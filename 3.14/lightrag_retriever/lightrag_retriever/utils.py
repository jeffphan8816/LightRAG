"""
Utility functions for the lightrag_retriever package.

Extracted from LightRAG's utils.py — only the functions needed
by the retrieval pipeline are included here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Callable, List, Protocol, Sequence

import numpy as np

from lightrag_retriever.constants import (
    DEFAULT_MAX_TOTAL_TOKENS,
)
from lightrag_retriever.types import CacheData

logger = logging.getLogger("lightrag_retriever")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TokenizerInterface(Protocol):
    def encode(self, content: str) -> List[int]: ...
    def decode(self, tokens: List[int]) -> str: ...


class Tokenizer:
    """Wrapper around a tokenizer providing a consistent encode/decode interface."""

    def __init__(self, model_name: str, tokenizer: TokenizerInterface):
        self.model_name = model_name
        self.tokenizer = tokenizer

    def encode(self, content: str) -> List[int]:
        try:
            return self.tokenizer.encode(content)
        except ValueError as e:
            if "special token" not in str(e):
                raise
            try:
                return self.tokenizer.encode(content, disallowed_special=())
            except TypeError:
                raise e

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)


class TiktokenTokenizer(Tokenizer):
    """Tokenizer implementation using tiktoken."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is not installed. Install with `pip install tiktoken`."
            )
        try:
            tokenizer = tiktoken.encoding_for_model(model_name)
            super().__init__(model_name=model_name, tokenizer=tokenizer)
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}.")


# ---------------------------------------------------------------------------
# Hashing & cache identity
# ---------------------------------------------------------------------------


def compute_args_hash(*args: Any) -> str:
    """Compute an MD5 hash for the given arguments."""
    args_str = "".join([str(arg) for arg in args])
    try:
        return hashlib.md5(args_str.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        safe_bytes = args_str.encode("utf-8", errors="replace")
        return hashlib.md5(safe_bytes).hexdigest()


def _serialize_cache_variant(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            value = value.model_dump(mode="json")
        except TypeError:
            value = value.model_dump()
    if hasattr(value, "model_json_schema") and callable(value.model_json_schema):
        value = value.model_json_schema()
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=repr
        )
    except (TypeError, ValueError):
        return repr(value)


def get_llm_cache_identity(
    global_config: dict[str, Any] | None, role: str
) -> dict[str, Any]:
    config = global_config or {}
    identities = config.get("llm_cache_identities")
    if isinstance(identities, dict):
        identity = identities.get(role)
        if isinstance(identity, dict):
            return dict(identity)
    return {
        "role": role,
        "binding": None,
        "model": config.get("llm_model_name"),
        "host": None,
    }


def serialize_llm_cache_identity(identity: Any) -> str:
    return _serialize_cache_variant(identity)


def generate_cache_key(mode: str, cache_type: str, hash_value: str) -> str:
    return f"{mode}:{cache_type}:{hash_value}"


# ---------------------------------------------------------------------------
# Cache handling
# ---------------------------------------------------------------------------


async def handle_cache(
    hashing_kv,
    args_hash: str,
    prompt: str,
    mode: str = "default",
    cache_type: str = "unknown",
) -> tuple[str, int] | None:
    if hashing_kv is None:
        return None
    if mode != "default":
        if not hashing_kv.global_config.get("enable_llm_cache"):
            return None
    else:
        if not hashing_kv.global_config.get("enable_llm_cache_for_entity_extract"):
            return None
    flattened_key = generate_cache_key(mode, cache_type, args_hash)
    cache_entry = await hashing_kv.get_by_id(flattened_key)
    if cache_entry:
        logger.debug(f"Flattened cache hit(key:{flattened_key})")
        content = cache_entry["return"]
        timestamp = cache_entry.get("create_time", 0)
        return content, timestamp
    logger.debug(f"Cache missed(mode:{mode} type:{cache_type})")
    return None


async def save_to_cache(hashing_kv, cache_data: CacheData) -> None:
    if hashing_kv is None or not cache_data.content:
        return
    if hasattr(cache_data.content, "__aiter__"):
        logger.debug("Streaming response detected, skipping cache")
        return
    flattened_key = generate_cache_key(
        cache_data.mode, cache_data.cache_type, cache_data.args_hash
    )
    existing_cache = await hashing_kv.get_by_id(flattened_key)
    if existing_cache:
        existing_content = existing_cache.get("return")
        if existing_content == cache_data.content:
            logger.warning(
                f"Cache duplication detected for {flattened_key}, skipping update"
            )
            return
    cache_entry = {
        "return": cache_data.content,
        "cache_type": cache_data.cache_type,
        "chunk_id": cache_data.chunk_id if cache_data.chunk_id is not None else None,
        "original_prompt": cache_data.prompt,
        "queryparam": cache_data.queryparam if cache_data.queryparam is not None else None,
    }
    logger.info(f" == LLM cache == saving: {flattened_key}")
    await hashing_kv.upsert({flattened_key: cache_entry})


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers."""
    if not markers:
        return [content]
    result = re.split("|".join(re.escape(m) for m in markers), content)
    return [r.strip() for r in result if r.strip()]


def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> tags and their content."""
    text = re.sub(r"^((?!<think>).)*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def truncate_list_by_token_size(
    list_data: list[Any],
    key: Callable[[Any], str],
    max_token_size: int,
    tokenizer: Tokenizer,
) -> list[Any]:
    """Truncate a list of data by token size."""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(tokenizer.encode(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def cosine_similarity(v1, v2) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return float(dot_product / (norm1 * norm2))


# ---------------------------------------------------------------------------
# Chunk selection algorithms
# ---------------------------------------------------------------------------


def pick_by_weighted_polling(
    entities_or_relations: list[dict],
    max_related_chunks: int,
    min_related_chunks: int = 1,
) -> list[str]:
    """Linear gradient weighted polling algorithm for text chunk selection."""
    if not entities_or_relations:
        return []
    n = len(entities_or_relations)
    if n == 1:
        entity_chunks = entities_or_relations[0].get("sorted_chunks", [])
        return entity_chunks[:max_related_chunks]
    expected_counts = []
    for i in range(n):
        ratio = i / (n - 1) if n > 1 else 0
        expected = max_related_chunks - ratio * (max_related_chunks - min_related_chunks)
        expected_counts.append(int(round(expected)))
    selected_chunks: list[str] = []
    used_counts: list[int] = []
    total_remaining = 0
    for i, entity_rel in enumerate(entities_or_relations):
        entity_chunks = entity_rel.get("sorted_chunks", [])
        expected = expected_counts[i]
        actual = min(expected, len(entity_chunks))
        selected_chunks.extend(entity_chunks[:actual])
        used_counts.append(actual)
        remaining = expected - actual
        if remaining > 0:
            total_remaining += remaining
    for _ in range(total_remaining):
        allocated = False
        for i, entity_rel in enumerate(entities_or_relations):
            entity_chunks = entity_rel.get("sorted_chunks", [])
            if used_counts[i] < len(entity_chunks):
                selected_chunks.append(entity_chunks[used_counts[i]])
                used_counts[i] += 1
                allocated = True
                break
        if not allocated:
            break
    return selected_chunks


async def pick_by_vector_similarity(
    query: str,
    text_chunks_storage,
    chunks_vdb,
    num_of_chunks: int,
    entity_info: list[dict[str, Any]],
    embedding_func,
    query_embedding=None,
) -> list[str]:
    """Vector similarity-based text chunk selection."""
    if not entity_info or num_of_chunks <= 0:
        return []
    all_chunk_ids: set[str] = set()
    for entity in entity_info:
        chunk_ids = entity.get("sorted_chunks", [])
        all_chunk_ids.update(chunk_ids)
    if not all_chunk_ids:
        return []
    all_chunk_ids = list(all_chunk_ids)
    try:
        if query_embedding is None:
            query_embedding = await embedding_func([query], context="query")
            query_embedding = query_embedding[0]
        chunk_vectors = await chunks_vdb.get_vectors_by_ids(all_chunk_ids)
        if not chunk_vectors or len(chunk_vectors) != len(all_chunk_ids):
            return []
        similarities: list[tuple[str, float]] = []
        for chunk_id in all_chunk_ids:
            if chunk_id in chunk_vectors:
                chunk_embedding = chunk_vectors[chunk_id]
                try:
                    similarity = cosine_similarity(query_embedding, chunk_embedding)
                    similarities.append((chunk_id, similarity))
                except Exception:
                    pass
        similarities.sort(key=lambda x: x[1], reverse=True)
        selected = [chunk_id for chunk_id, _ in similarities[:num_of_chunks]]
        return selected
    except Exception as e:
        logger.error(f"Error in vector similarity sorting: {e}")
        return all_chunk_ids[:num_of_chunks]


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


async def apply_rerank_if_enabled(
    query: str,
    retrieved_docs: list[dict],
    global_config: dict,
    enable_rerank: bool = True,
    top_n: int | None = None,
) -> list[dict]:
    if not enable_rerank or not retrieved_docs:
        return retrieved_docs
    rerank_func = global_config.get("rerank_model_func")
    if not rerank_func:
        logger.warning(
            "Rerank is enabled but no rerank model is configured. "
            "Set up a rerank model or set enable_rerank=False."
        )
        return retrieved_docs
    try:
        document_texts = []
        for doc in retrieved_docs:
            content = (
                doc.get("content")
                or doc.get("text")
                or doc.get("chunk_content")
                or doc.get("document")
                or str(doc)
            )
            document_texts.append(content)
        rerank_results = await rerank_func(query=query, documents=document_texts, top_n=top_n)
        if rerank_results and len(rerank_results) > 0:
            if isinstance(rerank_results[0], dict) and "index" in rerank_results[0]:
                reranked_docs = []
                for result in rerank_results:
                    index = result["index"]
                    relevance_score = result["relevance_score"]
                    if 0 <= index < len(retrieved_docs):
                        doc = retrieved_docs[index].copy()
                        doc["rerank_score"] = relevance_score
                        reranked_docs.append(doc)
                logger.info(
                    f"Successfully reranked: {len(reranked_docs)} chunks from {len(retrieved_docs)} original chunks"
                )
                return reranked_docs
            else:
                return rerank_results[:top_n] if top_n else rerank_results
        else:
            logger.warning("Rerank returned empty results, using original chunks")
            return retrieved_docs
    except Exception as e:
        logger.error(f"Error during reranking: {e}, using original chunks")
        return retrieved_docs


# ---------------------------------------------------------------------------
# Unified chunk processing
# ---------------------------------------------------------------------------


async def process_chunks_unified(
    query: str,
    unique_chunks: list[dict],
    query_param,
    global_config: dict,
    source_type: str = "mixed",
    chunk_token_limit: int | None = None,
) -> list[dict]:
    """Unified processing: deduplication, chunk_top_k limiting, reranking, token truncation."""
    if not unique_chunks:
        return []
    origin_count = len(unique_chunks)
    if query_param.enable_rerank and query and unique_chunks:
        rerank_top_k = query_param.chunk_top_k or len(unique_chunks)
        unique_chunks = await apply_rerank_if_enabled(
            query=query,
            retrieved_docs=unique_chunks,
            global_config=global_config,
            enable_rerank=query_param.enable_rerank,
            top_n=rerank_top_k,
        )
    if query_param.enable_rerank and unique_chunks:
        min_rerank_score = global_config.get("min_rerank_score", 0.5)
        if min_rerank_score > 0.0:
            filtered_chunks = []
            for chunk in unique_chunks:
                rerank_score = chunk.get("rerank_score", 1.0)
                if rerank_score >= min_rerank_score:
                    filtered_chunks.append(chunk)
            unique_chunks = filtered_chunks
            if not unique_chunks:
                return []
    if query_param.chunk_top_k is not None and query_param.chunk_top_k > 0:
        if len(unique_chunks) > query_param.chunk_top_k:
            unique_chunks = unique_chunks[: query_param.chunk_top_k]
        logger.debug(
            f"Kept chunk_top-k: {len(unique_chunks)} chunks (deduplicated original: {origin_count})"
        )
    tokenizer = global_config.get("tokenizer")
    if tokenizer and unique_chunks:
        if chunk_token_limit is None:
            chunk_token_limit = getattr(
                query_param,
                "max_total_tokens",
                global_config.get("MAX_TOTAL_TOKENS", DEFAULT_MAX_TOTAL_TOKENS),
            )
        original_count = len(unique_chunks)
        unique_chunks = truncate_list_by_token_size(
            unique_chunks,
            key=lambda x: "\n".join(json.dumps(item, ensure_ascii=False) for item in [x]),
            max_token_size=chunk_token_limit,
            tokenizer=tokenizer,
        )
        logger.debug(
            f"Token truncation: {len(unique_chunks)} chunks from {original_count} "
            f"(chunk available tokens: {chunk_token_limit}, source: {source_type})"
        )
    final_chunks = []
    for i, chunk in enumerate(unique_chunks):
        chunk_with_id = chunk.copy()
        chunk_with_id["id"] = f"DC{i + 1}"
        final_chunks.append(chunk_with_id)
    return final_chunks


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------


def convert_to_user_format(
    entities_context: list[dict],
    relations_context: list[dict],
    chunks: list[dict],
    references: list[dict],
    query_mode: str,
    entity_id_to_original: dict | None = None,
    relation_id_to_original: dict | None = None,
) -> dict[str, Any]:
    """Convert internal data format to user-friendly format using original database data."""
    formatted_entities = []
    for entity in entities_context:
        entity_name = entity.get("entity", "")
        original_entity = None
        if entity_id_to_original and entity_name in entity_id_to_original:
            original_entity = entity_id_to_original[entity_name]
        if original_entity:
            formatted_entities.append(
                {
                    "entity_name": original_entity.get("entity_name", entity_name),
                    "entity_type": original_entity.get("entity_type", "UNKNOWN"),
                    "description": original_entity.get("description", ""),
                    "source_id": original_entity.get("source_id", ""),
                    "file_path": original_entity.get("file_path", "unknown_source"),
                    "created_at": original_entity.get("created_at", ""),
                }
            )
        else:
            formatted_entities.append(
                {
                    "entity_name": entity_name,
                    "entity_type": entity.get("type", "UNKNOWN"),
                    "description": entity.get("description", ""),
                    "source_id": entity.get("source_id", ""),
                    "file_path": entity.get("file_path", "unknown_source"),
                    "created_at": entity.get("created_at", ""),
                }
            )
    formatted_relationships = []
    for relation in relations_context:
        entity1 = relation.get("entity1", "")
        entity2 = relation.get("entity2", "")
        relation_key = (entity1, entity2)
        original_relation = None
        if relation_id_to_original and relation_key in relation_id_to_original:
            original_relation = relation_id_to_original[relation_key]
        if original_relation:
            formatted_relationships.append(
                {
                    "src_id": original_relation.get("src_id", entity1),
                    "tgt_id": original_relation.get("tgt_id", entity2),
                    "description": original_relation.get("description", ""),
                    "keywords": original_relation.get("keywords", ""),
                    "weight": original_relation.get("weight", 1.0),
                    "source_id": original_relation.get("source_id", ""),
                    "file_path": original_relation.get("file_path", "unknown_source"),
                    "created_at": original_relation.get("created_at", ""),
                }
            )
        else:
            formatted_relationships.append(
                {
                    "src_id": entity1,
                    "tgt_id": entity2,
                    "description": relation.get("description", ""),
                    "keywords": relation.get("keywords", ""),
                    "weight": relation.get("weight", 1.0),
                    "source_id": relation.get("source_id", ""),
                    "file_path": relation.get("file_path", "unknown_source"),
                    "created_at": relation.get("created_at", ""),
                }
            )
    formatted_chunks = []
    for chunk in chunks:
        chunk_data = {
            "reference_id": chunk.get("reference_id", ""),
            "content": chunk.get("content", ""),
            "file_path": chunk.get("file_path", "unknown_source"),
            "chunk_id": chunk.get("chunk_id", ""),
        }
        formatted_chunks.append(chunk_data)
    metadata = {
        "query_mode": query_mode,
        "keywords": {"high_level": [], "low_level": []},
    }
    return {
        "status": "success",
        "message": "Query processed successfully",
        "data": {
            "entities": formatted_entities,
            "relationships": formatted_relationships,
            "chunks": formatted_chunks,
            "references": references,
        },
        "metadata": metadata,
    }


def generate_reference_list_from_chunks(
    chunks: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Generate reference list from chunks, prioritizing by occurrence frequency."""
    if not chunks:
        return [], []
    file_path_counts: dict[str, int] = {}
    for chunk in chunks:
        file_path = chunk.get("file_path", "")
        if file_path and file_path != "unknown_source":
            file_path_counts[file_path] = file_path_counts.get(file_path, 0) + 1
    file_path_with_indices: list[tuple[str, int, int]] = []
    seen_paths: set[str] = set()
    for i, chunk in enumerate(chunks):
        file_path = chunk.get("file_path", "")
        if file_path and file_path != "unknown_source" and file_path not in seen_paths:
            file_path_with_indices.append((file_path, file_path_counts[file_path], i))
            seen_paths.add(file_path)
    sorted_file_paths = sorted(file_path_with_indices, key=lambda x: (-x[1], x[2]))
    unique_file_paths = [item[0] for item in sorted_file_paths]
    file_path_to_ref_id: dict[str, str] = {}
    for i, file_path in enumerate(unique_file_paths):
        file_path_to_ref_id[file_path] = str(i + 1)
    updated_chunks = []
    for chunk in chunks:
        chunk_copy = chunk.copy()
        file_path = chunk_copy.get("file_path", "")
        if file_path and file_path != "unknown_source":
            chunk_copy["reference_id"] = file_path_to_ref_id[file_path]
        else:
            chunk_copy["reference_id"] = ""
        updated_chunks.append(chunk_copy)
    reference_list = []
    for i, file_path in enumerate(unique_file_paths):
        reference_list.append({"reference_id": str(i + 1), "file_path": file_path})
    return reference_list, updated_chunks
