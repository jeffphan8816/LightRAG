"""
Core retrieval operations for the lightrag_retriever package.

Extracted from LightRAG's operate.py — contains all retrieval logic:
keyword extraction, KG search, vector search, chunk merging, context
building, and the main kg_query / naive_query entry points.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from functools import partial
from typing import Any, AsyncIterator, overload, Literal

import json_repair

from lightrag_retriever.base import (
    GraphStorage,
    KVStorage,
    VectorStorage,
)
from lightrag_retriever.constants import (
    DEFAULT_KG_CHUNK_PICK_METHOD,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_SECTION_CONTEXT_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_QUERY_PRIORITY,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_SUMMARY_LANGUAGE,
    GRAPH_FIELD_SEP,
)
from lightrag_retriever.prompts import PROMPTS
from lightrag_retriever.types import (
    CacheData,
    QueryContextResult,
    QueryParam,
    QueryResult,
)
from lightrag_retriever.utils import (
    Tokenizer,
    compute_args_hash,
    convert_to_user_format,
    generate_reference_list_from_chunks,
    get_llm_cache_identity,
    handle_cache,
    logger,
    pick_by_vector_similarity,
    pick_by_weighted_polling,
    process_chunks_unified,
    remove_think_tags,
    save_to_cache,
    serialize_llm_cache_identity,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)

# ---------------------------------------------------------------------------
# Heading helpers (simplified from lightrag.chunk_schema)
# ---------------------------------------------------------------------------

HEADING_BREADCRUMB_SEP = " → "


def _truncate_section_context(
    heading_path: str,
    tokenizer: Tokenizer | None,
    max_tokens: int,
) -> str:
    if not heading_path or tokenizer is None or max_tokens <= 0:
        return heading_path
    if len(tokenizer.encode(heading_path)) <= max_tokens:
        return heading_path
    levels = heading_path.split(HEADING_BREADCRUMB_SEP)
    if len(levels) >= 3:
        heading_path = f"{levels[0]}{HEADING_BREADCRUMB_SEP}…{HEADING_BREADCRUMB_SEP}{levels[-1]}"
    tokens = tokenizer.encode(heading_path)
    if len(tokens) > max_tokens:
        ellipsis = "…"
        ellipsis_token_count = len(tokenizer.encode(ellipsis))
        if ellipsis_token_count <= max_tokens:
            for keep in range(max_tokens - ellipsis_token_count, -1, -1):
                candidate = tokenizer.decode(tokens[:keep]).rstrip() + ellipsis
                if len(tokenizer.encode(candidate)) <= max_tokens:
                    return candidate
        for keep in range(max_tokens, -1, -1):
            candidate = tokenizer.decode(tokens[:keep]).rstrip()
            if len(tokenizer.encode(candidate)) <= max_tokens:
                return candidate
        return ""
    return heading_path


def _format_parent_headings(dp: dict[str, Any]) -> str:
    """Join a chunk's parent heading chain into h1 → h2 → h3."""
    nested = dp.get("heading")
    if isinstance(nested, dict):
        parents_raw = nested.get("parent_headings") or []
    else:
        parents_raw = dp.get("parent_headings") or []
    parent_headings: list[str] = []
    if isinstance(parents_raw, list):
        for entry in parents_raw:
            text = str(entry or "").strip()
            if text:
                parent_headings.append(text)
    if not parent_headings:
        return ""
    return HEADING_BREADCRUMB_SEP.join(parent_headings)


async def _attach_content_headings(
    chunks: list[dict], text_chunks_db: KVStorage | None
) -> None:
    """Backfill content_headings field onto chunks in place."""
    if not text_chunks_db or not chunks:
        return
    tokenizer = text_chunks_db.global_config.get("tokenizer")
    chunk_ids = [c.get("chunk_id") for c in chunks]
    chunk_data_list = await text_chunks_db.get_by_ids(chunk_ids)
    for chunk, data in zip(chunks, chunk_data_list):
        if not isinstance(data, dict):
            continue
        headings = _truncate_section_context(
            _format_parent_headings(data),
            tokenizer,
            DEFAULT_MAX_SECTION_CONTEXT_TOKENS,
        )
        if headings:
            chunk["content_headings"] = headings


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


def _normalize_keyword_list(raw_values: Any, field_name: str) -> list[str]:
    if raw_values is None:
        return []
    if isinstance(raw_values, str):
        raw_values = [
            part.strip()
            for part in re.split(r"[\n,;]+", raw_values)
            if part and part.strip()
        ]
    if not isinstance(raw_values, list):
        logger.warning("Keyword field '%s' is not a list: %r", field_name, raw_values)
        return []
    normalized: list[str] = []
    for value in raw_values:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                normalized.append(cleaned)
    return normalized


_CODE_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL
)


def _strip_markdown_code_fence(text: str) -> str:
    match = _CODE_FENCE_PATTERN.match(text)
    return match.group(1) if match else text


def _parse_keywords_payload(result: Any) -> tuple[bool, list[str], list[str]]:
    if result is None:
        return False, [], []
    payload: Any
    if hasattr(result, "model_dump") and callable(result.model_dump):
        payload = result.model_dump()
    elif isinstance(result, dict):
        payload = result
    elif isinstance(result, str):
        cleaned_result = remove_think_tags(result)
        unfenced_result = _strip_markdown_code_fence(cleaned_result)
        if unfenced_result is not cleaned_result:
            cleaned_result = unfenced_result
        try:
            payload = json.loads(cleaned_result)
        except json.JSONDecodeError as strict_error:
            try:
                payload = json_repair.loads(cleaned_result)
                logger.warning(
                    "Keyword extraction required JSON repair: %s", strict_error
                )
            except Exception:
                return False, [], []
    else:
        return False, [], []
    if not isinstance(payload, dict):
        return False, [], []
    hl_keywords = _normalize_keyword_list(payload.get("high_level_keywords"), "high_level_keywords")
    ll_keywords = _normalize_keyword_list(payload.get("low_level_keywords"), "low_level_keywords")
    return True, hl_keywords, ll_keywords


async def extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict[str, Any],
    hashing_kv: KVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """Extract high-level and low-level keywords from text using LLM."""
    examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    addon_params = global_config.get("addon_params") or {}
    language = global_config.get("_resolved_summary_language")
    if language is None:
        language = addon_params.get("language", DEFAULT_SUMMARY_LANGUAGE)
    llm_cache_identity = get_llm_cache_identity(global_config, "keyword")
    args_hash = compute_args_hash(
        param.mode, text, language,
        "\n<llm_identity>\n",
        serialize_llm_cache_identity(llm_cache_identity),
    )
    cached_result = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_result is not None:
        cached_response, _ = cached_result
        is_valid, hl, ll = _parse_keywords_payload(cached_response)
        if is_valid:
            return hl, ll
    kw_prompt = PROMPTS["keywords_extraction"].format(
        query=text, examples=examples, language=language
    )
    use_model_func = partial(
        global_config["role_llm_funcs"]["keyword"], _priority=DEFAULT_QUERY_PRIORITY
    )
    result = await use_model_func(kw_prompt, response_format={"type": "json_object"})
    _, hl_keywords, ll_keywords = _parse_keywords_payload(result)
    if hl_keywords or ll_keywords:
        cache_data = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            queryparam_dict = {
                "mode": param.mode,
                "response_type": param.response_type,
                "top_k": param.top_k,
                "chunk_top_k": param.chunk_top_k,
                "max_entity_tokens": param.max_entity_tokens,
                "max_relation_tokens": param.max_relation_tokens,
                "max_total_tokens": param.max_total_tokens,
                "user_prompt": param.user_prompt or "",
                "enable_rerank": param.enable_rerank,
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=json.dumps(cache_data),
                    prompt=text,
                    mode=param.mode,
                    cache_type="keywords",
                    queryparam=queryparam_dict,
                ),
            )
    return hl_keywords, ll_keywords


async def get_keywords_from_query(
    query: str,
    query_param: QueryParam,
    global_config: dict[str, Any],
    hashing_kv: KVStorage | None = None,
) -> tuple[list[str], list[str]]:
    if query_param.hl_keywords or query_param.ll_keywords:
        return query_param.hl_keywords, query_param.ll_keywords
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )
    return hl_keywords, ll_keywords


# ---------------------------------------------------------------------------
# Vector context retrieval
# ---------------------------------------------------------------------------


async def _get_vector_context(
    query: str,
    chunks_vdb: VectorStorage,
    query_param: QueryParam,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """Retrieve text chunks from vector database without reranking or truncation."""
    try:
        search_top_k = query_param.chunk_top_k or query_param.top_k
        results = await chunks_vdb.query(
            query, top_k=search_top_k, query_embedding=query_embedding
        )
        if not results:
            logger.info(f"Naive query: 0 chunks (chunk_top_k:{search_top_k})")
            return []
        valid_chunks = []
        for result in results:
            if "content" in result:
                chunk_with_metadata = {
                    "content": result["content"],
                    "created_at": result.get("created_at", None),
                    "file_path": result.get("file_path", "unknown_source"),
                    "source_type": "vector",
                    "chunk_id": result.get("id"),
                }
                valid_chunks.append(chunk_with_metadata)
        logger.info(f"Naive query: {len(valid_chunks)} chunks (chunk_top_k:{search_top_k})")
        return valid_chunks
    except Exception as e:
        logger.error(f"Error in _get_vector_context: {e}")
        return []


# ---------------------------------------------------------------------------
# Entity / edge retrieval
# ---------------------------------------------------------------------------


async def _get_node_data(
    query: str,
    knowledge_graph_inst: GraphStorage,
    entities_vdb: VectorStorage,
    query_param: QueryParam,
    query_embedding=None,
) -> tuple[list[dict], list[dict]]:
    logger.info(
        f"Query nodes: {query} (top_k:{query_param.top_k}, "
        f"cosine:{entities_vdb.cosine_better_than_threshold})"
    )
    results = await entities_vdb.query(
        query, top_k=query_param.top_k, query_embedding=query_embedding
    )
    if not len(results):
        return [], []
    node_ids = [r["entity_name"] for r in results]
    nodes_dict, degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_nodes_batch(node_ids),
        knowledge_graph_inst.node_degrees_batch(node_ids),
    )
    node_datas = [nodes_dict.get(nid) for nid in node_ids]
    node_degrees = [degrees_dict.get(nid, 0) for nid in node_ids]
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_datas = [
        {
            **n,
            "entity_name": k["entity_name"],
            "rank": d,
            "created_at": k.get("created_at"),
        }
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(f"Local query: {len(node_datas)} entities, {len(use_relations)} relations")
    return node_datas, use_relations


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: GraphStorage,
) -> list[dict]:
    node_names = [dp["entity_name"] for dp in node_datas]
    batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)
    all_edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for node_name in node_names:
        this_edges = batch_edges_dict.get(node_name, [])
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)
    edge_pairs_dicts = [{"src": e[0], "tgt": e[1]} for e in all_edges]
    edge_pairs_tuples = list(all_edges)
    edge_data_dict, edge_degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
        knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
    )
    all_edges_data = []
    for pair in all_edges:
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                edge_props["weight"] = 1.0
            combined = {
                "src_tgt": pair,
                "rank": edge_degrees_dict.get(pair, 0),
                **edge_props,
            }
            all_edges_data.append(combined)
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    return all_edges_data


async def _get_edge_data(
    keywords: str,
    knowledge_graph_inst: GraphStorage,
    relationships_vdb: VectorStorage,
    query_param: QueryParam,
    query_embedding=None,
) -> tuple[list[dict], list[dict]]:
    logger.info(
        f"Query edges: {keywords} (top_k:{query_param.top_k}, "
        f"cosine:{relationships_vdb.cosine_better_than_threshold})"
    )
    results = await relationships_vdb.query(
        keywords, top_k=query_param.top_k, query_embedding=query_embedding
    )
    if not len(results):
        return [], []
    edge_pairs_dicts = [{"src": r["src_id"], "tgt": r["tgt_id"]} for r in results]
    edge_data_dict = await knowledge_graph_inst.get_edges_batch(edge_pairs_dicts)
    edge_datas = []
    for k in results:
        pair = (k["src_id"], k["tgt_id"])
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                edge_props["weight"] = 1.0
            combined = {
                "src_id": k["src_id"],
                "tgt_id": k["tgt_id"],
                "created_at": k.get("created_at", None),
                **edge_props,
            }
            edge_datas.append(combined)
    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )
    logger.info(f"Global query: {len(use_entities)} entities, {len(edge_datas)} relations")
    return edge_datas, use_entities


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: GraphStorage,
) -> list[dict]:
    entity_names: list[str] = []
    seen: set[str] = set()
    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])
    nodes_dict = await knowledge_graph_inst.get_nodes_batch(entity_names)
    node_datas = []
    for entity_name in entity_names:
        node = nodes_dict.get(entity_name)
        if node is None:
            continue
        combined = {**node, "entity_name": entity_name}
        node_datas.append(combined)
    return node_datas


# ---------------------------------------------------------------------------
# Chunk finding from entities / relations
# ---------------------------------------------------------------------------


async def _find_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: KVStorage,
    knowledge_graph_inst: GraphStorage,
    query: str | None = None,
    chunks_vdb: VectorStorage | None = None,
    chunk_tracking: dict | None = None,
    query_embedding=None,
) -> list[dict]:
    if not node_datas:
        return []
    entities_with_chunks = []
    for entity in node_datas:
        if entity.get("source_id"):
            chunks = split_string_by_multi_markers(entity["source_id"], [GRAPH_FIELD_SEP])
            if chunks:
                entities_with_chunks.append(
                    {
                        "entity_name": entity["entity_name"],
                        "chunks": chunks,
                        "entity_data": entity,
                    }
                )
    if not entities_with_chunks:
        logger.warning("No entities with text chunks found")
        return []
    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )
    chunk_occurrence_count: dict[str, int] = {}
    for entity_info in entities_with_chunks:
        deduplicated_chunks = []
        for chunk_id in entity_info["chunks"]:
            chunk_occurrence_count[chunk_id] = chunk_occurrence_count.get(chunk_id, 0) + 1
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
        entity_info["chunks"] = deduplicated_chunks
    total_entity_chunks = 0
    for entity_info in entities_with_chunks:
        sorted_chunks = sorted(
            entity_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        entity_info["sorted_chunks"] = sorted_chunks
        total_entity_chunks += len(sorted_chunks)
    selected_chunk_ids: list[str] = []
    if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
        num_of_chunks = int(max_related_chunks * len(entities_with_chunks) / 2)
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=entities_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                )
                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} "
                        f"entity-related chunks by vector similarity"
                    )
            except Exception as e:
                logger.error(f"Error in vector similarity sorting: {e}, falling back to WEIGHT")
                kg_chunk_pick_method = "WEIGHT"
    if kg_chunk_pick_method == "WEIGHT":
        selected_chunk_ids = pick_by_weighted_polling(
            entities_with_chunks, max_related_chunks, min_related_chunks=1
        )
        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} "
            f"entity-related chunks by weighted polling"
        )
    if not selected_chunk_ids:
        return []
    unique_chunk_ids = list(dict.fromkeys(selected_chunk_ids))
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "entity"
            chunk_data_copy["chunk_id"] = chunk_id
            result_chunks.append(chunk_data_copy)
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "E",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,
                }
    return result_chunks


async def _find_related_text_unit_from_relations(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: KVStorage,
    entity_chunks: list[dict] | None = None,
    query: str | None = None,
    chunks_vdb: VectorStorage | None = None,
    chunk_tracking: dict | None = None,
    query_embedding=None,
) -> list[dict]:
    if not edge_datas:
        return []
    relations_with_chunks = []
    for relation in edge_datas:
        if relation.get("source_id"):
            chunks = split_string_by_multi_markers(relation["source_id"], [GRAPH_FIELD_SEP])
            if chunks:
                if "src_tgt" in relation:
                    rel_key = tuple(sorted(relation["src_tgt"]))
                else:
                    rel_key = tuple(sorted([relation.get("src_id"), relation.get("tgt_id")]))
                relations_with_chunks.append(
                    {"relation_key": rel_key, "chunks": chunks, "relation_data": relation}
                )
    if not relations_with_chunks:
        logger.warning("No relation-related chunks found")
        return []
    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", DEFAULT_RELATED_CHUNK_NUMBER
    )
    entity_chunk_ids: set[str] = set()
    if entity_chunks:
        for chunk in entity_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                entity_chunk_ids.add(chunk_id)
    chunk_occurrence_count: dict[str, int] = {}
    for relation_info in relations_with_chunks:
        deduplicated_chunks = []
        for chunk_id in relation_info["chunks"]:
            if chunk_id in entity_chunk_ids:
                continue
            chunk_occurrence_count[chunk_id] = chunk_occurrence_count.get(chunk_id, 0) + 1
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
        relation_info["chunks"] = deduplicated_chunks
    relations_with_chunks = [r for r in relations_with_chunks if r["chunks"]]
    if not relations_with_chunks:
        logger.info(f"Find no additional relations-related chunks from {len(edge_datas)} relations")
        return []
    total_relation_chunks = 0
    for relation_info in relations_with_chunks:
        sorted_chunks = sorted(
            relation_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        relation_info["sorted_chunks"] = sorted_chunks
        total_relation_chunks += len(sorted_chunks)
    selected_chunk_ids: list[str] = []
    if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
        num_of_chunks = int(max_related_chunks * len(relations_with_chunks) / 2)
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=relations_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                )
                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} "
                        f"relation-related chunks by vector similarity"
                    )
            except Exception as e:
                logger.error(f"Error in vector similarity sorting: {e}, falling back to WEIGHT")
                kg_chunk_pick_method = "WEIGHT"
    if kg_chunk_pick_method == "WEIGHT":
        selected_chunk_ids = pick_by_weighted_polling(
            relations_with_chunks, max_related_chunks, min_related_chunks=1
        )
        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} "
            f"relation-related chunks by weighted polling"
        )
    if not selected_chunk_ids:
        return []
    unique_chunk_ids = list(dict.fromkeys(selected_chunk_ids))
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "relationship"
            chunk_data_copy["chunk_id"] = chunk_id
            result_chunks.append(chunk_data_copy)
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "R",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,
                }
    return result_chunks


# ---------------------------------------------------------------------------
# KG search (Stage 1)
# ---------------------------------------------------------------------------


async def _perform_kg_search(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: GraphStorage,
    entities_vdb: VectorStorage,
    relationships_vdb: VectorStorage,
    text_chunks_db: KVStorage,
    query_param: QueryParam,
    chunks_vdb: VectorStorage | None = None,
) -> dict[str, Any]:
    """Pure search logic — retrieves raw entities, relations, and vector chunks."""
    local_entities: list[dict] = []
    local_relations: list[dict] = []
    global_entities: list[dict] = []
    global_relations: list[dict] = []
    vector_chunks: list[dict] = []
    chunk_tracking: dict[str, dict] = {}

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", DEFAULT_KG_CHUNK_PICK_METHOD
    )
    actual_embedding_func = text_chunks_db.embedding_func
    query_embedding = None
    ll_embedding = None
    hl_embedding = None

    mode = query_param.mode
    need_ll = mode in ("local", "hybrid", "mix") and bool(ll_keywords)
    need_hl = mode in ("global", "hybrid", "mix") and bool(hl_keywords)

    if actual_embedding_func:
        texts_to_embed: list[str] = []
        text_purposes: list[str] = []
        if query and (kg_chunk_pick_method == "VECTOR" or chunks_vdb):
            texts_to_embed.append(query)
            text_purposes.append("query")
        if need_ll:
            texts_to_embed.append(ll_keywords)
            text_purposes.append("ll")
        if need_hl:
            texts_to_embed.append(hl_keywords)
            text_purposes.append("hl")
        if texts_to_embed:
            try:
                all_embeddings = await actual_embedding_func(
                    texts_to_embed, context="query", _priority=DEFAULT_QUERY_PRIORITY
                )
                for i, purpose in enumerate(text_purposes):
                    if purpose == "query":
                        query_embedding = all_embeddings[i]
                    elif purpose == "ll":
                        ll_embedding = all_embeddings[i]
                    elif purpose == "hl":
                        hl_embedding = all_embeddings[i]
            except Exception as e:
                logger.warning(f"Failed to batch pre-compute embeddings: {e}")

    if query_param.mode == "local" and len(ll_keywords) > 0:
        local_entities, local_relations = await _get_node_data(
            ll_keywords, knowledge_graph_inst, entities_vdb, query_param,
            query_embedding=ll_embedding,
        )
    elif query_param.mode == "global" and len(hl_keywords) > 0:
        global_relations, global_entities = await _get_edge_data(
            hl_keywords, knowledge_graph_inst, relationships_vdb, query_param,
            query_embedding=hl_embedding,
        )
    else:  # hybrid or mix
        if len(ll_keywords) > 0:
            local_entities, local_relations = await _get_node_data(
                ll_keywords, knowledge_graph_inst, entities_vdb, query_param,
                query_embedding=ll_embedding,
            )
        if len(hl_keywords) > 0:
            global_relations, global_entities = await _get_edge_data(
                hl_keywords, knowledge_graph_inst, relationships_vdb, query_param,
                query_embedding=hl_embedding,
            )
        if query_param.mode == "mix" and chunks_vdb:
            vector_chunks = await _get_vector_context(
                query, chunks_vdb, query_param, query_embedding
            )
            for i, chunk in enumerate(vector_chunks):
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                if chunk_id:
                    chunk_tracking[chunk_id] = {
                        "source": "C",
                        "frequency": 1,
                        "order": i + 1,
                    }

    # Round-robin merge entities
    final_entities: list[dict] = []
    seen_entities: set[str] = set()
    max_len = max(len(local_entities), len(global_entities))
    for i in range(max_len):
        if i < len(local_entities):
            entity = local_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name and entity_name not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_name)
        if i < len(global_entities):
            entity = global_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name and entity_name not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_name)

    # Round-robin merge relations
    final_relations: list[dict] = []
    seen_relations: set[tuple[str, str]] = set()
    max_len = max(len(local_relations), len(global_relations))
    for i in range(max_len):
        if i < len(local_relations):
            relation = local_relations[i]
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(sorted([relation.get("src_id"), relation.get("tgt_id")]))
            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)
        if i < len(global_relations):
            relation = global_relations[i]
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(sorted([relation.get("src_id"), relation.get("tgt_id")]))
            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

    logger.info(
        f"Raw search results: {len(final_entities)} entities, "
        f"{len(final_relations)} relations, {len(vector_chunks)} vector chunks"
    )
    return {
        "final_entities": final_entities,
        "final_relations": final_relations,
        "vector_chunks": vector_chunks,
        "chunk_tracking": chunk_tracking,
        "query_embedding": query_embedding,
    }


# ---------------------------------------------------------------------------
# Token truncation (Stage 2)
# ---------------------------------------------------------------------------


async def _apply_token_truncation(
    search_result: dict[str, Any],
    query_param: QueryParam,
    global_config: dict[str, Any],
) -> dict[str, Any]:
    tokenizer = global_config.get("tokenizer")
    if not tokenizer:
        logger.warning("No tokenizer found, skipping truncation")
        return {
            "entities_context": [],
            "relations_context": [],
            "filtered_entities": search_result["final_entities"],
            "filtered_relations": search_result["final_relations"],
            "entity_id_to_original": {},
            "relation_id_to_original": {},
        }
    max_entity_tokens = getattr(
        query_param, "max_entity_tokens",
        global_config.get("max_entity_tokens", DEFAULT_MAX_ENTITY_TOKENS),
    )
    max_relation_tokens = getattr(
        query_param, "max_relation_tokens",
        global_config.get("max_relation_tokens", DEFAULT_MAX_RELATION_TOKENS),
    )
    final_entities = search_result["final_entities"]
    final_relations = search_result["final_relations"]
    entity_id_to_original: dict[str, dict] = {}
    relation_id_to_original: dict[tuple[str, str], dict] = {}
    entities_context: list[dict] = []
    for entity in final_entities:
        entity_name = entity["entity_name"]
        created_at = entity.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        entity_id_to_original[entity_name] = entity
        entities_context.append(
            {
                "entity": entity_name,
                "type": entity.get("entity_type", "UNKNOWN"),
                "description": entity.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": entity.get("file_path", "unknown_source"),
            }
        )
    relations_context: list[dict] = []
    for relation in final_relations:
        created_at = relation.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        if "src_tgt" in relation:
            entity1, entity2 = relation["src_tgt"]
        else:
            entity1, entity2 = relation.get("src_id"), relation.get("tgt_id")
        relation_key = (entity1, entity2)
        relation_id_to_original[relation_key] = relation
        relations_context.append(
            {
                "entity1": entity1,
                "entity2": entity2,
                "description": relation.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": relation.get("file_path", "unknown_source"),
            }
        )
    if entities_context:
        entities_context_for_truncation = []
        for entity in entities_context:
            entity_copy = entity.copy()
            entity_copy.pop("file_path", None)
            entity_copy.pop("created_at", None)
            entities_context_for_truncation.append(entity_copy)
        entities_context = truncate_list_by_token_size(
            entities_context_for_truncation,
            key=lambda x: "\n".join(json.dumps(item, ensure_ascii=False) for item in [x]),
            max_token_size=max_entity_tokens,
            tokenizer=tokenizer,
        )
    if relations_context:
        relations_context_for_truncation = []
        for relation in relations_context:
            relation_copy = relation.copy()
            relation_copy.pop("file_path", None)
            relation_copy.pop("created_at", None)
            relations_context_for_truncation.append(relation_copy)
        relations_context = truncate_list_by_token_size(
            relations_context_for_truncation,
            key=lambda x: "\n".join(json.dumps(item, ensure_ascii=False) for item in [x]),
            max_token_size=max_relation_tokens,
            tokenizer=tokenizer,
        )
    filtered_entities: list[dict] = []
    filtered_entity_id_to_original: dict[str, dict] = {}
    if entities_context:
        final_entity_names = {e["entity"] for e in entities_context}
        seen_nodes: set[str] = set()
        for entity in final_entities:
            name = entity.get("entity_name")
            if name in final_entity_names and name not in seen_nodes:
                filtered_entities.append(entity)
                filtered_entity_id_to_original[name] = entity
                seen_nodes.add(name)
    filtered_relations: list[dict] = []
    filtered_relation_id_to_original: dict[tuple[str, str], dict] = {}
    if relations_context:
        final_relation_pairs = {(r["entity1"], r["entity2"]) for r in relations_context}
        seen_edges: set[tuple[str, str]] = set()
        for relation in final_relations:
            src, tgt = relation.get("src_id"), relation.get("tgt_id")
            if src is None or tgt is None:
                src, tgt = relation.get("src_tgt", (None, None))
            pair = (src, tgt)
            if pair in final_relation_pairs and pair not in seen_edges:
                filtered_relations.append(relation)
                filtered_relation_id_to_original[pair] = relation
                seen_edges.add(pair)
    return {
        "entities_context": entities_context,
        "relations_context": relations_context,
        "filtered_entities": filtered_entities,
        "filtered_relations": filtered_relations,
        "entity_id_to_original": filtered_entity_id_to_original,
        "relation_id_to_original": filtered_relation_id_to_original,
    }


# ---------------------------------------------------------------------------
# Chunk merging (Stage 3)
# ---------------------------------------------------------------------------


async def _merge_all_chunks(
    filtered_entities: list[dict],
    filtered_relations: list[dict],
    vector_chunks: list[dict],
    query: str = "",
    knowledge_graph_inst: GraphStorage | None = None,
    text_chunks_db: KVStorage | None = None,
    query_param: QueryParam | None = None,
    chunks_vdb: VectorStorage | None = None,
    chunk_tracking: dict | None = None,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    if chunk_tracking is None:
        chunk_tracking = {}
    entity_chunks: list[dict] = []
    if filtered_entities and text_chunks_db:
        entity_chunks = await _find_related_text_unit_from_entities(
            filtered_entities, query_param, text_chunks_db, knowledge_graph_inst,
            query, chunks_vdb, chunk_tracking=chunk_tracking, query_embedding=query_embedding,
        )
    relation_chunks: list[dict] = []
    if filtered_relations and text_chunks_db:
        relation_chunks = await _find_related_text_unit_from_relations(
            filtered_relations, query_param, text_chunks_db, entity_chunks,
            query, chunks_vdb, chunk_tracking=chunk_tracking, query_embedding=query_embedding,
        )
    merged_chunks: list[dict] = []
    seen_chunk_ids: set[str] = set()
    max_len = max(len(vector_chunks), len(entity_chunks), len(relation_chunks))
    origin_len = len(vector_chunks) + len(entity_chunks) + len(relation_chunks)
    for i in range(max_len):
        for source_list in (vector_chunks, entity_chunks, relation_chunks):
            if i < len(source_list):
                chunk = source_list[i]
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    merged_chunks.append(
                        {
                            "content": chunk["content"],
                            "file_path": chunk.get("file_path", "unknown_source"),
                            "chunk_id": chunk_id,
                        }
                    )
    logger.info(
        f"Round-robin merged chunks: {origin_len} -> {len(merged_chunks)} "
        f"(deduplicated {origin_len - len(merged_chunks)})"
    )
    if text_chunks_db and text_chunks_db.global_config.get("enable_content_headings", False):
        await _attach_content_headings(merged_chunks, text_chunks_db)
    return merged_chunks


# ---------------------------------------------------------------------------
# Context string building (Stage 4)
# ---------------------------------------------------------------------------


async def _build_context_str(
    entities_context: list[dict],
    relations_context: list[dict],
    merged_chunks: list[dict],
    query: str,
    query_param: QueryParam,
    global_config: dict[str, Any],
    chunk_tracking: dict | None = None,
    entity_id_to_original: dict | None = None,
    relation_id_to_original: dict | None = None,
) -> tuple[str, dict[str, Any]]:
    tokenizer = global_config.get("tokenizer")
    if not tokenizer:
        empty_raw_data = convert_to_user_format([], [], [], [], query_param.mode)
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Missing tokenizer, cannot build LLM context."
        return "", empty_raw_data
    max_total_tokens = getattr(
        query_param, "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )
    sys_prompt_template = global_config.get("system_prompt_template", PROMPTS["rag_response"])
    kg_context_template = PROMPTS["kg_query_context"]
    user_prompt = query_param.user_prompt if query_param.user_prompt else ""
    response_type = query_param.response_type if query_param.response_type else "Multiple Paragraphs"
    entities_str = "\n".join(json.dumps(e, ensure_ascii=False) for e in entities_context)
    relations_str = "\n".join(json.dumps(r, ensure_ascii=False) for r in relations_context)
    pre_kg_context = kg_context_template.format(
        entities_str=entities_str, relations_str=relations_str,
        text_chunks_str="", reference_list_str="",
    )
    kg_context_tokens = len(tokenizer.encode(pre_kg_context))
    pre_sys_prompt = sys_prompt_template.format(
        context_data="", response_type=response_type, user_prompt=user_prompt,
    )
    sys_prompt_tokens = len(tokenizer.encode(pre_sys_prompt))
    query_tokens = len(tokenizer.encode(query))
    buffer_tokens = 200
    available_chunk_tokens = max_total_tokens - (
        sys_prompt_tokens + kg_context_tokens + query_tokens + buffer_tokens
    )
    truncated_chunks = await process_chunks_unified(
        query=query, unique_chunks=merged_chunks, query_param=query_param,
        global_config=global_config, source_type=query_param.mode,
        chunk_token_limit=available_chunk_tokens,
    )
    reference_list, truncated_chunks = generate_reference_list_from_chunks(truncated_chunks)
    chunks_context = []
    for chunk in truncated_chunks:
        entry = {"reference_id": chunk["reference_id"], "content": chunk["content"]}
        if chunk.get("content_headings"):
            entry["content_headings"] = chunk["content_headings"]
        chunks_context.append(entry)
    text_units_str = "\n".join(json.dumps(tu, ensure_ascii=False) for tu in chunks_context)
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}" for ref in reference_list if ref["reference_id"]
    )
    logger.info(
        f"Final context: {len(entities_context)} entities, "
        f"{len(relations_context)} relations, {len(chunks_context)} chunks"
    )
    if not entities_context and not relations_context and not chunks_context:
        empty_raw_data = convert_to_user_format([], [], [], [], query_param.mode)
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Query returned empty dataset."
        return "", empty_raw_data
    result = kg_context_template.format(
        entities_str=entities_str, relations_str=relations_str,
        text_chunks_str=text_units_str, reference_list_str=reference_list_str,
    )
    final_data = convert_to_user_format(
        entities_context, relations_context, truncated_chunks,
        reference_list, query_param.mode,
        entity_id_to_original, relation_id_to_original,
    )
    return result, final_data


# ---------------------------------------------------------------------------
# Build query context (orchestrates Stages 1-4)
# ---------------------------------------------------------------------------


async def _build_query_context(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: GraphStorage,
    entities_vdb: VectorStorage,
    relationships_vdb: VectorStorage,
    text_chunks_db: KVStorage,
    query_param: QueryParam,
    chunks_vdb: VectorStorage | None = None,
) -> QueryContextResult | None:
    if not query:
        return None
    search_result = await _perform_kg_search(
        query, ll_keywords, hl_keywords, knowledge_graph_inst,
        entities_vdb, relationships_vdb, text_chunks_db, query_param, chunks_vdb,
    )
    if not search_result["final_entities"] and not search_result["final_relations"]:
        if query_param.mode != "mix":
            return None
        else:
            if not search_result["chunk_tracking"]:
                return None
    truncation_result = await _apply_token_truncation(
        search_result, query_param, text_chunks_db.global_config
    )
    merged_chunks = await _merge_all_chunks(
        filtered_entities=truncation_result["filtered_entities"],
        filtered_relations=truncation_result["filtered_relations"],
        vector_chunks=search_result["vector_chunks"],
        query=query, knowledge_graph_inst=knowledge_graph_inst,
        text_chunks_db=text_chunks_db, query_param=query_param,
        chunks_vdb=chunks_vdb, chunk_tracking=search_result["chunk_tracking"],
        query_embedding=search_result["query_embedding"],
    )
    if (
        not merged_chunks
        and not truncation_result["entities_context"]
        and not truncation_result["relations_context"]
    ):
        return None
    context, raw_data = await _build_context_str(
        entities_context=truncation_result["entities_context"],
        relations_context=truncation_result["relations_context"],
        merged_chunks=merged_chunks, query=query, query_param=query_param,
        global_config=text_chunks_db.global_config,
        chunk_tracking=search_result["chunk_tracking"],
        entity_id_to_original=truncation_result["entity_id_to_original"],
        relation_id_to_original=truncation_result["relation_id_to_original"],
    )
    hl_keywords_list = hl_keywords.split(", ") if hl_keywords else []
    ll_keywords_list = ll_keywords.split(", ") if ll_keywords else []
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}
    raw_data["metadata"]["keywords"] = {
        "high_level": hl_keywords_list,
        "low_level": ll_keywords_list,
    }
    raw_data["metadata"]["processing_info"] = {
        "total_entities_found": len(search_result.get("final_entities", [])),
        "total_relations_found": len(search_result.get("final_relations", [])),
        "entities_after_truncation": len(truncation_result.get("filtered_entities", [])),
        "relations_after_truncation": len(truncation_result.get("filtered_relations", [])),
        "merged_chunks_count": len(merged_chunks),
        "final_chunks_count": len(raw_data.get("data", {}).get("chunks", [])),
    }
    return QueryContextResult(context=context, raw_data=raw_data)


# ---------------------------------------------------------------------------
# Main query functions
# ---------------------------------------------------------------------------


async def kg_query(
    query: str,
    knowledge_graph_inst: GraphStorage,
    entities_vdb: VectorStorage,
    relationships_vdb: VectorStorage,
    text_chunks_db: KVStorage,
    query_param: QueryParam,
    global_config: dict[str, Any],
    hashing_kv: KVStorage | None = None,
    system_prompt: str | None = None,
    chunks_vdb: VectorStorage | None = None,
) -> QueryResult | None:
    """Execute knowledge graph query and return unified QueryResult."""
    if not query:
        return QueryResult(content=PROMPTS["fail_response"])
    use_model_func = partial(
        global_config["role_llm_funcs"]["query"], _priority=DEFAULT_QUERY_PRIORITY
    )
    llm_cache_identity = get_llm_cache_identity(global_config, "query")
    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, query_param, global_config, hashing_kv
    )
    if ll_keywords == [] and query_param.mode in ["local", "hybrid", "mix"]:
        logger.warning("low_level_keywords is empty")
    if hl_keywords == [] and query_param.mode in ["global", "hybrid", "mix"]:
        logger.warning("high_level_keywords is empty")
    if hl_keywords == [] and ll_keywords == []:
        if len(query) < 50:
            ll_keywords = [query]
        else:
            return QueryResult(content=PROMPTS["fail_response"])
    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""
    context_result = await _build_query_context(
        query, ll_keywords_str, hl_keywords_str, knowledge_graph_inst,
        entities_vdb, relationships_vdb, text_chunks_db, query_param, chunks_vdb,
    )
    if context_result is None:
        return None
    if query_param.only_need_context and not query_param.only_need_prompt:
        return QueryResult(content=context_result.context, raw_data=context_result.raw_data)
    user_prompt = f"\n\n{query_param.user_prompt}" if query_param.user_prompt else "n/a"
    response_type = query_param.response_type if query_param.response_type else "Multiple Paragraphs"
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        response_type=response_type, user_prompt=user_prompt,
        context_data=context_result.context,
    )
    user_query = query
    if query_param.only_need_prompt:
        prompt_content = "\n\n".join([sys_prompt, "---User Query---", user_query])
        return QueryResult(content=prompt_content, raw_data=context_result.raw_data)
    tokenizer: Tokenizer = global_config["tokenizer"]
    args_hash = compute_args_hash(
        query_param.mode, query, query_param.response_type, query_param.top_k,
        query_param.chunk_top_k, query_param.max_entity_tokens,
        query_param.max_relation_tokens, query_param.max_total_tokens,
        hl_keywords_str, ll_keywords_str, query_param.user_prompt or "",
        query_param.enable_rerank, global_config.get("enable_content_headings", False),
        "\n<llm_identity>\n", serialize_llm_cache_identity(llm_cache_identity),
    )
    cached_result = await handle_cache(
        hashing_kv, args_hash, user_query, query_param.mode, cache_type="query"
    )
    if cached_result is not None:
        response = cached_result[0]
    else:
        response = await use_model_func(
            user_query, system_prompt=sys_prompt,
            history_messages=query_param.conversation_history,
            enable_cot=True, stream=query_param.stream,
        )
        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            queryparam_dict = {
                "mode": query_param.mode,
                "response_type": query_param.response_type,
                "top_k": query_param.top_k,
                "chunk_top_k": query_param.chunk_top_k,
                "max_entity_tokens": query_param.max_entity_tokens,
                "max_relation_tokens": query_param.max_relation_tokens,
                "max_total_tokens": query_param.max_total_tokens,
                "hl_keywords": hl_keywords_str,
                "ll_keywords": ll_keywords_str,
                "user_prompt": query_param.user_prompt or "",
                "enable_rerank": query_param.enable_rerank,
                "enable_content_headings": global_config.get("enable_content_headings", False),
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash, content=response, prompt=query,
                    mode=query_param.mode, cache_type="query", queryparam=queryparam_dict,
                ),
            )
    if isinstance(response, str):
        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "").replace("user", "")
                .replace("model", "").replace(query, "")
                .replace("<system>", "").replace("</system>", "").strip()
            )
        return QueryResult(content=response, raw_data=context_result.raw_data)
    else:
        return QueryResult(
            response_iterator=response, raw_data=context_result.raw_data, is_streaming=True
        )


async def naive_query(
    query: str,
    chunks_vdb: VectorStorage,
    query_param: QueryParam,
    global_config: dict[str, Any],
    hashing_kv: KVStorage | None = None,
    system_prompt: str | None = None,
    text_chunks_db: KVStorage | None = None,
) -> QueryResult | None:
    """Execute naive query (vector-only) and return unified QueryResult."""
    if not query:
        return QueryResult(content=PROMPTS["fail_response"])
    use_model_func = partial(
        global_config["role_llm_funcs"]["query"], _priority=DEFAULT_QUERY_PRIORITY
    )
    llm_cache_identity = get_llm_cache_identity(global_config, "query")
    tokenizer: Tokenizer = global_config["tokenizer"]
    if not tokenizer:
        logger.error("Tokenizer not found in global configuration.")
        return QueryResult(content=PROMPTS["fail_response"])
    chunks = await _get_vector_context(query, chunks_vdb, query_param, None)
    if chunks is None or len(chunks) == 0:
        return None
    if global_config.get("enable_content_headings", False):
        await _attach_content_headings(chunks, text_chunks_db)
    max_total_tokens = getattr(
        query_param, "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )
    user_prompt = f"\n\n{query_param.user_prompt}" if query_param.user_prompt else "n/a"
    response_type = query_param.response_type if query_param.response_type else "Multiple Paragraphs"
    sys_prompt_template = system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    pre_sys_prompt = sys_prompt_template.format(
        response_type=response_type, user_prompt=user_prompt, content_data="",
    )
    sys_prompt_tokens = len(tokenizer.encode(pre_sys_prompt))
    query_tokens = len(tokenizer.encode(query))
    buffer_tokens = 200
    available_chunk_tokens = max_total_tokens - (sys_prompt_tokens + query_tokens + buffer_tokens)
    processed_chunks = await process_chunks_unified(
        query=query, unique_chunks=chunks, query_param=query_param,
        global_config=global_config, source_type="vector",
        chunk_token_limit=available_chunk_tokens,
    )
    reference_list, processed_chunks_with_ref_ids = generate_reference_list_from_chunks(
        processed_chunks
    )
    raw_data = convert_to_user_format(
        [], [], processed_chunks_with_ref_ids, reference_list, "naive"
    )
    if "metadata" not in raw_data:
        raw_data["metadata"] = {}
    raw_data["metadata"]["keywords"] = {"high_level": [], "low_level": []}
    raw_data["metadata"]["processing_info"] = {
        "total_chunks_found": len(chunks),
        "final_chunks_count": len(processed_chunks_with_ref_ids),
    }
    chunks_context = []
    for chunk in processed_chunks_with_ref_ids:
        entry = {"reference_id": chunk["reference_id"], "content": chunk["content"]}
        if chunk.get("content_headings"):
            entry["content_headings"] = chunk["content_headings"]
        chunks_context.append(entry)
    text_units_str = "\n".join(json.dumps(tu, ensure_ascii=False) for tu in chunks_context)
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}" for ref in reference_list if ref["reference_id"]
    )
    context_content = PROMPTS["naive_query_context"].format(
        text_chunks_str=text_units_str, reference_list_str=reference_list_str,
    )
    if query_param.only_need_context and not query_param.only_need_prompt:
        return QueryResult(content=context_content, raw_data=raw_data)
    sys_prompt = sys_prompt_template.format(
        response_type=query_param.response_type, user_prompt=user_prompt,
        content_data=context_content,
    )
    user_query = query
    if query_param.only_need_prompt:
        prompt_content = "\n\n".join([sys_prompt, "---User Query---", user_query])
        return QueryResult(content=prompt_content, raw_data=raw_data)
    args_hash = compute_args_hash(
        query_param.mode, query, query_param.response_type, query_param.top_k,
        query_param.chunk_top_k, query_param.max_entity_tokens,
        query_param.max_relation_tokens, query_param.max_total_tokens,
        query_param.user_prompt or "", query_param.enable_rerank,
        global_config.get("enable_content_headings", False),
        "\n<llm_identity>\n", serialize_llm_cache_identity(llm_cache_identity),
    )
    cached_result = await handle_cache(
        hashing_kv, args_hash, user_query, query_param.mode, cache_type="query"
    )
    if cached_result is not None:
        response = cached_result[0]
    else:
        response = await use_model_func(
            user_query, system_prompt=sys_prompt,
            history_messages=query_param.conversation_history,
            enable_cot=True, stream=query_param.stream,
        )
        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            queryparam_dict = {
                "mode": query_param.mode,
                "response_type": query_param.response_type,
                "top_k": query_param.top_k,
                "chunk_top_k": query_param.chunk_top_k,
                "max_entity_tokens": query_param.max_entity_tokens,
                "max_relation_tokens": query_param.max_relation_tokens,
                "max_total_tokens": query_param.max_total_tokens,
                "user_prompt": query_param.user_prompt or "",
                "enable_rerank": query_param.enable_rerank,
                "enable_content_headings": global_config.get("enable_content_headings", False),
            }
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash, content=response, prompt=query,
                    mode=query_param.mode, cache_type="query", queryparam=queryparam_dict,
                ),
            )
    if isinstance(response, str):
        if len(response) > len(sys_prompt):
            response = (
                response[len(sys_prompt):].replace(sys_prompt, "")
                .replace("user", "").replace("model", "")
                .replace(query, "").replace("<system>", "")
                .replace("</system>", "").strip()
            )
        return QueryResult(content=response, raw_data=raw_data)
    else:
        return QueryResult(response_iterator=response, raw_data=raw_data, is_streaming=True)
