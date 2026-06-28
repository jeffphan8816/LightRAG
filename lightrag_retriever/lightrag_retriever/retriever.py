"""
Retriever class — the main entry point for the lightrag_retriever package.

Accepts injected dependencies (storage backends, LLM functions, embedding
function, tokenizer) and exposes aquery / aquery_data / aquery_llm APIs
matching LightRAG's query interface.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, AsyncIterator, Callable, Dict

from lightrag_retriever.base import (
    EmbeddingFunc,
    GraphStorage,
    KVStorage,
    VectorStorage,
)
from lightrag_retriever.constants import (
    DEFAULT_KG_CHUNK_PICK_METHOD,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_QUERY_PRIORITY,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_SUMMARY_LANGUAGE,
)
from lightrag_retriever.operate import (
    kg_query,
    naive_query,
)
from lightrag_retriever.prompts import PROMPTS
from lightrag_retriever.types import QueryParam, QueryResult
from lightrag_retriever.utils import (
    Tokenizer,
    convert_to_user_format,
    logger,
)

logger = logging.getLogger("lightrag_retriever")


class Retriever:
    """Standalone retriever with dependency injection.

    Provides the same query API as LightRAG (aquery, aquery_data, aquery_llm)
    but without any dependency on the lightrag package. All storage backends,
    LLM functions, embedding functions, and tokenizer are injected.

    Args:
        knowledge_graph: Graph storage for entity/relation graph.
        entities_vdb: Vector storage for entity embeddings.
        relationships_vdb: Vector storage for relationship embeddings.
        text_chunks: KV storage for text chunks.
        chunks_vdb: Vector storage for chunk embeddings (naive/mix modes).
        query_llm_func: Async callable for query LLM generation.
        keyword_llm_func: Async callable for keyword extraction LLM.
        embedding_func: EmbeddingFunc instance for query embeddings.
        tokenizer: Tokenizer instance for token budget management.
        llm_response_cache: Optional KV storage for LLM response caching.
        rerank_model_func: Optional async callable for reranking.
        language: Summary language for keyword extraction.
        enable_llm_cache: Whether LLM caching is enabled.
        enable_content_headings: Whether to include content headings.
        kg_chunk_pick_method: Chunk selection method ("VECTOR" or "WEIGHT").
        related_chunk_number: Max related chunks per entity/relation.
        min_rerank_score: Minimum rerank score threshold.
        llm_model_name: LLM model name (for cache identity).
        addon_params: Additional parameters dict.
    """

    def __init__(
        self,
        knowledge_graph: GraphStorage,
        entities_vdb: VectorStorage,
        relationships_vdb: VectorStorage,
        text_chunks: KVStorage,
        chunks_vdb: VectorStorage | None,
        query_llm_func: Callable[..., Any],
        keyword_llm_func: Callable[..., Any],
        embedding_func: EmbeddingFunc,
        tokenizer: Tokenizer,
        llm_response_cache: KVStorage | None = None,
        rerank_model_func: Callable[..., Any] | None = None,
        language: str = DEFAULT_SUMMARY_LANGUAGE,
        enable_llm_cache: bool = False,
        enable_content_headings: bool = False,
        kg_chunk_pick_method: str = DEFAULT_KG_CHUNK_PICK_METHOD,
        related_chunk_number: int = DEFAULT_RELATED_CHUNK_NUMBER,
        min_rerank_score: float = 0.0,
        llm_model_name: str | None = None,
        addon_params: dict[str, Any] | None = None,
    ) -> None:
        self.knowledge_graph = knowledge_graph
        self.entities_vdb = entities_vdb
        self.relationships_vdb = relationships_vdb
        self.text_chunks = text_chunks
        self.chunks_vdb = chunks_vdb
        self.llm_response_cache = llm_response_cache
        self.tokenizer = tokenizer
        self.embedding_func = embedding_func
        self.rerank_model_func = rerank_model_func
        self.language = language
        self.llm_model_name = llm_model_name
        self.addon_params = addon_params or {}

        # Build global_config that operate.py expects
        self.global_config: dict[str, Any] = {
            "tokenizer": tokenizer,
            "role_llm_funcs": {
                "query": query_llm_func,
                "keyword": keyword_llm_func,
            },
            "rerank_model_func": rerank_model_func,
            "enable_llm_cache": enable_llm_cache,
            "enable_content_headings": enable_content_headings,
            "kg_chunk_pick_method": kg_chunk_pick_method,
            "related_chunk_number": related_chunk_number,
            "min_rerank_score": min_rerank_score,
            "max_entity_tokens": DEFAULT_MAX_ENTITY_TOKENS,
            "max_relation_tokens": DEFAULT_MAX_RELATION_TOKENS,
            "max_total_tokens": DEFAULT_MAX_TOTAL_TOKENS,
            "llm_model_name": llm_model_name,
            "addon_params": self.addon_params,
            "_resolved_summary_language": language,
        }

        # Wire global_config into text_chunks so operate.py can access it
        self.text_chunks.global_config = self.global_config
        self.text_chunks.embedding_func = embedding_func

        # Wire global_config into cache if present
        if self.llm_response_cache is not None:
            self.llm_response_cache.global_config = self.global_config

    def _build_global_config(self) -> dict[str, Any]:
        return self.global_config

    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """Async query — returns LLM response content (backward-compatible wrapper)."""
        result = await self.aquery_llm(query, param, system_prompt)
        llm_response = result.get("llm_response", {})
        if llm_response.get("is_streaming"):
            return llm_response.get("response_iterator")
        return llm_response.get("content", "")

    async def aquery_data(
        self,
        query: str,
        param: QueryParam = QueryParam(),
    ) -> dict[str, Any]:
        """Async data retrieval — returns structured retrieval results without LLM generation."""
        global_config = self._build_global_config()
        data_param = QueryParam(
            mode=param.mode,
            only_need_context=True,
            only_need_prompt=False,
            response_type=param.response_type,
            stream=False,
            top_k=param.top_k,
            chunk_top_k=param.chunk_top_k,
            max_entity_tokens=param.max_entity_tokens,
            max_relation_tokens=param.max_relation_tokens,
            max_total_tokens=param.max_total_tokens,
            hl_keywords=param.hl_keywords,
            ll_keywords=param.ll_keywords,
            conversation_history=param.conversation_history,
            user_prompt=param.user_prompt,
            enable_rerank=param.enable_rerank,
        )
        query_result: QueryResult | None = None
        if data_param.mode in ["local", "global", "hybrid", "mix"]:
            query_result = await kg_query(
                query.strip(),
                self.knowledge_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                data_param,
                global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=None,
                chunks_vdb=self.chunks_vdb,
            )
        elif data_param.mode == "naive":
            query_result = await naive_query(
                query.strip(),
                self.chunks_vdb,
                data_param,
                global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=None,
                text_chunks_db=self.text_chunks,
            )
        elif data_param.mode == "bypass":
            empty_raw_data = convert_to_user_format([], [], [], [], "bypass")
            query_result = QueryResult(content="", raw_data=empty_raw_data)
        else:
            raise ValueError(f"Unknown mode {data_param.mode}")
        if query_result is None:
            no_result_message = "Query returned no results"
            if data_param.mode == "naive":
                no_result_message = "No relevant document chunks found."
            return {
                "status": "failure",
                "message": no_result_message,
                "data": {},
                "metadata": {"failure_reason": "no_results", "mode": data_param.mode},
            }
        return query_result.raw_data or {}

    async def aquery_llm(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Async complete query — returns structured retrieval results with LLM generation."""
        global_config = self._build_global_config()
        try:
            query_result: QueryResult | None = None
            if param.mode in ["local", "global", "hybrid", "mix"]:
                query_result = await kg_query(
                    query.strip(),
                    self.knowledge_graph,
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.text_chunks,
                    param,
                    global_config,
                    hashing_kv=self.llm_response_cache,
                    system_prompt=system_prompt,
                    chunks_vdb=self.chunks_vdb,
                )
            elif param.mode == "naive":
                query_result = await naive_query(
                    query.strip(),
                    self.chunks_vdb,
                    param,
                    global_config,
                    hashing_kv=self.llm_response_cache,
                    system_prompt=system_prompt,
                    text_chunks_db=self.text_chunks,
                )
            elif param.mode == "bypass":
                use_llm_func = partial(
                    global_config["role_llm_funcs"]["query"],
                    _priority=8,
                )
                param.stream = True if param.stream is None else param.stream
                response = await use_llm_func(
                    query.strip(),
                    system_prompt=system_prompt,
                    history_messages=param.conversation_history,
                    enable_cot=True,
                    stream=param.stream,
                )
                if type(response) is str:
                    return {
                        "status": "success",
                        "message": "Bypass mode LLM non streaming response",
                        "data": {},
                        "metadata": {},
                        "llm_response": {
                            "content": response,
                            "response_iterator": None,
                            "is_streaming": False,
                        },
                    }
                else:
                    return {
                        "status": "success",
                        "message": "Bypass mode LLM streaming response",
                        "data": {},
                        "metadata": {},
                        "llm_response": {
                            "content": None,
                            "response_iterator": response,
                            "is_streaming": True,
                        },
                    }
            else:
                raise ValueError(f"Unknown mode {param.mode}")

            if query_result is None:
                return {
                    "status": "failure",
                    "message": "Query returned no results",
                    "data": {},
                    "metadata": {"failure_reason": "no_results", "mode": param.mode},
                    "llm_response": {
                        "content": PROMPTS["fail_response"],
                        "response_iterator": None,
                        "is_streaming": False,
                    },
                }
            raw_data = query_result.raw_data or {}
            raw_data["llm_response"] = {
                "content": query_result.content if not query_result.is_streaming else None,
                "response_iterator": query_result.response_iterator if query_result.is_streaming else None,
                "is_streaming": query_result.is_streaming,
            }
            return raw_data
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "status": "failure",
                "message": f"Query failed: {str(e)}",
                "data": {},
                "metadata": {},
                "llm_response": {
                    "content": None,
                    "response_iterator": None,
                    "is_streaming": False,
                },
            }
