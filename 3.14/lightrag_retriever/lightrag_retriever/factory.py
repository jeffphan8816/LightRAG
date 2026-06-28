"""Factory for creating a fully-wired Retriever backed by PostgreSQL.

Usage:
    from lightrag_retriever import create_postgres_retriever

    retriever = await create_postgres_retriever(
        workspace="default",
        llm_model_name="gpt-4o-mini",
        embedding_model_name="text-embedding-3-small",
        embedding_dim=1536,
    )
    result = await retriever.aquery_data("What is X?", QueryParam(mode="hybrid"))
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from typing import Any

from lightrag_retriever.base import EmbeddingFunc
from lightrag_retriever.constants import DEFAULT_SUMMARY_LANGUAGE
from lightrag_retriever.llm import openai_complete_if_cache, openai_embed
from lightrag_retriever.pg.db import PostgreSQLDB, get_pg_config
from lightrag_retriever.pg.graph_storage import PGGraphStorage
from lightrag_retriever.pg.kv_storage import PGKVStorage
from lightrag_retriever.pg.namespace import NameSpace
from lightrag_retriever.pg.vector_storage import PGVectorStorage
from lightrag_retriever.retriever import Retriever
from lightrag_retriever.utils import TiktokenTokenizer

logger = logging.getLogger("lightrag_retriever")


async def create_postgres_retriever(
    *,
    workspace: str | None = None,
    llm_model_name: str | None = None,
    embedding_model_name: str | None = None,
    embedding_dim: int = 1536,
    embedding_max_token_size: int = 8192,
    cosine_better_than_threshold: float = 0.2,
    language: str = DEFAULT_SUMMARY_LANGUAGE,
    enable_llm_cache: bool = False,
    enable_content_headings: bool = False,
    pg_config: dict[str, Any] | None = None,
    **retriever_kwargs: Any,
) -> Retriever:
    """Build a Retriever with PostgreSQL storage backends and OpenAI LLM/embedding.

    All configuration is read from environment variables unless explicitly
    passed. The PG connection pool is shared across all storage instances.

    Args:
        workspace: PG workspace isolation key (default: env POSTGRES_WORKSPACE or "default").
        llm_model_name: OpenAI chat model name (default: env LLM_MODEL or "gpt-4o-mini").
        embedding_model_name: OpenAI embedding model (default: env EMBEDDING_MODEL or "text-embedding-3-small").
        embedding_dim: Embedding vector dimension.
        embedding_max_token_size: Max tokens per text for embedding truncation.
        cosine_better_than_threshold: Vector similarity threshold for search.
        language: Summary language for keyword extraction.
        enable_llm_cache: Whether to enable LLM response caching.
        enable_content_headings: Whether to include content headings in context.
        pg_config: Optional PG config dict (overrides env vars). If None, built from env.
        **retriever_kwargs: Additional kwargs passed to Retriever.

    Returns:
        A fully initialized Retriever instance.
    """
    # Resolve config
    if pg_config is None:
        pg_config = get_pg_config()
    if workspace is not None:
        pg_config["workspace"] = workspace
    ws = pg_config.get("workspace", "default")

    if llm_model_name is None:
        llm_model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    if embedding_model_name is None:
        embedding_model_name = os.environ.get(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )

    # Build shared PG connection pool
    db = PostgreSQLDB(pg_config)
    await db.initdb()

    # Build embedding function
    embedding_func = EmbeddingFunc(
        func=functools.partial(
            openai_embed,
            model=embedding_model_name,
            embedding_dim=embedding_dim,
            max_token_size=embedding_max_token_size,
        ),
        embedding_dim=embedding_dim,
        model_name=embedding_model_name,
    )

    # Build LLM functions
    query_llm_func = functools.partial(openai_complete_if_cache, model=llm_model_name)
    keyword_llm_func = functools.partial(openai_complete_if_cache, model=llm_model_name)

    # Build tokenizer
    tokenizer = TiktokenTokenizer(model_name=llm_model_name)

    # Shared global_config for storages
    shared_global_config: dict[str, Any] = {
        "_pg_config": pg_config,
        "embedding_func": embedding_func,
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": cosine_better_than_threshold,
        },
    }

    # Build storage instances — initialize in parallel to reduce cold start
    knowledge_graph = PGGraphStorage(
        namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
        workspace=ws,
        global_config=shared_global_config,
        db=db,
    )

    entities_vdb = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_ENTITIES,
        workspace=ws,
        global_config=shared_global_config,
        embedding_func=embedding_func,
        cosine_better_than_threshold=cosine_better_than_threshold,
        db=db,
    )

    relationships_vdb = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_RELATIONSHIPS,
        workspace=ws,
        global_config=shared_global_config,
        embedding_func=embedding_func,
        cosine_better_than_threshold=cosine_better_than_threshold,
        db=db,
    )

    chunks_vdb = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        workspace=ws,
        global_config=shared_global_config,
        embedding_func=embedding_func,
        cosine_better_than_threshold=cosine_better_than_threshold,
        db=db,
    )

    text_chunks = PGKVStorage(
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        workspace=ws,
        global_config=shared_global_config,
        embedding_func=embedding_func,
        db=db,
    )

    llm_response_cache = None
    if enable_llm_cache:
        llm_response_cache = PGKVStorage(
            namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
            workspace=ws,
            global_config=shared_global_config,
            db=db,
        )

    # Initialize all storages concurrently
    init_tasks = [
        knowledge_graph.initialize(),
        entities_vdb.initialize(),
        relationships_vdb.initialize(),
        chunks_vdb.initialize(),
        text_chunks.initialize(),
    ]
    if llm_response_cache is not None:
        init_tasks.append(llm_response_cache.initialize())
    await asyncio.gather(*init_tasks)

    # Build the Retriever
    retriever = Retriever(
        knowledge_graph=knowledge_graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks=text_chunks,
        chunks_vdb=chunks_vdb,
        query_llm_func=query_llm_func,
        keyword_llm_func=keyword_llm_func,
        embedding_func=embedding_func,
        tokenizer=tokenizer,
        llm_response_cache=llm_response_cache,
        language=language,
        enable_llm_cache=enable_llm_cache,
        enable_content_headings=enable_content_headings,
        llm_model_name=llm_model_name,
        **retriever_kwargs,
    )

    return retriever
