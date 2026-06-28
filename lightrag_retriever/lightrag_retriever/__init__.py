"""
lightrag_retriever — Standalone retriever package extracted from LightRAG.

Graph-based retrieval with multiple query modes (local, global, hybrid,
mix, naive, bypass) and full pipeline support (keyword extraction,
search, truncation, chunk merging, context building, LLM generation).

Usage:
    from lightrag_retriever import Retriever, QueryParam

    retriever = Retriever(
        knowledge_graph=graph_storage,
        entities_vdb=entities_vector_storage,
        relationships_vdb=relations_vector_storage,
        text_chunks=text_chunk_kv,
        chunks_vdb=chunks_vector_storage,
        query_llm_func=my_query_llm,
        keyword_llm_func=my_keyword_llm,
        embedding_func=embedding_func,
        tokenizer=tokenizer,
    )

    result = await retriever.aquery("What is LightRAG?", QueryParam(mode="hybrid"))
"""

from lightrag_retriever.base import (
    CacheStorage,
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
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    GRAPH_FIELD_SEP,
)
from lightrag_retriever.operate import (
    extract_keywords_only,
    get_keywords_from_query,
    kg_query,
    naive_query,
)
from lightrag_retriever.prompts import PROMPTS
from lightrag_retriever.retriever import Retriever
from lightrag_retriever.types import (
    CacheData,
    QueryContextResult,
    QueryParam,
    QueryResult,
)
from lightrag_retriever.utils import (
    Tokenizer,
    TiktokenTokenizer,
    compute_args_hash,
    convert_to_user_format,
    generate_reference_list_from_chunks,
    handle_cache,
    process_chunks_unified,
    save_to_cache,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)

# PG storage backends (optional — requires asyncpg + pgvector)
try:
    from lightrag_retriever.pg import (
        PGGraphStorage,
        PGKVStorage,
        PGVectorStorage,
        PostgreSQLDB,
    )
    from lightrag_retriever.factory import create_postgres_retriever
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

# LLM functions (optional — requires openai)
try:
    from lightrag_retriever.llm import openai_complete_if_cache, openai_embed
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

# S3 Vectors storage (optional — requires boto3)
try:
    from lightrag_retriever.s3_vectors import S3VectorsStorage
    _S3_VECTORS_AVAILABLE = True
except ImportError:
    _S3_VECTORS_AVAILABLE = False

__version__ = "0.1.0"

__all__ = [
    # Core
    "Retriever",
    "QueryParam",
    "QueryResult",
    "QueryContextResult",
    "CacheData",
    # Abstract interfaces
    "GraphStorage",
    "VectorStorage",
    "KVStorage",
    "CacheStorage",
    "EmbeddingFunc",
    # Tokenizer
    "Tokenizer",
    "TiktokenTokenizer",
    # Query functions
    "kg_query",
    "naive_query",
    "get_keywords_from_query",
    "extract_keywords_only",
    # Utilities
    "PROMPTS",
    "compute_args_hash",
    "convert_to_user_format",
    "generate_reference_list_from_chunks",
    "handle_cache",
    "process_chunks_unified",
    "save_to_cache",
    "split_string_by_multi_markers",
    "truncate_list_by_token_size",
    # Constants
    "GRAPH_FIELD_SEP",
    "DEFAULT_TOP_K",
    "DEFAULT_CHUNK_TOP_K",
    "DEFAULT_MAX_ENTITY_TOKENS",
    "DEFAULT_MAX_RELATION_TOKENS",
    "DEFAULT_MAX_TOTAL_TOKENS",
    "DEFAULT_RELATED_CHUNK_NUMBER",
    "DEFAULT_KG_CHUNK_PICK_METHOD",
]

# Conditionally extend __all__ with PG and LLM exports
if _PG_AVAILABLE:
    __all__.extend([
        "PGKVStorage",
        "PGVectorStorage",
        "PGGraphStorage",
        "PostgreSQLDB",
        "create_postgres_retriever",
    ])
if _LLM_AVAILABLE:
    __all__.extend([
        "openai_complete_if_cache",
        "openai_embed",
    ])
if _S3_VECTORS_AVAILABLE:
    __all__.extend([
        "S3VectorsStorage",
    ])
