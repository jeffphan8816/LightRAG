"""
Abstract base classes for the lightrag_retriever package.

These interfaces decouple the retrieval logic from any specific storage,
LLM, embedding, or caching implementation. Users inject concrete
instances when constructing a Retriever.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class EmbeddingFunc:
    """Wrapper for an async embedding function with metadata."""

    func: Callable[..., Any]
    embedding_dim: int = 0
    model_name: str = ""
    max_async: int = 8

    async def __call__(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return await self.func(texts, **kwargs)


class GraphStorage(ABC):
    """Abstract interface for knowledge graph storage."""

    @abstractmethod
    async def has_node(self, node_id: str) -> bool: ...

    @abstractmethod
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool: ...

    @abstractmethod
    async def node_degree(self, node_id: str) -> int: ...

    @abstractmethod
    async def edge_degree(self, src_id: str, tgt_id: str) -> int: ...

    @abstractmethod
    async def get_node(self, node_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict[str, Any]]: ...

    @abstractmethod
    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]: ...

    @abstractmethod
    async def get_nodes_edges_batch(self, node_names: list[str]) -> dict[str, list[tuple[str, str]]]: ...

    @abstractmethod
    async def get_edges_batch(self, edge_pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]: ...

    @abstractmethod
    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]: ...


class VectorStorage(ABC):
    """Abstract interface for vector database storage."""

    embedding_func: EmbeddingFunc
    cosine_better_than_threshold: float = 0.2

    @abstractmethod
    async def query(
        self, query: str, top_k: int, query_embedding: list[float] | None = None
    ) -> list[dict[str, Any]]: ...


class KVStorage(ABC):
    """Abstract interface for key-value storage (text chunks, cache)."""

    global_config: dict[str, Any]
    embedding_func: EmbeddingFunc | None = None

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]: ...

    @abstractmethod
    async def filter_keys(self, keys: list[str]) -> set[str]: ...

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None: ...


class CacheStorage(KVStorage):
    """Abstract interface for cache-specific KV storage.

    Adds ``enable_llm_cache`` flag access via ``global_config``.
    """

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None: ...
