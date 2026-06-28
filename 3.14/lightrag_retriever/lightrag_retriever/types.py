"""
Data types for the lightrag_retriever package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Literal, Optional


@dataclass
class QueryParam:
    """Configuration parameters for query execution."""

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    only_need_context: bool = False
    only_need_prompt: bool = False
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    top_k: int = int(os.getenv("TOP_K", "40"))
    chunk_top_k: int = int(os.getenv("CHUNK_TOP_K", "20"))
    max_entity_tokens: int = int(os.getenv("MAX_ENTITY_TOKENS", "6000"))
    max_relation_tokens: int = int(os.getenv("MAX_RELATION_TOKENS", "8000"))
    max_total_tokens: int = int(os.getenv("MAX_TOTAL_TOKENS", "30000"))
    hl_keywords: list[str] = field(default_factory=list)
    ll_keywords: list[str] = field(default_factory=list)
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    user_prompt: str | None = None
    enable_rerank: bool = os.getenv("RERANK_BY_DEFAULT", "true").lower() == "true"
    include_references: bool = False


@dataclass
class QueryResult:
    """Unified query result for all query modes."""

    content: Optional[str] = None
    response_iterator: Optional[AsyncIterator[str]] = None
    raw_data: Optional[Dict[str, Any]] = None
    is_streaming: bool = False

    @property
    def reference_list(self) -> List[Dict[str, str]]:
        if self.raw_data:
            return self.raw_data.get("data", {}).get("references", [])
        return []

    @property
    def metadata(self) -> Dict[str, Any]:
        if self.raw_data:
            return self.raw_data.get("metadata", {})
        return {}


@dataclass
class QueryContextResult:
    """Query context result (intermediate, before LLM generation)."""

    context: str
    raw_data: Dict[str, Any]

    @property
    def reference_list(self) -> List[Dict[str, str]]:
        return self.raw_data.get("data", {}).get("references", [])


@dataclass
class CacheData:
    """Data structure for cache entries."""

    args_hash: str
    content: str
    prompt: str
    mode: str = "default"
    cache_type: str = "query"
    chunk_id: str | None = None
    queryparam: dict | None = None
