"""
Tests for the lightrag_retriever package using mock storage implementations.
"""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag_retriever import (
    EmbeddingFunc,
    QueryParam,
    Retriever,
    TiktokenTokenizer,
)
from lightrag_retriever.base import GraphStorage, KVStorage, VectorStorage


# ---------------------------------------------------------------------------
# Mock storage implementations
# ---------------------------------------------------------------------------


class MockGraphStorage(GraphStorage):
    """In-memory mock graph storage for testing."""

    def __init__(self):
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: dict[tuple[str, str], dict[str, Any]] = {}

    async def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return (source_node_id, target_node_id) in self._edges

    async def node_degree(self, node_id: str) -> int:
        return sum(
            1
            for (s, t) in self._edges
            if s == node_id or t == node_id
        )

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return await self.node_degree(src_id) + await self.node_degree(tgt_id)

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        return self._nodes.get(node_id)

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {nid: self._nodes[nid] for nid in node_ids if nid in self._nodes}

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        return {nid: await self.node_degree(nid) for nid in node_ids}

    async def get_nodes_edges_batch(self, node_names: list[str]) -> dict[str, list[tuple[str, str]]]:
        result: dict[str, list[tuple[str, str]]] = {}
        for name in node_names:
            result[name] = [
                (s, t) for (s, t) in self._edges if s == name or t == name
            ]
        return result

    async def get_edges_batch(self, edge_pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
        result = {}
        for pair in edge_pairs:
            key = (pair["src"], pair["tgt"])
            if key in self._edges:
                result[key] = self._edges[key]
            else:
                rev_key = (pair["tgt"], pair["src"])
                if rev_key in self._edges:
                    result[key] = self._edges[rev_key]
        return result

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        return {pair: await self.edge_degree(*pair) for pair in edge_pairs}


class MockVectorStorage(VectorStorage):
    """Mock vector storage returning pre-configured results."""

    def __init__(self, results: list[dict[str, Any]] | None = None, cosine_threshold: float = 0.2):
        self._results = results or []
        self.cosine_better_than_threshold = cosine_threshold
        self.embedding_func = None

    async def query(self, query: str, top_k: int, query_embedding=None) -> list[dict[str, Any]]:
        return self._results[:top_k]

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        return {id_: [0.1] * 10 for id_ in ids}


class MockKVStorage(KVStorage):
    """In-memory mock KV storage for testing."""

    def __init__(self):
        self._data: dict[str, dict[str, Any]] = {}
        self.global_config: dict[str, Any] = {}
        self.embedding_func = None

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        return self._data.get(id)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]:
        return [self._data.get(id_) for id_ in ids]

    async def filter_keys(self, keys: list[str]) -> set[str]:
        return {k for k in keys if k not in self._data}

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        self._data.update(data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedding_func():
    async def _embed(texts: list[str], **kwargs) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]
    return EmbeddingFunc(func=_embed, embedding_dim=10, model_name="mock")


@pytest.fixture
def mock_tokenizer():
    return TiktokenTokenizer(model_name="gpt-4o-mini")


@pytest.fixture
def mock_query_llm():
    async def _query(prompt, system_prompt=None, history_messages=None, enable_cot=True, stream=False, **kwargs):
        return "This is a mock LLM response."
    return _query


@pytest.fixture
def mock_keyword_llm():
    async def _keyword(prompt, response_format=None, **kwargs):
        return json.dumps({
            "high_level_keywords": ["technology", "AI"],
            "low_level_keywords": ["LightRAG", "retrieval"],
        })
    return _keyword


@pytest.fixture
def graph_storage():
    gs = MockGraphStorage()
    gs._nodes = {
        "LightRAG": {
            "entity_name": "LightRAG",
            "entity_type": "Technology",
            "description": "A RAG framework using knowledge graphs.",
            "source_id": "chunk1<SEP>chunk2",
            "file_path": "doc1.md",
            "created_at": 1700000000,
        },
        "AI": {
            "entity_name": "AI",
            "entity_type": "Concept",
            "description": "Artificial Intelligence.",
            "source_id": "chunk3",
            "file_path": "doc2.md",
            "created_at": 1700000001,
        },
    }
    gs._edges = {
        ("LightRAG", "AI"): {
            "weight": 1.5,
            "description": "LightRAG uses AI for retrieval.",
            "keywords": "uses",
            "source_id": "chunk1",
            "file_path": "doc1.md",
            "created_at": 1700000000,
        },
    }
    return gs


@pytest.fixture
def entities_vdb():
    return MockVectorStorage(
        results=[
            {"entity_name": "LightRAG", "id": "ent1", "created_at": 1700000000},
            {"entity_name": "AI", "id": "ent2", "created_at": 1700000001},
        ]
    )


@pytest.fixture
def relationships_vdb():
    return MockVectorStorage(
        results=[
            {"src_id": "LightRAG", "tgt_id": "AI", "id": "rel1", "created_at": 1700000000},
        ]
    )


@pytest.fixture
def text_chunks():
    kv = MockKVStorage()
    kv._data = {
        "chunk1": {
            "content": "LightRAG is a retrieval-augmented generation framework.",
            "file_path": "doc1.md",
        },
        "chunk2": {
            "content": "It uses knowledge graphs for enhanced retrieval.",
            "file_path": "doc1.md",
        },
        "chunk3": {
            "content": "AI is a broad field of computer science.",
            "file_path": "doc2.md",
        },
    }
    return kv


@pytest.fixture
def chunks_vdb():
    return MockVectorStorage(
        results=[
            {"content": "LightRAG is a RAG framework.", "id": "chunk1", "file_path": "doc1.md"},
            {"content": "It uses knowledge graphs.", "id": "chunk2", "file_path": "doc1.md"},
        ]
    )


@pytest.fixture
def retriever(
    graph_storage,
    entities_vdb,
    relationships_vdb,
    text_chunks,
    chunks_vdb,
    mock_embedding_func,
    mock_tokenizer,
    mock_query_llm,
    mock_keyword_llm,
):
    return Retriever(
        knowledge_graph=graph_storage,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks=text_chunks,
        chunks_vdb=chunks_vdb,
        query_llm_func=mock_query_llm,
        keyword_llm_func=mock_keyword_llm,
        embedding_func=mock_embedding_func,
        tokenizer=mock_tokenizer,
        enable_llm_cache=False,
        enable_content_headings=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_naive_query_data(retriever):
    """Test naive mode data retrieval returns chunks."""
    result = await retriever.aquery_data(
        "What is LightRAG?", QueryParam(mode="naive")
    )
    assert result["status"] == "success"
    assert len(result["data"]["chunks"]) > 0
    assert result["data"]["entities"] == []
    assert result["data"]["relationships"] == []


@pytest.mark.asyncio
async def test_naive_query_llm(retriever):
    """Test naive mode with LLM generation."""
    result = await retriever.aquery_llm(
        "What is LightRAG?", QueryParam(mode="naive")
    )
    assert result["status"] == "success"
    assert result["llm_response"]["content"] == "This is a mock LLM response."
    assert result["llm_response"]["is_streaming"] is False


@pytest.mark.asyncio
async def test_local_query_data(retriever):
    """Test local mode data retrieval returns entities and chunks."""
    result = await retriever.aquery_data(
        "What is LightRAG?", QueryParam(mode="local")
    )
    assert result["status"] == "success"
    assert len(result["data"]["entities"]) > 0
    assert result["metadata"]["query_mode"] == "local"


@pytest.mark.asyncio
async def test_global_query_data(retriever):
    """Test global mode data retrieval returns relationships."""
    result = await retriever.aquery_data(
        "What is LightRAG?", QueryParam(mode="global")
    )
    assert result["status"] == "success"
    assert len(result["data"]["relationships"]) > 0


@pytest.mark.asyncio
async def test_hybrid_query_data(retriever):
    """Test hybrid mode combines local and global."""
    result = await retriever.aquery_data(
        "What is LightRAG?", QueryParam(mode="hybrid")
    )
    assert result["status"] == "success"
    assert len(result["data"]["entities"]) > 0
    assert len(result["data"]["relationships"]) > 0


@pytest.mark.asyncio
async def test_mix_query_data(retriever):
    """Test mix mode includes KG + vector chunks."""
    result = await retriever.aquery_data(
        "What is LightRAG?", QueryParam(mode="mix")
    )
    assert result["status"] == "success"
    assert len(result["data"]["chunks"]) > 0


@pytest.mark.asyncio
async def test_bypass_query_llm(retriever):
    """Test bypass mode directly calls LLM without retrieval."""
    result = await retriever.aquery_llm(
        "Hello", QueryParam(mode="bypass")
    )
    assert result["status"] == "success"
    assert result["llm_response"]["content"] == "This is a mock LLM response."


@pytest.mark.asyncio
async def test_aquery_returns_string(retriever):
    """Test aquery wrapper returns string content."""
    result = await retriever.aquery(
        "What is LightRAG?", QueryParam(mode="naive")
    )
    assert isinstance(result, str)
    assert result == "This is a mock LLM response."


@pytest.mark.asyncio
async def test_unknown_mode_raises(retriever):
    """Test unknown mode raises ValueError."""
    param = QueryParam(mode="invalid")  # type: ignore
    with pytest.raises(ValueError):
        await retriever.aquery_data("test", param)


@pytest.mark.asyncio
async def test_empty_query(retriever):
    """Test empty query returns fail response."""
    result = await retriever.aquery_llm("", QueryParam(mode="naive"))
    # Empty query in naive_query returns fail_response content
    assert result["llm_response"]["content"] == "Sorry, I'm not able to provide an answer to that question.[no-context]"


@pytest.mark.asyncio
async def test_query_with_predefined_keywords(retriever):
    """Test query with pre-defined keywords skips LLM keyword extraction."""
    param = QueryParam(
        mode="local",
        ll_keywords=["LightRAG"],
        hl_keywords=["AI"],
    )
    result = await retriever.aquery_data("test", param)
    assert result["status"] == "success"
    assert result["metadata"]["keywords"]["low_level"] == ["LightRAG"]
    assert result["metadata"]["keywords"]["high_level"] == ["AI"]
