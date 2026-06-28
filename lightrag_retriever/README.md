# lightrag-retriever

Standalone retriever package extracted from [LightRAG](https://github.com/HKUDS/LightRAG). Provides graph-based retrieval with multiple query modes — no dependency on the `lightrag` package required.

## Features

- **6 query modes**: `local`, `global`, `hybrid`, `mix`, `naive`, `bypass`
- **Full pipeline**: keyword extraction → KG/vector search → token truncation → chunk merging → context building → LLM generation
- **Caching**: LLM response caching via injectable cache storage
- **Reranking**: Optional rerank model support
- **Token budget management**: Configurable entity/relation/total token limits
- **Dependency injection**: All storage backends, LLM functions, and embedding functions are injected — no coupling to specific implementations

## Installation

```bash
pip install lightrag-retriever
```

## Quick Start

```python
import asyncio
from lightrag_retriever import Retriever, QueryParam, EmbeddingFunc, TiktokenTokenizer

# 1. Implement the abstract interfaces (or use adapters for your existing storages)
from my_storages import (
    MyGraphStorage, MyVectorStorage, MyKVStorage
)

# 2. Create the retriever
retriever = Retriever(
    knowledge_graph=MyGraphStorage(),
    entities_vdb=MyVectorStorage(),
    relationships_vdb=MyVectorStorage(),
    text_chunks=MyKVStorage(),
    chunks_vdb=MyVectorStorage(),
    query_llm_func=my_query_llm_func,
    keyword_llm_func=my_keyword_llm_func,
    embedding_func=EmbeddingFunc(func=my_embedding_func, embedding_dim=1536),
    tokenizer=TiktokenTokenizer(),
)

# 3. Query
async def main():
    # Full query with LLM generation
    result = await retriever.aquery("What is LightRAG?", QueryParam(mode="hybrid"))
    print(result)

    # Data-only retrieval (no LLM generation)
    data = await retriever.aquery_data("What is LightRAG?", QueryParam(mode="local"))
    print(data["data"]["entities"])

asyncio.run(main())
```

## Query Modes

| Mode | Description |
|------|-------------|
| `local` | Entity-focused retrieval using low-level keywords |
| `global` | Relationship-focused retrieval using high-level keywords |
| `hybrid` | Combines local and global with round-robin merging |
| `mix` | KG retrieval + vector chunk retrieval (recommended with reranker) |
| `naive` | Direct vector search without knowledge graph |
| `bypass` | Direct LLM call without retrieval |

## Architecture

```
lightrag_retriever/
├── base.py        # Abstract interfaces (GraphStorage, VectorStorage, KVStorage, EmbeddingFunc)
├── types.py       # QueryParam, QueryResult, QueryContextResult, CacheData
├── constants.py   # Retrieval constants
├── prompts.py     # Prompt templates
├── utils.py       # Utility functions (tokenization, caching, chunk processing, format conversion)
├── operate.py     # Core retrieval logic (search, truncation, merging, context building)
└── retriever.py   # Retriever class — main entry point with dependency injection
```

## License

MIT
