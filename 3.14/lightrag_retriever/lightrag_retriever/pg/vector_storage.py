"""PostgreSQL vector storage for lightrag_retriever.

Read-only implementation of the VectorStorage interface for entity,
relationship, and chunk vector similarity search.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from lightrag_retriever.base import EmbeddingFunc, VectorStorage
from lightrag_retriever.constants import DEFAULT_QUERY_PRIORITY
from lightrag_retriever.pg.db import PostgreSQLDB
from lightrag_retriever.pg.namespace import namespace_to_table_name
from lightrag_retriever.pg.sql_templates import DDL_TEMPLATES, SQL_TEMPLATES

logger = logging.getLogger("lightrag_retriever")

PG_MAX_IDENTIFIER_LENGTH = 63


@dataclass
class PGVectorStorage(VectorStorage):
    namespace: str
    workspace: str = "default"
    global_config: dict[str, Any] = field(default_factory=dict)
    embedding_func: EmbeddingFunc = field(default=None)
    cosine_better_than_threshold: float = 0.2
    db: PostgreSQLDB = field(default=None)
    table_name: str = ""
    model_suffix: str = ""

    def __post_init__(self):
        self.model_suffix = self._generate_collection_suffix()
        base_table = namespace_to_table_name(self.namespace)
        if not base_table:
            raise ValueError(f"Unknown namespace: {self.namespace}")
        if self.model_suffix:
            self.table_name = f"{base_table}_{self.model_suffix}"
        else:
            self.table_name = base_table

        if len(self.table_name) > PG_MAX_IDENTIFIER_LENGTH:
            raise ValueError(
                f"PostgreSQL table name exceeds {PG_MAX_IDENTIFIER_LENGTH} chars: '{self.table_name}'"
            )

    def _generate_collection_suffix(self) -> str:
        import re

        model_name = ""
        if self.embedding_func is not None:
            model_name = getattr(self.embedding_func, "model_name", "") or ""
        if not model_name:
            return ""
        safe_model_name = re.sub(r"[^a-zA-Z0-9_]", "_", model_name.lower())
        embedding_dim = getattr(self.embedding_func, "embedding_dim", 1536)
        return f"{safe_model_name}_{embedding_dim}d"

    async def initialize(self):
        if self.db is None:
            self.db = PostgreSQLDB(self.global_config.get("_pg_config", {}))
            await self.db.initdb()
        if self.db.workspace:
            self.workspace = self.db.workspace

        # Create table if it doesn't exist (idempotent)
        base_table = namespace_to_table_name(self.namespace)
        if base_table in DDL_TEMPLATES:
            embedding_dim = getattr(self.embedding_func, "embedding_dim", 1536)
            vector_type = "VECTOR"
            if getattr(self.db, "vector_index_type", None) == "HNSW_HALFVEC":
                vector_type = "HALFVEC"

            ddl = DDL_TEMPLATES[base_table].replace(
                "VECTOR(dimension)", f"{vector_type}({embedding_dim})"
            )
            ddl = ddl.replace(base_table, self.table_name)
            ddl = ddl.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1)
            # Also replace PK constraint name to match new table name
            ddl = ddl.replace(
                f"{base_table}_PK", f"{self.table_name}_PK"
            )
            await self.db.execute(ddl)

            # Create indexes
            try:
                await self.db.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table_name.lower()}_id "
                    f"ON {self.table_name}(id)"
                )
            except Exception:
                pass
            try:
                await self.db.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table_name.lower()}_workspace_id "
                    f"ON {self.table_name}(workspace, id)"
                )
            except Exception:
                pass

    async def finalize(self):
        if self.db is not None:
            await self.db.close()
            self.db = None

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embeddings = await self.embedding_func(
                [query], context="query", _priority=DEFAULT_QUERY_PRIORITY
            )
            embedding = embeddings[0]

        vector_cast = (
            "halfvec"
            if getattr(self.db, "vector_index_type", None) == "HNSW_HALFVEC"
            else "vector"
        )
        sql = SQL_TEMPLATES[self.namespace].format(
            table_name=self.table_name, vector_cast=vector_cast
        )
        params = {
            "workspace": self.workspace,
            "closer_than_threshold": 1 - self.cosine_better_than_threshold,
            "top_k": top_k,
            "embedding": embedding,
        }
        results = await self.db.query(sql, params=list(params.values()), multirows=True)
        return results

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vector embeddings by ID (read-only, no pending buffer)."""
        if not ids:
            return {}
        query = (
            f"SELECT id, content_vector FROM {self.table_name} "
            f"WHERE workspace=$1 AND id = ANY($2)"
        )
        try:
            results = await self.db.query(
                query, [self.workspace, ids], multirows=True
            )
            result: dict[str, list[float]] = {}
            for row in results or []:
                if not row or "content_vector" not in row or "id" not in row:
                    continue
                vector_data = row["content_vector"]
                try:
                    if isinstance(vector_data, (list, tuple)):
                        result[row["id"]] = list(vector_data)
                    elif hasattr(vector_data, "tolist"):
                        result[row["id"]] = vector_data.tolist()
                    elif hasattr(vector_data, "to_list") and callable(
                        vector_data.to_list
                    ):
                        result[row["id"]] = vector_data.to_list()
                except (TypeError, ValueError) as e:
                    logger.warning(
                        f"Failed to parse vector data for ID {row['id']}: {e}"
                    )
            return result
        except Exception as e:
            logger.error(f"get_vectors_by_ids error: {e}")
            raise
