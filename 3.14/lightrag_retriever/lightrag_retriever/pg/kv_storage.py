"""PostgreSQL KV storage for lightrag_retriever.

Read-only implementation of the KVStorage interface for text chunks and
LLM response cache. Supports get_by_id, get_by_ids, filter_keys, and
upsert (for LLM cache writes).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from lightrag_retriever.base import KVStorage
from lightrag_retriever.pg.db import PostgreSQLDB
from lightrag_retriever.pg.namespace import NameSpace, is_namespace, namespace_to_table_name
from lightrag_retriever.pg.sql_templates import SQL_TEMPLATES

logger = logging.getLogger("lightrag_retriever")


@dataclass
class PGKVStorage(KVStorage):
    namespace: str
    workspace: str = "default"
    global_config: dict[str, Any] = field(default_factory=dict)
    embedding_func: Any = None
    db: PostgreSQLDB = field(default=None)

    def __post_init__(self):
        pass

    async def initialize(self):
        if self.db is None:
            self.db = PostgreSQLDB(self.global_config.get("_pg_config", {}))
            await self.db.initdb()
        if self.db.workspace:
            self.workspace = self.db.workspace

    async def finalize(self):
        if self.db is not None:
            await self.db.close()
            self.db = None

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        sql = SQL_TEMPLATES["get_by_id_" + self.namespace]
        params = {"workspace": self.workspace, "id": id}
        response = await self.db.query(sql, list(params.values()))

        if response and is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            llm_cache_list = response.get("llm_cache_list", [])
            if isinstance(llm_cache_list, str):
                try:
                    llm_cache_list = json.loads(llm_cache_list)
                except json.JSONDecodeError:
                    llm_cache_list = []
            response["llm_cache_list"] = llm_cache_list

            heading = response.get("heading")
            if isinstance(heading, str):
                try:
                    heading = json.loads(heading)
                except json.JSONDecodeError:
                    heading = {}
            if not isinstance(heading, dict):
                heading = {}
            response["heading"] = heading

            sidecar = response.get("sidecar")
            if isinstance(sidecar, str):
                try:
                    sidecar = json.loads(sidecar)
                except json.JSONDecodeError:
                    sidecar = {}
            if not isinstance(sidecar, dict):
                sidecar = {}
            response["sidecar"] = sidecar

            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        if response and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            chunk_options = response.get("chunk_options")
            if isinstance(chunk_options, str):
                try:
                    chunk_options = json.loads(chunk_options)
                except json.JSONDecodeError:
                    chunk_options = {}
            if not isinstance(chunk_options, dict):
                chunk_options = {}
            response["chunk_options"] = chunk_options

        if response and is_namespace(
            self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
        ):
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            queryparam = response.get("queryparam")
            if isinstance(queryparam, str):
                try:
                    queryparam = json.loads(queryparam)
                except json.JSONDecodeError:
                    queryparam = None
            response = {
                **response,
                "return": response.get("return_value", ""),
                "cache_type": response.get("cache_type"),
                "original_prompt": response.get("original_prompt", ""),
                "chunk_id": response.get("chunk_id"),
                "queryparam": queryparam,
                "create_time": create_time,
                "update_time": create_time if update_time == 0 else update_time,
            }

        return response if response else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        sql = SQL_TEMPLATES["get_by_ids_" + self.namespace]
        params = {"workspace": self.workspace, "ids": ids}
        results = await self.db.query(sql, list(params.values()), multirows=True)

        def _order_results(rows):
            if not rows:
                return [None for _ in ids]
            id_map = {}
            for row in rows:
                if row is None:
                    continue
                row_id = row.get("id")
                if row_id is not None:
                    id_map[str(row_id)] = row
            return [id_map.get(str(requested_id)) for requested_id in ids]

        if results and is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            for result in results:
                llm_cache_list = result.get("llm_cache_list", [])
                if isinstance(llm_cache_list, str):
                    try:
                        llm_cache_list = json.loads(llm_cache_list)
                    except json.JSONDecodeError:
                        llm_cache_list = []
                result["llm_cache_list"] = llm_cache_list

                heading = result.get("heading")
                if isinstance(heading, str):
                    try:
                        heading = json.loads(heading)
                    except json.JSONDecodeError:
                        heading = {}
                if not isinstance(heading, dict):
                    heading = {}
                result["heading"] = heading

                sidecar = result.get("sidecar")
                if isinstance(sidecar, str):
                    try:
                        sidecar = json.loads(sidecar)
                    except json.JSONDecodeError:
                        sidecar = {}
                if not isinstance(sidecar, dict):
                    sidecar = {}
                result["sidecar"] = sidecar

                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        if results and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            for result in results:
                chunk_options = result.get("chunk_options")
                if isinstance(chunk_options, str):
                    try:
                        chunk_options = json.loads(chunk_options)
                    except json.JSONDecodeError:
                        chunk_options = {}
                if not isinstance(chunk_options, dict):
                    chunk_options = {}
                result["chunk_options"] = chunk_options

        if results and is_namespace(
            self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
        ):
            processed = []
            for row in results:
                create_time = row.get("create_time", 0)
                update_time = row.get("update_time", 0)
                queryparam = row.get("queryparam")
                if isinstance(queryparam, str):
                    try:
                        queryparam = json.loads(queryparam)
                    except json.JSONDecodeError:
                        queryparam = None
                processed.append({
                    **row,
                    "return": row.get("return_value", ""),
                    "cache_type": row.get("cache_type"),
                    "original_prompt": row.get("original_prompt", ""),
                    "chunk_id": row.get("chunk_id"),
                    "queryparam": queryparam,
                    "create_time": create_time,
                    "update_time": create_time if update_time == 0 else update_time,
                })
            return _order_results(processed)

        return _order_results(results)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        table_name = namespace_to_table_name(self.namespace)
        sql = f"SELECT id FROM {table_name} WHERE workspace=$1 AND id = ANY($2)"
        params = {"workspace": self.workspace, "ids": list(keys)}
        try:
            res = await self.db.query(sql, list(params.values()), multirows=True)
            if res:
                exist_keys = [key["id"] for key in res]
            else:
                exist_keys = []
            return set(s for s in keys if s not in exist_keys)
        except Exception as e:
            logger.error(f"filter_keys error: {e}")
            raise

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        if not is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            logger.warning(f"upsert not supported for namespace {self.namespace}")
            return

        upsert_sql = SQL_TEMPLATES["upsert_llm_response_cache"]
        for k, v in data.items():
            chunk_id = v.get("chunk_id")
            cache_type = v.get("cache_type")
            queryparam = v.get("queryparam")
            if isinstance(queryparam, dict):
                queryparam = json.dumps(queryparam, ensure_ascii=False)
            values = {
                "workspace": self.workspace,
                "id": k,
                "original_prompt": v.get("original_prompt", ""),
                "return_value": v.get("return", v.get("return_value", "")),
                "chunk_id": chunk_id,
                "cache_type": cache_type,
                "queryparam": queryparam,
            }
            await self.db.execute(upsert_sql, values)
