"""PostgreSQL graph storage (Apache AGE) for lightrag_retriever.

Read-only implementation of the GraphStorage interface for entity/relation
graph queries via Cypher on Apache AGE.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Union

import asyncpg

from lightrag_retriever.base import GraphStorage
from lightrag_retriever.pg.db import PostgreSQLDB, _dollar_quote

logger = logging.getLogger("lightrag_retriever")


class PGGraphQueryException(Exception):
    def __init__(self, exception: Union[str, dict[str, Any]]) -> None:
        if isinstance(exception, dict):
            self.message = exception.get("message", "unknown")
            self.details = exception.get("details", "unknown")
        else:
            self.message = exception
            self.details = "unknown"

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details


@dataclass
class PGGraphStorage(GraphStorage):
    namespace: str = "chunk_entity_relation"
    workspace: str = "default"
    global_config: dict[str, Any] = field(default_factory=dict)
    db: PostgreSQLDB = field(default=None)
    graph_name: str = ""

    def __post_init__(self):
        pass

    def _get_workspace_graph_name(self) -> str:
        workspace = self.workspace
        namespace = self.namespace
        if workspace and workspace.strip() and workspace.strip().lower() != "default":
            safe_workspace = re.sub(r"[^a-zA-Z0-9_]", "_", workspace.strip())
            safe_namespace = re.sub(r"[^a-zA-Z0-9_]", "_", namespace)
            return f"{safe_workspace}_{safe_namespace}"
        else:
            return re.sub(r"[^a-zA-Z0-9_]", "_", namespace)

    @staticmethod
    def _normalize_node_id(node_id: str) -> str:
        normalized_id = node_id.replace("\x00", "")
        normalized_id = normalized_id.replace("\\", "\\\\")
        normalized_id = normalized_id.replace('"', '\\"')
        return normalized_id

    async def initialize(self):
        if self.db is None:
            self.db = PostgreSQLDB(self.global_config.get("_pg_config", {}))
            await self.db.initdb()
        if self.db.workspace:
            self.workspace = self.db.workspace

        self.graph_name = self._get_workspace_graph_name()
        logger.info(
            f"[{self.workspace}] PG graph initialized: graph_name='{self.graph_name}'"
        )

        async def _do_configure_age_extension(connection: asyncpg.Connection) -> None:
            await PostgreSQLDB.configure_age_extension(connection)

        await self.db._run_with_retry(_do_configure_age_extension)

        queries = [
            f"SELECT create_graph('{self.graph_name}')",
            f"SELECT create_vlabel('{self.graph_name}', 'base');",
            f"SELECT create_elabel('{self.graph_name}', 'DIRECTED');",
            f'CREATE INDEX CONCURRENTLY IF NOT EXISTS vertex_idx_node_id ON {self.graph_name}."_ag_label_vertex" (ag_catalog.agtype_access_operator(properties, \'"entity_id"\'::agtype))',
            f'CREATE INDEX CONCURRENTLY IF NOT EXISTS edge_sid_idx ON {self.graph_name}."_ag_label_edge" (start_id)',
            f'CREATE INDEX CONCURRENTLY IF NOT EXISTS edge_eid_idx ON {self.graph_name}."_ag_label_edge" (end_id)',
            f'CREATE INDEX CONCURRENTLY IF NOT EXISTS edge_seid_idx ON {self.graph_name}."_ag_label_edge" (start_id,end_id)',
            f'CREATE INDEX CONCURRENTLY IF NOT EXISTS directed_p_idx ON {self.graph_name}."DIRECTED" (id)',
            f'CREATE INDEX CONCURRENTLY IF NOT EXISTS directed_eid_idx ON {self.graph_name}."DIRECTED" (end_id)',
            f'CREATE INDEX CONCURRENTLY IF NOT EXISTS directed_sid_idx ON {self.graph_name}."DIRECTED" (start_id)',
            f'CREATE INDEX CONCURRENTLY IF NOT EXISTS directed_seid_idx ON {self.graph_name}."DIRECTED" (start_id,end_id)',
            f'CREATE INDEX CONCURRENTLY IF NOT EXISTS entity_p_idx ON {self.graph_name}."base" (id)',
            f'CREATE INDEX CONCURRENTLY IF NOT EXISTS entity_idx_node_id ON {self.graph_name}."base" (ag_catalog.agtype_access_operator(properties, \'"entity_id"\'::agtype))',
            f'CREATE INDEX CONCURRENTLY IF NOT EXISTS entity_node_id_gin_idx ON {self.graph_name}."base" using gin(properties)',
            f'ALTER TABLE {self.graph_name}."DIRECTED" CLUSTER ON directed_sid_idx',
        ]

        for query in queries:
            await self.db.execute(
                query,
                upsert=True,
                ignore_if_exists=True,
                with_age=True,
                graph_name=self.graph_name,
            )

    async def finalize(self):
        if self.db is not None:
            await self.db.close()
            self.db = None

    @staticmethod
    def _record_to_dict(record: asyncpg.Record) -> dict[str, Any]:
        @staticmethod
        def parse_agtype_string(agtype_str: str) -> tuple[str, str]:
            if not isinstance(agtype_str, str) or "::" not in agtype_str:
                return agtype_str, ""
            last_double_colon = agtype_str.rfind("::")
            if last_double_colon == -1:
                return agtype_str, ""
            json_content = agtype_str[:last_double_colon]
            type_identifier = agtype_str[last_double_colon + 2:]
            return json_content, type_identifier

        @staticmethod
        def safe_json_parse(json_str: str, context: str = "") -> dict:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None

        d = {}
        vertices = {}

        for k in record.keys():
            v = record[k]
            if isinstance(v, str) and "::" in v:
                if v.startswith("[") and v.endswith("]"):
                    json_content, type_id = parse_agtype_string(v)
                    if type_id == "vertex":
                        vertexes = safe_json_parse(json_content, f"vertices array for {k}")
                        if vertexes:
                            for vertex in vertexes:
                                vertices[vertex["id"]] = vertex.get("properties")
                else:
                    json_content, type_id = parse_agtype_string(v)
                    if type_id == "vertex":
                        vertex = safe_json_parse(json_content, f"single vertex for {k}")
                        if vertex:
                            vertices[vertex["id"]] = vertex.get("properties")

        for k in record.keys():
            v = record[k]
            if isinstance(v, str) and "::" in v:
                if v.startswith("[") and v.endswith("]"):
                    json_content, type_id = parse_agtype_string(v)
                    if type_id in ["vertex", "edge"]:
                        parsed_data = safe_json_parse(json_content, f"array {type_id} for field {k}")
                        d[k] = parsed_data if parsed_data is not None else None
                    else:
                        d[k] = None
                else:
                    json_content, type_id = parse_agtype_string(v)
                    if type_id in ["vertex", "edge"]:
                        parsed_data = safe_json_parse(json_content, f"single {type_id} for field {k}")
                        d[k] = parsed_data if parsed_data is not None else None
                    else:
                        d[k] = v
            else:
                d[k] = v

        return d

    async def _query(
        self,
        query: str,
        readonly: bool = True,
        upsert: bool = False,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            if readonly:
                data = await self.db.query(
                    query,
                    list(params.values()) if params else None,
                    multirows=True,
                    with_age=True,
                    graph_name=self.graph_name,
                )
            else:
                data = await self.db.execute(
                    query,
                    data=params,
                    upsert=upsert,
                    with_age=True,
                    graph_name=self.graph_name,
                )
        except Exception as e:
            raise PGGraphQueryException(
                {
                    "message": f"Error executing graph query: {query}",
                    "wrapped": query,
                    "detail": repr(e),
                    "error_type": e.__class__.__name__,
                }
            ) from e

        if data is None:
            result = []
        else:
            result = [self._record_to_dict(d) for d in data]
        return result

    async def has_node(self, node_id: str) -> bool:
        query = f"""
            SELECT EXISTS (
              SELECT 1
              FROM {self.graph_name}.base
              WHERE ag_catalog.agtype_access_operator(
                      VARIADIC ARRAY[properties, '"entity_id"'::agtype]
                    ) = (to_json($1::text)::text)::agtype
              LIMIT 1
            ) AS node_exists;
        """
        params = {"node_id": node_id}
        row = (await self._query(query, params=params))[0]
        return bool(row["node_exists"])

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        query = f"""
            WITH a AS (
              SELECT id AS vid
              FROM {self.graph_name}.base
              WHERE ag_catalog.agtype_access_operator(
                      VARIADIC ARRAY[properties, '"entity_id"'::agtype]
                    ) = (to_json($1::text)::text)::agtype
            ),
            b AS (
              SELECT id AS vid
              FROM {self.graph_name}.base
              WHERE ag_catalog.agtype_access_operator(
                      VARIADIC ARRAY[properties, '"entity_id"'::agtype]
                    ) = (to_json($2::text)::text)::agtype
            )
            SELECT EXISTS (
              SELECT 1
              FROM {self.graph_name}."DIRECTED" d
              JOIN a ON d.start_id = a.vid
              JOIN b ON d.end_id   = b.vid
              LIMIT 1
            )
            OR EXISTS (
              SELECT 1
              FROM {self.graph_name}."DIRECTED" d
              JOIN a ON d.end_id   = a.vid
              JOIN b ON d.start_id = b.vid
              LIMIT 1
            ) AS edge_exists;
        """
        params = {
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
        }
        row = (await self._query(query, params=params))[0]
        return bool(row["edge_exists"])

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        result = await self.get_nodes_batch(node_ids=[node_id])
        if result and node_id in result:
            return result[node_id]
        return None

    async def node_degree(self, node_id: str) -> int:
        result = await self.node_degrees_batch(node_ids=[node_id])
        if result and node_id in result:
            return result[node_id]
        return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        result = await self.edge_degrees_batch(edge_pairs=[(src_id, tgt_id)])
        if result and (src_id, tgt_id) in result:
            return result[(src_id, tgt_id)]
        return 0

    async def get_nodes_batch(
        self, node_ids: list[str], batch_size: int = 1000
    ) -> dict[str, dict]:
        if not node_ids:
            return {}

        seen: set[str] = set()
        unique_ids: list[str] = []
        lookup: dict[str, str] = {}
        requested: set[str] = set()
        for nid in node_ids:
            if nid not in seen:
                seen.add(nid)
                unique_ids.append(nid)
            requested.add(nid)
            lookup[nid] = nid
            lookup[self._normalize_node_id(nid)] = nid

        nodes_dict = {}

        for i in range(0, len(unique_ids), batch_size):
            batch = unique_ids[i: i + batch_size]

            query = f"""
                WITH input(v, ord) AS (
                  SELECT v, ord
                  FROM unnest($1::text[]) WITH ORDINALITY AS t(v, ord)
                ),
                ids(node_id, ord) AS (
                  SELECT (to_json(v)::text)::agtype AS node_id, ord
                  FROM input
                )
                SELECT i.node_id::text AS node_id,
                       b.properties
                FROM {self.graph_name}.base AS b
                JOIN ids i
                  ON ag_catalog.agtype_access_operator(
                       VARIADIC ARRAY[b.properties, '"entity_id"'::agtype]
                     ) = i.node_id
                ORDER BY i.ord;
            """

            results = await self._query(query, params={"ids": batch})

            for result in results:
                if result["node_id"] and result["properties"]:
                    node_dict = result["properties"]
                    if isinstance(node_dict, str):
                        try:
                            node_dict = json.loads(node_dict)
                        except json.JSONDecodeError:
                            pass
                    node_key = result["node_id"]
                    original_key = lookup.get(node_key, node_key)
                    if original_key in requested:
                        nodes_dict[original_key] = node_dict

        return nodes_dict

    async def node_degrees_batch(
        self, node_ids: list[str], batch_size: int = 500
    ) -> dict[str, int]:
        if not node_ids:
            return {}

        seen: set[str] = set()
        unique_ids: list[str] = []
        lookup: dict[str, str] = {}
        requested: set[str] = set()
        for nid in node_ids:
            if nid not in seen:
                seen.add(nid)
                unique_ids.append(nid)
            requested.add(nid)
            lookup[nid] = nid
            lookup[self._normalize_node_id(nid)] = nid

        out_degrees = {}
        in_degrees = {}

        for i in range(0, len(unique_ids), batch_size):
            batch = unique_ids[i: i + batch_size]

            query = f"""
                    WITH input(v, ord) AS (
                      SELECT v, ord
                      FROM unnest($1::text[]) WITH ORDINALITY AS t(v, ord)
                    ),
                    ids(node_id, ord) AS (
                      SELECT (to_json(v)::text)::agtype AS node_id, ord
                      FROM input
                    ),
                    vids AS (
                      SELECT b.id AS vid, i.node_id, i.ord
                      FROM {self.graph_name}.base AS b
                      JOIN ids i
                        ON ag_catalog.agtype_access_operator(
                             VARIADIC ARRAY[b.properties, '"entity_id"'::agtype]
                           ) = i.node_id
                    ),
                    deg_out AS (
                      SELECT d.start_id AS vid, COUNT(*)::bigint AS out_degree
                      FROM {self.graph_name}."DIRECTED" AS d
                      JOIN vids v ON v.vid = d.start_id
                      GROUP BY d.start_id
                    ),
                    deg_in AS (
                      SELECT d.end_id AS vid, COUNT(*)::bigint AS in_degree
                      FROM {self.graph_name}."DIRECTED" AS d
                      JOIN vids v ON v.vid = d.end_id
                      GROUP BY d.end_id
                    )
                    SELECT v.node_id::text AS node_id,
                           COALESCE(o.out_degree, 0) AS out_degree,
                           COALESCE(n.in_degree, 0)  AS in_degree
                    FROM vids v
                    LEFT JOIN deg_out o ON o.vid = v.vid
                    LEFT JOIN deg_in  n ON n.vid = v.vid
                    ORDER BY v.ord;
                """

            combined_results = await self._query(query, params={"ids": batch})

            for row in combined_results:
                node_id = row["node_id"]
                if not node_id:
                    continue
                original_key = lookup.get(node_id, node_id)
                if original_key in requested:
                    out_degrees[original_key] = int(row.get("out_degree", 0) or 0)
                    in_degrees[original_key] = int(row.get("in_degree", 0) or 0)

        degrees_dict = {}
        for node_id in node_ids:
            degrees_dict[node_id] = out_degrees.get(node_id, 0) + in_degrees.get(node_id, 0)

        return degrees_dict

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        if not edge_pairs:
            return {}

        all_nodes = set()
        for src, tgt in edge_pairs:
            all_nodes.add(src)
            all_nodes.add(tgt)

        node_degrees = await self.node_degrees_batch(list(all_nodes))

        edge_degrees_dict = {}
        for src, tgt in edge_pairs:
            edge_degrees_dict[(src, tgt)] = (
                node_degrees.get(src, 0) + node_degrees.get(tgt, 0)
            )

        return edge_degrees_dict

    async def get_edges_batch(
        self, pairs: list[dict[str, str]], batch_size: int = 500
    ) -> dict[tuple[str, str], dict]:
        if not pairs:
            return {}

        seen = set()
        uniq_pairs: list[dict[str, str]] = []
        for p in pairs:
            s = self._normalize_node_id(p["src"])
            t = self._normalize_node_id(p["tgt"])
            key = (s, t)
            if s and t and key not in seen:
                seen.add(key)
                uniq_pairs.append(p)

        edges_dict: dict[tuple[str, str], dict] = {}

        for i in range(0, len(uniq_pairs), batch_size):
            batch = uniq_pairs[i: i + batch_size]
            pairs_param = [{"src": p["src"], "tgt": p["tgt"]} for p in batch]

            forward_cypher = """
                         UNWIND $pairs AS p
                         WITH p.src AS src_eid, p.tgt AS tgt_eid
                         MATCH (a:base {entity_id: src_eid})
                         MATCH (b:base {entity_id: tgt_eid})
                         MATCH (a)-[r]->(b)
                         RETURN src_eid AS source, tgt_eid AS target, properties(r) AS edge_properties"""
            backward_cypher = """
                         UNWIND $pairs AS p
                         WITH p.src AS src_eid, p.tgt AS tgt_eid
                         MATCH (a:base {entity_id: src_eid})
                         MATCH (b:base {entity_id: tgt_eid})
                         MATCH (a)<-[r]-(b)
                         RETURN src_eid AS source, tgt_eid AS target, properties(r) AS edge_properties"""

            sql_fwd = f"""
            SELECT * FROM cypher({_dollar_quote(self.graph_name)}::name,
                                 {_dollar_quote(forward_cypher)}::cstring,
                                 $1::agtype)
              AS (source text, target text, edge_properties agtype)
            """
            sql_bwd = f"""
            SELECT * FROM cypher({_dollar_quote(self.graph_name)}::name,
                                 {_dollar_quote(backward_cypher)}::cstring,
                                 $1::agtype)
              AS (source text, target text, edge_properties agtype)
            """

            pg_params = {"params": json.dumps({"pairs": pairs_param}, ensure_ascii=False)}

            forward_results = await self._query(sql_fwd, params=pg_params)
            backward_results = await self._query(sql_bwd, params=pg_params)

            for result in forward_results:
                if result["source"] and result["target"] and result["edge_properties"]:
                    edge_props = result["edge_properties"]
                    if isinstance(edge_props, str):
                        try:
                            edge_props = json.loads(edge_props)
                        except json.JSONDecodeError:
                            continue
                    edges_dict[(result["source"], result["target"])] = edge_props

            for result in backward_results:
                if result["source"] and result["target"] and result["edge_properties"]:
                    edge_props = result["edge_properties"]
                    if isinstance(edge_props, str):
                        try:
                            edge_props = json.loads(edge_props)
                        except json.JSONDecodeError:
                            continue
                    edges_dict[(result["source"], result["target"])] = edge_props

        return edges_dict

    async def get_nodes_edges_batch(
        self, node_ids: list[str], batch_size: int = 500
    ) -> dict[str, list[tuple[str, str]]]:
        if not node_ids:
            return {}

        seen = set()
        unique_ids: list[str] = []
        for nid in node_ids:
            if nid and nid not in seen:
                seen.add(nid)
                unique_ids.append(nid)

        edges_norm: dict[str, list[tuple[str, str]]] = {n: [] for n in unique_ids}

        for i in range(0, len(unique_ids), batch_size):
            batch = unique_ids[i: i + batch_size]
            pg_params = {"params": json.dumps({"node_ids": batch}, ensure_ascii=False)}

            outgoing_cypher = """UNWIND $node_ids AS node_id
                         MATCH (n:base {entity_id: node_id})
                         OPTIONAL MATCH (n:base)-[]->(connected:base)
                         RETURN node_id, connected.entity_id AS connected_id"""

            incoming_cypher = """UNWIND $node_ids AS node_id
                         MATCH (n:base {entity_id: node_id})
                         OPTIONAL MATCH (n:base)<-[]-(connected:base)
                         RETURN node_id, connected.entity_id AS connected_id"""

            outgoing_query = f"SELECT * FROM cypher({_dollar_quote(self.graph_name)}::name, {_dollar_quote(outgoing_cypher)}::cstring, $1::agtype) AS (node_id text, connected_id text)"
            incoming_query = f"SELECT * FROM cypher({_dollar_quote(self.graph_name)}::name, {_dollar_quote(incoming_cypher)}::cstring, $1::agtype) AS (node_id text, connected_id text)"

            outgoing_results = await self._query(outgoing_query, params=pg_params)
            incoming_results = await self._query(incoming_query, params=pg_params)

            for result in outgoing_results:
                if result["node_id"] and result["connected_id"]:
                    edges_norm[result["node_id"]].append(
                        (result["node_id"], result["connected_id"])
                    )

            for result in incoming_results:
                if result["node_id"] and result["connected_id"]:
                    edges_norm[result["node_id"]].append(
                        (result["connected_id"], result["node_id"])
                    )

        out: dict[str, list[tuple[str, str]]] = {}
        for orig in node_ids:
            out[orig] = edges_norm.get(orig, [])

        return out
