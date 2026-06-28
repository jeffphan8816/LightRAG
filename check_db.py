"""Quick DB diagnostic — checks table counts, workspaces, and graph vertices."""
import asyncio
import asyncpg

PG_CONFIG = {
    "host": "hayabusa.proxy.rlwy.net",
    "port": 30426,
    "user": "rag",
    "password": "CHANGE_ME_STRONG_PASSWORD",
    "database": "rag",
}

async def check():
    conn = await asyncpg.connect(**PG_CONFIG, timeout=10)

    # Workspaces + counts in vector tables
    for table in [
        "lightrag_vdb_entity_text_embedding_3_small_1536d",
        "lightrag_vdb_relation_text_embedding_3_small_1536d",
        "lightrag_vdb_chunks_text_embedding_3_small_1536d",
    ]:
        rows = await conn.fetch(f"SELECT DISTINCT workspace FROM {table}")
        workspaces = [r["workspace"] for r in rows]
        rows = await conn.fetch(f"SELECT COUNT(*) as cnt FROM {table}")
        print(f"{table}: {rows[0]['cnt']} rows, workspaces={workspaces}")

    # Graph schemas
    rows = await conn.fetch(
        "SELECT nspname FROM pg_namespace WHERE nspname NOT LIKE 'pg_%' AND nspname != 'information_schema'"
    )
    print(f"\nSchemas: {[r['nspname'] for r in rows]}")

    # Graph vertices and edges
    try:
        await conn.execute('SET search_path = ag_catalog, "$user", public')
        rows = await conn.fetch(
            "SELECT * FROM cypher('chunk_entity_relation', $$MATCH (n) RETURN count(n) as cnt$$) AS (cnt agtype)"
        )
        print(f"Graph vertices: {rows[0]['cnt'] if rows else 'N/A'}")
        rows = await conn.fetch(
            "SELECT * FROM cypher('chunk_entity_relation', $$MATCH ()-[r]->() RETURN count(r) as cnt$$) AS (cnt agtype)"
        )
        print(f"Graph edges: {rows[0]['cnt'] if rows else 'N/A'}")
    except Exception as e:
        print(f"Graph query error: {e}")

    # Sample entity names
    rows = await conn.fetch(
        "SELECT entity_name FROM lightrag_vdb_entity_text_embedding_3_small_1536d LIMIT 5"
    )
    print(f"\nSample entities: {[r['entity_name'] for r in rows]}")

    await conn.close()

asyncio.run(check())
