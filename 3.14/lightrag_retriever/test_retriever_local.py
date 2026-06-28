"""Test the retriever locally against the real DB."""
import asyncio
import os
import sys

# Set env vars from secrets.json
import json
with open("deploy/lambda/secrets.json") as f:
    secrets = json.load(f)
for k, v in secrets.items():
    if k not in os.environ:
        os.environ[k] = str(v)

# Also set PG env vars that get_pg_config reads
os.environ["POSTGRES_HOST"] = secrets["POSTGRES_HOST"]
os.environ["POSTGRES_PORT"] = secrets["POSTGRES_PORT"]
os.environ["POSTGRES_USER"] = secrets["POSTGRES_USER"]
os.environ["POSTGRES_PASSWORD"] = secrets["POSTGRES_PASSWORD"]
os.environ["POSTGRES_DATABASE"] = secrets["POSTGRES_DATABASE"]
os.environ["POSTGRES_WORKSPACE"] = "default"
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

sys.path.insert(0, "lightrag_retriever")

from lightrag_retriever import QueryParam, create_postgres_retriever

async def test():
    print("Creating retriever...")
    retriever = await create_postgres_retriever(
        workspace="default",
        llm_model_name="gpt-4o-mini",
        embedding_model_name="text-embedding-3-small",
        embedding_dim=1536,
        cosine_better_than_threshold=0.2,
    )
    print("Retriever created. Running query...")

    param = QueryParam(mode="local", top_k=40)
    result = await retriever.aquery_data("List all see listing_type of Kevadia", param)
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")

asyncio.run(test())
