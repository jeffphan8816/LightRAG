"""AWS Lambda retriever-only entrypoint using lightrag_retriever.

Returns structured retrieval context (entities, relationships, chunks) from the
shared Railway Postgres knowledge graph. No answer generation.

Uses lightrag_retriever.create_postgres_retriever for the full retrieval
pipeline with self-contained PG storage backends and OpenAI LLM/embedding.
No imports from lightrag.

Reads all connection/model config from environment variables (set on the Lambda,
sourced from Secrets Manager). Never bake secrets into the image.
"""

import asyncio
import json
import os

import boto3

from lightrag_retriever import QueryParam, create_postgres_retriever

_secrets_loaded = False


def _load_secrets() -> None:
    """Load secrets from AWS Secrets Manager into os.environ (idempotent)."""
    global _secrets_loaded
    if _secrets_loaded:
        return
    secret_name = os.getenv("SECRET_NAME", "lightrag/retriever")
    region = os.getenv("AWS_REGION", "us-east-1")
    try:
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        secrets = json.loads(response["SecretString"])
        for key, value in secrets.items():
            if key not in os.environ:
                os.environ[key] = str(value)
    except Exception as exc:
        print(f"Failed to load secrets from {secret_name}: {exc}")
    _secrets_loaded = True


# Copy pre-baked tiktoken cache from read-only /opt to writable /tmp
# (Lambda's /opt is read-only; tiktoken needs to write .tmp files at runtime)
import shutil
_opt_cache = "/opt/tiktoken"
_tmp_cache = "/tmp/tiktoken"
if os.path.isdir(_opt_cache) and not os.path.isdir(_tmp_cache):
    shutil.copytree(_opt_cache, _tmp_cache)
os.environ["TIKTOKEN_CACHE_DIR"] = _tmp_cache

# Reused across warm invocations. Initialized once at first request.
_retriever = None
_init_lock = asyncio.Lock()
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


async def _get_retriever():
    """Build a lightrag_retriever.Retriever backed by self-contained PG storages."""
    global _retriever
    if _retriever is not None:
        return _retriever
    async with _init_lock:
        if _retriever is not None:
            return _retriever

        _load_secrets()
        _retriever = await create_postgres_retriever(
            workspace=os.getenv("POSTGRES_WORKSPACE", "default"),
            llm_model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1536")),
            embedding_max_token_size=int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "8192")),
            cosine_better_than_threshold=float(
                os.getenv("COSINE_BETTER_THAN_THRESHOLD", "0.2")
            ),
            language=os.getenv("SUMMARY_LANGUAGE", "English"),
            enable_llm_cache=os.getenv("ENABLE_LLM_CACHE", "false").lower() == "true",
            enable_content_headings=False,
        )
        return _retriever


async def _run(query: str, mode: str, top_k: int, hl_keywords, ll_keywords) -> dict:
    retriever = await _get_retriever()
    param = QueryParam(mode=mode, top_k=top_k, enable_rerank=False)
    if hl_keywords:
        param.hl_keywords = hl_keywords
    if ll_keywords:
        param.ll_keywords = ll_keywords
    return await retriever.aquery_data(query, param)


def handler(event, context):
    # Accept direct invoke (dict) or API Gateway/Function URL (body string).
    payload = event
    if isinstance(event, dict) and "body" in event and "query" not in event:
        body = event["body"]
        payload = json.loads(body) if isinstance(body, str) else body
    elif isinstance(event, str):
        payload = json.loads(event)

    query = payload.get("query")
    if not query:
        return {"statusCode": 400, "body": json.dumps({"error": "missing 'query'"})}

    mode = payload.get("mode", "mix")
    top_k = int(payload.get("top_k", 20))
    hl_keywords = payload.get("hl_keywords")
    ll_keywords = payload.get("ll_keywords")

    try:
        result = _loop.run_until_complete(
            _run(query, mode, top_k, hl_keywords, ll_keywords)
        )
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result, default=str),
        }
    except Exception as exc:  # surface error to caller, keep container alive
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(exc)}),
        }
