# Retriever-only LightRAG on AWS Lambda (container image).
# Uses lightrag_retriever exclusively — no lightrag dependency.
# Build context = repo root:  docker build -f deploy/lambda/Dockerfile -t lightrag-retriever .
#
# arm64 is cheaper on Lambda; ensure the ECR repo + Lambda function use the same arch.

FROM public.ecr.aws/lambda/python:3.12

# uv for fast, resolved installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

# lightrag_retriever — standalone retrieval package (includes PG storage + LLM)
COPY lightrag_retriever/pyproject.toml ${LAMBDA_TASK_ROOT}/lightrag_retriever/pyproject.toml
COPY lightrag_retriever/lightrag_retriever/ ${LAMBDA_TASK_ROOT}/lightrag_retriever/lightrag_retriever/
COPY lightrag_retriever/README.md ${LAMBDA_TASK_ROOT}/lightrag_retriever/README.md

WORKDIR ${LAMBDA_TASK_ROOT}

# Install lightrag_retriever with postgres + llm extras (asyncpg, pgvector, openai, tenacity).
# All dependencies pre-baked so no runtime pip install triggers
# (Lambda's filesystem is read-only outside /tmp).
RUN uv pip install --system --no-cache "./lightrag_retriever[all]" boto3

# Pre-bake tiktoken BPE so the first cold query doesn't fetch over the network.
RUN mkdir -p ${TIKTOKEN_CACHE_DIR} \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" || true

COPY deploy/lambda/handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]
