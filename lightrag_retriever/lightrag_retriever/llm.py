"""Self-contained OpenAI LLM and embedding functions for lightrag_retriever.

No dependency on lightrag. Provides:
- openai_complete_if_cache: async chat completion with retry, COT support
- openai_embed: async text embedding with truncation and asymmetric prefix support
- create_openai_async_client: client factory
"""

from __future__ import annotations

import base64
import logging
import os
import warnings
from collections.abc import AsyncIterator
from typing import Any, Union

import numpy as np
import tiktoken
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("lightrag_retriever")

EMBEDDING_USE_BASE64: bool = os.getenv("EMBEDDING_USE_BASE64", "true").lower() in (
    "true",
    "1",
    "yes",
)

_TIKTOKEN_ENCODING_CACHE: dict[str, Any] = {}


def _get_tiktoken_encoding_for_model(model: str) -> Any:
    if model not in _TIKTOKEN_ENCODING_CACHE:
        try:
            _TIKTOKEN_ENCODING_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _TIKTOKEN_ENCODING_CACHE[model] = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENCODING_CACHE[model]


class InvalidResponseError(Exception):
    pass


class TransientBadRequestError(Exception):
    pass


def create_openai_async_client(
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int | None = None,
    client_configs: dict[str, Any] | None = None,
) -> AsyncOpenAI:
    if not api_key:
        api_key = os.environ["OPENAI_API_KEY"]

    default_headers = {
        "User-Agent": "lightrag_retriever/0.1.0",
        "Content-Type": "application/json",
    }

    if client_configs is None:
        client_configs = {}

    merged_configs = {
        **client_configs,
        "default_headers": default_headers,
        "api_key": api_key,
    }

    if base_url is not None:
        merged_configs["base_url"] = base_url
    else:
        merged_configs["base_url"] = os.environ.get(
            "OPENAI_API_BASE", "https://api.openai.com/v1"
        )

    if timeout is not None:
        merged_configs["timeout"] = timeout

    return AsyncOpenAI(**merged_configs)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(InvalidResponseError)
        | retry_if_exception_type(InternalServerError)
        | retry_if_exception_type(TransientBadRequestError)
    ),
)
async def openai_complete_if_cache(
    prompt: str,
    *,
    model: str = "gpt-4o-mini",
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    stream: bool | None = None,
    timeout: int | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> str:
    if history_messages is None:
        history_messages = []

    kwargs.pop("hashing_kv", None)
    kwargs.pop("_priority", None)
    client_configs = kwargs.pop("openai_client_configs", {})

    entity_extraction = kwargs.pop("entity_extraction", False)
    if entity_extraction and kwargs.get("response_format") is None:
        kwargs["response_format"] = {"type": "json_object"}
    if keyword_extraction and kwargs.get("response_format") is None:
        kwargs["response_format"] = {"type": "json_object"}

    if kwargs.get("response_format") is not None:
        enable_cot = False

    openai_async_client = create_openai_async_client(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        client_configs=client_configs,
    )

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    messages = kwargs.pop("messages", messages)

    if stream is not None:
        kwargs["stream"] = stream
    if timeout is not None:
        kwargs["timeout"] = timeout

    try:
        response = await openai_async_client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
    except (APITimeoutError, APIConnectionError, RateLimitError):
        try:
            await openai_async_client.close()
        except Exception:
            pass
        raise
    except BadRequestError as e:
        try:
            await openai_async_client.close()
        except Exception:
            pass
        if "could not parse" in str(e).lower():
            raise TransientBadRequestError(str(e)) from e
        raise
    except Exception:
        try:
            await openai_async_client.close()
        except Exception:
            pass
        raise

    if hasattr(response, "__aiter__"):
        async def inner():
            cot_active = False
            cot_started = False
            initial_content_seen = False

            try:
                async for chunk in response:
                    if not hasattr(chunk, "choices") or not chunk.choices:
                        continue
                    if not hasattr(chunk.choices[0], "delta"):
                        continue

                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    reasoning_content = getattr(delta, "reasoning_content", "")

                    if enable_cot:
                        if content:
                            if not initial_content_seen:
                                initial_content_seen = True
                                if reasoning_content:
                                    cot_active = False
                                    cot_started = False
                            if cot_active:
                                yield "</think>"
                                cot_active = False
                            yield content
                        elif reasoning_content:
                            if not initial_content_seen and not cot_started:
                                if not cot_active:
                                    yield "<think>"
                                    cot_active = True
                                    cot_started = True
                            if cot_active:
                                yield reasoning_content
                    else:
                        if content:
                            yield content

                if enable_cot and cot_active:
                    yield "</think>"
            finally:
                try:
                    await response.aclose()
                except Exception:
                    pass
                try:
                    await openai_async_client.close()
                except Exception:
                    pass

        return inner()

    else:
        try:
            if (
                not response
                or not response.choices
                or not hasattr(response.choices[0], "message")
            ):
                raise InvalidResponseError("Invalid response from OpenAI API")

            message = response.choices[0].message

            if hasattr(message, "parsed") and message.parsed is not None:
                final_content = message.parsed.model_dump_json()
            else:
                content = getattr(message, "content", None)
                reasoning_content = getattr(message, "reasoning_content", "")

                final_content = ""
                if enable_cot:
                    if reasoning_content and reasoning_content.strip():
                        if not content or content.strip() == "":
                            final_content = content or ""
                            final_content = f"<think>{reasoning_content}</think>{final_content}"
                        else:
                            final_content = content
                    else:
                        final_content = content or ""
                else:
                    final_content = content or ""

                if not final_content or final_content.strip() == "":
                    raise InvalidResponseError("Received empty content from OpenAI API")

            return final_content
        finally:
            try:
                await openai_async_client.close()
            except Exception:
                pass


async def openai_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str | None = None,
    api_key: str | None = None,
    embedding_dim: int | None = None,
    max_token_size: int | None = None,
    client_configs: dict[str, Any] | None = None,
    context: str = "document",
    query_prefix: str | None = None,
    document_prefix: str | None = None,
    **kwargs: Any,
) -> np.ndarray:
    if context == "query" and query_prefix:
        texts = [query_prefix + text for text in texts]
    elif context == "document" and document_prefix:
        texts = [document_prefix + text for text in texts]

    if max_token_size is not None and max_token_size > 0:
        encoding = _get_tiktoken_encoding_for_model(model)
        truncated_texts = []
        for text in texts:
            if not text:
                truncated_texts.append(text)
                continue
            tokens = encoding.encode(text)
            if len(tokens) > max_token_size:
                truncated_texts.append(encoding.decode(tokens[:max_token_size]))
            else:
                truncated_texts.append(text)
        texts = truncated_texts

    openai_async_client = create_openai_async_client(
        api_key=api_key,
        base_url=base_url,
        client_configs=client_configs,
    )

    async with openai_async_client:
        api_params: dict[str, Any] = {
            "model": model,
            "input": texts,
        }
        api_params["encoding_format"] = "base64" if EMBEDDING_USE_BASE64 else "float"
        if embedding_dim is not None:
            api_params["dimensions"] = embedding_dim

        response = await openai_async_client.embeddings.create(**api_params)

        return np.array(
            [
                np.array(dp.embedding, dtype=np.float32)
                if isinstance(dp.embedding, list)
                else np.frombuffer(base64.b64decode(dp.embedding), dtype=np.float32)
                for dp in response.data
            ]
        )
