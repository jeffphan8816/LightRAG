"""S3 Vectors vector storage for lightrag_retriever (Preview — experimental).

Standalone implementation that does NOT import from the ``lightrag`` package.
Uses AWS S3 Vectors (s3vectors service) for vector storage with metadata
filtering for workspace isolation.

Requires ``boto3``:  pip install boto3

Configuration (environment variables):
    S3_VECTORS_BUCKET   — vector bucket name (required)
    S3_VECTORS_REGION   — AWS region (optional, uses boto3 default if unset)
    S3_VECTORS_WORKSPACE — override workspace for all instances (optional)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lightrag_retriever.base import EmbeddingFunc, VectorStorage
from lightrag_retriever.constants import DEFAULT_QUERY_PRIORITY

logger = logging.getLogger("lightrag_retriever")

try:
    import boto3  # type: ignore
    from botocore.exceptions import ClientError  # type: ignore

    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False

# S3 Vectors API batch limits
_S3_VECTORS_PUT_MAX_BATCH = 500
_S3_VECTORS_DELETE_MAX_BATCH = 500
_S3_VECTORS_GET_MAX_BATCH = 100
_S3_VECTORS_LIST_PAGE_SIZE = 1000


def _compute_mdhash_id(text: str, prefix: str = "") -> str:
    """Compute an MD5 hash ID with optional prefix."""
    return prefix + hashlib.md5(text.encode("utf-8")).hexdigest()


def _generate_collection_suffix(embedding_func: EmbeddingFunc) -> str:
    """Generate a model-based suffix from embedding_func metadata."""
    model_name = getattr(embedding_func, "model_name", "") or ""
    if not model_name:
        return ""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", model_name.lower())
    dim = getattr(embedding_func, "embedding_dim", 1536)
    return f"{safe}_{dim}d"


def _resolve_workspace(workspace: str) -> str:
    """Resolve effective workspace from env or parameter."""
    env_ws = os.environ.get("S3_VECTORS_WORKSPACE")
    if env_ws and env_ws.strip():
        logger.info(
            f"Using S3_VECTORS_WORKSPACE: '{env_ws.strip()}' (overriding '{workspace}')"
        )
        return env_ws.strip()
    return workspace


@dataclass
class _PendingS3VectorDoc:
    """Buffered vector upsert waiting for embedding and flush."""

    source: dict[str, Any]
    content: str
    vector: list[float] | None = None


class S3VectorsStorage(VectorStorage):
    """Vector storage using AWS S3 Vectors (Preview — experimental, no SLA).

    Uses deferred embedding and server-backed flush.  Workspace isolation is
    achieved via a ``workspace`` filterable metadata field per vector.  Each
    namespace gets its own index within a shared vector bucket.

    A local metadata cache (``_relation_index``) maps entity_name to the set
    of relation vector keys where that entity is src_id or tgt_id, enabling
    O(1) ``delete_entity_relation`` lookups.

    Requires:
        - boto3 (pip install boto3)
        - S3_VECTORS_BUCKET env var set to a vector bucket name
        - AWS credentials configured via standard boto3 credential chain
    """

    namespace: str
    workspace: str = ""
    global_config: dict[str, Any] = field(default_factory=dict)
    embedding_func: EmbeddingFunc = field(default=None)
    cosine_better_than_threshold: float = 0.2
    meta_fields: set[str] = field(default_factory=set)

    def __post_init__(self):
        if self.embedding_func is None:
            raise ValueError("embedding_func is required for S3VectorsStorage")

        self.workspace = _resolve_workspace(self.workspace or "")

        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is not None:
            self.cosine_better_than_threshold = cosine_threshold

        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)

        suffix = _generate_collection_suffix(self.embedding_func)
        if suffix:
            self._index_name = f"{self.namespace}_{suffix}"
        else:
            self._index_name = self.namespace
            logger.warning(
                f"S3 Vectors index '{self._index_name}' missing model suffix."
            )

        bucket = os.environ.get("S3_VECTORS_BUCKET")
        if not bucket:
            raise ValueError(
                "S3_VECTORS_BUCKET environment variable is required for S3VectorsStorage"
            )
        self._vector_bucket_name = bucket
        self._region = os.environ.get("S3_VECTORS_REGION")

        self._pending_vector_docs: dict[str, _PendingS3VectorDoc] = {}
        self._pending_vector_deletes: set[str] = set()
        self._relation_index: dict[str, set[str]] = {}
        self._flush_lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._client = None
        self._initialized = False

    async def initialize(self):
        """Initialize boto3 client, create vector bucket and index if needed."""
        async with self._init_lock:
            if self._initialized:
                return

            try:
                client_kwargs: dict[str, Any] = {}
                if self._region:
                    client_kwargs["region_name"] = self._region
                self._client = await asyncio.to_thread(
                    lambda: boto3.client("s3vectors", **client_kwargs)
                )

                await self._create_vector_bucket_if_not_exists()
                await self._create_index_if_not_exists()
                await self._load_relation_index()

                self._initialized = True
                logger.info(
                    f"S3 Vectors storage initialized: "
                    f"bucket={self._vector_bucket_name}, index={self._index_name}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize S3 Vectors storage: {e}")
                raise

    async def _create_vector_bucket_if_not_exists(self):
        def _create():
            try:
                self._client.get_vector_bucket(
                    vectorBucketName=self._vector_bucket_name
                )
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("NotFoundException", "ResourceNotFoundException"):
                    self._client.create_vector_bucket(
                        bucketName=self._vector_bucket_name
                    )
                    logger.info(
                        f"Created S3 Vectors bucket: {self._vector_bucket_name}"
                    )
                else:
                    raise

        await asyncio.to_thread(_create)

    async def _create_index_if_not_exists(self):
        def _create():
            try:
                self._client.get_index(
                    vectorBucketName=self._vector_bucket_name,
                    indexName=self._index_name,
                )
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("NotFoundException", "ResourceNotFoundException"):
                    self._client.create_index(
                        vectorBucketName=self._vector_bucket_name,
                        indexName=self._index_name,
                        dataType="float32",
                        dimension=self.embedding_func.embedding_dim,
                        distanceMetric="cosine",
                        metadataConfiguration={
                            "nonFilterableMetadataKeys": ["content"]
                        },
                    )
                    logger.info(
                        f"Created S3 Vectors index: {self._index_name} "
                        f"(dim={self.embedding_func.embedding_dim})"
                    )
                else:
                    raise

        await asyncio.to_thread(_create)

    async def _load_relation_index(self):
        if not self.namespace.endswith("relationships"):
            return

        def _load():
            relation_index: dict[str, set[str]] = {}
            next_token = None
            while True:
                list_kwargs: dict[str, Any] = {
                    "vectorBucketName": self._vector_bucket_name,
                    "indexName": self._index_name,
                    "returnMetadata": True,
                    "maxResults": _S3_VECTORS_LIST_PAGE_SIZE,
                    "filter": {"workspace": self.workspace},
                }
                if next_token:
                    list_kwargs["nextToken"] = next_token

                response = self._client.list_vectors(**list_kwargs)
                for vec in response.get("vectors", []):
                    vec_id = vec.get("id")
                    metadata = vec.get("metadata", {})
                    src_id = metadata.get("src_id")
                    tgt_id = metadata.get("tgt_id")
                    if src_id:
                        relation_index.setdefault(src_id, set()).add(vec_id)
                    if tgt_id:
                        relation_index.setdefault(tgt_id, set()).add(vec_id)

                next_token = response.get("nextToken")
                if not next_token:
                    break
            return relation_index

        self._relation_index = await asyncio.to_thread(_load)
        if self._relation_index:
            logger.info(
                f"Loaded {len(self._relation_index)} entities into relation index cache"
            )

    # ------------------------------------------------------------------
    # Write path (buffered + deferred embedding)
    # ------------------------------------------------------------------

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Buffer vector docs for embedding and batched flush."""
        if not data:
            return

        current_time = int(time.time())
        pending_docs: list[tuple[str, _PendingS3VectorDoc]] = []
        for k, v in data.items():
            content = v["content"]
            source = {
                "created_at": current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            pending_docs.append(
                (k, _PendingS3VectorDoc(source=source, content=content))
            )

        async with self._flush_lock:
            for doc_id, pdoc in pending_docs:
                self._pending_vector_deletes.discard(doc_id)
                self._pending_vector_docs[doc_id] = pdoc

    async def _flush_pending_vector_ops(self) -> None:
        """Flush buffered vector upserts and deletes to S3 Vectors."""
        async with self._flush_lock:
            if not self._pending_vector_docs and not self._pending_vector_deletes:
                return
            if self._client is None:
                return

            pending_docs = self._pending_vector_docs
            pending_deletes = self._pending_vector_deletes

            docs_to_embed = [
                (doc_id, pdoc)
                for doc_id, pdoc in pending_docs.items()
                if pdoc.vector is None
            ]
            if docs_to_embed:
                contents = [pdoc.content for _, pdoc in docs_to_embed]
                batches = [
                    contents[i : i + self._max_batch_size]
                    for i in range(0, len(contents), self._max_batch_size)
                ]
                logger.info(
                    f"{self.namespace} flush: embedding "
                    f"{len(docs_to_embed)} vectors in {len(batches)} batch(es)"
                )
                try:
                    embeddings_list = await asyncio.gather(
                        *[
                            self.embedding_func(batch, context="document")
                            for batch in batches
                        ]
                    )
                except Exception as e:
                    logger.error(f"Error embedding pending vector ops: {e}")
                    raise
                embeddings = np.concatenate(embeddings_list)
                if len(embeddings) != len(docs_to_embed):
                    raise RuntimeError(
                        f"Embedding count mismatch: expected {len(docs_to_embed)}, "
                        f"got {len(embeddings)}"
                    )
                for (doc_id, pdoc), embedding in zip(docs_to_embed, embeddings):
                    pdoc.vector = embedding.tolist()

            if pending_deletes:
                delete_ids = list(pending_deletes)
                for i in range(0, len(delete_ids), _S3_VECTORS_DELETE_MAX_BATCH):
                    chunk = delete_ids[i : i + _S3_VECTORS_DELETE_MAX_BATCH]
                    try:
                        await asyncio.to_thread(
                            self._client.delete_vectors,
                            vectorBucketName=self._vector_bucket_name,
                            indexName=self._index_name,
                            ids=chunk,
                        )
                    except Exception as e:
                        logger.error(f"Error deleting vectors: {e}")
                        raise

            committed_ids: list[str] = []
            vectors_to_put: list[dict[str, Any]] = []
            for doc_id, pdoc in pending_docs.items():
                if pdoc.vector is None:
                    continue
                committed_ids.append(doc_id)
                metadata: dict[str, Any] = {"workspace": self.workspace}
                metadata.update(pdoc.source)
                metadata["content"] = pdoc.content
                vectors_to_put.append(
                    {
                        "id": doc_id,
                        "vector": {"float32": pdoc.vector},
                        "metadata": metadata,
                    }
                )

            if vectors_to_put:
                for i in range(0, len(vectors_to_put), _S3_VECTORS_PUT_MAX_BATCH):
                    chunk = vectors_to_put[i : i + _S3_VECTORS_PUT_MAX_BATCH]
                    try:
                        await asyncio.to_thread(
                            self._client.put_vectors,
                            vectorBucketName=self._vector_bucket_name,
                            indexName=self._index_name,
                            vectors=chunk,
                        )
                    except Exception as e:
                        logger.error(f"Error putting vectors: {e}")
                        raise

            if self.namespace.endswith("relationships"):
                for doc_id in committed_ids:
                    pdoc = pending_docs.get(doc_id)
                    if pdoc is not None:
                        src_id = pdoc.source.get("src_id")
                        tgt_id = pdoc.source.get("tgt_id")
                        if src_id:
                            self._relation_index.setdefault(src_id, set()).add(doc_id)
                        if tgt_id:
                            self._relation_index.setdefault(tgt_id, set()).add(doc_id)

            for doc_id in committed_ids:
                pending_docs.pop(doc_id, None)
            pending_deletes.clear()

    async def index_done_callback(self) -> None:
        """Flush pending vector ops to S3 Vectors."""
        await self._flush_pending_vector_ops()

    async def drop_pending_index_ops(self) -> None:
        """Discard buffered upserts/deletes."""
        async with self._flush_lock:
            self._pending_vector_docs.clear()
            self._pending_vector_deletes.clear()

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """Similarity search with cosine distance → similarity conversion."""
        if self._client is None:
            return []

        if query_embedding is not None:
            query_vector = (
                query_embedding.tolist()
                if hasattr(query_embedding, "tolist")
                else list(query_embedding)
            )
        else:
            embedding = await self.embedding_func(
                [query], context="query", _priority=DEFAULT_QUERY_PRIORITY
            )
            query_vector = embedding[0].tolist()

        def _query():
            return self._client.query_vectors(
                vectorBucketName=self._vector_bucket_name,
                indexName=self._index_name,
                queryVector={"float32": query_vector},
                topK=top_k,
                filter={"workspace": self.workspace},
                returnMetadata=True,
                returnDistance=True,
            )

        try:
            response = await asyncio.to_thread(_query)
        except Exception as e:
            logger.error(f"Error querying S3 Vectors: {e}")
            return []

        results = []
        vectors = response.get("vectors", [])
        for vec in vectors:
            distance = vec.get("distance", 1.0)
            similarity = 1.0 - distance
            if similarity >= self.cosine_better_than_threshold:
                doc = dict(vec.get("metadata", {}))
                doc["id"] = vec.get("id")
                doc["distance"] = similarity
                results.append(doc)

        logger.info(
            f"S3 Vectors query on {self._index_name}: "
            f"top_k={top_k}, threshold={self.cosine_better_than_threshold}, "
            f"total_hits={len(vectors)}, passed_filter={len(results)}"
        )
        return results

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get a vector document by ID, with read-your-writes against the buffer."""
        async with self._flush_lock:
            if id in self._pending_vector_deletes:
                return None
            pending = self._pending_vector_docs.get(id)
            if pending is not None:
                doc = dict(pending.source)
                doc["id"] = id
                return doc

        if self._client is None:
            return None

        def _get():
            return self._client.get_vectors(
                vectorBucketName=self._vector_bucket_name,
                indexName=self._index_name,
                ids=[id],
                returnMetadata=True,
                returnData=False,
            )

        try:
            response = await asyncio.to_thread(_get)
        except Exception as e:
            logger.error(f"Error getting vector {id}: {e}")
            return None

        vectors = response.get("vectors", [])
        if not vectors:
            return None
        vec = vectors[0]
        doc = dict(vec.get("metadata", {}))
        doc["id"] = vec.get("id")
        return doc

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector documents by IDs (read-your-writes), preserving order."""
        if not ids:
            return []

        buffered: dict[str, dict[str, Any] | None] = {}
        remaining: list[str] = []
        async with self._flush_lock:
            for doc_id in ids:
                if doc_id in self._pending_vector_deletes:
                    buffered[doc_id] = None
                    continue
                pending = self._pending_vector_docs.get(doc_id)
                if pending is not None:
                    doc = dict(pending.source)
                    doc["id"] = doc_id
                    buffered[doc_id] = doc
                    continue
                remaining.append(doc_id)

        doc_map: dict[str, dict[str, Any]] = {}
        if remaining and self._client is not None:
            for i in range(0, len(remaining), _S3_VECTORS_GET_MAX_BATCH):
                chunk = remaining[i : i + _S3_VECTORS_GET_MAX_BATCH]

                def _get(chunk=chunk):
                    return self._client.get_vectors(
                        vectorBucketName=self._vector_bucket_name,
                        indexName=self._index_name,
                        ids=chunk,
                        returnMetadata=True,
                        returnData=False,
                    )

                try:
                    response = await asyncio.to_thread(_get)
                    for vec in response.get("vectors", []):
                        doc = dict(vec.get("metadata", {}))
                        doc["id"] = vec.get("id")
                        doc_map[vec.get("id")] = doc
                except Exception as e:
                    logger.error(f"Error getting vectors by ids: {e}")

        return [
            buffered[doc_id] if doc_id in buffered else doc_map.get(doc_id)
            for doc_id in ids
        ]

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vector embeddings for given IDs, with read-your-writes.

        Pending docs with ``vector is None`` trigger a lazy embed inside the
        lock; the resulting vector is cached on the buffered doc.
        """
        if not ids:
            return {}

        result: dict[str, list[float]] = {}
        remaining: list[str] = []
        async with self._flush_lock:
            docs_to_embed: list[tuple[str, _PendingS3VectorDoc]] = []
            for doc_id in ids:
                if doc_id in self._pending_vector_deletes:
                    continue
                pending = self._pending_vector_docs.get(doc_id)
                if pending is not None:
                    if pending.vector is None:
                        docs_to_embed.append((doc_id, pending))
                    else:
                        result[doc_id] = pending.vector
                    continue
                remaining.append(doc_id)

            if docs_to_embed:
                contents = [pdoc.content for _, pdoc in docs_to_embed]
                batches = [
                    contents[i : i + self._max_batch_size]
                    for i in range(0, len(contents), self._max_batch_size)
                ]
                try:
                    embeddings_list = await asyncio.gather(
                        *[
                            self.embedding_func(batch, context="document")
                            for batch in batches
                        ]
                    )
                except Exception as e:
                    logger.error(f"Error lazily embedding pending vectors: {e}")
                    raise
                embeddings = np.concatenate(embeddings_list)
                if len(embeddings) != len(docs_to_embed):
                    raise RuntimeError(
                        f"Embedding count mismatch: expected {len(docs_to_embed)}, "
                        f"got {len(embeddings)}"
                    )
                for (doc_id, pdoc), embedding in zip(docs_to_embed, embeddings):
                    pdoc.vector = embedding.tolist()
                    result[doc_id] = pdoc.vector

        if not remaining or self._client is None:
            return result

        for i in range(0, len(remaining), _S3_VECTORS_GET_MAX_BATCH):
            chunk = remaining[i : i + _S3_VECTORS_GET_MAX_BATCH]

            def _get(chunk=chunk):
                return self._client.get_vectors(
                    vectorBucketName=self._vector_bucket_name,
                    indexName=self._index_name,
                    ids=chunk,
                    returnMetadata=False,
                    returnData=True,
                )

            try:
                response = await asyncio.to_thread(_get)
                for vec in response.get("vectors", []):
                    vec_data = vec.get("vector", {})
                    float32_data = vec_data.get("float32")
                    if float32_data is not None:
                        result[vec.get("id")] = list(float32_data)
            except Exception as e:
                logger.error(f"Error retrieving vectors by IDs: {e}")

        return result

    # ------------------------------------------------------------------
    # Delete path
    # ------------------------------------------------------------------

    async def delete(self, ids: list[str]) -> None:
        """Buffer vector deletes for batched flush."""
        if not ids:
            return
        if isinstance(ids, set):
            ids = list(ids)
        async with self._flush_lock:
            for doc_id in ids:
                self._pending_vector_docs.pop(doc_id, None)
                self._pending_vector_deletes.add(doc_id)

    async def delete_entity(self, entity_name: str) -> None:
        """Buffer an entity vector delete by computing its hash ID."""
        entity_id = _compute_mdhash_id(entity_name, prefix="ent-")
        async with self._flush_lock:
            self._pending_vector_docs.pop(entity_id, None)
            self._pending_vector_deletes.add(entity_id)

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relation vectors where entity appears as src or tgt.

        Uses the local ``_relation_index`` cache for O(1) lookup.
        """

        def _prune_pending() -> None:
            for doc_id in [
                k
                for k, v in self._pending_vector_docs.items()
                if v.source.get("src_id") == entity_name
                or v.source.get("tgt_id") == entity_name
            ]:
                self._pending_vector_docs.pop(doc_id, None)

        async with self._flush_lock:
            relation_keys = self._relation_index.pop(entity_name, set())

            if relation_keys and self._client is not None:
                keys_list = list(relation_keys)
                deleted_keys: set[str] = set()
                for i in range(0, len(keys_list), _S3_VECTORS_DELETE_MAX_BATCH):
                    chunk = keys_list[i : i + _S3_VECTORS_DELETE_MAX_BATCH]
                    try:
                        await asyncio.to_thread(
                            self._client.delete_vectors,
                            vectorBucketName=self._vector_bucket_name,
                            indexName=self._index_name,
                            ids=chunk,
                        )
                        deleted_keys.update(chunk)
                    except Exception as e:
                        self._relation_index.setdefault(entity_name, set()).update(
                            k for k in relation_keys if k not in deleted_keys
                        )
                        logger.error(
                            f"Error deleting relations for {entity_name}: {e}"
                        )
                        raise

            _prune_pending()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def finalize(self):
        """Flush pending writes and close the boto3 client."""
        flush_error: Exception | None = None
        try:
            await self._flush_pending_vector_ops()
        except Exception as e:
            flush_error = e

        if self._client is not None:
            try:
                await asyncio.to_thread(self._client.close)
            except Exception:
                pass
            self._client = None

        async with self._flush_lock:
            pending_docs = len(self._pending_vector_docs)
            pending_deletes = len(self._pending_vector_deletes)

        if flush_error is not None:
            raise RuntimeError(
                f"S3VectorsStorage.finalize() flush raised; "
                f"{pending_docs} pending upserts and {pending_deletes} pending "
                f"deletes were left buffered (data lost)"
            ) from flush_error
        if pending_docs or pending_deletes:
            raise RuntimeError(
                f"S3VectorsStorage.finalize() left {pending_docs} pending upserts "
                f"and {pending_deletes} pending deletes buffered"
            )

    async def drop(self) -> dict[str, str]:
        """Delete and recreate the vector index, discarding pending buffers."""
        async with self._flush_lock:
            self._pending_vector_docs.clear()
            self._pending_vector_deletes.clear()
            self._relation_index.clear()

            try:
                if self._client is not None:
                    def _delete_index():
                        try:
                            self._client.delete_index(
                                vectorBucketName=self._vector_bucket_name,
                                indexName=self._index_name,
                            )
                        except ClientError as e:
                            code = e.response.get("Error", {}).get("Code", "")
                            if code not in ("NotFoundException", "ResourceNotFoundException"):
                                raise

                    await asyncio.to_thread(_delete_index)

                    def _create_index():
                        self._client.create_index(
                            vectorBucketName=self._vector_bucket_name,
                            indexName=self._index_name,
                            dataType="float32",
                            dimension=self.embedding_func.embedding_dim,
                            distanceMetric="cosine",
                            metadataConfiguration={
                                "nonFilterableMetadataKeys": ["content"]
                            },
                        )

                    await asyncio.to_thread(_create_index)

                return {"status": "success", "message": "index dropped and recreated"}
            except Exception as e:
                logger.error(f"Error dropping S3 Vectors index: {e}")
                return {"status": "error", "message": str(e)}
