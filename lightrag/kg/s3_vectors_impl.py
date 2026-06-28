"""
S3 Vectors Storage Implementation for LightRAG (Preview)

This module provides an S3 Vectors-based vector storage backend for LightRAG,
using AWS S3 Vectors (s3vectors service) for vector storage with metadata
filtering for workspace isolation.

Requirements:
    - boto3 (pip install boto3)
    - AWS S3 Vectors service access (preview)

Note: This backend is marked as experimental/preview. The S3 Vectors API may
change before GA. Not for production use without SLA guarantees.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, final
import configparser

import numpy as np

from ..base import BaseVectorStorage
from ..utils import (
    logger,
    compute_mdhash_id,
    _cooperative_yield,
    validate_workspace,
)
from ..constants import DEFAULT_QUERY_PRIORITY
from ..kg.shared_storage import get_data_init_lock, get_namespace_lock

import pipmaster as pm

if not pm.is_installed("boto3"):
    pm.install("boto3")

import boto3  # type: ignore
from botocore.exceptions import ClientError  # type: ignore

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

# S3 Vectors API batch limits
S3_VECTORS_PUT_MAX_BATCH = 500
S3_VECTORS_DELETE_MAX_BATCH = 500
S3_VECTORS_GET_MAX_BATCH = 100
S3_VECTORS_LIST_PAGE_SIZE = 1000


@dataclass
class _PendingS3VectorDoc:
    """Buffered vector upsert waiting for embedding and flush."""

    source: dict[str, Any]
    content: str
    vector: list[float] | None = None


def _get_s3_vectors_env(key: str, fallback: str | None = None) -> str | None:
    """Get S3 Vectors config from env or config.ini."""
    cfg_key = key.replace("S3_VECTORS_", "").lower()
    val = os.environ.get(key)
    if val is not None:
        return val
    return config.get("s3vectors", cfg_key, fallback=fallback)


def _resolve_workspace(workspace: str) -> str:
    """Resolve effective workspace from env or parameter."""
    s3vectors_workspace = os.environ.get("S3_VECTORS_WORKSPACE")
    if s3vectors_workspace and s3vectors_workspace.strip():
        effective = s3vectors_workspace.strip()
        logger.info(
            f"Using S3_VECTORS_WORKSPACE: '{effective}' (overriding '{workspace}')"
        )
        return effective
    return workspace


@final
@dataclass
class S3VectorsStorage(BaseVectorStorage):
    """Vector storage using AWS S3 Vectors (Preview — experimental, no SLA).

    Uses deferred embedding and server-backed flush, following the pattern
    established by OpenSearchVectorDBStorage and MilvusVectorDBStorage.
    Workspace isolation is achieved via metadata filtering on a 'workspace'
    filterable field per vector, not separate indexes. Each namespace gets
    its own index within a shared vector bucket.

    A local metadata cache (``_relation_index``) maps entity_name to the set
    of relation vector keys where that entity is src_id or tgt_id, enabling
    O(1) ``delete_entity_relation`` lookups instead of O(n) ListVectors scans.

    Requirements:
        - boto3 (pip install boto3)
        - S3_VECTORS_BUCKET env var set to a vector bucket name
        - AWS credentials configured via standard boto3 credential chain
    """

    def __init__(
        self, namespace, global_config, embedding_func, workspace=None, meta_fields=None
    ):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
            meta_fields=meta_fields or set(),
        )
        self.__post_init__()

    def __post_init__(self):
        validate_workspace(self.workspace)
        self._validate_embedding_func()

        # Resolve workspace (env override)
        self.workspace = _resolve_workspace(self.workspace)

        # Read cosine threshold
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        # Embedding batch size
        self._max_batch_size = self.global_config["embedding_batch_num"]

        # Generate model suffix and build index name
        self.model_suffix = self._generate_collection_suffix()
        if self.model_suffix:
            self._index_name = f"{self.namespace}_{self.model_suffix}"
        else:
            self._index_name = self.namespace
            logger.warning(
                f"[{self.workspace}] S3 Vectors index '{self._index_name}' missing model suffix. "
                f"Please add model_name to embedding_func for proper model-based data isolation."
            )

        # Vector bucket from env (required)
        bucket = _get_s3_vectors_env("S3_VECTORS_BUCKET")
        if not bucket:
            raise ValueError(
                "S3_VECTORS_BUCKET environment variable is required for S3VectorsStorage"
            )
        self._vector_bucket_name = bucket

        # AWS region
        self._region = _get_s3_vectors_env("S3_VECTORS_REGION")

        # Pending buffers
        self._pending_vector_docs: dict[str, _PendingS3VectorDoc] = {}
        self._pending_vector_deletes: set[str] = set()

        # Local metadata cache for O(1) delete_entity_relation lookups
        self._relation_index: dict[str, set[str]] = {}

        # Lock and client initialized in initialize()
        self._flush_lock = None
        self._client = None
        self._initialized = False

    async def initialize(self):
        """Initialize boto3 client, create vector bucket and index if needed."""
        async with get_data_init_lock():
            if self._initialized:
                return

            try:
                # Create boto3 client
                def _create_client():
                    client_kwargs: dict[str, Any] = {}
                    if self._region:
                        client_kwargs["region_name"] = self._region
                    return boto3.client("s3vectors", **client_kwargs)

                self._client = await asyncio.to_thread(_create_client)

                # Create vector bucket if not exists
                await self._create_vector_bucket_if_not_exists()

                # Create index if not exists
                await self._create_index_if_not_exists()

                # Load relation metadata cache for relationship namespaces
                await self._load_relation_index()

                self._initialized = True
                logger.info(
                    f"[{self.workspace}] S3 Vectors storage initialized: "
                    f"bucket={self._vector_bucket_name}, index={self._index_name}"
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Failed to initialize S3 Vectors storage: {e}"
                )
                raise

        if self._flush_lock is None:
            self._flush_lock = get_namespace_lock(
                self.namespace, workspace=self.workspace
            )

    async def _create_vector_bucket_if_not_exists(self):
        """Create the S3 Vectors vector bucket if it doesn't exist."""

        def _create():
            try:
                self._client.get_vector_bucket(
                    vectorBucketName=self._vector_bucket_name
                )
                logger.debug(
                    f"[{self.workspace}] S3 Vectors bucket exists: {self._vector_bucket_name}"
                )
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("NotFoundException", "ResourceNotFoundException"):
                    self._client.create_vector_bucket(
                        bucketName=self._vector_bucket_name
                    )
                    logger.info(
                        f"[{self.workspace}] Created S3 Vectors bucket: {self._vector_bucket_name}"
                    )
                else:
                    raise

        await asyncio.to_thread(_create)

    async def _create_index_if_not_exists(self):
        """Create the S3 Vectors index if it doesn't exist."""

        def _create():
            try:
                self._client.get_index(
                    vectorBucketName=self._vector_bucket_name,
                    indexName=self._index_name,
                )
                logger.debug(
                    f"[{self.workspace}] S3 Vectors index exists: {self._index_name}"
                )
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("NotFoundException", "ResourceNotFoundException"):
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
                        f"[{self.workspace}] Created S3 Vectors index: {self._index_name} "
                        f"(dim={self.embedding_func.embedding_dim})"
                    )
                else:
                    raise

        await asyncio.to_thread(_create)

    async def _load_relation_index(self):
        """Load existing relation metadata into _relation_index cache.

        Only loads for relationship namespaces. Paginates through all vectors
        with returnMetadata=True to build the entity_name -> {vector_key, ...} mapping.
        """
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
                    "maxResults": S3_VECTORS_LIST_PAGE_SIZE,
                    "filter": {"workspace": self.workspace},
                }
                if next_token:
                    list_kwargs["nextToken"] = next_token

                response = self._client.list_vectors(**list_kwargs)
                vectors = response.get("vectors", [])
                for vec in vectors:
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
                f"[{self.workspace}] Loaded {len(self._relation_index)} entities "
                f"into relation index cache"
            )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Buffer vector docs for embedding and batched flush.

        Docs are buffered in ``self._pending_vector_docs`` and flushed in a
        single batch during ``index_done_callback()`` / ``finalize()``.
        Embedding is deferred to the flush path so repeated upserts of the
        same id and many small upsert calls can be embedded once in a single
        batch.
        """
        if not data:
            return

        current_time = int(time.time())

        pending_docs: list[tuple[str, _PendingS3VectorDoc]] = []
        for i, (k, v) in enumerate(data.items(), start=1):
            content = v["content"]
            source = {
                "created_at": current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            pending_docs.append(
                (
                    k,
                    _PendingS3VectorDoc(source=source, content=content),
                )
            )
            await _cooperative_yield(i)

        # Buffer: an upsert overrides a pending delete on the same id.
        async with self._flush_lock:
            for doc_id, pending_doc in pending_docs:
                self._pending_vector_deletes.discard(doc_id)
                self._pending_vector_docs[doc_id] = pending_doc

    async def _flush_pending_vector_ops(self) -> None:
        """Flush buffered vector upserts and deletes to S3 Vectors.

        Concurrency contract: the entire flush, including deferred embedding,
        runs under ``_flush_lock`` so it is sequential within the process and
        ordered against concurrent cross-worker flushes.

        Failure handling: on any error (embedding or server write), the
        buffers are left intact and the next flush will retry.
        """
        async with self._flush_lock:
            if not self._pending_vector_docs and not self._pending_vector_deletes:
                return
            if self._client is None:
                return

            pending_docs = self._pending_vector_docs
            pending_deletes = self._pending_vector_deletes

            # Deferred embedding
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
                    f"[{self.workspace}] {self.namespace} flush: embedding "
                    f"{len(docs_to_embed)} vectors in {len(batches)} batch(es) "
                    f"(batch_num={self._max_batch_size})"
                )
                try:
                    embeddings_list = await asyncio.gather(
                        *[
                            self.embedding_func(batch, context="document")
                            for batch in batches
                        ]
                    )
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error embedding pending vector ops "
                        f"(upserts={len(docs_to_embed)}): {e}"
                    )
                    raise
                embeddings = np.concatenate(embeddings_list)
                if len(embeddings) != len(docs_to_embed):
                    raise RuntimeError(
                        f"[{self.workspace}] Embedding count mismatch: expected "
                        f"{len(docs_to_embed)}, got {len(embeddings)}"
                    )
                for i, ((_, pdoc), embedding) in enumerate(
                    zip(docs_to_embed, embeddings), start=1
                ):
                    pdoc.vector = embedding.tolist()
                    await _cooperative_yield(i)

            # Execute deletes (chunked, max 500)
            if pending_deletes:
                delete_ids = list(pending_deletes)
                for i in range(0, len(delete_ids), S3_VECTORS_DELETE_MAX_BATCH):
                    chunk = delete_ids[i : i + S3_VECTORS_DELETE_MAX_BATCH]
                    try:
                        await asyncio.to_thread(
                            self._client.delete_vectors,
                            vectorBucketName=self._vector_bucket_name,
                            indexName=self._index_name,
                            ids=chunk,
                        )
                    except Exception as e:
                        logger.error(
                            f"[{self.workspace}] Error deleting vectors: {e}"
                        )
                        raise

            # Execute upserts (chunked, max 500)
            committed_ids: list[str] = []
            vectors_to_put: list[dict[str, Any]] = []
            for doc_id, pdoc in pending_docs.items():
                if pdoc.vector is None:
                    continue
                committed_ids.append(doc_id)
                # Build metadata: workspace (filterable) + source fields (filterable)
                # + content (non-filterable)
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
                for i in range(0, len(vectors_to_put), S3_VECTORS_PUT_MAX_BATCH):
                    chunk = vectors_to_put[i : i + S3_VECTORS_PUT_MAX_BATCH]
                    try:
                        await asyncio.to_thread(
                            self._client.put_vectors,
                            vectorBucketName=self._vector_bucket_name,
                            indexName=self._index_name,
                            vectors=chunk,
                        )
                    except Exception as e:
                        logger.error(
                            f"[{self.workspace}] Error putting vectors: {e}"
                        )
                        raise

            # Update relation index cache for flushed relation vectors
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

            # Clear successful entries
            for doc_id in committed_ids:
                pending_docs.pop(doc_id, None)
            pending_deletes.clear()

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """Similarity search against S3 Vectors with cosine distance conversion.

        S3 Vectors returns cosine distance; we convert to similarity:
        ``similarity = 1 - distance``. Results are post-filtered by
        ``cosine_better_than_threshold``.
        """
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
            logger.error(f"[{self.workspace}] Error querying S3 Vectors: {e}")
            return []

        results = []
        vectors = response.get("vectors", [])
        for vec in vectors:
            distance = vec.get("distance", 1.0)
            # S3 Vectors returns cosine distance; convert to similarity
            similarity = 1.0 - distance
            if similarity >= self.cosine_better_than_threshold:
                doc = dict(vec.get("metadata", {}))
                doc["id"] = vec.get("id")
                doc["distance"] = similarity
                results.append(doc)

        logger.info(
            f"[{self.workspace}] S3 Vectors query on {self._index_name}: "
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
            logger.error(f"[{self.workspace}] Error getting vector {id}: {e}")
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
            for i in range(0, len(remaining), S3_VECTORS_GET_MAX_BATCH):
                chunk = remaining[i : i + S3_VECTORS_GET_MAX_BATCH]

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
                    logger.error(
                        f"[{self.workspace}] Error getting vectors by ids: {e}"
                    )

        return [
            buffered[doc_id] if doc_id in buffered else doc_map.get(doc_id)
            for doc_id in ids
        ]

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vector embeddings for given IDs, with read-your-writes.

        Pending docs with ``vector is None`` trigger a lazy embed inside the
        lock; the resulting vector is cached on the buffered ``_PendingS3VectorDoc``
        so the next flush won't re-embed the same content.
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
                    logger.error(
                        f"[{self.workspace}] Error lazily embedding pending vectors "
                        f"(upserts={len(docs_to_embed)}): {e}"
                    )
                    raise
                embeddings = np.concatenate(embeddings_list)
                if len(embeddings) != len(docs_to_embed):
                    raise RuntimeError(
                        f"[{self.workspace}] Embedding count mismatch: expected "
                        f"{len(docs_to_embed)}, got {len(embeddings)}"
                    )
                for i, ((doc_id, pdoc), embedding) in enumerate(
                    zip(docs_to_embed, embeddings), start=1
                ):
                    pdoc.vector = embedding.tolist()
                    result[doc_id] = pdoc.vector
                    await _cooperative_yield(i)

        if not remaining or self._client is None:
            return result

        for i in range(0, len(remaining), S3_VECTORS_GET_MAX_BATCH):
            chunk = remaining[i : i + S3_VECTORS_GET_MAX_BATCH]

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
                logger.error(
                    f"[{self.workspace}] Error retrieving vectors by IDs from {self.namespace}: {e}"
                )

        return result

    async def delete(self, ids: list[str]) -> None:
        """Buffer vector deletes for batched flush.

        A delete cancels any pending upsert for the same id; the actual
        delete is performed by ``_flush_pending_vector_ops`` during the next
        ``index_done_callback`` / ``finalize`` call.
        """
        if not ids:
            return
        if isinstance(ids, set):
            ids = list(ids)
        async with self._flush_lock:
            for doc_id in ids:
                self._pending_vector_docs.pop(doc_id, None)
                self._pending_vector_deletes.add(doc_id)
        logger.debug(
            f"[{self.workspace}] Buffered delete for {len(ids)} vectors in {self.namespace}"
        )

    async def delete_entity(self, entity_name: str) -> None:
        """Buffer an entity vector delete by computing its hash ID."""
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        async with self._flush_lock:
            self._pending_vector_docs.pop(entity_id, None)
            self._pending_vector_deletes.add(entity_id)
        logger.debug(f"[{self.workspace}] Buffered delete for entity {entity_name}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relation vectors where entity appears as src or tgt.

        Uses the local ``_relation_index`` cache for O(1) lookup instead of
        scanning the server with ListVectors. The whole method runs under
        ``_flush_lock`` so the server-side delete cannot interleave with an
        in-flight flush.

        Buffer semantics: matching pending upserts are pruned **only after**
        the server-side delete succeeds. On failure the pending buffer stays
        intact and the exception propagates so the caller can short-circuit
        before ``index_done_callback`` flushes a half-cleaned buffer.
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
            # Look up relation keys from local cache
            relation_keys = self._relation_index.pop(entity_name, set())

            if relation_keys and self._client is not None:
                # Delete from server (chunked, max 500)
                keys_list = list(relation_keys)
                deleted_keys: set[str] = set()
                for i in range(0, len(keys_list), S3_VECTORS_DELETE_MAX_BATCH):
                    chunk = keys_list[i : i + S3_VECTORS_DELETE_MAX_BATCH]
                    try:
                        await asyncio.to_thread(
                            self._client.delete_vectors,
                            vectorBucketName=self._vector_bucket_name,
                            indexName=self._index_name,
                            ids=chunk,
                        )
                        deleted_keys.update(chunk)
                    except Exception as e:
                        # Put un-deleted keys back in cache on failure
                        self._relation_index.setdefault(entity_name, set()).update(
                            k for k in relation_keys if k not in deleted_keys
                        )
                        logger.error(
                            f"[{self.workspace}] Error deleting relations for {entity_name}: {e}"
                        )
                        raise

            # Prune pending buffer
            _prune_pending()
            logger.debug(
                f"[{self.workspace}] Deleted {len(relation_keys)} relations for entity {entity_name}"
            )

    async def index_done_callback(self) -> None:
        """Flush pending vector ops to S3 Vectors.

        S3 Vectors writes are immediately durable, so no refresh step is
        needed (unlike OpenSearch).
        """
        await self._flush_pending_vector_ops()

    async def drop_pending_index_ops(self) -> None:
        """Discard buffered upserts/deletes (pipeline aborting on error)."""
        async with self._flush_lock:
            self._pending_vector_docs.clear()
            self._pending_vector_deletes.clear()

    async def finalize(self):
        """Flush pending writes and close the boto3 client.

        Regular flush failures are captured and re-surfaced as a
        ``RuntimeError`` that names the unflushed buffer counts.
        """
        flush_error: Exception | None = None
        try:
            await self._flush_pending_vector_ops()
        except Exception as e:
            flush_error = e

        # Close client
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
                f"[{self.workspace}] S3VectorsStorage.finalize() flush raised; "
                f"{pending_docs} pending upserts and {pending_deletes} pending "
                f"deletes were left buffered (data lost)"
            ) from flush_error
        if pending_docs or pending_deletes:
            raise RuntimeError(
                f"[{self.workspace}] S3VectorsStorage.finalize() left "
                f"{pending_docs} pending upserts and {pending_deletes} pending "
                f"deletes buffered after final flush attempt (these writes have been lost)"
            )

    async def drop(self) -> dict[str, str]:
        """Delete and recreate the vector index, discarding pending buffers.

        Runs entirely under ``_flush_lock`` so a concurrent flush / upsert
        cannot land writes against an index that is being deleted and rebuilt.
        """
        async with self._flush_lock:
            # Pending writes are meaningless once the index is dropped.
            self._pending_vector_docs.clear()
            self._pending_vector_deletes.clear()
            self._relation_index.clear()

            try:
                if self._client is not None:
                    # Delete the index
                    def _delete_index():
                        try:
                            self._client.delete_index(
                                vectorBucketName=self._vector_bucket_name,
                                indexName=self._index_name,
                            )
                            logger.info(
                                f"[{self.workspace}] Dropped S3 Vectors index: {self._index_name}"
                            )
                        except ClientError as e:
                            error_code = e.response.get("Error", {}).get("Code", "")
                            if error_code in ("NotFoundException", "ResourceNotFoundException"):
                                logger.info(
                                    f"[{self.workspace}] S3 Vectors index already missing during drop: {self._index_name}"
                                )
                            else:
                                raise

                    await asyncio.to_thread(_delete_index)

                    # Recreate the index
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

                logger.info(
                    f"[{self.workspace}] Dropped and recreated S3 Vectors index: {self._index_name}"
                )
                return {
                    "status": "success",
                    "message": f"S3 Vectors index {self._index_name} dropped and recreated",
                }
            except Exception as e:
                logger.error(f"[{self.workspace}] Error dropping S3 Vectors index: {e}")
                return {"status": "error", "message": str(e)}
