"""PostgreSQL connection pool manager for lightrag_retriever.

Provides asyncpg-based connection pooling with retry logic, pgvector codec
registration, and Apache AGE graph configuration. This is a self-contained
port of lightrag.kg.postgres_impl.PostgreSQLDB with only the read-path
features needed by the retriever.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import ssl
import time
from typing import Any, Awaitable, Callable, TypeVar

import asyncpg
from asyncpg import Pool
from pgvector.asyncpg import register_vector
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

logger = logging.getLogger("lightrag_retriever")

T = TypeVar("T")


def _dollar_quote(s: str, tag_prefix: str = "AGE") -> str:
    s = "" if s is None else str(s)
    for i in itertools.count(1):
        tag = f"{tag_prefix}{i}"
        wrapper = f"${tag}$"
        if wrapper not in s:
            return f"{wrapper}{s}{wrapper}"


class PostgreSQLDB:
    """Asyncpg connection pool with retry, pgvector, and AGE support."""

    def __init__(self, config: dict[str, Any], **kwargs: Any):
        self.host = config["host"]
        self.port = config["port"]
        self.user = config["user"]
        self.password = config["password"]
        self.database = config["database"]
        self.workspace = config["workspace"]
        self.max = int(config["max_connections"])
        self.pool: Pool | None = None

        self.ssl_mode = config.get("ssl_mode")
        self.ssl_cert = config.get("ssl_cert")
        self.ssl_key = config.get("ssl_key")
        self.ssl_root_cert = config.get("ssl_root_cert")
        self.ssl_crl = config.get("ssl_crl")

        _ev = config.get("enable_vector", True)
        self.enable_vector = (
            _ev if isinstance(_ev, bool) else str(_ev).lower() in ("true", "1", "yes", "on")
        )
        self.vector_index_type = config.get("vector_index_type")

        self.server_settings = config.get("server_settings")
        self.statement_cache_size = config.get("statement_cache_size")

        if self.user is None or self.password is None or self.database is None:
            raise ValueError("Missing database user, password, or database")

        self._pool_reconnect_lock = asyncio.Lock()

        self._transient_exceptions = (
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
            OSError,
            asyncpg.exceptions.InterfaceError,
            asyncpg.exceptions.TooManyConnectionsError,
            asyncpg.exceptions.CannotConnectNowError,
            asyncpg.exceptions.PostgresConnectionError,
            asyncpg.exceptions.ConnectionDoesNotExistError,
            asyncpg.exceptions.ConnectionFailureError,
        )

        self.connection_retry_attempts = config["connection_retry_attempts"]
        self.connection_retry_backoff = config["connection_retry_backoff"]
        self.connection_retry_backoff_max = max(
            self.connection_retry_backoff, config["connection_retry_backoff_max"]
        )
        self.pool_close_timeout = config["pool_close_timeout"]

    def _create_ssl_context(self) -> ssl.SSLContext | None:
        if not self.ssl_mode:
            return None
        ssl_mode = self.ssl_mode.lower()
        if ssl_mode in ["disable", "allow", "prefer", "require"]:
            return None
        if ssl_mode in ["verify-ca", "verify-full"]:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            if ssl_mode == "verify-ca":
                context.check_hostname = False
            elif ssl_mode == "verify-full":
                context.check_hostname = True
            if self.ssl_root_cert and os.path.exists(self.ssl_root_cert):
                context.load_verify_locations(cafile=self.ssl_root_cert)
            if self.ssl_cert and self.ssl_key:
                if os.path.exists(self.ssl_cert) and os.path.exists(self.ssl_key):
                    context.load_cert_chain(self.ssl_cert, self.ssl_key)
            if self.ssl_crl and os.path.exists(self.ssl_crl):
                context.load_verify_locations(crlfile=self.ssl_crl)
            return context
        return None

    async def initdb(self):
        connection_params: dict[str, Any] = {
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "host": self.host,
            "port": self.port,
            "min_size": 1,
            "max_size": self.max,
        }

        if self.statement_cache_size is not None:
            connection_params["statement_cache_size"] = int(self.statement_cache_size)

        ssl_context = self._create_ssl_context()
        if ssl_context is not None:
            connection_params["ssl"] = ssl_context
        elif self.ssl_mode:
            if self.ssl_mode.lower() in ["require", "prefer"]:
                connection_params["ssl"] = True
            elif self.ssl_mode.lower() == "disable":
                connection_params["ssl"] = False

        if self.server_settings:
            try:
                settings = {}
                pairs = self.server_settings.split("&")
                for pair in pairs:
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        settings[key] = value
                if settings:
                    connection_params["server_settings"] = settings
            except Exception:
                pass

        wait_strategy = (
            wait_exponential(
                multiplier=self.connection_retry_backoff,
                min=self.connection_retry_backoff,
                max=self.connection_retry_backoff_max,
            )
            if self.connection_retry_backoff > 0
            else wait_fixed(0)
        )

        async def _init_connection(connection: asyncpg.Connection) -> None:
            if self.enable_vector:
                await register_vector(connection)

        async def _reset_connection(connection: asyncpg.Connection) -> None:
            try:
                reset_query = connection.get_reset_query()
                if reset_query:
                    await connection.execute(reset_query)
            except Exception:
                raise

        async def _create_pool_once() -> None:
            if self.enable_vector:
                bootstrap_conn = await asyncpg.connect(
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    host=self.host,
                    port=self.port,
                    ssl=connection_params.get("ssl"),
                )
                try:
                    await bootstrap_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                finally:
                    await bootstrap_conn.close()

            pool = await asyncpg.create_pool(
                **connection_params,
                init=_init_connection,
                reset=_reset_connection,
            )
            self.pool = pool

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.connection_retry_attempts),
                retry=retry_if_exception_type(self._transient_exceptions),
                wait=wait_strategy,
                before_sleep=self._before_sleep,
                reraise=True,
            ):
                with attempt:
                    await _create_pool_once()

            logger.info(
                f"PostgreSQL connected to {self.host}:{self.port}/{self.database}"
            )
        except Exception as e:
            logger.error(f"PostgreSQL failed to connect: {e}")
            raise

    async def _ensure_pool(self) -> None:
        if self.pool is None:
            async with self._pool_reconnect_lock:
                if self.pool is None:
                    await self.initdb()

    async def _reset_pool(self) -> None:
        async with self._pool_reconnect_lock:
            if self.pool is not None:
                try:
                    await asyncio.wait_for(
                        self.pool.close(), timeout=self.pool_close_timeout
                    )
                except Exception:
                    pass
            self.pool = None

    async def _before_sleep(self, retry_state: RetryCallState) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "PostgreSQL transient issue attempt %s/%s: %r",
            retry_state.attempt_number,
            self.connection_retry_attempts,
            exc,
        )
        await self._reset_pool()

    async def _run_with_retry(
        self,
        operation: Callable[[asyncpg.Connection], Awaitable[T]],
        *,
        with_age: bool = False,
        graph_name: str | None = None,
    ) -> T:
        wait_strategy = (
            wait_exponential(
                multiplier=self.connection_retry_backoff,
                min=self.connection_retry_backoff,
                max=self.connection_retry_backoff_max,
            )
            if self.connection_retry_backoff > 0
            else wait_fixed(0)
        )

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.connection_retry_attempts),
            retry=retry_if_exception_type(self._transient_exceptions),
            wait=wait_strategy,
            before_sleep=self._before_sleep,
            reraise=True,
        ):
            with attempt:
                await self._ensure_pool()
                assert self.pool is not None
                async with self.pool.acquire() as connection:
                    if with_age and graph_name:
                        await self.configure_age(connection, graph_name)
                    elif with_age and not graph_name:
                        raise ValueError("Graph name is required when with_age is True")
                    return await operation(connection)

    @staticmethod
    async def configure_age_extension(connection: asyncpg.Connection) -> None:
        try:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS AGE CASCADE")
            logger.info("PostgreSQL AGE extension enabled")
        except Exception as e:
            logger.warning(f"Could not create AGE extension: {e}")

    @staticmethod
    async def configure_age(connection: asyncpg.Connection, graph_name: str) -> None:
        try:
            await connection.execute('SET search_path = ag_catalog, "$user", public')
            await connection.execute(f"select create_graph('{graph_name}')")
        except (
            asyncpg.exceptions.InvalidSchemaNameError,
            asyncpg.exceptions.UniqueViolationError,
        ):
            pass

    async def query(
        self,
        sql: str,
        params: list[Any] | None = None,
        multirows: bool = False,
        with_age: bool = False,
        graph_name: str | None = None,
    ) -> dict[str, Any] | None | list[dict[str, Any]]:
        async def _operation(connection: asyncpg.Connection) -> Any:
            prepared_params = tuple(params) if params else ()
            if prepared_params:
                rows = await connection.fetch(sql, *prepared_params)
            else:
                rows = await connection.fetch(sql)

            if multirows:
                if rows:
                    columns = [col for col in rows[0].keys()]
                    return [dict(zip(columns, row)) for row in rows]
                return []

            if rows:
                columns = rows[0].keys()
                return dict(zip(columns, rows[0]))
            return None

        try:
            return await self._run_with_retry(
                _operation, with_age=with_age, graph_name=graph_name
            )
        except Exception as e:
            logger.error(f"PostgreSQL query error: {e}")
            raise

    async def execute(
        self,
        sql: str,
        data: dict[str, Any] | None = None,
        upsert: bool = False,
        ignore_if_exists: bool = False,
        with_age: bool = False,
        graph_name: str | None = None,
    ):
        async def _operation(connection: asyncpg.Connection) -> Any:
            prepared_values = tuple(data.values()) if data else ()
            try:
                if not data:
                    result = await connection.execute(sql)
                else:
                    result = await connection.execute(sql, *prepared_values)
            except (
                asyncpg.exceptions.UniqueViolationError,
                asyncpg.exceptions.DuplicateTableError,
                asyncpg.exceptions.DuplicateObjectError,
                asyncpg.exceptions.InvalidSchemaNameError,
            ) as e:
                if ignore_if_exists or upsert:
                    result = None
                else:
                    raise
            return result

        try:
            await self._run_with_retry(
                _operation, with_age=with_age, graph_name=graph_name
            )
        except Exception as e:
            logger.error(f"PostgreSQL execute error: {e}")
            raise

    async def close(self):
        if self.pool is not None:
            await self.pool.close()
            self.pool = None


def get_pg_config() -> dict[str, Any]:
    """Build PostgreSQL connection config from environment variables."""
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": os.environ.get("POSTGRES_PORT", 5432),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD"),
        "database": os.environ.get("POSTGRES_DATABASE", "postgres"),
        "workspace": os.environ.get("POSTGRES_WORKSPACE", "default"),
        "max_connections": os.environ.get("POSTGRES_MAX_CONNECTIONS", 50),
        "ssl_mode": os.environ.get("POSTGRES_SSL_MODE"),
        "ssl_cert": os.environ.get("POSTGRES_SSL_CERT"),
        "ssl_key": os.environ.get("POSTGRES_SSL_KEY"),
        "ssl_root_cert": os.environ.get("POSTGRES_SSL_ROOT_CERT"),
        "ssl_crl": os.environ.get("POSTGRES_SSL_CRL"),
        "enable_vector": True,
        "vector_index_type": os.environ.get("POSTGRES_VECTOR_INDEX_TYPE", "HNSW"),
        "server_settings": os.environ.get("POSTGRES_SERVER_SETTINGS"),
        "statement_cache_size": os.environ.get("POSTGRES_STATEMENT_CACHE_SIZE"),
        "connection_retry_attempts": min(
            100, int(os.environ.get("POSTGRES_CONNECTION_RETRIES", 10))
        ),
        "connection_retry_backoff": min(
            300.0, float(os.environ.get("POSTGRES_CONNECTION_RETRY_BACKOFF", 3.0))
        ),
        "connection_retry_backoff_max": min(
            600.0,
            float(os.environ.get("POSTGRES_CONNECTION_RETRY_BACKOFF_MAX", 60.0)),
        ),
        "pool_close_timeout": float(
            os.environ.get("POSTGRES_POOL_CLOSE_TIMEOUT", 30.0)
        ),
    }
