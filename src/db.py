"""Database configuration and async session management.

Provides SQLAlchemy async engine and session factory for PostgreSQL
via the psycopg3 driver. Supports lazy initialization and clean shutdown.
"""

from collections.abc import AsyncGenerator

import structlog
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from settings import get_settings

logger = structlog.stdlib.get_logger(__name__)

# ---------------------------------------------------------------------------
# ORM Base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


# ---------------------------------------------------------------------------
# Idempotent additive migrations (mirrors metering_service/src/db.py)
# ---------------------------------------------------------------------------
#
# ``init_db`` only calls ``Base.metadata.create_all`` — Alembic is NOT auto-
# applied at startup. ``create_all`` creates *missing tables*, but it does
# NOT add columns to existing tables and it does NOT retroactively add
# tables that the ORM gained after the live DB was first provisioned.
#
# Result: long-running production DBs miss recent ORM additions until an
# operator hand-runs Alembic. Writes against missing columns/tables silently
# fail in services that swallow SQLAlchemyError. Fresh DBs (CI, local, new
# clusters) are unaffected because ``create_all`` already populates them
# from the current ORM metadata.
#
# Each statement below MUST be idempotent and MUST match the ORM model
# declarations in ``src/models/db.py`` exactly. A statement that fails (e.g.
# multi-replica startup race, partial state from a prior crash) is logged
# at debug and the batch continues — the readiness probe still catches a
# real connectivity problem.
ADDITIVE_MIGRATIONS: tuple[str, ...] = (
    # --- ModelVersionORM: deploy lifecycle columns ---
    # Without these, /api/deploy/* status updates silently lose their
    # state on legacy DBs (deploy_status / deploy_error written but column
    # doesn't exist → INSERT or UPDATE raises and the request 500s, or
    # the write is swallowed by an outer except).
    "ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS deployed_at TIMESTAMPTZ",
    "ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS deploy_status VARCHAR(32)",
    "ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS deploy_error TEXT",
    "ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS deploy_started_at TIMESTAMPTZ",
    "ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS deploy_completed_at TIMESTAMPTZ",
    # --- model_aliases (ModelAliasORM) ---
    # Composite PK (schema_id, alias_name) + secondary index on model_id.
    # Required for /api/aliases/* endpoints (champion / rollback markers).
    "CREATE TABLE IF NOT EXISTS model_aliases ("
    "schema_id VARCHAR(256) NOT NULL, "
    "alias_name VARCHAR(64) NOT NULL, "
    "model_id UUID NOT NULL, "
    "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
    "updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
    "PRIMARY KEY (schema_id, alias_name)"
    ")",
    "CREATE INDEX IF NOT EXISTS ix_model_aliases_model_id ON model_aliases (model_id)",
    # --- model_tags (ModelTagORM) ---
    # Composite PK (model_id, key) + (key, value) lookup index.
    "CREATE TABLE IF NOT EXISTS model_tags ("
    "model_id UUID NOT NULL, "
    "key VARCHAR(128) NOT NULL, "
    "value VARCHAR(512) NOT NULL, "
    "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
    "PRIMARY KEY (model_id, key)"
    ")",
    "CREATE INDEX IF NOT EXISTS ix_model_tags_key_value ON model_tags (key, value)",
    # --- audit_events (AuditEventORM) ---
    # SERIAL PK mirrors SQLAlchemy ``Integer + autoincrement=True`` on PG.
    # JSONB matches SQLAlchemy's ``JSON`` adaptation on the psycopg dialect.
    "CREATE TABLE IF NOT EXISTS audit_events ("
    "id SERIAL PRIMARY KEY, "
    "timestamp TIMESTAMPTZ NOT NULL DEFAULT now(), "
    "event_type VARCHAR(64) NOT NULL, "
    "entity_type VARCHAR(32) NOT NULL, "
    "entity_id VARCHAR(256) NOT NULL, "
    "actor VARCHAR(256) NOT NULL DEFAULT 'system', "
    "detail JSONB"
    ")",
    "CREATE INDEX IF NOT EXISTS ix_audit_events_event_type ON audit_events (event_type)",
    "CREATE INDEX IF NOT EXISTS ix_audit_events_entity_id ON audit_events (entity_id)",
    "CREATE INDEX IF NOT EXISTS ix_audit_events_timestamp ON audit_events (timestamp)",
    "CREATE INDEX IF NOT EXISTS ix_audit_entity_type_id ON audit_events (entity_type, entity_id)",
    "CREATE INDEX IF NOT EXISTS ix_audit_type_timestamp ON audit_events (event_type, timestamp)",
)


# ---------------------------------------------------------------------------
# Engine & session factory singletons
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine():
    """Get or lazily create the async SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        # NullPool: pgbouncer (transaction-mode, default_pool_size=2) is the
        # connection pool. App-side pooling on top would multiplex sessions
        # through a shared backend and break session-scoped state. Every
        # service that fronts via pgbouncer.pgbouncer.svc.example.com:6432
        # uses NullPool — see ingestion deployment for the reference pattern.
        _engine = create_async_engine(
            settings.database_url,
            future=True,
            echo=False,
            poolclass=NullPool,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or lazily create the async session factory.

    Returns:
        The singleton async_sessionmaker bound to the engine.
    """
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _async_session_factory


async def get_session() -> AsyncGenerator[AsyncSession]:
    """Yield a database session for FastAPI dependency injection.

    Usage::

        @app.get("/items")
        async def list_items(session: AsyncSession = Depends(get_session)):
            ...

    Yields:
        An AsyncSession that is automatically closed after use.
    """
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def init_db() -> None:
    """Create all tables from ORM metadata and apply additive migrations.

    ``create_all`` only creates *missing* tables, so long-running production
    DBs do not pick up new columns or new tables that the ORM gained after
    the DB was first provisioned. ``apply_additive_migrations`` closes that
    drift gap with idempotent ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``
    and ``CREATE TABLE IF NOT EXISTS`` statements. Fresh DBs are unaffected
    because ``create_all`` already populated them from current metadata.

    Note:
        Alembic remains the source of truth for downward-compatible schema
        changes. The additive migrations here are a safety net for the
        startup-time gap between an ORM addition and an operator hand-
        running Alembic.
    """
    from models.db import (  # noqa: F401
        AuditEventORM,
        ModelAliasORM,
        ModelTagORM,
        ModelVersionORM,
        RoutingRuleORM,
    )

    engine = get_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except SQLAlchemyError:
        # Benign in multi-replica startups: another replica may have
        # created the tables between our check and our CREATE TABLE.
        # The readiness probe still catches a real connectivity problem.
        logger.debug("table_creation_skipped_likely_race_condition", exc_info=True)

    await apply_additive_migrations(engine)


async def apply_additive_migrations(engine: AsyncEngine) -> None:
    """Apply the idempotent additive migrations against ``engine``.

    Each statement runs in its own transaction so a single benign failure
    (e.g. a column already added by a parallel replica, or a partial
    constraint left from a prior crash) does not abort the whole batch.
    Failures are logged at debug — the readiness probe surfaces real
    connectivity problems separately.
    """
    for stmt in ADDITIVE_MIGRATIONS:
        try:
            async with engine.begin() as conn:
                await conn.exec_driver_sql(stmt)
        except SQLAlchemyError:
            logger.debug("additive_migration_skipped", stmt=stmt, exc_info=True)


async def close_db() -> None:
    """Dispose the engine and release all connections."""
    global _engine, _async_session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
