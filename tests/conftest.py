"""Shared test fixtures for model-registry tests.

Requires PostgreSQL. The connection is sourced from DATABASE_URL (set by the
python-service archetype CI in ``postgres_isolation: schema`` mode, which routes
every connection — via libpq PGOPTIONS ``search_path`` — to a per-job
``ci_run_<run_id>_<run_attempt>`` schema in the shared ``model_registry_ci`` DB),
then TEST_DATABASE_URL, then a local docker-compose default.

Tables are reset per test by dropping the model's own tables (not the schema
object), so the suite is schema-agnostic: it works whether search_path is
``public`` (local) or the archetype's ``ci_run_*`` schema (CI). No database is
ever created from the test process — the old per-xdist-worker ``CREATE DATABASE``
path connected to the ``postgres`` maintenance DB, which pgbouncer routed onto
prod :5432 and leaked per-run databases; it has been removed.
"""

import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from sqlalchemy import text

# Archetype CI sets DATABASE_URL on $GITHUB_ENV (see
# gitops-production/.github/workflows/python-service.yml "Prepare CI database").
# In schema-isolation mode it also exports PGOPTIONS with
# ``search_path="ci_run_<run_id>_<run_attempt>"``, which libpq applies to every
# psycopg connection — so honouring DATABASE_URL is all the conftest needs to
# land in the per-job schema. Fall back to TEST_DATABASE_URL, then a local
# docker-compose default.
TEST_DATABASE_URL = (
    os.getenv("DATABASE_URL")
    or os.getenv("TEST_DATABASE_URL")
    or "postgresql+psycopg://postgres:postgres@localhost:5453/model_registry"
)

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Set test environment variables BEFORE importing application modules
os.environ["MODEL_REGISTRY_ENVIRONMENT"] = "test"
os.environ["MODEL_REGISTRY_LOG_FORMAT"] = "console"
os.environ["MODEL_REGISTRY_LOG_LEVEL"] = "DEBUG"
os.environ["MODEL_REGISTRY_DATABASE_URL"] = TEST_DATABASE_URL
os.environ["TEST_DATABASE_URL"] = TEST_DATABASE_URL
os.environ["MODEL_REGISTRY_GITOPS_REPO_PATH"] = "/tmp/test-gitops"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Import ORM models so Base.metadata is populated before create_all
import models.db as _orm_models  # noqa: E402, F401
from db import Base, get_session  # noqa: E402
from main import create_app  # noqa: E402
from services.alias_service import AliasService  # noqa: E402
from services.audit_service import AuditService  # noqa: E402
from services.global_adapter_service import GlobalAdapterService  # noqa: E402
from services.model_service import ModelService  # noqa: E402
from services.routing_service import RoutingService  # noqa: E402


@asynccontextmanager
async def _noop_lifespan(app: Any) -> AsyncIterator[None]:
    """No-op lifespan that skips database connectivity checks during tests.

    Args:
        app: The FastAPI application instance (unused).

    Yields:
        None
    """
    yield


@pytest_asyncio.fixture
async def engine():
    """Create a PostgreSQL async engine for testing.

    Resets state per test by dropping the model's own tables (schema-agnostic)
    and recreating them, so it works whether ``search_path`` is ``public``
    (local) or the archetype's per-job ``ci_run_*`` schema (CI schema-isolation
    mode). Dropping tables rather than the schema object keeps any installed
    extensions intact and never touches a schema another job may share.

    Yields:
        The test engine instance.
    """
    test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with test_engine.begin() as conn:
        # Drop our tables first to clear orphans from crashed runs, then build.
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(text(f'DROP TABLE IF EXISTS "{table.name}" CASCADE'))
        await conn.run_sync(Base.metadata.create_all)

    yield test_engine

    async with test_engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(text(f'DROP TABLE IF EXISTS "{table.name}" CASCADE'))
    await test_engine.dispose()


@pytest_asyncio.fixture
async def session(engine) -> AsyncGenerator[AsyncSession]:
    """Yield an async session bound to the test engine.

    Args:
        engine: The test engine fixture.

    Yields:
        An AsyncSession for database operations.
    """
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as sess:
        yield sess


@pytest_asyncio.fixture
async def client(engine) -> AsyncGenerator[AsyncClient]:
    """Create an httpx.AsyncClient configured to use the test database.

    Overrides the get_session dependency to use the test engine's sessions,
    and uses a no-op lifespan to skip real database initialization.

    Args:
        engine: The test engine fixture.

    Yields:
        An AsyncClient bound to the test FastAPI application.
    """
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _test_get_session() -> AsyncGenerator[AsyncSession]:
        async with factory() as sess:
            yield sess

    app = create_app(custom_lifespan=_noop_lifespan)

    # Override the database session dependency
    app.dependency_overrides[get_session] = _test_get_session

    # Wire up services with audit/alias support (mirrors main.py lifespan)
    from routers import aliases as aliases_router_mod
    from routers import audit as audit_router_mod
    from routers import deploy as deploy_router_mod
    from routers import global_adapter as global_adapter_router_mod
    from routers import models as models_router_mod
    from routers import routing as routing_router_mod

    audit_svc = AuditService()
    model_svc = ModelService(audit_service=audit_svc)
    routing_svc = RoutingService(audit_service=audit_svc)
    alias_svc = AliasService(audit_service=audit_svc)
    global_adapter_svc = GlobalAdapterService(audit_service=audit_svc)

    models_router_mod.set_model_service(model_svc)
    routing_router_mod.set_routing_service(routing_svc)
    audit_router_mod.set_audit_service(audit_svc)
    aliases_router_mod.set_alias_service(alias_svc)
    deploy_router_mod.set_services(model_svc, routing_svc, alias_svc)
    global_adapter_router_mod.set_global_adapter_service(global_adapter_svc)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def sample_model_data() -> dict[str, Any]:
    """Return a sample model registration payload.

    Returns:
        A dictionary suitable for POST /api/models.
    """
    return {
        "schema_id": "ks_EntitySpotterAgent",
        "version": "v1",
        "base_model": "Qwen3-4B-AWQ-270m",
        "quantization": "Q4_K_M",
        "training_samples": 500,
        "status": "training",
    }


@pytest.fixture
def sample_routing_data() -> dict[str, Any]:
    """Return a sample routing rule update payload.

    Returns:
        A dictionary suitable for PUT /api/routing/{schema_id}.
    """
    return {
        "active_model_id": str(uuid4()),
        "fallback_chain": ["pgx-local", "deepseek", "xai"],
        "endpoint": "fg-ks-entityspotteragent.llm-inference.svc.example.com:8080",
    }
