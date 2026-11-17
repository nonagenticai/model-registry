"""Unit tests for main.py (lifespan, create_app, exception handler, health endpoints)."""

import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from db import Base, get_session
from main import create_app, lifespan

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5453/model_registry",
)


class TestCreateApp:
    """Tests for create_app()."""

    def test_create_app_returns_fastapi(self) -> None:
        """create_app returns a FastAPI app with the expected configuration."""

        @asynccontextmanager
        async def noop_lifespan(app: Any) -> AsyncIterator[None]:
            yield

        app = create_app(custom_lifespan=noop_lifespan)
        assert app.title == "model-registry"
        assert app.version == "0.1.0"

    def test_create_app_with_default_lifespan(self) -> None:
        """create_app with no custom lifespan uses the default production lifespan."""
        app = create_app()
        assert app is not None
        assert app.title == "model-registry"
        # The router_lifespan wraps the real lifespan; verify routers are included.
        # fastapi 0.137 wraps include_router results in _IncludedRouter objects
        # that have no `.path`, so derive registered paths from the OpenAPI schema
        # (stable across versions) plus any top-level routes that still expose .path.
        paths = set(app.openapi().get("paths", {}).keys())
        for route in app.routes:
            p = cast(str | None, getattr(route, "path", None))
            if p:
                paths.add(p)
        assert "/api/models" in paths or any("/api/models" in p for p in paths)

    def test_create_app_docs_enabled_in_test(self) -> None:
        """In test environment, docs endpoints are available."""

        @asynccontextmanager
        async def noop_lifespan(app: Any) -> AsyncIterator[None]:
            yield

        app = create_app(custom_lifespan=noop_lifespan)
        assert app.openapi_url == "/openapi.json"


class TestLifespan:
    """Tests for the lifespan context manager."""

    async def test_lifespan_startup_and_shutdown(self) -> None:
        """Lifespan initializes services and disposes DB resources on shutdown."""
        # Create a real test engine for the lifespan to connect to
        test_engine = create_async_engine(TEST_DATABASE_URL)

        mock_app = MagicMock()

        from routers import aliases as aliases_mod
        from routers import audit as audit_mod
        from routers import models as models_mod
        from routers import routing as routing_mod

        with (
            patch("main.setup_telemetry", return_value=False) as mock_telemetry,
            patch("db.get_engine", return_value=test_engine),
            patch("db.close_db", new_callable=AsyncMock) as mock_close_db,
        ):
            async with lifespan(mock_app):
                # Inside lifespan: services must be wired into router modules
                # so that any request handler gets a fully constructed graph.
                assert models_mod.get_model_service() is not None
                assert routing_mod.get_routing_service() is not None
                assert aliases_mod.get_alias_service() is not None
                assert audit_mod.get_audit_service() is not None
                # Telemetry was attempted exactly once during startup
                mock_telemetry.assert_called_once()
                # DB cleanup hasn't run yet
                mock_close_db.assert_not_called()
            # On shutdown, close_db must run exactly once
            mock_close_db.assert_called_once()

        await test_engine.dispose()

    async def test_real_lifespan_creates_tables_and_services(self) -> None:
        """Real lifespan creates DB tables and injects services into routers.

        This test exercises the actual production lifespan (not _noop_lifespan)
        to verify that startup creates all ORM tables and wires up services.
        """
        import db as db_module
        from routers import aliases as aliases_mod
        from routers import audit as audit_mod
        from routers import models as models_mod
        from routers import routing as routing_mod

        # Save and restore global engine state
        saved_engine = db_module._engine
        saved_factory = db_module._async_session_factory
        db_module._engine = None
        db_module._async_session_factory = None

        test_engine = create_async_engine(TEST_DATABASE_URL)
        mock_app = MagicMock()

        try:
            with (
                patch("main.setup_telemetry", return_value=False),
                patch("db.get_engine", return_value=test_engine),
            ):
                async with lifespan(mock_app):
                    # Inside lifespan: DB tables should exist
                    async with test_engine.connect() as conn:

                        def check_tables(sync_conn):
                            from sqlalchemy import inspect

                            inspector = inspect(sync_conn)
                            return inspector.get_table_names()

                        tables = await conn.run_sync(check_tables)
                        assert "model_versions" in tables
                        assert "routing_rules" in tables
                        assert "audit_events" in tables
                        assert "model_aliases" in tables
                        assert "model_tags" in tables

                    # Services should have been injected into routers
                    model_svc = models_mod.get_model_service()
                    assert model_svc is not None
                    assert model_svc._audit is not None  # Audit service wired

                    routing_svc = routing_mod.get_routing_service()
                    assert routing_svc is not None
                    assert routing_svc._audit is not None

                    alias_svc = aliases_mod.get_alias_service()
                    assert alias_svc is not None
                    assert alias_svc._audit is not None

                    audit_svc = audit_mod.get_audit_service()
                    assert audit_svc is not None
        finally:
            db_module._engine = saved_engine
            db_module._async_session_factory = saved_factory
            await test_engine.dispose()

    async def test_lifespan_db_failure(self) -> None:
        """Lifespan raises RuntimeError chained from the underlying DB error.

        The startup body must never run because ``__aenter__`` should fail before
        yield. We assert both the wrapping RuntimeError message and the chained
        cause to lock in the user-facing error contract. The raised exception is
        an ``OSError`` (host unreachable) — one of the two narrowed types the
        lifespan handler catches alongside ``SQLAlchemyError``.
        """
        mock_engine = MagicMock()
        mock_connect_ctx = AsyncMock()
        mock_connect_ctx.__aenter__ = AsyncMock(side_effect=OSError("DB down"))
        mock_connect_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_engine.connect.return_value = mock_connect_ctx

        mock_app = MagicMock()

        with (
            patch("main.setup_telemetry", return_value=False),
            patch("db.get_engine", return_value=mock_engine),
        ):
            with pytest.raises(RuntimeError, match="Database initialization failed") as excinfo:
                async with lifespan(mock_app):
                    pytest.fail("lifespan body must not execute when DB connect fails")

        # The original DB exception must be chained as __cause__ for debuggability
        assert excinfo.value.__cause__ is not None
        assert isinstance(excinfo.value.__cause__, OSError)
        assert "DB down" in str(excinfo.value.__cause__)
        assert "DB down" in str(excinfo.value)

    async def test_lifespan_db_failure_sqlalchemy_error(self) -> None:
        """Lifespan also narrows on ``SQLAlchemyError`` from the driver layer.

        Mirrors ``test_lifespan_db_failure`` but raises
        ``sqlalchemy.exc.OperationalError`` (a ``SQLAlchemyError`` subclass)
        to confirm both branches of the narrowed ``except`` are exercised.
        """
        from sqlalchemy.exc import OperationalError

        mock_engine = MagicMock()
        mock_connect_ctx = AsyncMock()
        mock_connect_ctx.__aenter__ = AsyncMock(
            side_effect=OperationalError("SELECT 1", {}, Exception("auth failed"))
        )
        mock_connect_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_engine.connect.return_value = mock_connect_ctx

        mock_app = MagicMock()

        with (
            patch("main.setup_telemetry", return_value=False),
            patch("db.get_engine", return_value=mock_engine),
        ):
            with pytest.raises(RuntimeError, match="Database initialization failed") as excinfo:
                async with lifespan(mock_app):
                    pytest.fail("lifespan body must not execute when DB connect fails")

        assert isinstance(excinfo.value.__cause__, OperationalError)

    async def test_lifespan_programming_error_propagates(self) -> None:
        """A non-narrow exception (e.g. ``RuntimeError``) must NOT be caught.

        The narrowed ``(SQLAlchemyError, OSError)`` handler exists to
        translate infra failures into a user-friendly RuntimeError. A
        programming error (TypeError / RuntimeError) must bypass that
        translation so the original stack reaches the caller.
        """
        mock_engine = MagicMock()
        mock_connect_ctx = AsyncMock()
        mock_connect_ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("bug in code"))
        mock_connect_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_engine.connect.return_value = mock_connect_ctx

        mock_app = MagicMock()

        with (
            patch("main.setup_telemetry", return_value=False),
            patch("db.get_engine", return_value=mock_engine),
        ):
            with pytest.raises(RuntimeError, match="bug in code"):
                async with lifespan(mock_app):
                    pytest.fail("lifespan body must not execute when DB connect fails")

    async def test_lifespan_with_otel_enabled(self) -> None:
        """Lifespan completes successfully when setup_telemetry returns True.

        Verifies that the OTel-enabled startup branch still wires services and
        defers DB cleanup until after the body exits.
        """
        test_engine = create_async_engine(TEST_DATABASE_URL)
        mock_app = MagicMock()

        from routers import models as models_mod

        with (
            patch("main.setup_telemetry", return_value=True) as mock_telemetry,
            patch("db.get_engine", return_value=test_engine),
            patch("db.close_db", new_callable=AsyncMock) as mock_close_db,
        ):
            async with lifespan(mock_app):
                # OTel branch must still pass through service wiring
                assert models_mod.get_model_service() is not None
                # setup_telemetry was called with the app and the service name kwarg
                mock_telemetry.assert_called_once()
                call_args = mock_telemetry.call_args
                assert call_args.args[0] is mock_app
                # close_db has not run yet — only after body exits
                mock_close_db.assert_not_called()
            # On shutdown close_db must run exactly once
            mock_close_db.assert_called_once()

        await test_engine.dispose()


class TestExceptionHandler:
    """Tests for the global exception handler."""

    async def test_unhandled_exception_returns_500(self) -> None:
        """An unhandled exception in an endpoint returns 500."""

        @asynccontextmanager
        async def noop_lifespan(app: Any) -> AsyncIterator[None]:
            yield

        app = create_app(custom_lifespan=noop_lifespan)

        @app.get("/test-exception")
        async def raise_error():
            raise RuntimeError("unexpected boom")

        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/test-exception")
            assert resp.status_code == 500
            assert resp.json()["error"]["message"] == "Internal server error"

    async def test_http_exception_propagates(self) -> None:
        """An HTTPException is re-raised and not caught by the global handler."""
        from fastapi import HTTPException

        @asynccontextmanager
        async def noop_lifespan(app: Any) -> AsyncIterator[None]:
            yield

        app = create_app(custom_lifespan=noop_lifespan)

        @app.get("/test-http-exception")
        async def raise_http_error():
            raise HTTPException(status_code=418, detail="I'm a teapot")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/test-http-exception")
            assert resp.status_code == 418
            assert resp.json()["detail"] == "I'm a teapot"


class TestHealthReadinessEndpoint:
    """Tests for GET /health/ready."""

    async def test_readiness_check_when_db_available(self) -> None:
        """Readiness check returns 200 when DB is reachable."""

        @asynccontextmanager
        async def noop_lifespan(app: Any) -> AsyncIterator[None]:
            yield

        app = create_app(custom_lifespan=noop_lifespan)

        # Create a real test engine
        test_engine = create_async_engine(TEST_DATABASE_URL)
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        factory = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)

        async def test_get_session() -> AsyncGenerator[AsyncSession]:
            async with factory() as sess:
                yield sess

        app.dependency_overrides[get_session] = test_get_session

        # The /health/ready endpoint imports get_engine from db inside the function
        with patch("db.get_engine", return_value=test_engine), patch("db._engine", test_engine):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health/ready")
                assert resp.status_code == 200
                assert resp.json()["ready"] is True
                assert resp.json()["checks"]["database"] is True

        app.dependency_overrides.clear()
        await test_engine.dispose()

    async def test_readiness_check_when_db_unavailable(self) -> None:
        """Readiness check returns 503 when DB is unreachable.

        The mocked engine raises ``OSError`` (host unreachable) which is
        one of the two types the narrowed readiness handler catches.
        """

        @asynccontextmanager
        async def noop_lifespan(app: Any) -> AsyncIterator[None]:
            yield

        app = create_app(custom_lifespan=noop_lifespan)

        # Mock engine that fails to connect
        mock_engine = MagicMock()
        mock_connect_ctx = AsyncMock()
        mock_connect_ctx.__aenter__ = AsyncMock(side_effect=OSError("DB down"))
        mock_connect_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_engine.connect.return_value = mock_connect_ctx

        with patch("db.get_engine", return_value=mock_engine), patch("db._engine", mock_engine):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health/ready")
                assert resp.status_code == 503
                body = resp.json()
                assert body["ready"] is False
                assert body["checks"]["database"] is False

    async def test_readiness_check_when_db_unavailable_sqlalchemy(self) -> None:
        """Readiness check also reports 503 on a ``SQLAlchemyError``.

        Mirrors ``test_readiness_check_when_db_unavailable`` but raises a
        driver-level ``OperationalError`` to exercise the other branch of
        the narrowed ``(SQLAlchemyError, OSError)`` handler.
        """
        from sqlalchemy.exc import OperationalError

        @asynccontextmanager
        async def noop_lifespan(app: Any) -> AsyncIterator[None]:
            yield

        app = create_app(custom_lifespan=noop_lifespan)

        mock_engine = MagicMock()
        mock_connect_ctx = AsyncMock()
        mock_connect_ctx.__aenter__ = AsyncMock(
            side_effect=OperationalError("SELECT 1", {}, Exception("auth failed"))
        )
        mock_connect_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_engine.connect.return_value = mock_connect_ctx

        with patch("db.get_engine", return_value=mock_engine), patch("db._engine", mock_engine):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health/ready")
                assert resp.status_code == 503
                body = resp.json()
                assert body["ready"] is False
                assert body["checks"]["database"] is False

    async def test_liveness_endpoint(self) -> None:
        """Liveness endpoint returns 200 with status alive."""

        @asynccontextmanager
        async def noop_lifespan(app: Any) -> AsyncIterator[None]:
            yield

        app = create_app(custom_lifespan=noop_lifespan)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/live")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "alive"

    async def test_root_endpoint(self) -> None:
        """Root endpoint returns 200 with service info."""

        @asynccontextmanager
        async def noop_lifespan(app: Any) -> AsyncIterator[None]:
            yield

        app = create_app(custom_lifespan=noop_lifespan)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert "service" in data
