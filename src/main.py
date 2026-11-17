"""FastAPI application entry point for the model-registry service.

Provides the central catalog of model versions and routing table for
the Pi inference cluster. Uses async lifespan for startup/shutdown,
structured logging via structlog, and OpenTelemetry with graceful degradation.
"""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import sqlalchemy as sa
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.types import Lifespan

from logging_config import configure_logging, get_logger
from models.domain import HealthResponse, LivenessResponse, ReadinessResponse
from routers import aliases, audit, deploy, flywheel, global_adapter, models, routing
from settings import get_settings
from telemetry import setup_telemetry

# Configure structured logging early
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan: startup and shutdown hooks.

    On startup:
    - Initialise OpenTelemetry tracing (graceful degradation)
    - Validate database connectivity

    On shutdown:
    - Dispose database connections

    Args:
        app: The FastAPI application instance.

    Yields:
        None
    """
    settings = get_settings()

    logger.info(f"Starting {settings.service_name} v{settings.version} ({settings.environment})")

    # Initialise telemetry
    otel_enabled = setup_telemetry(app, settings.service_name)
    if otel_enabled:
        logger.info("OpenTelemetry instrumentation enabled")
    else:
        logger.info(
            "OpenTelemetry instrumentation disabled (endpoint unreachable or packages missing)"
        )

    # Validate database connectivity and create tables
    from db import get_engine, init_db

    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(sa.text("SELECT 1"))
        logger.info("Database connection pool initialized")
        await init_db()
        logger.info("Database tables created/verified")
    except (SQLAlchemyError, OSError) as e:
        # Narrowed from bare ``Exception`` to the two failure modes the
        # startup probe can legitimately hit:
        # - ``SQLAlchemyError`` — any DB-driver level failure (bad URL,
        #   auth rejected, schema permissions, pool exhaustion).
        # - ``OSError`` — low-level socket/connect errors (DNS miss,
        #   refused connection, host unreachable) raised before SQLAlchemy
        #   gets a chance to wrap them.
        # Programming bugs and ``asyncio.CancelledError`` deliberately
        # propagate unwrapped so pod startup crashes loudly.
        logger.error(f"Database connection failed: {e}")
        raise RuntimeError(f"Database initialization failed: {e}") from e

    # Construct shared services with audit dependency
    from services.alias_service import AliasService
    from services.audit_service import AuditService
    from services.global_adapter_service import GlobalAdapterService
    from services.model_service import ModelService
    from services.routing_service import RoutingService

    audit_service = AuditService()
    model_service = ModelService(
        audit_service=audit_service,
        auto_approve_score=settings.auto_approve_score,
    )
    routing_service = RoutingService(audit_service=audit_service)
    alias_service = AliasService(audit_service=audit_service)
    global_adapter_service = GlobalAdapterService(audit_service=audit_service)

    # Inject into routers' module-level singletons
    models.set_model_service(model_service)
    routing.set_routing_service(routing_service)
    audit.set_audit_service(audit_service)
    aliases.set_alias_service(alias_service)
    deploy.set_services(model_service, routing_service, alias_service)
    global_adapter.set_global_adapter_service(global_adapter_service)

    # Start the DETECTIVE half of the unroutable-chain control. The
    # registration guard (services/chain_validation.py) can only reject a bad
    # chain at write time; a chain that was valid when written and rotted
    # later -- because a provider was flipped to `enabled: false` in
    # inference-router -- involves no write at all, so no guard ever runs. This
    # sweep re-checks every STORED rule against the current catalog, once now
    # and then on an interval. It reports and never repairs.
    #
    # Deliberately started AFTER the readiness-critical work above and never
    # awaited: a reporting control must not be able to delay or fail startup.
    from db import get_session_factory

    routing.get_chain_reconciler().start(get_session_factory())

    logger.info(f"{settings.service_name} ready on port {settings.port}")

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.service_name}")
    await routing.get_chain_reconciler().stop()
    from db import close_db

    await close_db()
    logger.info("Resources cleaned up")


def create_app(custom_lifespan: Lifespan[FastAPI] | None = None) -> FastAPI:
    """Construct and configure the FastAPI application.

    Args:
        custom_lifespan: Optional lifespan override (used in tests to skip
            real database initialization). Pass ``None`` to use the default
            production lifespan.

    Returns:
        The fully configured FastAPI app instance.
    """
    settings = get_settings()

    docs_enabled = (
        settings.environment in ("development", "test")
        or os.environ.get("ENABLE_DOCS", "").lower() == "true"
    )

    active_lifespan: Lifespan[FastAPI] = (
        custom_lifespan if custom_lifespan is not None else lifespan
    )

    app = FastAPI(
        title=settings.service_name,
        description="Central catalog of model versions and routing table for Pi inference cluster",
        version=settings.version,
        lifespan=active_lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(models.router)
    app.include_router(models.tags_router)
    app.include_router(routing.router)
    app.include_router(deploy.router)
    app.include_router(audit.router)
    app.include_router(aliases.router)
    app.include_router(flywheel.router)
    app.include_router(global_adapter.router)

    # ---------- Health endpoints (no prefix) ----------

    @app.get("/health", response_model=HealthResponse, tags=["health"], summary="Health check")
    async def health_check() -> HealthResponse:
        """Basic health check endpoint.

        Returns:
            Service health information.
        """
        return HealthResponse(
            status="healthy",
            service=settings.service_name,
            version=settings.version,
            timestamp=datetime.now(UTC),
        )

    @app.get(
        "/health/ready",
        response_model=ReadinessResponse,
        tags=["health"],
        summary="Readiness probe",
    )
    async def readiness_check() -> ReadinessResponse | JSONResponse:
        """Kubernetes readiness probe. Checks database connectivity.

        Returns:
            Readiness status with dependency check results.
        """
        from db import get_engine

        db_ok = False

        try:
            engine = get_engine()
            async with engine.connect() as conn:
                await conn.execute(sa.text("SELECT 1"))
            db_ok = True
        except (SQLAlchemyError, OSError) as e:
            # Same narrowing rationale as the lifespan probe above: only
            # DB-driver errors and raw socket failures indicate "the
            # database is down", which is what the readiness probe is
            # meant to report. Programming errors must still bubble up
            # as a 500 via the global exception handler.
            logger.warning("readiness_check_failed", error=str(e))

        if not db_ok:
            response = ReadinessResponse(ready=False, checks={"database": db_ok})
            return JSONResponse(
                status_code=503,
                content=response.model_dump(mode="json"),
            )

        return ReadinessResponse(ready=True, checks={"database": db_ok})

    @app.get(
        "/health/live",
        response_model=LivenessResponse,
        tags=["health"],
        summary="Liveness probe",
    )
    async def liveness_check() -> LivenessResponse:
        """Kubernetes liveness probe. Always returns 200 if process is alive.

        Returns:
            Liveness status.
        """
        return LivenessResponse()

    # ---------- Root ----------

    @app.get("/", tags=["health"])
    async def root() -> dict[str, str]:
        """Root endpoint with service information.

        Returns:
            Service metadata.
        """
        return {
            "service": settings.service_name,
            "version": settings.version,
            "docs": "/docs",
            "health": "/health",
        }

    # ---------- Global exception handler ----------

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all handler for unhandled exceptions.

        Args:
            request: The incoming request.
            exc: The unhandled exception.

        Returns:
            A 500 JSON error response.
        """
        if isinstance(exc, HTTPException):
            raise exc

        logger.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "status": 500}},
        )

    return app


# Module-level app for uvicorn
app = create_app()
