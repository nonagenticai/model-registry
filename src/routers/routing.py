"""Routing table API endpoints.

Provides endpoints for querying and updating the routing table that maps
schema_ids to active models and fallback inference chains.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from models.domain import (
    ChainHealthResponse,
    EnsureRoutingRequest,
    ErrorResponse,
    RoutingRule,
    UpdateRoutingRequest,
)
from services.chain_reconciliation import ChainReconciler
from services.chain_validation import UnroutableChainError
from services.routing_service import RoutingService

router = APIRouter(prefix="/api/routing", tags=["routing"])

# Shared routing service instance (maintains the TTL cache)
_routing_service = RoutingService()

# Shared reconciler. Module-level like the service above so the background
# sweep started in main.py's lifespan and the /chain-health endpoint read the
# SAME cached report -- two instances would let the endpoint answer "clean"
# while the loop was logging findings.
_chain_reconciler = ChainReconciler()


def get_routing_service() -> RoutingService:
    """Return the singleton RoutingService instance.

    Exposed as a dependency so tests can override it.

    Returns:
        The shared RoutingService instance.
    """
    return _routing_service


def set_routing_service(service: RoutingService) -> None:
    """Replace the module-level RoutingService (used by main.py lifespan)."""
    global _routing_service
    _routing_service = service


def get_chain_reconciler() -> ChainReconciler:
    """Return the singleton ChainReconciler holding the last sweep's report.

    Exposed as a dependency so tests can override it.

    Returns:
        The shared ChainReconciler instance.
    """
    return _chain_reconciler


@router.get(
    "",
    response_model=list[RoutingRule],
    summary="Get full routing table (cached)",
)
async def get_routing_table(
    session: AsyncSession = Depends(get_session),
    routing_service: RoutingService = Depends(get_routing_service),
) -> list[RoutingRule]:
    """Return the complete routing table.

    Results are served from an in-memory TTL cache when available.

    Args:
        session: Injected database session.
        routing_service: Injected routing service.

    Returns:
        List of all routing rules.
    """
    return await routing_service.get_routing_table(session)


# ⚠️ DECLARATION ORDER IS LOAD-BEARING. Starlette matches routes in the order
# they are added, and `GET /{schema_id}` below is a catch-all that happily
# matches the literal segment "chain-health". Declared after it, this endpoint
# is unreachable and answers `404 No routing rule found for schema_id:
# chain-health` -- a 404 that reads like "no such rule", not like "you shadowed
# the route", which is how this kind of mistake survives review. Measured, not
# theorised: it returned exactly that until the block was moved up here.
# `test_chain_health_is_not_shadowed_by_the_schema_id_route` pins it.
@router.get(
    "/chain-health",
    response_model=ChainHealthResponse,
    summary="Report stored fallback chains that the current catalog cannot route",
)
async def chain_health(
    refresh: bool = False,
    session: AsyncSession = Depends(get_session),
    reconciler: ChainReconciler = Depends(get_chain_reconciler),
) -> ChainHealthResponse:
    """Return the detective sweep's view of the STORED routing table.

    This is the readable surface of the control that covers the class
    registration-time validation cannot: a chain that was valid when written
    and rotted later because a provider was disabled in inference-router.

    By default it serves the last completed sweep, so polling this endpoint
    costs nothing and repeated callers all see the same numbers. It runs a
    fresh sweep only when there is no report yet, or when ``refresh=true`` is
    asked for explicitly.

    It is a REPORT, not a repair and not a gate: it returns 200 whatever it
    finds. Returning a non-200 for unroutable rules would make a monitoring
    endpoint able to take the service out of service, which is the failure
    mode this whole change exists to avoid.

    Args:
        refresh: Force a fresh sweep instead of serving the cached report.
        session: Injected database session.
        reconciler: Injected reconciler holding the cached report.

    Returns:
        The report, including status ``"unknown"`` when the provider catalog
        could not be read -- which is deliberately NOT an error, because
        "I could not check" is a real and useful answer.
    """
    report = reconciler.last_report
    if refresh or report is None:
        report = await reconciler.reconcile(session)
    return ChainHealthResponse.model_validate(report, from_attributes=True)


@router.get(
    "/{schema_id}",
    response_model=RoutingRule,
    responses={404: {"model": ErrorResponse}},
    summary="Get route for specific schema",
)
async def get_route(
    schema_id: str,
    session: AsyncSession = Depends(get_session),
    routing_service: RoutingService = Depends(get_routing_service),
) -> RoutingRule:
    """Look up the routing rule for a specific schema_id.

    Args:
        schema_id: The schema identifier to look up.
        session: Injected database session.
        routing_service: Injected routing service.

    Returns:
        The routing rule for the given schema.
    """
    rule = await routing_service.get_route(session, schema_id)
    if rule is None:
        raise HTTPException(
            status_code=404,
            detail=f"No routing rule found for schema_id: {schema_id}",
        )
    return rule


@router.put(
    "/{schema_id}",
    response_model=RoutingRule,
    responses={422: {"model": ErrorResponse}},
    summary="Create or update routing rule",
)
async def update_route(
    schema_id: str,
    request: UpdateRoutingRequest,
    session: AsyncSession = Depends(get_session),
    routing_service: RoutingService = Depends(get_routing_service),
) -> RoutingRule:
    """Create or update the routing rule for a schema_id.

    Args:
        schema_id: The schema identifier.
        request: The routing update request.
        session: Injected database session.
        routing_service: Injected routing service.

    Returns:
        The created or updated routing rule.

    Raises:
        HTTPException: 422 when ``fallback_chain`` names an unknown provider or
            cannot fail over (fewer than ``routing_min_viable_hops`` distinct
            enabled providers). Rejected rather than repaired -- see
            ``services/chain_validation``.
    """
    try:
        return await routing_service.upsert_route(session, schema_id, request)
    except UnroutableChainError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete(
    "/{schema_id}",
    status_code=204,
    responses={404: {"model": ErrorResponse}},
    summary="Delete routing rule",
)
async def delete_route(
    schema_id: str,
    session: AsyncSession = Depends(get_session),
    routing_service: RoutingService = Depends(get_routing_service),
) -> None:
    """Delete the routing rule for a schema_id.

    Returns 204 on success, 404 if no rule exists for the given schema_id.

    Args:
        schema_id: The schema identifier to delete.
        session: Injected database session.
        routing_service: Injected routing service.
    """
    deleted = await routing_service.delete_route(session, schema_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No routing rule found for schema_id: {schema_id}",
        )


@router.post(
    "/ensure",
    response_model=RoutingRule,
    responses={422: {"model": ErrorResponse}},
    summary="Look up or create a routing rule by schema fingerprint",
)
async def ensure_route(
    request: EnsureRoutingRequest,
    session: AsyncSession = Depends(get_session),
    routing_service: RoutingService = Depends(get_routing_service),
) -> RoutingRule:
    """Return the routing rule whose fingerprint matches; create if missing.

    Called by inference-router on requests that carry an OpenAI
    ``response_format`` json_schema. The caller never has to pick a
    ``schema_id``: model-registry allocates a uuid7 on first sighting and
    returns the same rule (with that same uuid7) on every subsequent
    sighting of the same schema.

    Raises:
        HTTPException: 422 when a NEW rule would be created with a
            ``fallback_chain`` that cannot fail over. A fingerprint HIT
            returns the existing rule unchanged and is never validated --
            reading an already-broken rule must not 422.
    """
    try:
        return await routing_service.ensure_by_fingerprint(session, request)
    except UnroutableChainError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
