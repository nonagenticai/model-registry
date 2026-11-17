"""Audit trail API endpoints.

Provides endpoints for querying the audit event log to understand
what changed, when, and by whom.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from models.domain import AuditEventListResponse, AuditEventType
from services.audit_service import AuditService

router = APIRouter(prefix="/api/audit", tags=["audit"])

_audit_service = AuditService()


def get_audit_service() -> AuditService:
    """Return the singleton AuditService instance.

    Returns:
        The shared AuditService instance.
    """
    return _audit_service


def set_audit_service(service: AuditService) -> None:
    """Replace the module-level AuditService (used by main.py lifespan)."""
    global _audit_service
    _audit_service = service


@router.get(
    "",
    response_model=AuditEventListResponse,
    summary="List audit events with filtering",
)
async def list_audit_events(
    entity_type: str | None = Query(default=None, description="Filter by entity type"),
    entity_id: str | None = Query(default=None, description="Filter by entity ID"),
    event_type: AuditEventType | None = Query(default=None, description="Filter by event type"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    session: AsyncSession = Depends(get_session),
    audit_service: AuditService = Depends(get_audit_service),
) -> AuditEventListResponse:
    """List audit events, optionally filtered by entity and event type.

    Args:
        entity_type: Optional entity type filter.
        entity_id: Optional entity ID filter.
        event_type: Optional event type filter.
        limit: Maximum number of results to return.
        offset: Number of results to skip.
        session: Injected database session.
        audit_service: Injected audit service.

    Returns:
        An AuditEventListResponse containing matching events and total count.
    """
    events, total = await audit_service.list_events(
        session,
        entity_type=entity_type,
        entity_id=entity_id,
        event_type=event_type,
        limit=limit,
        offset=offset,
    )
    return AuditEventListResponse(events=events, total=total)


@router.get(
    "/models/{model_id}",
    response_model=AuditEventListResponse,
    summary="Get audit history for a specific model",
)
async def get_model_audit_history(
    model_id: UUID,
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    session: AsyncSession = Depends(get_session),
    audit_service: AuditService = Depends(get_audit_service),
) -> AuditEventListResponse:
    """Get all audit events for a specific model version.

    Args:
        model_id: UUID of the model version.
        limit: Maximum number of results.
        offset: Pagination offset.
        session: Injected database session.
        audit_service: Injected audit service.

    Returns:
        An AuditEventListResponse for the model.
    """
    events, total = await audit_service.get_model_history(
        session, str(model_id), limit=limit, offset=offset
    )
    return AuditEventListResponse(events=events, total=total)


@router.get(
    "/schemas/{schema_id}",
    response_model=AuditEventListResponse,
    summary="Get audit history for a schema",
)
async def get_schema_audit_history(
    schema_id: str,
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    session: AsyncSession = Depends(get_session),
    audit_service: AuditService = Depends(get_audit_service),
) -> AuditEventListResponse:
    """Get all audit events related to a schema_id.

    Args:
        schema_id: The schema identifier.
        limit: Maximum number of results.
        offset: Pagination offset.
        session: Injected database session.
        audit_service: Injected audit service.

    Returns:
        An AuditEventListResponse for the schema.
    """
    events, total = await audit_service.get_schema_history(
        session, schema_id, limit=limit, offset=offset
    )
    return AuditEventListResponse(events=events, total=total)
