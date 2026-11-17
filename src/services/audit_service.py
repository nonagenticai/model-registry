"""Audit trail service for recording and querying registry mutations."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from logging_config import get_logger
from models.db import AuditEventORM
from models.domain import AuditEvent, AuditEventType

logger = get_logger(__name__)


class AuditService:
    """Records and queries audit events for registry mutations.

    All ``record()`` calls should happen within the same DB session/transaction
    as the mutation they audit, so that a failed mutation does not leave
    orphaned audit records.
    """

    async def record(
        self,
        session: AsyncSession,
        event_type: AuditEventType,
        entity_type: str,
        entity_id: str,
        actor: str = "api",
        detail: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Record an audit event.

        Args:
            session: Active database session (should be the same as the mutation).
            event_type: The type of event being recorded.
            entity_type: The type of entity affected.
            entity_id: The identifier of the affected entity.
            actor: Who/what triggered the event.
            detail: Optional JSON-serialisable detail dict.

        Returns:
            The created AuditEvent.
        """
        now = datetime.now(UTC)
        orm_obj = AuditEventORM(
            timestamp=now,
            event_type=event_type.value,
            entity_type=entity_type,
            entity_id=entity_id,
            actor=actor,
            detail=detail,
        )
        session.add(orm_obj)
        await session.flush()
        await session.refresh(orm_obj)

        logger.info(
            "audit_event_recorded",
            event_type=event_type.value,
            entity_type=entity_type,
            entity_id=entity_id,
            actor=actor,
        )

        return AuditEvent.model_validate(orm_obj)

    async def list_events(
        self,
        session: AsyncSession,
        entity_type: str | None = None,
        entity_id: str | None = None,
        event_type: AuditEventType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[AuditEvent], int]:
        """List audit events with optional filtering.

        Args:
            session: Active database session.
            entity_type: Filter by entity type.
            entity_id: Filter by entity ID.
            event_type: Filter by event type.
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            A tuple of (list of AuditEvent, total count).
        """
        stmt = select(AuditEventORM)

        if entity_type is not None:
            stmt = stmt.where(AuditEventORM.entity_type == entity_type)
        if entity_id is not None:
            stmt = stmt.where(AuditEventORM.entity_id == entity_id)
        if event_type is not None:
            stmt = stmt.where(AuditEventORM.event_type == event_type.value)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await session.execute(count_stmt)
        total = total_result.scalar() or 0

        stmt = stmt.order_by(AuditEventORM.timestamp.desc())
        stmt = stmt.limit(limit).offset(offset)
        result = await session.execute(stmt)
        orm_objects = result.scalars().all()

        events = [AuditEvent.model_validate(obj) for obj in orm_objects]
        return events, total

    async def get_model_history(
        self,
        session: AsyncSession,
        model_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[AuditEvent], int]:
        """Get all audit events for a specific model version.

        Args:
            session: Active database session.
            model_id: UUID string of the model version.
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            A tuple of (list of AuditEvent, total count).
        """
        return await self.list_events(
            session,
            entity_type="model_version",
            entity_id=model_id,
            limit=limit,
            offset=offset,
        )

    async def get_schema_history(
        self,
        session: AsyncSession,
        schema_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[AuditEvent], int]:
        """Get all audit events related to a schema_id.

        Searches across entity types: routing_rule (entity_id=schema_id)
        and alias (entity_id starts with schema_id:).

        Args:
            session: Active database session.
            schema_id: The schema identifier.
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            A tuple of (list of AuditEvent, total count).
        """
        stmt = select(AuditEventORM).where(AuditEventORM.entity_id.like(f"{schema_id}%"))

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await session.execute(count_stmt)
        total = total_result.scalar() or 0

        stmt = stmt.order_by(AuditEventORM.timestamp.desc())
        stmt = stmt.limit(limit).offset(offset)
        result = await session.execute(stmt)
        orm_objects = result.scalars().all()

        events = [AuditEvent.model_validate(obj) for obj in orm_objects]
        return events, total
