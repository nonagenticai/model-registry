"""Service layer for the system-wide global LoRA adapter.

Single-row table (``global_adapter``) holds the adapter the
inference-router falls through to when no enabled per-schema rule matches.
Phase 1 of the single-LoRA rollout — substrate-only; consumers wire up in
later phases.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from logging_config import get_logger
from models.db import GlobalAdapterORM, ModelVersionORM
from models.domain import (
    AuditEventType,
    GlobalAdapter,
    UpsertGlobalAdapterRequest,
)

if TYPE_CHECKING:
    from services.audit_service import AuditService

logger = get_logger(__name__)


class GlobalAdapterService:
    """Service layer for the singleton global adapter row."""

    def __init__(self, audit_service: AuditService | None = None) -> None:
        from services.audit_service import AuditService as _AuditService

        self._audit: AuditService | None = (
            audit_service if isinstance(audit_service, _AuditService) else None
        )

    async def get(self, session: AsyncSession) -> GlobalAdapter | None:
        """Return the current global adapter row, or None if none set yet.

        Args:
            session: Active database session.

        Returns:
            The current ``GlobalAdapter`` if set, else ``None``.
        """
        stmt = select(GlobalAdapterORM).where(GlobalAdapterORM.id == 1)
        result = await session.execute(stmt)
        orm_obj = result.scalar_one_or_none()
        if orm_obj is None:
            return None
        return GlobalAdapter.model_validate(orm_obj)

    async def upsert(
        self,
        session: AsyncSession,
        request: UpsertGlobalAdapterRequest,
    ) -> GlobalAdapter:
        """Create or update the singleton global adapter row.

        Validates that the referenced ``model_version_id`` exists in
        ``model_versions``; rejects with 400 otherwise (FK is not enforced
        at the DB level for cross-dialect compat, so we check explicitly).

        Args:
            session: Active database session.
            request: Upsert payload.

        Returns:
            The current ``GlobalAdapter`` after the upsert.
        """
        # Validate referenced model_version exists.
        mv_stmt = select(ModelVersionORM.id).where(ModelVersionORM.id == request.model_version_id)
        mv_result = await session.execute(mv_stmt)
        if mv_result.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown model_version_id: {request.model_version_id}. "
                    "Register the model version first."
                ),
            )

        now = datetime.now(UTC)

        stmt = select(GlobalAdapterORM).where(GlobalAdapterORM.id == 1)
        result = await session.execute(stmt)
        orm_obj = result.scalar_one_or_none()

        if orm_obj is None:
            orm_obj = GlobalAdapterORM(
                id=1,
                model_version_id=request.model_version_id,
                minio_uri=request.minio_uri,
                adapter_hash=request.adapter_hash,
                notes=request.notes,
                created_at=now,
                updated_at=now,
            )
            session.add(orm_obj)
            logger.info(
                "global_adapter_created",
                model_version_id=str(request.model_version_id),
                adapter_hash=request.adapter_hash,
            )
        else:
            orm_obj.model_version_id = request.model_version_id
            orm_obj.minio_uri = request.minio_uri
            orm_obj.adapter_hash = request.adapter_hash
            orm_obj.notes = request.notes
            orm_obj.updated_at = now
            logger.info(
                "global_adapter_updated",
                model_version_id=str(request.model_version_id),
                adapter_hash=request.adapter_hash,
            )

        await session.flush()
        await session.refresh(orm_obj)

        if self._audit is not None:
            await self._audit.record(
                session=session,
                event_type=AuditEventType.GLOBAL_ADAPTER_SET,
                entity_type="global_adapter",
                entity_id="1",
                actor="api",
                detail={
                    "model_version_id": str(request.model_version_id),
                    "adapter_hash": request.adapter_hash,
                    "minio_uri": request.minio_uri,
                },
            )

        await session.commit()

        return GlobalAdapter.model_validate(orm_obj)
