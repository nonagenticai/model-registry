"""Model lifecycle management service.

Handles CRUD operations on model versions and enforces state transition
validation rules.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from services.audit_service import AuditService

from logging_config import get_logger
from models.db import ModelAliasORM, ModelTagORM, ModelVersionORM
from models.domain import (
    VALID_TRANSITIONS,
    AuditEventType,
    DeployFeedbackRequest,
    DeployStatus,
    ModelStatus,
    ModelVersion,
    ModelVersionSummary,
    QualityResponse,
    RegisterModelRequest,
    UpdateStatusRequest,
    is_valid_transition,
)

logger = get_logger(__name__)


class ModelService:
    """Service layer for model version lifecycle operations.

    All methods accept an AsyncSession and perform database operations
    within it. Transaction management (commit/rollback) is the caller's
    responsibility.
    """

    def __init__(
        self,
        audit_service: AuditService | None = None,
        auto_approve_score: float = 0.95,
    ) -> None:
        from services.audit_service import AuditService as _AuditService

        self._audit: AuditService | None = (
            audit_service if isinstance(audit_service, _AuditService) else None
        )
        self._auto_approve_score = auto_approve_score

    async def register_model(
        self, session: AsyncSession, request: RegisterModelRequest
    ) -> ModelVersion:
        """Register a new model version in the catalog.

        Args:
            session: Active database session.
            request: The registration request with model metadata.

        Returns:
            The newly created ModelVersion.

        Raises:
            ValueError: If a model with the same schema_id and version already exists.
        """
        # Check for duplicate schema_id + version
        stmt = select(ModelVersionORM).where(
            ModelVersionORM.schema_id == request.schema_id,
            ModelVersionORM.version == request.version,
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing is not None:
            raise ValueError(
                f"Model version already exists: schema_id={request.schema_id}, "
                f"version={request.version}"
            )

        now = datetime.now(UTC)
        orm_obj = ModelVersionORM(
            schema_id=request.schema_id,
            version=request.version,
            status=request.status.value,
            base_model=request.base_model,
            quantization=request.quantization,
            training_samples=request.training_samples,
            gguf_file=request.gguf_file,
            gguf_sha256=request.gguf_sha256,
            adapter_path=request.adapter_path,
            model_format=request.model_format,
            created_at=now,
            updated_at=now,
        )
        session.add(orm_obj)
        await session.flush()
        await session.refresh(orm_obj)

        # Write tags if provided
        if request.tags:
            tag_now = datetime.now(UTC)
            for key, value in request.tags.items():
                tag_orm = ModelTagORM(
                    model_id=orm_obj.id,
                    key=key,
                    value=value,
                    created_at=tag_now,
                )
                session.add(tag_orm)
            await session.flush()

        if self._audit is not None:
            await self._audit.record(
                session=session,
                event_type=AuditEventType.MODEL_REGISTERED,
                entity_type="model_version",
                entity_id=str(orm_obj.id),
                actor="api",
                detail={"schema_id": request.schema_id, "version": request.version},
            )

        await session.commit()

        logger.info(
            "model_registered",
            model_id=str(orm_obj.id),
            schema_id=orm_obj.schema_id,
            version=orm_obj.version,
        )

        return ModelVersion.model_validate(orm_obj)

    async def get_model(self, session: AsyncSession, model_id: UUID) -> ModelVersion | None:
        """Retrieve a model version by its UUID.

        Args:
            session: Active database session.
            model_id: The UUID of the model to retrieve.

        Returns:
            The ModelVersion if found, None otherwise.
        """
        stmt = select(ModelVersionORM).where(ModelVersionORM.id == model_id)
        result = await session.execute(stmt)
        orm_obj = result.scalar_one_or_none()
        if orm_obj is None:
            return None
        return ModelVersion.model_validate(orm_obj)

    async def list_models(
        self,
        session: AsyncSession,
        schema_id: str | None = None,
        status: ModelStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[ModelVersion], int]:
        """List model versions with optional filtering.

        Args:
            session: Active database session.
            schema_id: Filter by schema identifier.
            status: Filter by lifecycle status.
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            A tuple of (list of ModelVersion, total count).
        """
        stmt = select(ModelVersionORM)

        if schema_id is not None:
            stmt = stmt.where(ModelVersionORM.schema_id == schema_id)
        if status is not None:
            stmt = stmt.where(ModelVersionORM.status == status.value)

        # Count total matching records
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await session.execute(count_stmt)
        total = total_result.scalar() or 0

        # Fetch paginated results
        stmt = stmt.order_by(ModelVersionORM.created_at.desc())
        stmt = stmt.limit(limit).offset(offset)
        result = await session.execute(stmt)
        orm_objects = result.scalars().all()

        models = [ModelVersion.model_validate(obj) for obj in orm_objects]
        return models, total

    async def update_status(
        self, session: AsyncSession, model_id: UUID, request: UpdateStatusRequest
    ) -> ModelVersion:
        """Update the lifecycle status of a model version.

        Validates that the requested transition is allowed before applying.

        Args:
            session: Active database session.
            model_id: The UUID of the model to update.
            request: The status update request.

        Returns:
            The updated ModelVersion.

        Raises:
            ValueError: If the model is not found or the transition is invalid.
        """
        stmt = select(ModelVersionORM).where(ModelVersionORM.id == model_id)
        result = await session.execute(stmt)
        orm_obj = result.scalar_one_or_none()

        if orm_obj is None:
            raise ValueError(f"Model not found: {model_id}")

        current_status = ModelStatus(orm_obj.status)
        target_status = request.status

        if not is_valid_transition(current_status, target_status):
            raise ValueError(
                f"Invalid status transition: {current_status.value} -> {target_status.value}. "
                f"Allowed transitions from {current_status.value}: "
                f"{', '.join(s.value for s in VALID_TRANSITIONS.get(current_status, set()))}"
            )

        now = datetime.now(UTC)
        orm_obj.status = target_status.value
        orm_obj.updated_at = now

        # Apply optional fields from the update request
        if request.eval_score is not None:
            orm_obj.eval_score = request.eval_score
        if request.eval_metrics is not None:
            orm_obj.eval_metrics = request.eval_metrics
        if request.gguf_file is not None:
            orm_obj.gguf_file = request.gguf_file
        if request.gguf_sha256 is not None:
            orm_obj.gguf_sha256 = request.gguf_sha256

        # Record deployment timestamp
        if target_status == ModelStatus.DEPLOYED:
            orm_obj.deployed_at = now

        await session.flush()
        await session.refresh(orm_obj)

        if self._audit is not None:
            await self._audit.record(
                session=session,
                event_type=AuditEventType.STATUS_CHANGED,
                entity_type="model_version",
                entity_id=str(model_id),
                actor="api",
                detail={
                    "from_status": current_status.value,
                    "to_status": target_status.value,
                },
            )

        # Auto-approve: when transitioning to EVALUATED with a qualifying eval_score,
        # automatically advance to APPROVED.
        auto_approved = False
        if (
            target_status == ModelStatus.EVALUATED
            and orm_obj.eval_score is not None
            and orm_obj.eval_score >= self._auto_approve_score
        ):
            orm_obj.status = ModelStatus.APPROVED.value
            orm_obj.updated_at = datetime.now(UTC)
            auto_approved = True
            await session.flush()
            await session.refresh(orm_obj)

            if self._audit is not None:
                await self._audit.record(
                    session=session,
                    event_type=AuditEventType.AUTO_APPROVED,
                    entity_type="model_version",
                    entity_id=str(model_id),
                    actor="system",
                    detail={
                        "eval_score": orm_obj.eval_score,
                        "threshold": self._auto_approve_score,
                    },
                )

        await session.commit()

        if auto_approved:
            logger.info(
                "model_auto_approved",
                model_id=str(model_id),
                eval_score=orm_obj.eval_score,
                threshold=self._auto_approve_score,
            )
        else:
            logger.info(
                "model_status_updated",
                model_id=str(model_id),
                from_status=current_status.value,
                to_status=target_status.value,
            )

        return ModelVersion.model_validate(orm_obj)

    async def get_latest_quality(self, session: AsyncSession, schema_id: str) -> QualityResponse:
        """Return the latest quality score for a schema.

        Queries the most recent model version for the given schema_id that
        has a non-null eval_score, ordered by created_at descending.

        Args:
            session: Active database session.
            schema_id: The schema identifier to look up.

        Returns:
            A QualityResponse with the quality info, or one with null fields
            if no evaluated model exists for this schema.
        """
        stmt = (
            select(ModelVersionORM)
            .where(
                ModelVersionORM.schema_id == schema_id,
                ModelVersionORM.eval_score.isnot(None),
            )
            .order_by(ModelVersionORM.created_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        orm_obj = result.scalar_one_or_none()

        if orm_obj is None:
            logger.info(
                "quality_lookup_no_result",
                schema_id=schema_id,
            )
            return QualityResponse(schema_id=schema_id)

        logger.info(
            "quality_lookup_found",
            schema_id=schema_id,
            model_id=str(orm_obj.id),
            eval_score=orm_obj.eval_score,
        )
        return QualityResponse(
            schema_id=schema_id,
            quality_score=orm_obj.eval_score,
            model_id=orm_obj.id,
            version=orm_obj.version,
        )

    # ----- Tag operations -----

    async def set_tags(
        self,
        session: AsyncSession,
        model_id: UUID,
        tags: dict[str, str],
    ) -> dict[str, str]:
        """Set or update tags on a model version.

        Args:
            session: Active database session.
            model_id: UUID of the model version.
            tags: Key-value pairs to set.

        Returns:
            All current tags for the model after the update.

        Raises:
            ValueError: If the model doesn't exist.
        """
        # Verify model exists
        model = await self.get_model(session, model_id)
        if model is None:
            raise ValueError(f"Model not found: {model_id}")

        now = datetime.now(UTC)
        for key, value in tags.items():
            stmt = select(ModelTagORM).where(
                ModelTagORM.model_id == model_id,
                ModelTagORM.key == key,
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing is not None:
                existing.value = value
            else:
                session.add(
                    ModelTagORM(
                        model_id=model_id,
                        key=key,
                        value=value,
                        created_at=now,
                    )
                )

        if self._audit is not None:
            await self._audit.record(
                session=session,
                event_type=AuditEventType.TAG_SET,
                entity_type="tag",
                entity_id=str(model_id),
                actor="api",
                detail={"tags": tags},
            )

        await session.flush()
        await session.commit()
        return await self.get_tags(session, model_id)

    async def get_tags(
        self,
        session: AsyncSession,
        model_id: UUID,
    ) -> dict[str, str]:
        """Get all tags for a model version.

        Args:
            session: Active database session.
            model_id: UUID of the model version.

        Returns:
            Dict of key-value tag pairs.
        """
        stmt = select(ModelTagORM).where(ModelTagORM.model_id == model_id).order_by(ModelTagORM.key)
        result = await session.execute(stmt)
        return {t.key: t.value for t in result.scalars().all()}

    async def delete_tag(
        self,
        session: AsyncSession,
        model_id: UUID,
        key: str,
    ) -> bool:
        """Delete a specific tag from a model version.

        Args:
            session: Active database session.
            model_id: UUID of the model version.
            key: Tag key to delete.

        Returns:
            True if deleted, False if not found.
        """
        stmt = select(ModelTagORM).where(
            ModelTagORM.model_id == model_id,
            ModelTagORM.key == key,
        )
        result = await session.execute(stmt)
        if result.scalar_one_or_none() is None:
            return False

        del_stmt = delete(ModelTagORM).where(
            ModelTagORM.model_id == model_id,
            ModelTagORM.key == key,
        )
        await session.execute(del_stmt)

        if self._audit is not None:
            await self._audit.record(
                session=session,
                event_type=AuditEventType.TAG_DELETED,
                entity_type="tag",
                entity_id=str(model_id),
                actor="api",
                detail={"key": key},
            )

        await session.commit()
        return True

    async def search_by_tag(
        self,
        session: AsyncSession,
        key: str,
        value: str | None = None,
    ) -> list[ModelVersion]:
        """Find model versions by tag key and optionally value.

        Args:
            session: Active database session.
            key: Tag key to search for.
            value: Optional tag value to match exactly.

        Returns:
            List of matching ModelVersion objects.
        """
        tag_stmt = select(ModelTagORM.model_id).where(ModelTagORM.key == key)
        if value is not None:
            tag_stmt = tag_stmt.where(ModelTagORM.value == value)

        stmt = (
            select(ModelVersionORM)
            .where(ModelVersionORM.id.in_(tag_stmt))
            .order_by(ModelVersionORM.created_at.desc())
        )
        result = await session.execute(stmt)
        return [ModelVersion.model_validate(obj) for obj in result.scalars().all()]

    # ----- Model comparison -----

    async def compare_versions(
        self,
        session: AsyncSession,
        schema_id: str,
        versions: list[str] | None = None,
        limit: int = 10,
    ) -> list[ModelVersionSummary]:
        """Compare model versions for a schema, including aliases and tags.

        Args:
            session: Active database session.
            schema_id: The schema identifier.
            versions: Optional list of version labels to compare. If None, latest ``limit`` versions are compared.
            limit: Maximum number of versions to compare when versions is None.

        Returns:
            List of ModelVersionSummary with aliases and tags populated.
        """

        stmt = select(ModelVersionORM).where(ModelVersionORM.schema_id == schema_id)
        if versions:
            stmt = stmt.where(ModelVersionORM.version.in_(versions))
        else:
            stmt = stmt.order_by(ModelVersionORM.created_at.desc()).limit(limit)

        result = await session.execute(stmt)
        orm_objects = result.scalars().all()

        summaries = []
        for obj in orm_objects:
            # Get aliases for this model
            alias_stmt = select(ModelAliasORM.alias_name).where(ModelAliasORM.model_id == obj.id)
            alias_result = await session.execute(alias_stmt)
            alias_names = [r[0] for r in alias_result.fetchall()]

            # Get tags for this model
            tag_stmt = select(ModelTagORM).where(ModelTagORM.model_id == obj.id)
            tag_result = await session.execute(tag_stmt)
            tags = {t.key: t.value for t in tag_result.scalars().all()}

            summaries.append(
                ModelVersionSummary(
                    id=obj.id,
                    schema_id=obj.schema_id,
                    version=obj.version,
                    status=ModelStatus(obj.status),
                    eval_score=obj.eval_score,
                    eval_metrics=obj.eval_metrics,
                    created_at=obj.created_at,
                    deployed_at=obj.deployed_at,
                    deploy_status=DeployStatus(obj.deploy_status) if obj.deploy_status else None,
                    aliases=alias_names,
                    tags=tags,
                )
            )

        return summaries

    # ----- Deployment tracking -----

    async def update_deploy_status(
        self,
        session: AsyncSession,
        model_id: UUID,
        request: DeployFeedbackRequest,
    ) -> ModelVersion:
        """Update the deployment status of a model version (external feedback).

        Args:
            session: Active database session.
            model_id: UUID of the model version.
            request: Deployment feedback with status and optional error.

        Returns:
            The updated ModelVersion.

        Raises:
            ValueError: If the model is not found.
        """
        stmt = select(ModelVersionORM).where(ModelVersionORM.id == model_id)
        result = await session.execute(stmt)
        orm_obj = result.scalar_one_or_none()

        if orm_obj is None:
            raise ValueError(f"Model not found: {model_id}")

        now = datetime.now(UTC)
        orm_obj.deploy_status = request.deploy_status.value
        orm_obj.updated_at = now

        if request.deploy_error is not None:
            orm_obj.deploy_error = request.deploy_error

        if request.deploy_status in (DeployStatus.SUCCEEDED, DeployStatus.FAILED):
            orm_obj.deploy_completed_at = now

        await session.flush()
        await session.refresh(orm_obj)
        await session.commit()

        logger.info(
            "deploy_status_updated",
            model_id=str(model_id),
            deploy_status=request.deploy_status.value,
        )

        return ModelVersion.model_validate(orm_obj)
