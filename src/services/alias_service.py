"""Model alias management service.

Provides named aliases (e.g. champion, rollback, candidate) that point to
specific model versions within a schema. Aliases decouple identity from
version numbers, enabling zero-downtime rollback.
"""

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from logging_config import get_logger
from models.db import ModelAliasORM, ModelVersionORM
from models.domain import AuditEventType, ModelAlias, ModelVersion

if TYPE_CHECKING:
    from services.audit_service import AuditService

logger = get_logger(__name__)

ALIAS_PATTERN = re.compile(r"^[a-z][a-z0-9\-]{0,62}$")


class AliasService:
    """Service layer for model alias operations.

    All methods accept an AsyncSession. Audit events are emitted
    within the same transaction as the alias mutation.
    """

    def __init__(self, audit_service: AuditService | None = None) -> None:
        from services.audit_service import AuditService as _AuditService

        self._audit: AuditService | None = (
            audit_service if isinstance(audit_service, _AuditService) else None
        )

    @staticmethod
    def _validate_alias_name(alias_name: str) -> None:
        """Validate that alias_name matches the allowed pattern.

        Args:
            alias_name: The alias name to validate.

        Raises:
            ValueError: If the name doesn't match the pattern.
        """
        if not ALIAS_PATTERN.match(alias_name):
            raise ValueError(
                f"Invalid alias name '{alias_name}'. Must match pattern: ^[a-z][a-z0-9\\-]{{0,62}}$"
            )

    async def set_alias(
        self,
        session: AsyncSession,
        schema_id: str,
        alias_name: str,
        model_id: UUID,
        actor: str = "api",
    ) -> ModelAlias:
        """Set or update a named alias for a model version.

        Args:
            session: Active database session.
            schema_id: The schema identifier.
            alias_name: The alias name (e.g. 'champion').
            model_id: UUID of the model version.
            actor: Who triggered this action.

        Returns:
            The created or updated ModelAlias.

        Raises:
            ValueError: If alias_name is invalid or model doesn't exist.
        """
        self._validate_alias_name(alias_name)

        # Verify the model exists
        model_stmt = select(ModelVersionORM).where(ModelVersionORM.id == model_id)
        model_result = await session.execute(model_stmt)
        model = model_result.scalar_one_or_none()
        if model is None:
            raise ValueError(f"Model not found: {model_id}")

        now = datetime.now(UTC)
        stmt = select(ModelAliasORM).where(
            ModelAliasORM.schema_id == schema_id,
            ModelAliasORM.alias_name == alias_name,
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        old_model_id = None
        if existing is not None:
            old_model_id = existing.model_id
            existing.model_id = model_id
            existing.updated_at = now
        else:
            orm_obj = ModelAliasORM(
                schema_id=schema_id,
                alias_name=alias_name,
                model_id=model_id,
                created_at=now,
                updated_at=now,
            )
            session.add(orm_obj)

        await session.flush()

        if self._audit is not None:
            detail: dict = {"alias": alias_name, "to_model_id": str(model_id)}
            if old_model_id is not None:
                detail["from_model_id"] = str(old_model_id)
            await self._audit.record(
                session=session,
                event_type=AuditEventType.ALIAS_SET,
                entity_type="alias",
                entity_id=f"{schema_id}:{alias_name}",
                actor=actor,
                detail=detail,
            )

        await session.commit()

        logger.info(
            "alias_set",
            schema_id=schema_id,
            alias_name=alias_name,
            model_id=str(model_id),
        )

        # Re-fetch to return clean object
        stmt = select(ModelAliasORM).where(
            ModelAliasORM.schema_id == schema_id,
            ModelAliasORM.alias_name == alias_name,
        )
        result = await session.execute(stmt)
        return ModelAlias.model_validate(result.scalar_one())

    async def get_alias(
        self,
        session: AsyncSession,
        schema_id: str,
        alias_name: str,
    ) -> ModelAlias | None:
        """Get a specific alias.

        Args:
            session: Active database session.
            schema_id: The schema identifier.
            alias_name: The alias name.

        Returns:
            The ModelAlias if found, None otherwise.
        """
        stmt = select(ModelAliasORM).where(
            ModelAliasORM.schema_id == schema_id,
            ModelAliasORM.alias_name == alias_name,
        )
        result = await session.execute(stmt)
        orm_obj = result.scalar_one_or_none()
        if orm_obj is None:
            return None
        return ModelAlias.model_validate(orm_obj)

    async def list_aliases(
        self,
        session: AsyncSession,
        schema_id: str,
    ) -> list[ModelAlias]:
        """List all aliases for a schema.

        Args:
            session: Active database session.
            schema_id: The schema identifier.

        Returns:
            List of ModelAlias objects.
        """
        stmt = (
            select(ModelAliasORM)
            .where(ModelAliasORM.schema_id == schema_id)
            .order_by(ModelAliasORM.alias_name)
        )
        result = await session.execute(stmt)
        return [ModelAlias.model_validate(obj) for obj in result.scalars().all()]

    async def delete_alias(
        self,
        session: AsyncSession,
        schema_id: str,
        alias_name: str,
        actor: str = "api",
    ) -> bool:
        """Delete an alias.

        Args:
            session: Active database session.
            schema_id: The schema identifier.
            alias_name: The alias name.
            actor: Who triggered this action.

        Returns:
            True if deleted, False if not found.
        """
        stmt = select(ModelAliasORM).where(
            ModelAliasORM.schema_id == schema_id,
            ModelAliasORM.alias_name == alias_name,
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing is None:
            return False

        old_model_id = existing.model_id
        del_stmt = delete(ModelAliasORM).where(
            ModelAliasORM.schema_id == schema_id,
            ModelAliasORM.alias_name == alias_name,
        )
        await session.execute(del_stmt)

        if self._audit is not None:
            await self._audit.record(
                session=session,
                event_type=AuditEventType.ALIAS_DELETED,
                entity_type="alias",
                entity_id=f"{schema_id}:{alias_name}",
                actor=actor,
                detail={"alias": alias_name, "model_id": str(old_model_id)},
            )

        await session.commit()

        logger.info(
            "alias_deleted",
            schema_id=schema_id,
            alias_name=alias_name,
        )
        return True

    async def resolve_alias(
        self,
        session: AsyncSession,
        schema_id: str,
        alias_name: str,
    ) -> ModelVersion | None:
        """Resolve an alias to its full ModelVersion.

        Args:
            session: Active database session.
            schema_id: The schema identifier.
            alias_name: The alias name to resolve.

        Returns:
            The ModelVersion if found, None otherwise.
        """
        alias = await self.get_alias(session, schema_id, alias_name)
        if alias is None:
            return None

        stmt = select(ModelVersionORM).where(ModelVersionORM.id == alias.model_id)
        result = await session.execute(stmt)
        orm_obj = result.scalar_one_or_none()
        if orm_obj is None:
            return None
        return ModelVersion.model_validate(orm_obj)

    async def get_aliases_for_model(
        self,
        session: AsyncSession,
        model_id: UUID,
    ) -> list[ModelAlias]:
        """Get all aliases pointing to a specific model version.

        Args:
            session: Active database session.
            model_id: UUID of the model version.

        Returns:
            List of ModelAlias objects pointing to this model.
        """
        stmt = (
            select(ModelAliasORM)
            .where(ModelAliasORM.model_id == model_id)
            .order_by(ModelAliasORM.schema_id, ModelAliasORM.alias_name)
        )
        result = await session.execute(stmt)
        return [ModelAlias.model_validate(obj) for obj in result.scalars().all()]
