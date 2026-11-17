"""Routing table management service with TTL caching.

Provides cached lookups of routing rules so the inference-router can
quickly determine where to send each request. Cache is invalidated
on routing rule updates.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid7

from cachetools import TTLCache
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from logging_config import get_logger
from models.db import RoutingRuleORM
from models.domain import (
    AuditEventType,
    EnsureRoutingRequest,
    RoutingRule,
    UpdateRoutingRequest,
)
from services.chain_validation import UnroutableChainError, validate_fallback_chain
from settings import DEFAULT_FALLBACK_CHAIN, get_settings

if TYPE_CHECKING:
    from services.audit_service import AuditService

logger = get_logger(__name__)


class RoutingService:
    """Service layer for routing table operations with in-memory TTL cache.

    The cache stores RoutingRule instances keyed by schema_id. On any write
    operation the affected cache entry is evicted so the next read fetches
    fresh data from the database.
    """

    def __init__(self, audit_service: AuditService | None = None) -> None:
        from services.audit_service import AuditService as _AuditService

        self._audit: AuditService | None = (
            audit_service if isinstance(audit_service, _AuditService) else None
        )
        settings = get_settings()
        self._cache: TTLCache[str, RoutingRule] = TTLCache(
            maxsize=1024, ttl=settings.routing_cache_ttl
        )
        self._full_table_cache: list[RoutingRule] | None = None
        self._full_table_cache_time: datetime | None = None

    @staticmethod
    def _validated_chain(chain: list[str], *, schema_id: str, actor: str) -> list[str]:
        """Return ``chain`` unchanged, or raise if it cannot fail over.

        The single chokepoint for the two write paths (``upsert_route`` and
        ``ensure_by_fingerprint``). Deliberately returns the chain UNMODIFIED
        on success: repairing a bad chain here would hide the writer's
        misconfiguration, and the writer is the only thing that can fix it
        durably (the three ``["gemma","deepseek","claude"]`` rules are
        re-asserted nightly by a Job, so any repair would be undone within
        24h and re-repaired forever).

        Logs at ERROR before raising so the rejection is visible in the
        service's own logs and not only in the caller's, then lets
        ``UnroutableChainError`` propagate to the 422 mapping in
        ``routers/routing.py``.

        Args:
            chain: The resolved chain (caller-supplied, or the server default
                filled in for it).
            schema_id: Rule being written -- for the log line only.
            actor: Which write path is running -- for the log line only.

        Returns:
            The same list, when it is routable.

        Raises:
            UnroutableChainError: when the chain names an unknown provider, or
                has fewer than ``routing_min_viable_hops`` distinct enabled
                providers.
        """
        try:
            validate_fallback_chain(chain)
        except UnroutableChainError as exc:
            logger.error(
                "routing_rule_rejected_unroutable_chain",
                schema_id=schema_id,
                actor=actor,
                fallback_chain=chain,
                reason=str(exc),
            )
            raise
        return chain

    def invalidate(self, schema_id: str | None = None) -> None:
        """Invalidate cache entries.

        Args:
            schema_id: If provided, invalidate only this key. Otherwise
                       invalidate the entire cache.
        """
        if schema_id is not None:
            self._cache.pop(schema_id, None)
        else:
            self._cache.clear()
        # Always clear the full table cache on any invalidation
        self._full_table_cache = None
        self._full_table_cache_time = None

    async def get_routing_table(self, session: AsyncSession) -> list[RoutingRule]:
        """Return the full routing table, served from cache if fresh.

        Args:
            session: Active database session.

        Returns:
            List of all routing rules.
        """
        settings = get_settings()

        # Check if full table cache is still valid
        if self._full_table_cache is not None and self._full_table_cache_time is not None:
            age = (datetime.now(UTC) - self._full_table_cache_time).total_seconds()
            if age < settings.routing_cache_ttl:
                return self._full_table_cache

        stmt = select(RoutingRuleORM).order_by(RoutingRuleORM.schema_id)
        result = await session.execute(stmt)
        orm_objects = result.scalars().all()

        rules = [RoutingRule.model_validate(obj) for obj in orm_objects]

        # Populate both caches
        self._full_table_cache = rules
        self._full_table_cache_time = datetime.now(UTC)
        for rule in rules:
            self._cache[rule.schema_id] = rule

        return rules

    async def get_route(self, session: AsyncSession, schema_id: str) -> RoutingRule | None:
        """Look up the routing rule for a specific schema_id.

        Checks the per-key TTL cache first, falling back to the database.

        Args:
            session: Active database session.
            schema_id: The schema identifier to look up.

        Returns:
            The RoutingRule if found, None otherwise.
        """
        # Try cache first
        cached = self._cache.get(schema_id)
        if cached is not None:
            return cached

        stmt = select(RoutingRuleORM).where(RoutingRuleORM.schema_id == schema_id)
        result = await session.execute(stmt)
        orm_obj = result.scalar_one_or_none()

        if orm_obj is None:
            return None

        rule = RoutingRule.model_validate(orm_obj)
        self._cache[schema_id] = rule
        return rule

    async def upsert_route(
        self,
        session: AsyncSession,
        schema_id: str,
        request: UpdateRoutingRequest,
    ) -> RoutingRule:
        """Create or update a routing rule for the given schema_id.

        Args:
            session: Active database session.
            schema_id: The schema identifier to create/update.
            request: The routing update request.

        Returns:
            The created or updated RoutingRule.
        """
        stmt = select(RoutingRuleORM).where(RoutingRuleORM.schema_id == schema_id)
        result = await session.execute(stmt)
        orm_obj = result.scalar_one_or_none()

        now = datetime.now(UTC)

        if orm_obj is None:
            orm_obj = RoutingRuleORM(
                schema_id=schema_id,
                active_model_id=request.active_model_id,
                fallback_chain=self._validated_chain(
                    request.fallback_chain or list(DEFAULT_FALLBACK_CHAIN),
                    schema_id=schema_id,
                    actor="upsert_route/create",
                ),
                endpoint=request.endpoint,
                json_schema=request.json_schema,
                gbnf_grammar=request.gbnf_grammar,
                system_prompt=request.system_prompt,
                adapter_path=request.adapter_path,
                enabled=request.enabled if request.enabled is not None else True,
                created_at=now,
                updated_at=now,
            )
            session.add(orm_obj)
            logger.info("routing_rule_created", schema_id=schema_id)
        else:
            if request.active_model_id is not None:
                orm_obj.active_model_id = request.active_model_id
            if request.fallback_chain is not None:
                # Only validate a chain the caller actually sent. A partial
                # PUT that omits fallback_chain must not be rejected because
                # of a pre-existing bad value it is not touching -- that would
                # wedge every unrelated update to the 64 already-broken rules.
                orm_obj.fallback_chain = self._validated_chain(
                    request.fallback_chain,
                    schema_id=schema_id,
                    actor="upsert_route/update",
                )
            if request.endpoint is not None:
                orm_obj.endpoint = request.endpoint
            if request.json_schema is not None:
                orm_obj.json_schema = request.json_schema
            if request.gbnf_grammar is not None:
                orm_obj.gbnf_grammar = request.gbnf_grammar
            if request.system_prompt is not None:
                orm_obj.system_prompt = request.system_prompt
            if request.adapter_path is not None:
                orm_obj.adapter_path = request.adapter_path
            if request.enabled is not None:
                orm_obj.enabled = request.enabled
            orm_obj.updated_at = now
            logger.info("routing_rule_updated", schema_id=schema_id)

        await session.flush()
        await session.refresh(orm_obj)

        if self._audit is not None:
            await self._audit.record(
                session=session,
                event_type=AuditEventType.ROUTE_UPDATED,
                entity_type="routing_rule",
                entity_id=schema_id,
                actor="api",
                detail={
                    "active_model_id": str(request.active_model_id)
                    if request.active_model_id
                    else None,
                    "endpoint": request.endpoint,
                },
            )

        await session.commit()

        # Invalidate cache for this schema
        self.invalidate(schema_id)

        return RoutingRule.model_validate(orm_obj)

    async def delete_route(self, session: AsyncSession, schema_id: str) -> bool:
        """Delete the routing rule for the given schema_id.

        Args:
            session: Active database session.
            schema_id: The schema identifier to delete.

        Returns:
            True if a rule was deleted, False if no rule existed.
        """
        stmt = select(RoutingRuleORM).where(RoutingRuleORM.schema_id == schema_id)
        result = await session.execute(stmt)
        orm_obj = result.scalar_one_or_none()

        if orm_obj is None:
            return False

        fingerprint = orm_obj.schema_fingerprint
        await session.delete(orm_obj)

        if self._audit is not None:
            await self._audit.record(
                session=session,
                event_type=AuditEventType.ROUTE_DELETED,
                entity_type="routing_rule",
                entity_id=schema_id,
                actor="api",
                detail={"fingerprint": fingerprint},
            )

        await session.commit()
        self.invalidate(schema_id)
        logger.info("routing_rule_deleted", schema_id=schema_id)
        return True

    async def ensure_by_fingerprint(
        self,
        session: AsyncSession,
        request: EnsureRoutingRequest,
    ) -> RoutingRule:
        """Return the rule for a given schema fingerprint, creating one if absent.

        Lookup path:
            1. Match ``routing_rules.schema_fingerprint`` against the request.
               If hit, return the existing rule unchanged (idempotent).
            2. Otherwise insert a new rule with a server-generated uuid7
               ``schema_id`` and the caller-supplied json_schema + defaults.

        Args:
            session: Active database session.
            request: Ensure payload with fingerprint + schema metadata.

        Returns:
            The pre-existing or newly created routing rule.
        """
        stmt = select(RoutingRuleORM).where(
            RoutingRuleORM.schema_fingerprint == request.fingerprint
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing is not None:
            rule = RoutingRule.model_validate(existing)
            self._cache[existing.schema_id] = rule
            logger.debug(
                "routing_rule_fingerprint_hit",
                schema_id=existing.schema_id,
                fingerprint=request.fingerprint,
            )
            return rule

        schema_id = str(uuid7())
        now = datetime.now(UTC)
        orm_obj = RoutingRuleORM(
            schema_id=schema_id,
            schema_fingerprint=request.fingerprint,
            active_model_id=None,
            fallback_chain=self._validated_chain(
                request.fallback_chain or list(DEFAULT_FALLBACK_CHAIN),
                schema_id=schema_id,
                actor="ensure_by_fingerprint",
            ),
            endpoint=None,
            json_schema=request.json_schema,
            gbnf_grammar=None,
            system_prompt=request.system_prompt,
            adapter_path=None,
            created_at=now,
            updated_at=now,
        )
        session.add(orm_obj)
        await session.flush()
        await session.refresh(orm_obj)

        if self._audit is not None:
            await self._audit.record(
                session=session,
                event_type=AuditEventType.ROUTE_UPDATED,
                entity_type="routing_rule",
                entity_id=schema_id,
                actor="inference-router",
                detail={
                    "auto_registered": True,
                    "fingerprint": request.fingerprint,
                },
            )

        await session.commit()

        logger.info(
            "routing_rule_auto_registered",
            schema_id=schema_id,
            fingerprint=request.fingerprint,
        )
        rule = RoutingRule.model_validate(orm_obj)
        self._cache[schema_id] = rule
        self._full_table_cache = None
        self._full_table_cache_time = None
        return rule

    async def update_route_for_deployment(
        self,
        session: AsyncSession,
        schema_id: str,
        model_id: UUID,
        endpoint: str,
        adapter_path: str | None = None,
    ) -> RoutingRule:
        """Update or create a routing rule when a model is deployed.

        Convenience method called by the deploy service to wire up the
        routing after a successful deployment.

        Args:
            session: Active database session.
            schema_id: The schema identifier.
            model_id: UUID of the newly deployed model.
            endpoint: In-cluster endpoint URL.
            adapter_path: MinIO path to PEFT adapter (for vLLM-hosted LoRA models).

        Returns:
            The updated RoutingRule.
        """
        request = UpdateRoutingRequest(
            active_model_id=model_id,
            endpoint=endpoint,
            adapter_path=adapter_path,
        )
        return await self.upsert_route(session, schema_id, request)
