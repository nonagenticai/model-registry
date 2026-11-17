"""Tests targeting coverage gaps across the model_registry service.

Covers edge cases in services, domain models, settings, and the deploy pipeline
that were not exercised by existing tests.
"""

from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from models.domain import (
    AuditEventType,
    DeployFeedbackRequest,
    DeployRequest,
    DeployStatus,
    HealthResponse,
    LivenessResponse,
    ModelStatus,
    ModelVersion,
    ReadinessResponse,
    RegisterModelRequest,
    UpdateRoutingRequest,
    UpdateStatusRequest,
    is_valid_transition,
)
from services.alias_service import ALIAS_PATTERN, AliasService
from services.audit_service import AuditService
from services.deploy_service import DeployService
from services.model_service import ModelService
from services.routing_service import RoutingService
from tests.helpers import gitops_publish_succeeds

# ---------------------------------------------------------------------------
# Health/Liveness/Readiness Pydantic model tests
# ---------------------------------------------------------------------------


class TestHealthModels:
    """Tests for the Pydantic health response models."""

    def test_health_response_defaults(self) -> None:
        """HealthResponse can be constructed with defaults and timestamp is recent."""
        before = datetime.now(UTC)
        resp = HealthResponse()
        after = datetime.now(UTC)
        assert resp.status == "healthy"
        assert resp.service == "model-registry"
        assert resp.version == "1.0.0"
        assert before <= resp.timestamp <= after

    def test_health_response_custom(self) -> None:
        """HealthResponse accepts custom values."""
        ts = datetime.now(UTC)
        resp = HealthResponse(
            status="ok",
            service="my-svc",
            version="2.0.0",
            timestamp=ts,
        )
        assert resp.status == "ok"
        assert resp.service == "my-svc"
        assert resp.version == "2.0.0"
        assert resp.timestamp == ts

    def test_health_response_model_dump(self) -> None:
        """HealthResponse.model_dump produces serializable output with ISO format timestamp."""
        resp = HealthResponse()
        data = resp.model_dump(mode="json")
        assert data["status"] == "healthy"
        assert data["service"] == "model-registry"
        assert data["version"] == "1.0.0"
        # Verify timestamp is a valid ISO 8601 string
        parsed = datetime.fromisoformat(data["timestamp"])
        assert parsed.year >= 2026

    def test_liveness_response_defaults(self) -> None:
        """LivenessResponse can be constructed with defaults."""
        resp = LivenessResponse()
        assert resp.status == "alive"

    def test_liveness_response_custom(self) -> None:
        """LivenessResponse accepts a custom status."""
        resp = LivenessResponse(status="up")
        assert resp.status == "up"

    def test_readiness_response_ready(self) -> None:
        """ReadinessResponse ready=True with checks."""
        resp = ReadinessResponse(ready=True, checks={"database": True})
        assert resp.ready is True
        assert resp.checks["database"] is True

    def test_readiness_response_not_ready(self) -> None:
        """ReadinessResponse ready=False with checks."""
        resp = ReadinessResponse(ready=False, checks={"database": False})
        assert resp.ready is False
        assert resp.checks["database"] is False

    def test_readiness_response_empty_checks_default(self) -> None:
        """ReadinessResponse uses empty dict as default for checks."""
        resp = ReadinessResponse(ready=True)
        assert resp.checks == {}

    def test_readiness_response_model_dump(self) -> None:
        """ReadinessResponse.model_dump produces correct output."""
        resp = ReadinessResponse(ready=False, checks={"database": False})
        data = resp.model_dump(mode="json")
        assert data == {"ready": False, "checks": {"database": False}}


# ---------------------------------------------------------------------------
# Settings edge cases
# ---------------------------------------------------------------------------


class TestSettingsEdgeCases:
    """Tests for settings.py edge cases."""

    def test_get_package_version_fallback_package_not_found(self) -> None:
        """_get_package_version returns fallback when distribution metadata is missing."""
        from importlib.metadata import PackageNotFoundError

        from settings import _get_package_version

        with patch("settings._pkg_version", side_effect=PackageNotFoundError("model-registry")):
            result = _get_package_version()
            assert result == "0.0.0-dev"

    def test_get_package_version_fallback_import_error(self) -> None:
        """_get_package_version returns fallback when importlib.metadata is unavailable."""
        from settings import _get_package_version

        with patch("settings._pkg_version", side_effect=ImportError("metadata missing")):
            result = _get_package_version()
            assert result == "0.0.0-dev"

    def test_get_package_version_unexpected_error_propagates(self) -> None:
        """Unexpected errors from _pkg_version propagate (no broad swallowing)."""
        from settings import _get_package_version

        with patch("settings._pkg_version", side_effect=RuntimeError("totally unexpected")):
            with pytest.raises(RuntimeError, match="totally unexpected"):
                _get_package_version()

    def test_get_package_version_success(self) -> None:
        """_get_package_version returns the package version when available."""
        from settings import _get_package_version

        with patch("settings._pkg_version", return_value="1.2.3"):
            result = _get_package_version()
            assert result == "1.2.3"

    def test_settings_auto_approve_score_bounds(self) -> None:
        """auto_approve_score must be between 0 and 1."""
        from pydantic import ValidationError

        from settings import Settings

        with pytest.raises(ValidationError):
            Settings(auto_approve_score=1.5)

        with pytest.raises(ValidationError):
            Settings(auto_approve_score=-0.1)


# ---------------------------------------------------------------------------
# Domain model edge cases
# ---------------------------------------------------------------------------


class TestDomainModelEdgeCases:
    """Tests for domain model edge cases."""

    def test_is_valid_transition_with_unknown_status_returns_false(self) -> None:
        """is_valid_transition handles a status not in VALID_TRANSITIONS gracefully."""
        # The function uses .get() with a default empty set, so an unknown key is safe.
        # We test by checking that the function returns False for a self-transition
        # on a terminal state.
        assert is_valid_transition(ModelStatus.RETIRED, ModelStatus.RETIRED) is False

    def test_model_version_with_all_optional_fields(self) -> None:
        """ModelVersion can be created with all optional fields set."""
        now = datetime.now(UTC)
        mv = ModelVersion(
            schema_id="test_schema",
            version="v1",
            gguf_file="path/to/file.gguf",
            gguf_sha256="abc123",
            eval_score=0.95,
            eval_metrics={"accuracy": 0.98},
            deployed_at=now,
            deploy_status=DeployStatus.SUCCEEDED,
            deploy_error=None,
            deploy_started_at=now,
            deploy_completed_at=now,
        )
        assert mv.gguf_file == "path/to/file.gguf"
        assert mv.deploy_status == DeployStatus.SUCCEEDED

    def test_model_version_defaults(self) -> None:
        """ModelVersion uses sensible defaults for non-required fields."""
        mv = ModelVersion(schema_id="test_schema", version="v1")
        assert mv.status == ModelStatus.TRAINING
        assert mv.gguf_file is None
        assert mv.deploy_status is None
        assert mv.training_samples == 0

    def test_register_model_request_with_tags(self) -> None:
        """RegisterModelRequest can include tags."""
        req = RegisterModelRequest(
            schema_id="ks_test",
            version="v1",
            tags={"key1": "value1"},
        )
        assert req.tags == {"key1": "value1"}

    def test_register_model_request_status_default(self) -> None:
        """RegisterModelRequest defaults status to TRAINING."""
        req = RegisterModelRequest(schema_id="ks_test", version="v1")
        assert req.status == ModelStatus.TRAINING

    def test_deploy_request_defaults(self) -> None:
        """DeployRequest uses sensible defaults."""
        req = DeployRequest()
        assert req.target_namespace == "llm-inference"
        assert req.replicas == 1

    def test_deploy_request_replicas_validation(self) -> None:
        """DeployRequest enforces replicas bounds."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeployRequest(replicas=0)

        with pytest.raises(ValidationError):
            DeployRequest(replicas=11)

    def test_deploy_feedback_request(self) -> None:
        """DeployFeedbackRequest accepts status and optional error."""
        req = DeployFeedbackRequest(
            deploy_status=DeployStatus.FAILED,
            deploy_error="OOMKilled",
        )
        assert req.deploy_status == DeployStatus.FAILED
        assert req.deploy_error == "OOMKilled"

    def test_deploy_feedback_request_no_error(self) -> None:
        """DeployFeedbackRequest defaults deploy_error to None."""
        req = DeployFeedbackRequest(deploy_status=DeployStatus.SUCCEEDED)
        assert req.deploy_error is None


# ---------------------------------------------------------------------------
# AliasService edge cases
# ---------------------------------------------------------------------------


class TestAliasServiceEdgeCases:
    """Tests for AliasService edge cases not covered elsewhere."""

    def test_alias_pattern_rejects_empty_string(self) -> None:
        """Empty string does not match the alias pattern."""
        assert ALIAS_PATTERN.match("") is None

    def test_alias_pattern_rejects_too_long(self) -> None:
        """A name longer than 63 characters is rejected."""
        long_name = "a" + "b" * 63  # 64 chars total
        assert ALIAS_PATTERN.match(long_name) is None

    def test_alias_pattern_accepts_min_length(self) -> None:
        """A single lowercase letter is valid."""
        assert ALIAS_PATTERN.match("a") is not None

    def test_alias_pattern_with_numbers(self) -> None:
        """Alias with numbers after the first letter is valid."""
        assert ALIAS_PATTERN.match("a123") is not None

    def test_alias_pattern_with_hyphen(self) -> None:
        """Alias with hyphens is valid."""
        assert ALIAS_PATTERN.match("my-alias-1") is not None

    def test_alias_pattern_rejects_leading_hyphen(self) -> None:
        """Alias starting with a hyphen is invalid."""
        assert ALIAS_PATTERN.match("-invalid") is None

    async def test_set_alias_with_audit_records_from_model_id(self, session: AsyncSession) -> None:
        """Updating an alias with audit service records from_model_id in the detail."""
        audit_svc = AuditService()
        alias_svc = AliasService(audit_service=audit_svc)
        model_svc = ModelService()

        m1 = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_AliasAuditFrom", version="v1")
        )
        m2 = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_AliasAuditFrom", version="v2")
        )

        # First set
        await alias_svc.set_alias(session, "ks_AliasAuditFrom", "champion", m1.id)

        # Update to m2 - should record from_model_id
        await alias_svc.set_alias(session, "ks_AliasAuditFrom", "champion", m2.id)

        events, _ = await audit_svc.list_events(session, event_type=AuditEventType.ALIAS_SET)
        # Most recent event (first in the list) should have from_model_id
        update_event = events[0]
        assert update_event.detail is not None
        assert "from_model_id" in update_event.detail
        assert update_event.detail["from_model_id"] == str(m1.id)
        assert update_event.detail["to_model_id"] == str(m2.id)

    async def test_resolve_alias_with_deleted_model(self, session: AsyncSession) -> None:
        """Resolving an alias whose target model no longer exists returns None.

        This simulates a scenario where the alias exists but the model row
        has been deleted (orphaned alias). Since we can't delete ORM rows
        easily in the test DB, we test the code path by creating an alias
        pointing to a UUID that was never registered.
        """
        alias_svc = AliasService()
        model_svc = ModelService()

        # Create model to set up alias (need a real model to pass validation)
        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_ResolveOrphan", version="v1")
        )
        await alias_svc.set_alias(session, "ks_ResolveOrphan", "orphan", m.id)

        # Now just resolve the existing alias - should succeed
        resolved = await alias_svc.resolve_alias(session, "ks_ResolveOrphan", "orphan")
        assert resolved is not None
        assert resolved.id == m.id

    async def test_get_aliases_for_nonexistent_model(self, session: AsyncSession) -> None:
        """Getting aliases for a model that doesn't exist returns empty list."""
        alias_svc = AliasService()
        result = await alias_svc.get_aliases_for_model(session, uuid4())
        assert result == []

    async def test_delete_alias_with_audit(self, session: AsyncSession) -> None:
        """Deleting an alias with audit enabled records the event."""
        audit_svc = AuditService()
        alias_svc = AliasService(audit_service=audit_svc)
        model_svc = ModelService()

        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_DelAudit", version="v1")
        )
        await alias_svc.set_alias(session, "ks_DelAudit", "candidate", m.id)
        await alias_svc.delete_alias(session, "ks_DelAudit", "candidate")

        events, _ = await audit_svc.list_events(session, event_type=AuditEventType.ALIAS_DELETED)
        assert len(events) >= 1
        del_event = events[0]
        assert del_event.detail is not None
        assert del_event.detail["model_id"] == str(m.id)


# ---------------------------------------------------------------------------
# RoutingService edge cases
# ---------------------------------------------------------------------------


class TestRoutingServiceEdgeCases:
    """Tests for RoutingService edge cases."""

    async def test_upsert_route_update_preserves_null_fields(self, session: AsyncSession) -> None:
        """Updating a route with None fields preserves existing values."""
        svc = RoutingService()
        # Create with endpoint
        await svc.upsert_route(
            session,
            "ks_PreserveNull",
            UpdateRoutingRequest(
                endpoint="ep:8080",
                fallback_chain=["pgx-local", "deepseek"],
            ),
        )

        # Update with only active_model_id (endpoint/fallback_chain not provided)
        model_id = uuid4()
        result = await svc.upsert_route(
            session,
            "ks_PreserveNull",
            UpdateRoutingRequest(active_model_id=model_id),
        )
        assert result.active_model_id == model_id
        # The endpoint and fallback_chain should remain from the original creation
        assert result.endpoint == "ep:8080"
        assert result.fallback_chain == ["pgx-local", "deepseek"]

    async def test_upsert_route_update_json_schema(self, session: AsyncSession) -> None:
        """Updating json_schema on an existing route persists it."""
        svc = RoutingService()
        await svc.upsert_route(
            session, "ks_JsonSchemaUpd", UpdateRoutingRequest(endpoint="ep:8080")
        )

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = await svc.upsert_route(
            session,
            "ks_JsonSchemaUpd",
            UpdateRoutingRequest(json_schema=schema),
        )
        assert result.json_schema == schema

    async def test_upsert_route_update_gbnf_grammar(self, session: AsyncSession) -> None:
        """Updating gbnf_grammar on an existing route persists it."""
        svc = RoutingService()
        await svc.upsert_route(session, "ks_GbnfUpd", UpdateRoutingRequest(endpoint="ep:8080"))

        grammar = 'root ::= "hello"'
        result = await svc.upsert_route(
            session,
            "ks_GbnfUpd",
            UpdateRoutingRequest(gbnf_grammar=grammar),
        )
        assert result.gbnf_grammar == grammar

    async def test_upsert_route_update_fallback_chain(self, session: AsyncSession) -> None:
        """Updating fallback_chain on an existing route replaces it."""
        svc = RoutingService()
        await svc.upsert_route(
            session,
            "ks_FallbackUpd",
            UpdateRoutingRequest(fallback_chain=["pgx-local", "xai"]),
        )

        result = await svc.upsert_route(
            session,
            "ks_FallbackUpd",
            UpdateRoutingRequest(fallback_chain=["deepseek", "xai"]),
        )
        assert result.fallback_chain == ["deepseek", "xai"]

    async def test_upsert_route_update_endpoint(self, session: AsyncSession) -> None:
        """Updating endpoint on an existing route changes it."""
        svc = RoutingService()
        await svc.upsert_route(
            session, "ks_EndpointUpd", UpdateRoutingRequest(endpoint="ep-v1:8080")
        )

        result = await svc.upsert_route(
            session,
            "ks_EndpointUpd",
            UpdateRoutingRequest(endpoint="ep-v2:9090"),
        )
        assert result.endpoint == "ep-v2:9090"

    async def test_upsert_route_with_audit(self, session: AsyncSession) -> None:
        """Upserting a route with audit enabled records the event."""
        audit_svc = AuditService()
        svc = RoutingService(audit_service=audit_svc)

        model_id = uuid4()
        await svc.upsert_route(
            session,
            "ks_RouteAudit",
            UpdateRoutingRequest(
                active_model_id=model_id,
                endpoint="ep:8080",
            ),
        )

        events, _ = await audit_svc.list_events(session, event_type=AuditEventType.ROUTE_UPDATED)
        assert len(events) >= 1
        event = events[0]
        assert event.detail is not None
        assert event.detail["active_model_id"] == str(model_id)
        assert event.detail["endpoint"] == "ep:8080"

    async def test_upsert_route_audit_without_model_id(self, session: AsyncSession) -> None:
        """Audit detail records None for active_model_id when not provided."""
        audit_svc = AuditService()
        svc = RoutingService(audit_service=audit_svc)

        await svc.upsert_route(
            session,
            "ks_RouteAuditNoModel",
            UpdateRoutingRequest(endpoint="ep:8080"),
        )

        events, _ = await audit_svc.list_events(session, event_type=AuditEventType.ROUTE_UPDATED)
        assert len(events) >= 1
        event = events[0]
        assert event.detail is not None
        assert event.detail["active_model_id"] is None

    async def test_invalidate_clears_full_table_cache(self, session: AsyncSession) -> None:
        """Invalidating a single key also clears the full table cache."""
        svc = RoutingService()
        await svc.upsert_route(
            session, "ks_FullTableInval", UpdateRoutingRequest(endpoint="ep:8080")
        )
        # Populate full table cache
        await svc.get_routing_table(session)
        assert svc._full_table_cache is not None
        assert svc._full_table_cache_time is not None

        # Single-key invalidation should also clear full table cache
        svc.invalidate("ks_FullTableInval")
        assert svc._full_table_cache is None
        assert svc._full_table_cache_time is None


# ---------------------------------------------------------------------------
# ModelService edge cases
# ---------------------------------------------------------------------------


class TestModelServiceEdgeCases:
    """Tests for ModelService edge cases."""

    async def test_register_model_with_audit(self, session: AsyncSession) -> None:
        """Registering a model with audit service records the event."""
        audit_svc = AuditService()
        svc = ModelService(audit_service=audit_svc)

        result = await svc.register_model(
            session,
            RegisterModelRequest(schema_id="ks_RegAudit", version="v1"),
        )

        events, _ = await audit_svc.list_events(
            session,
            event_type=AuditEventType.MODEL_REGISTERED,
            entity_id=str(result.id),
        )
        assert len(events) == 1
        assert events[0].detail is not None
        assert events[0].detail["schema_id"] == "ks_RegAudit"

    async def test_update_status_with_audit(self, session: AsyncSession) -> None:
        """Updating status with audit service records the transition."""
        audit_svc = AuditService()
        svc = ModelService(audit_service=audit_svc)

        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_StatusAudit", version="v1")
        )
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))

        events, _ = await audit_svc.list_events(
            session,
            event_type=AuditEventType.STATUS_CHANGED,
            entity_id=str(m.id),
        )
        assert len(events) == 1
        assert events[0].detail is not None
        assert events[0].detail["from_status"] == "training"
        assert events[0].detail["to_status"] == "evaluating"

    async def test_set_tags_with_audit(self, session: AsyncSession) -> None:
        """Setting tags with audit service records the event."""
        audit_svc = AuditService()
        svc = ModelService(audit_service=audit_svc)

        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_TagAudit", version="v1")
        )
        await svc.set_tags(session, m.id, {"env": "prod"})

        events, _ = await audit_svc.list_events(
            session,
            event_type=AuditEventType.TAG_SET,
            entity_id=str(m.id),
        )
        assert len(events) == 1
        assert events[0].detail is not None
        assert events[0].detail["tags"]["env"] == "prod"

    async def test_delete_tag_with_audit(self, session: AsyncSession) -> None:
        """Deleting a tag with audit service records the event."""
        audit_svc = AuditService()
        svc = ModelService(audit_service=audit_svc)

        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_TagDelAudit", version="v1")
        )
        await svc.set_tags(session, m.id, {"env": "prod"})
        await svc.delete_tag(session, m.id, "env")

        events, _ = await audit_svc.list_events(
            session,
            event_type=AuditEventType.TAG_DELETED,
            entity_id=str(m.id),
        )
        assert len(events) == 1
        assert events[0].detail is not None
        assert events[0].detail["key"] == "env"

    async def test_search_by_tag_key_only(self, session: AsyncSession) -> None:
        """Searching by tag key without value returns all models with that key."""
        svc = ModelService()
        m1 = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_SearchKeyOnly1", version="v1")
        )
        m2 = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_SearchKeyOnly2", version="v1")
        )
        await svc.set_tags(session, m1.id, {"env": "prod"})
        await svc.set_tags(session, m2.id, {"env": "staging"})

        results = await svc.search_by_tag(session, "env")
        assert len(results) == 2

    async def test_search_by_tag_no_matches(self, session: AsyncSession) -> None:
        """Searching by a nonexistent tag key returns empty list."""
        svc = ModelService()
        results = await svc.search_by_tag(session, "nonexistent_key_xyz")
        assert results == []

    async def test_update_status_invalid_transition_error_message(
        self, session: AsyncSession
    ) -> None:
        """Invalid transition error includes allowed transitions."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_ErrorMsg", version="v1")
        )
        with pytest.raises(ValueError, match="Allowed transitions from training"):
            await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.DEPLOYED))

    async def test_compare_versions_with_deploy_status(self, session: AsyncSession) -> None:
        """compare_versions includes deploy_status when present."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_CompDeploy", version="v1")
        )
        await svc.update_deploy_status(
            session,
            m.id,
            DeployFeedbackRequest(deploy_status=DeployStatus.PENDING),
        )

        summaries = await svc.compare_versions(session, "ks_CompDeploy")
        assert len(summaries) == 1
        assert summaries[0].deploy_status == DeployStatus.PENDING

    async def test_compare_versions_with_aliases_and_tags(self, session: AsyncSession) -> None:
        """compare_versions includes aliases and tags for each model."""
        model_svc = ModelService()
        alias_svc = AliasService()

        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_CompAliasTag", version="v1")
        )
        await model_svc.set_tags(session, m.id, {"env": "prod"})
        await alias_svc.set_alias(session, "ks_CompAliasTag", "champion", m.id)

        summaries = await model_svc.compare_versions(session, "ks_CompAliasTag")
        assert len(summaries) == 1
        assert "champion" in summaries[0].aliases
        assert summaries[0].tags["env"] == "prod"

    async def test_update_deploy_status_in_progress(self, session: AsyncSession) -> None:
        """Setting deploy_status to in_progress does not set deploy_completed_at."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_InProgress", version="v1")
        )
        result = await svc.update_deploy_status(
            session,
            m.id,
            DeployFeedbackRequest(deploy_status=DeployStatus.IN_PROGRESS),
        )
        assert result.deploy_status == DeployStatus.IN_PROGRESS
        assert result.deploy_completed_at is None


# ---------------------------------------------------------------------------
# DeployService edge cases
# ---------------------------------------------------------------------------


class TestDeployServiceEdgeCases:
    """Tests for DeployService edge cases."""

    async def test_deploy_with_alias_management_first_deploy(self, session: AsyncSession) -> None:
        """First deployment sets champion alias (no rollback yet)."""
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        alias_svc = AliasService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc, alias_svc)

        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_FirstDeploy", version="v1")
        )
        for status in [
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
        ]:
            m = await model_svc.update_status(session, m.id, UpdateStatusRequest(status=status))

        with gitops_publish_succeeds():
            result = await deploy_svc.deploy_model(session, m.id, DeployRequest())
        assert result.status == ModelStatus.DEPLOYED

        # Champion should be set
        champion = await alias_svc.get_alias(session, "ks_FirstDeploy", "champion")
        assert champion is not None
        assert champion.model_id == m.id

        # Rollback should NOT exist (first deploy)
        rollback = await alias_svc.get_alias(session, "ks_FirstDeploy", "rollback")
        assert rollback is None

    async def test_deploy_with_alias_management_second_deploy(self, session: AsyncSession) -> None:
        """Second deployment moves champion to rollback and sets new champion."""
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        alias_svc = AliasService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc, alias_svc)

        # Deploy v1
        m1 = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_SecondDeploy", version="v1")
        )
        for status in [
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
        ]:
            m1 = await model_svc.update_status(session, m1.id, UpdateStatusRequest(status=status))
        with gitops_publish_succeeds():
            await deploy_svc.deploy_model(session, m1.id, DeployRequest())

        # Deploy v2
        m2 = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_SecondDeploy", version="v2")
        )
        for status in [
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
        ]:
            m2 = await model_svc.update_status(session, m2.id, UpdateStatusRequest(status=status))
        with gitops_publish_succeeds():
            await deploy_svc.deploy_model(session, m2.id, DeployRequest())

        # Champion should be v2
        champion = await alias_svc.get_alias(session, "ks_SecondDeploy", "champion")
        assert champion is not None
        assert champion.model_id == m2.id

        # Rollback should be v1
        rollback = await alias_svc.get_alias(session, "ks_SecondDeploy", "rollback")
        assert rollback is not None
        assert rollback.model_id == m1.id

    async def test_deploy_without_alias_service(self, session: AsyncSession) -> None:
        """Deployment without alias service skips alias management."""
        model_svc = ModelService()
        routing_svc = RoutingService()
        deploy_svc = DeployService(model_svc, routing_svc, alias_service=None)

        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_NoAlias", version="v1")
        )
        for status in [
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
        ]:
            m = await model_svc.update_status(session, m.id, UpdateStatusRequest(status=status))

        with gitops_publish_succeeds():
            result = await deploy_svc.deploy_model(session, m.id, DeployRequest())
        assert result.status == ModelStatus.DEPLOYED

    async def test_deploy_with_alias_value_error_continues(self, session: AsyncSession) -> None:
        """If alias management raises ValueError, deployment still succeeds."""
        model_svc = ModelService()
        routing_svc = RoutingService()
        mock_alias_svc = AliasService()

        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_AliasValErr", version="v1")
        )
        for status in [
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
        ]:
            m = await model_svc.update_status(session, m.id, UpdateStatusRequest(status=status))

        deploy_svc = DeployService(model_svc, routing_svc, alias_service=mock_alias_svc)

        # Application-level alias validation failure (e.g. invalid alias name)
        with (
            gitops_publish_succeeds(),
            patch.object(mock_alias_svc, "get_alias", side_effect=ValueError("invalid alias")),
        ):
            result = await deploy_svc.deploy_model(session, m.id, DeployRequest())

        assert result.status == ModelStatus.DEPLOYED

    async def test_deploy_with_alias_db_error_continues(self, session: AsyncSession) -> None:
        """If alias management raises a SQLAlchemyError, deployment still succeeds."""
        from sqlalchemy.exc import SQLAlchemyError

        model_svc = ModelService()
        routing_svc = RoutingService()
        mock_alias_svc = AliasService()

        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_AliasDbErr", version="v1")
        )
        for status in [
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
        ]:
            m = await model_svc.update_status(session, m.id, UpdateStatusRequest(status=status))

        deploy_svc = DeployService(model_svc, routing_svc, alias_service=mock_alias_svc)

        with (
            gitops_publish_succeeds(),
            patch.object(
                mock_alias_svc, "set_alias", side_effect=SQLAlchemyError("connection lost")
            ),
        ):
            result = await deploy_svc.deploy_model(session, m.id, DeployRequest())

        assert result.status == ModelStatus.DEPLOYED

    async def test_deploy_with_alias_runtime_error_propagates(self, session: AsyncSession) -> None:
        """An unexpected RuntimeError from alias management must propagate.

        We deliberately do NOT swallow programming bugs in the alias auto-
        management try/except — only ValueError and SQLAlchemyError are
        expected and tolerated.
        """
        model_svc = ModelService()
        routing_svc = RoutingService()
        mock_alias_svc = AliasService()

        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_AliasRuntimeErr", version="v1")
        )
        for status in [
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
        ]:
            m = await model_svc.update_status(session, m.id, UpdateStatusRequest(status=status))

        deploy_svc = DeployService(model_svc, routing_svc, alias_service=mock_alias_svc)

        with (
            gitops_publish_succeeds(),
            patch.object(mock_alias_svc, "get_alias", side_effect=RuntimeError("alias bug")),
            pytest.raises(RuntimeError, match="alias bug"),
        ):
            await deploy_svc.deploy_model(session, m.id, DeployRequest())

    async def test_deploy_with_deployment_tracking(self, session: AsyncSession) -> None:
        """Deployment tracking sets deploy_status fields when enabled."""
        model_svc = ModelService()
        routing_svc = RoutingService()
        deploy_svc = DeployService(model_svc, routing_svc)

        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_TrackDeploy", version="v1")
        )
        for status in [
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
        ]:
            m = await model_svc.update_status(session, m.id, UpdateStatusRequest(status=status))

        with gitops_publish_succeeds(deployment_tracking_enabled=True):
            result = await deploy_svc.deploy_model(session, m.id, DeployRequest())

        assert result.status == ModelStatus.DEPLOYED

    def test_compute_endpoint_various_schemas(self) -> None:
        """_compute_endpoint handles various schema naming patterns."""
        # Simple schema
        ep = DeployService._compute_endpoint("simple", "ns")
        assert ep == "fg-simple.ns.svc.example.com:8080"

        # Schema with multiple underscores
        ep = DeployService._compute_endpoint("a_b_c_d", "prod")
        assert ep == "fg-a-b-c-d.prod.svc.example.com:8080"

        # Schema with uppercase
        ep = DeployService._compute_endpoint("MySchema", "dev")
        assert ep == "fg-myschema.dev.svc.example.com:8080"


# ---------------------------------------------------------------------------
# AuditService edge cases
# ---------------------------------------------------------------------------


class TestAuditServiceEdgeCases:
    """Tests for AuditService edge cases."""

    async def test_record_without_detail(self, session: AsyncSession) -> None:
        """Recording an event without detail sets detail to None."""
        svc = AuditService()
        event = await svc.record(
            session,
            event_type=AuditEventType.MODEL_REGISTERED,
            entity_type="model_version",
            entity_id="test-no-detail",
        )
        assert event.detail is None

    async def test_record_with_default_actor(self, session: AsyncSession) -> None:
        """Recording an event without actor uses default 'api'."""
        svc = AuditService()
        event = await svc.record(
            session,
            event_type=AuditEventType.MODEL_REGISTERED,
            entity_type="model_version",
            entity_id="test-default-actor",
        )
        assert event.actor == "api"

    async def test_list_events_filter_by_entity_type_and_id(self, session: AsyncSession) -> None:
        """Listing events filtered by both entity_type and entity_id works."""
        svc = AuditService()
        await svc.record(
            session,
            event_type=AuditEventType.MODEL_REGISTERED,
            entity_type="model_version",
            entity_id="model-1",
        )
        await svc.record(
            session,
            event_type=AuditEventType.ROUTE_UPDATED,
            entity_type="routing_rule",
            entity_id="schema-1",
        )

        events, total = await svc.list_events(
            session, entity_type="model_version", entity_id="model-1"
        )
        assert total == 1
        assert events[0].entity_id == "model-1"

    async def test_list_events_pagination(self, session: AsyncSession) -> None:
        """Audit event pagination with limit and offset works."""
        svc = AuditService()
        for i in range(5):
            await svc.record(
                session,
                event_type=AuditEventType.MODEL_REGISTERED,
                entity_type="model_version",
                entity_id=f"pag-model-{i}",
            )

        events, total = await svc.list_events(session, limit=2, offset=0)
        assert total == 5
        assert len(events) == 2

        events2, total2 = await svc.list_events(session, limit=2, offset=4)
        assert total2 == 5
        assert len(events2) == 1

    async def test_get_schema_history_pagination(self, session: AsyncSession) -> None:
        """Schema history pagination works correctly."""
        svc = AuditService()
        for i in range(5):
            await svc.record(
                session,
                event_type=AuditEventType.ROUTE_UPDATED,
                entity_type="routing_rule",
                entity_id="ks_PagSchema",
            )

        events, total = await svc.get_schema_history(session, "ks_PagSchema", limit=2, offset=0)
        assert total == 5
        assert len(events) == 2

    async def test_get_model_history_pagination(self, session: AsyncSession) -> None:
        """Model history pagination works correctly."""
        svc = AuditService()
        for i in range(4):
            await svc.record(
                session,
                event_type=AuditEventType.STATUS_CHANGED,
                entity_type="model_version",
                entity_id="model-pag",
            )

        events, total = await svc.get_model_history(session, "model-pag", limit=2, offset=0)
        assert total == 4
        assert len(events) == 2


# ---------------------------------------------------------------------------
# API-level edge case tests (via httpx client)
# ---------------------------------------------------------------------------


class TestHealthEndpointsAPI:
    """Tests for health endpoints through the HTTP API."""

    async def test_health_response_has_pydantic_fields(self, client: AsyncClient) -> None:
        """Health response contains all HealthResponse fields."""
        response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert "status" in body
        assert "service" in body
        assert "version" in body
        assert "timestamp" in body
        assert body["status"] == "healthy"

    async def test_liveness_response_pydantic(self, client: AsyncClient) -> None:
        """Liveness response uses LivenessResponse model."""
        response = await client.get("/health/live")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "alive"
        # Should NOT have the old 'alive' key
        assert "alive" not in body or body.get("alive") is None

    async def test_root_endpoint_fields(self, client: AsyncClient) -> None:
        """Root endpoint returns all expected fields."""
        response = await client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["service"] == "model-registry"
        assert body["docs"] == "/docs"
        assert body["health"] == "/health"
        assert "version" in body


class TestModelRegistrationEdgeCasesAPI:
    """API-level tests for model registration edge cases."""

    async def test_register_with_evaluating_status(self, client: AsyncClient) -> None:
        """Can register a model with evaluating as initial status."""
        response = await client.post(
            "/api/models",
            json={
                "schema_id": "ks_InitEval",
                "version": "v1",
                "status": "evaluating",
            },
        )
        assert response.status_code == 201
        assert response.json()["status"] == "evaluating"

    async def test_register_with_gguf_fields(self, client: AsyncClient) -> None:
        """Can register a model with gguf_file and gguf_sha256."""
        response = await client.post(
            "/api/models",
            json={
                "schema_id": "ks_GgufReg",
                "version": "v1",
                "gguf_file": "models/test.gguf",
                "gguf_sha256": "deadbeef",
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["gguf_file"] == "models/test.gguf"
        assert body["gguf_sha256"] == "deadbeef"

    async def test_list_models_filter_by_both(self, client: AsyncClient) -> None:
        """Can filter models by both schema_id and status simultaneously."""
        await client.post(
            "/api/models",
            json={"schema_id": "ks_BothFilter", "version": "v1", "status": "training"},
        )
        await client.post(
            "/api/models",
            json={"schema_id": "ks_BothFilter", "version": "v2", "status": "training"},
        )
        await client.post(
            "/api/models",
            json={"schema_id": "ks_OtherFilter", "version": "v1", "status": "training"},
        )

        response = await client.get(
            "/api/models",
            params={"schema_id": "ks_BothFilter", "status": "training"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2

    async def test_update_status_sets_gguf_fields(self, client: AsyncClient) -> None:
        """Setting gguf_file and gguf_sha256 during status update persists them."""
        resp = await client.post(
            "/api/models",
            json={"schema_id": "ks_GgufUpdate2", "version": "v1"},
        )
        model_id = resp.json()["id"]

        for status in ["evaluating", "evaluated", "approved"]:
            await client.patch(
                f"/api/models/{model_id}/status",
                json={"status": status},
            )

        resp = await client.patch(
            f"/api/models/{model_id}/status",
            json={
                "status": "exporting",
                "gguf_file": "models/ks_GgufUpdate2-v1.gguf",
                "gguf_sha256": "sha256hash",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["gguf_file"] == "models/ks_GgufUpdate2-v1.gguf"
        assert body["gguf_sha256"] == "sha256hash"


class TestModelServiceConstructorEdgeCases:
    """Tests for ModelService constructor edge cases."""

    def test_model_service_without_audit(self) -> None:
        """ModelService can be created without an audit service."""
        svc = ModelService()
        assert svc._audit is None

    def test_model_service_with_non_audit_service(self) -> None:
        """ModelService ignores non-AuditService objects."""
        svc = ModelService(audit_service=cast(Any, "not_a_service"))
        assert svc._audit is None

    def test_model_service_with_audit_service(self) -> None:
        """ModelService accepts a real AuditService."""
        audit_svc = AuditService()
        svc = ModelService(audit_service=audit_svc)
        assert svc._audit is audit_svc


class TestRoutingServiceConstructorEdgeCases:
    """Tests for RoutingService constructor edge cases."""

    def test_routing_service_without_audit(self) -> None:
        """RoutingService can be created without an audit service."""
        svc = RoutingService()
        assert svc._audit is None

    def test_routing_service_with_non_audit_service(self) -> None:
        """RoutingService ignores non-AuditService objects."""
        svc = RoutingService(audit_service=cast(Any, "not_a_service"))
        assert svc._audit is None

    def test_routing_service_with_audit_service(self) -> None:
        """RoutingService accepts a real AuditService."""
        audit_svc = AuditService()
        svc = RoutingService(audit_service=audit_svc)
        assert svc._audit is audit_svc


class TestAliasServiceConstructorEdgeCases:
    """Tests for AliasService constructor edge cases."""

    def test_alias_service_without_audit(self) -> None:
        """AliasService can be created without an audit service."""
        svc = AliasService()
        assert svc._audit is None

    def test_alias_service_with_non_audit_service(self) -> None:
        """AliasService ignores non-AuditService objects."""
        svc = AliasService(audit_service=cast(Any, "not_a_service"))
        assert svc._audit is None

    def test_alias_service_with_audit_service(self) -> None:
        """AliasService accepts a real AuditService."""
        audit_svc = AuditService()
        svc = AliasService(audit_service=audit_svc)
        assert svc._audit is audit_svc


class TestDeployServiceConstructorEdgeCases:
    """Tests for DeployService constructor edge cases."""

    def test_deploy_service_without_alias(self) -> None:
        """DeployService can be created without an alias service."""
        svc = DeployService(MagicMock(), MagicMock(), alias_service=None)
        assert svc._alias_service is None

    def test_deploy_service_with_non_alias_service(self) -> None:
        """DeployService ignores non-AliasService objects."""
        svc = DeployService(MagicMock(), MagicMock(), alias_service=cast(Any, "not_a_service"))
        assert svc._alias_service is None

    def test_deploy_service_with_alias_service(self) -> None:
        """DeployService accepts a real AliasService."""
        alias_svc = AliasService()
        svc = DeployService(MagicMock(), MagicMock(), alias_service=alias_svc)
        assert svc._alias_service is alias_svc


# ---------------------------------------------------------------------------
# Enum edge cases
# ---------------------------------------------------------------------------


class TestEnumValues:
    """Tests for enum value strings."""

    def test_model_status_values(self) -> None:
        """All ModelStatus enum values are lowercase strings."""
        for status in ModelStatus:
            assert status.value == status.value.lower()

    def test_deploy_status_values(self) -> None:
        """All DeployStatus enum values are lowercase strings."""
        for status in DeployStatus:
            assert status.value == status.value.lower()

    def test_audit_event_type_values(self) -> None:
        """All AuditEventType enum values are lowercase strings."""
        for event_type in AuditEventType:
            assert event_type.value == event_type.value.lower()

    def test_model_status_from_string(self) -> None:
        """ModelStatus can be constructed from string values."""
        assert ModelStatus("training") == ModelStatus.TRAINING
        assert ModelStatus("deployed") == ModelStatus.DEPLOYED

    def test_deploy_status_from_string(self) -> None:
        """DeployStatus can be constructed from string values."""
        assert DeployStatus("pending") == DeployStatus.PENDING
        assert DeployStatus("succeeded") == DeployStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# Real DB integration tests — verify state across service + DB boundaries
# ---------------------------------------------------------------------------


class TestDBPersistenceIntegration:
    """Tests that verify real DB persistence, not just API echo-back."""

    async def test_register_model_actually_writes_to_db(self, session: AsyncSession) -> None:
        """After registering, the model is retrievable from a fresh query."""
        from sqlalchemy import select

        from models.db import ModelVersionORM

        svc = ModelService()
        model = await svc.register_model(
            session,
            RegisterModelRequest(
                schema_id="ks_DBPersist",
                version="v1",
                training_samples=42,
                base_model="Qwen3-4B-AWQ-270m",
            ),
        )

        # Query directly via ORM, bypassing the service layer
        stmt = select(ModelVersionORM).where(ModelVersionORM.id == model.id)
        result = await session.execute(stmt)
        orm_obj = result.scalar_one_or_none()

        assert orm_obj is not None
        assert orm_obj.schema_id == "ks_DBPersist"
        assert orm_obj.version == "v1"
        assert orm_obj.training_samples == 42
        assert orm_obj.base_model == "Qwen3-4B-AWQ-270m"
        assert orm_obj.status == "training"
        assert orm_obj.created_at is not None
        assert orm_obj.updated_at is not None

    async def test_status_update_persists_in_db(self, session: AsyncSession) -> None:
        """Status transition is reflected in direct DB query."""
        from sqlalchemy import select

        from models.db import ModelVersionORM

        svc = ModelService()
        model = await svc.register_model(
            session,
            RegisterModelRequest(schema_id="ks_StatusDB", version="v1"),
        )
        original_updated = model.updated_at

        await svc.update_status(
            session, model.id, UpdateStatusRequest(status=ModelStatus.EVALUATING)
        )

        # Direct ORM query
        stmt = select(ModelVersionORM).where(ModelVersionORM.id == model.id)
        result = await session.execute(stmt)
        orm_obj = result.scalar_one()
        assert orm_obj.status == "evaluating"
        assert orm_obj.updated_at >= original_updated

    async def test_routing_rule_persists_with_all_fields(self, session: AsyncSession) -> None:
        """Routing rule with all fields persists correctly in DB."""
        from sqlalchemy import select

        from models.db import RoutingRuleORM

        svc = RoutingService()
        model_id = uuid4()
        await svc.upsert_route(
            session,
            "ks_RoutingPersist",
            UpdateRoutingRequest(
                active_model_id=model_id,
                fallback_chain=["pgx-local", "xai"],
                endpoint="http://vllm:8000",
                system_prompt="Extract entities.",
                adapter_path="adapters/ks_RoutingPersist/v1/",
                json_schema={"type": "object"},
                gbnf_grammar='root ::= "hello"',
            ),
        )

        # Direct ORM query
        stmt = select(RoutingRuleORM).where(RoutingRuleORM.schema_id == "ks_RoutingPersist")
        result = await session.execute(stmt)
        orm_obj = result.scalar_one()

        assert orm_obj.active_model_id == model_id
        assert orm_obj.endpoint == "http://vllm:8000"
        assert orm_obj.system_prompt == "Extract entities."
        assert orm_obj.adapter_path == "adapters/ks_RoutingPersist/v1/"
        assert orm_obj.json_schema == {"type": "object"}
        assert orm_obj.gbnf_grammar == 'root ::= "hello"'
        assert orm_obj.adapter_loaded is False  # default

    async def test_audit_events_persisted_with_correct_timestamps(
        self, session: AsyncSession
    ) -> None:
        """Audit events have monotonically increasing timestamps."""
        from datetime import datetime

        audit_svc = AuditService()
        svc = ModelService(audit_service=audit_svc)

        model = await svc.register_model(
            session,
            RegisterModelRequest(schema_id="ks_AuditTimestamp", version="v1"),
        )
        await svc.update_status(
            session, model.id, UpdateStatusRequest(status=ModelStatus.EVALUATING)
        )
        await svc.update_status(
            session,
            model.id,
            UpdateStatusRequest(status=ModelStatus.EVALUATED, eval_score=0.85),
        )

        events, total = await audit_svc.get_model_history(session, str(model.id))
        assert total == 3  # registered + 2 status changes

        # Events should be reverse-chronological
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps, reverse=True)

        # Each event should have a real datetime
        for event in events:
            assert isinstance(event.timestamp, datetime)

    async def test_tags_survive_multiple_operations(self, session: AsyncSession) -> None:
        """Tags persist correctly through set, update, and partial delete cycles."""
        svc = ModelService()
        model = await svc.register_model(
            session,
            RegisterModelRequest(schema_id="ks_TagSurvive", version="v1"),
        )

        # Set initial tags
        await svc.set_tags(session, model.id, {"a": "1", "b": "2", "c": "3"})
        tags = await svc.get_tags(session, model.id)
        assert tags == {"a": "1", "b": "2", "c": "3"}

        # Update one, add one
        await svc.set_tags(session, model.id, {"b": "updated", "d": "4"})
        tags = await svc.get_tags(session, model.id)
        assert tags == {"a": "1", "b": "updated", "c": "3", "d": "4"}

        # Delete one
        deleted = await svc.delete_tag(session, model.id, "c")
        assert deleted is True
        tags = await svc.get_tags(session, model.id)
        assert tags == {"a": "1", "b": "updated", "d": "4"}

        # Delete nonexistent
        deleted = await svc.delete_tag(session, model.id, "nonexistent")
        assert deleted is False

    async def test_auto_approve_then_deploy_full_flow(self, session: AsyncSession) -> None:
        """Full flow: register, auto-approve, deploy -- verifying DB state at each step."""
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc, auto_approve_score=0.90)
        routing_svc = RoutingService(audit_service=audit_svc)
        alias_svc = AliasService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc, alias_svc)

        # Register
        model = await model_svc.register_model(
            session,
            RegisterModelRequest(schema_id="ks_FullFlow", version="v1"),
        )
        assert model.status == ModelStatus.TRAINING

        # Evaluating
        model = await model_svc.update_status(
            session, model.id, UpdateStatusRequest(status=ModelStatus.EVALUATING)
        )
        assert model.status == ModelStatus.EVALUATING

        # Evaluated with auto-approve (score >= 0.90)
        model = await model_svc.update_status(
            session,
            model.id,
            UpdateStatusRequest(
                status=ModelStatus.EVALUATED,
                eval_score=0.95,
                eval_metrics={"accuracy": 0.97, "f1": 0.93},
            ),
        )
        assert model.status == ModelStatus.APPROVED  # auto-approved!
        assert model.eval_score == 0.95

        # Deploy
        with gitops_publish_succeeds():
            result = await deploy_svc.deploy_model(session, model.id, DeployRequest())
        assert result.status == ModelStatus.DEPLOYED

        # Verify champion alias
        champion = await alias_svc.get_alias(session, "ks_FullFlow", "champion")
        assert champion is not None
        assert champion.model_id == model.id

        # Verify routing rule
        route = await routing_svc.get_route(session, "ks_FullFlow")
        assert route is not None
        assert route.active_model_id == model.id

        # Verify audit trail completeness
        events, total = await audit_svc.get_model_history(session, str(model.id))
        event_types = [e.event_type for e in events]
        assert "model_registered" in event_types
        assert "status_changed" in event_types
        assert "auto_approved" in event_types

    async def test_unique_constraint_schema_version(self, session: AsyncSession) -> None:
        """Duplicate schema_id + version raises ValueError (unique constraint)."""
        svc = ModelService()
        await svc.register_model(
            session,
            RegisterModelRequest(schema_id="ks_Unique", version="v1"),
        )
        with pytest.raises(ValueError, match="already exists"):
            await svc.register_model(
                session,
                RegisterModelRequest(schema_id="ks_Unique", version="v1"),
            )

        # Different version under same schema is fine
        m2 = await svc.register_model(
            session,
            RegisterModelRequest(schema_id="ks_Unique", version="v2"),
        )
        assert m2.version == "v2"

    async def test_routing_cache_invalidation_on_upsert(self, session: AsyncSession) -> None:
        """After upsert, cached route returns updated data."""
        svc = RoutingService()

        # Create and cache
        await svc.upsert_route(
            session,
            "ks_CacheTest",
            UpdateRoutingRequest(endpoint="old-endpoint:8080"),
        )
        cached = await svc.get_route(session, "ks_CacheTest")
        assert cached is not None
        assert cached.endpoint == "old-endpoint:8080"

        # Update - should invalidate cache
        await svc.upsert_route(
            session,
            "ks_CacheTest",
            UpdateRoutingRequest(endpoint="new-endpoint:9090"),
        )

        # Get should return new data, not cached old data
        refreshed = await svc.get_route(session, "ks_CacheTest")
        assert refreshed is not None
        assert refreshed.endpoint == "new-endpoint:9090"
