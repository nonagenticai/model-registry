"""Unit tests for service layer classes.

Tests ModelService, RoutingService, and DeployService directly
(not through the HTTP API) to cover service-layer logic and edge cases.
"""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from models.domain import (
    VALID_TRANSITIONS,
    DeployRequest,
    ModelStatus,
    ModelVersion,
    QualityResponse,
    RegisterModelRequest,
    UpdateRoutingRequest,
    UpdateStatusRequest,
    is_valid_transition,
)
from services.alias_service import AliasService
from services.audit_service import AuditService
from services.deploy_service import DeployService
from services.errors import DeployError
from services.model_service import ModelService
from services.routing_service import RoutingService
from tests.helpers import gitops_publish_succeeds, gitops_push_fails

# ---------------------------------------------------------------------------
# ModelService unit tests
# ---------------------------------------------------------------------------


class TestModelServiceRegister:
    """Tests for ModelService.register_model."""

    async def test_register_model_persists_all_fields(self, session: AsyncSession) -> None:
        """Registering a model persists all fields and can be read back from DB."""
        svc = ModelService()
        request = RegisterModelRequest(
            schema_id="ks_TestRegister",
            version="v1",
            base_model="Qwen3-4B-AWQ-270m",
            quantization="Q4_K_M",
            training_samples=100,
        )
        result = await svc.register_model(session, request)

        assert result.schema_id == "ks_TestRegister"
        assert result.version == "v1"
        assert result.status == ModelStatus.TRAINING
        assert result.base_model == "Qwen3-4B-AWQ-270m"
        assert result.quantization == "Q4_K_M"
        assert result.training_samples == 100
        assert result.created_at is not None
        assert result.updated_at is not None

        # Verify round-trip: re-read from DB to confirm persistence
        refetched = await svc.get_model(session, result.id)
        assert refetched is not None
        assert refetched.schema_id == "ks_TestRegister"
        assert refetched.version == "v1"
        assert refetched.base_model == "Qwen3-4B-AWQ-270m"
        assert refetched.quantization == "Q4_K_M"
        assert refetched.training_samples == 100

    async def test_register_duplicate_raises_value_error(self, session: AsyncSession) -> None:
        """Registering a duplicate schema_id + version raises ValueError."""
        svc = ModelService()
        request = RegisterModelRequest(schema_id="ks_Dup", version="v1")
        await svc.register_model(session, request)

        with pytest.raises(ValueError, match="already exists"):
            await svc.register_model(session, request)

    async def test_register_with_gguf_fields(self, session: AsyncSession) -> None:
        """Registering with gguf_file and gguf_sha256 persists them."""
        svc = ModelService()
        request = RegisterModelRequest(
            schema_id="ks_Gguf",
            version="v1",
            gguf_file="models/ks_Gguf-v1.gguf",
            gguf_sha256="abc123def456",  # pragma: allowlist secret
        )
        result = await svc.register_model(session, request)
        assert result.gguf_file == "models/ks_Gguf-v1.gguf"
        assert result.gguf_sha256 == "abc123def456"  # pragma: allowlist secret


class TestModelServiceGetModel:
    """Tests for ModelService.get_model."""

    async def test_get_existing_model(self, session: AsyncSession) -> None:
        """Getting an existing model returns all persisted fields matching what was registered."""
        svc = ModelService()
        request = RegisterModelRequest(
            schema_id="ks_Get",
            version="v1",
            base_model="Qwen3-4B-AWQ-270m",
            quantization="Q4_K_M",
            training_samples=42,
        )
        created = await svc.register_model(session, request)

        result = await svc.get_model(session, created.id)
        assert result is not None
        assert result.id == created.id
        assert result.schema_id == "ks_Get"
        assert result.version == "v1"
        assert result.base_model == "Qwen3-4B-AWQ-270m"
        assert result.quantization == "Q4_K_M"
        assert result.training_samples == 42
        assert result.status == ModelStatus.TRAINING
        assert result.created_at == created.created_at

    async def test_get_nonexistent_model_returns_none(self, session: AsyncSession) -> None:
        """Getting a nonexistent model returns None."""
        svc = ModelService()
        result = await svc.get_model(session, uuid4())
        assert result is None


class TestModelServiceListModels:
    """Tests for ModelService.list_models."""

    async def test_list_empty(self, session: AsyncSession) -> None:
        """Listing with no models returns empty list and zero total."""
        svc = ModelService()
        models, total = await svc.list_models(session)
        assert models == []
        assert total == 0

    async def test_list_returns_all(self, session: AsyncSession) -> None:
        """Listing returns all registered models."""
        svc = ModelService()
        for i in range(3):
            await svc.register_model(
                session, RegisterModelRequest(schema_id=f"ks_List{i}", version="v1")
            )
        models, total = await svc.list_models(session)
        assert total == 3
        assert len(models) == 3

    async def test_list_filter_by_schema_id(self, session: AsyncSession) -> None:
        """Filtering by schema_id returns only matching models."""
        svc = ModelService()
        await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_FilterA", version="v1")
        )
        await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_FilterB", version="v1")
        )
        models, total = await svc.list_models(session, schema_id="ks_FilterA")
        assert total == 1
        assert models[0].schema_id == "ks_FilterA"

    async def test_list_filter_by_status(self, session: AsyncSession) -> None:
        """Filtering by status returns only matching models."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_StatusFilter", version="v1")
        )
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))
        models, total = await svc.list_models(session, status=ModelStatus.EVALUATING)
        assert total == 1
        assert models[0].status == ModelStatus.EVALUATING

        models2, total2 = await svc.list_models(session, status=ModelStatus.DEPLOYED)
        assert total2 == 0

    async def test_list_pagination(self, session: AsyncSession) -> None:
        """Pagination with limit and offset works."""
        svc = ModelService()
        for i in range(5):
            await svc.register_model(
                session, RegisterModelRequest(schema_id=f"ks_Pag{i}", version="v1")
            )
        models, total = await svc.list_models(session, limit=2, offset=0)
        assert len(models) == 2
        assert total == 5

        models2, total2 = await svc.list_models(session, limit=2, offset=4)
        assert len(models2) == 1
        assert total2 == 5


class TestModelServiceUpdateStatus:
    """Tests for ModelService.update_status."""

    async def test_valid_transition(self, session: AsyncSession) -> None:
        """A valid transition updates the status."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_Trans", version="v1")
        )
        updated = await svc.update_status(
            session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATING)
        )
        assert updated.status == ModelStatus.EVALUATING

    async def test_invalid_transition_raises(self, session: AsyncSession) -> None:
        """An invalid transition raises ValueError."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_BadTrans", version="v1")
        )
        with pytest.raises(ValueError, match="Invalid status transition"):
            await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.DEPLOYED))

    async def test_update_nonexistent_raises(self, session: AsyncSession) -> None:
        """Updating a nonexistent model raises ValueError."""
        svc = ModelService()
        with pytest.raises(ValueError, match="Model not found"):
            await svc.update_status(
                session, uuid4(), UpdateStatusRequest(status=ModelStatus.EVALUATING)
            )

    async def test_eval_fields_applied(self, session: AsyncSession) -> None:
        """Setting eval_score and eval_metrics during transition persists them."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_EvalFields", version="v1")
        )
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))
        updated = await svc.update_status(
            session,
            m.id,
            UpdateStatusRequest(
                status=ModelStatus.EVALUATED,
                eval_score=0.88,
                eval_metrics={"accuracy": 0.92},
            ),
        )
        assert updated.eval_score == pytest.approx(0.88)
        assert updated.eval_metrics == {"accuracy": 0.92}

    async def test_gguf_fields_applied(self, session: AsyncSession) -> None:
        """Setting gguf_file and gguf_sha256 during status update persists them."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_GgufUpdate", version="v1")
        )
        # Walk to exporting
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATED))
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.APPROVED))
        updated = await svc.update_status(
            session,
            m.id,
            UpdateStatusRequest(
                status=ModelStatus.EXPORTING,
                gguf_file="models/ks_GgufUpdate-v1.gguf",
                gguf_sha256="sha256hash",
            ),
        )
        assert updated.gguf_file == "models/ks_GgufUpdate-v1.gguf"
        assert updated.gguf_sha256 == "sha256hash"

    async def test_deployed_sets_deployed_at(self, session: AsyncSession) -> None:
        """Transitioning to deployed sets the deployed_at timestamp."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_DeployAt", version="v1")
        )
        for status in [
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
            ModelStatus.EXPORTING,
            ModelStatus.DEPLOYED,
        ]:
            m = await svc.update_status(session, m.id, UpdateStatusRequest(status=status))
        assert m.deployed_at is not None

    async def test_full_lifecycle(self, session: AsyncSession) -> None:
        """Walking through the full lifecycle succeeds."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_FullLC", version="v1")
        )
        for status in [
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
            ModelStatus.EXPORTING,
            ModelStatus.DEPLOYED,
            ModelStatus.RETIRED,
        ]:
            m = await svc.update_status(session, m.id, UpdateStatusRequest(status=status))
        assert m.status == ModelStatus.RETIRED

    async def test_any_state_to_retired(self, session: AsyncSession) -> None:
        """Any non-retired state can transition to retired."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_AnyRetired", version="v1")
        )
        # Direct training -> retired
        updated = await svc.update_status(
            session, m.id, UpdateStatusRequest(status=ModelStatus.RETIRED)
        )
        assert updated.status == ModelStatus.RETIRED

    async def test_retired_is_terminal(self, session: AsyncSession) -> None:
        """Retired is a terminal state - no further transitions."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_TermRetire", version="v1")
        )
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.RETIRED))
        with pytest.raises(ValueError, match="Invalid status transition"):
            await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.TRAINING))


class TestModelServiceAutoApprove:
    """Tests for automatic approval when eval_score meets threshold."""

    async def test_auto_approve_when_score_meets_threshold(self, session: AsyncSession) -> None:
        """Transitioning to EVALUATED with eval_score >= threshold auto-approves."""
        svc = ModelService(auto_approve_score=0.90)
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_AutoApprove", version="v1")
        )
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))
        result = await svc.update_status(
            session,
            m.id,
            UpdateStatusRequest(status=ModelStatus.EVALUATED, eval_score=0.95),
        )
        assert result.status == ModelStatus.APPROVED
        assert result.eval_score == pytest.approx(0.95)

    async def test_no_auto_approve_when_score_below_threshold(self, session: AsyncSession) -> None:
        """Transitioning to EVALUATED with eval_score < threshold stays EVALUATED."""
        svc = ModelService(auto_approve_score=0.90)
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_BelowThresh", version="v1")
        )
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))
        result = await svc.update_status(
            session,
            m.id,
            UpdateStatusRequest(status=ModelStatus.EVALUATED, eval_score=0.85),
        )
        assert result.status == ModelStatus.EVALUATED
        assert result.eval_score == pytest.approx(0.85)

    async def test_auto_approve_at_exact_threshold(self, session: AsyncSession) -> None:
        """Auto-approve triggers when eval_score equals the threshold exactly."""
        svc = ModelService(auto_approve_score=0.90)
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_ExactThresh", version="v1")
        )
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))
        result = await svc.update_status(
            session,
            m.id,
            UpdateStatusRequest(status=ModelStatus.EVALUATED, eval_score=0.90),
        )
        assert result.status == ModelStatus.APPROVED

    async def test_no_auto_approve_without_eval_score(self, session: AsyncSession) -> None:
        """Transitioning to EVALUATED without eval_score does not auto-approve."""
        svc = ModelService(auto_approve_score=0.90)
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_NoScore", version="v1")
        )
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))
        result = await svc.update_status(
            session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATED)
        )
        assert result.status == ModelStatus.EVALUATED

    async def test_auto_approve_with_default_threshold(self, session: AsyncSession) -> None:
        """Default auto_approve_score (0.95) requires >= 0.95 to trigger."""
        svc = ModelService()  # default threshold 0.95
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_DefaultThresh", version="v1")
        )
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))
        # Score of 0.94 should NOT auto-approve
        result = await svc.update_status(
            session,
            m.id,
            UpdateStatusRequest(status=ModelStatus.EVALUATED, eval_score=0.94),
        )
        assert result.status == ModelStatus.EVALUATED

    async def test_auto_approve_default_threshold_met(self, session: AsyncSession) -> None:
        """Default threshold is met at 0.95."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_DefaultMet", version="v1")
        )
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))
        result = await svc.update_status(
            session,
            m.id,
            UpdateStatusRequest(status=ModelStatus.EVALUATED, eval_score=0.96),
        )
        assert result.status == ModelStatus.APPROVED

    async def test_auto_approve_creates_audit_events(self, session: AsyncSession) -> None:
        """Auto-approval records both STATUS_CHANGED and AUTO_APPROVED audit events."""
        audit_svc = AuditService()
        svc = ModelService(audit_service=audit_svc, auto_approve_score=0.90)
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_AutoAudit", version="v1")
        )
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))
        await svc.update_status(
            session,
            m.id,
            UpdateStatusRequest(status=ModelStatus.EVALUATED, eval_score=0.95),
        )
        # Query audit events for this model
        events, total = await audit_svc.list_events(session, entity_id=str(m.id), limit=100)
        event_types = [e.event_type for e in events]
        assert "auto_approved" in event_types
        # Find the auto_approved event and verify detail
        auto_event = next(e for e in events if e.event_type == "auto_approved")
        assert auto_event.detail is not None
        assert auto_event.detail["eval_score"] == pytest.approx(0.95)
        assert auto_event.detail["threshold"] == pytest.approx(0.90)
        assert auto_event.actor == "system"

    async def test_auto_approve_disabled_with_threshold_1(self, session: AsyncSession) -> None:
        """Setting auto_approve_score=1.0 effectively disables auto-approval."""
        svc = ModelService(auto_approve_score=1.0)
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_Disabled", version="v1")
        )
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))
        result = await svc.update_status(
            session,
            m.id,
            UpdateStatusRequest(status=ModelStatus.EVALUATED, eval_score=0.99),
        )
        assert result.status == ModelStatus.EVALUATED

    async def test_auto_approve_perfect_score_with_threshold_1(self, session: AsyncSession) -> None:
        """Even with threshold=1.0, a perfect score of 1.0 auto-approves."""
        svc = ModelService(auto_approve_score=1.0)
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_Perfect", version="v1")
        )
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))
        result = await svc.update_status(
            session,
            m.id,
            UpdateStatusRequest(status=ModelStatus.EVALUATED, eval_score=1.0),
        )
        assert result.status == ModelStatus.APPROVED


class TestModelServiceQuality:
    """Tests for ModelService.get_latest_quality."""

    async def test_no_models_returns_null_quality(self, session: AsyncSession) -> None:
        """Querying a schema with no models returns null quality fields."""
        svc = ModelService()
        result = await svc.get_latest_quality(session, "ks_NoModels")
        assert isinstance(result, QualityResponse)
        assert result.schema_id == "ks_NoModels"
        assert result.quality_score is None
        assert result.model_id is None
        assert result.version is None

    async def test_model_without_eval_returns_null(self, session: AsyncSession) -> None:
        """A model without eval_score returns null quality."""
        svc = ModelService()
        await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_NoEvalQ", version="v1")
        )
        result = await svc.get_latest_quality(session, "ks_NoEvalQ")
        assert result.quality_score is None

    async def test_model_with_eval_returns_score(self, session: AsyncSession) -> None:
        """A model with eval_score returns the quality score."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_WithEval", version="v1")
        )
        await svc.update_status(session, m.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))
        await svc.update_status(
            session,
            m.id,
            UpdateStatusRequest(status=ModelStatus.EVALUATED, eval_score=0.85),
        )
        result = await svc.get_latest_quality(session, "ks_WithEval")
        assert result.quality_score == pytest.approx(0.85)
        assert result.model_id == m.id
        assert result.version == "v1"

    async def test_quality_returns_latest(self, session: AsyncSession) -> None:
        """When multiple models have eval scores, the most recent is returned."""
        svc = ModelService()
        # Older model
        m1 = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_LatestQ", version="v1")
        )
        await svc.update_status(session, m1.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))
        await svc.update_status(
            session, m1.id, UpdateStatusRequest(status=ModelStatus.EVALUATED, eval_score=0.70)
        )
        # Newer model
        m2 = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_LatestQ", version="v2")
        )
        await svc.update_status(session, m2.id, UpdateStatusRequest(status=ModelStatus.EVALUATING))
        await svc.update_status(
            session, m2.id, UpdateStatusRequest(status=ModelStatus.EVALUATED, eval_score=0.95)
        )

        result = await svc.get_latest_quality(session, "ks_LatestQ")
        assert result.quality_score == pytest.approx(0.95)
        assert result.model_id == m2.id


# ---------------------------------------------------------------------------
# Domain model unit tests (is_valid_transition, VALID_TRANSITIONS)
# ---------------------------------------------------------------------------


class TestIsValidTransition:
    """Tests for is_valid_transition() domain logic."""

    def test_training_to_evaluating_is_valid(self) -> None:
        """Training -> evaluating is a valid transition."""
        assert is_valid_transition(ModelStatus.TRAINING, ModelStatus.EVALUATING) is True

    def test_training_to_retired_is_valid(self) -> None:
        """Training -> retired is valid (cleanup escape hatch)."""
        assert is_valid_transition(ModelStatus.TRAINING, ModelStatus.RETIRED) is True

    def test_training_to_deployed_is_invalid(self) -> None:
        """Training -> deployed is not a valid transition (skips steps)."""
        assert is_valid_transition(ModelStatus.TRAINING, ModelStatus.DEPLOYED) is False

    def test_training_to_approved_is_invalid(self) -> None:
        """Training -> approved is not a valid transition (skips steps)."""
        assert is_valid_transition(ModelStatus.TRAINING, ModelStatus.APPROVED) is False

    def test_evaluating_to_evaluated_is_valid(self) -> None:
        """Evaluating -> evaluated is a valid transition."""
        assert is_valid_transition(ModelStatus.EVALUATING, ModelStatus.EVALUATED) is True

    def test_evaluated_to_approved_is_valid(self) -> None:
        """Evaluated -> approved is a valid transition."""
        assert is_valid_transition(ModelStatus.EVALUATED, ModelStatus.APPROVED) is True

    def test_approved_to_exporting_is_valid(self) -> None:
        """Approved -> exporting is a valid transition."""
        assert is_valid_transition(ModelStatus.APPROVED, ModelStatus.EXPORTING) is True

    def test_exporting_to_deployed_is_valid(self) -> None:
        """Exporting -> deployed is a valid transition."""
        assert is_valid_transition(ModelStatus.EXPORTING, ModelStatus.DEPLOYED) is True
        # Failed-deploy rollback: exporting must be able to return to approved,
        # otherwise a failed deploy strands the model (retired is the only
        # other exit and deploy requires approved).
        assert is_valid_transition(ModelStatus.EXPORTING, ModelStatus.APPROVED) is True

    def test_deployed_to_retired_is_valid(self) -> None:
        """Deployed -> retired is a valid transition."""
        assert is_valid_transition(ModelStatus.DEPLOYED, ModelStatus.RETIRED) is True

    def test_retired_is_terminal(self) -> None:
        """Retired -> anything is invalid (terminal state)."""
        for target in ModelStatus:
            assert is_valid_transition(ModelStatus.RETIRED, target) is False

    def test_every_state_can_reach_retired(self) -> None:
        """Every non-retired state can transition to retired."""
        for status in ModelStatus:
            if status != ModelStatus.RETIRED:
                assert is_valid_transition(status, ModelStatus.RETIRED) is True

    def test_backward_transition_is_invalid(self) -> None:
        """Backward transitions (e.g., deployed -> training) are invalid."""
        assert is_valid_transition(ModelStatus.DEPLOYED, ModelStatus.TRAINING) is False
        assert is_valid_transition(ModelStatus.EVALUATED, ModelStatus.EVALUATING) is False


class TestValidTransitions:
    """Tests for the VALID_TRANSITIONS mapping."""

    def test_all_statuses_have_entries(self) -> None:
        """Every ModelStatus value has an entry in VALID_TRANSITIONS."""
        for status in ModelStatus:
            assert status in VALID_TRANSITIONS, f"Missing VALID_TRANSITIONS entry for {status}"

    def test_retired_has_no_transitions(self) -> None:
        """RETIRED is a terminal state with no valid transitions."""
        assert VALID_TRANSITIONS[ModelStatus.RETIRED] == set()

    def test_training_transitions(self) -> None:
        """TRAINING can only go to EVALUATING or RETIRED."""
        assert VALID_TRANSITIONS[ModelStatus.TRAINING] == {
            ModelStatus.EVALUATING,
            ModelStatus.RETIRED,
        }

    def test_forward_chain_is_complete(self) -> None:
        """The full forward lifecycle chain is valid end-to-end."""
        chain = [
            ModelStatus.TRAINING,
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
            ModelStatus.EXPORTING,
            ModelStatus.DEPLOYED,
            ModelStatus.RETIRED,
        ]
        for current, target in zip(chain[:-1], chain[1:]):
            assert target in VALID_TRANSITIONS[current], f"{current} -> {target} should be valid"


# ---------------------------------------------------------------------------
# RoutingService unit tests
# ---------------------------------------------------------------------------


class TestRoutingServiceCache:
    """Tests for RoutingService TTL cache behavior."""

    async def test_invalidate_single_key(self, session: AsyncSession) -> None:
        """Invalidating a single key removes only that entry."""
        svc = RoutingService()
        # Create a route to populate the cache
        await svc.upsert_route(
            session,
            "ks_CacheA",
            UpdateRoutingRequest(endpoint="ep-a:8080"),
        )
        await svc.upsert_route(
            session,
            "ks_CacheB",
            UpdateRoutingRequest(endpoint="ep-b:8080"),
        )
        # Populate cache by reading
        await svc.get_route(session, "ks_CacheA")
        await svc.get_route(session, "ks_CacheB")
        assert "ks_CacheA" in svc._cache
        assert "ks_CacheB" in svc._cache

        # Invalidate only A
        svc.invalidate("ks_CacheA")
        assert "ks_CacheA" not in svc._cache
        assert "ks_CacheB" in svc._cache

    async def test_invalidate_all(self, session: AsyncSession) -> None:
        """Invalidating without a key clears the entire cache."""
        svc = RoutingService()
        await svc.upsert_route(session, "ks_ClearAll", UpdateRoutingRequest(endpoint="ep:8080"))
        await svc.get_route(session, "ks_ClearAll")
        assert len(svc._cache) > 0

        svc.invalidate()
        assert len(svc._cache) == 0
        assert svc._full_table_cache is None
        assert svc._full_table_cache_time is None

    async def test_get_routing_table_caches(self, session: AsyncSession) -> None:
        """The full routing table is cached after first fetch."""
        svc = RoutingService()
        await svc.upsert_route(session, "ks_TableCache", UpdateRoutingRequest(endpoint="ep:8080"))
        # First call populates cache
        result1 = await svc.get_routing_table(session)
        assert svc._full_table_cache is not None
        assert svc._full_table_cache_time is not None

        # Second call should use cache (same result)
        result2 = await svc.get_routing_table(session)
        assert result1 == result2

    async def test_get_routing_table_expired_cache(self, session: AsyncSession) -> None:
        """When the full table cache is expired, it refetches from DB."""
        svc = RoutingService()
        await svc.upsert_route(session, "ks_Expired", UpdateRoutingRequest(endpoint="ep:8080"))
        # First call
        await svc.get_routing_table(session)
        # Force cache expiry by setting time in the past
        svc._full_table_cache_time = datetime.now(UTC) - timedelta(seconds=300)

        # Should refetch from DB
        result = await svc.get_routing_table(session)
        assert len(result) >= 1

    async def test_get_route_from_cache(self, session: AsyncSession) -> None:
        """A cached route is returned directly without DB query."""
        svc = RoutingService()
        await svc.upsert_route(session, "ks_CacheHit", UpdateRoutingRequest(endpoint="ep:8080"))
        # First fetch populates cache
        await svc.get_route(session, "ks_CacheHit")
        assert "ks_CacheHit" in svc._cache

        # Second fetch should come from cache
        result = await svc.get_route(session, "ks_CacheHit")
        assert result is not None
        assert result.schema_id == "ks_CacheHit"

    async def test_get_route_not_found(self, session: AsyncSession) -> None:
        """Getting a nonexistent route returns None."""
        svc = RoutingService()
        result = await svc.get_route(session, "nonexistent_schema")
        assert result is None


class TestRoutingServiceUpsert:
    """Tests for RoutingService.upsert_route."""

    async def test_create_new_route(self, session: AsyncSession) -> None:
        """Creating a new route returns a RoutingRule with correct fields persisted to DB."""
        svc = RoutingService()
        model_id = uuid4()
        result = await svc.upsert_route(
            session,
            "ks_NewRoute",
            UpdateRoutingRequest(
                active_model_id=model_id,
                fallback_chain=["pgx-local", "xai"],
                endpoint="fg-ks-newroute.ns.svc.example.com:8080",
            ),
        )
        assert result.schema_id == "ks_NewRoute"
        assert result.active_model_id == model_id
        assert result.fallback_chain == ["pgx-local", "xai"]
        assert result.endpoint == "fg-ks-newroute.ns.svc.example.com:8080"
        assert result.created_at is not None
        assert result.updated_at is not None

        # Verify round-trip: re-read from DB to confirm persistence
        svc.invalidate("ks_NewRoute")
        refetched = await svc.get_route(session, "ks_NewRoute")
        assert refetched is not None
        assert refetched.active_model_id == model_id
        assert refetched.fallback_chain == ["pgx-local", "xai"]
        assert refetched.endpoint == "fg-ks-newroute.ns.svc.example.com:8080"

    async def test_create_with_defaults(self, session: AsyncSession) -> None:
        """Creating a route with no fields uses defaults."""
        svc = RoutingService()
        result = await svc.upsert_route(session, "ks_Defaults", UpdateRoutingRequest())
        assert result.schema_id == "ks_Defaults"
        assert result.fallback_chain == ["pgx-local", "deepseek"]
        assert result.active_model_id is None
        assert result.endpoint is None

    async def test_update_existing_route(self, session: AsyncSession) -> None:
        """Updating an existing route modifies only the provided fields."""
        svc = RoutingService()
        # Create
        await svc.upsert_route(
            session,
            "ks_UpdRoute",
            UpdateRoutingRequest(endpoint="ep-v1:8080"),
        )
        # Update
        model_id = uuid4()
        result = await svc.upsert_route(
            session,
            "ks_UpdRoute",
            UpdateRoutingRequest(active_model_id=model_id, endpoint="ep-v2:8080"),
        )
        assert result.active_model_id == model_id
        assert result.endpoint == "ep-v2:8080"

    async def test_update_with_json_schema_and_grammar(self, session: AsyncSession) -> None:
        """Setting json_schema and gbnf_grammar on update persists correctly."""
        svc = RoutingService()
        await svc.upsert_route(
            session, "ks_SchemaGrammar", UpdateRoutingRequest(endpoint="ep:8080")
        )
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        grammar = 'root ::= "{" name "}"'
        result = await svc.upsert_route(
            session,
            "ks_SchemaGrammar",
            UpdateRoutingRequest(json_schema=schema, gbnf_grammar=grammar),
        )
        assert result.json_schema == schema
        assert result.gbnf_grammar == grammar

    async def test_upsert_invalidates_cache(self, session: AsyncSession) -> None:
        """Upserting a route invalidates the cache for that schema_id."""
        svc = RoutingService()
        await svc.upsert_route(session, "ks_Inval", UpdateRoutingRequest(endpoint="ep:8080"))
        # Populate cache
        await svc.get_route(session, "ks_Inval")
        assert "ks_Inval" in svc._cache

        # Upsert should invalidate cache
        await svc.upsert_route(session, "ks_Inval", UpdateRoutingRequest(endpoint="ep-v2:8080"))
        assert "ks_Inval" not in svc._cache


class TestRoutingServiceDeployment:
    """Tests for RoutingService.update_route_for_deployment."""

    async def test_update_route_for_deployment(self, session: AsyncSession) -> None:
        """update_route_for_deployment creates/updates a routing rule."""
        svc = RoutingService()
        model_id = uuid4()
        result = await svc.update_route_for_deployment(
            session, "ks_Deploy", model_id, "fg-ks-deploy.ns.svc.example.com:8080"
        )
        assert result.schema_id == "ks_Deploy"
        assert result.active_model_id == model_id
        assert result.endpoint == "fg-ks-deploy.ns.svc.example.com:8080"


# ---------------------------------------------------------------------------
# DeployService unit tests
# ---------------------------------------------------------------------------


class TestDeployServiceHelpers:
    """Tests for DeployService static/helper methods."""

    def test_compute_endpoint(self) -> None:
        """_compute_endpoint builds the correct in-cluster endpoint."""
        result = DeployService._compute_endpoint("ks_EntitySpotterAgent", "llm-inference")
        assert result == "fg-ks-entityspotteragent.llm-inference.svc.example.com:8080"

    def test_compute_endpoint_underscores(self) -> None:
        """Underscores are replaced with hyphens and the name is lowercased."""
        result = DeployService._compute_endpoint("wm_Parser_ABC", "staging")
        assert result == "fg-wm-parser-abc.staging.svc.example.com:8080"

    def test_compute_endpoint_uses_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cluster DNS suffix + listen port come from settings, not literals."""
        from settings import get_settings

        monkeypatch.setenv("MODEL_REGISTRY_CLUSTER_DNS_DOMAIN", "cluster.example.com")
        monkeypatch.setenv("MODEL_REGISTRY_DEPLOYED_SERVICE_PORT", "9090")
        monkeypatch.setenv("MODEL_REGISTRY_ENVIRONMENT", "development")
        get_settings.cache_clear()
        try:
            result = DeployService._compute_endpoint("simple", "ns")
            assert result == "fg-simple.ns.svc.cluster.example.com:9090"
        finally:
            get_settings.cache_clear()

    def test_generate_values_yaml(self) -> None:
        """_generate_values_yaml produces valid YAML content."""
        model = ModelVersion(
            id=uuid4(),
            schema_id="ks_TestYaml",
            version="v2",
            status=ModelStatus.EXPORTING,
            base_model="Qwen3-4B-AWQ-270m",
            quantization="Q4_K_M",
            gguf_file="models/ks_TestYaml-v2.gguf",
            gguf_sha256="abc123",
        )
        request = DeployRequest(target_namespace="llm-inference", replicas=3)
        endpoint = "fg-ks-testyaml.llm-inference.svc.example.com:8080"

        svc = DeployService(MagicMock(), MagicMock())
        yaml_content = svc._generate_values_yaml(model, request, endpoint)

        assert 'modelName: "ks_TestYaml"' in yaml_content
        assert 'modelVersion: "v2"' in yaml_content
        assert 'ggufPath: "models/ks_TestYaml-v2.gguf"' in yaml_content
        assert 'ggufSha256: "abc123"' in yaml_content
        assert 'endpoint: "fg-ks-testyaml.llm-inference.svc.example.com:8080"' in yaml_content
        assert "replicas: 3" in yaml_content

    def test_generate_values_yaml_without_gguf(self) -> None:
        """When gguf_file is None, a default path is generated."""
        model = ModelVersion(
            id=uuid4(),
            schema_id="ks_NoGguf",
            version="v1",
            status=ModelStatus.EXPORTING,
        )
        request = DeployRequest()
        endpoint = "fg-ks-nogguf.llm-inference.svc.example.com:8080"

        svc = DeployService(MagicMock(), MagicMock())
        yaml_content = svc._generate_values_yaml(model, request, endpoint)

        assert "ks_NoGguf-v1-f16.gguf" in yaml_content
        assert 'ggufSha256: "pending"' in yaml_content

    def test_write_values_file(self, tmp_path: Path) -> None:
        """_write_values_file writes the YAML content to the correct path."""
        svc = DeployService(MagicMock(), MagicMock())
        mock_settings = MagicMock()
        mock_settings.gitops_repo_path = str(tmp_path)

        with patch("services.deploy_service.get_settings", return_value=mock_settings):
            path = svc._write_values_file("ks_TestWrite", "yaml content here")

        expected = (
            tmp_path / "clusters" / "production" / "llm-inference" / "values-ks_TestWrite.yaml"
        )
        assert Path(path) == expected
        assert expected.read_text() == "yaml content here"


class TestDeployServiceGitOps:
    """Tests for DeployService._git_commit_and_push."""

    async def test_git_commit_and_push_success(self) -> None:
        """Successful git operations return True."""
        fake_proc = AsyncMock()
        fake_proc.communicate.return_value = (b"ok", b"")
        fake_proc.returncode = 0

        with patch(
            "services.deploy_service.asyncio.create_subprocess_exec",
            return_value=fake_proc,
        ) as mock_exec:
            result = await DeployService._git_commit_and_push(
                "/tmp/test/values.yaml", "deploy: test"
            )

        assert result is True
        assert mock_exec.call_count == 3

    async def test_git_commit_and_push_failure(self) -> None:
        """If a git command fails, returns False."""
        fake_proc = AsyncMock()
        fake_proc.communicate.return_value = (b"", b"error")
        fake_proc.returncode = 1

        with patch(
            "services.deploy_service.asyncio.create_subprocess_exec",
            return_value=fake_proc,
        ):
            result = await DeployService._git_commit_and_push(
                "/tmp/test/values.yaml", "deploy: test"
            )

        assert result is False

    async def test_git_commit_and_push_exception(self) -> None:
        """If subprocess spawn raises OSError, returns False."""
        with patch(
            "services.deploy_service.asyncio.create_subprocess_exec",
            side_effect=OSError("no git"),
        ):
            result = await DeployService._git_commit_and_push(
                "/tmp/test/values.yaml", "deploy: test"
            )

        assert result is False

    async def test_git_commit_and_push_file_not_found(self) -> None:
        """FileNotFoundError (git binary missing) is treated as a soft failure."""
        with patch(
            "services.deploy_service.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("git: command not found"),
        ):
            result = await DeployService._git_commit_and_push(
                "/tmp/test/values.yaml", "deploy: test"
            )

        assert result is False

    async def test_git_commit_and_push_permission_error(self) -> None:
        """PermissionError is treated as a soft failure."""
        with patch(
            "services.deploy_service.asyncio.create_subprocess_exec",
            side_effect=PermissionError("no exec"),
        ):
            result = await DeployService._git_commit_and_push(
                "/tmp/test/values.yaml", "deploy: test"
            )

        assert result is False

    async def test_git_commit_and_push_unexpected_error_propagates(self) -> None:
        """Non-OSError exceptions (e.g. programming bugs) must propagate."""
        with patch(
            "services.deploy_service.asyncio.create_subprocess_exec",
            side_effect=RuntimeError("unexpected bug"),
        ):
            with pytest.raises(RuntimeError, match="unexpected bug"):
                await DeployService._git_commit_and_push("/tmp/test/values.yaml", "deploy: test")

    async def test_git_push_retries_on_transient_failure(self) -> None:
        """git push is retried and eventually succeeds after two failures.

        Mocks ``create_subprocess_exec`` so that ``git add`` and
        ``git commit`` succeed once (returncode 0), then ``git push`` fails
        twice (returncode 1) before finally succeeding. We expect a total
        of 5 subprocess invocations (add, commit, push, push, push), the
        final return value to be True, and the ``asyncio.sleep`` backoff
        to be called twice with the documented (1s, 4s) exponential
        sequence.
        """

        def make_proc(returncode: int) -> AsyncMock:
            p = AsyncMock()
            p.communicate.return_value = (b"", b"")
            p.returncode = returncode
            return p

        # add ok, commit ok, push fail, push fail, push ok
        proc_sequence = [
            make_proc(0),
            make_proc(0),
            make_proc(1),
            make_proc(1),
            make_proc(0),
        ]

        with (
            patch(
                "services.deploy_service.asyncio.create_subprocess_exec",
                side_effect=proc_sequence,
            ) as mock_exec,
            patch(
                "services.deploy_service.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            result = await DeployService._git_commit_and_push(
                "/tmp/test/values.yaml", "deploy: test"
            )

        assert result is True
        assert mock_exec.call_count == 5
        # Only the push step is retried — 2 failed pushes trigger 2 sleeps.
        assert mock_sleep.await_count == 2
        # Exponential backoff: 1s, then 4s.
        sleep_delays = [call.args[0] for call in mock_sleep.await_args_list]
        assert sleep_delays == [1.0, 4.0]

    async def test_git_push_retries_exhausted(self) -> None:
        """git push returns False after exhausting all retry attempts.

        Mocks ``create_subprocess_exec`` so add/commit succeed and push
        fails every time. After 3 failed push attempts the function must
        return False and sleep must have been awaited exactly twice
        (between attempt 1-2 and 2-3, but NOT after the final failure).
        """

        def make_proc(returncode: int) -> AsyncMock:
            p = AsyncMock()
            p.communicate.return_value = (b"", b"boom")
            p.returncode = returncode
            return p

        # add ok, commit ok, push fail x3
        proc_sequence = [
            make_proc(0),
            make_proc(0),
            make_proc(1),
            make_proc(1),
            make_proc(1),
        ]

        with (
            patch(
                "services.deploy_service.asyncio.create_subprocess_exec",
                side_effect=proc_sequence,
            ) as mock_exec,
            patch(
                "services.deploy_service.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            result = await DeployService._git_commit_and_push(
                "/tmp/test/values.yaml", "deploy: test"
            )

        assert result is False
        assert mock_exec.call_count == 5
        # Between attempts 1-2 and 2-3, but not after attempt 3.
        assert mock_sleep.await_count == 2

    async def test_git_push_not_retried_if_commit_fails(self) -> None:
        """A failing ``git commit`` short-circuits before any push attempt.

        Verifies that the retry logic only applies to ``git push``: if
        ``git commit`` fails, we return False immediately without invoking
        ``git push`` or sleeping.
        """

        def make_proc(returncode: int) -> AsyncMock:
            p = AsyncMock()
            p.communicate.return_value = (b"", b"nothing to commit")
            p.returncode = returncode
            return p

        # add ok, commit fails — push must never run
        proc_sequence = [make_proc(0), make_proc(1)]

        with (
            patch(
                "services.deploy_service.asyncio.create_subprocess_exec",
                side_effect=proc_sequence,
            ) as mock_exec,
            patch(
                "services.deploy_service.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            result = await DeployService._git_commit_and_push(
                "/tmp/test/values.yaml", "deploy: test"
            )

        assert result is False
        # Only add + commit ran — no push, no retry, no sleep.
        assert mock_exec.call_count == 2
        mock_sleep.assert_not_awaited()


class TestDeployServiceDeployModel:
    """Tests for DeployService.deploy_model."""

    async def test_deploy_model_not_found(self, session: AsyncSession) -> None:
        """Deploying a nonexistent model raises ValueError."""
        model_svc = ModelService()
        routing_svc = RoutingService()
        deploy_svc = DeployService(model_svc, routing_svc)

        with pytest.raises(ValueError, match="Model not found"):
            await deploy_svc.deploy_model(session, uuid4(), DeployRequest())

    async def test_deploy_model_not_approved(self, session: AsyncSession) -> None:
        """Deploying a model not in 'approved' status raises ValueError."""
        model_svc = ModelService()
        routing_svc = RoutingService()
        deploy_svc = DeployService(model_svc, routing_svc)

        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_NotApproved", version="v1")
        )
        with pytest.raises(ValueError, match="approved"):
            await deploy_svc.deploy_model(session, m.id, DeployRequest())

    async def test_deploy_model_success(self, session: AsyncSession) -> None:
        """Full deployment pipeline succeeds: model deployed, routing set, values file written."""
        model_svc = ModelService()
        routing_svc = RoutingService()
        deploy_svc = DeployService(model_svc, routing_svc)

        # Register and walk to approved
        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_DeployOK", version="v1")
        )
        for status in [
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
        ]:
            m = await model_svc.update_status(session, m.id, UpdateStatusRequest(status=status))

        # Deploy
        with gitops_publish_succeeds():
            result = await deploy_svc.deploy_model(session, m.id, DeployRequest())

        assert result.model_id == m.id
        assert result.schema_id == "ks_DeployOK"
        assert result.status == ModelStatus.DEPLOYED
        assert result.values_file == "values-ks_DeployOK.yaml"
        assert "Deployed" in result.message or "deployed" in result.message

        # Verify the model status and deployed_at in DB
        updated = await model_svc.get_model(session, m.id)
        assert updated is not None
        assert updated.status == ModelStatus.DEPLOYED
        assert updated.deployed_at is not None

        # Verify routing rule was created with correct endpoint
        route = await routing_svc.get_route(session, "ks_DeployOK")
        assert route is not None
        assert route.active_model_id == m.id
        assert route.endpoint is not None
        assert "fg-ks-deployok" in route.endpoint
        assert "svc.example.com:8080" in route.endpoint

    async def test_deploy_with_gitops_enabled(self, session: AsyncSession) -> None:
        """When gitops is enabled, git operations are called."""
        model_svc = ModelService()
        routing_svc = RoutingService()
        deploy_svc = DeployService(model_svc, routing_svc)

        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_GitEnabled", version="v1")
        )
        for status in [
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
        ]:
            m = await model_svc.update_status(session, m.id, UpdateStatusRequest(status=status))

        fake_proc = AsyncMock()
        fake_proc.communicate.return_value = (b"ok", b"")
        fake_proc.returncode = 0

        mock_settings = MagicMock()
        mock_settings.gitops_repo_path = os.environ.get(
            "MODEL_REGISTRY_GITOPS_REPO_PATH", "/tmp/test-gitops"
        )
        mock_settings.gitops_enabled = True

        with (
            patch("services.deploy_service.get_settings", return_value=mock_settings),
            patch(
                "services.deploy_service.asyncio.create_subprocess_exec",
                return_value=fake_proc,
            ) as mock_exec,
        ):
            result = await deploy_svc.deploy_model(
                session, m.id, DeployRequest(target_namespace="llm-inference", replicas=2)
            )

        assert result.status == ModelStatus.DEPLOYED
        assert mock_exec.call_count == 3

    async def test_deploy_with_gitops_failure(self, session: AsyncSession) -> None:
        """When the gitops push fails, the deploy fails closed.

        Previously asserted DEPLOYED — a failed push was downgraded to a log
        line and the model was still reported deployed with a routing endpoint
        that the cluster would never serve.
        """
        model_svc = ModelService()
        routing_svc = RoutingService()
        deploy_svc = DeployService(model_svc, routing_svc)

        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_GitFail", version="v1")
        )
        for status in [
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
        ]:
            m = await model_svc.update_status(session, m.id, UpdateStatusRequest(status=status))

        with (
            gitops_push_fails(),
            pytest.raises(DeployError, match="did not reach gitops"),
        ):
            await deploy_svc.deploy_model(session, m.id, DeployRequest())

        updated = await model_svc.get_model(session, m.id)
        assert updated is not None
        assert updated.status == ModelStatus.APPROVED

        # And no routing rule was written for a deploy that did not happen.
        assert await routing_svc.get_route(session, "ks_GitFail") is None


# ---------------------------------------------------------------------------
# AuditService unit tests
# ---------------------------------------------------------------------------


class TestAuditServiceRecord:
    """Tests for AuditService.record."""

    async def test_record_creates_event_and_persists(self, session: AsyncSession) -> None:
        """Recording an audit event persists it to the DB with correct fields."""
        from models.domain import AuditEventType

        svc = AuditService()
        event = await svc.record(
            session,
            event_type=AuditEventType.MODEL_REGISTERED,
            entity_type="model_version",
            entity_id="test-id",
            actor="test",
            detail={"key": "value"},
        )
        assert event.event_type == AuditEventType.MODEL_REGISTERED
        assert event.entity_type == "model_version"
        assert event.entity_id == "test-id"
        assert event.actor == "test"
        assert event.detail == {"key": "value"}
        assert event.id > 0
        assert event.timestamp is not None

        # Verify round-trip: re-read from DB to confirm persistence
        events, total = await svc.list_events(session, entity_id="test-id")
        assert total == 1
        persisted = events[0]
        assert persisted.event_type == AuditEventType.MODEL_REGISTERED
        assert persisted.entity_type == "model_version"
        assert persisted.entity_id == "test-id"
        assert persisted.actor == "test"
        assert persisted.detail == {"key": "value"}

    async def test_list_events_empty(self, session: AsyncSession) -> None:
        """Listing events when none exist returns empty."""
        svc = AuditService()
        events, total = await svc.list_events(session)
        assert events == []
        assert total == 0

    async def test_list_events_with_filter(self, session: AsyncSession) -> None:
        """Listing events with filters returns matching events."""
        from models.domain import AuditEventType

        svc = AuditService()
        await svc.record(
            session,
            event_type=AuditEventType.MODEL_REGISTERED,
            entity_type="model_version",
            entity_id="a",
        )
        await svc.record(
            session,
            event_type=AuditEventType.STATUS_CHANGED,
            entity_type="model_version",
            entity_id="b",
        )

        events, total = await svc.list_events(session, event_type=AuditEventType.MODEL_REGISTERED)
        assert total == 1
        assert events[0].entity_id == "a"

    async def test_get_model_history(self, session: AsyncSession) -> None:
        """get_model_history returns events for a specific model."""
        from models.domain import AuditEventType

        svc = AuditService()
        await svc.record(
            session,
            event_type=AuditEventType.MODEL_REGISTERED,
            entity_type="model_version",
            entity_id="model-x",
        )
        await svc.record(
            session,
            event_type=AuditEventType.STATUS_CHANGED,
            entity_type="model_version",
            entity_id="model-x",
        )
        await svc.record(
            session,
            event_type=AuditEventType.MODEL_REGISTERED,
            entity_type="model_version",
            entity_id="model-y",
        )

        events, total = await svc.get_model_history(session, "model-x")
        assert total == 2

    async def test_get_schema_history(self, session: AsyncSession) -> None:
        """get_schema_history returns events related to a schema."""
        from models.domain import AuditEventType

        svc = AuditService()
        await svc.record(
            session,
            event_type=AuditEventType.ROUTE_UPDATED,
            entity_type="routing_rule",
            entity_id="ks_Schema1",
        )
        await svc.record(
            session,
            event_type=AuditEventType.ALIAS_SET,
            entity_type="alias",
            entity_id="ks_Schema1:champion",
        )
        await svc.record(
            session,
            event_type=AuditEventType.ROUTE_UPDATED,
            entity_type="routing_rule",
            entity_id="ks_Schema2",
        )

        events, total = await svc.get_schema_history(session, "ks_Schema1")
        assert total == 2


# ---------------------------------------------------------------------------
# AliasService unit tests
# ---------------------------------------------------------------------------


class TestAliasServiceSetAlias:
    """Tests for AliasService.set_alias."""

    async def test_set_alias_creates_new(self, session: AsyncSession) -> None:
        """Setting a new alias creates it and persists to DB."""
        model_svc = ModelService()
        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_AliasNew", version="v1")
        )

        alias_svc = AliasService()
        alias = await alias_svc.set_alias(session, "ks_AliasNew", "champion", m.id)
        assert alias.schema_id == "ks_AliasNew"
        assert alias.alias_name == "champion"
        assert alias.model_id == m.id
        assert alias.created_at is not None
        assert alias.updated_at is not None

        # Verify round-trip: re-read from DB to confirm persistence
        refetched = await alias_svc.get_alias(session, "ks_AliasNew", "champion")
        assert refetched is not None
        assert refetched.model_id == m.id
        assert refetched.alias_name == "champion"

    async def test_set_alias_updates_existing(self, session: AsyncSession) -> None:
        """Setting an existing alias updates the model_id."""
        model_svc = ModelService()
        m1 = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_AliasUpd", version="v1")
        )
        m2 = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_AliasUpd", version="v2")
        )

        alias_svc = AliasService()
        await alias_svc.set_alias(session, "ks_AliasUpd", "champion", m1.id)
        alias = await alias_svc.set_alias(session, "ks_AliasUpd", "champion", m2.id)
        assert alias.model_id == m2.id

    async def test_set_alias_invalid_name(self, session: AsyncSession) -> None:
        """Setting an alias with an invalid name raises ValueError."""
        alias_svc = AliasService()
        with pytest.raises(ValueError, match="Invalid alias name"):
            await alias_svc.set_alias(session, "ks_Test", "INVALID", uuid4())

    async def test_set_alias_model_not_found(self, session: AsyncSession) -> None:
        """Setting an alias to a nonexistent model raises ValueError."""
        alias_svc = AliasService()
        with pytest.raises(ValueError, match="Model not found"):
            await alias_svc.set_alias(session, "ks_Test", "champion", uuid4())


class TestAliasServiceGet:
    """Tests for AliasService.get_alias and list_aliases."""

    async def test_get_alias_not_found(self, session: AsyncSession) -> None:
        """Getting a nonexistent alias returns None."""
        alias_svc = AliasService()
        result = await alias_svc.get_alias(session, "ks_None", "champion")
        assert result is None

    async def test_list_aliases_empty(self, session: AsyncSession) -> None:
        """Listing aliases for a schema with none returns empty list."""
        alias_svc = AliasService()
        result = await alias_svc.list_aliases(session, "ks_NoAliases")
        assert result == []

    async def test_list_aliases(self, session: AsyncSession) -> None:
        """Listing aliases returns all for the schema."""
        model_svc = ModelService()
        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_AliasList", version="v1")
        )

        alias_svc = AliasService()
        await alias_svc.set_alias(session, "ks_AliasList", "champion", m.id)
        await alias_svc.set_alias(session, "ks_AliasList", "rollback", m.id)

        result = await alias_svc.list_aliases(session, "ks_AliasList")
        assert len(result) == 2

    async def test_get_aliases_for_model(self, session: AsyncSession) -> None:
        """Getting aliases for a model returns all aliases pointing to it."""
        model_svc = ModelService()
        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_AliasForModel", version="v1")
        )

        alias_svc = AliasService()
        await alias_svc.set_alias(session, "ks_AliasForModel", "champion", m.id)
        await alias_svc.set_alias(session, "ks_AliasForModel", "candidate", m.id)

        result = await alias_svc.get_aliases_for_model(session, m.id)
        assert len(result) == 2


class TestAliasServiceDelete:
    """Tests for AliasService.delete_alias."""

    async def test_delete_alias_success(self, session: AsyncSession) -> None:
        """Deleting an existing alias returns True."""
        model_svc = ModelService()
        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_AliasDel", version="v1")
        )

        alias_svc = AliasService()
        await alias_svc.set_alias(session, "ks_AliasDel", "champion", m.id)
        result = await alias_svc.delete_alias(session, "ks_AliasDel", "champion")
        assert result is True

        # Verify it's gone
        assert await alias_svc.get_alias(session, "ks_AliasDel", "champion") is None

    async def test_delete_alias_not_found(self, session: AsyncSession) -> None:
        """Deleting a nonexistent alias returns False."""
        alias_svc = AliasService()
        result = await alias_svc.delete_alias(session, "ks_None", "champion")
        assert result is False


class TestAliasServiceResolve:
    """Tests for AliasService.resolve_alias."""

    async def test_resolve_alias_success(self, session: AsyncSession) -> None:
        """Resolving an alias returns the full ModelVersion."""
        model_svc = ModelService()
        m = await model_svc.register_model(
            session, RegisterModelRequest(schema_id="ks_AliasRes", version="v1")
        )

        alias_svc = AliasService()
        await alias_svc.set_alias(session, "ks_AliasRes", "champion", m.id)

        resolved = await alias_svc.resolve_alias(session, "ks_AliasRes", "champion")
        assert resolved is not None
        assert resolved.id == m.id

    async def test_resolve_alias_not_found(self, session: AsyncSession) -> None:
        """Resolving a nonexistent alias returns None."""
        alias_svc = AliasService()
        result = await alias_svc.resolve_alias(session, "ks_None", "champion")
        assert result is None


# ---------------------------------------------------------------------------
# ModelService tag unit tests
# ---------------------------------------------------------------------------


class TestModelServiceTags:
    """Tests for ModelService tag operations."""

    async def test_set_and_get_tags(self, session: AsyncSession) -> None:
        """Setting and getting tags works correctly."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_Tags", version="v1")
        )
        tags = await svc.set_tags(session, m.id, {"env": "prod", "team": "ml"})
        assert tags == {"env": "prod", "team": "ml"}

        result = await svc.get_tags(session, m.id)
        assert result == {"env": "prod", "team": "ml"}

    async def test_set_tags_updates_existing(self, session: AsyncSession) -> None:
        """Setting a tag that exists updates it."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_TagUpd", version="v1")
        )
        await svc.set_tags(session, m.id, {"env": "staging"})
        tags = await svc.set_tags(session, m.id, {"env": "prod"})
        assert tags["env"] == "prod"

    async def test_set_tags_model_not_found(self, session: AsyncSession) -> None:
        """Setting tags on a nonexistent model raises ValueError."""
        svc = ModelService()
        with pytest.raises(ValueError, match="Model not found"):
            await svc.set_tags(session, uuid4(), {"env": "test"})

    async def test_get_tags_empty(self, session: AsyncSession) -> None:
        """Getting tags for a model with none returns empty dict."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_NoTags", version="v1")
        )
        assert await svc.get_tags(session, m.id) == {}

    async def test_delete_tag(self, session: AsyncSession) -> None:
        """Deleting a tag removes it."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_TagDel", version="v1")
        )
        await svc.set_tags(session, m.id, {"env": "prod", "team": "ml"})
        result = await svc.delete_tag(session, m.id, "env")
        assert result is True
        tags = await svc.get_tags(session, m.id)
        assert "env" not in tags
        assert tags["team"] == "ml"

    async def test_delete_tag_not_found(self, session: AsyncSession) -> None:
        """Deleting a nonexistent tag returns False."""
        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_TagDelNF", version="v1")
        )
        assert await svc.delete_tag(session, m.id, "nonexistent") is False

    async def test_search_by_tag(self, session: AsyncSession) -> None:
        """Searching by tag returns matching models."""
        import uuid

        tag_key = f"team_{uuid.uuid4().hex[:8]}"
        svc = ModelService()
        m1 = await svc.register_model(
            session, RegisterModelRequest(schema_id=f"ks_TagSearch1_{tag_key}", version="v1")
        )
        m2 = await svc.register_model(
            session, RegisterModelRequest(schema_id=f"ks_TagSearch2_{tag_key}", version="v1")
        )
        await svc.set_tags(session, m1.id, {tag_key: "alpha"})
        await svc.set_tags(session, m2.id, {tag_key: "beta"})

        results = await svc.search_by_tag(session, tag_key)
        assert len(results) == 2

        results_exact = await svc.search_by_tag(session, tag_key, "alpha")
        assert len(results_exact) == 1
        assert results_exact[0].id == m1.id

    async def test_register_with_tags(self, session: AsyncSession) -> None:
        """Registering a model with tags persists them."""
        svc = ModelService()
        request = RegisterModelRequest(
            schema_id="ks_InlineTag",
            version="v1",
            tags={"job_id": "123", "batch_id": "456"},
        )
        m = await svc.register_model(session, request)
        tags = await svc.get_tags(session, m.id)
        assert tags == {"batch_id": "456", "job_id": "123"}


# ---------------------------------------------------------------------------
# ModelService comparison unit tests
# ---------------------------------------------------------------------------


class TestModelServiceCompare:
    """Tests for ModelService.compare_versions."""

    async def test_compare_no_models(self, session: AsyncSession) -> None:
        """Comparing versions for a schema with no models returns empty."""
        svc = ModelService()
        result = await svc.compare_versions(session, "ks_NoCompare")
        assert result == []

    async def test_compare_all(self, session: AsyncSession) -> None:
        """Comparing without specific versions returns latest."""
        svc = ModelService()
        for i in range(3):
            await svc.register_model(
                session, RegisterModelRequest(schema_id="ks_CompAll", version=f"v{i + 1}")
            )
        result = await svc.compare_versions(session, "ks_CompAll")
        assert len(result) == 3

    async def test_compare_specific_versions(self, session: AsyncSession) -> None:
        """Comparing specific versions returns only those."""
        svc = ModelService()
        for i in range(4):
            await svc.register_model(
                session, RegisterModelRequest(schema_id="ks_CompSpec", version=f"v{i + 1}")
            )
        result = await svc.compare_versions(session, "ks_CompSpec", ["v1", "v3"])
        versions = {s.version for s in result}
        assert versions == {"v1", "v3"}


# ---------------------------------------------------------------------------
# ModelService deploy status unit tests
# ---------------------------------------------------------------------------


class TestModelServiceDeployStatus:
    """Tests for ModelService.update_deploy_status."""

    async def test_update_deploy_status(self, session: AsyncSession) -> None:
        """Updating deploy status sets the fields correctly."""
        from models.domain import DeployFeedbackRequest, DeployStatus

        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_DepStatus", version="v1")
        )
        result = await svc.update_deploy_status(
            session, m.id, DeployFeedbackRequest(deploy_status=DeployStatus.PENDING)
        )
        assert result.deploy_status == DeployStatus.PENDING

    async def test_update_deploy_status_succeeded(self, session: AsyncSession) -> None:
        """Setting deploy_status to succeeded sets deploy_completed_at."""
        from models.domain import DeployFeedbackRequest, DeployStatus

        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_DepOK", version="v1")
        )
        result = await svc.update_deploy_status(
            session, m.id, DeployFeedbackRequest(deploy_status=DeployStatus.SUCCEEDED)
        )
        assert result.deploy_status == DeployStatus.SUCCEEDED
        assert result.deploy_completed_at is not None

    async def test_update_deploy_status_failed_with_error(self, session: AsyncSession) -> None:
        """Setting deploy_status to failed records the error."""
        from models.domain import DeployFeedbackRequest, DeployStatus

        svc = ModelService()
        m = await svc.register_model(
            session, RegisterModelRequest(schema_id="ks_DepFail", version="v1")
        )
        result = await svc.update_deploy_status(
            session,
            m.id,
            DeployFeedbackRequest(
                deploy_status=DeployStatus.FAILED,
                deploy_error="CrashLoopBackOff",
            ),
        )
        assert result.deploy_status == DeployStatus.FAILED
        assert result.deploy_error == "CrashLoopBackOff"
        assert result.deploy_completed_at is not None

    async def test_update_deploy_status_not_found(self, session: AsyncSession) -> None:
        """Updating deploy status on a nonexistent model raises ValueError."""
        from models.domain import DeployFeedbackRequest, DeployStatus

        svc = ModelService()
        with pytest.raises(ValueError, match="Model not found"):
            await svc.update_deploy_status(
                session, uuid4(), DeployFeedbackRequest(deploy_status=DeployStatus.PENDING)
            )
