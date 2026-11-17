"""Tests for deployment tracking feature."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from models.domain import (
    DeployRequest,
    DeployStatus,
    ModelStatus,
    RegisterModelRequest,
    UpdateStatusRequest,
)
from services.audit_service import AuditService
from services.deploy_service import DeployService
from services.errors import DeployError
from services.model_service import ModelService
from services.routing_service import RoutingService
from tests.helpers import gitops_publish_succeeds, gitops_push_fails


class TestDeployStatusFeedback:
    """Tests for PATCH /api/models/{model_id}/deploy-status."""

    async def test_deploy_status_update(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Updating deploy status succeeds."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        status_resp = await client.patch(
            f"/api/models/{model_id}/deploy-status",
            json={"deploy_status": "pending"},
        )
        assert status_resp.status_code == 200
        body = status_resp.json()
        assert body["deploy_status"] == "pending"
        assert body["deploy_started_at"] is None

    async def test_deploy_status_succeeded_sets_completed_at(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Setting deploy_status to succeeded sets deploy_completed_at."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        await client.patch(
            f"/api/models/{model_id}/deploy-status",
            json={"deploy_status": "in_progress"},
        )
        status_resp = await client.patch(
            f"/api/models/{model_id}/deploy-status",
            json={"deploy_status": "succeeded"},
        )
        assert status_resp.status_code == 200
        body = status_resp.json()
        assert body["deploy_status"] == "succeeded"
        assert body["deploy_completed_at"] is not None
        assert body["deploy_error"] is None

    async def test_deploy_status_failed_with_error(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Setting deploy_status to failed stores the error message."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        status_resp = await client.patch(
            f"/api/models/{model_id}/deploy-status",
            json={
                "deploy_status": "failed",
                "deploy_error": "Pod CrashLoopBackOff",
            },
        )
        assert status_resp.status_code == 200
        body = status_resp.json()
        assert body["deploy_status"] == "failed"
        assert body["deploy_error"] == "Pod CrashLoopBackOff"
        assert body["deploy_completed_at"] is not None
        assert body["deploy_started_at"] is None

    async def test_deploy_status_not_found(self, client: AsyncClient) -> None:
        """Updating deploy status for nonexistent model returns 404."""
        resp = await client.patch(
            "/api/models/00000000-0000-0000-0000-000000000000/deploy-status",
            json={"deploy_status": "pending"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    async def test_deploy_fields_on_model_version(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Deploy fields are visible on the ModelVersion response."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        # Initially null
        get_resp = await client.get(f"/api/models/{model_id}")
        body = get_resp.json()
        assert body["deploy_status"] is None
        assert body["deploy_error"] is None
        assert body["deploy_started_at"] is None
        assert body["deploy_completed_at"] is None

        # After update
        await client.patch(
            f"/api/models/{model_id}/deploy-status",
            json={"deploy_status": "in_progress"},
        )
        get_resp2 = await client.get(f"/api/models/{model_id}")
        assert get_resp2.json()["deploy_status"] == "in_progress"

    async def test_deploy_status_overwrite(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Deploy status can be freely overwritten with no state machine enforcement."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        # Set to in_progress
        r1 = await client.patch(
            f"/api/models/{model_id}/deploy-status",
            json={"deploy_status": "in_progress"},
        )
        assert r1.status_code == 200
        assert r1.json()["deploy_status"] == "in_progress"

        # Set to succeeded
        r2 = await client.patch(
            f"/api/models/{model_id}/deploy-status",
            json={"deploy_status": "succeeded"},
        )
        assert r2.status_code == 200
        assert r2.json()["deploy_status"] == "succeeded"

        # Set back to pending (no state machine enforcement)
        r3 = await client.patch(
            f"/api/models/{model_id}/deploy-status",
            json={"deploy_status": "pending"},
        )
        assert r3.status_code == 200
        assert r3.json()["deploy_status"] == "pending"

    async def test_deploy_status_without_error_clears_previous_error(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Setting succeeded after failed does NOT auto-clear deploy_error."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        # Set failed with error
        await client.patch(
            f"/api/models/{model_id}/deploy-status",
            json={
                "deploy_status": "failed",
                "deploy_error": "OOMKilled",
            },
        )

        # Set succeeded without explicitly clearing deploy_error
        r2 = await client.patch(
            f"/api/models/{model_id}/deploy-status",
            json={"deploy_status": "succeeded"},
        )
        assert r2.status_code == 200
        body = r2.json()
        assert body["deploy_status"] == "succeeded"
        # deploy_error persists unless explicitly set to null
        assert body["deploy_error"] == "OOMKilled"

    async def test_deploy_not_approved_returns_400(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Deploying a model in training status returns 400."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        # Model is still in 'training' status
        deploy_resp = await client.post(f"/api/deploy/{model_id}")
        assert deploy_resp.status_code == 400

    async def test_deploy_nonexistent_model_returns_404(self, client: AsyncClient) -> None:
        """Deploying a non-existent model UUID returns 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        deploy_resp = await client.post(f"/api/deploy/{fake_id}")
        assert deploy_resp.status_code == 404


class TestDeployAutoChampionRollback:
    """Tests that deployment auto-sets champion/rollback aliases."""

    async def test_deploy_sets_champion_alias(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """After deploying a model, the champion alias points to it."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]
        schema_id = sample_model_data["schema_id"]

        # Walk to approved
        for status in ["evaluating", "evaluated", "approved"]:
            await client.patch(
                f"/api/models/{model_id}/status",
                json={"status": status},
            )

        # Deploy
        with gitops_publish_succeeds():
            deploy_resp = await client.post(f"/api/deploy/{model_id}")
        assert deploy_resp.status_code == 200

        # Check champion alias
        alias_resp = await client.get(f"/api/aliases/{schema_id}/champion")
        assert alias_resp.status_code == 200
        assert alias_resp.json()["model_id"] == model_id

        # Verify routing table has been updated for the schema
        routing_resp = await client.get(f"/api/routing/{schema_id}")
        assert routing_resp.status_code == 200
        assert routing_resp.json()["active_model_id"] == model_id

        # Verify model status is now "deployed"
        model_resp = await client.get(f"/api/models/{model_id}")
        assert model_resp.status_code == 200
        assert model_resp.json()["status"] == "deployed"

    async def test_deploy_moves_previous_champion_to_rollback(self, client: AsyncClient) -> None:
        """When deploying a new model, the previous champion becomes rollback."""
        schema_id = "ks_RollbackTest"

        # Deploy model v1
        resp1 = await client.post(
            "/api/models",
            json={"schema_id": schema_id, "version": "v1"},
        )
        mid1 = resp1.json()["id"]
        for status in ["evaluating", "evaluated", "approved"]:
            await client.patch(f"/api/models/{mid1}/status", json={"status": status})
        with gitops_publish_succeeds():
            await client.post(f"/api/deploy/{mid1}")

        # Deploy model v2
        resp2 = await client.post(
            "/api/models",
            json={"schema_id": schema_id, "version": "v2"},
        )
        mid2 = resp2.json()["id"]
        for status in ["evaluating", "evaluated", "approved"]:
            await client.patch(f"/api/models/{mid2}/status", json={"status": status})
        with gitops_publish_succeeds():
            await client.post(f"/api/deploy/{mid2}")

        # Champion should be v2
        champ_resp = await client.get(f"/api/aliases/{schema_id}/champion")
        assert champ_resp.status_code == 200
        assert champ_resp.json()["model_id"] == mid2

        # Rollback should be v1
        roll_resp = await client.get(f"/api/aliases/{schema_id}/rollback")
        assert roll_resp.status_code == 200
        assert roll_resp.json()["model_id"] == mid1

        # Verify v1 still has status "deployed" (not changed by v2 deployment)
        model1_resp = await client.get(f"/api/models/{mid1}")
        assert model1_resp.status_code == 200
        assert model1_resp.json()["status"] == "deployed"

        # Both models can be resolved via their respective aliases
        champ_alias = await client.get(f"/api/aliases/{schema_id}/champion")
        assert champ_alias.status_code == 200
        assert champ_alias.json()["model_id"] == mid2

        roll_alias = await client.get(f"/api/aliases/{schema_id}/rollback")
        assert roll_alias.status_code == 200
        assert roll_alias.json()["model_id"] == mid1


class TestDeployStatusTransitions:
    """Tests for deploy_status transitions (SUCCEEDED/FAILED) in deploy_model()."""

    @staticmethod
    async def _create_approved_model(
        session: AsyncSession,
        model_svc: ModelService,
        schema_id: str = "ks_DeployTransition",
        version: str = "v1",
    ) -> Any:
        """Register a model and walk it to approved status.

        Returns:
            The model domain object in 'approved' status.
        """
        model = await model_svc.register_model(
            session,
            RegisterModelRequest(schema_id=schema_id, version=version),
        )
        for status in [
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
        ]:
            model = await model_svc.update_status(
                session, model.id, UpdateStatusRequest(status=status)
            )
        return model

    async def test_successful_deploy_sets_status_succeeded(self, session: AsyncSession) -> None:
        """After a successful deployment, deploy_status is SUCCEEDED.

        NOTE: this test previously ran with ``gitops_enabled=False`` and still
        asserted SUCCEEDED — i.e. it encoded the defect that a gitops push
        which was never attempted counts as a successful deploy. "Successful"
        now requires the values file to have actually reached gitops.
        """
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc)

        model = await self._create_approved_model(session, model_svc)
        with gitops_publish_succeeds(deployment_tracking_enabled=True):
            result = await deploy_svc.deploy_model(session, model.id, DeployRequest())

        assert result.status == ModelStatus.DEPLOYED

        # Fetch the model and verify deploy_status is succeeded
        updated = await model_svc.get_model(session, model.id)
        assert updated is not None
        assert updated.deploy_status == DeployStatus.SUCCEEDED

    async def test_successful_deploy_sets_completed_at(self, session: AsyncSession) -> None:
        """After a successful deployment, deploy_completed_at is set."""
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc)

        model = await self._create_approved_model(
            session, model_svc, schema_id="ks_CompletedAt", version="v1"
        )
        with gitops_publish_succeeds(deployment_tracking_enabled=True):
            await deploy_svc.deploy_model(session, model.id, DeployRequest())

        updated = await model_svc.get_model(session, model.id)
        assert updated is not None
        assert updated.deploy_completed_at is not None

    async def test_successful_deploy_has_no_error(self, session: AsyncSession) -> None:
        """After a successful deployment, deploy_error is None."""
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc)

        model = await self._create_approved_model(
            session, model_svc, schema_id="ks_NoError", version="v1"
        )
        with gitops_publish_succeeds(deployment_tracking_enabled=True):
            await deploy_svc.deploy_model(session, model.id, DeployRequest())

        updated = await model_svc.get_model(session, model.id)
        assert updated is not None
        assert updated.deploy_error is None

    async def test_failed_routing_update_sets_status_failed(self, session: AsyncSession) -> None:
        """When routing update fails, deploy_status is set to FAILED."""
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc)

        model = await self._create_approved_model(
            session, model_svc, schema_id="ks_FailRouting", version="v1"
        )
        # Make routing update raise an exception
        with (
            gitops_publish_succeeds(deployment_tracking_enabled=True),
            patch.object(
                routing_svc,
                "update_route_for_deployment",
                new=AsyncMock(side_effect=RuntimeError("Routing table update failed")),
            ),
            pytest.raises(RuntimeError, match="Routing table update failed"),
        ):
            await deploy_svc.deploy_model(session, model.id, DeployRequest())

        updated = await model_svc.get_model(session, model.id)
        assert updated is not None
        assert updated.deploy_status == DeployStatus.FAILED

    async def test_failed_deploy_sets_completed_at(self, session: AsyncSession) -> None:
        """When deployment fails, deploy_completed_at is still set."""
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc)

        model = await self._create_approved_model(
            session, model_svc, schema_id="ks_FailCompleted", version="v1"
        )
        with (
            gitops_publish_succeeds(deployment_tracking_enabled=True),
            patch.object(
                routing_svc,
                "update_route_for_deployment",
                new=AsyncMock(side_effect=RuntimeError("Connection refused")),
            ),
            pytest.raises(RuntimeError, match="Connection refused"),
        ):
            await deploy_svc.deploy_model(session, model.id, DeployRequest())

        updated = await model_svc.get_model(session, model.id)
        assert updated is not None
        assert updated.deploy_completed_at is not None

    async def test_failed_deploy_stores_error_message(self, session: AsyncSession) -> None:
        """When deployment fails, deploy_error contains the exception message."""
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc)

        model = await self._create_approved_model(
            session, model_svc, schema_id="ks_ErrorMsg", version="v1"
        )
        error_msg = "Database connection lost during routing update"

        with (
            gitops_publish_succeeds(deployment_tracking_enabled=True),
            patch.object(
                routing_svc,
                "update_route_for_deployment",
                new=AsyncMock(side_effect=RuntimeError(error_msg)),
            ),
            pytest.raises(RuntimeError, match=error_msg),
        ):
            await deploy_svc.deploy_model(session, model.id, DeployRequest())

        updated = await model_svc.get_model(session, model.id)
        assert updated is not None
        assert updated.deploy_error == error_msg

    async def test_failed_deploy_re_raises_exception(self, session: AsyncSession) -> None:
        """After setting FAILED status, the exception is re-raised."""
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc)

        model = await self._create_approved_model(
            session, model_svc, schema_id="ks_ReRaise", version="v1"
        )
        with (
            gitops_publish_succeeds(deployment_tracking_enabled=True),
            patch.object(
                routing_svc,
                "update_route_for_deployment",
                new=AsyncMock(side_effect=ValueError("Something went wrong")),
            ),
            pytest.raises(ValueError, match="Something went wrong"),
        ):
            await deploy_svc.deploy_model(session, model.id, DeployRequest())


class TestGgufDeployFailsClosed:
    """A GGUF deploy that never reached gitops must not report success."""

    @staticmethod
    async def _approved(session: AsyncSession, model_svc: ModelService, schema_id: str) -> Any:
        """Register a GGUF model and walk it to approved.

        Returns:
            The approved model domain object.
        """
        model = await model_svc.register_model(
            session, RegisterModelRequest(schema_id=schema_id, version="v1")
        )
        for status in [ModelStatus.EVALUATING, ModelStatus.EVALUATED, ModelStatus.APPROVED]:
            model = await model_svc.update_status(
                session, model.id, UpdateStatusRequest(status=status)
            )
        return model

    async def test_gguf_deploy_raises_when_gitops_disabled(self, session: AsyncSession) -> None:
        """gitops_enabled=False → DeployError, deploy_status FAILED, error recorded."""
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc)

        model = await self._approved(session, model_svc, "ks_GitopsOff")

        with (
            gitops_publish_succeeds(deployment_tracking_enabled=True, gitops_enabled=False),
            pytest.raises(DeployError, match="did not reach gitops"),
        ):
            await deploy_svc.deploy_model(session, model.id, DeployRequest())

        updated = await model_svc.get_model(session, model.id)
        assert updated is not None
        # Back to APPROVED, not stranded in EXPORTING — the deploy is retryable.
        assert updated.status == ModelStatus.APPROVED
        assert updated.deploy_status == DeployStatus.FAILED
        assert updated.deploy_error

    async def test_gguf_deploy_raises_when_push_fails(self, session: AsyncSession) -> None:
        """gitops_enabled=True but push fails → DeployError, deploy_status FAILED."""
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc)

        model = await self._approved(session, model_svc, "ks_PushFails")

        with (
            gitops_push_fails(deployment_tracking_enabled=True),
            pytest.raises(DeployError, match="git commit/push failed"),
        ):
            await deploy_svc.deploy_model(session, model.id, DeployRequest())

        updated = await model_svc.get_model(session, model.id)
        assert updated is not None
        # Back to APPROVED, not stranded in EXPORTING — the deploy is retryable.
        assert updated.status == ModelStatus.APPROVED
        assert updated.deploy_status == DeployStatus.FAILED
        assert updated.deploy_error

    async def test_failed_gitops_does_not_write_routing_rule(self, session: AsyncSession) -> None:
        """No routing rule may be written for a deploy that did not happen."""
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc)

        model = await self._approved(session, model_svc, "ks_NoRouteOnFail")

        with (
            gitops_push_fails(deployment_tracking_enabled=True),
            pytest.raises(DeployError),
        ):
            await deploy_svc.deploy_model(session, model.id, DeployRequest())

        route = await routing_svc.get_route(session, "ks_NoRouteOnFail")
        assert route is None


class TestRecordDeployFailure:
    """The failure recorder must never replace the failure it is recording."""

    async def test_unknown_model_is_a_no_op(self, session: AsyncSession) -> None:
        """Recording against a model that no longer exists returns quietly."""
        await DeployService._record_deploy_failure(session, uuid4(), RuntimeError("original"))

    async def test_recording_error_does_not_mask_the_original(self, session: AsyncSession) -> None:
        """A DB error while recording is logged and swallowed, not raised.

        The caller re-raises the real exception immediately afterwards; a
        recording failure that propagated would replace a meaningful
        DeployError with a confusing SQLAlchemyError.
        """
        with patch.object(session, "rollback", side_effect=SQLAlchemyError("session poisoned")):
            await DeployService._record_deploy_failure(session, uuid4(), RuntimeError("original"))


class TestPeftDeployment:
    """Tests for PEFT (vLLM LoRA adapter) deployment path."""

    @staticmethod
    async def _create_approved_peft_model(
        session: AsyncSession,
        model_svc: ModelService,
        schema_id: str = "ks_PeftTest",
        version: str = "v1",
        adapter_path: str = "adapters/ks_PeftTest/v1/",
    ) -> Any:
        """Register a PEFT model and walk it to approved status."""
        model = await model_svc.register_model(
            session,
            RegisterModelRequest(
                schema_id=schema_id,
                version=version,
                model_format="peft",
                adapter_path=adapter_path,
            ),
        )
        for status in [
            ModelStatus.EVALUATING,
            ModelStatus.EVALUATED,
            ModelStatus.APPROVED,
        ]:
            model = await model_svc.update_status(
                session, model.id, UpdateStatusRequest(status=status)
            )
        return model

    async def test_peft_deploy_calls_load_adapter(self, session: AsyncSession) -> None:
        """PEFT deployment calls _load_adapter with the adapter path."""
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc)

        model = await self._create_approved_peft_model(session, model_svc)

        mock_settings = MagicMock()
        mock_settings.gitops_repo_path = "/tmp/test-gitops"
        mock_settings.gitops_enabled = False
        mock_settings.deployment_tracking_enabled = False
        mock_settings.rtx_endpoint = "http://vllm:8000"
        mock_settings.adapter_manager_url = "http://vllm:8001"

        with (
            patch("services.deploy_service.get_settings", return_value=mock_settings),
            patch.object(DeployService, "_load_adapter", new=AsyncMock()) as mock_load,
        ):
            result = await deploy_svc.deploy_model(session, model.id, DeployRequest())

        assert result.status == ModelStatus.DEPLOYED
        assert result.values_file is None
        assert "vLLM LoRA adapter" in result.message
        mock_load.assert_called_once_with("ks_PeftTest", "adapters/ks_PeftTest/v1/")

    async def test_peft_deploy_uses_shared_rtx_endpoint(self, session: AsyncSession) -> None:
        """PEFT deployment routes to the shared vLLM endpoint, not per-schema."""
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc)

        model = await self._create_approved_peft_model(
            session, model_svc, schema_id="ks_PeftEndpoint"
        )

        mock_settings = MagicMock()
        mock_settings.gitops_repo_path = "/tmp/test-gitops"
        mock_settings.gitops_enabled = False
        mock_settings.deployment_tracking_enabled = False
        mock_settings.rtx_endpoint = "http://vllm:8000"
        mock_settings.adapter_manager_url = "http://vllm:8001"

        with (
            patch("services.deploy_service.get_settings", return_value=mock_settings),
            patch.object(DeployService, "_load_adapter", new=AsyncMock()),
        ):
            await deploy_svc.deploy_model(session, model.id, DeployRequest())

        route = await routing_svc.get_route(session, "ks_PeftEndpoint")
        assert route is not None
        assert route.endpoint == "http://vllm:8000"
        assert route.adapter_path == "adapters/ks_PeftTest/v1/"

    async def test_peft_deploy_without_adapter_path_raises(self, session: AsyncSession) -> None:
        """PEFT model without adapter_path is a failed deploy, not a warning.

        Previously this asserted DEPLOYED with only a ``logger.warning``: the
        model was routed to the shared vLLM endpoint with no adapter loaded, so
        the base weights served silently under the fine-tune's name.
        """
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc)

        model = await self._create_approved_peft_model(
            session,
            model_svc,
            schema_id="ks_PeftNoPath",
            adapter_path="",
        )

        mock_settings = MagicMock()
        mock_settings.gitops_repo_path = "/tmp/test-gitops"
        mock_settings.gitops_enabled = False
        mock_settings.deployment_tracking_enabled = True
        mock_settings.rtx_endpoint = "http://vllm:8000"
        mock_settings.adapter_manager_url = "http://vllm:8001"

        with (
            patch("services.deploy_service.get_settings", return_value=mock_settings),
            patch.object(DeployService, "_load_adapter", new=AsyncMock()) as mock_load,
            pytest.raises(DeployError, match="no adapter_path"),
        ):
            await deploy_svc.deploy_model(session, model.id, DeployRequest())

        mock_load.assert_not_called()

        updated = await model_svc.get_model(session, model.id)
        assert updated is not None
        assert updated.status == ModelStatus.APPROVED
        assert updated.deploy_status == DeployStatus.FAILED

        # No routing rule for a model that was never actually served.
        assert await routing_svc.get_route(session, "ks_PeftNoPath") is None

    async def test_peft_deploy_raises_when_adapter_manager_url_empty(
        self, session: AsyncSession
    ) -> None:
        """An unset adapter-manager URL fails closed and is recorded as FAILED.

        Proves the widened try-boundary: this failure happens during the
        placement step (step 3), which used to sit outside the failure-tracking
        guard, leaving the model stranded in EXPORTING with deploy_status
        untouched.
        """
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc)

        model = await self._create_approved_peft_model(
            session,
            model_svc,
            schema_id="ks_PeftNoAdapterMgr",
            adapter_path="adapters/ks_PeftNoAdapterMgr/v1/",
        )

        mock_settings = MagicMock()
        mock_settings.gitops_repo_path = "/tmp/test-gitops"
        mock_settings.gitops_enabled = False
        mock_settings.deployment_tracking_enabled = True
        mock_settings.rtx_endpoint = "http://vllm:8000"
        mock_settings.adapter_manager_url = ""

        with (
            patch("services.deploy_service.get_settings", return_value=mock_settings),
            pytest.raises(DeployError, match="ADAPTER_MANAGER_URL"),
        ):
            await deploy_svc.deploy_model(session, model.id, DeployRequest())

        updated = await model_svc.get_model(session, model.id)
        assert updated is not None
        assert updated.deploy_status == DeployStatus.FAILED
        assert updated.deploy_error

    async def test_peft_deploy_no_values_file_generated(self, session: AsyncSession) -> None:
        """PEFT deployment does not generate Helm values YAML."""
        audit_svc = AuditService()
        model_svc = ModelService(audit_service=audit_svc)
        routing_svc = RoutingService(audit_service=audit_svc)
        deploy_svc = DeployService(model_svc, routing_svc)

        model = await self._create_approved_peft_model(
            session, model_svc, schema_id="ks_PeftNoValues"
        )

        mock_settings = MagicMock()
        mock_settings.gitops_repo_path = "/tmp/test-gitops"
        mock_settings.gitops_enabled = False
        mock_settings.deployment_tracking_enabled = False
        mock_settings.rtx_endpoint = "http://vllm:8000"
        mock_settings.adapter_manager_url = "http://vllm:8001"

        with (
            patch("services.deploy_service.get_settings", return_value=mock_settings),
            patch.object(DeployService, "_load_adapter", new=AsyncMock()),
            patch.object(DeployService, "_write_values_file") as mock_write,
        ):
            result = await deploy_svc.deploy_model(session, model.id, DeployRequest())

        assert result.values_file is None
        mock_write.assert_not_called()

    async def test_load_adapter_calls_adapter_manager(self) -> None:
        """_load_adapter POSTs to the adapter-manager load endpoint."""
        mock_settings = MagicMock()
        mock_settings.adapter_manager_url = "http://adapter-mgr:8001"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with (
            patch("services.deploy_service.get_settings", return_value=mock_settings),
            patch("services.deploy_service.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await DeployService._load_adapter("ks_Test", "adapters/ks_Test/v1/")

        mock_client.post.assert_called_once_with(
            "http://adapter-mgr:8001/api/adapters/ks_Test/load",
            json={"adapter_path": "adapters/ks_Test/v1/"},
            timeout=60.0,
        )
        mock_response.raise_for_status.assert_called_once()
