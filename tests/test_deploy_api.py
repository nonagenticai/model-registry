"""Tests for deployment trigger API endpoints (/api/deploy)."""

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient

from tests.helpers import gitops_publish_succeeds, gitops_push_fails


class TestDeployModel:
    """Tests for POST /api/deploy/{model_id}."""

    async def _create_and_approve_model(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> str:
        """Helper to register a model and walk it to 'approved' status.

        Args:
            client: The test HTTP client.
            sample_model_data: Model registration payload.

        Returns:
            The model UUID as a string.
        """
        create_resp = await client.post("/api/models", json=sample_model_data)
        model_id = create_resp.json()["id"]

        for status in ["evaluating", "evaluated", "approved"]:
            resp = await client.patch(
                f"/api/models/{model_id}/status",
                json={"status": status},
            )
            assert resp.status_code == 200, f"Failed moving to {status}: {resp.json()}"

        return model_id

    async def test_deploy_approved_model_success(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Deploying an approved model succeeds and returns deployment details."""
        model_id = await self._create_and_approve_model(client, sample_model_data)

        with gitops_publish_succeeds():
            response = await client.post(
                f"/api/deploy/{model_id}",
                json={"target_namespace": "llm-inference", "replicas": 2},
            )
        assert response.status_code == 200

        body = response.json()
        # Verify all expected fields are present with correct values
        assert body["model_id"] == model_id
        assert body["schema_id"] == sample_model_data["schema_id"]
        assert body["status"] == "deployed"
        assert isinstance(body["values_file"], str)
        assert len(body["values_file"]) > 0
        assert "values-" in body["values_file"]
        assert "deployed" in body["message"].lower() or "Deployed" in body["message"]

        # Verify the response contains exactly the expected fields
        assert set(body.keys()) == {"model_id", "schema_id", "status", "values_file", "message"}

    async def test_deploy_creates_values_file(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Deployment writes a values YAML file to the gitops repo path."""
        model_id = await self._create_and_approve_model(client, sample_model_data)

        gitops_path = os.environ.get("MODEL_REGISTRY_GITOPS_REPO_PATH", "/tmp/test-gitops")
        values_dir = Path(gitops_path) / "clusters" / "production" / "llm-inference"

        with gitops_publish_succeeds():
            response = await client.post(f"/api/deploy/{model_id}", json={})
        assert response.status_code == 200

        # Check that the values file was created
        expected_file = values_dir / f"values-{sample_model_data['schema_id']}.yaml"
        assert expected_file.exists(), f"Expected values file at {expected_file}"

        content = expected_file.read_text()
        assert sample_model_data["schema_id"] in content
        assert "ggufPath" in content
        assert "replicas" in content
        assert "endpoint:" in content
        assert "svc.example.com:8080" in content

        # Cleanup
        expected_file.unlink(missing_ok=True)

    async def test_deploy_updates_routing_table(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """After deployment, the routing table is updated for the schema."""
        model_id = await self._create_and_approve_model(client, sample_model_data)

        with gitops_publish_succeeds():
            await client.post(f"/api/deploy/{model_id}", json={})

        # Check routing rule was created/updated
        routing_resp = await client.get(f"/api/routing/{sample_model_data['schema_id']}")
        assert routing_resp.status_code == 200

        rule = routing_resp.json()
        assert rule["active_model_id"] == model_id
        assert isinstance(rule["endpoint"], str)
        assert len(rule["endpoint"]) > 0
        assert "svc.example.com" in rule["endpoint"]
        # Endpoint must contain the schema name (lowercased/hyphenated form)
        schema_lower = sample_model_data["schema_id"].replace("_", "-").lower()
        assert schema_lower in rule["endpoint"]

    async def test_deploy_model_transitions_to_deployed(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """After deployment, the model status is 'deployed'."""
        model_id = await self._create_and_approve_model(client, sample_model_data)

        with gitops_publish_succeeds():
            await client.post(f"/api/deploy/{model_id}", json={})

        model_resp = await client.get(f"/api/models/{model_id}")
        assert model_resp.json()["status"] == "deployed"

    async def test_deploy_non_approved_model_returns_400(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Deploying a model that is not in 'approved' status returns 400."""
        create_resp = await client.post("/api/models", json=sample_model_data)
        model_id = create_resp.json()["id"]

        # Model is still in 'training' status
        response = await client.post(f"/api/deploy/{model_id}", json={})
        assert response.status_code == 400
        assert "approved" in response.json()["detail"].lower()

    async def test_deploy_nonexistent_model_returns_404(self, client: AsyncClient) -> None:
        """Deploying a non-existent model returns 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.post(f"/api/deploy/{fake_id}", json={})
        assert response.status_code == 404

    async def test_deploy_already_deployed_returns_400(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Deploying an already-deployed model returns 400."""
        model_id = await self._create_and_approve_model(client, sample_model_data)

        # First deploy succeeds
        with gitops_publish_succeeds():
            resp1 = await client.post(f"/api/deploy/{model_id}", json={})
        assert resp1.status_code == 200

        # Second deploy should fail because model is now in 'deployed' status
        with gitops_publish_succeeds():
            resp2 = await client.post(f"/api/deploy/{model_id}", json={})
        assert resp2.status_code == 400

        # Verify the error detail mentions the current status
        detail = resp2.json()["detail"]
        assert "deployed" in detail.lower()

        # Verify the model was not modified by the failed deploy attempt
        model_resp = await client.get(f"/api/models/{model_id}")
        assert model_resp.status_code == 200
        assert model_resp.json()["status"] == "deployed"

    async def test_deploy_with_custom_namespace_and_replicas(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Deploy with custom target_namespace and replicas."""
        # Use a unique schema_id to avoid conflicts
        sample_model_data["schema_id"] = "ks_CustomDeploy"
        model_id = await self._create_and_approve_model(client, sample_model_data)

        with gitops_publish_succeeds():
            response = await client.post(
                f"/api/deploy/{model_id}",
                json={"target_namespace": "llm-staging", "replicas": 3},
            )
        assert response.status_code == 200

        body = response.json()
        assert body["status"] == "deployed"

        # Check values file references the custom namespace
        gitops_path = os.environ.get("MODEL_REGISTRY_GITOPS_REPO_PATH", "/tmp/test-gitops")
        values_dir = Path(gitops_path) / "clusters" / "production" / "llm-inference"
        values_file = values_dir / "values-ks_CustomDeploy.yaml"

        if values_file.exists():
            content = values_file.read_text()
            assert "replicas: 3" in content
            values_file.unlink(missing_ok=True)

    async def test_deploy_calls_git_when_gitops_enabled(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """When gitops_enabled=True, git add/commit/push are executed."""
        sample_model_data["schema_id"] = "ks_GitOpsEnabled"
        model_id = await self._create_and_approve_model(client, sample_model_data)

        # Build a fake process that always succeeds
        fake_proc = AsyncMock()
        fake_proc.communicate.return_value = (b"ok", b"")
        fake_proc.returncode = 0

        mock_settings = MagicMock()
        mock_settings.gitops_repo_path = os.environ.get(
            "MODEL_REGISTRY_GITOPS_REPO_PATH", "/tmp/test-gitops"
        )
        mock_settings.gitops_enabled = True

        with (
            patch(
                "services.deploy_service.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "services.deploy_service.asyncio.create_subprocess_exec",
                return_value=fake_proc,
            ) as mock_exec,
        ):
            response = await client.post(
                f"/api/deploy/{model_id}",
                json={"target_namespace": "llm-inference", "replicas": 1},
            )

        assert response.status_code == 200

        # git add, git commit, git push = 3 calls
        assert mock_exec.call_count == 3

        # Verify the command sequences
        calls = mock_exec.call_args_list
        assert calls[0].args[0] == "git" and calls[0].args[1] == "add"
        assert calls[1].args[0] == "git" and calls[1].args[1] == "commit"
        assert calls[2].args[0] == "git" and calls[2].args[1] == "push"

    async def test_deploy_returns_502_when_gitops_disabled(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """gitops_enabled=False (the shipped default) is a failed deploy, not a success.

        Previously this asserted 200: a GGUF deploy where the push was never
        even attempted reported ``deployed``. The fg-* Service it names can
        never exist, so the honest answer is 502.
        """
        sample_model_data["schema_id"] = "ks_GitOpsDisabled"
        model_id = await self._create_and_approve_model(client, sample_model_data)

        with patch(
            "services.deploy_service.asyncio.create_subprocess_exec",
        ) as mock_exec:
            response = await client.post(
                f"/api/deploy/{model_id}",
                json={"target_namespace": "llm-inference", "replicas": 1},
            )

        assert response.status_code == 502
        detail = response.json()["detail"]
        assert "gitops" in detail.lower()
        assert "MODEL_REGISTRY_GITOPS_ENABLED" in detail
        mock_exec.assert_not_called()

        # The model must NOT be marked deployed.
        model_resp = await client.get(f"/api/models/{model_id}")
        assert model_resp.json()["status"] != "deployed"

    async def test_failed_gitops_does_not_write_routing_rule(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """A deploy that never reached gitops must not poison the routing table.

        This is the load-bearing regression test: an endpoint pointing at a
        Service that will never exist is worse than no endpoint at all, because
        consumers dial a black hole with no upstream signal.
        """
        sample_model_data["schema_id"] = "ks_NoPoisonedRoute"
        model_id = await self._create_and_approve_model(client, sample_model_data)

        with gitops_push_fails():
            response = await client.post(
                f"/api/deploy/{model_id}",
                json={"target_namespace": "llm-inference", "replicas": 1},
            )

        assert response.status_code == 502

        routing_resp = await client.get("/api/routing/ks_NoPoisonedRoute")
        assert routing_resp.status_code == 404, (
            f"routing rule was written for a deploy that never happened: {routing_resp.json()}"
        )

    async def test_failed_deploy_is_persisted_and_retryable(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """A failed deploy is COMMITTED and leaves the model retryable.

        Deliberately API-level: every request gets its own session, so the
        read-backs below run on a fresh connection. A service-level test that
        reads through the same uncommitted session would pass even though
        ``get_session`` rolls the record back on close — the failure record
        would never reach the database.

        Also pins the retry path: ``exporting``'s only other legal exit is
        ``retired`` and deploy requires ``approved``, so a model left in
        ``exporting`` by a failed deploy would be stranded forever.
        """
        sample_model_data["schema_id"] = "ks_RetryAfterFailure"
        model_id = await self._create_and_approve_model(client, sample_model_data)

        with gitops_push_fails(deployment_tracking_enabled=True):
            fail_resp = await client.post(f"/api/deploy/{model_id}", json={})
        assert fail_resp.status_code == 502

        # Fresh connection: proves the failure record was committed, not flushed.
        body = (await client.get(f"/api/models/{model_id}")).json()
        assert body["deploy_status"] == "failed", body
        assert body["deploy_error"], body
        assert body["deploy_completed_at"] is not None, body
        assert body["status"] == "approved", body

        # And the deploy is actually retryable.
        with gitops_publish_succeeds():
            retry_resp = await client.post(f"/api/deploy/{model_id}", json={})
        assert retry_resp.status_code == 200
        assert (await client.get(f"/api/models/{model_id}")).json()["status"] == "deployed"

    async def test_deploy_full_side_effects(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """End-to-end deploy verifies ALL side effects across the system."""
        # Use a unique schema_id to avoid collisions with other tests
        schema_id = "ks_FullSideEffects"
        sample_model_data["schema_id"] = schema_id

        # --- Step 1: Register model ---
        create_resp = await client.post("/api/models", json=sample_model_data)
        assert create_resp.status_code == 201
        model_id = create_resp.json()["id"]

        # --- Step 2: Walk through status transitions to approved ---
        for status in ["evaluating", "evaluated", "approved"]:
            resp = await client.patch(
                f"/api/models/{model_id}/status",
                json={"status": status},
            )
            assert resp.status_code == 200, f"Failed moving to {status}: {resp.json()}"

        # --- Step 3: Deploy ---
        with gitops_publish_succeeds():
            deploy_resp = await client.post(
                f"/api/deploy/{model_id}",
                json={"target_namespace": "llm-inference", "replicas": 2},
            )
        assert deploy_resp.status_code == 200
        deploy_body = deploy_resp.json()
        assert deploy_body["status"] == "deployed"
        assert deploy_body["model_id"] == model_id
        assert deploy_body["schema_id"] == schema_id

        # --- Step 4: Verify model status is "deployed" and deployed_at is set ---
        model_resp = await client.get(f"/api/models/{model_id}")
        assert model_resp.status_code == 200
        model_body = model_resp.json()
        assert model_body["status"] == "deployed"
        assert model_body["deployed_at"] is not None
        assert isinstance(model_body["deployed_at"], str)
        assert len(model_body["deployed_at"]) > 0

        # --- Step 5: Verify routing table has the model ---
        routing_resp = await client.get(f"/api/routing/{schema_id}")
        assert routing_resp.status_code == 200
        rule = routing_resp.json()
        assert rule["active_model_id"] == model_id
        assert isinstance(rule["endpoint"], str)
        assert len(rule["endpoint"]) > 0
        schema_lower = schema_id.replace("_", "-").lower()
        assert schema_lower in rule["endpoint"]

        # --- Step 6: Verify champion alias is set ---
        champion_resp = await client.get(f"/api/aliases/{schema_id}/champion")
        assert champion_resp.status_code == 200
        champion = champion_resp.json()
        assert champion["model_id"] == model_id
        assert champion["schema_id"] == schema_id
        assert champion["alias_name"] == "champion"

        # --- Step 7: Verify audit trail has events for each step ---
        # Model-level audit trail (entity_type=model_version, entity_id=model_id)
        model_audit_resp = await client.get(
            f"/api/audit/models/{model_id}",
            params={"limit": 100},
        )
        assert model_audit_resp.status_code == 200
        model_events = model_audit_resp.json()["events"]
        model_event_types = [e["event_type"] for e in model_events]

        assert "model_registered" in model_event_types
        status_changes = [e for e in model_events if e["event_type"] == "status_changed"]
        # evaluating, evaluated, approved, exporting, deployed = 5 status changes
        assert len(status_changes) >= 5

        # Schema-level audit trail captures routing and alias events
        schema_audit_resp = await client.get(
            f"/api/audit/schemas/{schema_id}",
            params={"limit": 100},
        )
        assert schema_audit_resp.status_code == 200
        schema_events = schema_audit_resp.json()["events"]
        schema_event_types = [e["event_type"] for e in schema_events]

        assert "route_updated" in schema_event_types
        assert "alias_set" in schema_event_types


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    async def test_health_returns_200(self, client: AsyncClient) -> None:
        """The /health endpoint returns 200 with service info."""
        response = await client.get("/health")
        assert response.status_code == 200

        body = response.json()
        assert body["status"] == "healthy"
        assert body["service"] == "model-registry"
        assert isinstance(body["version"], str)
        assert len(body["version"]) > 0
        assert "timestamp" in body
        assert isinstance(body["timestamp"], str)
        assert len(body["timestamp"]) > 0

    async def test_liveness_returns_200(self, client: AsyncClient) -> None:
        """The /health/live endpoint returns 200 with JSON body."""
        response = await client.get("/health/live")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert response.json()["status"] == "alive"

    async def test_root_returns_service_info(self, client: AsyncClient) -> None:
        """The root endpoint returns service metadata with correct values."""
        response = await client.get("/")
        assert response.status_code == 200

        body = response.json()
        assert body["service"] == "model-registry"
        assert isinstance(body["version"], str)
        assert len(body["version"]) > 0
        assert body["docs"] == "/docs"
        assert body["health"] == "/health"
