"""Tests for model CRUD API endpoints (/api/models)."""

from datetime import datetime
from typing import Any

import pytest
from httpx import AsyncClient


class TestRegisterModel:
    """Tests for POST /api/models."""

    async def test_register_model_success(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Registering a new model returns 201 and persists correctly in the DB."""
        response = await client.post("/api/models", json=sample_model_data)
        assert response.status_code == 201

        body = response.json()
        assert body["schema_id"] == sample_model_data["schema_id"]
        assert body["version"] == sample_model_data["version"]
        assert body["status"] == "training"
        assert body["base_model"] == "Qwen3-4B-AWQ-270m"
        assert body["quantization"] == "Q4_K_M"
        assert body["training_samples"] == 500
        assert body["id"] is not None
        assert body["created_at"] is not None
        assert body["updated_at"] is not None
        # created_at and updated_at should be parseable ISO timestamps
        created_at = datetime.fromisoformat(body["created_at"])
        updated_at = datetime.fromisoformat(body["updated_at"])
        assert created_at == updated_at  # Same on creation

        # Verify the model is actually persisted by fetching it back
        model_id = body["id"]
        get_resp = await client.get(f"/api/models/{model_id}")
        assert get_resp.status_code == 200
        persisted = get_resp.json()
        assert persisted["id"] == model_id
        assert persisted["schema_id"] == sample_model_data["schema_id"]
        assert persisted["version"] == sample_model_data["version"]
        assert persisted["training_samples"] == 500

    async def test_register_model_defaults(self, client: AsyncClient) -> None:
        """Registering with minimal fields uses default values."""
        response = await client.post(
            "/api/models",
            json={"schema_id": "wm_parser_abc123", "version": "v1"},
        )
        assert response.status_code == 201

        body = response.json()
        assert body["base_model"] == "unspecified"
        assert body["quantization"] == "unspecified"
        assert body["training_samples"] == 0
        assert body["status"] == "training"

    async def test_register_duplicate_returns_409(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Registering a duplicate schema_id + version returns 409."""
        response1 = await client.post("/api/models", json=sample_model_data)
        assert response1.status_code == 201

        response2 = await client.post("/api/models", json=sample_model_data)
        assert response2.status_code == 409
        assert "already exists" in response2.json()["detail"]

    async def test_register_model_missing_required_field(self, client: AsyncClient) -> None:
        """Missing required fields return 422."""
        response = await client.post("/api/models", json={"version": "v1"})
        assert response.status_code == 422


class TestListModels:
    """Tests for GET /api/models."""

    async def test_list_models_empty(self, client: AsyncClient) -> None:
        """Listing models when none exist returns an empty list."""
        response = await client.get("/api/models")
        assert response.status_code == 200

        body = response.json()
        assert body["models"] == []
        assert body["total"] == 0

    async def test_list_models_returns_registered(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Listing models returns previously registered models."""
        await client.post("/api/models", json=sample_model_data)

        response = await client.get("/api/models")
        assert response.status_code == 200

        body = response.json()
        assert body["total"] == 1
        assert len(body["models"]) == 1
        assert body["models"][0]["schema_id"] == sample_model_data["schema_id"]

    async def test_list_models_filter_by_schema_id(self, client: AsyncClient) -> None:
        """Filtering by schema_id returns only matching models."""
        await client.post(
            "/api/models",
            json={"schema_id": "ks_Alpha", "version": "v1"},
        )
        await client.post(
            "/api/models",
            json={"schema_id": "ks_Beta", "version": "v1"},
        )

        response = await client.get("/api/models", params={"schema_id": "ks_Alpha"})
        assert response.status_code == 200

        body = response.json()
        assert body["total"] == 1
        assert body["models"][0]["schema_id"] == "ks_Alpha"

    async def test_list_models_filter_by_status(self, client: AsyncClient) -> None:
        """Filtering by status returns only matching models."""
        await client.post(
            "/api/models",
            json={"schema_id": "ks_Gamma", "version": "v1", "status": "training"},
        )

        response = await client.get("/api/models", params={"status": "training"})
        assert response.status_code == 200
        assert response.json()["total"] == 1

        response = await client.get("/api/models", params={"status": "deployed"})
        assert response.status_code == 200
        assert response.json()["total"] == 0

    async def test_list_models_pagination(self, client: AsyncClient) -> None:
        """Pagination with limit and offset works correctly."""
        for i in range(5):
            await client.post(
                "/api/models",
                json={"schema_id": f"ks_Pag{i}", "version": "v1"},
            )

        response = await client.get("/api/models", params={"limit": 2, "offset": 0})
        body = response.json()
        assert len(body["models"]) == 2
        assert body["total"] == 5

        response = await client.get("/api/models", params={"limit": 2, "offset": 4})
        body = response.json()
        assert len(body["models"]) == 1


class TestGetModel:
    """Tests for GET /api/models/{model_id}."""

    async def test_get_model_success(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Getting an existing model returns its full details."""
        create_resp = await client.post("/api/models", json=sample_model_data)
        model_id = create_resp.json()["id"]

        response = await client.get(f"/api/models/{model_id}")
        assert response.status_code == 200

        body = response.json()
        assert body["id"] == model_id
        assert body["schema_id"] == sample_model_data["schema_id"]

    async def test_get_model_not_found(self, client: AsyncClient) -> None:
        """Getting a non-existent model returns 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.get(f"/api/models/{fake_id}")
        assert response.status_code == 404


class TestUpdateStatus:
    """Tests for PATCH /api/models/{model_id}/status."""

    async def test_valid_transition_training_to_evaluating(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Transitioning from training to evaluating succeeds and updates timestamp."""
        create_resp = await client.post("/api/models", json=sample_model_data)
        created = create_resp.json()
        model_id = created["id"]
        original_updated_at = created["updated_at"]

        response = await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "evaluating"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "evaluating"
        # updated_at must change on status transition
        assert body["updated_at"] >= original_updated_at
        # created_at must NOT change
        assert body["created_at"] == created["created_at"]

        # Verify DB persistence: GET should reflect new status
        get_resp = await client.get(f"/api/models/{model_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["status"] == "evaluating"

    async def test_valid_full_lifecycle(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Walking through the full lifecycle succeeds at each step."""
        create_resp = await client.post("/api/models", json=sample_model_data)
        model_id = create_resp.json()["id"]

        transitions = [
            "evaluating",
            "evaluated",
            "approved",
            "exporting",
            "deployed",
            "retired",
        ]
        for target_status in transitions:
            response = await client.patch(
                f"/api/models/{model_id}/status",
                json={"status": target_status},
            )
            assert response.status_code == 200, (
                f"Failed transition to {target_status}: {response.json()}"
            )
            assert response.json()["status"] == target_status

    async def test_invalid_transition_returns_400(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """An invalid status transition returns 400."""
        create_resp = await client.post("/api/models", json=sample_model_data)
        model_id = create_resp.json()["id"]

        # training -> deployed is not allowed
        response = await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "deployed"},
        )
        assert response.status_code == 400
        assert "Invalid status transition" in response.json()["detail"]

    async def test_skip_status_returns_400(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Skipping a status step returns 400."""
        create_resp = await client.post("/api/models", json=sample_model_data)
        model_id = create_resp.json()["id"]

        # training -> approved (skipping evaluating and evaluated)
        response = await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "approved"},
        )
        assert response.status_code == 400

    async def test_any_state_to_retired(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Any non-retired state can transition to retired."""
        create_resp = await client.post("/api/models", json=sample_model_data)
        model_id = create_resp.json()["id"]

        # training -> retired (cleanup escape hatch)
        response = await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "retired"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "retired"

    async def test_retired_is_terminal(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Once retired, no further transitions are allowed."""
        create_resp = await client.post("/api/models", json=sample_model_data)
        model_id = create_resp.json()["id"]

        # Move to retired
        await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "retired"},
        )

        # Try to go back to training
        response = await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "training"},
        )
        assert response.status_code == 400

    async def test_update_status_with_eval_fields(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Setting eval_score and eval_metrics when transitioning to evaluated."""
        create_resp = await client.post("/api/models", json=sample_model_data)
        model_id = create_resp.json()["id"]

        # training -> evaluating
        await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "evaluating"},
        )

        # evaluating -> evaluated with metrics
        response = await client.patch(
            f"/api/models/{model_id}/status",
            json={
                "status": "evaluated",
                "eval_score": 0.92,
                "eval_metrics": {"accuracy": 0.95, "f1": 0.89},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["eval_score"] == 0.92
        assert body["eval_metrics"]["accuracy"] == 0.95

    async def test_invalid_status_value_returns_422(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Sending an invalid status value returns 422 validation error."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        status_resp = await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "totally_invalid_status"},
        )
        assert status_resp.status_code == 422

    async def test_update_status_not_found(self, client: AsyncClient) -> None:
        """Updating status for a non-existent model returns 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.patch(
            f"/api/models/{fake_id}/status",
            json={"status": "evaluating"},
        )
        assert response.status_code == 404

    async def test_deployed_sets_deployed_at(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Transitioning to deployed sets the deployed_at timestamp."""
        create_resp = await client.post("/api/models", json=sample_model_data)
        model_id = create_resp.json()["id"]

        for status in ["evaluating", "evaluated", "approved", "exporting", "deployed"]:
            await client.patch(
                f"/api/models/{model_id}/status",
                json={"status": status},
            )

        get_resp = await client.get(f"/api/models/{model_id}")
        assert get_resp.json()["deployed_at"] is not None


class TestQualityEndpoint:
    """Tests for GET /api/models/{schema_id}/quality."""

    async def test_quality_nonexistent_schema_returns_nulls(self, client: AsyncClient) -> None:
        """Querying quality for a schema with no models returns null score."""
        response = await client.get("/api/models/nonexistent_schema/quality")
        assert response.status_code == 200

        body = response.json()
        assert body["schema_id"] == "nonexistent_schema"
        assert body["quality_score"] is None
        assert body["model_id"] is None
        assert body["version"] is None

    async def test_quality_returns_eval_score(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Registering a model and setting eval_score, then verifying quality endpoint."""
        # Register a model
        create_resp = await client.post("/api/models", json=sample_model_data)
        assert create_resp.status_code == 201
        model_id = create_resp.json()["id"]
        schema_id = sample_model_data["schema_id"]

        # Walk it through training -> evaluating -> evaluated with an eval_score
        await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "evaluating"},
        )
        await client.patch(
            f"/api/models/{model_id}/status",
            json={
                "status": "evaluated",
                "eval_score": 0.85,
                "eval_metrics": {"accuracy": 0.90},
            },
        )

        # Now query the quality endpoint
        response = await client.get(f"/api/models/{schema_id}/quality")
        assert response.status_code == 200

        body = response.json()
        assert body["schema_id"] == schema_id
        assert body["quality_score"] == pytest.approx(0.85)
        assert 0.849 <= body["quality_score"] <= 0.851
        assert body["model_id"] == model_id
        assert body["version"] == sample_model_data["version"]

    async def test_quality_returns_latest_model(self, client: AsyncClient) -> None:
        """When multiple models have eval scores, the most recent is returned."""
        schema_id = "ks_MultiQuality"

        # Register first model (older)
        resp1 = await client.post(
            "/api/models",
            json={"schema_id": schema_id, "version": "v1"},
        )
        model_id_1 = resp1.json()["id"]
        await client.patch(
            f"/api/models/{model_id_1}/status",
            json={"status": "evaluating"},
        )
        await client.patch(
            f"/api/models/{model_id_1}/status",
            json={"status": "evaluated", "eval_score": 0.70},
        )

        # Register second model (newer)
        resp2 = await client.post(
            "/api/models",
            json={"schema_id": schema_id, "version": "v2"},
        )
        model_id_2 = resp2.json()["id"]
        await client.patch(
            f"/api/models/{model_id_2}/status",
            json={"status": "evaluating"},
        )
        await client.patch(
            f"/api/models/{model_id_2}/status",
            json={"status": "evaluated", "eval_score": 0.95},
        )

        # Quality endpoint should return the newer model's score
        response = await client.get(f"/api/models/{schema_id}/quality")
        assert response.status_code == 200

        body = response.json()
        assert body["schema_id"] == schema_id
        assert body["quality_score"] == pytest.approx(0.95)
        assert body["model_id"] == model_id_2
        assert body["version"] == "v2"

    async def test_quality_ignores_models_without_eval_score(self, client: AsyncClient) -> None:
        """Models without an eval_score are skipped by the quality endpoint."""
        schema_id = "ks_NoEval"

        # Register a model but do not set an eval_score
        await client.post(
            "/api/models",
            json={"schema_id": schema_id, "version": "v1"},
        )

        response = await client.get(f"/api/models/{schema_id}/quality")
        assert response.status_code == 200

        body = response.json()
        assert body["schema_id"] == schema_id
        assert body["quality_score"] is None
        assert body["model_id"] is None
        assert body["version"] is None


class TestAutoApproveAPI:
    """Tests for auto-approval behavior through the HTTP API."""

    async def test_auto_approve_via_api(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Setting eval_score >= 0.95 via API auto-transitions to approved."""
        create_resp = await client.post("/api/models", json=sample_model_data)
        model_id = create_resp.json()["id"]

        await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "evaluating"},
        )
        response = await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "evaluated", "eval_score": 0.96},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "approved"
        assert body["eval_score"] == pytest.approx(0.96)

    async def test_no_auto_approve_below_threshold_via_api(self, client: AsyncClient) -> None:
        """eval_score below threshold stays evaluated."""
        create_resp = await client.post(
            "/api/models",
            json={"schema_id": "ks_NoAutoAPI", "version": "v1"},
        )
        model_id = create_resp.json()["id"]

        await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "evaluating"},
        )
        response = await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "evaluated", "eval_score": 0.90},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "evaluated"

    async def test_auto_approved_model_can_continue_lifecycle(self, client: AsyncClient) -> None:
        """An auto-approved model can continue through exporting -> deployed."""
        create_resp = await client.post(
            "/api/models",
            json={"schema_id": "ks_AutoLC", "version": "v1"},
        )
        model_id = create_resp.json()["id"]

        await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "evaluating"},
        )
        # This auto-approves
        resp = await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "evaluated", "eval_score": 0.98},
        )
        assert resp.json()["status"] == "approved"

        # Continue lifecycle from approved
        for status in ["exporting", "deployed"]:
            resp = await client.patch(
                f"/api/models/{model_id}/status",
                json={"status": status},
            )
            assert resp.status_code == 200
        assert resp.json()["status"] == "deployed"
        assert resp.json()["deployed_at"] is not None


class TestPeftModelRegistration:
    """Tests for PEFT model format registration and lifecycle."""

    async def test_register_peft_model(self, client: AsyncClient) -> None:
        """Registering a PEFT model stores model_format and adapter_path."""
        response = await client.post(
            "/api/models",
            json={
                "schema_id": "ks_PeftReg",
                "version": "v1",
                "model_format": "peft",
                "adapter_path": "adapters/ks_PeftReg/v1/",
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["model_format"] == "peft"
        assert body["adapter_path"] == "adapters/ks_PeftReg/v1/"

        # Verify persistence
        get_resp = await client.get(f"/api/models/{body['id']}")
        assert get_resp.status_code == 200
        assert get_resp.json()["model_format"] == "peft"
        assert get_resp.json()["adapter_path"] == "adapters/ks_PeftReg/v1/"

    async def test_register_with_tags_persists_tags(self, client: AsyncClient) -> None:
        """Registering a model with inline tags stores them in the DB."""
        response = await client.post(
            "/api/models",
            json={
                "schema_id": "ks_TagsOnReg",
                "version": "v1",
                "tags": {"job_id": "j-123", "dataset": "d-456"},
            },
        )
        assert response.status_code == 201
        model_id = response.json()["id"]

        # Verify tags are actually persisted via the tags endpoint
        tag_resp = await client.get(f"/api/models/{model_id}/tags")
        assert tag_resp.status_code == 200
        tags = tag_resp.json()
        assert tags["job_id"] == "j-123"
        assert tags["dataset"] == "d-456"


class TestStatusTransitionErrorMessages:
    """Tests verifying specific error messages on invalid transitions."""

    async def test_invalid_transition_mentions_allowed_states(self, client: AsyncClient) -> None:
        """Error message for invalid transition lists the allowed target states."""
        resp = await client.post(
            "/api/models",
            json={"schema_id": "ks_ErrMsgCheck", "version": "v1"},
        )
        model_id = resp.json()["id"]

        response = await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "deployed"},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "Invalid status transition" in detail
        assert "training -> deployed" in detail
        # Should mention allowed transitions from training
        assert "evaluating" in detail
        assert "retired" in detail

    async def test_retired_transition_error_message(self, client: AsyncClient) -> None:
        """Error message for retired state says no transitions allowed."""
        resp = await client.post(
            "/api/models",
            json={"schema_id": "ks_RetiredMsg", "version": "v1"},
        )
        model_id = resp.json()["id"]

        await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "retired"},
        )

        response = await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "evaluating"},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "retired -> evaluating" in detail

    async def test_duplicate_registration_error_includes_schema_and_version(
        self, client: AsyncClient
    ) -> None:
        """Duplicate registration error mentions both schema_id and version."""
        data = {"schema_id": "ks_DupMsg", "version": "v7"}
        await client.post("/api/models", json=data)

        response = await client.post("/api/models", json=data)
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "ks_DupMsg" in detail
        assert "v7" in detail

    async def test_missing_schema_id_returns_422_with_field_info(self, client: AsyncClient) -> None:
        """Missing schema_id returns 422 with field-specific error info."""
        response = await client.post("/api/models", json={"version": "v1"})
        assert response.status_code == 422
        body = response.json()
        # FastAPI validation error includes detail array with field-specific info
        assert "detail" in body
        errors = body["detail"]
        field_names = [e.get("loc", [])[-1] for e in errors if "loc" in e]
        assert "schema_id" in field_names

    async def test_missing_version_returns_422_with_field_info(self, client: AsyncClient) -> None:
        """Missing version returns 422 with field-specific error info."""
        response = await client.post("/api/models", json={"schema_id": "ks_MissingVer"})
        assert response.status_code == 422
        body = response.json()
        errors = body["detail"]
        field_names = [e.get("loc", [])[-1] for e in errors if "loc" in e]
        assert "version" in field_names


class TestModelVersionOrdering:
    """Tests verifying model version ordering in list and quality endpoints."""

    async def test_list_models_ordered_by_created_at_desc(self, client: AsyncClient) -> None:
        """Models are listed in reverse chronological order (newest first)."""
        schema_id = "ks_OrderCheck"
        for i in range(3):
            resp = await client.post(
                "/api/models",
                json={"schema_id": schema_id, "version": f"v{i + 1}"},
            )
            assert resp.status_code == 201

        list_resp = await client.get("/api/models", params={"schema_id": schema_id})
        assert list_resp.status_code == 200
        models = list_resp.json()["models"]
        assert len(models) == 3
        # Newest should be first
        assert models[0]["version"] == "v3"
        assert models[1]["version"] == "v2"
        assert models[2]["version"] == "v1"

    async def test_different_schema_same_version_allowed(self, client: AsyncClient) -> None:
        """Same version label under different schema_ids is allowed."""
        resp1 = await client.post(
            "/api/models",
            json={"schema_id": "ks_SchemaA", "version": "v1"},
        )
        assert resp1.status_code == 201

        resp2 = await client.post(
            "/api/models",
            json={"schema_id": "ks_SchemaB", "version": "v1"},
        )
        assert resp2.status_code == 201

        # Both should exist
        assert resp1.json()["id"] != resp2.json()["id"]
