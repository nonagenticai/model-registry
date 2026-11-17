"""Tests for the global-adapter API endpoints (/api/global-adapter).

Phase 1 substrate of the single-LoRA rollout — substrate only, no
behaviour change for downstream consumers.
"""

from typing import Any
from uuid import uuid4

import pytest
from httpx import AsyncClient


@pytest.fixture
async def registered_model(client: AsyncClient) -> dict[str, Any]:
    """Register a fresh model_version and return its serialised body."""
    resp = await client.post(
        "/api/models",
        json={"schema_id": "ks_GlobalAdapterTest", "version": "v1"},
    )
    assert resp.status_code == 201
    body: dict[str, Any] = resp.json()
    return body


class TestGetGlobalAdapter:
    async def test_get_returns_204_when_unset(self, client: AsyncClient) -> None:
        """When no global adapter is set, GET returns 204 No Content."""
        resp = await client.get("/api/global-adapter")
        assert resp.status_code == 204
        # Body for 204 should be empty
        assert resp.content == b""


class TestUpsertGlobalAdapter:
    async def test_post_then_get_round_trip(
        self, client: AsyncClient, registered_model: dict[str, Any]
    ) -> None:
        """POST sets the global adapter; subsequent GET returns it."""
        payload = {
            "model_version_id": registered_model["id"],
            "minio_uri": "s3://models/global/v1/",
            "adapter_hash": "sha256:abc123",
            "notes": "first global",
        }
        post_resp = await client.post("/api/global-adapter", json=payload)
        assert post_resp.status_code == 200
        body = post_resp.json()
        assert body["id"] == 1
        assert body["model_version_id"] == registered_model["id"]
        assert body["minio_uri"] == payload["minio_uri"]
        assert body["adapter_hash"] == payload["adapter_hash"]
        assert body["notes"] == payload["notes"]
        assert body["created_at"]
        assert body["updated_at"]

        get_resp = await client.get("/api/global-adapter")
        assert get_resp.status_code == 200
        get_body = get_resp.json()
        assert get_body["model_version_id"] == registered_model["id"]
        assert get_body["adapter_hash"] == "sha256:abc123"

    async def test_post_idempotent_upsert(
        self, client: AsyncClient, registered_model: dict[str, Any]
    ) -> None:
        """Calling POST twice with the same body returns the same singleton row."""
        payload = {
            "model_version_id": registered_model["id"],
            "minio_uri": "s3://models/global/v1/",
            "adapter_hash": "sha256:def456",
        }
        first = (await client.post("/api/global-adapter", json=payload)).json()
        second = (await client.post("/api/global-adapter", json=payload)).json()
        assert first["id"] == 1 == second["id"]
        assert first["model_version_id"] == second["model_version_id"]
        assert first["adapter_hash"] == second["adapter_hash"]

    async def test_post_updates_existing_row(
        self, client: AsyncClient, registered_model: dict[str, Any]
    ) -> None:
        """Subsequent POST with new values updates the same singleton row."""
        await client.post(
            "/api/global-adapter",
            json={
                "model_version_id": registered_model["id"],
                "minio_uri": "s3://models/global/v1/",
                "adapter_hash": "sha256:v1",
            },
        )
        # Register a second model_version for the update target
        resp2 = await client.post(
            "/api/models",
            json={"schema_id": "ks_GlobalAdapterTest", "version": "v2"},
        )
        assert resp2.status_code == 201
        v2_id = resp2.json()["id"]

        updated = (
            await client.post(
                "/api/global-adapter",
                json={
                    "model_version_id": v2_id,
                    "minio_uri": "s3://models/global/v2/",
                    "adapter_hash": "sha256:v2",
                    "notes": "promoted",
                },
            )
        ).json()
        assert updated["id"] == 1
        assert updated["model_version_id"] == v2_id
        assert updated["adapter_hash"] == "sha256:v2"
        assert updated["minio_uri"] == "s3://models/global/v2/"
        assert updated["notes"] == "promoted"

        # Verify only one row exists by GET
        get_body = (await client.get("/api/global-adapter")).json()
        assert get_body["model_version_id"] == v2_id

    async def test_post_rejects_unknown_model_version_id(self, client: AsyncClient) -> None:
        """POST with a model_version_id that does not exist is rejected."""
        bogus_id = str(uuid4())
        resp = await client.post(
            "/api/global-adapter",
            json={
                "model_version_id": bogus_id,
                "minio_uri": "s3://models/global/v1/",
                "adapter_hash": "sha256:nope",
            },
        )
        assert resp.status_code == 400
        assert bogus_id in resp.json()["detail"]

    async def test_post_emits_audit_event(
        self, client: AsyncClient, registered_model: dict[str, Any]
    ) -> None:
        """Successful upsert emits a global_adapter_set audit event."""
        await client.post(
            "/api/global-adapter",
            json={
                "model_version_id": registered_model["id"],
                "minio_uri": "s3://models/global/v1/",
                "adapter_hash": "sha256:audit-test",
            },
        )

        audit_resp = await client.get("/api/audit?event_type=global_adapter_set")
        assert audit_resp.status_code == 200
        events = audit_resp.json()["events"]
        assert len(events) >= 1
        latest = events[0]
        assert latest["event_type"] == "global_adapter_set"
        assert latest["entity_type"] == "global_adapter"
        assert latest["entity_id"] == "1"
        assert latest["detail"]["adapter_hash"] == "sha256:audit-test"
