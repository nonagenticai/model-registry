"""Tests for model alias API endpoints."""

from typing import Any

from httpx import AsyncClient


class TestSetAlias:
    """Tests for PUT /api/aliases/{schema_id}/{alias_name}."""

    async def test_set_alias_success(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Setting an alias returns the created alias."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]
        schema_id = sample_model_data["schema_id"]

        alias_resp = await client.put(
            f"/api/aliases/{schema_id}/champion",
            json={"model_id": model_id},
        )
        assert alias_resp.status_code == 200
        body = alias_resp.json()
        assert body["schema_id"] == schema_id
        assert body["alias_name"] == "champion"
        assert body["model_id"] == model_id
        assert "created_at" in body
        assert "updated_at" in body

    async def test_set_alias_updates_existing(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Setting an existing alias updates it to the new model."""
        resp1 = await client.post("/api/models", json=sample_model_data)
        model_id_1 = resp1.json()["id"]
        schema_id = sample_model_data["schema_id"]

        resp2 = await client.post(
            "/api/models",
            json={**sample_model_data, "version": "v2"},
        )
        model_id_2 = resp2.json()["id"]

        first_resp = await client.put(
            f"/api/aliases/{schema_id}/champion",
            json={"model_id": model_id_1},
        )
        assert first_resp.status_code == 200
        assert first_resp.json()["model_id"] == model_id_1

        alias_resp = await client.put(
            f"/api/aliases/{schema_id}/champion",
            json={"model_id": model_id_2},
        )
        assert alias_resp.status_code == 200
        assert alias_resp.json()["model_id"] == model_id_2
        assert alias_resp.json()["updated_at"] is not None

    async def test_set_alias_invalid_name_returns_400(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Invalid alias names are rejected."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        alias_resp = await client.put(
            f"/api/aliases/{sample_model_data['schema_id']}/INVALID_NAME",
            json={"model_id": model_id},
        )
        assert alias_resp.status_code == 400
        assert "Invalid alias name" in alias_resp.json()["detail"]

    async def test_set_alias_nonexistent_model_returns_404(self, client: AsyncClient) -> None:
        """Setting an alias to a nonexistent model returns 404."""
        alias_resp = await client.put(
            "/api/aliases/ks_NoModel/champion",
            json={"model_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert alias_resp.status_code == 404
        assert "not found" in alias_resp.json()["detail"].lower()

    async def test_alias_name_validation_edge_cases(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Test valid and invalid alias name patterns."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]
        schema_id = sample_model_data["schema_id"]

        # Valid: single lowercase letter
        r = await client.put(
            f"/api/aliases/{schema_id}/a",
            json={"model_id": model_id},
        )
        assert r.status_code == 200

        # Valid: hyphenated lowercase
        r = await client.put(
            f"/api/aliases/{schema_id}/my-model",
            json={"model_id": model_id},
        )
        assert r.status_code == 200

        # Valid: max length (63 chars: 1 leading letter + 62 trailing)
        max_name = "a" + "b" * 62
        r = await client.put(
            f"/api/aliases/{schema_id}/{max_name}",
            json={"model_id": model_id},
        )
        assert r.status_code == 200

        # Invalid: starts with a number
        r = await client.put(
            f"/api/aliases/{schema_id}/1bad",
            json={"model_id": model_id},
        )
        assert r.status_code == 400
        assert "Invalid alias name" in r.json()["detail"]

        # Invalid: uppercase letters
        r = await client.put(
            f"/api/aliases/{schema_id}/BadName",
            json={"model_id": model_id},
        )
        assert r.status_code == 400
        assert "Invalid alias name" in r.json()["detail"]

        # Invalid: special characters (underscore)
        r = await client.put(
            f"/api/aliases/{schema_id}/bad_name",
            json={"model_id": model_id},
        )
        assert r.status_code == 400
        assert "Invalid alias name" in r.json()["detail"]

        # Invalid: special characters (dot)
        r = await client.put(
            f"/api/aliases/{schema_id}/bad.name",
            json={"model_id": model_id},
        )
        assert r.status_code == 400
        assert "Invalid alias name" in r.json()["detail"]


class TestListAliases:
    """Tests for GET /api/aliases/{schema_id}."""

    async def test_list_aliases_empty(self, client: AsyncClient) -> None:
        """Listing aliases for a schema with none returns empty."""
        resp = await client.get("/api/aliases/ks_NoAliases")
        assert resp.status_code == 200
        assert resp.json()["aliases"] == []

    async def test_list_aliases_returns_all(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Listing aliases returns all set aliases."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]
        schema_id = sample_model_data["schema_id"]

        await client.put(
            f"/api/aliases/{schema_id}/champion",
            json={"model_id": model_id},
        )
        await client.put(
            f"/api/aliases/{schema_id}/rollback",
            json={"model_id": model_id},
        )

        list_resp = await client.get(f"/api/aliases/{schema_id}")
        assert list_resp.status_code == 200
        aliases = list_resp.json()["aliases"]
        assert len(aliases) == 2
        names = {a["alias_name"] for a in aliases}
        assert names == {"champion", "rollback"}
        for alias in aliases:
            assert alias["model_id"] == model_id


class TestGetAlias:
    """Tests for GET /api/aliases/{schema_id}/{alias_name}."""

    async def test_get_alias_success(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Getting an existing alias returns its details."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]
        schema_id = sample_model_data["schema_id"]

        await client.put(
            f"/api/aliases/{schema_id}/candidate",
            json={"model_id": model_id},
        )

        get_resp = await client.get(f"/api/aliases/{schema_id}/candidate")
        assert get_resp.status_code == 200
        assert get_resp.json()["model_id"] == model_id

    async def test_get_alias_not_found(self, client: AsyncClient) -> None:
        """Getting a nonexistent alias returns 404."""
        resp = await client.get("/api/aliases/ks_None/champion")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


class TestDeleteAlias:
    """Tests for DELETE /api/aliases/{schema_id}/{alias_name}."""

    async def test_delete_alias_success(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Deleting an existing alias returns 204."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]
        schema_id = sample_model_data["schema_id"]

        await client.put(
            f"/api/aliases/{schema_id}/shadow",
            json={"model_id": model_id},
        )

        del_resp = await client.delete(f"/api/aliases/{schema_id}/shadow")
        assert del_resp.status_code == 204

        # Verify it's gone
        get_resp = await client.get(f"/api/aliases/{schema_id}/shadow")
        assert get_resp.status_code == 404

    async def test_delete_alias_not_found(self, client: AsyncClient) -> None:
        """Deleting a nonexistent alias returns 404."""
        resp = await client.delete("/api/aliases/ks_None/champion")
        assert resp.status_code == 404


class TestResolveAlias:
    """Tests for GET /api/aliases/{schema_id}/{alias_name}/resolve."""

    async def test_resolve_alias_returns_full_model(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Resolving an alias returns the full ModelVersion."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]
        schema_id = sample_model_data["schema_id"]

        await client.put(
            f"/api/aliases/{schema_id}/champion",
            json={"model_id": model_id},
        )

        resolve_resp = await client.get(f"/api/aliases/{schema_id}/champion/resolve")
        assert resolve_resp.status_code == 200
        body = resolve_resp.json()
        assert body["id"] == model_id
        assert body["schema_id"] == schema_id
        assert body["version"] == sample_model_data["version"]
        assert body["status"] == sample_model_data["status"]
        assert body["base_model"] == sample_model_data["base_model"]

    async def test_resolve_alias_not_found(self, client: AsyncClient) -> None:
        """Resolving a nonexistent alias returns 404."""
        resp = await client.get("/api/aliases/ks_None/champion/resolve")
        assert resp.status_code == 404


class TestAliasAuditIntegration:
    """Tests that alias operations emit audit events."""

    async def test_set_alias_emits_audit(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Setting an alias creates an audit event."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]
        schema_id = sample_model_data["schema_id"]

        await client.put(
            f"/api/aliases/{schema_id}/champion",
            json={"model_id": model_id},
        )

        audit_resp = await client.get(
            "/api/audit",
            params={"event_type": "alias_set"},
        )
        assert audit_resp.status_code == 200
        body = audit_resp.json()
        assert body["total"] >= 1
        event = body["events"][0]
        assert event["entity_id"] == f"{schema_id}:champion"
        assert "alias" in event["detail"]
        assert "to_model_id" in event["detail"]

    async def test_set_alias_emits_audit_with_from_model_id_on_update(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Updating an alias records both from_model_id and to_model_id in the audit event."""
        resp1 = await client.post("/api/models", json=sample_model_data)
        model_id_1 = resp1.json()["id"]
        schema_id = sample_model_data["schema_id"]

        resp2 = await client.post(
            "/api/models",
            json={**sample_model_data, "version": "v2"},
        )
        model_id_2 = resp2.json()["id"]

        # First set
        await client.put(
            f"/api/aliases/{schema_id}/champion",
            json={"model_id": model_id_1},
        )
        # Update to second model
        await client.put(
            f"/api/aliases/{schema_id}/champion",
            json={"model_id": model_id_2},
        )

        audit_resp = await client.get(
            "/api/audit",
            params={"event_type": "alias_set"},
        )
        assert audit_resp.status_code == 200
        events = audit_resp.json()["events"]
        # The most recent event should be the update (events are newest-first)
        update_event = events[0]
        assert update_event["detail"]["from_model_id"] == model_id_1
        assert update_event["detail"]["to_model_id"] == model_id_2

    async def test_delete_alias_emits_audit(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Deleting an alias creates an alias_deleted audit event."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]
        schema_id = sample_model_data["schema_id"]

        await client.put(
            f"/api/aliases/{schema_id}/candidate",
            json={"model_id": model_id},
        )

        del_resp = await client.delete(f"/api/aliases/{schema_id}/candidate")
        assert del_resp.status_code == 204

        audit_resp = await client.get(
            "/api/audit",
            params={"event_type": "alias_deleted"},
        )
        assert audit_resp.status_code == 200
        body = audit_resp.json()
        assert body["total"] >= 1
        event = body["events"][0]
        assert event["entity_id"] == f"{schema_id}:candidate"
