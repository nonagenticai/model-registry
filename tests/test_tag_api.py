"""Tests for model tag API endpoints."""

from typing import Any

from httpx import AsyncClient


class TestSetTags:
    """Tests for PUT /api/models/{model_id}/tags."""

    async def test_set_tags_success(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Setting tags returns the full tag set."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        tag_resp = await client.put(
            f"/api/models/{model_id}/tags",
            json={"tags": {"env": "staging", "team": "ml"}},
        )
        assert tag_resp.status_code == 200
        body = tag_resp.json()
        assert body["env"] == "staging"
        assert body["team"] == "ml"

    async def test_set_tags_updates_existing(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Setting a tag that already exists updates its value."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        await client.put(
            f"/api/models/{model_id}/tags",
            json={"tags": {"env": "staging"}},
        )
        tag_resp = await client.put(
            f"/api/models/{model_id}/tags",
            json={"tags": {"env": "production"}},
        )
        assert tag_resp.status_code == 200
        assert tag_resp.json()["env"] == "production"

    async def test_set_tags_model_not_found(self, client: AsyncClient) -> None:
        """Setting tags on a nonexistent model returns 404."""
        resp = await client.put(
            "/api/models/00000000-0000-0000-0000-000000000000/tags",
            json={"tags": {"env": "test"}},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    async def test_set_tags_multiple_calls_merge(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Multiple set-tag calls merge keys, updating existing and adding new."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        await client.put(
            f"/api/models/{model_id}/tags",
            json={"tags": {"a": "1", "b": "2"}},
        )
        await client.put(
            f"/api/models/{model_id}/tags",
            json={"tags": {"b": "3", "c": "4"}},
        )

        tag_resp = await client.get(f"/api/models/{model_id}/tags")
        assert tag_resp.status_code == 200
        tags = tag_resp.json()
        assert tags["a"] == "1"
        assert tags["b"] == "3"
        assert tags["c"] == "4"

    async def test_set_tags_empty_dict(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Setting an empty tags dict returns current tags without modification."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        tag_resp = await client.put(
            f"/api/models/{model_id}/tags",
            json={"tags": {}},
        )
        assert tag_resp.status_code == 200
        assert isinstance(tag_resp.json(), dict)

        # Verify tags are actually empty
        get_resp = await client.get(f"/api/models/{model_id}/tags")
        assert get_resp.status_code == 200
        assert get_resp.json() == {}


class TestGetTags:
    """Tests for GET /api/models/{model_id}/tags."""

    async def test_get_tags_empty(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Getting tags for a model with no tags returns empty dict."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        tag_resp = await client.get(f"/api/models/{model_id}/tags")
        assert tag_resp.status_code == 200
        assert isinstance(tag_resp.json(), dict)
        assert tag_resp.json() == {}

    async def test_get_tags_returns_set_tags(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Getting tags returns previously set tags."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        await client.put(
            f"/api/models/{model_id}/tags",
            json={"tags": {"foo": "bar"}},
        )

        tag_resp = await client.get(f"/api/models/{model_id}/tags")
        assert tag_resp.status_code == 200
        assert tag_resp.json()["foo"] == "bar"


class TestDeleteTag:
    """Tests for DELETE /api/models/{model_id}/tags/{key}."""

    async def test_delete_tag_success(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Deleting an existing tag returns 204."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        await client.put(
            f"/api/models/{model_id}/tags",
            json={"tags": {"env": "staging"}},
        )

        del_resp = await client.delete(f"/api/models/{model_id}/tags/env")
        assert del_resp.status_code == 204

        # Verify it's gone
        tag_resp = await client.get(f"/api/models/{model_id}/tags")
        assert "env" not in tag_resp.json()

    async def test_delete_tag_not_found(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Deleting a nonexistent tag returns 404."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        del_resp = await client.delete(f"/api/models/{model_id}/tags/nonexistent")
        assert del_resp.status_code == 404
        detail = del_resp.json()["detail"].lower()
        assert "not found" in detail or "nonexistent" in detail

    async def test_delete_last_tag_leaves_empty(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Deleting the only tag leaves the model with an empty tag dict."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        await client.put(
            f"/api/models/{model_id}/tags",
            json={"tags": {"only_key": "only_value"}},
        )

        del_resp = await client.delete(f"/api/models/{model_id}/tags/only_key")
        assert del_resp.status_code == 204

        tag_resp = await client.get(f"/api/models/{model_id}/tags")
        assert tag_resp.status_code == 200
        assert tag_resp.json() == {}


class TestSearchByTag:
    """Tests for GET /api/tags/search."""

    async def test_search_by_tag_key(self, client: AsyncClient) -> None:
        """Searching by tag key returns matching models."""
        resp = await client.post(
            "/api/models",
            json={"schema_id": "ks_TagSearch", "version": "v1"},
        )
        model_id = resp.json()["id"]

        await client.put(
            f"/api/models/{model_id}/tags",
            json={"tags": {"team": "alpha"}},
        )

        search_resp = await client.get("/api/tags/search", params={"key": "team"})
        assert search_resp.status_code == 200
        models = search_resp.json()
        assert len(models) >= 1
        assert any(m["id"] == model_id for m in models)

        # Verify returned models contain correct values
        matched = next(m for m in models if m["id"] == model_id)
        assert matched["schema_id"] == "ks_TagSearch"
        assert matched["version"] == "v1"
        assert "status" in matched

    async def test_search_by_tag_key_and_value(self, client: AsyncClient) -> None:
        """Searching by key+value returns only exact matches."""
        resp1 = await client.post(
            "/api/models",
            json={"schema_id": "ks_TagSearchKV1", "version": "v1"},
        )
        resp2 = await client.post(
            "/api/models",
            json={"schema_id": "ks_TagSearchKV2", "version": "v1"},
        )
        mid1 = resp1.json()["id"]
        mid2 = resp2.json()["id"]

        await client.put(f"/api/models/{mid1}/tags", json={"tags": {"env": "prod"}})
        await client.put(f"/api/models/{mid2}/tags", json={"tags": {"env": "staging"}})

        search_resp = await client.get("/api/tags/search", params={"key": "env", "value": "prod"})
        assert search_resp.status_code == 200
        models = search_resp.json()
        assert len(models) == 1
        assert models[0]["id"] == mid1
        assert models[0]["schema_id"] == "ks_TagSearchKV1"

    async def test_search_no_matches(self, client: AsyncClient) -> None:
        """Searching for a non-existent tag returns empty list."""
        resp = await client.get("/api/tags/search", params={"key": "nonexistent_key_xyz"})
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_search_returns_empty_for_existing_key_wrong_value(
        self, client: AsyncClient
    ) -> None:
        """Searching with correct key but wrong value returns empty list."""
        resp = await client.post(
            "/api/models",
            json={"schema_id": "ks_TagSearchWrongVal", "version": "v1"},
        )
        model_id = resp.json()["id"]

        await client.put(
            f"/api/models/{model_id}/tags",
            json={"tags": {"team": "alpha"}},
        )

        search_resp = await client.get("/api/tags/search", params={"key": "team", "value": "beta"})
        assert search_resp.status_code == 200
        # The model with team=alpha should NOT appear when searching for team=beta
        models = search_resp.json()
        assert all(m["id"] != model_id for m in models)


class TestRegisterWithTags:
    """Tests for registering models with inline tags."""

    async def test_register_with_tags(self, client: AsyncClient) -> None:
        """Registering a model with tags persists them."""
        resp = await client.post(
            "/api/models",
            json={
                "schema_id": "ks_InlineTags",
                "version": "v1",
                "tags": {
                    "training_job_id": "job-123",
                    "training_batch_id": "batch-456",
                },
            },
        )
        assert resp.status_code == 201
        model_id = resp.json()["id"]

        tag_resp = await client.get(f"/api/models/{model_id}/tags")
        assert tag_resp.status_code == 200
        tags = tag_resp.json()
        assert tags["training_job_id"] == "job-123"
        assert tags["training_batch_id"] == "batch-456"

        # Verify the model itself was created correctly
        model_resp = await client.get(f"/api/models/{model_id}")
        assert model_resp.status_code == 200
        model = model_resp.json()
        assert model["schema_id"] == "ks_InlineTags"
        assert model["version"] == "v1"
