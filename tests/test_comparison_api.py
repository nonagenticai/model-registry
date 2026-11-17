"""Tests for model comparison API endpoint."""

from httpx import AsyncClient


class TestCompareVersions:
    """Tests for GET /api/models/{schema_id}/compare."""

    async def test_compare_no_versions(self, client: AsyncClient) -> None:
        """Comparing versions for a schema with no models returns empty."""
        resp = await client.get("/api/models/ks_NoneSchema/compare")
        assert resp.status_code == 200
        body = resp.json()
        assert body["schema_id"] == "ks_NoneSchema"
        assert body["versions"] == []

    async def test_compare_all_versions(self, client: AsyncClient) -> None:
        """Comparing without specifying versions returns latest models."""
        schema_id = "ks_CompareAll"
        for i in range(3):
            await client.post(
                "/api/models",
                json={"schema_id": schema_id, "version": f"v{i + 1}"},
            )

        resp = await client.get(f"/api/models/{schema_id}/compare")
        assert resp.status_code == 200
        body = resp.json()
        assert body["schema_id"] == schema_id
        assert len(body["versions"]) == 3
        for version in body["versions"]:
            assert version["schema_id"] == schema_id
            assert "status" in version
            assert "eval_score" in version
            assert "aliases" in version
            assert "tags" in version

    async def test_compare_specific_versions(self, client: AsyncClient) -> None:
        """Comparing specific versions returns only those."""
        schema_id = "ks_CompareSpec"
        for i in range(4):
            await client.post(
                "/api/models",
                json={"schema_id": schema_id, "version": f"v{i + 1}"},
            )

        resp = await client.get(
            f"/api/models/{schema_id}/compare",
            params={"versions": "v1,v3"},
        )
        assert resp.status_code == 200
        body = resp.json()
        versions_labels = {v["version"] for v in body["versions"]}
        assert versions_labels == {"v1", "v3"}
        for version in body["versions"]:
            assert "aliases" in version
            assert "tags" in version

    async def test_compare_includes_eval_metrics(self, client: AsyncClient) -> None:
        """Comparison includes eval_score and eval_metrics."""
        schema_id = "ks_CompareMetrics"
        resp = await client.post(
            "/api/models",
            json={"schema_id": schema_id, "version": "v1"},
        )
        model_id = resp.json()["id"]

        await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "evaluating"},
        )
        await client.patch(
            f"/api/models/{model_id}/status",
            json={
                "status": "evaluated",
                "eval_score": 0.88,
                "eval_metrics": {"accuracy": 0.92, "f1": 0.85},
            },
        )

        resp = await client.get(f"/api/models/{schema_id}/compare")
        assert resp.status_code == 200
        version = resp.json()["versions"][0]
        assert version["eval_score"] == 0.88
        assert version["eval_metrics"]["accuracy"] == 0.92

    async def test_compare_includes_aliases_and_tags(self, client: AsyncClient) -> None:
        """Comparison includes aliases and tags correctly mapped per version."""
        schema_id = "ks_CompareAliasTags"

        # Create v1
        resp_v1 = await client.post(
            "/api/models",
            json={"schema_id": schema_id, "version": "v1"},
        )
        model_id_v1 = resp_v1.json()["id"]

        # Create v2
        resp_v2 = await client.post(
            "/api/models",
            json={"schema_id": schema_id, "version": "v2"},
        )
        model_id_v2 = resp_v2.json()["id"]

        # Set alias "champion" and tag env=prod on v1
        await client.put(
            f"/api/aliases/{schema_id}/champion",
            json={"model_id": model_id_v1},
        )
        await client.put(
            f"/api/models/{model_id_v1}/tags",
            json={"tags": {"env": "prod"}},
        )

        # Set alias "candidate" and tag env=staging on v2
        await client.put(
            f"/api/aliases/{schema_id}/candidate",
            json={"model_id": model_id_v2},
        )
        await client.put(
            f"/api/models/{model_id_v2}/tags",
            json={"tags": {"env": "staging"}},
        )

        resp = await client.get(f"/api/models/{schema_id}/compare")
        assert resp.status_code == 200
        versions = resp.json()["versions"]
        assert len(versions) == 2

        # Build a lookup by version label
        by_version = {v["version"]: v for v in versions}

        # v1 should have alias "champion" and tag env=prod
        v1 = by_version["v1"]
        assert "champion" in v1["aliases"]
        assert "candidate" not in v1["aliases"]
        assert v1["tags"]["env"] == "prod"

        # v2 should have alias "candidate" and tag env=staging
        v2 = by_version["v2"]
        assert "candidate" in v2["aliases"]
        assert "champion" not in v2["aliases"]
        assert v2["tags"]["env"] == "staging"

    async def test_compare_includes_deploy_status(self, client: AsyncClient) -> None:
        """Comparison includes deploy_status for each version."""
        schema_id = "ks_CompareDeployStatus"
        resp = await client.post(
            "/api/models",
            json={"schema_id": schema_id, "version": "v1"},
        )
        model_id = resp.json()["id"]

        # Update deploy-status to pending
        await client.patch(
            f"/api/models/{model_id}/deploy-status",
            json={"deploy_status": "pending"},
        )

        resp = await client.get(f"/api/models/{schema_id}/compare")
        assert resp.status_code == 200
        versions = resp.json()["versions"]
        assert len(versions) == 1
        assert "deploy_status" in versions[0]
        assert versions[0]["deploy_status"] == "pending"

    async def test_compare_empty_for_nonexistent_versions(self, client: AsyncClient) -> None:
        """Comparing specific versions that don't exist returns empty list."""
        schema_id = "ks_CompareNonexistVer"
        # Register one real version so the schema exists
        await client.post(
            "/api/models",
            json={"schema_id": schema_id, "version": "v1"},
        )

        resp = await client.get(
            f"/api/models/{schema_id}/compare",
            params={"versions": "v99,v100"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["schema_id"] == schema_id
        assert body["versions"] == []

    async def test_compare_default_limit(self, client: AsyncClient) -> None:
        """Comparing without versions returns at most 10 (default limit)."""
        schema_id = "ks_CompareLimit"
        for i in range(12):
            await client.post(
                "/api/models",
                json={"schema_id": schema_id, "version": f"v{i + 1}"},
            )

        resp = await client.get(f"/api/models/{schema_id}/compare")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["versions"]) == 10
