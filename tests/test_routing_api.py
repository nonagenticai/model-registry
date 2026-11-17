"""Tests for routing table API endpoints (/api/routing)."""

from uuid import uuid4

from httpx import AsyncClient


class TestGetRoutingTable:
    """Tests for GET /api/routing."""

    async def test_empty_routing_table(self, client: AsyncClient) -> None:
        """Getting the routing table when empty returns an empty list."""
        response = await client.get("/api/routing")
        assert response.status_code == 200
        assert response.json() == []

    async def test_routing_table_returns_created_rules(self, client: AsyncClient) -> None:
        """Created routing rules appear in the full routing table."""
        # Create two routing rules
        await client.put(
            "/api/routing/ks_Alpha",
            json={
                "fallback_chain": ["pgx-local", "deepseek"],
                "endpoint": "fg-ks-alpha.llm-inference.svc.example.com:8080",
            },
        )
        await client.put(
            "/api/routing/ks_Beta",
            json={
                "fallback_chain": ["deepseek", "xai"],
                "endpoint": "fg-ks-beta.llm-inference.svc.example.com:8080",
            },
        )

        response = await client.get("/api/routing")
        assert response.status_code == 200

        rules = response.json()
        assert len(rules) == 2
        schema_ids = {r["schema_id"] for r in rules}
        assert schema_ids == {"ks_Alpha", "ks_Beta"}


class TestGetRoute:
    """Tests for GET /api/routing/{schema_id}."""

    async def test_get_route_success(self, client: AsyncClient) -> None:
        """Getting an existing route returns its details."""
        model_id = str(uuid4())
        await client.put(
            "/api/routing/ks_EntitySpotterAgent",
            json={
                "active_model_id": model_id,
                "fallback_chain": ["pgx-local", "deepseek", "xai"],
                "endpoint": "fg-ks-entityspotteragent.llm-inference.svc.example.com:8080",
            },
        )

        response = await client.get("/api/routing/ks_EntitySpotterAgent")
        assert response.status_code == 200

        body = response.json()
        assert body["schema_id"] == "ks_EntitySpotterAgent"
        assert body["active_model_id"] == model_id
        assert body["fallback_chain"] == ["pgx-local", "deepseek", "xai"]
        assert isinstance(body["fallback_chain"], list)
        assert body["endpoint"] == "fg-ks-entityspotteragent.llm-inference.svc.example.com:8080"

    async def test_get_route_not_found(self, client: AsyncClient) -> None:
        """Getting a non-existent route returns 404."""
        response = await client.get("/api/routing/nonexistent_schema")
        assert response.status_code == 404
        assert "nonexistent_schema" in response.json()["detail"]

    async def test_get_route_cached(self, client: AsyncClient) -> None:
        """Subsequent gets for the same route should be served from cache."""
        await client.put(
            "/api/routing/ks_Cached",
            json={
                "fallback_chain": ["pgx-local", "deepseek"],
                "endpoint": "fg-ks-cached.llm-inference.svc.example.com:8080",
            },
        )

        # First request (cache miss, hits DB)
        resp1 = await client.get("/api/routing/ks_Cached")
        assert resp1.status_code == 200

        # Second request (should be cached)
        resp2 = await client.get("/api/routing/ks_Cached")
        assert resp2.status_code == 200
        assert resp1.json() == resp2.json()


class TestUpdateRoute:
    """Tests for PUT /api/routing/{schema_id}."""

    async def test_create_new_route(self, client: AsyncClient) -> None:
        """PUT to a new schema_id creates a routing rule."""
        model_id = str(uuid4())
        response = await client.put(
            "/api/routing/ks_NewAgent",
            json={
                "active_model_id": model_id,
                "fallback_chain": ["pgx-local", "deepseek"],
                "endpoint": "fg-ks-newagent.llm-inference.svc.example.com:8080",
            },
        )
        assert response.status_code == 200

        body = response.json()
        assert body["schema_id"] == "ks_NewAgent"
        assert body["active_model_id"] == model_id

    async def test_update_existing_route(self, client: AsyncClient) -> None:
        """PUT to an existing schema_id updates the routing rule."""
        # Create initial rule
        model_id1 = str(uuid4())
        await client.put(
            "/api/routing/ks_Update",
            json={
                "active_model_id": model_id1,
                "endpoint": "fg-ks-update-v1.llm-inference.svc.example.com:8080",
            },
        )

        # Update with new model
        model_id2 = str(uuid4())
        response = await client.put(
            "/api/routing/ks_Update",
            json={
                "active_model_id": model_id2,
                "endpoint": "fg-ks-update-v2.llm-inference.svc.example.com:8080",
            },
        )
        assert response.status_code == 200

        body = response.json()
        assert body["active_model_id"] == model_id2
        assert "v2" in body["endpoint"]

    async def test_create_route_with_defaults(self, client: AsyncClient) -> None:
        """Creating a route with minimal fields uses sensible defaults."""
        response = await client.put(
            "/api/routing/ks_Minimal",
            json={},
        )
        assert response.status_code == 200

        body = response.json()
        assert body["schema_id"] == "ks_Minimal"
        assert body["fallback_chain"] == ["pgx-local", "deepseek"]
        assert body["active_model_id"] is None

    async def test_update_route_with_json_schema(self, client: AsyncClient) -> None:
        """Setting a JSON schema on a routing rule persists correctly."""
        output_schema = {
            "type": "object",
            "properties": {
                "entities": {"type": "array", "items": {"type": "string"}},
            },
        }
        response = await client.put(
            "/api/routing/ks_WithSchema",
            json={
                "json_schema": output_schema,
                "gbnf_grammar": 'root ::= "{" entities "}"',
            },
        )
        assert response.status_code == 200

        body = response.json()
        assert body["json_schema"] == output_schema
        assert body["gbnf_grammar"] == 'root ::= "{" entities "}"'

    async def test_create_route_with_all_fields(self, client: AsyncClient) -> None:
        """Creating a route with all fields returns them all correctly."""
        model_id = str(uuid4())
        output_schema = {
            "type": "object",
            "properties": {
                "entities": {"type": "array", "items": {"type": "string"}},
            },
        }
        gbnf = 'root ::= "{" entities "}"'

        response = await client.put(
            "/api/routing/ks_FullFieldAgent",
            json={
                "active_model_id": model_id,
                "fallback_chain": ["pgx-local", "xai"],
                "endpoint": "fg-ks-fullfieldagent.llm-inference.svc.example.com:8080",
                "json_schema": output_schema,
                "gbnf_grammar": gbnf,
            },
        )
        assert response.status_code == 200

        body = response.json()
        assert body["schema_id"] == "ks_FullFieldAgent"
        assert body["active_model_id"] == model_id
        assert body["fallback_chain"] == ["pgx-local", "xai"]
        assert isinstance(body["fallback_chain"], list)
        assert body["endpoint"] == "fg-ks-fullfieldagent.llm-inference.svc.example.com:8080"
        assert body["json_schema"] == output_schema
        assert body["gbnf_grammar"] == gbnf
        assert body["created_at"] is not None
        assert body["updated_at"] is not None

    async def test_update_route_invalidates_cache(self, client: AsyncClient) -> None:
        """Updating a route returns fresh data on the next GET."""
        await client.put(
            "/api/routing/ks_CacheInvalidate",
            json={"endpoint": "endpoint-v1:8080"},
        )

        # Prime the cache
        resp1 = await client.get("/api/routing/ks_CacheInvalidate")
        assert "v1" in resp1.json()["endpoint"]

        # Update
        await client.put(
            "/api/routing/ks_CacheInvalidate",
            json={"endpoint": "endpoint-v2:8080"},
        )

        # Should get new data
        resp2 = await client.get("/api/routing/ks_CacheInvalidate")
        assert "v2" in resp2.json()["endpoint"]

    async def test_create_route_with_system_prompt(self, client: AsyncClient) -> None:
        """Creating a route with system_prompt persists it correctly."""
        prompt = "You are a function-calling assistant. Extract entities from the input."
        response = await client.put(
            "/api/routing/ks_SysPrompt",
            json={
                "endpoint": "ep:8080",
                "system_prompt": prompt,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["system_prompt"] == prompt

        # Verify persistence via GET
        get_resp = await client.get("/api/routing/ks_SysPrompt")
        assert get_resp.status_code == 200
        assert get_resp.json()["system_prompt"] == prompt

    async def test_create_route_with_adapter_path(self, client: AsyncClient) -> None:
        """Creating a route with adapter_path persists it correctly."""
        adapter = "adapters/ks_AdapterRoute/v1/"
        response = await client.put(
            "/api/routing/ks_AdapterRoute",
            json={
                "endpoint": "http://vllm:8000",
                "adapter_path": adapter,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["adapter_path"] == adapter

        # Verify persistence via GET
        get_resp = await client.get("/api/routing/ks_AdapterRoute")
        assert get_resp.status_code == 200
        assert get_resp.json()["adapter_path"] == adapter

    async def test_update_route_partial_does_not_clear_other_fields(
        self, client: AsyncClient
    ) -> None:
        """Updating only endpoint preserves existing fallback_chain and schema fields."""
        # Create with all fields
        await client.put(
            "/api/routing/ks_PartialUpdate",
            json={
                "fallback_chain": ["pgx-local", "xai"],
                "endpoint": "ep-v1:8080",
                "system_prompt": "Be precise.",
            },
        )

        # Update only the endpoint
        resp = await client.put(
            "/api/routing/ks_PartialUpdate",
            json={"endpoint": "ep-v2:9090"},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Endpoint should be updated
        assert body["endpoint"] == "ep-v2:9090"
        # Other fields should be preserved
        assert body["fallback_chain"] == ["pgx-local", "xai"]
        assert body["system_prompt"] == "Be precise."

    async def test_routing_table_sorted_by_schema_id(self, client: AsyncClient) -> None:
        """Full routing table is returned sorted by schema_id."""
        for name in ["ks_Zebra", "ks_Alpha", "ks_Middle"]:
            await client.put(
                f"/api/routing/{name}",
                json={"endpoint": f"ep-{name}:8080"},
            )

        response = await client.get("/api/routing")
        assert response.status_code == 200
        rules = response.json()
        schema_ids = [r["schema_id"] for r in rules]
        assert schema_ids == sorted(schema_ids)


class TestDeleteRoute:
    """Tests for DELETE /api/routing/{schema_id}."""

    async def test_delete_route_success(self, client: AsyncClient) -> None:
        """Deleting an existing route returns 204 and removes it."""
        await client.put(
            "/api/routing/ks_DeleteMe",
            json={"endpoint": "ep:8080"},
        )

        resp = await client.delete("/api/routing/ks_DeleteMe")
        assert resp.status_code == 204
        assert resp.content == b""

        # Confirm it is gone
        get_resp = await client.get("/api/routing/ks_DeleteMe")
        assert get_resp.status_code == 404

    async def test_delete_route_not_found(self, client: AsyncClient) -> None:
        """Deleting a non-existent route returns 404."""
        resp = await client.delete("/api/routing/nonexistent_schema")
        assert resp.status_code == 404
        assert "nonexistent_schema" in resp.json()["detail"]

    async def test_delete_route_invalidates_cache(self, client: AsyncClient) -> None:
        """A cached GET returns 404 after the rule is deleted."""
        await client.put(
            "/api/routing/ks_CacheDelete",
            json={"endpoint": "ep:8080"},
        )

        # Prime cache
        get1 = await client.get("/api/routing/ks_CacheDelete")
        assert get1.status_code == 200

        # Delete
        del_resp = await client.delete("/api/routing/ks_CacheDelete")
        assert del_resp.status_code == 204

        # Should not be served from stale cache
        get2 = await client.get("/api/routing/ks_CacheDelete")
        assert get2.status_code == 404

    async def test_delete_route_removes_from_table_listing(self, client: AsyncClient) -> None:
        """Deleted rule no longer appears in GET /api/routing."""
        await client.put("/api/routing/ks_KeepMe", json={"endpoint": "keep:8080"})
        await client.put("/api/routing/ks_DropMe", json={"endpoint": "drop:8080"})

        await client.delete("/api/routing/ks_DropMe")

        list_resp = await client.get("/api/routing")
        assert list_resp.status_code == 200
        ids = {r["schema_id"] for r in list_resp.json()}
        assert "ks_KeepMe" in ids
        assert "ks_DropMe" not in ids


class TestEnsureRoute:
    """Tests for POST /api/routing/ensure (fingerprint-keyed upsert)."""

    async def test_first_call_creates_rule_with_uuid7_schema_id(self, client: AsyncClient) -> None:
        """New fingerprint allocates a fresh uuid7 schema_id and stores the fp."""
        resp = await client.post(
            "/api/routing/ensure",
            json={
                "fingerprint": "order-v1-abcd1234",
                "json_schema": {"type": "object"},
                "system_prompt": "Be terse.",
                "fallback_chain": ["pgx-local", "deepseek"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["schema_fingerprint"] == "order-v1-abcd1234"
        assert body["fallback_chain"] == ["pgx-local", "deepseek"]
        assert body["json_schema"] == {"type": "object"}
        assert body["system_prompt"] == "Be terse."
        # schema_id is a fresh uuid (36 chars with hyphens); don't hard-code
        # but assert it's not empty and not the fingerprint
        assert body["schema_id"]
        assert body["schema_id"] != "order-v1-abcd1234"

    async def test_repeat_fingerprint_returns_same_rule(self, client: AsyncClient) -> None:
        """Posting the same fingerprint twice returns the identical rule."""
        body = {"fingerprint": "stable-fp-1", "json_schema": {"type": "string"}}
        first = (await client.post("/api/routing/ensure", json=body)).json()
        second = (await client.post("/api/routing/ensure", json=body)).json()
        assert first["schema_id"] == second["schema_id"]
        assert first["schema_fingerprint"] == second["schema_fingerprint"]

    async def test_different_fingerprints_get_different_ids(self, client: AsyncClient) -> None:
        """Distinct fingerprints allocate distinct uuid7 ids."""
        a = (
            await client.post(
                "/api/routing/ensure",
                json={"fingerprint": "fp-a", "json_schema": {"type": "object"}},
            )
        ).json()
        b = (
            await client.post(
                "/api/routing/ensure",
                json={"fingerprint": "fp-b", "json_schema": {"type": "string"}},
            )
        ).json()
        assert a["schema_id"] != b["schema_id"]
        assert a["schema_fingerprint"] == "fp-a"
        assert b["schema_fingerprint"] == "fp-b"

    async def test_default_chain_applied_when_not_provided(self, client: AsyncClient) -> None:
        """Omitting fallback_chain fills in the server default."""
        resp = await client.post(
            "/api/routing/ensure",
            json={"fingerprint": "no-chain-fp", "json_schema": {}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["fallback_chain"]  # non-empty default chain
