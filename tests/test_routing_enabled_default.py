"""Tests for the routing_rules.enabled flag (Phase 1 substrate).

The migration adds ``enabled BOOLEAN NOT NULL DEFAULT TRUE`` so existing
behaviour is preserved. New rules can opt out by passing ``enabled: false``
through the upsert endpoint.
"""

from httpx import AsyncClient


class TestRoutingEnabledDefault:
    async def test_new_rule_defaults_enabled_true(self, client: AsyncClient) -> None:
        """A freshly-created rule has enabled=true without the caller specifying it."""
        await client.put(
            "/api/routing/ks_DefaultEnabled",
            json={"endpoint": "ep:8080"},
        )
        resp = await client.get("/api/routing/ks_DefaultEnabled")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is True

    async def test_post_routing_with_enabled_false_persists(self, client: AsyncClient) -> None:
        """Passing enabled=false on create persists; round-trips on GET."""
        await client.put(
            "/api/routing/ks_DisabledNew",
            json={"endpoint": "ep:8080", "enabled": False},
        )
        resp = await client.get("/api/routing/ks_DisabledNew")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    async def test_partial_update_does_not_clobber_enabled(self, client: AsyncClient) -> None:
        """Updating an unrelated field with omitted enabled preserves prior value."""
        await client.put(
            "/api/routing/ks_PreserveEnabled",
            json={"endpoint": "ep:8080", "enabled": False},
        )
        # Update only the endpoint; enabled flag is omitted.
        await client.put(
            "/api/routing/ks_PreserveEnabled",
            json={"endpoint": "ep-v2:9090"},
        )
        body = (await client.get("/api/routing/ks_PreserveEnabled")).json()
        assert body["endpoint"] == "ep-v2:9090"
        assert body["enabled"] is False

    async def test_can_flip_enabled_back_to_true(self, client: AsyncClient) -> None:
        """A disabled rule can be re-enabled by sending enabled=true."""
        await client.put(
            "/api/routing/ks_FlipBack",
            json={"endpoint": "ep:8080", "enabled": False},
        )
        await client.put(
            "/api/routing/ks_FlipBack",
            json={"enabled": True},
        )
        body = (await client.get("/api/routing/ks_FlipBack")).json()
        assert body["enabled"] is True
