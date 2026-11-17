"""Tests for audit trail API endpoints and audit event emission."""

from typing import Any

from httpx import AsyncClient


class TestAuditEventEmission:
    """Tests that audit events are emitted when mutations occur."""

    async def test_register_model_emits_audit_event(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Registering a model creates an audit event with correct fields."""
        resp = await client.post("/api/models", json=sample_model_data)
        assert resp.status_code == 201
        model_id = resp.json()["id"]

        audit_resp = await client.get(
            "/api/audit", params={"entity_type": "model_version", "entity_id": model_id}
        )
        assert audit_resp.status_code == 200
        body = audit_resp.json()
        assert body["total"] == 1
        events = body["events"]
        reg_events = [e for e in events if e["event_type"] == "model_registered"]
        assert len(reg_events) == 1
        event = reg_events[0]
        assert event["entity_type"] == "model_version"
        assert event["entity_id"] == model_id
        assert event["actor"] == "api"
        assert "schema_id" in event["detail"]
        assert "version" in event["detail"]
        assert event["detail"]["schema_id"] == sample_model_data["schema_id"]
        assert event["detail"]["version"] == sample_model_data["version"]

    async def test_status_change_emits_audit_event(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Changing model status creates an audit event with from/to detail."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        await client.patch(f"/api/models/{model_id}/status", json={"status": "evaluating"})

        audit_resp = await client.get(
            "/api/audit",
            params={"event_type": "status_changed", "entity_id": model_id},
        )
        assert audit_resp.status_code == 200
        body = audit_resp.json()
        assert body["total"] >= 1
        event = body["events"][0]
        assert event["entity_type"] == "model_version"
        assert event["entity_id"] == model_id
        assert event["detail"]["from_status"] == "training"
        assert event["detail"]["to_status"] == "evaluating"

    async def test_route_update_emits_audit_event(self, client: AsyncClient) -> None:
        """Updating a routing rule creates an audit event with endpoint detail."""
        await client.put(
            "/api/routing/ks_AuditRoute",
            json={"endpoint": "ep:8080"},
        )

        audit_resp = await client.get(
            "/api/audit",
            params={"entity_type": "routing_rule", "entity_id": "ks_AuditRoute"},
        )
        assert audit_resp.status_code == 200
        body = audit_resp.json()
        assert body["total"] >= 1
        route_events = [e for e in body["events"] if e["event_type"] == "route_updated"]
        assert len(route_events) >= 1
        assert route_events[0]["detail"]["endpoint"] == "ep:8080"

    async def test_alias_operations_emit_audit(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Setting and deleting an alias creates alias_set and alias_deleted events."""
        resp = await client.post("/api/models", json=sample_model_data)
        assert resp.status_code == 201
        model_id = resp.json()["id"]
        schema_id = sample_model_data["schema_id"]

        # Set an alias
        alias_resp = await client.put(
            f"/api/aliases/{schema_id}/champion",
            json={"model_id": model_id},
        )
        assert alias_resp.status_code == 200

        # Delete the alias
        del_resp = await client.delete(f"/api/aliases/{schema_id}/champion")
        assert del_resp.status_code == 204

        # Verify alias_set event
        set_audit = await client.get(
            "/api/audit",
            params={"event_type": "alias_set"},
        )
        assert set_audit.status_code == 200
        set_events = set_audit.json()["events"]
        matching_set = [e for e in set_events if e["entity_id"] == f"{schema_id}:champion"]
        assert len(matching_set) >= 1
        assert matching_set[0]["detail"]["alias"] == "champion"
        assert matching_set[0]["detail"]["to_model_id"] == model_id

        # Verify alias_deleted event
        del_audit = await client.get(
            "/api/audit",
            params={"event_type": "alias_deleted"},
        )
        assert del_audit.status_code == 200
        del_events = del_audit.json()["events"]
        matching_del = [e for e in del_events if e["entity_id"] == f"{schema_id}:champion"]
        assert len(matching_del) >= 1
        assert matching_del[0]["detail"]["alias"] == "champion"
        assert matching_del[0]["detail"]["model_id"] == model_id

    async def test_tag_operations_emit_audit(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Setting and deleting tags creates tag_set and tag_deleted events."""
        resp = await client.post("/api/models", json=sample_model_data)
        assert resp.status_code == 201
        model_id = resp.json()["id"]

        # Set tags
        tag_resp = await client.put(
            f"/api/models/{model_id}/tags",
            json={"tags": {"env": "staging", "team": "ml"}},
        )
        assert tag_resp.status_code == 200

        # Delete a tag
        del_resp = await client.delete(f"/api/models/{model_id}/tags/env")
        assert del_resp.status_code == 204

        # Verify tag_set event
        set_audit = await client.get(
            "/api/audit",
            params={"event_type": "tag_set", "entity_id": model_id},
        )
        assert set_audit.status_code == 200
        set_body = set_audit.json()
        assert set_body["total"] >= 1
        tag_set_event = set_body["events"][0]
        assert tag_set_event["entity_type"] == "tag"
        assert tag_set_event["detail"]["tags"]["env"] == "staging"
        assert tag_set_event["detail"]["tags"]["team"] == "ml"

        # Verify tag_deleted event
        del_audit = await client.get(
            "/api/audit",
            params={"event_type": "tag_deleted", "entity_id": model_id},
        )
        assert del_audit.status_code == 200
        del_body = del_audit.json()
        assert del_body["total"] >= 1
        tag_del_event = del_body["events"][0]
        assert tag_del_event["entity_type"] == "tag"
        assert tag_del_event["detail"]["key"] == "env"


class TestAuditListEndpoint:
    """Tests for GET /api/audit."""

    async def test_list_events_empty(self, client: AsyncClient) -> None:
        """Listing events when none exist returns empty."""
        resp = await client.get("/api/audit")
        assert resp.status_code == 200
        body = resp.json()
        assert body["events"] == []
        assert body["total"] == 0

    async def test_list_events_filter_by_event_type(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Filtering by event_type returns only matching events."""
        await client.post("/api/models", json=sample_model_data)

        resp = await client.get("/api/audit", params={"event_type": "model_registered"})
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

        resp2 = await client.get("/api/audit", params={"event_type": "alias_set"})
        assert resp2.status_code == 200
        assert resp2.json()["total"] == 0

    async def test_list_events_filter_by_entity_type(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Filtering by entity_type returns only events of that entity type."""
        # Create a model (entity_type=model_version)
        await client.post("/api/models", json=sample_model_data)

        # Create a routing rule (entity_type=routing_rule)
        await client.put(
            "/api/routing/ks_EntityTypeFilter",
            json={"endpoint": "ep:9090"},
        )

        # Filter for model_version events only
        model_resp = await client.get("/api/audit", params={"entity_type": "model_version"})
        assert model_resp.status_code == 200
        model_events = model_resp.json()["events"]
        assert len(model_events) >= 1
        assert all(e["entity_type"] == "model_version" for e in model_events)

        # Filter for routing_rule events only
        route_resp = await client.get("/api/audit", params={"entity_type": "routing_rule"})
        assert route_resp.status_code == 200
        route_events = route_resp.json()["events"]
        assert len(route_events) >= 1
        assert all(e["entity_type"] == "routing_rule" for e in route_events)

    async def test_list_events_pagination(self, client: AsyncClient) -> None:
        """Pagination limit/offset work correctly."""
        for i in range(3):
            await client.post(
                "/api/models",
                json={"schema_id": f"ks_AuditPag{i}", "version": "v1"},
            )

        resp = await client.get("/api/audit", params={"limit": 2, "offset": 0})
        assert resp.status_code == 200
        assert len(resp.json()["events"]) == 2

        resp2 = await client.get("/api/audit", params={"limit": 2, "offset": 2})
        assert resp2.status_code == 200
        assert len(resp2.json()["events"]) >= 1


class TestAuditModelHistory:
    """Tests for GET /api/audit/models/{model_id}."""

    async def test_model_history(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Model history shows registration and status changes."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        await client.patch(f"/api/models/{model_id}/status", json={"status": "evaluating"})
        await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "evaluated", "eval_score": 0.9},
        )

        history_resp = await client.get(f"/api/audit/models/{model_id}")
        assert history_resp.status_code == 200
        body = history_resp.json()
        # Exactly: model_registered, 2x status_changed
        assert body["total"] == 3
        event_types = [e["event_type"] for e in body["events"]]
        assert sorted(event_types) == sorted(
            ["model_registered", "status_changed", "status_changed"]
        )

    async def test_model_history_returns_events_in_reverse_chronological_order(
        self, client: AsyncClient, sample_model_data: dict[str, Any]
    ) -> None:
        """Model history events are returned newest-first."""
        resp = await client.post("/api/models", json=sample_model_data)
        model_id = resp.json()["id"]

        await client.patch(f"/api/models/{model_id}/status", json={"status": "evaluating"})
        await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "evaluated", "eval_score": 0.8},
        )

        history_resp = await client.get(f"/api/audit/models/{model_id}")
        assert history_resp.status_code == 200
        body = history_resp.json()
        events = body["events"]
        assert len(events) >= 3

        # Verify reverse chronological order (newest first)
        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps, reverse=True)

        # The first event should be the most recent status change
        assert events[0]["event_type"] == "status_changed"
        assert events[0]["detail"]["to_status"] == "evaluated"

        # The last event should be the original registration
        assert events[-1]["event_type"] == "model_registered"

    async def test_model_history_empty_for_unknown_model(self, client: AsyncClient) -> None:
        """Querying history for a non-existent model UUID returns empty."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = await client.get(f"/api/audit/models/{fake_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["events"] == []
        assert body["total"] == 0


class TestAuditSchemaHistory:
    """Tests for GET /api/audit/schemas/{schema_id}."""

    async def test_schema_history_includes_routing(self, client: AsyncClient) -> None:
        """Schema history includes routing rule changes with correct fields."""
        schema_id = "ks_SchemaAudit"
        await client.put(
            f"/api/routing/{schema_id}",
            json={"endpoint": "ep:8080"},
        )

        resp = await client.get(f"/api/audit/schemas/{schema_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        route_events = [e for e in body["events"] if e["event_type"] == "route_updated"]
        assert len(route_events) == 1
        assert route_events[0]["entity_id"] == schema_id
