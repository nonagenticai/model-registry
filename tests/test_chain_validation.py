"""Registration-time rejection of unroutable fallback chains.

Pins the guard added for the 2026-08-16 production finding: 64 of 163
``routing_rules`` (39%) held a ``fallback_chain`` that inference-router could
not fail over with, accumulated over three months with nothing reporting it.

The two live broken shapes are asserted verbatim, because they are the whole
point of the guard:

* ``["pgx-local"]``                    -- 61 rules; one hop, written before
                                          ``deepseek`` was enabled 2026-06-16
* ``["gemma","deepseek","claude"]``    -- 3 rules; ``gemma`` is not a provider
                                          name and ``claude`` is
                                          ``enabled: false``, so IR's
                                          ``_build_provider_chain`` filtered
                                          both out and the chain collapsed to
                                          the single hop ``["deepseek"]``

Both must be rejected, not repaired: the ``gemma`` rules are re-asserted every
~24h by the ``coding-llm-bootstrap`` Job, so a silent repair would be silently
undone every night.
"""

from collections.abc import Iterator
from typing import Any

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.db import RoutingRuleORM
from services.chain_validation import (
    UnroutableChainError,
    parse_provider_catalog,
    validate_fallback_chain,
)
from settings import DEFAULT_FALLBACK_CHAIN, get_settings

# The live catalog, read from `kubectl -n inference-router get configmap
# inference-router-providers` on 2026-08-17. Keep this literal in the test so a
# change to the shipped default has to be acknowledged here.
LIVE_CATALOG = "pgx-local=true,deepseek=true,xai=true,claude=false"

# The live default_chain from the same ConfigMap, and the chain 99 of the 163
# production rules already carry.
HEALTHY_CHAIN = ["pgx-local", "deepseek"]


@pytest.fixture
def live_catalog(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the provider catalog to the live values for the duration of a test.

    ``get_settings`` is ``lru_cache``d, so the cache has to be cleared BOTH
    before (so the env var is read) and after (so a later test in the same
    worker does not inherit this instance). Skipping the teardown clear is how
    a settings-mutating test poisons every subsequent test in the worker.
    """
    monkeypatch.setenv("MODEL_REGISTRY_ROUTING_PROVIDER_CATALOG", LIVE_CATALOG)
    monkeypatch.setenv("MODEL_REGISTRY_ROUTING_MIN_VIABLE_HOPS", "2")
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest_asyncio.fixture
async def broken_rule(session: AsyncSession) -> RoutingRuleORM:
    """Insert a rule with an unroutable chain, bypassing the API guard.

    Reproduces one of the 61 frozen ``["pgx-local"]`` rows that already exist
    in production. Written through the ORM on purpose: the guard sits on the
    write path, so the only way to test behaviour *around* pre-existing bad
    rows is to create one beneath it.
    """
    orm = RoutingRuleORM(
        schema_id="already-broken",
        schema_fingerprint="outputdata-deadbeef",
        fallback_chain=["pgx-local"],
        endpoint="fg-legacy.llm-inference.svc.example.com:8080",
    )
    session.add(orm)
    await session.commit()
    return orm


# ---------------------------------------------------------------------------
# parse_provider_catalog
# ---------------------------------------------------------------------------


class TestParseProviderCatalog:
    """The catalog string mirrors IR's providers.yaml name/enabled pairs."""

    def test_parses_enabled_and_disabled(self) -> None:
        assert parse_provider_catalog(LIVE_CATALOG) == {
            "pgx-local": True,
            "deepseek": True,
            "xai": True,
            "claude": False,
        }

    def test_bare_name_means_enabled(self) -> None:
        assert parse_provider_catalog("pgx-local, deepseek") == {
            "pgx-local": True,
            "deepseek": True,
        }

    def test_blank_and_empty_entries_are_skipped(self) -> None:
        assert parse_provider_catalog("  ,pgx-local=true, ,deepseek=true,") == {
            "pgx-local": True,
            "deepseek": True,
        }

    def test_empty_string_yields_empty_catalog(self) -> None:
        assert parse_provider_catalog("") == {}

    def test_unparseable_flag_raises_rather_than_guessing(self) -> None:
        """A typo'd flag must not silently read as one of the two booleans."""
        with pytest.raises(UnroutableChainError, match="unparseable enabled flag"):
            parse_provider_catalog("pgx-local=maybe")


# ---------------------------------------------------------------------------
# validate_fallback_chain
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("live_catalog")
class TestValidateFallbackChain:
    """Unit-level: the three properties, against the live catalog."""

    def test_healthy_two_hop_chain_passes(self) -> None:
        validate_fallback_chain(HEALTHY_CHAIN)

    def test_any_two_enabled_providers_pass(self) -> None:
        validate_fallback_chain(["deepseek", "xai"])

    def test_extra_enabled_hops_pass(self) -> None:
        validate_fallback_chain(["pgx-local", "deepseek", "xai"])

    def test_the_shipped_default_chain_is_itself_valid(self) -> None:
        """A server default its own guard rejects is a startup contradiction.

        This is not a tautology: ``DEFAULT_FALLBACK_CHAIN`` was
        ``["deepseek", "claude"]`` until 2026-08-17, which this guard rejects
        (``claude`` is disabled → one routable hop). Every rule created without
        an explicit chain inherited that.
        """
        validate_fallback_chain(list(DEFAULT_FALLBACK_CHAIN))

    # -- the live broken shapes ------------------------------------------

    def test_single_element_chain_is_rejected(self) -> None:
        """THE 61 ROWS: ``["pgx-local"]`` has nowhere to fail over to."""
        with pytest.raises(UnroutableChainError) as excinfo:
            validate_fallback_chain(["pgx-local"])
        message = str(excinfo.value)
        assert "cannot fail over" in message
        # The error has to be actionable on its own: it names what is viable.
        assert "deepseek" in message

    def test_chain_naming_an_unknown_provider_is_rejected(self) -> None:
        """THE 3 ROWS: ``gemma`` is not a provider name in any catalog entry."""
        with pytest.raises(UnroutableChainError) as excinfo:
            validate_fallback_chain(["gemma", "deepseek", "claude"])
        message = str(excinfo.value)
        assert "gemma" in message
        assert "absent from the catalog" in message

    def test_chain_of_disabled_providers_only_is_rejected(self) -> None:
        """``claude`` is ``enabled: false`` in IR, so it is not a hop."""
        with pytest.raises(UnroutableChainError) as excinfo:
            validate_fallback_chain(["deepseek", "claude"])
        message = str(excinfo.value)
        assert "claude" in message
        assert "1 routable" in message

    def test_only_disabled_providers_is_rejected(self) -> None:
        with pytest.raises(UnroutableChainError, match="0 routable"):
            validate_fallback_chain(["claude"])

    def test_duplicate_padding_does_not_buy_a_hop(self) -> None:
        """Two list entries, one provider: still no failover.

        Counting ``len(chain)`` instead of distinct enabled names would let
        ``["pgx-local", "pgx-local"]`` through, which is the same defect with a
        longer list.
        """
        with pytest.raises(UnroutableChainError, match="1 routable"):
            validate_fallback_chain(["pgx-local", "pgx-local"])

    def test_empty_chain_is_rejected(self) -> None:
        with pytest.raises(UnroutableChainError, match="empty"):
            validate_fallback_chain([])

    def test_unknown_provider_reported_even_when_two_enabled_hops_exist(self) -> None:
        """A typo must not be excused by the rest of the chain being fine.

        ``["pgx-local", "deepseek", "gemmma"]`` would fail over correctly today,
        so a viability-only check would accept it and the typo would sit there
        until someone deleted the working hops.
        """
        with pytest.raises(UnroutableChainError, match="gemmma"):
            validate_fallback_chain(["pgx-local", "deepseek", "gemmma"])


class TestValidateFallbackChainConfiguration:
    """Catalog-configuration behaviour: fail closed, and the hop knob."""

    def test_empty_catalog_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset catalog must NOT read as 'validation off'.

        A guard that skips itself when its config is missing passes in exactly
        the environment nobody configured.
        """
        monkeypatch.setenv("MODEL_REGISTRY_ROUTING_PROVIDER_CATALOG", "")
        get_settings.cache_clear()
        try:
            with pytest.raises(UnroutableChainError, match="is empty"):
                validate_fallback_chain(HEALTHY_CHAIN)
        finally:
            get_settings.cache_clear()

    def test_min_hops_one_permits_a_single_provider_chain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The no-failover policy is expressible, but only deliberately.

        Some rules may legitimately want to stay on the local provider and
        never egress to a paid external one. That must be an explicit
        fleet-level setting, not the accident of an un-migrated 2026-05 row.
        """
        monkeypatch.setenv("MODEL_REGISTRY_ROUTING_PROVIDER_CATALOG", LIVE_CATALOG)
        monkeypatch.setenv("MODEL_REGISTRY_ROUTING_MIN_VIABLE_HOPS", "1")
        get_settings.cache_clear()
        try:
            validate_fallback_chain(["pgx-local"])
            # Still not a licence for unknown names.
            with pytest.raises(UnroutableChainError, match="gemma"):
                validate_fallback_chain(["gemma"])
        finally:
            get_settings.cache_clear()


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("live_catalog")
class TestRoutingApiRejectsUnroutableChains:
    """End-to-end through the HTTP layer: 422, and nothing written."""

    async def test_put_single_element_chain_returns_422_and_writes_nothing(
        self, client: AsyncClient, session: AsyncSession
    ) -> None:
        response = await client.put(
            "/api/routing/new-rule",
            json={"fallback_chain": ["pgx-local"]},
        )
        assert response.status_code == 422, response.text
        assert "cannot fail over" in response.json()["detail"]

        # A rejected registration must not have half-created the rule.
        result = await session.execute(
            select(RoutingRuleORM).where(RoutingRuleORM.schema_id == "new-rule")
        )
        assert result.scalar_one_or_none() is None

    async def test_put_unknown_provider_chain_returns_422(self, client: AsyncClient) -> None:
        """The exact payload the coding-llm-bootstrap Job sends nightly."""
        response = await client.put(
            "/api/routing/test_failure_analysis",
            json={"fallback_chain": ["gemma", "deepseek", "claude"]},
        )
        assert response.status_code == 422, response.text
        assert "gemma" in response.json()["detail"]

    async def test_put_healthy_chain_still_succeeds(self, client: AsyncClient) -> None:
        response = await client.put(
            "/api/routing/healthy-rule",
            json={"fallback_chain": HEALTHY_CHAIN},
        )
        assert response.status_code == 200, response.text
        assert response.json()["fallback_chain"] == HEALTHY_CHAIN

    async def test_put_omitting_chain_gets_the_valid_server_default(
        self, client: AsyncClient
    ) -> None:
        response = await client.put(
            "/api/routing/defaulted-rule",
            json={"endpoint": "fg-x.llm-inference.svc.example.com:8080"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["fallback_chain"] == list(DEFAULT_FALLBACK_CHAIN)

    async def test_ensure_rejects_an_unroutable_chain_on_create(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/routing/ensure",
            json={
                "fingerprint": "outputdata-abc12345",
                "json_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
                "fallback_chain": ["gemma", "deepseek", "claude"],
            },
        )
        assert response.status_code == 422, response.text
        assert "gemma" in response.json()["detail"]

    async def test_ensure_accepts_the_chain_inference_router_actually_sends(
        self, client: AsyncClient
    ) -> None:
        """IR passes its own filtered ``default_chain``; that must not 422."""
        response = await client.post(
            "/api/routing/ensure",
            json={
                "fingerprint": "outputdata-cafe1234",
                "json_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
                "fallback_chain": HEALTHY_CHAIN,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["fallback_chain"] == HEALTHY_CHAIN


@pytest.mark.usefixtures("live_catalog", "broken_rule")
class TestGuardDoesNotBreakExistingBrokenRules:
    """The 64 already-broken rows must stay readable and updatable.

    A write-path guard that also rejects READS of pre-existing bad data turns a
    silent 39%-no-failover problem into a 39%-of-requests-503 outage.
    """

    async def test_get_a_broken_rule_still_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/api/routing/already-broken")
        assert response.status_code == 200, response.text
        assert response.json()["fallback_chain"] == ["pgx-local"]

    async def test_get_table_containing_a_broken_rule_still_returns_200(
        self, client: AsyncClient
    ) -> None:
        response = await client.get("/api/routing")
        assert response.status_code == 200, response.text
        chains = {r["schema_id"]: r["fallback_chain"] for r in response.json()}
        assert chains["already-broken"] == ["pgx-local"]

    async def test_partial_update_omitting_the_chain_is_not_rejected(
        self, client: AsyncClient
    ) -> None:
        """Only a chain the caller SENT is validated.

        Otherwise every unrelated field update to the 64 broken rules (an
        endpoint change, a deploy wiring up active_model_id) would 422 on a
        value it never touched.
        """
        response = await client.put(
            "/api/routing/already-broken",
            json={"endpoint": "fg-moved.llm-inference.svc.example.com:8080"},
        )
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        assert body["endpoint"] == "fg-moved.llm-inference.svc.example.com:8080"
        assert body["fallback_chain"] == ["pgx-local"]

    async def test_ensure_fingerprint_hit_on_a_broken_rule_returns_it_unchanged(
        self, client: AsyncClient
    ) -> None:
        """A fingerprint HIT is a read, not a registration -- never a 422.

        This is the path every karma/w2v request takes. 422-ing here would
        take down every caller of the 64 broken rules.
        """
        response = await client.post(
            "/api/routing/ensure",
            json={"fingerprint": "outputdata-deadbeef"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["schema_id"] == "already-broken"
        assert response.json()["fallback_chain"] == ["pgx-local"]

    async def test_repairing_a_broken_rule_to_a_healthy_chain_is_allowed(
        self, client: AsyncClient
    ) -> None:
        """The backfill path: the guard must permit the fix, not just the break."""
        response = await client.put(
            "/api/routing/already-broken",
            json={"fallback_chain": HEALTHY_CHAIN},
        )
        assert response.status_code == 200, response.text
        assert response.json()["fallback_chain"] == HEALTHY_CHAIN
