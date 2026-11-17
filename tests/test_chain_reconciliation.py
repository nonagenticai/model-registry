"""The DETECTIVE half: stored chains that rotted after they were written.

``test_chain_validation.py`` pins the preventive guard -- an unroutable chain
cannot be REGISTERED. This file pins the class that guard structurally cannot
see: a chain that was valid at write time and became unroutable later because
a provider was flipped to ``enabled: false`` in inference-router's catalog.
No write happens, so no guard runs, and the rule silently loses a hop.

That is not a hypothetical failure mode, it is the recorded history of the
three ``["gemma","deepseek","claude"]`` rules -- ``claude`` was enabled when
they were written and disabled afterwards.

The three properties asserted throughout, because each one is a way this
control could be worse than the defect:

* it REPORTS, it never repairs -- every test that finds a rotted rule also
  asserts the stored row is byte-identical afterwards;
* it never fails CLOSED -- an unreadable catalog yields ``status="unknown"``
  with ``None`` counts, never "all 163 rules are broken";
* it never breaks the service -- the endpoint is 200 whatever it finds, and
  the background loop survives an exception without losing its schedule.
"""

import asyncio
from collections.abc import Iterator
from typing import cast

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from models.db import RoutingRuleORM
from routers import routing as routing_router
from services.chain_reconciliation import ChainReconciler
from services.chain_validation import chain_unroutable_reason, parse_provider_catalog
from settings import get_settings

# The live catalog, read from `kubectl -n inference-router get configmap
# inference-router-providers` on 2026-08-17 -- the same literal
# test_chain_validation.py pins.
LIVE_CATALOG = "pgx-local=true,deepseek=true,xai=true,claude=false"

# The catalog BEFORE claude was disabled. This is the state of the world in
# which ["pgx-local","claude"] was a perfectly good two-hop chain, and the
# reason a rule can rot without anyone touching it.
CATALOG_WITH_CLAUDE_ENABLED = "pgx-local=true,deepseek=true,xai=true,claude=true"

HEALTHY_CHAIN = ["pgx-local", "deepseek"]

# The chain at the centre of this file: valid under
# CATALOG_WITH_CLAUDE_ENABLED, one hop under LIVE_CATALOG.
ROTTABLE_CHAIN = ["pgx-local", "claude"]


def _set_catalog(monkeypatch: pytest.MonkeyPatch, catalog: str, **extra: str) -> None:
    """Point settings at ``catalog``, clearing the lru_cache so it is read.

    ``get_settings`` is ``lru_cache``d. Clearing only before the change leaves
    the mutated instance in the cache for every later test in the same worker,
    which is a well-known way to poison a suite; the fixtures below always
    clear on teardown too.
    """
    monkeypatch.setenv("MODEL_REGISTRY_ROUTING_PROVIDER_CATALOG", catalog)
    monkeypatch.setenv("MODEL_REGISTRY_ROUTING_MIN_VIABLE_HOPS", "2")
    for key, value in extra.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


@pytest.fixture
def live_catalog(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the catalog to today's live values (claude DISABLED)."""
    _set_catalog(monkeypatch, LIVE_CATALOG)
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest.fixture
def reconciler() -> Iterator[ChainReconciler]:
    """A fresh reconciler installed as the module singleton, then restored.

    The endpoint and the background loop deliberately share ONE instance, so
    a test that left its report behind would be read by the next test as if
    the sweep had run. Swapping the module global keeps that sharing intact
    while isolating each test.
    """
    original = routing_router._chain_reconciler
    fresh = ChainReconciler()
    routing_router._chain_reconciler = fresh
    try:
        yield fresh
    finally:
        routing_router._chain_reconciler = original


@pytest_asyncio.fixture
async def rotted_rule(session: AsyncSession) -> RoutingRuleORM:
    """A rule written while ``claude`` was still enabled.

    Inserted through the ORM, beneath the registration guard, because that is
    exactly how it got there in production: it passed validation on the day it
    was written, and the catalog changed afterwards. There is no API call that
    can create this state today, which is the entire point.
    """
    orm = RoutingRuleORM(
        schema_id="written-when-claude-was-enabled",
        schema_fingerprint="outputdata-c1a0de",
        fallback_chain=list(ROTTABLE_CHAIN),
        endpoint="fg-legacy.llm-inference.svc.example.com:8080",
    )
    session.add(orm)
    await session.commit()
    return orm


async def _stored_chain(session: AsyncSession, schema_id: str) -> list[str] | None:
    """Read a rule's chain straight from the table, bypassing every cache."""
    result = await session.execute(
        select(RoutingRuleORM.fallback_chain).where(RoutingRuleORM.schema_id == schema_id)
    )
    return result.scalar_one()


# ---------------------------------------------------------------------------
# THE DEFECT: valid when written, unroutable now
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRuleRotsWhenAProviderIsDisabled:
    """The class registration-time validation cannot reach."""

    async def test_chain_is_valid_while_the_provider_is_enabled(
        self, monkeypatch: pytest.MonkeyPatch, session: AsyncSession, rotted_rule: RoutingRuleORM
    ) -> None:
        """Baseline, and the half that makes the next test non-vacuous.

        Without this, "the sweep reports it" proves nothing -- the chain could
        have been unroutable for some unrelated reason all along.
        """
        _set_catalog(monkeypatch, CATALOG_WITH_CLAUDE_ENABLED)
        try:
            report = await ChainReconciler().reconcile(session)
            assert report.status == "ok"
            assert report.unroutable == 0
            assert report.findings == []
        finally:
            get_settings.cache_clear()

    async def test_same_rule_is_reported_once_the_provider_flips_to_disabled(
        self, monkeypatch: pytest.MonkeyPatch, session: AsyncSession, rotted_rule: RoutingRuleORM
    ) -> None:
        """THE TEST. Nothing wrote to the rule; only the catalog changed."""
        _set_catalog(monkeypatch, LIVE_CATALOG)
        try:
            report = await ChainReconciler().reconcile(session)
        finally:
            get_settings.cache_clear()

        assert report.status == "unroutable_rules_found"
        assert report.unroutable == 1
        assert report.unroutable_enabled == 1
        assert report.unroutable_disabled == 0

        (finding,) = report.findings
        assert finding.schema_id == "written-when-claude-was-enabled"
        # Reported AS STORED -- not normalised, not trimmed of the dead hop.
        assert finding.fallback_chain == ROTTABLE_CHAIN
        assert "claude" in finding.reason
        assert "1 routable" in finding.reason

    async def test_reporting_it_does_not_change_it(
        self, monkeypatch: pytest.MonkeyPatch, session: AsyncSession, rotted_rule: RoutingRuleORM
    ) -> None:
        """⛔ No auto-repair.

        A silent repair of a value some Job re-asserts nightly is undone
        forever and hides the real misconfiguration -- which lives in the
        WRITER's config, where the sweep cannot reach.
        """
        _set_catalog(monkeypatch, LIVE_CATALOG)
        try:
            await ChainReconciler().reconcile(session)
        finally:
            get_settings.cache_clear()

        session.expire_all()
        assert await _stored_chain(session, "written-when-claude-was-enabled") == ROTTABLE_CHAIN

    async def test_repeated_sweeps_keep_reporting_the_same_rule(
        self, monkeypatch: pytest.MonkeyPatch, session: AsyncSession, rotted_rule: RoutingRuleORM
    ) -> None:
        """The finding must not decay.

        A control that reports a problem once and then goes quiet is
        indistinguishable from one that fixed it. It did not fix it.
        """
        _set_catalog(monkeypatch, LIVE_CATALOG)
        try:
            rec = ChainReconciler()
            first = await rec.reconcile(session)
            second = await rec.reconcile(session)
        finally:
            get_settings.cache_clear()

        assert first.unroutable == second.unroutable == 1
        assert await _stored_chain(session, "written-when-claude-was-enabled") == ROTTABLE_CHAIN


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("live_catalog")
class TestCounting:
    """The numbers an alert would be written against."""

    async def test_healthy_table_reports_ok(self, session: AsyncSession) -> None:
        for i in range(3):
            session.add(
                RoutingRuleORM(schema_id=f"healthy-{i}", fallback_chain=list(HEALTHY_CHAIN))
            )
        await session.commit()

        report = await ChainReconciler().reconcile(session)
        assert report.status == "ok"
        assert report.scanned == 3
        assert report.unroutable == 0
        assert "all 3" in report.detail

    async def test_empty_table_reports_ok_not_unknown(self, session: AsyncSession) -> None:
        """Nothing to check is genuinely clean -- distinct from cannot check."""
        report = await ChainReconciler().reconcile(session)
        assert report.status == "ok"
        assert report.scanned == 0
        assert report.unroutable == 0

    async def test_enabled_and_disabled_findings_are_counted_separately(
        self, session: AsyncSession
    ) -> None:
        """The two halves warrant different urgency, so they are two numbers.

        A disabled rule with a dead chain is harmless today but is a revival
        hazard: flipping ``enabled`` back on is not a chain write, so nothing
        re-validates it on the way in.
        """
        session.add(RoutingRuleORM(schema_id="live-single-hop", fallback_chain=["pgx-local"]))
        session.add(
            RoutingRuleORM(
                schema_id="inert-single-hop", fallback_chain=["pgx-local"], enabled=False
            )
        )
        session.add(RoutingRuleORM(schema_id="fine", fallback_chain=list(HEALTHY_CHAIN)))
        await session.commit()

        report = await ChainReconciler().reconcile(session)
        assert report.scanned == 3
        assert report.unroutable == 2
        assert report.unroutable_enabled == 1
        assert report.unroutable_disabled == 1

    async def test_the_live_single_hop_shape_is_reported(self, session: AsyncSession) -> None:
        """The 56 production rules still carrying ``["pgx-local"]``."""
        session.add(RoutingRuleORM(schema_id="one-of-the-56", fallback_chain=["pgx-local"]))
        await session.commit()

        report = await ChainReconciler().reconcile(session)
        assert report.unroutable == 1
        assert report.findings[0].fallback_chain == ["pgx-local"]
        assert "1 routable" in report.findings[0].reason
        assert "cannot " in report.findings[0].reason

    async def test_counts_stay_exact_when_the_finding_list_is_capped(
        self, monkeypatch: pytest.MonkeyPatch, session: AsyncSession
    ) -> None:
        """Truncation must never reach the COUNTS.

        Capping the list keeps one bad catalog change from emitting a
        163-entry payload; capping the count would understate the blast
        radius, which is the number a human acts on.
        """
        for i in range(7):
            session.add(RoutingRuleORM(schema_id=f"broken-{i}", fallback_chain=["pgx-local"]))
        await session.commit()

        _set_catalog(monkeypatch, LIVE_CATALOG, MODEL_REGISTRY_ROUTING_RECONCILE_REPORT_LIMIT="3")
        try:
            report = await ChainReconciler().reconcile(session)
        finally:
            get_settings.cache_clear()

        assert report.unroutable == 7
        assert report.unroutable_enabled == 7
        assert len(report.findings) == 3
        assert report.findings_truncated is True


# ---------------------------------------------------------------------------
# ⛔ Never fail closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDegradesToUnknown:
    """An unreadable catalog is 'I could not check', never 'all broken'."""

    async def test_empty_catalog_reports_unknown_not_all_unroutable(
        self, monkeypatch: pytest.MonkeyPatch, session: AsyncSession
    ) -> None:
        """The inverse of the WRITE guard, on purpose.

        Refusing one write costs a retry. Declaring every stored rule
        unroutable because a config string went missing would manufacture a
        fleet-wide false alarm over rules that are serving traffic fine.
        """
        session.add(RoutingRuleORM(schema_id="fine", fallback_chain=list(HEALTHY_CHAIN)))
        session.add(RoutingRuleORM(schema_id="broken", fallback_chain=["pgx-local"]))
        await session.commit()

        _set_catalog(monkeypatch, "")
        try:
            report = await ChainReconciler().reconcile(session)
        finally:
            get_settings.cache_clear()

        assert report.status == "unknown"
        assert report.unroutable is None
        assert report.unroutable_enabled is None
        assert report.unroutable_disabled is None
        assert report.scanned == 0
        assert "empty" in report.detail

    async def test_unknown_counts_are_none_never_zero(
        self, monkeypatch: pytest.MonkeyPatch, session: AsyncSession
    ) -> None:
        """``0`` would render as 'checked and clean'. It was not checked.

        Asserted separately from the status because ``0 == False`` and
        ``None`` are all falsy: a reader (or an alert expression) that tests
        truthiness cannot tell them apart, so the TYPE has to carry the
        distinction.
        """
        _set_catalog(monkeypatch, "")
        try:
            report = await ChainReconciler().reconcile(session)
        finally:
            get_settings.cache_clear()
        assert report.unroutable is None
        assert report.unroutable != 0

    async def test_unparseable_catalog_reports_unknown(
        self, monkeypatch: pytest.MonkeyPatch, session: AsyncSession
    ) -> None:
        """A typo'd enabled flag must not read as one of the two booleans."""
        _set_catalog(monkeypatch, "pgx-local=maybe,deepseek=true")
        try:
            report = await ChainReconciler().reconcile(session)
        finally:
            get_settings.cache_clear()

        assert report.status == "unknown"
        assert report.unroutable is None
        assert "unparseable" in report.detail

    async def test_an_unknown_sweep_does_not_touch_the_table(
        self, monkeypatch: pytest.MonkeyPatch, session: AsyncSession
    ) -> None:
        session.add(RoutingRuleORM(schema_id="untouched", fallback_chain=["pgx-local"]))
        await session.commit()

        _set_catalog(monkeypatch, "")
        try:
            await ChainReconciler().reconcile(session)
        finally:
            get_settings.cache_clear()

        session.expire_all()
        assert await _stored_chain(session, "untouched") == ["pgx-local"]

    async def test_reads_still_work_while_the_catalog_is_unreadable(
        self, monkeypatch: pytest.MonkeyPatch, client: AsyncClient, session: AsyncSession
    ) -> None:
        """⛔ The sweep must not be able to take routing offline.

        Whatever the reconciler thinks, GET /api/routing keeps serving. This
        is the difference between a monitoring control and an outage.
        """
        session.add(RoutingRuleORM(schema_id="served", fallback_chain=["pgx-local"]))
        await session.commit()

        _set_catalog(monkeypatch, "")
        try:
            response = await client.get("/api/routing")
        finally:
            get_settings.cache_clear()

        assert response.status_code == 200
        assert [r["schema_id"] for r in response.json()] == ["served"]


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("live_catalog")
class TestChainHealthEndpoint:
    """GET /api/routing/chain-health -- a report, not a gate."""

    async def test_chain_health_is_not_shadowed_by_the_schema_id_route(
        self, client: AsyncClient, reconciler: ChainReconciler
    ) -> None:
        """Route-order regression.

        `GET /api/routing/{schema_id}` is a catch-all declared in the same
        router. While `/chain-health` sat after it, this request returned
        `404 No routing rule found for schema_id: chain-health` -- a 404 that
        reads like a missing rule rather than a shadowed route, so the
        endpoint would have shipped dead and looked merely unused.
        """
        response = await client.get("/api/routing/chain-health")
        assert response.status_code == 200
        assert "schema_id" not in response.json()
        assert response.json()["status"] in ("ok", "unroutable_rules_found", "unknown")

    async def test_returns_200_even_when_rules_are_unroutable(
        self, client: AsyncClient, session: AsyncSession, reconciler: ChainReconciler
    ) -> None:
        """A non-200 here would let a monitoring endpoint fail the service."""
        session.add(RoutingRuleORM(schema_id="broken", fallback_chain=["pgx-local"]))
        await session.commit()

        response = await client.get("/api/routing/chain-health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "unroutable_rules_found"
        assert body["unroutable"] == 1
        assert body["unroutable_enabled"] == 1
        assert body["findings"][0]["schema_id"] == "broken"
        assert body["findings"][0]["fallback_chain"] == ["pgx-local"]

    async def test_returns_200_and_unknown_when_the_catalog_is_unreadable(
        self, monkeypatch: pytest.MonkeyPatch, client: AsyncClient, reconciler: ChainReconciler
    ) -> None:
        _set_catalog(monkeypatch, "")
        try:
            response = await client.get("/api/routing/chain-health")
        finally:
            get_settings.cache_clear()

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "unknown"
        assert body["unroutable"] is None
        assert body["catalog"] is None

    async def test_reports_the_catalog_it_judged_against(
        self, client: AsyncClient, reconciler: ChainReconciler
    ) -> None:
        """So a reader can tell a regression from a catalog that moved."""
        body = (await client.get("/api/routing/chain-health")).json()
        assert body["catalog"] == {
            "pgx-local": True,
            "deepseek": True,
            "xai": True,
            "claude": False,
        }

    async def test_serves_the_cached_report_until_refresh_is_asked_for(
        self, client: AsyncClient, session: AsyncSession, reconciler: ChainReconciler
    ) -> None:
        """Polling is free; a fresh sweep is opt-in."""
        first = (await client.get("/api/routing/chain-health")).json()
        assert first["unroutable"] == 0

        session.add(RoutingRuleORM(schema_id="broken-later", fallback_chain=["pgx-local"]))
        await session.commit()

        cached = (await client.get("/api/routing/chain-health")).json()
        assert cached["completed_at"] == first["completed_at"]
        assert cached["unroutable"] == 0

        refreshed = (await client.get("/api/routing/chain-health?refresh=true")).json()
        assert refreshed["unroutable"] == 1

    async def test_requesting_the_report_repairs_nothing(
        self, client: AsyncClient, session: AsyncSession, reconciler: ChainReconciler
    ) -> None:
        session.add(RoutingRuleORM(schema_id="broken", fallback_chain=["pgx-local"]))
        await session.commit()

        await client.get("/api/routing/chain-health?refresh=true")

        session.expire_all()
        assert await _stored_chain(session, "broken") == ["pgx-local"]


# ---------------------------------------------------------------------------
# The background loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("live_catalog")
class TestBackgroundLoop:
    """It must sweep at startup, survive failures, and never be silently off."""

    async def test_start_runs_a_sweep_without_waiting_for_the_interval(
        self, engine, session: AsyncSession
    ) -> None:
        """The first pass is immediate, so a roll is itself the trigger."""
        from sqlalchemy.ext.asyncio import async_sessionmaker

        session.add(RoutingRuleORM(schema_id="broken", fallback_chain=["pgx-local"]))
        await session.commit()

        factory = async_sessionmaker(engine, expire_on_commit=False)
        rec = ChainReconciler()
        rec.start(factory)
        try:
            for _ in range(100):
                if rec.last_report is not None:
                    break
                await asyncio.sleep(0.01)
        finally:
            await rec.stop()

        assert rec.last_report is not None
        assert rec.last_report.unroutable == 1

    async def test_a_failing_sweep_does_not_kill_the_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A DB blip must not permanently silence the only rotted-chain watch.

        A detective control that died quietly reads exactly like one
        reporting 'clean'.

        The interval is pinned to its 1s floor so the retry is observable;
        at the 3600s default the first failure and the second attempt are an
        hour apart and this test would pass vacuously on a dead loop.
        """
        calls = 0

        class _Boom:
            def __call__(self) -> "_Boom":  # noqa: UP037
                nonlocal calls
                calls += 1
                raise RuntimeError("database went away")

        _set_catalog(
            monkeypatch, LIVE_CATALOG, MODEL_REGISTRY_ROUTING_RECONCILE_INTERVAL_SECONDS="1"
        )
        try:
            rec = ChainReconciler()
            factory = cast("async_sessionmaker[AsyncSession]", _Boom())
            task = asyncio.create_task(rec.run_forever(factory))
            try:
                for _ in range(400):
                    if calls >= 2:
                        break
                    await asyncio.sleep(0.01)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        finally:
            get_settings.cache_clear()

        # It came back for a second attempt instead of dying on the first.
        assert calls >= 2

    async def test_a_zero_interval_starts_nothing_and_says_so(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Disabling the sweep is allowed, but never silently.

        The factory is a sentinel that would raise if it were ever called, so
        "no task was started" is proven by the sentinel staying untouched
        rather than only by the ``_task is None`` read.
        """
        _set_catalog(
            monkeypatch, LIVE_CATALOG, MODEL_REGISTRY_ROUTING_RECONCILE_INTERVAL_SECONDS="0"
        )
        touched = False

        def _never_called() -> None:
            nonlocal touched
            touched = True
            raise AssertionError("the factory was used despite a 0 interval")

        try:
            rec = ChainReconciler()
            rec.start(cast("async_sessionmaker[AsyncSession]", _never_called))
            assert rec._task is None
            assert rec.last_report is None
            assert touched is False
        finally:
            get_settings.cache_clear()

    async def test_last_report_is_none_before_any_sweep(self) -> None:
        """``None`` is 'no sweep has run', and must not be read as healthy."""
        assert ChainReconciler().last_report is None


# ---------------------------------------------------------------------------
# The two controls must agree
# ---------------------------------------------------------------------------


class TestTheTwoHalvesShareOnePredicate:
    """A guard that disagrees with its own sweep is worse than either alone."""

    @pytest.mark.parametrize(
        ("chain", "expected_routable"),
        [
            (["pgx-local", "deepseek"], True),
            (["deepseek", "xai"], True),
            (["pgx-local", "deepseek", "xai"], True),
            (["pgx-local"], False),
            (["deepseek", "claude"], False),
            (["claude"], False),
            (["pgx-local", "pgx-local"], False),
            (["gemma", "deepseek", "claude"], False),
            ([], False),
        ],
    )
    def test_sweep_verdict_matches_the_registration_guard(
        self, monkeypatch: pytest.MonkeyPatch, chain: list[str], expected_routable: bool
    ) -> None:
        """Same shapes test_chain_validation.py pins, through the shared path.

        These are the live production shapes. If the sweep and the guard ever
        diverged on one, the estate would get a 422 on a chain its own
        monitoring called healthy, and no operator could act on that.
        """
        from services.chain_validation import UnroutableChainError, validate_fallback_chain

        _set_catalog(monkeypatch, LIVE_CATALOG)
        try:
            catalog = parse_provider_catalog(LIVE_CATALOG)
            sweep_reason = chain_unroutable_reason(chain, catalog, 2)

            guard_rejected = False
            try:
                validate_fallback_chain(chain)
            except UnroutableChainError:
                guard_rejected = True
        finally:
            get_settings.cache_clear()

        assert (sweep_reason is None) is expected_routable
        assert guard_rejected is not expected_routable
        # And when both say no, they say the same thing.
        assert guard_rejected == (sweep_reason is not None)
