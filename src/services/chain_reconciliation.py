"""Detective control: re-check STORED fallback chains against the live catalog.

Why a second control exists
---------------------------
``services.chain_validation`` is preventive: it rejects an unroutable
``fallback_chain`` at the moment someone writes it (HTTP 422). That closes the
door on new damage, and it is the only control that can, because
model-registry is the one chokepoint both writers pass through.

It cannot close the door on damage that arrives with **no write at all**. A
chain is validated once, at registration. Everything it was validated against
-- the set of provider names, and which of them are ``enabled`` -- lives in
inference-router's ``providers.yaml`` and changes independently. Flip a
provider to ``enabled: false`` there and every stored rule that leaned on it
silently loses a hop. No request touched model-registry; no guard ran; the
rules simply stopped being able to fail over.

That is not a hypothetical either. It is the exact history of the three
``["gemma","deepseek","claude"]`` rules: ``claude`` was enabled when they were
written and disabled later, which collapsed them to a single hop, and nothing
in the system said so for three months.

So: the guard covers "bad when written", this covers "rotted since". Neither
subsumes the other.

Three properties this deliberately has
--------------------------------------
1. **It never repairs.** ``reconcile`` issues one ``SELECT`` and writes
   nothing -- not the rules, not an audit row, not a "fixed" chain. A sweep
   that quietly rewrote a rotted chain would erase the evidence that a provider
   flip broke 56 rules, and for the rules re-asserted nightly by
   ``coding-llm-bootstrap`` the repair would be undone within 24h and redone
   forever, so the log would show perpetual success while the real
   misconfiguration -- in the *writer's* config -- was never fixed. Report,
   name the rules, let a human decide.

2. **It never fails closed.** If the catalog is empty or unparseable the sweep
   reports ``status="unknown"`` with ``unroutable=None``. It does NOT mark all
   163 rules unroutable, and nothing here can block a read or a write. The
   failure mode being avoided is concrete: an unreadable config turning into a
   fleet-wide false alarm over rules that are serving traffic perfectly well.
   "I could not tell" and "everything is broken" are different answers and the
   caller gets to see which one it is. (The preventive guard is the opposite --
   it fails CLOSED on an empty catalog, because refusing one write is cheap.
   The asymmetry is intentional; see ``chain_validation``.)

3. **It never breaks startup or the request path.** The periodic task swallows
   every non-cancellation exception, logs it, and keeps its schedule. A
   reporting control that can crash the pod it reports on is a worse defect
   than the one it reports.

What it counts
--------------
``unroutable`` is split by the rule's own ``enabled`` column, because the two
halves warrant different urgency and an alert needs to tell them apart:

* ``unroutable_enabled`` -- rules that can be SERVED right now with no
  failover. This is the number worth paging on.
* ``unroutable_disabled`` -- rules that are switched off. Harmless today, but
  they are the revival hazard: re-enabling one hands traffic straight to a
  chain that cannot fail over, and nothing on the re-enable path re-validates
  the chain (flipping ``enabled`` is not a chain write).

Measured 2026-08-16, after the ``gemma`` rules were repaired at source: 169
rules, 113 healthy ``["pgx-local","deepseek"]``, 56 single-hop
``["pgx-local"]``. The sweep's job is to make that 56 a number somebody sees
before the next provider flip makes it larger.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from logging_config import get_logger
from models.db import RoutingRuleORM
from services.chain_validation import (
    UnroutableChainError,
    chain_unroutable_reason,
    parse_provider_catalog,
)
from settings import get_settings

logger = get_logger(__name__)

__all__ = [
    "ChainReconciler",
    "ReconciliationReport",
    "UnroutableRule",
]

ReconciliationStatus = Literal["ok", "unroutable_rules_found", "unknown"]


@dataclass(frozen=True)
class UnroutableRule:
    """One stored rule the current catalog cannot route.

    Attributes:
        schema_id: Primary key of the offending rule.
        fallback_chain: The chain AS STORED. Never a repaired version -- the
            point of the report is to show what is actually in the table.
        enabled: The rule's own ``enabled`` column. Drives which of the two
            counters this finding lands in.
        reason: Why it is unroutable, from
            :func:`services.chain_validation.chain_unroutable_reason` -- the
            same sentence the registration guard would have produced.
    """

    schema_id: str
    fallback_chain: list[str]
    enabled: bool
    reason: str


@dataclass(frozen=True)
class ReconciliationReport:
    """Outcome of one sweep. Purely descriptive -- nothing acts on it.

    Attributes:
        status: ``"ok"`` (swept, nothing unroutable),
            ``"unroutable_rules_found"`` (swept, findings below), or
            ``"unknown"`` (could NOT sweep -- catalog empty or unparseable).
            ``"unknown"`` is a distinct value precisely so it can never be
            mistaken for ``"ok"`` by a reader or an alert expression.
        scanned: Rules examined. ``0`` when status is ``"unknown"``.
        unroutable: Total findings, or ``None`` when status is ``"unknown"``.
            ``None`` rather than ``0``: "I could not check" must not render as
            "I checked and it was clean".
        unroutable_enabled: Findings on rules with ``enabled=True`` -- the
            page-worthy count. ``None`` when unknown.
        unroutable_disabled: Findings on switched-off rules -- the revival
            hazard. ``None`` when unknown.
        catalog: The catalog the sweep resolved, so a reader can tell a real
            regression from a catalog that changed under it. ``None`` when
            unknown.
        findings: Every offending rule, capped by ``report_limit``.
        findings_truncated: True when ``findings`` was capped, so a reader
            never mistakes the list length for the count.
        detail: One sentence a human can act on. Always populated, and it is
            the only place the ``"unknown"`` cause is spelled out.
        completed_at: When the sweep finished (UTC). Lets a reader spot a
            reconciler that silently stopped running -- a stale report is its
            own defect and is invisible without a timestamp.
    """

    status: ReconciliationStatus
    scanned: int
    unroutable: int | None
    unroutable_enabled: int | None
    unroutable_disabled: int | None
    catalog: dict[str, bool] | None
    detail: str
    completed_at: datetime
    findings: list[UnroutableRule] = field(default_factory=list)
    findings_truncated: bool = False


class ChainReconciler:
    """Runs the sweep on startup and on an interval, and caches the last report.

    Holds no database handle of its own: ``reconcile`` takes a session so the
    request path, the tests and the background task all drive the identical
    code, rather than the background task getting a second, differently-behaved
    implementation nobody exercises.
    """

    def __init__(self) -> None:
        self._last_report: ReconciliationReport | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def last_report(self) -> ReconciliationReport | None:
        """The most recent report, or ``None`` if no sweep has completed yet.

        ``None`` is not "healthy". A caller rendering this must say "no sweep
        has run", never fall back to zero.
        """
        return self._last_report

    async def reconcile(self, session: AsyncSession) -> ReconciliationReport:
        """Re-evaluate every stored rule against the current catalog.

        Read-only by construction: the single statement is a ``SELECT`` and no
        ORM instance is mutated, so there is nothing for a stray ``flush`` to
        write back.

        Args:
            session: Database session to read the routing table through.

        Returns:
            The report, which is also cached on :attr:`last_report`.
        """
        settings = get_settings()

        # Resolve the catalog FIRST. If we cannot, we stop here and say so --
        # we do not fall through and judge 163 rules against an empty dict,
        # which would call every one of them unroutable.
        try:
            catalog = parse_provider_catalog(settings.routing_provider_catalog)
        except UnroutableChainError as exc:
            return self._unknown(
                f"provider catalog is unparseable, so no stored chain could be re-checked: {exc}"
            )
        if not catalog:
            return self._unknown(
                "MODEL_REGISTRY_ROUTING_PROVIDER_CATALOG is empty, so no stored "
                "chain could be re-checked; set it to mirror inference-router's "
                "provider catalog"
            )

        min_hops = settings.routing_min_viable_hops
        rows = (
            (
                await session.execute(
                    sa.select(
                        RoutingRuleORM.schema_id,
                        RoutingRuleORM.fallback_chain,
                        RoutingRuleORM.enabled,
                    ).order_by(RoutingRuleORM.schema_id)
                )
            )
            .tuples()
            .all()
        )

        findings: list[UnroutableRule] = []
        n_enabled = 0
        n_disabled = 0
        for schema_id, chain, is_enabled in rows:
            # A NULL chain is column-illegal (nullable=False) but is treated as
            # empty rather than crashing the sweep: a reporting control must
            # survive the bad data it exists to report.
            stored_chain = list(chain or [])
            reason = chain_unroutable_reason(stored_chain, catalog, min_hops)
            if reason is None:
                continue
            if is_enabled:
                n_enabled += 1
            else:
                n_disabled += 1
            if len(findings) < settings.routing_reconcile_report_limit:
                findings.append(
                    UnroutableRule(
                        schema_id=schema_id,
                        fallback_chain=stored_chain,
                        enabled=bool(is_enabled),
                        reason=reason,
                    )
                )

        total = n_enabled + n_disabled
        report = ReconciliationReport(
            status="unroutable_rules_found" if total else "ok",
            scanned=len(rows),
            unroutable=total,
            unroutable_enabled=n_enabled,
            unroutable_disabled=n_disabled,
            catalog=dict(catalog),
            findings=findings,
            findings_truncated=total > len(findings),
            detail=(
                f"{total} of {len(rows)} stored routing rules cannot fail over "
                f"against the current catalog ({n_enabled} enabled, "
                f"{n_disabled} disabled). NOT repaired -- fix the writer's "
                f"configuration, or the provider catalog."
                if total
                else f"all {len(rows)} stored routing rules are routable"
            ),
            completed_at=datetime.now(UTC),
        )
        self._last_report = report
        self._log(report)
        return report

    def _unknown(self, detail: str) -> ReconciliationReport:
        """Build, cache and log an "I could not check" report.

        Every count is ``None``, never ``0`` -- see
        :class:`ReconciliationReport`.
        """
        report = ReconciliationReport(
            status="unknown",
            scanned=0,
            unroutable=None,
            unroutable_enabled=None,
            unroutable_disabled=None,
            catalog=None,
            detail=detail,
            completed_at=datetime.now(UTC),
        )
        self._last_report = report
        self._log(report)
        return report

    @staticmethod
    def _log(report: ReconciliationReport) -> None:
        """Emit the report at a level matching what a human must do about it.

        ERROR when a rule that can serve traffic has no failover, WARNING when
        only switched-off rules are affected or the catalog was unreadable,
        INFO when clean. The event name is stable so it is greppable and
        alertable; the counts are structured fields, not prose.
        """
        fields = {
            "status": report.status,
            "scanned": report.scanned,
            "unroutable": report.unroutable,
            "unroutable_enabled": report.unroutable_enabled,
            "unroutable_disabled": report.unroutable_disabled,
            "detail": report.detail,
        }
        if report.status == "unknown":
            logger.warning("routing_chain_reconcile_unknown", **fields)
            return
        if report.unroutable_enabled:
            logger.error(
                "routing_chain_reconcile_unroutable",
                schema_ids=[f.schema_id for f in report.findings if f.enabled],
                **fields,
            )
        elif report.unroutable:
            logger.warning(
                "routing_chain_reconcile_unroutable_disabled_only",
                schema_ids=[f.schema_id for f in report.findings],
                **fields,
            )
        else:
            logger.info("routing_chain_reconcile_clean", **fields)

    async def run_forever(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        """Sweep once immediately, then every ``routing_reconcile_interval_seconds``.

        The immediate first pass is the point: a provider flip is discovered on
        the next roll rather than up to an interval later, and the rollout
        itself becomes the trigger.

        Every exception except cancellation is caught and logged, and the loop
        keeps its schedule. A transient database blip must not permanently
        silence the only control that watches for rotted chains -- a detective
        control that dies quietly is indistinguishable from one reporting
        "clean".

        Args:
            session_factory: ``async_sessionmaker`` yielding an
                :class:`AsyncSession`. Passed in rather than resolved here so
                this module never imports the engine, and so tests can drive
                the loop without a database.
        """
        interval = get_settings().routing_reconcile_interval_seconds
        while True:
            try:
                async with session_factory() as session:
                    await self.reconcile(session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - see docstring
                logger.error("routing_chain_reconcile_failed", error=str(exc))
            await asyncio.sleep(interval)

    def start(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        """Start the background loop, unless the interval is ``0`` (disabled).

        Args:
            session_factory: Passed through to :meth:`run_forever`.
        """
        interval = get_settings().routing_reconcile_interval_seconds
        if interval <= 0:
            logger.warning(
                "routing_chain_reconcile_disabled",
                detail=(
                    "MODEL_REGISTRY_ROUTING_RECONCILE_INTERVAL_SECONDS is 0; "
                    "stored chains will NOT be re-checked against the catalog"
                ),
            )
            return
        self._task = asyncio.create_task(self.run_forever(session_factory))
        logger.info("routing_chain_reconcile_started", interval_seconds=interval)

    async def stop(self) -> None:
        """Cancel the background loop and wait for it to unwind."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
