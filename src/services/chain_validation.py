"""Registration-time validation of routing-rule fallback chains.

Why this exists
---------------
A routing rule's ``fallback_chain`` is a list of inference-router provider
NAMES. inference-router resolves each name against its own provider catalog
(``inference-router-providers`` ConfigMap) in
``RouterService._build_provider_chain``, which **drops** any name that is not
in the catalog or whose entry is ``enabled: false`` and logs a WARNING. The
caller that wrote the rule never sees that. So a chain can look like failover
while being a single hop, and nothing in the system says so.

That is not hypothetical. Measured on 2026-08-16 across
``model_registry.routing_rules`` (163 rules):

===============================  =====  =====================================
fallback_chain                   rules  why it cannot fail over
===============================  =====  =====================================
["pgx-local", "deepseek"]           99  (healthy -- 2 enabled hops)
["pgx-local"]                       61  one hop; written 2026-05-15..06-08,
                                        before ``deepseek`` was enabled
                                        (2026-06-16), and never migrated
["gemma", "deepseek", "claude"]      3  ``gemma`` is not a provider name at
                                        all and ``claude`` is
                                        ``enabled: false`` -- IR filters both
                                        out and the chain collapses to
                                        ``["deepseek"]``
===============================  =====  =====================================

64 of 163 rules (39%) therefore had no working failover, for three months,
with nothing reporting it.

Design choice: reject, never repair
-----------------------------------
This validator raises. It does **not** rewrite the caller's chain into
something routable. A guard that silently repairs a misconfiguration hides
it: the caller keeps shipping the wrong value, the repair becomes load-bearing,
and the next reader believes the stored chain is what was asked for. The three
``gemma`` rules are the proof -- they are re-asserted by a Job every ~24h
(``coding-llm-bootstrap``, ``ttlSecondsAfterFinished: 86400`` + Flux reapply),
so a silent repair would have been silently undone every night forever.

Rejecting makes the writer fail loudly at the moment it writes the bad value,
which is the only moment anyone is looking.

Scope
-----
This validates *registrability*. It cannot detect a chain that was valid when
written and became unroutable later because a provider was disabled in IR's
catalog (which is precisely how the three ``gemma`` rules rotted): the rule is
never rewritten, so no write path runs and there is nothing to reject.

That residual class is covered by the DETECTIVE half,
``services.chain_reconciliation``, which re-evaluates every STORED rule against
the current catalog on startup and on an interval. The two halves share the
predicate :func:`chain_unroutable_reason` on purpose -- if they each had their
own copy they would drift, and a preventive guard that disagrees with its own
detective sweep is worse than either alone.
"""

from __future__ import annotations

from settings import get_settings

__all__ = [
    "UnroutableChainError",
    "chain_unroutable_reason",
    "parse_provider_catalog",
    "validate_fallback_chain",
]


class UnroutableChainError(ValueError):
    """A fallback chain that inference-router could not fail over with.

    Raised at registration time by :func:`validate_fallback_chain`. Mapped to
    HTTP 422 by ``routers/routing.py`` -- the request is syntactically fine,
    the *value* is unusable, so it is a client error and the caller has to fix
    its config.

    Subclasses ``ValueError`` so callers that already funnel ``ValueError``
    into a 4xx keep working.
    """


def parse_provider_catalog(raw: str) -> dict[str, bool]:
    """Parse the ``name=true|false`` provider catalog string into a mapping.

    Mirrors the ``name`` / ``enabled`` pair of each entry in
    inference-router's ``providers.yaml``.

    Args:
        raw: Comma-separated ``name=true``/``name=false`` pairs. A bare
            ``name`` with no ``=`` is read as enabled, so the shorter
            ``"pgx-local,deepseek"`` form also works.

    Returns:
        Mapping of provider name to its enabled flag. Empty when ``raw`` is
        blank -- callers must treat that as a configuration error, not as
        "validation off" (see :func:`validate_fallback_chain`).

    Raises:
        UnroutableChainError: if an entry names a provider with an
            unparseable enabled flag. A typo'd flag must not silently read as
            one of the two booleans.
    """
    catalog: dict[str, bool] = {}
    for chunk in raw.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        name, sep, flag = entry.partition("=")
        name = name.strip()
        if not name:
            continue
        if not sep:
            catalog[name] = True
            continue
        normalized = flag.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            catalog[name] = True
        elif normalized in ("false", "0", "no", "off"):
            catalog[name] = False
        else:
            raise UnroutableChainError(
                f"provider catalog entry {entry!r} has an unparseable enabled "
                f"flag {flag.strip()!r}; expected true or false"
            )
    return catalog


def chain_unroutable_reason(
    chain: list[str], catalog: dict[str, bool], min_hops: int
) -> str | None:
    """Return why ``chain`` is unroutable against ``catalog``, or ``None`` if it is fine.

    The single predicate behind BOTH controls on this class of defect:

    * :func:`validate_fallback_chain` -- the preventive, registration-time
      guard, which turns a reason into an ``UnroutableChainError`` -> HTTP 422.
    * ``services.chain_reconciliation`` -- the detective, periodic sweep, which
      turns a reason into a counted finding on an already-stored rule.

    It is factored out rather than reimplemented because the two controls MUST
    agree. A rule the guard would reject today and the sweep calls healthy (or
    vice versa) is a contradiction the estate cannot act on -- and two copies of
    a three-clause predicate is precisely the duplicated-value drift this whole
    change exists to stop. Extracting it also makes the sweep cheap: the caller
    parses the catalog ONCE and applies it to every row, instead of re-reading
    settings per rule.

    Pure: no settings read, no I/O, no mutation of ``chain`` or ``catalog``.

    Args:
        chain: Ordered provider names as stored on (or submitted for) a rule.
        catalog: Provider name -> enabled flag, as produced by
            :func:`parse_provider_catalog`. Must be non-empty; an empty catalog
            cannot distinguish "routable" from "unroutable" and the two callers
            handle that case differently on purpose (see
            :func:`validate_fallback_chain`).
        min_hops: Minimum distinct ENABLED providers required.

    Returns:
        A human-readable reason naming the offending providers, or ``None``
        when the chain is routable.
    """
    enabled = sorted(name for name, is_enabled in catalog.items() if is_enabled)

    if not chain:
        return (
            f"fallback_chain is empty; name at least {min_hops} of the enabled providers {enabled}"
        )

    unknown = [name for name in chain if name not in catalog]
    if unknown:
        return (
            f"fallback_chain names provider(s) absent from the catalog: "
            f"{unknown}. inference-router would silently drop them. "
            f"Known providers: {sorted(catalog)} (enabled: {enabled})"
        )

    # Distinct, not len(): duplicates are one hop however many times they are
    # listed, and IR walks providers, not list positions.
    viable = sorted({name for name in chain if catalog[name]})
    if len(viable) < min_hops:
        disabled = sorted({name for name in chain if not catalog[name]})
        return (
            f"fallback_chain {chain} resolves to {len(viable)} routable "
            f"provider(s) {viable}, below the required {min_hops}: it cannot "
            f"fail over. Disabled providers in the chain: {disabled}. "
            f"Enabled providers to choose from: {enabled}"
        )

    return None


def validate_fallback_chain(chain: list[str]) -> None:
    """Reject a fallback chain inference-router cannot actually fail over with.

    Three properties are enforced, in the order a reader would check them:

    1. **Non-empty.** An empty chain leaves IR with nothing but its own
       ``default_chain``, which makes the rule a no-op that looks configured.
    2. **Every name is known.** A name absent from the catalog is dropped by
       IR with only a WARNING, so ``["gemma", "deepseek", "claude"]`` silently
       becomes one hop. Unknown names are almost always typos or
       already-removed providers (e.g. ``mlx_ring``).
    3. **At least ``routing_min_viable_hops`` DISTINCT ENABLED providers.**
       Counting distinct enabled names is what catches all three live broken
       shapes: the single-hop ``["pgx-local"]``, the
       collapses-to-one-hop ``["gemma", "deepseek", "claude"]``, and the
       duplicate-padded ``["pgx-local", "pgx-local"]`` (two entries, one hop).

    Args:
        chain: Ordered provider names as supplied by the caller (or the
            server default filled in for it).

    Raises:
        UnroutableChainError: on any of the three violations, or when the
            configured catalog is empty. The message names the offending
            providers and the viable set, so the caller can fix its config
            from the error alone.
    """
    settings = get_settings()
    catalog = parse_provider_catalog(settings.routing_provider_catalog)

    if not catalog:
        # Fail CLOSED **on writes**. Treating an unset catalog as "skip
        # validation" is how a guard becomes decorative: it would pass in
        # exactly the environment whose config nobody filled in. The field
        # ships with a correct non-empty default, so an empty value can only be
        # a deliberate (and wrong) override.
        #
        # ⚠️ The detective half deliberately does the OPPOSITE. A *write* that
        # cannot be checked is safe to refuse -- the caller retries, nothing
        # already in service is affected. A *sweep* that cannot read the
        # catalog must NOT declare all 163 stored rules unroutable: that would
        # convert an unreadable config into a fleet-wide false alarm (and, if
        # anything ever gated on it, an outage) over rules that are serving
        # traffic perfectly well. It reports status "unknown" instead. See
        # ``services.chain_reconciliation.ChainReconciler.reconcile``.
        raise UnroutableChainError(
            "MODEL_REGISTRY_ROUTING_PROVIDER_CATALOG is empty, so no fallback "
            "chain can be validated; set it to mirror inference-router's "
            "provider catalog (e.g. 'pgx-local=true,deepseek=true,claude=false')"
        )

    reason = chain_unroutable_reason(chain, catalog, settings.routing_min_viable_hops)
    if reason is not None:
        raise UnroutableChainError(reason)
