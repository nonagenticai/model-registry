#!/usr/bin/env python3
"""Backfill the routing rules whose fallback_chain cannot fail over.

⚠️ HELD FOR OPERATOR APPROVAL. This is a write to PRODUCTION data. It is
dry-run by default and refuses to write without an explicit ``--apply``.

What is broken
--------------
Measured against ``model_registry.routing_rules`` (prod, worker4) on
2026-08-17, 163 rules total:

    fallback_chain                     rules  created
    ["pgx-local", "deepseek"]             99  2026-05-12 .. 2026-08-16  OK
    ["pgx-local"]                         61  2026-05-15 .. 2026-06-08  BROKEN
    ["gemma", "deepseek", "claude"]        3  2026-05-10                BROKEN

64 rules (39%) resolve to a single routable provider in inference-router, so
they have no failover at all. The 61 were frozen before ``deepseek`` was
enabled (2026-06-16); the 3 name ``gemma`` (not a provider) and ``claude``
(``enabled: false``), both of which IR silently filters out.

Who gets backfilled, and why not all 64
---------------------------------------
A rule is selected by schema FINGERPRINT, so a rule no live caller matches is
inert and rewriting it is churn. Reachability was measured by joining the 64
against ``data_collector.inference_samples`` (which IR writes on every
provider attempt) over 30 days:

* ``--tier reachable`` (5 rules) -- had traffic in the last 30 days. These are
  actively serving with no failover today. Backfill is a real fix.
* ``--tier hazard`` (9 rules) -- zero traffic in 30 days, BUT their JSON
  schema shape still matches a fingerprint that IS live under a different
  fingerprint. They are inert only because the schema moved on. Any revert of
  that schema silently re-targets them and loses failover with no signal.
  ``019e5a77-9f43-751b-a965-07782a9d66e6`` (``outputdata-48a78d63``, 16821
  lifetime samples) is karma's previous relationship-extraction schema; the
  live one is ``outputdata-f8cf0dad``. Cheap to disarm, zero blast radius.
* The remaining 50 are pure-inert: no traffic in 30 days and no live schema
  twin. Left alone deliberately.

⚠️ This is a DATA-EGRESS POLICY change, not just a repair
---------------------------------------------------------
``["pgx-local"]`` was the DELIBERATE fleet policy when these rows were
written -- inference-router's providers ConfigMap records the 2026-05-15
"local-only stance", superseded on 2026-06-16 when deepseek was enabled as
the failover tier. Adding ``deepseek`` means that on a pgx-local outage these
schemas' payloads leave the tailnet for ``api.deepseek.com``. That is already
true of the 99 healthy rules and of ``default_chain``, so it is the current
policy -- but it is a policy decision per rule, which is the main reason this
script does not run itself.

Rollback
--------
``--apply`` writes ``<--before-image>`` (default
``./backfill_unroutable_chains.before.json``) BEFORE issuing any PUT, and a
matching ``.rollback.sql``. Undo is:

    psql "$MODEL_REGISTRY_DATABASE_URL" -f backfill_unroutable_chains.rollback.sql

Raw-SQL equivalent (only if the API is unreachable -- it skips the audit_event
and the routing TTL cache invalidation that the API path gives you):

    -- before-image
    \\copy (select schema_id, fallback_chain::text from routing_rules
            where schema_id in (...)) to 'before.csv' csv header
    -- idempotent apply
    update routing_rules
       set fallback_chain = '["pgx-local","deepseek"]'::json,
           updated_at     = now()
     where schema_id in (...)
       and fallback_chain::text <> '["pgx-local", "deepseek"]';
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

# The chain 99 of the 163 production rules already carry, and the live
# ``default_chain`` in inference-router's providers ConfigMap.
TARGET_CHAIN: list[str] = ["pgx-local", "deepseek"]

# Explicit allowlists. Deliberately NOT a `where fallback_chain = '...'`
# predicate: a predicate would sweep up all 64 including the 50 rules nothing
# has called in three months. Every id here was individually shown reachable
# (or shown to be a live-schema twin) against data_collector.inference_samples.
#
# 30-day sample counts as measured 2026-08-17.
REACHABLE: dict[str, str] = {
    "019e4a07-17ec-7273-a90c-83c941d3736f": "outputdata-8f824a8b  n30=702",
    "019e5c65-5477-73ee-827e-b91ebcf8f4db": "outputdata-903850bb  n30=255",
    "019e5619-e129-7470-8acb-5d640bed4ef9": "outputdata-df262cc4  n30=176",
    "019e2bf0-7581-75f6-80c3-eee87f0bfd08": "outputdata-b6a01ef1  n30=43",
    "019e7f51-f75e-702e-bee5-1ffbcc4de2c4": "outputdata-2f02d083  n30=9",
}

# Inert today, but a schema-shape twin of a fingerprint that IS live. Reverting
# that schema re-targets these with no signal.
HAZARD: dict[str, str] = {
    "019e5a77-9f43-751b-a965-07782a9d66e6": "outputdata-48a78d63  twin of live outputdata-f8cf0dad (karma relationships)",
    "019ea60a-6172-74fb-99da-dce6dbcbd58d": "outputdata-a9541437  twin of live outputdata-abd03a04",
    "019e2b7e-b579-751f-b755-95f6b16b6e43": "outputdata-50165e40  twin of live dgf007-induce-726f01d8",
    "019e5c65-59a2-76f8-b461-3de135bb6915": "outputdata-85f875b3  twin of live outputdata-2f02d083",
    "019e61ad-c5e3-705d-a4c6-8301bfd98f45": "list-1db7330c        twin of live list-2edd1adf / int-8a0d4ba5",
    "019e5fd4-e3e5-770e-9c88-6a3c637de265": "outputdata-d20a6307  twin of live outputdata-f1fbe818",
    "019e82f6-02b8-729d-901a-35cc6f594715": "outputdata-5b55f7a0  twin of live outputdata-2f02d083",
    "019e5a78-46e9-710c-8d27-a22b3acea0a4": "outputdata-ac120813  twin of live outputdata-51d5d899",
    "019e3b83-3dbc-708c-bc01-8da1aeef122e": "x-8e3cfb13           twin of live outputdata-b6a01ef1",
}

# NOT in either list, on purpose: `test_failure_analysis`,
# `code_quality_analysis`, `performance_analysis` -- the 3
# ["gemma","deepseek","claude"] rules. They have ZERO inference samples ever,
# and they are re-asserted every ~24h by the coding-llm-bootstrap Job
# (ttlSecondsAfterFinished: 86400 + Flux reapply), which reads
# LLM_ROUTING_DEFAULT_FALLBACK_CHAIN="gemma,deepseek,claude" from
# GitOps reconciliation would undo them within a day. Backfilling them Their fix
# is that gitops value, not this script.


def _tier_ids(tier: str) -> dict[str, str]:
    if tier == "reachable":
        return dict(REACHABLE)
    if tier == "hazard":
        return dict(HAZARD)
    return {**REACHABLE, **HAZARD}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-url",
        default="http://model-registry.model-registry.svc.example.com:8155",
        help="model-registry base URL",
    )
    parser.add_argument(
        "--tier",
        choices=["reachable", "hazard", "both"],
        default="reachable",
        help="Which measured set to backfill (default: reachable only)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually issue the PUTs. Without this the script only reports.",
    )
    parser.add_argument(
        "--before-image",
        default="backfill_unroutable_chains.before.json",
        help="Where to write the pre-change values + rollback SQL",
    )
    args = parser.parse_args()

    base_url = args.registry_url.rstrip("/")
    targets = _tier_ids(args.tier)

    before: list[dict[str, object]] = []
    to_change: list[str] = []
    with httpx.Client(timeout=30.0) as client:
        for schema_id, note in targets.items():
            resp = client.get(f"{base_url}/api/routing/{schema_id}")
            if resp.status_code == 404:
                print(f"  MISSING {schema_id}  ({note}) -- rule no longer exists, skipping")
                continue
            resp.raise_for_status()
            rule = resp.json()
            chain = rule.get("fallback_chain") or []
            before.append({"schema_id": schema_id, "fallback_chain": chain, "note": note})
            if chain == TARGET_CHAIN:
                print(f"  OK      {schema_id}  already {chain}")
                continue
            to_change.append(schema_id)
            print(f"  CHANGE  {schema_id}  {chain} -> {TARGET_CHAIN}   ({note})")

    print()
    print(f"tier={args.tier}  inspected={len(targets)}  would change={len(to_change)}")

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to write.")
        return 0

    if not to_change:
        print("Nothing to do.")
        return 0

    # Capture the before-image + rollback BEFORE the first write, not after.
    image_path = Path(args.before_image)
    image_path.write_text(json.dumps(before, indent=2) + "\n")
    rollback_path = image_path.with_suffix(".rollback.sql")
    rollback_path.write_text(
        "-- Generated by scripts/backfill_unroutable_chains.py --apply\n"
        "BEGIN;\n"
        + "".join(
            "UPDATE routing_rules SET fallback_chain = "
            f"'{json.dumps(row['fallback_chain'])}'::json, updated_at = now() "
            f"WHERE schema_id = '{row['schema_id']}';\n"
            for row in before
            if row["schema_id"] in to_change
        )
        + "COMMIT;\n"
    )
    print(f"before-image -> {image_path}")
    print(f"rollback     -> {rollback_path}")

    changed = 0
    with httpx.Client(timeout=30.0) as client:
        for schema_id in to_change:
            resp = client.put(
                f"{base_url}/api/routing/{schema_id}",
                json={"fallback_chain": TARGET_CHAIN},
            )
            if resp.status_code == 422:
                # The registration guard rejected the repair itself -- that
                # means TARGET_CHAIN disagrees with the deployed provider
                # catalog. Stop rather than grind through 14 identical 422s.
                print(f"REJECTED {schema_id}: {resp.text}", file=sys.stderr)
                return 1
            resp.raise_for_status()
            changed += 1
            print(f"  wrote   {schema_id}  -> {resp.json()['fallback_chain']}")

    print(f"\nDone: {changed} rules updated. Undo with:\n  psql ... -f {rollback_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
