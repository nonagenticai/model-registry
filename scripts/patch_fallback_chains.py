#!/usr/bin/env python3
"""Patch routing rule fallback chains to remove mlx_ring and Qwen3-4B-AWQ (where no adapter).

Removes mlx_ring from all routing rules' fallback_chain.
For rules without an adapter_path, also removes Qwen3-4B-AWQ since the base model
cannot produce valid JSON for structured extraction.

Usage:
  # Dry run — show what would change:
  python scripts/patch_fallback_chains.py --dry-run

  # Apply changes:
  python scripts/patch_fallback_chains.py

  # Custom registry URL:
  python scripts/patch_fallback_chains.py --registry-url http://localhost:8155
"""

from __future__ import annotations

import argparse

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch routing rule fallback chains")
    parser.add_argument(
        "--registry-url",
        default="http://model-registry.model-registry.svc.example.com:8155",
        help="Model registry base URL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without applying",
    )
    args = parser.parse_args()

    base_url = args.registry_url.rstrip("/")

    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{base_url}/api/routing")
        resp.raise_for_status()
        rules = resp.json()

    print(f"Fetched {len(rules)} routing rules from {base_url}")
    print()

    changed = 0
    skipped = 0

    for rule in rules:
        schema_id = rule["schema_id"]
        chain = rule.get("fallback_chain") or []
        adapter_path = rule.get("adapter_path")

        new_chain = [p for p in chain if p != "mlx_ring"]

        if not adapter_path:
            new_chain = [p for p in new_chain if p not in ("Qwen3-4B-AWQ", "rtx")]

        if new_chain == chain:
            skipped += 1
            continue

        removed = set(chain) - set(new_chain)
        print(f"  {schema_id}:")
        print(f"    old: {chain}")
        print(f"    new: {new_chain}")
        print(f"    removed: {removed}  adapter: {'yes' if adapter_path else 'no'}")

        if not args.dry_run:
            with httpx.Client(timeout=30.0) as client:
                resp = client.put(
                    f"{base_url}/api/routing/{schema_id}",
                    json={"fallback_chain": new_chain},
                )
                resp.raise_for_status()
            print("    -> updated")

        changed += 1

    print()
    action = "would change" if args.dry_run else "changed"
    print(f"Done: {changed} {action}, {skipped} unchanged out of {len(rules)} total")

    if args.dry_run and changed > 0:
        print("\nRe-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
