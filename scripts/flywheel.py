#!/usr/bin/env python3
"""Qwen3-4B-AWQ Flywheel Orchestrator.

Automates the full training pipeline:
  1. Check triggers / generate training batch (training-data-generator)
  2. Submit training job (model-trainer on mini2)
  3. Wait for training to complete
  4. Advance model status (model-registry)
  5. Run evaluation (model-evaluator)
  6. Deploy approved model (model-registry)

Usage:
  # Run full pipeline for one schema:
  python scripts/flywheel.py --schema ks-entity-spotter

  # Run for all schemas with enough samples:
  python scripts/flywheel.py --all

  # Monitor existing training jobs:
  python scripts/flywheel.py --status

  # Skip batch generation (use existing batch):
  python scripts/flywheel.py --schema ks-entity-spotter --batch-id <id> --minio-path <path>

  # CronJob mode (JSON logging, no interactive output):
  python scripts/flywheel.py --all --cron

  # Dry run (show what would be done):
  python scripts/flywheel.py --all --dry-run

  # Prune old adapter/model versions from MinIO:
  python scripts/flywheel.py --prune

  # Prune with custom retention (keep 3 versions per schema):
  python scripts/flywheel.py --prune --keep-versions 3 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import httpx
from minio import Minio
from minio.error import S3Error

# In-cluster service URLs (K8s DNS).
# Override with env vars for local/dev usage.
TDG_URL = os.environ.get(
    "FLYWHEEL_TDG_URL",
    "http://training-data-generator.training-data-generator.svc.example.com:8152",
)
EVALUATOR_URL = os.environ.get(
    "FLYWHEEL_EVALUATOR_URL",
    "http://model-evaluator.model-evaluator.svc.example.com:8154",
)
REGISTRY_URL = os.environ.get(
    "FLYWHEEL_REGISTRY_URL",
    "http://model-registry.model-registry.svc.example.com:8155",
)
TRAINER_URL = os.environ.get(
    "FLYWHEEL_TRAINER_URL",
    "http://mini2.local:8153",
)
RTX_ENDPOINT = os.environ.get(
    "FLYWHEEL_RTX_URL",
    "http://vllm.llm-inference.svc.example.com:8000",
)

# MinIO settings (used for version pruning)
MINIO_ENDPOINT = os.environ.get("MODEL_REGISTRY_MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MODEL_REGISTRY_MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.environ.get("MODEL_REGISTRY_MINIO_SECRET_KEY", "")
MINIO_SECURE = os.environ.get("MODEL_REGISTRY_MINIO_SECURE", "false").lower() == "true"
ADAPTERS_BUCKET = "adapters"
MODELS_BUCKET = "models"
PRUNE_KEEP_VERSIONS = 2  # Keep latest N adapter versions per schema

# Base model directories that must NEVER be pruned from the models bucket.
# vllm reads both Qwen3-4B-AWQ-270m-it (CronJob training target) and
# gemma-4-e2b-it-text (production inference base). Deleting either causes
# pod CrashLoopBackOff with "No objects found in MinIO at models/<prefix>/".
# Override via FLYWHEEL_PROTECTED_MODEL_DIRS (comma-separated).
PROTECTED_MODEL_DIRS: frozenset[str] = frozenset(
    d.strip()
    for d in os.environ.get(
        "FLYWHEEL_PROTECTED_MODEL_DIRS",
        "Qwen3-4B-AWQ-270m-it,gemma-4-e2b-it-text",
    ).split(",")
    if d.strip()
)

TRAINING_POLL_INTERVAL = 30  # seconds
TRAINING_TIMEOUT = 3600  # 1 hour
EVAL_POLL_INTERVAL = 15  # seconds
EVAL_TIMEOUT = 1800  # 30 minutes
MAX_SCHEMAS_PER_RUN = 12  # Fits in 2h CronJob deadline
EVAL_MAX_RETRIES = 3  # Retry failed evaluations (adapter load / MinIO blips)
EVAL_RETRY_DELAY = 60  # seconds between eval retries

# Schemas with large outputs that need structural scoring
LARGE_OUTPUT_SCHEMAS: set[str] = set()

# Error patterns that indicate "nothing to do" rather than a real failure.
# Matched case-insensitively against error_message from TDG batch_service.
_INSUFFICIENT_SAMPLES_PHRASES = (
    "samples passed filtering",
    "need at least",
)

# Logging mode
_cron_mode = False


def _load_large_output_schemas() -> None:
    """Populate LARGE_OUTPUT_SCHEMAS from env var if set."""
    overrides = os.environ.get("FLYWHEEL_MAX_RESPONSE_LENGTH_OVERRIDES", "{}")
    try:
        parsed = json.loads(overrides)
        LARGE_OUTPUT_SCHEMAS.update(parsed.keys())
    except json.JSONDecodeError:
        pass
    except AttributeError:
        pass


def log(msg: str, level: str = "info", **kwargs) -> None:
    """Log a message, either as structured JSON (cron) or plain text."""
    if _cron_mode:
        entry = {
            "level": level,
            "message": msg,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        entry.update(kwargs)
        print(json.dumps(entry), flush=True)
    else:
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        print(f"  {msg} {extra}".rstrip(), flush=True)


def _is_insufficient_samples(error_msg: str) -> bool:
    """Return True if the error message indicates an insufficient-samples condition."""
    lower = (error_msg or "").lower()
    return any(phrase in lower for phrase in _INSUFFICIENT_SAMPLES_PHRASES)


@dataclass
class BatchResult:
    id: str
    schema_id: str
    train_samples: int
    valid_samples: int
    minio_path: str


@dataclass
class TrainingResult:
    job_id: str
    model_id: str | None
    adapter_path: str | None
    status: str


@dataclass
class SchemaResult:
    schema_id: str
    success: bool
    error: str | None = None
    model_id: str | None = None


async def generate_batch(
    client: httpx.AsyncClient,
    schema_id: str,
    min_samples: int = 20,
    max_samples: int = 500,
) -> BatchResult:
    """Step 1: Generate training batch via training-data-generator."""
    log(f"[1/6] Generating training batch for {schema_id}...", schema_id=schema_id)
    resp = await client.post(
        f"{TDG_URL}/api/batches/generate",
        json={
            "schema_id": schema_id,
            "min_samples": min_samples,
            "train_split": 0.9,
            "max_samples": max_samples,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") == "failed":
        raise RuntimeError(f"Batch generation failed: {data.get('error_message')}")
    result = BatchResult(
        id=data["id"],
        schema_id=schema_id,
        train_samples=data["train_samples"],
        valid_samples=data["valid_samples"],
        minio_path=data["minio_path"],
    )
    log(
        f"       Batch {result.id[:12]}: {result.train_samples} train, {result.valid_samples} valid",
        batch_id=result.id,
        train_samples=result.train_samples,
    )
    return result


async def submit_training(client: httpx.AsyncClient, batch: BatchResult) -> str:
    """Step 2: Submit training job to model-trainer."""
    log(f"[2/6] Submitting training job for {batch.schema_id}...", schema_id=batch.schema_id)
    resp = await client.post(
        f"{TRAINER_URL}/api/jobs",
        json={
            "schema_id": batch.schema_id,
            "training_batch_id": batch.id,
            "minio_path": batch.minio_path,
            "config": {
                "num_epochs": 3,
                "learning_rate": 1e-5,
                "batch_size": 2,
                "min_training_samples": 20,
            },
            "priority": "normal",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    job_id = data["id"]
    log(f"       Job {job_id[:12]} submitted ({data['status']})", job_id=job_id)
    return job_id


async def wait_for_training(client: httpx.AsyncClient, job_id: str) -> TrainingResult:
    """Step 3: Wait for training to complete."""
    log(f"[3/6] Waiting for training job {job_id[:12]}...", job_id=job_id)
    deadline = time.monotonic() + TRAINING_TIMEOUT
    while time.monotonic() < deadline:
        resp = await client.get(f"{TRAINER_URL}/api/jobs/{job_id}", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        status = data["status"]
        progress = data.get("progress", 0) * 100
        step = data.get("current_step", "unknown")

        if status == "completed":
            model_id = data.get("model_id")
            adapter_path = data.get("adapter_path") or data.get("output_adapter_path")
            log(
                f"       Training completed! model_id={model_id}",
                model_id=model_id,
                adapter_path=adapter_path,
            )
            return TrainingResult(
                job_id=job_id, model_id=model_id, adapter_path=adapter_path, status=status
            )
        elif status == "failed":
            error = data.get("error_message", "unknown error")
            log(f"       Training FAILED: {error}", level="error", error=error)
            return TrainingResult(job_id=job_id, model_id=None, adapter_path=None, status=status)
        else:
            if not _cron_mode:
                print(f"\r        {progress:.0f}% - {step}       ", end="", flush=True)
            await asyncio.sleep(TRAINING_POLL_INTERVAL)

    log("       Training timed out", level="error", timeout_seconds=TRAINING_TIMEOUT)
    return TrainingResult(job_id=job_id, model_id=None, adapter_path=None, status="timeout")


async def find_model_for_job(
    client: httpx.AsyncClient, schema_id: str, batch_id: str
) -> str | None:
    """Find model ID created by training, in case trainer didn't return it directly."""
    resp = await client.get(
        f"{REGISTRY_URL}/api/models", params={"schema_id": schema_id}, timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    models = data if isinstance(data, list) else data.get("models", data.get("items", []))
    # Find most recent model in training status for this schema
    for m in models:
        if m.get("status") == "training":
            return m["id"]
    return None


async def advance_to_evaluating(client: httpx.AsyncClient, model_id: str) -> None:
    """Step 4: Advance model status from training -> evaluating."""
    log(f"[4/6] Advancing model {model_id[:12]} to evaluating...", model_id=model_id)
    resp = await client.patch(
        f"{REGISTRY_URL}/api/models/{model_id}/status",
        json={"status": "evaluating"},
        timeout=15,
    )
    resp.raise_for_status()
    log("       Status -> evaluating")


async def run_evaluation(client: httpx.AsyncClient, model_id: str, schema_id: str) -> dict:
    """Step 5: Run evaluation via model-evaluator."""
    log(f"[5/6] Starting evaluation for {schema_id}...", schema_id=schema_id, model_id=model_id)

    # Get JSON schema from routing rule
    routing_resp = await client.get(f"{REGISTRY_URL}/api/routing/{schema_id}", timeout=15)
    routing_resp.raise_for_status()
    routing = routing_resp.json()
    json_schema = routing.get("json_schema")

    eval_data: dict = {
        "model_id": model_id,
        "schema_id": schema_id,
        "endpoint": RTX_ENDPOINT,
        "threshold": 0.85,
        "max_samples": 50,
    }
    if json_schema:
        eval_data["json_schema"] = json_schema

    # Use structural scoring for large-output schemas
    if schema_id in LARGE_OUTPUT_SCHEMAS:
        eval_data["scoring_profile"] = "structural"
        log(f"       Using structural scoring profile for {schema_id}")

    resp = await client.post(f"{EVALUATOR_URL}/api/evaluations", json=eval_data, timeout=30)
    resp.raise_for_status()
    eval_resp = resp.json()
    eval_id = eval_resp.get("id", eval_resp.get("eval_id"))
    log(f"       Evaluation {eval_id} started", eval_id=eval_id)

    # Poll for completion
    deadline = time.monotonic() + EVAL_TIMEOUT
    while time.monotonic() < deadline:
        poll_resp = await client.get(f"{EVALUATOR_URL}/api/evaluations/{eval_id}", timeout=15)
        poll_resp.raise_for_status()
        eval_result = poll_resp.json()
        eval_status = eval_result.get("status", "unknown")

        if eval_status in ("completed", "approved", "rejected"):
            score = eval_result.get("composite_score", eval_result.get("score", "N/A"))
            log(
                f"       Evaluation {eval_status}: score={score}",
                eval_status=eval_status,
                composite_score=score,
            )
            return eval_result
        elif eval_status == "failed":
            error = eval_result.get("error_message", eval_result.get("error", "unknown"))
            log(f"       Evaluation FAILED: {error}", level="error", error=error)
            return eval_result
        else:
            if not _cron_mode:
                print(f"\r        Evaluating... ({eval_status})       ", end="", flush=True)
            await asyncio.sleep(EVAL_POLL_INTERVAL)

    log("       Evaluation timed out", level="error", timeout_seconds=EVAL_TIMEOUT)
    return {"status": "timeout"}


async def deploy_model(client: httpx.AsyncClient, model_id: str, schema_id: str) -> None:
    """Step 6: Deploy approved model."""
    log(
        f"[6/6] Deploying model {model_id[:12]} for {schema_id}...",
        model_id=model_id,
        schema_id=schema_id,
    )

    # Check model status
    resp = await client.get(f"{REGISTRY_URL}/api/models/{model_id}", timeout=15)
    resp.raise_for_status()
    model = resp.json()
    status = model.get("status")

    if status == "evaluated":
        # Approve first
        log("       Approving model...")
        approve_resp = await client.patch(
            f"{REGISTRY_URL}/api/models/{model_id}/status",
            json={"status": "approved"},
            timeout=15,
        )
        approve_resp.raise_for_status()
    elif status != "approved":
        log(f"       Model status is '{status}', cannot deploy (need approved)", level="warning")
        return

    # Deploy
    deploy_resp = await client.post(
        f"{REGISTRY_URL}/api/deploy/{model_id}",
        json={},
        timeout=30,
    )
    deploy_resp.raise_for_status()
    result = deploy_resp.json()
    log(f"       Deployed! {result.get('message', result.get('status'))}")


async def run_flywheel(
    client: httpx.AsyncClient,
    schema_id: str,
    batch_id: str | None = None,
    minio_path: str | None = None,
    min_samples: int = 20,
    dry_run: bool = False,
) -> bool:
    """Run the full flywheel pipeline for a single schema."""
    log(f"\n{'=' * 60}")
    log(f"FLYWHEEL: {schema_id}", schema_id=schema_id)
    log(f"{'=' * 60}")

    if dry_run:
        log(f"DRY RUN: would generate batch, train, evaluate, deploy for {schema_id}")
        return True

    try:
        # Step 1: Generate batch (or use existing)
        if batch_id and minio_path:
            batch = BatchResult(
                id=batch_id,
                schema_id=schema_id,
                train_samples=0,
                valid_samples=0,
                minio_path=minio_path,
            )
            log(f"[1/6] Using existing batch {batch_id[:12]}")
        else:
            batch = await generate_batch(client, schema_id, min_samples=min_samples)

        # Step 2: Submit training
        job_id = await submit_training(client, batch)

        # Step 3: Wait for training
        result = await wait_for_training(client, job_id)
        if result.status != "completed":
            log(f"FAILED: Training did not complete ({result.status})", level="error")
            return False

        # Find model_id (trainer may not return it in the response)
        model_id = result.model_id
        if not model_id:
            model_id = await find_model_for_job(client, schema_id, batch.id)
        if not model_id:
            log("FAILED: Could not find model_id after training", level="error")
            return False

        # Step 4: Advance status
        await advance_to_evaluating(client, model_id)

        # Step 5: Evaluate (with retries for transient failures like MinIO outages)
        eval_result: dict = {}
        eval_status = "unknown"
        for attempt in range(1, EVAL_MAX_RETRIES + 1):
            eval_result = await run_evaluation(client, model_id, schema_id)
            eval_status = eval_result.get("status", "unknown")

            if eval_status != "failed":
                break  # Success or non-retryable terminal status

            # Check if this is a retryable failure (adapter load / infra issue)
            error_msg = eval_result.get("error_message", eval_result.get("error", ""))
            retryable = any(
                phrase in error_msg.lower()
                for phrase in ("adapter", "load", "connect", "timeout", "minio")
            )
            if not retryable or attempt == EVAL_MAX_RETRIES:
                if attempt > 1:
                    log(
                        f"       Evaluation failed after {attempt} attempts",
                        level="error",
                        attempts=attempt,
                    )
                break

            log(
                f"       Evaluation failed (attempt {attempt}/{EVAL_MAX_RETRIES}), "
                f"retrying in {EVAL_RETRY_DELAY}s: {error_msg}",
                level="warning",
                attempt=attempt,
                max_attempts=EVAL_MAX_RETRIES,
            )
            await asyncio.sleep(EVAL_RETRY_DELAY)

            # Reset model status to evaluating so a new evaluation can run
            try:
                await client.patch(
                    f"{REGISTRY_URL}/api/models/{model_id}/status",
                    json={"status": "evaluating"},
                    timeout=15,
                )
            except (httpx.HTTPError, OSError) as e:
                log(f"       Could not reset model status for retry: {e}", level="warning")

        if eval_status in ("approved", "completed"):
            # Step 6: Deploy
            model_resp = await client.get(f"{REGISTRY_URL}/api/models/{model_id}", timeout=15)
            model_resp.raise_for_status()
            model_data = model_resp.json()
            if model_data.get("status") == "approved":
                await deploy_model(client, model_id, schema_id)
            else:
                # Advance: evaluating -> evaluated -> approved
                await client.patch(
                    f"{REGISTRY_URL}/api/models/{model_id}/status",
                    json={
                        "status": "evaluated",
                        "eval_score": eval_result.get(
                            "composite_score", eval_result.get("score", 0)
                        ),
                        "eval_metrics": eval_result.get("metrics", {}),
                    },
                    timeout=15,
                )
                await deploy_model(client, model_id, schema_id)
        else:
            log(f"Evaluation result: {eval_status} -- not deploying", level="warning")
            return False

        log(f"SUCCESS: {schema_id} flywheel complete!", schema_id=schema_id)
        return True

    except (httpx.HTTPError, OSError, RuntimeError, ValueError, KeyError) as e:
        log(f"ERROR: {e}", level="error", error=str(e), schema_id=schema_id)
        return False


async def show_status(client: httpx.AsyncClient) -> None:
    """Show current status of all flywheel services."""
    print("\n=== Training Queue ===")
    try:
        resp = await client.get(f"{TRAINER_URL}/api/queue", timeout=10)
        resp.raise_for_status()
        queue = resp.json()
        current = queue.get("current_job")
        if current:
            print(f"  Active: {current['schema_id']} ({current['status']})")
        queued = queue.get("queued", [])
        for q in queued:
            print(f"  Queued: {q['schema_id']} (position {q['position']})")
        if not current and not queued:
            print("  No active or queued jobs")
    except (httpx.HTTPError, OSError, KeyError, ValueError) as e:
        print(f"  Error: {e}")

    print("\n=== Models in Registry ===")
    try:
        resp = await client.get(f"{REGISTRY_URL}/api/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = data if isinstance(data, list) else data.get("models", data.get("items", []))
        for m in models:
            print(
                f"  {m['schema_id']}: {m['status']} (v{m['version']}, score={m.get('eval_score', 'N/A')})"
            )
    except (httpx.HTTPError, OSError, KeyError, ValueError) as e:
        print(f"  Error: {e}")

    print("\n=== Routing Rules (with active models) ===")
    try:
        resp = await client.get(f"{REGISTRY_URL}/api/routing", timeout=10)
        resp.raise_for_status()
        rules = resp.json()
        active = [r for r in rules if r.get("active_model_id")]
        if active:
            for r in active:
                print(f"  {r['schema_id']}: model={r['active_model_id'][:12]}")
        else:
            print("  No schemas have active models yet")
    except (httpx.HTTPError, OSError, KeyError, ValueError) as e:
        print(f"  Error: {e}")


def prioritize(batches: list[dict], routing_rules: list[dict] | None = None) -> list[dict]:
    """Prioritize schemas for processing.

    Priority order:
    1. HIGH: schemas with NO deployed adapter (never trained)
    2. NORMAL: schemas with the most new available samples
    3. LOW: schemas triggered by time_elapsed (already have working adapter)
    """
    schemas_with_adapter: set[str] = set()
    if routing_rules:
        for rule in routing_rules:
            if rule.get("active_model_id"):
                schemas_with_adapter.add(rule.get("schema_id", ""))

    def sort_key(b: dict) -> tuple[int, int]:
        schema_id = b.get("schema_id", "")
        sample_count = b.get("total_samples", b.get("train_samples", 0))
        trigger_reason = b.get("trigger_reason", "")

        if schema_id not in schemas_with_adapter:
            priority = 0  # HIGH: no adapter yet
        elif trigger_reason == "time_elapsed":
            priority = 2  # LOW: time-based trigger
        else:
            priority = 1  # NORMAL: new samples available

        return (priority, -sample_count)

    return sorted(batches, key=sort_key)


async def _submit_and_pipeline(
    client: httpx.AsyncClient,
    batch_info: dict,
    min_samples: int,
    dry_run: bool,
    use_pregenerated: bool,
) -> SchemaResult:
    """Run the full pipeline for a single schema with error isolation."""
    schema_id = batch_info.get("schema_id", "unknown")
    try:
        if use_pregenerated and batch_info.get("minio_path"):
            success = await run_flywheel(
                client,
                schema_id,
                batch_id=batch_info["id"],
                minio_path=batch_info["minio_path"],
                min_samples=min_samples,
                dry_run=dry_run,
            )
        else:
            success = await run_flywheel(
                client,
                schema_id,
                min_samples=min_samples,
                dry_run=dry_run,
            )
        return SchemaResult(schema_id=schema_id, success=success)
    except RuntimeError as e:
        if _is_insufficient_samples(str(e)):
            log(
                f"SKIP {schema_id}: insufficient samples (not a failure)",
                level="info",
                schema_id=schema_id,
                reason="insufficient_samples",
            )
            return SchemaResult(schema_id=schema_id, success=True)
        log(f"ERROR for {schema_id}: {e}", level="error", schema_id=schema_id, error=str(e))
        return SchemaResult(schema_id=schema_id, success=False, error=str(e))
    except (httpx.HTTPError, OSError, ValueError, KeyError) as e:
        log(f"ERROR for {schema_id}: {e}", level="error", schema_id=schema_id, error=str(e))
        return SchemaResult(schema_id=schema_id, success=False, error=str(e))


async def run_all(client: httpx.AsyncClient, min_samples: int, dry_run: bool) -> dict[str, bool]:
    """Run flywheel for all schemas that need retraining.

    Processes schemas in parallel: submits all training jobs concurrently
    (trainer queues them internally), then polls all jobs concurrently.
    Each schema's pipeline is isolated — one failure doesn't block others.
    """
    batches: list[dict] = []
    use_pregenerated = False

    try:
        # check-and-generate: returns list of batch results directly
        resp = await client.post(
            f"{TDG_URL}/api/triggers/check-and-generate",
            json={"min_samples": min_samples},
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            # Completed batches -- use batch_id/minio_path directly
            batches = [b for b in data if b.get("status") == "completed" and b.get("minio_path")]
            use_pregenerated = True

            # Log failed/skipped batches so operators understand why schemas were excluded
            for b in data:
                if b.get("status") == "failed":
                    sid = b.get("schema_id", "unknown")
                    err = b.get("error_message", "unknown error")
                    if _is_insufficient_samples(err):
                        log(
                            f"SKIP {sid}: {err}",
                            level="info",
                            schema_id=sid,
                            reason="insufficient_samples",
                        )
                    else:
                        log(
                            f"Batch generation failed for {sid}: {err}",
                            level="warning",
                            schema_id=sid,
                            error=err,
                        )

            log(f"check-and-generate: {len(data)} results, {len(batches)} with completed batches")
        elif isinstance(data, dict) and "triggers" in data:
            triggers = data["triggers"]
            batches = [{"schema_id": t["schema_id"]} for t in triggers if t.get("should_retrain")]
        else:
            log(
                f"Unexpected response format from check-and-generate: {type(data)}", level="warning"
            )
    except (httpx.HTTPStatusError, httpx.RemoteProtocolError, httpx.ConnectError) as e:
        log(f"check-and-generate failed ({e}), falling back to trigger check", level="warning")
        resp = await client.get(f"{TDG_URL}/api/triggers/check", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        triggers = data.get("triggers", data) if isinstance(data, dict) else data
        batches = [{"schema_id": t["schema_id"]} for t in triggers if t.get("should_retrain")]

    if not batches:
        log("No schemas need retraining")
        return {}

    # Fetch routing rules for prioritization
    routing_rules: list[dict] | None = None
    try:
        rules_resp = await client.get(f"{REGISTRY_URL}/api/routing", timeout=15)
        rules_resp.raise_for_status()
        routing_rules = rules_resp.json()
    except (httpx.HTTPError, OSError) as e:
        log(f"Could not fetch routing rules for prioritization: {e}", level="warning")

    # Prioritize and cap at MAX_SCHEMAS_PER_RUN
    batches = prioritize(batches, routing_rules)[:MAX_SCHEMAS_PER_RUN]

    schemas = [b.get("schema_id", "?") for b in batches]
    log(
        f"Processing {len(schemas)} schemas (max {MAX_SCHEMAS_PER_RUN}): {', '.join(schemas)}",
        count=len(schemas),
    )

    # Run all schemas concurrently with error isolation
    schema_results = await asyncio.gather(
        *[
            _submit_and_pipeline(client, batch_info, min_samples, dry_run, use_pregenerated)
            for batch_info in batches
        ],
        return_exceptions=True,
    )

    results: dict[str, bool] = {}
    for i, sr in enumerate(schema_results):
        if isinstance(sr, BaseException):
            schema_id = batches[i].get("schema_id", "unknown")
            log(f"ERROR for {schema_id}: {sr}", level="error", schema_id=schema_id)
            results[schema_id] = False
        else:
            results[sr.schema_id] = sr.success

    log(f"\n{'=' * 60}")
    log("RESULTS")
    log(f"{'=' * 60}")
    succeeded = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    for schema, success in results.items():
        log(f"  {schema}: {'OK' if success else 'FAILED'}", schema_id=schema, success=success)
    log(f"Total: {succeeded} succeeded, {failed} failed out of {len(results)}")
    return results


def _get_minio_client() -> Minio:
    """Create a MinIO client from environment variables."""
    if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        raise RuntimeError(
            "MinIO credentials not configured. Set MODEL_REGISTRY_MINIO_ACCESS_KEY "
            "and MODEL_REGISTRY_MINIO_SECRET_KEY environment variables."
        )
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


async def _get_active_adapter_paths(client: httpx.AsyncClient) -> set[str]:
    """Fetch all currently active adapter paths from routing rules.

    These paths must never be deleted during pruning.
    """
    active_paths: set[str] = set()
    try:
        resp = await client.get(f"{REGISTRY_URL}/api/routing", timeout=15)
        resp.raise_for_status()
        rules = resp.json()
        for rule in rules:
            path = rule.get("adapter_path")
            if path:
                active_paths.add(path)
    except (httpx.HTTPError, OSError, KeyError) as e:
        log(f"Could not fetch active adapter paths: {e}", level="warning")
    return active_paths


def _discover_versions(
    minio_client: Minio, bucket: str, schema_prefix: str
) -> list[tuple[str, datetime | None]]:
    """List version directories under a schema prefix, sorted newest-first.

    Returns list of (version_id, last_modified) tuples.
    MinIO objects under adapters/{schema_id}/{version_id}/ — we extract
    unique version_id prefixes and use the newest object timestamp per version.
    """
    version_timestamps: dict[str, datetime | None] = {}

    objects = minio_client.list_objects(bucket, prefix=schema_prefix, recursive=True)
    for obj in objects:
        # Extract version_id from path: {schema_id}/{version_id}/...
        parts = obj.object_name[len(schema_prefix) :].split("/")
        if not parts or not parts[0]:
            continue
        version_id = parts[0]
        # Track the newest object timestamp per version
        existing = version_timestamps.get(version_id)
        if existing is None or (obj.last_modified and obj.last_modified > existing):
            version_timestamps[version_id] = obj.last_modified

    # Sort by timestamp descending (newest first)
    sorted_versions = sorted(
        version_timestamps.items(),
        key=lambda x: x[1] if x[1] else "",
        reverse=True,
    )
    return sorted_versions


def _delete_version(minio_client: Minio, bucket: str, prefix: str) -> int:
    """Delete all objects under a prefix. Returns count of deleted objects."""
    deleted = 0
    objects = minio_client.list_objects(bucket, prefix=prefix, recursive=True)
    for obj in objects:
        minio_client.remove_object(bucket, obj.object_name)
        deleted += 1
    return deleted


async def prune_adapter_versions(
    http_client: httpx.AsyncClient,
    dry_run: bool = False,
    keep: int = PRUNE_KEEP_VERSIONS,
) -> dict[str, int]:
    """Prune old adapter versions from MinIO, keeping the latest `keep` per schema.

    Protects any adapter path currently referenced by a routing rule.

    Returns dict mapping schema_id to number of versions deleted.
    """
    log("Pruning old adapter versions from MinIO...", bucket=ADAPTERS_BUCKET, keep=keep)

    minio_client = _get_minio_client()

    # Get active adapter paths so we never delete a deployed version
    active_paths = await _get_active_adapter_paths(http_client)
    if active_paths:
        log(f"Protected adapter paths (active): {active_paths}", count=len(active_paths))

    # Discover schema prefixes in adapters bucket
    schema_ids: set[str] = set()
    try:
        top_level = minio_client.list_objects(ADAPTERS_BUCKET, recursive=False)
        for obj in top_level:
            name = obj.object_name.rstrip("/")
            if name:
                schema_ids.add(name)
    except S3Error as e:
        log(f"Failed to list adapters bucket: {e}", level="error")
        return {}

    if not schema_ids:
        log("No schemas found in adapters bucket")
        return {}

    results: dict[str, int] = {}

    for schema_id in sorted(schema_ids):
        schema_prefix = f"{schema_id}/"
        versions = _discover_versions(minio_client, ADAPTERS_BUCKET, schema_prefix)

        if len(versions) <= keep:
            log(
                f"  {schema_id}: {len(versions)} version(s), nothing to prune",
                schema_id=schema_id,
                versions=len(versions),
            )
            continue

        to_keep = versions[:keep]
        to_delete = versions[keep:]

        # Filter out any version whose path is active
        safe_to_delete = []
        for version_id, ts in to_delete:
            version_prefix = f"{schema_id}/{version_id}/"
            # Check if any active path starts with this version prefix
            is_active = any(
                ap.startswith(version_prefix) or ap == f"{schema_id}/{version_id}"
                for ap in active_paths
            )
            if is_active:
                log(
                    f"  {schema_id}: SKIPPING {version_id} (active/deployed)",
                    level="warning",
                    schema_id=schema_id,
                    version_id=version_id,
                )
                continue
            safe_to_delete.append((version_id, ts))

        deleted_count = 0
        for version_id, ts in safe_to_delete:
            version_prefix = f"{schema_id}/{version_id}/"
            if dry_run:
                log(
                    f"  {schema_id}: would delete {version_id} (ts={ts})",
                    schema_id=schema_id,
                    version_id=version_id,
                )
                deleted_count += 1
            else:
                try:
                    obj_count = _delete_version(minio_client, ADAPTERS_BUCKET, version_prefix)
                    log(
                        f"  {schema_id}: deleted {version_id} ({obj_count} objects)",
                        schema_id=schema_id,
                        version_id=version_id,
                        objects_deleted=obj_count,
                    )
                    deleted_count += 1
                except S3Error as e:
                    log(
                        f"  {schema_id}: failed to delete {version_id}: {e}",
                        level="error",
                        schema_id=schema_id,
                        version_id=version_id,
                    )

        if deleted_count > 0:
            results[schema_id] = deleted_count
            log(
                f"  {schema_id}: pruned {deleted_count} version(s), kept {[v[0] for v in to_keep]}",
                schema_id=schema_id,
                pruned=deleted_count,
                kept=len(to_keep),
            )

    total = sum(results.values())
    log(f"Adapter pruning complete: {total} version(s) removed across {len(results)} schema(s)")
    return results


async def prune_old_models(dry_run: bool = False) -> int:
    """Prune old GGUF models from the models bucket.

    Keeps all directories in ``PROTECTED_MODEL_DIRS`` (base models that
    the vllm pod loads at startup). Returns number of objects deleted.
    """
    log(
        "Pruning old GGUF models from MinIO...",
        bucket=MODELS_BUCKET,
        protected=sorted(PROTECTED_MODEL_DIRS),
    )

    minio_client = _get_minio_client()

    deleted = 0
    try:
        top_level = minio_client.list_objects(MODELS_BUCKET, recursive=False)
        for obj in top_level:
            name = obj.object_name.rstrip("/")
            if not name:
                continue
            # Keep every protected base model
            if name in PROTECTED_MODEL_DIRS:
                log(f"  models: keeping protected base model {name}")
                continue
            # Delete everything else
            prefix = f"{name}/"
            if dry_run:
                log(f"  models: would delete {name}/", version_id=name)
                deleted += 1
            else:
                try:
                    obj_count = _delete_version(minio_client, MODELS_BUCKET, prefix)
                    # Also try deleting the object itself if it's a file, not a directory
                    try:
                        minio_client.remove_object(MODELS_BUCKET, obj.object_name)
                        obj_count += 1
                    except S3Error:
                        pass
                    log(
                        f"  models: deleted {name} ({obj_count} objects)",
                        version_id=name,
                        objects_deleted=obj_count,
                    )
                    deleted += 1
                except S3Error as e:
                    log(f"  models: failed to delete {name}: {e}", level="error")
    except S3Error as e:
        if e.code == "NoSuchBucket":
            log("Models bucket does not exist, nothing to prune", level="warning")
            return 0
        log(f"Failed to list models bucket: {e}", level="error")
        return 0

    log(f"Models pruning complete: {deleted} item(s) removed")
    return deleted


async def run_prune(
    http_client: httpx.AsyncClient,
    dry_run: bool = False,
    keep: int = PRUNE_KEEP_VERSIONS,
) -> bool:
    """Run all pruning tasks. Returns True if all succeeded."""
    log(f"\n{'=' * 60}")
    log("VERSION PRUNING")
    log(f"{'=' * 60}")

    success = True

    try:
        await prune_adapter_versions(http_client, dry_run=dry_run, keep=keep)
    except (httpx.HTTPError, OSError, S3Error, RuntimeError) as e:
        log(f"Adapter pruning failed: {e}", level="error", error=str(e))
        success = False

    try:
        await prune_old_models(dry_run=dry_run)
    except (OSError, S3Error, RuntimeError) as e:
        log(f"Models pruning failed: {e}", level="error", error=str(e))
        success = False

    return success


async def async_main(args: argparse.Namespace) -> int:
    global _cron_mode
    _cron_mode = args.cron
    _load_large_output_schemas()

    async with httpx.AsyncClient(timeout=30) as client:
        if args.status:
            await show_status(client)
            return 0

        if args.prune:
            ok = await run_prune(client, dry_run=args.dry_run, keep=args.keep_versions)
            return 0 if ok else 1

        if args.schema:
            success = await run_flywheel(
                client,
                args.schema,
                args.batch_id,
                args.minio_path,
                min_samples=args.min_samples,
                dry_run=args.dry_run,
            )
            return 0 if success else 1

        if getattr(args, "all"):
            results = await run_all(client, min_samples=args.min_samples, dry_run=args.dry_run)

            # Prune old versions after processing all schemas
            try:
                await run_prune(client, dry_run=args.dry_run, keep=args.keep_versions)
            except (httpx.HTTPError, OSError, S3Error, RuntimeError) as e:
                log(f"Post-run pruning failed (non-fatal): {e}", level="warning", error=str(e))

            if not results:
                return 0
            return 0 if all(results.values()) else 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-4B-AWQ Flywheel Orchestrator")
    parser.add_argument("--schema", help="Run flywheel for a specific schema")
    parser.add_argument(
        "--all", action="store_true", help="Run for all schemas with enough samples"
    )
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--batch-id", help="Use existing batch ID (skip generation)")
    parser.add_argument("--minio-path", help="MinIO path for existing batch")
    parser.add_argument(
        "--min-samples", type=int, default=20, help="Min samples for batch generation"
    )
    parser.add_argument("--cron", action="store_true", help="JSON-only logging (for CronJob)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without executing"
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Prune old adapter/model versions from MinIO (standalone mode)",
    )
    parser.add_argument(
        "--keep-versions",
        type=int,
        default=PRUNE_KEEP_VERSIONS,
        help=f"Number of adapter versions to keep per schema (default: {PRUNE_KEEP_VERSIONS})",
    )

    args = parser.parse_args()

    if not (args.status or args.schema or args.all or args.prune):
        parser.print_help()
        return

    sys.exit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    main()
