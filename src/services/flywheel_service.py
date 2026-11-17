"""Flywheel service — launches the flywheel script as an async subprocess.

Avoids refactoring the standalone ``scripts/flywheel.py`` orchestrator by
shelling out to it with the appropriate CLI flags.  The caller gets back a
simple status dict while the heavy work happens in the child process.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from logging_config import get_logger

logger = get_logger(__name__)

# Resolve the project root so we can find scripts/flywheel.py regardless of
# the working directory uvicorn is started from.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FLYWHEEL_SCRIPT = _PROJECT_ROOT / "scripts" / "flywheel.py"

# Hard upper bound on how long a single flywheel subprocess may run.
# Matches the existing ``flywheel`` CronJob ``activeDeadlineSeconds: 7200`` /
# the trainer's 1h training timeout; we pick the tighter of the two (1h) so
# the HTTP-triggered path cannot out-live the scheduled path.
_FLYWHEEL_SUBPROCESS_TIMEOUT_SECONDS = 3600


async def run_flywheel_subprocess(
    *,
    schema_id: str,
    batch_id: str,
    minio_path: str,
    min_samples: int = 20,
) -> dict[str, str]:
    """Launch ``scripts/flywheel.py`` for a single schema in a subprocess.

    The function is fire-and-forget safe — it logs the outcome but never
    raises on flywheel failure (the caller already returned 202 to the
    HTTP client).

    Args:
        schema_id: Schema to run the pipeline for.
        batch_id: Pre-generated batch ID from TDG.
        minio_path: MinIO object path for the batch data.
        min_samples: Minimum samples threshold.

    Returns:
        A dict with ``status`` ("completed" | "failed") and ``detail``.
    """
    python = sys.executable
    cmd = [
        python,
        str(_FLYWHEEL_SCRIPT),
        "--schema",
        schema_id,
        "--batch-id",
        batch_id,
        "--minio-path",
        minio_path,
        "--min-samples",
        str(min_samples),
        "--cron",
    ]

    logger.info(
        "flywheel_subprocess_start",
        schema_id=schema_id,
        batch_id=batch_id,
        minio_path=minio_path,
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ},
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=_FLYWHEEL_SUBPROCESS_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            # The child is still running — terminate it before surfacing the
            # timeout so we don't leak a detached flywheel process. The raise
            # is caught by the outer handler which converts it to the
            # fire-and-forget ``{"status": "failed"}`` contract.
            proc.kill()
            await proc.wait()
            logger.error(
                "flywheel_subprocess_timeout",
                schema_id=schema_id,
                timeout_seconds=_FLYWHEEL_SUBPROCESS_TIMEOUT_SECONDS,
            )
            raise

        output = stdout.decode(errors="replace") if stdout else ""

        if proc.returncode == 0:
            logger.info(
                "flywheel_subprocess_completed",
                schema_id=schema_id,
                returncode=0,
            )
            return {"status": "completed", "detail": "Flywheel finished successfully"}

        # Non-zero exit — grab last 10 lines for diagnostics
        tail = "\n".join(output.splitlines()[-10:])
        logger.error(
            "flywheel_subprocess_failed",
            schema_id=schema_id,
            returncode=proc.returncode,
            tail=tail,
        )
        return {"status": "failed", "detail": f"Exit code {proc.returncode}: {tail}"}

    except TimeoutError:
        # Re-raised from the inner ``wait_for`` handler after killing the
        # child process. ``asyncio.TimeoutError`` is an alias for the
        # built-in ``TimeoutError`` on Python 3.11+.
        return {
            "status": "failed",
            "detail": (
                f"Flywheel subprocess exceeded "
                f"{_FLYWHEEL_SUBPROCESS_TIMEOUT_SECONDS}s timeout and was killed"
            ),
        }
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        # Narrowed from bare ``Exception`` to the documented subprocess
        # failure modes:
        # - ``subprocess.CalledProcessError`` — any check-based helper escape.
        # - ``FileNotFoundError`` — spawn failure (missing python / flywheel
        #   script).
        # Programming bugs, ``OSError`` subclasses other than
        # ``FileNotFoundError``, and ``asyncio.CancelledError`` deliberately
        # propagate.
        logger.error(
            "flywheel_subprocess_error",
            schema_id=schema_id,
            error=str(exc),
        )
        return {"status": "failed", "detail": str(exc)}
