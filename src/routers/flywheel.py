"""Flywheel trigger API endpoint.

Exposes an HTTP interface to kick off the train-eval-deploy pipeline for
a single schema that already has a pre-generated TDG batch.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from logging_config import get_logger
from services.flywheel_service import run_flywheel_subprocess

logger = get_logger(__name__)

router = APIRouter(tags=["flywheel"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class FlywheelRunRequest(BaseModel):
    """Payload to trigger the flywheel for a single schema."""

    schema_id: str = Field(..., description="Target schema identifier")
    batch_id: str = Field(..., description="Pre-generated TDG batch ID")
    minio_path: str = Field(..., description="MinIO object path for the batch data")
    min_samples: int = Field(default=20, ge=1, description="Minimum samples threshold")


class FlywheelRunResponse(BaseModel):
    """Acknowledgement returned immediately (before the pipeline runs)."""

    status: str
    schema_id: str
    message: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


async def _run_and_log(
    schema_id: str,
    batch_id: str,
    minio_path: str,
    min_samples: int,
) -> None:
    """Background wrapper that calls the subprocess and logs the result."""
    result = await run_flywheel_subprocess(
        schema_id=schema_id,
        batch_id=batch_id,
        minio_path=minio_path,
        min_samples=min_samples,
    )
    if result["status"] == "completed":
        logger.info("flywheel_run_done", schema_id=schema_id)
    else:
        logger.error(
            "flywheel_run_failed",
            schema_id=schema_id,
            detail=result.get("detail", ""),
        )


@router.post(
    "/api/flywheel/run",
    response_model=FlywheelRunResponse,
    status_code=202,
    summary="Trigger flywheel for a single schema",
)
async def run_flywheel_endpoint(
    request: FlywheelRunRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """Trigger the flywheel pipeline for a single schema with a pre-generated batch.

    Runs asynchronously in the background -- returns immediately with 202 Accepted.

    Args:
        request: Flywheel run parameters.
        background_tasks: FastAPI background task manager.

    Returns:
        A 202 Accepted response with a job reference.
    """
    logger.info(
        "flywheel_run_requested",
        schema_id=request.schema_id,
        batch_id=request.batch_id,
        minio_path=request.minio_path,
    )

    background_tasks.add_task(
        _run_and_log,
        schema_id=request.schema_id,
        batch_id=request.batch_id,
        minio_path=request.minio_path,
        min_samples=request.min_samples,
    )

    body = FlywheelRunResponse(
        status="accepted",
        schema_id=request.schema_id,
        message=f"Flywheel triggered for schema {request.schema_id} with batch {request.batch_id}",
    )
    return JSONResponse(status_code=202, content=body.model_dump())
