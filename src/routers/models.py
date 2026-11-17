"""Model CRUD API endpoints.

Provides endpoints for registering, listing, retrieving, and updating
the lifecycle status of model versions.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from models.domain import (
    DeployFeedbackRequest,
    ErrorResponse,
    ModelComparison,
    ModelListResponse,
    ModelStatus,
    ModelVersion,
    QualityResponse,
    RegisterModelRequest,
    SetTagsRequest,
    UpdateStatusRequest,
)
from services.model_service import ModelService

router = APIRouter(prefix="/api/models", tags=["models"])

_model_service = ModelService()


def get_model_service() -> ModelService:
    """Return the singleton ModelService instance.

    Returns:
        The shared ModelService instance.
    """
    return _model_service


def set_model_service(service: ModelService) -> None:
    """Replace the module-level ModelService (used by main.py lifespan)."""
    global _model_service
    _model_service = service


@router.post(
    "",
    response_model=ModelVersion,
    status_code=201,
    responses={409: {"model": ErrorResponse}},
    summary="Register a new model version",
)
async def register_model(
    request: RegisterModelRequest,
    session: AsyncSession = Depends(get_session),
    model_service: ModelService = Depends(get_model_service),
) -> ModelVersion:
    """Register a new model version in the catalog.

    Args:
        request: Model registration details.
        session: Injected database session.

    Returns:
        The newly created ModelVersion.
    """
    try:
        return await model_service.register_model(session, request)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.get(
    "",
    response_model=ModelListResponse,
    summary="List model versions with filtering",
)
async def list_models(
    schema_id: str | None = Query(default=None, description="Filter by schema identifier"),
    status: ModelStatus | None = Query(default=None, description="Filter by lifecycle status"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    session: AsyncSession = Depends(get_session),
    model_service: ModelService = Depends(get_model_service),
) -> ModelListResponse:
    """List model versions, optionally filtered by schema_id and/or status.

    Args:
        schema_id: Optional schema identifier filter.
        status: Optional lifecycle status filter.
        limit: Maximum number of results to return.
        offset: Number of results to skip.
        session: Injected database session.

    Returns:
        A ModelListResponse containing the matching models and total count.
    """
    models, total = await model_service.list_models(
        session, schema_id=schema_id, status=status, limit=limit, offset=offset
    )
    return ModelListResponse(models=models, total=total)


@router.get(
    "/{schema_id}/quality",
    response_model=QualityResponse,
    summary="Get latest quality score for a schema",
)
async def get_quality(
    schema_id: str,
    session: AsyncSession = Depends(get_session),
    model_service: ModelService = Depends(get_model_service),
) -> QualityResponse:
    """Return the latest quality score for the given schema_id.

    Looks up the most recent model version with a non-null eval_score.
    If no evaluated model exists, returns a response with null quality fields.

    Args:
        schema_id: The schema identifier to query.
        session: Injected database session.

    Returns:
        A QualityResponse with the quality info.
    """
    return await model_service.get_latest_quality(session, schema_id)


@router.get(
    "/{model_id}",
    response_model=ModelVersion,
    responses={404: {"model": ErrorResponse}},
    summary="Get model version details",
)
async def get_model(
    model_id: UUID,
    session: AsyncSession = Depends(get_session),
    model_service: ModelService = Depends(get_model_service),
) -> ModelVersion:
    """Retrieve a specific model version by its UUID.

    Args:
        model_id: The UUID of the model to retrieve.
        session: Injected database session.

    Returns:
        The requested ModelVersion.
    """
    model = await model_service.get_model(session, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return model


@router.patch(
    "/{model_id}/status",
    response_model=ModelVersion,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    summary="Update model lifecycle status",
)
async def update_status(
    model_id: UUID,
    request: UpdateStatusRequest,
    session: AsyncSession = Depends(get_session),
    model_service: ModelService = Depends(get_model_service),
) -> ModelVersion:
    """Update the lifecycle status of a model version.

    Validates that the transition is allowed before applying.
    Valid transitions follow the lifecycle: training -> evaluating ->
    evaluated -> approved -> exporting -> deployed -> retired.
    Any state can also transition to 'retired' for cleanup.

    Args:
        model_id: The UUID of the model to update.
        request: The status update request.
        session: Injected database session.

    Returns:
        The updated ModelVersion.
    """
    try:
        return await model_service.update_status(session, model_id, request)
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg) from e
        raise HTTPException(status_code=400, detail=error_msg) from e


# ---------- Tag endpoints ----------


@router.put(
    "/{model_id}/tags",
    response_model=dict[str, str],
    responses={404: {"model": ErrorResponse}},
    summary="Set tags on a model version",
)
async def set_tags(
    model_id: UUID,
    request: SetTagsRequest,
    session: AsyncSession = Depends(get_session),
    model_service: ModelService = Depends(get_model_service),
) -> dict[str, str]:
    """Set or update tags on a model version.

    Args:
        model_id: UUID of the model version.
        request: The tag set request.
        session: Injected database session.
        model_service: Injected model service.

    Returns:
        All current tags for the model.
    """
    try:
        return await model_service.set_tags(session, model_id, request.tags)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get(
    "/{model_id}/tags",
    response_model=dict[str, str],
    summary="Get all tags for a model version",
)
async def get_tags(
    model_id: UUID,
    session: AsyncSession = Depends(get_session),
    model_service: ModelService = Depends(get_model_service),
) -> dict[str, str]:
    """Get all tags for a model version.

    Args:
        model_id: UUID of the model version.
        session: Injected database session.
        model_service: Injected model service.

    Returns:
        Dict of key-value tag pairs.
    """
    return await model_service.get_tags(session, model_id)


@router.delete(
    "/{model_id}/tags/{key}",
    status_code=204,
    responses={404: {"model": ErrorResponse}},
    summary="Delete a tag from a model version",
)
async def delete_tag(
    model_id: UUID,
    key: str,
    session: AsyncSession = Depends(get_session),
    model_service: ModelService = Depends(get_model_service),
) -> None:
    """Delete a specific tag from a model version.

    Args:
        model_id: UUID of the model version.
        key: Tag key to delete.
        session: Injected database session.
        model_service: Injected model service.
    """
    deleted = await model_service.delete_tag(session, model_id, key)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Tag not found: {key}")


# ---------- Tag search endpoint ----------

tags_router = APIRouter(prefix="/api/tags", tags=["tags"])


@tags_router.get(
    "/search",
    response_model=list[ModelVersion],
    summary="Search models by tag",
)
async def search_by_tag(
    key: str = Query(description="Tag key to search for"),
    value: str | None = Query(default=None, description="Optional tag value to match"),
    session: AsyncSession = Depends(get_session),
    model_service: ModelService = Depends(get_model_service),
) -> list[ModelVersion]:
    """Find model versions by tag key and optionally value.

    Args:
        key: Tag key to search for.
        value: Optional tag value to match exactly.
        session: Injected database session.
        model_service: Injected model service.

    Returns:
        List of matching ModelVersion objects.
    """
    return await model_service.search_by_tag(session, key, value)


# ---------- Comparison endpoint ----------


@router.get(
    "/{schema_id}/compare",
    response_model=ModelComparison,
    summary="Compare model versions for a schema",
)
async def compare_versions(
    schema_id: str,
    versions: str | None = Query(
        default=None, description="Comma-separated version labels to compare"
    ),
    session: AsyncSession = Depends(get_session),
    model_service: ModelService = Depends(get_model_service),
) -> ModelComparison:
    """Compare model versions for a schema side by side.

    Includes eval_score, eval_metrics, aliases, tags, and status for each version.

    Args:
        schema_id: The schema identifier.
        versions: Optional comma-separated version labels. If not provided, compares latest 10.
        session: Injected database session.
        model_service: Injected model service.

    Returns:
        A ModelComparison with version summaries.
    """
    version_list = [v.strip() for v in versions.split(",")] if versions else None
    summaries = await model_service.compare_versions(session, schema_id, version_list)
    return ModelComparison(schema_id=schema_id, versions=summaries)


# ---------- Deployment tracking endpoint ----------


@router.patch(
    "/{model_id}/deploy-status",
    response_model=ModelVersion,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    summary="Update deployment status (external feedback)",
)
async def update_deploy_status(
    model_id: UUID,
    request: DeployFeedbackRequest,
    session: AsyncSession = Depends(get_session),
    model_service: ModelService = Depends(get_model_service),
) -> ModelVersion:
    """Receive deployment status feedback from external systems (e.g. K8s controllers).

    Args:
        model_id: UUID of the model version.
        request: Deployment feedback.
        session: Injected database session.
        model_service: Injected model service.

    Returns:
        The updated ModelVersion.
    """
    try:
        return await model_service.update_deploy_status(session, model_id, request)
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg) from e
        raise HTTPException(status_code=400, detail=error_msg) from e
