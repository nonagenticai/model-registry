"""Deployment trigger API endpoints.

Provides the endpoint to trigger a GitOps deployment of an approved model.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from models.domain import DeployRequest, DeployResponse, ErrorResponse
from services.alias_service import AliasService
from services.deploy_service import DeployService
from services.errors import DeployError
from services.model_service import ModelService
from services.routing_service import RoutingService

router = APIRouter(prefix="/api/deploy", tags=["deploy"])

# Shared service instances
_model_service = ModelService()
_routing_service: RoutingService | None = None
_alias_service: AliasService | None = None


def _get_model_service() -> ModelService:
    """Return the module-level ModelService."""
    return _model_service


def _get_routing_service() -> RoutingService:
    """Import the shared routing service from the routing router.

    Returns:
        The singleton RoutingService instance.
    """
    if _routing_service is not None:
        return _routing_service
    from routers.routing import get_routing_service

    return get_routing_service()


def set_services(
    model_service: ModelService,
    routing_service: RoutingService,
    alias_service: AliasService | None = None,
) -> None:
    """Replace the module-level services (used by main.py lifespan)."""
    global _model_service, _routing_service, _alias_service
    _model_service = model_service
    _routing_service = routing_service
    _alias_service = alias_service


@router.post(
    "/{model_id}",
    response_model=DeployResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
    summary="Trigger deployment for an approved model",
)
async def deploy_model(
    model_id: UUID,
    request: DeployRequest = DeployRequest(),
    session: AsyncSession = Depends(get_session),
) -> DeployResponse:
    """Trigger the deployment pipeline for an approved model version.

    This endpoint:
    1. Validates the model is in 'approved' status
    2. Transitions through exporting -> deployed
    3. Generates a Helm values YAML for FluxCD and pushes it to gitops
    4. Updates the routing table
    5. Auto-manages champion/rollback aliases

    Returns 502 when the model could not actually be placed (values not pushed
    to gitops, PEFT adapter not loadable). The routing table is left untouched
    in that case — an endpoint that will never resolve is worse than none.

    Args:
        model_id: UUID of the model to deploy.
        request: Optional deployment parameters.
        session: Injected database session.

    Returns:
        A DeployResponse with deployment details.
    """
    routing_service = _get_routing_service()
    deploy_service = DeployService(
        model_service=_model_service,
        routing_service=routing_service,
        alias_service=_alias_service,
    )

    try:
        return await deploy_service.deploy_model(session, model_id, request)
    except DeployError as e:
        # The request was valid; the deployment itself did not happen. 502 —
        # never a 200 with a 'deployed' body the cluster cannot back up.
        raise HTTPException(status_code=502, detail=str(e)) from e
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg) from e
        raise HTTPException(status_code=400, detail=error_msg) from e
