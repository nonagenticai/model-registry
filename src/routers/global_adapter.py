"""Global adapter API endpoints.

Single endpoint pair (GET/POST ``/api/global-adapter``) over the
singleton ``global_adapter`` table. Substrate for the single-LoRA
rollout; consumers (adapter-manager, inference-router) wire up in later
phases.
"""

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from models.domain import GlobalAdapter, UpsertGlobalAdapterRequest
from services.global_adapter_service import GlobalAdapterService

router = APIRouter(prefix="/api/global-adapter", tags=["global-adapter"])

_global_adapter_service = GlobalAdapterService()


def get_global_adapter_service() -> GlobalAdapterService:
    """Return the singleton GlobalAdapterService (overridable in tests)."""
    return _global_adapter_service


def set_global_adapter_service(service: GlobalAdapterService) -> None:
    """Replace the module-level GlobalAdapterService (used by main.py lifespan)."""
    global _global_adapter_service
    _global_adapter_service = service


@router.get(
    "",
    response_model=GlobalAdapter | None,
    responses={
        200: {"description": "Current global adapter"},
        204: {"description": "No global adapter set"},
    },
    summary="Get the current global adapter",
)
async def get_global_adapter(
    session: AsyncSession = Depends(get_session),
    service: GlobalAdapterService = Depends(get_global_adapter_service),
) -> Response | GlobalAdapter:
    """Return the current global adapter, or 204 if none set yet.

    Args:
        session: Injected database session.
        service: Injected global-adapter service.

    Returns:
        ``GlobalAdapter`` (200) or empty ``Response`` (204).
    """
    adapter = await service.get(session)
    if adapter is None:
        return Response(status_code=204)
    return adapter


@router.post(
    "",
    response_model=GlobalAdapter,
    summary="Set/update the global adapter",
)
async def upsert_global_adapter(
    request: UpsertGlobalAdapterRequest,
    session: AsyncSession = Depends(get_session),
    service: GlobalAdapterService = Depends(get_global_adapter_service),
) -> GlobalAdapter:
    """Set or update the singleton global adapter row.

    Args:
        request: Upsert payload.
        session: Injected database session.
        service: Injected global-adapter service.

    Returns:
        The current ``GlobalAdapter`` after the upsert.
    """
    return await service.upsert(session, request)
