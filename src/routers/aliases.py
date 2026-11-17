"""Model alias API endpoints.

Provides endpoints for managing named aliases (e.g. champion, rollback)
that point to specific model versions within a schema.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from models.domain import (
    AliasListResponse,
    ErrorResponse,
    ModelAlias,
    ModelVersion,
    SetAliasRequest,
)
from services.alias_service import AliasService

router = APIRouter(prefix="/api/aliases", tags=["aliases"])

_alias_service = AliasService()


def get_alias_service() -> AliasService:
    """Return the singleton AliasService instance.

    Returns:
        The shared AliasService instance.
    """
    return _alias_service


def set_alias_service(service: AliasService) -> None:
    """Replace the module-level AliasService (used by main.py lifespan)."""
    global _alias_service
    _alias_service = service


@router.put(
    "/{schema_id}/{alias_name}",
    response_model=ModelAlias,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    summary="Set or update a model alias",
)
async def set_alias(
    schema_id: str,
    alias_name: str,
    request: SetAliasRequest,
    session: AsyncSession = Depends(get_session),
    alias_service: AliasService = Depends(get_alias_service),
) -> ModelAlias:
    """Set or update a named alias for a model version.

    Args:
        schema_id: The schema identifier.
        alias_name: The alias name (e.g. 'champion', 'rollback').
        request: The alias request body with model_id.
        session: Injected database session.
        alias_service: Injected alias service.

    Returns:
        The created or updated ModelAlias.
    """
    try:
        return await alias_service.set_alias(session, schema_id, alias_name, request.model_id)
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg) from e
        raise HTTPException(status_code=400, detail=error_msg) from e


@router.get(
    "/{schema_id}",
    response_model=AliasListResponse,
    summary="List all aliases for a schema",
)
async def list_aliases(
    schema_id: str,
    session: AsyncSession = Depends(get_session),
    alias_service: AliasService = Depends(get_alias_service),
) -> AliasListResponse:
    """List all aliases for a schema.

    Args:
        schema_id: The schema identifier.
        session: Injected database session.
        alias_service: Injected alias service.

    Returns:
        An AliasListResponse containing all aliases.
    """
    aliases = await alias_service.list_aliases(session, schema_id)
    return AliasListResponse(aliases=aliases)


@router.get(
    "/{schema_id}/{alias_name}",
    response_model=ModelAlias,
    responses={404: {"model": ErrorResponse}},
    summary="Get alias details",
)
async def get_alias(
    schema_id: str,
    alias_name: str,
    session: AsyncSession = Depends(get_session),
    alias_service: AliasService = Depends(get_alias_service),
) -> ModelAlias:
    """Get details of a specific alias.

    Args:
        schema_id: The schema identifier.
        alias_name: The alias name.
        session: Injected database session.
        alias_service: Injected alias service.

    Returns:
        The ModelAlias details.
    """
    alias = await alias_service.get_alias(session, schema_id, alias_name)
    if alias is None:
        raise HTTPException(
            status_code=404,
            detail=f"Alias not found: {schema_id}:{alias_name}",
        )
    return alias


@router.delete(
    "/{schema_id}/{alias_name}",
    status_code=204,
    responses={404: {"model": ErrorResponse}},
    summary="Delete an alias",
)
async def delete_alias(
    schema_id: str,
    alias_name: str,
    session: AsyncSession = Depends(get_session),
    alias_service: AliasService = Depends(get_alias_service),
) -> None:
    """Delete a specific alias.

    Args:
        schema_id: The schema identifier.
        alias_name: The alias name.
        session: Injected database session.
        alias_service: Injected alias service.
    """
    deleted = await alias_service.delete_alias(session, schema_id, alias_name)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Alias not found: {schema_id}:{alias_name}",
        )


@router.get(
    "/{schema_id}/{alias_name}/resolve",
    response_model=ModelVersion,
    responses={404: {"model": ErrorResponse}},
    summary="Resolve alias to full ModelVersion",
)
async def resolve_alias(
    schema_id: str,
    alias_name: str,
    session: AsyncSession = Depends(get_session),
    alias_service: AliasService = Depends(get_alias_service),
) -> ModelVersion:
    """Resolve an alias to the full ModelVersion it points to.

    Args:
        schema_id: The schema identifier.
        alias_name: The alias name to resolve.
        session: Injected database session.
        alias_service: Injected alias service.

    Returns:
        The full ModelVersion the alias points to.
    """
    model = await alias_service.resolve_alias(session, schema_id, alias_name)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not resolve alias: {schema_id}:{alias_name}",
        )
    return model
