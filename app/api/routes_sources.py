from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_internal_api_key
from app.db.session import get_db
from app.schemas.source_schema import (
    SourceIngestRequestSchema,
    SourceIngestResponseSchema,
    SourceListResponseSchema,
    SourceResponseSchema,
)
from app.services.source_service import get_source, ingest_sources, list_sources
from app.utils.errors import AppError

router = APIRouter(prefix="/sources", tags=["sources"])


@router.post(
    "/ingest",
    response_model=SourceIngestResponseSchema,
    dependencies=[Depends(verify_internal_api_key)],
)
async def ingest_sources_endpoint(
    payload: SourceIngestRequestSchema,
    session: AsyncSession = Depends(get_db),
) -> SourceIngestResponseSchema:
    items, created_count, updated_count = await ingest_sources(session, payload.items)
    return SourceIngestResponseSchema(
        items=items,
        created_count=created_count,
        updated_count=updated_count,
    )


@router.get("", response_model=SourceListResponseSchema)
async def list_sources_endpoint(session: AsyncSession = Depends(get_db)) -> SourceListResponseSchema:
    items, total = await list_sources(session)
    return SourceListResponseSchema(items=items, total=total)


@router.get("/{source_id}", response_model=SourceResponseSchema)
async def get_source_endpoint(
    source_id: str,
    session: AsyncSession = Depends(get_db),
) -> SourceResponseSchema:
    source = await get_source(session, source_id)
    if source is None:
        raise AppError(status_code=404, code="SOURCE_NOT_FOUND", message="Source not found.")
    return source
