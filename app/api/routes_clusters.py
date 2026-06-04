from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.cluster_schema import RumorClusterDetailResponseSchema, RumorClusterListResponseSchema
from app.services.cluster_service import get_cluster_detail, list_clusters
from app.utils.errors import AppError

router = APIRouter(prefix="/rumor-clusters", tags=["rumor-clusters"])


@router.get("", response_model=RumorClusterListResponseSchema)
async def list_rumor_clusters(session: AsyncSession = Depends(get_db)) -> RumorClusterListResponseSchema:
    items, total = await list_clusters(session)
    return RumorClusterListResponseSchema(items=items, total=total)


@router.get("/{cluster_id}", response_model=RumorClusterDetailResponseSchema)
async def get_rumor_cluster(
    cluster_id: str,
    session: AsyncSession = Depends(get_db),
) -> RumorClusterDetailResponseSchema:
    cluster, recent_claims = await get_cluster_detail(session, cluster_id)
    if cluster is None:
        raise AppError(status_code=404, code="CLUSTER_NOT_FOUND", message="Rumor cluster not found.")
    return RumorClusterDetailResponseSchema(cluster=cluster, recent_claims=recent_claims)
