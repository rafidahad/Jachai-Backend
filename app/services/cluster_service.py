from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.claim import Claim
from app.models.rumor_cluster import RumorCluster
from app.schemas.cluster_schema import RumorClusterSchema
from app.schemas.dashboard_schema import RecentClaimItemSchema


def to_cluster_schema(cluster: RumorCluster) -> RumorClusterSchema:
    return RumorClusterSchema(
        id=cluster.id,
        topic_hash=cluster.topic_hash,
        title=cluster.title,
        language=cluster.language,
        summary=cluster.summary,
        claim_count=cluster.claim_count,
        representative_claim_id=cluster.representative_claim_id,
        created_at=cluster.created_at,
        updated_at=cluster.updated_at,
    )


async def get_or_create_cluster(
    session: AsyncSession,
    topic_hash: str,
    title: str,
    language: str,
    summary: str,
) -> RumorCluster:
    cluster = await session.scalar(select(RumorCluster).where(RumorCluster.topic_hash == topic_hash))
    if cluster:
        cluster.claim_count += 1
        cluster.title = title
        cluster.summary = summary
        cluster.language = language
        return cluster

    cluster = RumorCluster(
        topic_hash=topic_hash,
        title=title,
        language=language,
        summary=summary,
        claim_count=1,
    )
    session.add(cluster)
    await session.flush()
    return cluster


async def list_clusters(session: AsyncSession) -> tuple[list[RumorClusterSchema], int]:
    total = await session.scalar(select(func.count()).select_from(RumorCluster)) or 0
    clusters = (
        await session.execute(select(RumorCluster).order_by(RumorCluster.claim_count.desc(), RumorCluster.updated_at.desc()))
    ).scalars().all()
    return [to_cluster_schema(cluster) for cluster in clusters], int(total)


async def get_cluster_detail(
    session: AsyncSession,
    cluster_id: str,
) -> tuple[RumorClusterSchema | None, list[RecentClaimItemSchema]]:
    try:
        parsed_id = UUID(cluster_id)
    except ValueError:
        return None, []
    cluster = await session.get(RumorCluster, parsed_id)
    if not cluster:
        return None, []
    claims = (
        await session.execute(
            select(Claim).where(Claim.cluster_id == cluster.id).order_by(Claim.created_at.desc()).limit(10)
        )
    ).scalars().all()
    recent = [
        RecentClaimItemSchema(
            id=str(claim.id),
            verdict=claim.verdict,
            language=claim.language,
            review_status=claim.review_status,
            created_at=claim.created_at,
            share_summary=claim.share_summary,
        )
        for claim in claims
    ]
    return to_cluster_schema(cluster), recent
