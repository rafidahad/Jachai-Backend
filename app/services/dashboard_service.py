from __future__ import annotations

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.claim import Claim
from app.models.evidence_source import EvidenceSource
from app.models.rumor_cluster import RumorCluster
from app.models.verification_job import VerificationJob
from app.services.cache_service import cache_service
from app.schemas.dashboard_schema import (
    ClaimsOverTimeItemSchema,
    DistributionItemSchema,
    RecentClaimItemSchema,
    SummaryMetricSchema,
)


async def get_summary_metrics(session: AsyncSession) -> SummaryMetricSchema:
    cache_key = "dashboard:summary:public"
    cached_metrics = await cache_service.get_json_payload(cache_key)
    if cached_metrics:
        return SummaryMetricSchema.model_validate(cached_metrics)

    top_language_subquery = (
        select(Claim.language)
        .group_by(Claim.language)
        .order_by(func.count().desc(), Claim.language.asc())
        .limit(1)
        .scalar_subquery()
    )
    summary_query = select(
        select(func.count()).select_from(Claim).scalar_subquery().label("total_claims"),
        select(func.count()).select_from(VerificationJob).scalar_subquery().label("verification_runs"),
        (
            select(func.count())
            .select_from(Claim)
            .where(func.date(Claim.created_at) == func.current_date())
            .scalar_subquery()
        ).label("claims_today"),
        (
            select(func.count())
            .select_from(Claim)
            .where(Claim.review_status == "reviewed")
            .scalar_subquery()
        ).label("reviewed_claims"),
        select(func.count()).select_from(EvidenceSource).scalar_subquery().label("total_sources"),
        select(func.count()).select_from(RumorCluster).scalar_subquery().label("total_clusters"),
        select(func.avg(Claim.confidence)).select_from(Claim).scalar_subquery().label("average_confidence"),
        top_language_subquery.label("top_language"),
        (
            select(func.count())
            .select_from(Claim)
            .where(Claim.input_type == "image")
            .scalar_subquery()
        ).label("ocr_submissions"),
    )
    row = (await session.execute(summary_query)).one()
    metrics = SummaryMetricSchema(
        total_claims=int(row.total_claims or 0),
        verification_runs=int(row.verification_runs or 0),
        claims_today=int(row.claims_today or 0),
        reviewed_claims=int(row.reviewed_claims or 0),
        total_sources=int(row.total_sources or 0),
        total_clusters=int(row.total_clusters or 0),
        average_confidence=round(float(row.average_confidence or 0) * 100, 1),
        top_language=str(row.top_language or "Unknown"),
        ocr_submissions=int(row.ocr_submissions or 0),
    )
    await cache_service.set_json_payload(
        cache_key,
        metrics.model_dump(mode="json"),
        settings.dashboard_summary_cache_ttl_seconds,
    )
    return metrics


async def get_verdict_distribution(session: AsyncSession) -> list[DistributionItemSchema]:
    rows = (
        await session.execute(select(Claim.verdict, func.count()).group_by(Claim.verdict).order_by(func.count().desc()))
    ).all()
    return [DistributionItemSchema(label=str(label), count=int(count)) for label, count in rows]


async def get_language_distribution(session: AsyncSession) -> list[DistributionItemSchema]:
    rows = (
        await session.execute(
            select(Claim.language, func.count()).group_by(Claim.language).order_by(func.count().desc())
        )
    ).all()
    return [DistributionItemSchema(label=str(label), count=int(count)) for label, count in rows]


async def get_claims_over_time(session: AsyncSession) -> list[ClaimsOverTimeItemSchema]:
    rows = (
        await session.execute(
            select(func.date(Claim.created_at).label("day"), func.count())
            .group_by("day")
            .order_by("day")
        )
    ).all()
    return [ClaimsOverTimeItemSchema(date=day or date.today(), count=int(count)) for day, count in rows]


async def get_recent_claims(session: AsyncSession, limit: int = 10) -> list[RecentClaimItemSchema]:
    claims = (await session.execute(select(Claim).order_by(Claim.created_at.desc()).limit(limit))).scalars().all()
    return [
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
