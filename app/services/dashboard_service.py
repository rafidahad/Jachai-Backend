from __future__ import annotations

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.claim import Claim
from app.models.evidence_source import EvidenceSource
from app.models.rumor_cluster import RumorCluster
from app.models.verification_job import VerificationJob
from app.schemas.dashboard_schema import (
    ClaimsOverTimeItemSchema,
    DistributionItemSchema,
    RecentClaimItemSchema,
    SummaryMetricSchema,
)


async def get_summary_metrics(session: AsyncSession) -> SummaryMetricSchema:
    total_claims = await session.scalar(select(func.count()).select_from(Claim)) or 0
    verification_runs = await session.scalar(select(func.count()).select_from(VerificationJob)) or 0
    claims_today = await session.scalar(
        select(func.count()).select_from(Claim).where(func.date(Claim.created_at) == func.current_date())
    ) or 0
    reviewed_claims = await session.scalar(
        select(func.count()).select_from(Claim).where(Claim.review_status == "reviewed")
    ) or 0
    total_sources = await session.scalar(select(func.count()).select_from(EvidenceSource)) or 0
    total_clusters = await session.scalar(select(func.count()).select_from(RumorCluster)) or 0
    average_confidence = await session.scalar(select(func.avg(Claim.confidence))) or 0
    top_language_row = (
        await session.execute(
            select(Claim.language, func.count())
            .group_by(Claim.language)
            .order_by(func.count().desc(), Claim.language.asc())
            .limit(1)
        )
    ).first()
    ocr_submissions = await session.scalar(
        select(func.count()).select_from(Claim).where(Claim.input_type == "image")
    ) or 0
    return SummaryMetricSchema(
        total_claims=int(total_claims),
        verification_runs=int(verification_runs),
        claims_today=int(claims_today),
        reviewed_claims=int(reviewed_claims),
        total_sources=int(total_sources),
        total_clusters=int(total_clusters),
        average_confidence=round(float(average_confidence) * 100, 1),
        top_language=str(top_language_row[0]) if top_language_row else "Unknown",
        ocr_submissions=int(ocr_submissions),
    )


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
