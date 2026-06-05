from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, not_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evidence_source import EvidenceSource

TEST_SEEDED_BY_MARKERS = {"postman", "sample", "test", "demo", "fixture"}


@dataclass(slots=True)
class EvidenceIndexStatus:
    total_sources: int
    real_sources: int
    sample_sources: int
    ready: bool
    message: str


def _sample_source_filter():
    seeded_by = func.lower(func.coalesce(EvidenceSource.source_meta["seeded_by"].astext, ""))
    publisher = func.lower(func.coalesce(EvidenceSource.publisher, ""))
    title = func.lower(func.coalesce(EvidenceSource.title, ""))
    snippet = func.lower(func.coalesce(EvidenceSource.snippet, ""))
    return or_(
        seeded_by.in_(tuple(TEST_SEEDED_BY_MARKERS)),
        publisher.like("%sample source%"),
        title.like("%sample%"),
        snippet.like("%used for postman verification flow tests%"),
    )


async def get_evidence_index_status(session: AsyncSession) -> EvidenceIndexStatus:
    total_sources = int(await session.scalar(select(func.count()).select_from(EvidenceSource)) or 0)
    sample_filter = _sample_source_filter()
    sample_sources = int(
        await session.scalar(select(func.count()).select_from(EvidenceSource).where(sample_filter)) or 0
    )
    real_sources = int(
        await session.scalar(select(func.count()).select_from(EvidenceSource).where(not_(sample_filter))) or 0
    )

    if total_sources == 0:
        return EvidenceIndexStatus(
            total_sources=0,
            real_sources=0,
            sample_sources=0,
            ready=False,
            message="No evidence sources have been ingested yet.",
        )

    if real_sources == 0:
        return EvidenceIndexStatus(
            total_sources=total_sources,
            real_sources=0,
            sample_sources=sample_sources,
            ready=False,
            message="The evidence index currently contains only sample/test sources.",
        )

    return EvidenceIndexStatus(
        total_sources=total_sources,
        real_sources=real_sources,
        sample_sources=sample_sources,
        ready=True,
        message=f"{real_sources} real evidence source(s) are available for retrieval.",
    )
