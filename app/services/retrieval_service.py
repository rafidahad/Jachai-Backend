from __future__ import annotations

from sqlalchemy import Select, not_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.evidence_source import EvidenceSource
from app.services.evidence_index_service import sample_source_filter


async def retrieve_evidence(
    session: AsyncSession,
    embedding: list[float],
    top_k: int | None = None,
) -> list[tuple[EvidenceSource, float]]:
    limit = top_k or settings.pgvector_top_k
    distance = EvidenceSource.embedding.cosine_distance(embedding)
    statement: Select = (
        select(EvidenceSource, distance.label("distance"))
        .where(EvidenceSource.embedding.is_not(None))
        .where(not_(sample_source_filter()))
        .order_by(distance)
        .limit(limit)
    )
    rows = (await session.execute(statement)).all()
    return [(source, max(0.0, 1.0 - float(distance_value))) for source, distance_value in rows]
