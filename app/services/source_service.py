from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog
from app.models.evidence_source import EvidenceSource
from app.schemas.source_schema import SourceIngestItemSchema, SourceResponseSchema
from app.services.embedding_service import embed_text
from app.services.language_service import detect_language
from app.services.text_cleaning_service import clean_text


def to_source_response(source: EvidenceSource) -> SourceResponseSchema:
    return SourceResponseSchema(
        id=source.id,
        title=source.title,
        url=source.url,
        publisher=source.publisher,
        language=source.language,
        source_type=source.source_type,
        snippet=source.snippet,
        text_content=source.text_content,
        metadata=source.source_meta,
        embedding_available=source.embedding is not None,
        created_at=source.created_at,
        updated_at=source.updated_at,
    )


async def ingest_sources(
    session: AsyncSession,
    items: list[SourceIngestItemSchema],
) -> tuple[list[SourceResponseSchema], int, int]:
    created = 0
    updated = 0
    responses: list[SourceResponseSchema] = []

    for item in items:
        cleaned_snippet = clean_text(item.snippet)
        cleaned_text = clean_text(item.text_content)
        language = item.language or detect_language(cleaned_text)
        embedding = await embed_text(cleaned_text)

        existing = await session.scalar(select(EvidenceSource).where(EvidenceSource.url == str(item.url)))
        if existing:
            existing.title = item.title
            existing.publisher = item.publisher
            existing.language = language
            existing.source_type = item.source_type
            existing.snippet = cleaned_snippet
            existing.text_content = cleaned_text
            existing.embedding = embedding
            existing.source_meta = item.metadata
            source = existing
            updated += 1
        else:
            source = EvidenceSource(
                title=item.title,
                url=str(item.url),
                publisher=item.publisher,
                language=language,
                source_type=item.source_type,
                snippet=cleaned_snippet,
                text_content=cleaned_text,
                embedding=embedding,
                source_meta=item.metadata,
            )
            session.add(source)
            created += 1
            await session.flush()

        session.add(
            AuditLog(
                actor="internal",
                action="source_ingested",
                entity_type="evidence_source",
                entity_id=str(source.id),
                details={"title": item.title, "language": language, "url": str(item.url)},
            )
        )

    await session.commit()

    refreshed = await session.execute(
        select(EvidenceSource).where(EvidenceSource.url.in_([str(item.url) for item in items]))
    )
    for source in refreshed.scalars().all():
        responses.append(to_source_response(source))

    return responses, created, updated


async def list_sources(session: AsyncSession) -> tuple[list[SourceResponseSchema], int]:
    total = await session.scalar(select(func.count()).select_from(EvidenceSource)) or 0
    sources = (await session.execute(select(EvidenceSource).order_by(EvidenceSource.created_at.desc()))).scalars().all()
    return [to_source_response(source) for source in sources], int(total)


async def get_source(session: AsyncSession, source_id: str) -> SourceResponseSchema | None:
    try:
        parsed_id = UUID(source_id)
    except ValueError:
        return None
    source = await session.get(EvidenceSource, parsed_id)
    return to_source_response(source) if source else None
