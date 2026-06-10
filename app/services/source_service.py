from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog
from app.models.evidence_source import EvidenceSource
from app.schemas.source_schema import SourceIngestItemSchema, SourceResponseSchema
from app.services.embedding_service import embed_texts
from app.services.language_service import detect_language
from app.services.text_cleaning_service import clean_text
from app.utils.hashing import normalized_hash


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
    urls = [str(item.url) for item in items]
    existing_rows = await session.execute(
        select(EvidenceSource).where(EvidenceSource.url.in_(urls))
    )
    existing_by_url = {source.url: source for source in existing_rows.scalars().all()}

    prepared_items: list[dict[str, object]] = []
    items_needing_embeddings: list[dict[str, object]] = []

    for item in items:
        url = str(item.url)
        cleaned_snippet = clean_text(item.snippet)
        cleaned_text = clean_text(item.text_content)
        language = item.language or detect_language(cleaned_text)
        metadata = dict(item.metadata)
        content_hash = str(metadata.get("content_hash") or normalized_hash(cleaned_text))
        metadata["content_hash"] = content_hash
        existing = existing_by_url.get(url)
        existing_hash = ""
        if existing and isinstance(existing.source_meta, dict):
            existing_hash = str(existing.source_meta.get("content_hash") or "").strip()

        prepared: dict[str, object] = {
            "item": item,
            "url": url,
            "cleaned_snippet": cleaned_snippet,
            "cleaned_text": cleaned_text,
            "language": language,
            "metadata": metadata,
            "existing": existing,
            "embedding": None,
        }
        if existing is None or existing.embedding is None or existing_hash != content_hash:
            items_needing_embeddings.append(prepared)
        prepared_items.append(prepared)

    if items_needing_embeddings:
        embeddings = await embed_texts(
            [str(prepared["cleaned_text"]) for prepared in items_needing_embeddings],
            task_type="RETRIEVAL_DOCUMENT",
            titles=[str(prepared["item"].title) for prepared in items_needing_embeddings],
        )
        for prepared, embedding in zip(items_needing_embeddings, embeddings, strict=True):
            prepared["embedding"] = embedding

    audit_sources: list[tuple[EvidenceSource, str, str, str]] = []
    new_sources: list[EvidenceSource] = []

    for prepared in prepared_items:
        item = prepared["item"]
        url = str(prepared["url"])
        cleaned_snippet = str(prepared["cleaned_snippet"])
        cleaned_text = str(prepared["cleaned_text"])
        language = str(prepared["language"])
        metadata = dict(prepared["metadata"])  # defensive copy for SQLAlchemy tracking
        existing = prepared["existing"]
        computed_embedding = prepared["embedding"]
        if existing:
            existing.title = item.title
            existing.publisher = item.publisher
            existing.language = language
            existing.source_type = item.source_type
            existing.snippet = cleaned_snippet
            existing.text_content = cleaned_text
            if computed_embedding is not None:
                existing.embedding = computed_embedding
            existing.source_meta = metadata
            source = existing
            updated += 1
        else:
            source = EvidenceSource(
                title=item.title,
                url=url,
                publisher=item.publisher,
                language=language,
                source_type=item.source_type,
                snippet=cleaned_snippet,
                text_content=cleaned_text,
                embedding=computed_embedding,
                source_meta=metadata,
            )
            session.add(source)
            new_sources.append(source)
            created += 1
        audit_sources.append((source, item.title, language, url))

    if new_sources:
        await session.flush()

    for source, title, language, url in audit_sources:
        session.add(
            AuditLog(
                actor="internal",
                action="source_ingested",
                entity_type="evidence_source",
                entity_id=str(source.id),
                details={"title": title, "language": language, "url": url},
            )
        )

    await session.commit()

    refreshed = await session.execute(
        select(EvidenceSource).where(EvidenceSource.url.in_(urls))
    )
    refreshed_by_url = {source.url: source for source in refreshed.scalars().all()}
    for url in urls:
        source = refreshed_by_url.get(url)
        if source is not None:
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
