from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.schemas.source_schema import SourceIngestItemSchema
from app.services.source_service import ingest_sources
from app.utils.hashing import normalized_hash


class _ScalarResult:
    def __init__(self, rows: list[object]):
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


class _ExecuteResult:
    def __init__(self, rows: list[object]):
        self._rows = rows

    def scalars(self) -> _ScalarResult:
        return _ScalarResult(self._rows)


def _make_source(
    *,
    url: str,
    title: str,
    text_content: str,
    embedding: list[float] | None,
    source_meta: dict[str, object],
) -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=uuid4(),
        title=title,
        url=url,
        publisher="Example News",
        language="English",
        source_type="tavily_search",
        snippet="Example snippet for tests.",
        text_content=text_content,
        embedding=embedding,
        source_meta=source_meta,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_ingest_sources_skips_reembedding_for_unchanged_content() -> None:
    text = "Government statement confirms the original claim."
    existing = _make_source(
        url="https://example.com/article",
        title="Original title",
        text_content=text,
        embedding=[0.25, 0.75],
        source_meta={"content_hash": normalized_hash(text)},
    )
    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[_ExecuteResult([existing]), _ExecuteResult([existing])]
    )
    session.add = MagicMock()

    item = SourceIngestItemSchema(
        title="Updated title",
        url="https://example.com/article",
        publisher="Example News",
        language="English",
        source_type="tavily_search",
        snippet="Example snippet for tests.",
        text_content=text,
        metadata={},
    )

    with patch("app.services.source_service.embed_texts", new_callable=AsyncMock) as mock_embed:
        items, created, updated = await ingest_sources(session, [item])

    mock_embed.assert_not_awaited()
    session.flush.assert_not_awaited()
    session.commit.assert_awaited_once()
    assert created == 0
    assert updated == 1
    assert items[0].embedding_available is True
    assert existing.title == "Updated title"


@pytest.mark.asyncio
async def test_ingest_sources_can_skip_embeddings_for_fast_live_evidence() -> None:
    text = "Tavily answer and source content used directly for live evidence."
    refreshed = _make_source(
        url="https://example.com/live",
        title="Live title",
        text_content=text,
        embedding=None,
        source_meta={"content_hash": normalized_hash(text), "embedding_deferred": True},
    )
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[_ExecuteResult([]), _ExecuteResult([refreshed])])
    session.add = MagicMock()

    item = SourceIngestItemSchema(
        title="Live title",
        url="https://example.com/live",
        publisher="Example News",
        language="English",
        source_type="tavily_search",
        snippet="Live source snippet for tests.",
        text_content=text,
        metadata={"embedding_deferred": True},
    )

    with patch("app.services.source_service.embed_texts", new_callable=AsyncMock) as mock_embed:
        responses, created, updated = await ingest_sources(
            session,
            [item],
            generate_embeddings=False,
        )

    mock_embed.assert_not_awaited()
    session.flush.assert_awaited_once()
    session.commit.assert_awaited_once()
    assert created == 1
    assert updated == 0
    assert responses[0].embedding_available is False


@pytest.mark.asyncio
async def test_ingest_sources_batches_embeddings_for_changed_and_new_sources() -> None:
    existing_text = "Old source text that has since changed."
    changed_text = "Updated source text with corrected details."
    new_text = "A brand new source with supporting details."
    existing = _make_source(
        url="https://example.com/existing",
        title="Old title",
        text_content=existing_text,
        embedding=[0.1, 0.9],
        source_meta={"content_hash": normalized_hash(existing_text)},
    )
    refreshed_existing = _make_source(
        url="https://example.com/existing",
        title="Updated title",
        text_content=changed_text,
        embedding=[0.3, 0.7],
        source_meta={"content_hash": normalized_hash(changed_text)},
    )
    refreshed_new = _make_source(
        url="https://example.com/new",
        title="New title",
        text_content=new_text,
        embedding=[0.8, 0.2],
        source_meta={"content_hash": normalized_hash(new_text)},
    )
    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[
            _ExecuteResult([existing]),
            _ExecuteResult([refreshed_existing, refreshed_new]),
        ]
    )
    session.add = MagicMock()

    items = [
        SourceIngestItemSchema(
            title="Updated title",
            url="https://example.com/existing",
            publisher="Example News",
            language="English",
            source_type="tavily_search",
            snippet="Changed existing source snippet.",
            text_content=changed_text,
            metadata={},
        ),
        SourceIngestItemSchema(
            title="New title",
            url="https://example.com/new",
            publisher="Example News",
            language="English",
            source_type="tavily_search",
            snippet="Brand new source snippet.",
            text_content=new_text,
            metadata={},
        ),
    ]

    with patch(
        "app.services.source_service.embed_texts",
        new_callable=AsyncMock,
        return_value=[[0.3, 0.7], [0.8, 0.2]],
    ) as mock_embed:
        responses, created, updated = await ingest_sources(session, items)

    mock_embed.assert_awaited_once()
    assert mock_embed.await_args.args[0] == [changed_text, new_text]
    assert mock_embed.await_args.kwargs["task_type"] == "RETRIEVAL_DOCUMENT"
    assert mock_embed.await_args.kwargs["titles"] == ["Updated title", "New title"]
    session.flush.assert_awaited_once()
    session.commit.assert_awaited_once()
    assert created == 1
    assert updated == 1
    assert [item.url for item in responses] == [
        "https://example.com/existing",
        "https://example.com/new",
    ]
