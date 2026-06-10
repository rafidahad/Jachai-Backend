from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.models.claim import Claim
from app.models.rumor_cluster import RumorCluster
from app.services.cluster_service import get_or_create_cluster


class _ExecuteResult:
    def __init__(self, rows: list[object]):
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


def _make_cluster(*, topic_hash: str, title: str, language: str = "English") -> RumorCluster:
    return RumorCluster(
        id=uuid4(),
        topic_hash=topic_hash,
        title=title,
        language=language,
        summary=title,
        claim_count=1,
        representative_claim_id=uuid4(),
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_claim(*, semantic_text: str, semantic_embedding: list[float] | None) -> Claim:
    context_payload = {
        "extracted_claim": semantic_text,
        "semantic_cluster_text": semantic_text,
    }
    if semantic_embedding is not None:
        context_payload["semantic_cluster_embedding"] = semantic_embedding
    return Claim(
        id=uuid4(),
        input_type="text",
        raw_text=semantic_text,
        cleaned_text=semantic_text,
        masked_text=semantic_text,
        normalized_hash=uuid4().hex,
        language="English",
        verdict="Not Enough Evidence",
        confidence=0.2,
        explanation="test",
        reasoning="test",
        share_summary="test",
        review_status="pending",
        context_payload=context_payload,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_get_or_create_cluster_reuses_exact_topic_hash() -> None:
    existing_cluster = _make_cluster(
        topic_hash="same-hash",
        title="Original cluster title",
    )
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=existing_cluster)

    cluster = await get_or_create_cluster(
        session,
        topic_hash="same-hash",
        title="Updated title",
        language="English",
        summary="Updated summary",
        semantic_text="Updated semantic text",
        semantic_embedding=[1.0, 0.0],
    )

    assert cluster is existing_cluster
    assert existing_cluster.claim_count == 2
    assert existing_cluster.title == "Updated title"
    assert existing_cluster.summary == "Updated summary"


@pytest.mark.asyncio
async def test_get_or_create_cluster_reuses_semantic_match() -> None:
    existing_cluster = _make_cluster(
        topic_hash="first-claim-hash",
        title="Buffalo named Donald Trump is in Dhaka Zoo",
    )
    representative_claim = _make_claim(
        semantic_text="Buffalo named Donald Trump is in Dhaka Zoo",
        semantic_embedding=[1.0, 0.0],
    )
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=None)
    session.execute = AsyncMock(
        return_value=_ExecuteResult([(existing_cluster, representative_claim)])
    )

    cluster = await get_or_create_cluster(
        session,
        topic_hash="second-claim-hash",
        title="Dhaka Zoo now has a buffalo called Donald Trump",
        language="English",
        summary="Same rumor, slightly different wording.",
        semantic_text="Dhaka Zoo now has a buffalo called Donald Trump",
        semantic_embedding=[1.0, 0.0],
    )

    assert cluster is existing_cluster
    assert existing_cluster.claim_count == 2
    assert existing_cluster.title == "Dhaka Zoo now has a buffalo called Donald Trump"


@pytest.mark.asyncio
async def test_get_or_create_cluster_backfills_missing_semantic_embedding() -> None:
    existing_cluster = _make_cluster(
        topic_hash="first-claim-hash",
        title="Buffalo named Donald Trump is in Dhaka Zoo",
    )
    representative_claim = _make_claim(
        semantic_text="Buffalo named Donald Trump is in Dhaka Zoo",
        semantic_embedding=None,
    )
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=None)
    session.execute = AsyncMock(
        return_value=_ExecuteResult([(existing_cluster, representative_claim)])
    )

    with patch(
        "app.services.cluster_service.embed_texts",
        new_callable=AsyncMock,
        return_value=[[1.0, 0.0]],
    ) as mock_embed:
        cluster = await get_or_create_cluster(
            session,
            topic_hash="second-claim-hash",
            title="Dhaka Zoo now has a buffalo called Donald Trump",
            language="English",
            summary="Same rumor, slightly different wording.",
            semantic_text="Dhaka Zoo now has a buffalo called Donald Trump",
            semantic_embedding=[1.0, 0.0],
        )

    assert cluster is existing_cluster
    mock_embed.assert_awaited_once()
    assert representative_claim.context_payload["semantic_cluster_embedding"] == [1.0, 0.0]


@pytest.mark.asyncio
async def test_get_or_create_cluster_creates_new_cluster_for_low_similarity() -> None:
    existing_cluster = _make_cluster(
        topic_hash="first-claim-hash",
        title="Buffalo named Donald Trump is in Dhaka Zoo",
    )
    representative_claim = _make_claim(
        semantic_text="Buffalo named Donald Trump is in Dhaka Zoo",
        semantic_embedding=[1.0, 0.0],
    )
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=None)
    session.execute = AsyncMock(
        return_value=_ExecuteResult([(existing_cluster, representative_claim)])
    )
    session.add = MagicMock()

    cluster = await get_or_create_cluster(
        session,
        topic_hash="brand-new-hash",
        title="NASA confirms asteroid will pass Earth safely this week",
        language="English",
        summary="Completely different rumor.",
        semantic_text="NASA confirms asteroid will pass Earth safely this week",
        semantic_embedding=[0.0, 1.0],
    )

    assert cluster is not existing_cluster
    assert cluster.topic_hash == "brand-new-hash"
    session.add.assert_called_once()
