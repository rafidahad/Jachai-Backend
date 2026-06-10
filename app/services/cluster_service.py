from __future__ import annotations

import math
import re
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.claim import Claim
from app.models.rumor_cluster import RumorCluster
from app.schemas.cluster_schema import RumorClusterSchema
from app.schemas.dashboard_schema import RecentClaimItemSchema
from app.services.embedding_service import embed_texts
from app.services.text_cleaning_service import clean_text

SEMANTIC_CLUSTER_MAX_CANDIDATES = 200
SEMANTIC_CLUSTER_STRONG_MATCH = 0.92
SEMANTIC_CLUSTER_MIN_MATCH = 0.88
SEMANTIC_CLUSTER_MIN_LEXICAL_OVERLAP = 0.30
_CLUSTER_TOKEN_PATTERN = re.compile(r"[\w']+", re.UNICODE)
_CLUSTER_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by",
    "for", "from", "has", "have", "in", "into", "is", "it",
    "its", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with",
}


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


def _dot_similarity(left: list[float], right: list[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _cluster_tokens(value: str) -> set[str]:
    return {
        token
        for token in _CLUSTER_TOKEN_PATTERN.findall(value.lower())
        if len(token) >= 3 and token not in _CLUSTER_STOPWORDS
    }


def _lexical_overlap(left: str, right: str) -> float:
    left_tokens = _cluster_tokens(left)
    right_tokens = _cluster_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _cluster_claim_payload(claim: Claim | None) -> dict[str, object]:
    if claim is None or not isinstance(claim.context_payload, dict):
        return {}
    return claim.context_payload


def _cluster_semantic_text(cluster: RumorCluster, claim: Claim | None) -> str:
    payload = _cluster_claim_payload(claim)
    for key in ("semantic_cluster_text", "retrieval_match_query", "extracted_claim"):
        value = clean_text(str(payload.get(key) or ""))
        if value:
            return value
    title = clean_text(cluster.title)
    if title:
        return title
    return clean_text(cluster.summary)


def _cluster_semantic_embedding(claim: Claim | None) -> list[float] | None:
    payload = _cluster_claim_payload(claim)
    raw_values = payload.get("semantic_cluster_embedding")
    if not isinstance(raw_values, list) or not raw_values:
        return None
    values: list[float] = []
    for value in raw_values:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return None
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude <= 0:
        return None
    return [float(value / magnitude) for value in values]


def _upsert_claim_cluster_metadata(
    claim: Claim | None,
    *,
    semantic_text: str,
    semantic_embedding: list[float],
) -> None:
    if claim is None:
        return
    payload = dict(_cluster_claim_payload(claim))
    payload["semantic_cluster_text"] = semantic_text
    payload["semantic_cluster_embedding"] = [float(value) for value in semantic_embedding]
    claim.context_payload = payload


async def _find_semantic_cluster(
    session: AsyncSession,
    *,
    semantic_text: str,
    semantic_embedding: list[float],
    language: str,
) -> RumorCluster | None:
    statement = (
        select(RumorCluster, Claim)
        .outerjoin(Claim, Claim.id == RumorCluster.representative_claim_id)
        .order_by(RumorCluster.updated_at.desc(), RumorCluster.claim_count.desc())
        .limit(SEMANTIC_CLUSTER_MAX_CANDIDATES)
    )
    rows = (await session.execute(statement)).all()
    if not rows:
        return None

    candidate_payloads: list[dict[str, object]] = []
    texts_to_embed: list[str] = []
    candidate_indexes_needing_embeddings: list[int] = []

    for cluster, representative_claim in rows:
        candidate_text = _cluster_semantic_text(cluster, representative_claim)
        if len(candidate_text) < 5:
            continue
        lexical_overlap = _lexical_overlap(semantic_text, candidate_text)
        candidate_embedding = _cluster_semantic_embedding(representative_claim)
        candidate_payloads.append(
            {
                "cluster": cluster,
                "claim": representative_claim,
                "text": candidate_text,
                "embedding": candidate_embedding,
                "lexical_overlap": lexical_overlap,
                "language": clean_text(cluster.language),
            }
        )
        if candidate_embedding is None:
            candidate_indexes_needing_embeddings.append(len(candidate_payloads) - 1)
            texts_to_embed.append(candidate_text)

    if texts_to_embed:
        generated_embeddings = await embed_texts(
            texts_to_embed,
            task_type="RETRIEVAL_QUERY",
        )
        for candidate_index, generated_embedding in zip(
            candidate_indexes_needing_embeddings,
            generated_embeddings,
            strict=True,
        ):
            candidate = candidate_payloads[candidate_index]
            candidate["embedding"] = generated_embedding
            _upsert_claim_cluster_metadata(
                candidate["claim"],  # type: ignore[arg-type]
                semantic_text=str(candidate["text"]),
                semantic_embedding=generated_embedding,
            )

    best_cluster: RumorCluster | None = None
    best_score = -1.0
    normalized_language = clean_text(language)

    for candidate in candidate_payloads:
        candidate_embedding = candidate["embedding"]
        if not isinstance(candidate_embedding, list):
            continue
        semantic_score = _dot_similarity(semantic_embedding, candidate_embedding)
        lexical_overlap = float(candidate["lexical_overlap"])
        same_language = normalized_language and normalized_language == candidate["language"]
        if semantic_score < SEMANTIC_CLUSTER_MIN_MATCH:
            continue
        if (
            semantic_score < SEMANTIC_CLUSTER_STRONG_MATCH
            and lexical_overlap < SEMANTIC_CLUSTER_MIN_LEXICAL_OVERLAP
        ):
            continue
        ranking_score = (
            semantic_score
            + (0.02 if same_language else 0.0)
            + min(0.05, lexical_overlap * 0.05)
        )
        if ranking_score > best_score:
            best_score = ranking_score
            best_cluster = candidate["cluster"]  # type: ignore[assignment]

    return best_cluster


async def get_or_create_cluster(
    session: AsyncSession,
    topic_hash: str,
    title: str,
    language: str,
    summary: str,
    *,
    semantic_text: str | None = None,
    semantic_embedding: list[float] | None = None,
) -> RumorCluster:
    cluster = await session.scalar(
        select(RumorCluster).where(RumorCluster.topic_hash == topic_hash)
    )
    if cluster:
        cluster.claim_count += 1
        cluster.title = title
        cluster.summary = summary
        cluster.language = language
        return cluster

    if semantic_text and semantic_embedding:
        semantic_match = await _find_semantic_cluster(
            session,
            semantic_text=semantic_text,
            semantic_embedding=semantic_embedding,
            language=language,
        )
        if semantic_match:
            semantic_match.claim_count += 1
            semantic_match.title = title
            semantic_match.summary = summary
            semantic_match.language = language
            return semantic_match

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
        await session.execute(
            select(RumorCluster).order_by(
                RumorCluster.claim_count.desc(),
                RumorCluster.updated_at.desc(),
            )
        )
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
            select(Claim)
            .where(Claim.cluster_id == cluster.id)
            .order_by(Claim.created_at.desc())
            .limit(10)
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
