from __future__ import annotations

import math
import re
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.services.ai_model_router import NVIDIAModelTask, get_model_for_task
from app.services.nvidia_client import call_nvidia_rerank

logger = get_logger(__name__)

# ── Local ranking helpers ─────────────────────────────────────────────────────

_TOKEN_PATTERN = re.compile(r"[\w']+", re.UNICODE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by",
    "for", "from", "has", "have", "in", "into", "is", "it",
    "its", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with",
}


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _TOKEN_PATTERN.findall(text.lower()):
        if len(token) >= 3 and token not in _STOPWORDS:
            tokens.add(token)
    return tokens


def _lexical_overlap(claim_tokens: set[str], text: str) -> float:
    if not claim_tokens:
        return 0.0
    doc_tokens = _tokenize(text)
    if not doc_tokens:
        return 0.0
    overlap = claim_tokens & doc_tokens
    return min(1.0, len(overlap) / len(claim_tokens))


def _entity_match_bonus(claim_tokens: set[str], chunk: dict[str, Any]) -> float:
    """Extra score if claim entities are prominent in the chunk title."""
    title = str(chunk.get("title") or "")
    return _lexical_overlap(claim_tokens, title) * 0.15


def _freshness_bonus(chunk: dict[str, Any]) -> float:
    """Mild bonus for chunks with a publication date."""
    return 0.05 if chunk.get("published_date") else 0.0


def _snippet_penalty(chunk: dict[str, Any]) -> float:
    """Penalty for snippet-only content."""
    return -0.15 if chunk.get("snippet_only") else 0.0


def _local_rank_chunk(chunk: dict[str, Any], *, claim_tokens: set[str]) -> float:
    text = str(chunk.get("text") or chunk.get("snippet") or "")
    lexical = _lexical_overlap(claim_tokens, text)
    entity_bonus = _entity_match_bonus(claim_tokens, chunk)
    freshness = _freshness_bonus(chunk)
    snippet_pen = _snippet_penalty(chunk)
    # Credibility/trust from search metadata
    try:
        trust = float(chunk.get("trust_score") or 0.5)
    except (TypeError, ValueError):
        trust = 0.5
    trust_bonus = (trust - 0.5) * 0.2  # ±0.10 adjustment

    score = lexical + entity_bonus + freshness + snippet_pen + trust_bonus
    return max(0.0, min(1.0, score))


def _local_ranked_chunks(
    chunks: list[dict[str, Any]],
    *,
    claim_text: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Sort chunks by local lexical+heuristic score, assign ranks."""
    claim_tokens = _tokenize(claim_text)
    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk in chunks:
        score = _local_rank_chunk(chunk, claim_tokens=claim_tokens)
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    result: list[dict[str, Any]] = []
    for final_rank, (score, chunk) in enumerate(scored[:limit], start=1):
        item = dict(chunk)
        item["relevance_score"] = round(score, 4)
        item["final_rank"] = final_rank
        item.setdefault("rerank_score", None)
        item["ranking_reason"] = "local_lexical"
        result.append(item)
    return result


def _passage_from_chunk(chunk: dict[str, Any]) -> str:
    title = str(chunk.get("title") or "Untitled")
    domain = str(chunk.get("domain") or "unknown")
    url = str(chunk.get("url") or "")
    text = " ".join(str(chunk.get("text") or chunk.get("snippet") or "").split())
    if len(text) > 600:
        text = text[:597].rstrip() + "..."
    return (
        f"TITLE: {title}\n"
        f"DOMAIN: {domain}\n"
        f"URL: {url}\n"
        f"TEXT: {text}"
    )


# Keep legacy function for backward compat with pgvector candidate dicts
def _passage_from_candidate(candidate: dict[str, Any]) -> str:
    title = str(candidate.get("title") or "Untitled")
    publisher = str(candidate.get("publisher") or "Unknown")
    source_type = str(candidate.get("source_type") or "unknown")
    url = str(candidate.get("url") or "")
    snippet = " ".join(str(candidate.get("snippet") or "").split())
    if len(snippet) > 420:
        snippet = snippet[:417].rstrip() + "..."
    return (
        f"TITLE: {title}\n"
        f"PUBLISHER: {publisher}\n"
        f"TYPE: {source_type}\n"
        f"URL: {url}\n"
        f"SNIPPET: {snippet}"
    )


async def rerank_evidence(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    claim_text: str | None = None,
    use_chunk_format: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Rerank evidence candidates using NVIDIA reranker with local fallback.

    Args:
        query: The retrieval query string.
        candidates: List of evidence candidates (pgvector EvidenceSource dicts)
                    or evidence chunks (from evidence_chunker).
        claim_text: Used for local fallback ranking. Falls back to query if None.
        use_chunk_format: If True, uses chunk passage format instead of legacy snippet format.

    Returns:
        (reranked_candidates, metadata_dict)
    """
    effective_claim = claim_text or query
    limit = settings.final_evidence_top_k

    if len(candidates) < settings.rerank_min_candidates:
        if use_chunk_format:
            ranked = _local_ranked_chunks(candidates, claim_text=effective_claim, limit=limit)
        else:
            ranked = _fallback_candidates(candidates)
        return ranked, {
            "applied": False,
            "skipped": True,
            "reason": "insufficient_candidates",
            "call_count": 0,
            "model": None,
            "candidate_count": len(candidates),
        }

    model = get_model_for_task(NVIDIAModelTask.EVIDENCE_RERANKING)
    if not settings.nvidia_api_key or not model:
        if use_chunk_format:
            ranked = _local_ranked_chunks(candidates, claim_text=effective_claim, limit=limit)
        else:
            ranked = _fallback_candidates(candidates)
        return ranked, {
            "applied": False,
            "skipped": True,
            "reason": "rerank_not_configured",
            "call_count": 0,
            "model": model,
            "candidate_count": len(candidates),
        }

    try:
        if use_chunk_format:
            passages = [_passage_from_chunk(c) for c in candidates]
        else:
            passages = [_passage_from_candidate(c) for c in candidates]

        response = await call_nvidia_rerank(
            task=NVIDIAModelTask.EVIDENCE_RERANKING,
            query_text=query,
            passages=passages,
            model=model,
        )
        rankings = response.get("rankings") or []
        reranked: list[dict[str, Any]] = []
        for final_rank, ranking in enumerate(rankings, start=1):
            index = ranking.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(candidates):
                continue
            item = dict(candidates[index])
            score = ranking.get("score", ranking.get("logit"))
            item["rerank_score"] = float(score) if score is not None else None
            item["final_rank"] = final_rank
            # Add relevance_score normalized from rerank_score
            if score is not None:
                # rerank logit scores can be large; sigmoid-normalize
                try:
                    item["relevance_score"] = round(1 / (1 + math.exp(-float(score))), 4)
                except (OverflowError, ValueError):
                    item["relevance_score"] = 0.5
            else:
                item["relevance_score"] = 0.5
            item["ranking_reason"] = "nvidia_reranker"
            reranked.append(item)

        if not reranked:
            raise RuntimeError("Rerank response did not contain usable rankings.")

        return reranked[:limit], {
            "applied": True,
            "skipped": False,
            "reason": None,
            "call_count": 1,
            "model": model,
            "candidate_count": len(candidates),
            "returned_count": len(reranked),
        }
    except Exception:
        logger.exception("nvidia_rerank_failed — falling back to local ranking")
        if use_chunk_format:
            ranked = _local_ranked_chunks(candidates, claim_text=effective_claim, limit=limit)
        else:
            ranked = _fallback_candidates(candidates)
        return ranked, {
            "applied": False,
            "skipped": False,
            "reason": "rerank_failed",
            "call_count": 1,
            "model": model,
            "candidate_count": len(candidates),
        }


def _fallback_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Legacy fallback: simple top-K slice with rank assignment."""
    fallback: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[: settings.final_evidence_top_k], start=1):
        item = dict(candidate)
        item["final_rank"] = index
        item.setdefault("initial_rank", index)
        item.setdefault("rerank_score", None)
        item.setdefault("relevance_score", 0.5)
        item.setdefault("ranking_reason", "local_fallback")
        fallback.append(item)
    return fallback
