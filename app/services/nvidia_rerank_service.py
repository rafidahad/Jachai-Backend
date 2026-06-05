from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.services.ai_model_router import NVIDIAModelTask, get_model_for_task
from app.services.nvidia_client import call_nvidia_rerank

logger = get_logger(__name__)


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


def _fallback_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fallback: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[: settings.final_evidence_top_k], start=1):
        item = dict(candidate)
        item["final_rank"] = index
        item.setdefault("initial_rank", index)
        item.setdefault("rerank_score", None)
        fallback.append(item)
    return fallback


async def rerank_evidence(
    query: str,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(candidates) < settings.rerank_min_candidates:
        return _fallback_candidates(candidates), {
            "applied": False,
            "skipped": True,
            "reason": "insufficient_candidates",
            "call_count": 0,
            "model": None,
            "candidate_count": len(candidates),
        }

    model = get_model_for_task(NVIDIAModelTask.EVIDENCE_RERANKING)
    if not settings.nvidia_api_key or not model:
        return _fallback_candidates(candidates), {
            "applied": False,
            "skipped": True,
            "reason": "rerank_not_configured",
            "call_count": 0,
            "model": model,
            "candidate_count": len(candidates),
        }

    try:
        passages = [_passage_from_candidate(candidate) for candidate in candidates]
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
            reranked.append(item)

        if not reranked:
            raise RuntimeError("Rerank response did not contain usable rankings.")

        return reranked[: settings.final_evidence_top_k], {
            "applied": True,
            "skipped": False,
            "reason": None,
            "call_count": 1,
            "model": model,
            "candidate_count": len(candidates),
            "returned_count": len(reranked),
        }
    except Exception:
        logger.exception("nvidia_rerank_failed")
        return _fallback_candidates(candidates), {
            "applied": False,
            "skipped": False,
            "reason": "rerank_failed",
            "call_count": 1,
            "model": model,
            "candidate_count": len(candidates),
        }
