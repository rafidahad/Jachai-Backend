from __future__ import annotations

from typing import Any

from app.core.config import settings


def _compact_snippet(snippet: str, *, limit: int = 500) -> str:
    normalized = " ".join(snippet.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _stance_summary(classified_chunks: list[dict[str, Any]]) -> str:
    """Summarize stance distribution across classified chunks."""
    if not classified_chunks:
        return "No stance data available."
    counts: dict[str, int] = {"supports": 0, "refutes": 0, "neutral": 0, "background": 0}
    for chunk in classified_chunks:
        stance = str(chunk.get("stance") or "neutral")
        if stance in counts:
            counts[stance] += 1
    parts = [f"{v} {k}" for k, v in counts.items() if v > 0]
    return "Stance distribution: " + ", ".join(parts) + "."


def _credibility_lookup(
    source_id: str,
    credibility_scores: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if not credibility_scores:
        return {}
    for entry in credibility_scores:
        if str(entry.get("source_id")) == source_id:
            return entry
    return {}


def build_evidence_context(
    evidence: list[dict[str, object]],
    *,
    classified_chunks: list[dict[str, Any]] | None = None,
    credibility_scores: list[dict[str, Any]] | None = None,
) -> str:
    """
    Build a structured evidence context string for the verdict LLM.

    When classified_chunks are provided, the context uses chunk-level
    stance/rationale data instead of raw snippets. Otherwise falls back
    to the original snippet-only format.
    """
    lines: list[str] = []

    # Add stance summary header if we have classified chunks
    if classified_chunks:
        lines.append(_stance_summary(classified_chunks))
        lines.append("")
        # Use classified chunks as the primary evidence representation
        top_chunks = classified_chunks[: settings.max_evidence_chunks]
        for i, chunk in enumerate(top_chunks, start=1):
            source_id = str(chunk.get("source_id") or "")
            cred = _credibility_lookup(source_id, credibility_scores)
            lines.extend([
                f"--- EVIDENCE {i} ---",
                f"SOURCE_ID: {source_id}",
                f"CHUNK_ID: {chunk.get('chunk_id', '')}",
                f"TITLE: {chunk.get('title', '')}",
                f"URL: {chunk.get('url', '')}",
                f"DOMAIN: {chunk.get('domain', '')}",
                f"PUBLISHED: {chunk.get('published_date') or 'unknown'}",
                f"SNIPPET_ONLY: {chunk.get('snippet_only', False)}",
                (
                    f"CREDIBILITY: {cred.get('source_type', 'unknown')} "
                    f"score={float(cred.get('credibility_score', 0.5)):.2f}"
                    if cred
                    else "CREDIBILITY: unknown"
                ),
                f"STANCE: {chunk.get('stance', 'neutral')} "
                f"(confidence={float(chunk.get('stance_confidence') or 0.0):.2f})",
                f"RATIONALE: {chunk.get('rationale', '')}",
                f"QUOTED_EVIDENCE: {chunk.get('quoted_evidence', '')}",
                f"RELEVANCE_SCORE: {float(chunk.get('relevance_score') or 0.0):.3f}",
                f"TEXT:\n{_compact_snippet(str(chunk.get('text') or ''))}",
                "",
            ])
        return "\n".join(lines).strip()

    # ── Legacy path: no classified chunks, use raw evidence items ────────────
    top_evidence = evidence[: settings.final_evidence_top_k]
    for i, item in enumerate(top_evidence, start=1):
        source_id = str(item.get("source_id") or "")
        cred = _credibility_lookup(source_id, credibility_scores)
        match_score = item.get("match_score")
        trust_score = item.get("trust_score")
        search_score = item.get("search_score")
        lines.extend([
            f"--- EVIDENCE {i} ---",
            f"SOURCE_ID: {source_id}",
            f"TITLE: {item['title']}",
            f"PUBLISHER: {item.get('publisher') or 'Unknown'}",
            f"TYPE: {item['source_type']}",
            f"SIMILARITY: {float(item['similarity_score']):.2f}",
            (
                f"MATCH_SCORE: {float(match_score):.2f}"
                if match_score is not None
                else "MATCH_SCORE: not_available"
            ),
            (
                f"TRUST_SCORE: {float(trust_score):.2f}"
                if trust_score is not None
                else "TRUST_SCORE: not_available"
            ),
            (
                f"SEARCH_SCORE: {float(search_score):.4f}"
                if search_score is not None
                else "SEARCH_SCORE: not_available"
            ),
            (
                f"RERANK_SCORE: {float(item['rerank_score']):.4f}"
                if item.get("rerank_score") is not None
                else "RERANK_SCORE: not_used"
            ),
            (
                f"CREDIBILITY: {cred.get('source_type', 'unknown')} "
                f"score={float(cred.get('credibility_score', float(trust_score or 0.5))):.2f}"
                if cred
                else "CREDIBILITY: not_available"
            ),
            f"URL: {item['url']}",
            f"SNIPPET: {_compact_snippet(str(item['snippet']))}",
            "",
        ])
    return "\n".join(lines).strip()
