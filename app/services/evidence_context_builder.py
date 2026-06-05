from __future__ import annotations

from app.core.config import settings


def _compact_snippet(snippet: str, *, limit: int = 420) -> str:
    normalized = " ".join(snippet.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def build_evidence_context(evidence: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for item in evidence[: settings.final_evidence_top_k]:
        lines.extend(
            [
                f"SOURCE_ID: {item['source_id']}",
                f"TITLE: {item['title']}",
                f"PUBLISHER: {item.get('publisher') or 'Unknown'}",
                f"TYPE: {item['source_type']}",
                f"SIMILARITY: {float(item['similarity_score']):.2f}",
                (
                    f"RERANK_SCORE: {float(item['rerank_score']):.4f}"
                    if item.get("rerank_score") is not None
                    else "RERANK_SCORE: not_used"
                ),
                f"URL: {item['url']}",
                f"SNIPPET: {_compact_snippet(str(item['snippet']))}",
                "",
            ]
        )
    return "\n".join(lines).strip()
