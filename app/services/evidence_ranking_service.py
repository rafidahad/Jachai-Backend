from __future__ import annotations
import math
import re
from typing import Any
from pydantic import BaseModel
from app.core.config import settings
from app.core.logging import get_logger
from app.services.nvidia_rerank_service import rerank_evidence
TOKEN_PATTERN = re.compile(r"[\w']+", re.UNICODE)
MATCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}
MATCH_TOKEN_ALIASES = {
    "buffaloes": "buffalo",
    "cattle": "buffalo",
    "cow": "buffalo",
    "cows": "buffalo",
    "mahish": "buffalo",
    "mohis": "buffalo",
    "mohish": "buffalo",
    "mohishh": "buffalo",
    "mosh": "buffalo",
    "মহিষ": "buffalo",
    "eidaladha": "sacrifice",
    "eiduladha": "sacrifice",
    "korban": "sacrifice",
    "korbani": "sacrifice",
    "kurban": "sacrifice",
    "qorbani": "sacrifice",
    "qurbani": "sacrifice",
    "sacrificed": "sacrifice",
    "sacrificing": "sacrifice",
    "কোরবানি": "sacrifice",
    "কুরবানি": "sacrifice",
}

def _canonical_match_token(token: str) -> str:
    normalized = token.lower().strip("'")
    if normalized.endswith("'s"):
        normalized = normalized[:-2]
    normalized = normalized.replace("-", "")
    return MATCH_TOKEN_ALIASES.get(normalized, normalized)

def _tokens_for_match(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in TOKEN_PATTERN.findall(text):
        canonical = _canonical_match_token(token)
        if len(canonical) < 3 or canonical in MATCH_STOPWORDS:
            continue
        tokens.add(canonical)
    return tokens

def _token_overlap_score(claim_text: str, evidence_text: str) -> float:
    claim_tokens = _tokens_for_match(claim_text)
    if not claim_tokens:
        return 0.0
    evidence_tokens = _tokens_for_match(evidence_text)
    if not evidence_tokens:
        return 0.0
    overlap = claim_tokens & evidence_tokens
    return min(1.0, len(overlap) / len(claim_tokens))

logger = get_logger(__name__)

class RankedEvidenceChunk(BaseModel):
    chunk_id: str
    source_id: str
    title: str
    url: str
    domain: str
    published_date: str | None = None
    text: str
    snippet_only: bool
    relevance_score: float
    ranking_reason: str

class EvidenceRankingService:
    @staticmethod
    async def rank(claim: str, chunks: list[Any], max_chunks: int = 12) -> tuple[list[RankedEvidenceChunk], list[str]]:
        warnings = []
        if not chunks:
            return [], warnings

        candidate_list = []
        for c in chunks:
            overlap_score = _token_overlap_score(claim, f"{c.title} {c.text}")
            
            score = overlap_score
            if c.snippet_only:
                score *= 0.8
                
            candidate_list.append({
                "chunk": c,
                "local_score": score
            })

        candidate_list.sort(key=lambda x: x["local_score"], reverse=True)

        nvidia_applied = False
        reranked_chunks = []
        
        if settings.nvidia_api_key:
            try:
                candidates_for_rerank = []
                for item in candidate_list:
                    c = item["chunk"]
                    candidates_for_rerank.append({
                        "source_id": c.source_id,
                        "title": c.title,
                        "url": c.url,
                        "domain": c.domain,
                        "publisher": c.domain,
                        "snippet": c.text,
                        "similarity_score": item["local_score"],
                        "match_score": item["local_score"]
                    })
                
                # Use a larger minimum candidates check to verify
                reranked_res, meta = await rerank_evidence(claim, candidates_for_rerank)
                if meta.get("applied"):
                    nvidia_applied = True
                    chunk_lookup = {c["chunk"].url: c["chunk"] for c in candidate_list}
                    for rank_idx, r in enumerate(reranked_res, start=1):
                        c = chunk_lookup.get(r["url"])
                        if c:
                            reranked_chunks.append(RankedEvidenceChunk(
                                chunk_id=c.chunk_id,
                                source_id=c.source_id,
                                title=c.title,
                                url=c.url,
                                domain=c.domain,
                                published_date=c.published_date,
                                text=c.text,
                                snippet_only=c.snippet_only,
                                relevance_score=float(r.get("rerank_score") or r.get("match_score") or 0.0),
                                ranking_reason=f"NVIDIA Reranker ranked at position {rank_idx}."
                            ))
            except Exception as exc:
                logger.warning("nvidia_reranker_failed_falling_back_to_local_scoring error=%s", str(exc))
                warnings.append("NVIDIA Reranker failed. Fell back to local lexical ranking.")

        if not nvidia_applied:
            for rank_idx, item in enumerate(candidate_list, start=1):
                c = item["chunk"]
                reranked_chunks.append(RankedEvidenceChunk(
                    chunk_id=c.chunk_id,
                    source_id=c.source_id,
                    title=c.title,
                    url=c.url,
                    domain=c.domain,
                    published_date=c.published_date,
                    text=c.text,
                    snippet_only=c.snippet_only,
                    relevance_score=item["local_score"],
                    ranking_reason=f"Local lexical matcher ranked at position {rank_idx}."
                ))

        reranked_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        return reranked_chunks[:max_chunks], warnings
