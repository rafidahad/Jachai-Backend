from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger
from app.services.text_cleaning_service import clean_text

logger = get_logger(__name__)

TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"
MIN_TAVILY_CONTENT_CHARACTERS = 150
DROP_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
    "spm",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}
_SEARCH_TOKEN_PATTERN = re.compile(r"[\w']+", re.UNICODE)
_SEARCH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by",
    "for", "from", "has", "have", "in", "into", "is", "it",
    "its", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with",
}

# ── Default trusted domain registry ──────────────────────────────────────────
# Extended via TRUSTED_DOMAINS env var. Entries are (domain, score, source_type).
_DEFAULT_TRUSTED_DOMAIN_SCORES: dict[str, tuple[float, str]] = {
    # Government / official
    "gov.bd": (0.95, "government"),
    "who.int": (0.95, "official"),
    "cdc.gov": (0.95, "official"),
    "nasa.gov": (0.95, "official"),
    "un.org": (0.93, "official"),
    "nih.gov": (0.93, "official"),
    "fda.gov": (0.92, "official"),
    # Reputable international news
    "reuters.com": (0.92, "reputable_news"),
    "apnews.com": (0.92, "reputable_news"),
    "bbc.com": (0.90, "reputable_news"),
    "afp.com": (0.90, "reputable_news"),
    "theguardian.com": (0.88, "reputable_news"),
    "nytimes.com": (0.87, "reputable_news"),
    "washingtonpost.com": (0.87, "reputable_news"),
    # Fact-check organisations
    "rumorscanner.com": (0.95, "fact_check"),
    "rumorscannerbd.com": (0.95, "fact_check"),
    "fact-watch.org": (0.95, "fact_check"),
    "boomlive.in": (0.92, "fact_check"),
    "altnews.in": (0.92, "fact_check"),
    "snopes.com": (0.90, "fact_check"),
    "factcheck.org": (0.90, "fact_check"),
    "fullfact.org": (0.90, "fact_check"),
    "factcheck.afp.com": (0.92, "fact_check"),
    "politifact.com": (0.88, "fact_check"),
    # Regional reputable news
    "prothomalo.com": (0.85, "reputable_news"),
    "thedailystar.net": (0.85, "reputable_news"),
    "tbsnews.net": (0.82, "reputable_news"),
    "bdnews24.com": (0.82, "reputable_news"),
    "dhakatribune.com": (0.82, "reputable_news"),
    # Lower trust
    "foxnews.com": (0.70, "reputable_news"),
    "greenwichtime.com": (0.65, "unknown"),
    "thesunchronicle.com": (0.60, "unknown"),
}

# ── Social / low-quality domain patterns ─────────────────────────────────────
_SOCIAL_DOMAINS = {
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "tiktok.com", "youtube.com", "reddit.com", "linkedin.com",
    "t.me", "telegram.org",
}
_LOW_QUALITY_PATTERNS = (
    "blogspot.", "wordpress.com", "medium.com", "substack.com",
    "wix.com", "weebly.com",
)


class TavilyResult(BaseModel):
    url: str
    title: str = ""
    content: str | None = None
    score: float | None = None
    raw_content: str | None = None
    favicon: str | None = None
    published_date: str | None = None


class TavilySearchResponse(BaseModel):
    query: str
    answer: str | None = None
    results: list[TavilyResult] = Field(default_factory=list)
    response_time: float | None = None
    request_id: str | None = None


class NormalizedSearchResult(BaseModel):
    query: str
    title: str
    url: str
    snippet: str | None = None
    content: str | None = None
    search_score: float | None = None
    raw_content: str | None = None
    favicon: str | None = None
    provider: str = "tavily"
    request_id: str | None = None
    domain: str | None = None
    trust_score: float = 0.50
    selected_for_crawl: bool = False
    query_list: list[str] = Field(default_factory=list)
    published_date: str | None = None


def normalize_search_url(url: str) -> str:
    parts = urlsplit(url.strip())
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in DROP_QUERY_KEYS
    ]
    return urlunsplit(parts._replace(query=urlencode(filtered_query, doseq=True), fragment=""))


def domain_from_url(url: str) -> str | None:
    hostname = urlsplit(url).hostname
    return hostname.lower().removeprefix("www.") if hostname else None


def _get_trusted_domain_entry(domain: str | None) -> tuple[float, str] | None:
    """Return (score, source_type) for a domain, checking settings overrides first."""
    if not domain:
        return None

    # Check settings-configured domains first (score 0.85 default for configured ones)
    for trusted in settings.trusted_domains:
        td = trusted.lower().strip()
        if domain == td or domain.endswith(f".{td}"):
            # Try to find it in default map for better score, else use 0.85
            entry = _DEFAULT_TRUSTED_DOMAIN_SCORES.get(td)
            return entry if entry else (0.85, "reputable_news")

    # Check default map
    for trusted_domain, entry in _DEFAULT_TRUSTED_DOMAIN_SCORES.items():
        if domain == trusted_domain or domain.endswith(f".{trusted_domain}"):
            return entry

    return None


def trust_score_for_domain(domain: str | None) -> float:
    entry = _get_trusted_domain_entry(domain)
    return entry[0] if entry else 0.50


def _search_tokens(text: str) -> set[str]:
    return {
        token
        for token in _SEARCH_TOKEN_PATTERN.findall(text.lower())
        if len(token) >= 3 and token not in _SEARCH_STOPWORDS
    }


def _search_overlap_score(claim_tokens: set[str], result: NormalizedSearchResult) -> float:
    if not claim_tokens:
        return 0.0
    haystack = " ".join(
        [
            result.title,
            result.snippet or "",
            result.content or "",
            result.query,
        ]
    )
    result_tokens = _search_tokens(haystack)
    if not result_tokens:
        return 0.0
    return min(1.0, len(claim_tokens & result_tokens) / len(claim_tokens))


def _normalized_result_search_score(result: NormalizedSearchResult) -> float:
    try:
        score = float(result.search_score or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if score <= 1:
        return max(0.0, min(1.0, score))
    return max(0.0, min(1.0, score / 100.0))


def select_reliable_matching_results(
    results: list[NormalizedSearchResult],
    *,
    claim_text: str,
    limit: int,
) -> list[NormalizedSearchResult]:
    """
    Prefer results that both match the claim text and come from trusted domains.

    If strict matching/trust would leave too few sources, this gracefully falls
    back to the best remaining Tavily results instead of returning an empty set.
    """
    claim_tokens = _search_tokens(clean_text(claim_text))
    scored: list[tuple[float, float, float, NormalizedSearchResult]] = []
    for result in results:
        overlap = _search_overlap_score(claim_tokens, result)
        search_score = _normalized_result_search_score(result)
        trust_score = max(0.0, min(1.0, float(result.trust_score or 0.5)))
        full_content_bonus = 0.05 if result.content and len(result.content) >= MIN_TAVILY_CONTENT_CHARACTERS else 0.0
        composite = (overlap * 0.45) + (trust_score * 0.35) + (search_score * 0.20) + full_content_bonus
        scored.append((composite, overlap, trust_score, result))

    enough_reliable_matches = [
        item
        for item in scored
        if item[1] >= 0.15 and item[2] >= 0.60
    ]
    pool = enough_reliable_matches if len(enough_reliable_matches) >= max(3, min(limit, 5)) else scored
    pool.sort(key=lambda item: (item[0], item[2], item[1]), reverse=True)
    return [item[3] for item in pool[:limit]]


def score_source_credibility(
    *,
    source_id: str,
    domain: str | None,
    snippet_only: bool = False,
    published_date: str | None = None,
    requires_freshness: bool = False,
) -> dict[str, Any]:
    """
    Compute a structured credibility assessment for a source.

    Returns a dict compatible with SourceCredibilitySchema.
    """
    if not domain:
        return {
            "source_id": source_id,
            "domain": domain or "unknown",
            "source_type": "unknown",
            "credibility_score": 0.30,
            "credibility_reason": "No domain information available.",
            "snippet_only": snippet_only,
        }

    # Social media
    if domain in _SOCIAL_DOMAINS:
        return {
            "source_id": source_id,
            "domain": domain,
            "source_type": "social_media",
            "credibility_score": 0.20,
            "credibility_reason": "Social media platform — not a primary source.",
            "snippet_only": snippet_only,
        }

    # Low-quality blog patterns
    if any(pat in domain for pat in _LOW_QUALITY_PATTERNS):
        return {
            "source_id": source_id,
            "domain": domain,
            "source_type": "blog",
            "credibility_score": 0.30,
            "credibility_reason": "User-generated blog platform — treat with caution.",
            "snippet_only": snippet_only,
        }

    # Trusted domain lookup
    entry = _get_trusted_domain_entry(domain)
    if entry:
        score, source_type = entry
        reason = f"Recognized {source_type.replace('_', ' ')} domain."
        # Penalty for snippet-only content
        if snippet_only:
            score = max(0.20, score - 0.20)
            reason += " Content is snippet-only — confidence reduced."
        # Penalty for missing date on freshness-sensitive claims
        if requires_freshness and not published_date:
            score = max(0.20, score - 0.15)
            reason += " No publication date found — freshness unverifiable."
        return {
            "source_id": source_id,
            "domain": domain,
            "source_type": source_type,
            "credibility_score": round(score, 3),
            "credibility_reason": reason,
            "snippet_only": snippet_only,
        }

    # Unknown domain — check TLD hints
    source_type = "unknown"
    base_score = 0.45
    if domain.endswith(".gov") or domain.endswith(".gov.bd"):
        source_type = "government"
        base_score = 0.88
    elif domain.endswith(".edu") or domain.endswith(".ac.uk"):
        source_type = "academic"
        base_score = 0.80
    elif domain.endswith(".org"):
        source_type = "primary_source"
        base_score = 0.60

    if snippet_only:
        base_score = max(0.20, base_score - 0.15)
    if requires_freshness and not published_date:
        base_score = max(0.20, base_score - 0.10)

    reason_parts = [f"Unknown domain ({domain})."]
    if source_type != "unknown":
        reason_parts.append(f"TLD suggests {source_type.replace('_', ' ')}.")
    if snippet_only:
        reason_parts.append("Snippet-only content.")

    return {
        "source_id": source_id,
        "domain": domain,
        "source_type": source_type,
        "credibility_score": round(base_score, 3),
        "credibility_reason": " ".join(reason_parts),
        "snippet_only": snippet_only,
    }


def normalize_tavily_response(response: TavilySearchResponse) -> list[NormalizedSearchResult]:
    normalized: list[NormalizedSearchResult] = []
    for result in response.results:
        url = normalize_search_url(result.url)
        domain = domain_from_url(url)
        content = clean_text(result.content or "")
        normalized.append(
            NormalizedSearchResult(
                query=response.query,
                title=clean_text(result.title) or url,
                url=url,
                snippet=content or None,
                content=content or None,
                search_score=result.score,
                raw_content=result.raw_content,
                favicon=clean_text(result.favicon or "") or None,
                request_id=response.request_id,
                domain=domain,
                trust_score=trust_score_for_domain(domain),
                selected_for_crawl=len(content) < MIN_TAVILY_CONTENT_CHARACTERS,
                query_list=[response.query],
                published_date=result.published_date,
            )
        )
    return normalized


def dedupe_search_results(results: list[NormalizedSearchResult]) -> list[NormalizedSearchResult]:
    deduped: dict[str, NormalizedSearchResult] = {}
    for result in results:
        existing = deduped.get(result.url)
        if existing is None:
            deduped[result.url] = result
            continue

        existing_queries = [*existing.query_list, *result.query_list]
        existing.query_list = list(dict.fromkeys(existing_queries))
        existing.query = existing.query_list[0] if existing.query_list else existing.query
        if (result.search_score or 0.0) > (existing.search_score or 0.0):
            result.query_list = existing.query_list
            result.query = result.query_list[0] if result.query_list else result.query
            deduped[result.url] = result

    ordered = sorted(
        deduped.values(),
        key=lambda item: (item.trust_score, _normalized_result_search_score(item)),
        reverse=True,
    )
    return ordered


async def search_with_tavily(query: str) -> TavilySearchResponse:
    api_key = settings.active_tavily_api_key
    if not settings.tavily_enabled or not api_key:
        return TavilySearchResponse(query=query)

    timeout = min(settings.request_timeout_seconds, settings.live_evidence_timeout_seconds)
    payload = {
        "query": query,
        "search_depth": settings.tavily_search_depth,
        "topic": settings.tavily_topic,
        "max_results": settings.tavily_max_results,
        "include_answer": settings.tavily_include_answer,
        "include_raw_content": settings.tavily_include_raw_content,
        "include_images": settings.tavily_include_images,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.post(TAVILY_SEARCH_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        return TavilySearchResponse.model_validate(response.json())


async def search_general_web(query: str, *, language: str | None = None) -> list[NormalizedSearchResult]:
    del language
    try:
        response = await search_with_tavily(query)
    except Exception:
        logger.exception("tavily_search_failed")
        return []
    return normalize_tavily_response(response)
