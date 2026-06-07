from __future__ import annotations

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
TRUSTED_DOMAIN_SCORES = {
    "gov.bd": 0.95,
    "who.int": 0.95,
    "reuters.com": 0.92,
    "apnews.com": 0.92,
    "bbc.com": 0.90,
    "afp.com": 0.90,
    "rumorscanner.com": 0.95,
    "rumorscannerbd.com": 0.95,
    "fact-watch.org": 0.95,
    "boomlive.in": 0.92,
    "altnews.in": 0.92,
    "prothomalo.com": 0.85,
    "thedailystar.net": 0.85,
    "tbsnews.net": 0.82,
    "bdnews24.com": 0.82,
    "dhakatribune.com": 0.82,
    "foxnews.com": 0.78,
    "greenwichtime.com": 0.74,
    "thesunchronicle.com": 0.70,
}


class TavilyResult(BaseModel):
    url: str
    title: str = ""
    content: str | None = None
    score: float | None = None
    raw_content: str | None = None
    favicon: str | None = None


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


def trust_score_for_domain(domain: str | None) -> float:
    if not domain:
        return 0.50
    for trusted_domain, score in TRUSTED_DOMAIN_SCORES.items():
        if domain == trusted_domain or domain.endswith(f".{trusted_domain}"):
            return score
    return 0.50


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
        key=lambda item: ((item.search_score or 0.0), item.trust_score),
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
