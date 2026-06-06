from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.services.trusted_publisher_registry import is_trusted_url
from app.services.text_cleaning_service import clean_text

logger = get_logger(__name__)

BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"
SERPER_SEARCH_ENDPOINT = "https://google.serper.dev/search"


@dataclass(slots=True)
class SearchResult:
    url: str
    title: str
    snippet: str
    provider: str


def _result_from_values(
    *,
    url: object,
    title: object,
    snippet: object,
    provider: str,
) -> SearchResult | None:
    url_text = clean_text(str(url or ""))
    if not url_text or not is_trusted_url(url_text):
        return None
    return SearchResult(
        url=url_text,
        title=clean_text(str(title or "")) or url_text,
        snippet=clean_text(str(snippet or "")),
        provider=provider,
    )


def _parse_brave(payload: dict[str, Any]) -> list[SearchResult]:
    raw_results = ((payload.get("web") or {}).get("results") or [])
    results: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        result = _result_from_values(
            url=item.get("url"),
            title=item.get("title"),
            snippet=item.get("description"),
            provider="brave",
        )
        if result:
            results.append(result)
    return results


def _parse_tavily(payload: dict[str, Any]) -> list[SearchResult]:
    raw_results = payload.get("results") or []
    results: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        result = _result_from_values(
            url=item.get("url"),
            title=item.get("title"),
            snippet=item.get("content"),
            provider="tavily",
        )
        if result:
            results.append(result)
    return results


def _parse_serper(payload: dict[str, Any]) -> list[SearchResult]:
    raw_results = payload.get("organic") or []
    results: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        result = _result_from_values(
            url=item.get("link"),
            title=item.get("title"),
            snippet=item.get("snippet"),
            provider="serper",
        )
        if result:
            results.append(result)
    return results


def _parse_custom(payload: dict[str, Any]) -> list[SearchResult]:
    raw_results = payload.get("results") or payload.get("items") or []
    results: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        result = _result_from_values(
            url=item.get("url") or item.get("link"),
            title=item.get("title") or item.get("name"),
            snippet=item.get("snippet") or item.get("description") or item.get("content"),
            provider="custom",
        )
        if result:
            results.append(result)
    return results


def _dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    deduped: dict[str, SearchResult] = {}
    for result in results:
        deduped.setdefault(result.url, result)
    return list(deduped.values())[: settings.general_search_max_results]


async def search_general_web(query: str, *, language: str | None = None) -> list[SearchResult]:
    provider = settings.normalized_search_provider
    if not settings.general_search_enabled:
        return []
    if not settings.general_search_api_key:
        logger.warning("general_search_skipped_missing_api_key provider=%s", provider)
        return []

    timeout = min(settings.request_timeout_seconds, settings.live_evidence_timeout_seconds)
    headers: dict[str, str] = {"Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            if provider == "brave":
                headers["X-Subscription-Token"] = settings.general_search_api_key
                response = await client.get(
                    BRAVE_SEARCH_ENDPOINT,
                    params={"q": query, "count": settings.general_search_max_results},
                    headers=headers,
                )
                response.raise_for_status()
                return _dedupe_results(_parse_brave(response.json()))

            if provider == "tavily":
                response = await client.post(
                    TAVILY_SEARCH_ENDPOINT,
                    json={
                        "api_key": settings.general_search_api_key,
                        "query": query,
                        "max_results": settings.general_search_max_results,
                        "include_answer": False,
                    },
                    headers=headers,
                )
                response.raise_for_status()
                return _dedupe_results(_parse_tavily(response.json()))

            if provider == "serper":
                headers["X-API-KEY"] = settings.general_search_api_key
                response = await client.post(
                    SERPER_SEARCH_ENDPOINT,
                    json={"q": query, "num": settings.general_search_max_results},
                    headers=headers,
                )
                response.raise_for_status()
                return _dedupe_results(_parse_serper(response.json()))

            if provider == "custom" and settings.general_search_base_url:
                headers["Authorization"] = f"Bearer {settings.general_search_api_key}"
                response = await client.get(
                    settings.general_search_base_url,
                    params={"q": query, "language": language, "limit": settings.general_search_max_results},
                    headers=headers,
                )
                response.raise_for_status()
                return _dedupe_results(_parse_custom(response.json()))

            logger.warning("general_search_skipped_unknown_provider provider=%s", provider)
            return []
    except Exception:
        logger.exception("general_search_failed provider=%s", provider)
        return []
