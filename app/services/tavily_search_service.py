from __future__ import annotations
import asyncio
from typing import Any
from urllib.parse import urlsplit
import httpx
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class SearchResultV2(BaseModel):
    title: str
    url: str
    snippet: str
    score: float
    published_date: str | None = None
    domain: str

class TavilySearchService:
    @staticmethod
    async def search(queries: list[dict[str, Any]], max_results: int = 8) -> tuple[list[SearchResultV2], list[str]]:
        warnings = []
        api_key = settings.active_tavily_api_key
        if not settings.tavily_enabled or not api_key:
            warnings.append("Tavily search is disabled or API key is missing.")
            return [], warnings

        timeout = min(settings.request_timeout_seconds, settings.live_evidence_timeout_seconds)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        async def query_tavily(query_str: str) -> dict[str, Any] | None:
            payload = {
                "query": query_str,
                "search_depth": settings.tavily_search_depth,
                "topic": settings.tavily_topic,
                "max_results": max_results,
                "include_answer": settings.tavily_include_answer,
                "include_raw_content": settings.tavily_include_raw_content,
            }
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    response = await client.post("https://api.tavily.com/search", json=payload, headers=headers)
                    if response.status_code == 401:
                        logger.error("tavily_unauthorized invalid_api_key")
                        return {"error": "Unauthorized API Key"}
                    response.raise_for_status()
                    return response.json()
            except Exception as exc:
                logger.warning("tavily_query_failed query=%s error=%s", query_str, str(exc))
                return {"error": str(exc)}

        query_texts = [q["query"] for q in queries]
        results = await asyncio.gather(*[query_tavily(q) for q in query_texts], return_exceptions=True)

        raw_results = []
        for q_text, res in zip(query_texts, results):
            if isinstance(res, Exception):
                warnings.append(f"Search for query '{q_text}' failed due to network exception.")
                continue
            if not res:
                continue
            if "error" in res:
                warnings.append(f"Search query '{q_text}' returned error: {res['error']}")
                continue
            raw_results.extend(res.get("results", []))

        unique_results = {}
        for r in raw_results:
            url = r.get("url")
            if not url:
                continue
            url = url.strip()
            if url in unique_results:
                if r.get("score", 0.0) > unique_results[url].get("score", 0.0):
                    unique_results[url] = r
            else:
                unique_results[url] = r

        filtered_results = []
        for r in unique_results.values():
            url = r.get("url")
            title = r.get("title", "") or url
            snippet = r.get("content", r.get("snippet", "")) or ""
            score = r.get("score", 0.5)
            
            try:
                domain = (urlsplit(url).hostname or "unknown").lower().removeprefix("www.")
            except Exception:
                domain = "unknown"

            if not url.startswith("http") or len(url) < 10:
                continue
            if not snippet or len(snippet) < 10:
                continue

            filtered_results.append(SearchResultV2(
                title=title,
                url=url,
                snippet=snippet,
                score=float(score),
                published_date=r.get("published_date"),
                domain=domain
            ))

        filtered_results.sort(key=lambda x: x.score, reverse=True)
        return filtered_results, warnings
