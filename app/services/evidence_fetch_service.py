from __future__ import annotations
import asyncio
from typing import Any
from urllib.parse import urlsplit
import httpx
from pydantic import BaseModel
from app.core.config import settings
from app.core.logging import get_logger
from app.services.input_normalization_service import fetch_url_content

logger = get_logger(__name__)

class FetchedEvidence(BaseModel):
    source_id: str
    title: str
    url: str
    canonical_url: str | None = None
    domain: str
    author: str | None = None
    published_date: str | None = None
    raw_text: str
    fetch_status: str  # success | failed | snippet_fallback
    snippet_only: bool

class EvidenceFetchService:
    @staticmethod
    async def fetch_all(candidates: list[Any], max_to_fetch: int = 5) -> tuple[list[FetchedEvidence], list[str]]:
        warnings = []
        fetched_items = []
        
        to_fetch = candidates[:max_to_fetch]
        remaining = candidates[max_to_fetch:]
        
        async def fetch_one(candidate: Any) -> FetchedEvidence:
            url = candidate.url
            source_id = f"src_{hash(url) & 0xffffffff}"
            domain = candidate.domain
            title = candidate.title
            snippet = candidate.snippet
            pub_date = candidate.published_date
            
            res = await fetch_url_content(url)
            if res.get("error"):
                return FetchedEvidence(
                    source_id=source_id,
                    title=title,
                    url=url,
                    canonical_url=url,
                    domain=domain,
                    published_date=pub_date,
                    raw_text=snippet,
                    fetch_status="snippet_fallback",
                    snippet_only=True
                )
            
            return FetchedEvidence(
                source_id=source_id,
                title=res.get("title") or title,
                url=url,
                canonical_url=res.get("canonical_url") or url,
                domain=domain,
                author=res.get("author"),
                published_date=res.get("published_date") or pub_date,
                raw_text=res.get("raw_text") or snippet,
                fetch_status="success",
                snippet_only=False
            )
            
        results = await asyncio.gather(*[fetch_one(c) for c in to_fetch], return_exceptions=True)
        
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                candidate = to_fetch[idx]
                warnings.append(f"Failed to fetch content from URL: {candidate.url} due to exception.")
                source_id = f"src_{hash(candidate.url) & 0xffffffff}"
                fetched_items.append(FetchedEvidence(
                    source_id=source_id,
                    title=candidate.title,
                    url=candidate.url,
                    domain=candidate.domain,
                    published_date=candidate.published_date,
                    raw_text=candidate.snippet,
                    fetch_status="snippet_fallback",
                    snippet_only=True
                ))
            else:
                fetched_items.append(res)
                if res.fetch_status == "snippet_fallback":
                    warnings.append(f"Fetching failed for {res.url}; using search snippet fallback.")
                    
        for c in remaining:
            source_id = f"src_{hash(c.url) & 0xffffffff}"
            fetched_items.append(FetchedEvidence(
                source_id=source_id,
                title=c.title,
                url=c.url,
                domain=c.domain,
                published_date=c.published_date,
                raw_text=c.snippet,
                fetch_status="snippet_fallback",
                snippet_only=True
            ))
            
        return fetched_items, warnings
