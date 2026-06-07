from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.search_run import SearchResultRecord, SearchRun
from app.schemas.source_schema import SourceIngestItemSchema
from app.services.language_service import detect_language
from app.services.search_service import (
    MIN_TAVILY_CONTENT_CHARACTERS,
    NormalizedSearchResult,
    TavilySearchResponse,
    dedupe_search_results,
    normalize_search_url,
    normalize_tavily_response,
    search_with_tavily,
)
from app.services.source_service import ingest_sources
from app.services.text_cleaning_service import clean_text
from app.utils.hashing import normalized_hash

logger = get_logger(__name__)

HTML_LANGUAGE_MAP = {
    "bn": "Bangla",
    "en": "English",
    "hi": "Hindi",
}


@dataclass(slots=True)
class LiveEvidenceDocument:
    title: str
    url: str
    publisher: str
    language: str
    source_type: str
    snippet: str
    text_content: str
    metadata: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _domain_from_url(url: str) -> str:
    return (urlsplit(url).hostname or "unknown").lower().removeprefix("www.")


def _build_request_headers() -> dict[str, str]:
    return {
        "User-Agent": "JachAI/0.1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8,bn;q=0.7,hi;q=0.6",
    }


def _normalize_html_language(value: str | None) -> str | None:
    if not value:
        return None
    prefix = value.split("-", 1)[0].strip().lower()
    return HTML_LANGUAGE_MAP.get(prefix)


def _build_live_search_queries(claim_text: str, search_queries: list[str] | None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for query in [claim_text, *(search_queries or [])]:
        cleaned = clean_text(query)
        if len(cleaned) < 5:
            continue
        normalized = cleaned.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(cleaned)
    return ordered[:6]


def _extract_meta_content(soup: BeautifulSoup, *keys: tuple[str, str]) -> str | None:
    for attr_name, attr_value in keys:
        tag = soup.find("meta", attrs={attr_name: attr_value})
        if tag and tag.get("content"):
            content = clean_text(str(tag["content"]))
            if content:
                return content
    return None


def _article_text_from_html(html: str) -> tuple[str, str, str | None, str | None]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    title = (
        _extract_meta_content(soup, ("property", "og:title"), ("name", "twitter:title"))
        or clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    )
    description = _extract_meta_content(
        soup,
        ("name", "description"),
        ("property", "og:description"),
        ("name", "twitter:description"),
    )
    language = None
    html_tag = soup.find("html")
    if html_tag and html_tag.get("lang"):
        language = clean_text(str(html_tag.get("lang")))

    container = soup.find("article") or soup.find("main") or soup.body or soup
    text_content = clean_text(container.get_text(separator=" "))
    return title, text_content, description, language


async def _crawl_result_url(client: httpx.AsyncClient, result: NormalizedSearchResult) -> tuple[str, str, str | None]:
    response = await client.get(result.url, headers=_build_request_headers(), follow_redirects=True)
    response.raise_for_status()
    final_url = normalize_search_url(str(response.url))
    title, text_content, description, html_language = _article_text_from_html(response.text)
    if final_url != result.url:
        result.url = final_url
        result.domain = _domain_from_url(final_url)
    return title, text_content, description or result.snippet, html_language


async def _document_from_search_result(
    client: httpx.AsyncClient,
    result: NormalizedSearchResult,
    *,
    search_answer: str | None,
    search_run_id: str,
) -> LiveEvidenceDocument | None:
    content = clean_text(result.content or "")
    title = result.title
    snippet = result.snippet or content[:320]
    html_language: str | None = None

    if len(content) < MIN_TAVILY_CONTENT_CHARACTERS:
        result.selected_for_crawl = True
        try:
            crawled_title, crawled_text, crawled_snippet, html_language = await _crawl_result_url(client, result)
        except Exception:
            logger.exception("tavily_result_crawl_failed url=%s", result.url)
            return None
        content = clean_text(crawled_text)
        if len(content) < settings.live_evidence_min_article_characters:
            return None
        title = crawled_title or title
        snippet = crawled_snippet or content[:320]

    domain = result.domain or _domain_from_url(result.url)
    language = detect_language(content)
    if not language or language == "Unknown":
        language = _normalize_html_language(html_language) or "Unknown"

    snippet = clean_text(snippet or content[:320])
    if len(snippet) > 320:
        snippet = snippet[:317].rstrip() + "..."

    metadata = {
        "seeded_by": "live_tavily_lookup",
        "source_channel": "tavily_search",
        "provider": "tavily",
        "request_id": result.request_id,
        "generated_query": result.query,
        "query_list": result.query_list or [result.query],
        "search_answer": search_answer,
        "search_score": result.search_score,
        "trust_score": result.trust_score,
        "domain": domain,
        "favicon": result.favicon,
        "selected_for_crawl": result.selected_for_crawl,
        "content_hash": normalized_hash(content),
        "fetched_at": _utc_now_iso(),
        "final_url": result.url,
        "search_run_id": search_run_id,
    }
    return LiveEvidenceDocument(
        title=(clean_text(title) or result.url)[:255],
        url=result.url,
        publisher=domain,
        language=language,
        source_type="tavily_search",
        snippet=snippet,
        text_content=content,
        metadata=metadata,
    )


async def _run_tavily_queries(query_texts: list[str]) -> tuple[list[TavilySearchResponse], list[str]]:
    responses: list[TavilySearchResponse] = []
    errors: list[str] = []
    results = await asyncio.gather(
        *[search_with_tavily(query) for query in query_texts],
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, Exception):
            logger.exception("tavily_query_failed", exc_info=result)
            errors.append(str(result))
            continue
        responses.append(result)
    return responses, errors


async def hydrate_live_evidence(
    session: AsyncSession,
    *,
    claim_text: str,
    language: str,
    normalized_hash: str,
    search_queries: list[str] | None = None,
) -> dict[str, Any]:
    if not settings.live_evidence_enabled:
        return {
            "enabled": False,
            "attempted": False,
            "ingested_count": 0,
            "created_count": 0,
            "updated_count": 0,
            "channels": [],
        }

    query_texts = _build_live_search_queries(claim_text, search_queries)
    if not query_texts:
        return {
            "enabled": True,
            "attempted": False,
            "ingested_count": 0,
            "created_count": 0,
            "updated_count": 0,
            "channels": [],
            "search_queries": [],
            "search_answer_context": None,
            "tavily_request_count": 0,
            "normalized_hash": normalized_hash,
            "documents": [],
        }

    tavily_responses, errors = await _run_tavily_queries(query_texts)
    normalized_results = dedupe_search_results(
        [
            result
            for response in tavily_responses
            for result in normalize_tavily_response(response)
        ]
    )
    normalized_results = normalized_results[: settings.live_evidence_max_documents]

    request_ids = [
        response.request_id
        for response in tavily_responses
        if response.request_id
    ]
    search_answers = [
        clean_text(response.answer or "")
        for response in tavily_responses
        if clean_text(response.answer or "")
    ]
    search_answer_context = "\n".join(search_answers) or None

    search_run = SearchRun(
        extracted_claim=claim_text,
        search_queries=query_texts,
        provider="tavily",
        search_answer=search_answer_context,
        request_ids=request_ids,
        result_count=len(normalized_results),
        crawled_count=sum(1 for result in normalized_results if result.selected_for_crawl),
        status="failed" if errors and not tavily_responses else "completed",
        error_message="; ".join(errors)[:2000] if errors else None,
    )
    session.add(search_run)
    await session.flush()

    timeout = min(settings.live_evidence_timeout_seconds, settings.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        document_results = await asyncio.gather(
            *[
                _document_from_search_result(
                    client,
                    result,
                    search_answer=search_answer_context,
                    search_run_id=str(search_run.id),
                )
                for result in normalized_results
            ],
            return_exceptions=True,
        )

    documents: list[LiveEvidenceDocument] = []
    for index, result in enumerate(normalized_results, start=1):
        session.add(
            SearchResultRecord(
                search_run_id=search_run.id,
                query=result.query,
                title=result.title[:255],
                url=result.url,
                snippet=result.snippet,
                content=result.content,
                provider=result.provider,
                search_score=result.search_score,
                trust_score=result.trust_score,
                favicon=result.favicon,
                rank=index,
                selected_for_crawl=result.selected_for_crawl,
            )
        )

    for result in document_results:
        if isinstance(result, Exception):
            logger.warning("tavily_document_build_failed error=%s", result)
            continue
        if result is not None:
            documents.append(result)

    search_run.crawled_count = sum(1 for result in normalized_results if result.selected_for_crawl)
    selected_documents = documents[: settings.live_evidence_max_documents]
    if not selected_documents:
        await session.commit()
        return {
            "enabled": True,
            "attempted": True,
            "ingested_count": 0,
            "created_count": 0,
            "updated_count": 0,
            "channels": ["tavily"],
            "search_queries": query_texts,
            "search_answer_context": search_answer_context,
            "tavily_request_count": len(tavily_responses),
            "tavily_result_count": len(normalized_results),
            "crawled_count": search_run.crawled_count,
            "search_run_id": str(search_run.id),
            "normalized_hash": normalized_hash,
            "documents": [],
        }

    ingest_items = [
        SourceIngestItemSchema(
            title=document.title,
            url=document.url,
            publisher=document.publisher,
            language=document.language,
            source_type=document.source_type,
            snippet=document.snippet,
            text_content=document.text_content,
            metadata=document.metadata,
        )
        for document in selected_documents
    ]
    items, created_count, updated_count = await ingest_sources(session, ingest_items)
    return {
        "enabled": True,
        "attempted": True,
        "ingested_count": len(items),
        "created_count": created_count,
        "updated_count": updated_count,
        "channels": ["tavily"],
        "search_queries": query_texts,
        "search_answer_context": search_answer_context,
        "tavily_request_count": len(tavily_responses),
        "tavily_result_count": len(normalized_results),
        "crawled_count": search_run.crawled_count,
        "search_run_id": str(search_run.id),
        "normalized_hash": normalized_hash,
        "documents": [
            {
                "title": item.title,
                "url": item.url,
                "publisher": item.publisher,
                "language": item.language,
                "source_type": item.source_type,
                "embedding_available": item.embedding_available,
            }
            for item in items
        ],
    }
