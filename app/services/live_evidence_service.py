from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
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
    select_reliable_matching_results,
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
    # New richer fields
    snippet_only: bool = False
    published_date: str | None = None
    fetch_status: str = "success"  # "success" | "failed" | "snippet_fallback" | "tavily_only"
    warnings: list[str] = field(default_factory=list)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _domain_from_url(url: str) -> str:
    return (urlsplit(url).hostname or "unknown").lower().removeprefix("www.")


def _build_request_headers() -> dict[str, str]:
    return {
        "User-Agent": "JachAI/0.1 (fact-checking bot; contact: support@jachai.ai)",
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


def _extract_published_date(soup: BeautifulSoup) -> str | None:
    """Try to extract publication date from common meta tags."""
    candidates = [
        ("property", "article:published_time"),
        ("name", "publishdate"),
        ("name", "date"),
        ("itemprop", "datePublished"),
        ("property", "og:article:published_time"),
        ("name", "DC.date"),
        ("name", "article.published"),
    ]
    date_raw = _extract_meta_content(soup, *candidates)
    if date_raw:
        # Truncate to date portion if ISO datetime
        return date_raw[:10] if len(date_raw) >= 10 else date_raw

    # Check time tags
    time_tag = soup.find("time")
    if time_tag:
        dt_attr = time_tag.get("datetime") or time_tag.get_text(strip=True)
        if dt_attr:
            dt_str = str(dt_attr)[:10]
            return dt_str if len(dt_str) >= 4 else None
    return None


def _article_text_from_html(
    html: str,
) -> tuple[str, str, str | None, str | None, str | None]:
    """
    Extract title, text_content, description, html_language, published_date from HTML.
    Returns (title, text_content, description, language, published_date).
    """
    soup = BeautifulSoup(html, "html.parser")
    # Remove boilerplate tags
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "nav", "footer", "header"]):
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

    published_date = _extract_published_date(soup)

    # Prefer article/main body for content
    container = soup.find("article") or soup.find("main") or soup.body or soup
    # Remove repeated nav/menu patterns inside container
    for tag in container(["nav", "menu", "aside"]):
        tag.decompose()
    text_content = clean_text(container.get_text(separator=" "))
    return title, text_content, description, language, published_date


async def _read_capped_response_bytes(
    response: httpx.Response,
    *,
    max_bytes: int,
) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if not chunk:
            continue
        remaining = max_bytes - len(body)
        if remaining <= 0:
            break
        body.extend(chunk[:remaining])
        if len(body) >= max_bytes:
            break
    return bytes(body)


async def _crawl_result_url(
    client: httpx.AsyncClient,
    result: NormalizedSearchResult,
) -> tuple[str, str, str | None, str | None, str | None]:
    """
    Fetch a URL and extract (title, text_content, description, html_language, published_date).
    Raises on failure.
    """
    # Enforce max fetch size via HEAD check or content-length
    max_bytes = settings.max_fetch_bytes

    async with client.stream(
        "GET",
        result.url,
        headers=_build_request_headers(),
        follow_redirects=True,
    ) as response:
        response.raise_for_status()

        # Check content-type — only parse HTML
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            raise ValueError(f"Non-HTML content-type: {content_type}")

        raw_bytes = await _read_capped_response_bytes(response, max_bytes=max_bytes)
        final_url = normalize_search_url(str(response.url))

    html = raw_bytes.decode("utf-8", errors="replace")
    title, text_content, description, html_language, published_date = await asyncio.to_thread(
        _article_text_from_html,
        html,
    )
    if final_url != result.url:
        result.url = final_url
        result.domain = _domain_from_url(final_url)
    return title, text_content, description or result.snippet, html_language, published_date


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
    published_date = result.published_date  # from Tavily
    snippet_only = False
    fetch_status = "success"
    doc_warnings: list[str] = []
    cleaned_search_answer = clean_text(search_answer or "")

    if len(content) < MIN_TAVILY_CONTENT_CHARACTERS and cleaned_search_answer:
        answer_context = f"Tavily answer: {cleaned_search_answer}"
        content = clean_text(
            "\n\n".join(part for part in [content or snippet, answer_context] if part)
        )
        snippet_only = True
        fetch_status = "tavily_only"

    if settings.live_evidence_crawl_enabled and len(content) < MIN_TAVILY_CONTENT_CHARACTERS:
        result.selected_for_crawl = True
        try:
            crawled_title, crawled_text, crawled_snippet, html_language, crawled_date = (
                await _crawl_result_url(client, result)
            )
            content = clean_text(crawled_text)
            if len(content) < settings.live_evidence_min_article_characters:
                snippet_only = True
                fetch_status = "snippet_fallback"
                doc_warnings.append(f"Crawled content too short ({len(content)} chars); using snippet.")
                content = clean_text(snippet or crawled_snippet or "")
            else:
                title = crawled_title or title
                snippet = crawled_snippet or content[:320]
                if crawled_date and not published_date:
                    published_date = crawled_date
        except Exception as exc:
            logger.warning("tavily_result_crawl_failed url=%s error=%s", result.url, exc)
            snippet_only = True
            fetch_status = "snippet_fallback"
            doc_warnings.append(f"Full content fetch failed ({exc}); using Tavily snippet.")
            content = clean_text(snippet or "")
    elif len(content) < settings.live_evidence_min_article_characters:
        snippet_only = True
        fetch_status = "tavily_only"
        content = clean_text(content or snippet or "")

    if not content and not snippet:
        return None

    domain = result.domain or _domain_from_url(result.url)
    language = detect_language(content or snippet)
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
        "content_hash": normalized_hash(content or snippet),
        "fetched_at": _utc_now_iso(),
        "final_url": result.url,
        "search_run_id": search_run_id,
        "snippet_only": snippet_only,
        "fetch_status": fetch_status,
        "published_date": published_date,
        "doc_warnings": doc_warnings,
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
        snippet_only=snippet_only,
        published_date=published_date,
        fetch_status=fetch_status,
        warnings=doc_warnings,
    )


async def _run_tavily_queries(query_texts: list[str]) -> tuple[list[TavilySearchResponse], list[str]]:
    responses: list[TavilySearchResponse] = []
    errors: list[str] = []
    timeout = min(settings.request_timeout_seconds, settings.live_evidence_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        results = await asyncio.gather(
            *[search_with_tavily(query, client=client) for query in query_texts],
            return_exceptions=True,
        )
    for result in results:
        if isinstance(result, Exception):
            logger.warning("tavily_query_failed error=%s", result)
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
    requires_freshness: bool = False,
) -> dict[str, Any]:
    """
    Run Tavily search for the claim, fetch/crawl selected pages,
    ingest as EvidenceSource rows, and return rich metadata.

    The returned dict includes 'evidence_documents' — a list of
    document dicts suitable for evidence_chunker.chunk_evidence_documents().
    """
    if not settings.live_evidence_enabled:
        return {
            "enabled": False,
            "attempted": False,
            "ingested_count": 0,
            "created_count": 0,
            "updated_count": 0,
            "channels": [],
            "evidence_documents": [],
            "pipeline_warnings": [],
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
            "evidence_documents": [],
            "pipeline_warnings": ["No valid search queries could be generated."],
        }

    logger.info(
        "live_evidence_start claim_text=%.80s query_count=%d",
        claim_text, len(query_texts),
    )

    tavily_responses, errors = await _run_tavily_queries(query_texts)
    normalized_results = dedupe_search_results(
        [
            result
            for response in tavily_responses
            for result in normalize_tavily_response(response)
        ]
    )
    cap = settings.max_sources_to_fetch
    normalized_results = select_reliable_matching_results(
        normalized_results,
        claim_text=claim_text,
        limit=cap,
        claim_texts=query_texts,
    )

    request_ids = [response.request_id for response in tavily_responses if response.request_id]
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

    timeout = min(settings.fetch_timeout_seconds, settings.live_evidence_timeout_seconds)
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
    pipeline_warnings: list[str] = []

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
            pipeline_warnings.append(f"Failed to process a search result: {type(result).__name__}")
            continue
        if result is not None:
            documents.append(result)
            if result.warnings:
                pipeline_warnings.extend(result.warnings)

    if errors:
        pipeline_warnings.extend([f"Tavily query failed: {e}" for e in errors])

    # Freshness warnings
    if requires_freshness:
        missing_dates = [d for d in documents if not d.published_date]
        if missing_dates:
            pipeline_warnings.append(
                f"{len(missing_dates)} of {len(documents)} source(s) have no publication date — "
                "freshness cannot be verified."
            )

    search_run.crawled_count = sum(1 for result in normalized_results if result.selected_for_crawl)
    selected_documents = documents[: settings.live_evidence_max_documents]

    if not selected_documents:
        await session.commit()
        pipeline_warnings.insert(0, "No evidence documents could be fetched from Tavily results.")
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
            "evidence_documents": [],
            "pipeline_warnings": pipeline_warnings,
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

    # Build evidence_documents list for evidence_chunker
    evidence_documents: list[dict[str, Any]] = []
    for item, doc in zip(items, selected_documents):
        evidence_documents.append({
            "source_id": str(item.id) if hasattr(item, "id") else str(item.url),
            "title": doc.title,
            "url": doc.url,
            "domain": doc.publisher,
            "published_date": doc.published_date,
            "text_content": doc.text_content,
            "snippet": doc.snippet,
            "snippet_only": doc.snippet_only,
            "fetch_status": doc.fetch_status,
            "trust_score": doc.metadata.get("trust_score"),
        })

    logger.info(
        "live_evidence_complete docs=%d created=%d updated=%d warnings=%d",
        len(selected_documents),
        created_count,
        updated_count,
        len(pipeline_warnings),
    )

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
        "evidence_documents": evidence_documents,
        "pipeline_warnings": pipeline_warnings,
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
