from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.source_schema import SourceIngestItemSchema
from app.services.google_fact_check_service import GoogleFactCheckMatch, search_google_fact_checks
from app.services.language_service import detect_language
from app.services.search_service import SearchResult, search_general_web
from app.services.source_service import ingest_sources
from app.services.text_cleaning_service import clean_text
from app.services.trusted_publisher_registry import (
    TrustedPublisherCatalog,
    get_catalog_for_url,
    hostname_matches_trusted_domain,
    is_trusted_url,
    iter_trusted_catalogs,
)

logger = get_logger(__name__)

TOKEN_RE = re.compile(r"[0-9A-Za-z\u0980-\u09FF\u0900-\u097F]{3,}")
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


@dataclass(slots=True)
class LinkCandidate:
    url: str
    anchor_text: str
    score: float


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in DROP_QUERY_KEYS
    ]
    normalized_parts = parts._replace(query=urlencode(filtered_query, doseq=True), fragment="")
    return urlunsplit(normalized_parts)


def _tokenize(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(value)}


def _token_overlap_score(query_tokens: set[str], value: str) -> float:
    if not query_tokens:
        return 0.0
    candidate_tokens = _tokenize(value)
    if not candidate_tokens:
        return 0.0
    intersection = len(query_tokens & candidate_tokens)
    if not intersection:
        return 0.0
    return intersection / math.sqrt(len(query_tokens) * len(candidate_tokens))


def _build_request_headers() -> dict[str, str]:
    return {
        "User-Agent": "JachAI/0.1 (+https://github.com/openai)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8,bn;q=0.7,hi;q=0.6",
    }


def _normalize_html_language(value: str | None) -> str | None:
    if not value:
        return None
    prefix = value.split("-", 1)[0].strip().lower()
    return HTML_LANGUAGE_MAP.get(prefix)


def _catalogs_for_language(language: str) -> list[TrustedPublisherCatalog]:
    normalized_language = (language or "").strip()
    selected: list[TrustedPublisherCatalog] = []
    for catalog in iter_trusted_catalogs():
        if catalog.region == "bangladesh":
            if normalized_language in {"Bangla", "Banglish", "Hindi", "Hinglish", "Mixed", "Unknown", ""}:
                selected.append(catalog)
            continue

        selected.append(catalog)
    return selected


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


async def _fetch_document(
    client: httpx.AsyncClient,
    url: str,
    *,
    catalog: TrustedPublisherCatalog | None,
    base_metadata: dict[str, Any],
) -> LiveEvidenceDocument | None:
    normalized_url = _normalize_url(url)
    if not is_trusted_url(normalized_url):
        return None

    try:
        response = await client.get(normalized_url, headers=_build_request_headers(), follow_redirects=True)
        response.raise_for_status()
    except Exception:
        logger.exception("trusted_evidence_fetch_failed url=%s", normalized_url)
        return None

    final_url = _normalize_url(str(response.url))
    final_catalog = get_catalog_for_url(final_url) or catalog
    if final_catalog is None:
        return None

    title, text_content, description, html_language = _article_text_from_html(response.text)
    if len(text_content) < settings.live_evidence_min_article_characters:
        return None

    snippet = description or text_content[:280]
    if len(snippet) > 320:
        snippet = snippet[:317].rstrip() + "..."

    language = detect_language(text_content)
    if not language or language == "Unknown":
        language = _normalize_html_language(html_language) or "Unknown"
    metadata = {
        **base_metadata,
        "fetched_at": _utc_now_iso(),
        "final_url": final_url,
        "catalog": final_catalog.name,
        "publisher_domain": final_catalog.base_domain,
    }
    return LiveEvidenceDocument(
        title=title or text_content[:120],
        url=final_url,
        publisher=final_catalog.publisher,
        language=language,
        source_type=final_catalog.source_type,
        snippet=clean_text(snippet),
        text_content=text_content,
        metadata=metadata,
    )


def _extract_same_domain_links(
    html: str,
    *,
    listing_url: str,
    catalog: TrustedPublisherCatalog,
    query_tokens: set[str],
) -> list[LinkCandidate]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: dict[str, LinkCandidate] = {}
    for anchor in soup.find_all("a", href=True):
        href = clean_text(str(anchor.get("href") or ""))
        if not href:
            continue
        absolute_url = _normalize_url(urljoin(listing_url, href))
        parsed = urlsplit(absolute_url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if not hostname_matches_trusted_domain(parsed.hostname, catalog.base_domain):
            continue
        if any(part in absolute_url for part in ("/tag/", "/author/", "/category/", "/video/", "/contact", "/about")):
            continue
        anchor_text = clean_text(anchor.get_text(" ", strip=True))
        if not anchor_text:
            continue
        score = _token_overlap_score(query_tokens, f"{anchor_text} {absolute_url}")
        existing = candidates.get(absolute_url)
        candidate = LinkCandidate(url=absolute_url, anchor_text=anchor_text, score=score)
        if existing is None or candidate.score > existing.score:
            candidates[absolute_url] = candidate

    ordered = sorted(candidates.values(), key=lambda item: (item.score, len(item.anchor_text)), reverse=True)
    if ordered:
        return ordered[: settings.live_evidence_max_listing_links_per_catalog]
    return []


async def _discover_catalog_documents(
    client: httpx.AsyncClient,
    *,
    catalog: TrustedPublisherCatalog,
    query_text: str,
) -> list[LiveEvidenceDocument]:
    query_tokens = _tokenize(query_text)
    listing_responses = await asyncio.gather(
        *[
            client.get(url, headers=_build_request_headers(), follow_redirects=True)
            for url in catalog.listing_urls
        ],
        return_exceptions=True,
    )

    link_candidates: dict[str, LinkCandidate] = {}
    for listing_url, result in zip(catalog.listing_urls, listing_responses, strict=False):
        if isinstance(result, Exception):
            logger.warning("trusted_listing_fetch_failed url=%s error=%s", listing_url, result)
            continue
        if result.status_code >= 400:
            continue
        for candidate in _extract_same_domain_links(
            result.text,
            listing_url=listing_url,
            catalog=catalog,
            query_tokens=query_tokens,
        ):
            existing = link_candidates.get(candidate.url)
            if existing is None or candidate.score > existing.score:
                link_candidates[candidate.url] = candidate

    ordered_candidates = sorted(
        link_candidates.values(),
        key=lambda item: (item.score, len(item.anchor_text)),
        reverse=True,
    )[: settings.live_evidence_max_listing_links_per_catalog]

    documents = await asyncio.gather(
        *[
            _fetch_document(
                client,
                candidate.url,
                catalog=catalog,
                base_metadata={
                    "seeded_by": "live_trusted_lookup",
                    "source_channel": "trusted_catalog",
                    "match_score": round(candidate.score, 4),
                    "anchor_text": candidate.anchor_text,
                },
            )
            for candidate in ordered_candidates
        ],
        return_exceptions=True,
    )

    ranked: list[tuple[float, LiveEvidenceDocument]] = []
    for result in documents:
        if isinstance(result, Exception) or result is None:
            continue
        score = _token_overlap_score(query_tokens, f"{result.title} {result.snippet} {result.text_content[:800]}")
        if score <= 0:
            continue
        ranked.append((score, result))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in ranked[: settings.live_evidence_max_documents_per_catalog]]


async def _documents_from_google_matches(
    client: httpx.AsyncClient,
    *,
    matches: list[GoogleFactCheckMatch],
) -> list[LiveEvidenceDocument]:
    documents = await asyncio.gather(
        *[
            _fetch_document(
                client,
                match.review_url,
                catalog=get_catalog_for_url(match.review_url),
                base_metadata={
                    "seeded_by": "live_trusted_lookup",
                    "source_channel": "google_fact_check_api",
                    "matched_claim": match.claim_text,
                    "matched_language_code": match.language_code,
                    "publisher_site": match.publisher_site,
                    "review_title": match.review_title,
                    "textual_rating": match.textual_rating,
                    "review_date": match.review_date,
                    "site_filter": match.matched_by_site_filter,
                },
            )
            for match in matches
            if is_trusted_url(match.review_url)
        ],
        return_exceptions=True,
    )
    return [document for document in documents if isinstance(document, LiveEvidenceDocument)]


async def _documents_from_search_results(
    client: httpx.AsyncClient,
    *,
    results: list[SearchResult],
) -> list[LiveEvidenceDocument]:
    documents = await asyncio.gather(
        *[
            _fetch_document(
                client,
                result.url,
                catalog=get_catalog_for_url(result.url),
                base_metadata={
                    "seeded_by": "live_trusted_lookup",
                    "source_channel": "general_search_api",
                    "search_provider": result.provider,
                    "search_title": result.title,
                    "search_snippet": result.snippet,
                },
            )
            for result in results
            if is_trusted_url(result.url)
        ],
        return_exceptions=True,
    )
    return [document for document in documents if isinstance(document, LiveEvidenceDocument)]


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
            "google_match_count": 0,
            "general_search_result_count": 0,
            "normalized_hash": normalized_hash,
            "documents": [],
        }
    channels = ["google_fact_check_api"]
    if settings.general_search_enabled:
        channels.append("general_search_api")
    channels.append("trusted_catalog")

    timeout = min(settings.live_evidence_timeout_seconds, settings.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        google_match_results = await asyncio.gather(
            *[search_google_fact_checks(query, language=language) for query in query_texts],
            return_exceptions=True,
        )
        google_matches_by_url: dict[str, GoogleFactCheckMatch] = {}
        for result in google_match_results:
            if isinstance(result, Exception):
                logger.warning("google_fact_check_query_failed error=%s", result)
                continue
            for match in result:
                google_matches_by_url.setdefault(match.review_url, match)
        google_matches = list(google_matches_by_url.values())
        google_documents = await _documents_from_google_matches(client, matches=google_matches)

        general_search_results_by_url: dict[str, SearchResult] = {}
        if settings.general_search_enabled:
            general_search_results_nested = await asyncio.gather(
                *[search_general_web(query, language=language) for query in query_texts],
                return_exceptions=True,
            )
            for result in general_search_results_nested:
                if isinstance(result, Exception):
                    logger.warning("general_search_query_failed error=%s", result)
                    continue
                for item in result:
                    general_search_results_by_url.setdefault(item.url, item)
        general_search_results = list(general_search_results_by_url.values())
        general_documents = await _documents_from_search_results(client, results=general_search_results)

        catalog_documents_nested = await asyncio.gather(
            *[
                _discover_catalog_documents(client, catalog=catalog, query_text=query_texts[0])
                for catalog in _catalogs_for_language(language)
            ],
            return_exceptions=True,
        )

    catalog_documents: list[LiveEvidenceDocument] = []
    for result in catalog_documents_nested:
        if isinstance(result, Exception):
            logger.warning("trusted_catalog_discovery_failed error=%s", result)
            continue
        catalog_documents.extend(result)

    deduped_documents: dict[str, LiveEvidenceDocument] = {}
    for document in [*google_documents, *general_documents, *catalog_documents]:
        deduped_documents.setdefault(document.url, document)

    selected_documents = list(deduped_documents.values())[: settings.live_evidence_max_documents]
    if not selected_documents:
        return {
            "enabled": True,
            "attempted": True,
            "ingested_count": 0,
            "created_count": 0,
            "updated_count": 0,
            "channels": channels,
            "search_queries": query_texts,
            "google_match_count": len(google_matches),
            "general_search_result_count": len(general_search_results),
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
        "channels": channels,
        "search_queries": query_texts,
        "google_match_count": len(google_matches),
        "general_search_result_count": len(general_search_results),
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
