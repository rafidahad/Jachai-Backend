from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.services.trusted_publisher_registry import trusted_google_review_sites

logger = get_logger(__name__)

GOOGLE_FACT_CHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
LANGUAGE_CODE_MAP = {
    "Bangla": "bn",
    "English": "en",
    "Hindi": "hi",
    "Banglish": "bn",
    "Hinglish": "hi",
    "Mixed": "en",
    "Unknown": None,
}


@dataclass(slots=True)
class GoogleFactCheckMatch:
    review_url: str
    claim_text: str
    review_title: str
    publisher_name: str
    publisher_site: str | None
    language_code: str | None
    textual_rating: str | None
    review_date: str | None
    matched_by_site_filter: str | None = None


def _parse_claim_reviews(payload: dict[str, Any], *, matched_by_site_filter: str | None) -> list[GoogleFactCheckMatch]:
    matches: list[GoogleFactCheckMatch] = []
    for claim in payload.get("claims", []):
        if not isinstance(claim, dict):
            continue
        claim_text = str(claim.get("text") or "").strip()
        language_code = str(claim.get("claimReviewLanguageCode") or "").strip() or None
        reviews = claim.get("claimReview") or []
        if not isinstance(reviews, list):
            continue
        for review in reviews:
            if not isinstance(review, dict):
                continue
            review_url = str(review.get("url") or "").strip()
            if not review_url:
                continue
            publisher = review.get("publisher") or {}
            publisher_name = str(publisher.get("name") or "").strip() or str(review.get("publisherName") or "").strip()
            publisher_site = str(publisher.get("site") or "").strip() or None
            review_title = str(review.get("title") or "").strip() or claim_text
            textual_rating = str(review.get("textualRating") or "").strip() or None
            review_date = str(review.get("reviewDate") or "").strip() or None
            matches.append(
                GoogleFactCheckMatch(
                    review_url=review_url,
                    claim_text=claim_text,
                    review_title=review_title,
                    publisher_name=publisher_name or publisher_site or "Fact check source",
                    publisher_site=publisher_site,
                    language_code=language_code,
                    textual_rating=textual_rating,
                    review_date=review_date,
                    matched_by_site_filter=matched_by_site_filter,
                )
            )
    return matches


async def search_google_fact_checks(query: str, *, language: str | None = None) -> list[GoogleFactCheckMatch]:
    if not settings.google_fact_check_enabled or not settings.google_fact_check_api_key:
        return []

    language_code = LANGUAGE_CODE_MAP.get(language or "", None)
    params_list: list[tuple[dict[str, Any], str | None]] = [
        (
            {
                "query": query,
                "languageCode": language_code,
                "pageSize": settings.live_evidence_max_google_results,
                "key": settings.google_fact_check_api_key,
            },
            None,
        )
    ]

    for review_site in trusted_google_review_sites()[:2]:
        params_list.append(
            (
                {
                    "query": query,
                    "languageCode": language_code,
                    "reviewPublisherSiteFilter": review_site,
                    "pageSize": max(2, settings.live_evidence_max_google_results // 2),
                    "key": settings.google_fact_check_api_key,
                },
                review_site,
            )
        )

    deduped: dict[str, GoogleFactCheckMatch] = {}
    timeout = min(settings.request_timeout_seconds, settings.live_evidence_timeout_seconds)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for params, matched_by_site_filter in params_list:
            params = {key: value for key, value in params.items() if value not in (None, "")}
            try:
                response = await client.get(GOOGLE_FACT_CHECK_ENDPOINT, params=params)
                response.raise_for_status()
                payload = response.json()
            except Exception:
                logger.exception(
                    "google_fact_check_lookup_failed site_filter=%s",
                    matched_by_site_filter or "none",
                )
                continue

            for match in _parse_claim_reviews(payload, matched_by_site_filter=matched_by_site_filter):
                deduped.setdefault(match.review_url, match)

    return list(deduped.values())
