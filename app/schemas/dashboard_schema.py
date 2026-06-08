from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import StrictInt, StrictStr, field_validator

from app.schemas.common import StrictBaseModel
from app.schemas.verdict_schema import PIPELINE_TO_LEGACY_VERDICT, LanguageLabel, VerdictLabel

_LEGACY_VERDICTS: frozenset[str] = frozenset(VerdictLabel.__args__)  # type: ignore[attr-defined]


class SummaryMetricSchema(StrictBaseModel):
    total_claims: StrictInt
    claims_today: StrictInt
    reviewed_claims: StrictInt
    total_sources: StrictInt
    total_clusters: StrictInt
    average_confidence: float
    top_language: StrictStr
    ocr_submissions: StrictInt


class DistributionItemSchema(StrictBaseModel):
    label: StrictStr
    count: StrictInt


class ClaimsOverTimeItemSchema(StrictBaseModel):
    date: date
    count: StrictInt


class RecentClaimItemSchema(StrictBaseModel):
    id: StrictStr
    verdict: VerdictLabel
    language: LanguageLabel | StrictStr
    review_status: Literal["pending", "reviewed", "flagged"]
    created_at: datetime
    share_summary: StrictStr

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, v: object) -> object:
        if isinstance(v, str):
            if v in _LEGACY_VERDICTS:
                return v
            if v in PIPELINE_TO_LEGACY_VERDICT:
                return PIPELINE_TO_LEGACY_VERDICT[v]
            return "Not Enough Evidence"
        return v


class DashboardSummaryResponseSchema(StrictBaseModel):
    metrics: SummaryMetricSchema


class DashboardDistributionResponseSchema(StrictBaseModel):
    items: list[DistributionItemSchema]


class DashboardClaimsOverTimeResponseSchema(StrictBaseModel):
    items: list[ClaimsOverTimeItemSchema]


class DashboardRecentClaimsResponseSchema(StrictBaseModel):
    items: list[RecentClaimItemSchema]
