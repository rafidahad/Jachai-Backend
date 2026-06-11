from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator

from app.schemas.common import StrictBaseModel
from app.schemas.dashboard_schema import RecentClaimItemSchema
from app.schemas.health_schema import HealthResponseSchema


class DocsNarrativeSectionSchema(StrictBaseModel):
    title: StrictStr
    body: StrictStr
    bullets: list[StrictStr] = Field(default_factory=list)


class DocsPitchSectionSchema(StrictBaseModel):
    id: StrictStr
    title: StrictStr
    eyebrow: StrictStr
    body: StrictStr
    bullets: list[StrictStr] = Field(default_factory=list)


class DocsSectionSettingSchema(StrictBaseModel):
    slug: StrictStr
    title: StrictStr
    section_type: Literal["pitch", "technical"]
    icon: StrictStr | None = None
    sort_order: StrictInt
    is_visible: StrictBool = True


class DocsTeamMemberSchema(StrictBaseModel):
    id: StrictStr
    full_name: StrictStr
    role: StrictStr
    email: StrictStr
    profile_image_url: StrictStr | None = None


class DocsFeatureEntrySchema(StrictBaseModel):
    id: StrictStr
    name: StrictStr
    category: Literal["upcoming", "planned"]
    status: Literal["building", "planned", "research", "blocked"]
    description: StrictStr


class DocsDiagramSchema(StrictBaseModel):
    title: StrictStr
    caption: StrictStr
    nodes: list[StrictStr] = Field(default_factory=list)


class DocsTechStackGroupSchema(StrictBaseModel):
    id: StrictStr
    category: StrictStr
    items: list[StrictStr] = Field(default_factory=list)


class DocsExternalApiSchema(StrictBaseModel):
    id: StrictStr
    name: StrictStr
    purpose: StrictStr
    auth: StrictStr


class DocsExposedApiSchema(StrictBaseModel):
    id: StrictStr
    method: StrictStr
    path: StrictStr
    audience: StrictStr
    auth: StrictStr
    description: StrictStr


class DocsRoadmapSchema(StrictBaseModel):
    short_term: list[StrictStr] = Field(default_factory=list)
    mid_term: list[StrictStr] = Field(default_factory=list)
    long_term: list[StrictStr] = Field(default_factory=list)


class DocsChangelogEntrySchema(StrictBaseModel):
    version: StrictStr
    date: StrictStr
    summary: StrictStr
    highlights: list[StrictStr] = Field(default_factory=list)


class DocsContentSchema(StrictBaseModel):
    hero_tagline: StrictStr
    hero_title: StrictStr
    hero_summary: StrictStr
    team_name: StrictStr | None = None
    team_summary: StrictStr
    section_settings: list[DocsSectionSettingSchema] = Field(default_factory=list)
    section_markdown_overrides: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    pitch_deck: list[DocsPitchSectionSchema] = Field(default_factory=list)
    product_overview: DocsNarrativeSectionSchema
    feature_matrix: list[DocsFeatureEntrySchema] = Field(default_factory=list)
    architecture_diagram: DocsDiagramSchema
    data_flow_diagram: DocsDiagramSchema
    technology_stack: list[DocsTechStackGroupSchema] = Field(default_factory=list)
    api_used: list[DocsExternalApiSchema] = Field(default_factory=list)
    api_exposed: list[DocsExposedApiSchema] = Field(default_factory=list)
    data_layer: DocsNarrativeSectionSchema
    ai_layer: DocsNarrativeSectionSchema
    roadmap: DocsRoadmapSchema
    performance: DocsNarrativeSectionSchema
    security: DocsNarrativeSectionSchema
    analytics: DocsNarrativeSectionSchema
    team_members: list[DocsTeamMemberSchema] = Field(default_factory=list)
    changelog: list[DocsChangelogEntrySchema] = Field(default_factory=list)


class DocsLiveMetricSchema(StrictBaseModel):
    id: StrictStr
    label: StrictStr
    value: StrictStr
    detail: StrictStr
    status: Literal["neutral", "good", "warn", "bad", "info"] = "neutral"


class DocsLiveFeatureSchema(StrictBaseModel):
    id: StrictStr
    name: StrictStr
    status: Literal["live", "building", "attention", "planned"]
    detail: StrictStr
    source: StrictStr


class DocsAccessStatusSchema(StrictBaseModel):
    available: StrictBool
    visibility_enabled: StrictBool
    status: Literal["live", "scheduled", "hidden", "expired"]
    publish_start_at: datetime | None = None
    publish_end_at: datetime | None = None
    effective_end_at: datetime | None = None
    publish_duration_minutes: StrictInt | None = None
    window_label: StrictStr
    note: StrictStr


class DocsLiveSnapshotSchema(StrictBaseModel):
    last_synced_at: datetime
    metrics: list[DocsLiveMetricSchema] = Field(default_factory=list)
    live_features: list[DocsLiveFeatureSchema] = Field(default_factory=list)
    recent_claims: list[RecentClaimItemSchema] = Field(default_factory=list)
    health: HealthResponseSchema


class DocsVersionEntrySchema(StrictBaseModel):
    id: StrictStr
    version: StrictInt
    snapshot_type: StrictStr
    created_by: StrictStr
    summary: StrictStr
    created_at: datetime


class DocsPublicResponseSchema(StrictBaseModel):
    access: DocsAccessStatusSchema
    content: DocsContentSchema | None = None
    live_data: DocsLiveSnapshotSchema | None = None
    published_version: StrictInt
    last_published_at: datetime | None = None


class DocsAdminDocumentSchema(StrictBaseModel):
    access: DocsAccessStatusSchema
    draft_content: DocsContentSchema
    published_content: DocsContentSchema
    live_data: DocsLiveSnapshotSchema
    versions: list[DocsVersionEntrySchema] = Field(default_factory=list)
    draft_revision: StrictInt
    published_version: StrictInt
    updated_at: datetime
    last_published_at: datetime | None = None


class DocsAdminUpdateRequestSchema(StrictBaseModel):
    visibility_enabled: StrictBool
    publish_start_at: datetime | None = None
    publish_end_at: datetime | None = None
    publish_duration_minutes: StrictInt | None = None
    draft_content: DocsContentSchema
    actor: StrictStr | None = None

    @field_validator("publish_duration_minutes")
    @classmethod
    def validate_duration(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("publish_duration_minutes must be greater than zero.")
        return value


class DocsAdminPublishRequestSchema(StrictBaseModel):
    actor: StrictStr | None = None


class DocsLegacyConfigSchema(StrictBaseModel):
    is_enabled: StrictBool
    start_at: datetime | None = None
    end_at: datetime | None = None
    updated_at: datetime


class DocsLegacySectionSchema(StrictBaseModel):
    id: StrictStr
    slug: StrictStr
    title: StrictStr
    section_type: Literal["pitch", "technical"]
    content: StrictStr
    icon: StrictStr | None = None
    sort_order: StrictInt
    is_visible: StrictBool
    created_at: datetime
    updated_at: datetime


class DocsLegacySectionListResponseSchema(StrictBaseModel):
    items: list[DocsLegacySectionSchema] = Field(default_factory=list)


class DocsLegacyTeamMemberSchema(StrictBaseModel):
    id: StrictStr
    full_name: StrictStr
    role: StrictStr
    email: StrictStr | None = None
    profile_picture_url: StrictStr | None = None
    sort_order: StrictInt
    created_at: datetime
    updated_at: datetime


class DocsLegacyTeamListResponseSchema(StrictBaseModel):
    items: list[DocsLegacyTeamMemberSchema] = Field(default_factory=list)


class DocsLegacyLiveDataSchema(StrictBaseModel):
    total_claims: StrictInt
    total_verdicts_true: StrictInt
    total_verdicts_false: StrictInt
    total_verdicts_misleading: StrictInt
    total_verdicts_unknown: StrictInt
    total_sources: StrictInt
    total_clusters: StrictInt
    average_confidence: float
    claims_today: StrictInt
    top_language: StrictStr


class DocsLegacyPublicSchema(StrictBaseModel):
    config: DocsLegacyConfigSchema
    sections: list[DocsLegacySectionSchema] = Field(default_factory=list)
    team_members: list[DocsLegacyTeamMemberSchema] = Field(default_factory=list)
    live_data: DocsLegacyLiveDataSchema


class DocsLegacyConfigUpdateRequestSchema(StrictBaseModel):
    is_enabled: StrictBool | None = None
    start_at: datetime | None = None
    end_at: datetime | None = None
    clear_schedule: StrictBool | None = None


class DocsLegacySectionUpdateRequestSchema(StrictBaseModel):
    title: StrictStr | None = None
    content: StrictStr | None = None
    icon: StrictStr | None = None
    sort_order: StrictInt | None = None
    is_visible: StrictBool | None = None


class DocsLegacySectionReorderItemSchema(StrictBaseModel):
    id: StrictStr
    sort_order: StrictInt


class DocsLegacySectionReorderRequestSchema(StrictBaseModel):
    order: list[DocsLegacySectionReorderItemSchema] = Field(default_factory=list)


class DocsLegacyTeamMemberCreateRequestSchema(StrictBaseModel):
    full_name: StrictStr
    role: StrictStr
    email: StrictStr | None = None
    profile_picture_url: StrictStr | None = None
    sort_order: StrictInt | None = None


class DocsLegacyTeamMemberUpdateRequestSchema(StrictBaseModel):
    full_name: StrictStr | None = None
    role: StrictStr | None = None
    email: StrictStr | None = None
    profile_picture_url: StrictStr | None = None
    sort_order: StrictInt | None = None
