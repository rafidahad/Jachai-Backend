from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
from app.models.docs_document import DocsDocument, DocsDocumentVersion
from app.schemas.docs_schema import (
    DocsAccessStatusSchema,
    DocsAdminDocumentSchema,
    DocsAdminUpdateRequestSchema,
    DocsChangelogEntrySchema,
    DocsContentSchema,
    DocsDiagramSchema,
    DocsExposedApiSchema,
    DocsExternalApiSchema,
    DocsFeatureEntrySchema,
    DocsLiveFeatureSchema,
    DocsLiveMetricSchema,
    DocsLiveSnapshotSchema,
    DocsLegacyConfigSchema,
    DocsLegacyLiveDataSchema,
    DocsLegacyPublicSchema,
    DocsLegacySectionListResponseSchema,
    DocsLegacySectionSchema,
    DocsLegacyTeamListResponseSchema,
    DocsLegacyTeamMemberSchema,
    DocsNarrativeSectionSchema,
    DocsPitchSectionSchema,
    DocsPublicResponseSchema,
    DocsRoadmapSchema,
    DocsSectionSettingSchema,
    DocsTeamMemberSchema,
    DocsTechStackGroupSchema,
    DocsVersionEntrySchema,
)
from app.services.dashboard_service import get_recent_claims, get_summary_metrics, get_verdict_distribution
from app.services.health_service import get_system_health
from app.utils.errors import AppError

DOCS_SLUG = "main"
SHOWCASE_TIMEZONE = ZoneInfo("Asia/Dhaka")
_docs_tables_ready = False
_docs_tables_lock = asyncio.Lock()


def _format_window(value: datetime | None) -> str:
    if value is None:
        return ""
    localized = value.astimezone(SHOWCASE_TIMEZONE)
    return localized.strftime("%b %d, %Y %I:%M %p %Z")


def _default_publish_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    current = (now or datetime.now(SHOWCASE_TIMEZONE)).astimezone(SHOWCASE_TIMEZONE)
    year = current.year
    start = datetime(year, 6, 10, 0, 0, tzinfo=SHOWCASE_TIMEZONE)
    end = datetime(year, 6, 14, 23, 59, tzinfo=SHOWCASE_TIMEZONE)
    if current > end:
        year += 1
        start = datetime(year, 6, 10, 0, 0, tzinfo=SHOWCASE_TIMEZONE)
        end = datetime(year, 6, 14, 23, 59, tzinfo=SHOWCASE_TIMEZONE)
    return start, end


def _default_section_settings() -> list[DocsSectionSettingSchema]:
    definitions = [
        ("problem", "Problem", "pitch", "warning"),
        ("solution", "Solution", "pitch", "lightbulb"),
        ("why-now", "Why Now", "pitch", "schedule"),
        ("product-demo", "Product Demo", "pitch", "play_circle"),
        ("market-opportunity", "Market Opportunity", "pitch", "public"),
        ("business-model", "Business Model", "pitch", "payments"),
        ("traction", "Traction", "pitch", "trending_up"),
        ("competition", "Competition", "pitch", "compare_arrows"),
        ("unique-advantage", "Unique Advantage", "pitch", "diamond"),
        ("go-to-market", "Go-To-Market", "pitch", "campaign"),
        ("team", "Team", "pitch", "groups"),
        ("vision", "Vision", "pitch", "visibility"),
        ("product-overview", "Product Overview", "technical", "fact_check"),
        ("feature-matrix", "Feature Matrix", "technical", "grid_view"),
        ("architecture", "Architecture", "technical", "hub"),
        ("data-flow", "Data Flow", "technical", "account_tree"),
        ("tech-stack", "Technology Stack", "technical", "layers"),
        ("api-documentation", "API Documentation", "technical", "api"),
        ("data-layer", "Data Layer", "technical", "database"),
        ("ai-layer", "AI Layer", "technical", "neurology"),
        ("roadmap", "Roadmap", "technical", "route"),
        ("performance", "Performance & Scalability", "technical", "speed"),
        ("security", "Security", "technical", "shield_lock"),
        ("analytics", "Analytics", "technical", "insights"),
        ("changelog", "Changelog", "technical", "history"),
    ]
    return [
        DocsSectionSettingSchema(
            slug=slug,
            title=title,
            section_type=section_type,
            icon=icon,
            sort_order=index,
            is_visible=True,
        )
        for index, (slug, title, section_type, icon) in enumerate(definitions)
    ]


def build_default_docs_content() -> DocsContentSchema:
    today = datetime.now(SHOWCASE_TIMEZONE).date().isoformat()
    return DocsContentSchema(
        hero_tagline="Live Product Dossier",
        hero_title="JachAI turns rumor verification into a real-time operating system for South Asia.",
        hero_summary=(
            "JachAI verifies text, image, and URL claims across Bangla, English, Hindi, and mixed-language signals. "
            "It combines live web evidence, retrieval, reranking, verdict generation, clustering, and an operator console "
            "so judges, investors, and technical reviewers can understand both the product and the machinery behind it."
        ),
        team_name="JachAI Core Team",
        team_summary=(
            "Publish named team profiles here before the judging or investor window. Cards stay uniform automatically, "
            "and missing headshots fall back to generated initials."
        ),
        section_settings=_default_section_settings(),
        pitch_deck=[
            DocsPitchSectionSchema(
                id="problem",
                title="Problem",
                eyebrow="YC Deck",
                body=(
                    "Misinformation in South Asia spreads through screenshots, chat apps, short links, and mixed-language posts faster "
                    "than newsroom or civic teams can respond. Existing tools are either generic chatbots, single-language fact-checkers, "
                    "or workflows that break when claims arrive as images and forwarded messages."
                ),
                bullets=[
                    "Rumors move through Bangla, English, Hindi, Banglish, and Hinglish.",
                    "Manual verification is too slow for community moderators and fact-check desks.",
                    "Operators need both verdicts and a live narrative view, not just one-off answers.",
                ],
            ),
            DocsPitchSectionSchema(
                id="solution",
                title="Solution",
                eyebrow="YC Deck",
                body=(
                    "JachAI is a multilingual, multimodal verification stack that accepts suspicious text, screenshots, and URLs; "
                    "extracts the core claim; retrieves trusted evidence; produces an evidence-grounded verdict; and groups related "
                    "claims into rumor clusters for operator follow-up."
                ),
                bullets=[
                    "Public intake for text, image, and URL claims.",
                    "Admin console for sources, clusters, health, and documentation publishing.",
                    "Automation layer through n8n for Telegram, alerts, and scheduled operations.",
                ],
            ),
            DocsPitchSectionSchema(
                id="why-now",
                title="Why Now",
                eyebrow="YC Deck",
                body=(
                    "Election cycles, generative AI, and mobile-first media have raised the cost of slow verification. Teams need a system "
                    "that can read screenshots, search live evidence, and explain results in the same languages where rumors spread."
                ),
                bullets=[
                    "AI-generated and screenshot-based rumors are harder to triage manually.",
                    "Regional-language moderation tooling is still underserved.",
                    "Judges and partners increasingly expect live, inspectable AI systems rather than opaque demos.",
                ],
            ),
            DocsPitchSectionSchema(
                id="product-demo",
                title="Product Demo",
                eyebrow="YC Deck",
                body=(
                    "A user submits a claim. JachAI cleans and masks the input, runs OCR when needed, extracts the central claim, "
                    "retrieves live and indexed evidence, reranks the context, generates a verdict with explanation, then stores the result "
                    "for the public result page, admin dashboard, and rumor intelligence layer."
                ),
                bullets=[
                    "Submission -> verification job -> evidence retrieval -> verdict -> cluster assignment.",
                    "System health and recent-claim telemetry stay visible to operators.",
                    "This docs surface mirrors that same live application state.",
                ],
            ),
            DocsPitchSectionSchema(
                id="market-opportunity",
                title="Market Opportunity",
                eyebrow="YC Deck",
                body=(
                    "JachAI serves fact-checking desks, publishers, civic integrity teams, platform trust groups, and community moderators "
                    "who need verifiable answers plus a broader narrative map."
                ),
                bullets=[
                    "Newsrooms and fact-checking nonprofits.",
                    "Election monitoring and civic integrity operations.",
                    "Messaging-platform communities and channel administrators.",
                ],
            ),
            DocsPitchSectionSchema(
                id="business-model",
                title="Business Model",
                eyebrow="YC Deck",
                body=(
                    "The long-term model is a mix of workflow SaaS, API access, and organization-level intelligence tooling. "
                    "The current build already supports the product surfaces needed for demos, pilots, and embedded verification flows."
                ),
                bullets=[
                    "Operator console subscriptions for teams and desks.",
                    "Usage-based verification and retrieval APIs.",
                    "Channel integrations and premium monitoring workflows.",
                ],
            ),
            DocsPitchSectionSchema(
                id="traction",
                title="Traction",
                eyebrow="YC Deck",
                body=(
                    "Current traction is product and systems traction: an end-to-end working verification pipeline, live evidence retrieval, "
                    "admin observability, semantic rumor clustering, and an automation bridge for external channels."
                ),
                bullets=[
                    "Live text, image, and URL verification already shipped.",
                    "Rumor cluster monitoring and source registry are active.",
                    "n8n workflows extend the backend to Telegram and operational alerts.",
                ],
            ),
            DocsPitchSectionSchema(
                id="competition",
                title="Competition",
                eyebrow="YC Deck",
                body=(
                    "Manual fact-checking workflows, generic LLM assistants, and single-language verifiers each solve part of the problem. "
                    "JachAI is built around the real workflow: ingest messy regional inputs, verify against evidence, and monitor spread."
                ),
                bullets=[
                    "More operationally specific than a general chatbot.",
                    "More multilingual and multimodal than typical public verifiers.",
                    "Adds cluster intelligence rather than stopping at a single answer.",
                ],
            ),
            DocsPitchSectionSchema(
                id="unique-advantage",
                title="Unique Advantage",
                eyebrow="YC Deck",
                body=(
                    "JachAI combines multilingual intake, screenshot handling, live search, indexed retrieval, semantic clustering, "
                    "and operator-grade visibility in one stack designed for South Asian misinformation patterns."
                ),
                bullets=[
                    "Built for Bangla + English + Hindi + mixed-language usage.",
                    "Public proof surface and admin operations live on the same backend truth.",
                    "Can extend into external messaging channels without moving the core reasoning out of FastAPI.",
                ],
            ),
            DocsPitchSectionSchema(
                id="go-to-market",
                title="Go-To-Market",
                eyebrow="YC Deck",
                body=(
                    "Start with journalism, civic integrity, and moderation teams that already face daily rumor triage pressure. "
                    "Use public verification, private admin tools, and messaging-channel workflows to prove value in real operator loops."
                ),
                bullets=[
                    "Demo-led adoption for judges, pilots, and newsroom partnerships.",
                    "Channel integrations through n8n reduce deployment friction.",
                    "Documentation and live system transparency shorten evaluation cycles.",
                ],
            ),
            DocsPitchSectionSchema(
                id="team",
                title="Team",
                eyebrow="YC Deck",
                body=(
                    "This section is powered by the team profile cards below. Each card supports a consistent avatar frame, role, and contact "
                    "surface so judges and partners can quickly understand ownership across product, AI, backend, frontend, and operations."
                ),
                bullets=[
                    "Admin-managed showcase cards with uniform presentation.",
                    "Fallback avatars when profile images are missing.",
                    "Linked directly to the technical contributor section later in the page.",
                ],
            ),
            DocsPitchSectionSchema(
                id="vision",
                title="Vision",
                eyebrow="YC Deck",
                body=(
                    "JachAI aims to become trust infrastructure for multilingual information ecosystems: a system that can verify, explain, "
                    "cluster, and operationalize rumor intelligence wherever misinformation actually moves."
                ),
                bullets=[
                    "From reactive fact-checking to continuous narrative monitoring.",
                    "From website-only interaction to channel-native verification experiences.",
                    "From static decks to live, inspectable product truth.",
                ],
            ),
        ],
        product_overview=DocsNarrativeSectionSchema(
            title="Product Overview",
            body=(
                "JachAI is a public verification product and an internal intelligence console. End users submit suspicious claims through a "
                "web interface. Operators review verdicts, sources, and cluster activity from the admin console. The same stack can power "
                "external automation via n8n."
            ),
            bullets=[
                "Target users: fact-checking teams, newsroom operators, civic moderators, and community admins.",
                "Core use cases: verify viral claims, review evidence, monitor repeating narratives, and trigger downstream workflows.",
                "Output: verdict, confidence, explanation, share summary, supporting evidence, and rumor-cluster context.",
            ],
        ),
        feature_matrix=[
            DocsFeatureEntrySchema(
                id="feature-telegram",
                name="Telegram workflow bridge",
                category="upcoming",
                status="building",
                description="n8n-mediated intake and response loops for Telegram-based verification requests.",
            ),
            DocsFeatureEntrySchema(
                id="feature-manual-review",
                name="Reviewer escalation workflows",
                category="upcoming",
                status="building",
                description="Tighter handoff between automated verdicting and manual review queues for high-risk claims.",
            ),
            DocsFeatureEntrySchema(
                id="feature-reports",
                name="Scheduled executive digests",
                category="planned",
                status="planned",
                description="Daily and weekly summaries for operators, judges, and partner teams.",
            ),
            DocsFeatureEntrySchema(
                id="feature-platforms",
                name="Broader channel adapters",
                category="planned",
                status="research",
                description="Expansion from Telegram-focused automation toward WhatsApp, Messenger, and partner-specific connectors.",
            ),
        ],
        architecture_diagram=DocsDiagramSchema(
            title="Architecture Diagram",
            caption="Editable SVG view of the current application topology.",
            nodes=[
                "Public Next.js UI",
                "Admin Console",
                "FastAPI REST + Webhooks",
                "Verification Pipeline Services",
                "Redis Cache + Rate Limits",
                "PostgreSQL + pgvector",
                "External Search + Model Providers",
                "n8n Automation Layer",
            ],
        ),
        data_flow_diagram=DocsDiagramSchema(
            title="Data Flow Diagram",
            caption="From raw input to verdict, storage, and operator feedback.",
            nodes=[
                "Input: text, image, URL",
                "Cleaning + PII Masking + OCR",
                "Claim Extraction + Query Generation",
                "Live Search + Crawl + Retrieval",
                "Rerank + Reasoning + Verdict",
                "Storage + Cluster Assignment",
                "Public Result + Admin Review + Feedback",
            ],
        ),
        technology_stack=[
            DocsTechStackGroupSchema(
                id="stack-frontend",
                category="Frontend",
                items=["Next.js 16", "React 19", "Tailwind CSS 4", "Recharts", "Server + client app router"],
            ),
            DocsTechStackGroupSchema(
                id="stack-backend",
                category="Backend",
                items=["FastAPI", "SQLAlchemy 2", "Pydantic 2", "Async background jobs", "Structured health endpoints"],
            ),
            DocsTechStackGroupSchema(
                id="stack-data",
                category="Data + Storage",
                items=["PostgreSQL", "pgvector", "Redis", "JSONB-backed document storage", "Audit logs + source registry"],
            ),
            DocsTechStackGroupSchema(
                id="stack-ai",
                category="AI + Retrieval",
                items=["BAAI/bge-m3 embeddings", "Gemini reasoning", "NVIDIA reranking + query models", "Kimi OCR fallback", "Tavily live search"],
            ),
            DocsTechStackGroupSchema(
                id="stack-infra",
                category="Infra + Automation",
                items=["Docker", "Dockploy on VPS", "n8n", "Backend API proxying through Next.js", "Session-based admin auth"],
            ),
        ],
        api_used=[
            DocsExternalApiSchema(
                id="api-tavily",
                name="Tavily Search API",
                purpose="Live evidence discovery for current web results and news context.",
                auth="API key",
            ),
            DocsExternalApiSchema(
                id="api-gemini",
                name="Google Gemini",
                purpose="Reasoning, extraction, and verdict generation fallback stack.",
                auth="API key",
            ),
            DocsExternalApiSchema(
                id="api-nvidia",
                name="NVIDIA Build / Inference Endpoints",
                purpose="Query generation, reranking, and optional model serving tasks.",
                auth="API key",
            ),
            DocsExternalApiSchema(
                id="api-telegram",
                name="Telegram via n8n",
                purpose="Channel-side automation and response delivery for bot workflows.",
                auth="Bot credential in n8n",
            ),
        ],
        api_exposed=[
            DocsExposedApiSchema(
                id="route-claims-text",
                method="POST",
                path="/api/v1/claims/text",
                audience="Public",
                auth="Rate limited",
                description="Submit raw text for verification and receive a verification job handle.",
            ),
            DocsExposedApiSchema(
                id="route-claims-image",
                method="POST",
                path="/api/v1/claims/image",
                audience="Public",
                auth="Rate limited",
                description="Submit an image claim with OCR support and optional context text.",
            ),
            DocsExposedApiSchema(
                id="route-claims-url",
                method="POST",
                path="/api/v1/claims/url",
                audience="Public",
                auth="Rate limited",
                description="Submit a URL and optional context for verification.",
            ),
            DocsExposedApiSchema(
                id="route-jobs",
                method="GET",
                path="/api/v1/verification-jobs/{job_id}",
                audience="Public",
                auth="None",
                description="Poll live verification progress for an active job.",
            ),
            DocsExposedApiSchema(
                id="route-dashboard-summary",
                method="GET",
                path="/api/v1/dashboard/summary",
                audience="Public",
                auth="None",
                description="Read the public metrics block used on the homepage and docs live dashboard.",
            ),
            DocsExposedApiSchema(
                id="route-admin",
                method="GET",
                path="/api/v1/admin/auth/session",
                audience="Admin",
                auth="Signed cookie session",
                description="Validate backend-admin access for protected console routes.",
            ),
        ],
        data_layer=DocsNarrativeSectionSchema(
            title="Data Layer",
            body=(
                "The data layer blends live search evidence with persistent internal records. Claims, sources, clusters, jobs, and docs "
                "content live in PostgreSQL. Vector similarity is handled by pgvector, while Redis supports caching, duplicate detection, "
                "live status, and rate limiting."
            ),
            bullets=[
                "Data sources: public web, crawled articles, operator-ingested evidence, and channel-triggered workflows.",
                "Normalization: input cleaning, language detection, PII masking, OCR fallback, and chunking before retrieval.",
                "Privacy handling: unnecessary personal identifiers are stripped before reasoning where possible.",
            ],
        ),
        ai_layer=DocsNarrativeSectionSchema(
            title="AI Layer",
            body=(
                "JachAI routes work across embeddings, live search, reranking, and reasoning models rather than relying on a single prompt "
                "hop. The design keeps retrieval and evidence selection explicit so the final verdict remains inspectable."
            ),
            bullets=[
                "Embeddings: BAAI/bge-m3 for multilingual retrieval and semantic clustering.",
                "Reasoning: Gemini-first flow with NVIDIA-backed query and rerank helpers, plus optional vision fallback.",
                "Explainability: verdict, confidence, explanation, reasoning trace, evidence snippets, and share summary are stored.",
            ],
        ),
        roadmap=DocsRoadmapSchema(
            short_term=[
                "Polish reviewer workflows and manual-escalation states.",
                "Expand docs authoring with richer previews and publishing history.",
                "Harden cluster evaluation and evidence freshness reporting.",
            ],
            mid_term=[
                "Deepen Telegram and external-channel automations.",
                "Introduce executive digests and partner-facing reporting views.",
                "Add broader narrative analytics across repeated claims and sources.",
            ],
            long_term=[
                "Become the operating layer for multilingual trust and rumor intelligence.",
                "Support cross-channel interventions, partner APIs, and organization workspaces.",
                "Extend from claim verification into continuous information-risk monitoring.",
            ],
        ),
        performance=DocsNarrativeSectionSchema(
            title="Performance & Scalability",
            body=(
                "The backend uses async request handling, cached dashboard summaries, Redis-backed rate limiting, and vector retrieval to "
                "keep verification responsive while preserving answer quality. Heavy content is loaded lazily on the docs surface, and the "
                "public docs page refreshes live state incrementally instead of forcing full reloads."
            ),
            bullets=[
                "Background verification jobs prevent user requests from blocking on full pipeline execution.",
                "Dashboard and docs telemetry reuse cached summary metrics where safe.",
                "Semantic clustering and evidence ingestion now avoid unnecessary recompute paths where possible.",
            ],
        ),
        security=DocsNarrativeSectionSchema(
            title="Security",
            body=(
                "Admin operations are protected by backend-issued signed cookies. Internal webhook entry points can require a separate API "
                "key, and the public claim surface is rate-limited. Evidence and verdict traces remain auditable inside the database."
            ),
            bullets=[
                "RBAC today is admin-session based rather than multi-role user accounts.",
                "Internal automation can authenticate via X-Internal-API-Key.",
                "PII reduction happens before downstream reasoning when possible.",
            ],
        ),
        analytics=DocsNarrativeSectionSchema(
            title="Analytics",
            body=(
                "The live analytics layer surfaces total claims, verification runs, review coverage, source inventory, rumor clusters, "
                "OCR usage, recent claim events, and runtime health. It is intentionally grounded in the backend's current telemetry rather "
                "than synthetic demo numbers."
            ),
            bullets=[
                "KPIs come directly from dashboard summary queries and health checks.",
                "Recent-claim feed doubles as an event log for this docs experience.",
                "This page updates from the same API surface used elsewhere in the product.",
            ],
        ),
        team_members=[],
        changelog=[
            DocsChangelogEntrySchema(
                version="v0.1",
                date=today,
                summary="Live documentation module introduced.",
                highlights=[
                    "Added public /docs route with scheduling and access control.",
                    "Added admin draft + publish workflow for product and technical documentation.",
                    "Connected docs metrics and recent activity to live backend state.",
                ],
            )
        ],
    )


async def ensure_docs_tables() -> None:
    global _docs_tables_ready
    if _docs_tables_ready:
        return
    async with _docs_tables_lock:
        if _docs_tables_ready:
            return
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda sync_connection: Base.metadata.create_all(
                    sync_connection,
                    tables=[DocsDocument.__table__, DocsDocumentVersion.__table__],
                )
            )
        _docs_tables_ready = True


def _validate_docs_content(raw_content: dict[str, object] | None) -> DocsContentSchema:
    fallback = build_default_docs_content()
    if not raw_content:
        return fallback
    try:
        return DocsContentSchema.model_validate(raw_content)
    except ValidationError:
        return fallback


def _effective_end_at(document: DocsDocument) -> datetime | None:
    if document.publish_end_at is not None:
        return document.publish_end_at
    if document.publish_start_at is not None and document.publish_duration_minutes:
        return document.publish_start_at + timedelta(minutes=document.publish_duration_minutes)
    return None


def _compute_access(document: DocsDocument, now: datetime | None = None) -> DocsAccessStatusSchema:
    current = now or datetime.now(SHOWCASE_TIMEZONE)
    effective_end = _effective_end_at(document)

    if not document.visibility_enabled:
        return DocsAccessStatusSchema(
            available=False,
            visibility_enabled=False,
            status="hidden",
            publish_start_at=document.publish_start_at,
            publish_end_at=document.publish_end_at,
            effective_end_at=effective_end,
            publish_duration_minutes=document.publish_duration_minutes,
            window_label="Public docs are currently disabled by an administrator.",
            note="Turn visibility on in the admin panel to expose /docs publicly.",
        )

    if document.publish_start_at and current < document.publish_start_at:
        return DocsAccessStatusSchema(
            available=False,
            visibility_enabled=True,
            status="scheduled",
            publish_start_at=document.publish_start_at,
            publish_end_at=document.publish_end_at,
            effective_end_at=effective_end,
            publish_duration_minutes=document.publish_duration_minutes,
            window_label=(
                f"Scheduled for {_format_window(document.publish_start_at)}"
                + (f" -> {_format_window(effective_end)}" if effective_end else "")
            ),
            note="This docs experience is scheduled but not yet public.",
        )

    if effective_end and current > effective_end:
        return DocsAccessStatusSchema(
            available=False,
            visibility_enabled=True,
            status="expired",
            publish_start_at=document.publish_start_at,
            publish_end_at=document.publish_end_at,
            effective_end_at=effective_end,
            publish_duration_minutes=document.publish_duration_minutes,
            window_label=f"Publishing window closed at {_format_window(effective_end)}",
            note="The public showcase window has expired.",
        )

    if document.publish_start_at or effective_end:
        label = (
            f"{_format_window(document.publish_start_at)} -> {_format_window(effective_end)}"
            if document.publish_start_at and effective_end
            else f"Live from {_format_window(document.publish_start_at)}"
            if document.publish_start_at
            else f"Live until {_format_window(effective_end)}"
        )
    else:
        label = "Always on until disabled."

    return DocsAccessStatusSchema(
        available=True,
        visibility_enabled=True,
        status="live",
        publish_start_at=document.publish_start_at,
        publish_end_at=document.publish_end_at,
        effective_end_at=effective_end,
        publish_duration_minutes=document.publish_duration_minutes,
        window_label=label,
        note="The docs experience is currently public.",
    )


def _validate_schedule(
    *,
    publish_start_at: datetime | None,
    publish_end_at: datetime | None,
    publish_duration_minutes: int | None,
) -> None:
    if publish_duration_minutes is not None and publish_start_at is None:
        raise AppError(
            status_code=422,
            code="DOCS_SCHEDULE_INVALID",
            message="Duration-based publishing requires a start date and time.",
        )
    if publish_start_at and publish_end_at and publish_end_at <= publish_start_at:
        raise AppError(
            status_code=422,
            code="DOCS_SCHEDULE_INVALID",
            message="Publish end time must be later than the start time.",
        )


async def _get_or_create_document(session: AsyncSession) -> DocsDocument:
    await ensure_docs_tables()
    statement = (
        select(DocsDocument)
        .options(selectinload(DocsDocument.versions))
        .where(DocsDocument.slug == DOCS_SLUG)
    )
    document = (await session.execute(statement)).scalar_one_or_none()
    if document is not None:
        return document

    default_content = build_default_docs_content()
    publish_start_at, publish_end_at = _default_publish_window()
    now = datetime.now(SHOWCASE_TIMEZONE)
    document = DocsDocument(
        slug=DOCS_SLUG,
        visibility_enabled=True,
        publish_start_at=publish_start_at,
        publish_end_at=publish_end_at,
        publish_duration_minutes=None,
        draft_content=default_content.model_dump(mode="json"),
        published_content=default_content.model_dump(mode="json"),
        draft_revision=1,
        published_version=1,
        created_by="system",
        updated_by="system",
        last_published_at=now,
    )
    session.add(document)
    await session.flush()
    session.add(
        DocsDocumentVersion(
            document_id=document.id,
            version=1,
            snapshot_type="published",
            created_by="system",
            summary="Initial public docs bootstrap",
            snapshot={
                "content": default_content.model_dump(mode="json"),
                "visibility_enabled": True,
                "publish_start_at": publish_start_at.isoformat(),
                "publish_end_at": publish_end_at.isoformat(),
                "publish_duration_minutes": None,
            },
        )
    )
    await session.commit()
    refreshed = await session.execute(statement)
    created = refreshed.scalar_one()
    return created


async def _build_live_snapshot(session: AsyncSession) -> DocsLiveSnapshotSchema:
    summary = await get_summary_metrics(session)
    recent_claims = await get_recent_claims(session, limit=6)
    health = await get_system_health()

    metrics = [
        DocsLiveMetricSchema(
            id="claims",
            label="Claims verified",
            value=str(summary.total_claims),
            detail="Total persisted claims handled by the backend.",
            status="info",
        ),
        DocsLiveMetricSchema(
            id="runs",
            label="Verification runs",
            value=str(summary.verification_runs),
            detail="Every completed or active verification workflow recorded so far.",
            status="neutral",
        ),
        DocsLiveMetricSchema(
            id="sources",
            label="Evidence sources",
            value=str(summary.total_sources),
            detail="Trusted or ingested evidence records available to retrieval.",
            status="good" if summary.total_sources > 0 else "warn",
        ),
        DocsLiveMetricSchema(
            id="clusters",
            label="Rumor clusters",
            value=str(summary.total_clusters),
            detail="Distinct narratives grouped by the backend clustering layer.",
            status="good" if summary.total_clusters > 0 else "warn",
        ),
        DocsLiveMetricSchema(
            id="reviewed",
            label="Reviewed claims",
            value=str(summary.reviewed_claims),
            detail="Claims explicitly reviewed in the admin workflow.",
            status="neutral",
        ),
        DocsLiveMetricSchema(
            id="confidence",
            label="Average confidence",
            value=f"{summary.average_confidence:.1f}%",
            detail="Average confidence across persisted claim outcomes.",
            status="good" if summary.average_confidence >= 70 else "warn",
        ),
    ]

    live_features = [
        DocsLiveFeatureSchema(
            id="live-claim-intake",
            name="Public claim intake",
            status="live" if health.status != "unavailable" else "attention",
            detail="Text, image, and URL routes are wired through the live verification pipeline.",
            source="Claims module",
        ),
        DocsLiveFeatureSchema(
            id="live-clusters",
            name="Rumor clustering",
            status="live" if summary.total_clusters > 0 else "building",
            detail="Semantic clustering groups related claims into operational narratives.",
            source="Rumor cluster service",
        ),
        DocsLiveFeatureSchema(
            id="live-sources",
            name="Evidence registry",
            status="live" if summary.total_sources > 0 else "building",
            detail="Persistent evidence source library connected to retrieval and admin review.",
            source="Sources module",
        ),
        DocsLiveFeatureSchema(
            id="live-search",
            name="Live search retrieval",
            status="live" if settings.tavily_enabled else "attention",
            detail="Current build uses Tavily-backed evidence discovery when configured.",
            source="Search service",
        ),
        DocsLiveFeatureSchema(
            id="live-admin-auth",
            name="Protected admin console",
            status="live" if settings.admin_auth_enabled else "attention",
            detail="Admin pages depend on backend-issued signed session cookies.",
            source="Admin auth",
        ),
        DocsLiveFeatureSchema(
            id="live-automation",
            name="Automation bridge",
            status="live" if bool(settings.internal_api_key) else "planned",
            detail="Webhook and internal API flows are ready for n8n-driven orchestration.",
            source="Webhooks + n8n",
        ),
    ]

    return DocsLiveSnapshotSchema(
        last_synced_at=datetime.now(SHOWCASE_TIMEZONE),
        metrics=metrics,
        live_features=live_features,
        recent_claims=recent_claims,
        health=health,
    )


async def _build_legacy_live_data(session: AsyncSession) -> DocsLegacyLiveDataSchema:
    summary = await get_summary_metrics(session)
    verdict_rows = await get_verdict_distribution(session)
    counts = {item.label: item.count for item in verdict_rows}
    return DocsLegacyLiveDataSchema(
        total_claims=summary.total_claims,
        total_verdicts_true=int(counts.get("Likely True", 0)),
        total_verdicts_false=int(counts.get("Likely False", 0)),
        total_verdicts_misleading=int(counts.get("Misleading", 0)),
        total_verdicts_unknown=int(counts.get("Not Enough Evidence", 0)),
        total_sources=summary.total_sources,
        total_clusters=summary.total_clusters,
        average_confidence=summary.average_confidence,
        claims_today=summary.claims_today,
        top_language=summary.top_language,
    )


def _version_entries(document: DocsDocument) -> list[DocsVersionEntrySchema]:
    return [
        DocsVersionEntrySchema(
            id=str(version.id),
            version=version.version,
            snapshot_type=version.snapshot_type,
            created_by=version.created_by,
            summary=version.summary,
            created_at=version.created_at,
        )
        for version in document.versions[:10]
    ]


def _markdown_block(body: str, bullets: list[str] | None = None) -> str:
    lines = [body.strip()]
    for bullet in bullets or []:
        lines.append(f"- {bullet}")
    return "\n\n".join(line for line in lines if line)


def _feature_matrix_markdown(content: DocsContentSchema) -> str:
    lines = [
        "Current live capabilities are reflected automatically below, while roadmap items stay editable through admin.",
        "",
        "| Capability | Track | Status | Description |",
        "| --- | --- | --- | --- |",
    ]
    for feature in content.feature_matrix:
        track = "Upcoming" if feature.category == "upcoming" else "Planned"
        status = (
            "In Progress"
            if feature.status == "building"
            else "Planned"
            if feature.status == "planned"
            else "Research"
            if feature.status == "research"
            else "Blocked"
        )
        lines.append(f"| {feature.name} | {track} | {status} | {feature.description} |")
    return "\n".join(lines)


def _api_documentation_markdown(content: DocsContentSchema) -> str:
    lines = [
        "### APIs Used",
        "| Provider | Purpose | Auth |",
        "| --- | --- | --- |",
    ]
    for api in content.api_used:
        lines.append(f"| {api.name} | {api.purpose} | {api.auth} |")

    lines.extend(
        [
            "",
            "### APIs Exposed",
            "| Method | Path | Audience | Auth | Description |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for endpoint in content.api_exposed:
        lines.append(
            f"| {endpoint.method} | {endpoint.path} | {endpoint.audience} | {endpoint.auth} | {endpoint.description} |"
        )
    return "\n".join(lines)


def _roadmap_markdown(content: DocsContentSchema) -> str:
    sections = [
        ("## In Progress (Current)", content.roadmap.short_term),
        ("## Planned (Mid Term)", content.roadmap.mid_term),
        ("## Planned (Long Term)", content.roadmap.long_term),
    ]
    lines: list[str] = []
    for heading, items in sections:
        lines.append(heading)
        for item in items:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).strip()


def _stack_markdown(content: DocsContentSchema) -> str:
    lines: list[str] = []
    for group in content.technology_stack:
        lines.append(f"### {group.category}")
        for item in group.items:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).strip()


def _diagram_markdown(diagram: DocsDiagramSchema) -> str:
    lines = [diagram.caption, "", "### Flow"]
    for node in diagram.nodes:
        lines.append(f"- {node}")
    return "\n".join(lines)


def _changelog_markdown(content: DocsContentSchema) -> str:
    lines: list[str] = []
    for entry in content.changelog:
        lines.append(f"## {entry.version} ({entry.date})")
        lines.append(entry.summary)
        for item in entry.highlights:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).strip()


def _section_content_by_slug(content: DocsContentSchema, slug: str) -> str:
    override = content.section_markdown_overrides.get(slug)
    if override:
        return override
    pitch_sections = {section.id: section for section in content.pitch_deck}
    if slug in pitch_sections:
        item = pitch_sections[slug]
        return _markdown_block(item.body, item.bullets)
    if slug == "product-overview":
        return _markdown_block(content.product_overview.body, content.product_overview.bullets)
    if slug == "feature-matrix":
        return _feature_matrix_markdown(content)
    if slug == "architecture":
        return _diagram_markdown(content.architecture_diagram)
    if slug == "data-flow":
        return _diagram_markdown(content.data_flow_diagram)
    if slug == "tech-stack":
        return _stack_markdown(content)
    if slug == "api-documentation":
        return _api_documentation_markdown(content)
    if slug == "data-layer":
        return _markdown_block(content.data_layer.body, content.data_layer.bullets)
    if slug == "ai-layer":
        return _markdown_block(content.ai_layer.body, content.ai_layer.bullets)
    if slug == "roadmap":
        return _roadmap_markdown(content)
    if slug == "performance":
        return _markdown_block(content.performance.body, content.performance.bullets)
    if slug == "security":
        return _markdown_block(content.security.body, content.security.bullets)
    if slug == "analytics":
        return _markdown_block(content.analytics.body, content.analytics.bullets)
    if slug == "team":
        return content.team_summary
    if slug == "changelog":
        return _changelog_markdown(content)
    return ""


def _legacy_sections(content: DocsContentSchema, updated_at: datetime) -> list[DocsLegacySectionSchema]:
    return [
        DocsLegacySectionSchema(
            id=setting.slug,
            slug=setting.slug,
            title=setting.title,
            section_type=setting.section_type,
            content=_section_content_by_slug(content, setting.slug),
            icon=setting.icon,
            sort_order=setting.sort_order,
            is_visible=setting.is_visible,
            created_at=updated_at,
            updated_at=updated_at,
        )
        for setting in sorted(content.section_settings, key=lambda item: item.sort_order)
    ]


def _legacy_team_members(content: DocsContentSchema, updated_at: datetime) -> list[DocsLegacyTeamMemberSchema]:
    return [
        DocsLegacyTeamMemberSchema(
            id=member.id,
            full_name=member.full_name,
            role=member.role,
            email=member.email,
            profile_picture_url=member.profile_image_url,
            sort_order=index,
            created_at=updated_at,
            updated_at=updated_at,
        )
        for index, member in enumerate(content.team_members)
    ]


async def get_public_docs(session: AsyncSession) -> DocsPublicResponseSchema:
    document = await _get_or_create_document(session)
    access = _compute_access(document)
    if not access.available:
        return DocsPublicResponseSchema(
            access=access,
            content=None,
            live_data=None,
            published_version=document.published_version,
            last_published_at=document.last_published_at,
        )

    return DocsPublicResponseSchema(
        access=access,
        content=_validate_docs_content(document.published_content),
        live_data=await _build_live_snapshot(session),
        published_version=document.published_version,
        last_published_at=document.last_published_at,
    )


async def get_admin_docs(session: AsyncSession) -> DocsAdminDocumentSchema:
    document = await _get_or_create_document(session)
    return DocsAdminDocumentSchema(
        access=_compute_access(document),
        draft_content=_validate_docs_content(document.draft_content),
        published_content=_validate_docs_content(document.published_content),
        live_data=await _build_live_snapshot(session),
        versions=_version_entries(document),
        draft_revision=document.draft_revision,
        published_version=document.published_version,
        updated_at=document.updated_at,
        last_published_at=document.last_published_at,
    )


async def save_docs_draft(
    session: AsyncSession,
    payload: DocsAdminUpdateRequestSchema,
) -> DocsAdminDocumentSchema:
    _validate_schedule(
        publish_start_at=payload.publish_start_at,
        publish_end_at=payload.publish_end_at,
        publish_duration_minutes=payload.publish_duration_minutes,
    )
    document = await _get_or_create_document(session)
    actor = (payload.actor or "Admin").strip() or "Admin"

    document.visibility_enabled = payload.visibility_enabled
    document.publish_start_at = payload.publish_start_at
    document.publish_end_at = payload.publish_end_at
    document.publish_duration_minutes = payload.publish_duration_minutes
    document.draft_content = payload.draft_content.model_dump(mode="json")
    document.draft_revision += 1
    document.updated_by = actor

    await session.commit()
    return await get_admin_docs(session)


async def publish_docs(session: AsyncSession, actor: str | None = None) -> DocsAdminDocumentSchema:
    document = await _get_or_create_document(session)
    access_actor = (actor or "Admin").strip() or "Admin"
    draft_content = _validate_docs_content(document.draft_content)
    document.published_content = draft_content.model_dump(mode="json")
    document.published_version += 1
    document.last_published_at = datetime.now(SHOWCASE_TIMEZONE)
    document.updated_by = access_actor

    session.add(
        DocsDocumentVersion(
            document_id=document.id,
            version=document.published_version,
            snapshot_type="published",
            created_by=access_actor,
            summary="Published docs update",
            snapshot={
                "content": draft_content.model_dump(mode="json"),
                "visibility_enabled": document.visibility_enabled,
                "publish_start_at": document.publish_start_at.isoformat() if document.publish_start_at else None,
                "publish_end_at": document.publish_end_at.isoformat() if document.publish_end_at else None,
                "publish_duration_minutes": document.publish_duration_minutes,
            },
        )
    )
    await session.commit()
    return await get_admin_docs(session)


def _legacy_config(document: DocsDocument, access: DocsAccessStatusSchema) -> DocsLegacyConfigSchema:
    return DocsLegacyConfigSchema(
        is_enabled=access.available,
        start_at=document.publish_start_at,
        end_at=_effective_end_at(document),
        updated_at=document.updated_at,
    )


async def get_legacy_public_docs(session: AsyncSession) -> DocsLegacyPublicSchema:
    document = await _get_or_create_document(session)
    access = _compute_access(document)
    content = _validate_docs_content(document.published_content)
    return DocsLegacyPublicSchema(
        config=_legacy_config(document, access),
        sections=_legacy_sections(content, document.updated_at),
        team_members=_legacy_team_members(content, document.updated_at),
        live_data=await _build_legacy_live_data(session),
    )


async def get_legacy_docs_config(session: AsyncSession) -> DocsLegacyConfigSchema:
    document = await _get_or_create_document(session)
    return DocsLegacyConfigSchema(
        is_enabled=document.visibility_enabled,
        start_at=document.publish_start_at,
        end_at=_effective_end_at(document),
        updated_at=document.updated_at,
    )


async def get_legacy_docs_sections(session: AsyncSession) -> DocsLegacySectionListResponseSchema:
    document = await _get_or_create_document(session)
    content = _validate_docs_content(document.draft_content)
    return DocsLegacySectionListResponseSchema(items=_legacy_sections(content, document.updated_at))


async def get_legacy_docs_team(session: AsyncSession) -> DocsLegacyTeamListResponseSchema:
    document = await _get_or_create_document(session)
    content = _validate_docs_content(document.draft_content)
    return DocsLegacyTeamListResponseSchema(items=_legacy_team_members(content, document.updated_at))


def _find_section_setting(content: DocsContentSchema, section_id: str) -> DocsSectionSettingSchema | None:
    for setting in content.section_settings:
        if setting.slug == section_id:
            return setting
    return None


async def update_legacy_docs_config(
    session: AsyncSession,
    *,
    is_enabled: bool | None,
    start_at: datetime | None,
    end_at: datetime | None,
    clear_schedule: bool | None,
) -> DocsLegacyConfigSchema:
    document = await _get_or_create_document(session)
    next_start = None if clear_schedule else start_at if start_at is not None else document.publish_start_at
    next_end = None if clear_schedule else end_at if end_at is not None else document.publish_end_at
    _validate_schedule(
        publish_start_at=next_start,
        publish_end_at=next_end,
        publish_duration_minutes=None,
    )
    if is_enabled is not None:
        document.visibility_enabled = is_enabled
    document.publish_start_at = next_start
    document.publish_end_at = next_end
    document.publish_duration_minutes = None
    document.draft_revision += 1
    document.updated_by = "Admin"
    await session.commit()
    return await get_legacy_docs_config(session)


async def update_legacy_docs_section(
    session: AsyncSession,
    section_id: str,
    *,
    title: str | None,
    content_markdown: str | None,
    icon: str | None,
    sort_order: int | None,
    is_visible: bool | None,
) -> DocsLegacySectionSchema:
    document = await _get_or_create_document(session)
    draft = _validate_docs_content(document.draft_content)
    setting = _find_section_setting(draft, section_id)
    if setting is None:
        raise AppError(status_code=404, code="DOCS_SECTION_NOT_FOUND", message="Docs section not found.")
    if title is not None:
        setting.title = title
    if icon is not None:
        setting.icon = icon
    if sort_order is not None:
        setting.sort_order = sort_order
    if is_visible is not None:
        setting.is_visible = is_visible
    if content_markdown is not None:
        draft.section_markdown_overrides[section_id] = content_markdown
        if section_id == "team":
            draft.team_summary = content_markdown

    document.draft_content = draft.model_dump(mode="json")
    document.draft_revision += 1
    document.updated_by = "Admin"
    await session.commit()

    refreshed = await get_legacy_docs_sections(session)
    for item in refreshed.items:
        if item.id == section_id:
            return item
    raise AppError(status_code=404, code="DOCS_SECTION_NOT_FOUND", message="Docs section not found.")


async def reorder_legacy_docs_sections(
    session: AsyncSession,
    order: list[tuple[str, int]],
) -> None:
    document = await _get_or_create_document(session)
    draft = _validate_docs_content(document.draft_content)
    order_map = {section_id: sort_order for section_id, sort_order in order}
    for setting in draft.section_settings:
        if setting.slug in order_map:
            setting.sort_order = order_map[setting.slug]
    draft.section_settings = sorted(draft.section_settings, key=lambda item: item.sort_order)
    document.draft_content = draft.model_dump(mode="json")
    document.draft_revision += 1
    document.updated_by = "Admin"
    await session.commit()


async def create_legacy_team_member(
    session: AsyncSession,
    *,
    full_name: str,
    role: str,
    email: str | None,
    profile_picture_url: str | None,
    sort_order: int | None,
) -> DocsLegacyTeamMemberSchema:
    document = await _get_or_create_document(session)
    draft = _validate_docs_content(document.draft_content)
    member_id = f"team-{uuid.uuid4().hex[:10]}"
    next_index = len(draft.team_members) if sort_order is None else max(0, sort_order)
    created_member = DocsTeamMemberSchema(
        id=member_id,
        full_name=full_name,
        role=role,
        email=email,
        profile_image_url=profile_picture_url,
    )
    draft.team_members.insert(
        min(next_index, len(draft.team_members)),
        created_member,
    )
    document.draft_content = draft.model_dump(mode="json")
    document.draft_revision += 1
    document.updated_by = "Admin"
    await session.commit()
    team = await get_legacy_docs_team(session)
    for item in team.items:
        if item.id == member_id:
            return item
    raise AppError(status_code=404, code="DOCS_TEAM_MEMBER_NOT_FOUND", message="Team member not found.")


async def update_legacy_team_member(
    session: AsyncSession,
    member_id: str,
    *,
    full_name: str | None,
    role: str | None,
    email: str | None,
    profile_picture_url: str | None,
    sort_order: int | None,
) -> DocsLegacyTeamMemberSchema:
    document = await _get_or_create_document(session)
    draft = _validate_docs_content(document.draft_content)
    member_index = next((index for index, member in enumerate(draft.team_members) if member.id == member_id), -1)
    if member_index < 0:
        raise AppError(status_code=404, code="DOCS_TEAM_MEMBER_NOT_FOUND", message="Team member not found.")
    member = draft.team_members[member_index]
    if full_name is not None:
        member.full_name = full_name
    if role is not None:
        member.role = role
    if email is not None:
        member.email = email
    if profile_picture_url is not None:
        member.profile_image_url = profile_picture_url
    if sort_order is not None:
        moved = draft.team_members.pop(member_index)
        draft.team_members.insert(min(max(sort_order, 0), len(draft.team_members)), moved)

    document.draft_content = draft.model_dump(mode="json")
    document.draft_revision += 1
    document.updated_by = "Admin"
    await session.commit()
    team = await get_legacy_docs_team(session)
    for item in team.items:
        if item.id == member_id:
            return item
    raise AppError(status_code=404, code="DOCS_TEAM_MEMBER_NOT_FOUND", message="Team member not found.")


async def delete_legacy_team_member(session: AsyncSession, member_id: str) -> None:
    document = await _get_or_create_document(session)
    draft = _validate_docs_content(document.draft_content)
    original_count = len(draft.team_members)
    draft.team_members = [member for member in draft.team_members if member.id != member_id]
    if len(draft.team_members) == original_count:
        raise AppError(status_code=404, code="DOCS_TEAM_MEMBER_NOT_FOUND", message="Team member not found.")
    document.draft_content = draft.model_dump(mode="json")
    document.draft_revision += 1
    document.updated_by = "Admin"
    await session.commit()


async def get_legacy_live_data(session: AsyncSession) -> DocsLegacyLiveDataSchema:
    return await _build_legacy_live_data(session)
