from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.models.docs_document import DocsDocument
from app.services.docs_service import (
    SHOWCASE_TIMEZONE,
    _compute_access,
    _legacy_sections,
    build_default_docs_content,
)


def build_document(
    *,
    visibility_enabled: bool = True,
    publish_start_at: datetime | None = None,
    publish_end_at: datetime | None = None,
    publish_duration_minutes: int | None = None,
) -> DocsDocument:
    now = datetime(2026, 6, 11, 12, 0, tzinfo=SHOWCASE_TIMEZONE)
    return DocsDocument(
        slug="main",
        visibility_enabled=visibility_enabled,
        publish_start_at=publish_start_at,
        publish_end_at=publish_end_at,
        publish_duration_minutes=publish_duration_minutes,
        draft_content={},
        published_content={},
        draft_revision=1,
        published_version=1,
        created_by="test",
        updated_by="test",
        last_published_at=now,
        created_at=now,
        updated_at=now,
    )


def test_compute_access_hidden_when_visibility_disabled():
    document = build_document(visibility_enabled=False)

    access = _compute_access(document, datetime(2026, 6, 11, 12, 0, tzinfo=SHOWCASE_TIMEZONE))

    assert access.available is False
    assert access.status == "hidden"


def test_compute_access_scheduled_before_window():
    start = datetime(2026, 6, 12, 0, 0, tzinfo=SHOWCASE_TIMEZONE)
    end = datetime(2026, 6, 14, 23, 59, tzinfo=SHOWCASE_TIMEZONE)
    document = build_document(publish_start_at=start, publish_end_at=end)

    access = _compute_access(document, datetime(2026, 6, 11, 12, 0, tzinfo=SHOWCASE_TIMEZONE))

    assert access.available is False
    assert access.status == "scheduled"


def test_compute_access_live_inside_window():
    start = datetime(2026, 6, 10, 0, 0, tzinfo=SHOWCASE_TIMEZONE)
    end = datetime(2026, 6, 14, 23, 59, tzinfo=SHOWCASE_TIMEZONE)
    document = build_document(publish_start_at=start, publish_end_at=end)

    access = _compute_access(document, datetime(2026, 6, 11, 12, 0, tzinfo=SHOWCASE_TIMEZONE))

    assert access.available is True
    assert access.status == "live"


def test_compute_access_expired_after_window():
    start = datetime(2026, 6, 10, 0, 0, tzinfo=SHOWCASE_TIMEZONE)
    end = datetime(2026, 6, 10, 12, 0, tzinfo=SHOWCASE_TIMEZONE)
    document = build_document(publish_start_at=start, publish_end_at=end)

    access = _compute_access(document, datetime(2026, 6, 11, 12, 0, tzinfo=SHOWCASE_TIMEZONE))

    assert access.available is False
    assert access.status == "expired"


def test_default_docs_content_exposes_expected_sections():
    content = build_default_docs_content()
    now = datetime(2026, 6, 11, 12, 0, tzinfo=ZoneInfo("Asia/Dhaka"))
    sections = _legacy_sections(content, now)
    slugs = [section.slug for section in sections]

    assert "problem" in slugs
    assert "feature-matrix" in slugs
    assert "architecture" in slugs
    assert "team" in slugs
    assert "changelog" in slugs
