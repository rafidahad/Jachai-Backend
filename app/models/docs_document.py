from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class DocsDocument(Base):
    __tablename__ = "docs_documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True, default="main")
    visibility_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    publish_start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    publish_end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    publish_duration_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    draft_content: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict)
    published_content: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict)
    draft_revision: Mapped[int] = mapped_column(Integer, default=1)
    published_version: Mapped[int] = mapped_column(Integer, default=1)
    created_by: Mapped[str] = mapped_column(String(128), default="system")
    updated_by: Mapped[str] = mapped_column(String(128), default="system")
    last_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    versions: Mapped[list["DocsDocumentVersion"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="desc(DocsDocumentVersion.created_at)",
    )


class DocsDocumentVersion(Base):
    __tablename__ = "docs_document_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("docs_documents.id", ondelete="CASCADE"),
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    snapshot_type: Mapped[str] = mapped_column(String(24), default="published")
    created_by: Mapped[str] = mapped_column(String(128), default="system")
    summary: Mapped[str] = mapped_column(String(255), default="Published docs update")
    snapshot: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[DocsDocument] = relationship(back_populates="versions")
