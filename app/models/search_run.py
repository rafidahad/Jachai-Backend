from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class SearchRun(Base):
    __tablename__ = "search_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    claim_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("claims.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    extracted_claim: Mapped[str] = mapped_column(Text())
    search_queries: Mapped[list[str]] = mapped_column(JSONB, default=list)
    provider: Mapped[str] = mapped_column(String(64), default="tavily", index=True)
    search_answer: Mapped[str | None] = mapped_column(Text(), nullable=True)
    request_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    result_count: Mapped[int] = mapped_column(Integer(), default=0)
    crawled_count: Mapped[int] = mapped_column(Integer(), default=0)
    status: Mapped[str] = mapped_column(String(32), default="completed", index=True)
    error_message: Mapped[str | None] = mapped_column(Text(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    results: Mapped[list["SearchResultRecord"]] = relationship(
        back_populates="search_run",
        cascade="all, delete-orphan",
    )


class SearchResultRecord(Base):
    __tablename__ = "search_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    search_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("search_runs.id", ondelete="CASCADE"),
        index=True,
    )
    query: Mapped[str] = mapped_column(Text())
    title: Mapped[str] = mapped_column(String(255))
    url: Mapped[str] = mapped_column(String(2048), index=True)
    snippet: Mapped[str | None] = mapped_column(Text(), nullable=True)
    content: Mapped[str | None] = mapped_column(Text(), nullable=True)
    provider: Mapped[str] = mapped_column(String(64), default="tavily", index=True)
    search_score: Mapped[float | None] = mapped_column(Float(), nullable=True)
    trust_score: Mapped[float] = mapped_column(Float(), default=0.50)
    favicon: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    rank: Mapped[int] = mapped_column(Integer())
    selected_for_crawl: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    search_run: Mapped[SearchRun] = relationship(back_populates="results")
