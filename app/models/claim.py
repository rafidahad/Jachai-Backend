from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    cluster_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rumor_clusters.id", ondelete="SET NULL"),
        nullable=True,
    )
    input_type: Mapped[str] = mapped_column(String(16), index=True)
    source_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    raw_text: Mapped[str] = mapped_column(Text())
    cleaned_text: Mapped[str] = mapped_column(Text())
    masked_text: Mapped[str] = mapped_column(Text())
    normalized_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    language: Mapped[str] = mapped_column(String(32), index=True)
    verdict: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float())
    explanation: Mapped[str] = mapped_column(Text())
    reasoning: Mapped[str] = mapped_column(Text())
    share_summary: Mapped[str] = mapped_column(Text())
    review_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    context_payload: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    cluster: Mapped["RumorCluster | None"] = relationship(back_populates="claims")
    evidence_links: Mapped[list["ClaimEvidenceLink"]] = relationship(
        back_populates="claim",
        cascade="all, delete-orphan",
    )
    verification_jobs: Mapped[list["VerificationJob"]] = relationship(back_populates="claim")


class ClaimEvidenceLink(Base):
    __tablename__ = "claim_evidence_links"
    __table_args__ = (UniqueConstraint("claim_id", "source_id", name="uq_claim_evidence_link"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    claim_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("claims.id", ondelete="CASCADE"),
        index=True,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evidence_sources.id", ondelete="CASCADE"),
        index=True,
    )
    rank: Mapped[int] = mapped_column(Integer())
    similarity_score: Mapped[float] = mapped_column(Float())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    claim: Mapped["Claim"] = relationship(back_populates="evidence_links")
    source: Mapped["EvidenceSource"] = relationship(back_populates="claim_links")
