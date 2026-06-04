from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


from app.models.audit_log import AuditLog  # noqa: E402,F401
from app.models.claim import Claim, ClaimEvidenceLink  # noqa: E402,F401
from app.models.evidence_source import EvidenceSource  # noqa: E402,F401
from app.models.rumor_cluster import RumorCluster  # noqa: E402,F401
from app.models.verification_job import VerificationJob  # noqa: E402,F401
