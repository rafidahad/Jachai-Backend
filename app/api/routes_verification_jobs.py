from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.claim_schema import VerificationJobSchema
from app.services.claim_pipeline import get_verification_job
from app.utils.errors import AppError

router = APIRouter(prefix="/verification-jobs", tags=["verification-jobs"])


@router.get("/{job_id}", response_model=VerificationJobSchema)
async def get_verification_job_endpoint(
    job_id: str,
    session: AsyncSession = Depends(get_db),
) -> VerificationJobSchema:
    job = await get_verification_job(session, job_id)
    if job is None:
        raise AppError(status_code=404, code="JOB_NOT_FOUND", message="Verification job not found.")
    return job
