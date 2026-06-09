from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
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


@router.get("/{job_id}/stream")
async def stream_verification_job_endpoint(
    job_id: str,
    session: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    async def event_stream():
        last_payload = ""
        idle_ticks = 0

        while idle_ticks < 180:
            job = await get_verification_job(session, job_id)
            if job is None:
                error_payload = json.dumps(
                    {"error": {"code": "JOB_NOT_FOUND", "message": "Verification job not found."}}
                )
                yield f"event: error\ndata: {error_payload}\n\n"
                return

            payload = json.dumps(job.model_dump(mode="json"), default=str)
            if payload != last_payload:
                last_payload = payload
                idle_ticks = 0
                yield f"event: job\ndata: {payload}\n\n"
            else:
                idle_ticks += 1
                if idle_ticks % 10 == 0:
                    yield "event: keepalive\ndata: {}\n\n"

            if job.status in {"completed", "failed"}:
                yield "event: end\ndata: {}\n\n"
                return

            await asyncio.sleep(1)

        yield "event: timeout\ndata: {}\n\n"

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
