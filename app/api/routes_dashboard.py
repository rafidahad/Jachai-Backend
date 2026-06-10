from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import require_admin_session
from app.db.session import get_db
from app.schemas.dashboard_schema import (
    DashboardClaimsOverTimeResponseSchema,
    DashboardDistributionResponseSchema,
    DashboardRecentClaimsResponseSchema,
    DashboardSummaryResponseSchema,
)
from app.services.dashboard_service import (
    get_claims_over_time,
    get_language_distribution,
    get_recent_claims,
    get_summary_metrics,
    get_verdict_distribution,
)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummaryResponseSchema)
async def dashboard_summary(
    response: Response,
    session: AsyncSession = Depends(get_db),
) -> DashboardSummaryResponseSchema:
    response.headers["Cache-Control"] = "public, max-age=15, stale-while-revalidate=60"
    return DashboardSummaryResponseSchema(metrics=await get_summary_metrics(session))


@router.get(
    "/verdict-distribution",
    response_model=DashboardDistributionResponseSchema,
    dependencies=[Depends(require_admin_session)],
)
async def verdict_distribution(session: AsyncSession = Depends(get_db)) -> DashboardDistributionResponseSchema:
    return DashboardDistributionResponseSchema(items=await get_verdict_distribution(session))


@router.get(
    "/language-distribution",
    response_model=DashboardDistributionResponseSchema,
    dependencies=[Depends(require_admin_session)],
)
async def language_distribution(session: AsyncSession = Depends(get_db)) -> DashboardDistributionResponseSchema:
    return DashboardDistributionResponseSchema(items=await get_language_distribution(session))


@router.get(
    "/claims-over-time",
    response_model=DashboardClaimsOverTimeResponseSchema,
    dependencies=[Depends(require_admin_session)],
)
async def claims_over_time(session: AsyncSession = Depends(get_db)) -> DashboardClaimsOverTimeResponseSchema:
    return DashboardClaimsOverTimeResponseSchema(items=await get_claims_over_time(session))


@router.get("/recent-claims", response_model=DashboardRecentClaimsResponseSchema)
async def recent_claims(session: AsyncSession = Depends(get_db)) -> DashboardRecentClaimsResponseSchema:
    return DashboardRecentClaimsResponseSchema(items=await get_recent_claims(session))
