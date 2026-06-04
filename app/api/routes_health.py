from __future__ import annotations

from fastapi import APIRouter

from app.schemas.health_schema import HealthResponseSchema
from app.services.health_service import get_system_health

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/system", response_model=HealthResponseSchema)
async def system_health() -> HealthResponseSchema:
    return await get_system_health()
