from __future__ import annotations

import shutil

from app.core.config import settings
from app.core.database import check_database_connection
from app.core.redis import ping_redis
from app.schemas.health_schema import ComponentHealthSchema, HealthResponseSchema


async def get_system_health() -> HealthResponseSchema:
    db_ok = await check_database_connection()
    redis_ok = await ping_redis()
    nvidia_ok = bool(
        settings.nvidia_api_key
        and settings.nvidia_base_url
        and settings.active_nvidia_reasoning_model
    )
    tesseract_ok = shutil.which("tesseract") is not None

    components = [
        ComponentHealthSchema(
            name="database",
            status="healthy" if db_ok else "unavailable",
            ok=db_ok,
            message="Neon PostgreSQL connection ok." if db_ok else "Database connection failed.",
        ),
        ComponentHealthSchema(
            name="redis",
            status="healthy" if redis_ok else "unavailable",
            ok=redis_ok,
            message="Redis ping ok." if redis_ok else "Redis ping failed.",
        ),
        ComponentHealthSchema(
            name="nvidia",
            status="healthy" if nvidia_ok else "degraded",
            ok=nvidia_ok,
            message=(
                "NVIDIA reasoning model is configured."
                if nvidia_ok
                else "NVIDIA reasoning model is not configured."
            ),
        ),
        ComponentHealthSchema(
            name="tesseract",
            status="healthy" if tesseract_ok else "degraded",
            ok=tesseract_ok,
            message="Tesseract is available." if tesseract_ok else "Tesseract is not installed.",
        ),
    ]

    overall = "healthy"
    if any(component.status == "unavailable" for component in components):
        overall = "unavailable"
    elif any(component.status == "degraded" for component in components):
        overall = "degraded"

    return HealthResponseSchema(status=overall, components=components)
