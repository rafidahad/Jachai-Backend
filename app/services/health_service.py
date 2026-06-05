from __future__ import annotations

import shutil

from app.core.config import settings
from app.core.database import check_database_connection
from app.core.redis import ping_redis
from app.db.session import SessionLocal
from app.services.evidence_index_service import get_evidence_index_status
from app.schemas.health_schema import ComponentHealthSchema, HealthResponseSchema


async def get_system_health() -> HealthResponseSchema:
    db_ok = await check_database_connection()
    redis_ok = await ping_redis()
    reasoning_ok = bool(
        settings.nvidia_api_key
        and settings.nvidia_base_url
        and settings.active_nvidia_reasoning_model
    )
    rerank_ok = bool(
        settings.nvidia_api_key
        and settings.nvidia_base_url
        and settings.active_nvidia_rerank_model
    )
    vision_enabled = settings.nvidia_enable_vision_fallback
    vision_ok = bool(
        settings.nvidia_vision_enabled
        and settings.nvidia_api_key
        and settings.nvidia_base_url
        and settings.active_nvidia_vision_model
    )
    tesseract_ok = shutil.which("tesseract") is not None
    evidence_index_ok = False
    evidence_index_message = "Evidence index could not be checked."

    if db_ok:
        async with SessionLocal() as session:
            evidence_index = await get_evidence_index_status(session)
        evidence_index_ok = evidence_index.ready
        evidence_index_message = evidence_index.message

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
            name="nvidia_reasoning",
            status="healthy" if reasoning_ok else "degraded",
            ok=reasoning_ok,
            message=(
                "NVIDIA reasoning model is configured."
                if reasoning_ok
                else "NVIDIA reasoning model is not configured."
            ),
        ),
        ComponentHealthSchema(
            name="nvidia_rerank",
            status="healthy" if rerank_ok else "degraded",
            ok=rerank_ok,
            message=(
                "NVIDIA rerank model is configured."
                if rerank_ok
                else "NVIDIA rerank model is not configured."
            ),
        ),
        ComponentHealthSchema(
            name="nvidia_vision",
            status="healthy" if (not vision_enabled or vision_ok) else "degraded",
            ok=(not vision_enabled or vision_ok),
            message=(
                "NVIDIA vision fallback is configured."
                if vision_ok
                else "NVIDIA vision fallback is disabled."
                if not vision_enabled
                else "NVIDIA vision fallback is enabled but not configured."
            ),
        ),
        ComponentHealthSchema(
            name="tesseract",
            status="healthy" if tesseract_ok else "degraded",
            ok=tesseract_ok,
            message="Tesseract is available." if tesseract_ok else "Tesseract is not installed.",
        ),
        ComponentHealthSchema(
            name="evidence_index",
            status="healthy" if evidence_index_ok else "degraded",
            ok=evidence_index_ok,
            message=evidence_index_message,
        ),
    ]

    overall = "healthy"
    if any(component.status == "unavailable" for component in components):
        overall = "unavailable"
    elif any(component.status == "degraded" for component in components):
        overall = "degraded"

    return HealthResponseSchema(status=overall, components=components)
