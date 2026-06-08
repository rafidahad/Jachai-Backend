from __future__ import annotations

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
    query_ok = bool(
        settings.nvidia_api_key
        and settings.nvidia_base_url
        and settings.active_nvidia_query_model
    )
    rerank_ok = bool(
        settings.nvidia_api_key
        and settings.nvidia_base_url
        and settings.active_nvidia_rerank_model
    )
    kimi_ocr_enabled = settings.nvidia_enable_vision_fallback
    kimi_ocr_ok = bool(
        settings.nvidia_vision_enabled
        and settings.nvidia_api_key
        and settings.nvidia_base_url
        and settings.active_nvidia_vision_model
    )
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
            name="nvidia_query_generation",
            status="healthy" if query_ok else "degraded",
            ok=query_ok,
            message=(
                "NVIDIA query generation model is configured."
                if query_ok
                else "NVIDIA query generation model is not configured."
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
            name="kimi_ocr",
            status="healthy" if (not kimi_ocr_enabled or kimi_ocr_ok) else "degraded",
            ok=(not kimi_ocr_enabled or kimi_ocr_ok),
            message=(
                "Kimi OCR is configured."
                if kimi_ocr_ok
                else "Kimi OCR is disabled."
                if not kimi_ocr_enabled
                else "Kimi OCR is enabled but not configured."
            ),
        ),
        ComponentHealthSchema(
            name="evidence_index",
            status="healthy" if evidence_index_ok else "degraded",
            ok=evidence_index_ok,
            message=evidence_index_message,
        ),
        ComponentHealthSchema(
            name="tavily_search",
            status="healthy" if settings.tavily_enabled else "degraded",
            ok=settings.tavily_enabled,
            message=(
                "Tavily search is configured."
                if settings.tavily_enabled
                else "Tavily search requires SEARCH_PROVIDER=tavily and TAVILY_API_KEY or GENERAL_SEARCH_API_KEY."
            ),
        ),
    ]

    overall = "healthy"
    if any(component.status == "unavailable" for component in components):
        overall = "unavailable"
    elif any(component.status == "degraded" for component in components):
        overall = "degraded"

    return HealthResponseSchema(status=overall, components=components)
