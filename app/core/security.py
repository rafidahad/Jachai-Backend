from __future__ import annotations

from fastapi import Header

from app.core.config import settings
from app.utils.errors import AppError


async def verify_internal_api_key(
    x_internal_api_key: str | None = Header(default=None, alias="X-Internal-API-Key"),
) -> None:
    if not settings.internal_api_key:
        raise AppError(
            status_code=503,
            code="INTERNAL_KEY_NOT_CONFIGURED",
            message="Internal API key is not configured.",
        )
    if x_internal_api_key != settings.internal_api_key:
        raise AppError(status_code=401, code="UNAUTHORIZED", message="Invalid internal API key.")
