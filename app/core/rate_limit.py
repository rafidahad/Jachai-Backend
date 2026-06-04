from __future__ import annotations

import hashlib

from fastapi import Request

from app.core.config import settings
from app.core.redis import get_redis
from app.utils.errors import AppError


def rate_limit_dependency(limit_per_minute: int):
    async def dependency(request: Request) -> None:
        redis = await get_redis()
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            identity = forwarded_for.split(",")[0].strip()
        elif request.client:
            identity = request.client.host
        else:
            identity = "unknown"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        window_key = f"rate_limit:{request.url.path}:{digest}"
        count = await redis.incr(window_key)
        if count == 1:
            await redis.expire(window_key, 60)
        if count > limit_per_minute:
            raise AppError(
                status_code=429,
                code="RATE_LIMITED",
                message="Too many requests. Please retry in a minute.",
            )

    return dependency


claim_submission_rate_limit = rate_limit_dependency(settings.claim_rate_limit_per_minute)
internal_rate_limit = rate_limit_dependency(settings.internal_rate_limit_per_minute)
