from __future__ import annotations

import json
from typing import Any

from app.core.config import settings
from app.core.redis import get_redis


class CacheService:
    async def get_json_payload(self, key: str) -> dict[str, Any] | None:
        redis = await get_redis()
        payload = await redis.get(key)
        return json.loads(payload) if payload else None

    async def set_json_payload(self, key: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        redis = await get_redis()
        await redis.setex(
            key,
            ttl_seconds,
            json.dumps(payload, default=str),
        )

    async def get_duplicate_claim_id(self, normalized_hash: str) -> str | None:
        redis = await get_redis()
        return await redis.get(f"claim:duplicate:{normalized_hash}")

    async def set_duplicate_claim_id(self, normalized_hash: str, claim_id: str) -> None:
        redis = await get_redis()
        await redis.setex(
            f"claim:duplicate:{normalized_hash}",
            settings.duplicate_cache_ttl_seconds,
            claim_id,
        )

    async def get_claim_result(self, normalized_hash: str) -> dict[str, Any] | None:
        redis = await get_redis()
        payload = await redis.get(f"claim:result:{normalized_hash}")
        return json.loads(payload) if payload else None

    async def set_claim_result(self, normalized_hash: str, payload: dict[str, Any]) -> None:
        redis = await get_redis()
        await redis.setex(
            f"claim:result:{normalized_hash}",
            settings.result_cache_ttl_seconds,
            json.dumps(payload, default=str),
        )

    async def get_ai_task_payload(self, task_name: str, cache_key: str) -> dict[str, Any] | None:
        redis = await get_redis()
        payload = await redis.get(f"ai:{task_name}:{cache_key}")
        return json.loads(payload) if payload else None

    async def set_ai_task_payload(self, task_name: str, cache_key: str, payload: dict[str, Any]) -> None:
        redis = await get_redis()
        await redis.setex(
            f"ai:{task_name}:{cache_key}",
            settings.result_cache_ttl_seconds,
            json.dumps(payload, default=str),
        )

    async def set_job_status(self, job_id: str, payload: dict[str, Any]) -> None:
        redis = await get_redis()
        await redis.setex(
            f"job:status:{job_id}",
            settings.job_status_ttl_seconds,
            json.dumps(payload, default=str),
        )

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        redis = await get_redis()
        payload = await redis.get(f"job:status:{job_id}")
        return json.loads(payload) if payload else None

    async def reserve_nvidia_slot(self) -> bool:
        redis = await get_redis()
        key = "nvidia:rpm"
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, 60)
        if count > settings.nvidia_rpm_safety_limit:
            return False
        return True

    async def reserve_uncached_claim_slot(self) -> bool:
        redis = await get_redis()
        key = "claims:uncached:rpm"
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, 60)
        if count > settings.nvidia_max_uncached_claims_per_minute:
            return False
        return True


cache_service = CacheService()
