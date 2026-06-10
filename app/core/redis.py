from __future__ import annotations

from redis.asyncio import Redis

from app.core.config import settings

redis_client: Redis | None = None


async def init_redis() -> Redis:
    global redis_client
    if redis_client is None:
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    return redis_client


async def get_redis() -> Redis:
    return await init_redis()


async def ping_redis() -> bool:
    try:
        client = await init_redis()
        return bool(await client.ping())
    except Exception:
        return False


async def close_redis() -> None:
    global redis_client
    if redis_client is not None:
        aclose = getattr(redis_client, "aclose", None)
        if callable(aclose):
            await aclose()
        else:
            await redis_client.close()
        redis_client = None
