from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.services.ai_model_router import NVIDIAModelTask, get_model_for_task
from app.services.cache_service import cache_service

logger = get_logger(__name__)
client = AsyncOpenAI(api_key=settings.nvidia_api_key, base_url=settings.nvidia_base_url)


async def call_nvidia_chat(
    *,
    task: NVIDIAModelTask,
    messages: list[dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: int = 1200,
    response_format: dict[str, str] | None = None,
    model: str | None = None,
) -> str:
    selected_model = model or get_model_for_task(task)
    if not settings.nvidia_api_key:
        raise RuntimeError("NVIDIA API key is not configured.")
    if not selected_model:
        raise RuntimeError(f"No NVIDIA model is configured for task '{task.value}'.")
    if not await cache_service.reserve_nvidia_slot():
        raise RuntimeError("Verification is temporarily rate limited.")

    last_error: Exception | None = None
    for attempt in range(1, 3):
        try:
            logger.info("nvidia_request task=%s model=%s attempt=%s", task.value, selected_model, attempt)
            completion_kwargs: dict[str, Any] = {
                "model": selected_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": settings.nvidia_timeout_seconds,
            }
            if response_format is not None:
                completion_kwargs["response_format"] = response_format
            response = await client.chat.completions.create(**completion_kwargs)
            logger.info(
                "nvidia_request_succeeded task=%s model=%s attempt=%s",
                task.value,
                selected_model,
                attempt,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
            logger.exception(
                "nvidia_request_failed task=%s model=%s attempt=%s",
                task.value,
                selected_model,
                attempt,
            )
    if last_error is not None:
        raise last_error
    raise RuntimeError("NVIDIA request failed unexpectedly.")
