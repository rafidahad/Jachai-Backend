from __future__ import annotations

from time import perf_counter
from typing import Any

import httpx
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
            started = perf_counter()
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
                "nvidia_request_succeeded task=%s model=%s attempt=%s latency_ms=%s",
                task.value,
                selected_model,
                attempt,
                round((perf_counter() - started) * 1000, 2),
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


async def call_nvidia_rerank(
    *,
    task: NVIDIAModelTask,
    query_text: str,
    passages: list[str],
    model: str | None = None,
) -> dict[str, Any]:
    selected_model = model or get_model_for_task(task)
    if not settings.nvidia_api_key:
        raise RuntimeError("NVIDIA API key is not configured.")
    if not selected_model:
        raise RuntimeError(f"No NVIDIA model is configured for task '{task.value}'.")
    if not await cache_service.reserve_nvidia_slot():
        raise RuntimeError("Verification is temporarily rate limited.")

    payload = {
        "model": selected_model,
        "query": {"text": query_text},
        "passages": [{"text": passage} for passage in passages],
        "truncate": "END",
    }
    headers = {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(1, 3):
        try:
            started = perf_counter()
            logger.info("nvidia_rerank_request task=%s model=%s attempt=%s", task.value, selected_model, attempt)
            async with httpx.AsyncClient(
                timeout=settings.nvidia_timeout_seconds,
                headers=headers,
            ) as client:
                response = await client.post(settings.nvidia_rerank_url, json=payload)
                response.raise_for_status()
                body = response.json()
            logger.info(
                "nvidia_rerank_request_succeeded task=%s model=%s attempt=%s latency_ms=%s",
                task.value,
                selected_model,
                attempt,
                round((perf_counter() - started) * 1000, 2),
            )
            return body
        except Exception as exc:
            last_error = exc
            logger.exception(
                "nvidia_rerank_request_failed task=%s model=%s attempt=%s",
                task.value,
                selected_model,
                attempt,
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError("NVIDIA rerank request failed unexpectedly.")
