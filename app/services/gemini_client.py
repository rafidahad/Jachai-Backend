from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _gemini_generate_content_url(model: str) -> str:
    base_url = settings.gemini_base_url.rstrip("/")
    return f"{base_url}/models/{model}:generateContent"


def _gemini_embed_content_url(model: str) -> str:
    base_url = settings.gemini_base_url.rstrip("/")
    return f"{base_url}/models/{model}:embedContent"


def _gemini_headers(api_key: str) -> dict[str, str]:
    if not api_key:
        raise RuntimeError("Gemini API key is not configured.")
    return {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("Gemini response did not include candidates.")

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        text_parts = [
            str(part.get("text"))
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        if text_parts:
            return "".join(text_parts)

    raise RuntimeError("Gemini response did not contain text output.")


def _extract_gemini_embedding(payload: dict[str, Any]) -> list[float]:
    embedding = payload.get("embedding")
    if not isinstance(embedding, dict):
        raise RuntimeError("Gemini embedding response did not include embedding.")

    values = embedding.get("values")
    if not isinstance(values, list) or not values:
        raise RuntimeError("Gemini embedding response did not include embedding values.")

    try:
        return [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Gemini embedding response contained invalid embedding values.") from exc


def _gemini_api_keys() -> list[str]:
    keys = settings.active_gemini_api_keys
    if not keys:
        raise RuntimeError("Gemini API key is not configured.")
    return keys


def _should_failover_to_backup_key(exc: Exception) -> bool:
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    return exc.response.status_code in {401, 403, 429}


def _retry_delay_seconds(exc: Exception, attempt: int) -> float | None:
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    if exc.response.status_code not in {429, 500, 502, 503, 504}:
        return None
    retry_after = exc.response.headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return min(8.0, float(2 ** max(0, attempt - 1)))


async def call_gemini_generate_content(
    *,
    system_instruction: str,
    user_content: str,
    model: str | None = None,
    response_mime_type: str | None = None,
    response_schema: dict[str, Any] | None = None,
    max_output_tokens: int | None = None,
    purpose: str = "claim_extraction",
) -> str:
    selected_model = model or settings.active_gemini_claim_extraction_model
    if not selected_model:
        raise RuntimeError("No Gemini model is configured for this Gemini request.")

    payload: dict[str, Any] = {
        "system_instruction": {
            "parts": [{"text": system_instruction}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_content}],
            }
        ],
    }

    generation_config: dict[str, Any] = {}
    if response_mime_type:
        generation_config["responseMimeType"] = response_mime_type
    if response_schema:
        generation_config["responseSchema"] = response_schema
    if max_output_tokens is not None:
        generation_config["maxOutputTokens"] = max_output_tokens
    if generation_config:
        payload["generationConfig"] = generation_config

    last_error: Exception | None = None
    api_keys = _gemini_api_keys()
    for key_index, api_key in enumerate(api_keys, start=1):
        for attempt in range(1, 3):
            try:
                started = perf_counter()
                logger.info(
                    "gemini_request purpose=%s model=%s key_index=%s/%s attempt=%s",
                    purpose,
                    selected_model,
                    key_index,
                    len(api_keys),
                    attempt,
                )
                async with httpx.AsyncClient(timeout=settings.gemini_timeout_seconds) as client:
                    response = await client.post(
                        _gemini_generate_content_url(selected_model),
                        json=payload,
                        headers=_gemini_headers(api_key),
                    )
                    response.raise_for_status()
                    body = response.json()
                logger.info(
                    "gemini_request_succeeded purpose=%s model=%s key_index=%s/%s attempt=%s latency_ms=%s",
                    purpose,
                    selected_model,
                    key_index,
                    len(api_keys),
                    attempt,
                    round((perf_counter() - started) * 1000, 2),
                )
                return _extract_gemini_text(body)
            except Exception as exc:
                last_error = exc
                logger.exception(
                    "gemini_request_failed purpose=%s model=%s key_index=%s/%s attempt=%s",
                    purpose,
                    selected_model,
                    key_index,
                    len(api_keys),
                    attempt,
                )
                if _should_failover_to_backup_key(exc) and key_index < len(api_keys):
                    logger.warning(
                        "gemini_key_failover purpose=%s model=%s from_key=%s to_key=%s",
                        purpose,
                        selected_model,
                        key_index,
                        key_index + 1,
                    )
                    break
                retry_delay = _retry_delay_seconds(exc, attempt)
                if retry_delay is not None and attempt < 2:
                    await asyncio.sleep(retry_delay)
                    continue
                if attempt >= 2:
                    break

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini request failed unexpectedly.")


async def call_gemini_embed_content(
    *,
    content: str,
    model: str,
    output_dimensionality: int | None = None,
    task_type: str | None = None,
    title: str | None = None,
) -> list[float]:
    if not model:
        raise RuntimeError("No Gemini embedding model is configured.")

    payload: dict[str, Any] = {
        "model": f"models/{model}",
        "content": {
            "parts": [{"text": content}],
        },
    }
    if output_dimensionality is not None:
        payload["outputDimensionality"] = output_dimensionality
    if task_type:
        payload["taskType"] = task_type
    if title:
        payload["title"] = title

    last_error: Exception | None = None
    api_keys = _gemini_api_keys()
    for key_index, api_key in enumerate(api_keys, start=1):
        for attempt in range(1, 3):
            try:
                started = perf_counter()
                logger.info(
                    "gemini_request purpose=embedding model=%s key_index=%s/%s attempt=%s task_type=%s dim=%s",
                    model,
                    key_index,
                    len(api_keys),
                    attempt,
                    task_type or "unspecified",
                    output_dimensionality or "default",
                )
                async with httpx.AsyncClient(timeout=settings.gemini_timeout_seconds) as client:
                    response = await client.post(
                        _gemini_embed_content_url(model),
                        json=payload,
                        headers=_gemini_headers(api_key),
                    )
                    response.raise_for_status()
                    body = response.json()
                logger.info(
                    "gemini_request_succeeded purpose=embedding model=%s key_index=%s/%s attempt=%s latency_ms=%s",
                    model,
                    key_index,
                    len(api_keys),
                    attempt,
                    round((perf_counter() - started) * 1000, 2),
                )
                return _extract_gemini_embedding(body)
            except Exception as exc:
                last_error = exc
                logger.exception(
                    "gemini_request_failed purpose=embedding model=%s key_index=%s/%s attempt=%s",
                    model,
                    key_index,
                    len(api_keys),
                    attempt,
                )
                if _should_failover_to_backup_key(exc) and key_index < len(api_keys):
                    logger.warning(
                        "gemini_key_failover purpose=embedding model=%s from_key=%s to_key=%s",
                        model,
                        key_index,
                        key_index + 1,
                    )
                    break
                retry_delay = _retry_delay_seconds(exc, attempt)
                if retry_delay is not None and attempt < 2:
                    await asyncio.sleep(retry_delay)
                    continue
                if attempt >= 2:
                    break

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini embedding request failed unexpectedly.")
