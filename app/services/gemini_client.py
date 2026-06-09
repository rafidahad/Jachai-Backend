from __future__ import annotations

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


def _gemini_headers() -> dict[str, str]:
    api_key = settings.gemini_api_key
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


async def call_gemini_generate_content(
    *,
    system_instruction: str,
    user_content: str,
    model: str | None = None,
    response_mime_type: str | None = None,
    response_schema: dict[str, Any] | None = None,
    max_output_tokens: int | None = None,
) -> str:
    selected_model = model or settings.active_gemini_claim_extraction_model
    if not selected_model:
        raise RuntimeError("No Gemini model is configured for claim extraction.")

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
    for attempt in range(1, 3):
        try:
            started = perf_counter()
            logger.info(
                "gemini_request purpose=claim_extraction model=%s attempt=%s",
                selected_model,
                attempt,
            )
            async with httpx.AsyncClient(timeout=settings.gemini_timeout_seconds) as client:
                response = await client.post(
                    _gemini_generate_content_url(selected_model),
                    json=payload,
                    headers=_gemini_headers(),
                )
                response.raise_for_status()
                body = response.json()
            logger.info(
                "gemini_request_succeeded purpose=claim_extraction model=%s attempt=%s latency_ms=%s",
                selected_model,
                attempt,
                round((perf_counter() - started) * 1000, 2),
            )
            return _extract_gemini_text(body)
        except Exception as exc:
            last_error = exc
            logger.exception(
                "gemini_request_failed purpose=claim_extraction model=%s attempt=%s",
                selected_model,
                attempt,
            )

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
    for attempt in range(1, 3):
        try:
            started = perf_counter()
            logger.info(
                "gemini_request purpose=embedding model=%s attempt=%s task_type=%s dim=%s",
                model,
                attempt,
                task_type or "unspecified",
                output_dimensionality or "default",
            )
            async with httpx.AsyncClient(timeout=settings.gemini_timeout_seconds) as client:
                response = await client.post(
                    _gemini_embed_content_url(model),
                    json=payload,
                    headers=_gemini_headers(),
                )
                response.raise_for_status()
                body = response.json()
            logger.info(
                "gemini_request_succeeded purpose=embedding model=%s attempt=%s latency_ms=%s",
                model,
                attempt,
                round((perf_counter() - started) * 1000, 2),
            )
            return _extract_gemini_embedding(body)
        except Exception as exc:
            last_error = exc
            logger.exception(
                "gemini_request_failed purpose=embedding model=%s attempt=%s",
                model,
                attempt,
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini embedding request failed unexpectedly.")
