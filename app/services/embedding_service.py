from __future__ import annotations

import asyncio
from functools import lru_cache
from math import sqrt

from app.core.config import settings
from app.services.gemini_client import call_gemini_embed_content


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        settings.embedding_model,
        device=settings.embedding_device,
    )


def _local_embedding_model_metadata() -> dict[str, object]:
    model = get_embedding_model()
    probe = model.encode("embedding warmup probe", normalize_embeddings=True)
    dimension = int(len(probe))
    return {
        "backend": "local",
        "model": settings.embedding_model,
        "device": settings.embedding_device,
        "dimension": dimension,
    }


def _normalize_embedding(values: list[float]) -> list[float]:
    magnitude = sqrt(sum(value * value for value in values))
    if magnitude <= 0:
        return values
    return [float(value / magnitude) for value in values]


async def embed_text(
    value: str,
    *,
    task_type: str | None = None,
    title: str | None = None,
) -> list[float]:
    if settings.embedding_uses_gemini:
        vector = await call_gemini_embed_content(
            content=value,
            model=settings.embedding_model,
            output_dimensionality=settings.embedding_dim,
            task_type=task_type,
            title=title,
        )
        return _normalize_embedding(vector)

    def _encode() -> list[float]:
        model = get_embedding_model()
        vector = model.encode(value, normalize_embeddings=True)
        return [float(item) for item in vector.tolist()]

    return await asyncio.to_thread(_encode)


async def preload_embedding_model() -> dict[str, object]:
    if settings.embedding_uses_gemini:
        return {
            "backend": "gemini",
            "model": settings.embedding_model,
            "device": "remote",
            "dimension": settings.embedding_dim,
        }

    return await asyncio.to_thread(_local_embedding_model_metadata)


async def check_embedding_backend() -> tuple[bool, str]:
    if settings.embedding_uses_gemini:
        if settings.embedding_model.strip() and settings.active_gemini_api_keys:
            return True, f"Gemini embedding model {settings.embedding_model} is configured."
        return False, "Gemini embedding model is not fully configured."

    try:
        metadata = await preload_embedding_model()
    except Exception as exc:
        return False, f"Local embedding model failed to load: {exc}"

    return (
        True,
        "Local embedding model "
        f"{metadata['model']} is ready on {metadata['device']} ({metadata['dimension']} dims).",
    )
