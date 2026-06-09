from __future__ import annotations

import asyncio
from functools import lru_cache
from math import sqrt

from app.core.config import settings
from app.services.gemini_client import call_gemini_embed_content


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.embedding_model)


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
