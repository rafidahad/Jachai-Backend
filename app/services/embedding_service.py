from __future__ import annotations

import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


async def embed_text(value: str) -> list[float]:
    def _encode() -> list[float]:
        model = get_embedding_model()
        vector = model.encode(value, normalize_embeddings=True)
        return [float(item) for item in vector.tolist()]

    return await asyncio.to_thread(_encode)
