from __future__ import annotations

import asyncio

from app.core.logging import configure_logging, get_logger
from app.services.embedding_service import preload_embedding_model

configure_logging()
logger = get_logger(__name__)


async def main() -> None:
    metadata = await preload_embedding_model()
    logger.info(
        "embedding_model_preloaded backend=%s model=%s device=%s dimension=%s",
        metadata["backend"],
        metadata["model"],
        metadata["device"],
        metadata["dimension"],
    )


if __name__ == "__main__":
    asyncio.run(main())
