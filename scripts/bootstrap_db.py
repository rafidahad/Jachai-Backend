from __future__ import annotations

import asyncio

from app.core.database import bootstrap_database, close_database
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


async def main() -> None:
    await bootstrap_database(create_tables=True)
    await close_database()
    logger.info("database_bootstrap_complete")


if __name__ == "__main__":
    asyncio.run(main())
