from __future__ import annotations

from sqlalchemy import text

from app.core.logging import get_logger
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
import app.models  # noqa: F401

logger = get_logger(__name__)


async def bootstrap_database(*, create_tables: bool) -> None:
    if not engine:
        return
    async with engine.begin() as connection:
        await connection.execute(text("SELECT 1"))
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
        if create_tables:
            await connection.run_sync(Base.metadata.create_all)
    logger.info("database_bootstrapped")


async def init_database() -> None:
    if not engine:
        return

    manage_schema = settings.bootstrap_database_on_startup or settings.auto_create_tables
    if manage_schema:
        await bootstrap_database(create_tables=settings.auto_create_tables)
        return

    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
    logger.info("database_connection_verified")


async def check_database_connection() -> bool:
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("database_health_check_failed")
        return False


async def close_database() -> None:
    await engine.dispose()
