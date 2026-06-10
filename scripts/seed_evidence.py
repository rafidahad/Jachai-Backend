from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.core.database import init_database
from app.db.session import SessionLocal
from app.schemas.source_schema import SourceIngestItemSchema
from app.services.source_service import ingest_sources


def load_items(path: Path) -> list[SourceIngestItemSchema]:
    if path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
    return [SourceIngestItemSchema.model_validate(item) for item in items]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed evidence sources into JachAI.")
    parser.add_argument("path", type=Path, help="Path to a JSON or JSONL file containing evidence items.")
    args = parser.parse_args()

    await init_database()
    items = load_items(args.path)
    async with SessionLocal() as session:
        sources, created_count, updated_count = await ingest_sources(session, items)
    print(
        json.dumps(
            {
                "created_count": created_count,
                "updated_count": updated_count,
                "source_ids": [str(item.id) for item in sources],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
