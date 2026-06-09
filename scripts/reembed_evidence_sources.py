from __future__ import annotations

import asyncio

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.evidence_source import EvidenceSource
from app.services.embedding_service import embed_text
from app.services.text_cleaning_service import clean_text


async def main() -> None:
    async with SessionLocal() as session:
        sources = (await session.execute(select(EvidenceSource).order_by(EvidenceSource.created_at.asc()))).scalars().all()
        total = len(sources)
        print(f"Re-embedding {total} evidence sources...")

        for index, source in enumerate(sources, start=1):
            source.embedding = await embed_text(
                clean_text(source.text_content),
                task_type="RETRIEVAL_DOCUMENT",
                title=source.title,
            )
            if index % 25 == 0 or index == total:
                await session.commit()
                print(f"Updated {index}/{total}")

        await session.commit()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
