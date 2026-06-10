from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import asyncpg
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.asyncio import create_async_engine

import app.models  # noqa: F401
from app.db.base import Base
from app.models.evidence_source import EvidenceSource

APP_TABLES = (
    "audit_logs",
    "rumor_clusters",
    "claims",
    "evidence_sources",
    "claim_evidence_links",
    "search_runs",
    "search_results",
    "verification_jobs",
)
TEMP_TABLES = (
    "_json_test",
    "_vector_test",
)
COMPAT_DDL = (
    """
    CREATE DOMAIN vector AS text
    """,
    """
    CREATE FUNCTION vector_to_float8_array(value vector)
    RETURNS double precision[]
    LANGUAGE sql
    IMMUTABLE
    RETURNS NULL ON NULL INPUT
    AS $$
        SELECT CASE
            WHEN btrim(value::text) = '[]' THEN ARRAY[]::double precision[]
            ELSE string_to_array(trim(both '[]' from value::text), ',')::double precision[]
        END
    $$
    """,
    """
    CREATE FUNCTION vector_cosine_distance(lhs vector, rhs vector)
    RETURNS double precision
    LANGUAGE sql
    IMMUTABLE
    RETURNS NULL ON NULL INPUT
    AS $$
        WITH lhs_values AS (
            SELECT vector_to_float8_array(lhs) AS arr
        ),
        rhs_values AS (
            SELECT vector_to_float8_array(rhs) AS arr
        ),
        dims AS (
            SELECT array_length(lhs_values.arr, 1) AS lhs_dim, array_length(rhs_values.arr, 1) AS rhs_dim
            FROM lhs_values
            CROSS JOIN rhs_values
        ),
        pairs AS (
            SELECT lhs_item, rhs_item
            FROM lhs_values
            CROSS JOIN rhs_values
            CROSS JOIN LATERAL unnest(lhs_values.arr, rhs_values.arr) AS values(lhs_item, rhs_item)
        )
        SELECT CASE
            WHEN dims.lhs_dim IS DISTINCT FROM dims.rhs_dim THEN NULL
            WHEN coalesce(dims.lhs_dim, 0) = 0 THEN 1.0
            ELSE 1 - (
                coalesce(sum(pairs.lhs_item * pairs.rhs_item), 0)
                / nullif(
                    sqrt(coalesce(sum(pairs.lhs_item * pairs.lhs_item), 0))
                    * sqrt(coalesce(sum(pairs.rhs_item * pairs.rhs_item), 0)),
                    0
                )
            )
        END
        FROM dims
        LEFT JOIN pairs ON TRUE
        GROUP BY dims.lhs_dim, dims.rhs_dim
    $$
    """,
    """
    CREATE FUNCTION vector_l2_distance(lhs vector, rhs vector)
    RETURNS double precision
    LANGUAGE sql
    IMMUTABLE
    RETURNS NULL ON NULL INPUT
    AS $$
        WITH lhs_values AS (
            SELECT vector_to_float8_array(lhs) AS arr
        ),
        rhs_values AS (
            SELECT vector_to_float8_array(rhs) AS arr
        ),
        dims AS (
            SELECT array_length(lhs_values.arr, 1) AS lhs_dim, array_length(rhs_values.arr, 1) AS rhs_dim
            FROM lhs_values
            CROSS JOIN rhs_values
        ),
        pairs AS (
            SELECT lhs_item, rhs_item
            FROM lhs_values
            CROSS JOIN rhs_values
            CROSS JOIN LATERAL unnest(lhs_values.arr, rhs_values.arr) AS values(lhs_item, rhs_item)
        )
        SELECT CASE
            WHEN dims.lhs_dim IS DISTINCT FROM dims.rhs_dim THEN NULL
            WHEN coalesce(dims.lhs_dim, 0) = 0 THEN 0.0
            ELSE sqrt(coalesce(sum(power(pairs.lhs_item - pairs.rhs_item, 2)), 0))
        END
        FROM dims
        LEFT JOIN pairs ON TRUE
        GROUP BY dims.lhs_dim, dims.rhs_dim
    $$
    """,
    """
    CREATE FUNCTION vector_max_inner_product(lhs vector, rhs vector)
    RETURNS double precision
    LANGUAGE sql
    IMMUTABLE
    RETURNS NULL ON NULL INPUT
    AS $$
        WITH lhs_values AS (
            SELECT vector_to_float8_array(lhs) AS arr
        ),
        rhs_values AS (
            SELECT vector_to_float8_array(rhs) AS arr
        ),
        dims AS (
            SELECT array_length(lhs_values.arr, 1) AS lhs_dim, array_length(rhs_values.arr, 1) AS rhs_dim
            FROM lhs_values
            CROSS JOIN rhs_values
        ),
        pairs AS (
            SELECT lhs_item, rhs_item
            FROM lhs_values
            CROSS JOIN rhs_values
            CROSS JOIN LATERAL unnest(lhs_values.arr, rhs_values.arr) AS values(lhs_item, rhs_item)
        )
        SELECT CASE
            WHEN dims.lhs_dim IS DISTINCT FROM dims.rhs_dim THEN NULL
            ELSE -coalesce(sum(pairs.lhs_item * pairs.rhs_item), 0)
        END
        FROM dims
        LEFT JOIN pairs ON TRUE
        GROUP BY dims.lhs_dim, dims.rhs_dim
    $$
    """,
    """
    CREATE FUNCTION vector_l1_distance(lhs vector, rhs vector)
    RETURNS double precision
    LANGUAGE sql
    IMMUTABLE
    RETURNS NULL ON NULL INPUT
    AS $$
        WITH lhs_values AS (
            SELECT vector_to_float8_array(lhs) AS arr
        ),
        rhs_values AS (
            SELECT vector_to_float8_array(rhs) AS arr
        ),
        dims AS (
            SELECT array_length(lhs_values.arr, 1) AS lhs_dim, array_length(rhs_values.arr, 1) AS rhs_dim
            FROM lhs_values
            CROSS JOIN rhs_values
        ),
        pairs AS (
            SELECT lhs_item, rhs_item
            FROM lhs_values
            CROSS JOIN rhs_values
            CROSS JOIN LATERAL unnest(lhs_values.arr, rhs_values.arr) AS values(lhs_item, rhs_item)
        )
        SELECT CASE
            WHEN dims.lhs_dim IS DISTINCT FROM dims.rhs_dim THEN NULL
            ELSE coalesce(sum(abs(pairs.lhs_item - pairs.rhs_item)), 0)
        END
        FROM dims
        LEFT JOIN pairs ON TRUE
        GROUP BY dims.lhs_dim, dims.rhs_dim
    $$
    """,
    """
    CREATE OPERATOR <=> (
        LEFTARG = vector,
        RIGHTARG = vector,
        PROCEDURE = vector_cosine_distance
    )
    """,
    """
    CREATE OPERATOR <-> (
        LEFTARG = vector,
        RIGHTARG = vector,
        PROCEDURE = vector_l2_distance
    )
    """,
    """
    CREATE OPERATOR <#> (
        LEFTARG = vector,
        RIGHTARG = vector,
        PROCEDURE = vector_max_inner_product
    )
    """,
    """
    CREATE OPERATOR <+> (
        LEFTARG = vector,
        RIGHTARG = vector,
        PROCEDURE = vector_l1_distance
    )
    """,
)
DROP_COMPAT_SQL = (
    "DROP OPERATOR IF EXISTS <+> (vector, vector)",
    "DROP OPERATOR IF EXISTS <#> (vector, vector)",
    "DROP OPERATOR IF EXISTS <-> (vector, vector)",
    "DROP OPERATOR IF EXISTS <=> (vector, vector)",
    "DROP FUNCTION IF EXISTS vector_l1_distance(vector, vector)",
    "DROP FUNCTION IF EXISTS vector_max_inner_product(vector, vector)",
    "DROP FUNCTION IF EXISTS vector_l2_distance(vector, vector)",
    "DROP FUNCTION IF EXISTS vector_cosine_distance(vector, vector)",
    "DROP FUNCTION IF EXISTS vector_to_float8_array(vector)",
    "DROP DOMAIN IF EXISTS vector",
)


async def list_public_tables(connection: asyncpg.Connection) -> list[str]:
    rows = await connection.fetch(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
    )
    return [str(row["table_name"]) for row in rows]


async def assert_supported_source_schema(connection: asyncpg.Connection) -> list[str]:
    source_tables = await list_public_tables(connection)
    unknown_tables = [table for table in source_tables if table not in APP_TABLES]
    if unknown_tables:
        raise RuntimeError(f"Source has unsupported tables for this migration script: {', '.join(unknown_tables)}")
    return source_tables


async def clear_target(connection: asyncpg.Connection) -> None:
    existing_tables = await list_public_tables(connection)
    unexpected_tables = [
        table
        for table in existing_tables
        if table not in APP_TABLES and table not in TEMP_TABLES
    ]
    if unexpected_tables:
        raise RuntimeError(
            "Target contains unexpected public tables and was not modified: "
            + ", ".join(unexpected_tables)
        )
    for table_name in sorted(existing_tables, reverse=True):
        await connection.execute(f'DROP TABLE IF EXISTS public."{table_name}" CASCADE')
    has_vector_type = bool(await connection.fetchval("SELECT to_regtype('vector') IS NOT NULL"))
    has_vector_extension = bool(
        await connection.fetchval("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
    )
    if has_vector_type and not has_vector_extension:
        for statement in DROP_COMPAT_SQL:
            await connection.execute(statement)


async def ensure_vector_support(connection: asyncpg.Connection) -> str:
    await connection.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    try:
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        return "native"
    except asyncpg.PostgresError:
        for statement in COMPAT_DDL:
            await connection.execute(statement)
        return "compat"


def configure_metadata_for_target(mode: str) -> None:
    if mode != "compat":
        return
    embedding_column = EvidenceSource.__table__.c.embedding
    embedding_column.type = Vector()
    compat_index = next(
        (index for index in EvidenceSource.__table__.indexes if index.name == "ix_evidence_sources_embedding_hnsw"),
        None,
    )
    if compat_index is not None:
        EvidenceSource.__table__.indexes.remove(compat_index)


def to_asyncpg_sqlalchemy_dsn(dsn: str) -> str:
    normalized = dsn.strip()
    if normalized.startswith("postgres://"):
        normalized = "postgresql://" + normalized[len("postgres://") :]
    if normalized.startswith("postgresql://"):
        normalized = normalized.replace("postgresql://", "postgresql+asyncpg://", 1)
    parts = urlsplit(normalized)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    if "sslmode" in query and "ssl" not in query:
        query["ssl"] = query.pop("sslmode")
    query.pop("channel_binding", None)
    return urlunsplit(parts._replace(query=urlencode(query)))


async def create_target_schema(target_dsn: str, mode: str) -> None:
    configure_metadata_for_target(mode)
    engine = create_async_engine(to_asyncpg_sqlalchemy_dsn(target_dsn))
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()


async def fetch_columns(connection: asyncpg.Connection, table_name: str) -> list[str]:
    rows = await connection.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = $1
        ORDER BY ordinal_position
        """,
        table_name,
    )
    return [str(row["column_name"]) for row in rows]


def batched(values: list[tuple[object, ...]], size: int) -> Iterable[list[tuple[object, ...]]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


async def copy_table(source: asyncpg.Connection, target: asyncpg.Connection, table_name: str) -> int:
    columns = await fetch_columns(source, table_name)
    if not columns:
        return 0
    rows = await source.fetch(f'SELECT * FROM public."{table_name}"')
    if not rows:
        return 0
    column_sql = ", ".join(f'"{column}"' for column in columns)
    value_sql = ", ".join(f"${index}" for index in range(1, len(columns) + 1))
    insert_sql = f'INSERT INTO public."{table_name}" ({column_sql}) VALUES ({value_sql})'
    records = [tuple(row[column] for column in columns) for row in rows]
    for chunk in batched(records, size=200):
        await target.executemany(insert_sql, chunk)
    return len(records)


async def collect_counts(connection: asyncpg.Connection, table_names: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table_name in sorted(table_names):
        counts[table_name] = int(await connection.fetchval(f'SELECT count(*) FROM public."{table_name}"'))
    return counts


async def verify_vector_query(connection: asyncpg.Connection) -> str:
    value = await connection.fetchval(
        """
        SELECT embedding <=> embedding
        FROM public.evidence_sources
        WHERE embedding IS NOT NULL
        LIMIT 1
        """
    )
    return "ok" if value is not None else "no-embeddings"


async def migrate(source_dsn: str, target_dsn: str) -> None:
    source = await asyncpg.connect(dsn=source_dsn)
    target = await asyncpg.connect(dsn=target_dsn)
    try:
        source_tables = await assert_supported_source_schema(source)
        await clear_target(target)
        mode = await ensure_vector_support(target)
        await create_target_schema(target_dsn, mode)
        inserted_counts: dict[str, int] = {}
        ordered_tables = [table for table in APP_TABLES if table in source_tables]
        for table_name in ordered_tables:
            inserted_counts[table_name] = await copy_table(source, target, table_name)
        source_counts = await collect_counts(source, ordered_tables)
        target_counts = await collect_counts(target, ordered_tables)
        if source_counts != target_counts:
            raise RuntimeError(
                f"Row count mismatch after migration. source={source_counts} target={target_counts}"
            )
        vector_status = await verify_vector_query(target)
        print(f"Migration complete. vector_mode={mode}")
        print(f"Inserted rows: {inserted_counts}")
        print(f"Verified counts: {target_counts}")
        print(f"Vector query check: {vector_status}")
    finally:
        await source.close()
        await target.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate the JachAI PostgreSQL database.")
    parser.add_argument("--source-dsn", required=True, help="Source PostgreSQL DSN")
    parser.add_argument("--target-dsn", required=True, help="Target PostgreSQL DSN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(migrate(args.source_dsn, args.target_dsn))


if __name__ == "__main__":
    main()
