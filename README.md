# JachAI Backend

FastAPI backend for JachAI, a trilingual and multimodal misinformation verification platform focused on South Asia.

## Features

- Async FastAPI APIs for text, image, and URL claim verification
- Neon PostgreSQL + `pgvector` retrieval with 1024-dim BGE-M3 embeddings
- Redis-backed duplicate detection, result caching, rate limiting, and job status
- OCR with Tesseract for uploaded screenshots
- Single NVIDIA LLM call per uncached claim after retrieval
- Dashboard, rumor cluster, source ingestion, and webhook endpoints

## Quick Start

1. Copy `.env.example` to `.env` and fill in your Neon, Redis, and NVIDIA credentials.
2. Enable extensions in your database:

```sql
\i scripts/init_db.sql
```

3. Install dependencies and run the app:

```bash
uv sync
uv run uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`, with docs at `/docs`.

## Expected Environment

- Python 3.11 or 3.12
- Tesseract OCR installed locally with Bangla and Hindi data
- Neon PostgreSQL with `pgvector`
- Redis

## Selected Endpoints

- `GET /health`
- `GET /api/v1/health/system`
- `POST /api/v1/claims/text`
- `POST /api/v1/claims/image`
- `POST /api/v1/claims/url`
- `GET /api/v1/dashboard/summary`
- `POST /api/v1/sources/ingest`

## Notes

- The backend keeps LLM usage rate-safe by checking cache and retrieval first.
- If the NVIDIA call fails twice or rate limit safety is exceeded, the app falls back to `Not Enough Evidence`.
- `POST /api/v1/sources/ingest`, `PATCH /api/v1/claims/{claim_id}/review-status`, and the webhook endpoint expect `X-Internal-API-Key`.
