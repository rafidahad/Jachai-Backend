## 18. Full Codex Prompt

Use this prompt in Codex/CodeX/agentic coding tool.

```txt
You are building the backend for an AI hackathon project called JachAI.

JachAI is a trilingual, multimodal misinformation verification platform for South Asia. It verifies suspicious text, URLs, and screenshots in Bangla, English, Hindi, Banglish, and Hinglish.

Important infrastructure:
- I already have NeonDB PostgreSQL.
- I already have online Redis.
- I already have an NVIDIA API key.
- The backend must use FastAPI.
- The database must use NeonDB PostgreSQL with pgvector.
- Redis must be used for duplicate claim cache, result cache, rate limiting, and job status.
- NVIDIA API has a 40 RPM limit, so use only one NVIDIA LLM call per claim after retrieval.
- Do not use NVIDIA for embeddings.
- Use local embedding model BAAI/bge-m3 with 1024-dimensional vectors.
- Use Tesseract OCR for screenshots.
- Use Pydantic strict schemas for LLM output and API responses.

Create a production-clean FastAPI backend with this structure:

backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── routes_claims.py
│   │   ├── routes_verification_jobs.py
│   │   ├── routes_dashboard.py
│   │   ├── routes_clusters.py
│   │   ├── routes_sources.py
│   │   ├── routes_health.py
│   │   └── routes_webhooks.py
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── redis.py
│   │   ├── security.py
│   │   ├── logging.py
│   │   └── rate_limit.py
│   ├── db/
│   │   ├── base.py
│   │   └── session.py
│   ├── models/
│   │   ├── claim.py
│   │   ├── evidence_source.py
│   │   ├── verification_job.py
│   │   ├── rumor_cluster.py
│   │   └── audit_log.py
│   ├── schemas/
│   │   ├── claim_schema.py
│   │   ├── source_schema.py
│   │   ├── verdict_schema.py
│   │   ├── dashboard_schema.py
│   │   ├── cluster_schema.py
│   │   └── health_schema.py
│   ├── services/
│   │   ├── claim_pipeline.py
│   │   ├── ocr_service.py
│   │   ├── text_cleaning_service.py
│   │   ├── pii_service.py
│   │   ├── language_service.py
│   │   ├── embedding_service.py
│   │   ├── retrieval_service.py
│   │   ├── nvidia_llm_service.py
│   │   ├── source_service.py
│   │   ├── dashboard_service.py
│   │   ├── cluster_service.py
│   │   ├── health_service.py
│   │   └── cache_service.py
│   └── utils/
│       ├── hashing.py
│       ├── time.py
│       └── errors.py
├── scripts/
│   ├── init_db.sql
│   ├── seed_evidence.py
│   └── test_pipeline.py
├── tests/
├── .env.example
├── pyproject.toml
├── alembic.ini
├── Dockerfile
└── README.md

Implement these API endpoints:

GET /health
GET /api/v1/health/system

POST /api/v1/claims/text
POST /api/v1/claims/image
POST /api/v1/claims/url
GET /api/v1/verification-jobs/{job_id}
GET /api/v1/claims/{claim_id}
GET /api/v1/claims/{claim_id}/share-summary
PATCH /api/v1/claims/{claim_id}/review-status
GET /api/v1/claims

POST /api/v1/sources/ingest
GET /api/v1/sources
GET /api/v1/sources/{source_id}

GET /api/v1/dashboard/summary
GET /api/v1/dashboard/verdict-distribution
GET /api/v1/dashboard/language-distribution
GET /api/v1/dashboard/claims-over-time
GET /api/v1/dashboard/recent-claims

GET /api/v1/rumor-clusters
GET /api/v1/rumor-clusters/{cluster_id}

POST /api/v1/webhooks/inbound-message

Database requirements:
- Use SQLAlchemy async.
- Use asyncpg.
- Use UUID primary keys.
- Evidence embeddings must be vector(1024).
- Include SQL script to enable vector and pgcrypto extensions.
- Include HNSW index for evidence embedding with vector_cosine_ops.

Claim pipeline:
1. Validate input.
2. Clean text.
3. Mask PII.
4. Detect language using local rules.
5. Generate normalized hash.
6. Check Redis cache before AI.
7. Generate embedding locally with BAAI/bge-m3.
8. Search top 5 evidence sources using pgvector cosine similarity.
9. Call NVIDIA LLM once with cleaned text and evidence.
10. Parse strict JSON response with Pydantic.
11. Save claim, verYou are building the backend for an AI hackathon project called JachAI.

JachAI is a trilingual, multimodal misinformation verification platform for South Asia. It verifies suspicious text, URLs, and screenshots in Bangla, English, Hindi, Banglish, and Hinglish.

Important infrastructure:
- I already have NeonDB PostgreSQL.
- I already have online Redis.
- I already have an NVIDIA API key.
- The backend must use FastAPI.
- The database must use NeonDB PostgreSQL with pgvector.
- Redis must be used for duplicate claim cache, result cache, rate limiting, and job status.
- NVIDIA API has a 40 RPM limit, so use only one NVIDIA LLM call per claim after retrieval.
- Do not use NVIDIA for embeddings.
- Use local embedding model BAAI/bge-m3 with 1024-dimensional vectors.
- Use Tesseract OCR for screenshots.
- Use Pydantic strict schemas for LLM output and API responses.

Create a production-clean FastAPI backend with this structure:

backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── routes_claims.py
│   │   ├── routes_verification_jobs.py
│   │   ├── routes_dashboard.py
│   │   ├── routes_clusters.py
│   │   ├── routes_sources.py
│   │   ├── routes_health.py
│   │   └── routes_webhooks.py
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── redis.py
│   │   ├── security.py
│   │   ├── logging.py
│   │   └── rate_limit.py
│   ├── db/
│   │   ├── base.py
│   │   └── session.py
│   ├── models/
│   │   ├── claim.py
│   │   ├── evidence_source.py
│   │   ├── verification_job.py
│   │   ├── rumor_cluster.py
│   │   └── audit_log.py
│   ├── schemas/
│   │   ├── claim_schema.py
│   │   ├── source_schema.py
│   │   ├── verdict_schema.py
│   │   ├── dashboard_schema.py
│   │   ├── cluster_schema.py
│   │   └── health_schema.py
│   ├── services/
│   │   ├── claim_pipeline.py
│   │   ├── ocr_service.py
│   │   ├── text_cleaning_service.py
│   │   ├── pii_service.py
│   │   ├── language_service.py
│   │   ├── embedding_service.py
│   │   ├── retrieval_service.py
│   │   ├── nvidia_llm_service.py
│   │   ├── source_service.py
│   │   ├── dashboard_service.py
│   │   ├── cluster_service.py
│   │   ├── health_service.py
│   │   └── cache_service.py
│   └── utils/
│       ├── hashing.py
│       ├── time.py
│       └── errors.py
├── scripts/
│   ├── init_db.sql
│   ├── seed_evidence.py
│   └── test_pipeline.py
├── tests/
├── .env.example
├── pyproject.toml
├── alembic.ini
├── Dockerfile
└── README.md

Implement these API endpoints:

GET /health
GET /api/v1/health/system

POST /api/v1/claims/text
POST /api/v1/claims/image
POST /api/v1/claims/url
GET /api/v1/verification-jobs/{job_id}
GET /api/v1/claims/{claim_id}
GET /api/v1/claims/{claim_id}/share-summary
PATCH /api/v1/claims/{claim_id}/review-status
GET /api/v1/claims

POST /api/v1/sources/ingest
GET /api/v1/sources
GET /api/v1/sources/{source_id}

GET /api/v1/dashboard/summary
GET /api/v1/dashboard/verdict-distribution
GET /api/v1/dashboard/language-distribution
GET /api/v1/dashboard/claims-over-time
GET /api/v1/dashboard/recent-claims

GET /api/v1/rumor-clusters
GET /api/v1/rumor-clusters/{cluster_id}

POST /api/v1/webhooks/inbound-message

Database requirements:
- Use SQLAlchemy async.
- Use asyncpg.
- Use UUID primary keys.
- Evidence embeddings must be vector(1024).
- Include SQL script to enable vector and pgcrypto extensions.
- Include HNSW index for evidence embedding with vector_cosine_ops.

Claim pipeline:
1. Validate input.
2. Clean text.
3. Mask PII.
4. Detect language using local rules.
5. Generate normalized hash.
6. Check Redis cache before AI.
7. Generate embedding locally with BAAI/bge-m3.
8. Search top 5 evidence sources using pgvector cosine similarity.
9. Call NVIDIA LLM once with cleaned text and evidence.
10. Parse strict JSON response with Pydantic.
11. Save claim, verdict, and linked evidence.
12. Cache final result in Redis.
13. Return frontend-compatible response.

NVIDIA LLM:
- Use OpenAI-compatible client.
- Base URL comes from NVIDIA_BASE_URL.
- API key comes from NVIDIA_API_KEY.
- Model name comes from NVIDIA_LLM_MODEL.
- Temperature should be low, around 0.1.
- Return strict JSON only.
- If model fails or returns invalid JSON, retry once.
- If still fails, return a safe fallback with verdict "Not Enough Evidence".

OCR:
- Use pytesseract and Pillow.
- Support multipart image upload.
- If OCR extracts too little text, return OCR_FAILED error.

Dashboard:
- Summary metrics.
- Verdict distribution.
- Language distribution.
- Claims over time.
- Recent claims.
- Rumor clusters.

Health:
- Check DB connection.
- Check Redis ping.
- Check NVIDIA API key configured.
- Check Tesseract availability.
- Return frontend-friendly status.

Create a clean .env.example with:
DATABASE_URL
DATABASE_SYNC_URL
REDIS_URL
NVIDIA_API_KEY
NVIDIA_BASE_URL
NVIDIA_LLM_MODEL
EMBEDDING_MODEL
EMBEDDING_DIM
INTERNAL_API_KEY
BACKEND_CORS_ORIGINS

Use uv for package management. Create pyproject.toml with all required dependencies.

Important:
- Do not hardcode secrets.
- Do not create local Postgres as the default.
- Do not use Gemini.
- Do not use NVIDIA for every small step.
- Keep LLM usage rate-limit safe.
- Make the backend runnable with:
  uv sync
  uv run uvicorn app.main:app --reload

Start by creating the project skeleton, config, database connection, Redis connection, health endpoint, and source ingestion endpoint. Then implement the claim pipeline.dict, and linked evidence.
12. Cache final result in Redis.
13. Return frontend-compatible response.

NVIDIA LLM:
- Use OpenAI-compatible client.
- Base URL comes from NVIDIA_BASE_URL.
- API key comes from NVIDIA_API_KEY.
- Model name comes from NVIDIA_LLM_MODEL.
- Temperature should be low, around 0.1.
- Return strict JSON only.
- If model fails or returns invalid JSON, retry once.
- If still fails, return a safe fallback with verdict "Not Enough Evidence".

OCR:
- Use pytesseract and Pillow.
- Support multipart image upload.
- If OCR extracts too little text, return OCR_FAILED error.

Dashboard:
- Summary metrics.
- Verdict distribution.
- Language distribution.
- Claims over time.
- Recent claims.
- Rumor clusters.

Health:
- Check DB connection.
- Check Redis ping.
- Check NVIDIA API key configured.
- Check Tesseract availability.
- Return frontend-friendly status.

Create a clean .env.example with:
DATABASE_URL
DATABASE_SYNC_URL
REDIS_URL
NVIDIA_API_KEY
NVIDIA_BASE_URL
NVIDIA_LLM_MODEL
EMBEDDING_MODEL
EMBEDDING_DIM
INTERNAL_API_KEY
BACKEND_CORS_ORIGINS

Use uv for package management. Create pyproject.toml with all required dependencies.

Important:
- Do not hardcode secrets.
- Do not create local Postgres as the default.
- Do not use Gemini.
- Do not use NVIDIA for every small step.
- Keep LLM usage rate-limit safe.
- Make the backend runnable with:
  uv sync
  uv run uvicorn app.main:app --reload

Start by creating the project skeleton, config, database connection, Redis connection, health endpoint, and source ingestion endpoint. Then implement the claim pipeline.
```

---

## 19. Smaller Codex Prompts

### Prompt 1: Backend Skeleton

```txt
Create the FastAPI backend skeleton for JachAI.

Use:
- FastAPI
- Pydantic Settings
- SQLAlchemy async
- asyncpg
- Redis async client
- uv package manager

Create folders:
app/api
app/core
app/db
app/models
app/schemas
app/services
app/utils
scripts
tests

Implement:
- app/main.py
- app/core/config.py
- app/core/database.py
- app/core/redis.py
- GET /health
- .env.example
- pyproject.toml

The database is NeonDB and comes from DATABASE_URL.
Redis is online and comes from REDIS_URL.
Do not hardcode secrets.
```

### Prompt 2: Database Models

```txt
Add SQLAlchemy async models and SQL init script for JachAI.

Tables:
- claims
- verification_jobs
- evidence_sources
- claim_evidence_links
- rumor_clusters
- audit_logs

Use UUID primary keys.
Use pgvector vector(1024) for evidence_sources.embedding.
Create scripts/init_db.sql with:
CREATE EXTENSION vector
CREATE EXTENSION pgcrypto
all tables
all indexes
HNSW index for evidence embedding using vector_cosine_ops.
```

### Prompt 3: Evidence Source Ingestion

```txt
Implement POST /api/v1/sources/ingest.

Request fields:
title, publisher, url, language, source_type, published_at, content.

Behavior:
- validate data
- compute content hash
- check duplicate by URL or content hash
- generate embedding using local BAAI/bge-m3
- save to evidence_sources
- return source_id and embedding_status

Also implement:
GET /api/v1/sources
GET /api/v1/sources/{source_id}
```

### Prompt 4: Retrieval Service

```txt
Implement pgvector retrieval service.

Function:
search_similar_evidence(embedding, top_k=5)

Use cosine distance:
embedding <=> query_embedding

Return:
id, title, url, publisher, language, source_type, content snippet, similarity, credibility_score, published_at.
```

### Prompt 5: NVIDIA LLM Service

```txt
Implement NVIDIA LLM service using OpenAI-compatible client.

Use:
NVIDIA_API_KEY
NVIDIA_BASE_URL
NVIDIA_LLM_MODEL

Function:
generate_verdict_with_nvidia(cleaned_text, detected_language, evidence)

Rules:
- low temperature
- strict JSON only
- parse with Pydantic VerdictLLMOutput
- retry once if invalid JSON
- fallback to Not Enough Evidence if still invalid
- do not reveal chain-of-thought
```

### Prompt 6: Claim Pipeline

```txt
Implement claim verification pipeline.

For text:
- validate text
- clean text
- mask PII
- detect language locally
- hash normalized text
- check Redis cache
- generate local embedding
- retrieve top 5 evidence from pgvector
- call NVIDIA once
- save claim
- save evidence links
- cache result
- return frontend response

For image:
- extract text using pytesseract
- merge caption
- call text pipeline

For URL:
- fetch URL with httpx
- parse title/body with BeautifulSoup
- call text pipeline
```

### Prompt 7: API Routes

```txt
Implement API routes:

POST /api/v1/claims/text
POST /api/v1/claims/image
POST /api/v1/claims/url
GET /api/v1/claims/{claim_id}
GET /api/v1/claims/{claim_id}/share-summary
PATCH /api/v1/claims/{claim_id}/review-status
GET /api/v1/claims

Connect them to the claim pipeline and database.
Make response shape match the frontend API contract.
```

### Prompt 8: Dashboard and Health APIs

```txt
Implement dashboard and health APIs.

Dashboard:
GET /api/v1/dashboard/summary
GET /api/v1/dashboard/verdict-distribution
GET /api/v1/dashboard/language-distribution
GET /api/v1/dashboard/claims-over-time
GET /api/v1/dashboard/recent-claims

Clusters:
GET /api/v1/rumor-clusters
GET /api/v1/rumor-clusters/{cluster_id}

Health:
GET /api/v1/health/system

Use database aggregations where possible.
Return frontend-friendly JSON.
```

---

## 20. MVP Seed Evidence Data

Create `scripts/seed_evidence.py` that ingests records through the backend service or directly through DB session.

Minimum demo dataset:

1. Fake free internet/data offer
2. Fake government cash support link
3. Fake bank app update warning
4. Election misinformation claim
5. Fake celebrity death claim
6. Health remedy misinformation
7. Disaster warning rumor
8. Fake school/university closure notice
9. Fake job circular scam
10. Fake news screenshot example

Each seed item should include:

```json
{
  "title": "Fact Check: No free mobile data campaign from government",
  "publisher": "Demo Fact Check",
  "url": "https://example.com/demo-free-data-factcheck",
  "language": "Bangla",
  "source_type": "fact_check",
  "published_at": "2026-06-01T10:00:00Z",
  "content": "A viral message claiming that the government is offering free mobile data through a WhatsApp link is false. No verified government or telecom source has announced such an offer. Similar messages are used for phishing."
}
```

---

## 21. Testing Checklist

### Backend

```txt
[ ] FastAPI starts
[ ] /health works
[ ] DB connection works
[ ] Redis ping works
[ ] pgvector extension enabled
[ ] Source ingestion works
[ ] Embedding generated and saved
[ ] pgvector retrieval returns similar evidence
[ ] Text claim verification works
[ ] Image OCR works
[ ] URL verification works
[ ] NVIDIA response parses as JSON
[ ] Cache prevents repeated LLM calls
[ ] Dashboard APIs return data
[ ] Health API returns service status
```

### Rate Limit

```txt
[ ] Repeated same claim returns from Redis
[ ] Backend does not call NVIDIA before cache check
[ ] Backend does not call NVIDIA for embeddings
[ ] Backend uses one LLM call per claim
[ ] Rate limiter blocks above configured RPM
```

---

## 22. Run Commands

Install:

```bash
uv sync
```

Run:

```bash
uv run uvicorn app.main:app --reload
```

Run tests:

```bash
uv run pytest
```

Run seed script:

```bash
uv run python scripts/seed_evidence.py
```

---

## 23. Deployment Notes

Backend deployment environment must set:

```env
DATABASE_URL=
REDIS_URL=
NVIDIA_API_KEY=
NVIDIA_BASE_URL=
NVIDIA_LLM_MODEL=
INTERNAL_API_KEY=
BACKEND_CORS_ORIGINS=
```

Recommended deployment:

- Backend: Render/Railway/Fly.io/VPS
- Database: NeonDB
- Redis: Upstash/Railway Redis
- Frontend: Vercel

---

## 24. Final MVP Success Criteria

The backend is MVP-ready when:

```txt
[ ] User can submit a text claim and receive a verdict.
[ ] User can upload a screenshot and receive a verdict.
[ ] Evidence is retrieved from pgvector.
[ ] NVIDIA is called only once per uncached claim.
[ ] Repeated claims are cached in Redis.
[ ] Claims and verdicts are stored in NeonDB.
[ ] Dashboard APIs show real data.
[ ] Frontend can integrate without changing response shapes.
[ ] System health API works.
```
