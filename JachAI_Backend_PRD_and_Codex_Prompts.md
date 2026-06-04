# JachAI Backend PRD & Codex Implementation Prompt

## 1. Project Overview

**Project name:** JachAI  
**Tagline:** Don’t forward it. JachAI it.

JachAI is a trilingual, multimodal misinformation verification system for South Asia. The backend receives suspicious text, URLs, or screenshots, extracts the core claim, retrieves trusted evidence from a local RAG database, uses an NVIDIA Build endpoint to generate an evidence-grounded verdict, and returns the result to the frontend and admin dashboard.

The backend must be designed for:

- FastAPI-based REST APIs
- NeonDB PostgreSQL with `pgvector`
- Online Redis for cache, duplicate detection, and rate limiting
- NVIDIA Build API for LLM reasoning
- Local OCR with Tesseract
- Local embedding model to reduce NVIDIA API calls
- Strict JSON responses for frontend integration
- 40 RPM NVIDIA model rate limit safety

---

## 2. Existing Infrastructure

You already have:

- NeonDB PostgreSQL database
- Online Redis instance
- NVIDIA API key
- Stitch-generated frontend design
- API contract planned for frontend integration

The backend should not create local PostgreSQL by default. It must connect to NeonDB through environment variables.

---

## 3. Backend Goals

The backend must:

1. Accept text, image, and URL-based suspicious claims.
2. Extract useful text from screenshots using OCR.
3. Clean and sanitize input.
4. Remove or mask personal identifiers.
5. Detect language/code-mix: Bangla, English, Hindi, Banglish, Hinglish, Mixed.
6. Retrieve similar trusted evidence from NeonDB using pgvector.
7. Use NVIDIA model only after retrieval to reduce API usage.
8. Return verdict: Likely True, Likely False, Misleading, or Not Enough Evidence.
9. Store claim result, evidence links, and audit logs.
10. Expose APIs required by the frontend.
11. Expose dashboard APIs.
12. Support seed evidence ingestion.
13. Be deployable to Render, Railway, Fly.io, or VPS.

---

## 4. Backend Tech Stack

### Core

- Python 3.11 or 3.12
- FastAPI
- Uvicorn
- Pydantic v2
- Pydantic Settings
- SQLAlchemy 2.x async
- asyncpg
- Alembic
- Redis async client
- httpx
- python-multipart
- Pillow
- pytesseract
- sentence-transformers
- NumPy
- scikit-learn
- beautifulsoup4
- OpenAI-compatible client for NVIDIA endpoint

### Database

- NeonDB PostgreSQL
- `pgvector` extension
- `pgcrypto` extension for UUID generation

### Cache

- Online Redis
- Used for duplicate claim cache, claim result cache, rate limiting, verification job status, and temporary workflow locks.

### AI Models

#### LLM

Use NVIDIA Build endpoint through OpenAI-compatible API.

The exact model ID should be configured from `.env`.

The LLM is responsible for:

- Claim extraction
- Evidence-based reasoning
- Verdict generation
- Final user response

Because the project has a 40 RPM NVIDIA limit, the backend must use only **one NVIDIA LLM call per claim** wherever possible.

#### Embeddings

Use a local embedding model to avoid consuming NVIDIA rate limit.

Recommended model:

```txt
BAAI/bge-m3
```

Important:

- BGE-M3 outputs 1024-dimensional embeddings.
- Database vector column should be `vector(1024)`.

Alternative if local machine is too slow:

```txt
sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

That model outputs 768-dimensional embeddings. If you use it, change the DB column to `vector(768)`.

For this PRD, use **BGE-M3 with `vector(1024)`**.

### OCR

- Tesseract OCR
- Bangla, English, Hindi language packages where available
- Do not use NVIDIA vision model in MVP unless OCR fails and the rate limit allows it.

---

## 5. Rate Limit Strategy: 40 RPM NVIDIA Limit

Backend rules:

1. Do not call NVIDIA for language detection.
2. Do not call NVIDIA for simple validation.
3. Do not call NVIDIA for embeddings.
4. Do not call NVIDIA before Redis duplicate check.
5. Do not call NVIDIA before pgvector retrieval.
6. Use a single LLM call for claim extraction, evidence reasoning, verdict generation, and final user response.
7. Keep a safety budget of 30 RPM in code.
8. Use Redis to cache repeated rumor results for 24 to 48 hours.
9. If rate limit is hit, return `status: queued` or a safe temporary error.
10. Add retry with exponential backoff only for temporary API failures.

Recommended throttle:

```txt
Max NVIDIA calls: 30 RPM
Minimum delay between uncached LLM calls: 2 seconds
```

---

## 6. High-Level Backend Architecture

```txt
Frontend / n8n / API Client
        ↓
FastAPI API Routes
        ↓
Claim Pipeline
        ├── Input validation
        ├── OCR if image
        ├── Text cleaning
        ├── PII masking
        ├── Rule-based language detection
        ├── Redis duplicate cache check
        ├── Local embedding generation
        ├── pgvector evidence retrieval
        ├── One NVIDIA LLM call
        ├── Save claim result
        ├── Save evidence links
        ├── Cache result
        └── Return response
        ↓
NeonDB PostgreSQL + pgvector
Redis
```

---

## 7. Recommended Repository Structure

```txt
backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes_claims.py
│   │   ├── routes_verification_jobs.py
│   │   ├── routes_dashboard.py
│   │   ├── routes_clusters.py
│   │   ├── routes_sources.py
│   │   ├── routes_health.py
│   │   └── routes_webhooks.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── redis.py
│   │   ├── security.py
│   │   ├── logging.py
│   │   └── rate_limit.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── session.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── claim.py
│   │   ├── evidence_source.py
│   │   ├── verification_job.py
│   │   ├── rumor_cluster.py
│   │   └── audit_log.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── claim_schema.py
│   │   ├── source_schema.py
│   │   ├── verdict_schema.py
│   │   ├── dashboard_schema.py
│   │   ├── cluster_schema.py
│   │   └── health_schema.py
│   ├── services/
│   │   ├── __init__.py
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
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── hashing.py
│   │   ├── time.py
│   │   └── errors.py
│   └── workers/
│       ├── __init__.py
│       └── seed_worker.py
├── alembic/
├── scripts/
│   ├── init_db.sql
│   ├── seed_evidence.py
│   └── test_pipeline.py
├── tests/
│   ├── test_health.py
│   ├── test_claims.py
│   └── test_sources.py
├── .env.example
├── pyproject.toml
├── alembic.ini
├── Dockerfile
└── README.md
```

---

## 8. Environment Variables

Create `.env.example`:

```env
APP_NAME=JachAI
ENVIRONMENT=development
DEBUG=true

API_V1_PREFIX=/api/v1
BACKEND_CORS_ORIGINS=http://localhost:3000,http://localhost:8501

DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@HOST/jachai_db?ssl=require
DATABASE_SYNC_URL=postgresql://USER:PASSWORD@HOST/jachai_db?sslmode=require

REDIS_URL=redis://default:PASSWORD@HOST:PORT/0

INTERNAL_API_KEY=change_this_internal_key

NVIDIA_API_KEY=nvapi-your-key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_LLM_MODEL=replace-with-model-id-from-nvidia-build
NVIDIA_MAX_RPM=30
NVIDIA_TIMEOUT_SECONDS=60

EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
EMBEDDING_DEVICE=cpu

TESSERACT_CMD=
OCR_LANGUAGES=eng+ben+hin

CACHE_TTL_SECONDS=172800
JOB_TTL_SECONDS=3600

LOG_LEVEL=INFO
```

---

## 9. Database Setup

Run this in Neon SQL editor before running the backend:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

### 9.1 Tables

```sql
CREATE TABLE IF NOT EXISTS claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    original_text TEXT,
    cleaned_text TEXT,
    extracted_claim TEXT NOT NULL,
    detected_language VARCHAR(50),
    source_type VARCHAR(50),
    source_url TEXT,
    category VARCHAR(100),
    verdict VARCHAR(50),
    confidence FLOAT,
    confidence_label VARCHAR(50),
    explanation TEXT,
    ai_analysis TEXT,
    user_response TEXT,
    review_status VARCHAR(50) DEFAULT 'auto_verified',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS verification_jobs (
    id VARCHAR(100) PRIMARY KEY,
    claim_id UUID,
    status VARCHAR(50) NOT NULL DEFAULT 'processing',
    current_step VARCHAR(100),
    progress INTEGER DEFAULT 0,
    error_code VARCHAR(100),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evidence_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    url TEXT UNIQUE,
    publisher VARCHAR(255),
    language VARCHAR(50),
    source_type VARCHAR(50),
    published_at TIMESTAMP,
    content TEXT NOT NULL,
    content_hash TEXT,
    embedding vector(1024),
    embedding_status VARCHAR(50) DEFAULT 'pending',
    credibility_score FLOAT DEFAULT 0.75,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS claim_evidence_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID REFERENCES claims(id) ON DELETE CASCADE,
    evidence_id UUID REFERENCES evidence_sources(id) ON DELETE CASCADE,
    similarity FLOAT,
    rank INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rumor_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT,
    representative_claim TEXT,
    language VARCHAR(50),
    category VARCHAR(100),
    risk_level VARCHAR(50) DEFAULT 'Monitoring',
    submission_count INTEGER DEFAULT 1,
    viral_velocity FLOAT DEFAULT 0,
    first_seen_at TIMESTAMP DEFAULT NOW(),
    last_seen_at TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID,
    job_id VARCHAR(100),
    step VARCHAR(100),
    status VARCHAR(50),
    message TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 9.2 Indexes

```sql
CREATE INDEX IF NOT EXISTS idx_claims_language ON claims(detected_language);
CREATE INDEX IF NOT EXISTS idx_claims_verdict ON claims(verdict);
CREATE INDEX IF NOT EXISTS idx_claims_created_at ON claims(created_at);
CREATE INDEX IF NOT EXISTS idx_claims_review_status ON claims(review_status);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON verification_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_claim_id ON verification_jobs(claim_id);

CREATE INDEX IF NOT EXISTS idx_evidence_publisher ON evidence_sources(publisher);
CREATE INDEX IF NOT EXISTS idx_evidence_language ON evidence_sources(language);
CREATE INDEX IF NOT EXISTS idx_evidence_source_type ON evidence_sources(source_type);
CREATE INDEX IF NOT EXISTS idx_evidence_hash ON evidence_sources(content_hash);

CREATE INDEX IF NOT EXISTS idx_clusters_risk_level ON rumor_clusters(risk_level);
CREATE INDEX IF NOT EXISTS idx_clusters_language ON rumor_clusters(language);
CREATE INDEX IF NOT EXISTS idx_clusters_last_seen ON rumor_clusters(last_seen_at);

CREATE INDEX IF NOT EXISTS evidence_embedding_hnsw_idx
ON evidence_sources
USING hnsw (embedding vector_cosine_ops);
```

---

## 10. Required API Endpoints

Base URL:

```txt
/api/v1
```

### Health

```http
GET /health
GET /api/v1/health/system
```

### Claims

```http
POST /api/v1/claims/text
POST /api/v1/claims/image
POST /api/v1/claims/url
GET /api/v1/verification-jobs/{job_id}
GET /api/v1/claims/{claim_id}
GET /api/v1/claims/{claim_id}/share-summary
PATCH /api/v1/claims/{claim_id}/review-status
GET /api/v1/claims?search=&verdict=&language=&source_type=&status=&page=1&page_size=20
```

### Dashboard

```http
GET /api/v1/dashboard/summary
GET /api/v1/dashboard/verdict-distribution?range=24h
GET /api/v1/dashboard/language-distribution?range=24h
GET /api/v1/dashboard/claims-over-time?range=24h&interval=hour
GET /api/v1/dashboard/recent-claims?limit=10
```

### Rumor Clusters

```http
GET /api/v1/rumor-clusters?risk=all&language=all&range=24h&page=1&page_size=20
GET /api/v1/rumor-clusters/{cluster_id}
```

### Evidence Sources

```http
POST /api/v1/sources/ingest
GET /api/v1/sources?search=&publisher=&language=&source_type=&page=1&page_size=20
GET /api/v1/sources/{source_id}
```

### Webhooks

```http
POST /api/v1/webhooks/inbound-message
```

---

## 11. Main API Contracts

### 11.1 Submit Text Claim

```http
POST /api/v1/claims/text
```

Request:

```json
{
  "text": "Ei news ta ki true? Govt free 50GB data dicche?",
  "user_language": "auto",
  "source_url": null,
  "client_source": "web"
}
```

MVP response should return completed result directly:

```json
{
  "claim_id": "uuid",
  "job_id": "verify_01HZABC123",
  "status": "completed",
  "extracted_claim": "The government is offering free 50GB mobile data.",
  "detected_language": "Banglish",
  "source_type": "text",
  "category": "Cybersecurity",
  "verdict": "Likely False",
  "confidence": 0.92,
  "confidence_label": "High",
  "explanation": "No verified government or telecom source confirms this offer. Similar messages are commonly used as phishing attempts.",
  "user_response": "JachAI Verdict: Likely False. No verified government source supports this free data offer.",
  "sources": [
    {
      "id": "source_uuid",
      "title": "Fact Check: No free WhatsApp data offer from Govt",
      "publisher": "BBC Bangla",
      "url": "https://example.com",
      "source_type": "fact_check",
      "snippet": "The viral free data message is baseless.",
      "similarity": 0.91,
      "credibility_score": 0.95,
      "published_at": "2026-06-01T10:00:00Z"
    }
  ],
  "created_at": "2026-06-05T12:00:00Z"
}
```

### 11.2 Submit Image Claim

```http
POST /api/v1/claims/image
```

Request:

```txt
multipart/form-data:
- file
- user_language
- client_source
- caption
```

Response: same as text claim, plus `ocr_text`.

### 11.3 Submit URL Claim

```http
POST /api/v1/claims/url
```

Request:

```json
{
  "url": "https://suspicious-site.com/free-data",
  "user_language": "auto",
  "client_source": "web"
}
```

MVP behavior:

- Fetch URL title/body if possible.
- If URL fetch fails, verify based on URL text/domain.
- Return same shape as text claim.

### 11.4 Ingest Evidence Source

```http
POST /api/v1/sources/ingest
```

Request:

```json
{
  "title": "Fact Check: Viral free internet message is fake",
  "publisher": "Prothom Alo",
  "url": "https://example.com/article",
  "language": "Bangla",
  "source_type": "fact_check",
  "published_at": "2026-06-01T10:00:00Z",
  "content": "Clean article body text..."
}
```

Backend behavior:

1. Validate input.
2. Compute content hash.
3. Check duplicate by URL/hash.
4. Generate embedding locally.
5. Store in `evidence_sources`.
6. Return source ID and embedding status.

Response:

```json
{
  "source_id": "uuid",
  "embedding_status": "embedded",
  "message": "Evidence source ingested successfully."
}
```

---

## 12. Core Claim Verification Pipeline

### Text Claim Pipeline

```txt
1. Validate text length.
2. Clean text.
3. Mask PII.
4. Detect language locally.
5. Generate normalized hash.
6. Check Redis duplicate cache.
7. If cached, return cached verdict.
8. Generate embedding locally.
9. Retrieve top 5 to 10 evidence records from pgvector.
10. Call NVIDIA LLM once with original text, cleaned text, detected language, and top evidence.
11. Parse strict JSON verdict.
12. Validate with Pydantic.
13. Save claim to NeonDB.
14. Save evidence links.
15. Update/create rumor cluster.
16. Cache verdict in Redis.
17. Return response to frontend.
```

### Image Claim Pipeline

```txt
1. Validate file type.
2. Read image.
3. Run Tesseract OCR.
4. If OCR text is too short, return OCR failure message.
5. Merge caption + OCR text.
6. Send merged text to text claim pipeline.
```

### URL Claim Pipeline

```txt
1. Validate URL.
2. Fetch page if possible.
3. Extract title and readable text with BeautifulSoup.
4. If extraction fails, use URL/domain text as input.
5. Send extracted text to text claim pipeline.
```

---

## 13. LLM Prompt Design

Because of the 40 RPM limit, use one LLM call.

### System Prompt

```txt
You are JachAI, a multilingual South Asian fact-checking assistant.

You verify claims in Bangla, English, Hindi, Banglish, and Hinglish.

Rules:
- Use only the provided evidence.
- Do not use unsupported assumptions.
- Do not invent sources.
- If evidence is weak, missing, or conflicting, return "Not Enough Evidence".
- Return strict JSON only.
- Keep the explanation short and understandable.
- The user response should match the user's language style when possible.
```

### User Prompt Template

```txt
Input text:
{cleaned_text}

Detected language:
{detected_language}

Evidence sources:
{evidence_json}

Return strict JSON with this schema:
{
  "extracted_claim": "clean factual claim",
  "detected_language": "Bangla | English | Hindi | Banglish | Hinglish | Mixed | Unknown",
  "category": "Politics | Health | Disaster | Cybersecurity | Finance | Crime | Entertainment | Other",
  "verdict": "Likely True | Likely False | Misleading | Not Enough Evidence",
  "confidence": 0.0,
  "confidence_label": "Low | Medium | High",
  "explanation": "short explanation using only evidence",
  "ai_analysis": "brief reasoning summary without hidden chain-of-thought",
  "user_response": "short user-facing verdict",
  "used_source_ids": ["source_id_1", "source_id_2"]
}
```

### Pydantic Verdict Schema

```python
from pydantic import BaseModel, Field
from typing import Literal

class VerdictLLMOutput(BaseModel):
    extracted_claim: str
    detected_language: Literal[
        "Bangla", "English", "Hindi", "Banglish", "Hinglish", "Mixed", "Unknown"
    ]
    category: Literal[
        "Politics", "Health", "Disaster", "Cybersecurity", "Finance",
        "Crime", "Entertainment", "Other"
    ]
    verdict: Literal["Likely True", "Likely False", "Misleading", "Not Enough Evidence"]
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_label: Literal["Low", "Medium", "High"]
    explanation: str
    ai_analysis: str
    user_response: str
    used_source_ids: list[str]
```

---

## 14. Service Responsibilities

### `text_cleaning_service.py`

```python
clean_text(text: str) -> str
normalize_unicode(text: str) -> str
remove_forwarded_markers(text: str) -> str
remove_extra_whitespace(text: str) -> str
```

### `pii_service.py`

```python
mask_pii(text: str) -> str
```

Mask phone numbers, email addresses, and obvious tracking IDs.

### `language_service.py`

```python
detect_language(text: str) -> str
```

Use simple script/rule-based detection.

### `ocr_service.py`

```python
extract_text_from_image(file_bytes: bytes) -> str
```

### `embedding_service.py`

```python
generate_embedding(text: str) -> list[float]
```

Load the model lazily and reuse a singleton model.

### `retrieval_service.py`

```python
search_similar_evidence(embedding: list[float], top_k: int = 5) -> list[EvidenceResult]
```

SQL:

```sql
SELECT id, title, url, publisher, language, source_type, published_at, content,
       credibility_score,
       1 - (embedding <=> :query_embedding) AS similarity
FROM evidence_sources
WHERE embedding IS NOT NULL
ORDER BY embedding <=> :query_embedding
LIMIT :top_k;
```

### `nvidia_llm_service.py`

```python
generate_verdict_with_nvidia(
    cleaned_text: str,
    detected_language: str,
    evidence: list[dict],
) -> VerdictLLMOutput
```

Use OpenAI-compatible client.

### `claim_pipeline.py`

```python
verify_text_claim(text: str, user_language: str | None, client_source: str) -> ClaimResponse
verify_image_claim(file, user_language: str | None, client_source: str, caption: str | None) -> ClaimResponse
verify_url_claim(url: str, user_language: str | None, client_source: str) -> ClaimResponse
```

### `cache_service.py`

```python
get_cached_claim(hash_value: str) -> dict | None
set_cached_claim(hash_value: str, result: dict, ttl: int) -> None
set_job_status(job_id: str, status: dict) -> None
get_job_status(job_id: str) -> dict | None
```

---

## 15. Dashboard API Logic

### Summary

```http
GET /api/v1/dashboard/summary
```

Return:

- total claims
- claims today
- active rumor clusters
- high-risk clusters
- average confidence
- OCR submissions
- top language
- cache hit rate if available

### Verdict Distribution

Group claims by verdict.

### Language Distribution

Group claims by detected language.

### Claims Over Time

Use `date_trunc('hour', created_at)` for hourly charts.

### Recent Claims

Return latest claims sorted by `created_at DESC`.

---

## 16. Health API Logic

```http
GET /api/v1/health/system
```

Check:

- backend
- NeonDB connection
- Redis connection
- embedding model load status
- NVIDIA API configured
- Tesseract availability

Return example:

```json
{
  "overall_status": "healthy",
  "services": [
    {
      "key": "core_backend",
      "label": "Core Backend",
      "status": "healthy",
      "latency_ms": 12
    },
    {
      "key": "primary_db",
      "label": "NeonDB",
      "status": "healthy"
    },
    {
      "key": "redis_cache",
      "label": "Redis Cache",
      "status": "healthy"
    }
  ]
}
```

---

## 17. Development Order for Codex

Build in this exact order:

1. Project skeleton: FastAPI app, config, DB, Redis, health route.
2. Database models and schemas: SQLAlchemy models, Pydantic schemas, init SQL.
3. Source ingestion: `/sources/ingest`, local embedding, duplicate handling.
4. Retrieval service: pgvector similarity search.
5. Text verification mock: `/claims/text`, mock verdict, DB save.
6. NVIDIA LLM service: OpenAI-compatible client, strict JSON parsing.
7. Real text pipeline: cleaning, PII, language, cache, embedding, retrieval, NVIDIA verdict, save result.
8. Image pipeline: upload, Tesseract OCR, call text pipeline.
9. URL pipeline: URL fetch, BeautifulSoup extraction, call text pipeline.
10. Dashboard APIs.
11. Health APIs.
12. Tests and docs.

---
