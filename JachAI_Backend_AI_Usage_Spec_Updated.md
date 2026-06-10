# JachAI Backend AI Usage Specification Updated

## 1. Purpose

This document defines the complete backend AI usage plan for **JachAI**.

JachAI is a trilingual, multimodal misinformation verification platform for South Asia. It verifies suspicious text claims, screenshots, URLs, and code-mixed messages across Bangla, English, Hindi, Banglish, and Hinglish.

The backend already has:

- NeonDB PostgreSQL database
- Online Redis
- NVIDIA API key
- NVIDIA endpoint with **40 RPM rate limit**

This document is updated for the selected AI stack:

```txt
LLM reasoning / verdict:   Llama Nemotron
Claim extraction:          Llama Nemotron
Embeddings:                BAAI/bge-m3 locally
Reranking:                 NVIDIA rerank-qa-mistral-4b
Vision fallback:           Kimi K2.6
OCR default:               Tesseract OCR
```

Because of the NVIDIA 40 RPM limit, the backend must avoid unnecessary external AI calls. The system should do local processing first, retrieve evidence locally, rerank only when needed, and call Llama Nemotron only once for final claim extraction and verdict generation.

---

## 2. Final AI Model Stack

### 2.1 LLM Reasoning / Verdict Model

```txt
Model: Llama Nemotron
Provider: NVIDIA Build / NVIDIA NIM
Purpose:
- Final fact-check reasoning
- Claim extraction from cleaned user text
- Evidence comparison
- Verdict generation
- Confidence estimation
- User-facing explanation
```

Use Llama Nemotron after local retrieval and optional reranking.

Do **not** use Llama Nemotron for every small task such as language detection, simple classification, text cleaning, or PII masking.

### 2.2 Claim Extraction Model

```txt
Model: Llama Nemotron
Provider: NVIDIA Build / NVIDIA NIM
Purpose:
- Extract the core claim
- Identify claim category
- Identify entities
- Produce strict JSON
```

Important optimization:

Even though claim extraction and verdict generation both use Llama Nemotron, they should be combined into **one LLM call** whenever possible.

Preferred:

```txt
One Llama Nemotron call:
  extracted_claim + category + verdict + confidence + user_response
```

Avoid:

```txt
Call 1: claim extraction
Call 2: verdict generation
```

### 2.3 Embedding Model

```txt
Model: BAAI/bge-m3
Provider: Local backend runtime
Purpose:
- Claim embedding
- Evidence source embedding
- Semantic search
- Similar rumor matching
- pgvector retrieval
```

Important:

`BAAI/bge-m3` outputs **1024-dimensional embeddings**.

Therefore, NeonDB pgvector column must be:

```sql
embedding vector(1024)
```

Do not use `vector(768)` unless the embedding model is changed.

### 2.4 Reranking Model

```txt
Model: NVIDIA rerank-qa-mistral-4b
Provider: NVIDIA Build / NVIDIA NIM
Purpose:
- Rerank top evidence from pgvector
- Improve relevance before final LLM reasoning
```

Reranking should be used like this:

```txt
pgvector retrieves top 20 sources
  ↓
NVIDIA rerank-qa-mistral-4b reranks them
  ↓
Top 5 sources go to Llama Nemotron
```

This improves result quality because pgvector finds semantically similar evidence, while the reranker chooses the evidence most relevant to the exact claim.

### 2.5 Vision Fallback Model

```txt
Model: Kimi K2.6
Provider: NVIDIA Build / NVIDIA NIM
Purpose:
- Fallback image understanding
- Extract claim/text from difficult screenshots
```

Use Kimi K2.6 only when Tesseract OCR fails or produces poor output.

Do not use Kimi K2.6 for every image because it consumes NVIDIA API quota.

### 2.6 Default OCR Engine

```txt
Engine: Tesseract OCR
Provider: Local backend runtime
Purpose:
- Extract text from screenshots
- Support Bangla, English, and Hindi OCR
```

Install:

```bash
sudo apt install -y tesseract-ocr tesseract-ocr-ben tesseract-ocr-hin
uv add pytesseract pillow
```

OCR language setting:

```env
OCR_LANGUAGES=ben+eng+hin
```

---

## 3. Updated AI Architecture

```txt
User text / screenshot / URL
  ↓
FastAPI backend
  ↓
Local preprocessing
  ├── text cleaning
  ├── PII masking
  ├── rule-based language detection
  ├── Tesseract OCR for screenshots
  ├── BAAI/bge-m3 local embedding
  ├── pgvector top 20 retrieval
  └── Redis duplicate cache
  ↓
Optional NVIDIA rerank-qa-mistral-4b
  └── top 20 evidence → top 5 evidence
  ↓
Llama Nemotron
  ├── extract claim
  ├── classify category
  ├── compare evidence
  ├── generate verdict
  └── generate user-facing answer
  ↓
Save result in NeonDB
  ↓
Cache result in Redis
  ↓
Return response to frontend
```

Core engineering rule:

> Use Tesseract first, local bge-m3 embeddings second, pgvector retrieval third, NVIDIA reranker fourth, and Llama Nemotron only for the final evidence-grounded answer.

---

## 4. NVIDIA 40 RPM Rate Limit Strategy

The NVIDIA endpoint has a **40 requests per minute** rate limit.

Because JachAI may use two NVIDIA calls per new claim:

```txt
1 call = rerank-qa-mistral-4b
1 call = Llama Nemotron verdict
```

A new claim may cost:

```txt
2 NVIDIA calls
```

Therefore:

```txt
40 RPM / 2 calls per claim = 20 new claims per minute theoretical max
```

To stay safe, enforce internal throughput:

```txt
15 new uncached claims per minute
30 NVIDIA calls per minute max
```

This keeps a 10 RPM buffer for retries, manual testing, and demo traffic.

---

## 5. Rate-Limit-Safe Pipeline

### 5.1 Always Check Redis Before NVIDIA

Before reranking or calling Llama Nemotron:

```txt
Clean claim
  ↓
Create SHA256 hash
  ↓
Check Redis cache
      ├── cache hit → return cached result
      └── cache miss → continue
```

Cache key:

```txt
claim:hash:{sha256(cleaned_text)}
```

TTL:

```txt
24 to 48 hours
```

Recommended:

```env
CLAIM_CACHE_TTL_SECONDS=172800
```

### 5.2 Reranker Usage Policy

Use reranker only if pgvector returns enough candidates.

Use reranker when:

```txt
pgvector returns >= 6 evidence sources
```

Skip reranker when:

```txt
pgvector returns 0 to 5 evidence sources
```

This saves NVIDIA calls.

Flow:

```txt
If retrieved evidence count > 5:
  rerank top 20 → top 5
Else:
  use retrieved evidence directly
```

### 5.3 Vision Fallback Usage Policy

Use Kimi K2.6 only when:

```txt
Tesseract OCR output length < 10 useful characters
OR OCR confidence is poor
OR OCR text is mostly symbols/noise
OR screenshot is critical for demo
```

Otherwise:

```txt
Tesseract OCR output → normal text pipeline
```

---

## 6. Full Text Claim Pipeline

```txt
POST /api/v1/claims/text
  ↓
Validate request
  ↓
Clean text
  ↓
Sanitize PII
  ↓
Rule-based language detection
  ↓
Create normalized SHA256 hash
  ↓
Check Redis cache
      ├── hit → return cached result
      └── miss → continue
  ↓
Generate BAAI/bge-m3 embedding locally
  ↓
Search NeonDB pgvector for top 20 evidence sources
  ↓
If evidence count > 5:
      call NVIDIA rerank-qa-mistral-4b
      select top 5 evidence
    Else:
      use available top evidence directly
  ↓
Build compact evidence context
  ↓
Call Llama Nemotron once:
      extract claim + classify + reason + verdict + response
  ↓
Validate JSON with Pydantic
  ↓
Apply backend guardrails
  ↓
Save claim, sources, verdict, and AI usage metadata in NeonDB
  ↓
Cache final result in Redis
  ↓
Return response to frontend
```

---

## 7. Full Image Claim Pipeline

```txt
POST /api/v1/claims/image
  ↓
Validate image type and size
  ↓
Run Tesseract OCR locally
  ↓
If OCR output is good:
      pass OCR text to text claim pipeline
  ↓
If OCR output is poor:
      optionally call Kimi K2.6 vision fallback
  ↓
Pass extracted claim/text to text claim pipeline
```

Image file types:

```txt
PNG
JPG
JPEG
WEBP
```

Recommended max file size:

```env
MAX_IMAGE_SIZE_MB=8
```

---

## 8. Full URL Claim Pipeline

```txt
POST /api/v1/claims/url
  ↓
Validate URL
  ↓
Fetch metadata or article body if allowed
  ↓
Clean extracted content
  ↓
Pass extracted text into text claim pipeline
```

For MVP, this can be simple:

```txt
URL text + page title + meta description → claim pipeline
```

Avoid building complex scraping into the synchronous verification path.

---

## 9. Updated Environment Variables

Create `.env.example`:

```env
# App
APP_NAME=JachAI
ENVIRONMENT=development
DEBUG=true
API_V1_PREFIX=/api/v1
INTERNAL_API_KEY=change_this_key

# NeonDB
DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@HOST/jachai_db?ssl=require

# Redis
REDIS_URL=rediss://USER:PASSWORD@HOST:PORT

# NVIDIA
NVIDIA_API_KEY=nvapi-your-key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

# Selected NVIDIA Models
NVIDIA_REASONING_MODEL=your-llama-nemotron-model-id
NVIDIA_CLAIM_EXTRACTION_MODEL=your-llama-nemotron-model-id
NVIDIA_RERANK_MODEL=nvidia/rerank-qa-mistral-4b
NVIDIA_VISION_MODEL=your-kimi-k2.6-model-id

# Rate Limit
NVIDIA_MAX_RPM=30
NVIDIA_MAX_UNCACHED_CLAIMS_PER_MINUTE=15
NVIDIA_TIMEOUT_SECONDS=60

# Local Embeddings
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
EMBEDDING_DEVICE=cpu

# Retrieval
PGVECTOR_TOP_K=20
FINAL_EVIDENCE_TOP_K=5
RERANK_MIN_CANDIDATES=6
MIN_RELEVANT_SIMILARITY=0.60

# OCR
OCR_ENGINE=tesseract
OCR_LANGUAGES=ben+eng+hin
MAX_IMAGE_SIZE_MB=8
ENABLE_VISION_FALLBACK=true

# Cache
CLAIM_CACHE_TTL_SECONDS=172800

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8501
```

---

## 10. Updated Backend AI File Structure

```txt
backend/app/services/
├── ai_model_router.py
├── nvidia_client.py
├── nvidia_rerank_service.py
├── nvidia_vision_service.py
├── claim_pipeline.py
├── embedding_service.py
├── retrieval_service.py
├── rerank_service.py
├── ocr_service.py
├── verdict_service.py
├── evidence_context_builder.py
├── cache_service.py
├── rate_limit_service.py
└── factcheck_service.py

backend/app/utils/
├── text_cleaning.py
├── pii.py
├── language.py
├── hashing.py
└── json_repair.py

backend/app/schemas/
├── ai_schema.py
├── claim_schema.py
├── source_schema.py
├── verdict_schema.py
├── rerank_schema.py
└── pipeline_schema.py
```

---

## 11. NeonDB Database Setup

Run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

### 11.1 Evidence Sources Table

```sql
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
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 11.2 Claims Table

```sql
CREATE TABLE IF NOT EXISTS claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    original_text TEXT,
    cleaned_text TEXT,
    extracted_claim TEXT NOT NULL,
    detected_language VARCHAR(50),
    source_type VARCHAR(50),
    category VARCHAR(100),
    verdict VARCHAR(50),
    confidence FLOAT,
    confidence_label VARCHAR(50),
    explanation TEXT,
    user_response TEXT,

    reasoning_model_used VARCHAR(255),
    rerank_model_used VARCHAR(255),
    vision_model_used VARCHAR(255),
    embedding_model_used VARCHAR(255),
    llm_call_count INTEGER DEFAULT 0,
    rerank_call_count INTEGER DEFAULT 0,
    vision_call_count INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 11.3 Claim Evidence Links Table

```sql
CREATE TABLE IF NOT EXISTS claim_evidence_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID REFERENCES claims(id) ON DELETE CASCADE,
    evidence_id UUID REFERENCES evidence_sources(id) ON DELETE CASCADE,
    similarity FLOAT,
    rerank_score FLOAT,
    initial_rank INTEGER,
    final_rank INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 11.4 Verification Jobs Table

```sql
CREATE TABLE IF NOT EXISTS verification_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID,
    status VARCHAR(50) DEFAULT 'processing',
    current_step VARCHAR(100),
    progress INTEGER DEFAULT 0,
    error_code VARCHAR(100),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 11.5 Rumor Clusters Table

```sql
CREATE TABLE IF NOT EXISTS rumor_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT,
    representative_claim TEXT,
    language VARCHAR(50),
    category VARCHAR(100),
    risk_level VARCHAR(50),
    submission_count INTEGER DEFAULT 1,
    viral_velocity FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 11.6 Indexes

```sql
CREATE INDEX IF NOT EXISTS idx_claims_language ON claims(detected_language);
CREATE INDEX IF NOT EXISTS idx_claims_verdict ON claims(verdict);
CREATE INDEX IF NOT EXISTS idx_claims_created_at ON claims(created_at);

CREATE INDEX IF NOT EXISTS idx_evidence_language ON evidence_sources(language);
CREATE INDEX IF NOT EXISTS idx_evidence_publisher ON evidence_sources(publisher);
CREATE INDEX IF NOT EXISTS idx_evidence_source_type ON evidence_sources(source_type);

CREATE INDEX IF NOT EXISTS evidence_embedding_hnsw_idx
ON evidence_sources
USING hnsw (embedding vector_cosine_ops);
```

---

## 12. Pydantic Schemas

### 12.1 Evidence Item

```python
from pydantic import BaseModel
from typing import Optional

class EvidenceItem(BaseModel):
    id: str
    title: str
    publisher: Optional[str] = None
    url: Optional[str] = None
    source_type: Optional[str] = None
    snippet: str
    similarity: float
    rerank_score: Optional[float] = None
    initial_rank: Optional[int] = None
    final_rank: Optional[int] = None
    published_at: Optional[str] = None
```

### 12.2 AI Verdict

```python
from pydantic import BaseModel, Field
from typing import Literal

VerdictLabel = Literal[
    "Likely True",
    "Likely False",
    "Misleading",
    "Not Enough Evidence"
]

ConfidenceLabel = Literal["Low", "Medium", "High"]

class AIVerdict(BaseModel):
    extracted_claim: str = Field(..., min_length=5)
    detected_language: str
    category: str
    verdict: VerdictLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_label: ConfidenceLabel
    explanation: str = Field(..., min_length=10)
    user_response: str = Field(..., min_length=10)
    used_source_ids: list[str] = []
```

### 12.3 Rerank Result

```python
from pydantic import BaseModel

class RerankResult(BaseModel):
    evidence_id: str
    rerank_score: float
    final_rank: int
```

---

## 13. NVIDIA Client

File:

```txt
backend/app/services/nvidia_client.py
```

Use OpenAI-compatible client:

```python
from openai import AsyncOpenAI
from app.core.config import settings

client = AsyncOpenAI(
    base_url=settings.NVIDIA_BASE_URL,
    api_key=settings.NVIDIA_API_KEY,
)
```

Chat completion:

```python
async def call_nvidia_chat(
    model: str,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 1200,
) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=settings.NVIDIA_TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content
```

Requirements:

- Respect internal 30 RPM
- Add one retry only
- Add timeout
- Do not log raw user text
- Log model name, latency, and call status

---

## 14. Rerank Service

File:

```txt
backend/app/services/nvidia_rerank_service.py
```

Purpose:

```txt
Input:
- cleaned user query
- top 20 pgvector evidence candidates

Output:
- top 5 evidence sources with rerank_score
```

Implementation behavior:

```txt
If candidate count <= 5:
  skip reranking

If candidate count >= 6:
  call NVIDIA rerank-qa-mistral-4b
```

Pseudo-flow:

```python
async def rerank_evidence(query: str, candidates: list[EvidenceItem]) -> list[EvidenceItem]:
    if len(candidates) <= 5:
        return candidates

    # call NVIDIA rerank model
    # sort by rerank score
    # return top 5
```

Important:

- Do not pass full articles
- Pass title + short snippet only
- Preserve source IDs
- Store rerank_score in claim_evidence_links

---

## 15. Vision Fallback Service

File:

```txt
backend/app/services/nvidia_vision_service.py
```

Use only when Tesseract fails.

Input:

```txt
image bytes or image URL
```

Output:

```json
{
  "extracted_text": "string",
  "visual_claim": "string",
  "confidence": 0.0
}
```

Rules:

- Only call when OCR output is poor
- Count this as a NVIDIA call
- Store `vision_model_used`
- Do not call if `ENABLE_VISION_FALLBACK=false`

---

## 16. Embedding Service

File:

```txt
backend/app/services/embedding_service.py
```

Use local BAAI/bge-m3:

```python
from sentence_transformers import SentenceTransformer

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-m3")
    return _model

def generate_embedding(text: str) -> list[float]:
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()
```

Rules:

- Load once
- Do not reload per request
- Normalize embeddings
- Confirm output dimension is 1024
- Use CPU for MVP unless GPU is available

---

## 17. pgvector Retrieval

File:

```txt
backend/app/services/retrieval_service.py
```

Use top 20 before reranking:

```sql
SELECT
    id,
    title,
    url,
    publisher,
    source_type,
    published_at,
    LEFT(content, 600) AS snippet,
    1 - (embedding <=> :query_embedding) AS similarity
FROM evidence_sources
WHERE embedding IS NOT NULL
ORDER BY embedding <=> :query_embedding
LIMIT 20;
```

Recommended thresholds:

```txt
similarity >= 0.70 = relevant
similarity 0.60 to 0.69 = weak but usable
similarity < 0.60 = weak
```

---

## 18. Evidence Context Builder

File:

```txt
backend/app/services/evidence_context_builder.py
```

Only pass top 5 reranked sources to Llama Nemotron.

Format:

```txt
SOURCE_ID: src_001
TITLE: Fact Check: No free WhatsApp data offer from Govt
PUBLISHER: BBC Bangla
TYPE: fact_check
SIMILARITY: 0.91
RERANK_SCORE: 0.96
URL: https://example.com/article
SNIPPET: The viral message claiming a government-backed WhatsApp data offer is baseless.
```

Rules:

- Top 5 only
- 300 to 500 character snippet
- Include similarity and rerank_score
- Include exact source IDs
- Do not invent missing metadata

---

## 19. Llama Nemotron Final Verdict Prompt

Use this after retrieval and reranking.

```txt
You are JachAI, a strict multilingual fact-checking assistant for South Asian misinformation.

The user input may contain Bangla, English, Hindi, Banglish, Hinglish, OCR noise, social media slang, or forwarded-message formatting.

Your job:
1. Extract the core factual claim.
2. Identify the language/style.
3. Classify the claim category.
4. Compare the claim only against the provided evidence.
5. Return a verdict.
6. Write a short user-facing explanation.

Important rules:
- Use ONLY the provided evidence.
- Do not use your internal knowledge as proof.
- If evidence is missing, weak, irrelevant, or conflicting, return "Not Enough Evidence".
- Do not invent sources.
- Do not overstate certainty.
- Keep the explanation short and clear.
- Return valid JSON only.
- Do not include markdown.
- Do not include extra text outside JSON.

Allowed verdicts:
- Likely True
- Likely False
- Misleading
- Not Enough Evidence

Allowed confidence labels:
- Low
- Medium
- High

Return JSON exactly in this schema:
{
  "extracted_claim": "string",
  "detected_language": "Bangla | English | Hindi | Banglish | Hinglish | Mixed | Unknown",
  "category": "Politics | Health | Disaster | Crime | Finance | Cybersecurity | Entertainment | Religion | Education | Other",
  "verdict": "Likely True | Likely False | Misleading | Not Enough Evidence",
  "confidence": 0.0,
  "confidence_label": "Low | Medium | High",
  "explanation": "string",
  "user_response": "string",
  "used_source_ids": ["string"]
}

User input:
{cleaned_text}

Detected language hint:
{language_hint}

Evidence:
{evidence_context}
```

---

## 20. Verdict Guardrails

Backend must apply these after Llama Nemotron returns a result.

```txt
If no evidence sources:
  verdict = Not Enough Evidence
  confidence <= 0.35
  confidence_label = Low

If all similarity scores < 0.60:
  verdict = Not Enough Evidence
  confidence <= 0.45

If no used_source_ids:
  confidence_label cannot be High

If verdict is Likely True / Likely False:
  at least one valid source must exist

If source ID in used_source_ids is not in retrieved evidence:
  remove it

Never allow:
  High confidence + zero sources

Never allow:
  invented URLs
```

---

## 21. Final API Response Shape

```json
{
  "claim_id": "uuid",
  "extracted_claim": "The government is offering free 50GB mobile data.",
  "detected_language": "Banglish",
  "source_type": "text",
  "category": "Cybersecurity",
  "verdict": "Likely False",
  "confidence": 0.91,
  "confidence_label": "High",
  "explanation": "No trusted source confirms this offer. Similar messages are reported as phishing attempts.",
  "user_response": "JachAI Verdict: Likely False. This free data offer is not supported by trusted sources and resembles known phishing messages.",
  "sources": [
    {
      "id": "src_001",
      "title": "Fact Check: No free WhatsApp data offer from Govt",
      "publisher": "BBC Bangla",
      "url": "https://example.com/article",
      "snippet": "The viral message claiming a government-backed WhatsApp data offer is baseless.",
      "similarity": 0.91,
      "rerank_score": 0.96
    }
  ],
  "ai_usage": {
    "embedding_model": "BAAI/bge-m3",
    "rerank_model": "nvidia/rerank-qa-mistral-4b",
    "reasoning_model": "Llama Nemotron",
    "vision_model": null,
    "llm_call_count": 1,
    "rerank_call_count": 1,
    "vision_call_count": 0
  },
  "created_at": "2026-06-05T12:00:00Z"
}
```

---

## 22. Error Handling

### 22.1 NVIDIA Rate Limit

```json
{
  "success": false,
  "error": {
    "code": "AI_RATE_LIMITED",
    "message": "Verification is busy right now. Please try again shortly."
  }
}
```

### 22.2 Reranker Failure

If reranker fails:

```txt
Continue with pgvector top 5
Do not fail the full pipeline
Log reranker failure
Set rerank_call_count accordingly
```

Response can still be generated.

### 22.3 OCR Failed

```json
{
  "success": false,
  "error": {
    "code": "OCR_FAILED",
    "message": "We could not read the screenshot clearly. Please paste the text manually."
  }
}
```

If vision fallback is enabled, try Kimi K2.6 once before returning OCR failure.

### 22.4 No Evidence

```json
{
  "verdict": "Not Enough Evidence",
  "confidence": 0.25,
  "confidence_label": "Low",
  "explanation": "JachAI could not find enough reliable evidence from trusted sources to verify this claim."
}
```

### 22.5 Invalid Llama Nemotron JSON

Process:

```txt
1. Try JSON extraction / repair once
2. Validate with Pydantic
3. If still invalid, return safe fallback
```

Fallback:

```json
{
  "verdict": "Not Enough Evidence",
  "confidence": 0.20,
  "confidence_label": "Low",
  "explanation": "The verification pipeline could not produce a reliable structured result."
}
```

---

## 23. Backend Build Order

Build in this exact order:

```txt
1. FastAPI skeleton
2. NeonDB connection
3. Redis connection
4. Text cleaning
5. PII sanitization
6. Rule-based language detection
7. Hashing and Redis cache
8. BAAI/bge-m3 embedding service
9. Source ingestion endpoint
10. pgvector retrieval top 20
11. Tesseract OCR service
12. NVIDIA client
13. NVIDIA rerank service
14. Evidence context builder
15. Llama Nemotron verdict service
16. Kimi K2.6 vision fallback service
17. Full claim pipeline
18. API routes
19. Dashboard APIs
20. Tests
```

---

## 24. Required API Endpoints

```txt
POST /api/v1/claims/text
POST /api/v1/claims/image
POST /api/v1/claims/url
GET  /api/v1/claims/{claim_id}

POST /api/v1/sources/ingest
GET  /api/v1/sources
GET  /api/v1/sources/{source_id}

GET  /api/v1/dashboard/summary
GET  /api/v1/dashboard/recent-claims
GET  /api/v1/dashboard/verdict-distribution
GET  /api/v1/dashboard/language-distribution
GET  /api/v1/dashboard/claims-over-time

GET  /api/v1/health/system
```

---

## 25. Tests

Create tests for:

```txt
Bangla text claim
Banglish text claim
English text claim
Hindi text claim
Screenshot OCR
Tesseract OCR failure
Kimi fallback condition
Repeated claim cache
pgvector retrieval top 20
reranker skip when <= 5 candidates
reranker fallback when NVIDIA rerank fails
Llama Nemotron invalid JSON
No evidence fallback
High confidence with zero sources blocked
NVIDIA rate limit fallback
Embedding dimension = 1024
```

---

## 26. Full Codex / CodeX Prompt

Use this prompt in Codex, CodeX, Antigravity, Cursor, or any agentic coding tool.

```txt
You are building the backend AI system for JachAI.

Project:
JachAI is a trilingual, multimodal misinformation verification platform for South Asia. It verifies suspicious text claims, screenshots, URLs, and code-mixed Bangla-English-Hindi messages.

Current infrastructure:
- NeonDB PostgreSQL is already created.
- Online Redis is already created.
- NVIDIA API key is available.
- NVIDIA endpoint has a 40 RPM limit.

Selected AI models:
- LLM reasoning / verdict: Llama Nemotron via NVIDIA
- Claim extraction: Llama Nemotron via NVIDIA
- Embeddings: BAAI/bge-m3 locally
- Reranking: NVIDIA rerank-qa-mistral-4b
- Vision fallback: Kimi K2.6 via NVIDIA
- OCR default: Tesseract OCR

Critical constraints:
- Use local processing before NVIDIA calls.
- Use Redis cache before any NVIDIA call.
- Use BAAI/bge-m3 locally for embeddings.
- Use pgvector top 20 retrieval.
- Use NVIDIA rerank-qa-mistral-4b only when evidence candidates > 5.
- Use Llama Nemotron only once per new claim for extraction + verdict + final explanation.
- Use Kimi K2.6 only when Tesseract OCR fails or output is poor.
- Internally throttle NVIDIA calls to max 30 RPM.
- Do not allow high-confidence verdicts with zero sources.
- Never invent source URLs.

Use FastAPI, Python, SQLAlchemy, asyncpg, Pydantic, Redis, sentence-transformers, pgvector, pytesseract, Pillow, and OpenAI-compatible NVIDIA client. Use uv as package manager.

Create this backend structure:

backend/
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── redis.py
│   │   └── logging.py
│   ├── api/
│   │   ├── routes_claims.py
│   │   ├── routes_sources.py
│   │   ├── routes_health.py
│   │   └── routes_dashboard.py
│   ├── models/
│   │   ├── claim.py
│   │   ├── source.py
│   │   ├── verification_job.py
│   │   └── cluster.py
│   ├── schemas/
│   │   ├── ai_schema.py
│   │   ├── claim_schema.py
│   │   ├── source_schema.py
│   │   ├── verdict_schema.py
│   │   └── rerank_schema.py
│   ├── services/
│   │   ├── ai_model_router.py
│   │   ├── nvidia_client.py
│   │   ├── nvidia_rerank_service.py
│   │   ├── nvidia_vision_service.py
│   │   ├── claim_pipeline.py
│   │   ├── embedding_service.py
│   │   ├── retrieval_service.py
│   │   ├── ocr_service.py
│   │   ├── verdict_service.py
│   │   ├── evidence_context_builder.py
│   │   └── cache_service.py
│   └── utils/
│       ├── text_cleaning.py
│       ├── pii.py
│       ├── language.py
│       ├── hashing.py
│       └── json_repair.py
├── tests/
├── pyproject.toml
├── Dockerfile
└── .env.example

Implement:
1. pydantic-settings config.
2. Async SQLAlchemy NeonDB connection.
3. Redis connection.
4. pgvector evidence_sources embedding vector(1024).
5. Source ingestion endpoint that generates bge-m3 embeddings.
6. Text cleaning, PII masking, language detection, hashing.
7. Redis cache lookup before NVIDIA calls.
8. Tesseract OCR with ben+eng+hin.
9. Optional Kimi K2.6 vision fallback when OCR fails.
10. pgvector retrieval top 20.
11. NVIDIA rerank-qa-mistral-4b reranking top 20 to top 5 when candidate count > 5.
12. Evidence context builder for top 5 sources.
13. Llama Nemotron verdict service with one JSON-only prompt.
14. Pydantic validation and JSON repair fallback.
15. Guardrails for no evidence, weak similarity, invalid source IDs, and high confidence with zero sources.
16. API endpoints:
    POST /api/v1/claims/text
    POST /api/v1/claims/image
    POST /api/v1/claims/url
    GET /api/v1/claims/{claim_id}
    POST /api/v1/sources/ingest
    GET /api/v1/health/system
17. Return frontend-ready JSON including ai_usage metadata.
18. Add tests for OCR, cache, retrieval, rerank skip/fallback, invalid JSON fallback, no evidence fallback, and embedding dimension.

Make the implementation modular, production-clean, typed, and easy for the frontend to consume.
```

---

## 27. Acceptance Checklist

```txt
[ ] NeonDB connects successfully
[ ] Redis connects successfully
[ ] pgvector extension enabled
[ ] evidence_sources table has vector(1024)
[ ] BAAI/bge-m3 embedding service works locally
[ ] Source ingestion creates 1024-dim embeddings
[ ] pgvector retrieves top 20 evidence sources
[ ] Reranker skips when candidates <= 5
[ ] Reranker works when candidates > 5
[ ] Reranker failure falls back to pgvector top 5
[ ] Llama Nemotron is called once per new claim
[ ] Tesseract OCR works
[ ] Kimi K2.6 fallback is optional and controlled by env
[ ] Redis cache prevents repeated NVIDIA calls
[ ] 30 RPM internal NVIDIA throttle exists
[ ] No-evidence fallback works
[ ] Invalid JSON fallback works
[ ] High confidence with zero sources is blocked
[ ] ai_usage metadata is returned
[ ] Frontend receives stable response format
[ ] Swagger/OpenAPI docs work
```

---

## 28. Final Engineering Rule

For JachAI, the backend AI system must follow this rule:

> Tesseract first.  
> bge-m3 locally.  
> pgvector top 20.  
> Rerank only when needed.  
> Llama Nemotron once.  
> Kimi only as fallback.  
> Cache before NVIDIA.  
> Never invent evidence.
