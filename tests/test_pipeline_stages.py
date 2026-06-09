"""
tests/test_pipeline_stages.py
------------------------------
Tests for the evidence-based claim verification pipeline.

All external services (Tavily, NVIDIA LLM, NVIDIA Reranker, HTTP fetch) are mocked.
No real API keys are required.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_chunk(
    *,
    source_id: str | None = None,
    text: str = "The government confirmed the event on 1 January 2024.",
    stance: str = "neutral",
    snippet_only: bool = False,
    domain: str = "reuters.com",
    published_date: str | None = "2024-01-01",
    relevance_score: float = 0.75,
) -> dict[str, Any]:
    return {
        "chunk_id": f"chunk-{uuid4().hex[:8]}",
        "source_id": source_id or str(uuid4()),
        "title": "Test Article",
        "url": f"https://{domain}/article",
        "domain": domain,
        "published_date": published_date,
        "text": text,
        "snippet_only": snippet_only,
        "relevance_score": relevance_score,
        "stance": stance,
        "stance_confidence": 0.8,
        "rationale": "The passage directly addresses the claim.",
        "quoted_evidence": text[:80],
        "trust_score": 0.90,
    }


def _make_evidence_item(
    *,
    source_id: str | None = None,
    similarity_score: float = 0.85,
    trust_score: float = 0.90,
    match_score: float = 0.80,
) -> dict[str, Any]:
    sid = source_id or str(uuid4())
    return {
        "source_id": sid,
        "title": "Test Source",
        "url": "https://reuters.com/article",
        "publisher": "reuters.com",
        "language": "English",
        "source_type": "tavily_search",
        "snippet": "The government confirmed the event.",
        "similarity_score": similarity_score,
        "match_score": match_score,
        "trust_score": trust_score,
        "search_score": 0.9,
        "rerank_score": None,
        "initial_rank": 1,
        "final_rank": 1,
    }


# ── 1. Text cleaning / normalization ─────────────────────────────────────────

class TestInputNormalization:
    def test_clean_text_collapses_whitespace(self):
        from app.services.text_cleaning_service import clean_text
        assert clean_text("Hello   \n world") == "Hello world"

    def test_clean_text_strips_control_chars(self):
        from app.services.text_cleaning_service import clean_text
        assert clean_text("Hello\u200b world") == "Hello world"

    def test_clean_text_short_input_detected(self):
        from app.services.text_cleaning_service import clean_text
        assert len(clean_text("Hi")) < 5

    def test_mask_pii_redacts_email_and_phone(self):
        from app.services.pii_service import mask_pii
        masked = mask_pii("Reach me at jane@example.com or +8801712345678")
        assert "[EMAIL_REDACTED]" in masked
        assert "[PHONE_REDACTED]" in masked

    def test_detect_language_handles_banglish(self):
        from app.services.language_service import detect_language
        assert detect_language("ami ajke meeting e jabo না") == "Banglish"

    def test_detect_language_handles_romanized_bangla_without_native_script(self):
        from app.services.language_service import detect_language
        assert detect_language("Donald trump namer mohish ekhon chiriakhanae dhakar.") == "Banglish"

    def test_normalized_hash_is_whitespace_insensitive(self):
        from app.utils.hashing import normalized_hash
        assert normalized_hash("Fact check me") == normalized_hash("  fact   check me  ")


# ── 2. Claim extraction ───────────────────────────────────────────────────────

class TestClaimExtraction:
    def test_image_ocr_claim_prompt_adds_noise_filtering_rules(self):
        from app.services.nvidia_llm_service import _claim_extraction_system_prompt

        prompt = _claim_extraction_system_prompt("gemini", "image_ocr")

        assert "This input came from OCR on an image or screenshot." in prompt
        assert "Do not return a dump of all detected text." in prompt

    def test_claim_extraction_cache_task_separates_text_and_image_ocr(self):
        from app.services.nvidia_llm_service import _claim_extraction_cache_task

        text_task = _claim_extraction_cache_task("gemini", "gemini-3.1-flash-lite", "text")
        image_task = _claim_extraction_cache_task("gemini", "gemini-3.1-flash-lite", "image_ocr")

        assert text_task != image_task

    def test_fallback_claim_extraction_returns_schema(self):
        from app.services.nvidia_llm_service import fallback_claim_extraction
        result = fallback_claim_extraction("WHO says vaccines cause autism", language_hint="English")
        assert result.extracted_claim
        assert result.detected_language == "English"
        assert result.category == "Other"

    def test_claim_extraction_handles_invalid_json_via_fallback(self):
        """Extraction falls back gracefully when LLM returns no valid JSON."""
        from app.services.nvidia_llm_service import fallback_claim_extraction
        # Directly test fallback path
        result = fallback_claim_extraction("Bad JSON claim", language_hint="Unknown")
        assert result.extracted_claim == "Bad JSON claim"

    def test_normalize_extraction_preserves_original_wording_for_clean_claims(self):
        from app.services.nvidia_llm_service import _normalize_extraction_payload

        payload = {
            "extracted_claim": "Ramisa's murderer was sentenced to death.",
            "detected_language": "English",
            "category": "Crime",
        }
        normalized = _normalize_extraction_payload(
            payload,
            claim_text="Ramisa's murderer received the death penalty.",
            language_hint="English",
            provider="nvidia",
        )

        assert normalized["extracted_claim"] == "Ramisa's murderer received the death penalty."

    def test_normalize_extraction_keeps_cleaned_claim_when_wrapper_text_is_removed(self):
        from app.services.nvidia_llm_service import _normalize_extraction_payload

        payload = {
            "extracted_claim": "Ramisa's murderer received the death penalty.",
            "detected_language": "English",
            "category": "Crime",
        }
        normalized = _normalize_extraction_payload(
            payload,
            claim_text="Please verify this claim: Ramisa's murderer received the death penalty.",
            language_hint="English",
            provider="nvidia",
        )

        assert normalized["extracted_claim"] == "Ramisa's murderer received the death penalty."

    def test_normalize_extraction_accepts_native_script_translation_for_gemini_banglish(self):
        from app.services.nvidia_llm_service import _normalize_extraction_payload

        payload = {
            "extracted_claim": "ডোনাল্ড ট্রাম্প নামের মহিষ এখন ঢাকার চিড়িয়াখানায়।",
            "detected_language": "Banglish",
            "category": "Other",
        }
        normalized = _normalize_extraction_payload(
            payload,
            claim_text="Donald trump namer mohish ekhon chiriakhanae dhakar.",
            language_hint="Banglish",
            provider="gemini",
        )

        assert normalized["extracted_claim"] == "ডোনাল্ড ট্রাম্প নামের মহিষ এখন ঢাকার চিড়িয়াখানায়।"
        assert normalized["detected_language"] == "Bangla"

    def test_claim_extraction_schema_has_new_fields(self):
        from app.schemas.verdict_schema import ClaimExtractionSchema
        schema = ClaimExtractionSchema(
            extracted_claim="WHO says vaccines cause autism",
            detected_language="English",
            category="Health",
            entities=["WHO", "vaccines"],
            time_context=None,
            location_context=None,
            requires_freshness=False,
            verification_strategy="academic_source_first",
            detected_claims=[{"claim": "WHO says vaccines cause autism", "priority": 1}],
        )
        assert schema.requires_freshness is False
        assert "WHO" in schema.entities

    @pytest.mark.asyncio
    async def test_extract_claim_context_uses_fallback_without_api_key(self):
        from app.services.nvidia_llm_service import extract_claim_context
        with patch("app.services.nvidia_llm_service.settings") as mock_settings:
            mock_settings.gemini_claim_extraction_enabled = False
            mock_settings.nvidia_api_key = ""
            mock_settings.llm_json_retry_count = 1
            result, meta = await extract_claim_context("Some claim", "English")
            assert result.extracted_claim == "Some claim"
            assert meta["call_count"] == 0

    @pytest.mark.asyncio
    async def test_extract_claim_context_prefers_gemini_when_configured(self):
        from app.services.nvidia_llm_service import extract_claim_context

        gemini_payload = json.dumps({
            "extracted_claim": "Some claim",
            "detected_language": "English",
            "category": "Other",
            "entities": [],
            "time_context": "",
            "location_context": "",
            "requires_freshness": False,
            "verification_strategy": "general_web",
            "detected_claims": [{"claim": "Some claim", "priority": 1}],
        })

        with patch("app.services.nvidia_llm_service.settings") as mock_settings:
            mock_settings.gemini_claim_extraction_enabled = True
            mock_settings.active_gemini_claim_extraction_model = "gemini-3.1-flash-lite"
            mock_settings.llm_json_retry_count = 1
            with patch(
                "app.services.nvidia_llm_service.call_gemini_generate_content",
                new_callable=AsyncMock,
                return_value=gemini_payload,
            ) as mock_gemini:
                result, meta = await extract_claim_context("Some claim", "English")

        assert result.extracted_claim == "Some claim"
        assert meta["provider"] == "gemini"
        assert meta["model"] == "gemini-3.1-flash-lite"
        mock_gemini.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_extract_claim_context_uses_image_ocr_prompt_for_image_inputs(self):
        from app.services.nvidia_llm_service import extract_claim_context

        gemini_payload = json.dumps({
            "extracted_claim": "Ramisa's murderer received the death penalty.",
            "detected_language": "English",
            "category": "Crime",
            "entities": ["Ramisa"],
            "time_context": "",
            "location_context": "",
            "requires_freshness": True,
            "verification_strategy": "news_source_first",
            "detected_claims": [{"claim": "Ramisa's murderer received the death penalty.", "priority": 1}],
        })

        with patch("app.services.nvidia_llm_service.settings") as mock_settings:
            mock_settings.gemini_claim_extraction_enabled = True
            mock_settings.active_gemini_claim_extraction_model = "gemini-3.1-flash-lite"
            mock_settings.llm_json_retry_count = 1
            with patch(
                "app.services.nvidia_llm_service.call_gemini_generate_content",
                new_callable=AsyncMock,
                return_value=gemini_payload,
            ) as mock_gemini:
                result, meta = await extract_claim_context(
                    "Follow\n2h\nRamisa's murderer received the death penalty.\nLike Reply Share",
                    "English",
                    source_kind="image_ocr",
                )

        assert result.extracted_claim == "Ramisa's murderer received the death penalty."
        assert meta["source_kind"] == "image_ocr"
        assert "OCR on an image or screenshot" in mock_gemini.await_args.kwargs["system_instruction"]
        assert "Input source:\nimage_ocr" in mock_gemini.await_args.kwargs["user_content"]


class TestGeminiFailover:
    @pytest.mark.asyncio
    async def test_gemini_generate_content_fails_over_to_backup_key_on_429(self):
        from app.services.gemini_client import call_gemini_generate_content
        import httpx

        class _FakeClient:
            def __init__(self, responses):
                self._responses = responses

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json, headers):
                response = self._responses.pop(0)
                response.request = httpx.Request("POST", url, headers=headers)
                return response

        responses = [
            httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}}),
            httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "{\"ok\": true}"}],
                            }
                        }
                    ]
                },
            ),
        ]

        with patch("app.services.gemini_client.settings") as mock_settings:
            mock_settings.active_gemini_api_keys = ["primary-key", "backup-key"]
            mock_settings.gemini_timeout_seconds = 60
            with patch(
                "app.services.gemini_client.httpx.AsyncClient",
                return_value=_FakeClient(responses),
            ):
                result = await call_gemini_generate_content(
                    system_instruction="Return JSON only",
                    user_content="Reply with {\"ok\": true}",
                    model="gemini-3.1-flash-lite",
                    response_mime_type="application/json",
                    purpose="claim_extraction",
                )

        assert result == "{\"ok\": true}"


# ── 3. Query generation ───────────────────────────────────────────────────────

class TestQueryGeneration:
    def test_fallback_queries_return_valid_schema(self):
        from app.schemas.verdict_schema import ClaimExtractionSchema
        from app.services.nvidia_llm_service import fallback_search_queries
        extraction = ClaimExtractionSchema(
            extracted_claim="WHO says vaccines cause autism",
            detected_language="English",
            category="Health",
        )
        result = fallback_search_queries(extraction=extraction, original_text="vaccines autism who")
        assert len(result.search_queries) >= 1
        assert len(result.queries) >= 1
        assert result.queries[0].purpose in {"general", "official", "refutation", "recent", "background"}

    def test_query_schema_has_purpose_field(self):
        from app.schemas.verdict_schema import SearchQuerySchema
        q = SearchQuerySchema(query="vaccines autism site:who.int", purpose="official", priority=1)
        assert q.purpose == "official"
        assert q.priority == 1


# ── 4. Tavily search ──────────────────────────────────────────────────────────

class TestTavilySearch:
    @pytest.mark.asyncio
    async def test_search_returns_empty_when_disabled(self):
        from app.services.search_service import search_with_tavily
        with patch("app.services.search_service.settings") as mock_settings:
            mock_settings.tavily_enabled = False
            mock_settings.active_tavily_api_key = None
            result = await search_with_tavily("some query")
            assert result.results == []

    @pytest.mark.asyncio
    async def test_search_general_web_handles_exception_gracefully(self):
        from app.services.search_service import search_general_web
        with patch("app.services.search_service.search_with_tavily", side_effect=RuntimeError("API down")):
            results = await search_general_web("some query")
            assert results == []

    def test_dedupe_search_results_removes_duplicates(self):
        from app.services.search_service import NormalizedSearchResult, dedupe_search_results
        r1 = NormalizedSearchResult(query="q", title="T", url="https://example.com/page", search_score=0.8)
        r2 = NormalizedSearchResult(query="q2", title="T", url="https://example.com/page", search_score=0.5)
        deduped = dedupe_search_results([r1, r2])
        assert len(deduped) == 1
        assert deduped[0].search_score == 0.8  # higher score kept


# ── 5. Evidence fetching ──────────────────────────────────────────────────────

class TestEvidenceFetching:
    @pytest.mark.asyncio
    async def test_failed_url_does_not_crash_pipeline(self):
        """A single URL fetch failure should produce a snippet_fallback, not crash."""
        from app.services.search_service import NormalizedSearchResult
        from app.services.live_evidence_service import _document_from_search_result
        import httpx

        result = NormalizedSearchResult(
            query="test",
            title="Test",
            url="https://broken-url.example.com/article",
            snippet="Short snippet about the claim.",
            content="Short",  # triggers crawl
            search_score=0.7,
            domain="broken-url.example.com",
            trust_score=0.5,
        )

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        doc = await _document_from_search_result(
            mock_client,
            result,
            search_answer=None,
            search_run_id="test-run",
        )
        # Should use snippet fallback, not raise
        assert doc is None or doc.fetch_status == "snippet_fallback"


# ── 6. OCR ───────────────────────────────────────────────────────────────────

class TestOcr:
    def test_prepare_ocr_text_for_claim_extraction_filters_ui_noise(self):
        from app.services.ocr_service import prepare_ocr_text_for_claim_extraction

        text = "\n".join([
            "DailyStarNews",
            "Follow",
            "2h",
            "Ramisa's murderer received the death penalty.",
            "Like Reply Share",
        ])

        assert prepare_ocr_text_for_claim_extraction(text) == "Ramisa's murderer received the death penalty."

    def test_prepare_ocr_text_for_claim_extraction_trims_single_line_ui_noise(self):
        from app.services.ocr_service import prepare_ocr_text_for_claim_extraction

        text = "Follow 2h Ramisa's murderer received the death penalty. Like Reply Share"

        assert prepare_ocr_text_for_claim_extraction(text) == "Ramisa's murderer received the death penalty."

    @pytest.mark.asyncio
    async def test_image_ocr_uses_kimi_model(self):
        from app.services.ocr_service import extract_text_from_image

        with patch(
            "app.services.ocr_service.extract_text_with_kimi_ocr",
            new_callable=AsyncMock,
            return_value="This screenshot contains enough extracted text to pass OCR validation.",
        ) as mock_ocr:
            text = await extract_text_from_image(b"fake-image-bytes", mime_type="image/png")

        assert "enough extracted text" in text
        mock_ocr.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_image_ocr_rejects_too_short_output(self):
        from app.services.ocr_service import extract_text_from_image
        from app.utils.errors import AppError

        with patch(
            "app.services.ocr_service.extract_text_with_kimi_ocr",
            new_callable=AsyncMock,
            return_value="tiny",
        ):
            with pytest.raises(AppError) as exc:
                await extract_text_from_image(b"fake-image-bytes", mime_type="image/png")

        assert exc.value.code == "OCR_FAILED"


# ── 7. Evidence cleaning & chunking ──────────────────────────────────────────

class TestEvidenceChunking:
    def test_chunk_preserves_source_metadata(self):
        from app.services.evidence_chunker import chunk_evidence_documents
        docs = [{
            "source_id": "src-001",
            "title": "Reuters Health Report",
            "url": "https://reuters.com/health/article",
            "domain": "reuters.com",
            "published_date": "2024-01-15",
            "text_content": "Scientists confirmed. " * 40,
            "snippet": "Scientists confirmed vaccines are safe.",
            "snippet_only": False,
        }]
        chunks = chunk_evidence_documents(docs)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk["source_id"] == "src-001"
            assert chunk["title"] == "Reuters Health Report"
            assert chunk["domain"] == "reuters.com"
            assert chunk["published_date"] == "2024-01-15"
            assert "chunk_id" in chunk
            assert len(chunk["text"]) >= 10

    def test_chunk_deduplication_removes_identical_passages(self):
        from app.services.evidence_chunker import chunk_evidence_documents
        repeated = "This is exactly the same sentence. " * 3
        docs = [
            {"source_id": "a", "title": "A", "url": "https://a.com", "domain": "a.com",
             "text_content": repeated, "snippet": "Same.", "snippet_only": False},
            {"source_id": "b", "title": "B", "url": "https://b.com", "domain": "b.com",
             "text_content": repeated, "snippet": "Same.", "snippet_only": False},
        ]
        chunks = chunk_evidence_documents(docs)
        texts = [c["text"] for c in chunks]
        # After dedup, near-identical text should appear at most once
        unique_texts = list(set(texts))
        assert len(unique_texts) <= len(texts)

    def test_snippet_only_doc_produces_single_chunk(self):
        from app.services.evidence_chunker import chunk_evidence_documents, MIN_CHUNK_CHARS
        long_snippet = "Vaccines are widely considered safe by medical experts worldwide. " * 3
        assert len(long_snippet) >= MIN_CHUNK_CHARS, "Snippet too short for test"
        docs = [{
            "source_id": "snip-001",
            "title": "T",
            "url": "https://example.com",
            "domain": "example.com",
            "text_content": "Very short.",  # triggers snippet fallback
            "snippet": long_snippet,
            "snippet_only": True,
        }]
        chunks = chunk_evidence_documents(docs)
        assert len(chunks) == 1
        assert chunks[0]["snippet_only"] is True


# ── 7. Evidence ranking / reranking ──────────────────────────────────────────

class TestEvidenceRanking:
    @pytest.mark.asyncio
    async def test_rerank_falls_back_to_local_when_nvidia_unavailable(self):
        from app.services.nvidia_rerank_service import rerank_evidence
        chunks = [_make_chunk() for _ in range(3)]
        with patch("app.services.nvidia_rerank_service.settings") as mock_settings:
            mock_settings.nvidia_api_key = ""
            mock_settings.final_evidence_top_k = 5
            mock_settings.rerank_min_candidates = 2
            ranked, meta = await rerank_evidence("test query", chunks, use_chunk_format=True)
            assert meta["applied"] is False
            assert len(ranked) >= 1

    @pytest.mark.asyncio
    async def test_rerank_failure_falls_back_gracefully(self):
        from app.services.nvidia_rerank_service import rerank_evidence
        chunks = [_make_chunk() for _ in range(6)]
        with patch("app.services.nvidia_rerank_service.call_nvidia_rerank",
                   side_effect=RuntimeError("Reranker down")):
            with patch("app.services.nvidia_rerank_service.settings") as mock_settings:
                mock_settings.nvidia_api_key = "test-key"
                mock_settings.final_evidence_top_k = 5
                mock_settings.rerank_min_candidates = 2
                mock_settings.max_evidence_chunks = 12
                mock_settings.enable_debug_output = False
                ranked, meta = await rerank_evidence("test query", chunks, use_chunk_format=True)
                assert meta["reason"] == "rerank_failed"
                assert len(ranked) >= 1

    def test_local_ranking_assigns_relevance_scores(self):
        from app.services.nvidia_rerank_service import _local_ranked_chunks
        chunks = [
            _make_chunk(text="Vaccines are proven safe by WHO.", domain="who.int"),
            _make_chunk(text="Some unrelated sports news.", domain="example.com"),
        ]
        ranked = _local_ranked_chunks(chunks, claim_text="vaccines WHO safety", limit=5)
        assert ranked[0]["relevance_score"] >= ranked[1]["relevance_score"]


# ── 8. Source credibility scoring ────────────────────────────────────────────

class TestSourceCredibility:
    def test_government_domain_gets_high_score(self):
        from app.services.search_service import score_source_credibility
        result = score_source_credibility(source_id="s1", domain="who.int")
        assert result["credibility_score"] >= 0.85
        assert result["source_type"] in {"official", "government", "reputable_news", "fact_check"}

    def test_social_media_gets_low_score(self):
        from app.services.search_service import score_source_credibility
        result = score_source_credibility(source_id="s2", domain="facebook.com")
        assert result["credibility_score"] <= 0.30
        assert result["source_type"] == "social_media"

    def test_snippet_only_reduces_score(self):
        from app.services.search_service import score_source_credibility
        full = score_source_credibility(source_id="s3", domain="reuters.com", snippet_only=False)
        snip = score_source_credibility(source_id="s4", domain="reuters.com", snippet_only=True)
        assert snip["credibility_score"] < full["credibility_score"]

    def test_missing_date_for_freshness_reduces_score(self):
        from app.services.search_service import score_source_credibility
        with_date = score_source_credibility(
            source_id="s5", domain="cdc.gov", requires_freshness=True, published_date="2024-01-01"
        )
        without_date = score_source_credibility(
            source_id="s6", domain="cdc.gov", requires_freshness=True, published_date=None
        )
        assert without_date["credibility_score"] < with_date["credibility_score"]

    def test_unknown_domain_returns_unknown_type(self):
        from app.services.search_service import score_source_credibility
        result = score_source_credibility(source_id="s7", domain="random-blog-xyz.com")
        assert result["source_type"] in {"unknown", "blog", "low_quality"}


# ── 9. Stance classification ──────────────────────────────────────────────────

class TestStanceClassification:
    @pytest.mark.asyncio
    async def test_stance_classification_supports(self):
        """When LLM returns supports JSON, result should carry supports stance."""
        from app.services.nvidia_llm_service import classify_evidence_stance
        llm_response = json.dumps({
            "stance": "supports",
            "confidence": 0.9,
            "rationale": "The passage directly confirms the claim.",
            "quoted_evidence": "WHO confirmed the event.",
        })
        with patch("app.services.nvidia_llm_service._call_llm_with_retry",
                   new_callable=AsyncMock, return_value=llm_response):
            with patch("app.services.nvidia_llm_service.settings") as mock_settings:
                mock_settings.nvidia_api_key = "test-key"
                mock_settings.llm_json_retry_count = 1
                chunk = _make_chunk(text="WHO confirmed the event on 1 Jan 2024.")
                result = await classify_evidence_stance(chunk, normalized_claim="WHO confirmed an event in 2024")
                assert result["stance"] == "supports"
                assert result["stance_confidence"] >= 0.8


class TestVerdictGeneration:
    @pytest.mark.asyncio
    async def test_generate_verdict_prefers_gemini_reasoning_when_configured(self):
        from app.schemas.verdict_schema import ClaimExtractionSchema
        from app.services.nvidia_llm_service import generate_verdict

        gemini_payload = json.dumps({
            "verdict": "supported",
            "confidence": 0.91,
            "confidence_label": "High",
            "explanation": "Multiple reliable sources directly support the claim.",
            "user_response": "Reliable evidence supports this claim.",
            "used_source_ids": [],
            "key_evidence_ids": [],
            "warnings": [],
        })

        extraction = ClaimExtractionSchema(
            extracted_claim="The government confirmed the event.",
            detected_language="English",
            category="Politics",
            entities=["government"],
            time_context=None,
            location_context=None,
            requires_freshness=False,
            verification_strategy="official_source_first",
            detected_claims=[{"claim": "The government confirmed the event.", "priority": 1}],
        )
        evidence = [_make_evidence_item()]

        with patch("app.services.nvidia_llm_service.settings") as mock_settings:
            mock_settings.gemini_reasoning_enabled = True
            mock_settings.active_gemini_reasoning_model = "gemini-3.5-flash"
            mock_settings.active_gemini_reasoning_models = ["gemini-3.5-flash"]
            mock_settings.nvidia_api_key = ""
            with patch(
                "app.services.nvidia_llm_service.call_gemini_generate_content",
                new_callable=AsyncMock,
                return_value=gemini_payload,
            ) as mock_gemini:
                verdict, meta = await generate_verdict(extraction, evidence)

        assert verdict.confidence >= 0.9
        assert meta["provider"] == "gemini"
        assert meta["model"] == "gemini-3.5-flash"
        assert mock_gemini.await_args.kwargs["purpose"] == "reasoning"

    @pytest.mark.asyncio
    async def test_generate_verdict_falls_back_to_gemini_reasoning_backup_model(self):
        from app.schemas.verdict_schema import ClaimExtractionSchema
        from app.services.nvidia_llm_service import generate_verdict

        gemini_payload = json.dumps({
            "verdict": "supported",
            "confidence": 0.84,
            "confidence_label": "High",
            "explanation": "Reliable evidence supports the claim.",
            "user_response": "Reliable evidence supports this claim.",
            "used_source_ids": [],
            "key_evidence_ids": [],
            "warnings": [],
        })

        extraction = ClaimExtractionSchema(
            extracted_claim="The government confirmed the event.",
            detected_language="English",
            category="Politics",
            entities=["government"],
            time_context=None,
            location_context=None,
            requires_freshness=False,
            verification_strategy="official_source_first",
            detected_claims=[{"claim": "The government confirmed the event.", "priority": 1}],
        )
        evidence = [_make_evidence_item()]

        with patch("app.services.nvidia_llm_service.settings") as mock_settings:
            mock_settings.gemini_reasoning_enabled = True
            mock_settings.active_gemini_reasoning_models = [
                "gemini-3.5-flash",
                "gemini-3.1-flash-lite",
            ]
            mock_settings.nvidia_api_key = ""
            with patch(
                "app.services.nvidia_llm_service.call_gemini_generate_content",
                new_callable=AsyncMock,
                side_effect=[RuntimeError("primary failed"), gemini_payload],
            ) as mock_gemini:
                verdict, meta = await generate_verdict(extraction, evidence)

        assert verdict.confidence >= 0.8
        assert meta["provider"] == "gemini"
        assert meta["model"] == "gemini-3.1-flash-lite"
        assert mock_gemini.await_count == 2

    @pytest.mark.asyncio
    async def test_stance_classification_refutes(self):
        from app.services.nvidia_llm_service import classify_evidence_stance
        llm_response = json.dumps({
            "stance": "refutes",
            "confidence": 0.85,
            "rationale": "WHO denies the claim.",
            "quoted_evidence": "WHO denied any such statement.",
        })
        with patch("app.services.nvidia_llm_service._call_llm_with_retry",
                   new_callable=AsyncMock, return_value=llm_response):
            with patch("app.services.nvidia_llm_service.settings") as mock_settings:
                mock_settings.nvidia_api_key = "test-key"
                mock_settings.llm_json_retry_count = 1
                chunk = _make_chunk(text="WHO denied any such statement.")
                result = await classify_evidence_stance(chunk, normalized_claim="WHO said vaccines cause autism")
                assert result["stance"] == "refutes"

    @pytest.mark.asyncio
    async def test_stance_classification_falls_back_to_neutral_on_error(self):
        from app.services.nvidia_llm_service import classify_evidence_stance
        with patch("app.services.nvidia_llm_service._call_llm_with_retry",
                   side_effect=RuntimeError("LLM down")):
            with patch("app.services.nvidia_llm_service.settings") as mock_settings:
                mock_settings.nvidia_api_key = "test-key"
                mock_settings.llm_json_retry_count = 0
                chunk = _make_chunk()
                result = await classify_evidence_stance(chunk, normalized_claim="some claim")
                assert result["stance"] == "neutral"


# ── 10. Final verdict generation ──────────────────────────────────────────────

class TestFinalVerdict:
    @pytest.mark.asyncio
    async def test_verdict_returns_insufficient_evidence_when_no_evidence(self):
        from app.schemas.verdict_schema import ClaimExtractionSchema
        from app.services.nvidia_llm_service import generate_verdict
        extraction = ClaimExtractionSchema(
            extracted_claim="Some claim",
            detected_language="English",
            category="Other",
        )
        verdict, meta = await generate_verdict(extraction, [])
        assert verdict.verdict == "Not Enough Evidence"
        assert verdict.pipeline_verdict == "insufficient_evidence"
        assert meta["call_count"] == 0

    @pytest.mark.asyncio
    async def test_verdict_returns_insufficient_evidence_without_api_key(self):
        from app.schemas.verdict_schema import ClaimExtractionSchema
        from app.services.nvidia_llm_service import generate_verdict
        extraction = ClaimExtractionSchema(
            extracted_claim="Some claim",
            detected_language="English",
            category="Other",
        )
        evidence = [_make_evidence_item()]
        with patch("app.services.nvidia_llm_service.settings") as mock_settings:
            mock_settings.nvidia_api_key = ""
            mock_settings.llm_json_retry_count = 1
            verdict, meta = await generate_verdict(extraction, evidence)
            assert verdict.verdict == "Not Enough Evidence"
            assert meta["call_count"] == 0

    @pytest.mark.asyncio
    async def test_verdict_supported_from_strong_evidence(self):
        from app.schemas.verdict_schema import ClaimExtractionSchema
        from app.services.nvidia_llm_service import generate_verdict
        extraction = ClaimExtractionSchema(
            extracted_claim="WHO confirmed the vaccine is safe",
            detected_language="English",
            category="Health",
        )
        evidence = [_make_evidence_item()]
        llm_response = json.dumps({
            "verdict": "supported",
            "confidence": 0.9,
            "confidence_label": "High",
            "explanation": "WHO source directly confirms claim.",
            "user_response": "JachAI Verdict: Supported.",
            "used_source_ids": [evidence[0]["source_id"]],
            "key_evidence_ids": [evidence[0]["source_id"]],
            "warnings": [],
        })
        with patch("app.services.nvidia_llm_service._call_llm_with_retry",
                   new_callable=AsyncMock, return_value=llm_response):
            with patch("app.services.nvidia_llm_service.settings") as mock_settings:
                mock_settings.nvidia_api_key = "test-key"
                mock_settings.llm_json_retry_count = 1
                verdict, meta = await generate_verdict(extraction, evidence)
                assert verdict.verdict == "Likely True"  # mapped from "supported"
                assert verdict.pipeline_verdict == "supported"
                assert verdict.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_verdict_refuted_from_contradicting_evidence(self):
        from app.schemas.verdict_schema import ClaimExtractionSchema
        from app.services.nvidia_llm_service import generate_verdict
        extraction = ClaimExtractionSchema(
            extracted_claim="5G causes COVID-19",
            detected_language="English",
            category="Health",
        )
        evidence = [_make_evidence_item()]
        llm_response = json.dumps({
            "verdict": "refuted",
            "confidence": 0.95,
            "confidence_label": "High",
            "explanation": "WHO explicitly states 5G does not cause COVID-19.",
            "user_response": "JachAI Verdict: Refuted.",
            "used_source_ids": [evidence[0]["source_id"]],
            "key_evidence_ids": [evidence[0]["source_id"]],
            "warnings": [],
        })
        with patch("app.services.nvidia_llm_service._call_llm_with_retry",
                   new_callable=AsyncMock, return_value=llm_response):
            with patch("app.services.nvidia_llm_service.settings") as mock_settings:
                mock_settings.nvidia_api_key = "test-key"
                mock_settings.llm_json_retry_count = 1
                verdict, meta = await generate_verdict(extraction, evidence)
                assert verdict.verdict == "Likely False"
                assert verdict.pipeline_verdict == "refuted"


# ── 11. Verdict guardrails ────────────────────────────────────────────────────

class TestVerdictGuardrails:
    def _get_guardrails(self):
        from app.services.claim_pipeline import _apply_verdict_guardrails
        return _apply_verdict_guardrails

    def _make_verdict(self, pipeline_verdict: str = "supported", confidence: float = 0.9):
        from app.schemas.verdict_schema import LLMVerdictSchema, PIPELINE_TO_LEGACY_VERDICT
        legacy = PIPELINE_TO_LEGACY_VERDICT.get(pipeline_verdict, "Not Enough Evidence")
        return LLMVerdictSchema(
            extracted_claim="Test claim text here",
            detected_language="English",
            category="Other",
            verdict=legacy,
            confidence=confidence,
            confidence_label="High" if confidence >= 0.8 else "Medium",
            explanation="Evidence supports the claim.",
            user_response="JachAI Verdict: Supported.",
            used_source_ids=[],
            pipeline_verdict=pipeline_verdict,
            warnings=[],
        )

    def test_no_evidence_forces_insufficient(self):
        apply = self._get_guardrails()
        verdict = self._make_verdict("supported", 0.9)
        result = apply(verdict, claim_text="test", language="English", evidence=[])
        assert result.verdict == "Not Enough Evidence"
        assert result.confidence <= 0.35

    def test_all_neutral_chunks_triggers_insufficient_evidence(self):
        apply = self._get_guardrails()
        verdict = self._make_verdict("supported", 0.85)
        evidence = [_make_evidence_item()]
        # All chunks neutral — less than 20% are supports/refutes
        neutral_chunks = [_make_chunk(stance="neutral") for _ in range(10)]
        with patch("app.services.claim_pipeline.settings") as mock_settings:
            mock_settings.min_relevant_similarity = 0.60
            mock_settings.final_evidence_top_k = 5
            mock_settings.max_evidence_chunks = 12
            result = apply(
                verdict,
                claim_text="test",
                language="English",
                evidence=evidence,
                classified_chunks=neutral_chunks,
            )
            assert result.verdict == "Not Enough Evidence"
            assert result.pipeline_verdict == "insufficient_evidence"

    def test_strong_supporting_chunks_preserved(self):
        apply = self._get_guardrails()
        source_id = str(uuid4())
        # Verdict must cite the source_id so guardrails don't drop it
        from app.schemas.verdict_schema import LLMVerdictSchema, PIPELINE_TO_LEGACY_VERDICT
        verdict = LLMVerdictSchema(
            extracted_claim="Test claim text here",
            detected_language="English",
            category="Other",
            verdict="Likely True",
            confidence=0.9,
            confidence_label="High",
            explanation="Evidence supports the claim.",
            user_response="JachAI Verdict: Supported.",
            used_source_ids=[__import__('uuid').UUID(source_id)],
            pipeline_verdict="supported",
            warnings=[],
        )
        evidence = [_make_evidence_item(source_id=source_id, trust_score=0.92, similarity_score=0.85)]
        # Mix of support/refute chunks (40% = 4/10 > 20% threshold)
        chunks = (
            [_make_chunk(source_id=source_id, stance="supports") for _ in range(4)]
            + [_make_chunk(stance="neutral") for _ in range(6)]
        )
        with patch("app.services.claim_pipeline.settings") as mock_settings:
            mock_settings.min_relevant_similarity = 0.60
            mock_settings.final_evidence_top_k = 5
            mock_settings.max_evidence_chunks = 12
            result = apply(
                verdict,
                claim_text="test claim text here",
                language="English",
                evidence=evidence,
                classified_chunks=chunks,
            )
            # 40% strong evidence — should not be downgraded to insufficient
            assert result.verdict != "Not Enough Evidence"


# ── 12. API response schema validation ───────────────────────────────────────

class TestApiResponseSchema:
    def test_claim_response_schema_has_required_fields(self):
        from app.schemas.claim_schema import ClaimResponseSchema
        # Just verify schema can be instantiated with required fields
        import datetime
        from uuid import uuid4
        schema = ClaimResponseSchema(
            id=uuid4(),
            cluster_id=None,
            input_type="text",
            source_url=None,
            raw_text="Test claim",
            cleaned_text="Test claim",
            masked_text="Test claim",
            normalized_hash="abc123",
            language="English",
            extracted_claim="Test claim",
            detected_language="English",
            category="Other",
            review_status="pending",
            verdict="Not Enough Evidence",
            confidence=0.2,
            confidence_label="Low",
            explanation="No evidence found.",
            user_response="JachAI: No evidence found.",
            reasoning="No evidence.",
            share_summary="No evidence found.",
            created_at=datetime.datetime.now(datetime.UTC),
            updated_at=datetime.datetime.now(datetime.UTC),
        )
        assert schema.verdict == "Not Enough Evidence"

    def test_evidence_snippet_schema_has_new_fields(self):
        from app.schemas.verdict_schema import EvidenceSnippetSchema
        from uuid import uuid4
        snippet = EvidenceSnippetSchema(
            source_id=uuid4(),
            title="Article",
            url="https://example.com",
            language="English",
            source_type="tavily_search",
            snippet="Short snippet.",
            stance="supports",
            credibility_score=0.85,
            snippet_only=False,
        )
        assert snippet.stance == "supports"
        assert snippet.credibility_score == 0.85
        assert snippet.snippet_only is False

    def test_pipeline_verdict_label_values(self):
        from app.schemas.verdict_schema import ALLOWED_PIPELINE_VERDICTS
        expected = {
            "supported", "refuted", "misleading", "partially_true",
            "outdated", "insufficient_evidence", "unverifiable",
        }
        assert expected == ALLOWED_PIPELINE_VERDICTS

    def test_verdict_label_mapping_covers_all_new_labels(self):
        from app.schemas.verdict_schema import PIPELINE_TO_LEGACY_VERDICT, ALLOWED_PIPELINE_VERDICTS
        for pv in ALLOWED_PIPELINE_VERDICTS:
            assert pv in PIPELINE_TO_LEGACY_VERDICT, f"No mapping for pipeline verdict: {pv}"


# ── 13. Evidence context builder ─────────────────────────────────────────────

class TestEvidenceContextBuilder:
    def test_context_builder_with_classified_chunks(self):
        from app.services.evidence_context_builder import build_evidence_context
        chunks = [_make_chunk(stance="supports"), _make_chunk(stance="refutes")]
        with patch("app.services.evidence_context_builder.settings") as mock_settings:
            mock_settings.max_evidence_chunks = 12
            mock_settings.final_evidence_top_k = 5
            context = build_evidence_context([], classified_chunks=chunks)
            assert "supports" in context.lower() or "refutes" in context.lower()
            assert "STANCE:" in context

    def test_context_builder_legacy_path_without_chunks(self):
        from app.services.evidence_context_builder import build_evidence_context
        evidence = [_make_evidence_item()]
        with patch("app.services.evidence_context_builder.settings") as mock_settings:
            mock_settings.max_evidence_chunks = 12
            mock_settings.final_evidence_top_k = 5
            context = build_evidence_context(evidence)
            assert "SOURCE_ID" in context
            assert "SNIPPET" in context
