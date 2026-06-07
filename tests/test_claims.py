from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import app.api.routes_claims

from app.schemas.claim_schema import VerifyRequest, VerifyResponse, VerifyRequestOptions
from app.services.input_normalization_service import InputNormalizationService
from app.services.claim_extraction_service import ClaimExtractionService
from app.services.query_generation_service import QueryGenerationService
from app.services.tavily_search_service import TavilySearchService, SearchResultV2
from app.services.evidence_fetch_service import EvidenceFetchService
from app.services.evidence_cleaning_service import EvidenceCleaningService
from app.services.evidence_chunking_service import EvidenceChunkingService
from app.services.evidence_ranking_service import EvidenceRankingService
from app.services.source_credibility_service import SourceCredibilityService
from app.services.evidence_classification_service import EvidenceClassificationService
from app.services.verdict_service import VerdictService
from app.services.claim_pipeline import ClaimPipeline
from app.utils.errors import AppError

# 1. Text claim request works / normalizes correctly
@pytest.mark.asyncio
async def test_text_normalization() -> None:
    text, warnings, metadata = await InputNormalizationService.normalize("text", "  A very noisy   text claim   ")
    assert text == "A very noisy text claim"
    assert not warnings
    assert not metadata

# 2. URL input normalization works with mocked fetch
@pytest.mark.asyncio
@patch("app.services.input_normalization_service.fetch_url_content")
async def test_url_normalization(mock_fetch: MagicMock) -> None:
    mock_fetch.return_value = {
        "title": "NASA Landed on Mars",
        "author": "John Doe",
        "published_date": "2026-01-01",
        "canonical_url": "https://nasa.gov/mars",
        "raw_text": "Detailed article content about Mars rover landing.",
        "error": None
    }
    cleaned_text, warnings, metadata = await InputNormalizationService.normalize("url", "https://nasa.gov/mars")
    assert cleaned_text == "Detailed article content about Mars rover landing."
    assert not warnings
    assert metadata["title"] == "NASA Landed on Mars"
    assert metadata["domain"] == "nasa.gov"

# 3. Image OCR input normalization works
@pytest.mark.asyncio
async def test_image_ocr_normalization() -> None:
    cleaned_text, warnings, metadata = await InputNormalizationService.normalize("image_ocr", "Some OCR text with artifacts |\\/")
    assert "Some OCR text with artifacts" in cleaned_text
    assert warnings
    assert metadata["cleaned"] is True

# 4. Claim extraction JSON parsing works
@pytest.mark.asyncio
@patch("app.services.claim_extraction_service.call_nvidia_chat")
async def test_claim_extraction_parsing(mock_call: AsyncMock) -> None:
    mock_call.return_value = """{
        "normalized_claim": "Extracted Mars Claim",
        "claim_type": "scientific",
        "entities": ["NASA", "Mars"],
        "time_context": "2026",
        "location_context": "Mars",
        "requires_freshness": true,
        "verification_strategy": "official_source_first",
        "detected_claims": [{"claim": "Extracted Mars Claim", "priority": 1}]
    }"""
    res = await ClaimExtractionService.extract_claim("Raw noisy input text")
    assert res.normalized_claim == "Extracted Mars Claim"
    assert res.claim_type == "scientific"
    assert res.requires_freshness is True

# 5. Claim extraction handles invalid JSON retry
@pytest.mark.asyncio
@patch("app.services.claim_extraction_service.call_nvidia_chat")
async def test_claim_extraction_retry(mock_call: AsyncMock) -> None:
    # First response is invalid JSON, second is valid JSON
    mock_call.side_effect = [
        "INVALID JSON CODE BLOCK",
        """{
            "normalized_claim": "Mars Retry Claim",
            "claim_type": "scientific",
            "entities": [],
            "requires_freshness": false,
            "verification_strategy": "general_web",
            "detected_claims": []
        }"""
    ]
    res = await ClaimExtractionService.extract_claim("Raw text")
    assert res.normalized_claim == "Mars Retry Claim"
    assert mock_call.call_count == 2

# 6. Query generation returns valid queries
@pytest.mark.asyncio
@patch("app.services.query_generation_service.call_nvidia_chat")
async def test_query_generation(mock_call: AsyncMock) -> None:
    mock_call.return_value = """{
        "queries": [
            {"query": "nasa mars landing 2026", "purpose": "official", "priority": 1},
            {"query": "mars landing fake news", "purpose": "refutation", "priority": 2}
        ]
    }"""
    res = await QueryGenerationService.generate_queries("Mars Claim", {"claim_type": "scientific"})
    assert len(res.queries) == 2
    assert res.queries[0]["purpose"] == "official"

# 7. Tavily search service handles empty results
@pytest.mark.asyncio
@patch("app.services.tavily_search_service.settings")
async def test_tavily_search_empty(mock_settings: MagicMock) -> None:
    mock_settings.active_tavily_api_key = "fake_key"
    mock_settings.tavily_enabled = True
    mock_settings.request_timeout_seconds = 15.0
    mock_settings.live_evidence_timeout_seconds = 10.0
    mock_settings.tavily_search_depth = "basic"
    mock_settings.tavily_topic = "news"
    mock_settings.tavily_max_results = 5
    mock_settings.tavily_include_answer = True
    mock_settings.tavily_include_raw_content = False
    mock_settings.tavily_include_images = False
    
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"query": "test", "results": []}
        mock_post.return_value = mock_response
        
        results, warnings = await TavilySearchService.search([{"query": "test"}])
        assert not results
        assert not warnings

# 8. Tavily search service handles API failure
@pytest.mark.asyncio
@patch("app.services.tavily_search_service.settings")
async def test_tavily_search_api_failure(mock_settings: MagicMock) -> None:
    mock_settings.active_tavily_api_key = "fake_key"
    mock_settings.tavily_enabled = True
    mock_settings.request_timeout_seconds = 15.0
    mock_settings.live_evidence_timeout_seconds = 10.0
    mock_settings.tavily_search_depth = "basic"
    mock_settings.tavily_topic = "news"
    mock_settings.tavily_max_results = 5
    mock_settings.tavily_include_answer = True
    mock_settings.tavily_include_raw_content = False
    mock_settings.tavily_include_images = False
    
    with patch("httpx.AsyncClient.post", side_effect=Exception("API limit exceeded")) as mock_post:
        results, warnings = await TavilySearchService.search([{"query": "test"}])
        assert not results
        assert warnings

# 9. Evidence fetch handles failed URL
@pytest.mark.asyncio
@patch("app.services.evidence_fetch_service.fetch_url_content")
async def test_evidence_fetch_failed_url(mock_fetch: AsyncMock) -> None:
    mock_fetch.return_value = {"error": "Connection Timeout"}
    candidate = SearchResultV2(
        title="Fail Title",
        url="https://fail.com",
        snippet="This is a test snippet.",
        score=0.9,
        domain="fail.com"
    )
    results, warnings = await EvidenceFetchService.fetch_all([candidate], max_to_fetch=1)
    assert len(results) == 1
    assert results[0].fetch_status == "snippet_fallback"
    assert results[0].snippet_only is True
    assert len(warnings) == 1

# 10. Evidence cleaning removes junk text
def test_evidence_cleaning() -> None:
    class DummyFetched:
        raw_text = "   Junk   Header \n\n Real story content that is very long and detailed to ensure it passes the length checks of one hundred characters without being marked as weak. \n\n Junk Footer   "
        published_date = None
        domain = "test.com"
        source_id = "src_1"
        title = "Title"
        url = "https://test.com"
        snippet_only = False

    cleaned, warnings = EvidenceCleaningService.clean([DummyFetched()], requires_freshness=False)
    assert len(cleaned) == 1
    assert cleaned[0].cleaned_text == "Junk Header Real story content that is very long and detailed to ensure it passes the length checks of one hundred characters without being marked as weak. Junk Footer"
    assert not cleaned[0].is_weak

# 11. Chunking preserves source metadata
def test_chunking_preserves_metadata() -> None:
    class DummyCleaned:
        source_id = "src_123"
        title = "Mars Land"
        url = "https://test.com/mars"
        domain = "test.com"
        published_date = "2026-01-01"
        cleaned_text = "Paragraph one of text." * 50
        snippet_only = False
        is_weak = False

    chunks = EvidenceChunkingService.chunk([DummyCleaned()], max_chunks=3)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.source_id == "src_123"
        assert chunk.title == "Mars Land"
        assert chunk.published_date == "2026-01-01"

# 12. Ranking returns top chunks
@pytest.mark.asyncio
@patch("app.services.evidence_ranking_service.settings")
async def test_ranking_returns_top_chunks(mock_settings: MagicMock) -> None:
    mock_settings.nvidia_api_key = None  # Force local ranking
    class DummyChunk:
        chunk_id = "c1"
        source_id = "s1"
        title = "Target Term Claim"
        url = "https://t1.com"
        domain = "t1.com"
        published_date = "2026"
        text = "This text matches target term claim."
        snippet_only = False

    chunks = [DummyChunk()]
    ranked, warnings = await EvidenceRankingService.rank("Target Term Claim", chunks, max_chunks=1)
    assert len(ranked) == 1
    assert ranked[0].relevance_score > 0.0

# 13. Reranker failure falls back to local ranking
@pytest.mark.asyncio
@patch("app.services.evidence_ranking_service.settings")
@patch("app.services.evidence_ranking_service.rerank_evidence", side_effect=Exception("Rerank NIIM Error"))
async def test_reranker_failure_fallback(mock_rerank: AsyncMock, mock_settings: MagicMock) -> None:
    mock_settings.nvidia_api_key = "fake_key"
    class DummyChunk:
        chunk_id = "c1"
        source_id = "s1"
        title = "Claim Text"
        url = "https://t1.com"
        domain = "t1.com"
        published_date = "2026"
        text = "Matches Claim Text."
        snippet_only = False

    chunks = [DummyChunk()]
    ranked, warnings = await EvidenceRankingService.rank("Claim Text", chunks, max_chunks=1)
    assert len(ranked) == 1
    assert any("Reranker failed" in w for w in warnings)

# 14. Source credibility scoring works for trusted/unknown/social domains
def test_source_credibility_scoring() -> None:
    res_gov = SourceCredibilityService.score_credibility("s1", "Gov Site", "https://site.gov", "site.gov", snippet_only=False)
    assert res_gov.source_type == "government"
    assert res_gov.credibility_score == 0.95

    res_social = SourceCredibilityService.score_credibility("s2", "FB", "https://facebook.com/post", "facebook.com", snippet_only=False)
    assert res_social.source_type == "social_media"
    assert res_social.credibility_score == 0.30

# 15. Stance classification handles support/refute/neutral
@pytest.mark.asyncio
@patch("app.services.evidence_classification_service.call_nvidia_chat")
async def test_stance_classification(mock_call: AsyncMock) -> None:
    mock_call.return_value = """{
        "chunk_id": "c1",
        "stance": "supports",
        "confidence": 0.95,
        "rationale": "It matches.",
        "quoted_evidence": "Verbatim quote."
    }"""
    class DummyChunk:
        chunk_id = "c1"
        title = "Mars Land"
        text = "Verification passage"

    res = await EvidenceClassificationService.classify_chunks("Mars", [DummyChunk()])
    assert len(res) == 1
    assert res[0].stance == "supports"

# 16. Final verdict returns insufficient_evidence when evidence is weak
@pytest.mark.asyncio
@patch("app.services.verdict_service.call_nvidia_chat")
async def test_verdict_insufficient_evidence(mock_call: AsyncMock) -> None:
    mock_call.return_value = """{
        "verdict": "insufficient_evidence",
        "confidence": 0.2,
        "explanation": "No matching sources.",
        "key_evidence_ids": [],
        "warnings": ["Low source count"]
    }"""
    res = await VerdictService.generate_verdict("Claim", {}, [], [], [])
    assert res.verdict == "insufficient_evidence"

# 17. Final verdict returns refuted when credible evidence contradicts claim
@pytest.mark.asyncio
@patch("app.services.verdict_service.call_nvidia_chat")
async def test_verdict_refuted(mock_call: AsyncMock) -> None:
    mock_call.return_value = """{
        "verdict": "refuted",
        "confidence": 0.9,
        "explanation": "Credible sources contradict claim.",
        "key_evidence_ids": ["c1"],
        "warnings": []
    }"""
    class DummyClassified:
        chunk_id = "c1"
        stance = "refutes"
        rationale = "Contradiction."
        quoted_evidence = "Fake news."

    class DummyScore:
        source_id = "c1"
        credibility_score = 0.95
        source_type = "reputable_news"

    res = await VerdictService.generate_verdict(
        "Claim", {}, [DummyClassified()], [DummyScore()], []
    )
    assert res.verdict == "refuted"


# 18. API response matches schema
@pytest.mark.asyncio
@patch("app.services.claim_pipeline.InputNormalizationService.normalize")
@patch("app.services.claim_pipeline.ClaimExtractionService.extract_claim")
@patch("app.services.claim_pipeline.QueryGenerationService.generate_queries")
@patch("app.services.claim_pipeline.TavilySearchService.search")
@patch("app.services.claim_pipeline.EvidenceFetchService.fetch_all")
@patch("app.services.claim_pipeline.EvidenceCleaningService.clean")
@patch("app.services.claim_pipeline.ingest_sources")
@patch("app.services.claim_pipeline.EvidenceRankingService.rank")
@patch("app.services.claim_pipeline.EvidenceClassificationService.classify_chunks")
@patch("app.services.claim_pipeline.VerdictService.generate_verdict")
@patch("app.services.claim_pipeline.get_or_create_cluster")
@patch("app.services.claim_pipeline.cache_service")
async def test_full_pipeline_success(
    mock_cache: MagicMock,
    mock_cluster: MagicMock,
    mock_verdict: AsyncMock,
    mock_classify: AsyncMock,
    mock_rank: AsyncMock,
    mock_ingest: AsyncMock,
    mock_clean: MagicMock,
    mock_fetch: AsyncMock,
    mock_search: AsyncMock,
    mock_queries: AsyncMock,
    mock_extract: AsyncMock,
    mock_normalize: AsyncMock
) -> None:
    mock_normalize.return_value = ("Cleaned content", [], {})
    mock_cache.get_claim_result = AsyncMock(return_value=None)
    mock_cache.set_claim_result = AsyncMock()
    mock_cache.set_duplicate_claim_id = AsyncMock()
    
    mock_extract.return_value = MagicMock(
        normalized_claim="Extracted Claim",
        claim_type="general",
        entities=[],
        time_context=None,
        location_context=None,
        requires_freshness=False,
        verification_strategy="general_web",
        model_dump=lambda: {}
    )
    mock_queries.return_value = MagicMock(queries=[{"query": "Extracted Claim", "purpose": "general"}])
    mock_search.return_value = ([], [])
    mock_fetch.return_value = ([], [])
    mock_clean.return_value = ([], [])
    mock_ingest.return_value = ([], 0, 0)
    mock_rank.return_value = ([], [])
    mock_classify.return_value = []
    
    mock_verdict.return_value = MagicMock(
        verdict="insufficient_evidence",
        confidence=0.1,
        explanation="No sources found.",
        key_evidence_ids=[],
        warnings=[]
    )
    
    mock_cluster.return_value = MagicMock(id=MagicMock())
    
    mock_session = AsyncMock()
    mock_session.scalar.return_value = None
    
    request = VerifyRequest(
        input_type="text",
        content="Raw test input content",
        options=VerifyRequestOptions(max_search_results=3)
    )
    
    response = await ClaimPipeline.verify_claim(mock_session, request)
    assert isinstance(response, VerifyResponse)
    assert response.verdict == "insufficient_evidence"
    assert response.claim.original_input == "Raw test input content"
    assert response.claim.normalized_claim == "Extracted Claim"


@pytest.mark.asyncio
@patch("app.services.ocr_service.extract_text_from_image_with_fallback")
@patch("app.api.routes_claims.ClaimPipeline.verify_claim")
async def test_verify_image_claim_endpoint(
    mock_verify: AsyncMock,
    mock_extract: AsyncMock,
) -> None:
    from fastapi import UploadFile
    from app.api.routes_claims import verify_image_claim

    mock_extract.return_value = ("OCR text extracted", "tesseract")
    mock_verify.return_value = MagicMock(spec=VerifyResponse)

    mock_file = MagicMock(spec=UploadFile)
    mock_file.content_type = "image/png"
    mock_file.filename = "test.png"
    mock_file.read = AsyncMock(return_value=b"fake image bytes")

    mock_session = AsyncMock()

    res = await verify_image_claim(image=mock_file, session=mock_session)
    assert res is not None
    mock_extract.assert_called_once_with(b"fake image bytes", mime_type="image/png")
    mock_verify.assert_called_once()
