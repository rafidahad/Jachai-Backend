from __future__ import annotations
from time import perf_counter
from typing import Any
from uuid import UUID
from pydantic import HttpUrl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.services.cache_service import cache_service
from app.services.input_normalization_service import InputNormalizationService
from app.services.claim_extraction_service import ClaimExtractionService
from app.services.query_generation_service import QueryGenerationService
from app.services.tavily_search_service import TavilySearchService
from app.services.evidence_fetch_service import EvidenceFetchService
from app.services.evidence_cleaning_service import EvidenceCleaningService
from app.services.evidence_chunking_service import EvidenceChunkingService
from app.services.evidence_ranking_service import EvidenceRankingService
from app.services.source_credibility_service import SourceCredibilityService
from app.services.evidence_classification_service import EvidenceClassificationService
from app.services.verdict_service import VerdictService
from app.services.language_service import detect_language
from app.services.source_service import ingest_sources
from app.services.cluster_service import get_or_create_cluster
from app.models.claim import Claim, ClaimEvidenceLink
from app.schemas.source_schema import SourceIngestItemSchema
from app.schemas.claim_schema import (
    VerifyRequest,
    VerifyResponse,
    ClaimDetailsSchema,
    EvidenceDetailsSchema,
    SourceDetailsSchema,
    SearchQueryDetailsSchema
)
from app.utils.hashing import normalized_hash
from app.services.pii_service import mask_pii
from app.utils.errors import AppError

logger = get_logger(__name__)

class ClaimPipeline:
    @staticmethod
    async def verify_claim(session: AsyncSession, request: VerifyRequest) -> VerifyResponse:
        warnings = []
        debug_info = {}
        pipeline_started = perf_counter()

        # 1. Input Normalization
        started = perf_counter()
        cleaned_text, norm_warnings, norm_metadata = await InputNormalizationService.normalize(
            request.input_type, request.content
        )
        warnings.extend(norm_warnings)
        dur_normalization = perf_counter() - started
        logger.info("claim_pipeline normalization_finished duration_ms=%s", round(dur_normalization * 1000, 2))

        # Check Cache
        masked_for_hash = mask_pii(cleaned_text)
        content_hash = normalized_hash(masked_for_hash)
        hash_value = normalized_hash(f"v2-{settings.verification_pipeline_version}:{content_hash}")
        
        # Check if cached result is available in Redis
        cached_result = await cache_service.get_claim_result(f"v2:{hash_value}")
        if cached_result:
            logger.info("claim_pipeline cache_hit hash=%s", hash_value)
            try:
                # Return cached payload directly
                res_obj = VerifyResponse.model_validate(cached_result)
                if not res_obj.claim_id:
                    db_claim_id = await session.scalar(
                        select(Claim.id).where(Claim.normalized_hash == hash_value).limit(1)
                    )
                    if db_claim_id:
                        res_obj.claim_id = str(db_claim_id)
                return res_obj
            except Exception as exc:
                logger.warning("claim_pipeline invalid_cached_payload error=%s", str(exc))

        # Check existing claim in DB
        existing_claim_id = await session.scalar(
            select(Claim.id).where(Claim.normalized_hash == hash_value).limit(1)
        )
        if existing_claim_id:
            logger.info("claim_pipeline db_hit claim_id=%s", existing_claim_id)
            existing_claim = await session.get(Claim, existing_claim_id)
            if existing_claim and existing_claim.context_payload:
                try:
                    res_obj = VerifyResponse.model_validate(existing_claim.context_payload)
                    res_obj.claim_id = str(existing_claim_id)
                    return res_obj
                except Exception as exc:
                    logger.warning("claim_pipeline invalid_db_context_payload error=%s", str(exc))

        # 2. Claim Extraction
        started = perf_counter()
        extraction = await ClaimExtractionService.extract_claim(cleaned_text)
        dur_extraction = perf_counter() - started
        logger.info("claim_pipeline extraction_finished duration_ms=%s", round(dur_extraction * 1000, 2))

        # 3. Query Generation
        started = perf_counter()
        queries = await QueryGenerationService.generate_queries(extraction.normalized_claim, extraction.model_dump())
        dur_queries = perf_counter() - started
        logger.info("claim_pipeline query_gen_finished queries_count=%d duration_ms=%s", len(queries.queries), round(dur_queries * 1000, 2))

        # 4. Search via Tavily
        started = perf_counter()
        search_results, search_warnings = await TavilySearchService.search(
            queries.queries, max_results=request.options.max_search_results
        )
        warnings.extend(search_warnings)
        dur_search = perf_counter() - started
        logger.info("claim_pipeline search_finished results_count=%d duration_ms=%s", len(search_results), round(dur_search * 1000, 2))

        # 5. Fetch Candidate URLs
        started = perf_counter()
        fetched_evidence, fetch_warnings = await EvidenceFetchService.fetch_all(
            search_results, max_to_fetch=request.options.max_sources_to_fetch
        )
        warnings.extend(fetch_warnings)
        dur_fetch = perf_counter() - started
        logger.info("claim_pipeline fetching_finished duration_ms=%s", round(dur_fetch * 1000, 2))

        # 6. Evidence Cleaning
        started = perf_counter()
        cleaned_evidence, clean_warnings = EvidenceCleaningService.clean(
            fetched_evidence, requires_freshness=extraction.requires_freshness
        )
        warnings.extend(clean_warnings)
        dur_cleaning = perf_counter() - started
        logger.info("claim_pipeline cleaning_finished duration_ms=%s", round(dur_cleaning * 1000, 2))

        # DB Ingestion of Sources
        started = perf_counter()
        ingest_items = []
        for item in cleaned_evidence:
            try:
                pydantic_url = HttpUrl(item.url)
            except Exception:
                pydantic_url = item.url
            
            snippet = item.cleaned_text[:300]
            if len(snippet) < 10:
                snippet = (item.title or "Fallback Title") + " - snippet fallback context"
            
            body_text = item.cleaned_text
            if len(body_text) < 20:
                body_text = body_text + " - content fallback context metadata text"
                
            ingest_items.append(
                SourceIngestItemSchema(
                    title=item.title[:255] if item.title else "Untitled Source",
                    url=pydantic_url,
                    publisher=item.domain[:255],
                    language=detect_language(body_text),
                    source_type="tavily_search",
                    snippet=snippet,
                    text_content=body_text,
                    metadata={"domain": item.domain, "snippet_only": item.snippet_only}
                )
            )
        
        url_to_uuid = {}
        if ingest_items:
            try:
                db_sources, created, updated = await ingest_sources(session, ingest_items)
                url_to_uuid = {str(src.url): str(src.id) for src in db_sources}
            except Exception as exc:
                logger.warning("claim_pipeline db_ingest_sources_failed error=%s", str(exc))
                warnings.append("Failed to ingest evidence sources into DB. Proceeding with in-memory IDs.")
        dur_ingest = perf_counter() - started
        logger.info("claim_pipeline ingestion_finished duration_ms=%s", round(dur_ingest * 1000, 2))

        # Update cleaned_evidence source_id with actual DB source UUIDs
        for item in cleaned_evidence:
            item.source_id = url_to_uuid.get(item.url, item.source_id)

        # 7. Evidence Chunking
        started = perf_counter()
        chunks = EvidenceChunkingService.chunk(cleaned_evidence, max_chunks=request.options.max_evidence_chunks)
        dur_chunking = perf_counter() - started
        logger.info("claim_pipeline chunking_finished chunks_count=%d duration_ms=%s", len(chunks), round(dur_chunking * 1000, 2))

        # 8. Evidence Ranking / Reranking
        started = perf_counter()
        ranked_chunks, rank_warnings = await EvidenceRankingService.rank(
            extraction.normalized_claim, chunks, max_chunks=request.options.max_evidence_chunks
        )
        warnings.extend(rank_warnings)
        dur_ranking = perf_counter() - started
        logger.info("claim_pipeline ranking_finished duration_ms=%s", round(dur_ranking * 1000, 2))

        # 9. Source Credibility Scoring
        started = perf_counter()
        # Find unique sources among ranked chunks
        unique_sources = {}
        for rc in ranked_chunks:
            if rc.source_id not in unique_sources:
                unique_sources[rc.source_id] = (rc.title, rc.url, rc.domain, rc.snippet_only)
                
        credibility_scores = []
        for src_id, (title, url, domain, snippet_only) in unique_sources.items():
            credibility_scores.append(
                SourceCredibilityService.score_credibility(
                    src_id, title, url, domain, snippet_only
                )
            )
        dur_credibility = perf_counter() - started
        logger.info("claim_pipeline credibility_finished duration_ms=%s", round(dur_credibility * 1000, 2))

        # 10. Stance Classification
        started = perf_counter()
        classified_chunks = await EvidenceClassificationService.classify_chunks(
            extraction.normalized_claim, ranked_chunks
        )
        dur_stance = perf_counter() - started
        logger.info("claim_pipeline stance_classification_finished duration_ms=%s", round(dur_stance * 1000, 2))

        # 11. Final Verdict
        started = perf_counter()
        verdict_res = await VerdictService.generate_verdict(
            extraction.normalized_claim, extraction.model_dump(), classified_chunks, credibility_scores, warnings
        )
        dur_verdict = perf_counter() - started
        logger.info("claim_pipeline verdict_finished verdict=%s duration_ms=%s", verdict_res.verdict, round(dur_verdict * 1000, 2))

        # Compile final responses
        claim_schema = ClaimDetailsSchema(
            original_input=request.content,
            normalized_claim=extraction.normalized_claim,
            claim_type=extraction.claim_type,
            entities=extraction.entities,
            time_context=extraction.time_context,
            location_context=extraction.location_context,
            requires_freshness=extraction.requires_freshness
        )

        classified_lookup = {c.chunk_id: c for c in classified_chunks}
        evidence_schema_list = []
        for rc in ranked_chunks:
            stance_info = classified_lookup.get(rc.chunk_id)
            stance = stance_info.stance if stance_info else "neutral"
            
            score_info = next((cs for cs in credibility_scores if cs.source_id == rc.source_id), None)
            cred_score = score_info.credibility_score if score_info else 0.50
            
            evidence_schema_list.append(
                EvidenceDetailsSchema(
                    evidence_id=rc.source_id,
                    title=rc.title,
                    url=rc.url,
                    domain=rc.domain,
                    published_date=rc.published_date,
                    passage=rc.text,
                    stance=stance,
                    relevance_score=rc.relevance_score,
                    credibility_score=cred_score,
                    snippet_only=rc.snippet_only
                )
            )

        sources_schema_list = [
            SourceDetailsSchema(
                source_id=cs.source_id,
                title=cs.title,
                url=cs.url,
                domain=cs.domain,
                source_type=cs.source_type,
                credibility_score=cs.credibility_score,
                credibility_reason=cs.credibility_reason
            )
            for cs in credibility_scores
        ]

        queries_schema_list = [
            SearchQueryDetailsSchema(
                query=q["query"],
                purpose=q["purpose"]
            )
            for q in queries.queries
        ]

        pipeline_duration = perf_counter() - pipeline_started

        if settings.enable_debug_output:
            debug_info = {
                "durations": {
                    "input_normalization_ms": round(dur_normalization * 1000, 2),
                    "claim_extraction_ms": round(dur_extraction * 1000, 2),
                    "query_generation_ms": round(dur_queries * 1000, 2),
                    "tavily_search_ms": round(dur_search * 1000, 2),
                    "fetching_ms": round(dur_fetch * 1000, 2),
                    "cleaning_ms": round(dur_cleaning * 1000, 2),
                    "ingestion_ms": round(dur_ingest * 1000, 2),
                    "chunking_ms": round(dur_chunking * 1000, 2),
                    "ranking_ms": round(dur_ranking * 1000, 2),
                    "credibility_ms": round(dur_credibility * 1000, 2),
                    "stance_ms": round(dur_stance * 1000, 2),
                    "verdict_ms": round(dur_verdict * 1000, 2),
                    "total_pipeline_ms": round(pipeline_duration * 1000, 2)
                },
                "raw_extraction": extraction.model_dump(),
                "all_warnings": warnings
            }

        response = VerifyResponse(
            verdict=verdict_res.verdict,
            confidence=verdict_res.confidence,
            claim=claim_schema,
            explanation=verdict_res.explanation,
            evidence=evidence_schema_list,
            sources=sources_schema_list,
            search_queries=queries_schema_list,
            warnings=warnings,
            debug=debug_info if settings.enable_debug_output else None
        )

        # Save Claim record to Database
        try:
            cluster = await get_or_create_cluster(
                session,
                topic_hash=hash_value,
                title=extraction.normalized_claim[:120],
                language=detect_language(cleaned_text),
                summary=verdict_res.explanation
            )

            if existing_claim_id:
                claim_db = await session.get(Claim, existing_claim_id)
                if claim_db:
                    claim_db.verdict = verdict_res.verdict
                    claim_db.confidence = verdict_res.confidence
                    claim_db.explanation = verdict_res.explanation
                    claim_db.reasoning = verdict_res.explanation
                    claim_db.share_summary = verdict_res.explanation
                    
                    # Delete existing links to avoid duplicates
                    from sqlalchemy import delete
                    await session.execute(
                        delete(ClaimEvidenceLink).where(ClaimEvidenceLink.claim_id == claim_db.id)
                    )
            else:
                claim_db = Claim(
                    cluster_id=cluster.id,
                    input_type=request.input_type,
                    source_url=norm_metadata.get("url") if request.input_type == "url" else None,
                    raw_text=request.content,
                    cleaned_text=cleaned_text,
                    masked_text=mask_pii(cleaned_text),
                    normalized_hash=hash_value,
                    language=detect_language(cleaned_text),
                    verdict=verdict_res.verdict,
                    confidence=verdict_res.confidence,
                    explanation=verdict_res.explanation,
                    reasoning=verdict_res.explanation,
                    share_summary=verdict_res.explanation,
                    review_status="pending",
                    context_payload={}
                )
                session.add(claim_db)
            
            await session.flush()

            # Set response claim_id and update context_payload
            response.claim_id = str(claim_db.id)
            claim_db.context_payload = response.model_dump(mode="json")

            # Create ClaimEvidenceLink entries
            for pos, ev in enumerate(evidence_schema_list, start=1):
                # Only link if the ID is a valid DB UUID
                try:
                    uuid_id = UUID(ev.evidence_id)
                    session.add(
                        ClaimEvidenceLink(
                            claim_id=claim_db.id,
                            source_id=uuid_id,
                            rank=pos,
                            similarity_score=ev.relevance_score
                        )
                    )
                except ValueError:
                    # Not a UUID (temporary ID used as fallback)
                    continue
            await session.commit()
        except Exception as exc:
            logger.warning("claim_pipeline failed_to_persist_to_db error=%s", str(exc))
            await session.rollback()

        # Cache Result in Redis
        try:
            await cache_service.set_claim_result(f"v2:{hash_value}", response.model_dump(mode="json"))
            await cache_service.set_duplicate_claim_id(hash_value, hash_value) # register duplicate hash
        except Exception as exc:
            logger.warning("claim_pipeline failed_to_cache_redis error=%s", str(exc))

        return response


def _parse_ai_usage(context_payload: dict[str, Any]) -> Any | None:
    payload = context_payload.get("ai_usage")
    if not isinstance(payload, dict):
        return None
    try:
        from app.schemas.ai_schema import AIUsageSchema
        return AIUsageSchema.model_validate(payload, strict=False)
    except Exception:
        return None


def _parse_uuid(value: UUID | str, *, code: str, message: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except ValueError as exc:
        raise AppError(status_code=422, code=code, message=message) from exc


def _confidence_label_from_score(score: float) -> str:
    if score >= 0.8:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"


def serialize_claim(claim: Claim) -> ClaimResponseSchema:
    from app.schemas.claim_schema import ClaimResponseSchema
    from app.schemas.verdict_schema import EvidenceSnippetSchema

    context_payload = claim.context_payload or {}
    evidence_metadata = {}
    for item in context_payload.get("evidence", []):
        if isinstance(item, dict):
            sid = item.get("source_id") or item.get("evidence_id")
            if sid:
                evidence_metadata[str(sid)] = item

    evidence = [
        EvidenceSnippetSchema(
            source_id=link.source.id,
            title=link.source.title,
            url=link.source.url,
            publisher=link.source.publisher,
            language=link.source.language,
            source_type=link.source.source_type,
            snippet=link.source.snippet,
            similarity_score=float(
                evidence_metadata.get(str(link.source.id), {}).get("similarity_score")
                or evidence_metadata.get(str(link.source.id), {}).get("relevance_score")
                or link.similarity_score
            ),
            match_score=evidence_metadata.get(str(link.source.id), {}).get("match_score"),
            rerank_score=evidence_metadata.get(str(link.source.id), {}).get("rerank_score"),
            search_score=evidence_metadata.get(str(link.source.id), {}).get("search_score"),
            trust_score=evidence_metadata.get(str(link.source.id), {}).get("trust_score") or evidence_metadata.get(str(link.source.id), {}).get("credibility_score"),
            provider=evidence_metadata.get(str(link.source.id), {}).get("provider"),
            initial_rank=evidence_metadata.get(str(link.source.id), {}).get("initial_rank"),
            final_rank=evidence_metadata.get(str(link.source.id), {}).get("final_rank", link.rank),
        )
        for link in sorted(claim.evidence_links, key=lambda item: item.rank)
        if link.source is not None
    ]
    v2_claim = context_payload.get("claim") or {}
    return ClaimResponseSchema(
        id=claim.id,
        cluster_id=claim.cluster_id,
        input_type=claim.input_type,
        source_url=claim.source_url,
        raw_text=claim.raw_text,
        cleaned_text=claim.cleaned_text,
        masked_text=claim.masked_text,
        normalized_hash=claim.normalized_hash,
        language=claim.language,
        extracted_claim=str(
            context_payload.get("extracted_claim")
            or v2_claim.get("normalized_claim")
            or claim.cleaned_text
        ),
        detected_language=str(context_payload.get("detected_language") or claim.language),
        category=str(
            context_payload.get("category")
            or v2_claim.get("claim_type")
            or "Other"
        ),
        review_status=claim.review_status,
        verdict=claim.verdict,
        confidence=claim.confidence,
        confidence_label=str(context_payload.get("confidence_label") or _confidence_label_from_score(claim.confidence)),
        explanation=claim.explanation,
        user_response=str(
            context_payload.get("user_response")
            or context_payload.get("explanation")
            or claim.share_summary
        ),
        reasoning=claim.reasoning,
        share_summary=claim.share_summary,
        created_at=claim.created_at,
        updated_at=claim.updated_at,
        evidence=evidence,
        ai_usage=_parse_ai_usage(context_payload),
        context_payload=context_payload,
    )


async def _load_claim(session: AsyncSession, claim_id: UUID | str) -> Claim | None:
    from sqlalchemy.orm import selectinload
    statement = (
        select(Claim)
        .options(selectinload(Claim.evidence_links).selectinload(ClaimEvidenceLink.source))
        .where(Claim.id == _parse_uuid(claim_id, code="INVALID_CLAIM_ID", message="Claim ID must be a valid UUID."))
    )
    return await session.scalar(statement)


async def get_claim_by_id(session: AsyncSession, claim_id: UUID | str) -> ClaimResponseSchema | None:
    claim = await _load_claim(session, claim_id)
    return serialize_claim(claim) if claim else None


async def list_claims(
    session: AsyncSession,
    *,
    verdict: str | None = None,
    language: str | None = None,
    review_status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[ClaimResponseSchema], int]:
    from sqlalchemy import func
    from sqlalchemy.orm import selectinload

    filters = []
    if verdict:
        filters.append(Claim.verdict == verdict)
    if language:
        filters.append(Claim.language == language)
    if review_status:
        filters.append(Claim.review_status == review_status)

    total_statement = select(func.count()).select_from(Claim)
    if filters:
        total_statement = total_statement.where(*filters)
    total = await session.scalar(total_statement) or 0

    statement = (
        select(Claim)
        .options(selectinload(Claim.evidence_links).selectinload(ClaimEvidenceLink.source))
        .order_by(Claim.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if filters:
        statement = statement.where(*filters)
    claims = (await session.execute(statement)).scalars().all()
    return [serialize_claim(claim) for claim in claims], int(total)


async def update_review_status(
    session: AsyncSession,
    claim_id: UUID | str,
    review_status: str,
    reviewer: str,
) -> ClaimResponseSchema | None:
    from app.models.audit_log import AuditLog
    parsed_id = _parse_uuid(claim_id, code="INVALID_CLAIM_ID", message="Claim ID must be a valid UUID.")
    claim = await session.get(Claim, parsed_id)
    if claim is None:
        return None
    claim.review_status = review_status
    session.add(
        AuditLog(
            actor=reviewer,
            action="review_status_updated",
            entity_type="claim",
            entity_id=str(claim.id),
            details={"review_status": review_status},
        )
    )
    return serialize_claim(claim)
