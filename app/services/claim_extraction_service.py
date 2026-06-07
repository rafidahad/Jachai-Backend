from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logging import get_logger
from app.services.nvidia_client import call_nvidia_chat
from app.services.ai_model_router import NVIDIAModelTask
from app.utils.json_repair import load_json_with_repair

logger = get_logger(__name__)

CLAIM_EXTRACTION_SYSTEM_PROMPT = """
You are JachAI's claim extraction and classification agent.
Your task is to convert the raw, noisy normalized input text into a structured claim.

Rules:
- Extract the single clearest factual claim as `normalized_claim`.
- Classify the claim type into one of: event, statistic, quote, medical, political, scientific, financial, image_context, general.
- Identify named entities (people, organizations, locations, dates) mentioned.
- Extract `time_context` and `location_context` if specified (or null if not specified).
- Set `requires_freshness` to true if the claim is about a recent event, ongoing story, or news that requires up-to-date real-time search. Otherwise, false.
- Determine `verification_strategy`:
  - `official_source_first`: if the claim is about government actions, health regulations, laws.
  - `news_source_first`: if the claim is about recent general events, breaking news.
  - `academic_source_first`: if the claim is scientific, medical, or academic.
  - `general_web`: for general or unspecified claims.
- Return a list of `detected_claims` containing the claim text and a priority (starting at 1).

You must return valid JSON only. Do not include markdown code block formatting (like ```json). Use only the following schema:
{
  "normalized_claim": "string",
  "claim_type": "event | statistic | quote | medical | political | scientific | financial | image_context | general",
  "entities": ["string"],
  "time_context": "string or null",
  "location_context": "string or null",
  "requires_freshness": boolean,
  "verification_strategy": "official_source_first | news_source_first | academic_source_first | general_web",
  "detected_claims": [
    {
      "claim": "string",
      "priority": integer
    }
  ]
}
"""

class ClaimExtractionResult(BaseModel):
    normalized_claim: str
    claim_type: str
    entities: list[str] = Field(default_factory=list)
    time_context: str | None = None
    location_context: str | None = None
    requires_freshness: bool
    verification_strategy: str
    detected_claims: list[dict[str, Any]] = Field(default_factory=list)

class ClaimExtractionService:
    @staticmethod
    async def extract_claim(text: str) -> ClaimExtractionResult:
        user_prompt = f"Extract structured claim from this normalized text:\n\n{text}"
        
        max_attempts = settings.llm_json_retry_count + 1
        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = await call_nvidia_chat(
                    task=NVIDIAModelTask.CLAIM_EXTRACTION,
                    messages=[
                        {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                parsed = load_json_with_repair(raw_response)
                
                normalized_claim = str(parsed.get("normalized_claim", text[:200])).strip()
                claim_type = str(parsed.get("claim_type", "general")).strip()
                if claim_type not in ["event", "statistic", "quote", "medical", "political", "scientific", "financial", "image_context", "general"]:
                    claim_type = "general"
                
                entities = [str(e) for e in parsed.get("entities", [])]
                time_context = parsed.get("time_context")
                time_context = str(time_context) if time_context else None
                location_context = parsed.get("location_context")
                location_context = str(location_context) if location_context else None
                requires_freshness = bool(parsed.get("requires_freshness", False))
                verification_strategy = str(parsed.get("verification_strategy", "general_web"))
                if verification_strategy not in ["official_source_first", "news_source_first", "academic_source_first", "general_web"]:
                    verification_strategy = "general_web"
                detected_claims = parsed.get("detected_claims")
                if not isinstance(detected_claims, list) or not detected_claims:
                    detected_claims = [{"claim": normalized_claim, "priority": 1}]
                
                return ClaimExtractionResult(
                    normalized_claim=normalized_claim,
                    claim_type=claim_type,
                    entities=entities,
                    time_context=time_context,
                    location_context=location_context,
                    requires_freshness=requires_freshness,
                    verification_strategy=verification_strategy,
                    detected_claims=detected_claims
                )
            except Exception as exc:
                logger.warning("failed_claim_extraction attempt=%d error=%s", attempt, str(exc))
        
        logger.error("claim_extraction_failed_all_attempts falling_back_to_default_extraction")
        return ClaimExtractionResult(
            normalized_claim=text[:500],
            claim_type="general",
            entities=[],
            time_context=None,
            location_context=None,
            requires_freshness=False,
            verification_strategy="general_web",
            detected_claims=[{"claim": text[:500], "priority": 1}]
        )
