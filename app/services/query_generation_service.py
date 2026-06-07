from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logging import get_logger
from app.services.nvidia_client import call_nvidia_chat
from app.services.ai_model_router import NVIDIAModelTask
from app.utils.json_repair import load_json_with_repair

logger = get_logger(__name__)

QUERY_GENERATION_SYSTEM_PROMPT = """
You are JachAI's query generation agent.
Your task is to generate 3 to 5 search queries to verify a claim.

Rules:
- Generate target queries for specific purposes:
  - `general`: General lookup of the claim.
  - `official`: Queries targeting official agencies, fact-checkers, or government domains.
  - `refutation`: Queries designed to find refutations or exposes of the claim (e.g. including "hoax", "fake", "fact check").
  - `recent`: News or date-restricted queries if freshness is required.
  - `background`: Contextual query about the entities or event.
- Keep queries short and precise. Avoid search operator syntax unless standard double quotes.
- Do not output markdown code blocks. Return JSON matching:
{
  "queries": [
    {
      "query": "string",
      "purpose": "general | official | refutation | recent | background",
      "priority": integer
    }
  ]
}
"""

class QueryGenerationResult(BaseModel):
    queries: list[dict[str, Any]] = Field(default_factory=list)

class QueryGenerationService:
    @staticmethod
    async def generate_queries(claim: str, claim_metadata: dict[str, Any]) -> QueryGenerationResult:
        user_prompt = (
            f"Normalized Claim: {claim}\n"
            f"Claim Type: {claim_metadata.get('claim_type', 'general')}\n"
            f"Requires Freshness: {claim_metadata.get('requires_freshness', False)}\n"
            f"Location Context: {claim_metadata.get('location_context', '')}\n"
            f"Time Context: {claim_metadata.get('time_context', '')}\n"
        )
        
        max_attempts = settings.llm_json_retry_count + 1
        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = await call_nvidia_chat(
                    task=NVIDIAModelTask.SEARCH_QUERY_GENERATION,
                    messages=[
                        {"role": "system", "content": QUERY_GENERATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                parsed = load_json_with_repair(raw_response)
                queries = parsed.get("queries", [])
                if not isinstance(queries, list) or not queries:
                    raise ValueError("No queries returned in list format.")
                
                validated_queries = []
                for idx, q in enumerate(queries, start=1):
                    q_text = str(q.get("query", "")).strip()
                    purpose = str(q.get("purpose", "general")).strip()
                    if purpose not in ["general", "official", "refutation", "recent", "background"]:
                        purpose = "general"
                    priority = q.get("priority", idx)
                    if q_text:
                        validated_queries.append({
                            "query": q_text,
                            "purpose": purpose,
                            "priority": int(priority)
                        })
                
                return QueryGenerationResult(queries=validated_queries)
            except Exception as exc:
                logger.warning("failed_query_generation attempt=%d error=%s", attempt, str(exc))
        
        # Fallback queries
        fallback = [
            {"query": claim, "purpose": "general", "priority": 1},
            {"query": f"{claim} fact check", "purpose": "refutation", "priority": 2}
        ]
        return QueryGenerationResult(queries=fallback)
