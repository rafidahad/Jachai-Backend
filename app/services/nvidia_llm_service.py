from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from openai import AsyncOpenAI

from app.core.config import settings
from app.schemas.verdict_schema import LLMVerdictSchema
from app.services.cache_service import cache_service

client = AsyncOpenAI(api_key=settings.nvidia_api_key, base_url=settings.nvidia_base_url)

SYSTEM_PROMPT = """
You are a misinformation verification assistant for South Asia.
Return strict JSON only with keys:
verdict, confidence, explanation, reasoning, summary, source_ids.

Rules:
- Use only the supplied evidence.
- If evidence is weak or conflicting, choose "Not Enough Evidence" or "Misleading".
- Confidence must be between 0 and 1.
- source_ids must contain UUID strings chosen from the evidence list only.
""".strip()


def fallback_verdict(reason: str, source_ids: list[UUID]) -> LLMVerdictSchema:
    return LLMVerdictSchema(
        verdict="Not Enough Evidence",
        confidence=0.2,
        explanation=reason,
        reasoning="The system could not safely complete evidence-grounded reasoning.",
        summary="Not enough evidence is available to verify this claim right now.",
        source_ids=source_ids,
    )


async def generate_verdict(
    claim_text: str,
    language: str,
    evidence: list[dict[str, Any]],
) -> LLMVerdictSchema:
    source_ids = [UUID(item["source_id"]) for item in evidence if item.get("source_id")]
    if not settings.nvidia_api_key:
        return fallback_verdict("NVIDIA API key is not configured.", source_ids)
    if not await cache_service.reserve_nvidia_slot():
        return fallback_verdict("Verification is temporarily rate limited.", source_ids)

    evidence_blob = json.dumps(evidence, default=str)
    user_prompt = (
        f"Language: {language}\n"
        f"Claim: {claim_text}\n"
        f"Evidence: {evidence_blob}\n"
        "Respond with JSON only."
    )

    for _ in range(2):
        try:
            response = await client.chat.completions.create(
                model=settings.nvidia_llm_model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            return LLMVerdictSchema.model_validate(payload)
        except Exception:
            continue

    return fallback_verdict("The language model could not return a valid verdict.", source_ids)
