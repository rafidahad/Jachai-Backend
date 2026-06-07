from __future__ import annotations
import asyncio
from typing import Any
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logging import get_logger
from app.services.nvidia_client import call_nvidia_chat
from app.services.ai_model_router import NVIDIAModelTask
from app.utils.json_repair import load_json_with_repair

logger = get_logger(__name__)

STANCE_CLASSIFICATION_SYSTEM_PROMPT = """
You are JachAI's evidence stance classification agent.
Your task is to classify whether a given evidence chunk supports, refutes, or is neutral/background to a normalized claim.

Rules:
- Stance labels:
  - `supports`: the chunk directly validates or verifies the central factual assertion of the claim.
  - `refutes`: the chunk directly contradicts or disproves the central factual assertion.
  - `neutral`: the chunk relates to the entities or topic but does not prove or disprove the assertion.
  - `background`: the chunk provides helpful context but does not address the truth of the assertion.
- Quoted evidence: Extract a short, relevant verbatim excerpt or concise paraphrase from the chunk that justifies your stance.
- Do not make any claims or assumptions outside the provided chunk.
- Do not output markdown format. Return JSON matching:
{
  "chunk_id": "string",
  "stance": "supports | refutes | neutral | background",
  "confidence": float,
  "rationale": "string",
  "quoted_evidence": "string"
}
"""

class ClassifiedChunk(BaseModel):
    chunk_id: str
    stance: str
    confidence: float
    rationale: str
    quoted_evidence: str

class EvidenceClassificationService:
    @staticmethod
    async def classify_chunks(claim: str, chunks: list[Any]) -> list[ClassifiedChunk]:
        classified = []
        if not chunks:
            return classified
        
        async def classify_one(c: Any) -> ClassifiedChunk | None:
            user_prompt = (
                f"Normalized Claim: {claim}\n\n"
                f"Chunk ID: {c.chunk_id}\n"
                f"Source Title: {c.title}\n"
                f"Evidence Chunk Text:\n{c.text}\n"
            )
            
            max_attempts = settings.llm_json_retry_count + 1
            for attempt in range(1, max_attempts + 1):
                try:
                    raw_response = await call_nvidia_chat(
                        task=NVIDIAModelTask.CLAIM_REASONING,
                        messages=[
                            {"role": "system", "content": STANCE_CLASSIFICATION_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        response_format={"type": "json_object"}
                    )
                    parsed = load_json_with_repair(raw_response)
                    stance = str(parsed.get("stance", "neutral")).strip().lower()
                    if stance not in ["supports", "refutes", "neutral", "background"]:
                        stance = "neutral"
                        
                    return ClassifiedChunk(
                        chunk_id=c.chunk_id,
                        stance=stance,
                        confidence=float(parsed.get("confidence", 0.5)),
                        rationale=str(parsed.get("rationale", "No rationale provided.")),
                        quoted_evidence=str(parsed.get("quoted_evidence", ""))
                    )
                except Exception as exc:
                    logger.warning("failed_chunk_classification chunk_id=%s attempt=%d error=%s", c.chunk_id, attempt, str(exc))
            
            return ClassifiedChunk(
                chunk_id=c.chunk_id,
                stance="neutral",
                confidence=0.0,
                rationale="LLM classification failed.",
                quoted_evidence=""
            )

        results = await asyncio.gather(*[classify_one(c) for c in chunks], return_exceptions=True)
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                logger.error("error_during_classification error=%s", str(res))
                c = chunks[idx]
                classified.append(ClassifiedChunk(
                    chunk_id=c.chunk_id,
                    stance="neutral",
                    confidence=0.0,
                    rationale="Classification process crashed.",
                    quoted_evidence=""
                ))
            elif res is not None:
                classified.append(res)
                
        return classified
