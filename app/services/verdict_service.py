from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logging import get_logger
from app.services.nvidia_client import call_nvidia_chat
from app.services.ai_model_router import NVIDIAModelTask
from app.utils.json_repair import load_json_with_repair

logger = get_logger(__name__)

FINAL_VERDICT_SYSTEM_PROMPT = """
You are JachAI's senior fact-checking and consensus verdict model.
Your task is to assign a final, authoritative verification verdict to the normalized claim based ONLY on the provided stance-classified evidence chunks and credibility scores.

Verdict Options:
- `supported`: Credible sources directly confirm/support the claim.
- `refuted`: Credible sources directly contradict/refute the claim.
- `misleading`: The claim mixes some truth with unsupported or distorted details.
- `partially_true`: The claim is partially true but incomplete or misses important context.
- `outdated`: The claim was true in the past but is no longer current or accurate.
- `insufficient_evidence`: The provided evidence is weak, missing, unrelated, or too conflicting to verify.
- `unverifiable`: The claim is vague, purely subjective, or impossible to verify with factual search results.

Rules:
- Prefer `insufficient_evidence` over guessing when evidence is weak, missing, or mostly neutral/background.
- If there are contradictions among credible sources, explain this and assign `misleading`, `partially_true`, or `insufficient_evidence`.
- Reduce confidence if source quality (credibility score) is low, or if the evidence is snippet-only.
- Never cite source/evidence IDs that were not supplied in the input.
- Do not use markdown format. Return JSON matching:
{
  "verdict": "supported | refuted | misleading | partially_true | outdated | insufficient_evidence | unverifiable",
  "confidence": float,
  "explanation": "string",
  "key_evidence_ids": ["string"],
  "warnings": ["string"]
}
"""

class FinalVerdictResult(BaseModel):
    verdict: str
    confidence: float
    explanation: str
    key_evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class VerdictService:
    @staticmethod
    async def generate_verdict(
        claim: str,
        claim_metadata: dict[str, Any],
        classified_chunks: list[Any],
        credibility_scores: list[Any],
        pipeline_warnings: list[str]
    ) -> FinalVerdictResult:
        credibility_lookup = {score.source_id: score for score in credibility_scores}
        
        evidence_input = []
        for c in classified_chunks:
            score_info = credibility_lookup.get(c.chunk_id.split("_ch_")[0])
            cred_score = score_info.credibility_score if score_info else 0.50
            source_type = score_info.source_type if score_info else "unknown"
            
            evidence_input.append({
                "chunk_id": c.chunk_id,
                "stance": c.stance,
                "rationale": c.rationale,
                "quoted_evidence": c.quoted_evidence,
                "credibility_score": cred_score,
                "source_type": source_type
            })

        user_prompt = (
            f"Normalized Claim: {claim}\n"
            f"Claim Metadata: {claim_metadata}\n\n"
            f"Evidence Chunks and Classification: {evidence_input}\n"
            f"Existing Warnings: {pipeline_warnings}\n"
        )
        
        max_attempts = settings.llm_json_retry_count + 1
        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = await call_nvidia_chat(
                    task=NVIDIAModelTask.CLAIM_REASONING,
                    messages=[
                        {"role": "system", "content": FINAL_VERDICT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                parsed = load_json_with_repair(raw_response)
                
                verdict = str(parsed.get("verdict", "insufficient_evidence")).strip().lower()
                if verdict not in ["supported", "refuted", "misleading", "partially_true", "outdated", "insufficient_evidence", "unverifiable"]:
                    verdict = "insufficient_evidence"
                    
                confidence = float(parsed.get("confidence", 0.5))
                explanation = str(parsed.get("explanation", "Verification completed with insufficient evidence details."))
                key_evidence_ids = [str(eid) for eid in parsed.get("key_evidence_ids", [])]
                warnings = [str(w) for w in parsed.get("warnings", [])]
                
                has_supports_or_refutes = any(c["stance"] in ["supports", "refutes"] for c in evidence_input)
                if not has_supports_or_refutes and verdict in ["supported", "refuted"]:
                    verdict = "insufficient_evidence"
                    confidence = 0.3
                    explanation = "Automatic fallback: No directly supporting or refuting evidence found."

                return FinalVerdictResult(
                    verdict=verdict,
                    confidence=confidence,
                    explanation=explanation,
                    key_evidence_ids=key_evidence_ids,
                    warnings=warnings
                )
            except Exception as exc:
                logger.warning("failed_verdict_generation attempt=%d error=%s", attempt, str(exc))
                
        return FinalVerdictResult(
            verdict="insufficient_evidence",
            confidence=0.0,
            explanation="Failed to obtain verdict reasoning from the LLM.",
            key_evidence_ids=[],
            warnings=["Pipeline experienced technical issues generating the final verdict."]
        )
