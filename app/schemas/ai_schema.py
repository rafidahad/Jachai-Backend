from __future__ import annotations

from pydantic import StrictInt, StrictStr

from app.schemas.common import StrictBaseModel


class AIUsageSchema(StrictBaseModel):
    embedding_model: StrictStr
    claim_extraction_model: StrictStr | None = None
    query_generation_model: StrictStr | None = None
    rerank_model: StrictStr | None = None
    reasoning_model: StrictStr | None = None
    vision_model: StrictStr | None = None
    llm_call_count: StrictInt = 0
    rerank_call_count: StrictInt = 0
    vision_call_count: StrictInt = 0
