from __future__ import annotations
from typing import Any
from pydantic import BaseModel
from app.core.logging import get_logger
from app.services.text_cleaning_service import clean_text

logger = get_logger(__name__)

class CleanedEvidence(BaseModel):
    source_id: str
    title: str
    url: str
    domain: str
    published_date: str | None = None
    cleaned_text: str
    snippet_only: bool
    is_weak: bool

class EvidenceCleaningService:
    @staticmethod
    def clean(fetched_list: list[Any], requires_freshness: bool) -> tuple[list[CleanedEvidence], list[str]]:
        warnings = []
        cleaned_list = []
        
        for item in fetched_list:
            text = item.raw_text or ""
            cleaned = clean_text(text)
            
            is_weak = False
            if len(cleaned) < 100:
                is_weak = True
                
            if requires_freshness and not item.published_date:
                warnings.append(f"Evidence from {item.domain} has no publication date but claim is time-sensitive.")
                is_weak = True
                
            cleaned_list.append(CleanedEvidence(
                source_id=item.source_id,
                title=item.title,
                url=item.url,
                domain=item.domain,
                published_date=item.published_date,
                cleaned_text=cleaned,
                snippet_only=item.snippet_only,
                is_weak=is_weak
            ))
            
        return cleaned_list, warnings
