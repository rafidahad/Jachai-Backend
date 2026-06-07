from __future__ import annotations
from urllib.parse import urlsplit
from pydantic import BaseModel
from app.core.config import settings

class SourceCredibilityResult(BaseModel):
    source_id: str
    title: str
    url: str
    domain: str
    source_type: str
    credibility_score: float
    credibility_reason: str

class SourceCredibilityService:
    @staticmethod
    def score_credibility(source_id: str, title: str, url: str, domain: str, snippet_only: bool) -> SourceCredibilityResult:
        trusted_domains = settings.parsed_trusted_domains
        
        source_type = "unknown"
        credibility_score = 0.50
        reason = "Unknown domain category."
        
        if domain.endswith(".gov") or domain.endswith(".gov.bd") or domain.endswith(".mil"):
            source_type = "government"
            credibility_score = 0.95
            reason = "Official government domain."
        elif domain.endswith(".edu") or domain.endswith(".ac.bd") or "academic" in domain:
            source_type = "academic"
            credibility_score = 0.90
            reason = "Educational or academic institution."
        elif any(domain == td or domain.endswith(f".{td}") for td in trusted_domains):
            source_type = "fact_check" if ("factcheck" in domain or "scanner" in domain or "boomlive" in domain) else "reputable_news"
            credibility_score = 0.95 if source_type == "fact_check" else 0.92
            reason = f"Verified trusted publisher ({domain})."
        elif "wikipedia.org" in domain:
            source_type = "primary_source"
            credibility_score = 0.85
            reason = "Reputable crowd-sourced primary source."
        elif any(social in domain for social in ["facebook.com", "twitter.com", "x.com", "instagram.com", "reddit.com", "tiktok.com", "youtube.com"]):
            source_type = "social_media"
            credibility_score = 0.30
            reason = "User-generated content from social media."
        elif any(blog in domain for blog in ["blogspot.com", "wordpress.com", "medium.com"]):
            source_type = "blog"
            credibility_score = 0.45
            reason = "Personal blog / blogging platform."
        else:
            if any(kw in domain for kw in ["news", "times", "post", "tribune", "herald", "daily"]):
                source_type = "reputable_news"
                credibility_score = 0.80
                reason = "News organization domain."
            else:
                source_type = "unknown"
                credibility_score = 0.50
                reason = "Unclassified domain."

        if snippet_only:
            credibility_score = max(0.1, credibility_score - 0.15)
            reason += " (Penalty applied: only search snippet was available, page content could not be fetched)."
            
        return SourceCredibilityResult(
            source_id=source_id,
            title=title,
            url=url,
            domain=domain,
            source_type=source_type,
            credibility_score=round(credibility_score, 2),
            credibility_reason=reason
        )
