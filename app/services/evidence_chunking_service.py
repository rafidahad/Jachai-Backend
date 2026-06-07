from __future__ import annotations
import re
from typing import Any
from pydantic import BaseModel

class EvidenceChunk(BaseModel):
    chunk_id: str
    source_id: str
    title: str
    url: str
    domain: str
    published_date: str | None = None
    text: str
    snippet_only: bool

class EvidenceChunkingService:
    @staticmethod
    def chunk(cleaned_evidence: list[Any], max_chunks: int = 12) -> list[EvidenceChunk]:
        chunks = []
        
        for item in cleaned_evidence:
            text = item.cleaned_text
            source_id = item.source_id
            
            if item.snippet_only:
                chunks.append(EvidenceChunk(
                    chunk_id=f"{source_id}_ch_0",
                    source_id=source_id,
                    title=item.title,
                    url=item.url,
                    domain=item.domain,
                    published_date=item.published_date,
                    text=text,
                    snippet_only=True
                ))
                continue
                
            paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
            
            current_chunk = []
            current_len = 0
            sub_chunk_idx = 0
            
            for p in paragraphs:
                current_chunk.append(p)
                current_len += len(p)
                
                if current_len >= 2000:
                    chunk_text = "\n".join(current_chunk)
                    chunks.append(EvidenceChunk(
                        chunk_id=f"{source_id}_ch_{sub_chunk_idx}",
                        source_id=source_id,
                        title=item.title,
                        url=item.url,
                        domain=item.domain,
                        published_date=item.published_date,
                        text=chunk_text,
                        snippet_only=False
                    ))
                    sub_chunk_idx += 1
                    current_chunk = []
                    current_len = 0
                    
            if current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append(EvidenceChunk(
                    chunk_id=f"{source_id}_ch_{sub_chunk_idx}",
                    source_id=source_id,
                    title=item.title,
                    url=item.url,
                    domain=item.domain,
                    published_date=item.published_date,
                    text=chunk_text,
                    snippet_only=False
                ))
                
        seen_texts = set()
        unique_chunks = []
        for c in chunks:
            if c.text.lower() in seen_texts:
                continue
            seen_texts.add(c.text.lower())
            unique_chunks.append(c)
            
        return unique_chunks[:max_chunks]
