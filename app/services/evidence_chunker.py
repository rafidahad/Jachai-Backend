"""
evidence_chunker.py
-------------------
Splits fetched evidence documents into passage-level chunks for
ranking and stance classification.

Design decisions:
- Character-level approximation (~4 chars per token) avoids adding a tokenizer dep.
- Target: 300-600 chars per chunk (~75-150 tokens), hard cap at 700 chars.
- Splits on sentence boundaries (period/exclamation/question + whitespace).
- Preserves all source metadata in every chunk.
- Deduplicates near-identical chunks within the same request.
- snippet_only chunks use the raw snippet as the single chunk (no further split).
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Sentence boundary pattern: end of sentence punctuation followed by whitespace or end
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")

TARGET_CHUNK_CHARS = 500   # target characters per chunk
MAX_CHUNK_CHARS = 720      # hard cap per chunk
MIN_CHUNK_CHARS = 80       # discard tiny fragments


def _normalize_text(text: str) -> str:
    """Collapse whitespace for consistent chunk boundaries."""
    return _WHITESPACE.sub(" ", text).strip()


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries, returning non-empty sentences."""
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def _make_chunk_id(source_id: str, offset: int) -> str:
    """Deterministic chunk ID based on source_id + position."""
    raw = f"{source_id}:{offset}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]  # noqa: S324  (not crypto)


def _chunk_text(text: str, source_id: str) -> list[tuple[str, str]]:
    """
    Split a cleaned text body into (chunk_id, text) pairs.

    Returns list of (chunk_id, chunk_text).
    """
    sentences = _split_sentences(_normalize_text(text))
    chunks: list[tuple[str, str]] = []
    current_parts: list[str] = []
    current_len = 0
    chunk_index = 0

    for sentence in sentences:
        slen = len(sentence)
        # If a single sentence exceeds max, hard-split it
        if slen > MAX_CHUNK_CHARS:
            # Flush current buffer first
            if current_parts:
                chunk_text = " ".join(current_parts)
                if len(chunk_text) >= MIN_CHUNK_CHARS:
                    chunks.append((_make_chunk_id(source_id, chunk_index), chunk_text))
                    chunk_index += 1
                current_parts = []
                current_len = 0
            # Hard-split long sentence
            for start in range(0, slen, MAX_CHUNK_CHARS):
                part = sentence[start : start + MAX_CHUNK_CHARS].strip()
                if len(part) >= MIN_CHUNK_CHARS:
                    chunks.append((_make_chunk_id(source_id, chunk_index), part))
                    chunk_index += 1
            continue

        # Adding this sentence would exceed target — flush and start new chunk
        if current_len + slen > TARGET_CHUNK_CHARS and current_parts:
            chunk_text = " ".join(current_parts)
            if len(chunk_text) >= MIN_CHUNK_CHARS:
                chunks.append((_make_chunk_id(source_id, chunk_index), chunk_text))
                chunk_index += 1
            current_parts = []
            current_len = 0

        current_parts.append(sentence)
        current_len += slen + 1  # +1 for space

    # Flush remaining
    if current_parts:
        chunk_text = " ".join(current_parts)
        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunks.append((_make_chunk_id(source_id, chunk_index), chunk_text))

    return chunks


def _dedup_chunks(
    all_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove near-identical chunks by normalized text fingerprint."""
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for chunk in all_chunks:
        # Fingerprint: first 200 chars normalized
        fingerprint = _normalize_text(chunk["text"])[:200].lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(chunk)
    return deduped


def chunk_evidence_documents(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert a list of fetched evidence documents into passage-level chunks.

    Each document dict is expected to have:
      - source_id (str)
      - title (str)
      - url (str)
      - domain (str)
      - published_date (str | None)
      - text_content (str) — full page text
      - snippet (str) — short snippet (fallback)
      - snippet_only (bool)

    Returns a list of chunk dicts conforming to EvidenceChunkSchema.
    """
    all_chunks: list[dict[str, Any]] = []

    for doc in documents:
        source_id = str(doc.get("source_id") or doc.get("url") or "unknown")
        title = str(doc.get("title") or "")
        url = str(doc.get("url") or "")
        domain = str(doc.get("domain") or "")
        published_date = doc.get("published_date")
        snippet_only = bool(doc.get("snippet_only", False))
        text_content = str(doc.get("text_content") or doc.get("snippet") or "")
        snippet = str(doc.get("snippet") or "")

        base_meta = {
            "source_id": source_id,
            "title": title,
            "url": url,
            "domain": domain,
            "published_date": published_date,
            "snippet_only": snippet_only,
        }

        if snippet_only or len(text_content) < MIN_CHUNK_CHARS:
            # Use snippet as single weak chunk
            chunk_text = snippet or text_content
            if len(chunk_text) >= MIN_CHUNK_CHARS:
                all_chunks.append({
                    **base_meta,
                    "chunk_id": _make_chunk_id(source_id, 0),
                    "text": _normalize_text(chunk_text)[:MAX_CHUNK_CHARS],
                    "snippet_only": True,
                })
            continue

        text_to_chunk = text_content
        chunk_pairs = _chunk_text(text_to_chunk, source_id)

        if not chunk_pairs:
            # Fallback: use snippet
            if len(snippet) >= MIN_CHUNK_CHARS:
                all_chunks.append({
                    **base_meta,
                    "chunk_id": _make_chunk_id(source_id, 0),
                    "text": _normalize_text(snippet)[:MAX_CHUNK_CHARS],
                    "snippet_only": True,
                })
            continue

        for chunk_id, chunk_text in chunk_pairs:
            all_chunks.append({
                **base_meta,
                "chunk_id": chunk_id,
                "text": chunk_text,
            })

    deduped = _dedup_chunks(all_chunks)
    cap = settings.max_evidence_chunks
    if len(deduped) > cap:
        logger.debug("evidence_chunker capping chunks from %d to %d", len(deduped), cap)
        deduped = deduped[:cap]

    logger.info(
        "evidence_chunker docs=%d chunks_before_dedup=%d chunks_after_dedup=%d",
        len(documents),
        len(all_chunks),
        len(deduped),
    )
    return deduped
