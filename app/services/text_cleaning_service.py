from __future__ import annotations

import re
from html import unescape

from bs4 import BeautifulSoup

CONTROL_CHARS_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    value = unescape(value)
    value = CONTROL_CHARS_RE.sub(" ", value)
    value = value.replace("\x00", " ")
    value = WHITESPACE_RE.sub(" ", value).strip()
    return value


def text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return clean_text(soup.get_text(separator=" "))
