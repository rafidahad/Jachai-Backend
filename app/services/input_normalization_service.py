from __future__ import annotations
import re
from urllib.parse import urlsplit
import httpx
from bs4 import BeautifulSoup
from app.core.config import settings
from app.core.logging import get_logger
from app.utils.errors import AppError
from app.services.text_cleaning_service import clean_text

logger = get_logger(__name__)

URL_RE = re.compile(
    r'^(?:http|ftp)s?://' # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
    r'localhost|' #localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
    r'(?::\d+)?' # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)

def validate_url(url: str) -> bool:
    return bool(URL_RE.match(url.strip()))

async def fetch_url_content(url: str) -> dict[str, str | None]:
    headers = {
        "User-Agent": "JachAI-Verification-Agent/1.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=settings.fetch_timeout_seconds,
            headers=headers,
        ) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return {"error": f"Unsupported Content-Type: {content_type}"}
                
                body_bytes = b""
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    body_bytes += chunk
                    if len(body_bytes) > settings.max_fetch_bytes:
                        body_bytes = body_bytes[:settings.max_fetch_bytes]
                        break
                
                html = body_bytes.decode("utf-8", errors="replace")
                
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
                    tag.decompose()
                
                title = soup.title.get_text(" ", strip=True) if soup.title else None
                author = None
                meta_author = soup.find("meta", attrs={"name": "author"}) or soup.find("meta", attrs={"property": "article:author"})
                if meta_author and meta_author.get("content"):
                    author = str(meta_author["content"]).strip()
                
                published_date = None
                meta_date = (
                    soup.find("meta", attrs={"name": "pubdate"}) 
                    or soup.find("meta", attrs={"property": "article:published_time"})
                    or soup.find("meta", attrs={"name": "publish-date"})
                )
                if meta_date and meta_date.get("content"):
                    published_date = str(meta_date["content"]).strip()
                
                canonical_link = soup.find("link", rel="canonical")
                canonical_url = canonical_link.get("href") if canonical_link else url
                
                body_text = clean_text(soup.body.get_text(" ") if soup.body else soup.get_text(" "))
                
                return {
                    "title": title or url,
                    "author": author,
                    "published_date": published_date,
                    "canonical_url": str(canonical_url),
                    "raw_text": body_text,
                    "error": None
                }
    except Exception as exc:
        logger.warning("failed_to_fetch_url url=%s error=%s", url, str(exc))
        return {"error": str(exc)}

class InputNormalizationService:
    @staticmethod
    async def normalize(input_type: str, content: str) -> tuple[str, list[str], dict[str, object]]:
        warnings = []
        metadata = {}
        cleaned_content = ""
        
        if input_type == "text":
            cleaned_content = clean_text(content)
            if len(cleaned_content) < 5:
                raise AppError(status_code=400, code="INPUT_TOO_SHORT", message="Input content must be at least 5 characters long.")
            
        elif input_type == "url":
            url = content.strip()
            if not validate_url(url):
                raise AppError(status_code=400, code="INVALID_URL", message="Provided content is not a valid HTTP/HTTPS URL.")
            
            fetch_res = await fetch_url_content(url)
            if fetch_res.get("error"):
                warnings.append(f"Failed to fetch content from URL: {fetch_res['error']}. Falling back to URL text.")
                cleaned_content = url
                metadata = {
                    "title": url,
                    "domain": (urlsplit(url).hostname or "unknown").lower().removeprefix("www."),
                    "url": url,
                    "fetch_status": "failed",
                    "snippet_only": True
                }
            else:
                cleaned_content = fetch_res["raw_text"] or ""
                metadata = {
                    "title": fetch_res["title"],
                    "author": fetch_res["author"],
                    "published_date": fetch_res["published_date"],
                    "canonical_url": fetch_res["canonical_url"],
                    "domain": (urlsplit(url).hostname or "unknown").lower().removeprefix("www."),
                    "url": url,
                    "fetch_status": "success",
                    "snippet_only": False
                }
                if len(cleaned_content) < 50:
                    warnings.append("Fetched URL content is extremely short.")
                
        elif input_type == "image_ocr":
            cleaned_content = clean_text(content)
            cleaned_content = re.sub(r'[|\\/_~`@#$%^&*()_+={}\[\]:;"\'<>]', ' ', cleaned_content)
            cleaned_content = clean_text(cleaned_content)
            if len(cleaned_content) < 5:
                raise AppError(status_code=400, code="INPUT_TOO_SHORT", message="OCR text content must be at least 5 characters long.")
            warnings.append("OCR text is raw and may contain noise.")
            metadata = {
                "ocr_raw": content,
                "cleaned": True
            }
        else:
            raise AppError(status_code=400, code="INVALID_INPUT_TYPE", message=f"Unsupported input type: {input_type}")
            
        return cleaned_content, warnings, metadata
