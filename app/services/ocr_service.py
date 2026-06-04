from __future__ import annotations

import asyncio
from io import BytesIO

import pytesseract
from PIL import Image

from app.core.config import settings
from app.services.text_cleaning_service import clean_text
from app.utils.errors import AppError


async def extract_text_from_image(image_bytes: bytes) -> str:
    def _run() -> str:
        image = Image.open(BytesIO(image_bytes))
        return pytesseract.image_to_string(image, lang=settings.ocr_languages)

    text = clean_text(await asyncio.to_thread(_run))
    if len(text) < settings.ocr_min_characters:
        raise AppError(
            status_code=422,
            code="OCR_FAILED",
            message="OCR could not extract enough text from the image.",
        )
    return text
