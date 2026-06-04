from app.services.language_service import detect_language
from app.services.pii_service import mask_pii
from app.services.text_cleaning_service import clean_text
from app.utils.hashing import normalized_hash


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("Hello   \n world") == "Hello world"


def test_mask_pii_redacts_email_and_phone() -> None:
    masked = mask_pii("Reach me at jane@example.com or +8801712345678")
    assert "[EMAIL_REDACTED]" in masked
    assert "[PHONE_REDACTED]" in masked


def test_detect_language_handles_banglish() -> None:
    assert detect_language("ami ajke meeting e jabo না") == "Banglish"


def test_normalized_hash_is_whitespace_insensitive() -> None:
    assert normalized_hash("Fact check me") == normalized_hash("  fact   check me  ")
