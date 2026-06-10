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


def test_detect_language_handles_romanized_bangla_without_native_script() -> None:
    assert detect_language("Donald trump namer mohish ekhon chiriakhanae dhakar.") == "Banglish"


def test_normalized_hash_is_whitespace_insensitive() -> None:
    assert normalized_hash("Fact check me") == normalized_hash("  fact   check me  ")


def test_embed_text_uses_gemini_embeddings_with_retrieval_task_types() -> None:
    import asyncio
    from unittest.mock import AsyncMock, patch

    from app.services.embedding_service import embed_text

    async def run() -> list[float]:
        with patch("app.services.embedding_service.settings") as mock_settings:
            mock_settings.embedding_uses_gemini = True
            mock_settings.embedding_model = "gemini-embedding-2"
            mock_settings.embedding_dim = 1024
            with patch(
                "app.services.embedding_service.call_gemini_embed_content",
                new_callable=AsyncMock,
                return_value=[3.0, 4.0],
            ) as mock_embed:
                values = await embed_text(
                    "Bangla claim text",
                    task_type="RETRIEVAL_QUERY",
                )
                kwargs = mock_embed.await_args.kwargs
                assert kwargs["model"] == "gemini-embedding-2"
                assert kwargs["output_dimensionality"] == 1024
                assert kwargs["task_type"] == "RETRIEVAL_QUERY"
                return values

    normalized = asyncio.run(run())
    assert normalized == [0.6, 0.8]


def test_embed_texts_batches_local_embeddings() -> None:
    import asyncio
    from unittest.mock import MagicMock, patch

    from app.services.embedding_service import embed_texts

    async def run() -> list[list[float]]:
        with patch("app.services.embedding_service.settings") as mock_settings:
            mock_settings.embedding_uses_gemini = False
            model = MagicMock()
            model.encode.return_value = [[0.6, 0.8], [1.0, 0.0]]
            with patch("app.services.embedding_service.get_embedding_model", return_value=model):
                values = await embed_texts(
                    ["First claim", "Second claim"],
                    task_type="RETRIEVAL_DOCUMENT",
                    titles=["one", "two"],
                )
                model.encode.assert_called_once_with(
                    ["First claim", "Second claim"],
                    normalize_embeddings=True,
                )
                return values

    vectors = asyncio.run(run())
    assert vectors == [[0.6, 0.8], [1.0, 0.0]]
