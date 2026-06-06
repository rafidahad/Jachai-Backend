from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="JachAI Backend", validation_alias=AliasChoices("APP_NAME"))
    app_version: str = "0.1.0"
    app_env: str = Field(default="development", validation_alias=AliasChoices("APP_ENV", "ENVIRONMENT"))
    debug: bool = Field(default=False, validation_alias=AliasChoices("DEBUG"))
    api_v1_prefix: str = Field(default="/api/v1", validation_alias=AliasChoices("API_V1_PREFIX"))
    database_url: str = Field(validation_alias=AliasChoices("DATABASE_URL"))
    database_sync_url: str = Field(validation_alias=AliasChoices("DATABASE_SYNC_URL"))
    redis_url: str = Field(validation_alias=AliasChoices("REDIS_URL"))
    nvidia_api_key: str = Field(validation_alias=AliasChoices("NVIDIA_API_KEY"))
    nvidia_base_url: str = Field(validation_alias=AliasChoices("NVIDIA_BASE_URL"))
    google_fact_check_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_FACT_CHECK_API_KEY"),
    )
    google_fact_check_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("GOOGLE_FACT_CHECK_ENABLED"),
    )
    nvidia_llm_model: str | None = Field(default=None, validation_alias=AliasChoices("NVIDIA_LLM_MODEL"))
    nvidia_reasoning_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("NVIDIA_REASONING_MODEL"),
    )
    nvidia_claim_extraction_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("NVIDIA_CLAIM_EXTRACTION_MODEL"),
    )
    nvidia_query_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("NVIDIA_QUERY_MODEL", "NVIDIA_SEARCH_QUERY_MODEL"),
    )
    nvidia_rerank_model: str | None = Field(
        default="nvidia/rerank-qa-mistral-4b",
        validation_alias=AliasChoices("NVIDIA_RERANK_MODEL"),
    )
    nvidia_vision_model: str | None = Field(default=None, validation_alias=AliasChoices("NVIDIA_VISION_MODEL"))
    nvidia_enable_vision_fallback: bool = Field(
        default=False,
        validation_alias=AliasChoices("NVIDIA_ENABLE_VISION_FALLBACK", "ENABLE_VISION_FALLBACK"),
    )
    nvidia_rpm_safety_limit: int = Field(default=30, validation_alias=AliasChoices("NVIDIA_MAX_RPM"))
    nvidia_max_uncached_claims_per_minute: int = Field(
        default=15,
        validation_alias=AliasChoices("NVIDIA_MAX_UNCACHED_CLAIMS_PER_MINUTE"),
    )
    nvidia_timeout_seconds: float = Field(default=60.0, validation_alias=AliasChoices("NVIDIA_TIMEOUT_SECONDS"))
    embedding_model: str = Field(default="BAAI/bge-m3", validation_alias=AliasChoices("EMBEDDING_MODEL"))
    embedding_dim: int = Field(
        default=1024,
        validation_alias=AliasChoices("EMBEDDING_DIM", "EMBEDDING_DIMENSION"),
    )
    embedding_device: str = Field(default="cpu", validation_alias=AliasChoices("EMBEDDING_DEVICE"))
    internal_api_key: str = Field(validation_alias=AliasChoices("INTERNAL_API_KEY"))
    backend_cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        validation_alias=AliasChoices("BACKEND_CORS_ORIGINS", "CORS_ORIGINS"),
    )
    log_level: str = "INFO"
    auto_create_tables: bool = True
    result_cache_ttl_seconds: int = Field(
        default=60 * 60 * 24,
        validation_alias=AliasChoices("RESULT_CACHE_TTL_SECONDS", "CLAIM_CACHE_TTL_SECONDS"),
    )
    duplicate_cache_ttl_seconds: int = Field(
        default=60 * 60 * 48,
        validation_alias=AliasChoices("DUPLICATE_CACHE_TTL_SECONDS", "CLAIM_CACHE_TTL_SECONDS"),
    )
    job_status_ttl_seconds: int = 60 * 60 * 24
    claim_rate_limit_per_minute: int = 20
    internal_rate_limit_per_minute: int = 60
    pgvector_top_k: int = Field(default=20, validation_alias=AliasChoices("PGVECTOR_TOP_K"))
    final_evidence_top_k: int = Field(default=5, validation_alias=AliasChoices("FINAL_EVIDENCE_TOP_K"))
    rerank_min_candidates: int = Field(default=6, validation_alias=AliasChoices("RERANK_MIN_CANDIDATES"))
    min_relevant_similarity: float = Field(default=0.60, validation_alias=AliasChoices("MIN_RELEVANT_SIMILARITY"))
    ocr_engine: str = Field(default="tesseract", validation_alias=AliasChoices("OCR_ENGINE"))
    ocr_min_characters: int = Field(default=10, validation_alias=AliasChoices("OCR_MIN_CHARACTERS"))
    ocr_languages: str = Field(default="eng+ben+hin", validation_alias=AliasChoices("OCR_LANGUAGES"))
    max_image_size_mb: int = Field(default=8, validation_alias=AliasChoices("MAX_IMAGE_SIZE_MB"))
    request_timeout_seconds: float = 15.0
    search_provider: str = Field(default="google_fact_check", validation_alias=AliasChoices("SEARCH_PROVIDER"))
    general_search_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GENERAL_SEARCH_API_KEY"),
    )
    general_search_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GENERAL_SEARCH_BASE_URL"),
    )
    general_search_max_results: int = Field(default=8, validation_alias=AliasChoices("GENERAL_SEARCH_MAX_RESULTS"))
    live_evidence_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("LIVE_EVIDENCE_ENABLED"),
    )
    live_evidence_timeout_seconds: float = Field(
        default=12.0,
        validation_alias=AliasChoices("LIVE_EVIDENCE_TIMEOUT_SECONDS"),
    )
    live_evidence_max_documents: int = Field(
        default=10,
        validation_alias=AliasChoices("LIVE_EVIDENCE_MAX_DOCUMENTS"),
    )
    live_evidence_max_google_results: int = Field(
        default=6,
        validation_alias=AliasChoices("LIVE_EVIDENCE_MAX_GOOGLE_RESULTS"),
    )
    live_evidence_max_listing_links_per_catalog: int = Field(
        default=8,
        validation_alias=AliasChoices("LIVE_EVIDENCE_MAX_LISTING_LINKS_PER_CATALOG"),
    )
    live_evidence_max_documents_per_catalog: int = Field(
        default=2,
        validation_alias=AliasChoices("LIVE_EVIDENCE_MAX_DOCUMENTS_PER_CATALOG"),
    )
    live_evidence_min_article_characters: int = Field(
        default=240,
        validation_alias=AliasChoices("LIVE_EVIDENCE_MIN_ARTICLE_CHARACTERS"),
    )

    @field_validator("backend_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("["):
                parsed = json.loads(value)
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in value.split(",") if item.strip()]
        raise TypeError("Invalid BACKEND_CORS_ORIGINS value")

    @field_validator("database_url", mode="before")
    @classmethod
    def parse_database_url(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise TypeError("Invalid DATABASE_URL value")
        normalized = value.strip()
        if normalized.startswith("postgres://"):
            normalized = "postgresql://" + normalized[len("postgres://") :]
        if normalized.startswith("postgresql://"):
            normalized = normalized.replace("postgresql://", "postgresql+asyncpg://", 1)
        parts = urlsplit(normalized)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        if "sslmode" in query and "ssl" not in query:
            query["ssl"] = query.pop("sslmode")
        query.pop("channel_binding", None)
        normalized = urlunsplit(parts._replace(query=urlencode(query)))
        return normalized

    @field_validator("database_sync_url", mode="before")
    @classmethod
    def parse_database_sync_url(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise TypeError("Invalid DATABASE_SYNC_URL value")
        normalized = value.strip()
        if normalized.startswith("postgres://"):
            normalized = "postgresql://" + normalized[len("postgres://") :]
        if normalized.startswith("postgresql+asyncpg://"):
            normalized = normalized.replace("postgresql+asyncpg://", "postgresql://", 1)
        return normalized

    @field_validator(
        "debug",
        "nvidia_enable_vision_fallback",
        "google_fact_check_enabled",
        "live_evidence_enabled",
        mode="before",
    )
    @classmethod
    def parse_bool_flags(cls, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None or value == "":
            return False
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "development", "dev"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "production", "prod"}:
                return False
        raise TypeError("Invalid boolean flag value")

    @property
    def active_nvidia_reasoning_model(self) -> str | None:
        return self.nvidia_reasoning_model or self.nvidia_llm_model

    @property
    def active_nvidia_claim_extraction_model(self) -> str | None:
        return self.nvidia_claim_extraction_model or self.active_nvidia_reasoning_model

    @property
    def active_nvidia_query_model(self) -> str | None:
        return self.nvidia_query_model or self.active_nvidia_claim_extraction_model

    @property
    def active_nvidia_rerank_model(self) -> str | None:
        if not self.nvidia_rerank_model:
            return None
        normalized = self.nvidia_rerank_model.strip()
        return normalized or None

    @property
    def active_nvidia_vision_model(self) -> str | None:
        if not self.nvidia_vision_model:
            return None
        normalized = self.nvidia_vision_model.strip()
        return normalized or None

    @property
    def nvidia_vision_enabled(self) -> bool:
        return self.nvidia_enable_vision_fallback and self.active_nvidia_vision_model is not None

    @property
    def normalized_search_provider(self) -> str:
        return self.search_provider.strip().lower().replace("-", "_")

    @property
    def general_search_enabled(self) -> bool:
        return self.normalized_search_provider not in {"", "none", "disabled", "google_fact_check"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
