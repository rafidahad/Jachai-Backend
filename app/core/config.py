from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "JachAI Backend"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    database_url: str = Field(alias="DATABASE_URL")
    database_sync_url: str = Field(alias="DATABASE_SYNC_URL")
    redis_url: str = Field(alias="REDIS_URL")
    nvidia_api_key: str = Field(alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(alias="NVIDIA_BASE_URL")
    nvidia_llm_model: str | None = Field(default=None, alias="NVIDIA_LLM_MODEL")
    nvidia_reasoning_model: str | None = Field(default=None, alias="NVIDIA_REASONING_MODEL")
    nvidia_vision_model: str | None = Field(default=None, alias="NVIDIA_VISION_MODEL")
    nvidia_enable_vision_fallback: bool = Field(default=False, alias="NVIDIA_ENABLE_VISION_FALLBACK")
    nvidia_rpm_safety_limit: int = Field(default=30, alias="NVIDIA_MAX_RPM")
    nvidia_timeout_seconds: float = Field(default=60.0, alias="NVIDIA_TIMEOUT_SECONDS")
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM")
    internal_api_key: str = Field(alias="INTERNAL_API_KEY")
    backend_cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        alias="BACKEND_CORS_ORIGINS",
    )
    log_level: str = "INFO"
    auto_create_tables: bool = True
    result_cache_ttl_seconds: int = 60 * 60 * 24
    duplicate_cache_ttl_seconds: int = 60 * 60 * 48
    job_status_ttl_seconds: int = 60 * 60 * 24
    claim_rate_limit_per_minute: int = 20
    internal_rate_limit_per_minute: int = 60
    evidence_top_k: int = 5
    ocr_min_characters: int = 12
    ocr_languages: str = "eng+ben+hin"
    request_timeout_seconds: float = 15.0

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

    @field_validator("debug", "nvidia_enable_vision_fallback", mode="before")
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
    def active_nvidia_vision_model(self) -> str | None:
        if not self.nvidia_vision_model:
            return None
        normalized = self.nvidia_vision_model.strip()
        return normalized or None

    @property
    def nvidia_vision_enabled(self) -> bool:
        return self.nvidia_enable_vision_fallback and self.active_nvidia_vision_model is not None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
