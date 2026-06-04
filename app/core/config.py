from __future__ import annotations

import json
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    nvidia_llm_model: str = Field(alias="NVIDIA_LLM_MODEL")
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM")
    internal_api_key: str = Field(alias="INTERNAL_API_KEY")
    backend_cors_origins: list[str] = Field(default_factory=list, alias="BACKEND_CORS_ORIGINS")
    log_level: str = "INFO"
    auto_create_tables: bool = True
    result_cache_ttl_seconds: int = 60 * 60 * 24
    duplicate_cache_ttl_seconds: int = 60 * 60 * 48
    job_status_ttl_seconds: int = 60 * 60 * 24
    claim_rate_limit_per_minute: int = 20
    internal_rate_limit_per_minute: int = 60
    nvidia_rpm_safety_limit: int = 30
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
