from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gigachat_credentials: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_model: str = "GigaChat"
    gigachat_verify_ssl: bool = False

    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 60 * 60 * 24 * 7  # 7 дней

    model_path: Path = Path("models/subscription_catboost.cbm")
    model_config_path: Path = Path("models/model_config.json")


settings = Settings()
