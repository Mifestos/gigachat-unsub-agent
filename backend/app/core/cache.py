from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import redis

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class MerchantCache:
    """Кэш ответов агента по мерчантам: Redis (основной) + In-Memory словарь (fallback)."""

    def __init__(self, redis_url: str = settings.redis_url, ttl_seconds: int = settings.cache_ttl_seconds) -> None:
        self.ttl_seconds = ttl_seconds
        self._memory_store: dict[str, str] = {}
        self._redis_client: redis.Redis | None = None

        try:
            client = redis.Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1.5)
            client.ping()
            self._redis_client = client
            logger.info("Подключен Redis кэш (%s).", redis_url)
        except Exception:
            logger.info("Redis недоступен локально — используем In-Memory кэш (graceful fallback).")
            self._redis_client = None

    @staticmethod
    def _make_key(merchant_clean: str) -> str:
        digest = hashlib.sha256(merchant_clean.strip().upper().encode("utf-8")).hexdigest()[:16]
        return f"unsub:merchant:{digest}"

    def get(self, merchant_clean: str) -> dict[str, Any] | None:
        key = self._make_key(merchant_clean)
        raw: str | None = None

        if self._redis_client is not None:
            try:
                raw = self._redis_client.get(key)
            except Exception:
                raw = self._memory_store.get(key)
        else:
            raw = self._memory_store.get(key)

        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set(self, merchant_clean: str, payload: dict[str, Any]) -> None:
        key = self._make_key(merchant_clean)
        serialized = json.dumps(payload, ensure_ascii=False)
        self._memory_store[key] = serialized

        if self._redis_client is not None:
            try:
                self._redis_client.setex(key, self.ttl_seconds, serialized)
            except Exception as e:
                logger.warning("Не удалось записать ключ в Redis (%s), сохранено в памяти.", e)
