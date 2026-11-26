from __future__ import annotations

import json
import logging
import re
from typing import Any, TypedDict

from ddgs import DDGS
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from backend.app.core.cache import MerchantCache
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class SubscriptionGuidance(BaseModel):
    """Структурированный ответ AI-агента по отмене подписки."""

    merchant_clean: str
    service_name: str = Field(description="Понятное название сервиса или бренда на русском/английском")
    category: str = Field(description="Категория подписки: Кино и музыка, Облако, Образование, ПО и др.")
    unsubscribe_url: str | None = Field(default=None, description="Прямая ссылка на страницу отписки из поиска")
    steps: list[str] = Field(description="2-3 четких шага для отключения автопродления")
    source: str = Field(default="gigachat+web", description="Источник: cache | gigachat+web | web_fallback")


class AgentState(TypedDict, total=False):
    merchant_clean: str
    mcc_code: str
    amount: float
    search_results: list[dict[str, str]]
    guidance: dict[str, Any]
    from_cache: bool


SYSTEM_PROMPT = """Ты — финансовый AI-помощник российского банка (Умный анти-подписочный агент).
Твоя задача — по названию мерчанта из банковской выписки и результатам веб-поиска определить:
1. Понятное человеку название сервиса (service_name).
2. Категорию сервиса (category).
3. Прямую официальную ссылку на страницу управления подпиской или инструкции по отмене (unsubscribe_url) — БЕРИ ТОЛЬКО ИЗ ПЕРЕДАННЫХ ССЫЛОК ПОИСКА, не выдумывай URL! Если подходящей ссылки нет, верни null.
4. 2–3 конкретных коротких шага (steps), как пользователю отключить автопродление.

Отвечай СТРОГО в формате валидного JSON без markdown-оберток:
{
  "service_name": "...",
  "category": "...",
  "unsubscribe_url": "https://..." или null,
  "steps": ["Шаг 1...", "Шаг 2..."]
}"""


def _search_web_for_unsubscribe(merchant_clean: str) -> list[dict[str, str]]:
    """Выполняет целевой поиск в интернете по отмене подписки для конкретного мерчанта."""
    query = f"{merchant_clean} как отменить подписку личный кабинет"
    results: list[dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, region="ru-ru", max_results=4):
                results.append(
                    {
                        "title": str(item.get("title", ""))[:120],
                        "href": str(item.get("href", "")),
                        "body": str(item.get("body", ""))[:260],
                    }
                )
    except Exception as e:
        logger.warning("Ошибка веб-поиска DDGS для '%s': %s", merchant_clean, e)
    return results


def _extract_json_block(text: str) -> dict[str, Any]:
    """Безопасно извлекает JSON из ответа LLM."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("JSON-объект не найден в ответе LLM")
    return json.loads(match.group(0))


class UnsubAgentRunner:
    """Оркестратор графа LangGraph для поиска инструкций и ссылок на отмену подписок."""

    def __init__(self, cache: MerchantCache | None = None) -> None:
        self.cache = cache if cache is not None else MerchantCache()
        self.graph = self._build_graph()

    def _node_check_cache(self, state: AgentState) -> AgentState:
        cached = self.cache.get(state["merchant_clean"])
        if cached is not None:
            cached["source"] = "cache"
            return {"guidance": cached, "from_cache": True}
        return {"from_cache": False}

    def _node_web_search(self, state: AgentState) -> AgentState:
        results = _search_web_for_unsubscribe(state["merchant_clean"])
        return {"search_results": results}

    def _node_synthesize_gigachat(self, state: AgentState) -> AgentState:
        merchant = state["merchant_clean"]
        search_results = state.get("search_results", [])
        first_hit = next(iter(search_results), {})
        top_url = first_hit.get("href") or None

        # Graceful fallback, если ключ GigaChat не задан в .env
        if not settings.gigachat_credentials.strip():
            fallback_guidance = SubscriptionGuidance(
                merchant_clean=merchant,
                service_name=merchant.title(),
                category="Цифровые сервисы и подписки",
                unsubscribe_url=top_url,
                steps=[
                    f"Перейдите в личный кабинет сервиса {merchant} по ссылке или через поиск.",
                    "Откройте раздел «Подписки», «Оплата» или «Управление тарифом».",
                    "Отключите автопродление или отвяжите банковскую карту.",
                ],
                source="web_fallback",
            ).model_dump()
            self.cache.set(merchant, fallback_guidance)
            return {"guidance": fallback_guidance}

        user_prompt = (
            f"Мерчант в банковской выписке: {merchant}\n"
            f"MCC-код: {state.get('mcc_code', '5815')}, сумма списания: {state.get('amount', 0)} руб.\n\n"
            f"Результаты веб-поиска:\n{json.dumps(search_results, ensure_ascii=False, indent=2)}"
        )

        try:
            with GigaChat(
                credentials=settings.gigachat_credentials,
                scope=settings.gigachat_scope,
                model=settings.gigachat_model,
                verify_ssl_certs=settings.gigachat_verify_ssl,
            ) as giga:
                response = giga.chat(
                    Chat(
                        messages=[
                            Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT),
                            Messages(role=MessagesRole.USER, content=user_prompt),
                        ],
                        temperature=0.1,
                    )
                )
                first_choice = next(iter(response.choices))
                raw_content = first_choice.message.content
                parsed = _extract_json_block(raw_content)

                allowed_urls = {r["href"] for r in search_results if r.get("href")}
                candidate_url = parsed.get("unsubscribe_url")
                safe_url = candidate_url if candidate_url in allowed_urls else top_url

                guidance = SubscriptionGuidance(
                    merchant_clean=merchant,
                    service_name=str(parsed.get("service_name") or merchant.title()),
                    category=str(parsed.get("category") or "Цифровые сервисы"),
                    unsubscribe_url=safe_url,
                    steps=[str(s) for s in parsed.get("steps", [])][:4]
                    or ["Откройте личный кабинет сервиса и отключите автопродление в разделе «Подписки»."],
                    source="gigachat+web",
                ).model_dump()

        except Exception as e:
            logger.warning("Ошибка вызова GigaChat для '%s' (%s) — отдаем web_fallback.", merchant, e)
            guidance = SubscriptionGuidance(
                merchant_clean=merchant,
                service_name=merchant.title(),
                category="Цифровые сервисы и подписки",
                unsubscribe_url=top_url,
                steps=[
                    f"Авторизуйтесь в личном кабинете сервиса {merchant}.",
                    "Найдите раздел «Управление подпиской» и нажмите «Отменить автопродление».",
                ],
                source="web_fallback",
            ).model_dump()

        self.cache.set(merchant, guidance)
        return {"guidance": guidance}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("check_cache", self._node_check_cache)
        workflow.add_node("web_search", self._node_web_search)
        workflow.add_node("synthesize", self._node_synthesize_gigachat)

        workflow.set_entry_point("check_cache")
        workflow.add_conditional_edges(
            "check_cache",
            lambda state: "done" if state.get("from_cache") else "search",
            {"done": END, "search": "web_search"},
        )
        workflow.add_edge("web_search", "synthesize")
        workflow.add_edge("synthesize", END)
        return workflow.compile()

    def resolve_subscription(
        self, merchant_clean: str, mcc_code: str = "5815", amount: float = 0.0
    ) -> SubscriptionGuidance:
        """Запускает граф агента для одного мерчанта и возвращает структурированную карточку."""
        final_state = self.graph.invoke(
            {
                "merchant_clean": merchant_clean,
                "mcc_code": mcc_code,
                "amount": amount,
            }
        )
        return SubscriptionGuidance(**final_state["guidance"])
