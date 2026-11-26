from __future__ import annotations

from pathlib import Path

from backend.app.agent.graph import UnsubAgentRunner
from backend.app.core.parser import parse_alfa_statement
from backend.app.ml.scorer import SubscriptionMLScorer


def test_end_to_end_scoring_and_agent() -> None:
    sample_path = Path("data/sample_alfa_statement.csv")
    parsed = parse_alfa_statement(sample_path)

    scorer = SubscriptionMLScorer()
    subs = scorer.predict_subscriptions(parsed.df)

    # В демо-выписке должно быть найдено минимум 3 подписки
    assert len(subs) >= 3
    # Проверяем, что СБП-переводы НЕ попали в список подписок
    assert all("СБП" not in s.merchant_clean for s in subs)

    # Проверяем работу LangGraph-агента + кэширование на первой найденной подписке
    agent = UnsubAgentRunner()
    top_sub = next(iter(subs))
    guidance_1 = agent.resolve_subscription(
        merchant_clean=top_sub.merchant_clean,
        mcc_code=top_sub.mcc_code,
        amount=top_sub.last_amount,
    )
    assert guidance_1.service_name
    assert len(guidance_1.steps) >= 1

    # Повторный запрос должен мгновенно вернуться из кэша
    guidance_2 = agent.resolve_subscription(merchant_clean=top_sub.merchant_clean)
    assert guidance_2.source == "cache"
