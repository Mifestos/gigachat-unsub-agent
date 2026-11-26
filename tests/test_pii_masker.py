from __future__ import annotations

from backend.app.core.pii_masker import HybridPIIMasker, is_valid_luhn


def test_luhn_validation() -> None:
    """Проверяет математическую валидацию номеров карт по алгоритму Луна."""
    assert is_valid_luhn("4111111111111111") is True
    assert is_valid_luhn("4111111111111112") is False
    assert is_valid_luhn("1234") is False
    assert is_valid_luhn("abcd1234efgh5678") is False


def test_hybrid_masker_sbp_vs_acquiring() -> None:
    """Проверяет контекстную маскировку: в СБП вырезаются ФИО и телефон, в подписке сохраняется мерчант."""
    masker = HybridPIIMasker(use_ner=True)

    # 1. СБП перевод с латинскими инициалами
    sbp_raw = "Перевод по СБП: СБП +7916***4512 IVANOV A.V. от 15.10.2026 без НДС"
    sbp_clean, stats_sbp = masker.sanitize_text(sbp_raw, mcc="4829", is_p2p=True)
    assert "<PHONE>" in sbp_clean
    assert "<PERSON>" in sbp_clean
    assert "IVANOV" not in sbp_clean
    assert stats_sbp.phones_masked == 1
    assert stats_sbp.persons_masked >= 1

    # 2. Кириллическое ФИО в переводе
    ru_p2p_raw = "Перевод клиенту Сбербанка: Александр Сергеевич Пушкин за аренду"
    ru_p2p_clean, stats_ru = masker.sanitize_text(ru_p2p_raw, mcc="6538", is_p2p=True)
    assert "<PERSON>" in ru_p2p_clean
    assert "Пушкин" not in ru_p2p_clean
    assert stats_ru.persons_masked >= 1

    # 3. Карточный эквайринг подписки -> затирается карта, но сохраняется имя мерчанта для веб-поиска
    sub_raw = "548673++++++1234    482910\\RUS\\MOSCOW\\CLOUDPAYMENTS*IP IVANOV     15.10.2026 990.00 RUR MCC8299"
    sub_clean, stats_sub = masker.sanitize_text(sub_raw, mcc="8299", is_p2p=False)
    assert "<CARD>" in sub_clean
    assert "CLOUDPAYMENTS*IP IVANOV" in sub_clean
    assert stats_sub.cards_masked == 1
    assert stats_sub.persons_masked == 0
