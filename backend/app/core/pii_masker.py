from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

DEFAULT_NER_MODELS: Final[tuple[str, ...]] = (
    "Babelscape/wikineural-multilingual-ner",
)


P2P_MCC_CODES: Final[frozenset[str]] = frozenset({"4829", "6536", "6537", "6538", "6012"})

MASKED_CARD_RE: Final[re.Pattern[str]] = re.compile(r"\b\d{6}[+*]{4,6}\d{4}\b")
RAW_CARD_CANDIDATE_RE: Final[re.Pattern[str]] = re.compile(r"\b(?:\d[ -]?){15,18}\d\b")
RU_ACCOUNT_RE: Final[re.Pattern[str]] = re.compile(r"\b(?:408|407|406|423|426)\d{17}\b")
PHONE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:\+7|8)[\s\-()]*(?:\d{3}|\d{2}\*{1,3})[\s\-()]*(?:\d{3}|\*{3})[\s\-()]*(?:\d{2}|\*{2})[\s\-()]*(?:\d{2}|\*{2})"
)
# Покрывает: "NOVIKOV S.S.", "IVANOV A.V.", "Иванов И.И.", "Александр Сергеевич Пушкин"
STRUCTURED_FIO_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![А-ЯЁA-Za-z0-9_])(?:"
    r"[А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z]{1,20}\s+[А-ЯЁA-Z]\.\s*[А-ЯЁA-Z]\.|"
    r"[А-ЯЁ][а-яё]{1,20}\s+[А-ЯЁ][а-яё]{1,20}(?:вич|вна|ич|на|вича|вну|ича|ну|вичу|вне)\s+[А-ЯЁ][а-яё]{1,20}"
    r")(?=\s|$|[,;!?])"
)


def is_valid_luhn(digits_only: str) -> bool:
    """Проверяет контрольную сумму номера банковской карты по алгоритму Луна."""
    if not digits_only.isdigit() or not (13 <= len(digits_only) <= 19):
        return False
    total = 0
    for idx, ch in enumerate(digits_only[::-1]):
        n = int(ch)
        if idx % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


@dataclass
class MaskingStats:
    cards_masked: int = 0
    accounts_masked: int = 0
    phones_masked: int = 0
    persons_masked: int = 0



@lru_cache(maxsize=2)
def _load_shared_ner_pipeline(model_names: tuple[str, ...]):
    """Загружает NER-модель ровно один раз на весь процесс (Singleton в памяти)."""
    device = 0 if torch.cuda.is_available() else -1
    for model_name in model_names:
        try:
            logger.info("Загрузка RuBERT-NER (%s) на device=%s...", model_name, device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            ner = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=device,
            )
            logger.info("RuBERT-NER (%s) успешно загружен на device=%s.", model_name, device)
            return ner
        except Exception as e:
            logger.warning("Модель %s недоступна (%s).", model_name, e)

    logger.warning("Нейросетевой NER недоступен — используем быстрый детерминированный слой ФИО.")
    return None


class HybridPIIMasker:
    """Двухступенчатый PII-санитайзер уровня финтеха с LRU-кэшированием:
    1. Deterministic Layer: Regex + проверка карт по алгоритму Луна + структурные ФИО.
    2. Neural NER Layer: RuBERT-NER для извлечения нешаблонных имен в P2P/СБП-переводах.
    """

    def __init__(self, model_names: tuple[str, ...] = DEFAULT_NER_MODELS, use_ner: bool = True) -> None:
        self.model_names = model_names
        self.use_ner = use_ner
        self._ner_pipeline = _load_shared_ner_pipeline(model_names) if use_ner else None
        self._cached_sanitize = lru_cache(maxsize=4096)(self._sanitize_uncached)

    def _mask_card_candidates(self, text: str) -> tuple[str, int]:
        cards_count = 0

        def _replace_raw_card(match: re.Match[str]) -> str:
            nonlocal cards_count
            raw = match.group(0)
            digits = re.sub(r"\D", "", raw)
            if is_valid_luhn(digits):
                cards_count += 1
                return "<CARD>"
            return raw

        text, n_masked = MASKED_CARD_RE.subn("<CARD>", text)
        cards_count += n_masked
        text = RAW_CARD_CANDIDATE_RE.sub(_replace_raw_card, text)
        return text, cards_count

    def _mask_persons(self, text: str) -> tuple[str, int]:
        persons_count = 0
        if self._ner_pipeline is not None:
            entities = self._ner_pipeline(text)
            person_spans = [
                (ent["start"], ent["end"])
                for ent in entities
                if "PER" in str(ent.get("entity_group", "")).upper() and ent.get("score", 0.0) >= 0.65
            ]
            for start, end in sorted(person_spans, key=lambda x: x, reverse=True):
                text = text[:start] + "<PERSON>" + text[end:]
                persons_count += 1

        text, extra_count = STRUCTURED_FIO_RE.subn("<PERSON>", text)
        persons_count += extra_count
        return text, persons_count

    def _sanitize_uncached(self, text: str, mcc: str | None = None, is_p2p: bool = False) -> tuple[str, MaskingStats]:
        if not text:
            return text, MaskingStats()

        cleaned, cards_cnt = self._mask_card_candidates(text)
        cleaned, acc_cnt = RU_ACCOUNT_RE.subn("<ACCOUNT>", cleaned)
        cleaned, phones_cnt = PHONE_RE.subn("<PHONE>", cleaned)

        persons_cnt = 0
        p2p_context = (
            is_p2p
            or (mcc is not None and str(mcc).strip().zfill(4) in P2P_MCC_CODES)
            or ("СБП" in cleaned.upper())
            or ("ПЕРЕВОД" in cleaned.upper())
        )
        if p2p_context:
            cleaned, persons_cnt = self._mask_persons(cleaned)

        return cleaned, MaskingStats(
            cards_masked=cards_cnt,
            accounts_masked=acc_cnt,
            phones_masked=phones_cnt,
            persons_masked=persons_cnt,
        )

    def sanitize_text(self, text: str, mcc: str | None = None, is_p2p: bool = False) -> tuple[str, MaskingStats]:
        return self._cached_sanitize(text, mcc, is_p2p)
