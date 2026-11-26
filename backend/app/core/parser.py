from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend.app.core.pii_masker import HybridPIIMasker, MaskingStats, P2P_MCC_CODES

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: frozenset[str] = frozenset({"Дата операции", "Сумма операции", "Описание"})

# Регулярка для извлечения мерчанта из стандартной строки эквайринга Альфа-Банка:
# Пример: "548673++++++1234    482910\RUS\MOSCOW\YANDEX*PLUS                  15.10.2026 299.00 RUR MCC5815"
ALFA_ACQUIRING_RE: re.Pattern[str] = re.compile(
    r"\\(?:RUS|USA|IRL|NLD|LUX|GBR|CYP|DEU|ARE|[A-Z]{2,3})\\[^\\]+\\(.+?)\s+\d{2}\.\d{2}\.\d{4}\s+\d+(?:[.,]\d{2})?\s+[A-Z]{3}",
    re.IGNORECASE,
)
TAIL_NOISE_RE: re.Pattern[str] = re.compile(
    r"\s+\d{2}\.\d{2}\.\d{4}.*$|\s+MCC\d{4}.*$",
    re.IGNORECASE,
)


@dataclass
class ParsedStatementResult:
    df: pd.DataFrame
    masking_stats: MaskingStats
    total_rows: int
    expense_rows: int


def extract_clean_merchant(description: str, is_sbp: bool = False) -> str:
    """Извлекает очищенный идентификатор мерчанта из сырой строки выписки Альфа-Банка."""
    raw = str(description).strip()
    if not raw:
        return "UNKNOWN_MERCHANT"

    if is_sbp or raw.lower().startswith("перевод по сбп"):
        # Нормализуем СБП в единый токен получателя (без даты в конце)
        cleaned_sbp = re.sub(r"^Перевод по СБП:\s*", "", raw, flags=re.IGNORECASE)
        cleaned_sbp = re.sub(r"\s+от\s+\d{2}\.\d{2}\.\d{4}.*$", "", cleaned_sbp, flags=re.IGNORECASE)
        return cleaned_sbp.strip().upper() or "SBP_TRANSFER"

    match = ALFA_ACQUIRING_RE.search(raw)
    if match:
        merchant = match.group(1).strip()
    else:
        # Fallback для нестандартных строк: убираем ведущую маску карты и хвост с датой/суммой/MCC
        merchant = re.sub(r"^\d{6}[+*]{4,6}\d{4}\s+\d+\s*", "", raw)
        merchant = TAIL_NOISE_RE.sub("", merchant).strip()

    # Схлопываем множественные пробелы
    merchant = re.sub(r"\s+", " ", merchant).strip().upper()
    return merchant if len(merchant) >= 2 else raw[:28].upper()


def _decode_csv_bytes(raw_bytes: bytes) -> str:
    """Пробует декодировать CSV в стандартных кодировках российских банков (UTF-8-SIG, CP1251)."""
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Не удалось определить кодировку файла выписки (поддерживаются UTF-8 и Windows-1251).")


def _detect_delimiter(sample_text: str) -> str:
    """Определяет разделитель колонок (; или ,)."""
    first_line = sample_text.splitlines()[0] if sample_text.splitlines() else ""
    try:
        dialect = csv.Sniffer().sniff(first_line, delimiters=";,|\t")
        return dialect.delimiter
    except csv.Error:
        return ";" if first_line.count(";") >= first_line.count(",") else ","


def parse_alfa_statement(
    source: Path | bytes | io.BytesIO,
    masker: HybridPIIMasker | None = None,
) -> ParsedStatementResult:
    """Читает CSV-выписку Альфа-Банка, нормализует типы и выполняет PII-санитизацию."""
    if isinstance(source, Path):
        raw_bytes = source.read_bytes()
    elif isinstance(source, io.BytesIO):
        raw_bytes = source.getvalue()
    else:
        raw_bytes = source

    if not raw_bytes:
        raise ValueError("Передан пустой файл выписки.")

    text = _decode_csv_bytes(raw_bytes)
    delimiter = _detect_delimiter(text)

    df = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"В выписке отсутствуют обязательные колонки: {sorted(missing_cols)}")

    total_rows = len(df)
    if total_rows == 0:
        raise ValueError("Файл выписки не содержит транзакций.")

    # Фильтруем только выполненные операции (если колонка Статус присутствует)
    if "Статус" in df.columns:
        df = df[df["Статус"].fillna("Выполнено").str.strip().str.lower().isin({"выполнено", "executed", "ok"})].copy()

    # Парсим дату операции
    df["dt"] = pd.to_datetime(df["Дата операции"].str.strip(), dayfirst=True, errors="coerce")
    df = df.dropna(subset=["dt"]).copy()

    # Парсим сумму (поддержка запятой как десятичного разделителя и пробелов в числах)
    clean_amount_str = (
        df["Сумма операции"]
        .astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["amount_raw"] = pd.to_numeric(clean_amount_str, errors="coerce")
    df = df.dropna(subset=["amount_raw"]).copy()

    # Оставляем только расходные операции (< 0) и переводим в положительный модуль суммы
    df = df[df["amount_raw"] < 0].copy()
    df["amount"] = df["amount_raw"].abs()

    if df.empty:
        raise ValueError("В выписке не найдено расходных операций для анализа.")

    # Нормализуем MCC (если колонки нет, пытаемся вытащить из описания)
    if "MCC" in df.columns:
        df["MCC"] = df["MCC"].fillna("0000").astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(4)
    else:
        extracted_mcc = df["Описание"].astype(str).str.extract(r"MCC(\d{4})", expand=False)
        df["MCC"] = extracted_mcc.fillna("0000")

    # Определяем P2P / СБП операции
    desc_series = df["Описание"].fillna("").astype(str)
    df["is_sbp"] = (
        df["MCC"].isin(P2P_MCC_CODES)

        | desc_series.str.contains(r"СБП|ПЕРЕВОД|SBP", case=False, regex=True)
    )

    # Извлекаем чистое имя мерчанта ДО маскировки (для корректной группировки)
    df["merchant_clean"] = [
        extract_clean_merchant(desc, is_sbp=bool(sbp))
        for desc, sbp in zip(desc_series, df["is_sbp"])
    ]

    # Выполняем PII-маскировку описаний и названий СБП-получателей
    active_masker = masker if masker is not None else HybridPIIMasker(use_ner=True)
    total_stats = MaskingStats()
    sanitized_descriptions: list[str] = []
    sanitized_merchants: list[str] = []

    for desc, merch, mcc, is_sbp in zip(desc_series, df["merchant_clean"], df["MCC"], df["is_sbp"]):
        clean_desc, st1 = active_masker.sanitize_text(desc, mcc=mcc, is_p2p=bool(is_sbp))
        total_stats.cards_masked += st1.cards_masked
        total_stats.accounts_masked += st1.accounts_masked
        total_stats.phones_masked += st1.phones_masked
        total_stats.persons_masked += st1.persons_masked
        sanitized_descriptions.append(clean_desc)

        if is_sbp:
            clean_merch, _ = active_masker.sanitize_text(merch, mcc=mcc, is_p2p=True)
            sanitized_merchants.append(clean_merch)
        else:
            sanitized_merchants.append(merch)

    df["Описание_masked"] = sanitized_descriptions
    df["merchant_clean"] = sanitized_merchants

    df = df.sort_values("dt").reset_index(drop=True)
    return ParsedStatementResult(
        df=df,
        masking_stats=total_stats,
        total_rows=total_rows,
        expense_rows=len(df),
    )
