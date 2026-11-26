"""
Сборка синтетического датасета банковских транзакций в формате выписки Альфа-Банка.

Источники:
- MCC-справочник (RU): открытый GitHub-репозиторий;
- Пул мерчантов: открытый HF-датасет (без токена);
- Синтетика: клиенты, подписки, СБП, hard-negatives.

Выход:
- data/train_dataset.parquet — обучающий датасет;
- data/metadata.json         — параметры запуска и метрики;
- data/sample_alfa_statement.csv — демо-выписка самого "подписочного" клиента.
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import json
import logging
import urllib.request
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Константы
# =============================================================================

MCC_RUSSIAN_URLS: Final[tuple[str, ...]] = (
    "https://raw.githubusercontent.com/Oleksios/Merchant-Category-Codes/main/With%20groups/mcc-ru.json",
)

HF_TRANSACTIONS_DATASETS: Final[tuple[str, ...]] = (
    "DoDataThings/us-bank-transaction-categories",
    "mindweave/bank-transactions-us",
)

# Цифровые MCC — те, что в реальности ассоциируются с подписками.
DIGITAL_MCC_CODES: Final[frozenset[str]] = frozenset(
    {"4816", "4899", "5815", "5816", "5817", "5818", "7399", "8299", "7997", "7372", "5968"}
)
SBP_MCC: Final[str] = "4829"
FALLBACK_DEFAULT_MCC: Final[str] = "5411"
FALLBACK_DIGITAL_MCC: Final[str] = "5815"

# Окно генерации: 30 месяцев. Годовая подписка, стартовавшая
# в первые 90 дней, успевает дать 2–3 списания и пройти фильтр tx_counts >= 2.
DATE_START: Final[pd.Timestamp] = pd.Timestamp("2024-06-01")
DATE_END: Final[pd.Timestamp] = pd.Timestamp("2026-11-20")

# Циклы для ОБЫЧНЫХ подписок. 365 здесь нет — годовые идут отдельным путём.
POSSIBLE_CYCLES: Final[np.ndarray] = np.array(
    (7, 14, 14, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 90),
    dtype=int,
)
YEARLY_CYCLE_DAYS: Final[int] = 365

# Вероятностная генерация годовых подписок.
# p=0.03 → ~3% клиентов имеют одну годовую подписку → ~0.5–1% от всех подписок.
# Число годовых = binomial(n_users, p_yearly), с floor min_yearly.
P_YEARLY_PER_USER: Final[float] = 0.03
MIN_YEARLY: Final[int] = 1

POSSIBLE_PRICES: Final[np.ndarray] = np.array(
    [149.0, 199.0, 299.0, 399.0, 490.0, 690.0, 990.0, 1490.0], dtype=float
)
YEARLY_PRICES: Final[tuple[float, ...]] = (1990.0, 2990.0, 3990.0, 4990.0)
COFFEE_PRICES: Final[tuple[float, ...]] = (190.0, 250.0, 300.0)
TRANSIT_PRICES: Final[tuple[float, ...]] = (500.0, 650.0)
SBP_AMOUNTS: Final[tuple[float, ...]] = (1500.0, 3000.0, 5000.0, 12000.0)

SBP_NAMES: Final[tuple[str, ...]] = (
    "IVANOV A.V.", "SMIRNOVA E.S.", "KUZNETSOV D.M.", "POPOVA O.I.",
    "SOKOLOV I.P.", "LEBEDEV N.A.", "MOROZOVA K.V.", "NOVIKOV S.S.",
    "VOLKOVA M.Y.", "PETROV D.K.",
)

DIGITAL_CATEGORY_REGEX: Final[str] = (
    r"sub|stream|digital|software|utility|telecom|fitness|edu|cloud|media|service"
)

FALLBACK_MCC_RECORDS: Final[tuple[dict[str, str], ...]] = (
    {"MCC": "4816", "Категория": "Компьютерные сети и информационные услуги"},
    {"MCC": "4899", "Категория": "Кабельные и платные ТВ-услуги"},
    {"MCC": "5815", "Категория": "Цифровые товары: медиа, книги, фильмы, музыка"},
    {"MCC": "5816", "Категория": "Цифровые товары: игры"},
    {"MCC": "5817", "Категория": "Цифровые товары: приложения"},
    {"MCC": "5818", "Категория": "Цифровые товары: мультисервисные подписки"},
    {"MCC": "5968", "Категория": "Подписки и периодические издания"},
    {"MCC": "7372", "Категория": "Разработка ПО и обработка данных"},
    {"MCC": "7399", "Категория": "Бизнес-сервисы и онлайн-платформы"},
    {"MCC": "8299", "Категория": "Образовательные услуги и онлайн-курсы"},
    {"MCC": "7997", "Категория": "Клубы, спорт и фитнес-подписки"},
    {"MCC": "4829", "Категория": "Денежные переводы"},
    {"MCC": "5411", "Категория": "Супермаркеты и продуктовые магазины"},
    {"MCC": "5812", "Категория": "Рестораны и кафе"},
    {"MCC": "5814", "Категория": "Фастфуд и кофейни"},
    {"MCC": "4121", "Категория": "Такси и пассажирские перевозки"},
    {"MCC": "4131", "Категория": "Автобусные перевозки"},
    {"MCC": "4111", "Категория": "Городской транспорт"},
    {"MCC": "5399", "Категория": "Маркетплейсы и универмаги"},
    {"MCC": "5311", "Категория": "Универмаги"},
    {"MCC": "5912", "Категория": "Аптеки"},
    {"MCC": "5541", "Категория": "Автозаправочные станции"},
    {"MCC": "5691", "Категория": "Одежда и обувь"},
    {"MCC": "5651", "Категория": "Семейная одежда"},
    {"MCC": "5732", "Категория": "Электроника"},
    {"MCC": "5734", "Категория": "Магазины ПО"},
    {"MCC": "5942", "Категория": "Книжные магазины"},
    {"MCC": "5945", "Категория": "Хобби и игрушки"},
    {"MCC": "5977", "Категория": "Косметика и парфюмерия"},
    {"MCC": "7011", "Категория": "Гостиницы и отели"},
    {"MCC": "7512", "Категория": "Аренда автомобилей"},
    {"MCC": "7832", "Категория": "Кинотеатры"},
    {"MCC": "7996", "Категория": "Парки развлечений"},
    {"MCC": "8099", "Категория": "Медицинские услуги"},
    {"MCC": "5999", "Категория": "Прочая розничная торговля"},
)


# =============================================================================
# Справочник MCC
# =============================================================================

def _download_json_safely(url: str, target_path: Path, timeout: int = 15) -> object:
    """Скачивает JSON с User-Agent, атомарной записью и очисткой .tmp в finally."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; DatasetBuilder/1.0)"},
    )
    tmp_path = target_path.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8")
        data = json.loads(content)
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(target_path)
        return data
    finally:
        tmp_path.unlink(missing_ok=True)


def _parse_mcc_payload(raw_mcc: object) -> list[dict[str, str]]:
    """Парсит MCC JSON: список словарей либо dict по кодам."""
    records: list[dict[str, str]] = []
    if isinstance(raw_mcc, list):
        for item in raw_mcc:
            if not isinstance(item, dict):
                continue
            mcc_val = item.get("mcc") or item.get("code")
            if not mcc_val:
                continue
            mcc_code = str(mcc_val).strip().zfill(4)
            title = (
                item.get("shortDescription")
                or item.get("title")
                or item.get("name")
                or item.get("fullDescription")
                or "Операции по карте"
            )
            records.append({"MCC": mcc_code, "Категория": str(title)[:45]})
    elif isinstance(raw_mcc, dict):
        for code, val in raw_mcc.items():
            if code is None:
                continue
            mcc_code = str(code).strip().zfill(4)
            if isinstance(val, dict):
                title = val.get("title") or val.get("name") or "Операции по карте"
            else:
                title = str(val) if val else "Операции по карте"
            records.append({"MCC": mcc_code, "Категория": str(title)[:45]})
    return records


def load_russian_mcc_catalog(cache_path: Path) -> pd.DataFrame:
    """Загружает справочник MCC: кэш → сеть → встроенный fallback."""
    raw_mcc: object | None = None

    if cache_path.exists():
        try:
            raw_mcc = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Кэш MCC %s поврежден (%s), перекачиваем...", cache_path, e)
            cache_path.unlink(missing_ok=True)

    if raw_mcc is None:
        for url in MCC_RUSSIAN_URLS:
            try:
                logger.info("Скачиваем справочник MCC из %s ...", url)
                raw_mcc = _download_json_safely(url, cache_path)
                break
            except Exception as e:
                logger.warning("Источник %s недоступен (%s), пробуем следующий...", url, e)

    records = _parse_mcc_payload(raw_mcc) if raw_mcc is not None else []
    if not records:
        logger.warning("Все источники MCC недоступны — используем встроенный справочник.")
        records = list(FALLBACK_MCC_RECORDS)

    df_mcc = pd.DataFrame(records).drop_duplicates(subset=["MCC"]).reset_index(drop=True)
    if SBP_MCC not in set(df_mcc["MCC"]):
        df_mcc = pd.concat(
            [df_mcc, pd.DataFrame([{"MCC": SBP_MCC, "Категория": "Денежные переводы"}])],
            ignore_index=True,
        )
    logger.info("Справочник MCC готов: %d уникальных кодов.", len(df_mcc))
    return df_mcc


# =============================================================================
# Пул мерчантов из HF
# =============================================================================

def fetch_open_merchants_pool(
    sample_size: int = 30_000,
) -> tuple[list[str], list[str]]:
    """Возвращает два непересекающихся пула: сервисные и розничные мерчанты."""
    df = pd.DataFrame()
    last_error: Exception | None = None

    for ds_name in HF_TRANSACTIONS_DATASETS:
        try:
            logger.info("Потоковая загрузка до %d записей из %s ...", sample_size, ds_name)
            ds = load_dataset(ds_name, split="train", streaming=True)
            df = pd.DataFrame(list(ds.take(sample_size)))
            if not df.empty:
                logger.info("Успешно загружено %d строк из %s.", len(df), ds_name)
                break
            logger.warning("Датасет %s вернул 0 строк, пробуем следующий...", ds_name)
        except Exception as e:
            last_error = e
            logger.warning("Датасет %s недоступен (%s), пробуем следующий...", ds_name, e)

    if df.empty:
        raise RuntimeError(
            f"Ни один из открытых HF-датасетов {HF_TRANSACTIONS_DATASETS} не вернул записей."
            + (f" Последняя ошибка: {last_error}" if last_error else "")
        )

    desc_col = next(
        (c for c in ("description", "transaction_description", "merchant", "text") if c in df.columns),
        None,
    )
    if desc_col is None:
        desc_col = next(
            (c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object"),
            df.columns[0],
        )

    cat_col = next((c for c in ("category", "Category", "label") if c in df.columns), None)

    df["clean_merch"] = (
        df[desc_col]
        .astype(str)
        .str.upper()
        .str.replace(r"[^\w*._ -]", "", regex=True)
        .str.strip()
        .str.slice(0, 26)
    )
    df = (
        df[df["clean_merch"].str.len() >= 4]
        .drop_duplicates(subset=["clean_merch"])
        .reset_index(drop=True)
    )

    if cat_col is not None:
        sub_mask = (
            df[cat_col]
            .astype(str)
            .str.lower()
            .str.contains(DIGITAL_CATEGORY_REGEX, regex=True)
        )
        sub_merchants = df.loc[sub_mask, "clean_merch"].tolist()
        reg_merchants = df.loc[~sub_mask, "clean_merch"].tolist()
    else:
        split_idx = max(1, int(len(df) * 0.25))
        sub_merchants = df["clean_merch"].iloc[:split_idx].tolist()
        reg_merchants = df["clean_merch"].iloc[split_idx:].tolist()

    sub_set = set(sub_merchants)
    reg_merchants = [m for m in reg_merchants if m not in sub_set]

    if len(sub_merchants) < 100 and len(reg_merchants) > 200:
        transfer = min(200, len(reg_merchants) // 3)
        sub_merchants.extend(reg_merchants[:transfer])
        reg_merchants = reg_merchants[transfer:]

    if not sub_merchants or not reg_merchants:
        raise RuntimeError("Недостаточно уникальных мерчантов для формирования пулов.")

    logger.info(
        "Пулы мерчантов: %d сервисных / %d розничных.",
        len(sub_merchants),
        len(reg_merchants),
    )
    return sub_merchants, reg_merchants


# =============================================================================
# Сэмплирование времени
# =============================================================================

def _build_allowed_intervals(
    total_span_sec: int, vacations: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Строит интервалы (start, end), исключая отпускные окна. Всегда непустой."""
    intervals: list[tuple[int, int]] = []
    prev_end = 0
    for vs, vl in sorted(vacations):
        if vs > prev_end:
            intervals.append((prev_end, vs))
        prev_end = max(prev_end, vs + vl)
    if prev_end < total_span_sec:
        intervals.append((prev_end, total_span_sec))
    return intervals or [(0, total_span_sec)]


def _sample_from_intervals(
    rng: np.random.Generator, intervals: list[tuple[int, int]]
) -> int:
    """Равномерно (взвешенно по длине) сэмплирует секунду из объединения интервалов."""
    lengths = np.array([e - s for s, e in intervals], dtype=np.int64)
    r = int(rng.integers(0, int(lengths.sum())))
    cum = 0
    for (s, _), ln in zip(intervals, lengths):
        if r < cum + ln:
            return s + (r - cum)
        cum += int(ln)
    return intervals[-1][0]


# =============================================================================
# Генерация истории одного клиента
# =============================================================================

def _generate_single_user_rows(
    user_id: str,
    sub_merchants: list[str],
    reg_merchants: list[str],
    merchant_to_mcc: dict[str, str],
    rng: np.random.Generator,
    yearly_merchant: str | None = None,
) -> list[dict[str, Any]]:
    """Генерирует полную историю одного клиента.

    Args:
        yearly_merchant: если задан — добавляется ровно одна годовая подписка
                         на этого мерчанта. Решение принимает build_realistic_statements
                         через binomial(n_users, p_yearly).
    """
    total_span_sec = int((DATE_END - DATE_START).total_seconds())
    card_suffix = f"{int(rng.integers(0, 10000)):04d}"
    card_mask = f"548673******{card_suffix}"
    rows: list[dict[str, Any]] = []

    # --- Отпускные окна: 1–2 периода по 7–21 дню ---
    vacations: list[tuple[int, int]] = []
    max_vs = max(1, total_span_sec - 21 * 86400)
    for _ in range(int(rng.integers(1, 3))):
        vs = int(rng.integers(0, max_vs))
        vl = int(rng.integers(7, 22)) * 86400
        vacations.append((vs, vl))
    allowed_intervals = _build_allowed_intervals(total_span_sec, vacations)

    # === 1. Регулярные покупки (hard negatives) ===
    n_shops = min(22, len(reg_merchants))
    user_shops = rng.choice(reg_merchants, size=n_shops, replace=False)
    shop_preferred_hour = rng.integers(8, 22, size=n_shops)

    fixed_price_indices = {0, 1} if n_shops >= 2 else set()
    fixed_prices: dict[int, float] = {}
    if 0 in fixed_price_indices:
        fixed_prices[0] = float(rng.choice(COFFEE_PRICES))
    if 1 in fixed_price_indices:
        fixed_prices[1] = float(rng.choice(TRANSIT_PRICES))

    n_reg_tx = int(rng.integers(220, 360))
    shop_indices = rng.integers(0, n_shops, size=n_reg_tx)

    for i in range(n_reg_tx):
        s_idx = int(shop_indices[i])
        merch = str(user_shops[s_idx])

        day_offset_sec = _sample_from_intervals(rng, allowed_intervals)
        base_date = (DATE_START + pd.Timedelta(seconds=day_offset_sec)).normalize()

        # 75% — «любимый час» ± 2ч, 25% — вне расписания.
        # Это гарантирует regular.std_hour_of_day > subscription.std_hour_of_day,
        # чтобы модель не выучила ложное правило «std_hour ⇒ подписка».
        if rng.random() < 0.25:
            hour = int(rng.integers(8, 22))
        else:
            hour = int(np.clip(int(shop_preferred_hour[s_idx]) + int(rng.integers(-2, 3)), 0, 23))

        dt = base_date + pd.Timedelta(
            hours=hour,
            minutes=int(rng.integers(0, 60)),
            seconds=int(rng.integers(0, 60)),
        )

        amount = (
            fixed_prices[s_idx]
            if s_idx in fixed_price_indices
            else round(float(rng.lognormal(mean=6.7, sigma=0.85)), 2)
        )
        rows.append(
            {
                "user_id": user_id,
                "dt": dt,
                "merchant_raw": merch,
                "MCC": merchant_to_mcc.get(merch, FALLBACK_DEFAULT_MCC),
                "Сумма операции": -abs(amount),
                "is_subscription": 0,
                "card_mask": card_mask,
                "is_sbp": False,
            }
        )

    # === 2. Переводы СБП с ФИО (hard negatives) ===
    for _ in range(2):
        fio = str(rng.choice(SBP_NAMES))
        phone_mask = f"+79{int(rng.integers(10, 100))}***{int(rng.integers(1000, 10000))}"
        sbp_merch = f"СБП {phone_mask} {fio}"
        sbp_base_amount = float(rng.choice(SBP_AMOUNTS))

        is_fixed_rent = rng.random() < 0.60
        sbp_cycle = int(rng.choice((14, 30)))

        scheduled_dt = DATE_START + pd.Timedelta(
            days=int(rng.integers(1, 15)),
            hours=int(rng.integers(9, 20)),
            minutes=int(rng.integers(0, 60)),
        )
        step = 0
        while scheduled_dt <= DATE_END:
            actual_dt = scheduled_dt
            if step > 0:
                actual_dt = scheduled_dt + pd.Timedelta(
                    hours=int(rng.integers(-1, 2)),
                    minutes=int(rng.integers(-10, 11)),
                )

            amount = (
                sbp_base_amount
                if is_fixed_rent
                else round(sbp_base_amount * float(rng.uniform(0.85, 1.15)), 2)
            )
            rows.append(
                {
                    "user_id": user_id,
                    "dt": actual_dt,
                    "merchant_raw": sbp_merch,
                    "MCC": SBP_MCC,
                    "Сумма операции": -abs(amount),
                    "is_subscription": 0,
                    "card_mask": card_mask,
                    "is_sbp": True,
                }
            )
            scheduled_dt += pd.Timedelta(days=sbp_cycle + int(rng.integers(-2, 3)))
            step += 1

    # === 3. Годовая подписка (опционально, ровно одна на клиента) ===
    if yearly_merchant is not None:
        current_price = float(rng.choice(YEARLY_PRICES))
        base_hour = int(rng.integers(0, 24))
        base_minute = int(rng.integers(0, 60))

        # Старт в первые 90 дней окна → за 30 месяцев успевает 2–3 списания.
        scheduled_dt = DATE_START + pd.Timedelta(
            days=int(rng.integers(0, 90)),
            hours=base_hour,
            minutes=base_minute,
        )
        step = 0
        while scheduled_dt <= DATE_END:
            actual_dt = scheduled_dt
            if step > 0:
                actual_dt = actual_dt + pd.Timedelta(
                    hours=int(rng.integers(-1, 2)),
                    minutes=int(rng.integers(-15, 16)),
                )
            rows.append(
                {
                    "user_id": user_id,
                    "dt": actual_dt,
                    "merchant_raw": yearly_merchant,
                    "MCC": merchant_to_mcc.get(yearly_merchant, FALLBACK_DIGITAL_MCC),
                    "Сумма операции": -abs(current_price),
                    "is_subscription": 1,
                    "card_mask": card_mask,
                    "is_sbp": False,
                }
            )
            scheduled_dt += pd.Timedelta(days=YEARLY_CYCLE_DAYS + int(rng.integers(-3, 4)))
            step += 1

    # === 4. Обычные подписки (7–90 дней) ===
    n_subs = min(int(rng.integers(3, 7)), len(sub_merchants))
    candidates = [m for m in sub_merchants if m != yearly_merchant]
    user_subs = (
        rng.choice(candidates, size=min(n_subs, len(candidates)), replace=False)
        if candidates
        else []
    )

    for merch_obj in user_subs:
        merch = str(merch_obj)
        cycle_days = int(rng.choice(POSSIBLE_CYCLES))
        current_price = float(rng.choice(POSSIBLE_PRICES))
        price_increased = False

        base_hour = int(rng.integers(0, 24))
        base_minute = int(rng.integers(0, 60))
        scheduled_dt = DATE_START + pd.Timedelta(
            days=int(rng.integers(0, max(1, min(25, cycle_days)))),
            hours=base_hour,
            minutes=base_minute,
        )

        step = 0
        while scheduled_dt <= DATE_END:
            actual_dt = scheduled_dt
            skip_cycle = False

            if step > 0:
                roll = rng.random()
                if roll < 0.10:
                    # 10% — failed payment: ретрай через 2–5 дней, но не позже след. цикла.
                    retry_offset = pd.Timedelta(days=int(rng.integers(2, 6)))
                    max_retry = pd.Timedelta(days=cycle_days - 1)
                    actual_dt = scheduled_dt + min(retry_offset, max_retry)
                elif roll < 0.13:
                    # 3% — полный пропуск цикла.
                    skip_cycle = True

            if skip_cycle:
                scheduled_dt += pd.Timedelta(days=cycle_days + int(rng.integers(-1, 2)))
                step += 1
                continue

            if step > 0:
                actual_dt = actual_dt + pd.Timedelta(
                    hours=int(rng.integers(-1, 2)),
                    minutes=int(rng.integers(-15, 16)),
                )

            if step >= 1 and not price_increased and rng.random() < 0.2:
                current_price = round(current_price * 1.15, 0)
                price_increased = True

            rows.append(
                {
                    "user_id": user_id,
                    "dt": actual_dt,
                    "merchant_raw": merch,
                    "MCC": merchant_to_mcc.get(merch, FALLBACK_DIGITAL_MCC),
                    "Сумма операции": -abs(current_price),
                    "is_subscription": 1,
                    "card_mask": card_mask,
                    "is_sbp": False,
                }
            )
            scheduled_dt += pd.Timedelta(days=cycle_days + int(rng.integers(-1, 2)))
            step += 1

    return rows


# =============================================================================
# Диагностика качества
# =============================================================================

def _log_quality_report(
    df: pd.DataFrame,
    yearly_pairs: set[tuple[str, str]],
) -> None:
    """Печатает std_hour_of_day и std_delta_days по 4 сегментам.

    Сегменты:
    - subs_regular: обычные подписки (7–90 дней);
    - subs_yearly:  годовые подписки (запланированные в yearly_pairs);
    - sbp:          переводы СБП (hard negatives);
    - regular:      повседневные покупки.

    Считаем по кластерам (user_id, merchant_raw) — именно так учится CatBoost.
    Кластеры с 1 транзакцией не влияют на delta_std (NaN → skipna), но их
    количество видно в логе для прозрачности.
    """
    work = df[["user_id", "merchant_raw", "dt", "is_subscription", "is_sbp"]].copy()
    work["hour"] = work["dt"].dt.hour
    work = work.sort_values(["user_id", "merchant_raw", "dt"])
    work["delta_days"] = (
        work.groupby(["user_id", "merchant_raw"])["dt"]
        .diff()
        .dt.total_seconds()
        / 86400.0
    )

    # Размечаем годовые подписки по запланированным парам.
    yearly_set = set(yearly_pairs)
    work["is_yearly"] = [
        (u, m) in yearly_set
        for u, m in zip(work["user_id"], work["merchant_raw"])
    ]

    segments = (
        ("subs_regular", (work["is_subscription"] == 1) & (~work["is_yearly"])),
        ("subs_yearly",  (work["is_subscription"] == 1) & (work["is_yearly"])),
        ("sbp",          (work["is_subscription"] == 0) & (work["is_sbp"])),
        ("regular",      (work["is_subscription"] == 0) & (~work["is_sbp"])),
    )

    logger.info("Диагностика по сегментам (по кластерам user_id × merchant_raw):")
    for label, mask in segments:
        sub = work.loc[mask]
        if sub.empty:
            logger.info("  [%-12s] нет кластеров", label)
            continue
        grouped = sub.groupby(["user_id", "merchant_raw"])
        hour_std = grouped["hour"].std(ddof=0).mean()
        delta_std = grouped["delta_days"].std(ddof=0).mean()
        n_clusters = grouped.ngroups
        logger.info(
            "  [%-12s] clusters=%5d | std_hour=%.2f | std_delta_days=%.2f",
            label,
            n_clusters,
            float(hour_std) if pd.notna(hour_std) else 0.0,
            float(delta_std) if pd.notna(delta_std) else 0.0,
        )


# =============================================================================
# Сборка датасета
# =============================================================================

def build_realistic_statements(
    sub_merchants: list[str],
    reg_merchants: list[str],
    df_mcc: pd.DataFrame,
    n_users: int = 300,
    seed: int = 42,
    p_yearly: float = P_YEARLY_PER_USER,
    min_yearly: int = MIN_YEARLY,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Собирает датасет и возвращает (df, metadata).

    Годовые подписки генерируются вероятностно:
      n_yearly_raw = binomial(n_users, p_yearly),
      actual_n_yearly = min(max(min_yearly, n_yearly_raw), n_users, len(sub_merchants)).

    Воспроизводимость: при фиксированных (seed, n_users, p_yearly, min_yearly)
    результат детерминирован. Изменение p_yearly меняет не только ЧИСЛО годовых,
    но и КТО именно их получит — это by design, а не баг.
    """
    if len(reg_merchants) < 15:
        raise RuntimeError(f"Слишком мало розничных мерчантов: {len(reg_merchants)} (нужно ≥ 15)")
    if len(sub_merchants) < 3:
        raise RuntimeError(f"Слишком мало сервисных мерчантов: {len(sub_merchants)} (нужно ≥ 3)")

    rng = np.random.default_rng(seed)

    # --- MCC-пулы ---
    digital_mccs = df_mcc.loc[df_mcc["MCC"].isin(DIGITAL_MCC_CODES), "MCC"].tolist()
    if not digital_mccs:
        digital_mccs = df_mcc["MCC"].head(min(8, len(df_mcc))).tolist()
    retail_mccs = df_mcc.loc[~df_mcc["MCC"].isin(digital_mccs), "MCC"].tolist()
    if not retail_mccs:
        retail_mccs = df_mcc["MCC"].tolist()

    # Anti-leakage: 80% подписок → digital, 20% → retail; 85% покупок → retail, 15% → digital.
    merchant_to_mcc: dict[str, str] = {}
    for m in sub_merchants:
        pool = digital_mccs if rng.random() < 0.80 else retail_mccs
        merchant_to_mcc[m] = str(rng.choice(pool))
    for m in reg_merchants:
        if m in merchant_to_mcc:
            continue
        pool = retail_mccs if rng.random() < 0.85 else digital_mccs
        merchant_to_mcc[m] = str(rng.choice(pool))

    # --- Вероятностная генерация годовых подписок ---
    n_yearly_raw = int(rng.binomial(n_users, p_yearly))
    actual_n_yearly = min(
        max(min_yearly, n_yearly_raw),
        n_users,
        len(sub_merchants),
    )
    if p_yearly <= 0 and min_yearly <= 0:
        logger.info("Годовые подписки отключены (p_yearly=0, min_yearly=0).")
    else:
        logger.info(
            "Годовых подписок: binomial(%d, %.3f) → %d, с floor=%d → %d.",
            n_users, p_yearly, n_yearly_raw, min_yearly, actual_n_yearly,
        )

    if actual_n_yearly > 0:
        yearly_user_indices = rng.choice(n_users, size=actual_n_yearly, replace=False).tolist()
        yearly_merchants = rng.choice(sub_merchants, size=actual_n_yearly, replace=False).tolist()
        yearly_plan: dict[int, str] = {
            int(u_idx): str(m) for u_idx, m in zip(yearly_user_indices, yearly_merchants)
        }
        logger.info(
            "Запланировано годовых подписок: %d (юзеры: %s).",
            actual_n_yearly,
            sorted(yearly_plan.keys()),
        )
    else:
        yearly_plan = {}

    # --- Генерация по всем клиентам ---
    all_rows: list[dict[str, Any]] = []
    for u_idx in range(n_users):
        yearly_merchant = yearly_plan.get(u_idx)
        all_rows.extend(
            _generate_single_user_rows(
                f"client_{u_idx:04d}",
                sub_merchants,
                reg_merchants,
                merchant_to_mcc,
                rng,
                yearly_merchant=yearly_merchant,
            )
        )

    df = pd.DataFrame(all_rows)
    df["dt"] = pd.to_datetime(df["dt"])

    # --- Safety net: одиночные списания не считаем подписками ---
    tx_counts = df.groupby(["user_id", "merchant_raw"])["dt"].transform("count")
    df.loc[(df["is_subscription"] == 1) & (tx_counts < 2), "is_subscription"] = 0

    # --- Сколько годовых подписок реально выжило после фильтра ---
    yearly_pairs: set[tuple[str, str]] = set()
    yearly_in_dataset = 0
    if yearly_plan:
        for u_idx, merch in yearly_plan.items():
            uid = f"client_{u_idx:04d}"
            mask = (
                (df["user_id"] == uid)
                & (df["merchant_raw"] == merch)
                & (df["is_subscription"] == 1)
            )
            if mask.any():
                yearly_in_dataset += 1
                yearly_pairs.add((uid, merch))
        logger.info(
            "Годовых подписок в датасете: %d (из %d запланированных).",
            yearly_in_dataset,
            len(yearly_plan),
        )

    # --- Обогащение категорией по MCC ---
    df = df.merge(df_mcc[["MCC", "Категория"]], on="MCC", how="left")
    df["Категория"] = df["Категория"].fillna("Прочие операции")
    df = df.sort_values(["user_id", "dt"]).reset_index(drop=True)

    # --- Диагностика ДО drop'а is_sbp (нужна для разметки сегментов) ---
    _log_quality_report(df, yearly_pairs)

    # --- Формат выписки Альфа-Банка ---
    auth_codes = rng.integers(100000, 1000000, size=len(df)).astype(str)
    padded_merch = df["merchant_raw"].str.slice(0, 28).str.ljust(28)
    date_str = df["dt"].dt.strftime("%d.%m.%Y")
    amt_str = np.abs(df["Сумма операции"]).map("{:.2f}".format)
    bslash = chr(92)

    card_desc = (
        df["card_mask"].str.replace("******", "++++++", regex=False)
        + "    "
        + auth_codes
        + f"{bslash}RUS{bslash}MOSCOW{bslash}"
        + padded_merch
        + " "
        + date_str
        + " "
        + amt_str
        + " RUR MCC"
        + df["MCC"]
    )
    sbp_desc = "Перевод по СБП: " + df["merchant_raw"] + " от " + date_str + " без НДС"

    df["Дата операции"] = df["dt"].dt.strftime("%d.%m.%Y %H:%M:%S")
    df["Номер карты"] = df["card_mask"]
    df["Статус"] = "Выполнено"
    df["Валюта"] = "RUR"
    df["Описание"] = card_desc.where(~df["is_sbp"], sbp_desc)

    # --- Метрики для metadata ---
    n_subs_tx = int((df["is_subscription"] == 1).sum())
    n_sbp_tx = int(df["Описание"].str.contains("СБП", regex=False).sum())
    n_reg_tx = len(df) - n_subs_tx - n_sbp_tx
    n_unique_merchants = int(df["merchant_raw"].nunique())
    total_sub_clusters = int(
        df.loc[df["is_subscription"] == 1, ["user_id", "merchant_raw"]]
        .drop_duplicates()
        .shape[0]
    )
    yearly_share = (
        yearly_in_dataset / total_sub_clusters * 100.0
        if total_sub_clusters > 0
        else 0.0
    )

    metadata: dict[str, Any] = {
        "n_users": n_users,
        "seed": seed,
        "p_yearly": p_yearly,
        "min_yearly": min_yearly,
        "date_start": str(DATE_START.date()),
        "date_end": str(DATE_END.date()),
        "n_transactions": len(df),
        "n_transactions_regular": n_reg_tx,
        "n_transactions_sbp": n_sbp_tx,
        "n_transactions_subscriptions": n_subs_tx,
        "subscription_ratio_tx": round(n_subs_tx / len(df), 6),
        "n_unique_merchants": n_unique_merchants,
        "n_subscription_clusters": total_sub_clusters,
        "n_yearly_subscriptions": yearly_in_dataset,
        "n_yearly_planned": len(yearly_plan),
        "yearly_share_of_sub_clusters": round(yearly_share, 4),
        "n_service_merchants_pool": len(sub_merchants),
        "n_retail_merchants_pool": len(reg_merchants),
        "mcc_catalog_size": int(len(df_mcc)),
    }

    df = df.drop(columns=["dt", "card_mask", "is_sbp"])
    return df, metadata


# =============================================================================
# CLI и main
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сборка датасета транзакций в формате Альфа-Банка из открытых источников."
    )
    parser.add_argument("--n-users", type=int, default=300, help="Кол-во клиентов (300–400 — оптимум)")
    parser.add_argument("--sample-size", type=int, default=30_000, help="Размер сэмпла HF-датасета")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Папка для выходных файлов")
    parser.add_argument(
        "--p-yearly",
        type=float,
        default=P_YEARLY_PER_USER,
        help=f"Вероятность годовой подписки у клиента (по умолчанию {P_YEARLY_PER_USER})",
    )
    parser.add_argument(
        "--min-yearly",
        type=int,
        default=MIN_YEARLY,
        help=f"Минимум годовых подписок (floor для демо, по умолчанию {MIN_YEARLY})",
    )
    return parser.parse_args()


def _log_summary(metadata: dict[str, Any], train_path: Path, metadata_path: Path) -> None:
    """Печатает красивый итоговый блок."""
    n_tx = metadata["n_transactions"]
    n_reg = metadata["n_transactions_regular"]
    n_sbp = metadata["n_transactions_sbp"]
    n_subs = metadata["n_transactions_subscriptions"]
    n_sub_clusters = metadata["n_subscription_clusters"]
    n_yearly = metadata["n_yearly_subscriptions"]
    yearly_share = metadata["yearly_share_of_sub_clusters"]

    logger.info("=" * 72)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 72)
    logger.info("  Output files:")
    logger.info("    parquet : %s", train_path)
    logger.info("    metadata: %s", metadata_path)
    logger.info("  Parameters:")
    logger.info("    n_users     = %d", metadata["n_users"])
    logger.info("    seed        = %d", metadata["seed"])
    logger.info("    p_yearly    = %.3f", metadata["p_yearly"])
    logger.info("    min_yearly  = %d", metadata["min_yearly"])
    logger.info("    period      = %s → %s", metadata["date_start"], metadata["date_end"])
    logger.info("  Transactions (total %d):", n_tx)
    logger.info("    regular        : %7d (%.1f%%)", n_reg, 100.0 * n_reg / n_tx)
    logger.info("    SBP            : %7d (%.1f%%)", n_sbp, 100.0 * n_sbp / n_tx)
    logger.info("    subscriptions  : %7d (%.1f%%)", n_subs, 100.0 * n_subs / n_tx)
    logger.info("  Subscription clusters: %d", n_sub_clusters)
    logger.info("    regular        : %7d", n_sub_clusters - n_yearly)
    logger.info("    yearly         : %7d (%.2f%% от подписок)", n_yearly, yearly_share)
    logger.info("  Merchants:")
    logger.info("    unique in data : %d", metadata["n_unique_merchants"])
    logger.info("    service pool   : %d", metadata["n_service_merchants_pool"])
    logger.info("    retail pool    : %d", metadata["n_retail_merchants_pool"])
    logger.info("    MCC catalog    : %d", metadata["mcc_catalog_size"])
    logger.info("=" * 72)


def main() -> None:
    args = _parse_args()
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Справочник MCC.
    df_mcc = load_russian_mcc_catalog(data_dir / "mcc_ru.json")

    # 2. Пулы мерчантов.
    sub_merchants, reg_merchants = fetch_open_merchants_pool(sample_size=args.sample_size)

    # 3. Генерация.
    df_dataset, metadata = build_realistic_statements(
        sub_merchants,
        reg_merchants,
        df_mcc,
        n_users=args.n_users,
        seed=args.seed,
        p_yearly=args.p_yearly,
        min_yearly=args.min_yearly,
    )

    # 4. Data Quality Gate.
    assert not df_dataset.isna().any().any(), "В датасете обнаружены NaN!"
    sub_ratio = float(df_dataset["is_subscription"].mean())
    assert 0.02 <= sub_ratio <= 0.60, f"Аномальный дисбаланс классов: {sub_ratio:.2%}"

    # 5. Сохранение.
    train_path = data_dir / "train_dataset.parquet"
    metadata_path = data_dir / "metadata.json"
    df_dataset.to_parquet(train_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _log_summary(metadata, train_path, metadata_path)

    # 6. Демо-выписка самого «подписочного» клиента.
    demo_user = df_dataset.groupby("user_id")["is_subscription"].sum().idxmax()
    demo_df = df_dataset.loc[df_dataset["user_id"] == demo_user].drop(
        columns=["user_id", "merchant_raw", "is_subscription"]
    )
    demo_path = data_dir / "sample_alfa_statement.csv"
    demo_df.to_csv(demo_path, sep=";", index=False, encoding="utf-8-sig")
    logger.info(
        "Демо-выписка клиента %s → %s (%d строк).",
        demo_user,
        demo_path,
        len(demo_df),
    )


if __name__ == "__main__":
    main()