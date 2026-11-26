from __future__ import annotations

import numpy as np
import pandas as pd

BILLING_CYCLES: tuple[int, ...] = (7, 14, 30, 90, 365)

FEATURE_COLUMNS: list[str] = [
    "tx_count",
    "mean_delta_days",
    "median_delta_days",
    "std_delta_days",
    "min_delta_days",
    "max_delta_days",
    "periodicity_error",
    "mean_amount",
    "std_amount",
    "amount_cv",
    "amount_to_user_median",
    "std_hour_of_day",
    "is_sbp",
    "mcc_code",
]

CAT_FEATURES: list[str] = ["mcc_code"]


def _min_cycle_distance(median_days: float) -> float:
    """Минимальное абсолютное отклонение медианного интервала от стандартных циклов биллинга."""
    if np.isnan(median_days) or median_days <= 0:
        return 999.0
    return float(min(abs(median_days - cycle) for cycle in BILLING_CYCLES))


def build_merchant_features(df_tx: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует транзакции в поведенческие фичи на уровне (user_id, merchant_clean).

    Ожидаемые колонки в df_tx:
    - user_id (если нет — заполняется 'single_user')
    - merchant_clean (или merchant_raw)
    - dt (pd.Timestamp)
    - amount (положительный float, модуль суммы списания)
    - MCC (строка из 4 цифр)
    - is_sbp (bool)
    - is_subscription (опционально, целевая переменная для обучения)
    """
    work = df_tx.copy()

    if "user_id" not in work.columns:
        work["user_id"] = "single_user"

    merch_col = "merchant_clean" if "merchant_clean" in work.columns else "merchant_raw"
    work["merchant_key"] = work[merch_col].astype(str).str.strip()

    if "amount" not in work.columns:
        work["amount"] = work["Сумма операции"].astype(float).abs()

    if "dt" not in work.columns:
        work["dt"] = pd.to_datetime(work["Дата операции"], dayfirst=True)

    work["mcc_code"] = work["MCC"].fillna("0000").astype(str).str.strip().str.zfill(4)
    work["is_sbp_int"] = work["is_sbp"].astype(int) if "is_sbp" in work.columns else 0
    work["hour"] = work["dt"].dt.hour

    # Медианный чек каждого пользователя для нормализации суммы
    user_medians = work.groupby("user_id")["amount"].median().rename("user_median_amount")

    # Сортируем по времени и считаем интервалы между последовательными операциями у одного мерчанта
    work = work.sort_values(["user_id", "merchant_key", "dt"]).reset_index(drop=True)
    work["delta_days"] = (
        work.groupby(["user_id", "merchant_key"])["dt"]
        .diff()
        .dt.total_seconds()
        / 86400.0
    )

    agg_dict: dict[str, tuple[str, str | object]] = {
        "tx_count": ("amount", "count"),
        "mean_delta_days": ("delta_days", "mean"),
        "median_delta_days": ("delta_days", "median"),
        "std_delta_days": ("delta_days", lambda s: float(s.std(ddof=0)) if s.notna().sum() >= 1 else 999.0),
        "min_delta_days": ("delta_days", "min"),
        "max_delta_days": ("delta_days", "max"),
        "mean_amount": ("amount", "mean"),
        "std_amount": ("amount", lambda s: float(s.std(ddof=0))),
        "std_hour_of_day": ("hour", lambda s: float(s.std(ddof=0))),
        "is_sbp": ("is_sbp_int", "max"),
        "mcc_code": ("mcc_code", lambda s: str(next(iter(s.mode()), "0000"))),
        "last_dt": ("dt", "max"),
        "last_amount": ("amount", "last"),
    }

    if "is_subscription" in work.columns:
        agg_dict["target"] = ("is_subscription", "max")

    features_df = work.groupby(["user_id", "merchant_key"], as_index=False).agg(**agg_dict)
    features_df = features_df.merge(user_medians, on="user_id", how="left")

    # Заполняем пропуски для одиночных транзакций
    features_df["mean_delta_days"] = features_df["mean_delta_days"].fillna(999.0)
    features_df["median_delta_days"] = features_df["median_delta_days"].fillna(999.0)
    features_df["std_delta_days"] = features_df["std_delta_days"].fillna(999.0)
    features_df["min_delta_days"] = features_df["min_delta_days"].fillna(999.0)
    features_df["max_delta_days"] = features_df["max_delta_days"].fillna(999.0)

    # Производные доменные признаки
    features_df["periodicity_error"] = features_df["median_delta_days"].map(_min_cycle_distance)
    features_df["amount_cv"] = np.where(
        features_df["mean_amount"] > 0,
        features_df["std_amount"] / features_df["mean_amount"],
        0.0,
    )
    features_df["amount_to_user_median"] = np.where(
        features_df["user_median_amount"] > 0,
        features_df["mean_amount"] / features_df["user_median_amount"],
        1.0,
    )

    return features_df.drop(columns=["user_median_amount"])
