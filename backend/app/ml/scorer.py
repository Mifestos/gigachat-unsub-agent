from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from backend.app.core.config import settings
from backend.app.core.features import FEATURE_COLUMNS, build_merchant_features

logger = logging.getLogger(__name__)


@dataclass
class DetectedSubscription:
    merchant_clean: str
    probability: float
    tx_count: int
    last_amount: float
    mean_amount: float
    median_delta_days: float
    estimated_monthly_cost: float
    estimated_yearly_cost: float
    last_dt: str
    mcc_code: str


class SubscriptionMLScorer:
    """Продакшн-скорер подписок на базе обученной модели CatBoost."""

    def __init__(
        self,
        model_path: Path = settings.model_path,
        config_path: Path = settings.model_config_path,
    ) -> None:
        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"Артефакты модели не найдены ({model_path} / {config_path}). "
                "Сначала запустите: python -m ml_pipeline.train_catboost"
            )

        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.threshold: float = float(self.config.get("threshold", 0.5))
        self.feature_columns: list[str] = list(self.config.get("feature_columns", FEATURE_COLUMNS))

        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path))
        logger.info("CatBoost модель загружена (threshold=%.4f).", self.threshold)

    @staticmethod
    def _estimate_monthly_cost(mean_amount: float, median_delta_days: float) -> float:
        """Переводит стоимость цикла в эквивалент ежемесячных трат."""
        if median_delta_days <= 0 or median_delta_days >= 900:
            return round(mean_amount, 2)
        cycles_per_month = 30.0 / max(1.0, median_delta_days)
        return round(mean_amount * cycles_per_month, 2)

    def predict_subscriptions(
        self, df_parsed_tx: pd.DataFrame, custom_threshold: float | None = None
    ) -> list[DetectedSubscription]:
        """Возвращает отсортированный список обнаруженных подписок по распарсенным транзакциям."""
        if df_parsed_tx.empty:
            return []

        df_feat = build_merchant_features(df_parsed_tx)
        if df_feat.empty:
            return []

        probs = self.model.predict_proba(df_feat[self.feature_columns])[:, 1]
        df_feat["prob"] = probs

        active_threshold = custom_threshold if custom_threshold is not None else self.threshold
        subs_df = df_feat[df_feat["prob"] >= active_threshold].copy()
        subs_df = subs_df.sort_values(["prob", "mean_amount"], ascending=[False, False])

        results: list[DetectedSubscription] = []
        for row in subs_df.itertuples(index=False):
            monthly = self._estimate_monthly_cost(float(row.mean_amount), float(row.median_delta_days))
            results.append(
                DetectedSubscription(
                    merchant_clean=str(row.merchant_key),
                    probability=round(float(row.prob), 4),
                    tx_count=int(row.tx_count),
                    last_amount=round(float(row.last_amount), 2),
                    mean_amount=round(float(row.mean_amount), 2),
                    median_delta_days=round(float(row.median_delta_days), 1),
                    estimated_monthly_cost=monthly,
                    estimated_yearly_cost=round(monthly * 12.0, 2),
                    last_dt=pd.Timestamp(row.last_dt).strftime("%d.%m.%Y"),
                    mcc_code=str(row.mcc_code),
                )
            )
        return results
