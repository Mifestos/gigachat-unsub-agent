from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from backend.app.core.features import build_merchant_features

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "gigachat-unsub-subscription-classifier"


def setup_mlflow(mlruns_dir: Path = Path("mlruns")) -> None:
    """Настраивает локальный SQLite-бэкенд MLflow внутри папки mlruns/."""
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlruns_dir / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT_NAME)


def load_and_split_dataset(
    data_path: Path, test_size: float = 0.20, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает транзакции, строит фичи и делит по user_id без утечки клиентов."""
    logger.info("Загрузка датасета из %s ...", data_path)
    df_raw = pd.read_parquet(data_path)
    df_raw["is_sbp"] = df_raw["Описание"].astype(str).str.contains("СБП", regex=False)

    df_features = build_merchant_features(df_raw)
    logger.info(
        "Сформировано %d кластеров (подписки: %.2f%%).",
        len(df_features),
        df_features["target"].mean() * 100.0,
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(
        splitter.split(df_features, df_features["target"], groups=df_features["user_id"])
    )
    train_df = df_features.iloc[train_idx].reset_index(drop=True)
    test_df = df_features.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Подбирает порог вероятности, максимизирующий F1-score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = int(np.argmax(f1_scores))
    if best_idx < len(thresholds):
        return float(np.clip(thresholds[best_idx], 0.15, 0.85))
    return 0.50


def evaluate_predictions(
    y_true: np.ndarray, y_probs: np.ndarray, threshold: float
) -> dict[str, float]:
    """Считает метрики качества на отложенной тестовой выборке."""
    y_pred = (y_probs >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_probs)), 5),
        "pr_auc": round(float(average_precision_score(y_true, y_probs)), 5),
        "f1": round(float(f1_score(y_true, y_pred)), 5),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 5),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 5),
        "threshold": round(float(threshold), 4),
    }
