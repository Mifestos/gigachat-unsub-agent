from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupShuffleSplit

from backend.app.core.features import CAT_FEATURES, FEATURE_COLUMNS
from ml_pipeline.utils import (
    evaluate_predictions,
    find_optimal_threshold,
    load_and_split_dataset,
    setup_mlflow,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_catboost_model(data_path: Path, models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    mlruns_dir = Path("mlruns")
    setup_mlflow(mlruns_dir)

    train_df, test_df = load_and_split_dataset(data_path)

    # Выделяем 15% клиентов внутри train для early_stopping и калибровки порога
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_sub_idx, val_sub_idx = next(
        val_splitter.split(train_df, train_df["target"], groups=train_df["user_id"])
    )
    tr_sub = train_df.iloc[tr_sub_idx]
    val_sub = train_df.iloc[val_sub_idx]

    train_pool = Pool(tr_sub[FEATURE_COLUMNS], tr_sub["target"], cat_features=CAT_FEATURES)
    val_pool = Pool(val_sub[FEATURE_COLUMNS], val_sub["target"], cat_features=CAT_FEATURES)
    test_pool = Pool(test_df[FEATURE_COLUMNS], test_df["target"], cat_features=CAT_FEATURES)

    cb_params: dict[str, Any] = {
        "iterations": 800,
        "learning_rate": 0.04,
        "depth": 6,
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": 42,
        "train_dir": str(mlruns_dir / "catboost_info"),  # не засоряем корень проекта!
        "verbose": 100,
    }

    baseline_info: dict[str, Any] = {}
    baseline_path = models_dir / "baseline_metrics.json"
    if baseline_path.exists():
        baseline_info = json.loads(baseline_path.read_text(encoding="utf-8"))

    with mlflow.start_run(run_name="CatBoost_Production"):
        mlflow.log_params(cb_params)
        model = CatBoostClassifier(**cb_params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=60, verbose=100)

        val_probs = model.predict_proba(val_pool)[:, 1]
        best_threshold = find_optimal_threshold(val_sub["target"].to_numpy(), val_probs)

        test_probs = model.predict_proba(test_pool)[:, 1]
        cb_metrics = evaluate_predictions(test_df["target"].to_numpy(), test_probs, best_threshold)
        mlflow.log_metrics(cb_metrics)

        importances = dict(
            sorted(
                zip(FEATURE_COLUMNS, [round(float(x), 3) for x in model.get_feature_importance()]),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        model_path = models_dir / "subscription_catboost.cbm"
        config_path = models_dir / "model_config.json"
        model.save_model(str(model_path))

        config_payload = {
            "model_file": model_path.name,
            "threshold": round(best_threshold, 4),
            "feature_columns": FEATURE_COLUMNS,
            "cat_features": CAT_FEATURES,
            "holdout_metrics": cb_metrics,
            "baseline_comparison": baseline_info,
            "feature_importances": importances,
        }
        config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(config_path))

    logger.info("=" * 72)
    logger.info("PRODUCTION MODEL READY: %s", model_path)
    logger.info("Holdout Test Metrics: %s", cb_metrics)
    logger.info("Top Feature Importances: %s", importances)
    logger.info("=" * 72)


if __name__ == "__main__":
    train_catboost_model(Path("data/train_dataset.parquet"), Path("models"))
