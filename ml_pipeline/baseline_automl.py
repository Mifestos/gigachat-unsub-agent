from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

from backend.app.core.features import CAT_FEATURES, FEATURE_COLUMNS
from ml_pipeline.utils import (
    evaluate_predictions,
    find_optimal_threshold,
    load_and_split_dataset,
    setup_mlflow,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_automl_baseline(data_path: Path, models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    setup_mlflow(Path("mlruns"))

    train_df, test_df = load_and_split_dataset(data_path)
    y_train = train_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()

    lama_train = train_df[FEATURE_COLUMNS + ["target"]].copy()
    lama_test = test_df[FEATURE_COLUMNS].copy()
    for col in CAT_FEATURES:
        lama_train[col] = lama_train[col].astype(object)
        lama_test[col] = lama_test[col].astype(object)

    try:
        from lightautoml.automl.presets.tabular_presets import TabularAutoML
        from lightautoml.tasks import Task

        logger.info("Запуск LightAutoML (TabularAutoML) бейзлайна...")
        automl = TabularAutoML(
            task=Task("binary", metric="auc"),
            timeout=60,
            cpu_limit=4,
            reader_params={"n_jobs": 4, "cv": 3, "random_state": 42},
        )
        oof_preds = automl.fit_predict(
            lama_train,
            roles={"target": "target", "category": CAT_FEATURES},
            verbose=0,
        ).data[:, 0]
        test_preds = automl.predict(lama_test).data[:, 0]
        best_thr = find_optimal_threshold(y_train, oof_preds)
        metrics = evaluate_predictions(y_test, test_preds, best_thr)
        run_name = "LightAutoML_Tabular_Baseline"

    except Exception as e:
        logger.warning("LightAutoML недоступен (%s), запускаем HistGradientBoosting бейзлайн...", e)
        X_train = train_df[FEATURE_COLUMNS].copy()
        X_test = test_df[FEATURE_COLUMNS].copy()
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[CAT_FEATURES] = enc.fit_transform(X_train[CAT_FEATURES].astype(str))
        X_test[CAT_FEATURES] = enc.transform(X_test[CAT_FEATURES].astype(str))

        hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42)
        hgb.fit(X_train, y_train)
        test_preds = hgb.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, test_preds, threshold=0.50)
        run_name = "Sklearn_HistGB_Baseline"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"model_type": run_name, "n_features": len(FEATURE_COLUMNS)})
        mlflow.log_metrics(metrics)

    baseline_path = models_dir / "baseline_metrics.json"
    baseline_path.write_text(
        json.dumps({"model_name": run_name, "metrics": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("[%s] Holdout Test Metrics: %s -> сохранено в %s", run_name, metrics, baseline_path)


if __name__ == "__main__":
    run_automl_baseline(Path("data/train_dataset.parquet"), Path("models"))
