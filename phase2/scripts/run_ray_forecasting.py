"""Run the Phase 2 baseline and Ray pooled forecasting comparison."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PHASE2_ROOT = Path(__file__).resolve().parents[1]
DERIVED_DIR = PHASE2_ROOT / "data" / "derived"
FEATURE_TABLE = DERIVED_DIR / "top_tags_daily_features.parquet"

BASELINE_PREDICTIONS = DERIVED_DIR / "baseline_forecast_predictions.parquet"
BASELINE_METRICS_OVERALL = DERIVED_DIR / "baseline_forecast_metrics_overall.csv"
BASELINE_METRICS_BY_TAG = DERIVED_DIR / "baseline_forecast_metrics_by_tag.csv"
BASELINE_METRICS_BY_TAG_TYPE = DERIVED_DIR / "baseline_forecast_metrics_by_tag_type.csv"

RAY_MODEL_PREDICTIONS = DERIVED_DIR / "ray_model_predictions.parquet"
RAY_MODEL_METRICS_OVERALL = DERIVED_DIR / "ray_model_metrics_overall.csv"
RAY_MODEL_METRICS_BY_TAG = DERIVED_DIR / "ray_model_metrics_by_tag.csv"
RAY_MODEL_METRICS_BY_TAG_TYPE = DERIVED_DIR / "ray_model_metrics_by_tag_type.csv"
RAY_MODEL_RUN_SUMMARY = DERIVED_DIR / "ray_model_run_summary.json"
FORECAST_COMPARISON = DERIVED_DIR / "sentiment_forecast_comparison.csv"

TARGET_COLUMN = "target_next_count"
BASE_NUMERIC_FEATURES = [
    "count",
    "count_lag_1",
    "count_lag_2",
    "count_lag_3",
    "count_lag_7",
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_std_3",
    "rolling_std_7",
    "count_diff_1",
    "count_pct_change_1",
]
SENTIMENT_NUMERIC_FEATURES = [
    "sentiment_mean",
    "sentiment_median",
    "sentiment_std",
    "sentiment_tweet_count",
    "positive_share",
    "negative_share",
    "neutral_share",
    "avg_sentiment_confidence",
    "sentiment_mean_lag_1",
    "sentiment_mean_lag_3",
    "sentiment_tweet_count_lag_1",
    "positive_share_lag_1",
    "negative_share_lag_1",
    "sentiment_mean_rolling_3",
    "sentiment_mean_rolling_7",
    "positive_share_rolling_3",
    "negative_share_rolling_3",
]
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES
CATEGORICAL_FEATURES = ["tag", "tag_type"]
EXPECTED_SPLITS = {"train": 1440, "validation": 330, "test": 300}
OFFICIAL_BASELINE = "baseline_last_value"

MODEL_CONFIGS: list[dict[str, Any]] = [
    {"name": "linear_regression_pooled", "kind": "linear_regression"},
    {"name": "ridge_regression_pooled", "kind": "ridge", "alpha": 10.0},
    {
        "name": "random_forest_pooled",
        "kind": "random_forest",
        "n_estimators": 80,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "random_state": 42,
    },
    {
        "name": "hist_gradient_boosting_pooled",
        "kind": "hist_gradient_boosting",
        "max_iter": 120,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "l2_regularization": 0.1,
        "random_state": 42,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 Ray forecasting.")
    parser.add_argument("--features", type=Path, default=FEATURE_TABLE)
    parser.add_argument("--run-label", default=None, help="Optional label used to save run-specific outputs.")
    parser.add_argument("--include-sentiment", action="store_true", help="Use sentiment numeric features if present.")
    parser.add_argument("--num-workers", type=int, default=min(4, len(MODEL_CONFIGS)))
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Check data loading and Ray startup without training.")
    parser.add_argument("--use-gpu", action="store_true", help="Note GPU visibility; sklearn models still run on CPU.")
    parser.add_argument("--reserve-gpu-resource", action="store_true", help="Reserve Ray GPU resources for tasks if present.")
    return parser.parse_args()


def validate_run_label(run_label: str | None) -> str | None:
    if run_label is None:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_]+", run_label):
        raise ValueError("--run-label may contain only letters, numbers, and underscores.")
    return run_label


def labeled_path(path: Path, run_label: str | None) -> Path:
    if run_label is None:
        return path
    return path.with_name(f"{path.stem}_{run_label}{path.suffix}")


def output_paths(run_label: str | None) -> dict[str, Path]:
    return {
        "baseline_predictions": labeled_path(BASELINE_PREDICTIONS, run_label),
        "baseline_metrics_overall": labeled_path(BASELINE_METRICS_OVERALL, run_label),
        "baseline_metrics_by_tag": labeled_path(BASELINE_METRICS_BY_TAG, run_label),
        "baseline_metrics_by_tag_type": labeled_path(BASELINE_METRICS_BY_TAG_TYPE, run_label),
        "ray_predictions": labeled_path(RAY_MODEL_PREDICTIONS, run_label),
        "ray_metrics_overall": labeled_path(RAY_MODEL_METRICS_OVERALL, run_label),
        "ray_metrics_by_tag": labeled_path(RAY_MODEL_METRICS_BY_TAG, run_label),
        "ray_metrics_by_tag_type": labeled_path(RAY_MODEL_METRICS_BY_TAG_TYPE, run_label),
        "run_summary": labeled_path(RAY_MODEL_RUN_SUMMARY, run_label),
    }


def numeric_features(include_sentiment: bool) -> list[str]:
    if include_sentiment:
        return [*BASE_NUMERIC_FEATURES, *SENTIMENT_NUMERIC_FEATURES]
    return BASE_NUMERIC_FEATURES.copy()


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def smape_percent(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    values = np.where(denominator == 0, 0.0, np.abs(predicted - actual) / denominator * 100)
    return float(np.mean(values))


def metric_row(actual: pd.Series, predicted: pd.Series | np.ndarray) -> dict[str, float | int]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    return {
        "rows": int(len(actual_values)),
        "MAE": float(mean_absolute_error(actual_values, predicted_values)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "sMAPE": smape_percent(actual_values, predicted_values),
    }


def load_modeling_frame(feature_path: Path, include_sentiment: bool) -> tuple[pd.DataFrame, list[str]]:
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature table: {feature_path}")

    features = pd.read_parquet(feature_path)
    selected_numeric = numeric_features(include_sentiment)
    required = [
        "date",
        "tag",
        "tag_type",
        "count",
        TARGET_COLUMN,
        "split",
        "modeling_ready",
        *selected_numeric,
        *CATEGORICAL_FEATURES,
    ]
    require_columns(features, required, "feature table")

    modeling = (
        features.loc[features["modeling_ready"]]
        .copy()
        .sort_values(["tag_type", "tag", "date"])
        .reset_index(drop=True)
    )
    split_counts = modeling["split"].value_counts().to_dict()
    if split_counts != EXPECTED_SPLITS:
        raise ValueError(f"Unexpected modeling split counts: {split_counts}")
    return modeling, selected_numeric


def make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_model(config: dict[str, Any]):
    kind = config["kind"]
    if kind == "linear_regression":
        return LinearRegression()
    if kind == "ridge":
        return Ridge(alpha=config["alpha"])
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            random_state=config["random_state"],
            n_jobs=1,
        )
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=config["max_iter"],
            learning_rate=config["learning_rate"],
            max_leaf_nodes=config["max_leaf_nodes"],
            l2_regularization=config["l2_regularization"],
            random_state=config["random_state"],
        )
    raise ValueError(f"Unknown model kind: {kind}")


def make_pipeline(model_config: dict[str, Any], numeric_feature_names: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_feature_names,
            ),
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_encoder())]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", make_model(model_config))])


def make_baseline_outputs(modeling: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = modeling[["date", "tag", "tag_type", "count", TARGET_COLUMN, "split"]].copy()
    predictions["baseline_last_value"] = modeling["count"].astype(float)
    predictions["baseline_rolling_mean_3"] = modeling["rolling_mean_3"].astype(float)
    predictions["baseline_rolling_mean_7"] = modeling["rolling_mean_7"].astype(float)
    predictions["baseline_lag_7"] = modeling["count_lag_7"].astype(float)

    baseline_names = [
        "baseline_last_value",
        "baseline_rolling_mean_3",
        "baseline_rolling_mean_7",
        "baseline_lag_7",
    ]

    overall_rows = []
    for split_name in ["validation", "test"]:
        split_frame = predictions[predictions["split"] == split_name]
        for baseline in baseline_names:
            overall_rows.append(
                {
                    "split": split_name,
                    "baseline": baseline,
                    **metric_row(split_frame[TARGET_COLUMN], split_frame[baseline]),
                }
            )
    overall = pd.DataFrame(overall_rows).sort_values(["split", "MAE", "sMAPE"])

    by_tag_rows = []
    for (tag_type, tag), group in predictions[predictions["split"] == "test"].groupby(["tag_type", "tag"]):
        by_tag_rows.append(
            {
                "tag_type": tag_type,
                "tag": tag,
                "baseline": OFFICIAL_BASELINE,
                **metric_row(group[TARGET_COLUMN], group[OFFICIAL_BASELINE]),
            }
        )
    by_tag = pd.DataFrame(by_tag_rows).sort_values(["MAE", "sMAPE"], ascending=[False, False])

    by_type_rows = []
    scored = predictions[predictions["split"].isin(["validation", "test"])]
    for (split_name, tag_type), group in scored.groupby(["split", "tag_type"]):
        for baseline in baseline_names:
            by_type_rows.append(
                {
                    "split": split_name,
                    "tag_type": tag_type,
                    "baseline": baseline,
                    **metric_row(group[TARGET_COLUMN], group[baseline]),
                }
            )
    by_tag_type = pd.DataFrame(by_type_rows).sort_values(["split", "tag_type", "MAE"])
    return predictions, overall, by_tag, by_tag_type


@ray.remote
def train_model_worker(
    model_config: dict[str, Any],
    numeric_feature_names: list[str],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> dict[str, Any]:
    model_name = model_config["name"]
    feature_cols = numeric_feature_names + CATEGORICAL_FEATURES
    pipeline = make_pipeline(model_config, numeric_feature_names)
    pipeline.fit(train_frame[feature_cols], train_frame[TARGET_COLUMN])

    prediction_frames = []
    metric_rows = []
    for split_name, frame in [("validation", validation_frame), ("test", test_frame)]:
        predicted = np.clip(pipeline.predict(frame[feature_cols]), 0, None)
        split_predictions = frame[["date", "tag", "tag_type", TARGET_COLUMN, "split"]].copy()
        split_predictions["model"] = model_name
        split_predictions["prediction"] = predicted
        prediction_frames.append(split_predictions)
        metric_rows.append({"split": split_name, "model": model_name, **metric_row(frame[TARGET_COLUMN], predicted)})

    return {
        "model": model_name,
        "metrics": metric_rows,
        "predictions": pd.concat(prediction_frames, ignore_index=True),
    }


def start_ray(num_workers: int, num_cpus: int | None, use_gpu: bool, reserve_gpu_resource: bool) -> tuple[int, float]:
    if num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if num_cpus is not None and num_cpus < 1:
        raise ValueError("--num-cpus must be at least 1")

    total_cpus = num_cpus or max(1, num_workers)
    if ray.is_initialized():
        ray.shutdown()

    warnings.filterwarnings("ignore", message="Tip: In future versions of Ray.*", category=FutureWarning)
    ray.init(
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
        logging_level=logging.ERROR,
        num_cpus=total_cpus,
    )
    resources = ray.cluster_resources()
    visible_gpus = float(resources.get("GPU", 0.0))

    if use_gpu:
        print(f"Ray sees {visible_gpus:g} GPU resource(s); sklearn models remain CPU-bound.")

    gpus_per_worker = 0.0
    if reserve_gpu_resource and visible_gpus > 0:
        gpus_per_worker = min(1.0, visible_gpus / max(1, num_workers))

    cpus_per_worker = max(1, total_cpus // max(1, num_workers))
    return cpus_per_worker, gpus_per_worker


def run_ray_models(
    modeling: pd.DataFrame,
    numeric_feature_names: list[str],
    num_workers: int,
    num_cpus: int | None,
    use_gpu: bool,
    reserve_gpu_resource: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = modeling[modeling["split"] == "train"].copy()
    validation = modeling[modeling["split"] == "validation"].copy()
    test = modeling[modeling["split"] == "test"].copy()

    cpus_per_worker, gpus_per_worker = start_ray(num_workers, num_cpus, use_gpu, reserve_gpu_resource)
    try:
        futures = [
            train_model_worker.options(num_cpus=cpus_per_worker, num_gpus=gpus_per_worker).remote(
                model_config,
                numeric_feature_names,
                train,
                validation,
                test,
            )
            for model_config in MODEL_CONFIGS
        ]
        results = ray.get(futures)
    finally:
        ray.shutdown()

    metrics = pd.DataFrame([row for result in results for row in result["metrics"]])
    predictions = pd.concat([result["predictions"] for result in results], ignore_index=True)
    return predictions, metrics


def add_baseline_to_comparison(
    learned_predictions: pd.DataFrame,
    learned_metrics: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_comparison = (
        baseline_predictions[baseline_predictions["split"].isin(["validation", "test"])][
            ["date", "tag", "tag_type", TARGET_COLUMN, "split", OFFICIAL_BASELINE]
        ]
        .rename(columns={OFFICIAL_BASELINE: "prediction"})
        .copy()
    )
    baseline_comparison["model"] = OFFICIAL_BASELINE

    all_predictions = pd.concat(
        [
            learned_predictions,
            baseline_comparison[["date", "tag", "tag_type", TARGET_COLUMN, "split", "model", "prediction"]],
        ],
        ignore_index=True,
    )

    baseline_rows = baseline_metrics.rename(columns={"baseline": "model"})[
        ["split", "model", "rows", "MAE", "RMSE", "sMAPE"]
    ]
    all_metrics = pd.concat(
        [learned_metrics, baseline_rows[baseline_rows["model"] == OFFICIAL_BASELINE]],
        ignore_index=True,
    )
    return all_predictions, all_metrics


def summarize_by_tag(predictions: pd.DataFrame, best_model: str) -> pd.DataFrame:
    rows = []
    best_test = predictions[(predictions["split"] == "test") & (predictions["model"] == best_model)]
    for (tag_type, tag), group in best_test.groupby(["tag_type", "tag"]):
        rows.append({"tag_type": tag_type, "tag": tag, "model": best_model, **metric_row(group[TARGET_COLUMN], group["prediction"])})
    return pd.DataFrame(rows).sort_values(["MAE", "sMAPE"], ascending=[False, False])


def summarize_by_tag_type(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_name, tag_type, model), group in predictions.groupby(["split", "tag_type", "model"]):
        rows.append({"split": split_name, "tag_type": tag_type, "model": model, **metric_row(group[TARGET_COLUMN], group["prediction"])})
    return pd.DataFrame(rows).sort_values(["split", "tag_type", "sMAPE", "MAE"])


def save_outputs(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    by_tag: pd.DataFrame,
    by_tag_type: pd.DataFrame,
    run_summary: dict[str, Any],
    baseline_outputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    paths: dict[str, Path],
) -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    baseline_predictions, baseline_metrics, baseline_by_tag, baseline_by_type = baseline_outputs

    baseline_predictions.to_parquet(paths["baseline_predictions"], index=False)
    baseline_metrics.to_csv(paths["baseline_metrics_overall"], index=False)
    baseline_by_tag.to_csv(paths["baseline_metrics_by_tag"], index=False)
    baseline_by_type.to_csv(paths["baseline_metrics_by_tag_type"], index=False)

    predictions.to_parquet(paths["ray_predictions"], index=False)
    metrics.to_csv(paths["ray_metrics_overall"], index=False)
    by_tag.to_csv(paths["ray_metrics_by_tag"], index=False)
    by_tag_type.to_csv(paths["ray_metrics_by_tag_type"], index=False)
    with open(paths["run_summary"], "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, default=str)

    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Expected output was not saved: {path}")


def update_forecast_comparison(metrics: pd.DataFrame, run_label: str | None) -> None:
    if run_label is None:
        return

    rows = metrics[["split", "model", "rows", "MAE", "RMSE", "sMAPE"]].copy()
    rows.insert(0, "run_label", run_label)

    if FORECAST_COMPARISON.exists():
        comparison = pd.read_csv(FORECAST_COMPARISON)
        comparison = comparison[comparison["run_label"] != run_label]
        comparison = pd.concat([comparison, rows], ignore_index=True)
    else:
        comparison = rows

    comparison = comparison.sort_values(["run_label", "split", "sMAPE", "MAE"])
    comparison.to_csv(FORECAST_COMPARISON, index=False)
    if not FORECAST_COMPARISON.exists():
        raise FileNotFoundError(f"Comparison output was not saved: {FORECAST_COMPARISON}")


def main() -> int:
    args = parse_args()
    args.run_label = validate_run_label(args.run_label)
    paths = output_paths(args.run_label)

    print("Loading feature table...")
    modeling, selected_numeric = load_modeling_frame(args.features, args.include_sentiment)
    split_counts = modeling["split"].value_counts().reindex(["train", "validation", "test"]).to_dict()
    print(f"Modeling rows by split: {split_counts}")
    print(f"Numeric features: {len(selected_numeric)}")
    if args.include_sentiment:
        covered = int((modeling["sentiment_tweet_count"] > 0).sum())
        if covered / len(modeling) < 0.05:
            print(
                "Warning: sentiment coverage is low "
                f"({covered:,}/{len(modeling):,} modeling rows). Treat results as pipeline validation only."
            )

    if args.smoke_test:
        print("Starting Ray smoke test...")
        start_ray(args.num_workers, args.num_cpus, args.use_gpu, args.reserve_gpu_resource)
        ray.shutdown()
        print("Smoke test passed.")
        return 0

    print("Building baseline metrics...")
    baseline_outputs = make_baseline_outputs(modeling)
    baseline_predictions, baseline_metrics, _, _ = baseline_outputs

    print("Training Ray models...")
    learned_predictions, learned_metrics = run_ray_models(
        modeling,
        selected_numeric,
        args.num_workers,
        args.num_cpus,
        args.use_gpu,
        args.reserve_gpu_resource,
    )
    all_predictions, all_metrics = add_baseline_to_comparison(
        learned_predictions,
        learned_metrics,
        baseline_predictions,
        baseline_metrics,
    )

    validation_metrics = all_metrics[all_metrics["split"] == "validation"].sort_values(["sMAPE", "MAE"])
    test_metrics = all_metrics[all_metrics["split"] == "test"].sort_values(["sMAPE", "MAE"])
    learned_names = [model_config["name"] for model_config in MODEL_CONFIGS]
    best_model = validation_metrics[validation_metrics["model"].isin(learned_names)].iloc[0]["model"]

    best_validation = validation_metrics[validation_metrics["model"] == best_model].iloc[0].to_dict()
    best_test = test_metrics[test_metrics["model"] == best_model].iloc[0].to_dict()
    baseline_validation = validation_metrics[validation_metrics["model"] == OFFICIAL_BASELINE].iloc[0].to_dict()
    baseline_test = test_metrics[test_metrics["model"] == OFFICIAL_BASELINE].iloc[0].to_dict()

    by_tag = summarize_by_tag(all_predictions, best_model)
    by_tag_type = summarize_by_tag_type(all_predictions)

    expected_prediction_rows = (330 + 300) * (len(MODEL_CONFIGS) + 1)
    if len(all_predictions) != expected_prediction_rows:
        raise ValueError(f"Unexpected prediction row count: {len(all_predictions)}")

    run_summary = {
        "script": "phase2/scripts/run_ray_forecasting.py",
        "run_label": args.run_label,
        "features": str(args.features),
        "include_sentiment": bool(args.include_sentiment),
        "official_baseline": OFFICIAL_BASELINE,
        "model_selection_metric": "validation sMAPE",
        "best_learned_model": best_model,
        "best_learned_validation_metrics": best_validation,
        "best_learned_test_metrics": best_test,
        "baseline_validation_metrics": baseline_validation,
        "baseline_test_metrics": baseline_test,
        "best_learned_beats_baseline_validation_sMAPE": bool(best_validation["sMAPE"] < baseline_validation["sMAPE"]),
        "best_learned_beats_baseline_test_sMAPE": bool(best_test["sMAPE"] < baseline_test["sMAPE"]),
        "gpu_requested": bool(args.use_gpu),
        "gpu_note": "Sklearn model families are CPU-bound; GPU handling is Ray resource compatibility only.",
        "numeric_features": selected_numeric,
        "categorical_features": CATEGORICAL_FEATURES,
        "models_compared": learned_names,
    }

    print("Saving outputs...")
    save_outputs(all_predictions, all_metrics, by_tag, by_tag_type, run_summary, baseline_outputs, paths)
    update_forecast_comparison(all_metrics, args.run_label)

    print(f"Best learned model: {best_model}")
    print(f"Validation sMAPE: {best_validation['sMAPE']:.4f}; test sMAPE: {best_test['sMAPE']:.4f}")
    print(f"Saved predictions to {paths['ray_predictions']}")
    print(f"Saved metrics to {paths['ray_metrics_overall']}")
    if args.run_label is not None:
        print(f"Updated comparison CSV at {FORECAST_COMPARISON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
