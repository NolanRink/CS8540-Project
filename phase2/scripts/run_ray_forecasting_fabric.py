"""FABRIC-friendly Ray forecasting comparison for Phase 2.

This script keeps the accepted forecasting logic from the notebook and the
original command-line workflow, but makes the FABRIC execution path explicit:

- load the saved top-tag feature table
- rebuild the official baseline benchmark
- train one pooled sklearn model per Ray task
- compare learned models against `baseline_last_value`
- save predictions, metrics, and a compact run summary

The sklearn models used here are CPU-oriented. The `--use-gpu` flag only makes
Ray resource handling GPU-aware on a FABRIC node; it does not turn these model
families into GPU-accelerated estimators.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any


def log(message: str) -> None:
    print(message, flush=True)


log("Starting phase2/scripts/run_ray_forecasting_fabric.py")
log("Importing forecasting dependencies...")

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

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class tqdm:  # type: ignore[no-redef]
        """Small fallback for environments where tqdm is not installed."""

        def __init__(self, total: int, desc: str = "Progress", unit: str = "item") -> None:
            self.total = total
            self.desc = desc
            self.unit = unit
            self.current = 0

        def __enter__(self):
            log(f"{self.desc}: 0/{self.total} {self.unit}s")
            return self

        def update(self, amount: int = 1) -> None:
            self.current += amount
            log(f"{self.desc}: {self.current}/{self.total} {self.unit}s")

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

import config

log("Dependency imports complete.")


RUN_CONFIG: dict[str, Any] = {
    "target": config.TARGET_COLUMN,
    "official_baseline": "baseline_last_value",
    "selection_metric": "validation sMAPE",
    "expected_split_counts": {"train": 1440, "validation": 330, "test": 300},
}

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
    parser = argparse.ArgumentParser(
        description="Run the FABRIC-friendly Phase 2 Ray forecasting comparison."
    )
    parser.add_argument("--features", type=Path, default=config.FEATURE_TABLE)
    parser.add_argument("--num-workers", type=int, default=min(4, len(MODEL_CONFIGS)))
    parser.add_argument("--num-cpus", type=int, default=None, help="Total CPUs available to Ray.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate paths, data loading, environment reporting, and Ray startup without model training.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Report GPU/CUDA status. Sklearn models remain CPU-bound, so GPU is not reserved by default.",
    )
    parser.add_argument(
        "--reserve-gpu-resource",
        action="store_true",
        help="Reserve fractional Ray GPU resources for model tasks. Usually unnecessary for sklearn models.",
    )
    return parser.parse_args()


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def report_environment(use_gpu: bool) -> None:
    log("Environment:")
    log(f"  Python: {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    log(f"  pandas: {package_version('pandas')}")
    log(f"  numpy: {package_version('numpy')}")
    log(f"  scikit-learn: {package_version('scikit-learn')}")
    log(f"  pyarrow: {package_version('pyarrow')}")
    log(f"  ray: {package_version('ray')}")
    log(f"  torch: {package_version('torch')}")
    log(f"  GPU requested: {use_gpu}")

    if use_gpu:
        log("Checking torch CUDA availability...")
        try:
            import torch

            log(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
            log(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
            if torch.cuda.is_available():
                log(f"  CUDA device 0: {torch.cuda.get_device_name(0)}")
        except Exception as exc:  # pragma: no cover - environment-specific diagnostics
            log(f"  CUDA check failed: {exc}")

    log("Note: current sklearn forecasting models are CPU-bound even on GPU-capable VMs.")


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


def load_modeling_frame(feature_path: Path) -> pd.DataFrame:
    log("Checking feature table path...")
    log(f"  {feature_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature table: {feature_path}")

    log("Loading feature table parquet...")
    features = pd.read_parquet(feature_path)
    log(f"Loaded feature table with {len(features):,} rows and {len(features.columns):,} columns.")
    required = [
        "date",
        "tag",
        "tag_type",
        "count",
        config.TARGET_COLUMN,
        "split",
        "modeling_ready",
        *config.NUMERIC_FEATURES,
        *config.CATEGORICAL_FEATURES,
    ]
    require_columns(features, required, "feature table")
    log("Feature table columns validated.")

    modeling = (
        features.loc[features["modeling_ready"]]
        .copy()
        .sort_values(["tag_type", "tag", "date"])
        .reset_index(drop=True)
    )

    split_counts = modeling["split"].value_counts().to_dict()
    if split_counts != RUN_CONFIG["expected_split_counts"]:
        raise ValueError(f"Unexpected modeling split counts: {split_counts}")

    log(f"Modeling-ready rows: {len(modeling):,}")
    return modeling


def make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_model(model_config: dict[str, Any]):
    kind = model_config["kind"]
    if kind == "linear_regression":
        return LinearRegression()
    if kind == "ridge":
        return Ridge(alpha=model_config["alpha"])
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=model_config["n_estimators"],
            max_depth=model_config["max_depth"],
            min_samples_leaf=model_config["min_samples_leaf"],
            random_state=model_config["random_state"],
            n_jobs=1,
        )
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=model_config["max_iter"],
            learning_rate=model_config["learning_rate"],
            max_leaf_nodes=model_config["max_leaf_nodes"],
            l2_regularization=model_config["l2_regularization"],
            random_state=model_config["random_state"],
        )
    raise ValueError(f"Unknown model kind: {kind}")


def make_pipeline(model_config: dict[str, Any]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                config.NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_encoder()),
                    ]
                ),
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", make_model(model_config))])


def make_baseline_outputs(modeling: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("Building baseline predictions and metrics...")
    predictions = modeling[["date", "tag", "tag_type", "count", config.TARGET_COLUMN, "split"]].copy()
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
                    **metric_row(split_frame[config.TARGET_COLUMN], split_frame[baseline]),
                }
            )
    overall = pd.DataFrame(overall_rows).sort_values(["split", "MAE", "sMAPE"])

    test_frame = predictions[predictions["split"] == "test"]
    by_tag_rows = []
    for (tag_type, tag), group in test_frame.groupby(["tag_type", "tag"]):
        by_tag_rows.append(
            {
                "tag_type": tag_type,
                "tag": tag,
                "baseline": RUN_CONFIG["official_baseline"],
                **metric_row(group[config.TARGET_COLUMN], group[RUN_CONFIG["official_baseline"]]),
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
                    **metric_row(group[config.TARGET_COLUMN], group[baseline]),
                }
            )
    by_tag_type = pd.DataFrame(by_type_rows).sort_values(["split", "tag_type", "MAE"])
    log("Baseline outputs built.")
    return predictions, overall, by_tag, by_tag_type


@ray.remote
def train_model_worker(
    model_config: dict[str, Any],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Train one pooled model and return validation/test predictions."""
    model_name = model_config["name"]
    feature_cols = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES

    pipeline = make_pipeline(model_config)
    pipeline.fit(train_frame[feature_cols], train_frame[config.TARGET_COLUMN])

    prediction_frames = []
    metric_rows = []
    for split_name, frame in [("validation", validation_frame), ("test", test_frame)]:
        # Counts cannot be negative, so learned predictions are clipped to zero.
        predicted = np.clip(pipeline.predict(frame[feature_cols]), 0, None)
        split_predictions = frame[["date", "tag", "tag_type", config.TARGET_COLUMN, "split"]].copy()
        split_predictions["model"] = model_name
        split_predictions["prediction"] = predicted
        prediction_frames.append(split_predictions)

        metric_rows.append(
            {
                "split": split_name,
                "model": model_name,
                **metric_row(frame[config.TARGET_COLUMN], predicted),
            }
        )

    return {
        "model": model_name,
        "metrics": metric_rows,
        "predictions": pd.concat(prediction_frames, ignore_index=True),
    }


def choose_ray_resources(
    num_workers: int,
    num_cpus: int | None,
    use_gpu: bool,
    reserve_gpu_resource: bool,
) -> tuple[int, float]:
    """Start Ray and choose simple per-worker resources."""
    if num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if num_cpus is not None and num_cpus < 1:
        raise ValueError("--num-cpus must be at least 1 when provided")

    total_cpus = num_cpus or max(1, num_workers)
    if ray.is_initialized():
        log("Ray was already initialized; shutting it down before this run.")
        ray.shutdown()

    log("Initializing Ray...")
    log(f"  requested total CPUs: {total_cpus}")
    log(f"  requested workers: {num_workers}")
    log(f"  GPU status requested: {use_gpu}")
    log(f"  reserve GPU resource for sklearn tasks: {reserve_gpu_resource}")
    ray.init(
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
        num_cpus=total_cpus,
    )
    log("Ray initialized.")

    cluster_resources = ray.cluster_resources()
    log(f"Ray cluster resources: {cluster_resources}")
    visible_gpus = float(cluster_resources.get("GPU", 0.0))

    if reserve_gpu_resource and visible_gpus > 0:
        gpus_per_worker = min(1.0, visible_gpus / max(1, num_workers))
        log(
            f"Ray sees {visible_gpus:g} GPU resource(s); reserving "
            f"{gpus_per_worker:g} GPU per model task."
        )
        log("Note: the current sklearn models are CPU-bound and do not perform GPU training.")
    elif reserve_gpu_resource:
        gpus_per_worker = 0.0
        log("No Ray GPU resources were detected; continuing safely on CPU.")
    elif use_gpu and visible_gpus > 0:
        gpus_per_worker = 0.0
        log(f"Ray sees {visible_gpus:g} GPU resource(s), but GPU resources will not be reserved for CPU-bound sklearn tasks.")
    elif use_gpu:
        gpus_per_worker = 0.0
        log("No Ray GPU resources were detected; continuing safely on CPU.")
    else:
        gpus_per_worker = 0.0
        log("Running Ray model tasks with CPU resources.")

    cpus_per_worker = max(1, total_cpus // max(1, num_workers))
    log(f"Per-model task resources: {cpus_per_worker} CPU(s), {gpus_per_worker:g} GPU(s)")
    return cpus_per_worker, gpus_per_worker


def run_ray_model_comparison(
    modeling: pd.DataFrame,
    num_workers: int,
    num_cpus: int | None,
    use_gpu: bool,
    reserve_gpu_resource: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("Preparing train/validation/test frames for Ray model comparison...")
    train = modeling[modeling["split"] == "train"].copy()
    validation = modeling[modeling["split"] == "validation"].copy()
    test = modeling[modeling["split"] == "test"].copy()
    log(f"  train rows: {len(train):,}")
    log(f"  validation rows: {len(validation):,}")
    log(f"  test rows: {len(test):,}")

    cpus_per_worker, gpus_per_worker = choose_ray_resources(num_workers, num_cpus, use_gpu, reserve_gpu_resource)

    try:
        log("Submitting Ray model jobs...")
        futures = [
            train_model_worker.options(
                num_cpus=cpus_per_worker,
                num_gpus=gpus_per_worker,
            ).remote(model_config, train, validation, test)
            for model_config in MODEL_CONFIGS
        ]
        log(f"Submitted {len(futures)} model job(s): {[model_config['name'] for model_config in MODEL_CONFIGS]}")

        results = []
        with tqdm(total=len(futures), desc="Ray model jobs", unit="model") as progress:
            remaining = futures
            while remaining:
                done, remaining = ray.wait(remaining, num_returns=1)
                results.append(ray.get(done[0]))
                progress.update(1)
    finally:
        log("Shutting down Ray...")
        ray.shutdown()

    log("Ray model jobs complete.")
    metrics = pd.DataFrame([row for result in results for row in result["metrics"]])
    predictions = pd.concat([result["predictions"] for result in results], ignore_index=True)
    return predictions, metrics


def add_baseline_to_comparison(
    learned_predictions: pd.DataFrame,
    learned_metrics: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_name = RUN_CONFIG["official_baseline"]
    baseline_comparison = (
        baseline_predictions[baseline_predictions["split"].isin(["validation", "test"])][
            ["date", "tag", "tag_type", config.TARGET_COLUMN, "split", baseline_name]
        ]
        .rename(columns={baseline_name: "prediction"})
        .copy()
    )
    baseline_comparison["model"] = baseline_name

    all_predictions = pd.concat(
        [
            learned_predictions,
            baseline_comparison[["date", "tag", "tag_type", config.TARGET_COLUMN, "split", "model", "prediction"]],
        ],
        ignore_index=True,
    )

    baseline_rows = baseline_metrics.rename(columns={"baseline": "model"})[
        ["split", "model", "rows", "MAE", "RMSE", "sMAPE"]
    ]
    all_metrics = pd.concat(
        [learned_metrics, baseline_rows[baseline_rows["model"] == baseline_name]],
        ignore_index=True,
    )
    return all_predictions, all_metrics


def summarize_by_tag(predictions: pd.DataFrame, best_model: str) -> pd.DataFrame:
    rows = []
    best_test = predictions[(predictions["split"] == "test") & (predictions["model"] == best_model)]
    for (tag_type, tag), group in best_test.groupby(["tag_type", "tag"]):
        rows.append(
            {
                "tag_type": tag_type,
                "tag": tag,
                "model": best_model,
                **metric_row(group[config.TARGET_COLUMN], group["prediction"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["MAE", "sMAPE"], ascending=[False, False])


def summarize_by_tag_type(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_name, tag_type, model), group in predictions.groupby(["split", "tag_type", "model"]):
        rows.append(
            {
                "split": split_name,
                "tag_type": tag_type,
                "model": model,
                **metric_row(group[config.TARGET_COLUMN], group["prediction"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "tag_type", "sMAPE", "MAE"])


def save_outputs(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    by_tag: pd.DataFrame,
    by_tag_type: pd.DataFrame,
    run_summary: dict[str, Any],
    baseline_outputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    log("Saving output artifacts...")
    config.DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    baseline_predictions, baseline_metrics, baseline_by_tag, baseline_by_type = baseline_outputs
    baseline_predictions.to_parquet(config.BASELINE_PREDICTIONS, index=False)
    baseline_metrics.to_csv(config.BASELINE_METRICS_OVERALL, index=False)
    baseline_by_tag.to_csv(config.BASELINE_METRICS_BY_TAG, index=False)
    baseline_by_type.to_csv(config.BASELINE_METRICS_BY_TAG_TYPE, index=False)

    predictions.to_parquet(config.RAY_MODEL_PREDICTIONS, index=False)
    metrics.to_csv(config.RAY_MODEL_METRICS_OVERALL, index=False)
    by_tag.to_csv(config.RAY_MODEL_METRICS_BY_TAG, index=False)
    by_tag_type.to_csv(config.RAY_MODEL_METRICS_BY_TAG_TYPE, index=False)
    with open(config.RAY_MODEL_RUN_SUMMARY, "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, default=str)
    log(f"Saved predictions: {config.RAY_MODEL_PREDICTIONS}")
    log(f"Saved metrics: {config.RAY_MODEL_METRICS_OVERALL}")
    log(f"Saved run summary: {config.RAY_MODEL_RUN_SUMMARY}")


def main() -> int:
    args = parse_args()
    log("=" * 72)
    log("FABRIC Phase 2 Ray forecasting comparison")
    log("=" * 72)
    log("Configuration:")
    log(f"  feature table: {args.features}")
    log(f"  num workers: {args.num_workers}")
    log(f"  num CPUs: {args.num_cpus if args.num_cpus is not None else 'auto'}")
    log(f"  use GPU flag: {args.use_gpu}")
    log(f"  reserve GPU resource: {args.reserve_gpu_resource}")
    log(f"  smoke test only: {args.smoke_test}")
    report_environment(args.use_gpu)

    modeling = load_modeling_frame(args.features)
    split_counts = modeling["split"].value_counts().reindex(["train", "validation", "test"]).to_dict()
    log(f"Modeling rows by split: {split_counts}")

    if args.smoke_test:
        log("Smoke test requested: validating Ray startup without training models.")
        choose_ray_resources(
            num_workers=args.num_workers,
            num_cpus=args.num_cpus,
            use_gpu=args.use_gpu,
            reserve_gpu_resource=args.reserve_gpu_resource,
        )
        log("Smoke test Ray startup succeeded; shutting down Ray.")
        ray.shutdown()
        log("PASS")
        return 0

    baseline_outputs = make_baseline_outputs(modeling)
    baseline_predictions, baseline_metrics, _, _ = baseline_outputs

    learned_predictions, learned_metrics = run_ray_model_comparison(
        modeling=modeling,
        num_workers=args.num_workers,
        num_cpus=args.num_cpus,
        use_gpu=args.use_gpu,
        reserve_gpu_resource=args.reserve_gpu_resource,
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
    baseline_validation = validation_metrics[validation_metrics["model"] == RUN_CONFIG["official_baseline"]].iloc[0].to_dict()
    baseline_test = test_metrics[test_metrics["model"] == RUN_CONFIG["official_baseline"]].iloc[0].to_dict()

    by_tag = summarize_by_tag(all_predictions, best_model)
    by_tag_type = summarize_by_tag_type(all_predictions)

    expected_prediction_rows = (330 + 300) * (len(MODEL_CONFIGS) + 1)
    if len(all_predictions) != expected_prediction_rows:
        raise ValueError(f"Unexpected prediction row count: {len(all_predictions)}")

    run_summary = {
        "script": "phase2/scripts/run_ray_forecasting_fabric.py",
        "official_baseline": RUN_CONFIG["official_baseline"],
        "model_selection_metric": RUN_CONFIG["selection_metric"],
        "best_learned_model": best_model,
        "best_learned_validation_metrics": best_validation,
        "best_learned_test_metrics": best_test,
        "baseline_validation_metrics": baseline_validation,
        "baseline_test_metrics": baseline_test,
        "best_learned_beats_baseline_validation_sMAPE": bool(best_validation["sMAPE"] < baseline_validation["sMAPE"]),
        "best_learned_beats_baseline_test_sMAPE": bool(best_test["sMAPE"] < baseline_test["sMAPE"]),
        "gpu_requested": bool(args.use_gpu),
        "gpu_note": "Sklearn model families are CPU-bound; GPU handling is Ray resource compatibility only.",
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "models_compared": learned_names,
    }

    save_outputs(all_predictions, all_metrics, by_tag, by_tag_type, run_summary, baseline_outputs)

    log(f"Best learned model: {best_model}")
    log(f"Validation sMAPE: {best_validation['sMAPE']:.4f}")
    log(f"Test sMAPE: {best_test['sMAPE']:.4f}")
    log("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
