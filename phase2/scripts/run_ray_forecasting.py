"""Run Phase 2 baseline and Ray-based pooled forecasting models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

import config


MODEL_NAMES = [
    "linear_regression_pooled",
    "ridge_regression_pooled",
    "random_forest_pooled",
    "hist_gradient_boosting_pooled",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 Ray forecasting comparison.")
    parser.add_argument("--features", type=Path, default=config.FEATURE_TABLE)
    parser.add_argument("--num-cpus", type=int, default=min(4, len(MODEL_NAMES)))
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def smape_percent(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    values = np.where(denominator == 0, 0.0, np.abs(predicted - actual) / denominator * 100)
    return float(np.mean(values))


def metric_dict(actual, predicted) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "rows": int(len(actual)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "sMAPE": smape_percent(actual, predicted),
    }


def metrics_by_baseline(predictions: pd.DataFrame, split_name: str) -> pd.DataFrame:
    rows = []
    split_frame = predictions[predictions["split"] == split_name]
    baseline_cols = [
        "baseline_last_value",
        "baseline_rolling_mean_3",
        "baseline_rolling_mean_7",
        "baseline_lag_7",
    ]
    for baseline in baseline_cols:
        rows.append({"split": split_name, "baseline": baseline, **metric_dict(split_frame["target_next_count"], split_frame[baseline])})
    return pd.DataFrame(rows)


def make_baseline_predictions(modeling: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = modeling[["date", "tag", "tag_type", "count", "target_next_count", "split"]].copy()
    predictions["baseline_last_value"] = modeling["count"].astype(float)
    predictions["baseline_rolling_mean_3"] = modeling["rolling_mean_3"].astype(float)
    predictions["baseline_rolling_mean_7"] = modeling["rolling_mean_7"].astype(float)
    predictions["baseline_lag_7"] = modeling["count_lag_7"].astype(float)

    overall = pd.concat(
        [metrics_by_baseline(predictions, "validation"), metrics_by_baseline(predictions, "test")],
        ignore_index=True,
    ).sort_values(["split", "MAE", "sMAPE"])

    best_baseline = overall[overall["split"] == "validation"].sort_values(["MAE", "sMAPE"]).iloc[0]["baseline"]
    per_tag_rows = []
    for (tag_type, tag), group in predictions[predictions["split"] == "test"].groupby(["tag_type", "tag"]):
        per_tag_rows.append({"tag_type": tag_type, "tag": tag, "baseline": best_baseline, **metric_dict(group["target_next_count"], group[best_baseline])})
    by_tag = pd.DataFrame(per_tag_rows).sort_values(["MAE", "sMAPE"], ascending=[False, False])

    by_type_rows = []
    for (split_name, tag_type), group in predictions[predictions["split"].isin(["validation", "test"])].groupby(["split", "tag_type"]):
        for baseline in ["baseline_last_value", "baseline_rolling_mean_3", "baseline_rolling_mean_7", "baseline_lag_7"]:
            by_type_rows.append({"split": split_name, "tag_type": tag_type, "baseline": baseline, **metric_dict(group["target_next_count"], group[baseline])})
    by_tag_type = pd.DataFrame(by_type_rows).sort_values(["split", "tag_type", "MAE"])
    return predictions, overall, by_tag, by_tag_type


@ray.remote
def train_and_evaluate_model(model_name, train_frame, validation_frame, test_frame, numeric_features, categorical_features):
    import numpy as _np
    import pandas as _pd
    from sklearn.compose import ColumnTransformer as _ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor as _HistGradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor as _RandomForestRegressor
    from sklearn.impute import SimpleImputer as _SimpleImputer
    from sklearn.linear_model import LinearRegression as _LinearRegression
    from sklearn.linear_model import Ridge as _Ridge
    from sklearn.metrics import mean_absolute_error as _mean_absolute_error
    from sklearn.metrics import mean_squared_error as _mean_squared_error
    from sklearn.pipeline import Pipeline as _Pipeline
    from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    def _smape(actual, predicted):
        actual = _np.asarray(actual, dtype=float)
        predicted = _np.asarray(predicted, dtype=float)
        denominator = (_np.abs(actual) + _np.abs(predicted)) / 2
        values = _np.where(denominator == 0, 0.0, _np.abs(predicted - actual) / denominator * 100)
        return float(_np.mean(values))

    def _metrics(actual, predicted):
        actual = _np.asarray(actual, dtype=float)
        predicted = _np.asarray(predicted, dtype=float)
        return {
            "rows": int(len(actual)),
            "MAE": float(_mean_absolute_error(actual, predicted)),
            "RMSE": float(_np.sqrt(_mean_squared_error(actual, predicted))),
            "sMAPE": _smape(actual, predicted),
        }

    def _encoder():
        try:
            return _OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return _OneHotEncoder(handle_unknown="ignore", sparse=False)

    if model_name == "linear_regression_pooled":
        model = _LinearRegression()
    elif model_name == "ridge_regression_pooled":
        model = _Ridge(alpha=10.0)
    elif model_name == "random_forest_pooled":
        model = _RandomForestRegressor(n_estimators=80, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=1)
    elif model_name == "hist_gradient_boosting_pooled":
        model = _HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.1, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    preprocessor = _ColumnTransformer(
        [
            ("numeric", _Pipeline([("imputer", _SimpleImputer(strategy="median")), ("scaler", _StandardScaler())]), numeric_features),
            ("categorical", _Pipeline([("imputer", _SimpleImputer(strategy="most_frequent")), ("onehot", _encoder())]), categorical_features),
        ]
    )
    pipeline = _Pipeline([("preprocess", preprocessor), ("model", model)])
    feature_cols = numeric_features + categorical_features
    pipeline.fit(train_frame[feature_cols], train_frame["target_next_count"])

    outputs = []
    metric_rows = []
    for split_name, frame in [("validation", validation_frame), ("test", test_frame)]:
        predicted = _np.clip(pipeline.predict(frame[feature_cols]), 0, None)
        split_predictions = frame[["date", "tag", "tag_type", "target_next_count", "split"]].copy()
        split_predictions["model"] = model_name
        split_predictions["prediction"] = predicted
        outputs.append(split_predictions)
        metric_rows.append({"split": split_name, "model": model_name, **_metrics(frame["target_next_count"], predicted)})

    return {"model": model_name, "metrics": metric_rows, "predictions": _pd.concat(outputs, ignore_index=True)}


def run_ray_models(modeling: pd.DataFrame, num_cpus: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = modeling[modeling["split"] == "train"].copy()
    validation = modeling[modeling["split"] == "validation"].copy()
    test = modeling[modeling["split"] == "test"].copy()

    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=num_cpus, log_to_driver=False)
    try:
        futures = [
            train_and_evaluate_model.remote(
                model_name,
                train,
                validation,
                test,
                config.NUMERIC_FEATURES,
                config.CATEGORICAL_FEATURES,
            )
            for model_name in MODEL_NAMES
        ]
        results = ray.get(futures)
    finally:
        ray.shutdown()

    metrics = pd.DataFrame([row for result in results for row in result["metrics"]])
    predictions = pd.concat([result["predictions"] for result in results], ignore_index=True)
    return predictions, metrics


def summarize_by_tag(predictions: pd.DataFrame, best_model: str) -> pd.DataFrame:
    rows = []
    for (tag_type, tag), group in predictions[(predictions["split"] == "test") & (predictions["model"] == best_model)].groupby(["tag_type", "tag"]):
        rows.append({"tag_type": tag_type, "tag": tag, "model": best_model, **metric_dict(group["target_next_count"], group["prediction"])})
    return pd.DataFrame(rows).sort_values(["MAE", "sMAPE"], ascending=[False, False])


def summarize_by_tag_type(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_name, tag_type, model), group in predictions.groupby(["split", "tag_type", "model"]):
        rows.append({"split": split_name, "tag_type": tag_type, "model": model, **metric_dict(group["target_next_count"], group["prediction"])})
    return pd.DataFrame(rows).sort_values(["split", "tag_type", "sMAPE", "MAE"])


def main() -> int:
    args = parse_args()
    if not args.features.exists():
        raise FileNotFoundError(f"Missing feature table: {args.features}")

    print("Running Phase 2 baseline and Ray forecasting comparison")
    features = pd.read_parquet(args.features)
    required = ["date", "tag", "tag_type", "count", "target_next_count", "split", "modeling_ready"] + config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    require_columns(features, required, "feature table")
    modeling = features[features["modeling_ready"]].sort_values(["tag_type", "tag", "date"]).reset_index(drop=True)

    split_counts = modeling["split"].value_counts().to_dict()
    if split_counts != {"train": 1440, "validation": 330, "test": 300}:
        raise ValueError(f"Unexpected modeling split counts: {split_counts}")

    config.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    baseline_predictions, baseline_metrics, baseline_by_tag, baseline_by_type = make_baseline_predictions(modeling)
    baseline_predictions.to_parquet(config.BASELINE_PREDICTIONS, index=False)
    baseline_metrics.to_csv(config.BASELINE_METRICS_OVERALL, index=False)
    baseline_by_tag.to_csv(config.BASELINE_METRICS_BY_TAG, index=False)
    baseline_by_type.to_csv(config.BASELINE_METRICS_BY_TAG_TYPE, index=False)

    learned_predictions, learned_metrics = run_ray_models(modeling, args.num_cpus)

    baseline_for_comparison = baseline_predictions[baseline_predictions["split"].isin(["validation", "test"])][
        ["date", "tag", "tag_type", "target_next_count", "split", "baseline_last_value"]
    ].rename(columns={"baseline_last_value": "prediction"})
    baseline_for_comparison["model"] = "baseline_last_value"

    all_predictions = pd.concat(
        [learned_predictions, baseline_for_comparison[["date", "tag", "tag_type", "target_next_count", "split", "model", "prediction"]]],
        ignore_index=True,
    )
    baseline_rows = baseline_metrics.rename(columns={"baseline": "model"})[["split", "model", "rows", "MAE", "RMSE", "sMAPE"]]
    overall_metrics = pd.concat([learned_metrics, baseline_rows[baseline_rows["model"] == "baseline_last_value"]], ignore_index=True)

    validation_metrics = overall_metrics[overall_metrics["split"] == "validation"].sort_values(["sMAPE", "MAE"])
    test_metrics = overall_metrics[overall_metrics["split"] == "test"].sort_values(["sMAPE", "MAE"])
    best_model = validation_metrics[validation_metrics["model"].isin(MODEL_NAMES)].iloc[0]["model"]
    best_validation = validation_metrics[validation_metrics["model"] == best_model].iloc[0].to_dict()
    best_test = test_metrics[test_metrics["model"] == best_model].iloc[0].to_dict()
    baseline_validation = validation_metrics[validation_metrics["model"] == "baseline_last_value"].iloc[0].to_dict()
    baseline_test = test_metrics[test_metrics["model"] == "baseline_last_value"].iloc[0].to_dict()

    by_tag = summarize_by_tag(all_predictions, best_model)
    by_type = summarize_by_tag_type(all_predictions)

    all_predictions.to_parquet(config.RAY_MODEL_PREDICTIONS, index=False)
    overall_metrics.to_csv(config.RAY_MODEL_METRICS_OVERALL, index=False)
    by_tag.to_csv(config.RAY_MODEL_METRICS_BY_TAG, index=False)
    by_type.to_csv(config.RAY_MODEL_METRICS_BY_TAG_TYPE, index=False)

    run_summary = {
        "official_baseline": "baseline_last_value",
        "model_selection_metric": "validation sMAPE",
        "best_learned_model": best_model,
        "best_learned_validation_metrics": best_validation,
        "best_learned_test_metrics": best_test,
        "baseline_validation_metrics": baseline_validation,
        "baseline_test_metrics": baseline_test,
        "best_learned_beats_baseline_validation_sMAPE": bool(best_validation["sMAPE"] < baseline_validation["sMAPE"]),
        "best_learned_beats_baseline_test_sMAPE": bool(best_test["sMAPE"] < baseline_test["sMAPE"]),
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "models_compared": MODEL_NAMES,
    }
    with open(config.RAY_MODEL_RUN_SUMMARY, "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, default=str)

    expected_prediction_rows = (330 + 300) * (len(MODEL_NAMES) + 1)
    if len(all_predictions) != expected_prediction_rows:
        raise ValueError(f"Unexpected prediction row count: {len(all_predictions)}")

    print(f"Best learned model: {best_model}")
    print(f"Validation sMAPE: {best_validation['sMAPE']:.4f}; test sMAPE: {best_test['sMAPE']:.4f}")
    print(f"Saved predictions: {config.RAY_MODEL_PREDICTIONS}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
