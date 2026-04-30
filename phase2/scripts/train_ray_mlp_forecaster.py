"""Train a small Ray Train PyTorch MLP for next-day tag-count forecasting."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ray import train
from ray.exceptions import RayTaskError
from ray.train import ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer


PHASE2_ROOT = Path(__file__).resolve().parents[1]
DERIVED_DIR = PHASE2_ROOT / "data" / "derived"
TARGET_COLUMN = "target_next_count"
MODEL_NAME = "ray_train_mlp"

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
CATEGORICAL_FEATURES = ["tag", "tag_type"]


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Ray Train MLP forecaster.")
    parser.add_argument("--features", type=Path, default=DERIVED_DIR / "top_tags_daily_features.parquet")
    parser.add_argument("--run-label", default="mlp_count_only")
    parser.add_argument("--include-sentiment", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def smape_percent(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    values = np.where(denominator == 0, 0.0, np.abs(predicted - actual) / denominator * 100)
    return float(np.mean(values))


def metric_row(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict[str, float | int]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    return {
        "rows": int(len(actual_values)),
        "MAE": float(np.mean(np.abs(predicted_values - actual_values))),
        "RMSE": float(np.sqrt(np.mean((predicted_values - actual_values) ** 2))),
        "sMAPE": smape_percent(actual_values, predicted_values),
    }


def output_paths(run_label: str) -> dict[str, Path]:
    return {
        "predictions": DERIVED_DIR / f"ray_train_mlp_predictions_{run_label}.parquet",
        "metrics": DERIVED_DIR / f"ray_train_mlp_metrics_{run_label}.csv",
        "metrics_by_tag": DERIVED_DIR / f"ray_train_mlp_metrics_by_tag_{run_label}.csv",
        "metrics_by_tag_type": DERIVED_DIR / f"ray_train_mlp_metrics_by_tag_type_{run_label}.csv",
        "history": DERIVED_DIR / f"ray_train_mlp_training_history_{run_label}.csv",
        "training_time": DERIVED_DIR / f"ray_train_mlp_training_time_{run_label}.csv",
        "summary": DERIVED_DIR / f"ray_train_mlp_run_summary_{run_label}.json",
    }


def check_args(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.run_label):
        raise ValueError("--run-label may contain only letters, numbers, and underscores.")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1.")
    if args.num_workers != 1:
        raise ValueError("This small course-project MLP uses --num-workers 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.hidden_size < 4:
        raise ValueError("--hidden-size must be at least 4.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1).")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")


def load_modeling_frame(feature_path: Path, include_sentiment: bool) -> tuple[pd.DataFrame, list[str]]:
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature table: {feature_path}")

    features = pd.read_parquet(feature_path)
    numeric_features = BASE_NUMERIC_FEATURES.copy()
    if include_sentiment:
        numeric_features += SENTIMENT_NUMERIC_FEATURES

    required = [
        "date",
        "tag",
        "tag_type",
        TARGET_COLUMN,
        "split",
        "modeling_ready",
        *numeric_features,
        *CATEGORICAL_FEATURES,
    ]
    missing = [col for col in required if col not in features.columns]
    if missing:
        raise ValueError(f"feature table is missing columns: {missing}")

    modeling = (
        features.loc[features["modeling_ready"]]
        .copy()
        .sort_values(["tag_type", "tag", "date"])
        .reset_index(drop=True)
    )
    split_counts = modeling["split"].value_counts().to_dict()
    expected_splits = {"train": 1440, "validation": 330, "test": 300}
    if split_counts != expected_splits:
        raise ValueError(f"Unexpected modeling split counts: {split_counts}")
    return modeling, numeric_features


def prepare_arrays(modeling: pd.DataFrame, numeric_features: list[str]) -> dict[str, Any]:
    train_mask = modeling["split"].eq("train")
    train_frame = modeling.loc[train_mask].copy()

    numeric = modeling[numeric_features].replace([np.inf, -np.inf], np.nan)
    medians = train_frame[numeric_features].replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    numeric = numeric.fillna(medians)

    means = numeric.loc[train_mask].mean()
    stds = numeric.loc[train_mask].std(ddof=0).replace(0, 1).fillna(1.0)
    numeric_scaled = (numeric - means) / stds

    categories = pd.get_dummies(modeling[CATEGORICAL_FEATURES].astype(str), columns=CATEGORICAL_FEATURES)
    feature_frame = pd.concat([numeric_scaled, categories.astype(float)], axis=1)

    y = modeling[TARGET_COLUMN].astype(float).to_numpy()
    y_train = y[train_mask.to_numpy()]
    target_mean = float(np.mean(y_train))
    target_std = float(np.std(y_train))
    if target_std == 0:
        target_std = 1.0
    y_scaled = (y - target_mean) / target_std

    arrays: dict[str, Any] = {
        "feature_names": feature_frame.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_mean": target_mean,
        "target_std": target_std,
        "rows": modeling[["date", "tag", "tag_type", "split", TARGET_COLUMN]].copy(),
    }

    x_values = feature_frame.to_numpy(dtype="float32")
    y_values = y_scaled.astype("float32")
    for split_name in ["train", "validation", "test"]:
        mask = modeling["split"].eq(split_name).to_numpy()
        arrays[f"x_{split_name}"] = x_values[mask]
        arrays[f"y_{split_name}"] = y_values[mask]
        arrays[f"actual_{split_name}"] = y[mask]
    return arrays


def find_classical_reference() -> dict[str, Any] | None:
    path = DERIVED_DIR / "ray_model_metrics_overall_count_only.csv"
    if not path.exists():
        return None
    metrics = pd.read_csv(path)
    learned = metrics[(metrics["split"] == "test") & (metrics["model"] != "baseline_last_value")].copy()
    if learned.empty:
        return None
    best = learned.sort_values(["sMAPE", "MAE"]).iloc[0]
    return {
        "path": str(path),
        "best_test_model": str(best["model"]),
        "best_test_MAE": float(best["MAE"]),
        "best_test_RMSE": float(best["RMSE"]),
        "best_test_sMAPE": float(best["sMAPE"]),
    }


def summarize_by_tag(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test_predictions = predictions[predictions["split"].eq("test")]
    for (tag_type, tag), group in test_predictions.groupby(["tag_type", "tag"]):
        rows.append(
            {
                "tag_type": tag_type,
                "tag": tag,
                "model": MODEL_NAME,
                **metric_row(group[TARGET_COLUMN], group["prediction"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["MAE", "sMAPE"], ascending=[False, False])


def summarize_by_tag_type(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_name, tag_type), group in predictions.groupby(["split", "tag_type"]):
        rows.append(
            {
                "split": split_name,
                "tag_type": tag_type,
                "model": MODEL_NAME,
                **metric_row(group[TARGET_COLUMN], group["prediction"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "tag_type", "sMAPE", "MAE"])


def train_loop(config: dict[str, Any]) -> None:
    import ray.train.torch as ray_torch

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = ray_torch.get_device()
    if config["use_gpu"] and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("Ray Train was asked to use GPU, but CUDA is not available in the worker.")

    x_train = torch.tensor(config["x_train"], dtype=torch.float32)
    y_train = torch.tensor(config["y_train"], dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    train_loader = ray_torch.prepare_data_loader(train_loader)

    model = MLPRegressor(config["input_dim"], config["hidden_size"], config["dropout"]).to(device)
    if config["use_gpu"]:
        model = ray_torch.prepare_model(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = nn.SmoothL1Loss()

    def predict_counts(x_values: np.ndarray) -> np.ndarray:
        model.eval()
        x_tensor = torch.tensor(x_values, dtype=torch.float32, device=device)
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(x_tensor), 2048):
                batch = x_tensor[start : start + 2048]
                scaled = model(batch).detach().cpu().numpy()
                predictions.append(scaled)
        raw = np.concatenate(predictions) * config["target_std"] + config["target_mean"]
        return np.clip(raw, 0, None)

    history = []
    best_smape = float("inf")
    best_state = None
    start_time = time.perf_counter()

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_pred = predict_counts(config["x_validation"])
        validation_metrics = metric_row(config["actual_validation"], validation_pred)
        elapsed = time.perf_counter() - start_time
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_MAE": validation_metrics["MAE"],
            "validation_RMSE": validation_metrics["RMSE"],
            "validation_sMAPE": validation_metrics["sMAPE"],
            "elapsed_seconds": elapsed,
        }
        history.append(row)

        if validation_metrics["sMAPE"] < best_smape:
            best_smape = validation_metrics["sMAPE"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

        train.report(
            {
                "epoch": epoch,
                "train_loss": row["train_loss"],
                "validation_MAE": row["validation_MAE"],
                "validation_RMSE": row["validation_RMSE"],
                "validation_sMAPE": row["validation_sMAPE"],
            }
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    total_seconds = time.perf_counter() - start_time
    validation_pred = predict_counts(config["x_validation"])
    test_pred = predict_counts(config["x_test"])
    validation_metrics = metric_row(config["actual_validation"], validation_pred)
    test_metrics = metric_row(config["actual_test"], test_pred)

    metrics = pd.DataFrame(
        [
            {"split": "validation", "model": MODEL_NAME, **validation_metrics},
            {"split": "test", "model": MODEL_NAME, **test_metrics},
        ]
    )

    rows = config["rows"].copy()
    validation_predictions = rows[rows["split"].eq("validation")].copy()
    validation_predictions["model"] = MODEL_NAME
    validation_predictions["prediction"] = validation_pred
    test_predictions = rows[rows["split"].eq("test")].copy()
    test_predictions["model"] = MODEL_NAME
    test_predictions["prediction"] = test_pred
    predictions = pd.concat([validation_predictions, test_predictions], ignore_index=True)
    metrics_by_tag = summarize_by_tag(predictions)
    metrics_by_tag_type = summarize_by_tag_type(predictions)

    history_frame = pd.DataFrame(history)
    best_epoch = int(history_frame.sort_values(["validation_sMAPE", "validation_MAE"]).iloc[0]["epoch"])
    training_time = pd.DataFrame(
        [
            {
                "run_label": config["run_label"],
                "model": MODEL_NAME,
                "feature_set": "count_plus_sentiment" if config["include_sentiment"] else "count_only",
                "train_seconds": float(total_seconds),
                "epochs": int(config["epochs"]),
                "batch_size": int(config["batch_size"]),
                "num_workers": int(config["num_workers"]),
                "use_gpu": bool(config["use_gpu"]),
                "device_used": device.type,
            }
        ]
    )

    output_paths = {key: Path(value) for key, value in config["output_paths"].items()}
    output_paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_paths["predictions"], index=False)
    metrics.to_csv(output_paths["metrics"], index=False)
    metrics_by_tag.to_csv(output_paths["metrics_by_tag"], index=False)
    metrics_by_tag_type.to_csv(output_paths["metrics_by_tag_type"], index=False)
    history_frame.to_csv(output_paths["history"], index=False)
    training_time.to_csv(output_paths["training_time"], index=False)

    summary = {
        "run_label": config["run_label"],
        "features": config["features"],
        "include_sentiment": bool(config["include_sentiment"]),
        "model": MODEL_NAME,
        "gpu_requested": bool(config["use_gpu"]),
        "device_used": device.type,
        "cuda_available_in_worker": bool(torch.cuda.is_available()),
        "model_params": {
            "input_dim": int(config["input_dim"]),
            "hidden_size": int(config["hidden_size"]),
            "dropout": float(config["dropout"]),
            "epochs": int(config["epochs"]),
            "batch_size": int(config["batch_size"]),
            "learning_rate": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
            "loss": "SmoothL1Loss on standardized target_next_count",
        },
        "best_epoch": best_epoch,
        "best_validation_metrics": {"split": "validation", "model": MODEL_NAME, **validation_metrics},
        "test_metrics": {"split": "test", "model": MODEL_NAME, **test_metrics},
        "total_training_seconds": float(total_seconds),
        "rows": {
            "train": int(len(config["x_train"])),
            "validation": int(len(config["x_validation"])),
            "test": int(len(config["x_test"])),
        },
        "numeric_features": config["numeric_features"],
        "categorical_features": config["categorical_features"],
        "feature_count_after_encoding": int(config["input_dim"]),
        "classical_count_only_reference": config["classical_reference"],
        "outputs": {key: str(value) for key, value in output_paths.items()},
    }
    with open(output_paths["summary"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    train.report(
        {
            "done": 1,
            "best_epoch": best_epoch,
            "device_used": device.type,
            "validation_MAE": validation_metrics["MAE"],
            "validation_RMSE": validation_metrics["RMSE"],
            "validation_sMAPE": validation_metrics["sMAPE"],
            "test_MAE": test_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_sMAPE": test_metrics["sMAPE"],
            "total_training_seconds": float(total_seconds),
        }
    )


def is_local_shutdown_warning(error: RayTaskError, paths: dict[str, Path], run_started_at: float) -> bool:
    message = str(error)
    outputs_exist = all(path.exists() and path.stat().st_mtime >= run_started_at for path in paths.values())
    return outputs_exist and "_shutdown_torch" in message and "Expected a cuda device, but got: cpu" in message


def main() -> int:
    args = parse_args()
    check_args(args)
    os.environ.setdefault("USE_LIBUV", "0")
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    if not args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.use_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run without --use-gpu for CPU smoke testing.")

    print("Loading features")
    modeling, numeric_features = load_modeling_frame(args.features, args.include_sentiment)
    arrays = prepare_arrays(modeling, numeric_features)
    paths = output_paths(args.run_label)
    split_counts = modeling["split"].value_counts().reindex(["train", "validation", "test"]).to_dict()

    print(f"Rows by split: {split_counts}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Encoded feature count: {len(arrays['feature_names'])}")
    print(f"Ray Train GPU requested: {args.use_gpu}")

    train_config = {
        **arrays,
        "run_label": args.run_label,
        "features": str(args.features),
        "include_sentiment": bool(args.include_sentiment),
        "input_dim": len(arrays["feature_names"]),
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "use_gpu": bool(args.use_gpu),
        "output_paths": {key: str(value) for key, value in paths.items()},
        "classical_reference": find_classical_reference(),
    }

    print("Training Ray Train MLP")
    run_started_at = time.time()
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config=train_config,
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=args.use_gpu),
        torch_config=TorchConfig(backend="gloo"),
    )
    try:
        result = trainer.fit()
        result_metrics = result.metrics
    except RayTaskError as error:
        if not is_local_shutdown_warning(error, paths, run_started_at):
            raise
        with open(paths["summary"], "r", encoding="utf-8") as handle:
            saved_summary = json.load(handle)
        result_metrics = {
            "device_used": saved_summary.get("device_used", "unknown"),
            "validation_sMAPE": saved_summary["best_validation_metrics"]["sMAPE"],
            "test_sMAPE": saved_summary["test_metrics"]["sMAPE"],
        }
        print("Ray Train saved outputs, then hit the known local Windows Torch shutdown warning.")

    print(f"Device used: {result_metrics.get('device_used', 'unknown')}")
    print(f"Validation sMAPE: {result_metrics['validation_sMAPE']:.4f}")
    print(f"Test sMAPE: {result_metrics['test_sMAPE']:.4f}")
    print(f"Saved predictions to {paths['predictions']}")
    print(f"Saved metrics to {paths['metrics']}")
    print(f"Saved by-tag metrics to {paths['metrics_by_tag']}")
    print(f"Saved by-tag-type metrics to {paths['metrics_by_tag_type']}")
    print(f"Saved training history to {paths['history']}")
    print(f"Saved training time to {paths['training_time']}")
    print(f"Saved run summary to {paths['summary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
