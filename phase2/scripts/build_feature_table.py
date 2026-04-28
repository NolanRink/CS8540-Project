"""Build the selected-tag daily feature table for Phase 2 forecasting.

This script is the command-line equivalent of Milestones 1 and 2 in the
development notebook. It reads the Spark parquet outputs, selects dense top-N
hashtags/cashtags, builds a complete selected-tag panel, adds causal features,
and saves the derived feature table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 2 selected-tag feature table.")
    parser.add_argument("--spark-output-dir", type=Path, default=config.SPARK_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=config.FEATURE_TABLE)
    return parser.parse_args()


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def read_daily_counts(path: Path, tag_type: str) -> pd.DataFrame:
    require_path(path, f"{tag_type} daily parquet")
    table = pq.read_table(path, columns=["date", "tag", "count"])
    missing = {"date", "tag", "count"} - set(table.column_names)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    raw_days = pd.Series(table["date"].cast(pa.int32()).combine_chunks().to_pylist(), dtype="float64")
    dates = pd.to_datetime(raw_days, unit="D", origin="unix", errors="coerce")
    return pd.DataFrame(
        {
            "date": dates,
            "tag": table["tag"].to_pandas(),
            "tag_type": tag_type,
            "count": table["count"].to_pandas(),
        }
    )


def load_and_filter_counts(spark_output_dir: Path) -> pd.DataFrame:
    start = pd.Timestamp(config.START_DATE)
    end = pd.Timestamp(config.END_DATE)
    frames = [
        read_daily_counts(spark_output_dir / "daily_hashtag_counts.parquet", "hashtag"),
        read_daily_counts(spark_output_dir / "daily_cashtag_counts.parquet", "cashtag"),
    ]
    raw = pd.concat(frames, ignore_index=True)
    filtered = raw.loc[raw["date"].between(start, end, inclusive="both")].copy()
    filtered = filtered[["date", "tag", "tag_type", "count"]]
    filtered = filtered.sort_values(["tag_type", "tag", "date"]).reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No rows remained after date filtering.")
    return filtered


def select_top_tags(daily_counts: pd.DataFrame) -> pd.DataFrame:
    tag_summary = (
        daily_counts.groupby(["tag_type", "tag"], as_index=False)
        .agg(total_count=("count", "sum"), observed_days=("date", "nunique"), row_count=("count", "size"))
    )
    eligible = tag_summary[tag_summary["observed_days"] >= config.MIN_OBSERVED_DAYS].copy()
    selected = (
        eligible.sort_values(
            ["tag_type", "total_count", "observed_days", "tag"],
            ascending=[True, False, False, True],
        )
        .groupby("tag_type", group_keys=False)
        .head(config.TOP_N_PER_TAG_TYPE)
        .sort_values(["tag_type", "total_count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    counts = selected.groupby("tag_type")["tag"].nunique().to_dict()
    if counts.get("hashtag", 0) != config.TOP_N_PER_TAG_TYPE or counts.get("cashtag", 0) != config.TOP_N_PER_TAG_TYPE:
        raise ValueError(f"Expected 15 hashtags and 15 cashtags, got {counts}")
    return selected


def build_complete_panel(daily_counts: pd.DataFrame, selected_tags: pd.DataFrame) -> pd.DataFrame:
    observed_dates = pd.Index(sorted(daily_counts["date"].dropna().unique()))
    selected_keys = selected_tags[["tag_type", "tag"]].drop_duplicates().sort_values(["tag_type", "tag"])
    panel_index = (
        selected_keys.assign(_key=1)
        .merge(pd.DataFrame({"date": observed_dates}).assign(_key=1), on="_key")
        .drop(columns="_key")
        .sort_values(["tag_type", "tag", "date"])
        .reset_index(drop=True)
    )
    selected_counts = daily_counts.merge(selected_keys, on=["tag_type", "tag"], how="inner")
    panel = panel_index.merge(selected_counts, on=["tag_type", "tag", "date"], how="left")
    panel["count"] = panel["count"].fillna(0).astype("int64")
    return panel[["date", "tag", "tag_type", "count"]].sort_values(["tag_type", "tag", "date"]).reset_index(drop=True)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    features = panel.copy()
    group_cols = ["tag_type", "tag"]
    grouped_count = features.groupby(group_cols, sort=False)["count"]

    features["target_next_count"] = grouped_count.shift(-1)
    for lag in [1, 2, 3, 7]:
        features[f"count_lag_{lag}"] = grouped_count.shift(lag)

    for window in [3, 7]:
        rolling = grouped_count.rolling(window=window, min_periods=window)
        features[f"rolling_mean_{window}"] = rolling.mean().reset_index(level=group_cols, drop=True)
        features[f"rolling_std_{window}"] = rolling.std().reset_index(level=group_cols, drop=True)

    features["count_diff_1"] = features["count"] - features["count_lag_1"]
    features["count_pct_change_1"] = np.where(
        features["count_lag_1"].isna(),
        np.nan,
        np.where(features["count_lag_1"] == 0, 0.0, features["count_diff_1"] / features["count_lag_1"]),
    )
    features["day_of_week"] = features["date"].dt.dayofweek
    features["day_index"] = features.groupby(group_cols).cumcount()

    anomaly_roll = grouped_count.rolling(window=7, min_periods=7)
    features["anomaly_trailing_mean_7"] = anomaly_roll.mean().reset_index(level=group_cols, drop=True)
    features["anomaly_trailing_std_7"] = anomaly_roll.std().reset_index(level=group_cols, drop=True)
    features["anomaly_deviation_7"] = features["count"] - features["anomaly_trailing_mean_7"]
    features["anomaly_z_ready_7"] = np.where(
        features["anomaly_trailing_std_7"] > 0,
        features["anomaly_deviation_7"] / features["anomaly_trailing_std_7"],
        np.nan,
    )
    return features


def add_split_and_ready_flag(features: pd.DataFrame) -> pd.DataFrame:
    observed_dates = pd.Index(sorted(features["date"].dropna().unique()))
    split_dates = pd.DataFrame({"date": observed_dates}).reset_index(names="date_position")
    split_dates["split"] = np.select(
        [
            split_dates["date_position"] < config.TRAIN_DATES,
            split_dates["date_position"] < config.TRAIN_DATES + config.VALIDATION_DATES,
        ],
        ["train", "validation"],
        default="test",
    )
    features = features.merge(split_dates[["date", "split"]], on="date", how="left")
    required = [
        config.TARGET_COLUMN,
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
        "day_of_week",
        "day_index",
    ]
    features["modeling_ready"] = features[required].notna().all(axis=1)
    return features


def validate_feature_table(features: pd.DataFrame) -> None:
    tag_counts = features[["tag_type", "tag"]].drop_duplicates().groupby("tag_type").size().to_dict()
    observed_dates = features["date"].nunique()
    split_counts = features.loc[features["modeling_ready"], "split"].value_counts().to_dict()
    if tag_counts.get("hashtag") != 15 or tag_counts.get("cashtag") != 15:
        raise ValueError(f"Unexpected selected tag counts: {tag_counts}")
    if observed_dates != 77:
        raise ValueError(f"Expected 77 observed dates, found {observed_dates}")
    if features.duplicated(["date", "tag_type", "tag"]).any():
        raise ValueError("Feature table has duplicate date/tag_type/tag rows.")
    expected_splits = {"train": 1440, "validation": 330, "test": 300}
    if split_counts != expected_splits:
        raise ValueError(f"Unexpected modeling-ready split counts: {split_counts}")


def main() -> int:
    args = parse_args()
    print("Building Phase 2 selected-tag feature table")
    daily_counts = load_and_filter_counts(args.spark_output_dir)
    selected_tags = select_top_tags(daily_counts)
    panel = build_complete_panel(daily_counts, selected_tags)
    features = add_split_and_ready_flag(add_features(panel))
    validate_feature_table(features)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    print(f"Saved feature table: {args.output}")
    print(f"Rows: {len(features)}; modeling-ready rows: {int(features['modeling_ready'].sum())}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
