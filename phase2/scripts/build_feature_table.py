"""Build the daily feature table for the selected Phase 2 tags."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PHASE2_ROOT = Path(__file__).resolve().parents[1]
SPARK_OUTPUT_DIR = PHASE2_ROOT / "data" / "spark_output" / "output"
DERIVED_DIR = PHASE2_ROOT / "data" / "derived"
FEATURE_TABLE_PATH = DERIVED_DIR / "top_tags_daily_features.parquet"

START_DATE = "2020-04-09"
END_DATE = "2020-07-16"
TOP_N_PER_TAG_TYPE = 15
MIN_OBSERVED_DAYS = 60
TRAIN_DATES = 55
VALIDATION_DATES = 11
TARGET_COLUMN = "target_next_count"


def read_daily_counts(path: Path, tag_type: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {tag_type} counts: {path}")

    table = pq.read_table(path, columns=["date", "tag", "count"])
    missing = {"date", "tag", "count"} - set(table.column_names)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    raw_days = pd.Series(table["date"].cast(pa.int32()).combine_chunks().to_pylist(), dtype="float64")
    return pd.DataFrame(
        {
            "date": pd.to_datetime(raw_days, unit="D", origin="unix", errors="coerce"),
            "tag": table["tag"].to_pandas(),
            "tag_type": tag_type,
            "count": table["count"].to_pandas(),
        }
    )


def load_counts(spark_output_dir: Path) -> pd.DataFrame:
    frames = [
        read_daily_counts(spark_output_dir / "daily_hashtag_counts.parquet", "hashtag"),
        read_daily_counts(spark_output_dir / "daily_cashtag_counts.parquet", "cashtag"),
    ]
    counts = pd.concat(frames, ignore_index=True)
    counts = counts[counts["date"].between(START_DATE, END_DATE, inclusive="both")].copy()
    counts = counts[["date", "tag", "tag_type", "count"]].sort_values(["tag_type", "tag", "date"])

    if counts.empty:
        raise ValueError("No daily count rows remained after date filtering.")
    return counts.reset_index(drop=True)


def select_top_tags(daily_counts: pd.DataFrame) -> pd.DataFrame:
    summary = (
        daily_counts.groupby(["tag_type", "tag"], as_index=False)
        .agg(total_count=("count", "sum"), observed_days=("date", "nunique"))
    )
    selected = (
        summary[summary["observed_days"] >= MIN_OBSERVED_DAYS]
        .sort_values(["tag_type", "total_count", "observed_days", "tag"], ascending=[True, False, False, True])
        .groupby("tag_type", group_keys=False)
        .head(TOP_N_PER_TAG_TYPE)
        .sort_values(["tag_type", "total_count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    tag_counts = selected.groupby("tag_type")["tag"].nunique().to_dict()
    expected = {"hashtag": TOP_N_PER_TAG_TYPE, "cashtag": TOP_N_PER_TAG_TYPE}
    if tag_counts != expected:
        raise ValueError(f"Expected top tags {expected}, got {tag_counts}")
    return selected


def build_panel(daily_counts: pd.DataFrame, selected_tags: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DataFrame({"date": sorted(daily_counts["date"].dropna().unique())})
    tags = selected_tags[["tag_type", "tag"]].drop_duplicates().sort_values(["tag_type", "tag"])

    panel_index = tags.assign(_key=1).merge(dates.assign(_key=1), on="_key").drop(columns="_key")
    selected_counts = daily_counts.merge(tags, on=["tag_type", "tag"], how="inner")
    panel = panel_index.merge(selected_counts, on=["tag_type", "tag", "date"], how="left")
    panel["count"] = panel["count"].fillna(0).astype("int64")
    return panel[["date", "tag", "tag_type", "count"]].sort_values(["tag_type", "tag", "date"]).reset_index(drop=True)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    features = panel.copy()
    group_cols = ["tag_type", "tag"]
    counts = features.groupby(group_cols, sort=False)["count"]

    features[TARGET_COLUMN] = counts.shift(-1)
    for lag in [1, 2, 3, 7]:
        features[f"count_lag_{lag}"] = counts.shift(lag)

    for window in [3, 7]:
        rolling = counts.rolling(window=window, min_periods=window)
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

    anomaly_roll = counts.rolling(window=7, min_periods=7)
    features["anomaly_trailing_mean_7"] = anomaly_roll.mean().reset_index(level=group_cols, drop=True)
    features["anomaly_trailing_std_7"] = anomaly_roll.std().reset_index(level=group_cols, drop=True)
    features["anomaly_deviation_7"] = features["count"] - features["anomaly_trailing_mean_7"]
    features["anomaly_z_ready_7"] = np.where(
        features["anomaly_trailing_std_7"] > 0,
        features["anomaly_deviation_7"] / features["anomaly_trailing_std_7"],
        np.nan,
    )
    return features


def add_splits(features: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DataFrame({"date": sorted(features["date"].dropna().unique())}).reset_index(names="date_position")
    dates["split"] = np.select(
        [
            dates["date_position"] < TRAIN_DATES,
            dates["date_position"] < TRAIN_DATES + VALIDATION_DATES,
        ],
        ["train", "validation"],
        default="test",
    )

    features = features.merge(dates[["date", "split"]], on="date", how="left")
    required = [
        TARGET_COLUMN,
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
    split_counts = features.loc[features["modeling_ready"], "split"].value_counts().to_dict()
    expected_target = features.groupby(["tag_type", "tag"], sort=False)["count"].shift(-1)
    if tag_counts != {"hashtag": 15, "cashtag": 15}:
        raise ValueError(f"Unexpected selected tag counts: {tag_counts}")
    if features["date"].nunique() != 77:
        raise ValueError(f"Expected 77 observed dates, found {features['date'].nunique()}")
    if features.duplicated(["date", "tag_type", "tag"]).any():
        raise ValueError("Feature table has duplicate date/tag_type/tag rows.")
    if not np.allclose(features[TARGET_COLUMN].fillna(-1), expected_target.fillna(-1)):
        raise ValueError("Unexpected target_next_count values.")
    if split_counts != {"train": 1440, "validation": 330, "test": 300}:
        raise ValueError(f"Unexpected modeling-ready split counts: {split_counts}")


def main() -> int:
    print("Loading daily counts")
    daily_counts = load_counts(SPARK_OUTPUT_DIR)

    print("Building feature table")
    selected_tags = select_top_tags(daily_counts)
    panel = build_panel(daily_counts, selected_tags)
    features = add_features(panel)

    features = add_splits(features)
    validate_feature_table(features)

    FEATURE_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURE_TABLE_PATH, index=False)

    print(f"Saved {FEATURE_TABLE_PATH}")
    print(f"Rows: {len(features):,}; modeling-ready rows: {int(features['modeling_ready'].sum()):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
