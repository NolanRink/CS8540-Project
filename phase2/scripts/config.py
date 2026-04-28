"""Small shared configuration for the Phase 2 command-line workflow."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE2_ROOT = PROJECT_ROOT / "phase2"

SPARK_OUTPUT_DIR = PHASE2_ROOT / "data" / "spark_output" / "output"
DERIVED_DIR = PHASE2_ROOT / "data" / "derived"

DAILY_HASHTAG_COUNTS = SPARK_OUTPUT_DIR / "daily_hashtag_counts.parquet"
DAILY_CASHTAG_COUNTS = SPARK_OUTPUT_DIR / "daily_cashtag_counts.parquet"

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

START_DATE = "2020-04-09"
END_DATE = "2020-07-16"
TOP_N_PER_TAG_TYPE = 15
MIN_OBSERVED_DAYS = 60

TRAIN_DATES = 55
VALIDATION_DATES = 11

NUMERIC_FEATURES = [
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
CATEGORICAL_FEATURES = ["tag", "tag_type"]
TARGET_COLUMN = "target_next_count"
