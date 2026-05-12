#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "Setting up Python environment"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r phase2/requirements-phase2.txt
python -m pip install --force-reinstall "pyspark==3.5.1"

# Use the venv Spark, not any old system Spark install.
unset SPARK_HOME
unset PYTHONPATH
export PYSPARK_PYTHON="$(pwd)/.venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$(pwd)/.venv/bin/python"

SPARK_INPUT="phase2/data/spark_output/stock_tweets.csv"
SPARK_OUTPUT_DIR="phase2/data/spark_output/output"
FEATURE_TABLE="phase2/data/derived/top_tags_daily_features.parquet"
DATA_URL="https://huggingface.co/datasets/StephanAkkerman/stock-market-tweets-data/resolve/main/stock-market-tweets-data.csv"

mkdir -p phase2/data/spark_output phase2/data/derived "$SPARK_OUTPUT_DIR"

if [ ! -f "$SPARK_INPUT" ]; then
  echo "Downloading stock tweet data"
  curl -L "$DATA_URL" -o "$SPARK_INPUT"
fi

# Older copies of spark_pipeline.py read stock_tweets.csv from the repo root.
cp "$SPARK_INPUT" stock_tweets.csv

echo "Running Spark preprocessing"
.venv/bin/spark-submit phase2/spark_pipeline.py \
  --input "$SPARK_INPUT" \
  --output-dir "$SPARK_OUTPUT_DIR"

if [ -d output/daily_hashtag_counts.parquet ]; then
  echo "Copying Spark output into phase2 data folder"
  cp -r output/*.parquet "$SPARK_OUTPUT_DIR"/
fi

echo "Building feature table"
python phase2/scripts/build_feature_table.py \
  --spark-output-dir "$SPARK_OUTPUT_DIR" \
  --output "$FEATURE_TABLE"

echo "Running classical Ray forecasting"
python -u phase2/scripts/run_ray_forecasting.py \
  --features "$FEATURE_TABLE" \
  --run-label count_only

echo "Training Ray/PyTorch MLP forecaster"
python -u phase2/scripts/train_ray_mlp_forecaster.py \
  --features "$FEATURE_TABLE" \
  --run-label mlp_count_only \
  --num-workers 1 \
  --epochs 100 \
  --batch-size 64 \
  --use-gpu

echo "Phase 2 pipeline finished"
