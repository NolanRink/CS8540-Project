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

# The Spark script reads stock_tweets.csv from the repo root unless SPARK_INPUT is changed.
SPARK_INPUT="${SPARK_INPUT:-stock_tweets.csv}"
SPARK_OUTPUT_DIR="phase2/data/spark_output/output"
DATA_URL="https://huggingface.co/datasets/StephanAkkerman/stock-market-tweets-data/resolve/main/stock-market-tweets-data.csv"

if [ ! -f "$SPARK_INPUT" ]; then
  echo "Downloading stock tweet data"
  mkdir -p "$(dirname "$SPARK_INPUT")"
  curl -L "$DATA_URL" -o "$SPARK_INPUT"
fi

echo "Running Spark preprocessing"
.venv/bin/spark-submit phase2/spark_pipeline.py \
  --input "$SPARK_INPUT" \
  --output-dir "$SPARK_OUTPUT_DIR"

echo "Building feature table"
python phase2/scripts/build_feature_table.py

echo "Running classical Ray forecasting"
python -u phase2/scripts/run_ray_forecasting.py

echo "Training Ray/PyTorch MLP forecaster"
python -u phase2/scripts/train_ray_mlp_forecaster.py

echo "Phase 2 pipeline finished"
