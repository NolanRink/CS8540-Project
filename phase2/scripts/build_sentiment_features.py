"""Score cleaned tweets with a financial sentiment model and aggregate by day/tag."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PHASE2_ROOT = Path(__file__).resolve().parents[1]
SPARK_OUTPUT_DIR = PHASE2_ROOT / "data" / "spark_output" / "output"
DERIVED_DIR = PHASE2_ROOT / "data" / "derived"

CLEANED_TWEETS = SPARK_OUTPUT_DIR / "cleaned_tweets.parquet"
SELECTED_TAG_FEATURES = DERIVED_DIR / "top_tags_daily_features.parquet"
TWEET_OUTPUT = DERIVED_DIR / "tweet_sentiment.parquet"
DAILY_OUTPUT = DERIVED_DIR / "daily_tag_sentiment.parquet"

START_DATE = "2020-04-09"
END_DATE = "2020-07-16"
DEFAULT_MODEL = "StephanAkkerman/FinTwitBERT-sentiment"
DEFAULT_HF_CACHE_DIR = Path(os.environ.get("HF_HOME", "/mnt/project/cache/hf"))
REQUIRED_TWEET_COLUMNS = ["date", "text", "hashtags", "cashtags"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 2 tweet and daily tag sentiment features.")
    parser.add_argument("--input", type=Path, default=CLEANED_TWEETS)
    parser.add_argument("--selected-tags", type=Path, default=SELECTED_TAG_FEATURES)
    parser.add_argument("--tweet-output", type=Path, default=TWEET_OUTPUT)
    parser.add_argument("--daily-output", type=Path, default=DAILY_OUTPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional small-row smoke test limit after date filtering.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_HF_CACHE_DIR)
    parser.add_argument("--progress-every", type=int, default=500, help="Print progress every N batches; use 0 to disable.")
    return parser.parse_args()


def require_columns(columns: list[str], required: list[str], label: str) -> None:
    missing = [col for col in required if col not in columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def normalize_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist() if item is not None]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return []


def normalize_hashtag(tag: str) -> str:
    tag = str(tag).strip().lower()
    if tag and not tag.startswith("#"):
        tag = f"#{tag}"
    return tag


def normalize_cashtag(tag: str) -> str:
    tag = str(tag).strip().upper()
    if tag and not tag.startswith("$"):
        tag = f"${tag}"
    return tag


def load_cleaned_tweets(path: Path, max_rows: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing cleaned tweets parquet: {path}")

    table = pq.read_table(path, columns=REQUIRED_TWEET_COLUMNS)
    require_columns(table.column_names, REQUIRED_TWEET_COLUMNS, "cleaned tweets")

    raw_days = pd.Series(table["date"].cast(pa.int32()).combine_chunks().to_pylist(), dtype="float64")
    data = table.select(["text", "hashtags", "cashtags"]).to_pandas()
    tweets = pd.DataFrame(
        {
            "tweet_row_id": np.arange(len(data), dtype="int64"),
            "date": pd.to_datetime(raw_days, unit="D", origin="unix", errors="coerce"),
            "text": data["text"],
            "hashtags": data["hashtags"].apply(normalize_list),
            "cashtags": data["cashtags"].apply(normalize_list),
        }
    )

    tweets = tweets[tweets["date"].between(START_DATE, END_DATE, inclusive="both")].copy()
    tweets["text"] = tweets["text"].fillna("").astype(str).str.strip()
    tweets = tweets[tweets["text"] != ""].copy()
    tweets = tweets.sort_values(["date", "tweet_row_id"]).reset_index(drop=True)

    if max_rows is not None:
        if max_rows < 1:
            raise ValueError("--max-rows must be positive when provided.")
        tweets = tweets.head(max_rows).copy()

    if tweets.empty:
        raise ValueError("No cleaned tweet rows remained after date/text filtering.")
    return tweets


def load_selected_tags(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing selected tag feature table: {path}")

    selected = pd.read_parquet(path, columns=["tag", "tag_type"])
    require_columns(list(selected.columns), ["tag", "tag_type"], "selected tag feature table")
    selected = selected[["tag", "tag_type"]].drop_duplicates().copy()
    selected["tag"] = np.where(
        selected["tag_type"].eq("cashtag"),
        selected["tag"].map(normalize_cashtag),
        selected["tag"].map(normalize_hashtag),
    )

    tag_counts = selected.groupby("tag_type")["tag"].nunique().to_dict()
    if tag_counts != {"cashtag": 15, "hashtag": 15}:
        raise ValueError(f"Expected 15 selected hashtags and 15 selected cashtags, got {tag_counts}")
    return selected


def choose_device(requested: str):
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_hf_cache(cache_dir: Path) -> Path:
    cache_dir = cache_dir.expanduser().resolve()
    hub_dir = cache_dir / "hub"
    xet_dir = cache_dir / "xet"
    datasets_dir = cache_dir / "datasets"
    for path in [cache_dir, hub_dir, xet_dir, datasets_dir]:
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["HF_XET_CACHE"] = str(xet_dir)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_dir)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.pop("TRANSFORMERS_CACHE", None)
    return cache_dir


def get_label_mapping(model) -> dict[str, int]:
    labels = {int(index): str(label).lower() for index, label in model.config.id2label.items()}
    mapping: dict[str, int] = {}

    for index, label in labels.items():
        if "positive" in label or label in {"pos", "bullish", "bull"}:
            mapping["positive"] = index
        elif "negative" in label or label in {"neg", "bearish", "bear"}:
            mapping["negative"] = index
        elif "neutral" in label or label in {"neu", "none"}:
            mapping["neutral"] = index

    if set(mapping) != {"positive", "negative", "neutral"}:
        raise ValueError(
            "Could not identify positive/negative/neutral labels from model config "
            f"id2label={model.config.id2label}. Use a three-class sentiment model with clear labels."
        )
    return mapping


def run_sentiment_inference(
    texts: pd.Series,
    model_name: str,
    batch_size: int,
    device_name: str,
    max_length: int,
    cache_dir: Path,
    progress_every: int,
) -> tuple[pd.DataFrame, str]:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if max_length < 16:
        raise ValueError("--max-length is too small for tweet sentiment inference.")
    if progress_every < 0:
        raise ValueError("--progress-every cannot be negative.")

    cache_dir = prepare_hf_cache(cache_dir)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = choose_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
    model.to(device)
    model.eval()

    label_map = get_label_mapping(model)
    inverse_label_map = {index: label for label, index in label_map.items()}

    rows = []
    text_values = texts.tolist()
    total_batches = (len(text_values) + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch_number, start in enumerate(range(0, len(text_values), batch_size), start=1):
            batch_texts = text_values[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probabilities = torch.softmax(model(**encoded).logits, dim=1).cpu().numpy()

            for probs in probabilities:
                positive = float(probs[label_map["positive"]])
                neutral = float(probs[label_map["neutral"]])
                negative = float(probs[label_map["negative"]])
                best_index = int(np.argmax(probs))
                rows.append(
                    {
                        "sentiment_label": inverse_label_map[best_index],
                        "sentiment_score": positive - negative,
                        "positive_prob": positive,
                        "neutral_prob": neutral,
                        "negative_prob": negative,
                        "sentiment_confidence": float(np.max(probs)),
                    }
                )
            if progress_every and (batch_number % progress_every == 0 or batch_number == total_batches):
                print(f"Scored {min(start + batch_size, len(text_values)):,}/{len(text_values):,} tweets", flush=True)

    return pd.DataFrame(rows), str(device)


def build_tweet_sentiment(tweets: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    scores, device_used = run_sentiment_inference(
        tweets["text"],
        model_name=args.model_name,
        batch_size=args.batch_size,
        device_name=args.device,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
        progress_every=args.progress_every,
    )
    if len(scores) != len(tweets):
        raise ValueError(f"Scored row count mismatch: {len(scores)} scores for {len(tweets)} tweets.")

    tweet_sentiment = pd.concat(
        [
            tweets[["tweet_row_id", "date", "hashtags", "cashtags"]].reset_index(drop=True),
            scores.reset_index(drop=True),
        ],
        axis=1,
    )
    if tweet_sentiment.empty:
        raise ValueError("Tweet sentiment output is empty.")
    return tweet_sentiment, device_used


def explode_tag_rows(tweet_sentiment: pd.DataFrame, selected_tags: pd.DataFrame) -> pd.DataFrame:
    selected_hashtags = set(selected_tags.loc[selected_tags["tag_type"].eq("hashtag"), "tag"])
    selected_cashtags = set(selected_tags.loc[selected_tags["tag_type"].eq("cashtag"), "tag"])

    base_cols = [
        "date",
        "sentiment_label",
        "sentiment_score",
        "positive_prob",
        "neutral_prob",
        "negative_prob",
        "sentiment_confidence",
    ]

    hashtags = tweet_sentiment[base_cols + ["hashtags"]].copy()
    hashtags["tag"] = hashtags["hashtags"].apply(lambda values: [normalize_hashtag(tag) for tag in values])
    hashtags = hashtags.drop(columns="hashtags").explode("tag")
    hashtags = hashtags[hashtags["tag"].isin(selected_hashtags)].copy()
    hashtags["tag_type"] = "hashtag"

    cashtags = tweet_sentiment[base_cols + ["cashtags"]].copy()
    cashtags["tag"] = cashtags["cashtags"].apply(lambda values: [normalize_cashtag(tag) for tag in values])
    cashtags = cashtags.drop(columns="cashtags").explode("tag")
    cashtags = cashtags[cashtags["tag"].isin(selected_cashtags)].copy()
    cashtags["tag_type"] = "cashtag"

    tagged = pd.concat([hashtags, cashtags], ignore_index=True)
    if tagged.empty:
        raise ValueError("No sentiment rows matched the selected forecasting tags.")
    return tagged


def aggregate_daily_sentiment(tagged: pd.DataFrame) -> pd.DataFrame:
    tagged = tagged.copy()
    tagged["is_positive"] = tagged["sentiment_label"].eq("positive").astype(float)
    tagged["is_negative"] = tagged["sentiment_label"].eq("negative").astype(float)
    tagged["is_neutral"] = tagged["sentiment_label"].eq("neutral").astype(float)

    daily = (
        tagged.groupby(["date", "tag", "tag_type"], as_index=False)
        .agg(
            sentiment_tweet_count=("sentiment_score", "size"),
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_median=("sentiment_score", "median"),
            sentiment_std=("sentiment_score", "std"),
            positive_share=("is_positive", "mean"),
            negative_share=("is_negative", "mean"),
            neutral_share=("is_neutral", "mean"),
            avg_sentiment_confidence=("sentiment_confidence", "mean"),
        )
        .sort_values(["tag_type", "tag", "date"])
        .reset_index(drop=True)
    )
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)

    if daily.duplicated(["date", "tag", "tag_type"]).any():
        raise ValueError("Daily sentiment output has duplicate date/tag/tag_type rows.")
    return daily


def save_outputs(tweet_sentiment: pd.DataFrame, daily_sentiment: pd.DataFrame, args: argparse.Namespace) -> None:
    args.tweet_output.parent.mkdir(parents=True, exist_ok=True)
    args.daily_output.parent.mkdir(parents=True, exist_ok=True)

    tweet_sentiment.to_parquet(args.tweet_output, index=False)
    daily_sentiment.to_parquet(args.daily_output, index=False)

    if not args.tweet_output.exists():
        raise FileNotFoundError(f"Tweet sentiment output was not saved: {args.tweet_output}")
    if not args.daily_output.exists():
        raise FileNotFoundError(f"Daily sentiment output was not saved: {args.daily_output}")


def main() -> int:
    args = parse_args()

    print("Loading cleaned tweets...")
    tweets = load_cleaned_tweets(args.input, args.max_rows)
    selected_tags = load_selected_tags(args.selected_tags)
    print(f"Tweets to score: {len(tweets):,}")

    print(f"Running sentiment model: {args.model_name}")
    print(f"Using Hugging Face cache: {args.cache_dir}")
    tweet_sentiment, device_used = build_tweet_sentiment(tweets, args)

    print("Aggregating sentiment by day and tag...")
    tagged = explode_tag_rows(tweet_sentiment, selected_tags)
    daily_sentiment = aggregate_daily_sentiment(tagged)

    save_outputs(tweet_sentiment, daily_sentiment, args)

    print(f"Device used: {device_used}")
    print(f"Saved tweet sentiment rows: {len(tweet_sentiment):,} -> {args.tweet_output}")
    print(f"Saved daily tag sentiment rows: {len(daily_sentiment):,} -> {args.daily_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
