from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode, col, to_date, date_trunc
from pyspark.sql.types import ArrayType, StringType
import re, os

spark = SparkSession.builder \
    .appName("StockTweetPipeline") \
    .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED") \
    .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv("stock_tweets.csv", header=True, inferSchema=True)
print("Total rows:", df.count())

df = df.dropna(subset=["text"]).dropDuplicates(["text"])

def extract_hashtags(text):
    return re.findall(r"#\w+", text.lower()) if text else []

def extract_cashtags(text):
    return re.findall(r"\$[A-Z]{1,5}", text.upper()) if text else []

hashtag_udf = udf(extract_hashtags, ArrayType(StringType()))
cashtag_udf = udf(extract_cashtags, ArrayType(StringType()))

df = df.withColumn("hashtags", hashtag_udf("text"))
df = df.withColumn("cashtags", cashtag_udf("text"))
df = df.withColumn("date", to_date(col("created_at")))
df = df.withColumn("week", date_trunc("week", col("date")))

os.makedirs("output", exist_ok=True)

df.select("date", explode("hashtags").alias("tag")) \
    .groupBy("date", "tag").count().orderBy("date", col("count").desc()) \
    .write.mode("overwrite").parquet("output/daily_hashtag_counts.parquet")

df.select("date", explode("cashtags").alias("tag")) \
    .groupBy("date", "tag").count().orderBy("date", col("count").desc()) \
    .write.mode("overwrite").parquet("output/daily_cashtag_counts.parquet")

df.select("week", explode("hashtags").alias("tag")) \
    .groupBy("week", "tag").count().orderBy("week", col("count").desc()) \
    .write.mode("overwrite").parquet("output/weekly_hashtag_counts.parquet")

df.select("date", "text", "hashtags", "cashtags") \
    .write.mode("overwrite").parquet("output/cleaned_tweets.parquet")

print("\nTop 10 Hashtags:")
df.select(explode("hashtags").alias("tag")).groupBy("tag").count() \
    .orderBy(col("count").desc()).show(10)

print("\nTop 10 Cashtags:")
df.select(explode("cashtags").alias("tag")).groupBy("tag").count() \
    .orderBy(col("count").desc()).show(10)

print("\nDone. Outputs saved to ./output/")
spark.stop()
