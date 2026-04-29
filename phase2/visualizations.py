import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("plots", exist_ok=True)

# Load data
daily_h = pd.read_parquet("output/daily_hashtag_counts.parquet")
daily_c = pd.read_parquet("output/daily_cashtag_counts.parquet")

# 1. Bar chart - Top 20 Hashtags
top_hashtags = daily_h.groupby("tag")["count"].sum().sort_values(ascending=False).head(20)
plt.figure(figsize=(14,6))
sns.barplot(x=top_hashtags.values, y=top_hashtags.index, palette="Blues_r")
plt.title("Top 20 Hashtags in Stock Market Tweets")
plt.xlabel("Total Count")
plt.ylabel("Hashtag")
plt.tight_layout()
plt.savefig("plots/top20_hashtags.png", dpi=150)
plt.close()
print("Saved top20_hashtags.png")

# 2. Pie chart - Top 10 Cashtags
top_cashtags = daily_c.groupby("tag")["count"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,8))
plt.pie(top_cashtags.values, labels=top_cashtags.index, autopct='%1.1f%%', startangle=140)
plt.title("Top 10 Cashtags Distribution")
plt.tight_layout()
plt.savefig("plots/top10_cashtags_pie.png", dpi=150)
plt.close()
print("Saved top10_cashtags_pie.png")

# 3. Line chart - Daily tweet activity for top 5 hashtags
top5 = daily_h.groupby("tag")["count"].sum().sort_values(ascending=False).head(5).index.tolist()
filtered = daily_h[daily_h["tag"].isin(top5)].copy()
filtered["date"] = pd.to_datetime(filtered["date"])
pivot = filtered.pivot_table(index="date", columns="tag", values="count", fill_value=0)
plt.figure(figsize=(14,6))
pivot.plot(ax=plt.gca())
plt.title("Daily Trend of Top 5 Hashtags Over Time")
plt.xlabel("Date")
plt.ylabel("Count")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("plots/hashtag_trends.png", dpi=150)
plt.close()
print("Saved hashtag_trends.png")

# 4. Heatmap - Top 10 hashtags by week
weekly_h = pd.read_parquet("output/weekly_hashtag_counts.parquet")
top10 = weekly_h.groupby("tag")["count"].sum().sort_values(ascending=False).head(10).index.tolist()
filtered_w = weekly_h[weekly_h["tag"].isin(top10)].copy()
filtered_w["week"] = pd.to_datetime(filtered_w["week"]).dt.strftime("%Y-%m-%d")
pivot_w = filtered_w.pivot_table(index="tag", columns="week", values="count", fill_value=0)
plt.figure(figsize=(14,6))
sns.heatmap(pivot_w, cmap="YlOrRd", linewidths=0.5, annot=False)
plt.title("Weekly Hashtag Activity Heatmap")
plt.xlabel("Week")
plt.ylabel("Hashtag")
plt.tight_layout()
plt.savefig("plots/hashtag_heatmap.png", dpi=150)
plt.close()
print("Saved hashtag_heatmap.png")

print("\nAll 4 plots saved to ./plots/")
