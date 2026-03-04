import json, os

INPUT_FILE = "out.json"
os.makedirs("phase1/extracted", exist_ok=True)

hashtags, urls = [], []

def extract(tweet):
    # check both entity blocks
    for ent in [tweet.get("entities") or {}, (tweet.get("extended_entities") or {})]:

        # collect each hashtag text
        for h in ent.get("hashtags", [] or []):
            if h.get("text"): hashtags.append(f'#{h["text"].lower()}')

        # collect each expanded URL
        for u in ent.get("urls", [] or []):
            if u.get("expanded_url"): urls.append(u["expanded_url"])

        # collect each media file URL
        for m in ent.get("media", [] or []):
            if m.get("expanded_url"): urls.append(m["expanded_url"])
        
        # extract from nested tweets
    if tweet.get("retweeted_status"): extract(tweet["retweeted_status"])
    if tweet.get("quoted_status"):    extract(tweet["quoted_status"])

with open(INPUT_FILE, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            extract(json.loads(line))

open("phase1/extracted/hashtags.txt", "w").write("\n".join(hashtags))
open("phase1/extracted/urls.txt",     "w").write("\n".join(urls))
open("phase1/extracted/combined.txt", "w").write("\n".join(hashtags + urls))


print(f"Hashtags: \t{len(hashtags)}")
print(f"URLs: \t\t{len(urls)}")