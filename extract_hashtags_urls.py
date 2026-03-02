import pandas as pd
from pathlib import Path

df = pd.read_json('out.json', lines=True)

hashtags = []
urls = []

for ent in df['entities']:
    for all_hashtags in ent.get('hashtags'):
        hashtags.append(all_hashtags.get('text'))
    for all_urls in ent.get('urls'):
        urls.append(all_urls.get('expanded_url'))

out_dir = Path("extracted")
out_dir.mkdir(exist_ok=True)

(out_dir / "hashtags.txt").write_text("\n".join(hashtags) + ("\n" if hashtags else ""), encoding="utf-8")
(out_dir / "urls.txt").write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")
(out_dir / "all.txt").write_text("\n".join(hashtags + urls) + ("\n" if (hashtags or urls) else ""), encoding="utf-8")

print(f'# of urls = {len(urls)}')
print(f'# of hashtags = {len(hashtags)}')