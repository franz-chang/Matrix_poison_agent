from pathlib import Path
import re
import numpy as np
import pandas as pd

proj = Path('/root/matrix-poison-camel')

# -----------------------------
# LastFM (HetRec 2011 LastFM 2K)
# -----------------------------
lastfm_raw = proj / 'data/raw/lastfm/hetrec2011-lastfm-2k'
lastfm_out = proj / 'data/lastfm'
lastfm_out.mkdir(parents=True, exist_ok=True)

ua = pd.read_csv(lastfm_raw / 'user_artists.dat', sep='\t')
ua = ua.rename(columns={'userID': 'user_id', 'artistID': 'item_id', 'weight': 'weight'})
ua = ua[['user_id', 'item_id', 'weight']].copy()

w = ua['weight'].clip(lower=1).astype(float)
logw = np.log1p(w)
if float(logw.max()) > float(logw.min()):
    ua['rating'] = 1.0 + 4.0 * (logw - logw.min()) / (logw.max() - logw.min())
else:
    ua['rating'] = 5.0
ua['rating'] = ua['rating'].clip(1.0, 5.0).round(4)

ua = ua.sort_values(['user_id', 'weight', 'item_id'], ascending=[True, False, True]).reset_index(drop=True)
ua['timestamp'] = 1_500_000_000 + ua.groupby('user_id').cumcount().astype(int)
inter_lastfm = ua[['user_id', 'item_id', 'rating', 'timestamp']]
inter_lastfm.to_csv(lastfm_out / 'interactions.csv', index=False)

# Build artist category from dominant tag.
# tags.dat in latin1 encoding on this mirror.
tags = pd.read_csv(lastfm_raw / 'tags.dat', sep='\t', encoding='latin1')
tags = tags.rename(columns={'tagID': 'tag_id', 'tagValue': 'tag_value'})
uta = pd.read_csv(lastfm_raw / 'user_taggedartists-timestamps.dat', sep='\t')
uta = uta.rename(columns={'artistID': 'item_id', 'tagID': 'tag_id'})

merged = uta[['item_id', 'tag_id']].merge(tags[['tag_id', 'tag_value']], on='tag_id', how='left')

def clean_tag(x: str) -> str:
    if pd.isna(x):
        return 'UNKNOWN'
    s = str(x).strip().lower()
    s = re.sub(r'[^a-z0-9\s\-_/]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if not s:
        return 'UNKNOWN'
    return s[:40]

merged['category'] = merged['tag_value'].map(clean_tag)
merged = merged[(merged['category'] != 'UNKNOWN') & (merged['category'].str.len() >= 2)]

if not merged.empty:
    tag_counts = merged.groupby(['item_id', 'category']).size().reset_index(name='cnt')
    top_tag = (
        tag_counts.sort_values(['item_id', 'cnt', 'category'], ascending=[True, False, True])
        .drop_duplicates(subset=['item_id'], keep='first')
    )
    item_to_cat = dict(zip(top_tag['item_id'].astype(int), top_tag['category'].astype(str)))
else:
    item_to_cat = {}

items_lastfm = pd.DataFrame({'item_id': sorted(inter_lastfm['item_id'].astype(int).unique().tolist())})
items_lastfm['category'] = items_lastfm['item_id'].map(item_to_cat).fillna('UNKNOWN')

# Collapse sparse categories for stability.
cat_counts = items_lastfm['category'].value_counts()
keep = set(cat_counts[cat_counts >= 20].index.tolist())
items_lastfm.loc[~items_lastfm['category'].isin(keep), 'category'] = 'OTHER'
items_lastfm.to_csv(lastfm_out / 'items.csv', index=False)

# -----------------------------
# Amazon Automotive
# -----------------------------
auto_out = proj / 'data/automotive'
auto_out.mkdir(parents=True, exist_ok=True)

raw_candidates = [
    Path('/root/autodl-tmp/PC-Attack/data/automotive/raw/raw.data'),
    Path('/root/autodl-tmp/ShillingAttack/data/automotive/automotive_train.dat'),
]
raw_auto = None
for c in raw_candidates:
    if c.exists():
        raw_auto = c
        break
if raw_auto is None:
    raise FileNotFoundError('Automotive raw source not found in expected cloned repos.')

# raw.data format: user item rating (tab separated)
auto = pd.read_csv(raw_auto, sep='\t', header=None, names=['user_id', 'item_id', 'rating'])
auto = auto[['user_id', 'item_id', 'rating']].copy()
auto['user_id'] = auto['user_id'].astype(int)
auto['item_id'] = auto['item_id'].astype(int)
auto['rating'] = pd.to_numeric(auto['rating'], errors='coerce').fillna(0).astype(float).clip(1.0, 5.0)

# Deduplicate user-item by max rating.
auto = (
    auto.groupby(['user_id', 'item_id'], as_index=False)['rating']
    .max()
    .sort_values(['user_id', 'item_id'])
    .reset_index(drop=True)
)
auto['timestamp'] = 1_600_000_000 + auto.groupby('user_id').cumcount().astype(int)
auto[['user_id', 'item_id', 'rating', 'timestamp']].to_csv(auto_out / 'interactions.csv', index=False)

items_auto = pd.DataFrame({'item_id': sorted(auto['item_id'].unique().tolist())})
items_auto['category'] = 'Automotive'
items_auto.to_csv(auto_out / 'items.csv', index=False)

# -----------------------------
# Stats
# -----------------------------
def dataset_stats(name: str, inter_path: Path, items_path: Path):
    inter = pd.read_csv(inter_path)
    items = pd.read_csv(items_path)
    print(f"[{name}] interactions={len(inter):,} users={inter['user_id'].nunique():,} items={inter['item_id'].nunique():,}")
    print(f"[{name}] item_categories={items['category'].nunique():,} top_categories={items['category'].value_counts().head(10).to_dict()}")

print('Generated datasets:')
dataset_stats('lastfm', lastfm_out / 'interactions.csv', lastfm_out / 'items.csv')
dataset_stats('automotive', auto_out / 'interactions.csv', auto_out / 'items.csv')
