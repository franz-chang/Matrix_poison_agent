#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
import sys

ROOT = Path('/root/matrix-poison-camel')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_poison.data import prepare_data

configs = {
    'ml1m': ('data/ml1m/interactions.csv', 'data/ml1m/items.csv'),
    'lastfm': ('data/lastfm/interactions.csv', 'data/lastfm/items.csv'),
    'automotive': ('data/automotive/interactions.csv', 'data/automotive/items.csv'),
}

report = {}
for ds, (ip, mp) in configs.items():
    inter = pd.read_csv(ROOT / ip)
    items = pd.read_csv(ROOT / mp)
    d = prepare_data(str(ROOT / ip), str(ROOT / mp), min_positive_rating=1.0, seed=42)

    item_set = set(items['item_id'].astype(str).tolist())
    inter_item_set = set(inter['item_id'].astype(str).tolist())
    missing_meta = inter_item_set - item_set

    report[ds] = {
        'raw_interactions': int(len(inter)),
        'raw_users': int(inter['user_id'].nunique()),
        'raw_items_in_interactions': int(inter['item_id'].nunique()),
        'raw_items_in_meta': int(items['item_id'].nunique()),
        'missing_items_in_meta_count': int(len(missing_meta)),
        'prepared_total': int(len(d.interactions)),
        'train': int(len(d.train)),
        'val': int(len(d.val)),
        'test': int(len(d.test)),
        'prepared_users': int(d.num_users),
        'prepared_items': int(d.num_items),
        'categories': int(len(d.category_to_items)),
    }

print(json.dumps(report, ensure_ascii=False, indent=2))
