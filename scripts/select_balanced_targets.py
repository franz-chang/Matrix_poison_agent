#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path('/root/matrix-poison-camel')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix_poison.data import build_user_item_sets, build_user_sequences, prepare_data
from matrix_poison.models import TrainConfig, train_model


DATASETS = {
    'lastfm': {
        'interactions': ROOT / 'data/lastfm/interactions.csv',
        'items': ROOT / 'data/lastfm/items.csv',
        'target_category': 'rock',
        'clean_epochs': 28,
        'min_positive_rating': 1.0,
        'positive_threshold': 1.0,
    },
    'automotive': {
        'interactions': ROOT / 'data/automotive/interactions.csv',
        'items': ROOT / 'data/automotive/items.csv',
        'target_category': 'Automotive',
        'clean_epochs': 28,
        'min_positive_rating': 1.0,
        'positive_threshold': 1.0,
    },
}
MODELS = ['bpr', 'neumf', 'lightgcn', 'sasrec']


def train_clean_models(ds_key: str):
    cfg = DATASETS[ds_key]
    data = prepare_data(
        interactions_path=str(cfg['interactions']),
        items_path=str(cfg['items']),
        min_positive_rating=float(cfg['min_positive_rating']),
        seed=42,
    )
    train_user_items = build_user_item_sets(data.train)
    eval_users = sorted(set(data.test['user'].astype(int).tolist()) or set(data.train['user'].astype(int).tolist()))
    eval_users = eval_users[: min(len(eval_users), 1000)]

    model_topk: Dict[str, Dict[int, np.ndarray]] = {}
    all_items = np.arange(data.num_items, dtype=np.int64)

    for idx, model_name in enumerate(MODELS):
        batch = 3072 if model_name == 'sasrec' else 4096
        cfg_train = TrainConfig(
            model_name=model_name,
            epochs=int(cfg['clean_epochs']),
            batch_size=batch,
            lr=1e-3,
            weight_decay=1e-6,
            seed=42 + idx,
            device='cuda' if True else 'cpu',
            positive_threshold=float(cfg['positive_threshold']),
            use_explicit_neg=True,
            explicit_neg_bias=0.8,
        )
        model = train_model(
            train_df=data.train,
            num_users=data.num_users,
            num_items=data.num_items,
            cfg=cfg_train,
        )
        seqs = build_user_sequences(data.train, cfg_train.sasrec_max_len)

        topk_by_user: Dict[int, np.ndarray] = {}
        for u in eval_users:
            seen = train_user_items.get(int(u), set())
            mask = np.ones(data.num_items, dtype=bool)
            if seen:
                s = np.fromiter(seen, dtype=np.int64)
                s = s[(s >= 0) & (s < data.num_items)]
                mask[s] = False
            candidates = all_items[mask]
            if len(candidates) == 0:
                topk_by_user[int(u)] = np.array([], dtype=np.int64)
                continue
            users_arr = np.full(len(candidates), int(u), dtype=np.int64)
            scores = model.score(users_arr, candidates, user_sequences=seqs)
            k = min(10, len(candidates))
            top_idx = np.argpartition(-scores, kth=k - 1)[:k]
            ranked = candidates[top_idx][np.argsort(-scores[top_idx])]
            topk_by_user[int(u)] = ranked.astype(np.int64)
        model_topk[model_name] = topk_by_user
        print(f'[done] {ds_key} {model_name}', flush=True)

    return data, eval_users, model_topk


def evaluate_item(topk_by_model: Dict[str, Dict[int, np.ndarray]], eval_users: List[int], item_idx: int) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    denom = max(1, len(eval_users))
    for model, by_user in topk_by_model.items():
        hit_sum = 0.0
        ndcg_sum = 0.0
        for u in eval_users:
            ranked = by_user.get(int(u), np.array([], dtype=np.int64))
            if ranked.size == 0:
                continue
            pos = np.where(ranked == int(item_idx))[0]
            if len(pos) > 0:
                rank = int(pos[0])
                hit_sum += 1.0
                ndcg_sum += 1.0 / np.log2(rank + 2.0)
        out[model] = {
            'hit': float(hit_sum / denom),
            'ndcg': float(ndcg_sum / denom),
        }
    return out


def score_item(metric_by_model: Dict[str, Dict[str, float]]) -> float:
    hits = np.array([metric_by_model[m]['hit'] for m in MODELS], dtype=np.float64)
    mean_hit = float(hits.mean())
    std_hit = float(hits.std())
    min_hit = float(hits.min())
    max_hit = float(hits.max())

    target_mean = 0.06
    penalty = abs(mean_hit - target_mean)
    penalty += 0.8 * std_hit
    penalty += 2.5 * max(0.0, max_hit - 0.25)
    penalty += 2.0 * max(0.0, 0.005 - min_hit)
    return -penalty


def pick_for_dataset(ds_key: str):
    data, eval_users, model_topk = train_clean_models(ds_key)
    target_category = DATASETS[ds_key]['target_category']
    candidates = [int(x) for x in data.category_to_items.get(target_category, [])]
    pop = data.train['item'].value_counts().to_dict()
    candidates = sorted(candidates, key=lambda i: pop.get(int(i), 0), reverse=True)
    if not candidates:
        raise ValueError(f'No candidates for dataset={ds_key} category={target_category}')

    rows = []
    for item in candidates:
        metric_by_model = evaluate_item(model_topk, eval_users, int(item))
        s = score_item(metric_by_model)
        row = {
            'dataset': ds_key,
            'target_item_index': int(item),
            'target_item_raw_id': data.index_to_item_id.get(int(item), 'UNKNOWN'),
            'popularity': int(pop.get(int(item), 0)),
            'score': float(s),
        }
        for m in MODELS:
            row[f'{m}_hit'] = float(metric_by_model[m]['hit'])
            row[f'{m}_ndcg'] = float(metric_by_model[m]['ndcg'])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    best = df.iloc[0].to_dict()
    return best, df


def main():
    out_root = ROOT / 'outputs' / 'balanced_target_selection'
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for ds in ['lastfm', 'automotive']:
        print(f'=== selecting for {ds} ===', flush=True)
        best, df = pick_for_dataset(ds)
        df.to_csv(out_root / f'{ds}_candidates.csv', index=False)
        summary[ds] = best

    (out_root / 'selected_balanced_targets.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'saved in {out_root}')


if __name__ == '__main__':
    main()
