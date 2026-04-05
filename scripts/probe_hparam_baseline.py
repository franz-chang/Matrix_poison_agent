#!/usr/bin/env python3
from pathlib import Path
import json
import sys

ROOT = Path('/root/matrix-poison-camel')
sys.path.insert(0, str(ROOT))

from matrix_poison.data import prepare_data
from matrix_poison.agent import MatrixPoisonAgent, MatrixPoisonConfig


def pick_head(data, category):
    items = list(data.category_to_items[category])
    pop = data.train['item'].value_counts().to_dict()
    items = sorted(items, key=lambda i: pop.get(int(i), 0), reverse=True)
    return int(items[0])


def run_case(ds, cat, model, clean_epochs, lr, max_len=50):
    data = prepare_data(
        interactions_path=str(ROOT / f'data/{ds}/interactions.csv'),
        items_path=str(ROOT / f'data/{ds}/items.csv'),
        min_positive_rating=1.0,
        seed=42,
    )
    target_item = pick_head(data, cat)
    batch = 2048 if model == 'sasrec' else 4096
    cfg = MatrixPoisonConfig(
        intent='nuke',
        model_name=model,
        epsilon=0.02,
        max_iters=1,
        clean_epochs=clean_epochs,
        probe_epochs=2,
        judge_epochs=4,
        batch_size=batch,
        k=10,
        target_eval_size=1,
        positive_rating_threshold=1.0,
        lr=lr,
        sasrec_max_len=max_len,
        seed=42,
        device='auto',
        camel_enabled=False,
    )
    agent = MatrixPoisonAgent(data=data, target_category=cat, target_item=target_item, cfg=cfg)
    agent.target_items = {target_item}
    res = agent.run()
    return {
        'dataset': ds,
        'model': model,
        'clean_epochs': clean_epochs,
        'lr': lr,
        'sasrec_max_len': max_len,
        'hit': float(res.clean_metrics.get('hit', 0.0)),
        'ndcg': float(res.clean_metrics.get('ndcg', 0.0)),
    }


def main():
    rows = []
    # Automotive NeuMF
    for ep in [4, 8, 12, 20, 28]:
        for lr in [5e-4, 1e-3, 2e-3]:
            print('auto neumf', ep, lr, flush=True)
            rows.append(run_case('automotive', 'Automotive', 'neumf', ep, lr))

    # LastFM SASRec
    for ep in [8, 16, 24, 32, 40]:
        for lr in [5e-4, 1e-3, 2e-3]:
            for ml in [20, 30, 50]:
                print('lastfm sasrec', ep, lr, ml, flush=True)
                rows.append(run_case('lastfm', 'rock', 'sasrec', ep, lr, max_len=ml))

    out = ROOT / 'outputs' / 'probe_hparam_baseline.json'
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved', out)


if __name__ == '__main__':
    main()
