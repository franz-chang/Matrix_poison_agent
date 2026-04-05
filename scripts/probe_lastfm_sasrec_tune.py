#!/usr/bin/env python3
from pathlib import Path
import json
import sys

ROOT = Path('/root/matrix-poison-camel')
sys.path.insert(0, str(ROOT))

from matrix_poison.data import prepare_data
from matrix_poison.agent import MatrixPoisonAgent, MatrixPoisonConfig


def run_case(ep, lr, ml, batch):
    data = prepare_data(
        interactions_path=str(ROOT / 'data/lastfm/interactions.csv'),
        items_path=str(ROOT / 'data/lastfm/items.csv'),
        min_positive_rating=1.0,
        seed=42,
    )
    target_category = 'rock'
    target_item = data.item_id_to_index['511']
    cfg = MatrixPoisonConfig(
        intent='nuke',
        model_name='sasrec',
        epsilon=0.02,
        max_iters=1,
        clean_epochs=ep,
        probe_epochs=2,
        judge_epochs=4,
        batch_size=batch,
        k=10,
        target_eval_size=1,
        positive_rating_threshold=1.0,
        lr=lr,
        sasrec_max_len=ml,
        seed=42,
        device='auto',
        camel_enabled=False,
    )
    agent = MatrixPoisonAgent(data=data, target_category=target_category, target_item=target_item, cfg=cfg)
    agent.target_items = {int(target_item)}
    res = agent.run()
    return {
        'clean_epochs': ep,
        'lr': lr,
        'max_len': ml,
        'batch': batch,
        'hit': float(res.clean_metrics.get('hit', 0.0)),
        'ndcg': float(res.clean_metrics.get('ndcg', 0.0)),
    }


def main():
    rows = []
    for ep in [24, 32]:
        for lr in [5e-4, 1e-3, 2e-3]:
            for ml in [20, 30]:
                for batch in [1024, 2048]:
                    print('run', ep, lr, ml, batch, flush=True)
                    rows.append(run_case(ep, lr, ml, batch))
    rows = sorted(rows, key=lambda x: (x['hit'], x['ndcg']), reverse=True)
    out = ROOT / 'outputs' / 'probe_lastfm_sasrec_tune.json'
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print('top10')
    print(json.dumps(rows[:10], ensure_ascii=False, indent=2))
    print('saved', out)


if __name__ == '__main__':
    main()
