#!/usr/bin/env python3
from pathlib import Path
import json
import sys

ROOT = Path('/root/matrix-poison-camel')
sys.path.insert(0, str(ROOT))

from matrix_poison.data import prepare_data
from matrix_poison.agent import MatrixPoisonAgent, MatrixPoisonConfig


def pick_item(data, category, policy):
    items = list(data.category_to_items[category])
    pop = data.train['item'].value_counts().to_dict()
    items = sorted(items, key=lambda i: pop.get(int(i), 0), reverse=True)
    if policy == 'head':
        return int(items[0])
    if policy == 'tail':
        return int(items[-1])
    return int(items[len(items)//2])


def run_ds(ds, category, policy, model):
    data = prepare_data(
        interactions_path=str(ROOT / f'data/{ds}/interactions.csv'),
        items_path=str(ROOT / f'data/{ds}/items.csv'),
        min_positive_rating=1.0,
        seed=42,
    )
    target_item = pick_item(data, category, policy)
    batch = 3072 if model == 'sasrec' else 4096
    cfg = MatrixPoisonConfig(
        intent='nuke',
        model_name=model,
        epsilon=0.02,
        max_iters=1,
        clean_epochs=28,
        probe_epochs=2,
        judge_epochs=4,
        batch_size=batch,
        k=10,
        target_eval_size=1,
        positive_rating_threshold=1.0,
        seed=42,
        device='auto',
        camel_enabled=False,
    )
    agent = MatrixPoisonAgent(data=data, target_category=category, target_item=target_item, cfg=cfg)
    agent.target_items = {target_item}
    res = agent.run()
    return {
        'dataset': ds,
        'model': model,
        'policy': policy,
        'target_item_index': target_item,
        'target_item_raw_id': data.index_to_item_id.get(target_item),
        'clean_hit': float(res.clean_metrics.get('hit', 0.0)),
        'clean_ndcg': float(res.clean_metrics.get('ndcg', 0.0)),
    }


def main():
    rows = []
    for ds, cat in [('lastfm', 'rock'), ('automotive', 'Automotive')]:
        for policy in ['head', 'median', 'tail']:
            for model in ['bpr', 'neumf', 'lightgcn', 'sasrec']:
                print(f'run {ds} {policy} {model}', flush=True)
                rows.append(run_ds(ds, cat, policy, model))
    out = ROOT / 'outputs' / 'probe_strict_target_baseline.json'
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f'saved: {out}')


if __name__ == '__main__':
    main()
