import json
from pathlib import Path
import pandas as pd

base = Path('/root/matrix-poison-camel/outputs/table5_ablation_push_seed42')
variants = ['full','wo_analysis','wo_planning','wo_judgment','wo_self_evolving']
print('=== RUN SUMMARIES ===')
for k in variants:
    p = base / k / 'run_summary.json'
    o = json.loads(p.read_text())
    b = o.get('best_metrics', {})
    print('\n---', k)
    print('target_item_index:', o.get('target_item_index'))
    print('target_item_raw_id:', o.get('target_item_raw_id'))
    print('target_item_policy:', o.get('target_item_policy'))
    print('clean_metrics:', o.get('clean_metrics'))
    print('best_hit_ndcg_shift_obj:', b.get('hit'), b.get('ndcg'), b.get('prediction_shift'), b.get('objective'))
    print('best_iter/eps:', o.get('best_iteration'), o.get('best_epsilon'))

# popularity diagnostics
inter = pd.read_csv('/root/matrix-poison-camel/data/ml1m/interactions.csv')
items = pd.read_csv('/root/matrix-poison-camel/data/ml1m/items.csv')

target_raw = str(json.loads((base/'full'/'run_summary.json').read_text()).get('target_item_raw_id'))
cnt = inter['item_id'].astype(str).value_counts()
print('\n=== POPULARITY DIAGNOSTICS ===')
print('target raw id:', target_raw, 'popularity:', int(cnt.get(target_raw, 0)))

act = set(items[items['category'] == 'Action']['item_id'].astype(str))
act_counts = sorted([int(cnt.get(i, 0)) for i in act], reverse=True)
print('Action item count:', len(act_counts))
print('Action top-10 popularity:', act_counts[:10])
print('Action median popularity:', act_counts[len(act_counts)//2])
idx70 = round((len(act_counts)-1)*0.7)
print('Action 70% index popularity:', act_counts[idx70], 'index:', idx70)
print('Action tail popularity:', act_counts[-1])

# where target lies in action pop ranking (1 is highest)
trank = None
for i,v in enumerate(act_counts, start=1):
    if v == int(cnt.get(target_raw,0)):
        trank = i
        break
print('target popularity rank among Action (approx, ties collapse):', trank)
