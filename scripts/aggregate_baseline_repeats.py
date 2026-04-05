#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

root = Path('/root/matrix-poison-camel/outputs/baseline_repeats_20260405')
runs = sorted([p for p in root.glob('run_seed*/table_baseline_only.csv') if p.exists()])
if not runs:
    raise SystemExit('no run csv found')

records = []
for p in runs:
    seed = int(p.parent.name.replace('run_seed',''))
    df = pd.read_csv(p)
    for _, r in df.iterrows():
        records.append({
            'seed': seed,
            'model': r['Model'],
            'ml1m_hit': float(r['ML-1M Hit@10']),
            'ml1m_ndcg': float(r['ML-1M NDCG@10']),
            'lastfm_hit': float(r['LastFM Hit@10']),
            'lastfm_ndcg': float(r['LastFM NDCG@10']),
            'automotive_hit': float(r['Automotive Hit@10']),
            'automotive_ndcg': float(r['Automotive NDCG@10']),
        })

raw = pd.DataFrame(records)
raw = raw.sort_values(['model','seed']).reset_index(drop=True)
raw.to_csv(root / 'baseline_3runs_raw.csv', index=False)

agg = raw.groupby('model', as_index=False).agg(
    ml1m_hit_mean=('ml1m_hit','mean'),
    ml1m_hit_std=('ml1m_hit','std'),
    ml1m_ndcg_mean=('ml1m_ndcg','mean'),
    ml1m_ndcg_std=('ml1m_ndcg','std'),
    lastfm_hit_mean=('lastfm_hit','mean'),
    lastfm_hit_std=('lastfm_hit','std'),
    lastfm_ndcg_mean=('lastfm_ndcg','mean'),
    lastfm_ndcg_std=('lastfm_ndcg','std'),
    automotive_hit_mean=('automotive_hit','mean'),
    automotive_hit_std=('automotive_hit','std'),
    automotive_ndcg_mean=('automotive_ndcg','mean'),
    automotive_ndcg_std=('automotive_ndcg','std'),
)
agg.to_csv(root / 'baseline_3runs_mean_std.csv', index=False)

show = pd.DataFrame({
    'Model': agg['model'],
    'ML-1M Hit@10': agg.apply(lambda x: f"{x['ml1m_hit_mean']:.4f}±{x['ml1m_hit_std']:.4f}", axis=1),
    'ML-1M NDCG@10': agg.apply(lambda x: f"{x['ml1m_ndcg_mean']:.4f}±{x['ml1m_ndcg_std']:.4f}", axis=1),
    'LastFM Hit@10': agg.apply(lambda x: f"{x['lastfm_hit_mean']:.4f}±{x['lastfm_hit_std']:.4f}", axis=1),
    'LastFM NDCG@10': agg.apply(lambda x: f"{x['lastfm_ndcg_mean']:.4f}±{x['lastfm_ndcg_std']:.4f}", axis=1),
    'Automotive Hit@10': agg.apply(lambda x: f"{x['automotive_hit_mean']:.4f}±{x['automotive_hit_std']:.4f}", axis=1),
    'Automotive NDCG@10': agg.apply(lambda x: f"{x['automotive_ndcg_mean']:.4f}±{x['automotive_ndcg_std']:.4f}", axis=1),
})
show.to_csv(root / 'baseline_3runs_summary_table.csv', index=False)

print(show.to_string(index=False))
print('\nfiles:')
for name in ['baseline_3runs_raw.csv','baseline_3runs_mean_std.csv','baseline_3runs_summary_table.csv']:
    print(root / name)
