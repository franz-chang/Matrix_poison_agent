from pathlib import Path
import pandas as pd

root = Path('/root/matrix-poison/outputs/table3_xmrec_musical_seed42')
base = pd.read_csv(root / 'baseline_rerun_cuda_full_v2' / 'table4_baseline_seed42.csv')
base_map = {
    r['Model']: {
        'US Hit@10': float(r['US Hit@10']),
        'US NDCG@10': float(r['US NDCG@10']),
        'JP Hit@10': float(r['JP Hit@10']),
        'JP NDCG@10': float(r['JP NDCG@10']),
        'UK Hit@10': float(r['UK Hit@10']),
        'UK NDCG@10': float(r['UK NDCG@10']),
    }
    for _, r in base.iterrows()
}

files = sorted(root.glob('**/table4_with_matrix_poison_seed42.csv'))
print('num_files', len(files))
for f in files:
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    if 'Methods' not in df.columns:
        continue
    push = df[df['Methods'].astype(str).str.contains('Push', case=False, na=False)]
    if push.empty:
        continue
    ok = 0
    total = 0
    for _, r in push.iterrows():
        m = r['Model']
        if m not in base_map:
            continue
        for c in ['US Hit@10','US NDCG@10','JP Hit@10','JP NDCG@10','UK Hit@10','UK NDCG@10']:
            total += 1
            try:
                v = float(r[c])
            except Exception:
                continue
            if v > base_map[m][c]:
                ok += 1
    print(f'{f}\tbetter={ok}/{total}')
