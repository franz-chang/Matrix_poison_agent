from pathlib import Path
import pandas as pd

base = Path('/root/matrix-poison/outputs/table3_xmrec_musical_seed42/baseline_rerun_cuda_full_v2/table4_baseline_seed42.csv')
baseline = pd.read_csv(base)

bpr = pd.read_csv('/root/matrix-poison/outputs/table3_xmrec_musical_seed42/matrix_push_opt_probe_bpr_v3/table4_with_matrix_poison_seed42.csv')
neu_sas = pd.read_csv('/root/matrix-poison/outputs/table3_xmrec_musical_seed42/matrix_push_opt_neumf_sasrec_v3/table4_with_matrix_poison_seed42.csv')
lg_log = Path('/root/matrix-poison/outputs/table3_xmrec_musical_seed42/matrix_push_optimized_20260513.run.log')

# Parse lightgcn from log lines.
light = {'US': None, 'JP': None, 'UK': None}
for line in lg_log.read_text(encoding='utf-8', errors='ignore').splitlines():
    if '[table4][push]' in line and '[lightgcn]' in line:
        # Example: [table4][push][US][lightgcn] hit=0.0178 ndcg=0.0096 p_shift=0.0394 target=...
        try:
            market = line.split('[table4][push][')[1].split(']')[0]
            hit = float(line.split('hit=')[1].split()[0])
            ndcg = float(line.split('ndcg=')[1].split()[0])
            p_shift = float(line.split('p_shift=')[1].split()[0])
            light[market] = (hit, ndcg, p_shift)
        except Exception:
            pass

rows = []

# helper

def pick_push(df, model):
    x = df[(df['Methods'] == 'Matrix Poison-Push') & (df['Model'] == model)]
    if x.empty:
        raise ValueError(f'missing push row for {model}')
    return x.iloc[0]

for model in ['BPR', 'NeuMF', 'LightGCN', 'SASRec']:
    b = baseline[baseline['Model'] == model].iloc[0]
    row_b = {'Methods': 'Baseline', 'Model': model}
    for m in ['US','JP','UK']:
        row_b[f'{m} Hit@10'] = b[f'{m} Hit@10']
        row_b[f'{m} NDCG@10'] = b[f'{m} NDCG@10']
        row_b[f'{m} P. Shift'] = '-'
    rows.append(row_b)

    row_p = {'Methods': 'Matrix Poison-Push', 'Model': model}
    if model == 'BPR':
        p = pick_push(bpr, model)
        for m in ['US','JP','UK']:
            row_p[f'{m} Hit@10'] = p[f'{m} Hit@10']
            row_p[f'{m} NDCG@10'] = p[f'{m} NDCG@10']
            row_p[f'{m} P. Shift'] = p[f'{m} P. Shift']
    elif model in ('NeuMF','SASRec'):
        p = pick_push(neu_sas, model)
        for m in ['US','JP','UK']:
            row_p[f'{m} Hit@10'] = p[f'{m} Hit@10']
            row_p[f'{m} NDCG@10'] = p[f'{m} NDCG@10']
            row_p[f'{m} P. Shift'] = p[f'{m} P. Shift']
    else:
        for m in ['US','JP','UK']:
            if light[m] is None:
                row_p[f'{m} Hit@10'] = 'N/A'
                row_p[f'{m} NDCG@10'] = 'N/A'
                row_p[f'{m} P. Shift'] = 'N/A'
            else:
                h,n,s = light[m]
                row_p[f'{m} Hit@10'] = f'{h:.4f}'
                row_p[f'{m} NDCG@10'] = f'{n:.4f}'
                row_p[f'{m} P. Shift'] = f'{s:.4f}'
    rows.append(row_p)

out = pd.DataFrame(rows)
out_path = Path('/root/matrix-poison/outputs/table3_xmrec_musical_seed42/matrix_push_optimized_20260513_bestmix.csv')
out.to_csv(out_path, index=False)
print(out.to_string(index=False))
print('saved', out_path)

# compute win stats
wins = 0
total = 0
for model in ['BPR','NeuMF','LightGCN','SASRec']:
    b = out[(out['Methods']=='Baseline') & (out['Model']==model)].iloc[0]
    p = out[(out['Methods']=='Matrix Poison-Push') & (out['Model']==model)].iloc[0]
    for m in ['US','JP','UK']:
        for c in ['Hit@10','NDCG@10']:
            total += 1
            bv = float(b[f'{m} {c}'])
            pv = float(p[f'{m} {c}'])
            if pv > bv:
                wins += 1
print('wins', wins, '/', total)
