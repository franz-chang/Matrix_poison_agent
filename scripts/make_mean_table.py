#!/usr/bin/env python3
import pandas as pd

p='/root/matrix-poison-camel/outputs/baseline_repeats_20260405/baseline_3runs_mean_std.csv'
df=pd.read_csv(p)
out=pd.DataFrame({
 'Model':df['model'],
 'ML-1M Hit@10':df['ml1m_hit_mean'].map(lambda x:f'{x:.4f}'),
 'ML-1M NDCG@10':df['ml1m_ndcg_mean'].map(lambda x:f'{x:.4f}'),
 'LastFM Hit@10':df['lastfm_hit_mean'].map(lambda x:f'{x:.4f}'),
 'LastFM NDCG@10':df['lastfm_ndcg_mean'].map(lambda x:f'{x:.4f}'),
 'Automotive Hit@10':df['automotive_hit_mean'].map(lambda x:f'{x:.4f}'),
 'Automotive NDCG@10':df['automotive_ndcg_mean'].map(lambda x:f'{x:.4f}'),
})
out=out.set_index('Model').loc[['BPR','NeuMF','LightGCN','SASRec']].reset_index()
out_path='/root/matrix-poison-camel/outputs/baseline_repeats_20260405/baseline_3runs_mean_table.csv'
out.to_csv(out_path,index=False)
print(out.to_string(index=False))
print('saved',out_path)
