# Matrix-Poison Reproduction

基于论文 **Self-Evolving Matrix-Poison Agent on Recommender System** 的工程复现实现。  
该实现覆盖论文中的核心流程：

1. `Initialization`：场景统计 + 初始启发式攻击
2. `Analysis Phase`：对 `Random / Average / Bandwagon / Segmented` 做探测攻击并计算脆弱性排序
3. `Planning + Judging`：按排序生成混合策略，快速验证目标效果与隐蔽性
4. `Self-evolving`：根据反馈执行 `epsilon` 调节、`Mode Switching`、`Distribution Hybridization`

## 目录

- `matrix_poison/agent.py`：核心 Agent 闭环
- `matrix_poison/attacks.py`：四种基础攻击 + 混合器
- `matrix_poison/models.py`：`BPR / NeuMF / LightGCN / SASRec` 训练与打分
- `matrix_poison/metrics.py`：`Hit@K / NDCG@K / Prediction Shift / Stealth`
- `scripts/run_matrix_poison.py`：主实验脚本
- `scripts/preprocess_movielens1m.py`：MovieLens-1M 转换脚本
- `scripts/make_synthetic_dataset.py`：生成可快速验证的合成数据

## 数据格式

### interactions.csv

- `user_id`
- `item_id`
- `rating`
- `timestamp`（可选，建议保留）

### items.csv

- `item_id`
- `category`

## 快速开始

```bash
pip install -r requirements.txt
python scripts/make_synthetic_dataset.py --output-dir examples/synth
python scripts/run_matrix_poison.py \
  --interactions examples/synth/interactions.csv \
  --items examples/synth/items.csv \
  --target-category C1 \
  --intent nuke \
  --model bpr \
  --output-dir outputs/synth_run
```

## MovieLens-1M 示例

```bash
python scripts/preprocess_movielens1m.py \
  --ratings-dat /path/to/ml-1m/ratings.dat \
  --movies-dat /path/to/ml-1m/movies.dat \
  --output-dir data/ml1m

python scripts/run_matrix_poison.py \
  --interactions data/ml1m/interactions.csv \
  --items data/ml1m/items.csv \
  --target-category Action \
  --intent nuke \
  --model lightgcn \
  --max-iters 5 \
  --epsilon 0.005 \
  --clean-epochs 50 \
  --judge-epochs 12 \
  --output-dir outputs/ml1m_lightgcn_nuke
```

## 输出结果

- `run_summary.json`
  - clean 基线指标
  - 最优迭代、最优策略权重、最优 epsilon
  - 每轮迭代指标（objective / stealth / reward）
- `best_malicious_profiles.csv`
  - 最优毒化样本（可直接注入训练集）

## 说明

- 论文原文中的 LLM-分析器/规划器这里实现为可运行的“规则化语义梯度”闭环；  
  如果需要接入真实 LLM API，可在 `matrix_poison/agent.py` 的 `analysis_phase` 和 `self_evolve` 扩展提示优化逻辑。
- 默认遵循论文的灰盒设定：只基于可见交互数据训练本地 shadow/victim，不访问目标系统梯度与参数。

