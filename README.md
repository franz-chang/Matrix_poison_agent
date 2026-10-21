# Matrix-Poison Reproduction

An engineering reproduction based on the paper **Self-Evolving Matrix-Poison Agent on Recommender System**.

This implementation covers the paper’s core workflow:

1. `Initialization`: Scenario statistics and an initial heuristic attack
2. `Analysis Phase`: Probe attacks using `Random / Average / Bandwagon / Segmented`, followed by vulnerability ranking
3. `Planning + Judging`: Generate a hybrid strategy based on the ranking, then quickly validate its target impact and stealth
4. `Self-evolving`: Adjust `epsilon`, switch modes, and hybridize distributions based on feedback

## Method Overview

The figure below compares traditional item-level Nuke attacks with Matrix-Poison’s category-level Nuke setup:

![Matrix-Poison schematic diagram](schematic_diagram.png)

## Directory Structure

- `matrix_poison/agent.py`: Core Agent loop
- `matrix_poison/attacks.py`: Four basic attack methods and the hybridizer
- `matrix_poison/models.py`: Training and scoring for `BPR / NeuMF / LightGCN / SASRec`
- `matrix_poison/metrics.py`: `Hit@K / NDCG@K / Prediction Shift / Stealth`
- `scripts/run_matrix_poison.py`: Main experiment script
- `scripts/preprocess_movielens1m.py`: MovieLens-1M conversion script
- `scripts/make_synthetic_dataset.py`: Script to generate a synthetic dataset for quick validation

## Data Format

### `interactions.csv`

- `user_id`
- `item_id`
- `rating`
- `timestamp` (optional; recommended)

### `items.csv`

- `item_id`
- `category`

## Quick Start

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

## MovieLens-1M Example

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

## Output Files

- `run_summary.json`
  - Clean baseline metrics
  - Best iteration, best strategy weights, and best epsilon
  - Metrics for each iteration (`objective / stealth / reward`)
- `best_malicious_profiles.csv`
  - Best poisoning samples, ready to be injected into the training set

## Notes

- The LLM analyzer and planner in the original paper are implemented here as a runnable, rule-based “semantic gradient” loop.
  If you want to connect a real LLM API, extend the prompt-optimization logic in `analysis_phase` and `self_evolve` in `matrix_poison/agent.py`.
- The implementation follows the paper’s gray-box setting by default: it trains local shadow/victim models using only visible interaction data, without accessing the target system’s gradients or parameters.

## CAMEL-AI Multi-Agent Rewrite

The current implementation integrates CAMEL-AI as the orchestration layer for the analysis, planning, judgment, and self-evolving stages:

- `matrix_poison/camel_orchestrator.py`: CAMEL role orchestration and JSON decision interface
- `matrix_poison/agent.py`: Calls CAMEL for decisions on each iteration, with a robust fallback

Default behavior:

- If `OPENAI_API_KEY` is available in the environment, the model specified by `--camel-model` will be used (defaults to `openai:gpt-4o-mini`).
- If no API key is set, the CAMEL `STUB` backend is used by default, with automatic fallback to the rule-based strategy to ensure the experiment runs.

Available options in `scripts/run_matrix_poison.py`:

- `--disable-camel`
- `--camel-model openai:gpt-4o-mini`
- `--camel-no-stub-fallback`
- `--disable-analysis-phase / --disable-planning-phase / --disable-judgment-phase / --disable-self-evolving`
- `--fixed-attack-method random|average|bandwagon|segmented`

## Train Baseline Target Models

Use the following command to pre-train baseline target-model parameters and save them under the project directory:

```bash
python scripts/train_target_baselines.py \
  --interactions data/ml1m/interactions.csv \
  --items data/ml1m/items.csv \
  --output-dir checkpoints/target_models/ml1m
```

This produces:

- `checkpoints/target_models/ml1m/bpr_baseline.pt`
- `checkpoints/target_models/ml1m/neumf_baseline.pt`
- `checkpoints/target_models/ml1m/lightgcn_baseline.pt`
- `checkpoints/target_models/ml1m/sasrec_baseline.pt`
- `checkpoints/target_models/ml1m/manifest.json`

## Run Experiments in Paper Order

The new script `scripts/run_paper_order_experiments.py` runs experiments in the order of the paper’s sections:

1. `5.3.1` Attack effectiveness in a single market (RQ1)
2. `5.3.2` Comparison with baseline methods (RQ1 extension)
3. `5.4` Cross-market attack effectiveness (RQ2)
4. `5.5` Stealth evaluation (RQ3)
5. `5.6` Ablation study (also generates the RQ4 vulnerability summary)

Example (quick version):

```bash
python scripts/run_paper_order_experiments.py --quick
```

Outputs include:

- `paper_order_report.md`
- `paper_order_all_results.csv`
- `5_5_stealthiness.csv`
- `rq4_vulnerability_summary.csv`

## Additional Datasets (LastFM + Automotive)

The following datasets have been added and are ready to use:

- `data/lastfm/interactions.csv`
- `data/lastfm/items.csv`
- `data/automotive/interactions.csv`
- `data/automotive/items.csv`

Dataset build script:

```bash
python scripts/build_matrix_poison_datasets.py
```

Notes:

- LastFM uses `hetrec2011-lastfm-2k`, with item categories constructed from `user_taggedartists` and `tags`.
- Automotive uses the Amazon Automotive ratings source (saved locally as `data/raw/automotive/raw.data`), with the category standardized as `Automotive`.
