## Data

Raw CSV (Online Retail II from Kaggle) goes in `data/raw/`. From there:

- `src/data/loader.py` reads the CSV, renames columns, and cleans it: drops null
  CustomerIDs, drops returns (quantity ≤ 0) and zero/negative prices, drops the
  non-product stock codes (POSTAGE, BANK CHARGES, etc.), removes exact duplicates.
- `src/data/splits.py` builds three rolling 90-day windows (train/val/test) by
  walking back from the last transaction in the dataset.
- `src/data/labels.py` writes the churn label: for each customer active before
  `feature_end`, churn=1 if they made no purchase in the next 90 days.

The configurable bits (paths, horizon, number of windows) are in `configs/data.yaml`.

## Churn prediction

I built all three approaches the spec lists so I could compare them properly.

| Approach | File |
|---|---|
| Classification with stacking (XGB + LGBM + CatBoost + LR meta) | `src/models/churn/classification/stack.py` |
| BG-NBD (lifetimes lib) | `src/models/churn/bgnbd/model.py` |
| Survival / Cox PH (lifelines) | `src/models/churn/survival/model.py` |

Everything outputs `p(churn)` per customer so they're directly comparable. Numbers
land in `reports/churn_model_comparison.csv` after `make phase2`.

Features come from two places:

- `src/features/build_features.py` — the basic RFM + behavioral + windowed set
  (~30 columns). This was my first pass.
- `src/features/expanded.py` — ten Kaggle-style families on top: multi-horizon
  aggregates (7/14/30/60/90/180d), time-decay-weighted RFM, inter-purchase
  intervals, cohort, item-side rollups, basket stats (entropy, gini), sequence
  Jaccard, behavioral ratios, time-of-day / day-of-week. ~76 columns.
- `src/features/target_encoding.py` — out-of-fold mean encoder, used on
  `last_country` and `cohort_month`.

The baseline-vs-expanded comparison runs in `notebooks/02b_baseline_vs_expanded.ipynb`
and writes `reports/fe_comparison.csv`. On the test window the expanded set buys
about +0.01 AUC and a slightly better Brier; the deltas are real but small.

## Retrieval and ranking

SASRec for retrieval, in `src/models/retrieval/sasrec.py`. It's a transformer
encoder with causal attention and learned position embeddings, right-padded
sequences (left-pad blows up under the causal+padding-mask combo — that bit me
once). Trained in `src/models/retrieval/train.py`, customer purchase histories
are built by `src/models/retrieval/dataset.py`.

The spec lists ALS / Item2Vec / Two-Tower / Graph as suggested retrieval
options. SASRec isn't literally on that list but learns user + item embeddings
which is what the spec actually requires, and our teacher's example notebook
(`Sequential RecSys.ipynb`) uses SASRec, so I went with it.

`src/faiss/index.py` builds an `IndexFlatIP` over L2-normalized item embeddings
and exposes top-K search. Notebook 03 runs the offline retrieval and persists
`item_index.faiss` plus the top-50 candidates per test customer.

Two rankers are in the repo:

- **NeuMF** at `src/models/ranking/neumf.py` — the one that matches the spec's
  required list (BPR / NeuMF / Wide & Deep / DeepFM / Sequential). GMF tower
  plus MLP tower, BCE on positives + 4× negative sampling.
- **LightGBM LambdaRank** at `src/models/ranking/lgbm_ranker.py` — a strong
  tabular baseline that uses `p(churn)` as a feature explicitly. This is the
  one the API serves by default.

The pipeline ablation in `notebooks/05_pipeline_ablation.ipynb` shows the
LGBM-with-p_churn variant clearly winning on the high-churn segment.

## Reranker (optional)

`src/models/reranker/llm.py` ships a tiny client that sends the top-20 LGBM
candidates and the customer's recent history to an LLM (Anthropic API) and asks
it to reorder. Disabled by default; enable per-request via
`use_llm_reranker: true` in the API body and `ANTHROPIC_API_KEY` in the env.

## Decision layer

`src/decision/retention.py` maps `p(churn)` to one of three tiers:

- p ≥ 0.7 → high risk → 20% discount
- p ≥ 0.4 → medium risk → 10% discount
- otherwise → low risk → no offer

The thresholds are parameters so the business team can tune them without
touching the modeling code. The dashboard chart `churn_distribution.png`
shows where the test population sits relative to those cutoffs.

## API and serving

FastAPI app in `src/api/app.py`. Two endpoints:

- `GET /health` — reports whether each artifact loaded (churn scores, SASRec
  weights, FAISS index, transactions). Returns 503 from `/recommend` if not.
- `POST /recommend` — takes `{customer_id, top_k, use_llm_reranker}`, returns
  the top-K product list plus the risk tier and suggested discount.

Artifacts are loaded once at startup. Dockerfile is in `docker/`, and
`docker/docker-compose.yml` brings up the API plus a local MLflow server.

A separate Flask dashboard sits at `src/dashboard/app.py` and reads
`reports/*.csv` plus the parquet files to render the result charts. Run with
`make dashboard`, port 5050.

## Evaluation

| What | Where |
|---|---|
| AUC, PR-AUC, Brier, precision/recall/F1 | `src/evaluation/churn_metrics.py` (`evaluate()`) |
| Recall@K, NDCG@K | `src/evaluation/recsys_metrics.py` |
| Performance sliced by risk quantile + retention-impact simulation | `src/evaluation/business.py` |
| Plotting helpers (10 charts) | `src/visualization/plots.py` |

The three CSVs in `reports/` cover the three ablations the spec asks for:

- `churn_model_comparison.csv` — Classification vs BG-NBD vs Cox
- `fe_comparison.csv` — baseline FE vs expanded FE
- `pipeline_ablation.csv` — churn-only vs +retrieval vs +retrieval+ranker

PR-AUC and Brier aren't in the spec but I added them because just AUC + F1
doesn't tell you whether the probabilities are calibrated, which matters for
the discount-tier decision.

## Diagrams

`diagrams/` has both the Mermaid source and the rendered PNGs:

- `OfflineChurn.png` — churn pipeline
- `Recommendation_Pipeline.png` — retrieval → ranking → reranker
- `CICD_Deployment.png` — GH Actions → GPU runner → MLflow → Docker
- `System_Architecture.png` — the combined view, this is the deliverable
- `architecture.md` — Mermaid source for the combined view
- `DIAGRAMS.md` — how I built them, in case anyone wants to extend

## CI/CD

GitHub Actions workflows in `.github/workflows/`:

- `lint-test.yml` — ruff + pytest on every push and PR (ubuntu-latest)
- `train.yml` — `workflow_dispatch` or PR label `train`; runs on a self-hosted
  GPU runner and executes the relevant phase notebooks
- `drift.yml` — daily cron that runs Evidently against the latest features

Repo: https://github.com/PhamThinh31/churn_prevention_system

## Engineering checklist

Python 3.11 everywhere. Requirements split: `requirements-base.txt` for the
shared deps, `requirements-cpu.txt` for local dev, `requirements-gpu.txt` for
the GPU server (CUDA 12.1 default — edit if your driver differs).

The `make verify` target runs `ruff check`, `pytest tests/`, plus a separate
test file `tests/test_engineering_features.py` that asserts each spec section
is satisfied (right directories exist, the API has a `recommend` endpoint, the
churn metrics function returns the right keys, etc.). 36 checks, ~10 seconds.

## How to look around

Three quick ways to convince yourself the project works:

```bash
make verify          # asserts the spec is met, ~10s
make dashboard       # open http://localhost:5050, look at the charts
make api             # then POST a customer_id to /recommend
```

For a guided tour, open `notebooks/00_demo.ipynb` — it walks through every
component once.
