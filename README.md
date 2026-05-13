# Churn Prevention System

[![lint-test](https://github.com/PhamThinh31/churn_prevention_system/actions/workflows/lint-test.yml/badge.svg)](https://github.com/PhamThinh31/churn_prevention_system/actions)

End-to-end ML system that combines **churn prediction** with **personalized
recommendation** to power retention campaigns on the
[Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
dataset. Final project for the Customer Journey Bootcamp.

![System architecture](diagrams/System_Architecture.png)

Sub-system breakdowns:
[OfflineChurn.png](diagrams/OfflineChurn.png) ·
[Recommendation_Pipeline.png](diagrams/Recommendation_Pipeline.png) ·
[CICD_Deployment.png](diagrams/CICD_Deployment.png).
Mermaid source lives in [diagrams/architecture.md](diagrams/architecture.md).

---

## Table of contents

- [Quick start](#quick-start)
- [System overview](#system-overview)
- [Project layout](#project-layout)
- [Pipeline phases](#pipeline-phases)
- [Results](#results)
- [Where things live](#where-things-live)
- [Engineering feature checks](#engineering-feature-checks)
- [Documentation index](#documentation-index)

---

## Quick start

```bash
# 1. Environment (Python 3.11 required)
conda create -n churn-cpu python=3.11 -y && conda activate churn-cpu
pip install -r requirements-cpu.txt
# macOS only: conda install -c conda-forge llvm-openmp -y

# 2A. Skip training — use the pretrained artifacts (~30 MB)
make fetch-artifacts

# 2B. ...OR train from scratch (~20 min on GPU, ~1 hour on CPU)

curl -L -o online-retail-ii-uci.zip https://www.kaggle.com/api/v1/datasets/download/mashlyn/online-retail-ii-uci
mv online_retail_II.csv data/raw/
make pipeline

# 3. Inspect results
make dashboard     # http://localhost:5050  — Flask charts dashboard
make api           # http://localhost:8000  — FastAPI inference endpoint

# 4. Verify everything still works
make verify        # ruff + pytest
```

On a GPU server, swap `requirements-cpu.txt` → `requirements-gpu.txt` (CUDA 12.1
default; edit the `cu121` token if your driver differs — check with `nvidia-smi`).

---

## System overview

Two ML systems glued together, with a thin decision and serving layer on top.

The full architecture is in [diagrams/System_Architecture.png](diagrams/System_Architecture.png).
Sub-system breakdowns:

- Churn pipeline → [diagrams/OfflineChurn.png](diagrams/OfflineChurn.png)
- Recommendation pipeline → [diagrams/Recommendation_Pipeline.png](diagrams/Recommendation_Pipeline.png)
- CI/CD deployment → [diagrams/CICD_Deployment.png](diagrams/CICD_Deployment.png)


| Layer | What it does | Implementation |
|---|---|---|
| **Churn prediction** | Predict `p(churn)` per customer, three approaches | `src/models/churn/{classification,bgnbd,survival}/` |
| **Retrieval** | Learn user + item embeddings → top-50 candidates per customer | `src/models/retrieval/sasrec.py` + `src/faiss/index.py` |
| **Ranking** | Reorder candidates using `p(churn)` + customer/item features | `src/models/ranking/{lgbm_ranker,neumf}.py` |
| **Reranker (optional)** | LLM semantic rerank of the top-20 (Anthropic API) | `src/models/reranker/llm.py` |
| **Decision** | `p(churn)` × top-K → retention action (risk tier + discount) | `src/decision/retention.py` |
| **API** | `POST /recommend` and `GET /health` | `src/api/app.py` (FastAPI) |
| **Dashboard** | Business-facing chart UI on `:5050` | `src/dashboard/app.py` (Flask) |
| **MLOps** | Tracking, drift, retraining, CI/CD | `mlops/`, `.github/workflows/` |

A full spec → file map (with line numbers) is in [FEATURES.md](FEATURES.md).
The Mermaid source for the system diagram is in
[diagrams/architecture.md](diagrams/architecture.md); the four PNG diagrams in
`diagrams/` cover the churn / recsys / CI-CD breakdowns.

---

## Project layout

```
.
├── data/                          gitignored — raw / processed / features
├── notebooks/                     one notebook per phase
│   ├── 00_demo.ipynb              touches every component end-to-end
│   ├── 01_eda_and_labels.ipynb    EDA + churn label generation
│   ├── 01b_features.ipynb         baseline + expanded FE tables
│   ├── 02_churn_models.ipynb      3-approach churn comparison
│   ├── 02b_baseline_vs_expanded.ipynb   FE ablation
│   ├── 03_sasrec_retrieval.ipynb  SASRec + FAISS index
│   ├── 04_ranking_and_eval.ipynb  LGBM ranker + recsys metrics
│   ├── 05_pipeline_ablation.ipynb churn-only vs +retrieval vs +ranker
│   └── 06_results_dashboard.ipynb 10 charts for the report
├── src/
│   ├── data/                      loader, time-window splits, churn labels
│   ├── features/                  build_features (baseline) + expanded + target_encoding
│   ├── models/
│   │   ├── churn/                 classification (Kaggle-stack), bgnbd, survival
│   │   ├── retrieval/             SASRec + sequence dataset
│   │   ├── ranking/               LGBMRanker + NeuMF
│   │   └── reranker/              LLM reranker (Anthropic API)
│   ├── faiss/                     ANN index over item embeddings
│   ├── api/                       FastAPI inference service
│   ├── dashboard/                 Flask results dashboard
│   ├── decision/                  retention-action policy
│   ├── evaluation/                churn / recsys / business metrics
│   └── visualization/             matplotlib plotting helpers
├── tests/                         unit + integration + engineering features
├── mlops/                         MLflow + Evidently
├── docker/                        Dockerfile, docker-compose (api + mlflow)
├── .github/workflows/             lint-test.yml, train.yml, drift.yml
├── configs/                       data.yaml (paths, churn horizon, n_windows)
├── diagrams/                      Mermaid architecture + draw.io guide
├── reports/                       gitignored — CSVs + chart PNGs
├── scripts/                       run_pipeline.py (alternative to notebooks)
├── Makefile                       all phase / dev targets
├── requirements-base.txt          shared deps
├── requirements-cpu.txt           local Mac dev (+CPU torch + faiss-cpu)
├── requirements-gpu.txt           Linux + CUDA 12.1 (+pycox / DeepSurv)
├── pyproject.toml                 ruff + pytest config
├── FEATURES.md                    spec section → file mapping
├── TESTING.md                     how to run + interpret every check
├── README.md                      this file
└── ONBOARDING.md / NOTES (optional)
```

---

## Pipeline phases

Each phase is a Makefile target. `make pipeline` runs them in order.

| Phase | Target | Notebook | Output |
|---|---|---|---|
| 1 | `make phase1` | `01_eda_and_labels.ipynb` | `data/processed/transactions_clean.parquet`, `churn_labels_{train,val,test}.parquet` |
| 1b | `make phase1b` | `01b_features.ipynb` | `data/features/{baseline,expanded}_features_{train,val,test}.parquet` |
| 2 | `make phase2` | `02_churn_models.ipynb` | `reports/churn_model_comparison.csv` + `data/features/churn_scores_{gbm,bgnbd,cox}_test.parquet` |
| 2b | `make phase2b` | `02b_baseline_vs_expanded.ipynb` | `reports/fe_comparison.csv` + `reports/fe_top25_importances.png` |
| 3 | `make phase3` | `03_sasrec_retrieval.ipynb` | `data/features/sasrec/{sasrec.pt,vocab.pt}`, `item_index.faiss`, `retrieval_candidates_test.parquet` |
| 4 | `make phase4` | `04_ranking_and_eval.ipynb` | `reports/recsys_ablation.csv`, `churn_by_risk_segment.csv` |
| 5 | `make phase5` | `05_pipeline_ablation.ipynb` | `reports/pipeline_ablation.csv` |
| 6 | `make phase6` | `06_results_dashboard.ipynb` | `reports/charts/*.png` |

Serving:
- `make api` — FastAPI on `:8000`, endpoints `GET /health`, `POST /recommend`
- `make dashboard` — Flask on `:5050`, regenerates charts from CSVs on every request
- `make docker-up` — docker-compose brings up API + MLflow

---

## Results

Measured on the held-out test window (2011-09-10 → 2011-12-09).

### Churn models — three approaches compared

| model | AUC-ROC | PR-AUC | Brier | F1 |
|---|---|---|---|---|
| GBM Stack (XGB + LGBM + Cat + LR meta) | 0.768 | 0.778 | 0.202 | 0.777 |
| **BG-NBD** | **0.800** | **0.820** | 0.180 | 0.774 |
| Cox PH | 0.685 | 0.609 | 0.389 | 0.678 |

BG-NBD edges out the GBM stack on AUC and PR-AUC. Cox PH lags. The GBM stack
is what the API uses because it gives a richer per-customer feature set.

### Feature engineering — baseline (30 cols) vs expanded (76 cols)

| | test AUC | PR-AUC | Brier |
|---|---|---|---|
| baseline | 0.768 | 0.778 | 0.202 |
| expanded | **0.777 (+0.009)** | **0.785 (+0.008)** | **0.199 (better)** |

### Pipeline ablation on the high-churn segment

| variant | recall@10 | NDCG@5 |
|---|---|---|
| A. churn-only (popular items) | 0.030 | 0.103 |
| B. + SASRec retrieval | 0.019 | 0.103 |
| **C. + LGBM ranker that uses p_churn** | **0.064 (2.1×A)** | **0.417 (4.1×A)** |

Personalization *alone* is worse than popular-items on high-churn customers
(their histories are sparse). Combining personalization with a churn-aware
ranker buys a **4× NDCG@5 lift** — the headline number for stakeholders.

---

## Where things live

If you're looking for:

| You want… | Open… |
|---|---|
| The end-to-end demo | [notebooks/00_demo.ipynb](notebooks/00_demo.ipynb) |
| The architecture diagram | [diagrams/architecture.md](diagrams/architecture.md)|
| Spec § → file mapping | [FEATURES.md](FEATURES.md) |
| The decision threshold logic | [src/decision/retention.py](src/decision/retention.py) |
| The 10 Kaggle-style FE families | [src/features/expanded.py](src/features/expanded.py) |
| The OOF target encoder | [src/features/target_encoding.py](src/features/target_encoding.py) |
| The 3 churn approaches | [src/models/churn/](src/models/churn/) |
| The SASRec model | [src/models/retrieval/sasrec.py](src/models/retrieval/sasrec.py) |
| The NeuMF ranker (spec §4.5) | [src/models/ranking/neumf.py](src/models/ranking/neumf.py) |
| The LGBM ranker (Kaggle baseline) | [src/models/ranking/lgbm_ranker.py](src/models/ranking/lgbm_ranker.py) |
| The LLM reranker | [src/models/reranker/llm.py](src/models/reranker/llm.py) |
| The FAISS index build/search | [src/faiss/index.py](src/faiss/index.py) |

---

## Engineering feature checks

```bash
make verify         # ruff + pytest
make lint           # ruff check
make test           # pytest tests/
```

See [TESTING.md](TESTING.md) for the full table mapping each spec section to
its executable check.

CI runs the same on every push: `.github/workflows/lint-test.yml`.

---

## Documentation index

| Document | Purpose |
|---|---|
| [README.md](README.md) | This file — overview, setup, results, where things live |
| [FEATURES.md](FEATURES.md) | Spec § → file:line mapping, every requirement |
| [TESTING.md](TESTING.md) | How to run + interpret every engineering feature check |
| [diagrams/architecture.md](diagrams/architecture.md) | Mermaid source of truth for the system diagram |
