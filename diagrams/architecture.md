# Architecture Diagram — Churn Prevention System

## Offline + Online — combined view

```mermaid
flowchart LR
  subgraph SRC[Raw sources]
    DB[(Online Retail II<br/>transactions.csv)]
  end

  subgraph OFF[Offline Pipeline — runs on GPU server via GH Actions self-hosted runner]
    direction TB
    PRE[Preprocessing<br/>src/data/loader.py]
    LBL[Churn label generation<br/>src/data/labels.py]
    FE[Feature engineering<br/>src/features/build_features.py]

    subgraph CHURN[Churn — 3 approaches compared]
      C1[GBM Stack<br/>XGB + LGBM + Cat + LR meta]
      C2[BG-NBD<br/>lifetimes]
      C3[Cox PH<br/>lifelines]
    end

    SAS[SASRec<br/>src/models/retrieval/sasrec.py]
    IDX[FAISS index build<br/>src/faiss/index.py]
    RNK[LGBMRanker<br/>src/models/ranking/lgbm_ranker.py]

    DB --> PRE --> LBL --> FE
    FE --> C1
    FE --> C2
    FE --> C3
    DB --> SAS --> IDX
    C1 -.score.-> RNK
    IDX -.candidates.-> RNK

    MLF[(MLflow<br/>experiments + registry)]
    C1 --> MLF
    C2 --> MLF
    C3 --> MLF
    SAS --> MLF
    RNK --> MLF
  end

  subgraph ART[Model artifacts<br/>data/features/]
    A1[/churn_scores_*.parquet/]
    A2[/sasrec.pt + vocab.pt/]
    A3[/item_index.faiss/]
    A4[/lgbm_ranker.joblib/]
  end

  C1 --> A1
  SAS --> A2
  IDX --> A3
  RNK --> A4

  subgraph ON[Online Pipeline — FastAPI container]
    direction TB
    REQ[POST /recommend<br/>customer_id]
    CHK[Churn score lookup]
    UVEC[SASRec user vector<br/>last 50 items]
    ANN[FAISS top-50]
    LRNK[LGBM rerank]
    LLM{Use LLM<br/>reranker?}
    LLM_Y[LLM rerank top-20]
    DEC[Decision layer<br/>src/decision/retention.py]
    RESP[Top-K + risk tier<br/>+ discount]

    REQ --> CHK --> UVEC --> ANN --> LRNK --> LLM
    LLM -- yes --> LLM_Y --> DEC
    LLM -- no --> DEC
    DEC --> RESP
  end

  A1 -.load on startup.-> CHK
  A2 -.load on startup.-> UVEC
  A3 -.load on startup.-> ANN
  A4 -.load on startup.-> LRNK

  subgraph MON[Monitoring + CI/CD]
    EVI[Evidently drift report<br/>daily cron]
    GHA[GitHub Actions<br/>lint-test, train, drift]
    DOC[Docker images<br/>GHCR]
  end

  EVI -. drift detected .-> GHA
  GHA -. triggers retrain .-> OFF
  GHA -. publishes .-> DOC
  DOC -. deploys .-> ON
```

## Layer responsibilities

| Layer | What it does | Where it lives |
|---|---|---|
| Raw | UK e-commerce transactions | `data/raw/online_retail_II.csv` |
| Preprocess | Drop nulls, returns, non-product codes | `src/data/loader.py` |
| Label | 90-day no-purchase = churn | `src/data/labels.py` |
| Features | RFM + behavioral + windowed + ratios | `src/features/build_features.py` |
| Churn (×3) | GBM stack, BG-NBD, CoxPH | `src/models/churn/` |
| Retrieval | SASRec → item embeddings | `src/models/retrieval/` |
| ANN | FAISS top-50 | `src/faiss/` |
| Ranking | LGBMRanker with churn × retrieval features | `src/models/ranking/` |
| Rerank (opt) | LLM (Anthropic API) | `src/models/reranker/` |
| Decision | risk tier + discount | `src/decision/` |
| API | FastAPI single endpoint | `src/api/` |
| Tracking | MLflow | `mlops/mlflow/` |
| Drift | Evidently report | `mlops/evidently/` |
| CI/CD | GH Actions (lint-test, train, drift) | `.github/workflows/` |
```
