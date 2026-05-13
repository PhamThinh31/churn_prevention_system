# How to Test All Engineering Features

```bash
make verify     # ruff + pytest + engineering feature checks
```

What `make verify` covers:

| Spec requirement | Check | How |
|---|---|---|
| §6 Project structure (`data/`, `features/`, `models/{churn,retrieval,ranking}/`, `faiss/`, `api/`, `evaluation/`) | `test_required_directory_exists` | `pytest` |
| §6 PyTorch | `test_pytorch_available` | `pytest` |
| §6 FAISS | `test_faiss_available` | `pytest` |
| §6 FastAPI | `test_fastapi_available` | `pytest` |
| §4.2 Three churn approaches | `test_churn_{classification,bgnbd,survival}_module` | `pytest` |
| §4.3 PyTorch retrieval model | `test_retrieval_sasrec_uses_pytorch` (must subclass `nn.Module`) | `pytest` |
| §4.5 Ranking model in spec list | `test_ranking_neumf_in_spec_list` (NeuMF present) | `pytest` |
| §4.4 FAISS build + search | `test_faiss_module_exposes_build_and_search` | `pytest` |
| §4.8 REST inference endpoint | `test_api_has_inference_endpoint` (AST scans for `/recommend`) | `pytest` |
| §4.9 Churn metrics (AUC, P/R/F1) | `test_churn_metrics_include_auc_pr_brier` | `pytest` |
| §4.9 Recsys metrics (Recall@K, NDCG@K) | `test_recsys_metrics_recall_ndcg` | `pytest` |
| §4.9 Business eval | `test_business_evaluation_present` | `pytest` |
| §6 Reproducible pipeline | `test_makefile_pipeline_target` | `pytest` |
| §6 Dockerized | `test_docker_files_present` | `pytest` |
| §6 CI/CD | `test_ci_workflows_present` | `pytest` |
| §5 Architecture diagram | `test_architecture_diagram_present` (Mermaid block exists) | `pytest` |
| §6 Documentation | `test_readme_present`, `test_features_catalog_present` | `pytest` |
| §6 Code quality | `ruff check`, `test_no_top_level_print_in_src` | `pytest` + `ruff` |

## Reproducibility — the harder requirement

Spec §6 says *"Reproducible pipeline."* The minimum is "another engineer can run
this end-to-end and get similar numbers." We make that testable:

```bash
# On a clean machine:
conda create -n churn-cpu python=3.11 -y && conda activate churn-cpu
pip install -r requirements-cpu.txt
# (macOS only)
conda install -c conda-forge llvm-openmp -y
# Place dataset:
cp ~/Downloads/online_retail_II.csv data/raw/

make pipeline   # phases 1, 1b, 2, 2b, 3, 4, 5, 6 in sequence
make dashboard  # then open http://localhost:5050
make api        # then POST /recommend
```

A successful `make pipeline` writes:
- `data/processed/transactions_clean.parquet` + `churn_labels_{train,val,test}.parquet`
- `data/features/baseline_features_*.parquet`, `expanded_features_*.parquet`
- `data/features/churn_scores_{gbm,bgnbd,cox}_test.parquet`
- `data/features/sasrec/{sasrec.pt,vocab.pt}`, `item_index.faiss`
- `data/features/retrieval_candidates_test.parquet`
- `reports/{churn_model_comparison,fe_comparison,pipeline_ablation,recsys_ablation,churn_by_risk_segment}.csv`
- `reports/charts/*.png`

`make verify` doesn't run the pipeline (too slow for CI). It verifies the *infrastructure
that makes the pipeline reproducible* is in place. The CI workflow at
`.github/workflows/lint-test.yml` runs the same checks on every push.

## Manual smoke tests (after `make pipeline`)

1. **Dashboard** — `make dashboard`, open http://localhost:5050. Headline cards
   should show three numbers; ten charts should render; tables should populate.
2. **API health** — `curl http://localhost:8000/health` returns `{"status": "ok", ...}`.
3. **API inference** — `curl -X POST http://localhost:8000/recommend
   -H 'Content-Type: application/json'
   -d '{"customer_id": 12347, "top_k": 10}'` returns a `RecommendResponse`.
4. **Docker build** — `docker compose -f docker/docker-compose.yml up --build`
   should start both the API and MLflow on :8000 and :5000.
