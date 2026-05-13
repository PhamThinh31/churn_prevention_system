.PHONY: help setup-cpu setup-gpu fetch-artifacts upload-artifacts phase1 phase1b phase2 phase2b phase3 phase4 phase5 phase6 demo pipeline test lint verify api dashboard docker-up docker-down clean

PY ?= python
ENV_CPU := churn-cpu
ENV_GPU := churn-gpu

help:
	@echo "Targets:"
	@echo "  setup-cpu        Create + provision the local CPU conda env"
	@echo "  setup-gpu        Create + provision the GPU server conda env"
	@echo "  fetch-artifacts  Download pretrained models from the GitHub Release"
	@echo "  upload-artifacts Publish freshly-trained artifacts to a release tag"
	@echo "  phase1        Run EDA + churn labels notebook"
	@echo "  phase1b       Build baseline + expanded feature tables to parquet"
	@echo "  phase2        Run the 3-churn-models notebook"
	@echo "  phase2b       Baseline vs Expanded FE comparison (MLflow)"
	@echo "  phase3        Run SASRec retrieval + FAISS notebook"
	@echo "  phase4        Run LGBM ranker + recsys ablation notebook"
	@echo "  phase5        Run pipeline-level ablation (churn-only → +retrieval → +ranking → +LLM)"
	@echo "  phase6        Run results dashboard — 10 business-facing charts saved to reports/charts/"
	@echo "  demo          Run the end-to-end demo notebook touching every component"
	@echo "  pipeline      Run all phases sequentially"
	@echo "  test          Run pytest"
	@echo "  lint          Run ruff"
	@echo "  api           Run FastAPI (inference) locally on :8000"
	@echo "  dashboard     Run Flask results dashboard locally on :5050"
	@echo "  verify        Run lint + tests + engineering feature checks"
	@echo "  docker-up     Bring up api + mlflow via docker-compose"
	@echo "  docker-down   Stop the docker-compose stack"
	@echo "  clean         Remove generated data/processed and data/features"

setup-cpu:
	conda create -n $(ENV_CPU) python=3.11 -y
	conda run -n $(ENV_CPU) pip install -r requirements-cpu.txt

setup-gpu:
	conda create -n $(ENV_GPU) python=3.11 -y
	conda run -n $(ENV_GPU) pip install -r requirements-gpu.txt

fetch-artifacts:
	python scripts/download_artifacts.py

upload-artifacts:
	python scripts/upload_artifacts.py --tag $(TAG)

phase1:
	jupyter nbconvert --to notebook --execute notebooks/01_eda_and_labels.ipynb --inplace

phase1b:
	jupyter nbconvert --to notebook --execute notebooks/01b_features.ipynb --inplace

phase2:
	jupyter nbconvert --to notebook --execute notebooks/02_churn_models.ipynb --inplace

phase2b:
	jupyter nbconvert --to notebook --execute notebooks/02b_baseline_vs_expanded.ipynb --inplace

phase3:
	jupyter nbconvert --to notebook --execute notebooks/03_sasrec_retrieval.ipynb --inplace

phase4:
	jupyter nbconvert --to notebook --execute notebooks/04_ranking_and_eval.ipynb --inplace

phase5:
	jupyter nbconvert --to notebook --execute notebooks/05_pipeline_ablation.ipynb --inplace

phase6:
	jupyter nbconvert --to notebook --execute notebooks/06_results_dashboard.ipynb --inplace

demo:
	jupyter nbconvert --to notebook --execute notebooks/00_demo.ipynb --inplace

pipeline: phase1 phase1b phase2 phase2b phase3 phase4 phase5 phase6

test:
	pytest -q tests/

lint:
	ruff check src/ tests/

api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

dashboard:
	DASHBOARD_PORT=5050 python -m src.dashboard.app

verify:
	@echo "== ruff =="
	ruff check src/ tests/
	@echo "== pytest =="
	pytest -q tests/
	@echo "== feature checks =="
	pytest -q tests/test_engineering_features.py -v

docker-up:
	docker compose -f docker/docker-compose.yml up -d --build

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	rm -rf data/processed/* data/features/*
	touch data/processed/.gitkeep data/features/.gitkeep
