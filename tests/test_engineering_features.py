"""Engineering feature checks.

Maps the spec's §6 "Engineering Requirements" to executable checks. Run with:
    pytest -q tests/test_engineering_features.py -v
or:
    make verify
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ───────────────── Project structure (§6) ─────────────────


REQUIRED_DIRS = [
    "data",
    "src/data",
    "src/features",
    "src/models/churn",
    "src/models/retrieval",
    "src/models/ranking",
    "src/faiss",
    "src/api",
    "src/evaluation",
]


@pytest.mark.parametrize("rel", REQUIRED_DIRS)
def test_required_directory_exists(rel: str):
    assert (ROOT / rel).is_dir(), f"Spec §6 requires directory `{rel}`"


# ───────────────── Tech stack (§6) ─────────────────


def _module_importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def test_pytorch_available():
    assert _module_importable("torch"), "Spec §6: PyTorch required"


def test_faiss_available():
    assert _module_importable("faiss"), "Spec §6: FAISS required"


def test_fastapi_available():
    assert _module_importable("fastapi"), "Spec §6: FastAPI required"


def test_flask_available_for_dashboard():
    assert _module_importable("flask"), "Dashboard requires Flask"


# ───────────────── Modeling components (§4) ─────────────────


def test_churn_classification_module():
    from src.models.churn.classification import stack  # noqa: F401
    assert hasattr(stack, "ChurnStack")


def test_churn_bgnbd_module():
    from src.models.churn.bgnbd import model
    assert hasattr(model, "BGNBDChurn")


def test_churn_survival_module():
    from src.models.churn.survival import model
    assert hasattr(model, "CoxChurn")


def test_retrieval_sasrec_uses_pytorch():
    import torch.nn as nn

    from src.models.retrieval import sasrec
    assert issubclass(sasrec.SASRec, nn.Module), "SASRec must be a torch.nn.Module"


def test_ranking_neumf_in_spec_list():
    """§4.5 requires one of BPR / NeuMF / Wide&Deep / DeepFM / Sequential."""
    from src.models.ranking import neumf
    assert hasattr(neumf, "NeuMF")
    assert hasattr(neumf, "train_neumf")


def test_ranking_lgbm_present_too():
    from src.models.ranking import lgbm_ranker
    assert hasattr(lgbm_ranker, "train_ranker")


def test_faiss_module_exposes_build_and_search():
    from src.faiss import index
    assert callable(index.build_index)
    assert callable(index.topk)


def test_api_has_inference_endpoint():
    """§4.8: at least one inference endpoint, FastAPI."""
    src = (ROOT / "src" / "api" / "app.py").read_text()
    tree = ast.parse(src)
    routes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                txt = ast.unparse(dec) if hasattr(ast, "unparse") else ""
                if "@app." in txt or ".post(" in txt or ".get(" in txt:
                    routes.append(node.name)
    assert "recommend" in routes, "Spec §4.8 requires a recommend endpoint"
    assert "health" in routes, "Health endpoint expected"


# ───────────────── Evaluation (§4.9) ─────────────────


def test_churn_metrics_include_auc_pr_brier():
    """§4.9: AUC-ROC + Precision/Recall/F1. We also add PR-AUC and Brier."""
    import numpy as np

    from src.evaluation.churn_metrics import evaluate
    rng = np.random.default_rng(0)
    y = (rng.random(200) > 0.4).astype(int)
    p = np.clip(y + rng.normal(0, 0.3, 200), 0, 1)
    m = evaluate(y, p)
    for k in ["auc_roc", "pr_auc", "brier_score", "precision", "recall", "f1"]:
        assert k in m, f"evaluate() must return `{k}` (spec §4.9 + project extras)"


def test_recsys_metrics_recall_ndcg():
    from src.evaluation.recsys_metrics import ndcg_at_k, recall_at_k
    assert callable(recall_at_k)
    assert callable(ndcg_at_k)


def test_business_evaluation_present():
    """§4.9 business eval: high-churn segment + retention impact."""
    from src.evaluation.business import evaluate_by_risk_segment, retention_impact
    assert callable(evaluate_by_risk_segment)
    assert callable(retention_impact)


# ───────────────── Reproducibility + docs (§6) ─────────────────


def test_makefile_pipeline_target():
    mk = (ROOT / "Makefile").read_text()
    assert "pipeline:" in mk, "Makefile must define a `pipeline` target for reproducibility"
    for ph in ["phase1", "phase2", "phase3", "phase4"]:
        assert f"{ph}:" in mk, f"Makefile must define `{ph}` target"


def test_requirements_split():
    """CPU and GPU envs must be installable separately."""
    for f in ["requirements-base.txt", "requirements-cpu.txt", "requirements-gpu.txt"]:
        assert (ROOT / f).exists(), f"Spec §6 reproducibility: missing {f}"


def test_docker_files_present():
    """§6: Dockerize."""
    assert (ROOT / "docker" / "Dockerfile").exists()
    assert (ROOT / "docker" / "docker-compose.yml").exists()


def test_ci_workflows_present():
    """§6: CI/CD."""
    wf = ROOT / ".github" / "workflows"
    assert wf.is_dir()
    files = {p.name for p in wf.glob("*.yml")}
    assert files, "At least one GitHub Actions workflow required"


def test_architecture_diagram_present():
    """§5: system diagram."""
    d = ROOT / "diagrams" / "architecture.md"
    assert d.exists(), "Spec §5 requires an architecture diagram"
    assert "```mermaid" in d.read_text(), "Diagram should be machine-readable Mermaid"


def test_features_catalog_present():
    assert (ROOT / "FEATURES.md").exists(), "Spec → file map should be in FEATURES.md"


def test_readme_present():
    assert (ROOT / "README.md").exists()


def test_pyproject_lint_config():
    assert (ROOT / "pyproject.toml").exists()


# ───────────────── Code-quality smoke ─────────────────


def test_no_top_level_print_in_src():
    """Catch stray debug prints in library code (notebooks are OK)."""
    offenders = []
    for py in (ROOT / "src").rglob("*.py"):
        text = py.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent == 0 and stripped.startswith("print("):
                offenders.append(f"{py.relative_to(ROOT)}:{i}")
    assert not offenders, "Top-level print() found in src/: " + ", ".join(offenders)
