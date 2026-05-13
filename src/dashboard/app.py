"""Flask dashboard — business-facing view of the modeling results.

Loads the CSVs in `reports/` and the parquet files in `data/features/`, then
renders a single-page dashboard with 10 charts plus the underlying tables.
Charts are regenerated on-the-fly via matplotlib from the CSV/parquet inputs,
so the dashboard auto-updates whenever pipeline artifacts change.

Run with:
    python -m src.dashboard.app
or:
    make dashboard
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from flask import Flask, render_template, send_file
from matplotlib.figure import Figure

# Add project root to path so `src.*` imports work when run as a module
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS = PROJECT_ROOT / "reports"
FEAT = PROJECT_ROOT / "data" / "features"
PROC = PROJECT_ROOT / "data" / "processed"

app = Flask(__name__, template_folder=str(HERE.parent / "templates"))


def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=0)
    except Exception:
        return pd.read_csv(path)


def _png(fig: Figure):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    matplotlib.pyplot.close(fig)
    return send_file(buf, mimetype="image/png")


# ───────────────────────── Index page ─────────────────────────


@app.route("/")
def index():
    sections = {
        "overview": _overview_context(),
        "files_found": _files_found(),
        "tables": _tables_context(),
        "charts": [
            ("churn_model_comparison", "Churn-model approaches"),
            ("pipeline_ablation", "Pipeline ablation — high-churn segment"),
            ("fe_comparison", "Feature engineering: baseline vs expanded"),
            ("churn_distribution", "Churn risk distribution + tier shading"),
            ("roc_pr", "ROC + Precision-Recall"),
            ("calibration", "Calibration"),
            ("confusion", "Confusion matrix at deployed threshold"),
            ("feature_importance", "Top-25 feature importances"),
            ("risk_segment", "Performance by risk segment"),
            ("customer_funnel", "Customer pipeline funnel"),
        ],
    }
    return render_template("index.html", **sections)


# ───────────────────────── Aggregated context ─────────────────────────


def _files_found() -> dict:
    return {
        "reports": sorted(p.name for p in REPORTS.glob("*.csv")) if REPORTS.exists() else [],
        "features": sorted(p.name for p in FEAT.glob("*.parquet")) if FEAT.exists() else [],
        "processed": sorted(p.name for p in PROC.glob("*.parquet")) if PROC.exists() else [],
    }


def _overview_context() -> dict:
    """Headline numbers for the top of the page."""
    out = {
        "headline_metrics": [],
        "ablation_winner": None,
    }

    # Churn-model headline — best AUC among the 3 approaches on test
    cmp = _read_csv_safe(REPORTS / "churn_model_comparison.csv")
    if cmp is not None and "auc_roc" in cmp.columns:
        cmp_test = cmp[cmp.index.astype(str).str.contains("test", na=False)]
        if len(cmp_test):
            best = cmp_test["auc_roc"].idxmax()
            out["headline_metrics"].append({
                "label": "Best churn AUC (test)",
                "value": f"{cmp_test.loc[best, 'auc_roc']:.3f}",
                "subtitle": str(best).replace(" — test", ""),
            })

    # FE comparison delta
    fe = _read_csv_safe(REPORTS / "fe_comparison.csv")
    if fe is not None and "test_auc" in fe.columns and "expanded" in fe.index and "baseline" in fe.index:
        delta = float(fe.loc["expanded", "test_auc"]) - float(fe.loc["baseline", "test_auc"])
        sign = "+" if delta >= 0 else ""
        out["headline_metrics"].append({
            "label": "FE expansion test-AUC lift",
            "value": f"{sign}{delta:.4f}",
            "subtitle": "76 vs 30 features",
        })

    # Pipeline ablation — variant C lift over variant A
    abl = _read_csv_safe(REPORTS / "pipeline_ablation.csv")
    if abl is not None and "ndcg@5" in abl.columns and len(abl) >= 3:
        rows = list(abl.index)
        a_row = next((r for r in rows if r.startswith("A.")), None)
        c_row = next((r for r in rows if r.startswith("C.")), None)
        if a_row and c_row:
            a_score, c_score = float(abl.loc[a_row, "ndcg@5"]), float(abl.loc[c_row, "ndcg@5"])
            mult = c_score / a_score if a_score > 0 else float("nan")
            out["headline_metrics"].append({
                "label": "NDCG@5 lift (high-churn)",
                "value": f"{mult:.1f}×",
                "subtitle": "ranker+churn vs popular",
            })
            out["ablation_winner"] = c_row

    return out


def _tables_context() -> dict:
    """Render up to 4 small CSVs as HTML tables for inspection."""
    tables = {}
    for fname in ["churn_model_comparison.csv", "fe_comparison.csv",
                  "pipeline_ablation.csv", "recsys_ablation.csv",
                  "churn_by_risk_segment.csv"]:
        df = _read_csv_safe(REPORTS / fname)
        if df is not None:
            tables[fname] = df.round(4).to_html(classes="table table-sm", border=0)
    return tables


# ───────────────────────── Chart routes (lazy regeneration) ─────────────────────────


@app.route("/chart/<name>.png")
def chart(name: str):

    from src.visualization.plots import (
        plot_calibration,
        plot_churn_distribution,
        plot_churn_model_comparison,
        plot_confusion,
        plot_customer_funnel,
        plot_fe_comparison,
        plot_pipeline_ablation,
        plot_risk_segment,
        plot_roc_pr,
    )

    if name == "churn_model_comparison":
        cmp = _read_csv_safe(REPORTS / "churn_model_comparison.csv")
        return _png(plot_churn_model_comparison(cmp))

    if name == "pipeline_ablation":
        abl = _read_csv_safe(REPORTS / "pipeline_ablation.csv")
        return _png(plot_pipeline_ablation(abl))

    if name == "fe_comparison":
        fe = _read_csv_safe(REPORTS / "fe_comparison.csv")
        return _png(plot_fe_comparison(fe))

    if name == "churn_distribution":
        s = pd.read_parquet(FEAT / "churn_scores_gbm_test.parquet")
        return _png(plot_churn_distribution(s, "p_churn_gbm"))

    if name in {"roc_pr", "calibration", "confusion"}:
        labels = pd.read_parquet(PROC / "churn_labels_test.parquet")
        sets = {}
        for nm, fname, col in [
            ("GBM Stack", "churn_scores_gbm_test.parquet", "p_churn_gbm"),
            ("BG-NBD",    "churn_scores_bgnbd_test.parquet", "p_churn_bgnbd"),
            ("Cox PH",    "churn_scores_cox_test.parquet", "p_churn_cox"),
        ]:
            p = FEAT / fname
            if p.exists():
                df = pd.read_parquet(p)
                j = labels.merge(df, on="customer_id", how="inner")
                j[col] = j[col].fillna(j[col].median())
                sets[nm] = (j["churn"].values, j[col].values)
        if not sets:
            return ("no churn scores", 404)
        ref_y = sets["GBM Stack"][0]
        common = {k: v[1] for k, v in sets.items() if len(v[1]) == len(ref_y)}
        if name == "roc_pr":
            return _png(plot_roc_pr(ref_y, common))
        if name == "calibration":
            return _png(plot_calibration(ref_y, common))
        # confusion
        cmp = _read_csv_safe(REPORTS / "churn_model_comparison.csv")
        thr = 0.5
        if cmp is not None and "threshold" in cmp.columns:
            row_match = cmp.index.astype(str).str.contains("GBM.*test")
            if row_match.any():
                thr = float(cmp.loc[cmp.index[row_match][0], "threshold"])
        gbm_p = common["GBM Stack"]
        y_pred = (gbm_p >= thr).astype(int)
        return _png(plot_confusion(ref_y, y_pred))

    if name == "feature_importance":
        # Lightweight: load existing PNG if it was produced by phase 6; otherwise empty
        png = REPORTS / "charts" / "feature_importance.png"
        if png.exists():
            return send_file(png, mimetype="image/png")
        # otherwise refit on baseline features alone
        return _png(_quick_feature_importance())

    if name == "risk_segment":
        from src.evaluation.business import evaluate_by_risk_segment
        labels = pd.read_parquet(PROC / "churn_labels_test.parquet")
        scores = pd.read_parquet(FEAT / "churn_scores_gbm_test.parquet")
        j = labels.merge(scores, on="customer_id", how="inner")
        by_seg = evaluate_by_risk_segment(j["customer_id"].values, j["churn"].values, j["p_churn_gbm"].values)
        return _png(plot_risk_segment(by_seg))

    if name == "customer_funnel":
        labels = pd.read_parquet(PROC / "churn_labels_test.parquet")
        df_tx = pd.read_parquet(PROC / "transactions_clean.parquet")
        cand = pd.read_parquet(FEAT / "retrieval_candidates_test.parquet")
        scores = pd.read_parquet(FEAT / "churn_scores_gbm_test.parquet")
        stages = {
            "1. All test customers":            int(labels["customer_id"].nunique()),
            "2. Have purchase history":         int(df_tx["customer_id"].nunique()),
            "3. SASRec candidates (≥3 buys)":   int(cand["customer_id"].nunique()),
            "4. Top-10 recs delivered":         int(cand[cand["stock_code"].notna()]["customer_id"].nunique()),
            "5. High-risk tier (action sent)":  int((scores["p_churn_gbm"] >= 0.7).sum()),
        }
        return _png(plot_customer_funnel(stages))

    return ("unknown chart", 404)


def _quick_feature_importance() -> Figure:
    from lightgbm import LGBMClassifier

    from src.visualization.plots import plot_feature_importance

    labels = pd.read_parquet(PROC / "churn_labels_train.parquet")
    feats = pd.read_parquet(FEAT / "baseline_features_train.parquet")
    df = feats.merge(labels[["customer_id", "churn"]], on="customer_id", how="inner")
    cols = [c for c in df.columns if c not in {"customer_id", "churn", "country"}
            and pd.api.types.is_numeric_dtype(df[c])]
    m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63, n_jobs=-1, verbosity=-1)
    m.fit(df[cols].fillna(0), df["churn"])
    imp = pd.DataFrame({"feature": cols, "gain": m.booster_.feature_importance(importance_type="gain")})
    return plot_feature_importance(imp, k=25)


if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
