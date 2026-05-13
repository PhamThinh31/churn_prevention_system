"""Run the offline pipeline end-to-end as a script (alternative to notebooks).

Useful for CI / scheduled retraining where notebook execution is heavier.
Each phase logs to MLflow under experiment "churn-prevention".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from src.data.labels import make_churn_labels
from src.data.loader import clean, load_raw
from src.data.splits import build_windows
from src.evaluation.churn_metrics import evaluate
from src.features.build_features import build_customer_features
from src.models.churn.classification.stack import ChurnStack


def phase_1_data(root: Path) -> tuple[pd.DataFrame, list, dict]:
    print("[phase 1] loading + cleaning")
    df = clean(load_raw(root / "data" / "raw" / "online_retail_II.csv"))
    windows = build_windows(df, horizon_days=90, n_windows=3)
    labels = {w.name: make_churn_labels(df, w) for w in windows}

    proc = root / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    df.to_parquet(proc / "transactions_clean.parquet", index=False)
    for name, t in labels.items():
        t.to_parquet(proc / f"churn_labels_{name}.parquet", index=False)
    return df, windows, labels


def phase_2_churn(df: pd.DataFrame, windows: list, labels: dict, root: Path) -> dict:
    print("[phase 2] features + GBM stack")
    top_countries = df["country"].value_counts().head(8).index.tolist()
    features = {
        w.name: build_customer_features(df, w.feature_end, country_dummies=top_countries)
        for w in windows
    }

    feature_cols = [c for c in features["train"].columns if c not in {"customer_id", "country"}]

    def join(name):
        return features[name].merge(labels[name][["customer_id", "churn"]], on="customer_id")

    data = {n: join(n) for n in ["train", "val", "test"]}

    Xtr, ytr = data["train"][feature_cols].fillna(0), data["train"]["churn"]
    Xva, yva = data["val"][feature_cols].fillna(0), data["val"]["churn"]
    Xte, yte = data["test"][feature_cols].fillna(0), data["test"]["churn"]

    with mlflow.start_run(run_name="churn_gbm_stack"):
        stack = ChurnStack().fit(Xtr, ytr)
        m_val = evaluate(yva.values, stack.predict_proba(Xva))
        m_test = evaluate(yte.values, stack.predict_proba(Xte), threshold=m_val["threshold"])
        mlflow.log_metrics({f"val_{k}": v for k, v in m_val.items() if isinstance(v, float)})
        mlflow.log_metrics({f"test_{k}": v for k, v in m_test.items() if isinstance(v, float)})

    out = root / "data" / "features"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "customer_id": data["test"]["customer_id"],
        "p_churn_gbm": stack.predict_proba(Xte),
    }).to_parquet(out / "churn_scores_gbm_test.parquet", index=False)
    return {"val": m_val, "test": m_test}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--phase", choices=["1", "2", "all"], default="all")
    args = p.parse_args()

    mlflow.set_experiment("churn-prevention")
    df, windows, labels = phase_1_data(args.root)
    if args.phase in {"2", "all"}:
        metrics = phase_2_churn(df, windows, labels, args.root)
        print("[phase 2] metrics:", metrics)


if __name__ == "__main__":
    main()
