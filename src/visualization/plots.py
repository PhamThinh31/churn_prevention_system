"""Plotting helpers for the churn/recsys results dashboard.

All functions write a PNG to `out_path` and return the matplotlib Figure so the
caller can also embed it in a notebook. Color palette is colorblind-safe and
consistent across plots so the business audience can compare across charts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# Consistent palette — business-friendly: blue=baseline, orange=challenger, green=winner
PALETTE = {
    "baseline": "#4C78A8",
    "challenger": "#F58518",
    "winner": "#54A24B",
    "neutral": "#9C9C9C",
    "danger": "#E45756",
}
sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, out_path: str | Path | None) -> plt.Figure:
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


# ───────────────────────── Churn model comparison ─────────────────────────


def plot_churn_model_comparison(comparison_csv: pd.DataFrame, out_path=None) -> plt.Figure:
    """Bar chart: AUC / PR-AUC / Brier across the 3 churn models (test only)."""
    df = comparison_csv.copy()
    df = df[df.index.astype(str).str.contains("test")]
    df.index = df.index.str.replace(" — test", "", regex=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, title, fmt in [
        (axes[0], "auc_roc", "AUC-ROC (higher = better)", "{:.3f}"),
        (axes[1], "pr_auc", "PR-AUC (higher = better)", "{:.3f}"),
        (axes[2], "brier_score", "Brier (lower = better calibrated)", "{:.3f}"),
    ]:
        colors = [PALETTE["winner"] if v == df[metric].max() and metric != "brier_score"
                  else PALETTE["winner"] if v == df[metric].min() and metric == "brier_score"
                  else PALETTE["baseline"]
                  for v in df[metric]]
        bars = ax.bar(df.index, df[metric], color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(title)
        ax.set_ylim(0, max(df[metric].max() * 1.15, 0.05))
        for b, v in zip(bars, df[metric], strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
    fig.suptitle("Churn-prediction approaches on the test window", y=1.05, fontsize=14)
    plt.tight_layout()
    return _save(fig, out_path)


# ───────────────────────── Pipeline ablation ─────────────────────────


def plot_pipeline_ablation(ablation_csv: pd.DataFrame, out_path=None) -> plt.Figure:
    """Grouped bars: recall@10 & ndcg@5 across pipeline variants on high-churn segment."""
    df = ablation_csv.copy()
    df.index.name = "variant"
    df = df.reset_index()

    metrics = ["recall@5", "recall@10", "ndcg@5", "ndcg@10"]
    melted = df.melt(id_vars="variant", value_vars=metrics, var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    palette = [PALETTE["baseline"], PALETTE["challenger"], PALETTE["winner"]]
    sns.barplot(data=melted, x="metric", y="value", hue="variant", palette=palette, ax=ax)
    ax.set_title("System ablation on the high-churn segment\n"
                 "C (retrieval + churn-aware ranker) is the clear winner",
                 fontsize=13)
    ax.set_ylabel("score")
    ax.set_xlabel("")
    ax.legend(title="", loc="upper right", fontsize=10)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", padding=2, fontsize=9)
    plt.tight_layout()
    return _save(fig, out_path)


# ───────────────────────── FE comparison ─────────────────────────


def plot_fe_comparison(fe_csv: pd.DataFrame, out_path=None) -> plt.Figure:
    """Three columns: baseline / expanded / delta — for AUC, PR-AUC, Brier on test."""
    df = fe_csv.copy()
    df = df.loc[~df.index.astype(str).str.startswith("delta")]
    df.index = df.index.str.strip()

    metrics_with_dir = [
        ("test_auc", "AUC-ROC", "higher"),
        ("test_pr_auc", "PR-AUC", "higher"),
        ("test_brier", "Brier", "lower"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, label, direction) in zip(axes, metrics_with_dir, strict=True):
        vals = df[col]
        best = vals.min() if direction == "lower" else vals.max()
        colors = [PALETTE["winner"] if v == best else PALETTE["baseline"] for v in vals]
        bars = ax.bar(vals.index, vals.values, color=colors, edgecolor="white", linewidth=1.5)
        delta = vals.iloc[1] - vals.iloc[0]
        sign = "+" if delta >= 0 else ""
        good = (delta > 0 and direction == "higher") or (delta < 0 and direction == "lower")
        ax.set_title(f"{label}  ({sign}{delta:.4f} {'✓' if good else '✗'})",
                     color=PALETTE["winner"] if good else PALETTE["danger"])
        ax.set_ylim(0, max(vals.max() * 1.15, 0.05))
        for b, v in zip(bars, vals.values, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    fig.suptitle("Feature engineering: 30 → 76 columns. Net effect on test.", y=1.05, fontsize=14)
    plt.tight_layout()
    return _save(fig, out_path)


# ───────────────────────── Churn probability distribution ─────────────────────────


def plot_churn_distribution(scores_df: pd.DataFrame, score_col: str, out_path=None) -> plt.Figure:
    """Histogram of p(churn) on the test population, with risk-tier bands shaded."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(scores_df[score_col], bins=50, color=PALETTE["baseline"], edgecolor="white")
    ax.axvspan(0.7, 1.0, alpha=0.15, color=PALETTE["danger"], label="High risk (≥0.70 → 20% off)")
    ax.axvspan(0.4, 0.7, alpha=0.15, color=PALETTE["challenger"], label="Medium risk (0.40–0.70 → 10% off)")
    ax.axvspan(0.0, 0.4, alpha=0.15, color=PALETTE["winner"], label="Low risk (no action)")
    ax.set_xlabel("p(churn)")
    ax.set_ylabel("# customers")
    ax.set_title("Churn-risk distribution on the test window\n(decision-layer tier shading)", fontsize=13)
    ax.legend(loc="upper center")
    plt.tight_layout()
    return _save(fig, out_path)


# ───────────────────────── ROC + PR curves ─────────────────────────


def plot_roc_pr(
    y_true: np.ndarray,
    score_sets: dict[str, np.ndarray],
    out_path=None,
) -> plt.Figure:
    """Side-by-side ROC and PR curves for multiple model scores."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, y_score in score_sets.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC")
    axes[0].legend()

    for name, y_score in score_sets.items():
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        axes[1].plot(rec, prec, label=f"{name} (AP={auc(rec, prec):.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall")
    axes[1].legend()

    fig.suptitle("Churn models — ROC & PR on the test window", y=1.02, fontsize=14)
    plt.tight_layout()
    return _save(fig, out_path)


# ───────────────────────── Calibration ─────────────────────────


def plot_calibration(
    y_true: np.ndarray,
    score_sets: dict[str, np.ndarray],
    out_path=None,
) -> plt.Figure:
    """Reliability diagram — how trustworthy is each model's probability?"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, y_score in score_sets.items():
        x, y = calibration_curve(y_true, np.clip(y_score, 0, 1), n_bins=10, strategy="quantile")
        ax.plot(y, x, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration — does p(churn)=0.7 actually mean 70% churn rate?")
    ax.legend(loc="upper left")
    plt.tight_layout()
    return _save(fig, out_path)


# ───────────────────────── Confusion matrix ─────────────────────────


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path=None,
                   labels=("retained", "churned")) -> plt.Figure:
    """Confusion matrix expressed in retention-business terms."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"pred: {labels[0]}", f"pred: {labels[1]}"],
                yticklabels=[f"actual: {labels[0]}", f"actual: {labels[1]}"],
                cbar=False, ax=ax)
    ax.set_title("What the churn model gets right / wrong")
    plt.tight_layout()
    return _save(fig, out_path)


# ───────────────────────── Feature importance ─────────────────────────


def plot_feature_importance(importance_df: pd.DataFrame, k: int = 25, out_path=None) -> plt.Figure:
    """Top-k features by gain."""
    df = importance_df.sort_values("gain", ascending=False).head(k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * k)))
    ax.barh(df["feature"], df["gain"], color=PALETTE["baseline"])
    ax.set_xlabel("LightGBM gain")
    ax.set_title(f"Top-{k} features driving the churn prediction")
    plt.tight_layout()
    return _save(fig, out_path)


# ───────────────────────── Risk-segment table heatmap ─────────────────────────


def plot_risk_segment(by_segment_df: pd.DataFrame, out_path=None) -> plt.Figure:
    """Heatmap: quantile × metric. Quickly answers 'is the model better at high-risk?'"""
    keep = ["auc_roc", "pr_auc", "precision", "recall", "f1"]
    df = by_segment_df.set_index("quantile")[keep]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", cbar=False, ax=ax)
    ax.set_title("Performance by churn-risk quantile of the test population")
    ax.set_ylabel("Quantile threshold")
    plt.tight_layout()
    return _save(fig, out_path)


# ───────────────────────── Customer funnel ─────────────────────────


def plot_customer_funnel(stages: dict[str, int], out_path=None) -> plt.Figure:
    """Funnel: how many customers survive each pipeline stage."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = list(stages.keys())
    values = list(stages.values())
    colors = sns.color_palette("Blues_r", n_colors=len(names))
    bars = ax.barh(names, values, color=colors)
    for b, v in zip(bars, values, strict=True):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:,}", va="center")
    ax.invert_yaxis()
    ax.set_xlabel("# customers")
    ax.set_title("Customer pipeline — counts at each stage")
    plt.tight_layout()
    return _save(fig, out_path)
