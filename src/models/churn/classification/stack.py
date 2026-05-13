"""Kaggle-style stacked churn classifier: XGBoost + LightGBM + CatBoost + LR meta.

Time-aware OOF stacking:
  - Inside `fit`, each base model is trained with KFold on the (training) window.
  - OOF predictions feed a logistic-regression meta-learner.
  - For unseen rows (val / test windows) we average base predictions trained on full
    train fold + meta-learner uses those averaged predictions.

This module is intentionally framework-agnostic — call `fit(X_train, y_train)` then
`predict_proba(X)` on any later window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from xgboost import XGBClassifier


@dataclass
class StackConfig:
    n_splits: int = 5
    random_state: int = 42
    xgb_params: dict | None = None
    lgbm_params: dict | None = None
    cat_params: dict | None = None


def _default_params() -> tuple[dict, dict, dict]:
    xgb = dict(
        n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, eval_metric="auc", tree_method="hist", n_jobs=-1,
    )
    lgbm = dict(
        n_estimators=800, learning_rate=0.05, num_leaves=63, subsample=0.8,
        colsample_bytree=0.8, objective="binary", n_jobs=-1, verbosity=-1,
    )
    cat = dict(
        iterations=800, learning_rate=0.05, depth=6, loss_function="Logloss",
        verbose=False, allow_writing_files=False,
    )
    return xgb, lgbm, cat


class ChurnStack:
    """Stacked churn classifier following the standard Kaggle recipe.

    Three diverse gradient-boosted base learners (XGBoost, LightGBM, CatBoost) are
    each trained K-fold; their out-of-fold predictions become the features for a
    logistic-regression meta-learner. The meta-learner's role is to find the right
    convex combination of the base outputs — usually it picks a near-equal blend,
    occasionally upweights the strongest learner.

    Why this beats a single GBM:
    - **Diversity** — XGBoost, LightGBM, CatBoost split trees differently, handle
      categoricals differently, and regularize differently. Their errors decorrelate.
    - **OOF predictions are honest** — the meta-learner only sees predictions made
      on fold data the base didn't see, avoiding label leakage.
    - **Inference robustness** — at predict time, we average all K trained bases per
      family, which is a cheap form of bagging.

    Usage
    -----
    >>> stack = ChurnStack()
    >>> stack.fit(X_train, y_train)
    >>> p_test = stack.predict_proba(X_test)

    The default base-learner hyperparameters are in :func:`_default_params`. Pass
    a :class:`StackConfig` to override.
    """

    def __init__(self, config: StackConfig | None = None):
        self.config = config or StackConfig()
        xgb_p, lgbm_p, cat_p = _default_params()
        self.config.xgb_params = self.config.xgb_params or xgb_p
        self.config.lgbm_params = self.config.lgbm_params or lgbm_p
        self.config.cat_params = self.config.cat_params or cat_p

        self.base_models: list[tuple[str, list]] = []
        self.meta: LogisticRegression | None = None
        self.feature_names_: list[str] | None = None

    def _make_bases(self):
        return [
            ("xgb", XGBClassifier(**self.config.xgb_params)),
            ("lgbm", LGBMClassifier(**self.config.lgbm_params)),
            ("cat", CatBoostClassifier(**self.config.cat_params)),
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ChurnStack:
        self.feature_names_ = list(X.columns)
        kf = KFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.random_state)

        n = len(X)
        oof = np.zeros((n, 3))  # one column per base

        self.base_models = [(name, []) for name, _ in self._make_bases()]
        for tr, va in kf.split(X):
            for i, (_name, model_ctor) in enumerate(self._make_bases()):
                model = model_ctor
                model.fit(X.iloc[tr], y.iloc[tr])
                oof[va, i] = model.predict_proba(X.iloc[va])[:, 1]
                self.base_models[i][1].append(model)

        self.meta = LogisticRegression(C=1.0, max_iter=1000)
        self.meta.fit(oof, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.meta is not None, "Call fit() first."
        base_preds = np.column_stack([
            np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
            for _, models in self.base_models
        ])
        return self.meta.predict_proba(base_preds)[:, 1]
