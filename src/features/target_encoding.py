"""Out-of-fold mean target encoding.

To avoid leakage:
- On TRAIN: K-fold cross-fit. For each fold, mean target is computed on the OTHER
  folds and applied to held-out fold.
- On VAL/TEST: use the full-train group mean.

For unseen categories at predict time, fall back to the global mean.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import KFold


@dataclass
class OOFEncoder:
    group_col: str
    full_train_map: dict   # category -> mean target
    global_mean: float

    def transform(self, df: pd.DataFrame) -> pd.Series:
        s = df[self.group_col].map(self.full_train_map)
        return s.fillna(self.global_mean).astype("float32")


def fit_oof(
    train: pd.DataFrame,
    target_col: str,
    group_col: str,
    n_folds: int = 5,
    random_state: int = 42,
) -> tuple[pd.Series, OOFEncoder]:
    """Return OOF-encoded values for train rows, plus an encoder for val/test."""
    global_mean = float(train[target_col].mean())
    oof = pd.Series(global_mean, index=train.index, dtype="float32")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for fit_idx, hold_idx in kf.split(train):
        means = train.iloc[fit_idx].groupby(group_col)[target_col].mean()
        oof.iloc[hold_idx] = (
            train.iloc[hold_idx][group_col].map(means).fillna(global_mean).values.astype("float32")
        )

    full_map = train.groupby(group_col)[target_col].mean().to_dict()
    return oof, OOFEncoder(group_col=group_col, full_train_map=full_map, global_mean=global_mean)
