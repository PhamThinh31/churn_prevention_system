"""Drift monitoring with Evidently.

Run periodically (e.g. via a daily GH Actions cron) to compare current customer
features against the training distribution. If significant drift is detected on
any of the key features, the workflow opens an issue / triggers retraining.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def run_drift(reference: pd.DataFrame, current: pd.DataFrame, out_html: Path) -> bool:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_html))
    result = report.as_dict()
    drifted = result["metrics"][0]["result"]["dataset_drift"]
    return bool(drifted)


if __name__ == "__main__":
    ref_path, cur_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    drift = run_drift(pd.read_parquet(ref_path), pd.read_parquet(cur_path), Path(out))
    sys.exit(1 if drift else 0)
