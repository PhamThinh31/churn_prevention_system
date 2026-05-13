"""Download pretrained artifacts from the GitHub Release so you can skip training.

Usage:
    python scripts/download_artifacts.py
    # or
    make fetch-artifacts

Override the release tag / URL if you've published a newer version:
    python scripts/download_artifacts.py --tag v0.1.0
    python scripts/download_artifacts.py --base-url https://example.com/some/folder
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPO = "PhamThinh31/churn_prevention_system"
DEFAULT_TAG = "v0.1.0"

# local path  →  asset filename (in the release)
ARTIFACTS = {
    # processed
    "data/processed/transactions_clean.parquet":   "transactions_clean.parquet",
    "data/processed/churn_labels_train.parquet":   "churn_labels_train.parquet",
    "data/processed/churn_labels_val.parquet":     "churn_labels_val.parquet",
    "data/processed/churn_labels_test.parquet":    "churn_labels_test.parquet",
    # features
    "data/features/baseline_features_train.parquet": "baseline_features_train.parquet",
    "data/features/baseline_features_val.parquet":   "baseline_features_val.parquet",
    "data/features/baseline_features_test.parquet":  "baseline_features_test.parquet",
    "data/features/expanded_features_train.parquet": "expanded_features_train.parquet",
    "data/features/expanded_features_val.parquet":   "expanded_features_val.parquet",
    "data/features/expanded_features_test.parquet":  "expanded_features_test.parquet",
    # churn scores
    "data/features/churn_scores_gbm_test.parquet":   "churn_scores_gbm_test.parquet",
    "data/features/churn_scores_bgnbd_test.parquet": "churn_scores_bgnbd_test.parquet",
    "data/features/churn_scores_cox_test.parquet":   "churn_scores_cox_test.parquet",
    # SASRec + FAISS
    "data/features/sasrec/sasrec.pt":                "sasrec.pt",
    "data/features/sasrec/vocab.pt":                 "vocab.pt",
    "data/features/item_index.faiss":                "item_index.faiss",
    "data/features/retrieval_candidates_test.parquet": "retrieval_candidates_test.parquet",
    # reports
    "reports/churn_model_comparison.csv":            "churn_model_comparison.csv",
    "reports/fe_comparison.csv":                     "fe_comparison.csv",
    "reports/recsys_ablation.csv":                   "recsys_ablation.csv",
    "reports/pipeline_ablation.csv":                 "pipeline_ablation.csv",
    "reports/churn_by_risk_segment.csv":             "churn_by_risk_segment.csv",
}


def _release_base(repo: str, tag: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}"


def _download_one(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        seen = 0
        chunk = 64 * 1024
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            seen += len(buf)
            if total:
                pct = seen / total * 100
                print(f"\r    {dest.name:38s}  {seen/1e6:6.1f} / {total/1e6:6.1f} MB  ({pct:5.1f}%)",
                      end="", flush=True)
    print()  # newline
    tmp.rename(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(base_url: str, force: bool = False, only: list[str] | None = None) -> int:
    """Returns the number of files actually downloaded (skipped count = total - this)."""
    missing = []
    for local_path, asset_name in ARTIFACTS.items():
        if only and not any(needle in asset_name for needle in only):
            continue
        dest = PROJECT_ROOT / local_path
        if dest.exists() and not force:
            print(f"skip {dest.relative_to(PROJECT_ROOT)} (exists — pass --force to redownload)")
            continue
        missing.append((asset_name, dest))

    if not missing:
        print("Nothing to download.")
        return 0

    print(f"\nFetching {len(missing)} file(s) from {base_url} ...\n")
    for asset_name, dest in missing:
        url = f"{base_url}/{asset_name}"
        try:
            _download_one(url, dest)
        except urllib.error.HTTPError as e:
            print(f"  ✗ {asset_name}: HTTP {e.code} ({e.reason})", file=sys.stderr)
            return 1
        except urllib.error.URLError as e:
            print(f"  ✗ {asset_name}: {e}", file=sys.stderr)
            return 1
    print("\nDone.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"GitHub repo in OWNER/NAME form (default: {DEFAULT_REPO})")
    p.add_argument("--tag", default=DEFAULT_TAG,
                   help=f"Release tag to download from (default: {DEFAULT_TAG})")
    p.add_argument("--base-url",
                   help="Override the release URL completely (e.g. for a custom mirror)")
    p.add_argument("--force", action="store_true",
                   help="Redownload files that already exist locally")
    p.add_argument("--only", nargs="+",
                   help="Substring match — only download asset names containing any of these")
    args = p.parse_args()

    base = args.base_url or _release_base(args.repo, args.tag)
    return fetch(base, force=args.force, only=args.only)


if __name__ == "__main__":
    raise SystemExit(main())
