"""Upload trained artifacts to a GitHub Release using only the standard library.

Why this exists: many GPU servers don't have the GitHub CLI (`gh`) installed
and can't easily install it. This script does the same job using urllib +
a personal access token.

Setup (one time):
    1. Create a fine-grained PAT at https://github.com/settings/tokens
       Scope: "Repository contents: Read and write" on this repo.
    2. export GITHUB_TOKEN=ghp_XXXX        (or pat_XXXX for fine-grained)

Usage:
    python scripts/upload_artifacts.py --tag v0.1.0
    python scripts/upload_artifacts.py --tag v0.2.0 --notes "Note about what changed"

The script:
- creates the release if it doesn't exist
- deletes any previously-uploaded asset of the same name and replaces it
  (so reruns are idempotent)
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "PhamThinh31/churn_prevention_system"

# Files to upload. Must match the asset names downloaded by download_artifacts.py.
FILES = [
    "data/processed/transactions_clean.parquet",
    "data/processed/churn_labels_train.parquet",
    "data/processed/churn_labels_val.parquet",
    "data/processed/churn_labels_test.parquet",
    "data/features/baseline_features_train.parquet",
    "data/features/baseline_features_val.parquet",
    "data/features/baseline_features_test.parquet",
    "data/features/expanded_features_train.parquet",
    "data/features/expanded_features_val.parquet",
    "data/features/expanded_features_test.parquet",
    "data/features/churn_scores_gbm_test.parquet",
    "data/features/churn_scores_bgnbd_test.parquet",
    "data/features/churn_scores_cox_test.parquet",
    "data/features/sasrec/sasrec.pt",
    "data/features/sasrec/vocab.pt",
    "data/features/item_index.faiss",
    "data/features/retrieval_candidates_test.parquet",
    "reports/churn_model_comparison.csv",
    "reports/fe_comparison.csv",
    "reports/recsys_ablation.csv",
    "reports/pipeline_ablation.csv",
    "reports/churn_by_risk_segment.csv",
]


def _request(method: str, url: str, token: str, *, body: bytes | None = None,
             content_type: str = "application/json") -> dict | None:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "churn-prevention/upload-artifacts",
    }
    if body is not None:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            data = r.read()
            return json.loads(data) if data else None
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")[:500]
        raise SystemExit(f"\nGitHub API {method} {url} failed: HTTP {e.code} {e.reason}\n{msg}") from None


def get_or_create_release(repo: str, tag: str, notes: str, token: str) -> dict:
    try:
        rel = _request("GET", f"https://api.github.com/repos/{repo}/releases/tags/{tag}", token)
        print(f"Release {tag} already exists (id={rel['id']}).")
        return rel
    except SystemExit as e:
        if "HTTP 404" not in str(e):
            raise

    print(f"Creating release {tag}...")
    body = json.dumps({
        "tag_name": tag,
        "name": tag,
        "body": notes,
        "draft": False,
        "prerelease": False,
    }).encode()
    rel = _request("POST", f"https://api.github.com/repos/{repo}/releases", token, body=body)
    print(f"Created release id={rel['id']}.")
    return rel


def list_assets(repo: str, release_id: int, token: str) -> list[dict]:
    return _request("GET", f"https://api.github.com/repos/{repo}/releases/{release_id}/assets", token) or []


def delete_asset(repo: str, asset_id: int, token: str) -> None:
    _request("DELETE", f"https://api.github.com/repos/{repo}/releases/assets/{asset_id}", token)


def upload_one(repo: str, release_id: int, local: Path, token: str) -> None:
    name = local.name
    size = local.stat().st_size
    mime, _ = mimetypes.guess_type(str(local))
    content_type = mime or "application/octet-stream"
    url = (
        f"https://uploads.github.com/repos/{repo}/releases/{release_id}/assets"
        f"?name={urllib.parse.quote(name)}"
    )
    print(f"  uploading {name:42s} ({size/1e6:6.1f} MB) ... ", end="", flush=True)
    with open(local, "rb") as f:
        data = f.read()
    _request("POST", url, token, body=data, content_type=content_type)
    print("ok")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"GitHub repo in OWNER/NAME form (default: {DEFAULT_REPO})")
    p.add_argument("--tag", default="v0.1.0", help="Release tag (created if missing)")
    p.add_argument("--notes", default="Trained churn + recsys artifacts. "
                                      "Run scripts/download_artifacts.py to fetch.",
                   help="Release notes body")
    args = p.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: set GITHUB_TOKEN env var with a personal access token", file=sys.stderr)
        print("  (PAT scopes: Contents: read and write on this repo)", file=sys.stderr)
        return 1

    # Check all files exist before doing any network calls.
    locals_ = [PROJECT_ROOT / rel for rel in FILES]
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in locals_ if not p.exists()]
    if missing:
        print("Missing files (run `make pipeline` first):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    rel = get_or_create_release(args.repo, args.tag, args.notes, token)

    existing = {a["name"]: a for a in list_assets(args.repo, rel["id"], token)}
    print(f"\nUploading {len(locals_)} file(s):\n")
    for local in locals_:
        if local.name in existing:
            delete_asset(args.repo, existing[local.name]["id"], token)
        upload_one(args.repo, rel["id"], local, token)

    print(f"\nDone. Anyone can now run: python scripts/download_artifacts.py --tag {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
