#!/usr/bin/env bash
# Upload the freshly-trained artifacts to a GitHub Release.
#
# Run this AFTER `make pipeline` on the GPU server, when you have the trained
# models in data/features/ and the result CSVs in reports/.
#
# Requires the GitHub CLI (`gh`). Install with:
#   - macOS:   brew install gh
#   - Ubuntu:  sudo apt install gh
#   - other:   https://cli.github.com/
#
# Usage:
#   ./scripts/upload_artifacts.sh v0.1.0
#   ./scripts/upload_artifacts.sh v0.2.0 "Note about what changed in this release"

set -euo pipefail

TAG="${1:-v0.1.0}"
NOTES="${2:-Trained churn + recsys artifacts. See scripts/download_artifacts.py.}"

# Files to upload — must match the asset names in scripts/download_artifacts.py
FILES=(
  data/processed/transactions_clean.parquet
  data/processed/churn_labels_train.parquet
  data/processed/churn_labels_val.parquet
  data/processed/churn_labels_test.parquet

  data/features/baseline_features_train.parquet
  data/features/baseline_features_val.parquet
  data/features/baseline_features_test.parquet
  data/features/expanded_features_train.parquet
  data/features/expanded_features_val.parquet
  data/features/expanded_features_test.parquet

  data/features/churn_scores_gbm_test.parquet
  data/features/churn_scores_bgnbd_test.parquet
  data/features/churn_scores_cox_test.parquet

  data/features/sasrec/sasrec.pt
  data/features/sasrec/vocab.pt
  data/features/item_index.faiss
  data/features/retrieval_candidates_test.parquet

  reports/churn_model_comparison.csv
  reports/fe_comparison.csv
  reports/recsys_ablation.csv
  reports/pipeline_ablation.csv
  reports/churn_by_risk_segment.csv
)

missing=()
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || missing+=("$f")
done
if (( ${#missing[@]} > 0 )); then
  echo "Missing files (run 'make pipeline' first):" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

# Create the release if it doesn't exist, or upload to it if it does.
if gh release view "$TAG" >/dev/null 2>&1; then
  echo "Release $TAG exists — uploading (overwriting) assets..."
  gh release upload "$TAG" "${FILES[@]}" --clobber
else
  echo "Creating release $TAG..."
  gh release create "$TAG" "${FILES[@]}" --title "$TAG" --notes "$NOTES"
fi

echo "Done. Users can now run: python scripts/download_artifacts.py --tag $TAG"
