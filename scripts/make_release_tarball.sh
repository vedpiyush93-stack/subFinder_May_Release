#!/usr/bin/env bash
# Bundle heavy artifacts for distribution alongside a GitHub release.
#
# Three separate tarballs (so reviewers can download only what they need):
#
#   subfinder_classifiers.tar.gz    — all 29 configs × 25 trials of classifier weights
#                                     (joblib for shallow, keras for DL; ~1.2 GB compressed)
#                                     needed for: scripts/05_calibrate_best.py and
#                                                 scripts/07_build_paper_artifacts.py with --retrain-cal
#
#   subfinder_final_model.tar.gz    — the 180 MB deployed final_model.pkl + calibration npz
#                                     needed for: scripts/06_inference.py
#
#   subfinder_embeddings.tar.gz     — the per-fold gensim word-embedding cache
#                                     (~50-100 GB compressed depending on tarball compression)
#                                     needed for: scripts/01_train_embeddings.py --reuse-cache
#                                                 and any retrain of shallow/DL from scratch
#
# Usage:
#   bash scripts/make_release_tarball.sh
#
# Upload the resulting tarballs to a release page (GitHub release + Zenodo).
# Then update README §5 with the download URLs + SHA-256 checksums printed below.

set -euo pipefail
RELEASE_DIR="${1:-./releases}"
mkdir -p "$RELEASE_DIR"
cd "$(dirname "$0")/.."

echo "[release] Building classifier-weights tarball ..."
tar -czf "$RELEASE_DIR/subfinder_classifiers.tar.gz" \
    artifacts/predictions/*/r*_f*/classifier.joblib \
    artifacts/predictions/*/r*_f*/classifier.keras 2>/dev/null || true

echo "[release] Building final-model tarball ..."
tar -czf "$RELEASE_DIR/subfinder_final_model.tar.gz" \
    artifacts/final_model.pkl \
    artifacts/calibration/

echo "[release] Building embeddings-cache tarball (this is the big one) ..."
# Follow symlink for the embeddings cache (-h)
tar -czhf "$RELEASE_DIR/subfinder_embeddings.tar.gz" artifacts/embeddings_cache/

echo ""
echo "[release] Tarballs built. Sizes + SHA-256 (paste these into README §5):"
echo ""
for f in "$RELEASE_DIR"/*.tar.gz; do
    size=$(du -h "$f" | cut -f1)
    sha=$(shasum -a 256 "$f" | cut -d' ' -f1)
    printf "  %-50s %s   %s\n" "$(basename "$f")" "$size" "$sha"
done
