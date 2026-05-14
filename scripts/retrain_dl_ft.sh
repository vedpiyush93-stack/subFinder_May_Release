#!/usr/bin/env bash
# Re-train the 8 DL FastText configs with proper n-gram OOV fallback.
#
# Why we need this: scripts/03_train_deep.py was patched in commit 1ce3206 to
# (a) load the FULL gensim FastText model (n-gram OOV) instead of the npz-only
# shim, and (b) fix two latent indexing bugs that bit the original --retrain
# path. The shipped probs_*.npz came from the source repo's training; re-running
# here regenerates them with the new code so the release repo is self-contained.
#
# Order: phases run fastest → slowest so you see progress quickly.
#
# Usage (from a fresh terminal):
#   cd /Users/ved/subFinder_May_Release
#   bash scripts/retrain_dl_ft.sh
#
# In another terminal, watch progress with:
#   bash scripts/retrain_status.sh
#
# Total wall time: ~1.5-2 hours on M4 Max.
set -euo pipefail
cd "$(dirname "$0")/.."

# Point gensim loader at the source-of-truth uncompressed cache (no xz
# decompress overhead × 100 folds). Set this to your own regenerated cache
# if you don't have the author's machine layout.
export FT_FULL_DIR="${FT_FULL_DIR:-/Users/ved/subFinder/reproducibility/fold_cache_v2}"

if [ ! -d "$FT_FULL_DIR" ]; then
    echo "[retrain] FT_FULL_DIR=$FT_FULL_DIR does not exist."
    echo "[retrain] Either set FT_FULL_DIR to your local regenerated cache,"
    echo "[retrain] or rely on the release-repo's xz-compressed .npy.xz files"
    echo "[retrain] (slower: adds ~6 s decompress per fold per FT flavor)."
    export FT_FULL_DIR=""
fi

T0=$(date +%s)
# Write a start-time marker so the status script can show real-time progress
# (only count meta.json files modified AFTER this marker).
echo "$T0" > /tmp/retrain_start_time
echo "════════════════════════════════════════════════════════════════════"
echo "DL RETRAIN — 8 FastText DL configs × 25 folds = 200 fits"
echo "FT_FULL_DIR=$FT_FULL_DIR"
echo "start marker: /tmp/retrain_start_time = $T0"
echo "monitor with:  bash scripts/retrain_status.sh"
echo "════════════════════════════════════════════════════════════════════"

phase() {
    local name="$1"; shift
    echo
    echo "════ PHASE: $name ════ (cumulative $(($(date +%s)-T0))s)"
    "$@"
    echo "════ $name done (cumulative $(($(date +%s)-T0))s)"
}

# Phase 2 — JustAttn (~10 s/fold × 50 fits ≈ 8 min)
phase "DL JustAttn (ftCbow + ftSg)" \
    python3 scripts/03_train_deep.py --retrain --only \
        ftCbow__JustAttn ftSg__JustAttn

# Phase 3 — LSTM (~25 s/fold × 50 fits ≈ 21 min)
phase "DL LSTM (ftCbow + ftSg)" \
    python3 scripts/03_train_deep.py --retrain --only \
        ftCbow__LSTM ftSg__LSTM

# Phase 4 — LSTMattn (~25 s/fold × 50 fits ≈ 21 min)
phase "DL LSTMattn (ftCbow + ftSg)" \
    python3 scripts/03_train_deep.py --retrain --only \
        ftCbow__LSTMattn ftSg__LSTMattn

# Phase 5 — Trans (slowest: ~45 s/fold × 50 fits ≈ 38 min)
phase "DL Trans (ftCbow + ftSg) — SLOWEST" \
    python3 scripts/03_train_deep.py --retrain --only \
        ftCbow__Trans ftSg__Trans

echo
echo "════════════════════════════════════════════════════════════════════"
echo "DL RETRAIN COMPLETE — total wall: $(($(date +%s)-T0))s"
echo "════════════════════════════════════════════════════════════════════"
echo
echo "Next: tell Claude 'DL retrain done' and it will run the consolidation:"
echo "  - 04_benchmark.py + 05_calibrate_best.py + notebook end-to-end"
echo "  - 08/09 deck regeneration"
echo "  - copy paper .tex from source repo, recompile PDFs"
echo "  - final README pass + commit + push"
echo "  - resume LFS embedding-model push pipeline"
