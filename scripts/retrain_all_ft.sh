#!/usr/bin/env bash
# ONE COMMAND end-to-end FastText retrain.
#
# Retrains ALL 11 FastText-using configs (3 shallow + 8 DL) × 25 folds = 275
# trials, in fastest→slowest order. Auto-uses the local .venv (TF 2.18 +
# Apple Silicon Metal GPU). Writes a /tmp/retrain_start_time marker so
# scripts/retrain_status.sh can show real-time progress.
#
# Usage (single terminal):
#   cd /Users/ved/subFinder_May_Release
#   bash scripts/retrain_all_ft.sh
#
# In another terminal, watch the live monitor:
#   bash scripts/retrain_status.sh
#
# Total wall time on M4 Max: ~2 hours (5 min shallow + ~115 min DL).
set -euo pipefail
cd "$(dirname "$0")/.."

# Auto-use the local .venv if present (GPU-enabled TF + tensorflow-metal).
if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
    echo "[retrain] using local .venv python: $($PY -c 'import sys; print(sys.executable)')"
else
    PY=python3
    echo "[retrain] no local .venv — falling back to system python3 (may not have GPU)"
fi

# Point gensim loader at the source-of-truth uncompressed cache (no xz
# decompress overhead). Override with FT_FULL_DIR=/path/to/your/cache if
# you regenerated locally.
export FT_FULL_DIR="${FT_FULL_DIR:-/Users/ved/subFinder/reproducibility/fold_cache_v2}"
if [ ! -d "$FT_FULL_DIR" ]; then
    echo "[retrain] FT_FULL_DIR=$FT_FULL_DIR does not exist; will use release-repo's"
    echo "[retrain]   .npy.xz files (slower: ~6 s auto-decompress per fold per FT flavor)."
    export FT_FULL_DIR=""
fi

T0=$(date +%s)
# Start marker so retrain_status.sh shows real-time progress (counts only
# meta.json modified AFTER this marker).
echo "$T0" > /tmp/retrain_start_time

echo "════════════════════════════════════════════════════════════════════"
echo "RETRAIN ALL 11 FastText configs — 275 fits with n-gram OOV"
echo "  python:        $PY"
echo "  FT_FULL_DIR:   $FT_FULL_DIR"
echo "  start marker:  /tmp/retrain_start_time = $T0"
echo "  monitor:       bash scripts/retrain_status.sh"
echo "════════════════════════════════════════════════════════════════════"

phase() {
    local name="$1"; shift
    echo
    echo "════ PHASE: $name ════ (elapsed $(($(date +%s)-T0))s)"
    "$@"
    echo "════ $name done (cumulative $(($(date +%s)-T0))s)"
}

# Phase 1 — shallow FT (3 configs × 25 folds = 75 fits, ~5 min)
phase "[1/5] SHALLOW (ftCbow_MM + ftCbow__BRF + ftSg__BRF)" \
    $PY scripts/02_train_shallow.py --retrain --only \
        ftCbow_MM__ET500_sqrt ftCbow__BRF100 ftSg__BRF100

# Phase 2 — DL JustAttn (~10 s/fold × 50 fits ≈ 8 min)
phase "[2/5] DL JustAttn (ftCbow + ftSg)" \
    $PY scripts/03_train_deep.py --retrain --only \
        ftCbow__JustAttn ftSg__JustAttn

# Phase 3 — DL LSTM (~25 s/fold × 50 fits ≈ 21 min)
phase "[3/5] DL LSTM (ftCbow + ftSg)" \
    $PY scripts/03_train_deep.py --retrain --only \
        ftCbow__LSTM ftSg__LSTM

# Phase 4 — DL LSTMattn (~25 s/fold × 50 fits ≈ 21 min)
phase "[4/5] DL LSTMattn (ftCbow + ftSg)" \
    $PY scripts/03_train_deep.py --retrain --only \
        ftCbow__LSTMattn ftSg__LSTMattn

# Phase 5 — DL Trans (slowest: ~45 s/fold × 50 fits ≈ 38 min)
phase "[5/5] DL Trans (ftCbow + ftSg) — SLOWEST" \
    $PY scripts/03_train_deep.py --retrain --only \
        ftCbow__Trans ftSg__Trans

echo
echo "════════════════════════════════════════════════════════════════════"
echo "ALL FT RETRAIN COMPLETE — total wall: $(($(date +%s)-T0))s"
echo "════════════════════════════════════════════════════════════════════"
echo
echo "Next, run the consolidation pipeline:"
echo "  - 04_benchmark.py + 05_calibrate_best.py"
echo "  - notebook end-to-end (regen all paper tables + audit + figures)"
echo "  - 08/09 deck regeneration"
echo "  - copy paper .tex sources from source repo + recompile PDFs"
echo "  - final README pass + commit + push"
echo "  - resume LFS embedding-model push (32/100 → 100/100)"
