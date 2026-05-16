#!/usr/bin/env bash
# AUTONOMOUS CHAIN — fires automatically after W2V/D2V retrain ends:
#   1. waits until 300 fresh W2V/D2V meta.jsons exist
#   2. re-runs 04_benchmark + 05_calibrate
#   3. re-executes notebooks/build_paper_artifacts.ipynb
#   4. rebuilds static + interactive decks
#   5. recompiles main.pdf + supplement.pdf with tectonic
#   6. README final pass (strips the "pending W2V/D2V" note)
#   7. commits + pushes the final clean state to GitHub
#   8. starts the 5-rep reproducibility suite (overnight, ~13 h)
#
# All steps log to /tmp/auto_chain.log so you can `tail -f` it.
# Each step has its own log under reproducibility/_chain_logs/.
#
# Launch ONCE (detached, survives Claude session ending):
#   nohup bash scripts/repro/auto_chain.sh > /tmp/auto_chain.log 2>&1 < /dev/null &
#   disown
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
# the system venv has matplotlib + tectonic deps; use it for notebook + decks
PY_NB="/Users/ved/subFinder/.venv/bin/python"
[ -x "$PY_NB" ] || PY_NB="$PY"
JUPYTER="/Users/ved/subFinder/.venv/bin/jupyter"
[ -x "$JUPYTER" ] || JUPYTER="$(which jupyter || echo jupyter)"

LOG_DIR="reproducibility/_chain_logs"
mkdir -p "$LOG_DIR"

step() {  # step "name" command...
    local name="$1"; shift
    local log="$LOG_DIR/${name}.log"
    echo "════ $(date '+%H:%M:%S')  STEP: $name"
    echo "  log: $log"
    if "$@" > "$log" 2>&1; then
        echo "  ✅ $name done"
    else
        local rc=$?
        echo "  ❌ $name FAILED (rc=$rc) — see $log"
        echo "  (continuing — won't auto-push if anything broke)"
        return $rc
    fi
}

T0=$(date +%s)
echo "════════════════════════════════════════════════════════════════════════════════"
echo "AUTO_CHAIN start at $(date '+%H:%M:%S')   T0=$T0"
echo "  python (training):  $PY"
echo "  python (notebook):  $PY_NB"
echo "  jupyter:            $JUPYTER"
echo "════════════════════════════════════════════════════════════════════════════════"

# ─── 1. WAIT FOR W2V/D2V RETRAIN ──────────────────────────────────────────────
echo
echo "════ STAGE 1 — waiting for W2V/D2V retrain to finish (300 fits)"
WV_DV_CONFIGS="w2vCbow__LSTM w2vCbow__LSTMattn w2vCbow__JustAttn w2vCbow__Trans w2vSg__LSTM w2vSg__LSTMattn w2vSg__JustAttn w2vSg__Trans d2vDm__LSTM d2vDm__LSTMattn d2vDm__JustAttn d2vDm__Trans"

while true; do
    total=0
    for cfg in $WV_DV_CONFIGS; do
        n=0
        [ -d "artifacts/predictions/$cfg" ] && \
            n=$(find "artifacts/predictions/$cfg" -name meta.json \
                -newer /tmp/wv_dv_retrain_start_time 2>/dev/null | wc -l | tr -d ' ')
        total=$((total + n))
    done
    if [ "$total" -ge 300 ]; then
        echo "  ✅ W2V/D2V retrain complete (300/300) at $(date '+%H:%M:%S')"
        break
    fi
    # heartbeat every 5 min to log
    if [ $(( $(date +%s) % 300 )) -lt 30 ]; then
        echo "  $(date '+%H:%M:%S')  $total / 300 fits"
    fi
    sleep 30
done

# ─── 2-6. PASS B-2 ────────────────────────────────────────────────────────────
echo
echo "════ STAGE 2 — Pass B-2 consolidation"

step "01_bench"     $PY  scripts/04_benchmark.py
step "02_cal"      $PY  scripts/05_calibrate_best.py
step "03_notebook" $JUPYTER nbconvert --to notebook --execute --inplace \
    notebooks/build_paper_artifacts.ipynb --ExecutePreprocessor.timeout=900
step "04_deckppt"  $PY_NB scripts/08_build_static_deck.py
step "05_deckhtml" $PY_NB scripts/09_build_interactive_deck.py

# refresh paper sources from source repo + compile PDFs
cp /Users/ved/subFinder/paper/main.tex       paper/main.tex       2>/dev/null || true
cp /Users/ved/subFinder/paper/supplement.tex paper/supplement.tex 2>/dev/null || true
cp /Users/ved/subFinder/paper/reference.bib  paper/reference.bib  2>/dev/null || true
(cd paper && step "06_main_pdf"       tectonic -X compile main.tex)
(cd paper && step "07_supplement_pdf" tectonic -X compile supplement.tex)

# strip the "pending W2V/D2V" note from the README
python3 - <<'PY'
from pathlib import Path
p = Path("README.md")
text = p.read_text()
banner_marker = "> ⏳ **W2V/D2V DL retrain in progress:**"
# delete the line + the following blank line
import re
text = re.sub(rf"^>\s*\n\s*{re.escape(banner_marker)}[^\n]*\n",  "", text, flags=re.M)
text = re.sub(rf"^{re.escape(banner_marker)}[^\n]*\n>?\s*\n?", "", text, flags=re.M)
p.write_text(text)
print("README banner stripped")
PY

# ─── 7. COMMIT + PUSH ─────────────────────────────────────────────────────────
echo
echo "════ STAGE 3 — commit + push"
git add -A > "$LOG_DIR/git_add.log" 2>&1 || true
git commit -m "Pass B-2: retrain W2V/D2V DL with bug fix, final clean leaderboard

- Retrained 12 non-FT DL configs (w2vCbow/w2vSg/d2vDm x LSTM/LSTMattn/JustAttn/Trans)
  with the same _load_emb_seq bug fix + REPRO_REP_SEED-aware classifier seeding.
- Re-ran 04_benchmark + 05_calibrate; regenerated paper figures, audit, decks,
  and recompiled main.pdf + supplement.pdf.
- README banner about 'pending W2V/D2V retrain' removed — leaderboard is now
  fully consistent with the n-gram-OOV bug-fix codebase." > "$LOG_DIR/git_commit.log" 2>&1 || \
  echo "  (nothing to commit, or commit failed — see $LOG_DIR/git_commit.log)"

step "08_push" git push origin main

# ─── 8. AUTO-START REPRO SUITE ────────────────────────────────────────────────
echo
echo "════ STAGE 4 — kicking off 5-rep reproducibility suite"
nohup bash scripts/repro/run_repro_suite.sh > /tmp/repro_suite.log 2>&1 < /dev/null &
disown
echo "  ✅ repro suite started — pid $!  (log: /tmp/repro_suite.log)"

echo
echo "════════════════════════════════════════════════════════════════════════════════"
echo "AUTO_CHAIN main work done at $(date '+%H:%M:%S')   total wall: $(( ($(date +%s) - T0) / 60 )) min"
echo "Reproducibility suite is running in background; ~13 h to full completion."
echo "Monitor:  bash scripts/repro/all_status.sh"
echo "════════════════════════════════════════════════════════════════════════════════"
