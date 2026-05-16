#!/usr/bin/env bash
# ONE-COMMAND end-to-end reproducibility suite.
#
# Runs the full 29-config 5×5 RSKF benchmark FIVE independent times
# (rep_1 .. rep_5), each with a different model-init seed but the same
# data splits (5 RSKF seeds × 5 inner folds = 25 trials per config).
#
# Per-rep artifacts (under reproducibility/rep_<N>/):
#   predictions/<config>/r<seed>_f<fold>/{classifier, probs_*, meta}
#   leaderboard.csv          rep-specific leaderboard
#   final_model.pkl          rep-specific calibrated deployment model
#   calibration_report.csv
#   seed.txt                 the model-init seed used for this rep
#   rep_meta.json            wall time + completion timestamp
#   logs/                    per-phase logs
#
# Usage (single terminal):
#   cd /Users/ved/subFinder_May_Release
#   bash scripts/repro/run_repro_suite.sh                # all 5 reps
#   bash scripts/repro/run_repro_suite.sh 3              # only rep 3
#   REPS="1 3 5" bash scripts/repro/run_repro_suite.sh   # custom set
#
# Live monitor (another terminal):
#   bash scripts/repro/repro_status.sh
#
# Total wall time on M4 Max: ~13 h for all 5 reps (~2.5 h per rep).
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

# Default reps list = 1 2 3 4 5; CLI arg or REPS env var overrides
if [ $# -ge 1 ]; then
    REPS="$*"
elif [ -n "${REPS:-}" ]; then
    : # use env-provided
else
    REPS="1 2 3 4 5"
fi

T0=$(date +%s)
echo "$T0" > /tmp/repro_start_time

echo "════════════════════════════════════════════════════════════════════"
echo "REPRODUCIBILITY SUITE — reps: $REPS"
echo "  python:        $PY"
echo "  start marker:  /tmp/repro_start_time = $T0"
echo "  monitor:       bash scripts/repro/repro_status.sh"
echo "════════════════════════════════════════════════════════════════════"

for r in $REPS; do
    t_rep=$(date +%s)
    echo
    echo "════ REP $r START ($(date '+%H:%M:%S'),  elapsed $(($(date +%s)-T0))s) ════"
    $PY scripts/repro/run_one_rep.py --rep "$r"
    echo "════ REP $r END   ($(date '+%H:%M:%S'),  rep wall $(($(date +%s)-t_rep))s) ════"
done

echo
echo "════════════════════════════════════════════════════════════════════"
echo "REPRO SUITE done — total wall: $(($(date +%s)-T0))s"
echo "════════════════════════════════════════════════════════════════════"
echo
echo "Aggregating across reps ..."
$PY scripts/repro/compare_reps.py
