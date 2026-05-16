#!/usr/bin/env bash
# Live monitor for the W2V/D2V DL retrain (12 configs × 25 folds = 300 fits).
# Counts meta.json files newer than the retrain start marker.
#
# Usage:
#   bash scripts/repro/wv_dv_status.sh           # auto-refresh every 10s
#   bash scripts/repro/wv_dv_status.sh --once    # snapshot
set -euo pipefail
cd "$(dirname "$0")/../.."

INTERVAL=10
ONCE=0
if [ "${1:-}" = "--once" ]; then ONCE=1; fi

CONFIGS=(
  w2vCbow__LSTM w2vCbow__LSTMattn w2vCbow__JustAttn w2vCbow__Trans
  w2vSg__LSTM   w2vSg__LSTMattn   w2vSg__JustAttn   w2vSg__Trans
  d2vDm__LSTM   d2vDm__LSTMattn   d2vDm__JustAttn   d2vDm__Trans
)

render() {
    clear 2>/dev/null || true
    local now=$(date '+%H:%M:%S')
    local start=$(cat /tmp/wv_dv_retrain_start_time 2>/dev/null || echo 0)
    local elapsed=$(( $(date +%s) - start ))
    echo "W2V/D2V DL retrain — live monitor   (refresh ${INTERVAL}s)"
    echo "  Updated at: $now   ·   elapsed: ${elapsed}s ($(( elapsed / 60 )) min)"
    echo
    printf "  %-22s   trials   bar (25 folds = 5 seeds × 5 inner)\n" "config"
    printf "  %-22s   ------   ----------------------------------\n" ""

    local total=0
    for cfg in "${CONFIGS[@]}"; do
        local n=0
        [ -d "artifacts/predictions/$cfg" ] && \
            n=$(find "artifacts/predictions/$cfg" -name meta.json \
                  -newer /tmp/wv_dv_retrain_start_time 2>/dev/null | wc -l | tr -d ' ')
        total=$((total + n))
        local nb=$n
        local bar=""
        for i in $(seq 1 25); do
            if [ "$i" -le "$nb" ]; then bar="${bar}█"; else bar="${bar}░"; fi
        done
        local marker="  "
        [ "$n" -eq 25 ] && marker="✅"
        [ "$n" -gt 0 ] && [ "$n" -lt 25 ] && marker="🔄"
        printf "  %s %-22s  %4d/25   %s\n" "$marker" "$cfg" "$n" "$bar"
    done
    local pct=$((total * 100 / 300))
    echo
    echo "  TOTAL: $total / 300  (${pct}%)"
    if [ "$total" -lt 300 ]; then
        local rate=1
        [ "$elapsed" -gt 0 ] && rate=$(( total > 0 ? elapsed / (total > 0 ? total : 1) : 30 ))
        local remaining=$((300 - total))
        local eta=$((remaining * rate))
        echo "  ETA: ~$((eta / 60)) min  (avg $rate s/fit)"
    fi
}

if [ $ONCE -eq 1 ]; then render; exit 0; fi
while true; do
    render
    sleep $INTERVAL
done
