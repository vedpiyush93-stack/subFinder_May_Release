#!/usr/bin/env bash
# Live monitor for reproducibility suite. Counts per-rep, per-config completion
# (725 fits per rep = 29 configs × 25 folds) and shows progress bars.
#
# Usage:
#   bash scripts/repro/repro_status.sh           # auto-refresh every 15s
#   bash scripts/repro/repro_status.sh --once    # one-shot snapshot
set -euo pipefail
cd "$(dirname "$0")/../.."

INTERVAL=15
ONCE=0
if [ "${1:-}" = "--once" ]; then ONCE=1; fi
[ "${1:-}" -eq "${1:-}" ] 2>/dev/null && INTERVAL=$1   # numeric arg = refresh interval

CONFIGS=(
  cpu__ET500_log2 ftCbow_MM__ET500_sqrt cv__BRF100
  ftCbow__BRF100 ftSg__BRF100 w2vCbow__BRF100 w2vSg__BRF100
  d2vDm__BRF100 d2vDbow__BRF100
  ftCbow__LSTM   ftSg__LSTM   w2vCbow__LSTM   w2vSg__LSTM   d2vDm__LSTM
  ftCbow__LSTMattn ftSg__LSTMattn w2vCbow__LSTMattn w2vSg__LSTMattn d2vDm__LSTMattn
  ftCbow__JustAttn ftSg__JustAttn w2vCbow__JustAttn w2vSg__JustAttn d2vDm__JustAttn
  ftCbow__Trans  ftSg__Trans  w2vCbow__Trans  w2vSg__Trans  d2vDm__Trans
)

bar() {  # bar <done> <total>
    local d=$1 t=$2
    local n=$((d * 25 / t))
    printf "%0.s█" $(seq 1 $n) 2>/dev/null
    printf "%0.s░" $(seq 1 $((25-n))) 2>/dev/null
}

render() {
    clear 2>/dev/null || true
    local now=$(date '+%H:%M:%S')
    local start=$(cat /tmp/repro_start_time 2>/dev/null || echo 0)
    local elapsed=$(( $(date +%s) - start ))
    echo "Reproducibility suite — live monitor   (refresh ${INTERVAL}s, Ctrl+C to exit)"
    echo "  Updated at: $now   ·   elapsed: ${elapsed}s"
    echo

    local total_fits=0
    local done_fits=0
    local total_target=$(( 5 * 29 * 25 ))

    for r in 1 2 3 4 5; do
        local rep_dir="reproducibility/rep_${r}"
        local rep_done=0
        if [ ! -d "$rep_dir" ]; then continue; fi
        local has_lb=""
        [ -f "$rep_dir/leaderboard.csv" ] && has_lb="✅ leaderboard"
        echo "─── rep_${r}  $has_lb  ─────────────────────────────────────────"
        printf "  %-26s   trials   bar (25 folds = 5 seeds × 5 inner)\n" "config"
        printf "  %-26s   ------   ----------------------------------\n"   ""
        for cfg in "${CONFIGS[@]}"; do
            local n=0
            if [ -d "$rep_dir/predictions/$cfg" ]; then
                n=$(find "$rep_dir/predictions/$cfg" -name meta.json 2>/dev/null | wc -l | tr -d ' ')
            fi
            rep_done=$((rep_done + n))
            local marker=" "
            [ "$n" -eq 25 ] && marker="✅"
            [ "$n" -gt 0 ] && [ "$n" -lt 25 ] && marker="🔄"
            printf "  %s %-26s  %4d/25   %s\n" "$marker" "$cfg" "$n" "$(bar $n 25)"
        done
        echo "  rep_${r} subtotal: ${rep_done} / 725 fits"
        done_fits=$((done_fits + rep_done))
        echo
    done

    local pct=0
    [ "$total_target" -gt 0 ] && pct=$((done_fits * 100 / total_target))
    echo "═══ TOTAL across reps: ${done_fits} / ${total_target} fits  (${pct}%)"
}

if [ $ONCE -eq 1 ]; then render; exit 0; fi
while true; do
    render
    sleep "$INTERVAL"
done
