#!/usr/bin/env bash
# LIVE progress monitor for the FT retrain. Auto-refreshes every 10 seconds.
# Press Ctrl+C to exit.
#
# Usage:
#   bash scripts/retrain_status.sh            # default 10 s refresh
#   bash scripts/retrain_status.sh 5          # 5 s refresh
#   bash scripts/retrain_status.sh --once     # single snapshot (no loop)
cd "$(dirname "$0")/.."

CFG=(
  "shallow:ftCbow_MM__ET500_sqrt"
  "shallow:ftCbow__BRF100"
  "shallow:ftSg__BRF100"
  "DL_JustAttn:ftCbow__JustAttn"
  "DL_JustAttn:ftSg__JustAttn"
  "DL_LSTM:ftCbow__LSTM"
  "DL_LSTM:ftSg__LSTM"
  "DL_LSTMattn:ftCbow__LSTMattn"
  "DL_LSTMattn:ftSg__LSTMattn"
  "DL_Trans:ftCbow__Trans"
  "DL_Trans:ftSg__Trans"
)

# "Fresh" = meta.json modified AFTER the bug-fix-script's mtime (commit 1ce3206).
THRESH=$(stat -f "%m" scripts/02_train_shallow.py)

render() {
    clear
    NOW=$(date +%s)
    printf "\n  FastText retrain — live monitor   (refresh every ${1}s, Ctrl+C to exit)\n"
    printf "  Updated at: $(date '+%H:%M:%S')   ·   threshold mtime: $(date -r $THRESH '+%H:%M:%S')\n\n"
    printf "  %-28s  %-12s  %s\n" "config" "trials done" "bar (25 folds = 5 seeds × 5 inner)"
    printf "  %-28s  %-12s  %s\n" "------" "-----------" "------------------------------------"
    done_total=0
    for entry in "${CFG[@]}"; do
        phase="${entry%%:*}"
        cfg="${entry##*:}"
        done_n=0
        for seed in 42 43 44 45 46; do
            for fold in 0 1 2 3 4; do
                m="artifacts/predictions/$cfg/r${seed}_f${fold}/meta.json"
                [ -f "$m" ] || continue
                mtime=$(stat -f "%m" "$m" 2>/dev/null) || continue
                [ "$mtime" -gt "$THRESH" ] && done_n=$((done_n+1))
            done
        done
        bar=""; i=1
        while [ $i -le 25 ]; do
            if [ $i -le $done_n ]; then bar="${bar}█"; else bar="${bar}░"; fi
            i=$((i+1))
        done
        # Color cue for which phase is currently active
        emoji="  "
        [ $done_n -gt 0 ] && [ $done_n -lt 25 ] && emoji="🔄"
        [ $done_n -eq 25 ] && emoji="✅"
        printf "  %s %-26s  %4d/25      %s\n" "$emoji" "$cfg" "$done_n" "$bar"
        done_total=$((done_total + done_n))
    done

    pct=$((done_total * 100 / 275))
    printf "\n  TOTAL fresh retrains:  %d / 275  (%d%%)\n" "$done_total" "$pct"

    # Last few retrain log lines (if a retrain is currently running)
    if [ -f /tmp/retrain_start_time ]; then
        st=$(cat /tmp/retrain_start_time)
        elapsed=$((NOW - st))
        printf "  retrain elapsed:       %d s (%d min)\n" "$elapsed" "$((elapsed/60))"
        # show last log line from the running retrain (if logged)
        for log in /tmp/retrain.log /tmp/retrain_dl_ft.log; do
            [ -f "$log" ] && {
                last=$(tail -1 "$log" 2>/dev/null)
                [ -n "$last" ] && printf "  latest log:            %.110s\n" "$last"
                break
            }
        done
    fi
    echo
}

# Single-shot mode
if [ "${1:-}" = "--once" ]; then
    render 0
    exit 0
fi

REFRESH=${1:-10}
trap 'echo; echo "  monitor stopped."; exit 0' INT
while true; do
    render "$REFRESH"
    sleep "$REFRESH"
done
