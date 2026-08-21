#!/usr/bin/env bash
# UNIFIED LIVE MONITOR — W2V/D2V retrain + git/LFS push + reproducibility
#
# Shows in one screen, auto-refreshing every 15 s:
#   1. Which long-running processes are alive (W2V/D2V retrain, git push, etc.)
#   2. W2V/D2V retrain progress (12 configs × 25 folds = 300 fits, with bars)
#   3. Git push state (commits ahead/behind, LFS push count)
#   4. Reproducibility suite (per-rep, only renders if reps exist)
#
# Usage:
#   bash scripts/repro/all_status.sh             # refresh every 15s
#   bash scripts/repro/all_status.sh --once      # one-shot
#   bash scripts/repro/all_status.sh 30          # custom interval
set -euo pipefail
cd "$(dirname "$0")/../.."

INTERVAL=15
ONCE=0
if [ "${1:-}" = "--once" ]; then ONCE=1; fi
[ "${1:-}" -eq "${1:-}" ] 2>/dev/null && INTERVAL=$1

WV_DV_CONFIGS=(
  w2vCbow__LSTM w2vCbow__LSTMattn w2vCbow__JustAttn w2vCbow__Trans
  w2vSg__LSTM   w2vSg__LSTMattn   w2vSg__JustAttn   w2vSg__Trans
)

REPRO_CONFIGS=(
  cpu__ET500_log2 ftCbow_MM__ET500_sqrt cv__BRF100
  ftCbow__BRF100 ftSg__BRF100 w2vCbow__BRF100 w2vSg__BRF100
  d2vDm__BRF100 d2vDbow__BRF100
  ftCbow__LSTM   ftSg__LSTM   w2vCbow__LSTM   w2vSg__LSTM
  ftCbow__LSTMattn ftSg__LSTMattn w2vCbow__LSTMattn w2vSg__LSTMattn
  ftCbow__JustAttn ftSg__JustAttn w2vCbow__JustAttn w2vSg__JustAttn
  ftCbow__Trans  ftSg__Trans  w2vCbow__Trans  w2vSg__Trans
)

bar25() {
    local n=$1
    local s=""
    for i in $(seq 1 25); do
        if [ "$i" -le "$n" ]; then s="${s}█"; else s="${s}░"; fi
    done
    echo "$s"
}

render() {
    clear 2>/dev/null || true
    local now=$(date '+%H:%M:%S')
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  RELEASE-REPO PIPELINE — LIVE  ($now,  refresh ${INTERVAL}s, Ctrl+C to exit)"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo

    # ─── 1. PROCESS HEALTH ─────────────────────────────────────────────────────
    echo "▶ PROCESS HEALTH"
    local n_retrain=$(pgrep -fa "03_train_deep\.py" 2>/dev/null | wc -l | tr -d ' ')
    local n_push=$(pgrep -fa "git push" 2>/dev/null | wc -l | tr -d ' ')
    local n_lfs=$(pgrep -fa "git-lfs" 2>/dev/null | wc -l | tr -d ' ')
    local n_compress=$(pgrep -fa "nirvana_compress" 2>/dev/null | wc -l | tr -d ' ')
    local n_xz=$(pgrep -fa "^xz " 2>/dev/null | wc -l | tr -d ' ')
    local n_repro=$(pgrep -fa "run_one_rep\.py\|run_repro_suite" 2>/dev/null | wc -l | tr -d ' ')

    [ "$n_retrain" -gt 0 ] && echo "  ✅ W2V/D2V retrain     ($n_retrain proc)" || echo "  ❌ W2V/D2V retrain     (NOT RUNNING)"
    [ "$n_push" -gt 0 ] && echo "  ✅ git push             ($n_push proc)"     || echo "  ⏸  git push             (idle)"
    [ "$n_lfs" -gt 0 ] && echo "  ✅ git-lfs              ($n_lfs proc)"       || echo "  ⏸  git-lfs              (idle)"
    [ "$n_compress" -gt 0 ] && echo "  ✅ LFS compress         ($n_compress proc)" || echo "  ⏸  LFS compress         (idle)"
    [ "$n_xz" -gt 0 ] && echo "  ✅ xz                   ($n_xz proc)"           || true
    [ "$n_repro" -gt 0 ] && echo "  ✅ reproducibility      ($n_repro proc)"     || echo "  ⏸  reproducibility      (not yet started)"
    echo

    # ─── 2. W2V/D2V RETRAIN ────────────────────────────────────────────────────
    local start=$(cat /tmp/wv_dv_retrain_start_time 2>/dev/null || echo 0)
    local elapsed=$(( $(date +%s) - start ))
    echo "▶ W2V/D2V DL RETRAIN (target 300 fits)   started: $(date -r $start '+%H:%M:%S')   elapsed: $((elapsed / 60)) min"
    local total=0
    for cfg in "${WV_DV_CONFIGS[@]}"; do
        local n=0
        [ -d "artifacts/predictions/$cfg" ] && \
            n=$(find "artifacts/predictions/$cfg" -name meta.json \
                  -newer /tmp/wv_dv_retrain_start_time 2>/dev/null | wc -l | tr -d ' ')
        total=$((total + n))
        local marker="  "
        [ "$n" -eq 25 ] && marker="✅"
        [ "$n" -gt 0 ] && [ "$n" -lt 25 ] && marker="🔄"
        printf "  %s %-22s  %4d/25   %s\n" "$marker" "$cfg" "$n" "$(bar25 $n)"
    done
    local pct=$((total * 100 / 300))
    echo "  TOTAL: $total / 300  (${pct}%)"
    if [ "$total" -lt 300 ] && [ "$total" -gt 0 ]; then
        local rate=$((elapsed / total))
        local eta=$(( (300 - total) * rate / 60 ))
        echo "  ETA: ~${eta} min   (avg ${rate} s/fit)"
    fi
    echo

    # ─── 3. GIT PUSH STATE ─────────────────────────────────────────────────────
    echo "▶ GIT PUSH / GITHUB"
    git branch -vv 2>/dev/null | head -1 | sed 's/^/  /'
    local lfs_ct=$(git lfs ls-files 2>/dev/null | wc -l | tr -d ' ')
    echo "  LFS files tracked: $lfs_ct   (target: 101 = 1 final_model + 100 ngram .xz)"
    if [ -f /tmp/push.log ]; then
        echo "  last push.log lines:"
        tail -3 /tmp/push.log 2>/dev/null | sed 's/^/    /'
    fi
    echo

    # ─── 4. REPRO SUITE ────────────────────────────────────────────────────────
    if [ -d reproducibility ] && [ "$(ls reproducibility 2>/dev/null | grep -c '^rep_')" -gt 0 ]; then
        echo "▶ REPRODUCIBILITY SUITE (5 reps × 29 configs × 25 folds = 3625 fits target)"
        local total_repro=0
        for r in 1 2 3 4 5; do
            local rd="reproducibility/rep_$r"
            [ ! -d "$rd" ] && continue
            local rep_done=0
            for cfg in "${REPRO_CONFIGS[@]}"; do
                local n=0
                [ -d "$rd/predictions/$cfg" ] && \
                    n=$(find "$rd/predictions/$cfg" -name meta.json 2>/dev/null | wc -l | tr -d ' ')
                rep_done=$((rep_done + n))
            done
            total_repro=$((total_repro + rep_done))
            local lbm=""; [ -f "$rd/leaderboard.csv" ] && lbm="  ✅ leaderboard"
            echo "  rep_${r}:  ${rep_done} / 725  ${lbm}"
        done
        local pct=$((total_repro * 100 / 3625))
        echo "  TOTAL repro: $total_repro / 3625  (${pct}%)"
    else
        echo "▶ REPRODUCIBILITY SUITE — not yet started (auto-launches after W2V/D2V commit + push)"
    fi
    echo
    echo "════════════════════════════════════════════════════════════════════════════════"
}

if [ $ONCE -eq 1 ]; then render; exit 0; fi
while true; do
    render
    sleep "$INTERVAL"
done