#!/usr/bin/env bash
# Pretty live progress for the 11-config FT retrain. Run alongside
# scripts/retrain_dl_ft.sh in a separate terminal.
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
printf "\n%-28s  %-12s  %s\n" "config" "trials done" "bar (25 folds = 5 seeds × 5 inner folds)"
printf "%-28s  %-12s  %s\n"   "------" "-----------" "-----------------------------------------"
done_total=0
# "Done" = meta.json modified AFTER the bug-fix commit (= scripts/02_train_shallow.py
# mtime, which the bug-fix commit just touched). Any meta.json newer than the
# script was written by the new code with n-gram OOV.
THRESH=$(stat -f "%m" scripts/02_train_shallow.py)
for entry in "${CFG[@]}"; do
  cfg="${entry##*:}"
  done_n=0
  for seed in 42 43 44 45 46; do
    for fold in 0 1 2 3 4; do
      m="artifacts/predictions/$cfg/r${seed}_f${fold}/meta.json"
      [ -f "$m" ] || continue
      mtime=$(stat -f "%m" "$m")
      [ "$mtime" -gt "$THRESH" ] && done_n=$((done_n+1))
    done
  done
  bar=""; i=1
  while [ $i -le 25 ]; do
    if [ $i -le $done_n ]; then bar="${bar}█"; else bar="${bar}░"; fi
    i=$((i+1))
  done
  printf "%-28s  %4d/25      %s\n" "$cfg" "$done_n" "$bar"
  done_total=$((done_total + done_n))
done
printf "\nTOTAL fresh retrains: %d / 275\n\n" "$done_total"
