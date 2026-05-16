#!/usr/bin/env python3
"""Aggregate the 5 rep leaderboards into one cross-rep summary.

For each of the 29 configs:
  * mean & std mean_acc ACROSS reps (captures model-init variance)
  * mean & std std_acc within each rep
  * top1/top2 ranking stability (how often each config lands at rank ≤ k)

Outputs:
    reproducibility/summary/all_reps_leaderboard.csv   long form
    reproducibility/summary/rep_variance.csv           per-config across-rep stats
    reproducibility/summary/ranking_stability.csv      rank histogram per config
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
REPRO = ROOT / "reproducibility"
OUT   = REPRO / "summary"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    rep_dirs = sorted(REPRO.glob("rep_*"))
    if not rep_dirs:
        print("[compare] no rep_* folders yet — nothing to aggregate", file=sys.stderr)
        sys.exit(0)

    for rd in rep_dirs:
        lb = rd / "leaderboard.csv"
        if not lb.exists():
            print(f"[compare] {rd.name}: no leaderboard.csv (skipping)")
            continue
        rep_id = int(rd.name.removeprefix("rep_"))
        df = pd.read_csv(lb)
        df["rep_id"] = rep_id
        df["rep_rank"] = df["rank"]
        rows.append(df)

    if not rows:
        print("[compare] no leaderboards present — exit")
        return
    long_df = pd.concat(rows, ignore_index=True)
    long_df.to_csv(OUT / "all_reps_leaderboard.csv", index=False)

    # across-rep variance per config
    g = long_df.groupby("shorthand")
    summary = pd.DataFrame({
        "acc_mean_across_reps": g["mean_acc"].mean(),
        "acc_std_across_reps":  g["mean_acc"].std(),
        "acc_min_across_reps":  g["mean_acc"].min(),
        "acc_max_across_reps":  g["mean_acc"].max(),
        "within_rep_std_mean":  g["std_acc"].mean(),
        "n_reps":               g["mean_acc"].count(),
    }).sort_values("acc_mean_across_reps", ascending=False).round(5)
    summary.to_csv(OUT / "rep_variance.csv")
    print("--- across-rep summary (top 10):")
    print(summary.head(10))

    # ranking stability: how often each config is at rank ≤ k
    rank_hist = (long_df.assign(top1=lambda d: d["rank"] <= 1,
                                  top3=lambda d: d["rank"] <= 3,
                                  top5=lambda d: d["rank"] <= 5)
                  .groupby("shorthand")[["top1","top3","top5"]].mean()
                  .sort_values("top1", ascending=False).round(3))
    rank_hist.to_csv(OUT / "ranking_stability.csv")
    print("\n--- ranking stability (fraction of reps at rank ≤ k):")
    print(rank_hist.head(10))

    print(f"\n[compare] wrote {OUT}/all_reps_leaderboard.csv")
    print(f"[compare] wrote {OUT}/rep_variance.csv")
    print(f"[compare] wrote {OUT}/ranking_stability.csv")


if __name__ == "__main__":
    main()
