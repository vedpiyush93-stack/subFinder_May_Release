#!/usr/bin/env python3
"""Aggregate all 29 trained configurations into a single leaderboard.

Walks artifacts/predictions/*/r*_f*/meta.json, computes per-config 5×5 RSKF
means/stds and per-substrate F1 for the top model. Writes:

    artifacts/leaderboard.csv      29-row leaderboard, sorted by mean acc desc
    artifacts/per_fold_metrics.csv full per-trial CSV (already exists; verify)

Usage:
    python scripts/04_benchmark.py
    python scripts/04_benchmark.py --only-complete    # only configs with all 25 trials
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-dir", default=str(ROOT/"artifacts/predictions"))
    ap.add_argument("--only-complete", action="store_true",
                    help="Drop configs with fewer than 25 trials.")
    ap.add_argument("--out", default=str(ROOT/"artifacts/leaderboard.csv"))
    args = ap.parse_args()

    rows = [json.load(open(p)) for p in glob.glob(str(Path(args.predictions_dir)/"*/r*_f*/meta.json"))]
    if not rows:
        sys.exit(f"[04-bench] no meta.json files under {args.predictions_dir}")
    df = pd.DataFrame(rows)
    print(f"[04-bench] {len(df)} per-trial rows across {df.shorthand.nunique()} configs")

    agg = df.groupby("shorthand").agg(
        mean_acc=("test_acc","mean"),
        std_acc=("test_acc","std"),
        n_trials=("test_acc","count"),
        min_acc=("test_acc","min"),
        max_acc=("test_acc","max"),
        median_acc=("test_acc","median"),
    ).reset_index()
    if args.only_complete:
        agg = agg[agg.n_trials == 25].copy()
    agg = agg.sort_values("mean_acc", ascending=False).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg)+1)
    agg.to_csv(args.out, index=False)
    print(f"[04-bench] wrote {args.out} ({len(agg)} rows)")
    print()
    print(agg.head(10).to_string(index=False))
    print()
    if len(agg) >= 2:
        top1, top2 = agg.iloc[0], agg.iloc[1]
        print(f"  #1 {top1.shorthand}: {top1.mean_acc:.4f} ± {top1.std_acc:.4f}")
        print(f"  #2 {top2.shorthand}: {top2.mean_acc:.4f} ± {top2.std_acc:.4f}")
        if "cv__BRF100" in agg.shorthand.values:
            base = agg[agg.shorthand == "cv__BRF100"].iloc[0]
            print(f"  Paper baseline cv__BRF100: {base.mean_acc:.4f} ± {base.std_acc:.4f}")
            print(f"  Gap (top - baseline): +{(top1.mean_acc - base.mean_acc)*100:.2f} pp")
        if "ftSg__LSTMattn" in agg.shorthand.values:
            dl = agg[agg.shorthand == "ftSg__LSTMattn"].iloc[0]
            print(f"  Best paper DL ftSg__LSTMattn: {dl.mean_acc:.4f} ± {dl.std_acc:.4f}")
            print(f"  Gap (top - best DL): +{(top1.mean_acc - dl.mean_acc)*100:.2f} pp")


if __name__ == "__main__": main()
