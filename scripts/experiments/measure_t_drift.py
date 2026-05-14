#!/usr/bin/env python3
"""Measure how much the calibration temperature T drifts across reruns.

Background
----------
The deployed model in artifacts/final_model.pkl carries a single scalar T
(fit on the inner 5-fold OOF probabilities of all 1030 training rows).
Re-running scripts/05_calibrate_best.py from a clean checkout can produce
a slightly different T because the upstream NLL-minimizer's solution
depends on tiny numerical differences (sklearn/numpy/BLAS version).
The argmax predictions are unaffected — calibration only changes the
confidence values.

What this script does
---------------------
Shells out to ``scripts/05_calibrate_best.py`` N times back-to-back and
parses the printed per-fold T values + mean OOF T + deployment T from
the stdout. Reports mean ± std and full range across runs.

Usage
-----
    python3 scripts/experiments/measure_t_drift.py --n-runs 5
    python3 scripts/experiments/measure_t_drift.py --n-runs 10 --out drift.csv

Time
----
Each run = one full ``05_calibrate_best.py``. ~4 min on M4 Max.
``--n-runs 5`` ≈ 20 min.

Heads up — this script overwrites ``artifacts/final_model.pkl`` on every
call (that's what ``05_calibrate_best.py`` does). If you want to preserve
the canonical pkl, back it up first or restore from the Drive zip after.
"""
from __future__ import annotations
import argparse, re, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

# parse lines like "    fold 0: T = 0.6845  (...)"
FOLD_RE      = re.compile(r"fold\s+(\d+):\s*T\s*=\s*([\d.]+)")
DEPLOY_RE    = re.compile(r"deployment T\s*=\s*([\d.]+)")
MEAN_OOF_RE  = re.compile(r"mean T over 5 outer folds\s*=\s*([\d.]+)")


def one_run() -> dict:
    """Run 05_calibrate_best.py once, scrape T values from its stdout."""
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(ROOT/"scripts/05_calibrate_best.py")],
        capture_output=True, text=True, cwd=ROOT, check=True,
    )
    wall = time.time() - t0
    out = proc.stdout + proc.stderr
    fold_Ts  = [float(m.group(2)) for m in FOLD_RE.finditer(out)]
    deploy   = DEPLOY_RE.search(out)
    mean_oof = MEAN_OOF_RE.search(out)
    return {
        "per_fold_T":   fold_Ts,
        "mean_oof_T":   float(mean_oof.group(1)) if mean_oof else float(np.mean(fold_Ts)),
        "deployment_T": float(deploy.group(1))   if deploy   else float("nan"),
        "wall_sec":     wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    t_start = time.time()
    for run in range(args.n_runs):
        r = one_run()
        rows.append({"run": run, **r})
        print(f"[run {run}] mean_oof_T={r['mean_oof_T']:.4f}  "
              f"deployment_T={r['deployment_T']:.4f}  "
              f"per_fold={['%.4f'%t for t in r['per_fold_T']]}  "
              f"({r['wall_sec']:.1f}s)", flush=True)

    df = pd.DataFrame(rows)

    print("\n" + "=" * 64)
    print(f"N runs: {args.n_runs}   total wall: {time.time()-t_start:.1f}s")
    print("-" * 64)
    print(f"mean_oof_T:    mean={df['mean_oof_T'].mean():.4f}  "
          f"std={df['mean_oof_T'].std():.4f}  "
          f"range=[{df['mean_oof_T'].min():.4f}, {df['mean_oof_T'].max():.4f}]")
    print(f"deployment_T:  mean={df['deployment_T'].mean():.4f}  "
          f"std={df['deployment_T'].std():.4f}  "
          f"range=[{df['deployment_T'].min():.4f}, {df['deployment_T'].max():.4f}]")
    print("=" * 64)

    if args.out:
        # csv-friendly: flatten per_fold_T list
        df_out = df.copy()
        df_out["per_fold_T"] = df_out["per_fold_T"].apply(lambda l: ",".join(f"{t:.6f}" for t in l))
        df_out.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
