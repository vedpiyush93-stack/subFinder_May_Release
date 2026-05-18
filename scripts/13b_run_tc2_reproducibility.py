#!/usr/bin/env python3
"""Full 5-rep reproducibility study for the TC2 refinement — mirrors the
treatment we gave the original deployed model.

For each rep N in {1, 2, 3, 4, 5} with model-init seed = 1000 * N:
  - Run full 5x5 RSKF benchmark (cpu__ET500_log2 with tok_cpu_tc2)
  - Per-fold predictions saved to reproducibility/rep_<N>_tc2/predictions/
  - Per-fold temperature calibration
  - Deployment model on all 1030 rows + deployment T

Then aggregate cross-rep stats:
  - Per-rep mean acc (across 25 fits)
  - Cross-rep mean ± std (across 5 reps)
  - Comparison vs original tok_cpu deployed model

Saves:
  reproducibility/rep_<N>_tc2/leaderboard.csv       per-fold + per-config metrics
  reproducibility/rep_<N>_tc2/final_model.pkl       deployed TC2 model for that rep
  reproducibility/v2_cross_rep_summary.json        aggregate across 5 reps

Wall: ~20 min (5 reps x ~4 min each on M4 Max with n_jobs=-1).
"""
from __future__ import annotations
import sys, time, json, joblib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocessing.tokenizers import tok_cpu_v2
from src.splits import rskf_splits
from src.calibration.temperature import fit_temperature_inner_cv

REPRO = ROOT / "reproducibility"
print("[tc2-repro] loading data ...")
df = pd.read_csv(ROOT / "data/Train_data.csv")
X = df["sig_gene_seq"].fillna("").values
y = df["high_level_substr"].values

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import ExtraTreesClassifier

def make_pipe(seed):
    return Pipeline([
        ("cv", CountVectorizer(tokenizer=tok_cpu_v2, lowercase=False, token_pattern=None)),
        ("vr", OneVsRestClassifier(ExtraTreesClassifier(
            n_estimators=500, max_features="log2",
            class_weight="balanced", bootstrap=False,
            random_state=seed, n_jobs=-1))),
    ])

REP_SEEDS = [1000, 2000, 3000, 4000, 5000]
per_rep = []

for rep_i, repro_seed in enumerate(REP_SEEDS, start=1):
    rep_dir = REPRO / f"rep_{rep_i}_v2"
    rep_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[tc2-repro] === rep {rep_i}/5 (REPRO_REP_SEED={repro_seed}) ===")
    t_rep = time.time()
    fold_rows = []
    for seed, fold, tr_outer, te, _, _ in rskf_splits(y):
        pipe = make_pipe(seed=repro_seed + seed * 7 + fold)
        pipe.fit(X[tr_outer], y[tr_outer])
        pred = pipe.predict(X[te])
        acc = float((pred == y[te]).mean())
        fold_rows.append({"shorthand": "cpu__ET500_log2__v2",
                          "repeat_seed": seed, "fold": fold, "acc": acc,
                          "n_train": int(len(tr_outer)), "n_test": int(len(te))})
        print(f"  r{seed}_f{fold}: acc={acc:.4f}", flush=True)
    per_fold_df = pd.DataFrame(fold_rows)
    per_fold_df.to_csv(rep_dir / "per_fold_metrics.csv", index=False)
    leaderboard = per_fold_df.groupby("shorthand").agg(
        mean_acc=("acc", "mean"), std_acc=("acc", "std"), n=("acc", "count")
    ).reset_index()
    leaderboard.to_csv(rep_dir / "leaderboard.csv", index=False)
    rep_acc_mean = float(per_fold_df.acc.mean())
    rep_acc_std = float(per_fold_df.acc.std())
    print(f"  rep_{rep_i}_tc2 5x5 acc: {rep_acc_mean:.4f} ± {rep_acc_std:.4f}  "
          f"({time.time()-t_rep:.0f}s)")

    # Deployed model on all 1030 + temperature
    print(f"  [rep_{rep_i}] fitting deployment model + T ...")
    t_dep = time.time()
    pipe_dep = make_pipe(seed=repro_seed)
    pipe_dep.fit(X, y)
    T_dep, _ = fit_temperature_inner_cv(lambda: make_pipe(seed=repro_seed+1),
                                          X, y, n_inner_folds=5, random_state=42)
    print(f"  T_deploy={float(T_dep):.4f}  vocab={len(pipe_dep.named_steps['cv'].vocabulary_)}  "
          f"({time.time()-t_dep:.0f}s)")

    joblib.dump({
        "pipeline": pipe_dep, "T": float(T_dep),
        "classes": list(pipe_dep.named_steps["vr"].classes_),
        "config": "cpu__ET500_log2__v2_refinement",
        "trained_on": "all 1030 rows",
        "tokenizer": "tok_cpu_v2",
        "rep_id": rep_i, "repro_seed": repro_seed,
    }, rep_dir / "final_model.pkl", compress=("xz", 6))
    per_rep.append({
        "rep_id": rep_i, "repro_seed": repro_seed,
        "n_trials": int(len(per_fold_df)),
        "mean_acc": rep_acc_mean, "std_acc": rep_acc_std,
        "T_deploy": float(T_dep),
        "vocab_size": int(len(pipe_dep.named_steps['cv'].vocabulary_)),
        "wall_sec": round(time.time() - t_rep, 1),
    })

# Cross-rep aggregate
rep_means = [r["mean_acc"] for r in per_rep]
cross_rep_mean = float(np.mean(rep_means))
cross_rep_std = float(np.std(rep_means))
cross_rep_range = [float(min(rep_means)), float(max(rep_means))]
print(f"\n[tc2-repro] === CROSS-REP SUMMARY ===")
print(f"  per-rep means: {[round(r, 4) for r in rep_means]}")
print(f"  cross-rep mean = {cross_rep_mean:.4f} ± {cross_rep_std:.4f}")
print(f"  range = [{cross_rep_range[0]:.4f}, {cross_rep_range[1]:.4f}]")

summary = {
    "model": "cpu__ET500_log2 with tok_cpu_tc2 (post-hoc refinement)",
    "n_reps": len(per_rep),
    "rep_seeds": REP_SEEDS,
    "per_rep": per_rep,
    "cross_rep_mean": cross_rep_mean,
    "cross_rep_std": cross_rep_std,
    "cross_rep_min": cross_rep_range[0],
    "cross_rep_max": cross_rep_range[1],
    "original_5rep_mean_for_comparison": 0.9063,  # from earlier cross-rep work
    "delta_vs_original": round(cross_rep_mean - 0.9063, 4),
}
out = REPRO / "v2_cross_rep_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"\n[tc2-repro] wrote {out.relative_to(ROOT)}")
print(f"[tc2-repro] Δ vs original 5-rep mean (0.9063): {cross_rep_mean - 0.9063:+.4f}")
