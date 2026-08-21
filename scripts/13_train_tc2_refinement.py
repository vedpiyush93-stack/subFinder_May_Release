#!/usr/bin/env python3
"""Post-hoc refinement: retrain the deployed config cpu__ET500_log2 with the
refined tokenizer (tok_cpu_v2).

Same model architecture (CountVec + OvR ExtraTrees-500). Same training data
(all 1030 supervised PULs). Same calibration protocol (inner-CV5 temperature
scaling). Only difference: tokenizer is tok_cpu_v2 instead of tok_cpu, which
truncates TC numbers to their 3-level FAMILY (1.B.14.6.1 -> 1.B.14) and adds a
CAZy family fallback (GH13 -> GH13 + GH).

TC depth revised Aug 2026 from 2-level to 3-level. Level 3 is the TCDB family
(1.B.14 = the TonB-dependent SusC-like receptors) and is the depth the
unsupervised corpus natively uses (99.9% of its TC tokens), so it aligns the
corpora without discarding biology. The previous 2-level form collapsed 596
families into 26 tokens and was no more accurate (0.9151 vs 0.9163 on 5x5 RSKF).

ADDITIVE: this script writes a NEW file artifacts/final_model_v2.pkl
without touching the original artifacts/final_model.pkl. Both models remain
available; the original is still the default for scripts/06_inference.py.
Use --model artifacts/final_model_v2.pkl to load the refined version.

Result vs original tok_cpu (5x5 RSKF, OvR-ExtraTrees-500):
  - Supervised accuracy:  0.9066 -> 0.9163
  - Deployed vocab size:  517 -> 360 tokens
  - Unsupervised mean OOV: 21.3% -> 16.5%

Usage:
    python3 scripts/13_train_tc2_refinement.py
"""
from __future__ import annotations
import os, sys, time, joblib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocessing.tokenizers import tok_cpu_v2
from src.splits import rskf_splits
from src.calibration.temperature import fit_temperature_inner_cv

OUT_PKL = ROOT / "artifacts" / "final_model_v2.pkl"

print("[v2] loading data + setting up winner config with tok_cpu_v2 ...")
df = pd.read_csv(ROOT / "data/Train_data.csv")
X = df["sig_gene_seq"].fillna("").values
y = df["high_level_substr"].values
print(f"  N={len(X)}, classes={len(set(y))}")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import ExtraTreesClassifier

def make_pipe(seed=None):
    """Build the deployed pipeline: CountVectorizer(tok_cpu_v2) -> OvR(ExtraTrees-500).

    ``seed`` defaults to REPRO_REP_SEED, the same model-init seed every other
    configuration in the benchmark uses, so the accuracy reported here matches
    artifacts/leaderboard.csv exactly. It previously varied per fold
    (``seed*7+fold``), which made this script disagree with the leaderboard
    (0.9150 here vs 0.9163 there) for no reason other than model init.
    """
    if seed is None:
        seed = int(os.environ.get("REPRO_REP_SEED", "42"))
    return Pipeline([
        ("cv", CountVectorizer(tokenizer=tok_cpu_v2, lowercase=False, token_pattern=None)),
        ("vr", OneVsRestClassifier(ExtraTreesClassifier(
            n_estimators=500, max_features="log2",
            class_weight="balanced", bootstrap=False,
            random_state=seed, n_jobs=-1))),
    ])

# === 5x5 RSKF benchmark (same protocol as the original) ===
print("\n[tc2] running 5x5 RSKF benchmark with tok_cpu_tc2 ...")
t0 = time.time()
fold_accs, T_list = [], []
for seed, fold, tr_outer, te, _, _ in rskf_splits(y):
    pipe = make_pipe()
    pipe.fit(X[tr_outer], y[tr_outer])
    pred = pipe.predict(X[te])
    fold_accs.append(float((pred == y[te]).mean()))
mean = float(np.mean(fold_accs)); std = float(np.std(fold_accs))
print(f"  5x5 RSKF acc: {mean:.4f} ± {std:.4f}  ({time.time()-t0:.0f}s)")

# === Deployment training (on ALL 1030 rows) + per-fold temperature scaling ===
print("\n[tc2] fitting deployment model on all 1030 rows ...")
t1 = time.time()
pipe_deploy = make_pipe(seed=42)
pipe_deploy.fit(X, y)
print(f"  fit done in {time.time()-t1:.0f}s")
print(f"  deployed vocab size: {len(pipe_deploy.named_steps['cv'].vocabulary_)}")

print("\n[tc2] fitting deployment temperature (inner-CV5 on all rows) ...")
t2 = time.time()
T_deploy, _oof_probs = fit_temperature_inner_cv(make_pipe, X, y,
                                                  n_inner_folds=5, random_state=42)
# Second return is OOF probabilities (1030, 12), not per-fold T values.
# For deployment we only need T_deploy.
T_per_fold = []
print(f"  T_deploy={float(T_deploy):.4f}  ({time.time()-t2:.0f}s)")

# === Save NEW file (don't overwrite the original) ===
classes = list(pipe_deploy.named_steps["vr"].classes_)
bundle = {
    "pipeline": pipe_deploy,
    "T": T_deploy,
    "T_per_fold": T_per_fold,
    "classes": classes,
    "config": "cpu__ET500_log2__v2_refinement",
    "trained_on": "all 1030 rows",
    "calibration_method": "temperature_scaling_inner_cv5",
    "tokenizer": "tok_cpu_v2",
    "refinement_note": (
        "POST-HOC REFINEMENT (May 2026): collapses TC numbers to 2-level family "
        "(1.B.14.6.1 -> 1.B) to close the supervised/unsupervised format mismatch. "
        "All other tokens unchanged. Improves both supervised 5-fold acc (+0.6 pp) "
        "AND drops unsupervised mean OOV from 20% to 8%. See "
        "unravel/experiments/run_token_strategies.py for the strategy sweep."
    ),
    "metrics_5x5_rskf": {"mean": mean, "std": std, "n": len(fold_accs)},
}
joblib.dump(bundle, OUT_PKL, compress=("xz", 6))
print(f"\n[tc2] wrote {OUT_PKL.relative_to(ROOT)} ({OUT_PKL.stat().st_size//1024//1024} MB, xz-compressed; joblib.load auto-decompresses)")
print(f"[tc2] done. Use this with:  python3 scripts/06_inference.py --model {OUT_PKL.relative_to(ROOT)} --seq '...'")
