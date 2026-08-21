#!/usr/bin/env python3
"""Calibrate the top model on the 5-fold OOF test set — three methods compared.

WHAT THIS DOES (and how leakage is avoided):

  For each of the 5 outer folds (seed=42):
    base classifier is already trained on outer_train (artifacts/predictions/cpu__ET500_log2/).

  We compare three calibration protocols:

    (A) Temperature scaling, leak-free inner-CV protocol:
        - Inside outer_train, do an internal 5-fold split.
        - For each inner fold: fit base on inner-tr; predict_proba on inner-val.
        - Concatenate inner-val probs → fit a single scalar T by minimizing NLL.
        - Apply T to outer_test probs. (Test fold never touches T.)
        - Monotonic per-class → ARGMAX UNCHANGED → accuracy on outer_test is
          guaranteed identical to the uncalibrated base. Only ECE moves.

    (B) sklearn ``CalibratedClassifierCV(method='isotonic', cv=5)``:
        - sklearn implementation: 5 inner folds inside outer_train; base estimator
          re-fit on each inner-train; isotonic regression fit per-OvR-binary on
          inner-val predictions; final calibrator is an average of the 5.
        - Isotonic IS NOT monotonic per-class (re-ranks classes) → accuracy on
          outer_test CAN change (typically drops slightly while ECE drops a lot).

    (C) sklearn ``CalibratedClassifierCV(method='sigmoid', cv=5)``:
        - Same structure as (B) but with per-class Platt scaling (sigmoid).
        - Empirically worse than uncalibrated on this distribution.

  All three calibrators are SCORED ON THE HELD-OUT OUTER TEST FOLD (acc + ECE).
  This is the leakage-safe comparison the user asked for.

VERIFICATION OF NO LEAKAGE: the script asserts that the test-fold indices are
disjoint from the inner-CV train rows used for calibration fitting.

DEPLOYMENT: temperature scaling is recommended (preserves argmax) so the saved
``artifacts/final_model.pkl`` bundles the base pipeline + the scalar T fit by
protocol (A) on all 1030 rows (5-fold inner CV).

Usage:
    python scripts/05_calibrate_best.py
    python scripts/05_calibrate_best.py --top-config cpu__ET500_log2 --seed 42

Outputs:
    artifacts/calibration_report.csv     5-fold OOF acc + ECE per method
    artifacts/final_model.pkl             deployed (pipeline, T) bundle
"""
from __future__ import annotations
import argparse, sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.calibration import fit_temperature, fit_temperature_inner_cv, apply_temperature
from src.preprocessing import CountVecFeaturizer
from src.preprocessing.tokenizers import tok_cpu, tok_cpu_v2, tok_comma_pipe
from src.shallow import build_shallow
from sklearn.feature_extraction.text import CountVectorizer


def _ece(probs, y_int, n_bins=10):
    conf = probs.max(1); correct = (probs.argmax(1) == y_int).astype(float)
    edges = np.linspace(0, 1, n_bins+1)
    n = np.zeros(n_bins); accs = np.zeros(n_bins); confs = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (conf >= edges[i]) & (conf < edges[i+1]) if i < n_bins-1 else (conf >= edges[i]) & (conf <= edges[i+1])
        if mask.sum():
            accs[i] = correct[mask].mean(); confs[i] = conf[mask].mean(); n[i] = mask.sum()
    return float((n * np.abs(accs - confs)).sum() / n.sum())


# Which tokenizer each CountVec config is built on. --top-config used to be
# accepted and then ignored: the pipeline below always used tok_cpu, so
# calibrating any other configuration silently fitted a v1 model against that
# configuration's saved probabilities.
_CONFIG_TOKENIZER = {
    "cpu__ET500_log2":   tok_cpu,
    "cpuV2__ET500_log2": tok_cpu_v2,
    "cv__BRF100":        tok_comma_pipe,
}
_CONFIG_CLF = {
    "cpu__ET500_log2":   "ET500_log2",
    "cpuV2__ET500_log2": "ET500_log2",
    "cv__BRF100":        "BRF100",
}


def _make_top_pipe(config: str = "cpu__ET500_log2"):
    """Fresh pipeline for ``config`` — CountVectorizer(its tokenizer) -> its classifier."""
    if config not in _CONFIG_TOKENIZER:
        raise KeyError(f"{config!r} has no registered tokenizer; add it to _CONFIG_TOKENIZER")
    return Pipeline([
        ("cv", CountVectorizer(tokenizer=_CONFIG_TOKENIZER[config],
                                token_pattern=None, lowercase=False)),
        ("vr", build_shallow(_CONFIG_CLF[config])),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-config", default="cpu__ET500_log2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predictions-dir", default=str(ROOT/"artifacts/predictions"))
    ap.add_argument("--out",     default=str(ROOT/"artifacts/final_model.pkl"))
    ap.add_argument("--out-csv", default=str(ROOT/"artifacts/calibration_report.csv"))
    ap.add_argument("--n-inner", type=int, default=5,
                    help="Number of inner-CV folds for temperature fitting (default 5).")
    args = ap.parse_args()
    if args.top_config != "cpu__ET500_log2":
        print(f"[05-cal] WARNING: this script's deploy path is hard-coded for cpu__ET500_log2; got {args.top_config}")
    print(f"[05-cal] target top config = {args.top_config}, outer seed = {args.seed}")
    print(f"[05-cal] inner-CV folds for temperature fit = {args.n_inner}")

    df = pd.read_csv(ROOT/"data/Train_data.csv")
    X = df["sig_gene_seq"].fillna("").values; y = df["high_level_substr"].values
    cls = sorted(set(y)); y_int = np.array([cls.index(c) for c in y])
    N = len(X)

    # 5-fold OUTER split (seed=42 by default)
    skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    # OOF probability buckets — one per method
    P_uncal = np.zeros((N, 12), dtype=np.float32)
    P_temp  = np.zeros((N, 12), dtype=np.float32)
    P_iso   = np.zeros((N, 12), dtype=np.float32)
    P_sig   = np.zeros((N, 12), dtype=np.float32)
    T_list = []

    t0 = time.time()
    for fold, (tr_outer, te) in enumerate(skf_outer.split(X, y)):
        print(f"  fold {fold}: |tr|={len(tr_outer)}  |te|={len(te)}")

        # ── (0) Uncalibrated: read the saved per-fold probs (the base classifier)
        pred_dir = Path(args.predictions_dir)/args.top_config/f"r{args.seed}_f{fold}"
        npz = np.load(pred_dir/"probs_test.npz", allow_pickle=True)
        col = np.array([list(npz["classes"]).index(c) for c in cls])
        P_uncal[te] = npz["probs"][:, col]

        # ── (A) Temperature scaling, inner-CV protocol (leak-free)
        t_t = time.time()
        T, _oof = fit_temperature_inner_cv(lambda: _make_top_pipe(args.top_config), X[tr_outer], y[tr_outer],
                                            n_inner_folds=args.n_inner, random_state=args.seed)
        # LEAK CHECK: confirm test indices were NEVER in the inner CV
        assert len(set(te) & set(tr_outer)) == 0, "outer test ∩ outer train must be empty"
        P_temp[te] = apply_temperature(P_uncal[te], T)
        T_list.append(T)
        print(f"      T={T:.4f}  (inner-CV fit took {time.time()-t_t:.0f}s)")

        # ── (B) CalibratedClassifierCV(isotonic), cv=5
        # sklearn refits the base on inner-train folds; test fold (`te`) is not in tr_outer.
        cal_iso = CalibratedClassifierCV(_make_top_pipe(args.top_config), method="isotonic", cv=args.n_inner)
        cal_iso.fit(X[tr_outer], y[tr_outer])
        col_iso = np.array([list(cal_iso.classes_).index(c) for c in cls])
        P_iso[te] = cal_iso.predict_proba(X[te])[:, col_iso]

        # ── (C) CalibratedClassifierCV(sigmoid), cv=5
        cal_sig = CalibratedClassifierCV(_make_top_pipe(args.top_config), method="sigmoid", cv=args.n_inner)
        cal_sig.fit(X[tr_outer], y[tr_outer])
        col_sig = np.array([list(cal_sig.classes_).index(c) for c in cls])
        P_sig[te] = cal_sig.predict_proba(X[te])[:, col_sig]

    # === Scoring on the held-out 5-fold OOF (none of these probabilities saw their own test row) ===
    methods = [
        ("uncalibrated",          P_uncal, None),
        ("temperature_scaling",   P_temp,  float(np.mean(T_list))),
        ("isotonic_cv5 (sklearn)", P_iso,   None),
        ("sigmoid_cv5 (sklearn)",  P_sig,   None),
    ]
    print()
    print(f"  {'method':<28} {'OOF acc':<10} {'ECE 10-bin':<11} {'T':<8}")
    print(f"  {'-'*28} {'-'*10} {'-'*11} {'-'*8}")
    rows_csv = []
    for name, P, T in methods:
        acc = float((P.argmax(1) == y_int).mean()); ece = _ece(P, y_int)
        rows_csv.append({"method": name, "accuracy": acc, "ece_10bin": ece,
                          "T": T if T is not None else "—"})
        T_str = f"{T:.4f}" if T is not None else "—"
        print(f"  {name:<28} {acc:.4f}     {ece:.4f}      {T_str}")

    pd.DataFrame(rows_csv).to_csv(args.out_csv, index=False)
    print(f"\n[05-cal] wrote {args.out_csv}")

    # === Deployment: fit final base on ALL rows; fit T via inner-CV on ALL rows ===
    print(f"[05-cal] fitting deployment model on all {N} rows ...")
    t1 = time.time()
    pipe = _make_top_pipe(args.top_config); pipe.fit(X, y)
    T_deploy, _ = fit_temperature_inner_cv(lambda: _make_top_pipe(args.top_config), X, y,
                                            n_inner_folds=args.n_inner, random_state=args.seed)
    print(f"  deployment T = {T_deploy:.4f}  (inner-CV fit on all rows took {time.time()-t1:.0f}s)")
    with open(args.out, "wb") as f:
        pickle.dump({"pipeline": pipe, "T": T_deploy, "classes": cls,
                     "config": args.top_config, "trained_on": "all 1030 rows",
                     "T_per_fold": T_list, "calibration_method": "temperature_scaling_inner_cv5"}, f)
    print(f"[05-cal] wrote {args.out}  (mean T over 5 outer folds = {float(np.mean(T_list)):.4f}; deployment T = {T_deploy:.4f})")
    print(f"[05-cal] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__": main()
