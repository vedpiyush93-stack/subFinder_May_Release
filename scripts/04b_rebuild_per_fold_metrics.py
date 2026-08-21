#!/usr/bin/env python3
"""Rebuild artifacts/per_fold_metrics.csv from artifacts/predictions/.

Why this exists
---------------
``04_benchmark.py`` writes ``leaderboard.csv`` but treats ``per_fold_metrics.csv``
as pre-existing ("already exists; verify"). Nothing regenerated it, so after a
re-run the leaderboard was fresh while the per-fold CSV still held the previous
run's numbers — and ``07/08/09`` prefer that CSV over the raw ``meta.json``
files, so stale rows would silently propagate into the paper tables, figures
and decks.

This script recomputes every per-trial row from the shipped probability
matrices, preserving the original column schema. Static per-config descriptive
columns (family / role / featurizer / classifier / description) are carried
over from the previous CSV when available, since they describe the
configuration rather than the run.

    python scripts/04b_rebuild_per_fold_metrics.py
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COLS = ["config","shorthand","family","role","featurizer","classifier","description",
        "repeat_seed","fold","acc","classwise","precision_macro","recall_macro",
        "f1_macro","best_substrate","worst_substrate","wall_sec"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-dir", default=str(ROOT/"artifacts/predictions"))
    ap.add_argument("--out", default=str(ROOT/"artifacts/per_fold_metrics.csv"))
    args = ap.parse_args()

    y = pd.read_csv(ROOT/"data/Train_data.csv")["high_level_substr"].values

    # carry forward the descriptive columns keyed by shorthand
    meta_cols = {}
    prev = Path(args.out)
    if prev.exists():
        for r in csv.DictReader(open(prev)):
            meta_cols.setdefault(r["shorthand"],
                {k: r.get(k, "") for k in ("config","family","role","featurizer","classifier","description")})

    rows = []
    for mp in sorted(Path(args.predictions_dir).glob("*/r*_f*/meta.json")):
        m = json.load(open(mp))
        probs_p = mp.parent/"probs_test.npz"
        if not probs_p.exists():
            print(f"[04b] skip {mp.parent} (no probs_test.npz)", file=sys.stderr); continue
        z = np.load(probs_p, allow_pickle=True)
        classes = [str(c) for c in z["classes"]]
        idx = z["idx"]; pred = np.array(classes)[z["probs"].argmax(axis=1)]
        true = y[idx]
        p, r, f1, _ = precision_recall_fscore_support(true, pred, average="macro", zero_division=0)
        per = f1_score(true, pred, average=None, labels=classes, zero_division=0)
        order = np.argsort(per)
        d = dict(meta_cols.get(m["shorthand"], {}))
        d.update(shorthand=m["shorthand"], repeat_seed=m["seed"], fold=m["fold"],
                 acc=float((pred == true).mean()), classwise="",
                 precision_macro=float(p), recall_macro=float(r), f1_macro=float(f1),
                 best_substrate=f"{classes[order[-1]]}={per[order[-1]]:.2f}",
                 worst_substrate=f"{classes[order[0]]}={per[order[0]]:.2f}",
                 wall_sec=m.get("wall_sec", ""))
        rows.append({c: d.get(c, "") for c in COLS})

    df = pd.DataFrame(rows, columns=COLS).sort_values(["shorthand","repeat_seed","fold"])
    df.to_csv(args.out, index=False)
    print(f"[04b] wrote {args.out} ({len(df)} rows, {df.shorthand.nunique()} configs)")
    chk = df.groupby("shorthand").acc.mean().sort_values(ascending=False)
    print(f"[04b] top: {chk.index[0]} = {chk.iloc[0]:.4f}")


if __name__ == "__main__": main()
