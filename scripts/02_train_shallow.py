#!/usr/bin/env python3
"""Train shallow classifier configs across all (seed, fold) splits.

Usage:
    python scripts/02_train_shallow.py --reuse                          # skip; verify artifacts present
    python scripts/02_train_shallow.py --retrain                        # retrain all 10 shallow configs (~20min)
    python scripts/02_train_shallow.py --retrain --only cpu__ET500_log2  # retrain one config only

The 10 shallow configs are:
    cpuV2__ET500_log2        <-- deployed     (CountVec_cpu_v2 × OvR(ExtraTrees 500, log2))
    cpu__ET500_log2          <-- v1 winner    (CountVec_cpu × OvR(ExtraTrees 500, log2))
    ftCbow_MM__ET500_sqrt    <-- our second  (FastText CBOW mean+max × OvR(ExtraTrees 500, sqrt))
    cv__BRF100               <-- paper baseline (CountVec_paper × OvR(BalancedRF 100))
    ftCbow__BRF100, ftSg__BRF100, w2vCbow__BRF100, w2vSg__BRF100, d2vDm__BRF100, d2vDbow__BRF100

Embeddings (May 2026)
---------------------
The six embeddings are global — trained once on the unsupervised corpus and
frozen (scripts/01_train_embeddings.py). There is no per-fold variant to select,
so a config's feature matrix no longer depends on the split: we featurize all
1,030 PULs once per config and slice it per fold, instead of re-running the
featurizer 25 times over the same rows.

The two Doc2Vec configs featurize with *document* vectors via ``infer_vector``
(deterministically seeded), not with word vectors — that is what Doc2Vec is
trained to produce, and DBOW never trains word vectors at all.

Outputs per (config, seed, fold):
    artifacts/predictions/<config>/r<seed>_f<fold>/
      classifier.joblib  trained sklearn Pipeline
      probs_test.npz     test-fold probability matrix
      probs_train.npz    train-fold probability matrix
      meta.json          test_acc, wall_sec, hyperparams
"""
from __future__ import annotations
import argparse, sys, json, time
from pathlib import Path

import numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocessing import (CountVecFeaturizer, EmbeddingMeanFeaturizer,
                               EmbeddingMeanMaxFeaturizer, Doc2VecInferFeaturizer)
from src.embeddings.loader import load_word_vectors, load_doc2vec
from src.preprocessing.tokenizers import tok_cpu, tok_comma_pipe
from src.shallow import build_shallow
from src.splits import rskf_splits


SHALLOW_CONFIGS = {
    "cpu__ET500_log2":       dict(featurizer=("countvec", "cpu"),        clf="ET500_log2"),
    "cpuV2__ET500_log2":     dict(featurizer=("countvec", "cpu_v2"),     clf="ET500_log2"),
    "ftCbow_MM__ET500_sqrt": dict(featurizer=("ft_meanmax", "cbow"),     clf="ET500_sqrt"),
    "cv__BRF100":            dict(featurizer=("countvec", "comma_pipe"), clf="BRF100"),
    "ftCbow__BRF100":        dict(featurizer=("emb_mean", "fasttext_cbow"), clf="BRF100"),
    "ftSg__BRF100":          dict(featurizer=("emb_mean", "fasttext_sg"),   clf="BRF100"),
    "w2vCbow__BRF100":       dict(featurizer=("emb_mean", "word2vec_cbow"), clf="BRF100"),
    "w2vSg__BRF100":         dict(featurizer=("emb_mean", "word2vec_sg"),   clf="BRF100"),
    "d2vDm__BRF100":         dict(featurizer=("doc_infer", "doc2vec_dm"),   clf="BRF100"),
    "d2vDbow__BRF100":       dict(featurizer=("doc_infer", "doc2vec_dbow"), clf="BRF100"),
}


_FEATURIZER_CACHE: dict = {}


def _build_featurizer(spec, emb_dir):
    """Instantiate the featurizer for one config spec.

    Embeddings are global, so this no longer takes a fold key. Loaded models are
    memoised — a run touches each architecture once, not once per fold.
    """
    kind = spec[0]
    if kind == "countvec":
        return CountVecFeaturizer(tokenizer=spec[1])

    key = (kind, spec[1])
    if key in _FEATURIZER_CACHE:
        return _FEATURIZER_CACHE[key]

    if kind == "emb_mean":
        feat = EmbeddingMeanFeaturizer(load_word_vectors(spec[1], emb_dir), tokenizer="comma_pipe")
    elif kind == "ft_meanmax":
        feat = EmbeddingMeanMaxFeaturizer(load_word_vectors("fasttext_cbow", emb_dir), tokenizer="comma_pipe")
    elif kind == "doc_infer":
        feat = Doc2VecInferFeaturizer(load_doc2vec(spec[1], emb_dir), tokenizer="comma_pipe")
    else:
        raise ValueError(f"unknown featurizer kind {kind!r}")

    _FEATURIZER_CACHE[key] = feat
    return feat


_DENSE_CACHE: dict = {}


def _dense_features(spec, X, emb_dir):
    """Dense feature matrix for ALL rows, computed once per config spec.

    Safe because a global embedding is identical for every fold: the rows a
    fold trains on are selected afterwards by indexing. Nothing about a test
    row influences a train row's vector.
    """
    key = (spec[0], spec[1])
    if key not in _DENSE_CACHE:
        t0 = time.time()
        _DENSE_CACHE[key] = _build_featurizer(spec, emb_dir).transform(list(X))
        print(f"[02-shallow] featurized {len(X)} PULs with {key[1]} "
              f"({_DENSE_CACHE[key].shape[1]}-d, {time.time()-t0:.1f}s)", flush=True)
    return _DENSE_CACHE[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse",   action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--only",    nargs="+", default=None, help="subset of config shorthands to retrain")
    ap.add_argument("--only-folds", nargs="+", default=None, help="subset of fold keys like 'r42_f0' (default: all 25)")
    ap.add_argument("--emb-dir",   default=str(ROOT/"artifacts/embeddings"))
    ap.add_argument("--out-dir",   default=str(ROOT/"artifacts/predictions"))
    args = ap.parse_args()
    if not args.reuse and not args.retrain: ap.error("specify --reuse or --retrain")
    out_dir = Path(args.out_dir); emb_dir = Path(args.emb_dir)

    if args.reuse:
        ok = 0; total = 0
        for cfg in SHALLOW_CONFIGS:
            for seed in [42,43,44,45,46]:
                for fold in range(5):
                    total += 1
                    p = out_dir/cfg/f"r{seed}_f{fold}"/"meta.json"
                    if p.exists(): ok += 1
        print(f"[02-shallow] reuse check: {ok}/{total} shallow trials present in {out_dir}")
        return

    df = pd.read_csv(ROOT/"data/Train_data.csv")
    X = df["sig_gene_seq"].fillna("").values; y = df["high_level_substr"].values
    targets = args.only or list(SHALLOW_CONFIGS.keys())
    t0 = time.time()
    for seed, fold, tr_outer, te, _, _ in rskf_splits(y):
        fold_key = f"r{seed}_f{fold}"
        if args.only_folds and fold_key not in args.only_folds: continue
        for cfg in targets:
            spec = SHALLOW_CONFIGS[cfg]
            out = out_dir/cfg/fold_key
            out.mkdir(parents=True, exist_ok=True)
            # Resume-friendly: skip trials that already have meta.json
            # (the marker the trial wrote at the very end).
            if (out/"meta.json").exists():
                print(f"[02-shallow] {fold_key} {cfg}: SKIP (meta.json exists)", flush=True)
                continue
            t_t = time.time()
            # For sparse-CountVec we fit + transform via Pipeline; for dense embedding featurizers
            # we transform manually and feed dense matrix to the OvR classifier.
            clf = build_shallow(spec["clf"])
            if spec["featurizer"][0] == "countvec":
                feat = _build_featurizer(spec["featurizer"], emb_dir)
                pipe = Pipeline([("cv", feat._cv), ("vr", clf)])
                pipe.fit(X[tr_outer], y[tr_outer])
                joblib.dump(pipe, out/"classifier.joblib", compress=3)
                P_te = pipe.predict_proba(X[te])
                P_tr = pipe.predict_proba(X[tr_outer])
                classes = list(pipe.named_steps["vr"].classes_)
            else:
                F = _dense_features(spec["featurizer"], X, emb_dir)
                Xtr_d = F[tr_outer]; Xte_d = F[te]
                clf.fit(Xtr_d, y[tr_outer])
                joblib.dump({"featurizer": "dense", "clf": clf, "fold_key": fold_key,
                              "config": cfg}, out/"classifier.joblib", compress=3)
                P_te = clf.predict_proba(Xte_d); P_tr = clf.predict_proba(Xtr_d)
                classes = list(clf.classes_)
            np.savez(out/"probs_test.npz",  probs=P_te.astype(np.float32),
                     classes=np.array(classes, dtype=object), idx=te)
            np.savez(out/"probs_train.npz", probs=P_tr.astype(np.float32),
                     classes=np.array(classes, dtype=object), idx=tr_outer)
            acc = (np.array([classes[i] for i in P_te.argmax(axis=1)]) == y[te]).mean()
            json.dump({"shorthand": cfg, "seed": seed, "fold": fold,
                       "test_acc": float(acc), "wall_sec": time.time()-t_t,
                       "n_test": int(len(te)), "n_train": int(len(tr_outer))},
                      open(out/"meta.json","w"), indent=2)
            print(f"[02-shallow] {fold_key} {cfg}: acc={acc:.4f} ({time.time()-t_t:.1f}s)")
    print(f"[02-shallow] done in {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__": main()
