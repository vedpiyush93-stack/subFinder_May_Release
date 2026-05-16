#!/usr/bin/env python3
"""Train shallow classifier configs across all (seed, fold) splits.

Usage:
    python scripts/02_train_shallow.py --reuse                          # skip; verify artifacts present
    python scripts/02_train_shallow.py --retrain                        # retrain all 9 shallow configs (~30min)
    python scripts/02_train_shallow.py --retrain --only cpu__ET500_log2  # retrain one config only

The 9 shallow configs are:
    cpu__ET500_log2          <-- our winner  (CountVec_cpu × OvR(ExtraTrees 500, log2))
    ftCbow_MM__ET500_sqrt    <-- our second  (FastText CBOW mean+max × OvR(ExtraTrees 500, sqrt))
    cv__BRF100               <-- paper baseline (CountVec_paper × OvR(BalancedRF 100))
    ftCbow__BRF100, ftSg__BRF100, w2vCbow__BRF100, w2vSg__BRF100, d2vDm__BRF100, d2vDbow__BRF100

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
from src.preprocessing import CountVecFeaturizer, EmbeddingMeanFeaturizer, EmbeddingMeanMaxFeaturizer
from src.preprocessing.tokenizers import tok_cpu, tok_comma_pipe
from src.shallow import build_shallow
from src.splits import rskf_splits


SHALLOW_CONFIGS = {
    "cpu__ET500_log2":       dict(featurizer=("countvec", "cpu"),        clf="ET500_log2"),
    "ftCbow_MM__ET500_sqrt": dict(featurizer=("ft_meanmax", "cbow"),     clf="ET500_sqrt"),
    "cv__BRF100":            dict(featurizer=("countvec", "comma_pipe"), clf="BRF100"),
    "ftCbow__BRF100":        dict(featurizer=("emb_mean", "fasttext_cbow"), clf="BRF100"),
    "ftSg__BRF100":          dict(featurizer=("emb_mean", "fasttext_sg"),   clf="BRF100"),
    "w2vCbow__BRF100":       dict(featurizer=("emb_mean", "word2vec_cbow"), clf="BRF100"),
    "w2vSg__BRF100":         dict(featurizer=("emb_mean", "word2vec_sg"),   clf="BRF100"),
    "d2vDm__BRF100":         dict(featurizer=("emb_mean", "doc2vec_dm"),    clf="BRF100"),
    "d2vDbow__BRF100":       dict(featurizer=("emb_mean", "doc2vec_dbow"),  clf="BRF100"),
}


def _load_emb_npz(cache_dir: Path, fold_key: str, arch: str, kind: str = "shallow"):
    """Lightweight npz-only KeyedVectors shim — for W2V/D2V (no n-grams).

    For FastText we use _load_emb_ft (full model + n-gram fallback) instead.

    Compatible with both the shipped npz schema (``vocab`` / ``vectors``) and
    the legacy ``keys`` / ``vectors`` schema used by older versions of
    ``01_train_embeddings.py``.
    """
    npz = np.load(cache_dir/fold_key/f"{arch}_{kind}.npz", allow_pickle=True)
    vocab_field = "vocab" if "vocab" in npz.files else "keys"
    keys = npz[vocab_field]; vecs = npz["vectors"]
    class _WV:
        def __init__(self, keys, vecs):
            self.vector_size = vecs.shape[1] if vecs.shape[0] else 300
            self._idx = {str(k): i for i, k in enumerate(keys)}
            self.vectors = vecs
        def __getitem__(self, t):
            i = self._idx.get(t)
            if i is None: raise KeyError(t)
            return self.vectors[i]
        def __contains__(self, t): return t in self._idx
    return _WV(keys, vecs)


def _load_emb_ft(cache_dir: Path, fold_key: str, arch: str, kind: str = "shallow"):
    """Load the FULL FastText gensim model so wv[t] does n-gram fallback on OOV.

    Priority:
      1. If ``FT_FULL_DIR`` env var is set AND contains the model, use it
         (development convenience — avoids the ~70 s xz decompress per fold).
      2. Otherwise use the release-repo path with auto-decompress of .npy.xz
         (the normal reviewer path after ``git clone`` + ``git lfs pull``).
    """
    import os
    arch_stem = arch.replace("fasttext_", "")     # "cbow" or "sg"
    sub = f"{arch}_{kind}_model"
    model_fn = f"fasttext_{arch_stem}.model"

    ft_full = os.environ.get("FT_FULL_DIR")
    if ft_full:
        src_path = Path(ft_full)/fold_key/sub/model_fn
        if src_path.exists():
            from gensim.models import FastText
            return FastText.load(str(src_path)).wv

    from src.embeddings.loader import load_fasttext
    rel_path = cache_dir/fold_key/sub/model_fn
    if rel_path.exists():
        return load_fasttext(rel_path).wv

    raise FileNotFoundError(
        f"FastText full model not found at {rel_path}"
        + (f" or {Path(ft_full)/fold_key/sub/model_fn}" if ft_full else "")
        + ". Either pull from LFS (git lfs fetch) or set FT_FULL_DIR to a "
        "local regenerated cache (see scripts/01_train_embeddings.py).")


def _build_featurizer(spec, fold_key: str, cache_dir: Path):
    kind = spec[0]
    if kind == "countvec":
        return CountVecFeaturizer(tokenizer=spec[1])
    arch_alias = {"fasttext_cbow": "fasttext_cbow", "fasttext_sg": "fasttext_sg",
                  "word2vec_cbow": "word2vec_cbow", "word2vec_sg": "word2vec_sg",
                  "doc2vec_dm": "doc2vec_dm", "doc2vec_dbow": "doc2vec_dbow"}
    if kind == "emb_mean":
        arch = arch_alias[spec[1]]
        if arch.startswith("fasttext_"):
            wv = _load_emb_ft(cache_dir, fold_key, arch, kind="shallow")
        else:
            wv = _load_emb_npz(cache_dir, fold_key, arch, kind="shallow")
        return EmbeddingMeanFeaturizer(wv, tokenizer="comma_pipe")
    if kind == "ft_meanmax":
        wv = _load_emb_ft(cache_dir, fold_key, "fasttext_cbow", kind="shallow")
        return EmbeddingMeanMaxFeaturizer(wv, tokenizer="comma_pipe")
    raise ValueError(f"unknown featurizer kind {kind!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse",   action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--only",    nargs="+", default=None, help="subset of config shorthands to retrain")
    ap.add_argument("--only-folds", nargs="+", default=None, help="subset of fold keys like 'r42_f0' (default: all 25)")
    ap.add_argument("--cache-dir", default=str(ROOT/"artifacts/embeddings_cache"))
    ap.add_argument("--out-dir",   default=str(ROOT/"artifacts/predictions"))
    args = ap.parse_args()
    if not args.reuse and not args.retrain: ap.error("specify --reuse or --retrain")
    out_dir = Path(args.out_dir); cache_dir = Path(args.cache_dir)

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
            feat = _build_featurizer(spec["featurizer"], fold_key, cache_dir)
            # For sparse-CountVec we fit + transform via Pipeline; for dense embedding featurizers
            # we transform manually and feed dense matrix to the OvR classifier.
            clf = build_shallow(spec["clf"])
            if spec["featurizer"][0] == "countvec":
                pipe = Pipeline([("cv", feat._cv), ("vr", clf)])
                pipe.fit(X[tr_outer], y[tr_outer])
                joblib.dump(pipe, out/"classifier.joblib", compress=3)
                P_te = pipe.predict_proba(X[te])
                P_tr = pipe.predict_proba(X[tr_outer])
                classes = list(pipe.named_steps["vr"].classes_)
            else:
                Xtr_d = feat.transform(X[tr_outer]); Xte_d = feat.transform(X[te])
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
