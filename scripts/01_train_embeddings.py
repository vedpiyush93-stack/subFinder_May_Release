#!/usr/bin/env python3
"""Train per-fold word embeddings (6 architectures × 25 splits = 150 models).

Each fold gets its own embedding training corpus to keep test tokens out of
the embedding space (the leakage we fixed vs. the paper's pre-trained models).

Usage:
    python scripts/01_train_embeddings.py --reuse-cache    # default: load from artifacts/embeddings_cache (already shipped in repo)
    python scripts/01_train_embeddings.py --retrain        # retrain from scratch into artifacts/embeddings_cache (~6h)
    python scripts/01_train_embeddings.py --retrain --only-archs fasttext_cbow fasttext_sg
    python scripts/01_train_embeddings.py --unsupervised-csv data/unsupervised.csv  # use external corpus

Cache state in this repo
------------------------
The cache is the same fold_cache_v2/ layout as the original benchmark, with one
gensim model per (seed, fold, architecture). After a fresh ``git clone`` you
already have the reduced subset needed for downstream training and inference:

    * ``r*_f*/<arch>_<regime>.npz``               — vector lookup tables (regular git, ~446 MB)
    * ``r*_f*/fasttext_*_model/{*.model, *.npy.xz}`` — n-gram-OOV-ready FastText
      gensim model dirs (xz-compressed via Git LFS; ``src/embeddings/loader.py``
      auto-decompresses on first load)

The Word2Vec / Doc2Vec full model dirs are NOT shipped — those architectures
don't have n-gram OOV, so their ``.npz`` is functionally bit-identical to the
full model dir for any inference. Use ``--retrain`` only when you want to
rebuild the embeddings themselves from scratch (e.g. with a different
unsupervised corpus).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocessing.tokenizers import tok_comma_pipe
from src.embeddings import EMB_ARCHITECTURES, train_embedding, EMB_HYPERPARAMS
from src.splits import rskf_splits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse-cache", action="store_true",
                    help="Skip training; verify cache integrity only.")
    ap.add_argument("--retrain", action="store_true",
                    help="Retrain all (seed, fold, arch) embeddings from scratch.")
    ap.add_argument("--cache-dir", default=str(ROOT/"artifacts/embeddings_cache"),
                    help="Per-fold embedding cache directory.")
    ap.add_argument("--unsupervised-csv", default=None,
                    help="Optional external unsupervised corpus to augment per-fold embedding training.")
    ap.add_argument("--only-archs", nargs="+", default=None,
                    help="Subset of architectures to train (default: all 6).")
    ap.add_argument("--only-folds", nargs="+", type=str, default=None,
                    help="Subset like 'r42_f0' 'r42_f1' (default: all 25 = 5 seeds × 5 folds).")
    args = ap.parse_args()
    if not args.reuse_cache and not args.retrain:
        ap.error("specify either --reuse-cache or --retrain")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_cache:
        # Verify integrity of cached models
        n_total = n_ok = 0
        for fold_dir in sorted(cache_dir.glob("r*_f*")):
            for arch in EMB_ARCHITECTURES:
                n_total += 1
                kinds = ["shallow", "dl"]
                for kind in kinds:
                    if (fold_dir/f"{arch}_{kind}.npz").exists():
                        n_ok += 1; break
        print(f"[01-emb] cache check: {n_ok}/{n_total} (seed,fold,arch) entries present in {cache_dir}")
        if n_ok == 0:
            print("[01-emb] cache empty — download from README §5 or pass --retrain", file=sys.stderr)
            sys.exit(1)
        return

    # Retrain path
    df = pd.read_csv(ROOT/"data/Train_data.csv")
    X = df["sig_gene_seq"].fillna("").values; y = df["high_level_substr"].values
    sentences_sup = [tok_comma_pipe(s) for s in X]
    sentences_unsup = []
    if args.unsupervised_csv:
        df_u = pd.read_csv(args.unsupervised_csv)
        sentences_unsup = [tok_comma_pipe(s) for s in df_u["sig_gene_seq"].fillna("").values]
        print(f"[01-emb] augmenting with {len(sentences_unsup)} unsupervised PULs")

    archs = args.only_archs or list(EMB_ARCHITECTURES.keys())
    t_total = time.time()
    for seed, fold, tr_outer, te, tr_inner, val in rskf_splits(y):
        fold_key = f"r{seed}_f{fold}"
        if args.only_folds and fold_key not in args.only_folds: continue
        out_dir = cache_dir/fold_key; out_dir.mkdir(parents=True, exist_ok=True)
        # Save splits for audit
        np.savez(out_dir/"splits.npz",
                 outer_tr=tr_outer, te=te, tr_inner=tr_inner, val=val)
        for arch in archs:
            t0 = time.time()
            # Shallow embedding uses outer-train sentences; DL embedding excludes val
            shallow_corpus = sentences_unsup + [sentences_sup[i] for i in tr_outer]
            dl_corpus = sentences_unsup + [sentences_sup[i] for i in tr_inner]
            for kind, corpus, train_idx in [("shallow", shallow_corpus, tr_outer),
                                              ("dl", dl_corpus, tr_inner)]:
                model = train_embedding(arch, corpus)
                # Persist a thin npz with token vectors + the labeled-train-row indices
                if EMB_ARCHITECTURES[arch][1] == "wv":
                    keys = list(model.wv.key_to_index)
                    vecs = np.stack([model.wv[k] for k in keys])
                else:  # doc2vec — store inferred document vectors per outer-train row
                    keys, vecs = [], []
                    for i in train_idx:
                        keys.append(int(i))
                        vecs.append(model.infer_vector(sentences_sup[i]))
                    vecs = np.stack(vecs)
                np.savez(out_dir/f"{arch}_{kind}.npz",
                         keys=np.array(keys, dtype=object),
                         vectors=vecs,
                         train_indices=np.array(list(train_idx), dtype=np.int64))
            print(f"[01-emb] {fold_key} {arch} ({time.time()-t0:.0f}s)")
    print(f"[01-emb] done in {(time.time()-t_total)/60:.1f}min")


if __name__ == "__main__": main()
