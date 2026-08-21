#!/usr/bin/env python3
"""Train the six global embeddings — once, on the unsupervised corpus.

    python scripts/01_train_embeddings.py --reuse-cache   # verify what's shipped
    python scripts/01_train_embeddings.py --retrain       # rebuild all six (~25 min)
    python scripts/01_train_embeddings.py --retrain --only-archs fasttext_cbow

What changed (May 2026)
-----------------------
This script used to train one embedding per (seed, fold, architecture, regime)
— 300 gensim models, 186 GB, because each FastText carries a 2.24 GB n-gram
hash table. The per-fold split existed to keep test-fold tokens out of the
embedding space.

It bought nothing. Each fold's corpus was the unsupervised corpus plus that
fold's ~824 supervised training rows — 0.08 % of the corpus — and all 1,030
supervised PULs already appear *verbatim* in the unsupervised corpus, so
holding 206 of them out changed nothing that was not already there. Adding the
supervised rows raised the vocabulary by 44 tokens, 41 of which are 5-level TC
numbers that ``tok_cpu_v2`` truncates away anyway.

So: train once on the unsupervised corpus, freeze, and only ever *feed*
supervised sequences through. The embeddings never see a label, which is a
stronger and simpler guarantee than the per-fold scheme provided.

Outputs (``artifacts/embeddings/``, ~38 MB total)
-------------------------------------------------
  fasttext_cbow.npz   compacted collision-free store  (~14 MB)
  fasttext_sg.npz     ''
  word2vec_cbow.npz   token -> vector table            (~1.7 MB)
  word2vec_sg.npz     ''
  doc2vec_dm.model    gensim model, training doc-vectors stripped (~3.3 MB)
  doc2vec_dbow.model  ''

FastText is trained at the paper's ``bucket=2_000_000`` and then compacted:
only ~11 k of those 2 M rows are reachable by any token the corpus can produce,
so we store one row per real fragment, keyed by the fragment's text. Verified
equivalent at build time and in tests/verify_reduced_embedding_files.py.

Doc2Vec is featurized by *document* vectors (``infer_vector``), not word
vectors — see ``src.preprocessing.featurizers.Doc2VecInferFeaturizer``.
"""
from __future__ import annotations
import argparse, gzip, sys, time
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocessing.tokenizers import tok_comma_pipe
from src.embeddings import (EMB_ARCHITECTURES, train_embedding, EMB_HYPERPARAMS,
                            FASTTEXT_BUCKET, strip_training_docvecs,
                            build_compact, save_compact, save_word_vectors)

DEFAULT_CORPUS = ROOT / "data/unsupervised_corpus.txt.gz"
DEFAULT_OUT    = ROOT / "artifacts/embeddings"


def read_corpus(path: Path) -> list[list[str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return [ln.split() for ln in fh if ln.strip()]


def supervised_tokens() -> set[str]:
    df = pd.read_csv(ROOT / "data/Train_data.csv")
    return {t for s in df["sig_gene_seq"].fillna("").values for t in tok_comma_pipe(s)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse-cache", action="store_true", help="Verify the shipped embeddings, train nothing.")
    ap.add_argument("--retrain", action="store_true", help="Retrain all six from scratch.")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="Unsupervised corpus (one PUL per line).")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--only-archs", nargs="+", default=None)
    args = ap.parse_args()
    if not args.reuse_cache and not args.retrain:
        ap.error("specify either --reuse-cache or --retrain")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    archs = args.only_archs or list(EMB_ARCHITECTURES)

    if args.reuse_cache:
        from src.embeddings.loader import embedding_path
        missing = [a for a in archs if not embedding_path(a, out_dir).exists()]
        for a in archs:
            p = embedding_path(a, out_dir)
            mark = "ok " if p.exists() else "MISSING"
            size = f"{p.stat().st_size/1024**2:6.1f} MB" if p.exists() else ""
            print(f"[01-emb] {mark} {a:15s} {size}")
        if missing:
            print(f"[01-emb] {len(missing)} missing — rerun with --retrain", file=sys.stderr)
            sys.exit(1)
        return

    corpus = read_corpus(Path(args.corpus))
    sup_tokens = supervised_tokens()
    print(f"[01-emb] corpus: {len(corpus):,} unsupervised PULs from {args.corpus}")
    print(f"[01-emb] supervised tokens (for FastText fragment coverage): {len(sup_tokens):,}")
    print(f"[01-emb] hyperparameters: {EMB_HYPERPARAMS} | fasttext bucket={FASTTEXT_BUCKET:,}\n")

    t_all = time.time()
    for arch in archs:
        kind = EMB_ARCHITECTURES[arch][1]
        t0 = time.time()
        model = train_embedding(arch, corpus)
        t_train = time.time() - t0
        meta = dict(arch=arch, n_docs=len(corpus), trained_on="unsupervised_corpus_deduplicated",
                    **{k: v for k, v in EMB_HYPERPARAMS.items()})

        if arch.startswith("fasttext_"):
            wv = model.wv
            raw_mb = (wv.vectors_ngrams.nbytes + wv.vectors.nbytes) / 1024**2
            compact = build_compact(wv, extra_tokens=sup_tokens)
            # Build-time equivalence assertion over every token we will ever query.
            probe = list(wv.key_to_index) + sorted(sup_tokens)
            worst = max(float(np.abs(wv[t] - compact[t]).max()) for t in probe)
            assert worst < 1e-5, f"{arch}: compaction diverged by {worst}"
            out = out_dir / f"{arch}.npz"
            save_compact(out, compact, bucket=FASTTEXT_BUCKET, **meta)
            print(f"[01-emb] {arch:15s} trained {t_train/60:4.1f} min | "
                  f"{raw_mb/1024:.2f} GB -> {out.stat().st_size/1024**2:.1f} MB | "
                  f"{len(compact._fidx):,} fragments, {len(compact):,} words | "
                  f"max|diff| vs full model {worst:.2e}")

        elif arch.startswith("word2vec_"):
            out = out_dir / f"{arch}.npz"
            save_word_vectors(out, model.wv, **meta)
            print(f"[01-emb] {arch:15s} trained {t_train/60:4.1f} min | "
                  f"{out.stat().st_size/1024**2:.1f} MB | {len(model.wv):,} words")

        else:  # doc2vec — featurized by inferred document vectors
            probe_doc = sorted(sup_tokens)[:12]
            model.random = np.random.RandomState(42); before = model.infer_vector(probe_doc)
            dv_mb = model.dv.vectors.nbytes / 1024**2
            strip_training_docvecs(model)
            model.random = np.random.RandomState(42); after = model.infer_vector(probe_doc)
            drift = float(np.abs(before - after).max())
            assert drift == 0.0, f"{arch}: stripping training doc-vectors changed inference by {drift}"
            out = out_dir / f"{arch}.model"
            model.save(str(out))
            total = sum(p.stat().st_size for p in out_dir.glob(f"{arch}.model*")) / 1024**2
            print(f"[01-emb] {arch:15s} trained {t_train/60:4.1f} min | "
                  f"dropped {dv_mb:.0f} MB of training doc-vectors -> {total:.1f} MB | "
                  f"inference drift {drift:.1e}")

    print(f"\n[01-emb] done in {(time.time()-t_all)/60:.1f} min -> {out_dir}")


if __name__ == "__main__": main()
