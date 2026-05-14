#!/usr/bin/env python3
"""Train 20 deep configs (4 architectures × 5 embeddings) across (seed, fold) splits.

The 4 deep architectures:
    LSTM         vanilla single-layer LSTM(128) + dropout(0.6)
    LSTMattn     LSTM(128) + Bahdanau-style soft attention
    JustAttn     non-recurrent attention block (bag-of-vectors → attn → dense)
    Trans        4-block transformer encoder (4 heads × head=128 × ff=512 × dropout=0.5)

The 5 embeddings (per fold; loaded from --cache-dir):
    fasttext_cbow, fasttext_sg, word2vec_cbow, word2vec_sg, doc2vec_dm

DL_BATCH = {LSTM/LSTMattn/JustAttn: 1024, Trans: 4096} (M4 Max throughput).

Usage:
    python scripts/03_train_deep.py --reuse                   # verify artifacts
    python scripts/03_train_deep.py --retrain                 # retrain all 20 × 25 = 500 fits (~6h)
    python scripts/03_train_deep.py --retrain --only ftSg__LSTMattn
"""
from __future__ import annotations
import argparse, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocessing.tokenizers import tok_comma_pipe
from src.deep import build_dl, train_dl
from src.splits import rskf_splits

DL_CONFIGS = {
    f"{emb_short}__{arch}": (emb_arch, arch)
    for emb_short, emb_arch in [("ftCbow","fasttext_cbow"), ("ftSg","fasttext_sg"),
                                  ("w2vCbow","word2vec_cbow"), ("w2vSg","word2vec_sg"),
                                  ("d2vDm","doc2vec_dm")]
    for arch in ["LSTM", "LSTMattn", "JustAttn", "Trans"]
}


def _build_wv_for_arch(cache_dir, fold_key, arch):
    """Return a KeyedVectors-like object for ``arch`` at ``fold_key``.

    For FastText configs: loads the FULL gensim model (via load_fasttext, which
    auto-decompresses the .npy.xz sidecar from LFS) so ``wv[t]`` resolves OOV
    tokens via character n-gram fallback. Falls back to ``FT_FULL_DIR`` env
    var or ``/Users/ved/subFinder/reproducibility/fold_cache_v2/`` if the
    release-repo .xz isn't local yet.

    For W2V configs: uses the npz-only shim (Word2Vec has no n-gram OOV, so
    npz IS bit-identical to the full model for inference — verified empirically).

    For Doc2Vec: same npz-only path.
    """
    if arch.startswith("fasttext_"):
        import os
        arch_stem = arch.replace("fasttext_", "")
        sub = f"{arch}_dl_model"
        model_fn = f"fasttext_{arch_stem}.model"
        # Priority 1: FT_FULL_DIR (already-uncompressed, dev convenience)
        ft_full = os.environ.get("FT_FULL_DIR")
        if ft_full:
            src_path = Path(ft_full)/fold_key/sub/model_fn
            if src_path.exists():
                from gensim.models import FastText
                return FastText.load(str(src_path)).wv
        # Priority 2: release-repo path with auto-decompress
        from src.embeddings.loader import load_fasttext
        rel_path = cache_dir/fold_key/sub/model_fn
        if rel_path.exists():
            return load_fasttext(rel_path).wv
        raise FileNotFoundError(
            f"FastText DL model not at {rel_path}"
            + (f" or {Path(ft_full)/fold_key/sub/model_fn}" if ft_full else "")
            + ". Either pull from LFS or set FT_FULL_DIR.")
    # Non-FastText: npz lookup is functionally identical to full model
    npz = np.load(cache_dir/fold_key/f"{arch}_dl.npz", allow_pickle=True)
    class _NpzWV:
        def __init__(self, keys, vecs):
            self.vector_size = vecs.shape[1] if vecs.shape[0] else 300
            self._idx = {str(k): i for i, k in enumerate(keys)}
            self.vectors = vecs
        def __getitem__(self, t):
            i = self._idx.get(t)
            if i is None: raise KeyError(t)
            return self.vectors[i]
        def __contains__(self, t): return t in self._idx
    return _NpzWV(npz["keys"], npz["vectors"])


def _load_emb_seq(cache_dir, fold_key, arch, sentences, sentences_indices,
                   max_seq_len: int = 30, wv=None):
    """Return (len(indices), max_seq_len, vec_size) tensor of token vectors.

    ONLY the rows at ``sentences_indices`` are materialized, in that order
    (so downstream indexing matches the indices array). For FastText configs,
    OOV tokens get n-gram-resolved vectors (not zeros); for W2V/D2V, OOV
    tokens contribute a zero vector.

    Pass ``wv`` to reuse a previously-loaded model across calls in the same
    fold (avoids re-loading the gensim model twice for outer vs test).
    """
    if wv is None:
        wv = _build_wv_for_arch(cache_dir, fold_key, arch)
    vec_size = int(wv.vector_size)
    n = len(sentences_indices)
    out = np.zeros((n, max_seq_len, vec_size), dtype=np.float32)
    for i, idx in enumerate(sentences_indices):
        toks = sentences[idx][:max_seq_len]
        for j, t in enumerate(toks):
            try:
                out[i, j] = wv[t]              # n-gram fallback for FastText OOV
            except (KeyError, AttributeError):
                pass                            # zero for W2V/D2V OOV
    return out, wv


def main():
    try:
        import keras  # noqa: F401
    except ImportError:
        sys.exit("[03-deep] keras not installed (pip install keras tensorflow)")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse",   action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--only",    nargs="+", default=None)
    ap.add_argument("--cache-dir", default=str(ROOT/"artifacts/embeddings_cache"))
    ap.add_argument("--out-dir",   default=str(ROOT/"artifacts/predictions"))
    ap.add_argument("--max-seq-len", type=int, default=30)
    args = ap.parse_args()
    if not args.reuse and not args.retrain: ap.error("specify --reuse or --retrain")

    out_dir = Path(args.out_dir); cache_dir = Path(args.cache_dir)
    if args.reuse:
        ok = total = 0
        for cfg in DL_CONFIGS:
            for seed in [42,43,44,45,46]:
                for fold in range(5):
                    total += 1
                    if (out_dir/cfg/f"r{seed}_f{fold}"/"meta.json").exists(): ok += 1
        print(f"[03-deep] reuse check: {ok}/{total} deep trials present in {out_dir}")
        return

    df = pd.read_csv(ROOT/"data/Train_data.csv")
    X = df["sig_gene_seq"].fillna("").values; y = df["high_level_substr"].values
    substrates = sorted(set(y))
    sentences = [tok_comma_pipe(s) for s in X]
    targets = args.only or list(DL_CONFIGS.keys())

    import keras
    for seed, fold, tr_outer, te, tr_inner, val in rskf_splits(y):
        fold_key = f"r{seed}_f{fold}"
        for cfg in targets:
            emb_arch, dl_arch = DL_CONFIGS[cfg]
            out = out_dir/cfg/fold_key
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                # Materialize features ONLY for the rows we'll touch in this fold.
                # Shape: Xtr_outer = (len(tr_outer), max_seq_len, vec_size) ≈ (824, 30, 300)
                #        Xte_outer = (len(te), ...) ≈ (206, ...)
                Xtr_outer, wv = _load_emb_seq(cache_dir, fold_key, emb_arch, sentences,
                                                tr_outer, max_seq_len=args.max_seq_len)
                Xte_outer, _ = _load_emb_seq(cache_dir, fold_key, emb_arch, sentences,
                                                te, max_seq_len=args.max_seq_len, wv=wv)
                # Keras categorical labels (aligned with tr_outer order)
                cls = sorted(set(y))
                onehot_tr = np.eye(len(cls))[[cls.index(c) for c in y[tr_outer]]].astype(np.float32)
                model = build_dl(dl_arch, (args.max_seq_len, Xtr_outer.shape[-1]))
                # Inner train/val split for EarlyStopping. tr_inner ⊆ tr_outer are GLOBAL
                # indices; we need the POSITIONS within tr_outer that they map to.
                inner_pos = tr_outer.searchsorted(tr_inner)
                model, hist = train_dl(model, Xtr_outer[inner_pos],
                                       onehot_tr[inner_pos],
                                       dl_arch, max_epochs=2000, verbose=0)
                P_te = model.predict(Xte_outer, verbose=0)
                P_tr = model.predict(Xtr_outer, verbose=0)
                model.save(out/"classifier.keras")
                np.savez(out/"probs_test.npz",  probs=P_te.astype(np.float32),
                         classes=np.array(cls, dtype=object), idx=te)
                np.savez(out/"probs_train.npz", probs=P_tr.astype(np.float32),
                         classes=np.array(cls, dtype=object), idx=tr_outer)
                acc = (np.array(cls)[P_te.argmax(1)] == y[te]).mean()
                json.dump({"shorthand": cfg, "seed": seed, "fold": fold,
                           "test_acc": float(acc), "wall_sec": time.time()-t0,
                           "n_test": int(len(te)), "n_train": int(len(tr_outer))},
                          open(out/"meta.json","w"), indent=2)
                json.dump(hist, open(out/"history.json","w"))
                print(f"[03-deep] {fold_key} {cfg}: acc={acc:.4f} ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"[03-deep] FAIL {fold_key} {cfg}: {e}", file=sys.stderr)


if __name__ == "__main__": main()
