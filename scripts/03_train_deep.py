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


def _load_emb_seq(cache_dir, fold_key, arch, sentences, sentences_indices, max_seq_len: int = 30):
    """Return (n, max_seq_len, 300) tensor of stacked token vectors per PUL.

    For FastText/Word2Vec we look up per-token vectors; for Doc2Vec we'd need
    the document-vector model (not implemented in the lite cache); falls back
    to zero-vector if cache missing.
    """
    npz = np.load(cache_dir/fold_key/f"{arch}_dl.npz", allow_pickle=True)
    keys = npz["keys"]; vecs = npz["vectors"]
    vec_size = vecs.shape[1] if vecs.shape[0] else 300
    idx = {str(k): i for i, k in enumerate(keys)}
    out = np.zeros((len(sentences), max_seq_len, vec_size), dtype=np.float32)
    for i, s in enumerate(sentences):
        toks = sentences[i][:max_seq_len]
        for j, t in enumerate(toks):
            if t in idx: out[i, j] = vecs[idx[t]]
    return out


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
                Xtr_outer = _load_emb_seq(cache_dir, fold_key, emb_arch, sentences,
                                          tr_outer, max_seq_len=args.max_seq_len)
                Xte_outer = _load_emb_seq(cache_dir, fold_key, emb_arch, sentences,
                                          te, max_seq_len=args.max_seq_len)
                # Keras categorical labels
                cls = sorted(set(y))
                onehot_tr = np.eye(len(cls))[[cls.index(c) for c in y[tr_outer]]].astype(np.float32)
                model = build_dl(dl_arch, (args.max_seq_len, Xtr_outer.shape[-1]))
                # Inner train/val split for EarlyStopping
                model, hist = train_dl(model, Xtr_outer[tr_inner.searchsorted(tr_inner)],
                                       onehot_tr[tr_inner.searchsorted(tr_inner)],
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
