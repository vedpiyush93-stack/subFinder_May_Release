#!/usr/bin/env python3
"""Train 16 deep configs (4 architectures × 4 embeddings) across (seed, fold) splits.

The 4 deep architectures:
    LSTM         vanilla single-layer LSTM(128) + dropout(0.6)
    LSTMattn     LSTM(128) + Bahdanau-style soft attention
    JustAttn     non-recurrent attention block (bag-of-vectors → attn → dense)
    Trans        4-block transformer encoder (4 heads × head=128 × ff=512 × dropout=0.5)

The 4 embeddings (global; loaded from --emb-dir):
    fasttext_cbow, fasttext_sg, word2vec_cbow, word2vec_sg

Doc2Vec is absent by design. These are sequence models: they consume one vector
per token and attend/recur over the token axis. Doc2Vec's output is a single
vector for the whole PUL, so there is no sequence for an LSTM or a transformer
to run over — the four d2vDm__* configs were dropped in May 2026 rather than
fed Doc2Vec's word vectors, which DBOW never trains and which are not what
Doc2Vec is for. Doc2Vec is still benchmarked in the shallow configs, where a
document vector is exactly the right input.

Embeddings are global — trained once on the unsupervised corpus and frozen, so
the token tensor no longer depends on the split: it is built once for all 1,030
PULs and sliced per fold.

DL_BATCH = {LSTM/LSTMattn/JustAttn: 1024, Trans: 4096} (M4 Max throughput).

Usage:
    python scripts/03_train_deep.py --reuse                   # verify artifacts
    python scripts/03_train_deep.py --retrain                 # retrain all 16 × 25 = 400 fits (~5h)
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
from src.embeddings.loader import load_word_vectors
from src.deep import configure_device, device_summary
from src.splits import rskf_splits

DL_CONFIGS = {
    f"{emb_short}__{arch}": (emb_arch, arch)
    for emb_short, emb_arch in [("ftCbow","fasttext_cbow"), ("ftSg","fasttext_sg"),
                                  ("w2vCbow","word2vec_cbow"), ("w2vSg","word2vec_sg")]
    for arch in ["LSTM", "LSTMattn", "JustAttn", "Trans"]
}


_SEQ_CACHE: dict = {}


def _sequence_tensor(emb_dir, arch, sentences, max_seq_len: int = 30):
    """(n_rows, max_seq_len, 300) token-vector tensor for ALL rows, built once.

    Embeddings are global, so this tensor is identical for every fold — folds
    select rows from it by indexing. FastText resolves out-of-vocabulary tokens
    through its character fragments; Word2Vec leaves them as zeros.
    """
    key = (str(emb_dir), arch, max_seq_len)
    if key not in _SEQ_CACHE:
        t0 = time.time()
        wv = load_word_vectors(arch, emb_dir)
        vec_size = int(wv.vector_size)
        out = np.zeros((len(sentences), max_seq_len, vec_size), dtype=np.float32)
        for i, toks in enumerate(sentences):
            for j, t in enumerate(toks[:max_seq_len]):
                try:
                    out[i, j] = wv[t]
                except (KeyError, AttributeError):
                    pass                       # zero vector for W2V out-of-vocabulary
        _SEQ_CACHE[key] = out
        print(f"[03-deep] built {arch} tensor {out.shape} in {time.time()-t0:.1f}s", flush=True)
    return _SEQ_CACHE[key]


def main():
    try:
        import keras  # noqa: F401
    except ImportError:
        sys.exit("[03-deep] keras not installed (pip install keras tensorflow)")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse",   action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--only",    nargs="+", default=None)
    ap.add_argument("--only-folds", nargs="+", default=None, help="subset of fold keys like 'r42_f0' (default: all 25)")
    ap.add_argument("--emb-dir",   default=str(ROOT/"artifacts/embeddings"))
    ap.add_argument("--out-dir",   default=str(ROOT/"artifacts/predictions"))
    ap.add_argument("--max-seq-len", type=int, default=30)
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto",
                    help="auto = use the Metal GPU when the tensorflow-metal plugin is present")
    args = ap.parse_args()
    if not args.reuse and not args.retrain: ap.error("specify --reuse or --retrain")

    out_dir = Path(args.out_dir); emb_dir = Path(args.emb_dir)
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

    dev = configure_device(args.device)
    print(f"[03-deep] compute device: {device_summary(dev)}", flush=True)

    import keras
    for seed, fold, tr_outer, te, tr_inner, val in rskf_splits(y):
        fold_key = f"r{seed}_f{fold}"
        if args.only_folds and fold_key not in args.only_folds: continue
        for cfg in targets:
            emb_arch, dl_arch = DL_CONFIGS[cfg]
            out = out_dir/cfg/fold_key
            out.mkdir(parents=True, exist_ok=True)
            # Resume-friendly: skip trials that already have meta.json
            if (out/"meta.json").exists():
                print(f"[03-deep] {fold_key} {cfg}: SKIP (meta.json exists)", flush=True)
                continue
            t0 = time.time()
            try:
                # Global embedding -> one tensor for all rows, sliced per fold.
                # Xtr_outer = (824, max_seq_len, 300), Xte_outer = (206, ...)
                T = _sequence_tensor(emb_dir, emb_arch, sentences, max_seq_len=args.max_seq_len)
                Xtr_outer = T[tr_outer]; Xte_outer = T[te]
                # Keras categorical labels (aligned with tr_outer order)
                cls = sorted(set(y))
                onehot_tr = np.eye(len(cls))[[cls.index(c) for c in y[tr_outer]]].astype(np.float32)
                # Release the previous fit's graph. Without this, every build_dl()
                # leaks its model into the TF session: across 400 fits RSS reached
                # 46 GB and per-fit time drifted 13.7s -> 20.3s (+48%). Verified
                # bit-identical results with and without (build_dl reseeds anyway).
                keras.backend.clear_session()
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
                           "n_test": int(len(te)), "n_train": int(len(tr_outer)),
                           "device": dev["device"], "backend": dev["backend"],
                           "tensorflow": dev["tensorflow"]},
                          open(out/"meta.json","w"), indent=2)
                json.dump(hist, open(out/"history.json","w"))
                print(f"[03-deep] {fold_key} {cfg}: acc={acc:.4f} ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"[03-deep] FAIL {fold_key} {cfg}: {e}", file=sys.stderr)


if __name__ == "__main__": main()
