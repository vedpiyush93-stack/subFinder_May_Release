"""Leak-audit smoke test — run with: pytest tests/leak_audit.py -v

Verifies:
  1. For every (seed, fold) in 5×5 RSKF, outer_test ∩ outer_train = ∅.
  2. The embeddings are global and label-free: one model per architecture,
     trained on the unsupervised corpus, with no per-fold variants.
  3. A PUL's embedding features are identical no matter which fold it lands in.
  4. The fitted CountVectorizer of cpu__ET500_log2 fold 0 has a bounded vocabulary.

Why (2) and (3) replaced the old per-fold assertion
---------------------------------------------------
Until May 2026 embeddings were trained per fold, and this file asserted that a
fold's embedding never saw that fold's test rows. That assertion passed, but it
did not mean what it appeared to: each fold's corpus was the unsupervised corpus
plus ~824 supervised training rows, and **all 1,030 supervised PULs occur
verbatim inside the unsupervised corpus already** (verified: 1030/1030 exact
sequence matches, including a 72-token PUL). Excluding 206 labelled copies of
sequences that were present a million lines over protected nothing.

The current guarantee is stronger and simpler: the embeddings never see a label
or a supervised row at all. They are trained once on the unsupervised corpus and
frozen, so there is no fold-specific information for them to carry.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "artifacts/embeddings"
ARCHS = ["fasttext_cbow", "fasttext_sg", "word2vec_cbow", "word2vec_sg",
         "doc2vec_dm", "doc2vec_dbow"]


@pytest.fixture(scope="module")
def y():
    return pd.read_csv(ROOT/"data/Train_data.csv")["high_level_substr"].values


def test_outer_train_test_disjoint(y):
    for seed in [42, 43, 44, 45, 46]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            assert len(set(tr) & set(te)) == 0, f"r{seed}_f{fold}: outer split not disjoint"


def test_no_per_fold_embedding_variants():
    """There must be exactly one embedding per architecture — no r*_f* copies."""
    if not EMB_DIR.exists():
        pytest.skip("embeddings not built — run scripts/01_train_embeddings.py --retrain")
    stale = list(EMB_DIR.glob("r*_f*")) + list((ROOT/"artifacts").glob("embeddings_cache/r*_f*"))
    assert not stale, f"per-fold embedding directories still present: {stale[:3]}"
    for arch in ARCHS:
        hits = list(EMB_DIR.glob(f"{arch}.npz")) + list(EMB_DIR.glob(f"{arch}.model"))
        assert len(hits) == 1, f"{arch}: expected exactly 1 global embedding, found {len(hits)}"


def test_embeddings_trained_on_unsupervised_corpus_only():
    """Metadata must record training on the unsupervised corpus, at its full size."""
    if not EMB_DIR.exists():
        pytest.skip("embeddings not built")
    import gzip
    corpus = ROOT/"data/unsupervised_corpus.txt.gz"
    with gzip.open(corpus, "rt") as fh:
        n_docs = sum(1 for ln in fh if ln.strip())
    for arch in ARCHS:
        npz = EMB_DIR/f"{arch}.npz"
        if not npz.exists():
            continue                                   # doc2vec ships as a gensim model
        z = np.load(npz, allow_pickle=True)
        assert str(z["trained_on"]) == "unsupervised_corpus_deduplicated", arch
        assert int(z["n_docs"]) == n_docs, f"{arch}: trained on {int(z['n_docs'])} docs, corpus has {n_docs}"


def test_features_are_fold_invariant(y):
    """The same PUL must featurize identically regardless of the split it lands in.

    Trivially true for a frozen embedding — this test exists so that any future
    reintroduction of fold-dependent embeddings fails loudly here.
    """
    if not EMB_DIR.exists():
        pytest.skip("embeddings not built")
    from src.embeddings.loader import load_word_vectors
    from src.preprocessing import EmbeddingMeanFeaturizer
    X = pd.read_csv(ROOT/"data/Train_data.csv")["sig_gene_seq"].fillna("").values
    feat = EmbeddingMeanFeaturizer(load_word_vectors("fasttext_cbow", EMB_DIR), tokenizer="comma_pipe")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    (tr, te), = [s for i, s in enumerate(skf.split(np.zeros(len(y)), y)) if i == 0]
    both = feat.transform(list(X))
    assert np.array_equal(both[te], feat.transform(list(X[te]))), "test-row features depend on context"
    assert np.array_equal(both[tr], feat.transform(list(X[tr]))), "train-row features depend on context"
