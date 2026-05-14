"""Reviewer-runnable proof that the reduced embedding files shipped in this
repo produce BIT-IDENTICAL inference outputs to the full source-of-truth
gensim model directories.

What this verifies
------------------
For each (architecture, regime) under ``artifacts/embeddings_cache/r42_f0/``,
compares the REDUCED files we ship against a FULL-source gensim model dir:

  * Word2Vec  →  REDUCED = .npz only          (no n-gram OOV)
  * Doc2Vec   →  REDUCED = .npz only          (no n-gram OOV)
  * FastText  →  REDUCED = .npy.xz + .model   (n-gram OOV via wrapper auto-decompress)

Comparison dimensions:

  D1. Per-token vector equality across all 1430+ training-corpus tokens.
  D2. ``EmbeddingMeanFeaturizer`` output over ALL 1030 supervised PULs.
  D3. ``EmbeddingMeanFeaturizer`` output over 100 synthetic heavy-OOV PULs
       (every token in these is guaranteed unknown to the train vocab,
        stress-testing n-gram OOV fallback for FastText and zero-vector
        substitution for W2V/D2V).

Pass criterion
--------------
Every D-test must report ``max-abs-diff = 0.00e+00`` and
``feature-matrix bit-identical = True``. Any non-zero diff is a regression
and should be raised in review.

Run with
--------
    pytest -q tests/verify_reduced_embedding_files.py

or interactively:

    python3 tests/verify_reduced_embedding_files.py

Requires the **source-of-truth** model dir at ``/Users/ved/subFinder/...``
for the comparison side. On a reviewer's machine, point ``FULL_SRC`` at
their local regenerated cache (see README §"Regenerating the embedding cache").
"""
from __future__ import annotations
import os, random, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.featurizers import EmbeddingMeanFeaturizer
from src.embeddings.loader import load_fasttext
from gensim.models import FastText, Word2Vec, Doc2Vec

# point at your local regenerated fold_cache_v2 if you don't have the author's
FULL_SRC = os.environ.get("FULL_SRC",
                          "/Users/ved/subFinder/reproducibility/fold_cache_v2")
REL = str(ROOT / "artifacts/embeddings_cache")


def tok_cpu(s): return [t for t in re.split(r"[,|_]", str(s)) if t]


class _NpzKV:
    """KeyedVectors-shim wrapping a shipped .npz {vocab, vectors}."""
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self._table = {str(t): d["vectors"][i] for i, t in enumerate(d["vocab"])}
        self.vector_size = int(d["vectors"].shape[1])
    def __getitem__(self, key): return self._table[key]


@pytest.fixture(scope="module")
def corpus():
    df = pd.read_csv(ROOT / "data/Train_data.csv")
    X = df["sig_gene_seq"].astype(str).values.tolist()
    return X


@pytest.fixture(scope="module")
def stress_oov():
    random.seed(0)
    def fake(): return "OOV_" + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=8))
    return [",".join(fake() for _ in range(random.randint(3, 10))) for _ in range(100)]


def _assert_no_diff(label, full_kv, reduced_kv, corpus, stress):
    feat_full = EmbeddingMeanFeaturizer(full_kv,    tokenizer="cpu")
    feat_red  = EmbeddingMeanFeaturizer(reduced_kv, tokenizer="cpu")

    # D1: per-token equality (only for tokens that the .npz has)
    all_tokens = {t for s in corpus for t in tok_cpu(s)}
    miss = 0; max_diff = 0.0
    for t_ in all_tokens:
        try: a = full_kv[t_]
        except (KeyError, AttributeError): continue
        try: b = reduced_kv[t_]
        except (KeyError, AttributeError):
            miss += 1; continue
        max_diff = max(max_diff, float(np.abs(a-b).max()))
    print(f"  D1 {label}: max-abs-diff={max_diff:.2e}, missed={miss}")
    assert max_diff == 0.0 and miss == 0

    # D2: featurizer over the full training corpus
    a = feat_full.transform(corpus); b = feat_red.transform(corpus)
    print(f"  D2 {label}: featurizer over {len(corpus)} PULs → bit-identical={np.array_equal(a,b)}")
    assert np.array_equal(a, b)

    # D3: stress test with synthetic heavy-OOV
    a = feat_full.transform(stress); b = feat_red.transform(stress)
    print(f"  D3 {label}: featurizer over {len(stress)} OOV-PULs → bit-identical={np.array_equal(a,b)}")
    assert np.array_equal(a, b)


@pytest.mark.skipif(not Path(FULL_SRC).exists(),
                     reason="full source-of-truth cache not available locally")
def test_word2vec_cbow_shallow_reduced_matches_full(corpus, stress_oov):
    full = Word2Vec.load(f"{FULL_SRC}/r42_f0/word2vec_cbow_shallow_model/word2vec_cbow.model").wv
    reduced = _NpzKV(f"{REL}/r42_f0/word2vec_cbow_shallow.npz")
    _assert_no_diff("W2V_cbow_shallow", full, reduced, corpus, stress_oov)


@pytest.mark.skipif(not Path(FULL_SRC).exists(),
                     reason="full source-of-truth cache not available locally")
def test_doc2vec_dm_shallow_reduced_matches_full(corpus, stress_oov):
    full = Doc2Vec.load(f"{FULL_SRC}/r42_f0/doc2vec_dm_shallow_model/doc2vec_dm.model").wv
    reduced = _NpzKV(f"{REL}/r42_f0/doc2vec_dm_shallow.npz")
    _assert_no_diff("D2V_dm_shallow", full, reduced, corpus, stress_oov)


@pytest.mark.skipif(not Path(FULL_SRC).exists(),
                     reason="full source-of-truth cache not available locally")
def test_fasttext_cbow_shallow_xz_matches_full(corpus, stress_oov):
    full = FastText.load(f"{FULL_SRC}/r42_f0/fasttext_cbow_shallow_model/fasttext_cbow.model").wv
    # our wrapper auto-decompresses the .npy.xz from the release repo
    reduced_model = load_fasttext(f"{REL}/r42_f0/fasttext_cbow_shallow_model/fasttext_cbow.model")
    _assert_no_diff("FT_cbow_shallow (xz)", full, reduced_model.wv, corpus, stress_oov)


if __name__ == "__main__":
    # interactive mode
    df = pd.read_csv(ROOT / "data/Train_data.csv")
    X = df["sig_gene_seq"].astype(str).values.tolist()
    random.seed(0)
    def fake(): return "OOV_" + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=8))
    stress = [",".join(fake() for _ in range(random.randint(3, 10))) for _ in range(100)]

    for name, gensim_cls, regime in [
        ("word2vec_cbow_shallow", Word2Vec, "word2vec_cbow"),
        ("doc2vec_dm_shallow",    Doc2Vec,  "doc2vec_dm"),
    ]:
        full = gensim_cls.load(f"{FULL_SRC}/r42_f0/{name}_model/{regime}.model").wv
        reduced = _NpzKV(f"{REL}/r42_f0/{name}.npz")
        print(f"\n=== {name} (REDUCED=npz only) ===")
        _assert_no_diff(name, full, reduced, X, stress)

    print(f"\n=== fasttext_cbow_shallow (REDUCED=xz decompressed) ===")
    full = FastText.load(f"{FULL_SRC}/r42_f0/fasttext_cbow_shallow_model/fasttext_cbow.model").wv
    reduced_model = load_fasttext(f"{REL}/r42_f0/fasttext_cbow_shallow_model/fasttext_cbow.model")
    _assert_no_diff("fasttext_cbow_shallow", full, reduced_model.wv, X, stress)

    print("\n✅ All checks passed — reduced files in the repo are bit-identical to the full source.")
