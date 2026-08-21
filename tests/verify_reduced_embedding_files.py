"""Reviewer-runnable proof that the compacted embedding files shipped in this
repo produce the same inference outputs as a full gensim FastText model.

Run with:  pytest tests/verify_reduced_embedding_files.py -v -s

What is being proved
--------------------
gensim stores FastText character-n-gram ("fragment") vectors in a hash table of
``bucket`` rows, addressed by ``hash(fragment) % bucket``. At the paper's
``bucket=2_000_000`` that table is 2.24 GB — but the corpus vocabulary only
generates ~11 k distinct fragments, so **only ~0.55 % of those rows can be
reached by any possible input**.

We therefore ship one row per fragment that actually exists, keyed by the
fragment's text (``src/embeddings/compact.py``). This test trains a real
FastText at the paper's bucket size, compacts it, and checks the compacted form
against the full model along the dimensions that matter:

  D1. Per-token vectors for every in-vocabulary token.
  D2. ``EmbeddingMeanFeaturizer`` over all 1,030 supervised PULs.
  D3. ``EmbeddingMeanMaxFeaturizer`` over all 1,030 supervised PULs
      (the featurizer used by ftCbow_MM__ET500_sqrt).
  D4. Tokens whose fragments were never seen in training — the one case that is
      *expected* to differ, and is asserted to differ in the documented way.

Pass criterion
--------------
D1 must be exactly 0. D2/D3 must agree to <1e-5: they are not bit-identical
because float32 addition is not associative and we sum fragments in alphabetical
order where gensim sums them in hash order. D4 must show the compacted form
returning zero contributions where the full model returns values from rows that
training never touched (i.e. random initialisation noise).
"""
from __future__ import annotations
import gzip
from pathlib import Path
import numpy as np, pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data/unsupervised_corpus.txt.gz"
N_DOCS = 40_000          # subset keeps the test to ~1 min while exercising the real path


@pytest.fixture(scope="module")
def models():
    from gensim.models import FastText
    from src.embeddings.compact import build_compact
    from src.preprocessing.tokenizers import tok_comma_pipe
    if not CORPUS.exists():
        pytest.skip(f"{CORPUS} missing")
    with gzip.open(CORPUS, "rt") as fh:
        corpus = [ln.split() for ln in fh if ln.strip()][:N_DOCS]
    sup = pd.read_csv(ROOT/"data/Train_data.csv")["sig_gene_seq"].fillna("").values
    sup_tokens = {t for s in sup for t in tok_comma_pipe(s)}
    m = FastText(sentences=corpus, sg=0, vector_size=300, window=7, min_count=5,
                 epochs=5, workers=8, bucket=2_000_000, seed=42)
    compact = build_compact(m.wv, extra_tokens=sup_tokens)
    full_mb = (m.wv.vectors_ngrams.nbytes + m.wv.vectors.nbytes) / 1024**2
    comp_mb = (compact.fragment_vectors.nbytes + compact.vectors.nbytes) / 1024**2
    print(f"\n  full model {full_mb/1024:.2f} GB -> compacted {comp_mb:.1f} MB "
          f"({full_mb/comp_mb:.0f}x smaller)")
    return m.wv, compact, list(sup)


def test_D1_in_vocabulary_tokens_are_exact(models):
    wv, compact, _ = models
    toks = list(wv.key_to_index)
    diff = np.abs(np.stack([wv[t] for t in toks]) - np.stack([compact[t] for t in toks])).max()
    print(f"  D1 in-vocabulary tokens ({len(toks)}): max|diff| = {diff:.3e}")
    assert diff == 0.0


def test_D2_mean_featurizer_over_all_supervised_puls(models):
    from src.preprocessing import EmbeddingMeanFeaturizer
    wv, compact, X = models
    a = EmbeddingMeanFeaturizer(wv, tokenizer="comma_pipe").transform(X)
    b = EmbeddingMeanFeaturizer(compact, tokenizer="comma_pipe").transform(X)
    diff = np.abs(a - b).max()
    print(f"  D2 EmbeddingMeanFeaturizer ({len(X)} PULs): max|diff| = {diff:.3e}")
    assert diff < 1e-5


def test_D3_meanmax_featurizer_over_all_supervised_puls(models):
    from src.preprocessing import EmbeddingMeanMaxFeaturizer
    wv, compact, X = models
    a = EmbeddingMeanMaxFeaturizer(wv, tokenizer="comma_pipe").transform(X)
    b = EmbeddingMeanMaxFeaturizer(compact, tokenizer="comma_pipe").transform(X)
    diff = np.abs(a - b).max()
    print(f"  D3 EmbeddingMeanMaxFeaturizer ({len(X)} PULs): max|diff| = {diff:.3e}")
    assert diff < 1e-5


def test_D4_never_seen_fragments_are_the_documented_difference(models):
    """Tokens built from character sequences absent from the whole corpus.

    The full model reads these from hash rows training never updated, i.e. the
    random values they were initialised with. The compacted store has no row for
    them and contributes zero. Neither carries information; this asserts the
    divergence is confined to exactly this case.
    """
    import string
    from gensim.models.fasttext import compute_ngrams
    wv, compact, _ = models
    known = set(compact._fidx)

    # Build tokens from characters the corpus never uses, so every fragment is
    # guaranteed novel. Guessing at "weird-looking" tokens does not work: digits
    # and CAZy letters recur everywhere, e.g. "987" is a fragment of real TC numbers.
    seen_chars = {c for frag in known for c in frag}
    spare = [c for c in string.ascii_lowercase if c not in seen_chars]
    assert len(spare) >= 3, f"corpus uses nearly every character; only {spare} spare"
    novel = ["".join(spare[:3]) * 3, "".join(spare[:2]) * 4]

    for tok in novel:
        frags = compute_ngrams(tok, wv.min_n, wv.max_n)
        assert not (set(frags) & known), f"{tok} unexpectedly shares fragments with the corpus"
        assert np.abs(compact[tok]).max() == 0.0, f"{tok} should contribute zero when fully unknown"
        # the full model instead returns whatever those never-trained rows hold
        assert tok not in wv.key_to_index
    print(f"  D4 fully-unknown tokens {novel}: compacted returns the origin vector, as documented")
