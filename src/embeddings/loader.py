"""Load the six global embeddings from ``artifacts/embeddings/``.

Layout (one model per architecture — no per-fold copies)::

    artifacts/embeddings/
      fasttext_cbow.npz     compacted, collision-free (words + fragments)
      fasttext_sg.npz
      word2vec_cbow.npz     token -> vector table
      word2vec_sg.npz
      doc2vec_dm.model      gensim model, training doc-vectors stripped
      doc2vec_dbow.model

Every model is trained once on the unsupervised corpus and frozen. Supervised
sequences are only ever *fed through* these models, never trained into them,
so there is no per-fold variant to pick and nothing to keep out of a test fold.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = ROOT / "artifacts/embeddings"


def embedding_path(arch: str, emb_dir=None) -> Path:
    emb_dir = Path(emb_dir or DEFAULT_DIR)
    return emb_dir / (f"{arch}.model" if arch.startswith("doc2vec_") else f"{arch}.npz")


def load_word_vectors(arch: str, emb_dir=None):
    """Return a ``wv``-like object (``wv[token]``, ``token in wv``) for a
    FastText or Word2Vec architecture."""
    path = embedding_path(arch, emb_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build it with `python scripts/01_train_embeddings.py --retrain`")
    if arch.startswith("fasttext_"):
        from .compact import load_compact
        return load_compact(path)
    if arch.startswith("word2vec_"):
        from .compact import load_word_vectors as _load
        return _load(path)
    raise ValueError(f"{arch!r} has no word-vector form; use load_doc2vec() for Doc2Vec")


def load_doc2vec(arch: str, emb_dir=None):
    """Return the gensim ``Doc2Vec`` model. Doc2Vec configs featurize with
    ``infer_vector`` (a document vector), not with word vectors — see
    ``src.preprocessing.featurizers.Doc2VecInferFeaturizer``."""
    from gensim.models.doc2vec import Doc2Vec
    path = embedding_path(arch, emb_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build it with `python scripts/01_train_embeddings.py --retrain`")
    return Doc2Vec.load(str(path))


def load_embedding(arch: str, emb_dir=None):
    """Dispatch on architecture: Doc2Vec → model, others → word vectors."""
    if arch.startswith("doc2vec_"):
        return load_doc2vec(arch, emb_dir)
    return load_word_vectors(arch, emb_dir)
