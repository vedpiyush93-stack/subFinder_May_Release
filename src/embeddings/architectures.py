"""Six gensim embedding architectures, paper-verbatim hyperparameters.

Each entry in ``EMB_ARCHITECTURES`` is ``(constructor_fn, kind)``:
    constructor_fn(sentences) -> trained model
    kind                       -> "wv"  (FastText/Word2Vec — featurized by word vectors)
                                  "doc" (Doc2Vec — featurized by inferred document vectors)

The 6 architectures correspond to the paper's grid:
    {FastText, Word2Vec, Doc2Vec} x {cbow|dm, skip-gram|dbow}

All hyperparameters match the paper exactly (300-d, window=7, min_count=5,
60 epochs). ``bucket`` is stated explicitly rather than left to gensim's
default so the value is visible: it is the paper's 2,000,000 and it governs
*training* only — the trained FastText tables are compacted to collision-free
storage afterwards (see ``src.embeddings.compact``), which is a storage change,
not a retraining.

Since May 2026 these are trained ONCE on the unsupervised corpus and frozen;
supervised sequences are only fed through them. There are no per-fold variants.
"""
from __future__ import annotations
from gensim.models import FastText, Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument


EMB_HYPERPARAMS = dict(
    vector_size=300,
    window=7,
    min_count=5,
    epochs=60,
    workers=14,
    seed=42,
)

# Paper's FastText n-gram bucket count. Training-time only.
FASTTEXT_BUCKET = 2_000_000


def _fasttext(sentences, sg: int):
    return FastText(sentences=sentences, sg=sg, bucket=FASTTEXT_BUCKET, **EMB_HYPERPARAMS)

def _word2vec(sentences, sg: int):
    return Word2Vec(sentences=sentences, sg=sg, **EMB_HYPERPARAMS)

def _doc2vec(sentences, dm: int):
    tagged = [TaggedDocument(s, [i]) for i, s in enumerate(sentences)]
    return Doc2Vec(documents=tagged, dm=dm, **EMB_HYPERPARAMS)


EMB_ARCHITECTURES = {
    "fasttext_cbow":  (lambda s: _fasttext(s, sg=0), "wv"),
    "fasttext_sg":    (lambda s: _fasttext(s, sg=1), "wv"),
    "word2vec_cbow":  (lambda s: _word2vec(s, sg=0), "wv"),
    "word2vec_sg":    (lambda s: _word2vec(s, sg=1), "wv"),
    "doc2vec_dm":     (lambda s: _doc2vec(s, dm=1),  "doc"),
    "doc2vec_dbow":   (lambda s: _doc2vec(s, dm=0),  "doc"),
}


def train_embedding(name: str, sentences: list[list[str]]):
    """Train one named embedding. Returns the gensim model object."""
    if name not in EMB_ARCHITECTURES:
        raise KeyError(f"unknown embedding {name!r}; choose from {list(EMB_ARCHITECTURES)}")
    ctor, _kind = EMB_ARCHITECTURES[name]
    return ctor(sentences)


def strip_training_docvecs(model):
    """Drop a Doc2Vec model's per-training-document vectors.

    ``infer_vector`` reads only ``dv.vector_size`` — the training document
    matrix (359,763 x 300 = 412 MB) is never consulted when featurizing a new
    PUL. Dropping it leaves inference bitwise unchanged (asserted at build
    time in ``scripts/01_train_embeddings.py``) and takes each Doc2Vec model
    down to ~3.3 MB.
    """
    import numpy as np
    model.dv.vectors = np.zeros((1, model.dv.vector_size), dtype=np.float32)
    if hasattr(model.dv, "index_to_key"):
        model.dv.index_to_key = [0]
    if hasattr(model.dv, "key_to_index"):
        model.dv.key_to_index = {0: 0}
    return model
