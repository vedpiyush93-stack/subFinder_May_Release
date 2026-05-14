"""Six gensim embedding architectures, paper-verbatim hyperparameters.

Each entry in ``EMB_ARCHITECTURES`` is ``(constructor_fn, kind)``:
    constructor_fn(sentences) -> trained model
    kind                       -> "wv" (FastText/Word2Vec) or "doc" (Doc2Vec)

The 6 architectures correspond to paper's grid:
    {FastText, Word2Vec, Doc2Vec} × {cbow|dm,  skip-gram|dbow}

All hyperparameters match the paper exactly (300-d, window=7, min_count=5,
60 epochs, 15 workers); the only knob exposed at the script level is the
optional ``--retrain`` flag in 01_train_embeddings.py.
"""
from __future__ import annotations
from gensim.models import FastText, Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument


EMB_HYPERPARAMS = dict(
    vector_size=300,
    window=7,
    min_count=5,
    epochs=60,
    workers=15,
)


def _fasttext(sentences, sg: int):
    return FastText(sentences=sentences, sg=sg, **EMB_HYPERPARAMS)

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
