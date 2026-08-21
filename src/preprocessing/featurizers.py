"""Featurizers — turn a list of PUL token-strings into a (n, d) feature matrix.

Three families:
  1. ``CountVecFeaturizer``         sparse bag-of-tokens (the winner uses this).
  2. ``EmbeddingMeanFeaturizer``    300-d mean of token vectors (paper's FastText/W2V default).
  3. ``EmbeddingMeanMaxFeaturizer`` 600-d mean+max-concat (our second-best shallow).

All featurizers are leak-free by construction: ``fit`` learns vocabulary from
train rows only; ``transform`` silently drops out-of-vocabulary tokens.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from .tokenizers import tok_cpu, tok_comma_pipe, tok_cpu_v2


class CountVecFeaturizer:
    """Sparse bag-of-tokens featurizer wrapping sklearn's ``CountVectorizer``.

    Args:
        tokenizer: "cpu" (split on `,|_`), "comma_pipe" (split on `,|`), or
            "cpu_v2" (as "cpu", plus TC numbers truncated to their 3-level
            family and CAZy tokens augmented with a family-only fallback).
    """
    def __init__(self, tokenizer: str = "cpu"):
        tk = {"cpu": tok_cpu, "comma_pipe": tok_comma_pipe, "cpu_v2": tok_cpu_v2}[tokenizer]
        self._cv = CountVectorizer(tokenizer=tk, token_pattern=None, lowercase=False)
        self.tokenizer_name = tokenizer

    def fit(self, X_strings: list[str]):
        self._cv.fit(X_strings)
        return self

    def transform(self, X_strings: list[str]):
        return self._cv.transform(X_strings)

    def fit_transform(self, X_strings: list[str]):
        return self._cv.fit_transform(X_strings)

    @property
    def vocabulary_(self):
        return self._cv.vocabulary_


class EmbeddingMeanFeaturizer:
    """300-d mean-of-token-vectors. Pass a gensim ``KeyedVectors`` at construction.

    For FastText, OOV tokens are mapped via n-gram fallback (``model.wv[t]``).
    For Word2Vec / Doc2Vec, OOV tokens contribute a zero vector.
    """
    def __init__(self, wv, tokenizer: str = "comma_pipe", vector_size: int | None = None):
        self.wv = wv
        self._tok = {"cpu": tok_cpu, "comma_pipe": tok_comma_pipe}[tokenizer]
        self.vector_size = vector_size or wv.vector_size

    def _vec_one(self, s: str) -> np.ndarray:
        toks = self._tok(s)
        vecs = []
        for t in toks:
            try:
                vecs.append(self.wv[t])
            except (KeyError, AttributeError):
                vecs.append(np.zeros(self.vector_size, dtype=np.float32))
        return np.mean(vecs, axis=0) if vecs else np.zeros(self.vector_size, dtype=np.float32)

    def transform(self, X_strings: list[str]) -> np.ndarray:
        return np.array([self._vec_one(s) for s in X_strings], dtype=np.float32)


class EmbeddingMeanMaxFeaturizer(EmbeddingMeanFeaturizer):
    """600-d mean+max-concat — preserves the loudest signature gene per dim."""
    def _vec_one(self, s: str) -> np.ndarray:
        toks = self._tok(s)
        vecs = []
        for t in toks:
            try: vecs.append(self.wv[t])
            except (KeyError, AttributeError):
                vecs.append(np.zeros(self.vector_size, dtype=np.float32))
        if not vecs:
            return np.zeros(2 * self.vector_size, dtype=np.float32)
        arr = np.stack(vecs)
        return np.concatenate([arr.mean(axis=0), arr.max(axis=0)]).astype(np.float32)

    def transform(self, X_strings):
        out = np.array([self._vec_one(s) for s in X_strings], dtype=np.float32)
        return out  # shape (n, 2*vector_size)


class Doc2VecInferFeaturizer:
    """300-d *document* vector per PUL, via ``Doc2Vec.infer_vector``.

    Doc2Vec learns two things: word vectors, and a vector per training
    document. The document vector is the object the model is actually trained
    to produce, so that is what we featurize with — word vectors are what
    Word2Vec is for, and for DBOW they are never trained at all.

    A new PUL gets its own vector by gradient descent on *that vector only*:
    gensim runs the inference passes with ``learn_words=False,
    learn_hidden=False``, so the trained model is never updated by a sample it
    is asked to featurize.

    Determinism
    -----------
    ``infer_vector`` is stochastic out of the box — the same PUL yields a
    different vector on each call, because negative sampling draws from
    ``model.random``, which advances between calls. gensim's own docstring
    warns about this. We reseed that generator immediately before every call,
    which makes inference bitwise reproducible (verified across repeated calls,
    independent model reloads, and after stripping the training doc-vectors).
    """

    def __init__(self, model, tokenizer: str = "comma_pipe", seed: int = 42, epochs: int | None = None):
        self.model = model
        self._tok = {"cpu": tok_cpu, "comma_pipe": tok_comma_pipe}[tokenizer]
        self.seed = seed
        self.epochs = epochs                       # None -> the model's own training epochs
        self.vector_size = model.dv.vector_size

    def _vec_one(self, s: str) -> np.ndarray:
        toks = self._tok(s)
        if not toks:
            return np.zeros(self.vector_size, dtype=np.float32)
        self.model.random = np.random.RandomState(self.seed)   # <- makes inference deterministic
        return self.model.infer_vector(toks, epochs=self.epochs)

    def transform(self, X_strings: list[str]) -> np.ndarray:
        return np.array([self._vec_one(s) for s in X_strings], dtype=np.float32)
