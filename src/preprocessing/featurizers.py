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

from .tokenizers import tok_cpu, tok_comma_pipe


class CountVecFeaturizer:
    """Sparse bag-of-tokens featurizer wrapping sklearn's ``CountVectorizer``.

    Args:
        tokenizer: either "cpu" (split on `,|_`) or "comma_pipe" (split on `,|`).
    """
    def __init__(self, tokenizer: str = "cpu"):
        tk = {"cpu": tok_cpu, "comma_pipe": tok_comma_pipe}[tokenizer]
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
