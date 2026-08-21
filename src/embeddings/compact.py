"""Collision-free, compacted FastText storage.

Why this exists
---------------
gensim's FastText keeps character n-gram ("fragment") vectors in a hash table
of ``bucket`` rows and addresses it with ``hash(fragment) % bucket``. At the
paper's ``bucket=2_000_000`` that table is 2.24 GB per model — but our
vocabulary of ~1,500 tokens only ever generates ~11k distinct fragments, so
**0.55 % of the rows are reachable and the other 99.45 % cannot be addressed
by any possible input**.

This module drops the hash at *inference* time. We enumerate every fragment the
vocabulary can produce (using gensim's own chopper), copy each fragment's
trained row out of the big table, and store it keyed by the fragment's text.
Lookup becomes a dict hit on the fragment string, so two distinct fragments can
never share a slot.

Training is unchanged: gensim still trains through the hash at the paper's
bucket size, and whatever blending happened there (0.63 % of fragments shared a
row) is preserved faithfully — this is a storage change, not a retraining.

Equivalence is verified in ``tests/verify_reduced_embedding_files.py``:
in-vocabulary vectors match exactly, and the featurizers agree to ~6e-7
(float32 summation order — we sum fragments alphabetically, gensim sums them in
hash order). The only intentional divergence is a token containing fragments
that appeared *nowhere* in training: gensim reads those from rows training never
touched (random initialisation), we contribute zero.
"""
from __future__ import annotations
import numpy as np
from gensim.models.fasttext import compute_ngrams, ft_hash_bytes


class CompactFastText:
    """Drop-in replacement for a gensim ``FastTextKeyedVectors`` (``model.wv``).

    Implements exactly the contract the featurizers rely on: ``wv[token]``,
    ``token in wv`` and ``wv.vector_size``.
    """

    def __init__(self, words, word_vectors, fragments, fragment_vectors, min_n, max_n):
        self._widx = {str(w): i for i, w in enumerate(words)}
        self._fidx = {str(f): i for i, f in enumerate(fragments)}
        self.vectors = np.asarray(word_vectors, dtype=np.float32)
        self.fragment_vectors = np.asarray(fragment_vectors, dtype=np.float32)
        self.min_n, self.max_n = int(min_n), int(max_n)
        self.vector_size = self.vectors.shape[1]

    def __contains__(self, token) -> bool:
        return token in self._widx

    def __len__(self) -> int:
        return len(self._widx)

    @property
    def key_to_index(self):
        return self._widx

    def __getitem__(self, token):
        """Reproduce ``FastTextKeyedVectors.get_vector`` without the hash table."""
        i = self._widx.get(token)
        if i is not None:
            return self.vectors[i]
        ngrams = compute_ngrams(token, self.min_n, self.max_n)   # gensim's own chopper
        vec = np.zeros(self.vector_size, dtype=np.float32)
        if not ngrams:
            return vec                       # gensim returns the origin vector here too
        for g in ngrams:
            j = self._fidx.get(g)
            if j is not None:
                vec += self.fragment_vectors[j]
        # divisor is ALL ngrams, matching gensim — unknown fragments contribute zero
        return vec / len(ngrams)


def build_compact(wv, extra_tokens=()) -> CompactFastText:
    """Compact a trained ``model.wv`` into collision-free storage.

    ``extra_tokens`` lets callers guarantee coverage for a vocabulary the
    embedding corpus did not contain (e.g. the supervised tokens), so those
    fragments resolve to their trained rows rather than to zero.
    """
    fragments = set()
    for word in wv.key_to_index:
        fragments.update(compute_ngrams(word, wv.min_n, wv.max_n))
    for token in extra_tokens:
        fragments.update(compute_ngrams(token, wv.min_n, wv.max_n))
    fragments = sorted(fragments)

    table = np.zeros((len(fragments), wv.vector_size), dtype=np.float32)
    for i, frag in enumerate(fragments):
        row = ft_hash_bytes(frag.encode("utf8")) % wv.bucket
        table[i] = wv.vectors_ngrams[row]

    return CompactFastText(
        words=list(wv.key_to_index), word_vectors=np.asarray(wv.vectors, dtype=np.float32),
        fragments=fragments, fragment_vectors=table, min_n=wv.min_n, max_n=wv.max_n)


def save_compact(path, compact: CompactFastText, **meta) -> None:
    np.savez_compressed(
        path,
        words=np.array(list(compact.key_to_index), dtype=object),
        word_vectors=compact.vectors,
        fragments=np.array(list(compact._fidx), dtype=object),
        fragment_vectors=compact.fragment_vectors,
        min_n=compact.min_n, max_n=compact.max_n,
        **meta)


def load_compact(path) -> CompactFastText:
    z = np.load(path, allow_pickle=True)
    return CompactFastText(
        words=z["words"], word_vectors=z["word_vectors"],
        fragments=z["fragments"], fragment_vectors=z["fragment_vectors"],
        min_n=int(z["min_n"]), max_n=int(z["max_n"]))


class CompactWordVectors:
    """Plain token→vector table for Word2Vec (no character n-grams, so no OOV
    fallback — an unknown token raises ``KeyError`` and the featurizers
    substitute a zero vector, exactly as before)."""

    def __init__(self, words, vectors):
        self._widx = {str(w): i for i, w in enumerate(words)}
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.vector_size = self.vectors.shape[1]

    def __contains__(self, token): return token in self._widx
    def __len__(self): return len(self._widx)

    @property
    def key_to_index(self): return self._widx

    def __getitem__(self, token):
        i = self._widx.get(token)
        if i is None:
            raise KeyError(token)
        return self.vectors[i]


def save_word_vectors(path, wv, **meta) -> None:
    np.savez_compressed(
        path,
        words=np.array(list(wv.key_to_index), dtype=object),
        vectors=np.asarray(wv.vectors, dtype=np.float32),
        **meta)


def load_word_vectors(path) -> CompactWordVectors:
    z = np.load(path, allow_pickle=True)
    return CompactWordVectors(words=z["words"], vectors=z["vectors"])
