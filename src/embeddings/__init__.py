"""Word-embedding architectures, compacted storage, and loading utilities.

Architecture naming (used as keys throughout the codebase):
  fasttext_cbow   FastText CBOW              (300-d, character n-gram OOV fallback)
  fasttext_sg     FastText skip-gram         (300-d, character n-gram OOV fallback)
  word2vec_cbow   Word2Vec CBOW              (300-d, no OOV fallback)
  word2vec_sg     Word2Vec skip-gram         (300-d, no OOV fallback)
  doc2vec_dm      Doc2Vec distributed memory (300-d, per-document inference)
  doc2vec_dbow    Doc2Vec DBOW               (300-d, per-document inference)

All six are trained once on the unsupervised corpus and frozen.
"""
from .architectures import (
    EMB_ARCHITECTURES, train_embedding, EMB_HYPERPARAMS,
    FASTTEXT_BUCKET, strip_training_docvecs,
)
from .compact import (
    CompactFastText, CompactWordVectors, build_compact,
    save_compact, load_compact, save_word_vectors, load_word_vectors,
)
from .loader import load_embedding, load_doc2vec, embedding_path

__all__ = ["EMB_ARCHITECTURES", "train_embedding", "EMB_HYPERPARAMS",
           "FASTTEXT_BUCKET", "strip_training_docvecs",
           "CompactFastText", "CompactWordVectors", "build_compact",
           "save_compact", "load_compact", "save_word_vectors", "load_word_vectors",
           "load_embedding", "load_doc2vec", "embedding_path"]
