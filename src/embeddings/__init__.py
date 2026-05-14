"""Word-embedding architectures and per-fold training utilities.

All six paper-shipped embeddings are wrapped here as factory functions so that
the training script can iterate over them cleanly.

Architecture naming (used as keys throughout the codebase):
  fasttext_cbow   FastText CBOW              (300-d, n-gram OOV fallback)
  fasttext_sg     FastText skip-gram         (300-d, n-gram OOV fallback)
  word2vec_cbow   Word2Vec CBOW              (300-d, no OOV fallback)
  word2vec_sg     Word2Vec skip-gram         (300-d, no OOV fallback)
  doc2vec_dm      Doc2Vec distributed memory (300-d, per-document inference)
  doc2vec_dbow    Doc2Vec DBOW               (300-d, per-document inference)
"""
from .architectures import (
    EMB_ARCHITECTURES, train_embedding, EMB_HYPERPARAMS,
)
__all__ = ["EMB_ARCHITECTURES", "train_embedding", "EMB_HYPERPARAMS"]
