"""subFinder release package — PUL substrate prediction.

Modules (top-down):
  preprocessing      tokenization + sparse / dense featurizers
  embeddings         word-embedding architectures (FastText, Word2Vec, Doc2Vec)
  shallow            shallow ML architectures (ExtraTrees, Balanced RF)
  deep               deep architectures (LSTM, LSTM+attn, attention-only, transformer)
  calibration        temperature scaling
  ablation           leave-one-token-out signature-gene attribution
  lit_validation     curated CAZy-substrate canonical mapping + alias collapse
  inference          end-to-end inference for new PULs

Driver scripts in scripts/ wire these modules into reproducible CLI pipelines.
"""
__version__ = "1.0.0"
