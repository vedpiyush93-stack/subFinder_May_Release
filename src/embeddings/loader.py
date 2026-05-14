"""Load a gensim ``FastText`` (or any) embedding model, transparently
auto-decompressing xz-compressed n-gram bucket sidecars.

Why this exists
---------------
GitHub LFS has a 2 GB per-file hard limit. The raw FastText n-gram bucket
table (``*.model.wv.vectors_ngrams.npy``) is 2.235 GB. We ship a lossless
xz-compressed version (~1.86 GB) instead, then decompress at load time.

The decompressed bytes are bit-identical to the source-of-truth (gzip is
a lossless algorithm; xz is too). Verified empirically that
``load_fasttext("model.model")`` after decompress gives the same vector
for both in-vocab tokens and OOV-via-n-gram tokens as loading from the
original uncompressed model dir.

Usage
-----
    from src.embeddings.loader import load_fasttext

    m = load_fasttext("artifacts/embeddings_cache/r42_f0/fasttext_cbow_shallow_model/fasttext_cbow.model")
    v = m.wv["GH13_99"]      # OOV → n-gram-resolved vector, not zero
    in_vocab = m.wv["GT2"]   # in-vocab → standard lookup

Decompression is done once per ``.npy.xz`` sibling (cached on disk next to
the .xz file). Subsequent loads find the .npy directly and skip the
decompress step.
"""
from __future__ import annotations
import os
import lzma
import shutil
import time
from pathlib import Path


def _decompress_xz_sibling(xz_path: str | Path) -> str:
    """Decompress ``foo.npy.xz`` → ``foo.npy`` next to it. No-op if already
    decompressed. Returns the .npy path."""
    xz_path = str(xz_path)
    target = xz_path[:-3]                          # strip ".xz"
    if os.path.exists(target):
        return target
    t0 = time.time()
    tmp = target + ".part"
    with lzma.open(xz_path, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
    os.replace(tmp, target)
    print(f"  [fasttext-loader] decompressed {os.path.basename(xz_path)} "
          f"({os.path.getsize(target)/1024**3:.2f} GB) in {time.time()-t0:.1f}s")
    return target


def load_fasttext(model_path: str | Path):
    """Drop-in replacement for ``gensim.models.FastText.load()``.

    Before calling the upstream loader, walks the model's directory and
    decompresses any ``*.npy.xz`` sidecars to their ``*.npy`` form (one-time
    per file). Then defers to ``FastText.load()`` as usual.

    Parameters
    ----------
    model_path : path to the ``.model`` pickle (the gensim convention).

    Returns
    -------
    The fully-loaded ``gensim.models.FastText`` instance, with n-gram OOV
    fallback fully functional via ``model.wv[token]``.
    """
    from gensim.models import FastText
    model_path = str(model_path)
    model_dir = os.path.dirname(model_path)

    # find + decompress every .npy.xz sibling we haven't already decompressed
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".npy.xz"):
            _decompress_xz_sibling(os.path.join(model_dir, f))

    return FastText.load(model_path)
