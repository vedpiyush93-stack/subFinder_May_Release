"""Two PUL tokenizers used in the paper.

A PUL sequence is a comma/pipe-separated string of CAZy family tokens
(e.g. ``GH13_4``), CAZy modules (``CBM6``), Transporter Classification IDs
(``1.B.14.6.1``), transcription-factor types and ``null`` padding.

- ``tok_comma_pipe`` is the original paper's split: ``,`` and ``|`` only.
  Keeps ``GH5_4`` as one token.
- ``tok_cpu`` is our additional split on ``_`` so that ``GH5_4`` becomes
  ``GH5`` + ``4``. The family token ``GH5`` is now shared across all GH5
  subfamily variants which lifts in-distribution accuracy by ~1-2 pp.
"""
import re

_COMMA_PIPE = re.compile(r"[,|]")
_COMMA_PIPE_UNDERSCORE = re.compile(r"[,|_]")
_CAZY = re.compile(r"^(GH|PL|CE|CBM|GT|AA)[0-9]+$")


def tok_comma_pipe(s: str) -> list[str]:
    """Paper's original tokenizer — split on ',' and '|' only."""
    return [t for t in _COMMA_PIPE.split(str(s)) if t]


def tok_cpu(s: str) -> list[str]:
    """Our tokenizer — also splits on '_' so CAZy subfamilies share a family token.

    Example:
        "GH5_4,CBM6|null" -> ["GH5", "4", "CBM6", "null"]
    """
    return [t for t in _COMMA_PIPE_UNDERSCORE.split(str(s)) if t]


def is_cazy(tok: str) -> bool:
    """Return True iff *tok* is a CAZy family identifier (GH/PL/CE/CBM/GT/AA + digits)."""
    return bool(_CAZY.match(str(tok)))
