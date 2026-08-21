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


# --- Post-hoc refinement (May 2026) --------------------------------------
# After studying the cross-domain OOV between supervised (1030 PULs) and the
# unsupervised pre-training corpus (358,751 PULs), we found that the supervised
# data uses a MIX of 5-level (1.B.14.6.1) and 3-level (1.B.14) TC numbers but
# the unsupervised corpus has ONLY 3-level. That's a pure format mismatch.
#
# Collapsing TC numbers to their 2-level family (1.B, 2.A, etc.) closes the
# format gap, keeps all CAZy / regulator tokens unchanged, drops mean unsup
# OOV from 20% → 8%, and IMPROVES supervised 5-fold acc by +0.6 pp (the TC
# fragmentation was actually noise the model had to overfit through).
#
# This is an ADDITIVE refinement — the original tok_cpu still exists and is
# used by the original deployed model (artifacts/final_model.pkl). The TC2
# variant trains a separate refined model (artifacts/final_model_tc2.pkl).
_TC_HEAD = re.compile(r"^[0-9]+\.[A-Z]\.?")


def tok_cpu_tc2(s: str) -> list[str]:
    """Post-hoc refinement of tok_cpu: also collapse TC numbers to 2-level family.

    Example:
        "1.B.14.6.1,GH5_4,CBM6|null" -> ["1.B", "GH5", "4", "CBM6", "null"]

    Effect: closes the supervised/unsupervised TC-format mismatch.
    Supervised 5-fold CV acc: 0.9039 (tok_cpu) -> 0.9097 (tok_cpu_tc2), +0.6 pp.
    Unsupervised mean OOV: 20.23% (tok_cpu) -> 8.35% (tok_cpu_tc2).
    Share of unsup PULs at OOV<=10%: 30.8% -> 69.0% (+38 pp).
    """
    out = []
    for t in _COMMA_PIPE_UNDERSCORE.split(str(s)):
        if not t: continue
        if _TC_HEAD.match(t):
            parts = t.split(".")
            out.append(".".join(parts[:2]) if len(parts) >= 2 else t)
        else:
            out.append(t)
    return out


def tok_cpu_v2(s: str) -> list[str]:
    """Deployed tokenizer: tok_cpu, with TC numbers read at the 3-level family.

    Example:
        "1.B.14.6.1,GH5_4,CBM6|null" -> ["1.B.14", "GH5", "4", "CBM6", "null"]

    TC identifiers are class.subclass.family.subfamily.protein, so level 3 is the
    family: "1.B.14" is the Outer Membrane Receptor family, the TonB-dependent
    SusC-like receptors characteristic of a PUL. Three levels is also the depth
    the corpus is written at -- 99.9% of the 1,678,991 TC tokens in the
    unsupervised corpus are already 3-level, while the labelled set mixes 3- and
    5-level forms -- so reading both at the family level makes "1.B.14.6.1" and
    "1.B.14" recognisable as the same transporter instead of two unrelated
    strings.

    CAZy tokens are left exactly as they are. An earlier variant additionally
    emitted a family-only companion ("GH13" -> "GH13" plus "GH") to give unseen
    families something to match on; that is deliberately not done, because it
    conflates enzymes of very different specificity under one feature -- every
    GH is counted together regardless of what it acts on.

    A 2-level TC variant is kept as tok_cpu_tc2 for provenance; it collapses 596
    distinct transporter families into 26 labels, which is too coarse to carry
    substrate signal.
    """
    out = []
    for t in _COMMA_PIPE_UNDERSCORE.split(str(s)):
        if not t: continue
        if _TC_HEAD.match(t):
            parts = t.split(".")
            out.append(".".join(parts[:3]) if len(parts) >= 3 else t)
            continue
        out.append(t)
    return out
