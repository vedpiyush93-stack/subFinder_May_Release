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


_CAZY_FAMILY_RE = re.compile(r"^(GH|PL|CE|CBM|GT|AA)([0-9]+)$")


def tok_cpu_v2(s: str) -> list[str]:
    """v2 deployed tokenizer (May 2026 refinement; TC depth revised Aug 2026).

    Two augmentations on top of tok_cpu:

      1. TC truncation to the 3-level FAMILY: "1.B.14.6.1" -> "1.B.14"
         TCDB numbers are class.subclass.family.subfamily.protein. Level 3 is
         the family — "1.B.14" is the Outer Membrane Receptor family, i.e. the
         TonB-dependent SusC-like receptors that define a PUL. That is the unit
         that carries substrate meaning, and it is also the depth the
         unsupervised corpus natively uses (99.9% of its 1.68 M TC tokens are
         3-level, vs 32.4% of supervised ones), so truncating to 3 aligns the
         two corpora exactly.

         This replaces an earlier 2-level variant ("1.B"). Two levels collapsed
         596 distinct families into 26 tokens — 2.A alone swallowed 106
         families, 1.B swallowed 73 — which is biologically meaningless. It
         scored a flattering unsupervised OOV (7.4% vs 16.5%) precisely BECAUSE
         the vocabulary had been reduced to almost nothing, and it was no more
         accurate: 5x5 RSKF 0.9151 +/- 0.0189 at 2-level vs 0.9163 +/- 0.0167
         at 3-level. The 2-level form is retained as tok_cpu_tc2 for provenance.

      2. CAZy family augmentation (keep original AND add family-only):
         "GH13" -> ["GH13", "GH"]    "AA17" -> ["AA17", "AA"]
         Gives novel CAZy families never seen in supervised data a fallback
         match on the family prefix. Because this feeds a CountVectorizer, the
         family token's COUNT also becomes a feature ("how many GH genes").

    Example:
        "1.B.14.6.1,GH5_4,CBM6|null" -> ["1.B.14", "GH5", "GH", "4", "CBM6", "CBM", "null"]

    A matching TC-family fallback (emitting "1.B" alongside "1.B.14", mirroring
    GH13 -> GH) was tested and rejected: 0.9148 +/- 0.0132, slightly worse than
    plain 3-level. The coarse token only dilutes the signal.
    """
    out = []
    for t in _COMMA_PIPE_UNDERSCORE.split(str(s)):
        if not t: continue
        if _TC_HEAD.match(t):
            parts = t.split(".")
            out.append(".".join(parts[:3]) if len(parts) >= 3 else t)
            continue
        m = _CAZY_FAMILY_RE.match(t)
        if m:
            out.append(t)
            out.append(m.group(1))
            continue
        out.append(t)
    return out
