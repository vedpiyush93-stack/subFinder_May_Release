"""Parse a dbCAN-style ``cgc_standard.out`` TSV → per-CGC PUL token strings
that match the format the inference pipeline expects.

Input format
------------
Standard dbCAN ``cgc_standard.out``: 8 tab-separated columns ::

    CGC#  Gene Type  Contig ID  Protein ID  Gene Start  Gene Stop  Direction  Protein Family

One line per gene. Genes are grouped by ``(Contig ID, CGC#)``. Output is
a comma-joined annotation string per CGC, ready to feed to ``tok_cpu`` /
``CountVectorizer``.

Per-gene rules (mirror ``/Users/ved/subFinder/inference/cgc_loader.py``)
-------------------------------------------------------------------------
============  ============================================================
gene_type     transformation on the ``Protein Family`` column
============  ============================================================
``null``      always emit literal ``"null"``
``TF``        replace ``+`` (multi-domain separator) with ``|`` so that
``STP``       ``tok_cpu`` later splits the domains into separate tokens
``CAZyme``    keep as-is, just strip whitespace
``TC``        controlled by ``tc_mode`` (see below)
*(other)*     keep as-is
============  ============================================================

TC handling — why ``tc_mode="full"`` is the default
---------------------------------------------------
The deployed tokenizer ``tok_cpu_v2`` already reads a TC identifier at its
3-level family, so ``1.B.14.12.1`` and ``1.B.14`` become the same feature no
matter which form arrives. The reader therefore passes TC annotations through
untouched and lets the tokenizer do the truncation once.

An earlier default emitted BOTH forms separated by ``|`` (``"1.B.14.12.1" ->
"1.B.14.12.1|1.B.14"``). That was correct for the older ``tok_cpu``, which kept
the full identifier verbatim and so genuinely needed the truncated form spelled
out to match the training vocabulary's 3-level half. Under ``tok_cpu_v2`` it is
actively wrong: both halves collapse to ``1.B.14``, and because the featurizer
counts tokens, every transporter is counted twice. The same locus then produces
a different feature vector depending on whether it arrived as a token string or
as a CGC table -- measured at up to 0.251 of probability mass on the deployed
model. ``tests/verify_input_formats_agree.py`` pins this down.

``tc_mode`` options
-------------------
* ``"full"`` (default): pass the TC annotation through unchanged and let
  ``tok_cpu_v2`` read it at the family level. Predictions then match the
  token-string path exactly.
* ``"truncate"`` : emit only the 3-part form. Equivalent under ``tok_cpu_v2``;
  retained because it matches the legacy ``import_data.py`` convention.
* ``"both"``     : emit full + 3-part. Only correct for the older ``tok_cpu``;
  double-counts every transporter under ``tok_cpu_v2``.

This loader does **not** modify the legacy script; it's an inference-only
convention.

Featurizer gotcha to know
-------------------------
``tok_cpu_v2`` splits on ``,``, ``|`` and ``_``, so a subfamily index like
``GH43_34`` becomes ``[GH43, 34]``. That applies equally to the token-string
input and the CGC-parsed input, and the model learnt the convention at training
time. What matters is that the *multiset* of tokens matches: because the
featurizer counts occurrences, emitting a token twice is not harmless.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REQUIRED_COLS = 8


def _format_tc(ann: str, tc_mode: str) -> str:
    parts = ann.split(".")
    if tc_mode == "full" or len(parts) <= 3:
        return ann
    if tc_mode == "truncate":
        return ".".join(parts[:3])
    if tc_mode == "both":
        return f"{ann}|{'.'.join(parts[:3])}"
    raise ValueError(f"tc_mode must be 'both' | 'full' | 'truncate', got {tc_mode!r}")


def _parse_line(parts: list[str], tc_mode: str) -> tuple[str, str]:
    cgc_id     = f"{parts[2]}|{parts[0]}"
    gene_type  = parts[1]
    annotation = parts[7].replace(" ", "")
    if gene_type == "null":
        annotation = "null"
    elif gene_type == "TC":
        annotation = _format_tc(annotation, tc_mode)
    elif gene_type in ("TF", "STP"):
        annotation = annotation.replace("+", "|")
    return cgc_id, annotation


def cgc_to_pul_strings(filepath: str | Path,
                        tc_mode: str = "full") -> dict[str, str]:
    """Read a cgc_standard.out file → ``{cgc_id: pul_string}``.

    Parameters
    ----------
    filepath : path to a dbCAN cgc_standard.out (tab-separated, 8 cols).
    tc_mode  : how to render TC numbers — ``"full"`` (default), ``"truncate"``,
                or ``"both"``. See module docstring.

    Returns
    -------
    dict with one key per CGC (formatted ``"<contig_id>|<CGC#>"``) and
    value = comma-joined PUL annotation string ready for ``predict()``.

    Lines with fewer than ``REQUIRED_COLS`` columns are skipped silently
    (header line + any malformed rows).
    """
    with open(filepath) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    cgc_dict: dict[str, list[str]] = defaultdict(list)
    for line in lines[1:]:                       # skip header
        parts = line.split("\t")
        if len(parts) < REQUIRED_COLS:
            continue
        cgc_id, annotation = _parse_line(parts, tc_mode)
        cgc_dict[cgc_id].append(annotation)
    return {cid: ",".join(annots) for cid, annots in cgc_dict.items()}
