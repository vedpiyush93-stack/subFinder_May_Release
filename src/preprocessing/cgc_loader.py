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

TC handling — why ``tc_mode="both"`` by default
------------------------------------------------
The training corpus ``data/Train_data.csv`` contains BOTH full 5-part TC
numbers (e.g. ``1.B.14.12.1``) AND 3-part truncated TC (e.g. ``1.B.14``).
We measured 177 5-part vs. 52 3-part TC tokens in the training vocab,
with 41 TC families appearing in BOTH forms (e.g. ``1.B.14`` alongside
``1.B.14.6.1``, ``1.B.14.12.1``, ``1.B.14.10.1``).

The legacy parser ``Codes/import_data.py`` truncated TC to 3 parts — which
silently lost matches with the 5-part half of the training vocab. To
maximize vocab overlap at inference time, this loader's default is
``tc_mode="both"`` — emit BOTH forms separated by ``|``:

* ``"1.B.14.12.1"`` → ``"1.B.14.12.1|1.B.14"``
* ``tok_cpu`` later splits on ``|`` so both columns activate in CountVec.
* The classifier uses whichever it has weights for.

``tc_mode`` options
-------------------
* ``"both"`` (default): emit both full + 3-part. Recommended.
* ``"truncate"`` : emit only 3-part. Matches legacy ``import_data.py``.
* ``"full"``     : emit only the original (no transformation).

This loader does **not** modify the legacy script; it's an inference-only
convention.

Featurizer gotcha to know
-------------------------
``tok_cpu`` from :pyfile:`src/preprocessing/tokenizers.py` splits on three
characters: ``,``, ``|``, and ``_``. The ``_`` split means that subfamily
indices like ``GH43_34`` become two tokens ``[GH43, 34]`` after
tokenization. This applies to both the raw token-string input and the
CGC-parsed input — the model already learnt that convention at training
time. As long as the *atomic* output of ``tok_cpu`` matches what the model
expects, both input paths give identical predictions.
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
                        tc_mode: str = "both") -> dict[str, str]:
    """Read a cgc_standard.out file → ``{cgc_id: pul_string}``.

    Parameters
    ----------
    filepath : path to a dbCAN cgc_standard.out (tab-separated, 8 cols).
    tc_mode  : how to render TC numbers — ``"both"`` (default), ``"full"``,
                or ``"truncate"``. See module docstring.

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
