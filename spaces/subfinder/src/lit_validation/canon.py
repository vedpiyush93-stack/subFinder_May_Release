"""Build per-substrate canonical-CAZy-family sets from the curated lit DB."""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

from .alias_map import SUBSTRATE_ALIAS
from src.preprocessing.tokenizers import tok_cpu


_SPLIT_RE = re.compile(r",|and\s+")


def _split_lit_substr(s: str) -> list[str]:
    """Split a multi-substrate lit-row label on commas and the word 'and'."""
    return [p.strip() for p in _SPLIT_RE.split(str(s)) if p.strip()]


def build_canon(lit_tsv_path: Path | str,
                alias_map: dict[str, list[str]] = SUBSTRATE_ALIAS
                ) -> dict[str, set[str]]:
    """Build the (substrate → {canonical CAZy families}) mapping.

    Args:
        lit_tsv_path: path to ``Literature_Data_fam_substrate_mapping.tsv``.
        alias_map:    substrate → list of lit-name aliases.
    Returns:
        Dict mapping each of our 12 substrates to its canonical CAZy-family set.
    """
    lit = pd.read_csv(lit_tsv_path, sep="\t")
    lit.columns = [c.strip() for c in lit.columns]
    canon = {s: set() for s in alias_map}
    for _, row in lit.iterrows():
        parts = _split_lit_substr(row["Substrate_high_level"])
        for our_sub, aliases in alias_map.items():
            if set(aliases) & set(parts):
                canon[our_sub].add(row["Family"])
    return canon


def compute_in_scope(canon: dict[str, set[str]], pul_sequences, true_labels) -> dict[str, set[str]]:
    """For each substrate s, return the set of canon-CAZy that ACTUALLY appear in
    some test PUL with true substrate s. This is the 'in-scope' set used in the
    paper's scope-coverage metric (Table~S5 / Slide 11)."""
    scope = {s: set() for s in canon}
    for pul, true in zip(pul_sequences, true_labels):
        toks = set(tok_cpu(pul))
        if true in scope:
            scope[true] |= toks & canon[true]
    return scope


def compute_flagged(canon: dict[str, set[str]], oof_df, target_class_col: str = "true",
                    top_col: str = "top3") -> dict[str, set[str]]:
    """For each substrate s, return the set of canon-CAZy the model flagged as a
    top-K sig gene for class s in at least one OOF PUL.

    Args:
        canon:           output of ``build_canon``.
        oof_df:          DataFrame with columns [target_class_col, top_col].
        target_class_col: which column holds the substrate to score against
                          (default 'true' for true-class attribution; use 'pred'
                          for argmax-class attribution).
        top_col:         column holding ';'-joined top-K tokens.
    Returns:
        Dict mapping each substrate to its 'flagged' set (a subset of canon[s]).
    """
    flagged = {s: set() for s in canon}
    for _, r in oof_df.iterrows():
        s = r[target_class_col]
        if s not in flagged: continue
        toks = set(str(r[top_col]).split(";"))
        flagged[s] |= toks & canon[s]
    return flagged
