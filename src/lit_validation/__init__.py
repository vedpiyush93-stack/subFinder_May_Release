"""Curated CAZy-substrate canonical mapping with alias collapse + literature citations.

The hand-curated literature DB at ``data/Literature_Data_fam_substrate_mapping.tsv``
uses **75 fine-grained substrate names**. Our model output space is **12 classes**.
``SUBSTRATE_ALIAS`` is the audit-trail mapping that collapses the 75 lit names to
our 12 with primary-literature citations for every non-trivial collapse.

See ``alias_map.py`` for the full table with citations and ``canon.py`` for the
helpers that build the (substrate, CAZy-family) canonical sets.
"""
from .alias_map import SUBSTRATE_ALIAS, ALIAS_CITATIONS
from .canon import build_canon, compute_in_scope, compute_flagged
__all__ = ["SUBSTRATE_ALIAS", "ALIAS_CITATIONS",
           "build_canon", "compute_in_scope", "compute_flagged"]
