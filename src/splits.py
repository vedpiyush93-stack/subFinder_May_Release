"""5×5 RSKF splitter — yields (seed, fold, train_idx, test_idx, train_inner_idx, val_idx)."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def rskf_splits(y, seeds=DEFAULT_SEEDS, n_folds: int = 5):
    """Yield (seed, fold, tr_outer, te, tr_inner, val) for each of |seeds|×n_folds splits.

    * ``tr_outer`` / ``te``: 80/20 outer split (the canonical 5×5 RSKF protocol).
    * ``tr_inner`` / ``val``: 75/25 stratified subsplit of ``tr_outer`` for DL
      EarlyStopping — paper-verbatim, random_state=42.
    """
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr_outer, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            tr_inner, val = train_test_split(
                tr_outer, test_size=0.25, random_state=42, stratify=y[tr_outer])
            yield seed, fold, tr_outer, te, tr_inner, val
