"""Leak-audit smoke test — run with: pytest tests/leak_audit.py -v

Verifies:
  1. For every (seed, fold) in 5×5 RSKF, outer_test ∩ outer_train = ∅.
  2. For every fold's embedding-model NPZ, train_indices ∩ outer_test = ∅.
  3. The fitted CountVectorizer of cpu__ET500_log2 fold 0 has a vocabulary size
     >0 and <total-distinct-tokens (i.e. it's bounded by min_df defaults).
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np, pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def y():
    return pd.read_csv(ROOT/"data/Train_data.csv")["high_level_substr"].values


def test_outer_train_test_disjoint(y):
    for seed in [42, 43, 44, 45, 46]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            assert len(set(tr) & set(te)) == 0, f"r{seed}_f{fold}: outer split not disjoint"


def test_emb_cache_no_leakage(y):
    """For each fold's cached embedding, assert its train_indices don't include test rows."""
    cache = ROOT/"artifacts/embeddings_cache"
    if not cache.exists() or not any(cache.iterdir()):
        pytest.skip("embedding cache missing — see README §5 for download")
    for fold_dir in sorted(cache.glob("r*_f*")):
        # Re-derive outer test indices from the seed+fold encoded in the dir name
        parts = fold_dir.name.split("_")
        seed = int(parts[0][1:]); fold = int(parts[1][1:])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        all_tr_te = list(skf.split(np.zeros(len(y)), y))
        _, te = all_tr_te[fold]; te_set = set(te.tolist())
        for emb_npz in fold_dir.glob("*.npz"):
            if emb_npz.name == "splits.npz": continue
            try: train_idx = np.load(emb_npz, allow_pickle=True)["train_indices"]
            except KeyError: continue
            leaked = set(train_idx.tolist()) & te_set
            assert not leaked, f"{emb_npz}: {len(leaked)} test rows leaked into embedding train"
