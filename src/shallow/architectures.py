"""Three shallow classifier architectures wrapped in One-vs-Rest for 12 classes.

Architectures (used as keys throughout the codebase):
  ET500_log2   OvR(ExtraTrees n=500, max_features='log2')   <-- our winner
  ET500_sqrt   OvR(ExtraTrees n=500, max_features='sqrt')   <-- our second-best
  BRF100       OvR(BalancedRandomForest n=100)              <-- paper's baseline

All three use ``class_weight='balanced'``. ExtraTrees uses ``bootstrap=False``
(no bagging) so every tree sees the full outer-training fold.
"""
from __future__ import annotations
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except Exception:  # pragma: no cover
    BalancedRandomForestClassifier = None


def _ET(max_features: str):
    return OneVsRestClassifier(ExtraTreesClassifier(
        n_estimators=500, max_features=max_features,
        class_weight="balanced", bootstrap=False, random_state=42))

def _BRF():
    if BalancedRandomForestClassifier is None:
        raise RuntimeError("imblearn not installed — `pip install imbalanced-learn`")
    return OneVsRestClassifier(BalancedRandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        sampling_strategy="all", replacement=True, bootstrap=False, random_state=42))


SHALLOW_ARCHITECTURES = {
    "ET500_log2": lambda: _ET("log2"),
    "ET500_sqrt": lambda: _ET("sqrt"),
    "BRF100":     _BRF,
}


def build_shallow(name: str):
    """Return a freshly-instantiated shallow classifier."""
    if name not in SHALLOW_ARCHITECTURES:
        raise KeyError(f"unknown shallow arch {name!r}; choose from {list(SHALLOW_ARCHITECTURES)}")
    return SHALLOW_ARCHITECTURES[name]()
