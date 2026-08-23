"""Temperature scaling for the OvR(ExtraTrees) winner.

For an OvR-style probability vector p ∈ R^K with K classes, our calibration
protocol is:
    logit_c   = log(p_c / (1 - p_c))   # per-class binary logit
    p_T,c     = sigmoid(logit_c / T)   # divide by scalar temperature
    p_T,c    /= sum_c p_T,c            # renormalize across classes

A single scalar ``T`` is fit on inner-fold OOF probabilities. Temperature
scaling is monotonic per-class, so it preserves the argmax and cannot change a
prediction (verified: identical argmax on 100% of loci for every T we ship).

**Objective matters here.** ``fit_temperature`` minimizes NLL, which is the
conventional choice. On this model it is the wrong one. OvR-ExtraTrees outputs
are diffuse rather than peaked, so the model is systematically *under*-confident:
its observed accuracy exceeds its stated confidence by about 0.15 in every
repeat. Fixing that requires sharpening, T < 1. But NLL is dominated by the few
confident mistakes, and sharpening makes those far more expensive, so the NLL
optimum sits near T ≈ 0.96 and leaves the under-confidence essentially uncorrected
-- sometimes making 10-bin ECE worse than doing nothing.

``fit_temperature_ece`` optimizes ECE directly instead. Across the five repeats it
selects T between 0.57 and 0.68 and reduces ECE roughly four-fold (0.057-0.070 to
0.006-0.017) with predictions bit-identical. That is the objective we deploy.

Use ``CalibratedClassifier`` as a deployment-ready wrapper: it takes a fitted
sklearn pipeline and a temperature, and exposes a normal ``predict_proba``.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar


def _nll(probs: np.ndarray, y_int: np.ndarray) -> float:
    eps = 1e-9
    return float(-np.log(np.clip(probs[np.arange(len(y_int)), y_int], eps, 1.0)).mean())


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """Apply per-class-binary logit / T / sigmoid / renormalize. Vectorized over rows.

    Robust to extreme probs (0 or 1): clips before logit, and uses ``np.expm1`` /
    ``np.log1p`` style numerics implicitly via the wider eps.
    """
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    with np.errstate(over="ignore", invalid="ignore"):
        logits = np.log(p) - np.log(1 - p)
        p_cal = 1.0 / (1.0 + np.exp(-logits / T))
    s = p_cal.sum(axis=-1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return p_cal / s


def fit_temperature(probs: np.ndarray, y_int: np.ndarray,
                    bracket: tuple = (0.05, 5.0)) -> float:
    """Find the temperature T that minimizes multi-class NLL on (probs, y_int).

    IMPORTANT — leakage warning: ``probs`` must be **out-of-fold** probabilities
    obtained from an inner cross-validation on outer-train data. Using train
    predict_proba (which is overfit to the train labels) underestimates T and
    overstates calibration improvement. Use ``fit_temperature_inner_cv`` for the
    leak-free protocol.

    Args:
        probs:  (n, K) INNER-CV OOF probability matrix (uncalibrated).
        y_int:  (n,) integer class labels.
        bracket: search bracket for the scalar minimizer.
    Returns:
        Optimal T (positive scalar).
    """
    res = minimize_scalar(lambda T: _nll(apply_temperature(probs, T), y_int),
                          bracket=bracket, method="brent")
    return float(res.x)


def _ece(probs: np.ndarray, y_int: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: mean |stated confidence - observed accuracy|,
    over ``n_bins`` equal-width bins of the winning probability."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if m.sum():
            total += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(total / len(conf))


def fit_temperature_ece(probs: np.ndarray, y_int: np.ndarray,
                         bounds: tuple = (0.3, 2.0)) -> float:
    """Find T minimizing 10-bin ECE on (probs, y_int).

    Same leakage rule as ``fit_temperature``: ``probs`` must be out-of-fold.
    Prefer this over the NLL objective for this model -- see the module
    docstring for why the two disagree.
    """
    res = minimize_scalar(lambda T: _ece(apply_temperature(probs, T), y_int),
                          bounds=bounds, method="bounded")
    return float(res.x)


def fit_temperature_inner_cv(base_estimator_factory, X, y, n_inner_folds: int = 5,
                              random_state: int = 42,
                              objective: str = "ece") -> tuple[float, np.ndarray]:
    """Leak-free temperature fit: inner-CV OOF probs → fit T → return it.

    Implements the "fit on inner-CV OOF, deploy on outer-test" protocol used in
    the paper. For each fold of an internal k-fold split:
      - fit base estimator on inner-train rows
      - predict_proba on inner-val rows
    Concatenate the inner-val predictions; fit T on the resulting (n, K) matrix.

    Args:
        base_estimator_factory: zero-arg callable returning a fresh sklearn estimator
            (e.g. ``lambda: Pipeline([(\"cv\", CountVectorizer(...)), (\"vr\", OvR(...))])``).
        X, y: outer-train data (NOT the test fold).
        n_inner_folds: number of inner CV folds (5 in the paper).
        random_state: passed to inner StratifiedKFold.
        objective: ``"ece"`` (default, what we deploy) or ``"nll"`` (conventional).
    Returns:
        (T, oof_probs) — fitted temperature and the (n, K) inner-OOF probability
        matrix used to fit it (returned for ECE diagnostics).
    """
    from sklearn.model_selection import StratifiedKFold
    cls = sorted(set(y))
    K = len(cls)
    y_int = np.array([cls.index(c) for c in y])
    oof = np.zeros((len(X), K), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
    for inner_tr, inner_val in skf.split(X, y):
        m = base_estimator_factory()
        m.fit([X[i] for i in inner_tr], [y[i] for i in inner_tr])
        if hasattr(m, "classes_"): fold_classes = list(m.classes_)
        else: fold_classes = list(m.named_steps["vr"].classes_)
        col = np.array([fold_classes.index(c) for c in cls])
        p_val = m.predict_proba([X[i] for i in inner_val])[:, col]
        oof[inner_val] = p_val
    T = (fit_temperature_ece(oof, y_int) if objective == "ece"
         else fit_temperature(oof, y_int))
    return T, oof


class CalibratedClassifier:
    """Deployment-ready wrapper: ``predict_proba`` returns CALIBRATED probabilities.

    Construct via ``CalibratedClassifier(pipeline, T)`` where ``pipeline`` is a
    fitted sklearn ``Pipeline`` with steps ``[(cv, CountVectorizer), (vr, OvR)]``
    and ``T`` is a positive scalar. ``classes_`` is exposed for compatibility.
    """
    def __init__(self, pipeline, T: float):
        self.pipeline = pipeline
        self.T = float(T)
        self.classes_ = list(pipeline.named_steps["vr"].classes_)

    def predict_proba(self, X_strings):
        p_raw = self.pipeline.predict_proba(X_strings)
        return apply_temperature(p_raw, self.T)

    def predict(self, X_strings):
        return np.array(self.classes_)[self.predict_proba(X_strings).argmax(axis=1)]
