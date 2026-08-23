"""Leave-one-token-out Δ-prob signature-gene attribution.

For a PUL with tokens ``T = {t_1, …, t_n}`` and a class ``s`` of interest,
the **signature genes** are the tokens whose removal causes the largest drop
in ``P(s | PUL)``:

    Δ_s(t) = P(s | T) − P(s | T \\ {t})

Top-K tokens by ``Δ_s`` are the model's signature genes for substrate ``s``.

Two common choices of ``s``:
  * ``s = argmax_c P(c | T)`` — argmax-class attribution (deployment view).
  * ``s = true substrate``    — true-class attribution (clean attribution test;
                                 decouples attribution from classification).

The batched implementation calls ``predict_proba`` once on a giant stacked
sparse matrix containing one ablated copy per (PUL, token) pair. On the
seed-42 5-fold OOF of ``cpu__ET500_log2`` (1030 PULs × ~12 tokens/PUL =
~12k rows) the total wall time is ~8 sec.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse


def ablate_pul(pipeline, X_str: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Score (token, Δ_argmax) pairs for one PUL via argmax-class attribution.

    Args:
        pipeline: fitted sklearn Pipeline with steps (cv: CountVectorizer, vr: OvR classifier).
        X_str:    PUL token-string (e.g. "GH13,CBM6|null").
        top_k:    number of tokens to return.
    Returns:
        ``[(token, delta), …]`` sorted by Δ descending; max len = top_k.
    """
    cv = pipeline.named_steps["cv"]; clf = pipeline.named_steps["vr"]
    Xv = cv.transform([X_str]).tocsr()
    p_full = clf.predict_proba(Xv)
    ci = int(p_full.argmax(1)[0])
    p_full_class = float(p_full[0, ci])
    inv = {v: k for k, v in cv.vocabulary_.items()}
    deltas = []
    nnz_cols = Xv.getrow(0).nonzero()[1]
    if len(nnz_cols) == 0: return []
    mods = []
    for col in nnz_cols:
        mod = Xv.tolil(); mod[0, col] = 0
        mods.append(mod.tocsr())
    big = sparse.vstack(mods)
    p_ab = clf.predict_proba(big)
    for i, col in enumerate(nnz_cols):
        deltas.append((inv[col], p_full_class - float(p_ab[i, ci])))
    deltas.sort(key=lambda x: -x[1])
    return deltas[:top_k]


def ablate_pul_for_class(pipeline, X_str: str, class_name: str,
                         top_k: int = 5, apply_temp: float | None = None
                         ) -> list[tuple[str, float]]:
    """Score (token, Δ_class) pairs for one PUL given a target class.

    Args:
        pipeline:    fitted sklearn Pipeline.
        X_str:       PUL token-string.
        class_name:  target substrate (must be in pipeline.named_steps['vr'].classes_).
        top_k:       number of tokens to return.
        apply_temp:  if a positive float, apply temperature scaling with that T
                     to both the full and ablated probabilities before subtracting.
    Returns:
        ``[(token, delta_class), …]`` sorted by Δ descending; max len = top_k.
    """
    from src.calibration.temperature import apply_temperature
    cv = pipeline.named_steps["cv"]; clf = pipeline.named_steps["vr"]
    classes = list(clf.classes_)
    ci = classes.index(class_name)
    Xv = cv.transform([X_str]).tocsr()
    p_full = clf.predict_proba(Xv)
    if apply_temp is not None:
        p_full = apply_temperature(p_full, apply_temp)
    p_full_class = float(p_full[0, ci])
    inv = {v: k for k, v in cv.vocabulary_.items()}
    nnz_cols = Xv.getrow(0).nonzero()[1]
    if len(nnz_cols) == 0: return []
    mods = []
    for col in nnz_cols:
        mod = Xv.tolil(); mod[0, col] = 0
        mods.append(mod.tocsr())
    big = sparse.vstack(mods)
    p_ab = clf.predict_proba(big)
    if apply_temp is not None:
        p_ab = apply_temperature(p_ab, apply_temp)
    deltas = [(inv[col], p_full_class - float(p_ab[i, ci])) for i, col in enumerate(nnz_cols)]
    deltas.sort(key=lambda x: -x[1])
    return deltas[:top_k]


def batched_ablation(pipeline, X_strings: list[str], target: str = "argmax",
                     true_labels=None, apply_temp: float | None = None,
                     top_k: int = 5):
    """Vectorized ablation over many PULs.

    Args:
        pipeline:    fitted sklearn Pipeline.
        X_strings:   list of n PUL token-strings.
        target:      "argmax" (default) or "true" (requires ``true_labels``).
        true_labels: list of n substrate strings (used only when target="true").
        apply_temp:  optional scalar T to apply per-class binary temperature scaling.
        top_k:       number of tokens to return per PUL.
    Returns:
        List of n lists of ``(token, delta)`` tuples.
    """
    from src.calibration.temperature import apply_temperature
    cv = pipeline.named_steps["cv"]; clf = pipeline.named_steps["vr"]
    classes = list(clf.classes_)
    Xv = cv.transform(X_strings).tocsr()
    P = clf.predict_proba(Xv)
    if apply_temp is not None: P = apply_temperature(P, apply_temp)
    if target == "argmax":
        target_idx = P.argmax(axis=1)
    elif target == "true":
        if true_labels is None: raise ValueError("true_labels required for target='true'")
        target_idx = np.array([classes.index(t) for t in true_labels])
    else:
        raise ValueError(f"unknown target={target!r}")
    inv = {v: k for k, v in cv.vocabulary_.items()}
    rows_mod, meta = [], []
    for i in range(Xv.shape[0]):
        row = Xv.getrow(i)
        for col in row.nonzero()[1]:
            mod = row.tolil(); mod[0, col] = 0
            rows_mod.append(mod.tocsr())
            meta.append((i, inv[col]))
    big = sparse.vstack(rows_mod) if rows_mod else sparse.csr_matrix((0, Xv.shape[1]))
    Pa = clf.predict_proba(big)
    if apply_temp is not None: Pa = apply_temperature(Pa, apply_temp)
    out = [[] for _ in range(Xv.shape[0])]
    for r, (i_loc, tok) in enumerate(meta):
        out[i_loc].append((tok, float(P[i_loc, target_idx[i_loc]] - Pa[r, target_idx[i_loc]])))
    for i in range(Xv.shape[0]):
        out[i].sort(key=lambda x: -x[1])
        out[i] = out[i][:top_k]
    return out
