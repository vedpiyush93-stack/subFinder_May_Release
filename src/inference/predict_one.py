"""End-to-end inference on new PULs.

A ``PULPredictor`` bundles:
  * the fitted sklearn pipeline (CountVec_cpu × OvR(ExtraTrees 500, log2))
  * the temperature scalar T (one for deployment)
  * a Dirichlet-uniform null for per-substrate p-values

and exposes a single ``predict(seq)`` method that returns a dict:

    {
      "predicted":          "alginate",
      "confidence":          0.92,
      "probabilities":      {"alginate": 0.92, "alpha-glucan": 0.01, ...},
      "p_values":           {"alginate": 6.0e-5, "alpha-glucan": 0.91, ...},
      "is_significant":      True,
      "signature_genes":   [{"token": "PL7",  "delta": +0.18, "is_lit_canonical": True},
                              {"token": "PL17", "delta": +0.11, "is_lit_canonical": True},
                              ...],
      "oov_proportion":     0.0,     # fraction of PUL tokens NOT in train vocab
      "refuse_to_predict":  False    # True iff oov_proportion > 0.10
    }

The "refuse" flag is purely INFORMATIONAL — a "needs manual review" caveat.
The inference pipeline runs identically regardless of OOV: substrate +
calibrated probabilities + p-values + signature genes are always returned.
The flag and ``oov_proportion`` are just two extra fields that downstream
tooling can use to decide whether to trust the result.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
import numpy as np

from src.preprocessing.tokenizers import tok_cpu, tok_cpu_v2, is_cazy
from src.calibration.temperature import apply_temperature
from src.ablation.leave_one_token_out import ablate_pul_for_class
from src.lit_validation.canon import build_canon


def p_value_dirichlet_uniform(p: float, K: int = 12) -> float:
    """p-value under a uniform-Dirichlet null hypothesis over K classes."""
    return float((1.0 - p) ** (K - 1))


class PULPredictor:
    """End-to-end inference for one PUL."""

    OOV_REFUSE_THRESHOLD = 0.10

    def __init__(self, pipeline, T: float, lit_tsv_path: str | None = None):
        self.pipeline = pipeline
        self.T = float(T)
        self.classes_ = list(pipeline.named_steps["vr"].classes_)
        self._vocab = set(pipeline.named_steps["cv"].vocabulary_.keys())
        self.canon = build_canon(lit_tsv_path) if lit_tsv_path else None
        # Use whichever tokenizer the CountVectorizer was fit with (v2 bundles
        # are fit with tok_cpu_v2; original bundles use tok_cpu). Fallback to
        # tok_cpu for older pickles that don't carry a tokenizer attribute.
        self._tokenize = pipeline.named_steps["cv"].tokenizer or tok_cpu

    def oov_proportion(self, seq: str) -> float:
        toks = self._tokenize(seq)
        if not toks: return 0.0
        return sum(1 for t in toks if t not in self._vocab) / len(toks)

    def predict(self, seq: str, top_k: int = 5) -> dict:
        oov = self.oov_proportion(seq)
        refuse = oov > self.OOV_REFUSE_THRESHOLD

        # Probabilities (calibrated)
        p_raw = self.pipeline.predict_proba([seq])
        p_cal = apply_temperature(p_raw, self.T)[0]
        argmax_idx = int(p_cal.argmax())
        predicted = self.classes_[argmax_idx]
        confidence = float(p_cal[argmax_idx])

        probs_dict = {c: float(p_cal[i]) for i, c in enumerate(self.classes_)}
        pvals = {c: p_value_dirichlet_uniform(probs_dict[c]) for c in self.classes_}

        # Signature genes via leave-one-token-out ablation against the PREDICTED class,
        # on the CALIBRATED probabilities (matches the deployment story).
        # We run this regardless of OOV — the refuse flag is purely a caveat for the
        # caller, never a gate on the inference itself.
        sig = []
        try:
            deltas = ablate_pul_for_class(self.pipeline, seq, predicted,
                                          top_k=top_k, apply_temp=self.T)
            for tok, d in deltas:
                is_lit = (self.canon is not None
                          and is_cazy(tok)
                          and tok in self.canon.get(predicted, set()))
                sig.append({"token": tok, "delta": float(d),
                            "is_lit_canonical": bool(is_lit)})
        except Exception as e:
            sig = [{"error": str(e)}]

        return {
            "predicted":         predicted,
            "confidence":        confidence,
            "probabilities":     probs_dict,
            "p_values":          pvals,
            "is_significant":    pvals[predicted] < 0.05,
            "signature_genes":   sig,
            "oov_proportion":    oov,
            "refuse_to_predict": refuse,
        }


def load_predictor(model_pkl_path: str | Path,
                   lit_tsv_path: str | Path | None = None) -> PULPredictor:
    """Load a ``PULPredictor`` from a single .pkl file.

    The deployed final_model.pkl is a flat dict of components:
      {"classifier", "vectorizer", "label_encoder", "temperature",
       "class_names", ...}
    We wrap it back into an sklearn Pipeline with the named steps
    ``cv`` (CountVectorizer) and ``vr`` (OvR(ExtraTrees)) so ``PULPredictor``
    can treat it uniformly.
    """
    # The deployed pickles were saved from scripts where ``tok_cpu`` (or the
    # v2 refinement ``tok_cpu_v2``) lived in ``__main__``. Inject both so
    # unpickling resolves the reference regardless of which script (or notebook)
    # is loading the model. We use joblib.load so it also transparently
    # decompresses the xz-compressed v2 bundles.
    import sys, joblib
    import numpy as np
    from sklearn.pipeline import Pipeline
    sys.modules['__main__'].tok_cpu    = tok_cpu
    sys.modules['__main__'].tok_cpu_v2 = tok_cpu_v2
    obj = joblib.load(model_pkl_path)
    # Support both shapes: the deployed flat-dict and the older Pipeline form.
    if "pipeline" in obj and "T" in obj:
        pipeline, T = obj["pipeline"], obj["T"]
    else:
        pipeline = Pipeline([("cv", obj["vectorizer"]), ("vr", obj["classifier"])])
        T = float(obj["temperature"])
        # The classifier was trained on label-encoded ints (0..11); the deployed
        # pickle carries the human-readable substrate names separately. Re-label
        # the classes_ attribute so downstream code (and predict_proba output)
        # speaks substrate strings, not int64s.
        if "class_names" in obj:
            pipeline.named_steps["vr"].classes_ = np.array(list(obj["class_names"]))
    return PULPredictor(pipeline, T, str(lit_tsv_path) if lit_tsv_path else None)
