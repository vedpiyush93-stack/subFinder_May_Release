"""End-to-end inference on new PULs.

A ``PULPredictor`` bundles:
  * the fitted sklearn pipeline (CountVec_cpu × OvR(ExtraTrees 500, log2))
  * the temperature scalar T (one for deployment)
  * a binomial null on each substrate's forest votes, for per-substrate p-values

and exposes a single ``predict(seq)`` method that returns a dict:

    {
      "predicted":          "alginate",
      "confidence":          0.92,
      "probabilities":      {"alginate": 0.92, "alpha-glucan": 0.01, ...},
      "p_values":           {"alginate": 6.0e-5, "alpha-glucan": 0.91, ...},
      "is_significant":      True,   # p-value clears alpha AND there was enough to read
      "p_value_winner_adjusted": 6.0e-5,  # Bonferroni-corrected over the K substrates
      "vote_fractions":     {"alginate": 0.954, ...},  # yes-votes / n_trees per forest
      "n_trees":             500,
      "signature_genes":   [{"token": "PL7",  "delta": +0.18, "is_lit_canonical": True},
                              {"token": "PL17", "delta": +0.11, "is_lit_canonical": True},
                              ...],
      "oov_proportion":     0.0,     # fraction of PUL tokens NOT in train vocab
      "refuse_to_predict":  False,   # True iff oov_proportion > 0.10
      "n_informative_tokens": 7,     # in-vocabulary, non-null: what the model could read
      "insufficient_evidence": False # True iff n_informative_tokens < MIN_INFORMATIVE_TOKENS
    }

The "refuse" flag is purely INFORMATIONAL — a "needs manual review" caveat.
The inference pipeline runs identically regardless of OOV: substrate +
calibrated probabilities + p-values + signature genes are always returned.
The flag and ``oov_proportion`` are just two extra fields that downstream
tooling can use to decide whether to trust the result.

``is_significant`` is the one field that IS gated. It requires both a p-value
below alpha and at least ``MIN_INFORMATIVE_TOKENS`` readable genes, because the
p-value measures whether the probability vector is peaked and cannot measure
whether anything was read to peak it — see ``has_enough_evidence``. When the
verdict is withheld, ``insufficient_evidence`` says so and every other field,
including the p-value itself, is reported unchanged.
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


def vote_p_value(vote_fraction: float, n_trees: int, alpha_null: float = 0.5) -> float:
    """Binomial p-value for ONE substrate, from the votes that produced it.

    The classifier is one-vs-rest: each substrate has its own forest of
    ``n_trees`` trees, and each tree votes yes or no on that substrate alone.
    The test statistic is therefore a count of yes-votes out of a known number of
    trials, and the natural null is a forest with nothing to go on, whose trees
    split evenly. Class weights are balanced during training, so an uninformed
    tree has no majority class to fall back on and 0.5 is the right coin.

    p = P(at least this many yes-votes | fair coin), the upper tail of
    Binomial(n_trees, 0.5).

    Reading the votes directly rather than the normalised twelve-vector matters,
    because normalisation invents confidence that the votes do not contain. A
    locus whose genes are all unannotated draws almost no yes-votes anywhere,
    yet dividing those twelve near-zero values by their sum still yields a
    leading "probability" of 0.54. On the normalised vector that locus is
    significant; on the votes it is not, which is the honest answer.
    """
    from scipy.stats import binom
    k = int(round(float(vote_fraction) * n_trees))
    return float(binom.sf(k - 1, n_trees, alpha_null))


def winner_is_significant(p_value: float, K: int = 12, alpha: float = 0.05) -> bool:
    """Does the top substrate's vote count beat a fair coin, after correcting for K tests?

    One test is run per substrate, so the threshold is Bonferroni-corrected:
    reject when p < alpha/K. For K=12, alpha=0.05 and 500 trees this means a
    substrate needs at least 280 of its 500 trees voting yes.
    """
    return bool(p_value < alpha / K)


def has_enough_evidence(n_informative: int, minimum: int = 2) -> bool:
    """Is there enough in-vocabulary evidence for the significance test to mean anything?

    ``winner_is_significant`` asks whether the probability vector is more peaked
    than a random split of 1. It never asks whether anything was read to produce
    that peak, and an ExtraTrees ensemble answers a near-empty input with its
    training prior rather than with a flat vector. Unguarded, an empty PUL is
    reported as a significant "host glycan" hit at p = 2.6e-3, and a single GH20
    scores p = 4.1e-102 -- numerically identical to a complete nine-gene PUL.

    The cut is set at two informative tokens from the pooled 5x5 RSKF
    out-of-fold predictions (5,150 predictions), where the model's stated
    confidence is compared against the accuracy it actually achieves:

        informative tokens |    n | accuracy | stated conf |     gap
        -------------------|------|----------|-------------|--------
                         1 |   78 |   0.6154 |      0.8318 | -0.2164
                         2 |  321 |   0.8785 |      0.8974 | -0.0189
                         3 |  516 |   0.8876 |      0.8851 | +0.0025
                       >=6 | 3160 |   0.9358 |      0.9223 | +0.0135

    At one token the overconfidence is decisive -- bootstrap 95% CI on the gap
    is [-0.335, -0.098], excluding zero in 8,000 of 8,000 resamples. At two it
    is [-0.059, +0.020], which straddles zero: calibrated. The cliff sits
    between 1 and 2, so 2 is the threshold.

    Requiring 3 instead would suppress 7.7% of labelled loci that are still
    82.7% accurate -- discarding usable calls to buy nothing.

    Cost of the guard: 1.5% of labelled loci (accuracy 0.615, versus 0.918
    overall) and 14.1% of the 359,763-locus unsupervised corpus.

    "Informative" means non-``null`` and in the training vocabulary. ``null``
    is deliberately excluded: it is a padding token present in the vocabulary,
    so counting it would let a PUL of ten nulls pass this guard.
    """
    return bool(n_informative >= minimum)


class PULPredictor:
    """End-to-end inference for one PUL."""

    OOV_REFUSE_THRESHOLD = 0.10
    MIN_INFORMATIVE_TOKENS = 2
    # features the model uses but that name no gene, so they are never
    # reported as a reason for a call
    NOT_A_GENE = frozenset({"null", ""})

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

    def n_informative_tokens(self, seq: str) -> int:
        """Count tokens the model can actually read: in-vocabulary and not ``null``."""
        return sum(1 for t in self._tokenize(seq)
                   if t != "null" and t in self._vocab)

    def predict(self, seq: str, top_k: int = 5) -> dict:
        oov = self.oov_proportion(seq)
        refuse = oov > self.OOV_REFUSE_THRESHOLD
        n_inf = self.n_informative_tokens(seq)
        enough = has_enough_evidence(n_inf, self.MIN_INFORMATIVE_TOKENS)

        # Probabilities (calibrated)
        p_raw = self.pipeline.predict_proba([seq])
        p_cal = apply_temperature(p_raw, self.T)[0]
        argmax_idx = int(p_cal.argmax())
        predicted = self.classes_[argmax_idx]
        confidence = float(p_cal[argmax_idx])

        probs_dict = {c: float(p_cal[i]) for i, c in enumerate(self.classes_)}

        # p-values are computed from the VOTES, not from the normalised vector:
        # each substrate's forest votes yes or no on that substrate alone, so the
        # statistic is a count out of a known number of trees.
        Z = self.pipeline.named_steps["cv"].transform([seq])
        ests = self.pipeline.named_steps["vr"].estimators_
        votes = {c: float(ests[i].predict_proba(Z)[0, 1])
                 for i, c in enumerate(self.classes_)}
        n_trees = int(ests[0].n_estimators)
        pvals = {c: vote_p_value(votes[c], n_trees) for c in self.classes_}

        # Signature genes via leave-one-token-out ablation against the PREDICTED class,
        # on the CALIBRATED probabilities (matches the deployment story).
        # We run this regardless of OOV — the refuse flag is purely a caveat for the
        # caller, never a gate on the inference itself.
        sig = []
        try:
            # Ask for more than we return, because the two tokens filtered below
            # can occupy top slots; without the margin a caller asking for three
            # genes could receive one.
            deltas = ablate_pul_for_class(self.pipeline, seq, predicted,
                                          top_k=top_k + len(self.NOT_A_GENE) + 4,
                                          apply_temp=self.T)
            for tok, d in deltas:
                # `null` and a bare subfamily index are real features -- both are
                # in the vocabulary and both move the probability -- but neither
                # names a gene, so neither is reported as a reason. The ablation
                # still scores them; only this list is filtered.
                if tok in self.NOT_A_GENE or tok.isdigit():
                    continue
                is_lit = (self.canon is not None
                          and is_cazy(tok)
                          and tok in self.canon.get(predicted, set()))
                sig.append({"token": tok, "delta": float(d),
                            "is_lit_canonical": bool(is_lit)})
                if len(sig) == top_k:
                    break
        except Exception as e:
            sig = [{"error": str(e)}]

        return {
            "predicted":         predicted,
            "confidence":        confidence,
            "probabilities":     probs_dict,
            "p_values":          pvals,
            # The verdict needs BOTH a peaked output and something to have read.
            # The statistic below is always reported unchanged; only the verdict
            # is gated, so a caller can still see why it was withheld.
            "is_significant":    (winner_is_significant(pvals[predicted],
                                                         K=len(self.classes_))
                                  and enough),
            "p_value_winner_adjusted": min(1.0, len(self.classes_) * pvals[predicted]),
            "vote_fractions":    votes,
            "n_trees":           n_trees,
            "n_informative_tokens": n_inf,
            "insufficient_evidence": not enough,
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
