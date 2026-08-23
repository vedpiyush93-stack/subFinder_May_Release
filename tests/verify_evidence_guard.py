"""Prove a baseless prediction is never reported as significant.

Run with:  pytest tests/verify_evidence_guard.py -v -s

The Bonferroni-corrected p-value asks whether the twelve probabilities are more
peaked than a random split of 1. It cannot ask whether anything was read to
produce that peak, and an ExtraTrees ensemble answers a near-empty input with its
training prior, not with a flat vector. Before the evidence guard, an empty PUL
was reported as a significant "host glycan" hit at p = 2.6e-3, and a single GH20
scored p = 4.1e-102 -- numerically identical to a complete nine-gene PUL.

These tests pin the two halves of the guard: that the degenerate inputs are
suppressed, and that the threshold sits where the calibration evidence puts it
(the model is decisively overconfident on one informative token and calibrated on
two). They also pin that the p-value statistic is still reported when the verdict
is withheld, so a caller can see why.
"""
from __future__ import annotations
import sys
from pathlib import Path
import joblib, pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.inference.predict_one import PULPredictor, has_enough_evidence

REAL_PUL = ("1.B.14.12.1,GntR,PL6|PL6_1,PL17_2|PL17,AraC_binding,"
            "2.A.1.14.25,null,PfkB,null")


@pytest.fixture(scope="module")
def predictor():
    b = joblib.load(ROOT / "artifacts" / "final_model_v2.pkl")
    return PULPredictor(b["pipeline"], float(b["T"]))


@pytest.mark.parametrize("label,seq", [
    ("empty string",       ""),
    ("padding only",       ",".join(["null"] * 10)),
    ("all genes unknown",  "ZZZ1,ZZZ2,ZZZ3"),
    ("a single gene",      "GH20"),
])
def test_baseless_input_is_never_significant(predictor, label, seq):
    r = predictor.predict(seq)
    assert r["insufficient_evidence"], f"{label}: guard did not fire"
    assert not r["is_significant"], (
        f"{label}: reported significant on {r['n_informative_tokens']} "
        f"informative tokens (p = {r['p_value_winner_adjusted']:.2e})")


def test_the_statistic_survives_even_when_the_verdict_is_withheld(predictor):
    """Suppressing the verdict must not hide the number behind it."""
    r = predictor.predict("GH20")
    assert r["p_value_winner_adjusted"] < 0.05, "p-value should still be reported"
    assert r["n_informative_tokens"] == 1
    assert not r["is_significant"]


def test_a_real_pul_is_unaffected(predictor):
    r = predictor.predict(REAL_PUL)
    assert not r["insufficient_evidence"]
    assert r["is_significant"]
    assert r["n_informative_tokens"] >= 2


def test_null_is_not_counted_as_evidence(predictor):
    """``null`` is in the vocabulary, so it must be excluded explicitly."""
    assert predictor.n_informative_tokens(",".join(["null"] * 50)) == 0


@pytest.mark.parametrize("seq", [REAL_PUL, "GH20,1.B.14", "", "ZZZ1,ZZZ2"])
def test_served_probabilities_stay_normalised(predictor, seq):
    """The p-value reads raw votes; the probabilities a user sees must still sum to 1."""
    r = predictor.predict(seq)
    assert sum(r["probabilities"].values()) == pytest.approx(1.0, abs=1e-9)


def test_vote_fractions_are_not_normalised(predictor):
    """They are the pre-normalisation quantity, which is the point of reading them.

    Asserted on a locus the model is unsure about: when every forest is decisive
    the twelve fractions can happen to sum to 1, so a confident PUL proves nothing.
    """
    r = predictor.predict("")
    assert sum(r["vote_fractions"].values()) != pytest.approx(1.0, abs=1e-3)


def test_pvalue_describes_the_substrate_its_row_names(predictor):
    """Normalisation and temperature are monotone, so the two rankings cannot diverge.

    If they could, the p-value printed beside a substrate would belong to a
    different one.
    """
    for seq in (REAL_PUL, "GH20,1.B.14", "GH13,CBM20,2.A.1", "PL6,PL17,1.B.14,GntR"):
        r = predictor.predict(seq)
        by_prob = max(r["probabilities"], key=r["probabilities"].get)
        by_vote = max(r["vote_fractions"], key=r["vote_fractions"].get)
        assert by_prob == by_vote == r["predicted"], seq


def test_null_is_used_by_the_model_but_never_reported_as_a_gene(predictor):
    """`null` is a feature, not a gene.

    It is 30% of the training tokens, it is in the vocabulary, and removing it
    moves the probability -- so it must keep participating in the model. What it
    must never do is appear in a list offered to a user as the reason for a call,
    because it names nothing they can look up.
    """
    vocab = predictor.pipeline.named_steps["cv"].vocabulary_
    assert "null" in vocab, "null must remain a feature the model was fitted on"

    with_null = "1.B.14.12.1,GntR,PL6|PL6_1,PL17_2|PL17,2.A.1.14.25,null,null"
    r = predictor.predict(with_null, top_k=3)
    tokens = [g["token"] for g in r["signature_genes"]]
    assert tokens, "a real PUL should still yield signature genes"
    assert "null" not in tokens, f"null was reported as a reason: {tokens}"
    assert not any(t.isdigit() for t in tokens), f"bare index reported: {tokens}"

    # and it genuinely still influences the prediction
    import numpy as np
    from src.calibration.temperature import apply_temperature
    a = apply_temperature(predictor.pipeline.predict_proba([with_null]), predictor.T)
    b = apply_temperature(
        predictor.pipeline.predict_proba([with_null.replace(",null", "")]), predictor.T)
    assert np.abs(a - b).max() > 0, "removing null should change the probabilities"


def test_threshold_is_two(predictor):
    """One informative token is where the model is overconfident; two is calibrated."""
    assert not has_enough_evidence(1)
    assert has_enough_evidence(2)
    assert predictor.MIN_INFORMATIVE_TOKENS == 2
