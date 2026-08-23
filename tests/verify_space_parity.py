"""Prove the deployed web app and the command line return the same numbers.

Run with:  pytest tests/verify_space_parity.py -v -s

``spaces/subfinder/`` is pushed to Hugging Face as-is, so it carries its own copy of the
model bundle and of the inference modules. That independence is deliberate -- a refactor in
``src/`` cannot break the running app -- but it means the two trees can drift apart silently,
and a user who reads the paper, runs the CLI and then uses the website would get three
different answers without anything raising.

Each side is run in its own interpreter, from its own directory, so each resolves ``src``
the way it does in production: the Space ships a trimmed ``src/`` with no ``inference``
package, and importing both into one process would silently have the research tree answer
for the Space. Comparing across subprocesses is the only way the test means what it says.

Pinned here:

* the two bundles are the same bundle (temperature, classes, trees, vocabulary),
* every shared module that can move a prediction is byte-identical,
* the constants that gate a verdict agree,
* end to end, both paths return the same substrate, calibrated probability, p-value and
  significance verdict -- including on the degenerate inputs where the evidence guard,
  not the classifier, decides the answer,
* and the numbers the website advertises are the ones the paper reports.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "spaces" / "subfinder"

pytestmark = pytest.mark.skipif(
    not (SPACE / "engine.py").exists(),
    reason="the Space bundle is not present in this checkout")

# Loci chosen so each exercises a different branch: clean calls, a two-way split, a locus
# the guard withholds on, a locus with nothing readable, and labels the model does not know.
CASES = [
    ("clean alginate call", "1.B.14.12.1,GntR,PL6|PL6_1,PL17_2|PL17,2.A.1.14.25,null"),
    ("clean starch call",   "GH13_1,GH13,CBM48,SusC,1.B.14.6.1,GH31,null"),
    ("clean xylan call",    "GH10,GH43_4,CBM6,1.B.14.2.1,AraC,null,GH67"),
    ("two-way split",       "GH16,GH3,null,null"),
    ("one readable gene",   "GH13,foo_bar_baz"),
    ("no readable gene",    "foo_bar_baz,another_nonsense"),
    ("nulls only",          "null,null,null"),
    ("subfamily suffixes",  "PL17_2,GH43_4,GH13_1,CBM48"),
    ("bare TC family",      "1.B.14,2.A.1,GH92,null"),
    ("single informative",  "GH20"),
]

SHARED_MODULES = [
    "src/preprocessing/tokenizers.py",
    "src/preprocessing/cgc_loader.py",
    "src/calibration/temperature.py",
    "src/ablation/leave_one_token_out.py",
    "src/lit_validation/canon.py",
    "src/lit_validation/alias_map.py",
]

_CLI_PROBE = r"""
import json, sys, joblib
sys.path.insert(0, %(root)r)
from src.inference.predict_one import PULPredictor
b = joblib.load(%(root)r + "/artifacts/final_model_v2.pkl")
p = PULPredictor(b["pipeline"], float(b["T"]))
cases = json.loads(sys.argv[1])
out = {"meta": {"T": float(b["T"]), "classes": [str(c) for c in b["classes"]],
                "vocab": sorted(p._vocab),
                "min_informative": PULPredictor.MIN_INFORMATIVE_TOKENS,
                "not_a_gene": sorted(PULPredictor.NOT_A_GENE)},
       "loci": {}}
for name, seq in cases:
    r = p.predict(seq)
    out["loci"][name] = {
        "predicted": str(r["predicted"]),
        "confidence": float(r["confidence"]),
        "p_winner": float(r["p_value_winner_adjusted"]),
        "significant": bool(r["is_significant"]) and not bool(r["insufficient_evidence"]),
        "n_informative": int(r["n_informative_tokens"]),
        "probabilities": {str(k): float(v) for k, v in r["probabilities"].items()},
        "p_values": {str(k): float(v) for k, v in r["p_values"].items()},
        "n_trees": int(r["n_trees"]),
    }
print(json.dumps(out))
"""

_SPACE_PROBE = r"""
import json, sys
sys.path.insert(0, %(space)r)
import engine
from scipy.stats import binom
cases = json.loads(sys.argv[1])
df = engine.predict_frame([(n, s) for n, s in cases])
out = {"meta": {"T": float(engine.T), "classes": list(engine.CLASSES),
                "vocab": sorted(engine.VOCAB),
                "min_informative": int(engine.MIN_INFORMATIVE),
                "not_a_gene": sorted(engine.NOT_A_GENE),
                "alpha": float(engine.ALPHA), "n_trees": int(engine.NTREES),
                "vote_threshold": int(engine.VOTE_THRESHOLD)},
       "loci": {}}
for name, seq in cases:
    sub = df[df.locus == name]
    w = sub[sub.is_winner].iloc[0]
    out["loci"][name] = {
        "predicted": str(w.substrate),
        "confidence": float(w.probability),
        "p_winner": float(w.p_value),
        "significant": bool(w.significant),
        "n_informative": int(w.readable_genes),
        "probabilities": {str(r.substrate): float(r.probability) for _, r in sub.iterrows()},
        "p_values": {str(r.substrate): float(r.p_value) for _, r in sub.iterrows()},
        "n_rows": int(len(sub)),
        "signature_genes": {str(r.substrate): str(r.signature_genes or "")
                            for _, r in sub.iterrows()},
    }
print(json.dumps(out))
"""


def _probe(code: str, cwd: Path) -> dict:
    r = subprocess.run([sys.executable, "-c", code, json.dumps(CASES)],
                       cwd=str(cwd), capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(f"probe failed in {cwd}:\n{r.stderr[-3000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def cli():
    return _probe(_CLI_PROBE % {"root": str(ROOT)}, ROOT)


@pytest.fixture(scope="module")
def space():
    return _probe(_SPACE_PROBE % {"space": str(SPACE)}, SPACE)


# --------------------------------------------------------------------- the bundle
def test_bundle_is_byte_identical():
    """The Space ships a copy of the model. It has to be the same copy."""
    a = hashlib.sha256((ROOT / "artifacts" / "final_model_v2.pkl").read_bytes()).hexdigest()
    b = hashlib.sha256((SPACE / "final_model_v2.pkl").read_bytes()).hexdigest()
    assert a == b, (
        "spaces/subfinder/final_model_v2.pkl has drifted from artifacts/final_model_v2.pkl. "
        "Re-copy it before deploying, or the website will not match the paper.")


def test_bundle_contents_agree(cli, space):
    assert space["meta"]["T"] == cli["meta"]["T"]
    assert space["meta"]["classes"] == cli["meta"]["classes"]
    assert space["meta"]["vocab"] == cli["meta"]["vocab"]
    assert space["meta"]["n_trees"] == 500


# ------------------------------------------------------------------ shared modules
@pytest.mark.parametrize("rel", SHARED_MODULES)
def test_shared_modules_identical(rel):
    """Anything that can change a prediction must be the same file in both trees."""
    a = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    b = hashlib.sha256((SPACE / rel).read_bytes()).hexdigest()
    assert a == b, f"{rel} differs between the research tree and the deployed Space"


def test_literature_table_identical():
    a = hashlib.sha256((ROOT / "data" /
                        "Literature_Data_fam_substrate_mapping.tsv").read_bytes()).hexdigest()
    b = hashlib.sha256((SPACE /
                        "Literature_Data_fam_substrate_mapping.tsv").read_bytes()).hexdigest()
    assert a == b, "the curated literature table differs between the two trees"


# ---------------------------------------------------------------------- constants
def test_verdict_constants_agree(cli, space):
    from scipy.stats import binom
    assert space["meta"]["alpha"] == 0.05
    assert space["meta"]["min_informative"] == cli["meta"]["min_informative"]
    assert space["meta"]["not_a_gene"] == cli["meta"]["not_a_gene"]
    # 280/500: the smallest vote count clearing 0.05/12 under a fair coin
    assert space["meta"]["vote_threshold"] == int(binom.isf(0.05 / 12, 500, 0.5)) + 1


# ------------------------------------------------------------------- end to end
def test_winner_agrees(cli, space):
    """The number a user reads on the website is the number the CLI prints."""
    failures = []
    for name, _ in CASES:
        a, b = space["loci"][name], cli["loci"][name]
        for field in ("predicted", "significant", "n_informative"):
            if a[field] != b[field]:
                failures.append(f"{name}: {field} app={a[field]!r} cli={b[field]!r}")
        for field in ("confidence", "p_winner"):
            if abs(a[field] - b[field]) >= 1e-12:
                failures.append(f"{name}: {field} app={a[field]!r} cli={b[field]!r}")
    assert not failures, "the Space and the CLI disagree:\n  " + "\n  ".join(failures)


def test_all_twelve_agree(cli, space):
    """Not just the winner -- every substrate's probability and p-value."""
    worst_p = worst_pv = 0.0
    for name, _ in CASES:
        a, b = space["loci"][name], cli["loci"][name]
        assert a["n_rows"] == 12, f"{name}: expected 12 rows, got {a['n_rows']}"
        assert set(a["probabilities"]) == set(b["probabilities"])
        for sub in a["probabilities"]:
            worst_p = max(worst_p, abs(a["probabilities"][sub] - b["probabilities"][sub]))
            worst_pv = max(worst_pv, abs(a["p_values"][sub] - b["p_values"][sub]))
    assert worst_p < 1e-12, f"probability disagreement up to {worst_p:.2e}"
    assert worst_pv < 1e-12, f"p-value disagreement up to {worst_pv:.2e}"
    print(f"\n  12 substrates x {len(CASES)} loci: "
          f"max |dprob| = {worst_p:.1e}, max |dp-value| = {worst_pv:.1e}")


def test_no_null_in_signature_genes(space):
    """`null` is a real feature but names no gene, so it is never shown as a reason."""
    for name, _ in CASES:
        for cell in space["loci"][name]["signature_genes"].values():
            shown = [g.strip() for g in cell.split(",") if g.strip()]
            assert "null" not in shown, f"{name}: `null` surfaced as a signature gene"
            assert not any(g.isdigit() for g in shown), f"{name}: bare index surfaced: {cell!r}"


def test_app_headline_numbers_match_the_paper():
    """The accuracy the website advertises has to be the accuracy the paper reports."""
    import re
    numbers = ROOT / "paper" / "generated" / "numbers.tex"
    if not numbers.exists():
        pytest.skip("paper/generated/numbers.tex is not in this checkout")
    m = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", numbers.read_text()))
    app = (SPACE / "app.py").read_text()
    wanted = {k: m[k].replace("{,}", ",")
              for k in ("DepAcc", "DepStd", "NClasses", "NTrees", "NPuls", "HighConfAcc")}
    missing = {k: v for k, v in wanted.items() if v not in app}
    assert not missing, (
        f"the Space advertises numbers the paper does not support; missing {missing}")
