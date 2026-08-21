"""Prove the two input formats produce identical predictions.

Run with:  pytest tests/verify_input_formats_agree.py -v -s

The tool accepts a PUL either as the token string used throughout training, or
as a dbCAN CGC-finder output table. Those are different files describing the same
locus, so they must yield the same feature vector and therefore the same
probabilities, p-values and signature genes. Nothing enforces that automatically:
the CGC reader applies its own per-gene rules, and a mismatch there would show up
only as quietly different predictions.

This test round-trips real labelled loci: it renders each one as a CGC table,
reads it back through the production reader, and compares against the original
string at three levels -- tokens, feature counts, and final probabilities.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocessing.cgc_loader import cgc_to_pul_strings
from src.preprocessing.tokenizers import tok_cpu_v2

N_LOCI = 60
CAZY = ("GH", "PL", "CE", "CBM", "GT", "AA")


def _gene_type(tok: str) -> str:
    """Classify a token the way dbCAN's CGC table labels the gene it came from."""
    if tok == "null":                                   return "null"
    if tok.startswith(CAZY) and any(c.isdigit() for c in tok): return "CAZyme"
    head = tok.split(".")
    if len(head) >= 2 and head[0].isdigit() and head[1][:1].isalpha(): return "TC"
    return "TF"


def _write_cgc(puls: list[str], path: Path) -> None:
    """Render token strings as a dbCAN CGC-finder table, one row per gene."""
    rows = ["\t".join(["CGC#", "Gene Type", "Contig ID", "Protein ID",
                       "Gene Start", "Gene Stop", "Direction", "Protein Family"])]
    for i, pul in enumerate(puls):
        for j, gene in enumerate(str(pul).split(",")):
            gene = gene.strip()
            if not gene: continue
            gt = _gene_type(gene.split("|")[0])
            # a CGC table stores multi-domain genes with '+', which the reader
            # converts back to '|' for TF/STP rows
            fam = gene.replace("|", "+") if gt in ("TF", "STP") else gene
            rows.append("\t".join([f"CGC{i+1}", gt, f"contig{i+1}",
                                   f"prot{i+1}_{j}", "1", "2", "+", fam]))
    path.write_text("\n".join(rows) + "\n")


@pytest.fixture(scope="module")
def loci():
    df = pd.read_csv(ROOT/"data/Train_data.csv")
    return [s for s in df["sig_gene_seq"].fillna("").values[:N_LOCI] if s.strip()]


@pytest.fixture(scope="module")
def predictor():
    import joblib
    m = joblib.load(ROOT/"artifacts/final_model_v2.pkl")
    pipe, T, cls = m["pipeline"], m["T"], [str(c) for c in m["classes"]]
    def f(seq):
        p = pipe.predict_proba([seq])[0]
        z = np.log(np.clip(p, 1e-12, None))/T
        e = np.exp(z - z.max())
        return e/e.sum(), cls
    return f


@pytest.mark.parametrize("tc_mode", ["full", "truncate", "both"])
def test_tokens_match(loci, tmp_path, tc_mode):
    """Each locus must tokenize identically whichever file it arrived in."""
    p = tmp_path/f"cgc_{tc_mode}.out"; _write_cgc(loci, p)
    back = cgc_to_pul_strings(p, tc_mode=tc_mode)
    mismatch = []
    for i, original in enumerate(loci):
        got = back[f"contig{i+1}|CGC{i+1}"]
        a, b = tok_cpu_v2(original), tok_cpu_v2(got)
        if sorted(a) != sorted(b):
            mismatch.append((i, sorted(a), sorted(b)))
    print(f"\n  tc_mode={tc_mode!r}: {len(loci)-len(mismatch)}/{len(loci)} loci tokenize identically")
    if mismatch:
        i, a, b = mismatch[0]
        extra = [t for t in b if b.count(t) > a.count(t)]
        print(f"    first mismatch (locus {i}): CGC path has extra {sorted(set(extra))}")
    if tc_mode == "both":
        pytest.xfail("tc_mode='both' emits the full and truncated TC forms, which "
                     "tok_cpu_v2 collapses to the same token -- double-counting it")
    assert not mismatch, f"{len(mismatch)} loci tokenize differently under tc_mode={tc_mode!r}"


@pytest.mark.parametrize("tc_mode", ["full", "truncate"])
def test_predictions_match(loci, tmp_path, predictor, tc_mode):
    """Identical tokens must give identical probabilities, to machine precision."""
    p = tmp_path/f"cgc_pred_{tc_mode}.out"; _write_cgc(loci, p)
    back = cgc_to_pul_strings(p, tc_mode=tc_mode)
    worst = 0.0
    for i, original in enumerate(loci):
        pa, _ = predictor(original)
        pb, _ = predictor(back[f"contig{i+1}|CGC{i+1}"])
        worst = max(worst, float(np.abs(pa - pb).max()))
    print(f"  tc_mode={tc_mode!r}: max |probability difference| over {len(loci)} loci = {worst:.2e}")
    assert worst == 0.0, f"probabilities differ by up to {worst}"


def test_default_mode_is_safe_for_the_deployed_tokenizer(loci, tmp_path, predictor):
    """The reader's default must not change a prediction relative to the string path."""
    import inspect
    default = inspect.signature(cgc_to_pul_strings).parameters["tc_mode"].default
    p = tmp_path/"cgc_default.out"; _write_cgc(loci, p)
    back = cgc_to_pul_strings(p)
    worst = max(float(np.abs(predictor(o)[0] - predictor(back[f"contig{i+1}|CGC{i+1}"])[0]).max())
                for i, o in enumerate(loci))
    print(f"  default tc_mode={default!r}: max |probability difference| = {worst:.2e}")
    assert worst == 0.0, (
        f"the default tc_mode={default!r} changes predictions by up to {worst:.3f} "
        "relative to the token-string path")
