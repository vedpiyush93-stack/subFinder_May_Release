#!/usr/bin/env python3
"""Check that the numbers a reader sees in paper/main.pdf match the run artifacts.

The manuscript no longer contains literal numbers: they are \\newcommand macros
written by scripts/07c_build_paper_figures.py and expanded at typeset time. So
the meaningful check is on the compiled PDF rather than the source -- it verifies
both that the artifacts are right and that the macros expanded as intended.

Each value below is re-derived from artifacts/ and paper/audit_output.txt and
must appear in the rendered text. Exits non-zero on any mismatch.

    python3 scripts/07b_verify_paper_numbers.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT/"paper/main.pdf"
if not PDF.exists():
    sys.exit("paper/main.pdf not built yet")

try:
    import fitz
except ImportError:
    sys.exit("pymupdf required: pip install pymupdf")

doc = fitz.open(PDF)
TEXT = "".join(doc[i].get_text() for i in range(doc.page_count))
TEXT = TEXT.replace("−", "-").replace(" ", "").replace("\xa0", " ")

audit = dict(l.split("\t", 1) for l in (ROOT/"paper/audit_output.txt").read_text().splitlines()
             if "\t" in l and not l.startswith("#"))
lb   = pd.read_csv(ROOT/"artifacts/leaderboard.csv")
pfm  = pd.read_csv(ROOT/"artifacts/per_fold_metrics.csv")
cal  = pd.read_csv(ROOT/"artifacts/calibration_report.csv")
per  = pd.read_csv(ROOT/"paper/tables/table_per_substrate.csv")
fun  = pd.read_csv(ROOT/"paper/tables/table_sig_funnel.csv")

def acc(cfg):
    r = lb[lb.shorthand == cfg].iloc[0]; return float(r.mean_acc), float(r.std_acc)

checks, fails = [], 0
def check(label, needle):
    global fails
    ok = str(needle) in TEXT
    checks.append((label, str(needle), ok))
    if not ok: fails += 1

dep_a, dep_s = acc("cpuV2__ET500_log2")
brf_a, _     = acc("cv__BRF100")
dl = lb[lb.shorthand.str.contains("__LSTM|__Trans|__JustAttn")].iloc[0]

check("deployed accuracy",       f"{dep_a:.4f}")
check("deployed std",            f"{dep_s:.4f}")
check("baseline accuracy",       f"{brf_a:.4f}")
check("best deep accuracy",      f"{dl.mean_acc:.4f}")
check("gap vs best deep",        f"{(dep_a-dl.mean_acc)*100:.2f}")
check("gap vs baseline",         f"{(dep_a-brf_a)*100:.2f}")
check("held-out pass accuracy",  audit[[k for k in audit if k.startswith("oof_seed")][0]])
check("n configurations",        pfm.shorthand.nunique())
check("n runs",                  pfm.shorthand.nunique()*25)
check("n labelled loci",         f"{len(pd.read_csv(ROOT/'data/Train_data.csv')):,}")
check("signature-gene hits",     int(fun.hit.sum()))
check("signature-gene eligible", int(fun.eligible.sum()))
check("signature-gene rate",     f"{fun.hit.sum()/fun.eligible.sum()*100:.1f}")
check("gene-view flagged",       int(fun.flagged.sum()))
check("gene-view in-scope",      int(fun.in_scope.sum()))
check("lit canon pairs",         audit["lit_db_substrate_family_pairs_after_alias_collapse"])
for _, r in cal.iterrows():
    if r.method in ("uncalibrated", "temperature_scaling", "isotonic_cv5 (sklearn)"):
        check(f"ECE {r.method.split('_cv')[0]}", f"{r.ece_10bin:.3f}")
w = per.iloc[-1]
check("weakest substrate F1",    f"{w.f1:.2f}")
check("weakest substrate recall", f"{w.recall:.2f}")

print(f"{'check':30s} {'value in artifacts':22s} in PDF?")
print("-" * 66)
for label, needle, ok in checks:
    print(f"  {label:28s} {needle:22s} {'yes' if ok else 'NOT FOUND'}")
print("-" * 66)
if fails:
    print(f"{fails} of {len(checks)} values are missing from the rendered manuscript")
    sys.exit(1)
print(f"all {len(checks)} checked values appear in paper/main.pdf as rendered")
