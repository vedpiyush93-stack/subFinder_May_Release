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
import re
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# Both documents are checked together: some values (calibration detail, top-K,
# vocabulary behaviour) live in the supplement by design, and a value present in
# either is a value the reader can find.
PDFS = [ROOT/"paper/main.pdf", ROOT/"paper/supplement.pdf"]
missing = [q for q in PDFS if not q.exists()]
if missing:
    sys.exit(f"not built yet: {', '.join(str(q.name) for q in missing)}")

try:
    import fitz
except ImportError:
    sys.exit("pymupdf required: pip install pymupdf")

TEXT = ""
for q in PDFS:
    doc = fitz.open(q)
    TEXT += "".join(doc[i].get_text() for i in range(doc.page_count))
# Normalise away what the PDF layer adds or drops: line breaks inside a value,
# the various space characters, and the underscore, which pdf text extraction
# does not preserve (\texttt{PL6\_1} comes back as "PL61").
TEXT = re.sub(r"[\s\u00a0]+", "", TEXT.replace("\u2212", "-")).replace("_", "")

# The .tex sources, for the second half of the macro check. Searching the PDF for
# a bare value is not enough on its own: short numbers collide with page ranges in
# the bibliography and with digits inside larger figures -- "78" once passed on the
# 78 inside "1,678,991" while the macro was not used in the manuscript at all. A
# macro therefore has to be BOTH wired into the source AND expanded in the PDF.
SRC = "".join(q.read_text() for q in (ROOT/"paper/main.tex", ROOT/"paper/supplement.tex"))

audit = dict(l.split("\t", 1) for l in (ROOT/"paper/audit_output.txt").read_text().splitlines()
             if "\t" in l and not l.startswith("#"))
lb   = pd.read_csv(ROOT/"artifacts/leaderboard.csv")
pfm  = pd.read_csv(ROOT/"artifacts/per_fold_metrics.csv")
cal  = pd.read_csv(ROOT/"artifacts/calibration_report.csv")
per  = pd.read_csv(ROOT/"paper/tables/table_per_substrate.csv")
fun  = pd.read_csv(ROOT/"paper/tables/table_sig_funnel.csv")

def acc(cfg):
    r = lb[lb.shorthand == cfg].iloc[0]; return float(r.mean_acc), float(r.std_acc)

def _norm(v):
    """Reduce a value to what a reader actually sees, so both sides compare alike."""
    v = str(v).replace("{,}", ",").replace("\\texttt", "").replace("$|$", "|")
    # maths that the PDF renders as glyphs rather than as the source spells it
    v = v.replace("\\times", "\u00d7").replace("^", "")
    v = v.replace("{", "").replace("}", "").replace("\\_", "").replace("_", "")
    return re.sub(r"[\s\u00a0]+", "", v)

gen = (ROOT/"paper/generated/numbers.tex").read_text()
def _parse_macros(text):
    """Parse \\newcommand bodies with balanced braces.

    A naive [^}]* stops at the first closing brace, which truncates any value
    containing a group -- the digit-grouping macro \\NPuls is "1{,}030", and a
    naive parse reads it as "1{,". Every such value then fails to be found in the
    PDF for a reason that has nothing to do with the PDF.
    """
    out = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{", text):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            if text[i] == "{": depth += 1
            elif text[i] == "}": depth -= 1
            i += 1
        out[m.group(1)] = text[m.end():i-1]
    return out

M = _parse_macros(gen)

checks, fails = [], 0
def check(label, needle):
    global fails
    ok = _norm(needle) in TEXT
    checks.append((label, str(needle), "yes" if ok else "NOT FOUND"))
    if not ok: fails += 1

def check_macro(name, value):
    """A macro counts as verified only if the manuscript uses it AND the PDF shows it."""
    global fails
    used  = re.search(r"\\" + name + r"(?![A-Za-z])", SRC) is not None
    shown = _norm(value) in TEXT
    ok = used and shown
    verdict = "yes" if ok else ("NOT USED IN TEX" if shown else "NOT FOUND")
    checks.append((f"macro {name}", str(value), verdict))
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
# The manuscript no longer quotes a single-pass accuracy: every rate is a mean
# over the 25 fits. What must hold instead is that the three places the headline
# accuracy is derived independently all agree.
_lead = float(lb[lb.shorthand == "cpuV2__ET500_log2"].iloc[0].mean_acc)
_calib = float(cal[cal.method == "uncalibrated"].accuracy.iloc[0])
_macro = float(M["HeldOutAcc"])
_agree = abs(_lead - _calib) < 5e-5 and abs(_lead - _macro) < 5e-5
checks.append(("headline accuracy agrees across leaderboard/calibration/macro",
               f"{_lead:.4f}", "yes" if _agree else "MISMATCH"))
if not _agree: fails += 1
check("n configurations",        pfm.shorthand.nunique())
check("n runs",                  pfm.shorthand.nunique()*25)
check("n labelled loci",         f"{len(pd.read_csv(ROOT/'data/Train_data.csv')):,}")
check("signature-gene hits",     int(fun.hit.sum()))
check("signature-gene eligible", int(fun.eligible.sum()))
check("signature-gene rate",     f"{fun.hit.sum()/fun.eligible.sum()*100:.1f}")
check("gene-view flagged",       int(fun.flagged.sum()))
check("gene-view in-scope",      int(fun.in_scope.sum()))
check("lit canon pairs",         audit["lit_db_substrate_family_pairs_after_alias_collapse"])
# the calibration table renders four decimals; the prose quotes three
# The manuscript quotes the calibration numbers in prose at three decimals rather
# than in a table, so what must hold is that each macro equals the run artifact
# rounded -- the exhaustive macro sweep then confirms it reached the page.
for method, mac in (("uncalibrated", "EceRaw"), ("temperature_scaling", "EceTemp"),
                    ("isotonic_cv5 (sklearn)", "EceIso")):
    src_val = float(cal[cal.method == method].ece_10bin.iloc[0])
    ok = abs(float(M[mac]) - round(src_val, 3)) < 5e-4
    checks.append((f"ECE {method.split('_cv')[0]} matches the run artifact",
                   f"{src_val:.4f} -> {M[mac]}", "yes" if ok else "MISMATCH"))
    if not ok: fails += 1
w = per.iloc[-1]
check("weakest substrate F1",    f"{w.f1:.2f}")
check("weakest substrate recall", f"{w.recall:.2f}")
topk = pd.read_csv(ROOT/"paper/tables/table_per_substrate.csv")  # presence check only
# Every macro the manuscript actually uses must expand to its generated value in
# the rendered PDF. Sweeping all of them, rather than a hand-written list, means
# the check follows the text: macros dropped during an edit stop being checked
# automatically, and macros newly used start being checked without anyone
# remembering to add them.
_skip_ci = {"EvOneCI", "EvTwoCI"}          # rendered with LaTeX spacing; handled below
for k in sorted(M):
    if not M[k] or k in _skip_ci: continue
    if re.search(r"\\" + k + r"(?![A-Za-z])", SRC):
        check_macro(k, M[k])

# bootstrap intervals print with a thin space after the comma, so compare the ends
for k in sorted(_skip_ci):
    if M.get(k) and re.search(r"\\" + k + r"(?![A-Za-z])", SRC):
        lo, hi = M[k].strip("[]").split(",")
        check_macro(k, lo.strip())
        check(f"macro {k} upper bound", hi.strip())

unused = sorted(k for k, v in M.items()
                if v and not re.search(r"\\" + k + r"(?![A-Za-z])", SRC))
if unused:
    print(f"note: {len(unused)} generated macros are used by neither document")
    print("      " + ", ".join(unused) + "\n")

print(f"{'check':30s} {'value in artifacts':22s} in either PDF?")
print("-" * 66)
for label, needle, verdict in checks:
    print(f"  {label:28s} {needle:22s} {verdict}")
print("-" * 66)
if fails:
    print(f"{fails} of {len(checks)} values are missing from the rendered documents")
    sys.exit(1)
print(f"all {len(checks)} checked values appear in the rendered main paper or supplement")
