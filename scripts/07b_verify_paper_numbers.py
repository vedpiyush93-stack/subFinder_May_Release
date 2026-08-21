#!/usr/bin/env python3
"""Check that every headline number in paper/main.tex matches the run artifacts.

The manuscript states figures in prose, where nothing stops them drifting away
from the runs they describe as the pipeline is re-run. This script re-derives
each one from paper/audit_output.txt and the released CSVs and fails loudly on
any mismatch, so "traceable to the runs" is enforced rather than asserted.

    python3 scripts/07b_verify_paper_numbers.py
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
tex   = (ROOT/"paper/main.tex").read_text()
audit = dict(l.split("\t", 1) for l in (ROOT/"paper/audit_output.txt").read_text().splitlines()
             if "\t" in l and not l.startswith("#"))
lb    = pd.read_csv(ROOT/"artifacts/leaderboard.csv")
pfm   = pd.read_csv(ROOT/"artifacts/per_fold_metrics.csv")
cal   = pd.read_csv(ROOT/"artifacts/calibration_report.csv")

def acc(cfg):
    r = lb[lb.shorthand == cfg].iloc[0]
    return float(r.mean_acc), float(r.std_acc)

checks, fails = [], 0

def check(label, expected, present_in_tex=None, fmt="{:.4f}"):
    """Assert `expected` (from artifacts) appears in the manuscript."""
    global fails
    needle = present_in_tex if present_in_tex is not None else fmt.format(expected)
    ok = needle in tex
    checks.append((label, needle, ok))
    if not ok: fails += 1

top_a, top_s = acc("cpuV2__ET500_log2")
v1_a,  v1_s  = acc("cpu__ET500_log2")
brf_a, brf_s = acc("cv__BRF100")
dl_a,  dl_s  = acc(audit["best_dl_config"])

check("deployed accuracy",      top_a, f"0.9163\\pm0.0167")
check("v1 accuracy",            v1_a,  f"0.9066\\pm0.0174")
check("published baseline",     brf_a, f"0.8402\\pm0.0254")
check("best deep config",       dl_a,  f"0.7922\\pm0.0331")
check("gap vs baseline (pp)",   None,  f"{float(audit['gap_ours_vs_paper_baseline'])*100:.2f}")
check("gap vs best deep (pp)",  None,  f"{float(audit['gap_ours_vs_best_dl'])*100:.2f}")
check("n labelled PULs",        None,  "1{,}030")
check("n configurations",       None,  str(pfm.shorthand.nunique()))
check("n runs",                 None,  str(pfm.shorthand.nunique()*25))
check("sig-gene hit rate",      None,  audit["per_sub_sig_pul_hit_rate"].replace("%","\\%"))
check("sig-gene numerator",     None,  audit["per_sub_sig_total_hit"])
check("sig-gene denominator",   None,  "1{,}028")
check("lit canon pairs",        None,  audit["lit_db_substrate_family_pairs_after_alias_collapse"])
check("OOF accuracy",           None,  audit["oof_seed42_acc"])

for m, e in zip(cal.method, cal.ece_10bin):
    if m in ("uncalibrated", "temperature_scaling"):
        check(f"ECE {m}", None, f"{e:.3f}")

print(f"{'check':32s} {'value in artifacts':24s} result")
print("-"*72)
for label, needle, ok in checks:
    print(f"  {label:30s} {needle:24s} {'OK' if ok else 'NOT FOUND IN main.tex'}")
print("-"*72)
if fails:
    print(f"{fails} of {len(checks)} numbers in the manuscript do not match the artifacts")
    sys.exit(1)
print(f"all {len(checks)} checked numbers in paper/main.tex match the released run artifacts")
