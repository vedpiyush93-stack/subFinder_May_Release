#!/usr/bin/env python3
"""Build docs/per_pul_report.html — a per-substrate browse of every test PUL
from the seed-42 5-fold OOF (rep_1), with calibrated probabilities, p-values,
signature genes, literature-match badges, and per-fold OOV proportion.

The report has 13 tabs:
    Overview         5 demo PULs displayed in "full 12-row prediction form"
                     (each row = one substrate class, sorted by descending
                     calibrated probability, with sig genes for top-1 + TRUE)
    alginate         every test PUL with TRUE substrate = alginate
    alpha-glucan     every test PUL with TRUE substrate = alpha-glucan
    ...              (12 substrate tabs total)

Per-substrate tabs show, for each test PUL:
    - TRUE vs predicted, rank of TRUE in top-K (1 / 2 / 3 / >3)
    - 12-class calibrated probability vector as horizontal mini-bars
    - Top-5 signature genes (leave-one-token-out Δ on TRUE-class calibrated
      probs) with literature-match badges (exact | collapse | not canonical)
    - OOV proportion vs the PUL's own training fold vocab
    - Full PUL gene token sequence (collapsible)

Single-file self-contained HTML, no external deps (CSS + JS inline).

Usage:
    python3 scripts/12_build_per_pul_report.py
    open docs/per_pul_report.html
"""
from __future__ import annotations
import json, html, sys, re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocessing.tokenizers import tok_cpu
from src.lit_validation.canon import build_canon
from src.splits import rskf_splits
from src.ablation.leave_one_token_out import ablate_pul_for_class
import joblib

OUT_HTML = ROOT / "docs" / "per_pul_report.html"

# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------
print("[per-pul-report] loading data ...")
df_data = pd.read_csv(ROOT / "data/Train_data.csv")
sequences = df_data["sig_gene_seq"].fillna("").values
y_true_str = df_data["high_level_substr"].values
substrates = sorted(set(y_true_str))
n_classes = len(substrates)
sub2idx = {s: i for i, s in enumerate(substrates)}

# Calibrated OOF probabilities (rep_1 seed-42 5-fold OOF)
calib_npz = np.load(ROOT / "artifacts/calibration/oof_outer42_best_of_both.npz", allow_pickle=True)
P_cal = calib_npz["probs_temp"]         # (1030, 12)  — temperature-scaled
y_true_int = calib_npz["y_true"]        # (1030,)     — integer-encoded labels
T_per_fold = list(calib_npz["T_per_fold"])
print(f"  calibrated probs: shape={P_cal.shape}, T_per_fold={[f'{t:.4f}' for t in T_per_fold]}")

# Per-PUL signature genes (TRUE-class, calibrated)
sig_csv = ROOT / "artifacts/ablation/sig_gene_ablation_oof_outer42_groundtruth_calibrated.csv"
sig_df = pd.read_csv(sig_csv).set_index("idx")
print(f"  sig-gene ablation: {len(sig_df)} rows")

# Per-PUL signature genes (argmax-class, calibrated)
sig_arg_csv = ROOT / "artifacts/ablation/sig_gene_ablation_oof_outer42.csv"
sig_arg_df = pd.read_csv(sig_arg_csv).set_index("idx")

def _parse_top5_with_delta(s: str) -> list[tuple[str, float]]:
    """Parse 'tok:+0.1234;tok2:+0.0567;...' into [(tok, delta), ...]."""
    if not isinstance(s, str) or not s.strip(): return []
    pairs = []
    for part in s.split(";"):
        part = part.strip()
        if ":" not in part: continue
        tok, dv = part.rsplit(":", 1)
        try: pairs.append((tok, float(dv)))
        except ValueError: pass
    return pairs

# Literature canon (substrate → set of canonical CAZy families)
canon = build_canon(ROOT / "data/Literature_Data_fam_substrate_mapping.tsv")
print(f"  lit canon: {sum(len(v) for v in canon.values())} (substrate, CAZy) pairs across {len(canon)} classes")

# Build lit-substrate index for "match type" badges:
#   exact  : the canonical CAZy is in canon[predicted_substrate] AND substrate name = top-level (1:1)
#   collapse: matched via alias collapse (canonical in canon[substrate], but substrate is a collapsed name)
# For simplicity we treat canon membership as "exact or collapse" — fine-grained source comes from alias_map below
from src.lit_validation.alias_map import SUBSTRATE_ALIAS
EXACT_LIT = set()      # (high_class, lit_sub_name) where lit_sub matches high_class 1:1
COLLAPSE_LIT = set()   # (high_class, lit_sub_name) where lit_sub is collapsed into high_class
for high, aliases in SUBSTRATE_ALIAS.items():
    for a in aliases:
        if a == high: EXACT_LIT.add(high)
        else: COLLAPSE_LIT.add((high, a))

# Pre-compute per-PUL OOV vs that PUL's *training fold* vocab. The 5-fold OOF
# uses seed-42 only (matches calibration npz). For each fold, refit CountVec on
# outer_train, then compute OOV for every test row in that fold.
print("[per-pul-report] computing per-fold OOV (refitting CountVec for each of 5 folds) ...")
oov_per_pul = np.zeros(len(sequences), dtype=np.float32)
n_oov_tokens = np.zeros(len(sequences), dtype=np.int32)
n_total_tokens = np.zeros(len(sequences), dtype=np.int32)
fold_per_pul = np.full(len(sequences), -1, dtype=np.int32)
fold_train_vocab_size = {}  # fold_id (0-4) -> vocab size

for seed, fold, tr_outer, te, tr_inner, val in rskf_splits(y_true_str):
    if seed != 42: continue
    cv = CountVectorizer(tokenizer=tok_cpu, lowercase=False, token_pattern=None)
    cv.fit(sequences[tr_outer])
    vocab = set(cv.vocabulary_.keys())
    fold_train_vocab_size[fold] = len(vocab)
    for idx in te:
        toks = tok_cpu(sequences[idx])
        n_total_tokens[idx] = len(toks)
        n_oov_tokens[idx]   = sum(1 for t in toks if t not in vocab)
        oov_per_pul[idx]    = (n_oov_tokens[idx] / max(1, len(toks)))
        fold_per_pul[idx]   = fold
print(f"  per-fold vocab sizes: {dict(sorted(fold_train_vocab_size.items()))}")
print(f"  PULs with OOV > 0: {(oov_per_pul > 0).sum()}/{len(sequences)}  "
      f"(>10%: {(oov_per_pul > 0.10).sum()})")

# Dirichlet-uniform p-value (matches deployed PULPredictor)
def p_value_dirichlet_uniform(p: float) -> float:
    """P(X >= p) where X ~ Beta(1, K-1) under a uniform Dirichlet null."""
    # P(X >= p) = (1 - p) ** (K - 1)
    return float((1.0 - p) ** (n_classes - 1))


# ---------------------------------------------------------------------------
# Build per-PUL records
# ---------------------------------------------------------------------------
print("[per-pul-report] building per-PUL records ...")
PUL_RECORDS = []  # list of dicts; one per PUL (1030 of them)
for i in range(len(sequences)):
    probs = P_cal[i]
    order = np.argsort(probs)[::-1]
    ranked = [(substrates[j], float(probs[j])) for j in order]
    true_sub = y_true_str[i]
    pred_sub = substrates[int(probs.argmax())]
    rank_true = int(np.where(order == sub2idx[true_sub])[0][0]) + 1
    prob_true = float(probs[sub2idx[true_sub]])
    # Sig genes (TRUE-class)
    sig_true_str = sig_df.loc[i, "top5_with_delta"] if i in sig_df.index else ""
    sig_true = _parse_top5_with_delta(str(sig_true_str))
    # Sig genes (argmax/predicted-class)
    sig_arg_str = sig_arg_df.loc[i, "top5_with_delta"] if i in sig_arg_df.index else ""
    sig_argmax = _parse_top5_with_delta(str(sig_arg_str))

    PUL_RECORDS.append({
        "idx": i,
        "sequence": sequences[i],
        "tokens": tok_cpu(sequences[i]),
        "true": true_sub,
        "pred": pred_sub,
        "probs": {substrates[j]: float(probs[j]) for j in range(n_classes)},
        "ranked": ranked,         # [(substrate, prob), ...] sorted desc
        "rank_true": rank_true,
        "prob_true": prob_true,
        "fold": int(fold_per_pul[i]),
        "oov_pct": float(oov_per_pul[i]) * 100,
        "n_oov": int(n_oov_tokens[i]),
        "n_tok": int(n_total_tokens[i]),
        "sig_true": sig_true,         # for the TRUE class
        "sig_argmax": sig_argmax,     # for the predicted class
    })


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
NAVY   = "#1a3a5c"
SAGE   = "#27ae60"
AMBER  = "#f39c12"
RED    = "#c0392b"
GRAY   = "#7f8c8d"

# Color per substrate (12 distinct, colorblind-aware)
SUB_COLORS = {
    "alginate":         "#1f77b4",
    "alpha-glucan":     "#d62728",
    "alpha-mannan":     "#9467bd",
    "arabinogalactan":  "#8c564b",
    "beta-glucan":      "#ff7f0e",
    "beta-mannan":      "#e377c2",
    "chitin":           "#17becf",
    "fructan":          "#7f7f7f",
    "galactan":         "#bcbd22",
    "host glycan":      "#2ca02c",
    "pectin":           "#aec7e8",
    "xylan":            "#ffbb78",
}

def _tok_is_cazy(t: str) -> bool:
    return bool(re.match(r"^(GH|PL|CE|CBM|GT|AA)[0-9]+(_[0-9]+)?$", t))

def _lit_label_for(class_name: str, tok: str) -> tuple[str, str]:
    """Return (badge_class, badge_text) for how `tok` relates to literature for `class_name`."""
    if not _tok_is_cazy(tok):
        return ("badge-non", "non-CAZy")
    base = tok.split("_")[0]
    canon_set = canon.get(class_name, set())
    if tok in canon_set or base in canon_set:
        # Check if this match is via exact or collapse
        # heuristic: if substrate is in EXACT_LIT and there are no alias entries, it's exact-only
        # if SUBSTRATE_ALIAS[class_name] has > 1 entry, it's likely collapse
        aliases = SUBSTRATE_ALIAS.get(class_name, [class_name])
        if len(aliases) <= 1: return ("badge-exact", "LIT exact")
        return ("badge-collapse", "LIT collapse")
    return ("badge-miss", "not in lit canon")

def _prob_bar(prob: float, color: str, width_px: int = 110) -> str:
    """Render a single horizontal prob bar (inline-block)."""
    pct = max(1, int(prob * 100))
    return (f'<span class="bar-wrap" style="width:{width_px}px">'
            f'<span class="bar-fill" style="width:{pct}%;background:{color}"></span>'
            f'<span class="bar-num">{prob:.3f}</span>'
            f'</span>')

def _rank_badge(rank: int) -> str:
    if rank == 1: return f'<span class="rank rank-1">#1 ✓</span>'
    if rank == 2: return f'<span class="rank rank-2">#2</span>'
    if rank == 3: return f'<span class="rank rank-3">#3</span>'
    return f'<span class="rank rank-bad">#{rank}</span>'

def _sig_genes_html(class_name: str, sig: list[tuple[str, float]]) -> str:
    """Render top-5 sig genes for one class as colored chips."""
    if not sig: return '<span class="muted">—</span>'
    parts = []
    for tok, d in sig[:5]:
        badge_cls, badge_text = _lit_label_for(class_name, tok)
        parts.append(
            f'<span class="sig-chip {badge_cls}" title="Δ-prob = {d:+.4f}; {badge_text}">'
            f'<b>{html.escape(tok)}</b> <span class="dlt">Δ{d:+.3f}</span>'
            f'</span>'
        )
    return "".join(parts)

def _seq_html(tokens: list[str], oov_set: set[str]) -> str:
    """Render a PUL sequence as colored token chips; OOV tokens in red."""
    parts = []
    for t in tokens:
        cls = "tok-oov" if t in oov_set else ("tok-cazy" if _tok_is_cazy(t) else "tok-norm")
        parts.append(f'<span class="tok {cls}">{html.escape(t)}</span>')
    return '<span class="seq">' + " ".join(parts) + '</span>'

def _oov_set_for_pul(i: int) -> set[str]:
    """The set of tokens in PUL i that are not in its training fold's vocab.
    We don't store the per-fold vocab, so we re-derive on the fly."""
    return _OOV_LOOKUP.get(i, set())

# Pre-cache OOV token set per PUL (we already have count, but need the actual tokens for highlighting)
print("[per-pul-report] caching per-fold OOV token sets for highlighting ...")
_OOV_LOOKUP: dict[int, set[str]] = {}
for seed, fold, tr_outer, te, _, _ in rskf_splits(y_true_str):
    if seed != 42: continue
    cv = CountVectorizer(tokenizer=tok_cpu, lowercase=False, token_pattern=None)
    cv.fit(sequences[tr_outer])
    vocab = set(cv.vocabulary_.keys())
    for idx in te:
        toks = tok_cpu(sequences[idx])
        _OOV_LOOKUP[idx] = {t for t in toks if t not in vocab}

# ---------------------------------------------------------------------------
# Overview tab — pick 5 demo PULs, render full 12-row prediction form per PUL
# ---------------------------------------------------------------------------
def _pul_full_form(r: dict) -> str:
    """Render one PUL as a 12-row table (one row per substrate, sorted by prob desc).
    Each row: substrate · prob bar · p-value · sig genes for that substrate (if available).
    Sig genes are shown for the TRUE class and the predicted class only (we don't have
    ablation cached for the other 10 classes).
    """
    is_correct = r["pred"] == r["true"]
    rank_html = _rank_badge(r["rank_true"])
    correct_badge = ('<span class="ok-badge">✓ correct top-1</span>' if is_correct
                     else f'<span class="bad-badge">✗ TRUE at rank {r["rank_true"]}</span>')
    rows = []
    per_class_ablation = ABLATION_FOR_DEMO.get(r["idx"], {})
    for rank_pos, (sub, p) in enumerate(r["ranked"], start=1):
        color = SUB_COLORS.get(sub, GRAY)
        pval = p_value_dirichlet_uniform(p)
        is_true = (sub == r["true"])
        is_pred = (sub == r["pred"])
        flags = []
        if is_true: flags.append('<span class="flag-true">TRUE</span>')
        if is_pred: flags.append('<span class="flag-pred">argmax</span>')
        # Sig genes for this row: prefer the on-the-fly per-class ablation we
        # computed for the top-3 (+TRUE); fall back to the cached argmax/TRUE
        # csv data; finally, a clear "not computed" note for low-prob rows.
        if sub in per_class_ablation and per_class_ablation[sub]:
            sig_html = _sig_genes_html(sub, per_class_ablation[sub])
            sig_note = ""
        elif is_true and r["sig_true"]:
            sig_html = _sig_genes_html(sub, r["sig_true"])
            sig_note = ""
        elif is_pred and r["sig_argmax"]:
            sig_html = _sig_genes_html(sub, r["sig_argmax"])
            sig_note = ""
        else:
            sig_html = (f'<span class="muted">not computed (prob ≈ {p:.3f} — '
                        f'ablation is skipped for low-rank classes; rerun '
                        f'<code>ablate_pul_for_class(..., class_name=\"{sub}\")</code> for this row)</span>')
            sig_note = ""
        row_cls = "row-true" if is_true else ("row-pred" if is_pred else "")
        rows.append(
            f'<tr class="{row_cls}">'
            f'<td class="sub-cell"><span class="sub-dot" style="background:{color}"></span>'
            f'<b>{html.escape(sub)}</b> {" ".join(flags)}</td>'
            f'<td>{_prob_bar(p, color, width_px=180)}</td>'
            f'<td class="mono">{pval:.2e}</td>'
            f'<td class="sig-cell">{sig_html}</td>'
            f'</tr>'
        )
    return (
        f'<div class="pul-card">'
        f'<div class="pul-card-head">'
        f'<div><b>PUL idx {r["idx"]}</b> · fold {r["fold"]} · '
        f'TRUE = <b>{html.escape(r["true"])}</b> · pred = <b>{html.escape(r["pred"])}</b> '
        f'{rank_html} {correct_badge}</div>'
        f'<div class="meta-line">OOV: {r["oov_pct"]:.1f}% '
        f'({r["n_oov"]}/{r["n_tok"]} tokens unknown to training fold)</div>'
        f'</div>'
        f'<details><summary>PUL gene token sequence ({r["n_tok"]} tokens)</summary>'
        f'<div class="seq-box">{_seq_html(r["tokens"], _oov_set_for_pul(r["idx"]))}</div></details>'
        f'<table class="prob-table">'
        f'<thead><tr><th style="width:30%">substrate</th><th>calibrated prob</th>'
        f'<th>p-value</th><th>signature genes (top-5 by Δ on that class)</th></tr></thead>'
        f'<tbody>' + "".join(rows) + '</tbody>'
        f'</table>'
        f'</div>'
    )

# Pick demo PULs: 1 confident-correct, 1 medium-conf, 1 low-conf-correct,
# 1 rank-2 redemption, 1 rank-3 redemption, 1 confident-wrong
def _pick_demo_puls() -> list[dict]:
    candidates = {
        "Confident correct (top-1, high prob)":         lambda r: r["pred"] == r["true"] and r["prob_true"] > 0.95,
        "Medium-confidence correct":                     lambda r: r["pred"] == r["true"] and 0.45 < r["prob_true"] < 0.65,
        "Low-confidence correct (model knows it's unsure)": lambda r: r["pred"] == r["true"] and 0.30 < r["prob_true"] < 0.42,
        "Rank-2 redemption (TRUE at rank 2)":            lambda r: r["rank_true"] == 2 and r["prob_true"] > 0.10,
        "Rank-3 redemption (TRUE at rank 3)":            lambda r: r["rank_true"] == 3 and r["prob_true"] > 0.05,
        "Confident WRONG (TRUE missed top-3)":           lambda r: r["pred"] != r["true"] and r["probs"][r["pred"]] > 0.6 and r["rank_true"] > 3,
    }
    out = []
    for label, predicate in candidates.items():
        match = next((r for r in PUL_RECORDS if predicate(r)), None)
        if match: out.append({"label": label, "rec": match})
    return out

DEMO_PULS = _pick_demo_puls()
print(f"[per-pul-report] selected {len(DEMO_PULS)} demo PULs for Overview tab")

# For the Overview tab demos, ablation is cached only for argmax + TRUE classes
# (those are the headline use-cases). Compute it on-the-fly for the top-3 ranked
# classes too so the 12-row prediction form has sig genes for every plausible class.
print(f"[per-pul-report] computing top-3 per-class ablation for {len(DEMO_PULS)} demo PULs ...")
ABLATION_FOR_DEMO: dict[int, dict[str, list[tuple[str, float]]]] = {}
_pipeline_cache: dict[int, object] = {}
for d in DEMO_PULS:
    r = d["rec"]
    fold = r["fold"]
    if fold not in _pipeline_cache:
        clf_path = ROOT / f"artifacts/predictions/cpu__ET500_log2/r42_f{fold}/classifier.joblib"
        if clf_path.exists():
            _pipeline_cache[fold] = joblib.load(clf_path)
        else:
            print(f"  WARN: no classifier.joblib at {clf_path.relative_to(ROOT)}")
            _pipeline_cache[fold] = None
    pipeline = _pipeline_cache[fold]
    if pipeline is None:
        ABLATION_FOR_DEMO[r["idx"]] = {}
        continue
    T_fold = float(T_per_fold[fold])
    per_class: dict[str, list[tuple[str, float]]] = {}
    # ablate top-3 ranked classes + ensure TRUE class is included
    targets = {r["ranked"][0][0], r["ranked"][1][0], r["ranked"][2][0], r["true"]}
    for cls in targets:
        try:
            per_class[cls] = ablate_pul_for_class(pipeline, r["sequence"], cls,
                                                  top_k=5, apply_temp=T_fold)
        except Exception as e:
            print(f"  WARN: ablation failed for PUL {r['idx']} class {cls}: {e}")
            per_class[cls] = []
    ABLATION_FOR_DEMO[r["idx"]] = per_class
print(f"  computed ablation for {sum(len(v) for v in ABLATION_FOR_DEMO.values())} (PUL, class) cells")


def _overview_html() -> str:
    intro = (
        '<div class="intro">'
        '<h2>Overview — how to read this report</h2>'
        '<p>Every PUL is run through the deployed calibrated model '
        f'(<code>cpu__ET500_log2</code>, mean T = {float(np.mean(T_per_fold)):.3f}) and gets a '
        'probability distribution over all 12 substrate classes. The probabilities are '
        '<b>temperature-calibrated</b> (per-fold T on inner-CV5 — see paper §3.2), so they are '
        'safe to use as confidence values, not just rankings.</p>'
        '<p>Below are <b>6 hand-picked test PULs</b> in <i>full prediction form</i>: '
        'one row per substrate, sorted by descending calibrated probability. Columns: '
        '<b>substrate</b> · <b>calibrated prob</b> (with mini-bar) · '
        '<b>p-value</b> (uniform-Dirichlet null) · '
        '<b>signature genes for THAT substrate</b> '
        '(leave-one-token-out Δ on the calibrated probabilities). For the top-3 classes the '
        'sig-gene ablation is computed on-the-fly per row; for the bottom 9 (prob ≈ 0 — model is '
        'essentially ignoring them) the ablation is skipped since Δ is uninformative.</p>'
        '<p>Use the <b>per-substrate tabs above</b> to browse every test PUL grouped by ground-truth '
        'class — useful for, e.g., "show me every test PUL whose TRUE substrate is fructan and '
        'whether the model got each one right".</p>'
        '<div class="legend-inline">'
        '<div><b>Rank badges:</b> '
        '<span class="ok-badge">✓ correct top-1</span> '
        '<span class="bad-badge">✗ TRUE at rank N</span></div>'
        '<div><b>Row highlighting:</b> '
        '<span style="background:#d4edda;padding:1px 6px;border-radius:3px">green = TRUE row</span> '
        '<span style="background:#fef5e7;padding:1px 6px;border-radius:3px">amber = argmax row</span></div>'
        '<div><b>Signature gene chip color</b> (in the rightmost column — meaning is per-row, i.e. vs. the literature canon for THAT row\'s substrate):'
        '<br><span class="sig-chip badge-exact"><b>GH8</b></span>green = literature canonical, 1:1 exact match'
        ' &nbsp;·&nbsp; <span class="sig-chip badge-collapse"><b>GH16</b></span>light green = canonical via alias collapse (e.g. starch → α-glucan)'
        ' &nbsp;·&nbsp; <span class="sig-chip badge-miss"><b>GT2</b></span>red = CAZy but not in lit canon for that class'
        ' &nbsp;·&nbsp; <span class="sig-chip badge-non"><b>LacI</b></span>gray = non-CAZy (regulator / TC number / "null")'
        '</div>'
        '<div><b>Token color inside the collapsible "PUL gene token sequence":</b> '
        '<span class="tok tok-cazy">GH8</span>blue = CAZy enzyme family &nbsp;·&nbsp; '
        '<span class="tok tok-oov">novel_tok</span>red dashed = OOV (unknown to training fold) &nbsp;·&nbsp; '
        '<span class="tok tok-norm">1.B.55.3.1</span>gray = other</div>'
        '<div><b>Δ (delta)</b> = drop in calibrated P(class) when the token is removed. '
        'Δ &gt; 0 = the model relies on the token for that class. Top-5 sorted descending = '
        '5 strongest supporters.</div>'
        '</div>'
        '</div>'
    )
    cards = []
    for d in DEMO_PULS:
        cards.append(f'<div class="demo-section"><h3>{html.escape(d["label"])}</h3>'
                     + _pul_full_form(d["rec"]) + '</div>')
    return intro + "".join(cards)


# ---------------------------------------------------------------------------
# Per-substrate tabs
# ---------------------------------------------------------------------------
def _pul_compact_card(r: dict) -> str:
    """Compact card showing one test PUL within a substrate's tab.

    Layout: header bar (TRUE | pred | rank | OOV); 12-row prob mini-table sorted by prob;
    sig genes row (always TRUE-class; argmax-class also shown when prediction was wrong);
    collapsible sequence.
    """
    is_correct = r["pred"] == r["true"]
    rank_html = _rank_badge(r["rank_true"])
    border_color = SAGE if r["rank_true"] == 1 else (AMBER if r["rank_true"] <= 3 else RED)
    # Probability mini-bars (sorted desc; TRUE class highlighted)
    prob_rows = []
    for sub, p in r["ranked"]:
        color = SUB_COLORS.get(sub, GRAY)
        is_true = sub == r["true"]
        cls = "row-true" if is_true else ""
        marker = '<b>★</b>' if is_true else ''
        prob_rows.append(
            f'<tr class="{cls}"><td>{marker} {html.escape(sub)}</td>'
            f'<td>{_prob_bar(p, color, width_px=130)}</td></tr>'
        )
    # Sig-gene attribution — be EXPLICIT about which class each set attributes to.
    # TRUE-class sig genes are always shown ("what would have pushed the model
    # toward the right answer"). For incorrect predictions, also show argmax-class
    # sig genes ("what the model actually used to make its wrong call").
    sig_true_html = _sig_genes_html(r["true"], r["sig_true"])
    true_block = (
        f'<div class="sig-block">'
        f'<div class="sig-block-label">'
        f'Δ for <b>TRUE</b> class = <code>{html.escape(r["true"])}</code> '
        f'<span class="sig-hint">(what would have pushed toward the right answer)</span></div>'
        f'<div class="sig-row">{sig_true_html}</div>'
        f'</div>'
    )
    pred_block = ""
    if not is_correct:
        sig_pred_html = _sig_genes_html(r["pred"], r["sig_argmax"])
        pred_block = (
            f'<div class="sig-block sig-block-wrong">'
            f'<div class="sig-block-label">'
            f'Δ for <b>predicted (wrong)</b> class = <code>{html.escape(r["pred"])}</code> '
            f'<span class="sig-hint">(what the model latched onto for its wrong call)</span></div>'
            f'<div class="sig-row">{sig_pred_html}</div>'
            f'</div>'
        )
    return (
        f'<div class="card" style="border-left-color:{border_color}">'
        f'<div class="card-head">'
        f'<span class="card-idx">PUL #{r["idx"]}</span> · fold {r["fold"]} · '
        f'TRUE: <b>{html.escape(r["true"])}</b> · pred: <b>{html.escape(r["pred"])}</b> '
        f'{rank_html} · '
        f'<span class="oov-tag" title="OOV vs this PUL\'s training fold vocab">'
        f'OOV {r["oov_pct"]:.1f}% ({r["n_oov"]}/{r["n_tok"]})</span>'
        f'</div>'
        f'<div class="card-body">'
        f'<div class="card-col"><div class="col-label">All 12 calibrated probabilities (TRUE = ★)</div>'
        f'<table class="mini-prob">{"".join(prob_rows)}</table></div>'
        f'<div class="card-col">'
        f'<div class="col-label">Signature genes — leave-one-token-out Δ on calibrated probs</div>'
        f'{true_block}{pred_block}'
        f'<details class="seq-details"><summary>PUL gene token sequence ({r["n_tok"]} tokens)</summary>'
        f'<div class="seq-box">{_seq_html(r["tokens"], _oov_set_for_pul(r["idx"]))}</div></details>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

def _legend_strip() -> str:
    """Compact legend that explains every color/badge convention used on this tab.
    Reused per tab so any tab is self-contained."""
    return (
        '<div class="legend">'
        '<div class="legend-row">'
        '<span class="legend-label">Rank-of-TRUE badge (top of each card):</span> '
        '<span class="rank rank-1">#1 ✓</span> model predicted TRUE correctly &nbsp;·&nbsp; '
        '<span class="rank rank-2">#2</span> TRUE was rank 2 &nbsp;·&nbsp; '
        '<span class="rank rank-3">#3</span> rank 3 &nbsp;·&nbsp; '
        '<span class="rank rank-bad">#4+</span> TRUE missed top-3'
        '</div>'
        '<div class="legend-row">'
        '<span class="legend-label">Signature-gene chip color (right side, "literature status for THAT row\'s substrate"):</span><br>'
        '<span class="sig-chip badge-exact"><b>GH8</b></span> <b>green</b> = literature canonical, 1:1 exact match for this class in lit DB &nbsp;·&nbsp; '
        '<span class="sig-chip badge-collapse"><b>GH16</b></span> <b>light green</b> = literature canonical via alias collapse (e.g. starch is collapsed into α-glucan)<br>'
        '<span class="sig-chip badge-miss"><b>GT2</b></span> <b>red</b> = CAZy enzyme family but NOT in lit canon for this class (the model relies on it but no paper says it should) &nbsp;·&nbsp; '
        '<span class="sig-chip badge-non"><b>LacI</b></span> <b>gray</b> = non-CAZy gene (regulator / transporter / TC number / "null")'
        '</div>'
        '<div class="legend-row">'
        '<span class="legend-label">PUL gene token sequence highlighting (collapsible at bottom of each card):</span><br>'
        '<span class="tok tok-cazy">GH8</span> <b>blue</b> = CAZy enzyme family (GH/PL/CE/CBM/GT/AA prefix) &nbsp;·&nbsp; '
        '<span class="tok tok-oov">novel_tok</span> <b>red dashed</b> = OOV (token not seen in this PUL\'s training fold vocab) &nbsp;·&nbsp; '
        '<span class="tok tok-norm">1.B.55.3.1</span> <b>gray</b> = other (TC number, regulator, "null", non-CAZy gene)'
        '</div>'
        '<div class="legend-row legend-row-small">'
        '<b>Δ (delta) =</b> drop in calibrated P(class) when the token is removed. '
        '<b>Δ &gt; 0</b> means the model relies on that token for that class (support); '
        'top-5 are sorted Δ descending so they\'re the 5 strongest supporters. '
        'TRUE-class sig genes are always shown; for <b>incorrect</b> predictions the argmax-class sig genes are also '
        'shown side-by-side so you can compare "what would have helped" vs "what misled the model".'
        '</div>'
        '</div>'
    )

def _substrate_tab_html(class_name: str) -> str:
    recs = [r for r in PUL_RECORDS if r["true"] == class_name]
    if not recs: return f"<p>No test PULs for {class_name}.</p>"
    # Sort recs: correct first (by descending prob_true), then wrong (by rank asc, then descending prob_true)
    recs.sort(key=lambda r: (r["rank_true"], -r["prob_true"]))
    n = len(recs)
    top1 = sum(1 for r in recs if r["rank_true"] == 1) / n
    top3 = sum(1 for r in recs if r["rank_true"] <= 3) / n
    mean_oov = float(np.mean([r["oov_pct"] for r in recs]))
    n_wrong = sum(1 for r in recs if r["pred"] != r["true"])
    # Class summary banner
    canon_list = sorted(canon.get(class_name, set()))
    canon_html = ", ".join(f"<code>{html.escape(c)}</code>" for c in canon_list[:30])
    if len(canon_list) > 30: canon_html += f" <span class='muted'>(+{len(canon_list)-30} more)</span>"
    summary = (
        f'<div class="sub-summary">'
        f'<h2><span class="sub-dot" style="background:{SUB_COLORS.get(class_name, GRAY)}"></span>'
        f'{html.escape(class_name)}</h2>'
        f'<p class="tab-intro">'
        f'Every test PUL whose <b>TRUE substrate is {html.escape(class_name)}</b> '
        f'({n} PULs total from rep_1 seed-42 5-fold OOF). Each card shows the model\'s '
        f'12-class calibrated probability vector, the rank of the TRUE class, and the top-5 '
        f'signature genes attributing to <b>{html.escape(class_name)}</b> '
        f'(plus the predicted class on wrong calls).'
        f'</p>'
        f'<div class="stat-grid">'
        f'<div class="stat"><div class="stat-label">test PULs</div><div class="stat-val">{n}</div></div>'
        f'<div class="stat"><div class="stat-label">top-1 acc</div><div class="stat-val">{top1:.3f}</div></div>'
        f'<div class="stat"><div class="stat-label">top-3 acc</div><div class="stat-val">{top3:.3f}</div></div>'
        f'<div class="stat"><div class="stat-label">mispredicted</div><div class="stat-val">{n_wrong}/{n}</div></div>'
        f'<div class="stat"><div class="stat-label">mean OOV %</div><div class="stat-val">{mean_oov:.2f}%</div></div>'
        f'<div class="stat"><div class="stat-label">lit-canonical CAZy</div><div class="stat-val">{len(canon_list)}</div></div>'
        f'</div>'
        f'<div class="canon-strip"><b>Literature canon (after alias collapse):</b> {canon_html}</div>'
        f'</div>'
        + _legend_strip() +
        f'<div class="sort-note">'
        f'<b>Card order:</b> rank-1 (correct) first, then rank-2/3 (TRUE recovered in top-3), '
        f'then rank ≥ 4 (TRUE missed top-3). Border color: '
        f'<span class="border-sample" style="border-left-color:{SAGE}">green = #1</span> &nbsp;'
        f'<span class="border-sample" style="border-left-color:{AMBER}">amber = #2/#3</span> &nbsp;'
        f'<span class="border-sample" style="border-left-color:{RED}">red = #4+</span>. '
        f'Click "<i>PUL gene token sequence</i>" inside any card to expand the full token list '
        f'with OOV tokens highlighted in red.'
        f'</div>'
    )
    cards = "".join(_pul_compact_card(r) for r in recs)
    return summary + cards


# ---------------------------------------------------------------------------
# Compose the final HTML
# ---------------------------------------------------------------------------
print("[per-pul-report] composing HTML ...")

CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 0; background: #f6f7f9; color: #2c3e50; }
.page { max-width: 1280px; margin: 0 auto; padding: 0 24px 60px; }
header { background: #1a3a5c; color: white; padding: 24px 40px; }
header h1 { margin: 0 0 6px; font-size: 24px; font-weight: 700; }
header .sub { color: #c7d6e3; font-size: 13px; }
header .meta { color: #a3b8cc; font-size: 12px; margin-top: 8px; }
nav { position: sticky; top: 0; background: white; border-bottom: 2px solid #1a3a5c;
      padding: 8px 24px; z-index: 100; box-shadow: 0 2px 6px rgba(0,0,0,0.05);
      overflow-x: auto; white-space: nowrap; }
nav button { background: none; border: 1px solid transparent; border-radius: 8px;
             padding: 8px 14px; margin: 0 2px; cursor: pointer; font-size: 13px;
             color: #2c3e50; font-weight: 500; transition: all 0.12s; }
nav button:hover { background: #ecf0f1; }
nav button.active { background: #1a3a5c; color: white; border-color: #1a3a5c; }
nav .nav-count { font-size: 11px; opacity: 0.7; margin-left: 4px; }
section.tab { display: none; padding-top: 20px; }
section.tab.active { display: block; }
.intro { background: white; padding: 18px 24px; border-radius: 8px;
         border: 1px solid #d6dee5; margin-bottom: 24px; }
.intro h2 { margin-top: 0; color: #1a3a5c; font-size: 18px; }
.intro p { font-size: 13.5px; line-height: 1.6; margin: 8px 0; }
.demo-section { margin-bottom: 28px; }
.demo-section h3 { color: #1a3a5c; font-size: 16px; margin: 0 0 8px;
                   padding-bottom: 4px; border-bottom: 1px solid #d6dee5; }
.pul-card { background: white; border: 1px solid #d6dee5; border-radius: 8px;
            padding: 16px 20px; margin-bottom: 18px; }
.pul-card-head { font-size: 13px; margin-bottom: 10px; display: flex;
                 justify-content: space-between; align-items: center; gap: 12px;
                 flex-wrap: wrap; }
.pul-card-head .meta-line { font-size: 12px; color: #7f8c8d; }
.prob-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
.prob-table thead th { background: #f0f3f6; color: #1a3a5c; font-weight: 700;
                       padding: 6px 10px; text-align: left; border-bottom: 1px solid #d6dee5; }
.prob-table tbody td { padding: 4px 10px; border-bottom: 1px solid #f0f3f6; vertical-align: middle; }
.prob-table .row-true { background: #d4edda; }
.prob-table .row-pred:not(.row-true) { background: #fef5e7; }
.sub-cell .sub-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
                     margin-right: 6px; vertical-align: middle; }
.flag-true, .flag-pred { display: inline-block; font-size: 9.5px; font-weight: 700;
                         padding: 1px 5px; border-radius: 3px; margin-left: 6px;
                         vertical-align: middle; }
.flag-true { background: #155724; color: white; }
.flag-pred { background: #856404; color: white; }
.bar-wrap { position: relative; display: inline-block; height: 14px;
            background: #e8eced; border-radius: 3px; vertical-align: middle; }
.bar-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px;
            min-width: 1px; opacity: 0.75; }
.bar-num { position: absolute; left: 50%; top: -1px; transform: translateX(-50%);
           font-size: 10.5px; font-weight: 700; color: #1a1a1a; mix-blend-mode: difference;
           color: white; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.muted { color: #95a5a6; font-style: italic; font-size: 11px; }
.rank { display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: 700;
        font-size: 11px; margin-left: 6px; vertical-align: middle; }
.rank-1 { background: #d4edda; color: #155724; }
.rank-2 { background: #fff3cd; color: #856404; }
.rank-3 { background: #fde2cf; color: #7d4209; }
.rank-bad { background: #f5b7b1; color: #6e2820; }
.ok-badge { background: #155724; color: white; padding: 2px 8px; border-radius: 4px;
            font-size: 11px; font-weight: 700; margin-left: 6px; }
.bad-badge { background: #6e2820; color: white; padding: 2px 8px; border-radius: 4px;
             font-size: 11px; font-weight: 700; margin-left: 6px; }
.sig-chip { display: inline-block; font-size: 11px; padding: 2px 7px; border-radius: 4px;
            margin: 2px 3px 2px 0; border: 1px solid; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.sig-chip .dlt { color: rgba(0,0,0,0.55); font-weight: 400; font-size: 10px; margin-left: 4px; }
.badge-exact    { background: #d4edda; border-color: #28a745; color: #155724; }
.badge-collapse { background: #e0f3e8; border-color: #6dbf86; color: #2d5e3a; }
.badge-miss     { background: #f8d7da; border-color: #c0392b; color: #721c24; }
.badge-non      { background: #ecf0f1; border-color: #95a5a6; color: #4a5961; }
/* Per-substrate tab styles */
.sub-summary { background: white; padding: 18px 24px; border-radius: 8px;
               border: 1px solid #d6dee5; margin-bottom: 18px; }
.sub-summary h2 { margin: 0 0 12px; font-size: 22px; color: #1a3a5c; }
.sub-summary .sub-dot { display: inline-block; width: 14px; height: 14px;
                        border-radius: 50%; margin-right: 8px; vertical-align: middle; }
.stat-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-bottom: 10px; }
.stat { background: #f6f7f9; padding: 8px 12px; border-radius: 6px; text-align: center; }
.stat-label { font-size: 10.5px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-val { font-size: 19px; font-weight: 700; color: #1a3a5c; margin-top: 2px; }
.canon-strip { font-size: 12px; padding: 8px 0; border-top: 1px solid #f0f3f6;
               border-bottom: 1px solid #f0f3f6; margin-top: 6px; line-height: 1.7; }
.canon-strip code { background: #f0f3f6; padding: 1px 5px; border-radius: 3px;
                    font-size: 11px; }
.hint { font-size: 11px; color: #7f8c8d; margin: 10px 0 0; font-style: italic; }
.card { background: white; border: 1px solid #d6dee5; border-left: 4px solid #ddd;
        border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }
.card-head { font-size: 12.5px; margin-bottom: 10px; }
.card-idx { font-weight: 700; color: #1a3a5c; }
.oov-tag { display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 3px;
           background: #ecf0f1; color: #4a5961; margin-left: 6px; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.card-body { display: grid; grid-template-columns: 320px 1fr; gap: 20px; }
.col-label { font-size: 10.5px; color: #7f8c8d; text-transform: uppercase;
             letter-spacing: 0.5px; margin-bottom: 4px; font-weight: 700; }
.mini-prob { width: 100%; border-collapse: collapse; font-size: 11.5px; }
.mini-prob td { padding: 2px 6px; vertical-align: middle; }
.mini-prob td:first-child { white-space: nowrap; }
.mini-prob .row-true { background: #d4edda; font-weight: 600; }
.sig-row { line-height: 2; }
.seq-details { margin-top: 8px; font-size: 11px; }
.seq-details summary { cursor: pointer; color: #1a3a5c; font-weight: 600; }
.seq-box { padding: 8px 0 0; line-height: 1.8; }
.seq { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11px; }
.tok { display: inline-block; padding: 1px 5px; border-radius: 3px; margin: 1px 2px;
       background: #ecf0f1; color: #4a5961; }
.tok-cazy { background: #d6eaf8; color: #1f4e79; font-weight: 600; }
.tok-oov  { background: #fadbd8; color: #6e2820; border: 1px dashed #c0392b; }
.tok-norm { background: #ecf0f1; }
/* Per-tab intro + legend bars */
.tab-intro { font-size: 13px; line-height: 1.55; color: #34495e; margin: 6px 0 12px; }
.legend { background: #fbfcfd; border: 1px solid #d6dee5; border-left: 3px solid #1a3a5c;
          border-radius: 6px; padding: 10px 14px; margin: 0 0 14px; font-size: 12px; }
.legend-row { padding: 3px 0; line-height: 2.0; }
.legend-row-small { font-size: 11.5px; color: #5a6c7d; padding-top: 6px;
                    border-top: 1px solid #ecf0f1; margin-top: 4px; line-height: 1.5; }
.legend-label { font-weight: 700; color: #1a3a5c; margin-right: 8px; }
.legend-inline { background: #f6f7f9; padding: 10px 14px; border-radius: 6px;
                 border-left: 3px solid #1a3a5c; font-size: 12px; line-height: 1.9; }
.legend-inline div { margin: 4px 0; }
.sort-note { background: #fffbea; border: 1px solid #ecddb0; border-radius: 6px;
             padding: 8px 12px; margin: 0 0 14px; font-size: 11.5px; color: #6e5a18; }
.border-sample { display: inline-block; border-left: 4px solid; padding: 1px 6px;
                 background: white; border-radius: 2px; font-size: 11px; }
/* Sig-gene blocks (TRUE vs predicted attribution) */
.sig-block { padding: 6px 0; }
.sig-block + .sig-block { border-top: 1px dashed #ecf0f1; margin-top: 6px; }
.sig-block-label { font-size: 11px; color: #34495e; margin-bottom: 4px; }
.sig-block-label code { background: #ecf0f1; padding: 1px 5px; border-radius: 3px;
                        font-size: 11px; }
.sig-hint { color: #95a5a6; font-style: italic; font-size: 10.5px; margin-left: 4px; }
.sig-block-wrong .sig-block-label { color: #6e2820; }
.sig-block-wrong .sig-block-label code { background: #fadbd8; color: #6e2820; }
"""

JS = """
function showTab(name) {
  document.querySelectorAll('section.tab').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.querySelector('nav button[data-tab="' + name + '"]').classList.add('active');
  window.scrollTo({top: 0, behavior: 'instant'});
}
document.addEventListener('DOMContentLoaded', () => showTab('overview'));
"""

# Tab nav
substrate_counts = {s: sum(1 for r in PUL_RECORDS if r["true"] == s) for s in substrates}
nav_buttons = ['<button data-tab="overview" onclick="showTab(\'overview\')">Overview</button>']
for s in substrates:
    safe = re.sub(r"[^a-z0-9]", "-", s.lower())
    nav_buttons.append(
        f'<button data-tab="{safe}" onclick="showTab(\'{safe}\')">{html.escape(s)}'
        f'<span class="nav-count">({substrate_counts[s]})</span></button>'
    )

# Tab sections
sections = [f'<section id="tab-overview" class="tab">{_overview_html()}</section>']
for s in substrates:
    safe = re.sub(r"[^a-z0-9]", "-", s.lower())
    sections.append(f'<section id="tab-{safe}" class="tab">{_substrate_tab_html(s)}</section>')

# Header stats
n_pul = len(PUL_RECORDS)
overall_top1 = sum(1 for r in PUL_RECORDS if r["rank_true"] == 1) / n_pul
overall_top3 = sum(1 for r in PUL_RECORDS if r["rank_true"] <= 3) / n_pul
mean_T = float(np.mean(T_per_fold))

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
OUT_HTML.write_text(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>subFinder — per-PUL test-set report (rep_1, seed-42 5-fold OOF)</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>subFinder — per-PUL test-set report</h1>
  <div class="sub">All {n_pul} held-out PULs from the rep_1 seed-42 5-fold OOF run, with calibrated probabilities, p-values, signature genes, and literature-match badges.</div>
  <div class="meta">Model: <code>cpu__ET500_log2</code> · temperature scaling, mean T = {mean_T:.3f} · top-1 OOF acc = {overall_top1:.4f} · top-3 OOF acc = {overall_top3:.4f}</div>
</header>
<nav>{"".join(nav_buttons)}</nav>
<div class="page">
{"".join(sections)}
</div>
<script>{JS}</script>
</body>
</html>
""")
print(f"[per-pul-report] wrote {OUT_HTML.relative_to(ROOT)} ({OUT_HTML.stat().st_size//1024} KB)")
print(f"[per-pul-report] open in browser: file://{OUT_HTML}")
