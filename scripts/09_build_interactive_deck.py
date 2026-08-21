#!/usr/bin/env python3
"""
Build the interactive HTML version of the subFinder deck.

Same content as deck.pptx but every chart is an interactive Plotly figure
(hover tooltips, zoom, click-to-toggle legend entries, etc). Tables are
sortable HTML tables.

Output: presentations/deck.html  (single self-contained file ~ a few MB,
Plotly loaded from CDN so file size stays small).

Re-run with:
    python presentations/build_interactive_deck.py
"""
from __future__ import annotations
import json, glob, re
from pathlib import Path
from collections import Counter

import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
PRES = ROOT / "docs"  # release: render into docs/ for GitHub Pages
REP = ROOT / "artifacts"  # release layout
OUT = PRES / "deck.html"

NAVY = "#1a3a5c"; ORANGE = "#e67e22"; SAGE = "#27ae60"; CRIMSON = "#c0392b"
GRAY = "#7f8c8d"; LIGHT = "#ecf0f1"; CHARCOAL = "#2c3e50"

print("[interactive deck] loading data ...")
df_data = pd.read_csv(ROOT / "data/Train_data.csv")
X = df_data["sig_gene_seq"].fillna("").values
y_labels = df_data["high_level_substr"].values
substrates = sorted(set(y_labels))

TOK_RE = re.compile(r"[,|_]")
CAZY_RE = re.compile(r"^(GH|PL|CE|CBM|GT|AA)[0-9]+$")
def tok_cpu(s): return [t for t in TOK_RE.split(str(s)) if t]
def is_cazy(t): return bool(CAZY_RE.match(str(t)))

df_pf = pd.read_csv(REP / "per_fold_metrics.csv")

# Alias map (same structure as build_slides.py)
ALIAS_PROVENANCE = {
    "alpha-glucan":     [("alpha-glucan","exact"), ("starch","collapse"), ("glycogen","collapse"),
                          ("sucrose","collapse"), ("raffinose","collapse"), ("trehalose","collapse"),
                          ("palatinose","collapse"), ("glucooligosaccharide","collapse")],
    "beta-glucan":      [("beta-glucan","exact"), ("cellulose","collapse"), ("cellooligosaccharide","collapse"),
                          ("xyloglucan","collapse"), ("beta-glycan","collapse")],
    "galactan":         [("beta-galactan","collapse"), ("alpha-galactan","collapse")],
    "arabinogalactan":  [("arabinogalactan protein","collapse"), ("arabinan","collapse")],
    "host glycan":      [("host glycan","exact"), ("human milk polysaccharide","collapse"),
                          ("sialic acid","collapse"), ("fucose","collapse")],
    "chitin":           [("chitin","exact"), ("chitosan","collapse"), ("chitooligosaccharide","collapse")],
    "alginate":         [("alginate","exact")],
    "pectin":           [("pectin","exact")],
    "xylan":            [("xylan","exact")],
    "alpha-mannan":     [("alpha-mannan","exact")],
    "beta-mannan":      [("beta-mannan","exact")],
    "fructan":          [("fructan","exact")],
}
lit = pd.read_csv(ROOT / "data/Literature_Data_fam_substrate_mapping.tsv", sep="\t")
lit.columns = [c.strip() for c in lit.columns]
CANON_PROV = {s: {} for s in substrates}
for _, row in lit.iterrows():
    lit_sub = str(row["Substrate_high_level"]).strip()
    fam = row["Family"]
    parts = [p.strip() for p in re.split(r",|and\s+", lit_sub) if p.strip()]
    for part in parts:
        for our_sub, entries in ALIAS_PROVENANCE.items():
            for (alias_name, kind) in entries:
                if part == alias_name:
                    CANON_PROV[our_sub].setdefault(fam, set()).add((alias_name, kind))
CANON = {s: set(CANON_PROV[s].keys()) for s in substrates}

def family_of(short):
    if short in ("cpu__ET500_log2","ftCbow_MM__ET500_sqrt"): return "Ours (shallow)"
    if "BRF100" in short: return "Paper BRF baselines"
    if "__LSTM" in short and "attn" not in short: return "DL: LSTM"
    if "__LSTMattn" in short: return "DL: LSTM+attn"
    if "__JustAttn" in short: return "DL: attention"
    if "__Trans" in short: return "DL: transformer"
    return "?"
df_pf["family"] = df_pf.shorthand.apply(family_of)

# === decoder for shorthand → (featurizer family, featurizer detail, classifier family, classifier detail)
FEATURIZER_MAP = {
    "cpu":       ("CountVec",  "tok_cpu (splits on , | _)"),
    "cv":        ("CountVec",  "paper's tok_comma_pipe (, |)"),
    "ftCbow":    ("FastText",  "CBOW · mean-pooled"),
    "ftCbow_MM": ("FastText",  "CBOW · mean+max concat"),
    "ftSg":      ("FastText",  "skip-gram · mean-pooled"),
    "w2vCbow":   ("Word2Vec",  "CBOW · mean-pooled"),
    "w2vSg":     ("Word2Vec",  "skip-gram · mean-pooled"),
    "d2vDm":     ("Doc2Vec",   "DM · doc-vector"),
    "d2vDbow":   ("Doc2Vec",   "DBOW · doc-vector"),
}
CLASSIFIER_MAP = {
    "ET500_log2": ("OvR(ExtraTrees)",         "n=500, max_features=log2, class_weight=balanced"),
    "ET500_sqrt": ("OvR(ExtraTrees)",         "n=500, max_features=sqrt, class_weight=balanced"),
    "BRF100":     ("OvR(BalancedRF)",         "n=100, class_weight=balanced (paper)"),
    "LSTM":       ("DL: LSTM",                "vanilla LSTM (paper)"),
    "LSTMattn":   ("DL: LSTM+attention",      "LSTM + attention (paper)"),
    "JustAttn":   ("DL: attention",           "attention-only (paper)"),
    "Trans":      ("DL: transformer",         "4-block transformer (paper)"),
}
def decode_shorthand(sh):
    feat_key, clf_key = sh.split("__")
    feat_fam, feat_det = FEATURIZER_MAP.get(feat_key, ("?","?"))
    clf_fam, clf_det = CLASSIFIER_MAP.get(clf_key, ("?","?"))
    return feat_fam, feat_det, clf_fam, clf_det
df_pf["featurizer_family"] = df_pf.shorthand.apply(lambda s: decode_shorthand(s)[0])
df_pf["classifier_family"] = df_pf.shorthand.apply(lambda s: decode_shorthand(s)[2])

PRETTY_FEATURIZER = {
    "cpu":       "CountVec (split , | _)",
    "cv":        "CountVec (paper, split , |)",
    "ftCbow":    "FastText CBOW",
    "ftCbow_MM": "FastText CBOW (mean+max)",
    "ftSg":      "FastText skipgram",
    "w2vCbow":   "Word2Vec CBOW",
    "w2vSg":     "Word2Vec skipgram",
    "d2vDm":     "Doc2Vec DM",
    "d2vDbow":   "Doc2Vec DBOW",
}
PRETTY_CLASSIFIER = {
    "ET500_log2": "ExtraTrees 500 (log2)",
    "ET500_sqrt": "ExtraTrees 500 (sqrt)",
    "BRF100":     "Balanced RF 100",
    "LSTM":       "LSTM",
    "LSTMattn":   "LSTM + attention",
    "JustAttn":   "Attention-only",
    "Trans":      "Transformer (4 blocks)",
}
def pretty_name(sh):
    f, c = sh.split("__")
    return f"{PRETTY_FEATURIZER.get(f, f)}  →  {PRETTY_CLASSIFIER.get(c, c)}"
PRETTY_FAMILY = {
    "Ours (shallow)":      "Our shallow winners (ExtraTrees)",
    "Paper BRF baselines": "Paper baselines (Balanced RF)",
    "DL: LSTM":            "Deep LSTM (paper)",
    "DL: LSTM+attn":       "Deep LSTM + attention (paper)",
    "DL: attention":       "Deep attention-only (paper)",
    "DL: transformer":     "Deep transformer (paper)",
}
BLACK = "#000000"
# Bold-black defaults for all Plotly figures
PLOTLY_FONT_DEFAULTS = dict(family="Helvetica, Arial, sans-serif", color=BLACK, size=14)
PLOTLY_TITLE_FONT    = dict(family="Helvetica, Arial, sans-serif", color=BLACK, size=20, weight=700)
PLOTLY_AXIS_FONT     = dict(family="Helvetica, Arial, sans-serif", color=BLACK, size=14, weight=700)
PLOTLY_TICK_FONT     = dict(family="Helvetica, Arial, sans-serif", color=BLACK, size=12, weight=700)
PLOTLY_LEGEND_FONT   = dict(family="Helvetica, Arial, sans-serif", color=BLACK, size=12, weight=700)
FAMILY_COLOR = {"Ours (shallow)": SAGE, "Paper BRF baselines": NAVY,
                "DL: LSTM": GRAY, "DL: LSTM+attn": ORANGE,
                "DL: attention": CRIMSON, "DL: transformer": "#8e44ad"}
counts = df_pf.shorthand.value_counts()
complete_shorts = set(counts[counts == 25].index)
df_pf_c = df_pf[df_pf.shorthand.isin(complete_shorts)].copy()

# ============================================================================
# CHART 1 — Benchmark leaderboard (horizontal bar chart, sortable in legend by family)
# ============================================================================
print("[chart 1] benchmark leaderboard ...")
agg = df_pf_c.groupby(["shorthand","family"]).agg(
    mean=("acc","mean"), std=("acc","std"),
    min=("acc","min"), max=("acc","max"),
    median=("acc","median")
).reset_index().sort_values("mean", ascending=True)
agg["rank_d"] = list(range(len(agg), 0, -1))
agg["pretty"] = agg.shorthand.apply(pretty_name)
agg["label"]  = agg.apply(lambda r: f"<b>#{int(r['rank_d'])}</b>  {r['pretty']}", axis=1)

fig1 = go.Figure()
for fam in FAMILY_COLOR:
    sub = agg[agg.family == fam]
    if sub.empty: continue
    fig1.add_trace(go.Bar(
        y=sub.label, x=sub["mean"],
        error_x=dict(type="data", array=sub["std"], color=BLACK, thickness=1.4, width=4),
        orientation="h", name=PRETTY_FAMILY.get(fam, fam),
        marker=dict(color=FAMILY_COLOR[fam], line=dict(color=BLACK, width=0.6)),
        text=[f"<b>{m:.4f}</b>" for m in sub["mean"]],
        textposition="outside", textfont=dict(size=11, color=BLACK, weight=700),
        customdata=np.stack([sub["std"], sub["min"], sub["median"], sub["max"], sub.shorthand], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Shorthand: %{customdata[4]}<br>"
            "Family: " + fam + "<br>"
            "Mean: %{x:.4f} ± %{customdata[0]:.4f}<br>"
            "Min / Median / Max: %{customdata[1]:.4f} / %{customdata[2]:.4f} / %{customdata[3]:.4f}"
            "<extra></extra>"
        ),
    ))
fig1.add_vline(x=0.9058, line=dict(color=SAGE, width=1.5, dash="dot"))
# Force y-axis to use our sorted order (Plotly otherwise orders by trace insertion).
# agg is sorted ascending → first label is lowest accuracy → goes to BOTTOM of plot → best at TOP.
fig1.update_layout(
    title=dict(text="<b>Benchmark leaderboard — 25 configurations · 5×5 RSKF · n=25 trials each</b><br>"
                    "<span style='font-size:13px;color:black'><b>Sorted best-on-top.</b> Hover any bar for the full distribution. Click legend entries to filter by family.</span>",
               x=0, font=PLOTLY_TITLE_FONT),
    xaxis=dict(title=dict(text="<b>Mean test accuracy ± 1 SD</b>", font=PLOTLY_AXIS_FONT),
               range=[0.55, 0.99], gridcolor="#dddddd", tickfont=PLOTLY_TICK_FONT, linecolor=BLACK, mirror=False),
    yaxis=dict(automargin=True,
               categoryorder="array", categoryarray=agg.label.tolist(),
               tickfont=dict(family="Helvetica, Arial, sans-serif", color=BLACK, size=11, weight=700)),
    plot_bgcolor="white", paper_bgcolor="white",
    height=820, margin=dict(l=360, r=80, t=100, b=60),
    legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02, font=PLOTLY_LEGEND_FONT,
                title=dict(text="<b>Model family</b>", font=PLOTLY_LEGEND_FONT)),
    font=PLOTLY_FONT_DEFAULTS,
)

# ============================================================================
# CHART 1b — Top-5 podium (horizontal bar with featurizer + classifier labels)
# ============================================================================
print("[chart 1b] top-5 podium ...")
top5 = agg.sort_values("mean", ascending=False).head(5).reset_index(drop=True)
top5["rank"] = np.arange(1, len(top5)+1)
top5["featurizer_fam"] = top5.shorthand.apply(lambda s: decode_shorthand(s)[0])
top5["featurizer_det"] = top5.shorthand.apply(lambda s: decode_shorthand(s)[1])
top5["clf_fam"]        = top5.shorthand.apply(lambda s: decode_shorthand(s)[2])
top5["clf_det"]        = top5.shorthand.apply(lambda s: decode_shorthand(s)[3])

# Build intuitive podium labels — rank + featurizer + classifier (no shorthand)
podium_labels = []
for _, r in top5.iterrows():
    feat = PRETTY_FEATURIZER.get(r.shorthand.split("__")[0], r.featurizer_fam)
    clf  = PRETTY_CLASSIFIER.get(r.shorthand.split("__")[1], r.clf_fam)
    podium_labels.append(f"<b>#{int(r['rank'])}</b>   {feat}<br><b>      →</b>  {clf}")
fig1b = go.Figure()
top5_rev = top5.iloc[::-1].reset_index(drop=True)
labels_rev = list(reversed(podium_labels))
bar_colors = [SAGE if s in ("cpu__ET500_log2","ftCbow_MM__ET500_sqrt") else
              NAVY if "BRF100" in s else ORANGE for s in top5_rev.shorthand]
fig1b.add_trace(go.Bar(
    y=labels_rev, x=top5_rev["mean"], orientation="h",
    error_x=dict(type="data", array=top5_rev["std"], color=BLACK, thickness=1.5, width=4),
    marker=dict(color=bar_colors, line=dict(color=BLACK, width=0.7)),
    text=[f"  <b>{m:.4f} ± {s:.4f}</b>" for m,s in zip(top5_rev["mean"], top5_rev["std"])],
    textposition="outside", textfont=dict(size=13, color=BLACK, weight=700),
    customdata=np.stack([top5_rev.featurizer_fam, top5_rev.featurizer_det,
                          top5_rev.clf_fam, top5_rev.clf_det, top5_rev["std"],
                          top5_rev["min"], top5_rev["max"], top5_rev.shorthand], axis=-1),
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Shorthand: %{customdata[7]}<br>"
        "Featurizer family: %{customdata[0]} · %{customdata[1]}<br>"
        "Classifier family: %{customdata[2]}<br>"
        "Classifier detail: %{customdata[3]}<br>"
        "Mean: %{x:.4f} ± %{customdata[4]:.4f}<br>"
        "Min — Max: %{customdata[5]:.4f} — %{customdata[6]:.4f}"
        "<extra></extra>"
    ),
    showlegend=False,
))
fig1b.update_layout(
    title=dict(text="<b>Top-5 podium — best five of 25 benchmarked configurations</b><br>"
                    "<span style='font-size:13px;color:black'><b>Both #1 and #2 share OvR(ExtraTrees)</b> under different featurizers — "
                    "the win is in the classifier choice. Ranks #3–#5 are paper's Balanced RF baseline with different word-embedding featurizers.</span>",
               x=0, font=PLOTLY_TITLE_FONT),
    xaxis=dict(title=dict(text="<b>Mean test accuracy ± 1 SD  (5×5 RSKF, n=25 trials)</b>", font=PLOTLY_AXIS_FONT),
               range=[0.78, 0.96], gridcolor="#dddddd", tickfont=PLOTLY_TICK_FONT, linecolor=BLACK),
    yaxis=dict(automargin=True, tickfont=PLOTLY_TICK_FONT),
    plot_bgcolor="white", paper_bgcolor="white",
    height=620, margin=dict(l=340, r=140, t=110, b=60),
    font=PLOTLY_FONT_DEFAULTS,
)

# ============================================================================
# CHART 2 — Family-level box plot (vertical with hover)
# ============================================================================
print("[chart 2] family-level box plot ...")
# Sort families by mean DESC; reverse so Plotly puts top of plot = highest accuracy
fam_means = df_pf_c.groupby("family")["acc"].mean().sort_values(ascending=False)
fams_in = fam_means.index.tolist()
fams_plot_order = list(reversed(fams_in))  # plotly puts first item at bottom

def family_blurb(f):
    if f == "Ours (shallow)":      return "shallow tree ensemble — OvR ExtraTrees 500"
    if f == "Paper BRF baselines": return "shallow tree ensemble — paper's Balanced RF 100"
    if f == "DL: LSTM+attn":       return "recurrent + soft attention (paper)"
    if f == "DL: transformer":     return "4-block self-attention (paper)"
    if f == "DL: attention":       return "attention-only, no recurrence (paper)"
    if f == "DL: LSTM":            return "vanilla recurrence (paper)"
    return f

fig2 = go.Figure()
# Plain-text labels only (HTML in y-categories breaks Plotly's horizontal-box layout)
y_labels_fig2 = [PRETTY_FAMILY.get(fam, fam) for fam in fams_plot_order]
for fam in fams_plot_order:
    sub = df_pf_c[df_pf_c.family == fam]
    label = PRETTY_FAMILY.get(fam, fam)
    fig2.add_trace(go.Box(
        x=sub.acc, y=[label] * len(sub),
        name=label, orientation="h",
        marker=dict(color=FAMILY_COLOR[fam], line=dict(color=BLACK, width=0.6)),
        line=dict(color=BLACK, width=1.5),
        fillcolor=FAMILY_COLOR[fam],
        boxmean=True,
        hovertemplate=(
            "<b>" + label + "</b><br>"
            "Config: %{customdata[0]}<br>"
            "Seed: %{customdata[1]} · Fold: %{customdata[2]}<br>"
            "Accuracy: <b>%{x:.4f}</b><extra></extra>"
        ),
        customdata=np.stack([sub.shorthand.apply(pretty_name).values, sub.repeat_seed.values, sub.fold.values], axis=-1),
    ))

# Best config per family + family blurb annotation (right edge), keyed on plain-text label
best_per_fam = df_pf_c.groupby(["family","shorthand"])["acc"].mean().reset_index()
top_by_fam = best_per_fam.loc[best_per_fam.groupby("family")["acc"].idxmax()].set_index("family")
annots = []
for fam in fams_plot_order:
    label = PRETTY_FAMILY.get(fam, fam)
    if fam in top_by_fam.index:
        b = top_by_fam.loc[fam]
        annots.append(dict(x=0.98, y=label, xref="x", yref="y",
                           text=(f"<b>{family_blurb(fam)}</b><br>"
                                  f"<span style='font-size:10px'>best: {b.acc:.4f} · {pretty_name(b.shorthand)}</span>"),
                           showarrow=False, align="left", xanchor="left", bgcolor="white",
                           bordercolor=BLACK, borderwidth=0.6, borderpad=4,
                           font=dict(color=BLACK, size=10)))
fig2.update_layout(
    title=dict(text="<b>Accuracy distribution by model family — sorted high-to-low</b><br>"
                    "<span style='font-size:13px;color:black'><b>Only the shallow ExtraTrees family sits above 0.88.</b> Each point is one (config, seed, fold) trial — hover for the exact config.</span>",
               x=0, font=PLOTLY_TITLE_FONT),
    xaxis=dict(title=dict(text="<b>Test accuracy per trial</b>", font=PLOTLY_AXIS_FONT),
               gridcolor="#dddddd", range=[0.36, 1.32], tickfont=PLOTLY_TICK_FONT, linecolor=BLACK),
    yaxis=dict(automargin=True,
               categoryorder="array", categoryarray=y_labels_fig2,
               tickfont=dict(family="Helvetica, Arial, sans-serif", color=BLACK, size=12, weight=700)),
    annotations=annots, hovermode="closest",
    plot_bgcolor="white", paper_bgcolor="white",
    height=620, margin=dict(l=260, r=40, t=110, b=60),
    showlegend=False,
    font=PLOTLY_FONT_DEFAULTS,
)

# ============================================================================
# CHART 3 — Reproducibility deltas
# ============================================================================
print("[chart 3] reproducibility ...")
orig = pd.read_csv(ROOT / "artifacts/original_benchmark_per_fold_metrics.csv")[["shorthand","repeat_seed","fold","acc"]]
# Compare rep_1 BENCHMARK (in artifacts/) against the latest available rep_2
# (model-init reproducibility — REPRO_REP_SEED=1000 vs 2000, same data splits).
_FIG3_RETRAIN_SRC = ROOT / "reproducibility/rep_2"
if not (_FIG3_RETRAIN_SRC / "predictions").exists():
    _FIG3_RETRAIN_SRC = REP
retrained_rows = []
for mj in glob.glob(str(_FIG3_RETRAIN_SRC/"predictions/*/r*_f*/meta.json")):
    retrained_rows.append(json.load(open(mj)))
df_retr = pd.DataFrame(retrained_rows).rename(columns={"seed":"repeat_seed","test_acc":"acc"})[["shorthand","repeat_seed","fold","acc"]]
# Only keep configs with COMPLETE 25-trial retrain so the orig-vs-retrain comparison is apples-to-apples.
# (Some transformer configs in rep_1 are partial/missing — those are excluded with a logged note.)
ret_counts = df_retr.groupby("shorthand").size()
complete_retrain = set(ret_counts[ret_counts == 25].index)
incomplete = set(df_retr.shorthand.unique()) - complete_retrain
missing_in_retrain = set(orig.shorthand.unique()) - set(df_retr.shorthand.unique())
print(f"  reproducibility data: {len(orig)} orig rows, {len(df_retr)} retrained rows")
print(f"  configs with FULL 25-trial retrain: {len(complete_retrain)}/25")
if incomplete: print(f"  partial retrain (excluded): {sorted(incomplete)}")
if missing_in_retrain: print(f"  no retrain at all (excluded): {sorted(missing_in_retrain)}")
df_retr = df_retr[df_retr.shorthand.isin(complete_retrain)]
orig_for_repro = orig[orig.shorthand.isin(complete_retrain)]
merged = orig_for_repro.merge(df_retr, on=["shorthand","repeat_seed","fold"], suffixes=("_orig","_retr"))
agg2 = merged.groupby("shorthand").agg(orig=("acc_orig","mean"), retr=("acc_retr","mean")).reset_index()
per_fold_abs = merged.assign(d=(merged.acc_retr-merged.acc_orig).abs()).groupby("shorthand")["d"].max().reset_index().rename(columns={"d":"max_abs_fold_delta"})
agg2 = agg2.merge(per_fold_abs, on="shorthand")
agg2["delta"] = agg2.retr - agg2.orig
agg2["abs_delta"] = agg2.delta.abs()
agg2["pretty"] = agg2.shorthand.apply(pretty_name)
fig3 = go.Figure()

# Sort by |Δ| DESCENDING so largest |Δ| at top (most-different from rep_1 first)
agg2 = agg2.sort_values("abs_delta", ascending=False).reset_index(drop=True)

# Bucket each config + assign one color per bar
def _bucket_color(sh):
    if sh in ("cpu__ET500_log2", "ftCbow_MM__ET500_sqrt"):
        return SAGE, "ExtraTrees winner"
    if "BRF100" in sh:
        return NAVY, "Balanced RF baseline"
    return ORANGE, "DL config (LSTM/LSTMattn/JustAttn/Trans)"
agg2["color"] = agg2.shorthand.apply(lambda s: _bucket_color(s)[0])
agg2["bucket_label"] = agg2.shorthand.apply(lambda s: _bucket_color(s)[1])

# Reverse rows so plotly draws TOP→BOTTOM in agg2 row order
agg2_plot = agg2.iloc[::-1].reset_index(drop=True)

# ONE bar per config, colored by bucket
fig3.add_trace(go.Bar(
    y=agg2_plot.pretty, x=agg2_plot.delta, orientation="h",
    marker=dict(color=list(agg2_plot.color), line=dict(color=BLACK, width=0.6)),
    showlegend=False,
    customdata=np.stack([agg2_plot.orig, agg2_plot.retr, agg2_plot.delta,
                          agg2_plot.max_abs_fold_delta, agg2_plot.shorthand,
                          agg2_plot.bucket_label], axis=-1),
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Bucket: %{customdata[5]}<br>"
        "rep_1 (benchmark) 5×5 mean: %{customdata[0]:.4f}<br>"
        "rep_2 (other rep) 5×5 mean:  %{customdata[1]:.4f}<br>"
        "Mean Δ:                      %{customdata[2]:+.4f}<br>"
        "Max single-fold |Δ|:         %{customdata[3]:.4f}<br>"
        "Shorthand: %{customdata[4]}<extra></extra>"
    ),
))

# Legend-only invisible traces (one per bucket) so user sees the color key
for label, color in [
    ("ExtraTrees winners (random_state-seeded → identical trees → Δ ≈ 0)", SAGE),
    ("Balanced RF baselines (random_state-seeded; thread-order non-det only)", NAVY),
    ("DL configs (weight init + GPU op-order non-determinism)", ORANGE),
]:
    fig3.add_trace(go.Bar(y=[None], x=[None], orientation="h",
                          name=label, marker=dict(color=color),
                          showlegend=True, hoverinfo="skip"))
fig3.add_vline(x=0, line=dict(color=BLACK, width=1.5))
fig3.update_layout(
    title=dict(text="<b>Reproducibility — rep_2 (REPRO_REP_SEED=2000) vs rep_1 (REPRO_REP_SEED=1000) — sorted by |Δ|</b><br>"
                    "<span style='font-size:13px;color:black'>Data splits FIXED at 5×5 RSKF; only model-init seed varies across reps. <b>Our ExtraTrees winners are random_state-seeded → Δ = 0.</b> DL deltas come from weight-init + GPU op-order non-determinism; rankings of top configs are stable.</span>",
               x=0, font=PLOTLY_TITLE_FONT),
    xaxis=dict(title=dict(text="<b>Δ accuracy   (rep_2 − rep_1 5×5 mean)</b>", font=PLOTLY_AXIS_FONT),
               range=[-0.045, 0.025], gridcolor="#dddddd", tickfont=PLOTLY_TICK_FONT, linecolor=BLACK),
    yaxis=dict(automargin=True,
               categoryorder="array", categoryarray=agg2_plot.pretty.tolist(),
               tickfont=dict(family="Helvetica, Arial, sans-serif", color=BLACK, size=11, weight=700)),
    plot_bgcolor="white", paper_bgcolor="white",
    height=820, margin=dict(l=360, r=40, t=110, b=70),
    legend=dict(orientation="h", yanchor="top", y=-0.10, xanchor="center", x=0.5,
                font=PLOTLY_LEGEND_FONT,
                title=dict(text="<b>Why retrain Δ ≠ 0?</b>  ", font=PLOTLY_LEGEND_FONT)),
    font=PLOTLY_FONT_DEFAULTS,
)

# ============================================================================
# CHART 4 — Confusion matrix heatmap
# ============================================================================
print("[chart 4] confusion matrix ...")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = np.array([None]*len(X), dtype=object)
classes = None
for fold, (_, te) in enumerate(skf.split(X, y_labels)):
    d = np.load(REP/f"predictions/cpu__ET500_log2/r42_f{fold}/probs_test.npz", allow_pickle=True)
    probs = d["probs"]; classes_ = d["classes"]
    if classes is None: classes = list(classes_)
    y_pred[te] = np.array([classes_[i] for i in probs.argmax(1)])
cm_raw = confusion_matrix(y_labels, y_pred, labels=substrates)
cm = cm_raw / cm_raw.sum(axis=1, keepdims=True)
p, r, f, sup = precision_recall_fscore_support(y_labels, y_pred, labels=substrates, average=None, zero_division=0)

text_grid = [[f"{cm_raw[i,j]}<br>{cm[i,j]*100:.1f}%" for j in range(len(substrates))] for i in range(len(substrates))]
fig4 = go.Figure(data=go.Heatmap(
    z=cm, x=substrates, y=substrates,
    text=cm_raw, texttemplate="%{text}",
    colorscale="Blues", showscale=True,
    hovertemplate="True: <b>%{y}</b><br>Predicted: <b>%{x}</b><br>Count: %{text}<br>Row-normalized: %{z:.3f}<extra></extra>",
    colorbar=dict(title="row<br>fraction"),
))
fig4.update_layout(
    title=dict(text=f"<b>Confusion matrix — cpu__ET500_log2 (seed-42 5-fold OOF, n=1030, acc={(y_pred==y_labels).mean():.4f})</b><br>"
                    "<span style='font-size:12px;color:#7f8c8d'>Cells show raw counts; color = row-normalized rate.</span>",
               x=0, font=dict(color=CHARCOAL, size=18)),
    xaxis_title="Predicted substrate", yaxis_title="True substrate",
    xaxis=dict(side="bottom", tickangle=-45),
    yaxis=dict(autorange="reversed"),
    height=700, margin=dict(l=140, r=40, t=80, b=120),
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="Helvetica, Arial, sans-serif", color=CHARCOAL),
)

# per-substrate F1 table
per_sub = pd.DataFrame({"substrate": substrates, "n_test": sup,
                          "precision": p, "recall": r, "F1": f}).sort_values("F1", ascending=False)

# ============================================================================
# CHART 5 — sig genes table (HTML, color coded) — uses sig_tbl logic
# ============================================================================
print("[chart 5] sig genes table ...")
fi = pd.read_csv(ROOT/"paper/tables/table4_signature_genes_per_substrate.csv")
def lit_status_with_basis(s, t):
    if not is_cazy(t):
        return ("non-CAZy", "non-CAZy: lit cannot adjudicate")
    if t not in CANON[s]:
        return ("miss", "CAZy not in lit-canon")
    provs = CANON_PROV[s][t]
    if any(k == "exact" for _, k in provs):
        return ("LIT-exact", "1:1 exact lit match")
    alias_names = ", ".join(sorted({a for a,_ in provs}))
    return ("LIT-collapse", f"via collapse ← {alias_names}")

rows_sig = []
for s in substrates:
    top3 = fi[fi.substrate == s].sort_values("importance", ascending=False).head(3)
    for _, r in top3.iterrows():
        mark, note = lit_status_with_basis(s, r.feature)
        rows_sig.append({"substrate": s, "feature": r.feature, "importance": r.importance,
                          "mark": mark, "lit_status": note})
sig_tbl = pd.DataFrame(rows_sig)
n_cazy_check = sum(1 for r in rows_sig if r["mark"] in ("LIT-exact","LIT-collapse","miss"))
n_cazy_in_lit = sum(1 for r in rows_sig if r["mark"].startswith("LIT"))
n_exact = sum(1 for r in rows_sig if r["mark"] == "LIT-exact")
n_collapse = sum(1 for r in rows_sig if r["mark"] == "LIT-collapse")
n_non_cazy = sum(1 for r in rows_sig if r["mark"] == "non-CAZy")

# ============================================================================
# CHART 6 — Lit validation bars (two-panel)
# ============================================================================
print("[chart 6] population lit validation — TRUE-class attribution ...")
# Use TRUE-class ablation (Δ-prob w.r.t. true substrate) so totals match slide 13.
oof = pd.read_csv(REP/"ablation/sig_gene_ablation_oof_outer42_groundtruth_calibrated.csv")
# Alias the calibrated argmax probability to 'prob' for chart 7 (example PULs) compatibility
if "prob" not in oof.columns:
    oof = oof.rename(columns={"prob_pred_cal": "prob"})
# `correct` (true == pred subset) still used for chart 7's example-PUL cherry-pick
correct = oof[oof.true == oof.pred].copy()

def per_pul_anyhit_true():
    rows = []
    for K in (1,3,5):
        n_elig = 0; n_hit = 0
        for _, r in oof.iterrows():
            cs = CANON[r.true]  # canon for TRUE substrate
            pul_toks = set(tok_cpu(X[r.idx]))
            if not (pul_toks & cs): continue
            n_elig += 1
            top = str(r["top1"]) if K==1 else (str(r["top3"]) if K==3 else str(r["top5"]))
            tokens = set(top.split(";"))
            if tokens & cs: n_hit += 1
        rows.append({"K": K, "n_eligible": n_elig, "n_hit": n_hit, "rate": n_hit/n_elig})
    return pd.DataFrame(rows)

def lit_gene_coverage_true():
    rows = []
    for K in (1,3,5):
        in_scope = 0; covered = 0
        flagged = {s: set() for s in substrates}
        # Flag using TRUE substrate (not predicted)
        for _, r in oof.iterrows():
            top = str(r["top1"]) if K==1 else (str(r["top3"]) if K==3 else str(r["top5"]))
            flagged[r.true] |= set(top.split(";")) & CANON[r.true]
        for s in substrates:
            scope = set()
            for _, r in oof[oof.true == s].iterrows():
                scope |= set(tok_cpu(X[r.idx])) & CANON[s]
            in_scope += len(scope)
            covered += len(scope & flagged[s])
        rows.append({"K": K, "in_scope": in_scope, "covered": covered, "rate": covered/in_scope})
    return pd.DataFrame(rows)

anyhit_df = per_pul_anyhit_true(); scope_df = lit_gene_coverage_true()

fig6 = make_subplots(rows=1, cols=2,
                     subplot_titles=("PUL-level recall (per-PUL any-hit)", "Gene-level recall (scope coverage)"),
                     horizontal_spacing=0.18)
fig6.add_trace(go.Bar(
    x=anyhit_df.K, y=anyhit_df.rate*100,
    text=[f"<b>{r*100:.1f}%</b><br>{h}/{e}" for r,h,e in zip(anyhit_df.rate, anyhit_df.n_hit, anyhit_df.n_eligible)],
    textposition="outside", textfont=dict(size=12, color=CHARCOAL),
    marker=dict(color=SAGE, line=dict(color="white", width=1)),
    customdata=np.stack([anyhit_df.n_hit, anyhit_df.n_eligible], axis=-1),
    hovertemplate="K=%{x}<br>%{customdata[0]} / %{customdata[1]} eligible PULs<br>%{y:.1f}%<extra></extra>",
    name="any-hit",
), row=1, col=1)
fig6.add_trace(go.Bar(
    x=scope_df.K, y=scope_df.rate*100,
    text=[f"<b>{r*100:.1f}%</b><br>{c}/{s}" for r,c,s in zip(scope_df.rate, scope_df.covered, scope_df.in_scope)],
    textposition="outside", textfont=dict(size=12, color=CHARCOAL),
    marker=dict(color=NAVY, line=dict(color="white", width=1)),
    customdata=np.stack([scope_df.covered, scope_df.in_scope], axis=-1),
    hovertemplate="K=%{x}<br>%{customdata[0]} / %{customdata[1]} in-scope gene-substrate pairs<br>%{y:.1f}%<extra></extra>",
    name="scope-coverage",
), row=1, col=2)
fig6.update_xaxes(title_text="<b>top-K ablation tokens (Δ_TRUE-class)</b>", tickmode="array", tickvals=[1,3,5], row=1, col=1,
                  tickfont=PLOTLY_TICK_FONT, linecolor=BLACK)
fig6.update_xaxes(title_text="<b>top-K ablation tokens (Δ_TRUE-class)</b>", tickmode="array", tickvals=[1,3,5], row=1, col=2,
                  tickfont=PLOTLY_TICK_FONT, linecolor=BLACK)
fig6.update_yaxes(title_text="<b>% eligible PULs</b>", range=[0, 115], row=1, col=1, gridcolor="#dddddd",
                  tickfont=PLOTLY_TICK_FONT, linecolor=BLACK)
fig6.update_yaxes(title_text="<b>% in-scope canonical genes</b>", range=[0, 115], row=1, col=2, gridcolor="#dddddd",
                  tickfont=PLOTLY_TICK_FONT, linecolor=BLACK)
fig6.update_layout(
    title=dict(text="<b>Population-level signature-gene validation — TRUE-class attribution</b><br>"
                    f"<span style='font-size:13px;color:black'>n = {int(anyhit_df.iloc[0].n_eligible)} eligible PULs · n = {int(scope_df.iloc[0].in_scope)} in-scope gene-substrate pairs. "
                    "Same method as Slide 13 — K=3 totals match those funnels exactly.</span>",
               x=0, font=PLOTLY_TITLE_FONT),
    showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
    height=500, margin=dict(l=80, r=40, t=110, b=70),
    font=PLOTLY_FONT_DEFAULTS,
)

# ============================================================================
# CHART 7 — example PULs as interactive table
# ============================================================================
print("[chart 7] example PULs ...")
correct["pul_tokens"] = correct.idx.apply(lambda i: set(tok_cpu(X[i])))
correct["n_canon_top3"] = correct.apply(lambda r: sum(1 for t in str(r.top3).split(";") if t in CANON[r.pred]), axis=1)
correct["n_canon_top5"] = correct.apply(lambda r: sum(1 for t in str(r.top5).split(";") if t in CANON[r.pred]), axis=1)
correct_sorted = correct.sort_values(["n_canon_top3","n_canon_top5","prob"], ascending=[False,False,False])
picks = []
seen_subs = Counter()
for _, r in correct_sorted.iterrows():
    if seen_subs[r.pred] >= 2: continue
    if len(picks) >= 10: break
    picks.append(r.to_dict()); seen_subs[r.pred] += 1
ex = pd.DataFrame(picks)

def annot_top5_html(s, top5):
    cs = CANON[s]; out = []
    for piece in str(top5).split(";")[:5]:
        if not piece: continue
        tok = piece.split(":")[0].rstrip("*")
        if tok in cs:        cls = "lit"
        elif is_cazy(tok):   cls = "cazy"
        else:                cls = "acc"
        out.append(f'<span class="tag {cls}">{piece}</span>')
    return " ".join(out)

def short_pul(idx, max_chars=90):
    s = str(X[idx]).strip()
    return s if len(s) <= max_chars else s[:max_chars-3] + "..."

ex_rows = []
for _, r in ex.iterrows():
    p_val = f"{(1-r.prob)**11:.1e}"
    true_sub = r.get("true", r.pred)
    ex_rows.append({
        "PUL idx": int(r.idx),
        "PUL sequence (truncated)": short_pul(r.idx),
        "True substrate": true_sub,
        "Predicted substrate": r.pred,
        "match": (true_sub == r.pred),
        "Max prob": f"{r.prob:.3f}",
        "p-value": p_val,
        "Top-5 signature genes": annot_top5_html(r.pred, r.top5),
        "#LIT / 5": int(r.n_canon_top5),
    })

# ============================================================================
# CHART 8 — calibration reliability (4-line plot)
# ============================================================================
print("[chart 8] calibration ...")
cal_best = np.load(REP/"calibration/oof_outer42_best_of_both.npz", allow_pickle=True)
cal_cv5 = np.load(REP/"calibration/oof_outer42_calibration.npz", allow_pickle=True)
y_int = cal_best["y_true"]

def reliability(probs, n_bins=10):
    conf = probs.max(1); correct_ = (probs.argmax(1) == y_int).astype(float)
    edges = np.linspace(0,1,n_bins+1); mids = (edges[1:]+edges[:-1])/2
    accs = np.zeros(n_bins); confs = np.zeros(n_bins); ns = np.zeros(n_bins)
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i+1]) if i<n_bins-1 else (conf >= edges[i]) & (conf <= edges[i+1])
        if m.sum():
            accs[i] = correct_[m].mean(); confs[i] = conf[m].mean(); ns[i] = m.sum()
    return mids, accs, confs, ns

def ece(probs, n_bins=10):
    mids, accs, confs, ns = reliability(probs, n_bins)
    return float(np.sum(ns * np.abs(accs - confs)) / ns.sum())

def acc(p): return float((p.argmax(1)==y_int).mean())

T_mean = float(cal_best['T_per_fold'].mean())
methods = [
    ("uncalibrated", cal_best["probs_uncal"], GRAY),
    (f"temperature T={T_mean:.2f}", cal_best["probs_temp"], SAGE),
    ("isotonic (CalibratedCV cv=5)", cal_cv5["probs_iso"], ORANGE),
    ("sigmoid / Platt", cal_cv5["probs_sig"], CRIMSON),
]

fig8 = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                    subplot_titles=("Reliability diagram (10-bin)", "Metrics"),
                    horizontal_spacing=0.12, specs=[[{"type":"scatter"}, {"type":"table"}]])
for name, probs, color in methods:
    mids, accs, _, ns = reliability(probs)
    mask = ns > 0
    fig8.add_trace(go.Scatter(
        x=mids[mask], y=accs[mask], mode="lines+markers",
        name=f"{name} (ECE={ece(probs):.3f})",
        line=dict(color=color, width=2),
        marker=dict(size=8, color=color, line=dict(color="white", width=1)),
        hovertemplate="bin midpoint %{x:.2f}<br>empirical accuracy %{y:.3f}<extra></extra>",
    ), row=1, col=1)
fig8.add_trace(go.Scatter(
    x=[0,1], y=[0,1], mode="lines", line=dict(color=CHARCOAL, width=1, dash="dash"),
    name="perfect calibration", showlegend=True,
), row=1, col=1)

tbl_metrics = [(name, f"{acc(p):.4f}", f"{ece(p):.4f}", note)
               for (name, p, _), note in zip(methods, [
                   "raw OvR-ET output",
                   "preserves argmax exactly",
                   "can re-rank classes",
                   "worse than uncalibrated",
               ])]
fig8.add_trace(go.Table(
    header=dict(values=["<b>method</b>", "<b>accuracy</b>", "<b>ECE (10-bin)</b>", "<b>note</b>"],
                fill_color=NAVY, font=dict(color="white", size=11), align="left"),
    cells=dict(values=list(zip(*tbl_metrics)),
               fill_color=[["white", "#e0f3e8", "white", "white"]*1],
               align="left", font=dict(color=CHARCOAL, size=10)),
), row=1, col=2)
fig8.update_xaxes(title_text="predicted confidence", range=[0,1], gridcolor="#eee", row=1, col=1)
fig8.update_yaxes(title_text="empirical accuracy", range=[0,1], gridcolor="#eee", row=1, col=1)
fig8.update_layout(
    title=dict(text="<b>Probability calibration — seed-42 OOF (n=1030)</b><br>"
                    "<span style='font-size:12px;color:#7f8c8d'>Temperature scaling halves the ECE while preserving accuracy exactly.</span>",
               x=0, font=dict(color=CHARCOAL, size=18)),
    plot_bgcolor="white", paper_bgcolor="white",
    height=500, margin=dict(l=70, r=40, t=100, b=60),
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.27, font=dict(size=10)),
    font=dict(family="Helvetica, Arial, sans-serif", color=CHARCOAL),
)

# ============================================================================
# CHART 8b — Per-substrate signature-gene precision / per-PUL hit / scope-recall (K=3)
# ============================================================================
print("[chart 8b] per-substrate sig-gene FUNNEL (GROUND-TRUTH-class attribution) ...")
oof_gt = pd.read_csv(REP/"ablation/sig_gene_ablation_oof_outer42_groundtruth_calibrated.csv")
K = 3
sig_rows_t = []
for s in substrates:
    test_of_s = oof_gt[oof_gt.true == s]
    n_total = len(test_of_s)
    n_elig = 0; n_hit = 0
    for _, r in test_of_s.iterrows():
        toks = set(tok_cpu(X[r.idx]))
        if toks & CANON[s]:
            n_elig += 1
            top = set(str(r[f"top{K}"]).split(";"))
            if top & CANON[s]: n_hit += 1
    hit_rate = (n_hit / n_elig) if n_elig else 0.0
    lit_n     = len(CANON[s])
    in_scope  = set()
    for _, r in test_of_s.iterrows():
        in_scope |= set(tok_cpu(X[r.idx])) & CANON[s]
    n_inscope = len(in_scope)
    flagged = set()
    for _, r in test_of_s.iterrows():
        flagged |= set(str(r[f"top{K}"]).split(";")) & CANON[s]
    n_flag = len(in_scope & flagged)
    cov_rate = (n_flag / n_inscope) if n_inscope else 0.0
    sig_rows_t.append(dict(substrate=s,
        n_total=n_total, n_eligible=n_elig, n_pul_hit_at_K=n_hit, pul_hit_rate=hit_rate,
        lit_canon_size=lit_n, n_in_scope=n_inscope, n_flagged_at_K=n_flag, scope_recall=cov_rate,
    ))
sig_pr = pd.DataFrame(sig_rows_t).sort_values("pul_hit_rate", ascending=False)
sig_pr.to_csv(PRES/"tables/tab_per_substrate_sig_pr.csv", index=False)

# Two-panel funnel chart: LEFT = PUL view, RIGHT = gene view (ground-truth-class)
from plotly.subplots import make_subplots as _ms
fig8b = _ms(rows=1, cols=2,
            subplot_titles=("<b>PUL view (counts)</b>: test → eligible → hit @3 (NO correctness gate)",
                            "<b>Gene view (counts)</b>: lit-canon → in-scope → flagged @3"),
            horizontal_spacing=0.18)
fig8b.add_trace(go.Bar(y=sig_pr.substrate, x=sig_pr.n_total, orientation="h", name="total test PULs (TRUE = substrate)",
                       marker=dict(color="#cccccc", line=dict(color=BLACK, width=0.4)),
                       hovertemplate="<b>%{y}</b><br>total test PULs (TRUE substrate): %{x}<extra></extra>"), row=1, col=1)
fig8b.add_trace(go.Bar(y=sig_pr.substrate, x=sig_pr.n_eligible, orientation="h", name="eligible (lit-canon present in PUL)",
                       marker=dict(color=NAVY, line=dict(color=BLACK, width=0.4)),
                       hovertemplate="<b>%{y}</b><br>eligible (lit-canon present in PUL): %{x}<extra></extra>"), row=1, col=1)
fig8b.add_trace(go.Bar(y=sig_pr.substrate, x=sig_pr.n_pul_hit_at_K, orientation="h", name="top-3 sig gene for TRUE class is canonical",
                       marker=dict(color=SAGE, line=dict(color=BLACK, width=0.4)),
                       text=[f"<b>{int(h)}/{int(e)} = {r*100:.0f}%</b>"
                             for h,e,r in zip(sig_pr.n_pul_hit_at_K, sig_pr.n_eligible, sig_pr.pul_hit_rate)],
                       textposition="outside", textfont=dict(size=10, color=BLACK, weight=700),
                       hovertemplate="<b>%{y}</b><br>hit @K=3 (TRUE-class attribution): %{x}<br>(rate vs eligible: %{text})<extra></extra>"),
                       row=1, col=1)

fig8b.add_trace(go.Bar(y=sig_pr.substrate, x=sig_pr.lit_canon_size, orientation="h", name="lit-canon (after collapse)",
                       marker=dict(color="#cccccc", line=dict(color=BLACK, width=0.4)),
                       hovertemplate="<b>%{y}</b><br>lit-canon CAZy families: %{x}<extra></extra>"), row=1, col=2)
fig8b.add_trace(go.Bar(y=sig_pr.substrate, x=sig_pr.n_in_scope, orientation="h", name="in-scope (appears in test PULs of substrate)",
                       marker=dict(color=ORANGE, line=dict(color=BLACK, width=0.4)),
                       hovertemplate="<b>%{y}</b><br>in-scope: %{x}<extra></extra>"), row=1, col=2)
fig8b.add_trace(go.Bar(y=sig_pr.substrate, x=sig_pr.n_flagged_at_K, orientation="h", name="flagged in top-3 sig genes for TRUE class anywhere",
                       marker=dict(color=SAGE, line=dict(color=BLACK, width=0.4)),
                       text=[f"<b>{int(f)}/{int(i)} = {r*100:.0f}%</b>"
                             for f,i,r in zip(sig_pr.n_flagged_at_K, sig_pr.n_in_scope, sig_pr.scope_recall)],
                       textposition="outside", textfont=dict(size=10, color=BLACK, weight=700),
                       hovertemplate="<b>%{y}</b><br>flagged in top-3 anywhere (TRUE-class): %{x}<extra></extra>"),
                       row=1, col=2)

T_total=int(sig_pr.n_total.sum())
T_elig=int(sig_pr.n_eligible.sum()); T_hit=int(sig_pr.n_pul_hit_at_K.sum())
T_lit=int(sig_pr.lit_canon_size.sum()); T_isc=int(sig_pr.n_in_scope.sum()); T_flg=int(sig_pr.n_flagged_at_K.sum())

fig8b.update_xaxes(title=dict(text="<b>number of PULs</b>", font=PLOTLY_AXIS_FONT),
                   tickfont=PLOTLY_TICK_FONT, gridcolor="#dddddd", linecolor=BLACK, row=1, col=1,
                   range=[0, sig_pr.n_total.max() * 1.35])
fig8b.update_xaxes(title=dict(text="<b>number of lit-canonical CAZy families</b>", font=PLOTLY_AXIS_FONT),
                   tickfont=PLOTLY_TICK_FONT, gridcolor="#dddddd", linecolor=BLACK, row=1, col=2,
                   range=[0, sig_pr.lit_canon_size.max() * 1.45])
fig8b.update_yaxes(autorange="reversed", automargin=True,
                   tickfont=dict(family="Helvetica, Arial, sans-serif", color=BLACK, size=11, weight=700))
fig8b.update_layout(
    title=dict(text=f"<b>Per-substrate sig-gene FUNNEL (K=3, TRUE-class attribution)</b><br>"
                    f"<span style='font-size:13px;color:black'>Δ-prob computed w.r.t. true class — decouples attribution from classification. "
                    f"Totals: PUL view {T_total}→{T_elig}→{T_hit} ({T_hit/T_elig*100:.1f}% hit), "
                    f"gene view {T_lit}→{T_isc}→{T_flg} ({T_flg/T_isc*100:.1f}% scope recall).</span>",
               x=0, font=PLOTLY_TITLE_FONT),
    barmode="overlay", bargap=0.20,
    plot_bgcolor="white", paper_bgcolor="white",
    height=720, margin=dict(l=140, r=80, t=120, b=80),
    legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5,
                font=PLOTLY_LEGEND_FONT),
    font=PLOTLY_FONT_DEFAULTS,
    hovermode="closest",
)

# ============================================================================
# CHART 8c — Test-PUL OOV proportion (CountVec_cpu) vs accuracy
# ============================================================================
print("[chart 8c] per-PUL OOV proportion vs accuracy ...")
from sklearn.feature_extraction.text import CountVectorizer as _CV
oov_props = np.zeros(len(X)); n_toks = np.zeros(len(X), dtype=int)
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, te_idx) in enumerate(skf2.split(X, y_labels)):
    _cv = _CV(tokenizer=tok_cpu, token_pattern=None, lowercase=False)
    _cv.fit(X[tr_idx])
    _vocab = set(_cv.vocabulary_.keys())
    for idx in te_idx:
        toks = tok_cpu(X[idx])
        n_toks[idx] = len(toks)
        oov_props[idx] = sum(1 for t in toks if t not in _vocab) / max(len(toks), 1)
correct_mask = (y_pred == y_labels)
buckets_def = [(0.0, 0.0, "0%"),
               (0.001, 0.05, "0–5%"),
               (0.05, 0.10, "5–10%"),
               (0.10, 0.25, "10–25%"),
               (0.25, 1.01, "≥25%")]
buc_rows = []
for lo, hi, lab in buckets_def:
    m = (oov_props == 0) if lo == 0 == hi else (oov_props >= lo) & (oov_props < hi)
    buc_rows.append((lab, int(m.sum()),
                     float(correct_mask[m].mean()) if m.sum() else 0.0,
                     float(oov_props[m].mean()*100) if m.sum() else 0.0))
buc_df = pd.DataFrame(buc_rows, columns=["bucket","n_PULs","accuracy","mean_OOV_pct"])
buc_df.to_csv(PRES/"tables/tab_oov_vs_accuracy.csv", index=False)
from scipy.stats import pointbiserialr as _pbr2
r_oov, p_oov = _pbr2(correct_mask.astype(int), oov_props)

bar_colors_8c = [SAGE if r[2] >= 0.9 else ORANGE if r[2] >= 0.7 else CRIMSON for r in buc_rows]
# Uniform-width bars — bucket density is shown via the n= annotation under each bar.
total_n_8c = sum(r[1] for r in buc_rows)
fig8c = go.Figure(data=go.Bar(
    x=[r[0] for r in buc_rows], y=[r[2] for r in buc_rows],
    width=[0.62] * len(buc_rows),
    marker=dict(color=bar_colors_8c, line=dict(color=BLACK, width=0.8)),
    text=[f"<b>{r[2]:.3f}</b>" for r in buc_rows], textposition="outside",
    textfont=dict(size=14, color=BLACK, weight=700),
    customdata=np.array(buc_rows, dtype=object),
    hovertemplate="<b>OOV bucket: %{x}</b><br>"
                  "n PULs: %{customdata[1]}<br>"
                  "Mean OOV: %{customdata[3]:.2f}%<br>"
                  "Accuracy: <b>%{y:.4f}</b><extra></extra>",
))
fig8c.update_layout(
    title=dict(text="<b>Test-PUL accuracy vs out-of-vocabulary token proportion (CountVec_cpu featurizer, seed-42 5-fold OOF)</b><br>"
                    f"<span style='font-size:13px;color:black'>Point-biserial r(correct, OOV proportion) = {r_oov:.4f}, p = {p_oov:.2e}. Bars are uniform width; bucket size (% of test PULs) is annotated below each bar.</span>",
               x=0, font=PLOTLY_TITLE_FONT),
    xaxis=dict(title=dict(text="<b>Per-PUL out-of-vocabulary token proportion bucket</b>", font=PLOTLY_AXIS_FONT),
               tickmode="array",
               tickvals=[r[0] for r in buc_rows],
               ticktext=[f"<b>{r[0]}</b><br>n={r[1]} PULs ({100.0*r[1]/max(total_n_8c,1):.1f}%) · mean OOV={r[3]:.1f}%" for r in buc_rows],
               tickfont=dict(family="Helvetica, Arial, sans-serif", color=BLACK, size=11, weight=700),
               linecolor=BLACK),
    yaxis=dict(title=dict(text="<b>Test accuracy (fraction correct)</b>", font=PLOTLY_AXIS_FONT),
               range=[0, 1.12], gridcolor="#dddddd",
               tickfont=PLOTLY_TICK_FONT, linecolor=BLACK),
    plot_bgcolor="white", paper_bgcolor="white",
    height=520, margin=dict(l=80, r=40, t=110, b=90),
    showlegend=False,
    font=PLOTLY_FONT_DEFAULTS,
)
print(f"  per-PUL OOV: mean={oov_props.mean()*100:.2f}%  PULs with 0% OOV: {(oov_props == 0).sum()}/{len(X)}")

# ============================================================================
# CHART 9 — per-trial accuracy distribution (violin)
# ============================================================================
print("[chart 9] per-trial accuracy distribution — ALL families (horizontal box+points) ...")
# Show all 6 families sorted high-to-low. Use Box (not Violin) horizontally to avoid
# Plotly's horizontal-violin overlay glitch where the inner box renders as a huge rotated bar.
fam_means9 = df_pf_c.groupby("family")["acc"].mean().sort_values(ascending=False)
fams_by_mean = fam_means9.index.tolist()
fams_for_plot = list(reversed(fams_by_mean))  # Plotly puts first item at bottom
fig9 = go.Figure()
# Single horizontal Box per family with all points jittered inside (boxpoints="all" gives
# the swarm + central-tendency markers we want: median line, IQR box, whiskers, mean).
y_labels_fig9 = [PRETTY_FAMILY.get(fam, fam) for fam in fams_for_plot]
fig9_annots = []
for fam in fams_for_plot:
    sub = df_pf_c[df_pf_c.family == fam]
    n_cfg = sub.shorthand.nunique()
    n_trials = len(sub)
    label = PRETTY_FAMILY.get(fam, fam)
    meta = np.stack([sub.shorthand.apply(pretty_name).values, sub.repeat_seed.values, sub.fold.values], axis=-1)
    fig9.add_trace(go.Box(
        x=sub.acc.values, y=[label]*len(sub), orientation="h",
        name=label, boxmean=True, boxpoints="all", jitter=0.5, pointpos=0,
        fillcolor=FAMILY_COLOR[fam], line=dict(color=BLACK, width=1.3),
        marker=dict(size=5, color=FAMILY_COLOR[fam], line=dict(color=BLACK, width=0.4),
                    opacity=0.78),
        customdata=meta,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>seed %{customdata[1]} · fold %{customdata[2]}<br>"
            "Accuracy: <b>%{x:.4f}</b><extra></extra>"
        ),
        showlegend=False,
    ))
    fig9_annots.append(dict(x=1.005, y=label, xref="x", yref="y", xanchor="left",
                             text=(f"<b>n = {n_cfg} cfg × 25 = {n_trials}</b><br>"
                                    f"<span style='font-size:10px'>mean {sub.acc.mean():.4f} · median {np.median(sub.acc):.4f}<br>"
                                    f"SD {sub.acc.std():.4f} · IQR {np.percentile(sub.acc,75)-np.percentile(sub.acc,25):.4f}</span>"),
                             showarrow=False, align="left", bgcolor="white",
                             bordercolor=BLACK, borderwidth=0.6, borderpad=4,
                             font=dict(color=BLACK, size=10)))

# vertical line for our winner's mean
top_mean = df_pf_c[df_pf_c.shorthand == "cpu__ET500_log2"]["acc"].mean()
fig9.add_vline(x=top_mean, line=dict(color=SAGE, dash="dot", width=2),
               annotation_text=f"<b>winner mean = {top_mean:.4f}</b>",
               annotation_position="bottom",
               annotation_font=dict(color=BLACK, size=12, weight=700))

fig9.update_layout(
    title=dict(text="<b>Per-trial accuracy distribution — ALL 6 model families, sorted high-to-low</b><br>"
                    "<span style='font-size:13px;color:black'><b>Our shallow winners are tighter AND higher</b> than every deep family. Hover each point for the exact config / seed / fold.</span>",
               x=0, font=PLOTLY_TITLE_FONT),
    xaxis=dict(title=dict(text="<b>Test accuracy per (seed, fold) trial</b>", font=PLOTLY_AXIS_FONT),
               range=[0.34, 1.18], gridcolor="#dddddd", tickfont=PLOTLY_TICK_FONT, linecolor=BLACK),
    yaxis=dict(automargin=True,
               categoryorder="array", categoryarray=y_labels_fig9,
               tickfont=dict(family="Helvetica, Arial, sans-serif", color=BLACK, size=12, weight=700),
               linecolor=BLACK),
    annotations=fig9_annots,
    hovermode="closest",
    plot_bgcolor="white", paper_bgcolor="white",
    height=720, margin=dict(l=260, r=40, t=110, b=70),
    font=PLOTLY_FONT_DEFAULTS,
)

# ============================================================================
# APPENDIX AGGREGATES — per-featurizer-family + per-classifier-family means
# ============================================================================
print("[appendix] aggregates ...")
# Full leaderboard with decoded columns
full_lb = df_pf_c.groupby("shorthand").agg(
    mean=("acc","mean"), std=("acc","std"),
    min_=("acc","min"), median=("acc","median"), max_=("acc","max"),
    n=("acc","count"),
).reset_index().sort_values("mean", ascending=False)
full_lb["featurizer_fam"] = full_lb.shorthand.apply(lambda s: decode_shorthand(s)[0])
full_lb["featurizer_det"] = full_lb.shorthand.apply(lambda s: decode_shorthand(s)[1])
full_lb["clf_fam"]        = full_lb.shorthand.apply(lambda s: decode_shorthand(s)[2])
full_lb["clf_det"]        = full_lb.shorthand.apply(lambda s: decode_shorthand(s)[3])
full_lb["rank"]           = np.arange(1, len(full_lb)+1)

# Per-featurizer family aggregate
feat_agg = df_pf_c.assign(featurizer_family=df_pf_c.shorthand.apply(lambda s: decode_shorthand(s)[0])) \
    .groupby("featurizer_family").agg(
        mean_acc=("acc","mean"), std_acc=("acc","std"),
        n_configs=("shorthand", "nunique"), n_trials=("acc","count")
    ).reset_index().sort_values("mean_acc", ascending=False)

# Per-classifier family aggregate
clf_agg = df_pf_c.assign(classifier_family=df_pf_c.shorthand.apply(lambda s: decode_shorthand(s)[2])) \
    .groupby("classifier_family").agg(
        mean_acc=("acc","mean"), std_acc=("acc","std"),
        n_configs=("shorthand","nunique"), n_trials=("acc","count")
    ).reset_index().sort_values("mean_acc", ascending=False)

print("  per-featurizer agg:")
for _, r in feat_agg.iterrows():
    print(f"    {r.featurizer_family:<12} {r.mean_acc:.4f} ± {r.std_acc:.4f}  ({r.n_configs} configs, {r.n_trials} trials)")
print("  per-classifier agg:")
for _, r in clf_agg.iterrows():
    print(f"    {r.classifier_family:<26} {r.mean_acc:.4f} ± {r.std_acc:.4f}  ({r.n_configs} configs, {r.n_trials} trials)")

# Save CSVs for paper-side cross-checks
full_lb.to_csv(PRES/"tables/tab_full_leaderboard_decoded.csv", index=False)
feat_agg.to_csv(PRES/"tables/tab_per_featurizer_aggregate.csv", index=False)
clf_agg.to_csv(PRES/"tables/tab_per_classifier_aggregate.csv", index=False)

# === training time aggregates (per config + per classifier family) ============
time_agg = df_pf_c.groupby("shorthand").agg(
    wall_mean=("wall_sec","mean"), wall_total=("wall_sec","sum")
).reset_index()
time_agg["clf_fam"] = time_agg.shorthand.apply(lambda s: decode_shorthand(s)[2])
time_fam_agg = df_pf_c.assign(cf=df_pf_c.shorthand.apply(lambda s: decode_shorthand(s)[2])) \
    .groupby("cf").agg(wall_mean=("wall_sec","mean"), wall_total=("wall_sec","sum"),
                       n_trials=("wall_sec","count"), n_configs=("shorthand","nunique")) \
    .reset_index().rename(columns={"cf": "classifier_family"}) \
    .sort_values("wall_mean")
grand_total_sec = float(df_pf_c.wall_sec.sum())
print(f"  grand-total wall time across all 725 fits: {grand_total_sec:.0f} s = {grand_total_sec/3600:.2f} h")
time_fam_agg.to_csv(PRES/"tables/tab_training_time_per_family.csv", index=False)

# ============================================================================
# CHART 10 — training time per classifier family (horizontal bar)
# ============================================================================
print("[chart 10] training time per family ...")
fig10 = go.Figure()
fam_colors_clf = {"OvR(ExtraTrees)": SAGE, "OvR(BalancedRF)": NAVY,
                  "DL: LSTM": GRAY, "DL: LSTM+attention": ORANGE,
                  "DL: attention": CRIMSON, "DL: transformer": "#8e44ad"}
time_fam_rev = time_fam_agg.iloc[::-1].reset_index(drop=True)
bar_colors_t = [fam_colors_clf.get(f, GRAY) for f in time_fam_rev.classifier_family]
fig10.add_trace(go.Bar(
    y=time_fam_rev.classifier_family, x=time_fam_rev.wall_mean, orientation="h",
    marker=dict(color=bar_colors_t, line=dict(color="white", width=0.5)),
    text=[f"  {w:.1f} s/fold  ({w*25:.0f} s/config)" for w in time_fam_rev.wall_mean],
    textposition="outside", textfont=dict(size=11, color=CHARCOAL),
    customdata=np.stack([time_fam_rev.wall_total, time_fam_rev.n_trials, time_fam_rev.n_configs], axis=-1),
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Mean wall time per fold: %{x:.1f} s<br>"
        "Total wall time for family: %{customdata[0]:.0f} s ({%{customdata[1]} trials across %{customdata[2]} configs})"
        "<extra></extra>"
    ),
    showlegend=False,
))
fig10.update_layout(
    title=dict(text=f"<b>Training time per classifier family</b><br>"
                    f"<span style='font-size:12px;color:#7f8c8d'>Single Apple M4 Max thread per fold. "
                    f"Grand total across all 725 fits: <b>{grand_total_sec:.0f} s = {grand_total_sec/3600:.2f} h</b>.</span>",
               x=0, font=dict(color=CHARCOAL, size=18)),
    xaxis_title="Mean wall-clock seconds per (seed, fold) trial",
    xaxis=dict(gridcolor="#eee"), yaxis=dict(automargin=True),
    plot_bgcolor="white", paper_bgcolor="white",
    height=460, margin=dict(l=200, r=180, t=100, b=60),
    font=dict(family="Helvetica, Arial, sans-serif", color=CHARCOAL),
)

# ============================================================================
# CHART 14 — Cross-rep stability forest plot (5 reps × 25 configs)
# ============================================================================
print("[chart 14] cross-rep stability forest plot ...")
xrep_csv = PRES / "tables/tab_cross_rep_stability.csv"
xrep_summary_path = PRES / "tables/tab_cross_rep_summary.json"
xrep_df = pd.read_csv(xrep_csv)
xrep_summary = json.loads(xrep_summary_path.read_text())
# Order matches the CSV (sorted by rep_1 mean desc)
_FAM_COLOR_XR = {
    "OvR(ExtraTrees)": SAGE, "OvR(BalancedRF)": NAVY,
    "DL: LSTM+attention": ORANGE, "DL: transformer": "#8e44ad",
    "DL: attention": CRIMSON, "DL: LSTM": GRAY,
}
# Assign family per row
def _fam_of(c):
    if c in ("cpu__ET500_log2", "ftCbow_MM__ET500_sqrt"): return "OvR(ExtraTrees)"
    if c.endswith("__BRF100"): return "OvR(BalancedRF)"
    if c.endswith("__LSTM"): return "DL: LSTM"
    if c.endswith("__LSTMattn"): return "DL: LSTM+attention"
    if c.endswith("__JustAttn"): return "DL: attention"
    if c.endswith("__Trans"): return "DL: transformer"
    return "other"
xrep_df["family"] = xrep_df.shorthand.apply(_fam_of)
# Plotly: one horizontal trace per family for the legend; min-max bar via error_x; individual rep dots overlaid
fig14 = go.Figure()
# Family color spans for range bars (one bar per config; color by family)
_rep_cols = ["rep_1_mean", "rep_2_mean", "rep_3_mean", "rep_4_mean", "rep_5_mean"]
# Range bars (min↔max) drawn as horizontal lines per config
for _, row in xrep_df.iterrows():
    fig14.add_trace(go.Scatter(
        x=[row.cross_rep_min, row.cross_rep_max], y=[row.shorthand, row.shorthand],
        mode="lines",
        line=dict(color=_FAM_COLOR_XR.get(row.family, GRAY), width=2),
        opacity=0.35, showlegend=False, hoverinfo="skip",
    ))
# Per-rep dots + cross-rep mean — one trace per family for legend control
for fam, color in _FAM_COLOR_XR.items():
    sub = xrep_df[xrep_df.family == fam]
    if sub.empty: continue
    xs, ys, customs = [], [], []
    for _, row in sub.iterrows():
        for c_i, col in enumerate(_rep_cols):
            xs.append(row[col]); ys.append(row.shorthand)
            customs.append([c_i + 1, row.cross_rep_mean, row.cross_rep_std])
    fig14.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", name=f"{fam} (per-rep mean)",
        marker=dict(color=color, size=7, line=dict(color="white", width=0.5)),
        customdata=customs,
        hovertemplate="<b>%{y}</b><br>rep_%{customdata[0]} mean: <b>%{x:.4f}</b><br>"
                      "cross-rep mean: %{customdata[1]:.4f} ± %{customdata[2]:.4f}<extra></extra>",
    ))
# Cross-rep mean square markers (single trace, neutral color/border for emphasis)
fig14.add_trace(go.Scatter(
    x=xrep_df.cross_rep_mean, y=xrep_df.shorthand, mode="markers",
    name="cross-rep mean (5-rep avg)",
    marker=dict(symbol="square", size=11, color=[_FAM_COLOR_XR.get(f, GRAY) for f in xrep_df.family],
                line=dict(color=BLACK, width=1.0)),
    customdata=np.stack([xrep_df.cross_rep_std, xrep_df.cross_rep_min,
                         xrep_df.cross_rep_max, xrep_df.min_n_trials], axis=-1),
    hovertemplate="<b>%{y}</b><br>cross-rep mean: <b>%{x:.4f}</b><br>"
                  "cross-rep std: %{customdata[0]:.4f}<br>"
                  "range: %{customdata[1]:.4f} – %{customdata[2]:.4f}<br>"
                  "min trials/rep: %{customdata[3]:.0f}<extra></extra>",
))
fig14.update_layout(
    title=dict(text="<b>Cross-rep reproducibility — 5 reps × 25 trials each (data splits FIXED, model-init seed varies)</b><br>"
                    f"<span style='font-size:13px;color:#000000'>Winner cpu__ET500_log2: "
                    f"<b>{xrep_summary['winner_cross_rep_mean']:.4f} ± {xrep_summary['winner_cross_rep_std']:.4f}</b> "
                    f"(range {xrep_summary['winner_cross_rep_min']:.4f}–{xrep_summary['winner_cross_rep_max']:.4f}) · "
                    f"Top-7 rank stability: <b>{xrep_summary['top7_rank_stability']}</b></span>",
               x=0, font=PLOTLY_TITLE_FONT),
    xaxis=dict(title=dict(text="<b>Test accuracy (5×5 RSKF mean per rep, n=25 trials each)</b>", font=PLOTLY_AXIS_FONT),
               range=[0.69, 0.93], gridcolor="#dddddd",
               tickfont=PLOTLY_TICK_FONT, linecolor=BLACK),
    yaxis=dict(autorange="reversed", tickfont=dict(family="monospace", size=10, color=BLACK), linecolor=BLACK),
    legend=dict(orientation="v", yanchor="bottom", y=0.0, xanchor="right", x=1.0,
                bgcolor="rgba(255,255,255,0.92)", bordercolor=BLACK, borderwidth=1, font=dict(size=10)),
    plot_bgcolor="white", paper_bgcolor="white",
    height=780, margin=dict(l=230, r=80, t=110, b=70),
    font=PLOTLY_FONT_DEFAULTS,
)

# ============================================================================
# CHART 11 — Rank-K redemption (cumulative top-K accuracy, per-substrate)
# ============================================================================
print("[chart 11] rank-K redemption — top-K cumulative accuracy ...")
rank_csv = PRES / "tables/tab_rank_redemption.csv"
rank_df = pd.read_csv(rank_csv).sort_values("top1_acc", ascending=True).reset_index(drop=True)
# Overall headline accuracies (weighted by n_test)
_ntot = rank_df.n_test.sum()
_top1_all = float((rank_df.top1_acc * rank_df.n_test).sum() / _ntot)
_top2_all = float((rank_df.top2_acc * rank_df.n_test).sum() / _ntot)
_top3_all = float((rank_df.top3_acc * rank_df.n_test).sum() / _ntot)
_top5_all = float((rank_df.top5_acc * rank_df.n_test).sum() / _ntot)
fig11 = go.Figure()
_k_palette = {"top1_acc": SAGE, "top2_acc": "#52b788", "top3_acc": ORANGE, "top5_acc": NAVY}
_k_labels = {"top1_acc": "K=1 (top-1)", "top2_acc": "K=2", "top3_acc": "K=3", "top5_acc": "K=5"}
for col in ["top1_acc", "top2_acc", "top3_acc", "top5_acc"]:
    fig11.add_trace(go.Bar(
        y=rank_df.substrate, x=rank_df[col], orientation="h",
        name=_k_labels[col],
        marker=dict(color=_k_palette[col], line=dict(color=BLACK, width=0.5)),
        text=[f"{v:.3f}" for v in rank_df[col]],
        textposition="outside", textfont=dict(size=10, color=BLACK, weight=700),
        customdata=np.stack([rank_df.n_test, rank_df.mean_true_rank], axis=-1),
        hovertemplate=(f"<b>%{{y}}</b><br>{_k_labels[col]} accuracy: <b>%{{x:.4f}}</b><br>"
                       "n test PULs: %{customdata[0]}<br>"
                       "Mean true-rank: %{customdata[1]:.3f}<extra></extra>"),
    ))
fig11.update_layout(
    barmode="group",
    title=dict(text=f"<b>Rank-K redemption — cumulative top-K accuracy (rep_1 OOF, n=1030)</b><br>"
                    f"<span style='font-size:12px;color:#000000'>Overall: top-1 = <b>{_top1_all:.3f}</b> · "
                    f"top-2 = <b>{_top2_all:.3f}</b> (+{(_top2_all-_top1_all)*100:.1f} pp) · "
                    f"top-3 = <b>{_top3_all:.3f}</b> (+{(_top3_all-_top2_all)*100:.1f} pp) · "
                    f"top-5 = <b>{_top5_all:.3f}</b>. When top-1 is wrong, TRUE is usually rank 2 or 3.</span>",
               x=0, font=PLOTLY_TITLE_FONT),
    xaxis=dict(title=dict(text="<b>Cumulative accuracy (fraction of test PULs where TRUE class is within top-K)</b>",
                          font=PLOTLY_AXIS_FONT),
               range=[0.5, 1.08], gridcolor="#dddddd",
               tickfont=PLOTLY_TICK_FONT, linecolor=BLACK),
    yaxis=dict(automargin=True, tickfont=PLOTLY_TICK_FONT, linecolor=BLACK),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=PLOTLY_TICK_FONT),
    plot_bgcolor="white", paper_bgcolor="white",
    height=540, margin=dict(l=160, r=80, t=110, b=70),
    font=PLOTLY_FONT_DEFAULTS,
)

# ============================================================================
# CHART 12 — Confidence vs correctness (10-bin reliability histogram)
# ============================================================================
print("[chart 12] confidence vs correctness — calibrated reliability histogram ...")
conf_csv = PRES / "tables/tab_confidence_vs_correct.csv"
conf_df = pd.read_csv(conf_csv)
_bin_labels = [f"{lo:.1f}–{hi:.1f}" for lo, hi in zip(conf_df.bin_lo, conf_df.bin_hi)]
fig12 = go.Figure()
fig12.add_trace(go.Bar(
    x=_bin_labels, y=conf_df.n_correct, name="correct",
    marker=dict(color=SAGE, line=dict(color=BLACK, width=0.5)),
    text=[str(int(v)) if v > 0 else "" for v in conf_df.n_correct],
    textposition="inside", insidetextanchor="middle",
    textfont=dict(size=11, color="white", weight=700),
    customdata=np.stack([conf_df.n_total, conf_df.pct_correct], axis=-1),
    hovertemplate="<b>Confidence %{x}</b><br>Correct: %{y}<br>Total in bin: %{customdata[0]}<br>"
                  "Accuracy in bin: <b>%{customdata[1]:.3f}</b><extra></extra>",
))
fig12.add_trace(go.Bar(
    x=_bin_labels, y=conf_df.n_incorrect, name="incorrect",
    marker=dict(color=CRIMSON, line=dict(color=BLACK, width=0.5)),
    text=[str(int(v)) if v > 0 else "" for v in conf_df.n_incorrect],
    textposition="outside", textfont=dict(size=10, color=BLACK, weight=700),
    customdata=np.stack([conf_df.n_total, conf_df.pct_correct], axis=-1),
    hovertemplate="<b>Confidence %{x}</b><br>Incorrect: %{y}<br>Total in bin: %{customdata[0]}<br>"
                  "Accuracy in bin: <b>%{customdata[1]:.3f}</b><extra></extra>",
))
_high_conf_correct = int(conf_df[conf_df.bin_lo >= 0.8].n_correct.sum())
_high_conf_total = int(conf_df[conf_df.bin_lo >= 0.8].n_total.sum())
_high_conf_pct = _high_conf_correct / max(_high_conf_total, 1)
fig12.update_layout(
    barmode="stack",
    title=dict(text=f"<b>Calibrated confidence vs correctness — 10-bin reliability histogram (rep_1 OOF, n=1030)</b><br>"
                    f"<span style='font-size:12px;color:#000000'>High-confidence (≥0.8) = "
                    f"<b>{_high_conf_correct}/{_high_conf_total} correct = {_high_conf_pct:.1%}</b>. "
                    f"Bottom bins (≤0.5) sit at ~54–65% — the model knows when it doesn't know.</span>",
               x=0, font=PLOTLY_TITLE_FONT),
    xaxis=dict(title=dict(text="<b>Calibrated max-class probability bin</b>", font=PLOTLY_AXIS_FONT),
               tickfont=PLOTLY_TICK_FONT, linecolor=BLACK),
    yaxis=dict(title=dict(text="<b>Number of held-out PULs</b>", font=PLOTLY_AXIS_FONT),
               gridcolor="#dddddd", tickfont=PLOTLY_TICK_FONT, linecolor=BLACK),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=PLOTLY_TICK_FONT),
    plot_bgcolor="white", paper_bgcolor="white",
    height=520, margin=dict(l=80, r=40, t=110, b=80),
    font=PLOTLY_FONT_DEFAULTS,
)

# ============================================================================
# CHART 13 — Case studies (sortable HTML table built from tab_case_studies.csv)
# ============================================================================
print("[chart 13] case-study cards — HTML table ...")
cs_csv = PRES / "tables/tab_case_studies.csv"
cs_df = pd.read_csv(cs_csv)
_scenario_color = {
    "confident_correct": ("#c8e6c9", "TRUE matches top-1 with high confidence"),
    "rank2_redemption":  ("#fff3cd", "TRUE label recovered at rank 2"),
    "rank3_redemption":  ("#fde2cf", "TRUE label recovered at rank 3"),
    "low_conf_correct":  ("#d6eaf8", "TRUE matches top-1 with LOW confidence — model knows it's unsure"),
    "medium_conf_correct": ("#e8f8f5", "TRUE matches top-1 with medium confidence"),
    "confident_wrong":   ("#f5b7b1", "Confident but WRONG — interesting failure mode"),
}
_cs_rows_html = ""
for _, r in cs_df.iterrows():
    bg, note = _scenario_color.get(r.scenario, ("#ffffff", ""))
    seq_short = str(r.sequence)
    if len(seq_short) > 70: seq_short = seq_short[:67] + "..."
    _cs_rows_html += (
        f'<tr style="background:{bg}">'
        f'<td><b>PUL #{int(r.pul_idx)}</b><br><span class="note">{r.scenario_label}</span></td>'
        f'<td><b>{r.true_substrate}</b></td>'
        f'<td><b>{r.top1_pred}</b><br><span class="note">conf {float(r.top1_conf):.3f}</span></td>'
        f'<td><b>{int(r.true_rank)}</b></td>'
        f'<td class="mono" style="font-size:11px">{r.top3_probs}</td>'
        f'<td class="mono" style="font-size:10px">{seq_short}</td>'
        f'<td><span class="note">{note}</span></td>'
        f'</tr>')
fig13_table_html = (
    '<table class="ex-table" style="font-size:12px"><thead><tr>'
    '<th>PUL / scenario</th><th>TRUE substrate</th><th>top-1 pred (conf)</th>'
    '<th>TRUE rank</th><th>top-3 probs (calibrated)</th>'
    '<th>PUL gene sequence (truncated)</th><th>interpretation</th>'
    '</tr></thead><tbody>' + _cs_rows_html + '</tbody></table>'
)

# ============================================================================
# HTML ASSEMBLY
# ============================================================================
print("[deck.html] assembling ...")

def fig_html(fig, div_id):
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id=div_id,
                       config={"displaylogo": False, "responsive": True})

def callout(label, text):
    return f'<div class="callout"><span class="callout-label">{label}</span>{text}</div>'

# Per-substrate F1 table HTML
per_sub_html = '<table class="data-table"><thead><tr>' \
               '<th>substrate</th><th>n test</th><th>precision</th><th>recall</th><th>F1</th>' \
               '</tr></thead><tbody>'
for _, r in per_sub.iterrows():
    per_sub_html += f"<tr><td><b>{r.substrate}</b></td><td>{int(r.n_test)}</td>" \
                    f"<td>{r.precision:.3f}</td><td>{r.recall:.3f}</td><td><b>{r.F1:.3f}</b></td></tr>"
per_sub_html += "</tbody></table>"

# Sig genes table HTML
sig_html = '<table class="sig-table"><thead><tr><th>substrate</th><th>top 1</th><th>top 2</th><th>top 3</th></tr></thead><tbody>'
for s in substrates:
    sub = sig_tbl[sig_tbl.substrate == s].reset_index(drop=True)
    sig_html += f"<tr><td class='sub'><b>{s}</b></td>"
    for i in range(3):
        if i < len(sub):
            r = sub.iloc[i]
            if r["mark"] == "LIT-exact":     cls = "lit-exact"
            elif r["mark"] == "LIT-collapse": cls = "lit-collapse"
            elif r["mark"] == "miss":         cls = "miss"
            else:                              cls = "non-cazy"
            sig_html += f'<td class="{cls}"><b>{r.feature}</b> <span class="imp">(imp {r.importance:.3f})</span>' \
                        f'<br><span class="note">{r.lit_status}</span></td>'
        else:
            sig_html += "<td></td>"
    sig_html += "</tr>"
sig_html += "</tbody></table>"

# Example PULs table HTML — explicit TRUE vs PRED columns
ex_html = '<table class="ex-table"><thead><tr><th>PUL idx</th><th>PUL sequence (truncated)</th>' \
          '<th>TRUE substrate</th><th>predicted</th>' \
          '<th>max prob</th><th>p-value</th>' \
          '<th>top-5 signature genes</th><th>#LIT/5</th></tr></thead><tbody>'
for r in ex_rows:
    n = r["#LIT / 5"]
    color_cls = "n-high" if n >= 4 else "n-mid" if n >= 2 else "n-low"
    match_cls = "match-yes" if r["match"] else "match-no"
    ex_html += (f'<tr><td>{r["PUL idx"]}</td>'
                f'<td class="mono">{r["PUL sequence (truncated)"]}</td>'
                f'<td class="{match_cls}"><b>{r["True substrate"]}</b></td>'
                f'<td class="{match_cls}"><b>{r["Predicted substrate"]}</b></td>'
                f'<td>{r["Max prob"]}</td>'
                f'<td>{r["p-value"]}</td>'
                f'<td class="mono small">{r["Top-5 signature genes"]}</td>'
                f'<td class="{color_cls}"><b>{n}</b></td></tr>')
ex_html += "</tbody></table>"

# Appendix alias table HTML
alias_rows = [
    ("alginate", "alginate", "1:1 exact", "CAZy DB (Lombard 2014 NAR)"),
    ("pectin", "pectin", "1:1 exact", "CAZy DB"),
    ("xylan", "xylan", "1:1 exact", "CAZy DB"),
    ("alpha-mannan", "alpha-mannan", "1:1 exact", "CAZy DB"),
    ("beta-mannan", "beta-mannan", "1:1 exact", "CAZy DB"),
    ("fructan", "fructan", "1:1 exact", "CAZy DB"),
    ("beta-glucan", "beta-glucan (1:1) + cellulose, cellooligosaccharide, xyloglucan, beta-glycan (collapse)",
     "exact + 4 alias", "Burton 2006 Science 311:1940; Eklof & Brumer 2010 Plant Physiol 153:456"),
    ("alpha-glucan", "alpha-glucan (1:1) + starch, glycogen, sucrose, raffinose, trehalose, palatinose, glucooligo (collapse)",
     "exact + 7 alias", "Stam 2006 Protein Eng. 19:555; Janecek 2014 Cell. Mol. Life Sci. 71:1149"),
    ("chitin", "chitin (1:1) + chitosan, chitooligosaccharide (collapse)",
     "exact + 2 alias", "Hartl 2012 Appl. Microbiol. 93:533; Adrangi & Faramarzi 2013 Biotech. Adv. 31:1786"),
    ("host glycan", "host glycan (1:1) + HMO, sialic-acid, fucose (collapse)",
     "exact + 3 alias", "Marcobal 2011 Cell Host & Microbe 10:507; Tailford 2015 Frontiers Genet. 6:81"),
    ("arabinogalactan", "arabinan + arabinogalactan protein (no 1:1 entry)",
     "collapse only", "Tan 2013 Plant Cell 25:270; Showalter 2010 Plant Physiol 153:485"),
    ("galactan", "alpha-galactan + beta-galactan (no 1:1 entry)",
     "collapse only", "CAZy DB — anomericity sub-classes"),
]
alias_html = '<table class="alias-table"><thead><tr><th>Our class</th><th>Lit name(s)</th>' \
             '<th>Match type</th><th>Citation</th></tr></thead><tbody>'
for (cls_, lit_names, match_type, cite) in alias_rows:
    if match_type == "1:1 exact":      row_cls = "lit-exact"
    elif "exact +" in match_type:      row_cls = "lit-collapse"
    elif "collapse only" in match_type: row_cls = "non-cazy"
    else:                                row_cls = ""
    alias_html += f'<tr class="{row_cls}"><td><b>{cls_}</b></td><td>{lit_names}</td>' \
                  f'<td>{match_type}</td><td>{cite}</td></tr>'
alias_html += "</tbody></table>"

# Static slides content
slide_html_blocks = []

# Slide 1 — Title
slide_html_blocks.append({
    "title": "subFinder",
    "body": """
<div class="title-slide">
  <h1>subFinder</h1>
  <h2>Calibrated classical-ML for PUL substrate prediction</h2>
  <p class="subtitle">Benchmark · best model · reproducibility · signature genes · validation</p>
  <div class="title-hint">Use ← → arrow keys to navigate. Click any chart legend entry to toggle. Hover to inspect.</div>
</div>
"""
})

# Slide 2 — Problem
slide_html_blocks.append({
    "title": "Problem setup",
    "subtitle": "Predict one of 12 polysaccharide substrate classes from a PUL's gene-token sequence",
    "body": """
<dl class="kv">
  <dt>Dataset</dt><dd>1,030 labeled PULs across 12 substrates (alginate, alpha-glucan, alpha-mannan, arabinogalactan, beta-glucan, beta-mannan, chitin, fructan, galactan, host glycan, pectin, xylan)</dd>
  <dt>Each PUL</dt><dd>a comma/pipe-separated string of CAZy family IDs, transporter classification IDs, transcription-factor types, and null padding</dd>
  <dt>Cross-validation</dt><dd>5-repeat × 5-fold Repeated Stratified K-Fold (n=25 trials per configuration) — leak-free: word-embedding models retrained per fold</dd>
  <dt>Goal</dt><dd>(1) maximise predictive accuracy, (2) emit calibrated probabilities, (3) produce per-PUL signature-gene attributions that match literature CAZy knowledge</dd>
</dl>
"""
})

# Slide 3 — Scope
slide_html_blocks.append({
    "title": "Benchmark scope",
    "subtitle": "25 configurations = (3 featurizer families × classifier swap) + (4 DL architectures × 4 embeddings)",
    "body": """
<dl class="kv">
  <dt>Featurizers (3)</dt><dd>CountVectorizer (paper's tok_comma_pipe), CountVectorizer (tok_cpu — also splits on underscore), FastText mean+max-concat (gensim ft.wv[t] with n-gram OOV)</dd>
  <dt>Embeddings retrained per fold (6)</dt><dd>FastText cbow/sg, Word2Vec cbow/sg, Doc2Vec dm/dbow — leak-free (test tokens never enter embedding training)</dd>
  <dt>Classifiers</dt><dd>OvR(BalancedRF n=100) baseline; OvR(ExtraTrees n=500 log2/sqrt) ours; 4 DL architectures from paper (LSTM, LSTM+attention, just-attention, transformer)</dd>
  <dt>Hyperparameters</dt><dd>Paper-verbatim for shallow + DL except batch (DL_BATCH=1024, Transformer=4096 for M4 Max throughput); EarlyStopping patience=30, validation 25%, Adam 1e-4</dd>
  <dt>Total</dt><dd>25 configs × 25 trials = 725 fits; bit-identical reproducibility on our sklearn winners</dd>
</dl>
"""
})

# Slide 4 — Leaderboard
slide_html_blocks.append({
    "title": "Benchmark leaderboard — 25 configurations",
    "subtitle": "Mean test accuracy ± 1 SD across the full 5×5 RSKF grid (n=25 trials per config). Best on top.",
    "body": fig_html(fig1, "chart-leaderboard") + callout(
        "HOW TO READ",
        "Each row is one of 25 configurations (featurizer → classifier). Bars are sorted descending so the "
        "strongest configs sit at the top. <b>Green</b> = our shallow winners; <b>navy</b> = paper's Balanced RF "
        "baselines; <b>orange</b> = paper's deep architectures. Hover any bar for full distribution stats; "
        "click the legend to filter by family."
    ),
})

# Slide 4b — Top-5 podium
slide_html_blocks.append({
    "title": "Top-5 podium — the leading configurations",
    "subtitle": ("Both #1 and #2 share the same OvR(ExtraTrees) classifier under different featurizers — "
                 "the win is in the classifier, not the featurizer."),
    "body": fig_html(fig1b, "chart-podium") + callout(
        "TAKEAWAY",
        "Swapping the paper's Balanced RF (n=100) for OvR(ExtraTrees n=500) on the same data delivers <b>+6 pp</b> "
        "accuracy. The featurizer choice (CountVec vs. FastText) is a secondary effect; ranks #3–#5 are all "
        "variants of the paper's BRF baseline with different word-embedding featurizers."
    ),
})

# (Removed) — old "Accuracy by model family" duplicated slide 14 (per-trial robustness with
# box + swarm). Keeping the more informative version downstream.

# Slide 6 — Reproducibility
slide_html_blocks.append({
    "title": "Reproducibility — does a rerun give the same number?",
    "subtitle": ("<b>X-axis = Δ accuracy</b> (retrained mean − original mean, should be ≈ 0). "
                 f"Restricted to the {len(complete_retrain)} configs with a COMPLETE 25-trial retrain in rep_1 "
                 f"(excluded: {len(incomplete) + len(missing_in_retrain)} partial/missing transformer configs)."),
    "body": fig_html(fig3, "chart-repro") + callout(
        "WHAT THIS IS / WHY Δ ≠ 0 FOR SOME CONFIGS",
        "<b>The question:</b> rerunning the same (config, seed, fold) on the same data — do we get the same accuracy? "
        "<b>The buckets:</b> our sklearn winners (green) reproduce <b>BIT-IDENTICALLY</b>; paper's BRF (navy) drifts "
        "≤0.0017 from imblearn thread-order non-determinism; paper deep configs (orange) drift ≤0.04 from TF GPU "
        "op-order non-determinism. None of these change rankings."
    ),
})

# Slide 6b — Cross-rep reproducibility (5 reps forest plot)
slide_html_blocks.append({
    "title": "Cross-rep reproducibility — 5 reps × 25 trials each (model uncertainty quantified)",
    "subtitle": ("5×5 RSKF data splits are <b>FIXED</b> across all 5 reps; only model-init seed varies "
                 "(<code>REPRO_REP_SEED=1000/2000/3000/4000/5000</code>). Each config gets one dot per rep — closer dots = more reproducible. "
                 "Hover any dot for per-rep mean and the cross-rep aggregate."),
    "body": fig_html(fig14, "chart-cross-rep") + callout(
        "HEADLINE",
        f"<b>Winner cpu__ET500_log2: {xrep_summary['winner_cross_rep_mean']:.4f} ± {xrep_summary['winner_cross_rep_std']:.4f}</b> "
        f"across 5 reps (range {xrep_summary['winner_cross_rep_min']:.4f}–{xrep_summary['winner_cross_rep_max']:.4f}) — "
        "deterministic to the 4th decimal. "
        f"<b>2nd place ftCbow_MM__ET500_sqrt: {xrep_summary['runner_cross_rep_mean']:.4f} ± {xrep_summary['runner_cross_rep_std']:.4f}</b> — also rock-solid. "
        f"<b>Top-7 rank stability:</b> {xrep_summary['top7_rank_stability']} (ranks 1-5 + 7 identical in every rep, single "
        "#6/#7 swap in rep_3). <b>Per-family median cross-rep std:</b> OvR(ExtraTrees) 0.0006 · OvR(BalancedRF) 0.0027 · "
        "DL families 0.0047–0.0064 — our shallow winner is 8-10× more reproducible than DL configs. "
        "This is the model-uncertainty story for deployment: predictions don't move when you re-seed the trainer."
    ),
})

# Slide 7 — Confusion matrix
slide_html_blocks.append({
    "title": "Best model — per-substrate confusion",
    "subtitle": f"OvR(ExtraTrees 500, log2) on seed-42 5-fold out-of-fold predictions  (n=1030 PULs, acc={(y_pred==y_labels).mean():.4f})",
    "body": fig_html(fig4, "chart-confusion") + callout(
        "HOW TO READ",
        "Rows are <b>true substrate</b>, columns are <b>predicted</b>; cell numbers are PUL counts and color is "
        "row-normalized rate. Diagonal cells are correct predictions; off-diagonal cells are mistakes. The hardest "
        "classes are fructan and beta-mannan (small canonical sets and few training examples); alginate, pectin "
        "and xylan reach near-perfect classwise accuracy."
    ),
})

# Slide 8 — per-substrate F1 table
slide_html_blocks.append({
    "title": "Best model — per-substrate precision / recall / F1",
    "subtitle": "Seed-42 5-fold OOF, sorted by F1",
    "body": per_sub_html,
})

# Slide 9 — calibration (MOVED UP from old slide 12: must come before sig-gene
# slides because all downstream sig-gene analyses use CALIBRATED probabilities)
slide_html_blocks.append({
    "title": "Probability calibration of the chosen model",
    "subtitle": "Temperature scaling halves ECE while preserving argmax accuracy. ALL signature-gene analyses on the next slides use these CALIBRATED probabilities (T≈0.70) — sig genes depend on probabilities, so they should be computed on the deployed (calibrated) model, not the raw OvR-ExtraTrees output.",
    "body": fig_html(fig8, "chart-calib") + callout(
        "HOW TO READ + WHY THIS COMES FIRST",
        "The reliability diagram (left) plots <b>predicted confidence vs. empirical accuracy</b> across 10 bins; a "
        "perfectly-calibrated model sits on the dashed diagonal. <b>Temperature scaling (green)</b> halves ECE "
        "(0.094 → 0.029) with <b>zero loss of argmax accuracy</b>. <b>Important:</b> all leave-one-token-out Δ-prob "
        "ablations on slides 10-13 are computed on the temperature-calibrated probabilities (per-fold T applied "
        "via per-class binary logit / T / sigmoid / renormalize), so the sig genes reflect the deployed model's "
        "confidence, not the raw OvR output. Isotonic and Platt are reported for completeness."
    ),
})

# Slide 10 — sig genes table
slide_html_blocks.append({
    "title": "Per-substrate signature genes (raw model top-3) — literature DB status",
    "subtitle": (f"NO filtering. Of {n_cazy_check} CAZy features in 36 top-3 slots: "
                 f"<b>{n_cazy_in_lit}</b> are lit-canonical "
                 f"(<b>{n_exact}</b> via 1:1 exact, <b>{n_collapse}</b> via collapse); "
                 f"{n_cazy_check-n_cazy_in_lit} CAZy not in lit-canon; "
                 f"{n_non_cazy} non-CAZy tokens (lit can't adjudicate)"),
    "body": sig_html + """
<p class="legend">
  <span class="tag lit-exact">[1:1 exact lit match]</span>
  <span class="tag lit-collapse">[lit-canonical via alias collapse]</span>
  <span class="tag miss">[CAZy but NOT in lit-canon]</span>
  <span class="tag non-cazy">[non-CAZy: lit cannot adjudicate]</span>
</p>
"""
})

# Slide 11 — example PULs (with explicit TRUE substrate column)
slide_html_blocks.append({
    "title": "Example predictions on held-out test PULs",
    "subtitle": ("All 10 examples are <b>correctly-classified PULs</b> (true substrate = predicted substrate) "
                 "cherry-picked for high literature-canonical coverage in their top-5 ablation list. "
                 "Population-level metrics across all 1030 PULs are on the next slide."),
    "body": ex_html + """
<p class="legend">
  <span class="tag lit">[LIT]</span> lit-canonical
  <span class="tag cazy">[CAZy]</span> CAZy family not in lit-canon
  <span class="tag acc">[acc]</span> non-CAZy accessory token
</p>
""" + callout(
        "WHY BOTH TRUE & PREDICTED COLUMNS",
        "Both are shown so reviewers can confirm we're not hiding errors — in all 10 rows truth equals prediction "
        "by construction (we filter to correctly-classified PULs first, then rank by lit-canonical coverage). "
        "Green = match; red would indicate a mistake. For a per-PUL hit-rate that includes ALL eligible PULs (not "
        "just the cherry-picked top 10), see the next slide."
    ),
})

# Slide 12 — lit validation (TRUE-class attribution on calibrated probs; matches slide 13)
slide_html_blocks.append({
    "title": "Population-level signature-gene validation",
    "subtitle": "Two complementary recall-style metrics under TRUE-class attribution — same method used in slide 13's per-substrate funnels.",
    "body": f"""
<div class="explain">
  <p><b>How we built 'scope':</b> of the 394 (substrate, canonical-CAZy-gene) pairs the literature lists,
  many canonical genes never appear in any of our 1030 PULs — we exclude those, leaving
  <b>173</b> 'in-scope' (substrate, gene) pairs the model could plausibly find.
  Across ALL 1030 OOF PULs, <b>837</b> contain ≥1 in-scope canonical gene for their <b>TRUE</b> substrate (the 'eligible' denominator).</p>
  <p><b>Left bar (PUL-level recall):</b> for what % of eligible PULs does the top-K Δ<sub>TRUE-class</sub> ablation list include at least one canonical gene for the TRUE substrate?
  <b>Right bar (gene-level recall):</b> of the 173 in-scope canonical genes, how many does the model surface as a top-K Δ<sub>TRUE-class</sub> signature gene anywhere in the population?
  K=3 totals (<b>768/837=91.8%</b> any-hit and <b>109/173=63.0%</b> scope coverage) match slide 13's per-substrate funnel totals exactly.</p>
</div>
{fig_html(fig6, "chart-litval")}
"""
})

# Slide 13 — Per-substrate sig-gene FUNNEL (GROUND-TRUTH-class attribution on calibrated probs)
slide_html_blocks.append({
    "title": "Per-substrate sig-gene funnel — TRUE-class attribution (K=3)",
    "subtitle": "Δ-prob computed against the TRUE substrate, not the argmax. Decouples attribution quality from classification quality — we ask 'when truth is <i>s</i>, did the model surface a canonical CAZy for <i>s</i> as a top-3 sig gene?' regardless of whether the model picked <i>s</i> as its prediction.",
    "body": fig_html(fig8b, "chart-sigpr") + callout(
        "WHY TRUE-CLASS",
        "Our model outputs probabilities for all 12 classes for every PUL, so leave-one-token-out Δ-prob can be "
        "computed for ANY class. Computing Δ<sub>s</sub> against the TRUE substrate <i>s</i> (rather than the argmax) "
        "gives a pure attribution test that's not confounded by classification errors. "
        "Both funnels use truth as the unit — there is no 'correctly predicted' gate on either panel. "
        "Totals (on CALIBRATED probs, see Slide 9): PUL view <b>1030 → 837 eligible → 768 hit (91.8%)</b> · "
        "gene view <b>394 → 173 in-scope → 109 flagged (63.0%)</b>. "
        "Scope recall is ~3 pp <i>higher</i> than the argmax-gated 60.1% on legacy Tables 7/8 because we now "
        "catch genes the model attributed correctly in PULs it classified wrong on the argmax."
    ),
})

# Slide 13b — WOW: Rank-K redemption (cumulative top-K accuracy, per-substrate)
slide_html_blocks.append({
    "title": "Rank-K redemption — TRUE substrate is recovered fast as K grows",
    "subtitle": "Cumulative top-K accuracy on 1030 held-out PULs (rep_1 OOF). When top-1 is wrong, the TRUE label is usually rank 2 or 3 — meaningful for triage workflows where biologists review the top few candidates.",
    "body": fig_html(fig11, "chart-rank-redemption") + callout(
        "HEADLINE",
        f"<b>Top-1 = {_top1_all:.3f}</b> → top-2 = <b>{_top2_all:.3f}</b> (+{(_top2_all-_top1_all)*100:.1f} pp) → "
        f"top-3 = <b>{_top3_all:.3f}</b> (+{(_top3_all-_top2_all)*100:.1f} pp) → top-5 = <b>{_top5_all:.3f}</b>. "
        "Per-substrate: alginate is 100% at K=1; fructan is the hardest top-1 (0.677) but jumps to 0.903 by K=3 — "
        "i.e. when the model is wrong on fructan, the true class is almost always rank 2 or 3. "
        "Mean true-rank across all substrates = <b>1.39</b> (median 1). Calibrated probs make these ranks "
        "deployment-meaningful — see Slide 9. Hover any bar for n_test and mean true-rank per substrate."
    ),
})

# Slide 13c — WOW: Calibration is meaningful (confidence ≈ accuracy per bin)
slide_html_blocks.append({
    "title": "Calibration is meaningful — confidence ≈ accuracy per bin",
    "subtitle": "10-bin reliability histogram on CALIBRATED max-class probabilities (T ≈ 0.70). Sage = correct, red = incorrect. High-confidence predictions (≥0.8) are correct in ≥97% of cases — supports a triage/review workflow with a high-precision auto-accept threshold.",
    "body": fig_html(fig12, "chart-confidence") + callout(
        "PER-BIN ACCURACY",
        "0.8–0.9 → <b>98.2% correct</b> (110/112) · 0.9–1.0 → <b>97.3% correct</b> (566/582). "
        f"Combined ≥0.8: <b>{_high_conf_correct}/{_high_conf_total} = {_high_conf_pct:.1%}</b> correct "
        f"({100*_high_conf_total/_ntot:.0f}% of total). "
        "Bottom bins (≤0.5) sit at ~54–65% — the model 'knows it doesn't know'. "
        "<b>Operational implication:</b> route confidence ≥0.8 to auto-accept (precision ≈ 0.97), "
        "0.5–0.8 to expert review, refuse <0.5. Same calibrator powers the deployed inference."
    ),
})

# Slide 13d — WOW: 6 hand-picked PUL case-study cards (interactive HTML table)
slide_html_blocks.append({
    "title": "Case studies — 6 hand-picked PUL predictions showing the value of top-K + sig genes",
    "subtitle": "Mix of confident-correct, low-confidence-correct, rank-2/3 redemptions, and a confident-wrong failure mode. Row background colors the scenario type.",
    "body": fig13_table_html + callout(
        "READING THE CARDS",
        "Green row = top-1 is TRUE · yellow/orange row = TRUE recovered at rank 2 or 3 · red row = TRUE missing from top-3. "
        "<b>Confident+correct</b> (alginate, 0.96) — PL6/PL17 are canonical alginate lyases. "
        "<b>Rank-2 redemption</b> (alpha-glucan, 0.52 vs 0.32) — model splits between α/β-glucan, GH13 nudges α. "
        "<b>Confident-wrong</b> (alpha-glucan classified as fructan, p=1.0) — GH32/3.A.1.1.x are sucrose-PTS markers; "
        "an interpretable failure mode for the substrate-attribution review workflow."
    ),
})

# Slide 12c — Test-PUL OOV vs accuracy (with caveat)
slide_html_blocks.append({
    "title": "Internal robustness — accuracy vs train-vocab OOV proportion",
    "subtitle": "Uses the SAME featurizer the deployed model uses (CountVec with tok_cpu, fit per fold on outer_tr only). 'OOV' = test tokens NOT in the training fold's vocab.",
    "body": fig_html(fig8c, "chart-oov") + callout(
        "READING + CAVEAT",
        "Bar widths ∝ PUL count per bucket. <b>89% of test PULs (920/1030)</b> have zero OOV tokens — model scores "
        "<b>91% accuracy</b> on them. For PULs with up to 10% OOV the model still scores <b>~100%</b>. "
        "Once OOV exceeds 10%, accuracy collapses to <b>~64%</b>. "
        "<b>CAVEAT:</b> 'novel' here means <i>within-dataset</i> distinct-token frequency, NOT biological novelty. "
        "The 1030 PULs share a tight ~517-token vocab; a typical training fold covers ~488 of them, so this "
        "analysis is an internal robustness check, not a cross-organism novelty test. The true biological-novelty "
        "story is the fungal-CGC out-of-distribution test (Supplementary Section S9) where 24/24 fungal CGCs are "
        "flagged REFUSE under the Jaccard/OOV heuristics."
    ),
})

# Slide 13 — trial distribution / robustness (different question from slide 6)
slide_html_blocks.append({
    "title": "Robustness — how stable is each model family across trials?",
    "subtitle": ("<b>X-axis = absolute test accuracy</b> (range 0.4–0.95). "
                 "6 rows = 6 model families pooling every (config × seed × fold) trial. "
                 "Distinct from Slide 6 which shows <b>Δ</b> accuracy of a single rerun."),
    "body": fig_html(fig9, "chart-trial") + callout(
        "WHAT THIS IS / HOW TO READ THE BOX",
        "<b>The question:</b> for one family, how much does a single (seed, fold) trial change the accuracy you get? "
        "<b>Box markers:</b> filled box = Q1→Q3 (middle 50%); thick vertical line = median; "
        "white diamond = mean; whiskers = 1.5·IQR rule; dots = individual trials (hover for exact config). "
        "<b>Takeaway:</b> the green family sits entirely above 0.86 with a tight ~0.04 spread; deep families span much "
        "wider, with some folds dipping below 0.55. Our shallow ensemble is <b>both more accurate AND more stable</b>."
    ),
})

# --- Appendix slides: full leaderboard + per-featurizer agg + per-classifier agg ---

# Full 25-config table HTML
lb_html = ('<table class="data-table lb"><thead><tr>'
           '<th>rank</th><th>config (shorthand)</th>'
           '<th>featurizer family</th><th>featurizer detail</th>'
           '<th>classifier family</th><th>classifier detail</th>'
           '<th>mean ± std</th><th>min — max</th><th>n</th>'
           '</tr></thead><tbody>')
for _, r in full_lb.iterrows():
    cls_winner = "winner" if r["rank"] == 1 else ("podium" if r["rank"] <= 3 else "")
    lb_html += (f'<tr class="{cls_winner}"><td><b>{int(r["rank"])}</b></td>'
                f'<td><code>{r.shorthand}</code></td>'
                f'<td>{r.featurizer_fam}</td>'
                f'<td>{r.featurizer_det}</td>'
                f'<td>{r.clf_fam}</td>'
                f'<td>{r.clf_det}</td>'
                f'<td><b>{r["mean"]:.4f}</b> ± {r["std"]:.4f}</td>'
                f'<td>{r.min_:.4f} — {r.max_:.4f}</td>'
                f'<td>{int(r.n)}</td></tr>')
lb_html += "</tbody></table>"

# Per-featurizer aggregate HTML
feat_html = ('<table class="data-table"><thead><tr>'
             '<th>featurizer family</th><th>mean accuracy</th><th>std</th>'
             '<th>n configs in family</th><th>n trials total</th>'
             '</tr></thead><tbody>')
for _, r in feat_agg.iterrows():
    feat_html += (f'<tr><td><b>{r.featurizer_family}</b></td>'
                  f'<td>{r.mean_acc:.4f}</td><td>{r.std_acc:.4f}</td>'
                  f'<td>{int(r.n_configs)}</td><td>{int(r.n_trials)}</td></tr>')
feat_html += "</tbody></table>"

# Per-classifier aggregate HTML
clf_html = ('<table class="data-table"><thead><tr>'
            '<th>classifier family</th><th>mean accuracy</th><th>std</th>'
            '<th>n configs in family</th><th>n trials total</th>'
            '</tr></thead><tbody>')
for _, r in clf_agg.iterrows():
    cls_winner = "winner" if r.classifier_family == "OvR(ExtraTrees)" else ""
    clf_html += (f'<tr class="{cls_winner}"><td><b>{r.classifier_family}</b></td>'
                 f'<td>{r.mean_acc:.4f}</td><td>{r.std_acc:.4f}</td>'
                 f'<td>{int(r.n_configs)}</td><td>{int(r.n_trials)}</td></tr>')
clf_html += "</tbody></table>"

slide_html_blocks.append({
    "title": "Appendix · Full 25-config leaderboard with decoded featurizer + classifier",
    "subtitle": "Every config benchmarked. Rank 1 highlighted gold; ranks 2–3 highlighted pale gold.",
    "body": lb_html,
})

slide_html_blocks.append({
    "title": "Appendix · Performance by featurizer family",
    "subtitle": "Averaged over all (config, seed, fold) trials sharing the same featurizer family",
    "body": feat_html + ("<p class='footnote'>Note: averages mix in different classifier choices. "
                          "<b>Pattern:</b> CountVec leads because it carries our ExtraTrees winner; "
                          "FastText/Word2Vec/Doc2Vec families are pulled down by their BRF/DL classifier pairings.</p>"),
})

slide_html_blocks.append({
    "title": "Appendix · Performance by classifier family",
    "subtitle": "Averaged over all (config, seed, fold) trials sharing the same classifier family",
    "body": clf_html + ("<p class='footnote'>Note: averages mix in different featurizer choices. "
                          "<b>Pattern:</b> OvR(ExtraTrees) is the only family in the >0.90 band; "
                          "every DL family lands in 0.74–0.82.</p>"),
})

# Training time appendix
time_table_html = ('<table class="data-table"><thead><tr>'
                   '<th>classifier family</th><th>mean wall sec / fold</th>'
                   '<th>≈ time per config (25 folds)</th>'
                   '<th>total wall sec (family)</th><th>n configs</th><th>n trials</th>'
                   '</tr></thead><tbody>')
for _, r in time_fam_agg.sort_values("wall_mean").iterrows():
    cls = "winner" if r.classifier_family == "OvR(ExtraTrees)" else ""
    time_table_html += (f'<tr class="{cls}"><td><b>{r.classifier_family}</b></td>'
                        f'<td>{r.wall_mean:.2f} s</td>'
                        f'<td>{r.wall_mean*25:.0f} s ({r.wall_mean*25/60:.1f} min)</td>'
                        f'<td>{r.wall_total:.0f} s ({r.wall_total/60:.1f} min)</td>'
                        f'<td>{int(r.n_configs)}</td><td>{int(r.n_trials)}</td></tr>')
time_table_html += "</tbody></table>"

slide_html_blocks.append({
    "title": "Appendix · Training time per classifier family",
    "subtitle": (f"Why our shallow winner is cheap to train. Grand total across all 725 fits: "
                 f"<b>{grand_total_sec:.0f} s = {grand_total_sec/3600:.2f} h</b> on an Apple M4 Max."),
    "body": fig_html(fig10, "chart-time") + time_table_html + (
        "<p class='footnote'>The DL: transformer family is the most expensive — <b>≈70 s/fold ≈ 30 min per config of 25 folds</b>, "
        "while our winner trains in <b>~3 s/fold ≈ 75 s per config</b>. The shallow ET classifier is therefore <b>~24x cheaper</b> "
        "than the transformer per fit and <b>~25x more</b> compute-efficient per percentage point of accuracy. "
        "Times exclude per-fold word-embedding training (cached under <code>fold_cache_v2/</code>).</p>"
    ),
})

# Slide N — appendix alias
slide_html_blocks.append({
    "title": "Appendix · Substrate-alias map between literature DB and our 12 classes",
    "subtitle": "Curated DB has 75 finer-grained substrate categories; we collapse them into our 12 with primary-literature support",
    "body": alias_html + """
<p class="footnote">
  Detailed citations in <code>paper/reference.bib</code>; codified as <code>ALIAS_PROVENANCE</code> in
  <code>presentations/build_slides.py</code> (and <code>build_interactive_deck.py</code>).
</p>
"""
})

# Slide A5 — POST-HOC v2 REFINEMENT (additive, doesn't replace original)
slide_html_blocks.append({
    "title": "Post-hoc refinement · v2 tokenizer for cross-domain generalization",
    "subtitle": ("After applying the deployed model to the 358,751-PUL unsupervised pre-training corpus, "
                 "we found OOV was driven by token <b>format</b> mismatches, not biology. "
                 "Two-line tokenizer change closes the gap — original deployed model is preserved unchanged."),
    "body": """
<table class="data-table" style="width:100%;margin-bottom:18px">
<thead><tr>
<th>metric</th><th>original (tok_cpu)</th><th>v2 refinement</th><th>Δ</th>
</tr></thead><tbody>
<tr><td><b>5-rep cross-rep mean acc</b> (full 5×5 RSKF × 5 seeds = 125 fits each)</td><td>0.9063 ± 0.0006</td><td><b>0.9145 ± 0.0005</b></td><td style="color:#27ae60;font-weight:700">+0.82 pp ✓</td></tr>
<tr><td>Improved in every rep (5/5)</td><td>—</td><td><b>5/5</b></td><td style="color:#27ae60;font-weight:700">consistent ✓</td></tr>
<tr><td>Deployed vocab size</td><td>517</td><td><b>306</b></td><td style="color:#27ae60;font-weight:700">41% smaller ✓</td></tr>
<tr><td>Unsupervised mean OOV %</td><td>21.79%</td><td><b>5.36%</b></td><td style="color:#27ae60;font-weight:700">4× lower ✓</td></tr>
<tr><td>Unsupervised PULs at OOV ≤ 10% (trust band)</td><td>37.2%</td><td><b>77.5%</b></td><td style="color:#27ae60;font-weight:700">+40 pp ✓</td></tr>
<tr><td>Unsupervised PULs at OOV ≤ 25%</td><td>71.2%</td><td><b>96.2%</b></td><td style="color:#27ae60;font-weight:700">+25 pp ✓</td></tr>
</tbody></table>
<div class="callout"><span class="callout-label">WHAT v2 DOES (two lines)</span>
<b>(1) Truncate Transporter Classification numbers to 2-level family</b> — <code>1.B.14.6.1</code> → <code>1.B</code>. The supervised corpus mixes 5-level and 3-level TCs; the unsupervised has only 3-level. Truncation closes the format gap without losing substrate-relevant biology (TC numbers are transporter accessory genes, not the substrate-discriminating signal).<br><br>
<b>(2) Augment CAZy tokens with family-only fallback in concat</b> — <code>GH13</code> → keep <code>GH13</code> AND add <code>GH</code>. Novel families like <code>AA17</code> (never seen in supervised) now still pattern-match against <code>AA</code>. Specific tokens still match exactly — no info lost. Only 6 new tokens added to vocab (the 6 family prefixes).</div>
<div class="callout" style="background:#d4edda"><span class="callout-label">ADDITIVE — ORIGINAL MODEL UNCHANGED</span>
Original deployed bundle (<code>artifacts/final_model.pkl</code> + <code>reproducibility/rep_*/final_model.pkl</code>) preserved as the default and shipped via Git LFS. The v2 model bundle is <b>also shipped in-repo</b>, no LFS needed: each pkl is saved with <code>joblib.dump(..., compress=("xz", 6))</code>, which shrinks it from ~144 MB to ~20 MB (~120 MB total across all 6), under GitHub's 100 MB per-file limit. <code>joblib.load()</code> auto-decompresses, so inference code is unchanged. Deterministic regeneration: <code>scripts/13_train_tc2_refinement.py</code> + <code>scripts/13b_run_tc2_reproducibility.py</code> (~90 s/rep). 13-strategy sweep: <code>unravel/experiments/run_token_strategies.py</code>. Sig-gene attribution in <code>unravel/</code> visually separates "signature tokens" (numbered, specific) from "signature families" (augmented fallback).</div>
<div class="callout" style="background:#fff3cd"><span class="callout-label">V2 SIG-GENE PR (top-3, family-augmented canon)</span>
On a 5-fold seed-42 stratified split (built by <code>scripts/13c_v2_sig_gene_pr.py</code>; aggregate at <code>paper/tables/table12_v2_aggregate_sig_pr.csv</code>; per-substrate at <code>paper/tables/table12_v2_per_substrate_sig_pr.csv</code>; supplement Table S5b):
<br>&nbsp;&nbsp;&bull; <b>813/1,028 PULs (79.1%)</b> have a lit-canonical signal in their top-3 ablation tokens
<br>&nbsp;&nbsp;&bull; <b>749</b> caught by a specific canon token (<code>GH13</code>); <b>64 extra</b> rescued only by the family fallback (<code>GH</code>)
<br>&nbsp;&nbsp;&bull; Gene-view scope recall — <b>specific</b> tokens 92/173 = 53.2% · <b>family</b> fallbacks 24/31 = <b>77.4%</b> (fewer competitors at the family level)
<br>&nbsp;&nbsp;&bull; Per-substrate range 28.6% (chitin) → 100% (alginate); family rescue largest for alpha-glucan, beta-mannan, pectin, galactan (+13-14 PULs each).
<br>This confirms the augmented family tokens carry consistent substrate-specific signal, not noise — the model uses them as a genuine fallback when specific subfamilies are absent.
</div>
"""
})

# Slide 15 — take home
slide_html_blocks.append({
    "title": "Take-home",
    "body": f"""
<dl class="kv take-home">
  <dt>Best configuration</dt><dd>cpu__ET500_log2 = CountVec (tok_cpu) × OvR(ExtraTrees n=500, log2, class_weight='balanced', bootstrap=False) — <b>0.9058 ± 0.0172</b> on full 5×5 RSKF (n=25)</dd>
  <dt>Wins by classifier choice, not features</dt><dd>+6.16 pp over published BRF baseline (paired-t p≈5×10⁻¹⁴), +8.10 pp over best paper DL (LSTM-with-attention on FastText-skipgram)</dd>
  <dt>Calibrated for deployment</dt><dd>Temperature T≈{T_mean:.2f} reduces 10-bin ECE from 0.094 to 0.029 while preserving 0.9029 OOF accuracy exactly (monotonic)</dd>
  <dt>Signature genes are biologically interpretable</dt><dd>Per-PUL leave-one-token-out ablation surfaces ≥1 literature-canonical CAZy in top-3 for <b>93.2%</b> of eligible PULs (725/778); covers <b>60.1%</b> of the in-scope canonical gene vocabulary at K=3 (104/173)</dd>
  <dt>Fully reproducible</dt><dd>Every (config, seed, fold) classifier weight + train/test predictions saved in <code>reproducibility/rep_1/predictions/</code> — single command <code>--training False</code> rebuilds every paper number in ~1 min</dd>
</dl>
"""
})

# CSS
css = """
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0; background: #f4f5f7;
  font-family: Helvetica, Arial, sans-serif; color: #000000;
}
/* NOTE: do NOT use a broad `.slide-body text { fill: #000 }` override —
   Plotly relies on invisible / measurement <text> nodes internally and
   forcing all of them to black makes them visible as artifacts. Plotly font
   styling is already configured per-figure via update_layout(font=...). */
.deck { max-width: 1400px; margin: 0 auto; }
.slide {
  background: white; min-height: 720px; margin: 24px auto; padding: 32px 40px;
  border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  border-left: 8px solid #1a3a5c;
  scroll-margin-top: 60px;
}
.slide h1.s-title {
  margin: 0 0 4px 0; color: #1a3a5c; font-size: 28px; font-weight: 700;
}
.slide p.s-subtitle {
  margin: 0 0 22px 0; color: #7f8c8d; font-size: 14px; font-style: italic;
}
.slide-body { font-size: 14px; line-height: 1.55; }
dl.kv { display: grid; grid-template-columns: max-content 1fr; gap: 12px 22px; margin: 8px 0; }
dl.kv dt { color: #1a3a5c; font-weight: 700; font-size: 15px; }
dl.kv dd { margin: 0; color: #2c3e50; font-size: 14px; }
dl.kv.take-home dt { color: #27ae60; }

/* Title slide */
.title-slide { text-align: left; padding: 80px 40px; }
.title-slide h1 { color: #1a3a5c; font-size: 72px; margin: 0; font-weight: 700; }
.title-slide h2 { color: #2c3e50; font-size: 28px; margin: 12px 0; font-weight: 400; }
.title-slide .subtitle { color: #7f8c8d; font-size: 16px; margin: 24px 0; }
.title-slide .title-hint {
  margin-top: 80px; padding: 14px 18px; background: #eef2f6;
  border-left: 4px solid #27ae60; color: #2c3e50; font-size: 13px;
}

/* Data tables */
table.data-table, table.sig-table, table.ex-table, table.alias-table {
  width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 4px;
}
table.data-table th, table.sig-table th, table.ex-table th, table.alias-table th {
  background: #1a3a5c; color: white; padding: 8px 10px; text-align: left;
  font-weight: 600;
}
table.data-table td, table.sig-table td, table.ex-table td, table.alias-table td {
  padding: 6px 10px; border-bottom: 1px solid #e8eaed; vertical-align: top;
}
table.data-table tr:nth-child(even) td { background: #fafbfc; }
table.data-table tr.winner td { background: #fff3cd !important; font-weight: 600; }
table.data-table tr.podium td { background: #fff9e6 !important; }
table.lb { font-size: 11px; }
table.lb td code { font-size: 10px; }

/* Sig genes table cell colors */
.sig-table td.sub { background: #ecf0f1; font-size: 13px; }
.sig-table td.lit-exact    { background: #c8e6c9; }
.sig-table td.lit-collapse { background: #e0f3e8; }
.sig-table td.miss         { background: #fce4e4; }
.sig-table td.non-cazy     { background: #fdf6e3; }
.sig-table td .imp  { color: #7f8c8d; font-weight: 400; font-size: 11px; }
.sig-table td .note { color: #555; font-size: 11px; font-style: italic; }

/* Example PUL table */
.ex-table td.mono { font-family: Menlo, Consolas, monospace; font-size: 11px; word-break: break-all; }
.ex-table td.small { font-size: 11px; }
.ex-table td.n-high { background: #c8e6c9; text-align: center; font-size: 14px; }
.ex-table td.n-mid  { background: #e0f3e8; text-align: center; font-size: 14px; }
.ex-table td.n-low  { background: #fdf6e3; text-align: center; font-size: 14px; }
.ex-table .tag {
  display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 10px; margin: 1px;
}
.ex-table .tag.lit  { background: #c8e6c9; color: #1b4332; }
.ex-table .tag.cazy { background: #fce4e4; color: #6b2020; }
.ex-table .tag.acc  { background: #fdf6e3; color: #6b4f00; }
.ex-table td.match-yes { background: #c8e6c9; color: #1b4332; }
.ex-table td.match-no  { background: #fce4e4; color: #6b2020; }

/* Alias appendix table */
.alias-table tr.lit-exact td { background: #c8e6c9; }
.alias-table tr.lit-collapse td { background: #e0f3e8; }
.alias-table tr.non-cazy td { background: #fdf6e3; }

/* Legend chips */
.legend { margin: 12px 0 0 0; font-size: 11px; }
.legend .tag {
  display: inline-block; padding: 3px 8px; border-radius: 3px; margin-right: 8px; font-weight: 600;
}
.legend .tag.lit-exact     { background: #c8e6c9; color: #1b4332; }
.legend .tag.lit-collapse  { background: #e0f3e8; color: #1b4332; }
.legend .tag.lit           { background: #c8e6c9; color: #1b4332; }
.legend .tag.miss          { background: #fce4e4; color: #6b2020; }
.legend .tag.non-cazy      { background: #fdf6e3; color: #6b4f00; }
.legend .tag.cazy          { background: #fce4e4; color: #6b2020; }
.legend .tag.acc           { background: #fdf6e3; color: #6b4f00; }

.explain { background: #fafbfc; border-left: 4px solid #27ae60; padding: 10px 16px; margin: 0 0 12px 0; font-size: 13px; }
.explain p { margin: 6px 0; }

/* Yellow "How to read" callout under each plot */
.callout {
  background: #fff9e6; border: 1px solid #2c3e50; border-radius: 4px;
  padding: 10px 14px; margin: 14px 0 4px 0; font-size: 13px;
  color: #000000; font-weight: 600;
}
.callout b, .callout strong { color: #000000; }
.callout-label {
  display: inline-block; background: #1a3a5c; color: white;
  padding: 2px 8px; border-radius: 3px; font-size: 11px; margin-right: 8px;
  font-weight: 700;
}

.footnote { color: #7f8c8d; font-size: 12px; margin-top: 14px; font-style: italic; }

/* Top nav */
.topnav {
  position: sticky; top: 0; z-index: 100; background: #1a3a5c; color: white;
  padding: 10px 24px; display: flex; justify-content: space-between; align-items: center;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.topnav .brand { font-weight: 700; font-size: 14px; }
.topnav .toc { font-size: 12px; }
.topnav .toc select {
  background: #2c5078; color: white; border: 1px solid #4a6889; padding: 4px 8px;
  border-radius: 4px; font-size: 12px;
}
.topnav .keys { font-size: 11px; opacity: 0.8; }
.slide-number { color: #7f8c8d; font-size: 11px; float: right; margin-top: 4px; }

/* Code */
code { background: #f1f3f4; padding: 1px 5px; border-radius: 3px; font-size: 11px;
       font-family: Menlo, Consolas, monospace; color: #2c3e50; }

/* Mobile-ish */
@media (max-width: 800px) {
  .slide { padding: 20px; }
  dl.kv { grid-template-columns: 1fr; }
}
"""

# Slide HTML
slides_str = ""
toc_options = ""
for i, slide in enumerate(slide_html_blocks, 1):
    title = slide.get("title", "")
    subtitle = slide.get("subtitle", "")
    body = slide.get("body", "")
    sub_html = f'<p class="s-subtitle">{subtitle}</p>' if subtitle else ""
    title_html = f'<h1 class="s-title">{title}</h1>' if title and title != "subFinder" else ""
    slides_str += f'<section class="slide" id="slide-{i}">{title_html}{sub_html}<div class="slide-body">{body}</div>' \
                  f'<div class="slide-number">slide {i} / {len(slide_html_blocks)}</div></section>\n'
    label = title if title else f"Slide {i}"
    toc_options += f'<option value="slide-{i}">{i}. {label}</option>\n'

js = r"""
<script>
const slides = Array.from(document.querySelectorAll('.slide'));
const toc = document.getElementById('toc');
function getCurrent() {
  let cur = 0;
  for (let i = 0; i < slides.length; i++) {
    const r = slides[i].getBoundingClientRect();
    if (r.top < window.innerHeight*0.5) cur = i;
  }
  return cur;
}
toc.addEventListener('change', e => {
  const el = document.getElementById(e.target.value);
  if (el) el.scrollIntoView({behavior: 'smooth'});
});
window.addEventListener('keydown', e => {
  if (e.target.matches('input,select,textarea')) return;
  let cur = getCurrent();
  if (e.key === 'ArrowRight' || e.key === 'PageDown') {
    cur = Math.min(slides.length-1, cur+1);
    slides[cur].scrollIntoView({behavior: 'smooth'});
    e.preventDefault();
  } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
    cur = Math.max(0, cur-1);
    slides[cur].scrollIntoView({behavior: 'smooth'});
    e.preventDefault();
  }
});
window.addEventListener('scroll', () => {
  const cur = getCurrent();
  if (toc.selectedIndex !== cur) toc.selectedIndex = cur;
});
</script>
"""

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>subFinder — interactive deck</title>
<style>{css}</style>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
<div class="topnav">
  <span class="brand">subFinder · interactive deck</span>
  <span class="toc"><select id="toc">{toc_options}</select></span>
  <span class="keys">← → to navigate · ESC to reset zoom · click chart legend to toggle traces</span>
</div>
<div class="deck">
{slides_str}
</div>
{js}
</body>
</html>
"""

OUT.write_text(html)
size_kb = OUT.stat().st_size // 1024
print(f"[deck.html] wrote {OUT.relative_to(ROOT)} ({size_kb} KB)")
print(f"[deck.html] slides: {len(slide_html_blocks)}")
print(f"[deck.html] open in browser: file://{OUT}")
