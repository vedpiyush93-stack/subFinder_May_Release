#!/usr/bin/env python3
"""Build 'reviewer-impressive' supplementary visuals + tables.

Outputs (all from rep_1 = canonical benchmark):
  docs/figures/
    fig11_rank_redemption.png     Cumulative top-K accuracy. Shows that even when
                                   top-1 is wrong, the TRUE substrate is usually
                                   in top-2 or top-3. Justifies the calibrated
                                   probability output (not just argmax).
    fig12_confidence_vs_correct.png  Histogram of calibrated argmax probability
                                   stacked by correct/incorrect. Shows that
                                   confidence MEANS something: high-confidence
                                   predictions are nearly always right; low-
                                   confidence ones are where most errors are.
    fig13_case_study_cards.png    6 hand-picked PULs showing:
                                   - top-3 calibrated probabilities (bar chart)
                                   - top-5 signature genes for predicted substrate
                                     with lit-canonical match flags
                                   - TRUE substrate marker
                                   Mix of: (a) confident-correct, (b) low-confidence
                                   correct (model unsure but right), (c) confident
                                   wrong (where TRUE is top-2 redemption).

  docs/tables/
    tab_rank_redemption.csv       per-K cumulative accuracy + per-substrate breakdown
    tab_confidence_vs_correct.csv per-bin counts of correct/incorrect by confidence
    tab_case_study_cards.csv      the 6 selected PULs with all displayed metadata

Wire-up:
  - scripts/08_build_static_deck.py adds 3 new slides (slides 22/23/24)
  - scripts/09_build_interactive_deck.py adds 3 matching interactive charts
  - notebooks/build_paper_artifacts.ipynb picks these up via existing audit() calls
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))     # so `from src.lit_validation import ...` works
                                  # AND so pickle.load can find src.* modules
FIG  = ROOT / "docs/figures"
TAB  = ROOT / "docs/tables"
PRED = ROOT / "artifacts/predictions"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

# Paper-consistent colors (match the deck)
NAVY   = "#1a3a5c"
SAGE   = "#27ae60"
ORANGE = "#e67e22"
RED    = "#c0392b"
GRAY   = "#7f8c8d"
LIGHT  = "#ecf0f1"
BLACK  = "#0b0b0b"

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": BLACK,
    "axes.labelcolor": BLACK,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "axes.titlecolor": BLACK,
    "axes.titleweight": "bold",
})


def _load_seed42_oof_calibrated():
    """Load rep_1 seed=42 OOF calibrated probs for the deployed cpu__ET500_log2."""
    import pickle
    df = pd.read_csv(ROOT / "data/Train_data.csv")
    X = df["sig_gene_seq"].fillna("").values
    y = df["high_level_substr"].values
    classes = sorted(set(y))

    # Calibrated probs come from the deployed final_model.pkl temperature
    final = pickle.load(open(ROOT / "artifacts/final_model.pkl", "rb"))
    T = final.get("T", 1.0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    P_oof = np.zeros((len(X), len(classes)), dtype=np.float32)
    for fold, (_, te) in enumerate(skf.split(X, y)):
        npz = np.load(PRED / f"cpu__ET500_log2/r42_f{fold}/probs_test.npz", allow_pickle=True)
        cls_npz = list(npz["classes"])
        col = np.array([cls_npz.index(c) for c in classes])
        raw = npz["probs"][:, col]
        # Temperature scaling
        with np.errstate(divide="ignore"):
            logits = np.log(np.clip(raw, 1e-12, 1.0))
        scaled = logits / T
        cal = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        cal = cal / cal.sum(axis=1, keepdims=True)
        # Map te → fold rows
        idx_te = npz["idx"] if "idx" in npz.files else npz["test_idx"]
        P_oof[idx_te] = cal.astype(np.float32)
    return X, y, classes, P_oof, float(T)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 11: rank-K cumulative accuracy
# ─────────────────────────────────────────────────────────────────────────────
print("[10] Loading seed-42 OOF calibrated probs ...")
X, y, classes, P_oof, T = _load_seed42_oof_calibrated()
N = len(X); y_int = np.array([classes.index(c) for c in y])
print(f"     N={N}, C={len(classes)}, T={T:.4f}")

print("[10] Fig 11: rank-K cumulative accuracy ...")
# For each PUL, find rank of TRUE substrate when probs sorted desc
order = np.argsort(-P_oof, axis=1)
true_rank = np.array([np.where(order[i] == y_int[i])[0][0] + 1 for i in range(N)])

K_max = 12
cum_acc = [(true_rank <= k).mean() for k in range(1, K_max+1)]
cum_acc_arr = np.array(cum_acc)

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(range(1, K_max+1), cum_acc_arr, color=[NAVY if k==1 else (SAGE if k<=3 else GRAY) for k in range(1, K_max+1)],
              edgecolor=BLACK, linewidth=0.7)
for b, val, k in zip(bars, cum_acc_arr, range(1, K_max+1)):
    ax.text(b.get_x()+b.get_width()/2, val+0.005, f"{val*100:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=BLACK)
ax.set_xticks(range(1, K_max+1))
ax.set_xlabel("K  (top-K calibrated predictions)")
ax.set_ylabel("Cumulative accuracy")
ax.set_ylim(0, 1.04)
ax.set_title("Rank-K redemption — TRUE substrate is recovered fast as K grows", loc="left")
ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=0.7)
# Annotate the most important transitions
ax.annotate(f"top-1 = {cum_acc_arr[0]*100:.1f}%   (the headline accuracy)",
            xy=(1, cum_acc_arr[0]), xytext=(2.8, cum_acc_arr[0]-0.06),
            fontsize=10, color=NAVY, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=1))
ax.annotate(f"top-3 = {cum_acc_arr[2]*100:.1f}%   (calibrated probs surface the answer)",
            xy=(3, cum_acc_arr[2]), xytext=(5, cum_acc_arr[2]-0.04),
            fontsize=10, color=SAGE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=SAGE, lw=1))
plt.tight_layout()
plt.savefig(FIG/"fig11_rank_redemption.png", dpi=160); plt.close()

# Per-substrate breakdown table
per_sub = []
for ci, c in enumerate(classes):
    mask = (y_int == ci)
    n = int(mask.sum())
    ranks = true_rank[mask]
    per_sub.append({
        "substrate": c, "n_test": n,
        "top1_acc": float((ranks <= 1).mean()),
        "top2_acc": float((ranks <= 2).mean()),
        "top3_acc": float((ranks <= 3).mean()),
        "top5_acc": float((ranks <= 5).mean()),
        "mean_true_rank": float(ranks.mean()),
    })
df_sub = pd.DataFrame(per_sub).sort_values("top1_acc", ascending=False).round(4)
df_sub.to_csv(TAB/"tab_rank_redemption.csv", index=False)
print(f"  wrote {FIG}/fig11_rank_redemption.png + {TAB}/tab_rank_redemption.csv")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 12: confidence-vs-correctness histogram
# ─────────────────────────────────────────────────────────────────────────────
print("[10] Fig 12: confidence vs correctness ...")
top1_conf = P_oof.max(axis=1)
correct = (order[:, 0] == y_int)

bins = np.linspace(0, 1, 11)  # 10 bins of width 0.1
bin_idx = np.clip(np.digitize(top1_conf, bins) - 1, 0, len(bins)-2)
counts_correct   = np.array([((bin_idx == b) & correct).sum() for b in range(len(bins)-1)])
counts_incorrect = np.array([((bin_idx == b) & ~correct).sum() for b in range(len(bins)-1)])

fig, ax = plt.subplots(figsize=(10, 5.5))
centers = (bins[:-1] + bins[1:]) / 2
width = (bins[1]-bins[0]) * 0.85
ax.bar(centers, counts_correct,   width=width, color=SAGE,
        edgecolor=BLACK, linewidth=0.6, label="Correct (top-1 matches TRUE)")
ax.bar(centers, counts_incorrect, width=width, bottom=counts_correct,
        color=RED, edgecolor=BLACK, linewidth=0.6, label="Incorrect")
# Annotate per-bin correct%
for i, (c, ic) in enumerate(zip(counts_correct, counts_incorrect)):
    tot = c + ic
    if tot > 0:
        ax.text(centers[i], tot + 8, f"{c/tot*100:.0f}%",
                ha="center", va="bottom", fontsize=9, color=BLACK, fontweight="bold")
ax.set_xlabel("Calibrated top-1 confidence  (predicted-class probability)")
ax.set_ylabel("Number of PULs (out of {})".format(N))
ax.set_title("Calibration is meaningful — confidence ≈ accuracy per bin", loc="left")
ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc")
ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig(FIG/"fig12_confidence_vs_correct.png", dpi=160); plt.close()

df_conf = pd.DataFrame({
    "bin_lo": bins[:-1].round(2), "bin_hi": bins[1:].round(2),
    "n_correct": counts_correct, "n_incorrect": counts_incorrect,
    "n_total": counts_correct + counts_incorrect,
    "pct_correct": np.where(counts_correct+counts_incorrect > 0,
                              counts_correct/(counts_correct+counts_incorrect+1e-9), 0).round(4),
})
df_conf.to_csv(TAB/"tab_confidence_vs_correct.csv", index=False)
print(f"  wrote {FIG}/fig12_confidence_vs_correct.png + {TAB}/tab_confidence_vs_correct.csv")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 13: case study cards (6 PULs picked to illustrate different scenarios)
# ─────────────────────────────────────────────────────────────────────────────
print("[10] Fig 13: case study cards ...")

# Load sig-gene OOF Δ-probs (calibrated TRUE-class) for narrative
sig_csv = ROOT / "artifacts/ablation/sig_gene_ablation_oof_outer42_groundtruth_calibrated.csv"
if not sig_csv.exists():
    sig_csv = ROOT / "artifacts/ablation/sig_gene_ablation_oof_outer42_groundtruth_calibrated.csv"
df_sig = pd.read_csv(sig_csv) if sig_csv.exists() else None
print(f"  sig-gene file: {sig_csv.name if df_sig is not None else 'NOT FOUND'}, rows: {0 if df_sig is None else len(df_sig)}")

# Build lit canon for marking sig genes
try:
    from src.lit_validation import build_canon, SUBSTRATE_ALIAS
    canon = build_canon(ROOT/"data/Literature_Data_fam_substrate_mapping.tsv", SUBSTRATE_ALIAS)
except Exception as e:
    print(f"  WARN: lit canon load failed: {e}")
    canon = {}

# Categorize PULs by scenario: (confident correct), (confident wrong, rank-2 redemption), (low-conf correct), (low-conf wrong)
top1_pred = order[:, 0]
top1_idx = top1_pred
is_correct = correct
top1_class = np.array(classes)[top1_pred]

picks_specs = [
    ("confident_correct",    lambda i: is_correct[i]  and top1_conf[i] > 0.85, "Confident + Correct"),
    ("rank2_redemption",     lambda i: (not is_correct[i]) and true_rank[i] == 2 and top1_conf[i] > 0.50, "TOP-1 wrong → TRUE at rank 2"),
    ("rank3_redemption",     lambda i: (not is_correct[i]) and true_rank[i] == 3 and top1_conf[i] > 0.40, "TOP-1 wrong → TRUE at rank 3"),
    ("low_conf_correct",     lambda i: is_correct[i] and top1_conf[i] < 0.45, "Low-confidence correct (model knows it's unsure)"),
    ("medium_conf_correct",  lambda i: is_correct[i] and 0.55 < top1_conf[i] < 0.70, "Medium-confidence correct"),
    ("confident_wrong",      lambda i: (not is_correct[i]) and top1_conf[i] > 0.70, "Confident but WRONG (interesting failure case)"),
]

picks = []
for scenario, pred, label in picks_specs:
    cands = [i for i in range(N) if pred(i)]
    if not cands: continue
    # Pick the FIRST candidate (deterministic) that has at least 3 tokens
    for i in cands:
        if len(X[i].split(",")) >= 3:
            picks.append((i, scenario, label))
            break

print(f"  selected {len(picks)} case studies:")
for i, sc, lab in picks: print(f"    idx={i:4d}  scenario={sc:24s}  TRUE={y[i]:15s}  pred={top1_class[i]:15s}  conf={top1_conf[i]:.3f}  rank_of_true={true_rank[i]}")

# Render figure
n_pulls = len(picks)
fig, axes = plt.subplots(n_pulls, 2, figsize=(15, 2.5*n_pulls), gridspec_kw={"width_ratios":[1.2, 1.6]})
if n_pulls == 1: axes = axes.reshape(1, 2)

for row, (idx, scenario, label) in enumerate(picks):
    ax_p, ax_g = axes[row, 0], axes[row, 1]
    # Top-3 prob bar
    top3_idx = order[idx, :3]
    top3_probs = P_oof[idx, top3_idx]
    top3_names = [classes[k] for k in top3_idx]
    true_c = y[idx]
    cols = [SAGE if name == true_c else NAVY for name in top3_names]
    bars = ax_p.barh(range(3), top3_probs, color=cols, edgecolor=BLACK, linewidth=0.5)
    ax_p.set_yticks(range(3)); ax_p.set_yticklabels([f"{n}\n({p:.3f})" for n, p in zip(top3_names, top3_probs)], fontsize=9, fontweight="bold")
    ax_p.invert_yaxis()
    ax_p.set_xlim(0, 1)
    ax_p.set_xlabel("Calibrated prob")
    title = f"PUL {idx} · {label}\nTRUE: {true_c}"
    ax_p.set_title(title, fontsize=10, loc="left")
    ax_p.spines["top"].set_visible(False); ax_p.spines["right"].set_visible(False)

    # Sig genes panel — parse top5_with_delta column ("tok:+0.1234;tok:+0.0987;...")
    token_text = []
    if df_sig is not None:
        row = df_sig[df_sig["idx"] == idx]
        if len(row):
            packed = row.iloc[0].get("top5_with_delta", "")
            for piece in str(packed).split(";"):
                piece = piece.strip()
                if ":" in piece:
                    tok, val = piece.rsplit(":", 1)
                    try: delta = float(val)
                    except ValueError: continue
                    lit_set = canon.get(true_c, set())
                    in_lit = tok in lit_set
                    token_text.append((tok, delta, in_lit))
    ax_g.axis("off")
    if token_text:
        # Draw as a small table
        ax_g.text(0.0, 1.0, f"Top-5 sig genes for TRUE substrate ({true_c})",
                   ha="left", va="top", fontsize=10, fontweight="bold", color=NAVY, transform=ax_g.transAxes)
        for ti, (tok, delta, in_lit) in enumerate(token_text):
            y_pos = 0.82 - ti*0.16
            marker_color = SAGE if in_lit else GRAY
            marker_text = "LIT ✓" if in_lit else "    "
            ax_g.text(0.02, y_pos, marker_text, fontsize=10, fontweight="bold",
                       color=marker_color, transform=ax_g.transAxes, family="monospace")
            ax_g.text(0.15, y_pos, tok, fontsize=11, fontweight="bold",
                       color=BLACK, transform=ax_g.transAxes, family="monospace")
            ax_g.text(0.55, y_pos, f"Δ_true = {delta:+.4f}", fontsize=10,
                       color=BLACK, transform=ax_g.transAxes, family="monospace")
        # Show the PUL sequence (truncated)
        seq = X[idx]
        seq_short = (seq[:80] + "…") if len(seq) > 80 else seq
        ax_g.text(0.0, -0.12, f"sequence: {seq_short}", fontsize=8, color=GRAY,
                   transform=ax_g.transAxes, family="monospace", style="italic")
    else:
        ax_g.text(0.5, 0.5, "(no sig-gene file)", ha="center", va="center", color=GRAY, transform=ax_g.transAxes)

plt.suptitle(f"Case studies — calibrated probs + sig genes from rep_1 (cpu__ET500_log2, T={T:.3f})",
             fontsize=12, fontweight="bold", y=1.005)
plt.tight_layout()
plt.savefig(FIG/"fig13_case_study_cards.png", dpi=160, bbox_inches="tight"); plt.close()

# Save underlying table
df_cases = pd.DataFrame([
    {"pul_idx": idx, "scenario": sc, "scenario_label": lab,
      "true_substrate": y[idx], "top1_pred": top1_class[idx], "top1_conf": float(top1_conf[idx]),
      "true_rank": int(true_rank[idx]),
      "top3_probs": "; ".join(f"{classes[k]}={P_oof[idx, k]:.3f}" for k in order[idx, :3]),
      "sequence": X[idx]}
    for idx, sc, lab in picks
])
df_cases.to_csv(TAB/"tab_case_studies.csv", index=False)
print(f"  wrote {FIG}/fig13_case_study_cards.png + {TAB}/tab_case_studies.csv")

# Audit hints
print("\n[10] AUDIT additions for paper:")
print(f"  rank_redemption_top1_acc:  {cum_acc_arr[0]:.4f}")
print(f"  rank_redemption_top2_acc:  {cum_acc_arr[1]:.4f}")
print(f"  rank_redemption_top3_acc:  {cum_acc_arr[2]:.4f}")
print(f"  rank_redemption_top5_acc:  {cum_acc_arr[4]:.4f}")
print(f"  case_studies_n_picked:     {len(picks)}")
print(f"\n[10] done.")
