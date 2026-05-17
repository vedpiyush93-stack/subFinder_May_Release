#!/usr/bin/env python3
"""Aggregate per-config means across the 5 reproducibility reps to quantify model
uncertainty due to model-init seed (REPRO_REP_SEED=1000/2000/3000/4000/5000) with
the 5×5 RSKF data splits HELD FIXED.

Outputs (all in docs/):
    docs/tables/tab_cross_rep_stability.csv       29-config × 5-rep matrix + cross-rep mean/std
    docs/tables/tab_cross_rep_top7_ranking.csv    top-7 rank in every rep (stability check)
    docs/figures/fig14_cross_rep_stability.png    forest plot — 5 dots per config, sorted by rep_1 mean

Usage:
    python3 scripts/11_build_cross_rep_stability.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
OUT_FIG = ROOT / "docs" / "figures"; OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB = ROOT / "docs" / "tables";  OUT_TAB.mkdir(parents=True, exist_ok=True)

REP_SEEDS = {1: 1000, 2: 2000, 3: 3000, 4: 4000, 5: 5000}
REPS = list(REP_SEEDS.keys())

print("[cross-rep] loading 5 rep leaderboards ...")
rep_lbs = {i: pd.read_csv(ROOT / f"reproducibility/rep_{i}/leaderboard.csv") for i in REPS}
all_lb = pd.concat([df.assign(rep=i) for i, df in rep_lbs.items()])

# Per-config per-rep mean + n_trials
mean_pv = all_lb.pivot(index="shorthand", columns="rep", values="mean_acc")
std_pv  = all_lb.pivot(index="shorthand", columns="rep", values="std_acc")
n_pv    = all_lb.pivot(index="shorthand", columns="rep", values="n_trials")

# Cross-rep aggregate: take each rep's 25-trial mean, then mean/std across the 5 reps.
mean_pv["cross_rep_mean"] = mean_pv[REPS].mean(axis=1)
mean_pv["cross_rep_std"]  = mean_pv[REPS].std(axis=1)
mean_pv["cross_rep_min"]  = mean_pv[REPS].min(axis=1)
mean_pv["cross_rep_max"]  = mean_pv[REPS].max(axis=1)
mean_pv["min_n_trials"]   = n_pv[REPS].min(axis=1).astype(int)

# Sort by rep_1 (canonical benchmark) mean, descending — same order the paper leaderboard uses.
mean_pv = mean_pv.sort_values(1, ascending=False).reset_index()
# Rename rep columns for the CSV
csv_out = mean_pv.rename(columns={i: f"rep_{i}_mean" for i in REPS})
csv_path = OUT_TAB / "tab_cross_rep_stability.csv"
csv_out.to_csv(csv_path, index=False, float_format="%.4f")
print(f"  wrote {csv_path.relative_to(ROOT)} ({len(csv_out)} configs)")

# Top-7 ranking per rep
print("\n[cross-rep] top-7 ranking stability:")
rank_rows = []
for i in REPS:
    top_i = all_lb[all_lb.rep == i].sort_values("mean_acc", ascending=False).reset_index(drop=True)
    for rk in range(7):
        rank_rows.append({"rep": i, "rank": rk + 1, "config": top_i.shorthand.iloc[rk],
                          "mean_acc": top_i.mean_acc.iloc[rk]})
rank_df = pd.DataFrame(rank_rows)
rank_pv = rank_df.pivot(index="rank", columns="rep", values="config")
rank_path = OUT_TAB / "tab_cross_rep_top7_ranking.csv"
rank_pv.to_csv(rank_path)
print(rank_pv.to_string())
# Stability score: # of (rank, config) tuples that appear in all 5 reps
stable_count = sum(1 for rk in range(1, 8)
                   if all(rank_df[(rank_df["rank"] == rk) & (rank_df.rep == i)].config.iloc[0]
                          == rank_df[(rank_df["rank"] == rk) & (rank_df.rep == 1)].config.iloc[0] for i in REPS))
print(f"\n  STABLE: {stable_count}/7 top-7 ranks are IDENTICAL across all 5 reps")

# Headline numbers
winner = mean_pv[mean_pv.shorthand == "cpu__ET500_log2"].iloc[0]
runner = mean_pv[mean_pv.shorthand == "ftCbow_MM__ET500_sqrt"].iloc[0]
print(f"\n[cross-rep] HEADLINE (5-rep aggregate):")
print(f"  cpu__ET500_log2 (winner):      {winner.cross_rep_mean:.4f} ± {winner.cross_rep_std:.4f}  "
      f"(range {winner.cross_rep_min:.4f}–{winner.cross_rep_max:.4f})")
print(f"  ftCbow_MM__ET500_sqrt (2nd):   {runner.cross_rep_mean:.4f} ± {runner.cross_rep_std:.4f}  "
      f"(range {runner.cross_rep_min:.4f}–{runner.cross_rep_max:.4f})")

# Per-family cross-rep std (median) — quantify which families are most/least reproducible
fam_lookup = {
    "cpu__ET500_log2": "OvR(ExtraTrees)", "ftCbow_MM__ET500_sqrt": "OvR(ExtraTrees)",
    "cv__BRF100": "OvR(BalancedRF)",
}
for c in mean_pv.shorthand:
    if c in fam_lookup: continue
    if c.endswith("__BRF100"): fam_lookup[c] = "OvR(BalancedRF)"
    elif c.endswith("__LSTM"): fam_lookup[c] = "DL: LSTM"
    elif c.endswith("__LSTMattn"): fam_lookup[c] = "DL: LSTM+attention"
    elif c.endswith("__JustAttn"): fam_lookup[c] = "DL: attention"
    elif c.endswith("__Trans"): fam_lookup[c] = "DL: transformer"
mean_pv["family"] = mean_pv.shorthand.map(fam_lookup)
fam_std = mean_pv.groupby("family")["cross_rep_std"].agg(["median", "mean", "max"]).sort_values("median")
print("\n[cross-rep] per-family cross-rep std (median across configs in family):")
print(fam_std.to_string())

# ============================================================================
# FIGURE: forest plot — for each config, plot the 5 per-rep means as dots + a
# horizontal line at the cross-rep mean. Color by family. Order by rep_1 mean (desc).
# ============================================================================
print("\n[cross-rep] building fig14 forest plot ...")
FAM_COLOR = {"OvR(ExtraTrees)": "#27ae60", "OvR(BalancedRF)": "#1a3a5c",
             "DL: LSTM+attention": "#e67e22", "DL: transformer": "#8e44ad",
             "DL: attention": "#c0392b", "DL: LSTM": "#7f8c8d"}
n_cfg = len(mean_pv)
fig, ax = plt.subplots(figsize=(11.5, 9.5))
y_pos = np.arange(n_cfg)[::-1]  # top of chart = best config (rep_1)
for i, row in mean_pv.iterrows():
    y = y_pos[i]
    color = FAM_COLOR.get(row.family, "#7f8c8d")
    # Range bar
    ax.hlines(y, row.cross_rep_min, row.cross_rep_max, color=color, alpha=0.35, linewidth=2, zorder=1)
    # Individual rep dots
    for rep_i in REPS:
        ax.scatter(row[rep_i], y, s=24, color=color, edgecolor="white",
                   linewidth=0.6, zorder=3)
    # Cross-rep mean: small filled square
    ax.scatter(row.cross_rep_mean, y, s=42, marker="s", color=color,
               edgecolor="black", linewidth=0.8, zorder=4)
    # Right-side text: mean ± std + min_n note for partial reps
    note = "" if row.min_n_trials == 25 else f"  (min n={row.min_n_trials})"
    label = f"  {row.cross_rep_mean:.4f} ± {row.cross_rep_std:.4f}{note}"
    ax.text(row.cross_rep_max + 0.001, y, label, va="center", fontsize=7.5,
            family="monospace", color="#2c3e50")

# Y-axis: config name + family in parens
ytick_labels = [f"{r.shorthand} ({r.family.replace('OvR(', '').replace(')','')})" for _, r in mean_pv.iterrows()]
ax.set_yticks(y_pos); ax.set_yticklabels(ytick_labels, fontsize=8)
ax.set_xlabel("Test accuracy (mean across 25 trials per rep, 5 reps shown)", fontsize=10, fontweight="bold")
ax.set_title("Cross-rep reproducibility — 5 reps × 25 trials each, data splits FIXED, model-init seed varies (REPRO_REP_SEED=1000/2000/3000/4000/5000)\n"
             f"Winner cpu__ET500_log2: {winner.cross_rep_mean:.4f} ± {winner.cross_rep_std:.4f}  ·  Top-7 ranking identical in {stable_count}/7 positions across all 5 reps",
             fontsize=10, color="#1a3a5c", fontweight="bold", loc="left")
ax.set_xlim(0.69, 0.93)
ax.grid(axis="x", alpha=0.3, linewidth=0.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Family legend
legend_h = [Patch(facecolor=c, label=fam) for fam, c in FAM_COLOR.items()]
legend_h.append(Patch(facecolor="white", edgecolor="black", label="dot = single rep mean (5 dots/row) · square = cross-rep mean · bar = min↔max"))
ax.legend(handles=legend_h, loc="lower right", fontsize=7.5, framealpha=0.95,
          title="Classifier family", title_fontsize=8)
plt.tight_layout()
fig_path = OUT_FIG / "fig14_cross_rep_stability.png"
plt.savefig(fig_path, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"  wrote {fig_path.relative_to(ROOT)}")

# Write a JSON summary for downstream consumers (audit_output, README, paper)
summary = {
    "n_reps": len(REPS), "rep_seeds": list(REP_SEEDS.values()),
    "winner_config": "cpu__ET500_log2",
    "winner_cross_rep_mean":  round(float(winner.cross_rep_mean), 4),
    "winner_cross_rep_std":   round(float(winner.cross_rep_std),  4),
    "winner_cross_rep_min":   round(float(winner.cross_rep_min),  4),
    "winner_cross_rep_max":   round(float(winner.cross_rep_max),  4),
    "winner_per_rep_means":   {f"rep_{i}": round(float(winner[i]), 4) for i in REPS},
    "runner_config": "ftCbow_MM__ET500_sqrt",
    "runner_cross_rep_mean":  round(float(runner.cross_rep_mean), 4),
    "runner_cross_rep_std":   round(float(runner.cross_rep_std),  4),
    "top7_rank_stability":    f"{stable_count}/7 ranks identical across all 5 reps",
    "n_configs_min_n_25":     int((mean_pv.min_n_trials == 25).sum()),
    "n_configs_partial":      int((mean_pv.min_n_trials < 25).sum()),
}
sum_path = OUT_TAB / "tab_cross_rep_summary.json"
sum_path.write_text(json.dumps(summary, indent=2))
print(f"  wrote {sum_path.relative_to(ROOT)}")
print("\n[cross-rep] done.")
