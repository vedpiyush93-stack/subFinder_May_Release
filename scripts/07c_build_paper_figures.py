#!/usr/bin/env python3
"""Build every figure and table that appears in paper/main.tex.

All panels describe the deployed configuration (cpuV2__ET500_log2) and, where a
single split is shown, the same outer repeat, so the confusion matrix, the
per-substrate table, the signature-gene panel and the case studies all describe
one coherent set of predictions.

    python3 scripts/07c_build_paper_figures.py --split-seed 43

Outputs into paper/Fig/ :
    fig1_families.png     accuracy by model family over all 25 runs (box + points)
    fig2_confusion.png    row-normalised confusion matrix for one outer repeat
    fig3_siggenes.png     per-substrate top-3 signature genes, literature status
    fig4_funnel.png       per-substrate true-class attribution funnel at K=3
    fig5_cases.png        six worked examples
and into paper/tables/ :
    table_per_substrate.csv
"""
from __future__ import annotations
import argparse, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocessing.tokenizers import tok_cpu_v2
from src.lit_validation.canon import build_canon
from src.lit_validation.alias_map import SUBSTRATE_ALIAS

FIG = ROOT/"paper/Fig"; TAB = ROOT/"paper/tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
DEPLOYED = "cpuV2__ET500_log2"

# ---- house style -----------------------------------------------------------
INK, MUTED, RULE = "#1c2733", "#5b6b7c", "#d8dee5"
TEAL, AMBER, ROSE, SLATE = "#2a7f8f", "#c98a2b", "#b4585f", "#7a8794"
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "axes.grid": False,
})

def titled(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=14 if subtitle else 8)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.012), xycoords="axes fraction",
                    fontsize=8, color=MUTED, ha="left", va="bottom")

def family_of(short):
    if short.startswith(("cpu", "cpuV2", "ftCbow_MM")): return "Counts + ExtraTrees"
    if "BRF100" in short: return "Embedding + BalancedRF"
    if "__LSTMattn" in short: return "Deep: LSTM + attention"
    if "__LSTM" in short:    return "Deep: LSTM"
    if "__JustAttn" in short: return "Deep: attention"
    if "__Trans" in short:   return "Deep: transformer"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-seed", type=int, default=43)
    args = ap.parse_args()
    S = args.split_seed

    df = pd.read_csv(ROOT/"data/Train_data.csv")
    X = df["sig_gene_seq"].fillna("").values
    y = df["high_level_substr"].values
    subs = sorted(set(y))
    pfm = pd.read_csv(ROOT/"artifacts/per_fold_metrics.csv")

    # ---------------------------------------------------------------- FIG 1
    d = pfm[pfm.shorthand.isin(pfm.shorthand.unique())].copy()
    d["fam"] = d.shorthand.apply(family_of)
    order = d.groupby("fam").acc.mean().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    for i, fam in enumerate(order):
        v = d[d.fam == fam].acc.values
        bp = ax.boxplot(v, positions=[i], widths=0.52, vert=True, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, lw=1.4),
                        whiskerprops=dict(color=RULE, lw=1), capprops=dict(color=RULE, lw=1),
                        boxprops=dict(facecolor="#eef2f5", edgecolor=RULE, lw=1))
        rng = np.random.RandomState(0)
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=7,
                   color=TEAL if i == 0 else SLATE, alpha=0.55, zorder=3, linewidths=0)
        ax.scatter([i], [v.mean()], marker="D", s=22, color=AMBER, zorder=5,
                   edgecolor="white", linewidths=0.6)
    ax.set_xticks(range(len(order)))
    wrapped = [o.replace("Counts + ", "Counts +\n").replace("Embedding + ", "Embedding +\n")
                .replace("Deep: ", "Deep:\n") for o in order]
    ax.set_xticklabels(wrapped, fontsize=7.4)
    ax.set_ylabel("Test accuracy")
    ax.yaxis.grid(True, color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    titled(ax, "Accuracy by model family",
           f"every point is one of the 25 train/test runs; diamond = family mean; {len(d.shorthand.unique())} configurations")
    ax.legend(handles=[plt.Line2D([], [], marker="D", ls="", color=AMBER, label="family mean"),
                       plt.Line2D([], [], marker="o", ls="", color=SLATE, alpha=.6, label="single run")],
              loc="lower left", fontsize=7.5)
    plt.savefig(FIG/"fig1_families.png"); plt.close()

    # ------------------------------------------------- predictions for repeat S
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=S)
    y_pred = np.array([None]*len(X), dtype=object)
    for fold, (_, te) in enumerate(skf.split(X, y)):
        z = np.load(ROOT/f"artifacts/predictions/{DEPLOYED}/r{S}_f{fold}/probs_test.npz",
                    allow_pickle=True)
        cl = [str(c) for c in z["classes"]]
        y_pred[te] = np.array(cl)[z["probs"].argmax(1)]
    acc = float((y_pred == y).mean())

    # ---------------------------------------------------------------- FIG 2
    cm = confusion_matrix(y, y_pred, labels=subs).astype(float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(subs)):
        for j in range(len(subs)):
            v = cmn[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else INK,
                        fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(subs))); ax.set_yticks(range(len(subs)))
    ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(subs, fontsize=7.5)
    ax.set_xlabel("Predicted substrate"); ax.set_ylabel("True substrate")
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.03)
    cb.set_label("Row-normalised share", fontsize=8); cb.outline.set_visible(False)
    titled(ax, f"Confusion matrix, outer repeat {S}",
           f"cell = % of that true substrate; rows sum to 100. All {len(X):,} PULs, "
           f"each held out exactly once. Accuracy {acc:.4f}")
    plt.savefig(FIG/"fig2_confusion.png"); plt.close()

    p, r, f1, sup = precision_recall_fscore_support(y, y_pred, labels=subs,
                                                    average=None, zero_division=0)
    per = pd.DataFrame({"substrate": subs, "n": sup, "precision": p,
                        "recall": r, "f1": f1}).sort_values("f1", ascending=False)
    per.to_csv(TAB/"table_per_substrate.csv", index=False)

    # ---------------------------------------------------------------- FIG 3/4
    canon = build_canon(ROOT/"data/Literature_Data_fam_substrate_mapping.tsv", SUBSTRATE_ALIAS)
    import re as _re
    canon_aug = {s: fams | {_re.match(r"^(GH|PL|CE|CBM|GT|AA)", f).group(1)
                            for f in fams if _re.match(r"^(GH|PL|CE|CBM|GT|AA)[0-9]", f)}
                 for s, fams in canon.items()}
    abl = pd.read_csv(ROOT/f"artifacts/ablation/sig_gene_ablation_oof_outer{S}_v2.csv")
    abl["top3"] = abl["top3"].fillna("").astype(str)

    # top-3 tokens per substrate, ranked by how often they are a signature gene
    rows = []
    for s in subs:
        cnt = {}
        for t3 in abl[abl.true == s].top3.fillna('').astype(str):
            for tok in t3.split(";"):
                if tok and tok != "nan": cnt[tok] = cnt.get(tok, 0) + 1
        for tok, n in sorted(cnt.items(), key=lambda kv: -kv[1])[:3]:
            status = ("exact" if tok in canon[s] else
                      "family" if tok in canon_aug[s] else "not in DB")
            rows.append(dict(substrate=s, token=tok, n=n, status=status))
    sg = pd.DataFrame(rows)
    sg.to_csv(TAB/"table_top3_siggenes.csv", index=False)

    COLOR = {"exact": TEAL, "family": AMBER, "not in DB": SLATE}
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ypos, ylab = [], []
    for i, s in enumerate(subs[::-1]):
        block = sg[sg.substrate == s]
        for j, (_, rr) in enumerate(block.iterrows()):
            yy = i + (j - 1) * 0.28
            ax.barh(yy, rr.n, height=0.24, color=COLOR[rr.status], alpha=.92, linewidth=0)
            ax.text(rr.n + 2, yy, rr.token, va="center", fontsize=6.4, color=INK)
        ypos.append(i); ylab.append(s)
    ax.set_yticks(ypos); ax.set_yticklabels(ylab, fontsize=8)
    ax.set_xlabel("PULs in which the token was a top-3 signature gene")
    ax.xaxis.grid(True, color=RULE, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(0, sg.n.max()*1.22)
    titled(ax, "Top-3 signature genes per substrate, and whether the literature agrees",
           f"outer repeat {S}; colour = status in the curated enzyme table")
    ax.legend(handles=[Patch(facecolor=COLOR[k], label=v) for k, v in
                       [("exact", "listed for this substrate"),
                        ("family", "family-level match"),
                        ("not in DB", "not a CAZy family in the table")]],
              loc="lower right", fontsize=7.5)
    plt.savefig(FIG/"fig3_siggenes.png"); plt.close()

    # funnel: per substrate, eligible vs hit at K=3
    fr = []
    for s in subs:
        blk = abl[abl.true == s]
        elig = sum(1 for _, rr in blk.iterrows() if set(tok_cpu_v2(X[rr.idx])) & canon_aug[s])
        hit = sum(1 for _, rr in blk.iterrows()
                  if (set(tok_cpu_v2(X[rr.idx])) & canon_aug[s])
                  and (set(str(rr.top3).split(";")) & canon_aug[s]))
        fr.append(dict(substrate=s, n=len(blk), eligible=elig, hit=hit,
                       rate=hit/max(elig, 1)))
    fu = pd.DataFrame(fr).sort_values("rate", ascending=True)
    fu.to_csv(TAB/"table_sig_funnel.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    yy = np.arange(len(fu))
    ax.barh(yy, fu.eligible, height=.62, color="#e9eef2", linewidth=0, label="PULs containing a documented family")
    ax.barh(yy, fu.hit, height=.62, color=TEAL, linewidth=0, label="…and the model used one in its top 3")
    for i, (_, rr) in enumerate(fu.iterrows()):
        ax.text(rr.eligible + 3, i, f"{rr.rate*100:.0f}%", va="center", fontsize=7.5,
                color=INK, fontweight="bold")
    ax.set_yticks(yy); ax.set_yticklabels(fu.substrate, fontsize=8)
    ax.set_xlabel("Number of PULs")
    ax.xaxis.grid(True, color=RULE, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(0, fu.eligible.max()*1.16)
    tot_e, tot_h = int(fu.eligible.sum()), int(fu.hit.sum())
    titled(ax, "Does the model lean on the enzymes the literature says it should?",
           f"outer repeat {S}, true-substrate attribution, K=3 — overall {tot_h}/{tot_e} = {tot_h/tot_e*100:.1f}%")
    ax.legend(loc="lower right", fontsize=7.5)
    plt.savefig(FIG/"fig4_funnel.png"); plt.close()

    # ---------------------------------------------------------------- FIG 5
    # Worked examples must come from HELD-OUT predictions. The deployed model was
    # fitted on all 1,030 PULs, so scoring a training PUL with it reports
    # memorisation (probabilities pin to 1.00) rather than what a user would see.
    # We therefore reuse this repeat's saved out-of-fold probabilities and apply
    # the deployed temperature to them.
    import joblib
    T = joblib.load(ROOT/"artifacts/final_model_v2.pkl")["T"]
    P = np.zeros((len(X), len(subs))); cls = None
    for fold, (_, te) in enumerate(skf.split(X, y)):
        z = np.load(ROOT/f"artifacts/predictions/{DEPLOYED}/r{S}_f{fold}/probs_test.npz",
                    allow_pickle=True)
        cls = [str(c) for c in z["classes"]]
        P[te] = z["probs"]
    lg = np.log(np.clip(P, 1e-12, None))/T
    P = np.exp(lg - lg.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)

    top3 = {int(rr.idx): str(rr.top3) for _, rr in abl.iterrows()}
    C = pd.DataFrame([dict(
            idx=i, true=y[i], pred=cls[int(P[i].argmax())], conf=float(P[i].max()),
            rank=int(list(np.argsort(-P[i])).index(cls.index(y[i]))) + 1,
            sig=top3.get(i, ""))
        for i in range(len(X)) if top3.get(i)])

    used, seen_sub, picks = set(), set(), []
    def take(mask, label):
        """Pick one example, preferring a substrate not already illustrated so the
        panel shows six different biologies rather than six alginate loci."""
        sel = C[mask & ~C.idx.isin(used)]
        if not len(sel): return
        fresh = sel[~sel.true.isin(seen_sub)]
        rr = (fresh if len(fresh) else sel).iloc[0]
        used.add(rr.idx); seen_sub.add(rr.true); picks.append((rr, label))
    take((C.true == C.pred) & (C.conf > .90) & (C.sig != ""), "Confident and correct")
    take((C.true == C.pred) & C.conf.between(.40, .60), "Correct but hedged")
    take((C.true != C.pred) & (C["rank"] == 2), "Wrong; truth ranks 2nd")
    take((C.true != C.pred) & (C.conf > .60), "Confidently wrong")
    take((C.true == C.pred) & (C.true == "chitin"), "Right, no canonical enzyme")
    take((C.true == C.pred) & C.conf.between(.60, .85), "Typical case")
    while len(picks) < 6:                       # never leave an empty panel
        take(pd.Series(True, index=C.index), "Additional example")

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.8))
    fig.subplots_adjust(hspace=1.05, wspace=0.16, top=0.80)
    for ax, (rr, label) in zip(axes.ravel(), picks):
        order3 = np.argsort(-P[rr.idx])[:3]
        for k, t in enumerate(order3):
            v = float(P[rr.idx][t])
            good = cls[t] == rr.true
            ax.barh(2 - k, v, color=TEAL if good else ROSE, height=.62, linewidth=0)
            inside = v > 0.42
            ax.text(v - 0.03 if inside else v + 0.03, 2 - k,
                    f"{cls[t]} {v:.2f}", va="center",
                    ha="right" if inside else "left", fontsize=6.3,
                    color="white" if inside else INK)
        ax.set_xlim(0, 1.02); ax.set_ylim(-0.6, 2.6)
        ax.set_yticks([]); ax.set_xticks([0, .5, 1]); ax.tick_params(labelsize=6)
        ax.set_title(label, loc="left", fontsize=7.6, fontweight="bold", color=INK, pad=27)
        sig = rr.sig.replace(";", ", ") or "none above threshold"
        ax.annotate(f"true: {rr.true}\nsignature genes: {sig}",
                    xy=(0, 1.03), xycoords="axes fraction", fontsize=6.0,
                    color=MUTED, ha="left", va="bottom", linespacing=1.6)
        for sp in ax.spines.values(): sp.set_visible(False)
    fig.suptitle("Six worked examples, all held-out predictions", x=0.005, ha="left",
                 fontsize=10.5, fontweight="bold", color=INK, y=0.99)
    fig.text(0.005, 0.945, "three highest calibrated probabilities; teal = the true "
             "substrate, red = a wrong one", fontsize=7.6, color=MUTED, ha="left")
    plt.savefig(FIG/"fig5_cases.png", bbox_inches="tight"); plt.close()
    pd.DataFrame([dict(scenario=lab, idx=int(rr.idx), true=rr.true, pred=rr.pred,
                       confidence=round(float(rr.conf), 4), true_rank=int(rr["rank"]),
                       signature_genes=rr.sig) for rr, lab in picks]
                 ).to_csv(TAB/"table_case_studies.csv", index=False)

    print(f"[07c] outer repeat {S}: accuracy {acc:.4f}")
    print(f"[07c] funnel: {tot_h}/{tot_e} = {tot_h/tot_e*100:.1f}%")
    print(f"[07c] wrote 5 figures to {FIG} and 4 tables to {TAB}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
