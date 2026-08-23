#!/usr/bin/env python3
"""Build every figure, table and number that paper/main.tex renders.

Nothing in the manuscript is typed by hand. This script writes

    paper/Fig/*.png          the five figures
    paper/generated/*.tex    the tables, and \\newcommand macros for every number
    paper/tables/*.csv       the same content as data

and main.tex \\input's the generated .tex, so re-running this is the only way the
manuscript's contents change.

Single-split panels (confusion matrix, per-substrate table, signature genes,
worked examples) all describe the same set of held-out predictions: one complete
outer repeat, in which each of the five folds is held out in turn so that every
locus is predicted exactly once by a model that never saw it.

    python3 scripts/07c_build_paper_figures.py
"""
from __future__ import annotations
import argparse, re, sys, warnings
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

FIG = ROOT/"paper/Fig"; TAB = ROOT/"paper/tables"; GEN = ROOT/"paper/generated"
for d in (FIG, TAB, GEN): d.mkdir(parents=True, exist_ok=True)
DEPLOYED = "cpuV2__ET500_log2"
SKIP = {"null", ""}   # never shown to a user, in any displayed gene list

INK, MUTED, RULE = "#111820", "#3f4c59", "#c9d2da"
TEAL, AMBER, ROSE, SLATE = "#1f6f7f", "#b8791f", "#a8434b", "#63707d"
plt.rcParams.update({
    "figure.dpi": 320, "savefig.dpi": 320, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.edgecolor": RULE, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "legend.frameon": False, "legend.fontsize": 10.5, "axes.grid": False,
})

MACROS: dict[str, str] = {}
def macro(name, value):
    MACROS[name] = str(value)

def titled(ax, title, subtitle=None, tfs=14, sfs=11):
    ax.set_title(title, loc="left", fontsize=tfs, fontweight="bold",
                 color=INK, pad=20 if subtitle else 10)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.015), xycoords="axes fraction",
                    fontsize=sfs, color=MUTED, ha="left", va="bottom")

def family_of(short):
    """Group a configuration by what it actually is.

    The representation is the part before "__" and the classifier the part after, so
    both have to be read. An earlier version keyed on a prefix list, which put the
    embedding config ftCbow_MM__ET500_sqrt in the counts group and the counts
    baseline cv__BRF100 in the embedding group -- mislabelling both boxes of the
    benchmark figure.
    """
    rep = "Counts" if short.split("__")[0] in ("cpu", "cpuV2", "cv") else "Embedding"
    if "__LSTMattn" in short: return "Deep: LSTM + attention"
    if "__LSTM" in short:    return "Deep: LSTM"
    if "__JustAttn" in short: return "Deep: attention"
    if "__Trans" in short:   return "Deep: transformer"
    if "BRF100" in short: return f"{rep} + BalancedRF"
    if "ET500" in short:  return f"{rep} + ExtraTrees"
    return "other"



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-seed", type=int, default=43)
    args = ap.parse_args(); S = args.split_seed

    df = pd.read_csv(ROOT/"data/Train_data.csv")
    X = df["sig_gene_seq"].fillna("").values
    y = df["high_level_substr"].values
    subs = sorted(set(y))
    pfm = pd.read_csv(ROOT/"artifacts/per_fold_metrics.csv")
    lb  = pd.read_csv(ROOT/"artifacts/leaderboard.csv")

    macro("NPuls", f"{len(X):,}".replace(",", "{,}"))
    macro("NClasses", len(subs))
    macro("NConfigs", pfm.shorthand.nunique())
    macro("NRuns", pfm.shorthand.nunique()*25)

    # ============================================================ FIG 1
    d = pfm.copy(); d["fam"] = d.shorthand.apply(family_of)
    order = d.groupby("fam").acc.mean().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    for i, fam in enumerate(order):
        v = d[d.fam == fam].acc.values
        ax.boxplot(v, positions=[i], widths=0.55, patch_artist=True, showfliers=False,
                   medianprops=dict(color=INK, lw=1.8),
                   whiskerprops=dict(color=SLATE, lw=1.1), capprops=dict(color=SLATE, lw=1.1),
                   boxprops=dict(facecolor="#eef2f5", edgecolor=SLATE, lw=1.1))
        rng = np.random.RandomState(0)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=11,
                   color=TEAL if i == 0 else SLATE, alpha=0.5, zorder=3, linewidths=0)
        ax.scatter([i], [v.mean()], marker="D", s=42, color=AMBER, zorder=5,
                   edgecolor="white", linewidths=0.9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([o.replace("Counts + ", "Counts +\n").replace("Embedding + ", "Embedding +\n")
                        .replace("Deep: ", "Deep:\n") for o in order], fontsize=10.5)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.yaxis.grid(True, color=RULE, lw=0.7); ax.set_axisbelow(True)
    titled(ax, "Accuracy by model family",
           "each point is one train/test run; diamond marks the family mean")
    ax.legend(handles=[plt.Line2D([], [], marker="D", ls="", color=AMBER, label="family mean"),
                       plt.Line2D([], [], marker="o", ls="", color=SLATE, alpha=.6, label="single run")],
              loc="lower left")
    plt.savefig(FIG/"fig1_families.png"); plt.close()

    # -------------------------------------------------- held-out predictions
    # Every per-locus figure and rate below comes from the SAME protocol as the
    # benchmark table: 5 repeats x 5 folds. Reporting them from a single 5-fold
    # pass, as an earlier version did, put two different numbers on the same
    # quantity -- 0.9272 in the text against 0.9177 in the benchmark -- with
    # nothing to tell a reader they were measured differently. Each locus is
    # therefore held out REPS times and contributes that many predictions.
    import joblib
    REPS = (42, 43, 44, 45, 46)
    y_rep, P_list, inform_list, foldacc = [], [], [], []
    cls = None
    for seed in REPS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        order, Ps, ninf = [], [], []
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            z = np.load(ROOT/f"artifacts/predictions/{DEPLOYED}/r{seed}_f{fold}/probs_test.npz",
                        allow_pickle=True)
            cls = [str(c) for c in z["classes"]]
            order.append(te); Ps.append(z["probs"])
            foldacc.append(float((np.array(cls)[z["probs"].argmax(1)] == y[te]).mean()))
            # informative = in THIS fold's training vocabulary and not the padding
            # token. min_df=1, so the fitted vocabulary is exactly the training tokens.
            fv = set(t for i in tr for t in tok_cpu_v2(X[i]))
            ninf.append(np.array([sum(1 for t in tok_cpu_v2(X[i])
                                      if t != "null" and t in fv) for i in te]))
        idx = np.concatenate(order)
        P_list.append(np.concatenate(Ps)); y_rep.append(y[idx])
        inform_list.append(np.concatenate(ninf))
    P        = np.concatenate(P_list)            # 5150 x 12
    y_all    = np.concatenate(y_rep)             # the truth aligned to P
    n_inform = np.concatenate(inform_list)
    y_pred   = np.array(cls)[P.argmax(1)]
    # per-substrate vote fractions from the saved per-fold classifiers: the
    # p-values are computed from these counts, not from the normalised vector
    from scipy.stats import binom
    _cache = ROOT/"artifacts/oof_vote_fractions.npz"
    if _cache.exists():
        Vv = np.load(_cache)["V"]
    else:
        vv = []
        for seed in REPS:
            sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (_, te) in enumerate(sk.split(X, y)):
                pl = joblib.load(ROOT/f"artifacts/predictions/{DEPLOYED}/r{seed}_f{fold}/classifier.joblib")
                Zt = pl.named_steps["cv"].transform(X[te])
                vv.append(np.column_stack([e.predict_proba(Zt)[:, 1]
                                           for e in pl.named_steps["vr"].estimators_]))
        Vv = np.concatenate(vv); np.savez_compressed(_cache, V=Vv)
    NTREE = 500
    PV = binom.sf(np.rint(Vv*NTREE).astype(int) - 1, NTREE, 0.5)
    macro("NTrees", NTREE)
    macro("VoteThresh", int(binom.isf(0.05/len(subs), NTREE, 0.5)) + 1)
    macro("NReps", len(REPS))
    macro("NPredictions", f"{len(y_all):,}".replace(",", "{,}"))
    # quoted the way the benchmark quotes it: mean and s.d. over the 25 fits
    macro("HeldOutAcc", f"{np.mean(foldacc):.4f}")
    macro("HeldOutSd",  f"{np.std(foldacc, ddof=1):.4f}")

    # The six worked examples are individual loci, so they need one named pass
    # rather than a pooled estimate: a locus held out five times has five
    # probability vectors and no single one of them is "the" answer. Repeat S is
    # used, and the manuscript says so where the examples are presented.
    P_one = np.zeros((len(X), len(subs)))
    for fold, (_, te) in enumerate(StratifiedKFold(n_splits=5, shuffle=True,
                                                   random_state=S).split(X, y)):
        z = np.load(ROOT/f"artifacts/predictions/{DEPLOYED}/r{S}_f{fold}/probs_test.npz",
                    allow_pickle=True)
        P_one[te] = z["probs"]
    macro("CaseSeed", S)

    from src.calibration.temperature import apply_temperature
    _bundle = joblib.load(ROOT/"artifacts/final_model_v2.pkl")
    T = float(_bundle["T"]); pipe_final = _bundle["pipeline"]
    NTREES_DEPLOYED = int(pipe_final.named_steps["vr"].estimators_[0].n_estimators)
    # must be the same transform the deployed model applies (per-class logit / T,
    # sigmoid, renormalise) -- a softmax-style log(p)/T is a different operation
    # and would report probabilities no user ever sees.
    Pc = apply_temperature(P, T)
    Pc_one = apply_temperature(P_one, T)
    # Two index spaces exist from here on and mixing them is silent:
    #   pooled  (P, Pc, y_all, correct, Vv, PV) -- one row per PREDICTION
    #   by-locus(P_one, Pc_one, X, y)           -- one row per LOCUS
    assert P.shape[0] == len(y_all) == len(REPS)*len(X), "pooled arrays misaligned"
    assert P_one.shape[0] == len(X) == len(y), "per-locus arrays misaligned"
    assert Vv.shape[0] == len(y_all), "vote fractions not aligned to pooled truth"
    macro("DeployT", f"{T:.4f}")

    EXAMPLE_PUL = "1.B.14.12.1,GntR,PL6|PL6_1,PL17_2|PL17,2.A.1.14.25,null"

    # the winning panel's vote count for the worked example, quoted in the text
    _z0 = pipe_final.named_steps["cv"].transform([EXAMPLE_PUL])
    _v0 = np.column_stack([e.predict_proba(_z0)[:, 1]
                           for e in pipe_final.named_steps["vr"].estimators_])[0]
    macro("VoteExampleTop", f"{int(round(_v0.max()*500))}")

    # ============================================================ FIG 0 (workflow)
    # What the tool does to one locus, end to end. Every label here is produced by
    # the deployed pipeline, not drawn by hand, so the figure cannot drift from it.
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    ex_toks = tok_cpu_v2(EXAMPLE_PUL)
    _bx = pipe_final.named_steps["cv"].transform([EXAMPLE_PUL])
    _bv = np.column_stack([e.predict_proba(_bx)[:, 1]
                           for e in pipe_final.named_steps["vr"].estimators_])[0]
    _bp = apply_temperature(pipe_final.predict_proba([EXAMPLE_PUL]), T)[0]
    _bi = int(_bp.argmax())
    _btop = np.argsort(-_bp)[:3]

    from src.ablation.leave_one_token_out import ablate_pul_for_class
    _d = ablate_pul_for_class(pipe_final, EXAMPLE_PUL, cls[_bi], top_k=5, apply_temp=T)
    _ex_genes = ", ".join([t for t, _ in _d if t not in SKIP and not t.isdigit()][:3])

    fig = plt.figure(figsize=(13.4, 4.85))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 134); ax.set_ylim(0, 50); ax.axis("off")

    ACC  = "#0e5c6b"        # deeper than TEAL, for the accents
    WASH = "#eef4f5"
    def stage(x, w, n, title, sub, y=7.6, h=29.9):
        # a numbered band across the top of each panel carries the step and its title
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                    facecolor="white", edgecolor=RULE, lw=1.0, zorder=1))
        ax.add_patch(FancyBboxPatch((x, y + h - 6.6), w, 6.6,
                                    boxstyle="round,pad=0,rounding_size=1.4",
                                    facecolor=ACC, edgecolor="none", zorder=2))
        ax.add_patch(plt.Rectangle((x, y + h - 6.6), w, 1.6, facecolor=ACC,
                                   edgecolor="none", zorder=2))
        ax.text(x + 3.4, y + h - 3.3, n, ha="center", va="center", fontsize=10,
                fontweight="bold", color=ACC, zorder=4,
                bbox=dict(boxstyle="circle,pad=0.30", facecolor="white", edgecolor="none"))
        ax.text(x + 7.2, y + h - 3.3, title, ha="left", va="center", fontsize=11.2,
                fontweight="bold", color="white", zorder=4)
        ax.text(x + w/2, y + h - 9.6, sub, ha="center", va="center", fontsize=9.2,
                color=MUTED, style="italic", zorder=4)
    def flow(x0, x1):
        ax.add_patch(FancyArrowPatch((x0, 21.5), (x1, 21.5), arrowstyle="-|>",
                                     mutation_scale=17, lw=1.6, color=ACC, zorder=5))

    W = 29.5
    stage(1.5, W, "1", "The locus", "as an annotation pipeline reports it")
    for k, ln in enumerate(["1.B.14.12.1   GntR", "PL6|PL6_1   PL17_2|PL17",
                            "2.A.1.14.25   null"]):
        ax.add_patch(plt.Rectangle((3.6, 21.0 - k*5.0), 25.3, 3.9, facecolor=WASH,
                                   edgecolor="none", zorder=2))
        ax.text(16.25, 22.95 - k*5.0, ln, ha="center", va="center",
                fontsize=9.3, color=INK, family="monospace", zorder=3)
    flow(31.6, 35.4)

    stage(35.5, W, "2", "Its gene families", f"{len(ex_toks)} tokens, transporters at family level")
    for k in range(0, len(ex_toks), 3):
        row = ex_toks[k:k+3]
        for c, tk in enumerate(row):
            bx = 38.0 + c*8.6
            ax.add_patch(FancyBboxPatch((bx, 21.0 - (k//3)*5.0), 7.6, 3.9,
                                        boxstyle="round,pad=0,rounding_size=0.7",
                                        facecolor=WASH, edgecolor="none", zorder=2))
            ax.text(bx + 3.8, 22.95 - (k//3)*5.0, tk, ha="center", va="center",
                    fontsize=8.6, color=INK, family="monospace", zorder=3)
    flow(65.6, 69.4)

    stage(69.5, W, "3", "Twelve panels vote", f"{NTREE} trees each, yes or no")
    for k, ci in enumerate(_btop):
        v = float(_bv[ci]); yy = 22.9 - k*5.0
        ax.add_patch(plt.Rectangle((71.8, yy - 1.95), 24.9, 3.9, facecolor=WASH,
                                   edgecolor="none", zorder=2))
        ax.add_patch(plt.Rectangle((71.8, yy - 1.95), 24.9*v, 3.9,
                                   facecolor=ACC if ci == _bi else SLATE,
                                   alpha=1 if ci == _bi else .28, edgecolor="none", zorder=3))
        w_ = v > .5 and ci == _bi
        ax.text(73.0, yy, cls[ci], ha="left", va="center", fontsize=8.9, zorder=4,
                color="white" if w_ else INK, fontweight="bold" if ci == _bi else "normal")
        ax.text(95.6, yy, f"{int(round(v*NTREE))}/{NTREE}", ha="right", va="center",
                fontsize=8.9, zorder=4, color="white" if w_ else INK,
                fontweight="bold" if ci == _bi else "normal")
    flow(99.6, 103.4)

    stage(103.5, W, "4", "What you act on", "reported for all twelve")
    _pv = binom.sf(int(round(_bv[_bi]*NTREE)) - 1, NTREE, 0.5)
    rows = [("substrate", cls[_bi], True), ("probability", f"{_bp[_bi]:.2f}", True),
            ("$p$-value", f"{_pv:.0e}", False), ("driven by", _ex_genes, False)]
    for k, (lab, val, big) in enumerate(rows):
        yy = 21.9 - k*4.3
        ax.text(105.6, yy, lab, ha="left", va="center", fontsize=8.9, color=MUTED, zorder=3)
        ax.text(130.6, yy, val, ha="right", va="center", zorder=3,
                fontsize=10.2 if big else 9.0, color=ACC if big else INK, fontweight="bold")
        if k < len(rows) - 1:
            ax.plot([105.6, 130.6], [yy - 2.3, yy - 2.3], color=RULE, lw=0.7, zorder=2)

    ax.text(1.5, 43.4, "How subFinder reads one locus", ha="left", va="bottom",
            fontsize=15.5, fontweight="bold", color=INK)
    ax.text(1.5, 39.9, "every value shown is what the released model returns for this locus",
            ha="left", va="bottom", fontsize=10.5, color=MUTED)
    plt.savefig(FIG/"fig0_workflow.png", bbox_inches="tight"); plt.close()

    # ============================================================ FIG 2
    cm = confusion_matrix(y_all, y_pred, labels=subs).astype(float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(subs)):
        for j in range(len(subs)):
            v = cmn[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=9.5, color="white" if v > 0.55 else INK,
                        fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(subs))); ax.set_yticks(range(len(subs)))
    ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=10.5)
    ax.set_yticklabels(subs, fontsize=10.5)
    ax.set_xlabel("Predicted substrate", fontsize=12)
    ax.set_ylabel("True substrate", fontsize=12)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.03)
    cb.set_label("Share of the true substrate's loci", fontsize=11); cb.outline.set_visible(False)
    titled(ax, "Where the predictions go",
           "rows sum to 1.00; the diagonal is that substrate's accuracy")
    plt.savefig(FIG/"fig2_confusion.png"); plt.close()

    p, r, f1, sup = precision_recall_fscore_support(y_all, y_pred, labels=subs,
                                                    average=None, zero_division=0)
    # `sup` counts POOLED predictions, so it is REPS times the number of loci.
    # The table must show loci, or a reader comparing it with the class sizes in
    # the Data section sees 1200 against 240 for the same substrate.
    n_loci = (sup // len(REPS)).astype(int)
    assert (sup % len(REPS) == 0).all(), "each locus should appear once per repeat"
    assert n_loci.sum() == len(X), "per-substrate loci must sum to the corpus"
    per = pd.DataFrame({"substrate": subs, "loci": n_loci, "precision": p, "recall": r,
                        "f1": f1}).sort_values("f1", ascending=False)
    per.to_csv(TAB/"table_per_substrate.csv", index=False)
    body = "\n".join(f"{q.substrate} & {int(q.loci)} & {q.precision:.2f} & {q.recall:.2f} & {q.f1:.2f} \\\\"
                     for _, q in per.iterrows())
    (GEN/"table_persubstrate.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\nSubstrate & Loci & Precision & Recall & F1 \\\\\n"
        f"\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n")
    worst = per.iloc[-1]; best = per.iloc[0]
    macro("WorstSub", worst.substrate); macro("WorstF", f"{worst.f1:.2f}")
    second = per.iloc[-2]
    macro("SecondWorstSub", second.substrate)
    macro("SecondWorstF", f"{second.f1:.2f}")
    macro("SecondWorstPrec", f"{second.precision:.2f}")
    macro("SecondWorstRec", f"{second.recall:.2f}")
    macro("BestRec", f"{best.recall:.2f}")
    # the substrate the worst class most often loses loci to
    _cm = confusion_matrix(y_all, y_pred, labels=subs).astype(float)
    _wi = subs.index(worst.substrate)
    _off = [(_cm[_wi][j], subs[j]) for j in range(len(subs)) if j != _wi]
    macro("WorstConfusedWith", max(_off)[1])
    macro("WorstRec", f"{worst.recall:.2f}"); macro("WorstPrec", f"{worst.precision:.2f}")
    macro("BestSub", best.substrate)
    ch = per[per.substrate == "chitin"].iloc[0]; macro("ChitinF", f"{ch.f1:.2f}")

    # ============================================================ FIG 3 (deck-style table)
    canon = build_canon(ROOT/"data/Literature_Data_fam_substrate_mapping.tsv", SUBSTRATE_ALIAS)
    abl = pd.read_csv(ROOT/f"artifacts/ablation/sig_gene_ablation_oof_outer{S}_v2.csv")
    abl["top3"] = abl["top3"].fillna("").astype(str)
    CAZY = ("GH", "PL", "CE", "CBM", "GT", "AA")

    # "null" marks a gene the annotation pipeline could not label. It is a real
    # feature to the model -- an unannotated neighbour is weak evidence in itself --
    # but it names nothing, so it is excluded from a podium meant to be read
    # biologically. Bare digits are subfamily fragments left by splitting on "_"
    # and are likewise uninformative on their own.
    rows = []
    for s in subs:
        cnt = {}
        for t3 in abl[abl.true == s].top3:
            for tok in t3.split(";"):
                if tok in SKIP or tok.isdigit(): continue
                cnt[tok] = cnt.get(tok, 0) + 1
        for rank, (tok, n) in enumerate(sorted(cnt.items(), key=lambda kv: -kv[1])[:3], 1):
            if tok in canon[s]:                     status = "listed"
            elif tok.startswith(CAZY):              status = "cazy-not-listed"
            else:                                   status = "non-cazy"
            rows.append(dict(substrate=s, rank=rank, token=tok, n=n, status=status))
    sg = pd.DataFrame(rows); sg.to_csv(TAB/"table_top3_siggenes.csv", index=False)

    FILL = {"listed": "#d6ebe9", "cazy-not-listed": "#f7e2e2", "non-cazy": "#f0f2f4"}
    fig, ax = plt.subplots(figsize=(8.6, 3.55)); ax.axis("off")
    cell, colr = [], []
    for s in subs:
        blk = sg[sg.substrate == s]
        row, rowc = [s], ["white"]
        for k in range(1, 4):
            m = blk[blk["rank"] == k]
            if len(m):
                row.append(m.iloc[0].token); rowc.append(FILL[m.iloc[0].status])
            else:
                row.append(""); rowc.append("white")
        cell.append(row); colr.append(rowc)
    tb = ax.table(cellText=cell, colLabels=["Substrate", "1st", "2nd", "3rd"],
                  loc="upper left", cellLoc="left", colWidths=[0.30, 0.235, 0.235, 0.235])
    tb.auto_set_font_size(False); tb.set_fontsize(10.5); tb.scale(1, 1.30)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("white"); c.set_linewidth(1.4)
        if i == 0:
            c.set_facecolor(INK); c.set_text_props(color="white", weight="bold")
        else:
            c.set_facecolor(colr[i-1][j])
            c.set_text_props(color=INK, weight="bold" if j == 0 else "normal")
    titled(ax, "The three genes the model leans on most, per substrate",
           "colour shows whether the curated enzyme table lists that family for that substrate",
           tfs=13, sfs=10.5)
    ax.legend(handles=[Patch(facecolor=FILL["listed"], label="listed for this substrate"),
                       Patch(facecolor=FILL["cazy-not-listed"], label="a CAZy family, but not listed here"),
                       Patch(facecolor=FILL["non-cazy"], label="not a CAZy family: transporter, regulator, or unannotated")],
              loc="upper left", bbox_to_anchor=(-0.005, -0.015), ncol=3,
              handlelength=1.3, columnspacing=1.4, fontsize=9.5)
    plt.savefig(FIG/"fig3_siggenes.png"); plt.close()

    # ============================================================ FIG 4 (two-panel, eligible-scoped)
    fr = []
    for s in subs:
        blk = abl[abl.true == s]
        elig = [rr for _, rr in blk.iterrows() if set(tok_cpu_v2(X[rr.idx])) & canon[s]]
        hit = sum(1 for rr in elig if set(rr.top3.split(";")) & canon[s])
        scope, flag = set(), set()
        for rr in elig:
            scope |= set(tok_cpu_v2(X[rr.idx])) & canon[s]
            flag  |= set(rr.top3.split(";")) & canon[s]
        fr.append(dict(substrate=s, eligible=len(elig), hit=hit,
                       rate=hit/max(len(elig), 1), in_scope=len(scope),
                       flagged=len(scope & flag),
                       srate=len(scope & flag)/max(len(scope), 1)))
    fu = pd.DataFrame(fr).sort_values("rate"); fu.to_csv(TAB/"table_sig_funnel.csv", index=False)
    TE, TH = int(fu.eligible.sum()), int(fu.hit.sum())
    TS, TF = int(fu.in_scope.sum()), int(fu.flagged.sum())
    macro("SigElig", TE); macro("SigHit", TH); macro("SigRate", f"{TH/TE*100:.1f}")
    macro("SigScope", TS); macro("SigFlag", TF); macro("SigScopeRate", f"{TF/TS*100:.1f}")
    macro("SigSkipped", len(X)-TE)
    # The prose used to call chitin the least-agreeing substrate. It is not, and a
    # superlative typed by hand goes stale the moment the curated table changes, so
    # the extremes are emitted here and quoted rather than asserted.
    _lo = fu.iloc[0]                       # fu is sorted ascending by by-locus rate
    macro("FunnelLowSub", str(_lo.substrate))
    macro("FunnelLowRate", f"{_lo.rate*100:.0f}")
    _ch = fu[fu.substrate == "chitin"].iloc[0]
    macro("ChitinFunnelRate", f"{_ch.rate*100:.0f}")
    macro("ChitinFunnelRank", str(int((fu.rate < _ch.rate).sum()) + 1))
    _lo_fam = fu.sort_values("srate").iloc[0]
    macro("FunnelLowFamSub", str(_lo_fam.substrate))
    macro("FunnelLowFamRate", f"{_lo_fam.srate*100:.0f}")
    macro("ChitinFamRate", f"{_ch.srate*100:.0f}")

    fig, (aL, aR) = plt.subplots(1, 2, figsize=(11.0, 4.9))
    yy = np.arange(len(fu))
    aL.barh(yy, fu.eligible, height=.66, color="#e4eaef", label="loci containing a listed family")
    aL.barh(yy, fu.hit, height=.66, color=TEAL, label="…the model used one in its top 3")
    for i, (_, q) in enumerate(fu.iterrows()):
        aL.text(q.eligible + fu.eligible.max()*0.02, i, f"{int(q.hit)}/{int(q.eligible)}",
                va="center", fontsize=10, fontweight="bold", color=INK)
    aL.set_yticks(yy); aL.set_yticklabels(fu.substrate, fontsize=11)
    aL.set_xlabel("Loci", fontsize=12); aL.set_xlim(0, fu.eligible.max()*1.24)
    aL.xaxis.grid(True, color=RULE, lw=0.7); aL.set_axisbelow(True)
    titled(aL, "By locus", f"{TH} of {TE} = {TH/TE*100:.1f}%", tfs=13, sfs=11)
    aL.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)

    aR.barh(yy, fu.in_scope, height=.66, color="#e4eaef", label="listed families present in these loci")
    aR.barh(yy, fu.flagged, height=.66, color=AMBER, label="…surfaced by the model somewhere")
    for i, (_, q) in enumerate(fu.iterrows()):
        aR.text(q.in_scope + fu.in_scope.max()*0.02, i, f"{int(q.flagged)}/{int(q.in_scope)}",
                va="center", fontsize=10, fontweight="bold", color=INK)
    aR.set_yticks(yy); aR.set_yticklabels(fu.substrate, fontsize=11)
    aR.set_xlabel("Enzyme families", fontsize=12); aR.set_xlim(0, fu.in_scope.max()*1.30)
    aR.xaxis.grid(True, color=RULE, lw=0.7); aR.set_axisbelow(True)
    titled(aR, "By enzyme family", f"{TF} of {TS} = {TF/TS*100:.1f}%", tfs=13, sfs=11)
    aR.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig(FIG/"fig4_funnel.png"); plt.close()

    # ============================================================ FIG 5 (worked examples)
    # `null` marks an unannotated neighbour. The classifier uses it -- an
    # unannotated gene is weak evidence in itself -- but it names nothing, so it
    # is suppressed from every DISPLAYED gene list, exactly as it is from the
    # signature-gene figure. Drawing from top5 keeps three real genes where the
    # locus has three; a locus with fewer simply shows fewer.
    def _display_genes(row, k=3):
        pool = str(row.top5 if isinstance(row.top5, str) and row.top5 else row.top3)
        out = []
        for tok in pool.split(";"):
            tok = tok.strip()
            if tok and tok not in SKIP and not tok.isdigit() and tok not in out:
                out.append(tok)
            if len(out) == k: break
        return ";".join(out)
    top3 = {int(rr.idx): _display_genes(rr) for _, rr in abl.iterrows()}
    recs = []
    for i in range(len(X)):
        if i not in top3: continue
        o = np.argsort(-Pc_one[i]); tr = int(list(o).index(cls.index(y[i])))+1
        recs.append(dict(idx=i, true=y[i], pred=cls[int(o[0])], conf=float(Pc_one[i][o[0]]),
                         second=cls[int(o[1])], p2=float(Pc_one[i][o[1]]), rank=tr,
                         sig=top3[i], margin=float(Pc_one[i][o[0]]-Pc_one[i][o[1]])))
    C = pd.DataFrame(recs)
    # How many of the genes SHOWN are documented for the true substrate, and for
    # the predicted one. Every panel's caption is a claim about these, so the
    # selection below requires them rather than hoping for them.
    C["n_true_listed"] = [len(set(str(q.sig).split(";")) & canon[q.true]) for _, q in C.iterrows()]
    C["n_pred_listed"] = [len(set(str(q.sig).split(";")) & canon[q.pred]) for _, q in C.iterrows()]
    C["n_shown"] = [len([g for g in str(q.sig).split(";") if g]) for _, q in C.iterrows()]
    # a wrong call is "recoverable" if the truth is 2nd and the top two are close,
    # or if a listed enzyme for the TRUE substrate sits in the signature genes
    C["rescuable"] = [(q["rank"] == 2 and q.margin < 0.15) or
                      bool(set(str(q.sig).split(";")) & canon[q.true])
                      for _, q in C.iterrows()]
    used, seen, picks = set(), set(), []
    def take(mask, label):
        sel = C[mask & ~C.idx.isin(used)]
        if not len(sel): return
        fresh = sel[~sel.true.isin(seen)]
        q = (fresh if len(fresh) else sel).iloc[0]
        used.add(q.idx); seen.add(q.true); picks.append((q, label))
    # Two panels showing the tool working, three showing that when it is wrong the
    # output still carries the answer, and one honest limitation. Each condition
    # requires the evidence its caption claims.
    ok, wrong = C.true == C.pred, C.true != C.pred
    take(ok & (C.conf > .90) & (C.n_true_listed >= 2),
         "Correct, with reasons")
    take(ok & C.conf.between(.40, .75) & (C.n_true_listed >= 1),
         "Correct, and hedged")
    take(wrong & (C["rank"] == 2) & (C.margin < .15),
         "Wrong; truth is second")
    take(wrong & (C.n_true_listed >= 1) & (C["rank"] <= 3),
         "Wrong; genes name it")
    take(wrong & (C["rank"] == 3),
         "Wrong; truth is third")
    take(ok & (C.n_true_listed == 0) & (C.n_shown > 0),
         "Correct; table is silent")
    # fall back only if a category is genuinely empty, and say so
    while len(picks) < 6:
        before = len(picks); take(pd.Series(True, index=C.index), "Further example")
        if len(picks) == before: break

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.5))
    fig.subplots_adjust(hspace=1.12, wspace=0.20, top=0.775, bottom=0.08)
    for ax, (q, label) in zip(axes.ravel(), picks):
        right = (q.true == q.pred)
        order = np.argsort(-Pc_one[q.idx])[:3]
        for k, t in enumerate(order):
            v = float(Pc_one[q.idx][t]); is_true = cls[t] == q.true
            ax.barh(2 - k, v, color=TEAL if is_true else ROSE, height=.62,
                    linewidth=0, zorder=3)
            ins = v >= 0.58
            ax.text(v - 0.02 if ins else v + 0.02, 2 - k,
                    f"{cls[t]}  {v:.2f}", va="center",
                    ha="right" if ins else "left", fontsize=10.5,
                    color="white" if ins else INK,
                    fontweight="bold" if is_true else "normal", zorder=4)
        ax.set_xlim(0, 1.02); ax.set_ylim(-0.65, 2.65); ax.set_yticks([])
        ax.set_xticks([0, .5, 1]); ax.tick_params(labelsize=9.5, length=0)
        ax.grid(True, axis="x", color=RULE, lw=0.6, zorder=0); ax.set_axisbelow(True)

        # heading: the outcome, colour-coded, above the scenario
        ax.annotate("CORRECT" if right else "WRONG",
                    xy=(0, 1.42), xycoords="axes fraction",
                    fontsize=9, fontweight="bold", color=TEAL if right else ROSE,
                    ha="left", va="bottom")
        ax.annotate(label, xy=(0, 1.24), xycoords="axes fraction",
                    fontsize=11, fontweight="bold", color=INK, ha="left", va="bottom")
        # a filled marker means the curated table lists that gene for the TRUE
        # substrate, so a reader can see the recovery evidence at a glance
        genes = [g for g in str(q.sig).split(";") if g]
        parts = ["\u25cf " + g if g in canon[q.true] else "\u25cb " + g for g in genes]
        ax.annotate(f"true substrate: {q.true}", xy=(0, 1.06),
                    xycoords="axes fraction", fontsize=9.5, color=MUTED,
                    ha="left", va="bottom")
        ax.annotate("genes: " + "   ".join(parts) if parts else "genes: none",
                    xy=(0, -0.30), xycoords="axes fraction", fontsize=9.5,
                    color=MUTED, ha="left", va="top")
        for sp in ax.spines.values(): sp.set_visible(False)

    fig.suptitle("Six held-out predictions", x=0.005, ha="left", fontsize=15,
                 fontweight="bold", color=INK, y=0.995)
    fig.text(0.005, 0.948,
             "the three highest probabilities; teal is the true substrate. "
             "\u25cf a gene the curated table lists for that substrate, \u25cb one it does not",
             fontsize=11, color=MUTED, ha="left")
    plt.savefig(FIG/"fig5_cases.png", bbox_inches="tight"); plt.close()
    for _q, _l in picks:
        if _l.startswith("Wrong; truth is second"):
            macro("CaseSecondC", f"{_q.p2:.2f}")
    cs = pd.DataFrame([dict(scenario=l, idx=int(q.idx), true=q.true, pred=q.pred,
                            confidence=round(float(q.conf), 4), second=q.second,
                            p_second=round(float(q.p2), 4), true_rank=int(q["rank"]),
                            signature_genes=q.sig) for q, l in picks])
    cs.to_csv(TAB/"table_case_studies.csv", index=False)
    for k, (q, _) in enumerate(picks, 1):
        macro(f"CaseTrue{'ABCDEF'[k-1]}", q.true)
        macro(f"CasePred{'ABCDEF'[k-1]}", q.pred)
        macro(f"CaseConf{'ABCDEF'[k-1]}", f"{q.conf:.2f}")
        macro(f"CaseSig{'ABCDEF'[k-1]}", str(q.sig).replace(";", ", "))
        # Where the true substrate actually landed. The figure caption describes each
        # panel as a hit or a miss, and without a generated rank that description is
        # free to drift away from the panel it is describing.
        macro(f"CaseRank{'ABCDEF'[k-1]}",
              {1: "first", 2: "second", 3: "third"}.get(int(q["rank"]), str(int(q["rank"]))))

    # ============================================================ leaderboard + macros
    def row(cfg): 
        z = lb[lb.shorthand == cfg].iloc[0]; return float(z.mean_acc), float(z.std_acc)
    dep_a, dep_s = row(DEPLOYED); macro("DepAcc", f"{dep_a:.4f}"); macro("DepStd", f"{dep_s:.4f}")
    brf_a, brf_s = row("cv__BRF100"); macro("BrfAcc", f"{brf_a:.4f}"); macro("BrfStd", f"{brf_s:.4f}")
    # the counts-with-whole-transporter-identifiers variant, quoted in the text
    # as the before-and-after of swapping the classifier
    cpu_a, _ = row("cpu__ET500_log2"); macro("CpuAcc", f"{cpu_a:.4f}")
    macro("GapTok", f"{(dep_a-cpu_a)*100:.2f}")
    dl = lb[lb.shorthand.str.contains("__LSTM|__Trans|__JustAttn")].iloc[0]
    macro("DlAcc", f"{dl.mean_acc:.4f}"); macro("DlStd", f"{dl.std_acc:.4f}")
    macro("GapDl", f"{(dep_a-dl.mean_acc)*100:.2f}"); macro("GapBrf", f"{(dep_a-brf_a)*100:.2f}")
    macro("DlWorst", f"{lb[lb.shorthand.str.contains('__LSTM|__Trans|__JustAttn')].mean_acc.min():.4f}")

    import joblib as _j
    vocab = len(_j.load(ROOT/"artifacts/final_model_v2.pkl")["pipeline"].named_steps["cv"].vocabulary_)
    macro("Vocab", vocab)
    macro("TokPerPul", f"{np.mean([len(tok_cpu_v2(x)) for x in X]):.1f}")

    # The worked example in the main text is generated, not transcribed. It was
    # previously typed by hand and had drifted: it showed the bare subfamily
    # indices that the tokenizer discards, and the wrong token count.
    ex_toks = tok_cpu_v2(EXAMPLE_PUL)
    macro("ExampleRaw", EXAMPLE_PUL.replace("_", r"\_").replace("|", r"$|$"))
    macro("ExampleTokens", ", ".join(r"\texttt{" + t.replace("_", r"\_") + "}"
                                     for t in ex_toks))
    macro("ExampleNTokens", len(ex_toks))
    macro("ExampleNGenes", EXAMPLE_PUL.count(",") + 1)

    # Evidence for discarding the residual subfamily index: it is not unique to a
    # parent family, so retaining it would merge unrelated enzymes into one feature.
    import collections as _c
    _idx = _c.defaultdict(set)
    for _s in X:
        for _f in re.split(r"[,|]", str(_s)):
            _h, _u, _t = _f.partition("_")
            if _u and _t.isdigit(): _idx[_t].add(_h)
    macro("SubIdxTotal", len(_idx))
    macro("SubIdxShared", sum(1 for v in _idx.values() if len(v) > 1))
    macro("SubIdxWorst", len(_idx.get("1", ())))
    macro("CanonPairs", sum(len(v) for v in canon.values()))

    lead = lb.head(7).copy()
    # Human-readable names, derived rather than looked up, so a configuration
    # entering the top of the leaderboard can never appear as a raw shorthand.
    FEAT = {"cpuV2": ("Counts, transporters at family level", "counts"),
            "cpu": ("Counts, whole transporter identifiers", "counts"),
            "cv": ("Counts, commas and pipes only", "counts"),
            "ftCbow_MM": ("FastText CBOW mean$+$max", "embedding"),
            "ftCbow": ("FastText CBOW", "embedding"),
            "ftSg": ("FastText skip-gram", "embedding"),
            "w2vCbow": ("Word2Vec CBOW", "embedding"),
            "w2vSg": ("Word2Vec skip-gram", "embedding"),
            "d2vDm": ("Doc2Vec DM", "document vector"),
            "d2vDbow": ("Doc2Vec DBOW", "document vector")}
    CLF = {"ET500_log2": "ExtraTrees", "ET500_sqrt": "ExtraTrees",
           "BRF100": "Balanced Random Forest", "LSTM": "LSTM",
           "LSTMattn": "LSTM with attention", "JustAttn": "attention block",
           "Trans": "transformer"}
    body = ""
    for _, q in lead.iterrows():
        f_key, c_key = q.shorthand.split("__")
        fname, rep = FEAT.get(f_key, (f_key.replace("_", r"\_"), ""))
        cname = CLF.get(c_key, c_key.replace("_", r"\_"))
        nm = f"{fname} $+$ {cname}"
        if q.shorthand == DEPLOYED: nm = f"\\textbf{{{fname}}} $+$ \\textbf{{{cname}}}"
        val = f"{q.mean_acc:.4f}\\pm{q.std_acc:.4f}"
        if q.shorthand == DEPLOYED: val = f"\\mathbf{{{val}}}"
        body += f"{nm} & {rep} & ${val}$ \\\\\n"
    (GEN/"table_leaderboard.tex").write_text(
        "\\begin{tabular}{llr}\n\\toprule\nConfiguration & Representation & Accuracy \\\\\n"
        f"\\midrule\n{body}\\bottomrule\n\\end{{tabular}}\n")

    cal = pd.read_csv(ROOT/"artifacts/calibration_report.csv")
    LBL = {"uncalibrated": "Uncalibrated",
           "temperature_scaling": r"\textbf{Temperature scaling}",
           "isotonic_cv5 (sklearn)": r"Isotonic regression~\citep{zadrozny2002transforming}",
           "sigmoid_cv5 (sklearn)": r"Platt (sigmoid) scaling~\citep{platt1999probabilistic}"}
    cb = ""
    for _, q in cal.iterrows():
        a, e = f"{q.accuracy:.4f}", f"{q.ece_10bin:.4f}"
        if q.method == "temperature_scaling": a, e = f"\\textbf{{{a}}}", f"\\textbf{{{e}}}"
        cb += f"{LBL.get(q.method, q.method)} & {a} & {e} \\\\\n"
    # (no calibration table: the manuscript reports these three numbers in prose)
    u = cal[cal.method == "uncalibrated"].iloc[0]
    t = cal[cal.method == "temperature_scaling"].iloc[0]
    iso = cal[cal.method.str.startswith("isotonic")].iloc[0]
    macro("EceRaw", f"{u.ece_10bin:.3f}"); macro("EceTemp", f"{t.ece_10bin:.3f}")
    macro("EceIso", f"{iso.ece_10bin:.3f}"); macro("IsoAcc", f"{iso.accuracy:.4f}")
    macro("MeanT", f"{float(t['T']):.2f}")
    # claims the prose makes about the comparison, derived rather than typed
    sig_row = cal[cal.method.str.startswith("sigmoid")].iloc[0]
    macro("EceSigmoid", f"{sig_row.ece_10bin:.3f}")
    macro("EceReduction", f"{u.ece_10bin/t.ece_10bin:.1f}")
    macro("IsoAccCost", f"{(t.accuracy-iso.accuracy)*100:.1f}")

    # ======================================================== SUPPLEMENTARY
    from sklearn.feature_extraction.text import CountVectorizer

    # S1 reliability diagram --------------------------------------------------
    conf_raw = P.max(1); conf_cal = Pc.max(1); correct = (y_pred == y_all)
    # direction and size of the miscalibration, weighted by band population
    gap, ntot = 0.0, 0
    for lo in np.arange(0, 1, 0.1):
        m = (conf_raw >= lo) & (conf_raw < lo + 0.1)
        if m.sum() >= 5:
            gap += (correct[m].mean() - conf_raw[m].mean()) * m.sum(); ntot += int(m.sum())
    macro("ConfGap", f"{gap/ntot:+.2f}".replace("+", ""))
    macro("ConfDir", "under" if gap > 0 else "over")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for ax, cf, name in ((a1, conf_raw, "Uncalibrated"), (a2, conf_cal, "Temperature scaled")):
        edges = np.linspace(0, 1, 11); xs, ys, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (cf >= lo) & (cf < hi if hi < 1 else cf <= hi)
            if m.sum() >= 5:
                xs.append(cf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
        ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), color=SLATE, lw=1.2, label="perfect calibration")
        ax.plot(xs, ys, "-o", color=TEAL, lw=2.2, ms=7, label="observed")
        for x_, y_, n_ in zip(xs, ys, ns):
            ax.annotate(f"n={n_}", (x_, y_), textcoords="offset points", xytext=(0, -14),
                        fontsize=8, color=MUTED, ha="center")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.set_xlabel("Stated confidence"); ax.set_ylabel("Observed accuracy")
        ax.grid(True, color=RULE, lw=0.6); ax.set_axisbelow(True)
        titled(ax, name, tfs=12.5)
        ax.legend(loc="upper left")
    plt.tight_layout(); plt.savefig(FIG/"figS1_reliability.png"); plt.close()

    # S2 top-K cumulative accuracy -------------------------------------------
    ordr = np.argsort(-Pc, axis=1)
    topk = [float(np.mean([y_all[i] in [cls[j] for j in ordr[i][:k]] for i in range(len(y_all))]))
            for k in (1, 2, 3, 5)]
    # Both the mask and the value must come from the pooled space. Selecting with
    # `y[i]` (by-locus, CSV order) while scoring `y_all[i]`/`ordr[i]` (pooled, fold
    # order) reads a different locus on each side, which is why K=1 here disagreed
    # with the per-substrate recall in the main table. K=1 is that recall.
    _hit = np.array([[y_all[i] in [cls[j] for j in ordr[i][:k]] for k in (1, 2, 3, 5)]
                     for i in range(len(y_all))])
    _ya = np.asarray(y_all)
    per_sub_k = {s: _hit[_ya == s].mean(axis=0).tolist() for s in subs}
    # percentage forms, for the abstract, where a decimal reads oddly in prose
    macro("TopTwoPct", f"{topk[1]*100:.1f}"); macro("TopThreePct", f"{topk[2]*100:.1f}")
    macro("TopOne", f"{topk[0]:.3f}"); macro("TopTwo", f"{topk[1]:.3f}")
    macro("TopThree", f"{topk[2]:.3f}")
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ks = [1, 2, 3, 5]
    for s in subs:
        ax.plot(ks, per_sub_k[s], "-o", color=SLATE, alpha=.35, lw=1.2, ms=4)
    ax.plot(ks, topk, "-o", color=TEAL, lw=3, ms=9, label="all substrates")
    for k, v in zip(ks, topk):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points", xytext=(0, 11),
                    fontsize=11, fontweight="bold", color=INK, ha="center")
    ax.set_xticks(ks); ax.set_xlabel("K (substrates considered)"); ax.set_ylabel("Cumulative accuracy")
    ax.set_ylim(min(min(v) for v in per_sub_k.values())-0.04, 1.02)
    ax.grid(True, axis="y", color=RULE, lw=0.6); ax.set_axisbelow(True)
    titled(ax, "How often the truth is within the top K",
           "grey lines are individual substrates; reporting more than the winner recovers most misses")
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(FIG/"figS2_topk.png"); plt.close()

    # S3 confidence vs correctness -------------------------------------------
    # winner significance uses the Bonferroni-corrected threshold: the winner is
    # the max of 12 components, so the marginal null is anti-conservative.
    from scipy.optimize import brentq
    # significance is decided on the VOTE COUNTS, Bonferroni-corrected over the
    # 12 substrates -- not on the normalised probability, which manufactures a
    # confident winner out of twelve weak votes.
    ALPHA = 0.05
    win_pv = PV[np.arange(len(y_all)), Vv.argmax(1)]
    sig_vote = win_pv < ALPHA/len(subs)
    macro("SigThresh", f"{ALPHA/len(subs):.5f}")

    # Evidence guard. The p-value asks whether the 12 probabilities are more peaked
    # than a random split of 1; it cannot ask whether anything was read to produce
    # the peak. The two rules are largely, but not entirely, disjoint -- the overlap
    # is emitted below rather than assumed, because the supplement quotes it.
    MIN_INFORM = 2
    low_ev = n_inform < MIN_INFORM
    low_pr = ~sig_vote
    withheld = low_ev | low_pr
    macro("MinInform", f"{MIN_INFORM}")
    macro("LowEvidenceN", f"{int(low_ev.sum())}")
    macro("LowEvidencePct", f"{100*low_ev.mean():.1f}")
    macro("LowEvidenceAcc", f"{correct[low_ev].mean()*100:.0f}")
    macro("WithheldN", f"{int(withheld.sum())}")
    macro("WithheldPct", f"{100*withheld.mean():.1f}")
    macro("WithheldAcc", f"{correct[withheld].mean()*100:.0f}")
    macro("ReportedN", f"{int((~withheld).sum())}")
    macro("ReportedPct", f"{100*(~withheld).mean():.1f}")
    macro("ReportedAcc", f"{correct[~withheld].mean():.4f}")
    macro("WithheldBothN", f"{int((low_ev & low_pr).sum())}")
    macro("LowPrN", f"{int(low_pr.sum())}")

    # The "dividing by the total manufactures a leader" argument quotes a number for
    # a locus with nothing in it. Probe the deployed model rather than asserting one:
    # the normalised leader is well above chance (1/12) but nowhere near 0.5, and the
    # paper said 0.5 for long enough that it needs to be generated from here on.
    _null_seq = ",".join(["null"] * 9)
    _null_raw = pipe_final.predict_proba([_null_seq])[0]
    _null_votes = np.column_stack([e.predict_proba(
        pipe_final.named_steps["cv"].transform([_null_seq]))[:, 1]
        for e in pipe_final.named_steps["vr"].estimators_])[0]
    macro("NullLeader", f"{_null_raw.max():.2f}")
    macro("NullVotes", f"{int(round(_null_votes.max()*NTREES_DEPLOYED))}")
    macro("Chance", f"{1/len(subs):.2f}")

    bins = [(0, .4), (.4, .6), (.6, .8), (.8, .95), (.95, 1.01)]
    lab = ["below 0.40", "0.40-0.60", "0.60-0.80", "0.80-0.95", "0.95-1.00"]
    accs, cnts = [], []
    for lo, hi in bins:
        m = (conf_cal >= lo) & (conf_cal < hi)
        accs.append(float(correct[m].mean()) if m.sum() else 0.0); cnts.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    bars = ax.bar(range(len(bins)), accs, color=[ROSE] + [TEAL]*4, width=.62)
    for i, (a_, n_) in enumerate(zip(accs, cnts)):
        ax.annotate(f"{a_*100:.0f}%", (i, a_), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=12, fontweight="bold", color=INK)
        ax.annotate(f"{100*n_/len(y_all):.0f}%", (i, 0.02), ha="center", fontsize=10, color="white")
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(lab, fontsize=10)
    ax.set_ylabel("Fraction correct"); ax.set_ylim(0, 1.12)
    ax.grid(True, axis="y", color=RULE, lw=0.6); ax.set_axisbelow(True)
    titled(ax, "Confidence tracks correctness",
           "held-out predictions grouped by the winning calibrated probability")
    plt.tight_layout(); plt.savefig(FIG/"figS3_confidence.png"); plt.close()
    macro("HighConfAcc", f"{accs[-1]*100:.1f}"); macro("HighConfN", cnts[-1])
    macro("HighConfPct", f"{100*cnts[-1]/len(y_all):.1f}")
    # LaTeX control sequences are letters only, so the bands are named, not numbered
    for _i, _nm in enumerate(("Lowest", "Low", "Mid", "High", "Highest")):
        macro(f"BandAcc{_nm}", f"{accs[_i]*100:.0f}")
        macro(f"BandN{_nm}", f"{cnts[_i]}")
        macro(f"BandPct{_nm}", f"{100*cnts[_i]/len(y_all):.1f}")
    macro("BelowSigN", f"{int(low_pr.sum())}")
    macro("BelowSigPct", f"{100*low_pr.mean():.1f}")
    macro("BelowSigAcc", f"{correct[low_pr].mean()*100:.0f}")
    macro("AboveSigAcc", f"{correct[sig_vote].mean()*100:.1f}")
    macro("SigRateHeldOut", f"{100*sig_vote.mean():.1f}")

    # how the vote p-values order the substrates -- the behaviour the supplement claims
    _srt = np.sort(PV, axis=1)
    def _sci(v):
        e = int(np.floor(np.log10(v))) if v > 0 else 0
        return f"{v/10**e:.0f}\\times10^{{{e}}}"
    macro("PvWinnerMed", _sci(float(np.median(_srt[:, 0]))))
    macro("PvRunnerMed", f"{np.median(_srt[:, 1]):.3f}")
    macro("PvWinnerSmallest", f"{100*np.mean(_srt[:, 0] < _srt[:, 1]):.1f}")
    macro("PvCorrectMed", _sci(float(np.median(_srt[correct, 0]))))
    macro("PvWrongMed", f"{np.median(_srt[~correct, 0]):.2f}")
    _w = _srt[:, 0]
    macro("PvBandLowAcc",  f"{correct[_w < 1e-30].mean()*100:.1f}")
    macro("PvBandHighAcc", f"{correct[_w >= 0.05].mean()*100:.1f}")
    # the served probabilities are normalised; the p-values read the raw votes.
    # normalisation and the temperature step are both monotone and applied to all
    # twelve alike, so neither can reorder -- meaning the p-value always describes
    # the substrate the probabilities rank first. Measured, not assumed.
    macro("RankAgree", f"{100*np.mean(Vv.argmax(1) == Pc.argmax(1)):.1f}")

    # S3b evidence guard: what the model claims vs what it delivers, by how many
    # tokens it could actually read. Pooled over all 25 RSKF runs, because the
    # one-token group is small in any single split (15 loci) and the threshold
    # decision rests on it.
    EV, CR, PM = [], [], []
    for seed in (42, 43, 44, 45, 46):
        sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(sk.split(X, y)):
            z = np.load(ROOT/f"artifacts/predictions/{DEPLOYED}/r{seed}_f{fold}/probs_test.npz",
                        allow_pickle=True)
            c_ = [str(c) for c in z["classes"]]
            pc = apply_temperature(z["probs"], T)
            fv = set(t for i in tr for t in tok_cpu_v2(X[i]))
            EV.append(np.array([sum(1 for t in tok_cpu_v2(X[i]) if t != "null" and t in fv)
                                for i in te]))
            CR.append(np.array(c_)[pc.argmax(1)] == y[te]); PM.append(pc.max(1))
    ev_p = np.concatenate(EV); cr_p = np.concatenate(CR); pm_p = np.concatenate(PM)
    macro("PooledN", f"{len(ev_p):,}".replace(",", "{,}"))

    groups = [(1, "1"), (2, "2"), (3, "3"), (4, "4"), (5, "5"), (6, "$\\geq$6")]
    g_acc, g_conf, g_n = [], [], []
    for v, _ in groups:
        m = (ev_p == v) if v < 6 else (ev_p >= 6)
        g_acc.append(float(cr_p[m].mean())); g_conf.append(float(pm_p[m].mean()))
        g_n.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    xs = np.arange(len(groups))
    ax.bar(xs-0.19, g_conf, width=0.36, color=SLATE)
    ax.bar(xs+0.19, g_acc, width=0.36, color=[ROSE]+[TEAL]*(len(groups)-1))
    ax.annotate("", xy=(0.19, g_acc[0]), xytext=(0.19, g_conf[0]),
                arrowprops=dict(arrowstyle="<->", color=ROSE, lw=1.6))
    # sits in the headroom above the first pair, not across the neighbouring bars
    # short label beside the arrow; the caption carries the sentence
    ax.annotate(f"{(g_conf[0]-g_acc[0])*100:.0f} pts",
                (0.28, (g_acc[0]+g_conf[0])/2), ha="left", va="center",
                fontsize=10.5, color=ROSE, fontweight="bold")

    for i, n_ in enumerate(g_n):
        ax.annotate(f"{100*n_/len(ev_p):.0f}%", (i, 0.02), ha="center", fontsize=9.5, color="white")
    ax.set_xticks(xs); ax.set_xticklabels([lab for _, lab in groups])
    ax.set_xlabel("Informative tokens the model could read")
    ax.set_ylabel("Probability"); ax.set_ylim(0, 1.18)
    ax.grid(True, axis="y", color=RULE, lw=0.6); ax.set_axisbelow(True)
    # above the bars, not over the n= labels sitting on the axis
    # explicit handles: the accuracy series is teal except the withheld group,
    # so letting matplotlib pick the first colour would imply all of it is red
    ax.legend(handles=[Patch(facecolor=SLATE, label="confidence the model states"),
                       Patch(facecolor=TEAL,  label="accuracy it achieves"),
                       Patch(facecolor=ROSE,  label="withheld by the evidence rule")],
              loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    titled(ax, "One token is not enough to justify a claim",
           "pooled over all 25 runs; the bars match from two tokens onward")
    plt.tight_layout(); plt.savefig(FIG/"figS3b_evidence.png"); plt.close()

    rng = np.random.default_rng(0)
    def gap_ci(v):
        m = (ev_p == v)
        d = [rng.choice(cr_p[m], m.sum(), replace=True).mean()
             - rng.choice(pm_p[m], m.sum(), replace=True).mean() for _ in range(8000)]
        return np.quantile(d, .025), np.quantile(d, .975)
    lo1, hi1 = gap_ci(1); lo2, hi2 = gap_ci(2)
    macro("EvOneN", g_n[0]);  macro("EvOneAcc", f"{g_acc[0]*100:.1f}")
    macro("EvOneConf", f"{g_conf[0]*100:.1f}"); macro("EvOneGap", f"{(g_conf[0]-g_acc[0])*100:.0f}")
    macro("EvOneCI", f"[{lo1:+.3f}, {hi1:+.3f}]")
    macro("EvTwoN", g_n[1]);  macro("EvTwoAcc", f"{g_acc[1]*100:.1f}")
    macro("EvTwoConf", f"{g_conf[1]*100:.1f}")
    macro("EvTwoCI", f"[{lo2:+.3f}, {hi2:+.3f}]")
    # what a stricter cut would cost: loci it would suppress that are still usable
    m3 = ev_p < 3
    macro("EvThreeN", f"{m3.sum()}"); macro("EvThreePct", f"{100*m3.mean():.1f}")
    macro("EvThreeAcc", f"{cr_p[m3].mean()*100:.1f}")
    # the same rule applied to the unlabelled pre-training corpus. Only tokenisation
    # is needed, not inference: the guard counts readable genes, it does not use
    # the probabilities. Vocabulary here is the deployed model's, since that is
    # what a user actually runs against.
    import gzip
    dep_vocab = set(joblib.load(ROOT/"artifacts/final_model_v2.pkl")
                    ["pipeline"].named_steps["cv"].get_feature_names_out())
    n_corpus, n_thin = 0, 0
    with gzip.open(ROOT/"data/unsupervised_corpus.txt.gz", "rt") as fh:
        for line in fh:
            n_corpus += 1
            n_inf_u = sum(1 for t in tok_cpu_v2(line.strip().replace(" ", ","))
                          if t != "null" and t in dep_vocab)
            n_thin += (n_inf_u < MIN_INFORM)
    macro("UnsupCorpusSize", f"{n_corpus:,}".replace(",", "{,}"))
    macro("UnsupSuppressPct", f"{100*n_thin/n_corpus:.1f}")

    # the two views of the same corpus: normalised vector vs raw votes
    _lines = [l.strip().replace(" ", ",") for l in
              gzip.open(ROOT/"data/unsupervised_corpus.txt.gz", "rt")]
    _Zu = pipe_final.named_steps["cv"].transform(_lines)
    _Vu = np.column_stack([e.predict_proba(_Zu)[:, 1]
                           for e in pipe_final.named_steps["vr"].estimators_])
    _PVu = binom.sf(np.rint(_Vu*NTREE).astype(int) - 1, NTREE, 0.5)
    _Nu = apply_temperature(pipe_final.predict_proba(_lines), T)
    macro("UnsupSigVotes", f"{100*np.mean(_PVu.min(1) < ALPHA/len(subs)):.0f}")
    macro("UnsupSigNorm",  f"{100*np.mean(_Nu.max(1) > 0.392401):.0f}")

    # S4 out-of-vocabulary proportion vs correctness ---------------------------
    # measured against each fold's own training vocabulary, and pooled over the
    # same 5x5 protocol as the predictions it is compared against
    oov_list = []
    for seed in REPS:
        sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        per_rep = []
        for fold, (tr, te) in enumerate(sk.split(X, y)):
            cv = CountVectorizer(tokenizer=tok_cpu_v2, token_pattern=None, lowercase=False)
            cv.fit(X[tr]); V = set(cv.vocabulary_)
            per_rep.append(np.array([sum(1 for t in tok_cpu_v2(X[i]) if t not in V)
                                     / max(len(tok_cpu_v2(X[i])), 1) for i in te]))
        oov_list.append(np.concatenate(per_rep))
    oovs = np.concatenate(oov_list)
    ob = [(0, .001), (.001, .05), (.05, .15), (.15, 1.01)]
    olab = ["0%", "0-5%", "5-15%", ">15%"]
    oacc = [float(correct[(oovs >= lo) & (oovs < hi)].mean()) if ((oovs >= lo) & (oovs < hi)).sum() else 0
            for lo, hi in ob]
    on = [int(((oovs >= lo) & (oovs < hi)).sum()) for lo, hi in ob]
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    ax.bar(range(len(ob)), oacc, color=TEAL, width=.6)
    for i, (a_, n_) in enumerate(zip(oacc, on)):
        ax.annotate(f"{a_*100:.0f}%", (i, a_), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=12, fontweight="bold", color=INK)
        ax.annotate(f"{100*n_/len(y_all):.0f}%", (i, 0.02), ha="center", fontsize=10, color="white")
    ax.set_xticks(range(len(ob))); ax.set_xticklabels(olab)
    ax.set_ylabel("Fraction correct"); ax.set_ylim(0, 1.12)
    ax.set_xlabel("Share of the locus's tokens unseen in training")
    ax.grid(True, axis="y", color=RULE, lw=0.6); ax.set_axisbelow(True)
    titled(ax, "Accuracy against unfamiliar vocabulary",
           "loci whose genes the training fold never saw are predicted less reliably")
    plt.tight_layout(); plt.savefig(FIG/"figS4_oov.png"); plt.close()
    macro("ZeroOovN", on[0]); macro("ZeroOovAcc", f"{oacc[0]*100:.0f}")
    macro("ZeroOovPct", f"{100*on[0]/len(y_all):.0f}")

    # The temperature is chosen by minimising calibration error, not log-loss.
    # No figure is emitted: the finding is one sentence in the supplement, and an
    # orphan plot in paper/Fig would be flagged as generated-but-never-shown.
    from src.calibration.temperature import apply_temperature as _appT, _nll as _NLL
    yi = np.array([cls.index(v) for v in y_all])
    Ts = np.linspace(0.35, 1.6, 60)
    nll = [_NLL(_appT(P, t), yi) for t in Ts]
    eces = []
    for t in Ts:
        Q=_appT(P,t); c=Q.max(1); cr=(np.array(cls)[Q.argmax(1)]==y_all); e=0
        for lo in np.arange(0,1,.1):
            m=(c>=lo)&(c<lo+.1)
            if m.sum(): e+=m.sum()*abs(cr[m].mean()-c[m].mean())
        eces.append(e/len(c))
    t_nll, t_ece = Ts[int(np.argmin(nll))], Ts[int(np.argmin(eces))]
    macro("Tnll", f"{t_nll:.2f}"); macro("Tece", f"{t_ece:.2f}")

    # S5 training cost --------------------------------------------------------
    tt = pfm.copy(); tt["fam"] = tt.shorthand.apply(family_of)
    agg = tt.groupby("fam").wall_sec.mean().sort_values()
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.barh(range(len(agg)), agg.values, color=SLATE, height=.6)
    for i, v in enumerate(agg.values):
        ax.annotate(f"{v:.0f} s", (v, i), textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_yticks(range(len(agg))); ax.set_yticklabels(agg.index, fontsize=10.5)
    ax.set_xlabel("Mean seconds per training run"); ax.set_xlim(0, agg.values.max()*1.18)
    ax.grid(True, axis="x", color=RULE, lw=0.6); ax.set_axisbelow(True)
    titled(ax, "What each family costs to train", "mean wall-clock seconds for one train/test run")
    plt.tight_layout(); plt.savefig(FIG/"figS5_cost.png"); plt.close()
    # Name the deployed family: there are two "Counts" groups now, and taking the
    # first of a series sorted by time silently picks whichever is cheaper.
    _counts_fam = "Counts + ExtraTrees"
    assert _counts_fam in agg.index, sorted(agg.index)
    macro("CostCounts", f"{agg[_counts_fam]:.0f}")
    # The supplement used to call this "one to two orders of magnitude". It is a small
    # single-digit multiple, so the ratio is emitted from the same series the figure
    # plots rather than described in prose.
    _deep = agg[[i for i in agg.index if i.startswith("Deep")]]
    macro("CostDeepMin", f"{_deep.min()/agg[_counts_fam]:.1f}")
    macro("CostDeepMax", f"{_deep.max()/agg[_counts_fam]:.1f}")

    # supplementary tables ----------------------------------------------------
    full = lb.copy()
    fb = ""
    for i, q in full.iterrows():
        f_key, c_key = q.shorthand.split("__")
        fname, rep = FEAT.get(f_key, (f_key.replace("_", r"\_"), ""))
        cname = CLF.get(c_key, c_key)
        fb += (f"{i+1} & {fname} $+$ {cname} & {rep} & "
               f"${q.mean_acc:.4f}\\pm{q.std_acc:.4f}$ & {q.min_acc:.4f} & {q.max_acc:.4f} \\\\\n")
    (GEN/"table_full_leaderboard.tex").write_text(
        "\\begin{tabular}{rllrrr}\n\\toprule\nRank & Configuration & Representation & "
        "Accuracy & Min & Max \\\\\n\\midrule\n" + fb + "\\bottomrule\n\\end{tabular}\n")

    kb = "".join(f"{s} & {per_sub_k[s][0]:.3f} & {per_sub_k[s][1]:.3f} & "
                 f"{per_sub_k[s][2]:.3f} & {per_sub_k[s][3]:.3f} \\\\\n" for s in subs)
    kb += ("\\midrule\n\\textbf{all} & " +
           " & ".join(f"\\textbf{{{v:.3f}}}" for v in topk) + " \\\\\n")
    (GEN/"table_topk.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\nSubstrate & K=1 & K=2 & K=3 & K=5 \\\\\n"
        "\\midrule\n" + kb + "\\bottomrule\n\\end{tabular}\n")

    sb = "".join(f"{q.substrate} & {int(q.eligible)} & {int(q.hit)} & {q.rate*100:.0f}\\% & "
                 f"{int(q.in_scope)} & {int(q.flagged)} & {q.srate*100:.0f}\\% \\\\\n"
                 for _, q in fu.sort_values("substrate").iterrows())
    (GEN/"table_funnel.tex").write_text(
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{By locus} & "
        "\\multicolumn{3}{c}{By enzyme family} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "Substrate & Eligible & Hit & Rate & In scope & Surfaced & Rate \\\\\n\\midrule\n"
        + sb + "\\bottomrule\n\\end{tabular}\n")

    (GEN/"numbers.tex").write_text(
        "% Generated by scripts/07c_build_paper_figures.py. Do not edit.\n" +
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(MACROS.items())))

    print(f"[07c] held-out accuracy {np.mean(foldacc):.4f} +/- {np.std(foldacc, ddof=1):.4f} over {len(foldacc)} fits ({len(y_all):,} predictions)")
    print(f"[07c] signature genes {TH}/{TE} = {TH/TE*100:.1f}%  ({len(X)-TE} loci not answerable)")
    # counted, not asserted: a hardcoded figure count silently goes stale
    # the first time a panel is added or dropped
    print(f"[07c] wrote {len(list(FIG.glob('*.png')))} figures and "
          f"{len(MACROS)} macros to paper/generated/")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
