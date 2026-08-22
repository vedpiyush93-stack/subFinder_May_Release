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
import argparse, sys, warnings
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
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=S)
    y_pred = np.array([None]*len(X), dtype=object)
    P = np.zeros((len(X), len(subs))); cls = None
    for fold, (_, te) in enumerate(skf.split(X, y)):
        z = np.load(ROOT/f"artifacts/predictions/{DEPLOYED}/r{S}_f{fold}/probs_test.npz",
                    allow_pickle=True)
        cls = [str(c) for c in z["classes"]]
        P[te] = z["probs"]; y_pred[te] = np.array(cls)[z["probs"].argmax(1)]
    acc = float((y_pred == y).mean())
    macro("HeldOutAcc", f"{acc:.4f}")

    import joblib
    from src.calibration.temperature import apply_temperature
    T = float(joblib.load(ROOT/"artifacts/final_model_v2.pkl")["T"])
    # must be the same transform the deployed model applies (per-class logit / T,
    # sigmoid, renormalise) -- a softmax-style log(p)/T is a different operation
    # and would report probabilities no user ever sees.
    Pc = apply_temperature(P, T)
    macro("DeployT", f"{T:.4f}")

    # ============================================================ FIG 2
    cm = confusion_matrix(y, y_pred, labels=subs).astype(float)
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

    p, r, f1, sup = precision_recall_fscore_support(y, y_pred, labels=subs,
                                                    average=None, zero_division=0)
    per = pd.DataFrame({"substrate": subs, "n": sup, "precision": p, "recall": r,
                        "f1": f1}).sort_values("f1", ascending=False)
    per.to_csv(TAB/"table_per_substrate.csv", index=False)
    body = "\n".join(f"{q.substrate} & {int(q.n)} & {q.precision:.2f} & {q.recall:.2f} & {q.f1:.2f} \\\\"
                     for _, q in per.iterrows())
    (GEN/"table_persubstrate.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\nSubstrate & $n$ & Precision & Recall & F1 \\\\\n"
        f"\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n")
    worst = per.iloc[-1]; best = per.iloc[0]
    macro("WorstSub", worst.substrate); macro("WorstF", f"{worst.f1:.2f}")
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
    SKIP = {"null", ""}
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
    top3 = {int(rr.idx): rr.top3 for _, rr in abl.iterrows()}
    recs = []
    for i in range(len(X)):
        if i not in top3: continue
        o = np.argsort(-Pc[i]); tr = int(list(o).index(cls.index(y[i])))+1
        recs.append(dict(idx=i, true=y[i], pred=cls[int(o[0])], conf=float(Pc[i][o[0]]),
                         second=cls[int(o[1])], p2=float(Pc[i][o[1]]), rank=tr,
                         sig=top3[i], margin=float(Pc[i][o[0]]-Pc[i][o[1]])))
    C = pd.DataFrame(recs)
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
    take((C.true == C.pred) & (C.conf > .90), "Confident and correct")
    take((C.true == C.pred) & C.conf.between(.35, .60), "Correct, rightly hedged")
    take((C.true != C.pred) & (C["rank"] == 2) & (C.margin < .12), "Wrong; truth a near-tie 2nd")
    take((C.true != C.pred) & C.rescuable & (C["rank"] == 2), "Wrong; genes give it away")
    take((C.true == C.pred) & (C.true == "chitin"), "Right, no listed enzyme")
    take((C.true == C.pred) & C.conf.between(.60, .88), "A routine call")
    while len(picks) < 6: take(pd.Series(True, index=C.index), "Further example")

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.0))
    fig.subplots_adjust(hspace=1.05, wspace=0.14, top=0.82)
    for ax, (q, label) in zip(axes.ravel(), picks):
        o = np.argsort(-Pc[q.idx])[:3]
        for k, t in enumerate(o):
            v = float(Pc[q.idx][t]); good = cls[t] == q.true
            ax.barh(2-k, v, color=TEAL if good else ROSE, height=.66, linewidth=0)
            ins = v >= 0.62
            ax.text(v-0.025 if ins else v+0.025, 2-k, f"{cls[t]}  {v:.2f}",
                    va="center", ha="right" if ins else "left", fontsize=10.5,
                    color="white" if ins else INK,
                    fontweight="bold" if good else "normal")
        ax.set_xlim(0, 1.02); ax.set_ylim(-0.6, 2.6); ax.set_yticks([])
        ax.set_xticks([0, .5, 1]); ax.tick_params(labelsize=9.5)
        ax.set_title(label, loc="left", fontsize=12, fontweight="bold", color=INK, pad=30)
        ax.annotate(f"true substrate: {q.true}\nsignature genes: {str(q.sig).replace(';', ',  ')}",
                    xy=(0, 1.04), xycoords="axes fraction", fontsize=10, color=MUTED,
                    ha="left", va="bottom", linespacing=1.6)
        for sp in ax.spines.values(): sp.set_visible(False)
    fig.suptitle("Six held-out predictions", x=0.005, ha="left", fontsize=15,
                 fontweight="bold", color=INK, y=0.985)
    fig.text(0.005, 0.925, "the three highest calibrated probabilities; teal marks the true substrate",
             fontsize=11.5, color=MUTED, ha="left")
    plt.savefig(FIG/"fig5_cases.png", bbox_inches="tight"); plt.close()
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

    # ============================================================ leaderboard + macros
    def row(cfg): 
        z = lb[lb.shorthand == cfg].iloc[0]; return float(z.mean_acc), float(z.std_acc)
    dep_a, dep_s = row(DEPLOYED); macro("DepAcc", f"{dep_a:.4f}"); macro("DepStd", f"{dep_s:.4f}")
    brf_a, brf_s = row("cv__BRF100"); macro("BrfAcc", f"{brf_a:.4f}"); macro("BrfStd", f"{brf_s:.4f}")
    dl = lb[lb.shorthand.str.contains("__LSTM|__Trans|__JustAttn")].iloc[0]
    macro("DlAcc", f"{dl.mean_acc:.4f}"); macro("DlStd", f"{dl.std_acc:.4f}")
    macro("GapDl", f"{(dep_a-dl.mean_acc)*100:.2f}"); macro("GapBrf", f"{(dep_a-brf_a)*100:.2f}")
    macro("DlWorst", f"{lb[lb.shorthand.str.contains('__LSTM|__Trans|__JustAttn')].mean_acc.min():.4f}")

    import joblib as _j
    vocab = len(_j.load(ROOT/"artifacts/final_model_v2.pkl")["pipeline"].named_steps["cv"].vocabulary_)
    macro("Vocab", vocab)
    macro("TokPerPul", f"{np.mean([len(tok_cpu_v2(x)) for x in X]):.1f}")
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
    (GEN/"table_calibration.tex").write_text(
        "\\begin{tabular}{lrr}\n\\toprule\nMethod & Accuracy & ECE (10-bin) \\\\\n"
        f"\\midrule\n{cb}\\bottomrule\n\\end{{tabular}}\n")
    u = cal[cal.method == "uncalibrated"].iloc[0]
    t = cal[cal.method == "temperature_scaling"].iloc[0]
    iso = cal[cal.method.str.startswith("isotonic")].iloc[0]
    macro("EceRaw", f"{u.ece_10bin:.3f}"); macro("EceTemp", f"{t.ece_10bin:.3f}")
    macro("EceIso", f"{iso.ece_10bin:.3f}"); macro("IsoAcc", f"{iso.accuracy:.4f}")
    macro("MeanT", f"{float(t['T']):.2f}")

    # ======================================================== SUPPLEMENTARY
    from sklearn.feature_extraction.text import CountVectorizer

    # S1 reliability diagram --------------------------------------------------
    conf_raw = P.max(1); conf_cal = Pc.max(1); correct = (y_pred == y)
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
    topk = [float(np.mean([y[i] in [cls[j] for j in ordr[i][:k]] for i in range(len(y))]))
            for k in (1, 2, 3, 5)]
    per_sub_k = {s: [float(np.mean([y[i] in [cls[j] for j in ordr[i][:k]]
                                    for i in range(len(y)) if y[i] == s])) for k in (1, 2, 3, 5)]
                 for s in subs}
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
    SIG_TH = brentq(lambda q: 12*(1-q)**11 - 0.05, 1e-6, 0.999)
    macro("SigThresh", f"{SIG_TH:.3f}")
    bins = [(0, SIG_TH), (SIG_TH, .6), (.6, .8), (.8, .95), (.95, 1.01)]
    lab = ["below\nsignificance", f"{SIG_TH:.2f}-0.60", "0.60-0.80", "0.80-0.95", "0.95-1.00"]
    accs, cnts = [], []
    for lo, hi in bins:
        m = (conf_cal >= lo) & (conf_cal < hi)
        accs.append(float(correct[m].mean()) if m.sum() else 0.0); cnts.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    bars = ax.bar(range(len(bins)), accs, color=[ROSE] + [TEAL]*4, width=.62)
    for i, (a_, n_) in enumerate(zip(accs, cnts)):
        ax.annotate(f"{a_*100:.0f}%", (i, a_), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=12, fontweight="bold", color=INK)
        ax.annotate(f"{n_} loci", (i, 0.02), ha="center", fontsize=10, color="white")
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(lab, fontsize=10)
    ax.set_ylabel("Fraction correct"); ax.set_ylim(0, 1.12)
    ax.grid(True, axis="y", color=RULE, lw=0.6); ax.set_axisbelow(True)
    titled(ax, "Confidence tracks correctness",
           "loci grouped by the winning calibrated probability; red is the band the $p$-value rejects")
    plt.tight_layout(); plt.savefig(FIG/"figS3_confidence.png"); plt.close()
    macro("HighConfAcc", f"{accs[-1]*100:.1f}"); macro("HighConfN", cnts[-1])
    macro("BelowSigN", cnts[0]); macro("BelowSigAcc", f"{accs[0]*100:.0f}")

    # S4 out-of-vocabulary proportion vs correctness ---------------------------
    oovs = np.zeros(len(X))
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        cv = CountVectorizer(tokenizer=tok_cpu_v2, token_pattern=None, lowercase=False)
        cv.fit(X[tr]); V = set(cv.vocabulary_)
        for i in te:
            tk = tok_cpu_v2(X[i])
            oovs[i] = sum(1 for t in tk if t not in V)/max(len(tk), 1)
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
        ax.annotate(f"{n_} loci", (i, 0.02), ha="center", fontsize=10, color="white")
    ax.set_xticks(range(len(ob))); ax.set_xticklabels(olab)
    ax.set_ylabel("Fraction correct"); ax.set_ylim(0, 1.12)
    ax.set_xlabel("Share of the locus's tokens unseen in training")
    ax.grid(True, axis="y", color=RULE, lw=0.6); ax.set_axisbelow(True)
    titled(ax, "Accuracy against unfamiliar vocabulary",
           "loci whose genes the training fold never saw are predicted less reliably")
    plt.tight_layout(); plt.savefig(FIG/"figS4_oov.png"); plt.close()
    macro("ZeroOovN", on[0]); macro("ZeroOovAcc", f"{oacc[0]*100:.0f}")

    # S6 the calibration objective ------------------------------------------
    from src.calibration.temperature import apply_temperature as _appT, _nll as _NLL
    yi = np.array([cls.index(v) for v in y])
    Ts = np.linspace(0.35, 1.6, 60)
    nll = [_NLL(_appT(P, t), yi) for t in Ts]
    eces = []
    for t in Ts:
        Q=_appT(P,t); c=Q.max(1); cr=(np.array(cls)[Q.argmax(1)]==y); e=0
        for lo in np.arange(0,1,.1):
            m=(c>=lo)&(c<lo+.1)
            if m.sum(): e+=m.sum()*abs(cr[m].mean()-c[m].mean())
        eces.append(e/len(c))
    t_nll, t_ece = Ts[int(np.argmin(nll))], Ts[int(np.argmin(eces))]
    macro("Tnll", f"{t_nll:.2f}"); macro("Tece", f"{t_ece:.2f}")
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.plot(Ts, eces, color=TEAL, lw=2.6, label="calibration error (ECE)")
    ax.axvline(t_ece, color=TEAL, ls=(0,(4,3)), lw=1.4)
    ax.annotate(f"ECE best\nT={t_ece:.2f}", (t_ece, max(eces)*0.80), color=TEAL,
                fontsize=10.5, ha="center", fontweight="bold")
    ax.set_xlabel("Temperature $T$   (below 1 sharpens, above 1 flattens)")
    ax.set_ylabel("Calibration error", color=TEAL)
    ax.tick_params(axis="y", colors=TEAL)
    ax2 = ax.twinx(); ax2.plot(Ts, nll, color=AMBER, lw=2.6, label="log-loss (NLL)")
    ax2.axvline(t_nll, color=AMBER, ls=(0,(4,3)), lw=1.4)
    ax2.annotate(f"NLL best\nT={t_nll:.2f}", (t_nll, max(nll)*0.995), color=AMBER,
                 fontsize=10.5, ha="center", fontweight="bold")
    ax2.set_ylabel("Log-loss", color=AMBER); ax2.tick_params(axis="y", colors=AMBER)
    ax2.spines["top"].set_visible(False)
    ax.axvline(1.0, color=SLATE, lw=1.0)
    ax.annotate("no correction", (1.0, min(eces)), xytext=(6,4),
                textcoords="offset points", fontsize=9.5, color=SLATE)
    titled(ax, "The two objectives disagree, and only one of them calibrates",
           "fitting the temperature by log-loss barely sharpens; fitting by calibration error does")
    plt.tight_layout(); plt.savefig(FIG/"figS6_objective.png"); plt.close()

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
    macro("CostCounts", f"{agg[[i for i in agg.index if i.startswith('Counts')][0]]:.0f}")

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

    print(f"[07c] held-out accuracy {acc:.4f}")
    print(f"[07c] signature genes {TH}/{TE} = {TH/TE*100:.1f}%  ({len(X)-TE} loci not answerable)")
    print(f"[07c] wrote 11 figures and {len(MACROS)} macros to paper/generated/")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
