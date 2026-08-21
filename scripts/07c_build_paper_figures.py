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
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
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
    T = joblib.load(ROOT/"artifacts/final_model_v2.pkl")["T"]
    lg = np.log(np.clip(P, 1e-12, None))/T
    Pc = np.exp(lg - lg.max(1, keepdims=True)); Pc /= Pc.sum(1, keepdims=True)
    macro("DeployT", f"{T:.4f}")

    # ============================================================ FIG 2
    cm = confusion_matrix(y, y_pred, labels=subs).astype(float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
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

    rows = []
    for s in subs:
        cnt = {}
        for t3 in abl[abl.true == s].top3:
            for tok in t3.split(";"):
                if tok: cnt[tok] = cnt.get(tok, 0) + 1
        for rank, (tok, n) in enumerate(sorted(cnt.items(), key=lambda kv: -kv[1])[:3], 1):
            if tok in canon[s]:                     status = "listed"
            elif tok.startswith(CAZY):              status = "cazy-not-listed"
            else:                                   status = "non-cazy"
            rows.append(dict(substrate=s, rank=rank, token=tok, n=n, status=status))
    sg = pd.DataFrame(rows); sg.to_csv(TAB/"table_top3_siggenes.csv", index=False)

    FILL = {"listed": "#d6ebe9", "cazy-not-listed": "#f7e2e2", "non-cazy": "#f0f2f4"}
    fig, ax = plt.subplots(figsize=(9.0, 6.1)); ax.axis("off")
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
                  loc="upper left", cellLoc="left", colWidths=[0.34, 0.22, 0.22, 0.22])
    tb.auto_set_font_size(False); tb.set_fontsize(11.5); tb.scale(1, 1.62)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("white"); c.set_linewidth(1.4)
        if i == 0:
            c.set_facecolor(INK); c.set_text_props(color="white", weight="bold")
        else:
            c.set_facecolor(colr[i-1][j])
            c.set_text_props(color=INK, weight="bold" if j == 0 else "normal")
    titled(ax, "The three genes the model leans on most, per substrate",
           "colour shows whether the curated enzyme table lists that family for that substrate")
    ax.legend(handles=[Patch(facecolor=FILL["listed"], label="listed for this substrate"),
                       Patch(facecolor=FILL["cazy-not-listed"], label="a CAZy family, but not listed here"),
                       Patch(facecolor=FILL["non-cazy"], label="not a CAZy family: transporter, regulator, or unannotated")],
              loc="upper left", bbox_to_anchor=(0, -0.02), ncol=1, handlelength=1.6)
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

    fig, (aL, aR) = plt.subplots(1, 2, figsize=(11.4, 5.4))
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

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.6))
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

    (GEN/"numbers.tex").write_text(
        "% Generated by scripts/07c_build_paper_figures.py. Do not edit.\n" +
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(MACROS.items())))

    print(f"[07c] held-out accuracy {acc:.4f}")
    print(f"[07c] signature genes {TH}/{TE} = {TH/TE*100:.1f}%  ({len(X)-TE} loci not answerable)")
    print(f"[07c] wrote 5 figures, 4 tables, and {len(MACROS)} macros to paper/generated/")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
