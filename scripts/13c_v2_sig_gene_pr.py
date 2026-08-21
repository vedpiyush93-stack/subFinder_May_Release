#!/usr/bin/env python3
"""V2 signature-gene precision/recall analysis (mirrors notebook Section 8b for v1).

For each of the 5 seed-42 StratifiedKFold folds:
  1. Train cpu__ET500_log2 with tok_cpu_v2 on the train fold (824 PULs).
  2. Run BATCHED leave-one-token-out ablation against the TRUE class for
     each test PUL (206 PULs/fold). Top-K positive-delta tokens per PUL.
  3. Append per-PUL rows {idx, fold, true, pred, top{1,3,5}, top5_with_delta}.

Then build a *family-augmented* canon (for each specific canon token like
GH13, also add the family prefix GH). A v2 sig-gene chip like "GH" is now
treated as a family-fallback canon hit if any GH* is canonical for the
substrate. The per-substrate PR table reports three counts:

  hit_specific: top-K contains a specific canon token (e.g. GH13)
  hit_family:   top-K contains a family-fallback whose family has any
                canonical specific token (e.g. GH where GH13 is canon)
  hit_any:      hit_specific OR hit_family

Outputs (paper-table-shaped):
  artifacts/ablation/sig_gene_ablation_oof_outer42_v2.csv   per-PUL top-K
  paper/tables/table12_v2_per_substrate_sig_pr.csv          per-substrate PR
  paper/tables/table12_v2_aggregate_sig_pr.csv              all-substrates aggregate

Usage:  python3 scripts/13c_v2_sig_gene_pr.py
"""
from __future__ import annotations
import argparse, os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocessing.tokenizers import tok_cpu_v2

CAZY_FAMILY_RE = re.compile(r"^(GH|GT|PL|CE|CBM|AA)([0-9]+)(_[0-9]+)?$")
CAZY_FAMILY_ONLY_RE = re.compile(r"^(GH|GT|PL|CE|CBM|AA)$")

print("[13c] loading supervised training set ...")
df = pd.read_csv(ROOT / "data/Train_data.csv")
X = df["sig_gene_seq"].fillna("").values
y = df["high_level_substr"].values
substrates = sorted(set(y))
print(f"      {len(X):,} PULs, {len(substrates)} substrates")

_ap = argparse.ArgumentParser(description="v2 signature-gene ablation + literature PR")
_ap.add_argument("--split-seed", type=int, default=42,
                 help="Outer repeat to run. Its 5 folds together cover all 1,030 PULs. "
                      "Output filenames carry this seed so repeats do not overwrite each other.")
_args = _ap.parse_args()
SPLIT_SEED = _args.split_seed
MODEL_SEED = int(os.environ.get("REPRO_REP_SEED", "42"))

print(f"[13c] running 5-fold v2 OOF ablation (outer repeat seed={SPLIT_SEED}, "
      f"model seed={MODEL_SEED}, true-class attribution) ...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
all_rows = []
t0 = time.time()
for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
    t1 = time.time()
    cv = CountVectorizer(tokenizer=tok_cpu_v2, lowercase=False,
                          token_pattern=None, ngram_range=(1, 1))
    Xtr = cv.fit_transform(X[tr_idx])
    Xte = cv.transform(X[te_idx]).tocsr()
    inv_vocab = {v: k for k, v in cv.vocabulary_.items()}
    clf = OneVsRestClassifier(ExtraTreesClassifier(
        n_estimators=500, max_features="log2", class_weight="balanced",
        bootstrap=False, n_jobs=-1, random_state=MODEL_SEED))
    clf.fit(Xtr, y[tr_idx])
    classes_list = list(clf.classes_)
    P = clf.predict_proba(Xte)
    pred = P.argmax(1)

    rows_mod, meta = [], []
    for i_local in range(Xte.shape[0]):
        row = Xte.getrow(i_local)
        for col_i in row.nonzero()[1]:
            mod = row.tolil(); mod[0, col_i] = 0
            rows_mod.append(mod.tocsr())
            meta.append((i_local, inv_vocab[col_i]))
    big = sparse.vstack(rows_mod) if rows_mod else sparse.csr_matrix((0, Xte.shape[1]))
    Pa = clf.predict_proba(big)

    i_m = 0
    for i_local, gl in enumerate(te_idx):
        true_s = y[gl]
        if true_s in classes_list:
            ci_true = classes_list.index(true_s)
            p0_true = float(P[i_local, ci_true])
        else:
            ci_true = pred[i_local]; p0_true = float(P[i_local, ci_true])
        ci_pred = int(pred[i_local])
        pul_toks = []
        while i_m < len(meta) and meta[i_m][0] == i_local:
            pul_toks.append((meta[i_m][1], p0_true - float(Pa[i_m, ci_true])))
            i_m += 1
        pos = sorted([(t, d) for t, d in pul_toks if d > 0], key=lambda x: -x[1])[:5]
        all_rows.append({
            "idx": int(gl), "fold": fold_idx,
            "true": true_s, "pred": classes_list[ci_pred],
            "prob_true": p0_true,
            "top1": pos[0][0] if pos else "",
            "top3": ";".join(t for t, _ in pos[:3]),
            "top5": ";".join(t for t, _ in pos[:5]),
            "top5_with_delta": ";".join(f"{t}:{d:+.4f}" for t, d in pos[:5]),
        })
    print(f"  fold {fold_idx}: {len(te_idx):,} test PULs ablated in {time.time()-t1:.1f}s")

oof = pd.DataFrame(all_rows)
ablation_out = ROOT / f"artifacts/ablation/sig_gene_ablation_oof_outer{SPLIT_SEED}_v2.csv"
oof.to_csv(ablation_out, index=False)
print(f"[13c] wrote {ablation_out.relative_to(ROOT)}  ({len(oof):,} rows, {(oof['true']==oof['pred']).sum()} correct)")

# ---------------------------------------------------------------------------
# Literature canon: substrate -> {documented CAZy families}
#
# An earlier version also derived family prefixes ("GH13" -> "GH") so that the
# family-only tokens the tokenizer used to emit could be credited. The tokenizer
# no longer emits them, so the canon is matched exactly as curated. The
# canon_family bucket is retained and reported so the tables keep their shape;
# it is now expected to be empty.
# ---------------------------------------------------------------------------
print("[13c] building lit canon (exact families, no family-prefix augmentation) ...")
lit = pd.read_csv(ROOT / "data/Literature_Data_fam_substrate_mapping.tsv", sep="\t")
lit.columns = [c.strip() for c in lit.columns]
SA = {
    "alpha-glucan": ["alpha-glucan","starch","glycogen","sucrose","raffinose","trehalose","palatinose","glucooligosaccharide"],
    "beta-glucan":  ["beta-glucan","cellulose","cellooligosaccharide","xyloglucan","beta-glycan"],
    "galactan":     ["beta-galactan","alpha-galactan"],
    "arabinogalactan": ["arabinogalactan protein","arabinan"],
    "host glycan":  ["host glycan","human-milk-polysaccharide","human milk polysaccharide","sialic-acid","sialic acid","fucose"],
    "chitin":       ["chitin","chitooligosaccharide","chitosan"],
    "alginate":     ["alginate"], "pectin": ["pectin"], "xylan": ["xylan"],
    "alpha-mannan": ["alpha-mannan"], "beta-mannan": ["beta-mannan"], "fructan": ["fructan"],
}
SPLIT_RE = re.compile(r",|and\s+")
canon_specific = {s: set() for s in substrates}
for _, r in lit.iterrows():
    parts = [p.strip() for p in SPLIT_RE.split(str(r["Substrate_high_level"])) if p.strip()]
    for s, aliases in SA.items():
        if set(aliases) & set(parts):
            canon_specific[s].add(r["Family"])
canon_family = {s: set() for s in substrates}   # no family-prefix augmentation
print(f"      canon_specific (sample): alpha-glucan has {len(canon_specific['alpha-glucan'])} tokens "
      f"({sorted(canon_specific['alpha-glucan'])[:5]} ...)")
print(f"      canon_family: disabled (tokenizer emits no family-only tokens)")

# ---------------------------------------------------------------------------
# Per-substrate PR with three count buckets
# ---------------------------------------------------------------------------
K = 3
def is_family_token(t): return bool(CAZY_FAMILY_ONLY_RE.match(t))
def is_specific(t):     return bool(CAZY_FAMILY_RE.match(t))

rows = []
for s in substrates:
    test_of_s = oof[oof.true == s]
    n_total = len(test_of_s)
    n_elig = 0
    hit_specific = 0; hit_family = 0; hit_any = 0
    inscope_specific = set(); inscope_family = set()
    flag_specific = set(); flag_family = set()
    for _, r in test_of_s.iterrows():
        toks = set(tok_cpu_v2(X[r["idx"]]))
        spec_in_pul = toks & canon_specific[s]
        fam_in_pul  = {t for t in toks if is_family_token(t) and t in canon_family[s]}
        if spec_in_pul or fam_in_pul:
            n_elig += 1
        inscope_specific |= spec_in_pul
        inscope_family   |= fam_in_pul
        top = set(str(r[f"top{K}"]).split(";"))
        s_hit = bool(top & canon_specific[s])
        f_hit = any(is_family_token(t) and t in canon_family[s] for t in top)
        if s_hit: hit_specific += 1
        if f_hit: hit_family += 1
        if s_hit or f_hit: hit_any += 1
        flag_specific |= top & canon_specific[s]
        flag_family   |= {t for t in top if is_family_token(t) and t in canon_family[s]}
    rows.append({
        "substrate": s,
        "n_total": n_total, "n_eligible": n_elig,
        "hit_specific_at_K": hit_specific,
        "hit_family_at_K":   hit_family,
        "hit_any_at_K":      hit_any,
        "hit_rate_any":      hit_any / n_elig if n_elig else 0.0,
        "canon_specific_size": len(canon_specific[s]),
        "canon_family_size":   len(canon_family[s]),
        "in_scope_specific":   len(inscope_specific),
        "in_scope_family":     len(inscope_family),
        "flagged_specific":    len(flag_specific),
        "flagged_family":      len(flag_family),
        "scope_recall_specific": (len(flag_specific)/len(inscope_specific)) if inscope_specific else 0.0,
        "scope_recall_family":   (len(flag_family)/len(inscope_family))     if inscope_family   else 0.0,
    })
pr_df = pd.DataFrame(rows).sort_values("hit_rate_any", ascending=False)
pr_out = ROOT / f"paper/tables/table12_v2_per_substrate_sig_pr_outer{SPLIT_SEED}.csv"
pr_df.to_csv(pr_out, index=False)
print(f"[13c] wrote {pr_out.relative_to(ROOT)}")

agg = pd.DataFrame([{
    "n_total":       int(pr_df.n_total.sum()),
    "n_eligible":    int(pr_df.n_eligible.sum()),
    "hit_specific":  int(pr_df.hit_specific_at_K.sum()),
    "hit_family":    int(pr_df.hit_family_at_K.sum()),
    "hit_any":       int(pr_df.hit_any_at_K.sum()),
    "hit_rate_any":  pr_df.hit_any_at_K.sum() / pr_df.n_eligible.sum() if pr_df.n_eligible.sum() else 0.0,
    "hit_rate_specific_only": pr_df.hit_specific_at_K.sum() / pr_df.n_eligible.sum() if pr_df.n_eligible.sum() else 0.0,
    "in_scope_specific": int(pr_df.in_scope_specific.sum()),
    "in_scope_family":   int(pr_df.in_scope_family.sum()),
    "flagged_specific":  int(pr_df.flagged_specific.sum()),
    "flagged_family":    int(pr_df.flagged_family.sum()),
    "scope_recall_specific": pr_df.flagged_specific.sum() / pr_df.in_scope_specific.sum() if pr_df.in_scope_specific.sum() else 0.0,
    "scope_recall_family":   pr_df.flagged_family.sum()   / pr_df.in_scope_family.sum()   if pr_df.in_scope_family.sum()   else 0.0,
}])
agg_out = ROOT / f"paper/tables/table12_v2_aggregate_sig_pr_outer{SPLIT_SEED}.csv"
agg.to_csv(agg_out, index=False)
print(f"[13c] wrote {agg_out.relative_to(ROOT)}")

print(f"\n[v2 sig-gene PR @K={K}, true-class attribution, family-augmented canon]")
print(f"{'substrate':<18} {'n_tot':>6}{'elig':>6}{'spec':>6}{'fam':>5}{'any':>5}  rate")
print("-" * 60)
for _, r in pr_df.iterrows():
    print(f"{r.substrate:<18} {int(r.n_total):>6}{int(r.n_eligible):>6}{int(r.hit_specific_at_K):>6}"
          f"{int(r.hit_family_at_K):>5}{int(r.hit_any_at_K):>5}  {r.hit_rate_any*100:>5.1f}%")
print("-" * 60)
a = agg.iloc[0]
print(f"{'TOTAL':<18} {a.n_total:>6}{a.n_eligible:>6}{a.hit_specific:>6}{a.hit_family:>5}{a.hit_any:>5}  {a.hit_rate_any*100:>5.1f}%")
print(f"\nScope recall (gene-view):")
print(f"  specific lit-canon tokens: {a.flagged_specific}/{a.in_scope_specific} = {a.scope_recall_specific*100:.1f}%")
print(f"  family-fallback tokens:    {a.flagged_family}/{a.in_scope_family} = {a.scope_recall_family*100:.1f}%")
print(f"\n[13c] wall: {time.time()-t0:.0f}s")
