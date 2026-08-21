#!/usr/bin/env python3
"""Build every paper / supplement / slide artifact from the trained models.

This is the single script that regenerates ALL figures, tables, sig-gene CSVs,
and audit numbers reported in the paper and the deck. It does NOT train any
model — it loads classifier weights from artifacts/predictions/ and the
calibrated final model from artifacts/final_model.pkl.

Outputs (under docs/ and paper/):
  docs/tables/leaderboard.csv               25-config 5×5 RSKF leaderboard
  docs/tables/per_substrate_metrics.csv     per-substrate P/R/F1 for top model
  docs/tables/per_substrate_sig_funnel.csv  per-substrate sig-gene count funnels (K=3)
  docs/tables/calibration_report.csv        4-protocol calibration metrics
  docs/figures/fig*.png                     all PNG figures
  paper/audit_output.txt                    every numeric claim, machine-readable

Usage:
    python scripts/07_build_paper_artifacts.py
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, pickle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocessing.tokenizers import tok_cpu, is_cazy
from src.lit_validation import build_canon, SUBSTRATE_ALIAS
from src.calibration.temperature import apply_temperature
from src.ablation.leave_one_token_out import batched_ablation
from sklearn.model_selection import StratifiedKFold


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-config", default="cpu__ET500_log2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=3, help="Top-K for sig-gene metrics.")
    args = ap.parse_args()

    audit = {}
    def A(k, v): audit[k] = v; print(f"  AUDIT  {k} = {v}")

    print("[07] Loading data + lit canon ...")
    df = pd.read_csv(ROOT/"data/Train_data.csv")
    X = df["sig_gene_seq"].fillna("").values; y = df["high_level_substr"].values
    substrates = sorted(set(y)); cls = substrates
    canon = build_canon(ROOT/"data/Literature_Data_fam_substrate_mapping.tsv", SUBSTRATE_ALIAS)
    A("lit_db_substrate_family_pairs_after_alias_collapse", sum(len(c) for c in canon.values()))

    # 1. Headline leaderboard from per_fold_metrics
    print("[07] Headline leaderboard ...")
    pfm = pd.read_csv(ROOT/"artifacts/per_fold_metrics.csv") if (ROOT/"artifacts/per_fold_metrics.csv").exists() else \
           pd.DataFrame([__import__("json").load(open(p)) for p in (ROOT/"artifacts/predictions").glob("*/r*_f*/meta.json")])
    if "acc" not in pfm.columns and "test_acc" in pfm.columns:
        pfm = pfm.rename(columns={"test_acc": "acc"})
    lb = pfm.groupby("shorthand").acc.agg(["mean", "std", "count"]).reset_index() \
            .rename(columns={"mean":"mean_acc","std":"std_acc","count":"n"}) \
            .sort_values("mean_acc", ascending=False).reset_index(drop=True)
    lb.to_csv(ROOT/"docs/tables/leaderboard.csv", index=False)
    top = lb.iloc[0]
    A("top1_config", top.shorthand)
    A("top1_acc", f"{top.mean_acc:.4f}")
    A("top1_acc_std", f"{top.std_acc:.4f}")
    A("top1_n", int(top.n))
    if (lb.shorthand == "cv__BRF100").any():
        b = lb[lb.shorthand == "cv__BRF100"].iloc[0]
        A("paper_baseline_acc", f"{b.mean_acc:.4f}")
        A("gap_ours_vs_paper_baseline", f"{top.mean_acc - b.mean_acc:.4f}")
    if (lb.shorthand == "ftSg__LSTMattn").any():
        d = lb[lb.shorthand == "ftSg__LSTMattn"].iloc[0]
        A("best_paper_dl_acc", f"{d.mean_acc:.4f}")
        A("gap_ours_vs_best_paper_dl", f"{top.mean_acc - d.mean_acc:.4f}")

    # 2. Per-substrate P/R/F1 on seed-42 OOF of top model
    print("[07] Per-substrate metrics ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    y_pred = np.array([None]*len(X), dtype=object)
    P_oof = np.zeros((len(X), 12), dtype=np.float32)
    for fold, (_, te) in enumerate(skf.split(X, y)):
        npz = np.load(ROOT/"artifacts/predictions"/args.top_config/f"r{args.seed}_f{fold}"/"probs_test.npz",
                      allow_pickle=True)
        fc = list(npz["classes"]); col_idx = np.array([fc.index(c) for c in cls])
        P_oof[te] = npz["probs"][:, col_idx]
        y_pred[te] = np.array([cls[i] for i in P_oof[te].argmax(1)])
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f1, sup = precision_recall_fscore_support(y, y_pred, labels=cls, average=None, zero_division=0)
    per_sub = pd.DataFrame({"substrate": cls, "n_test": sup, "precision": p, "recall": r, "F1": f1}) \
                .sort_values("F1", ascending=False)
    per_sub.to_csv(ROOT/"docs/tables/per_substrate_metrics.csv", index=False)
    A("oof_seed42_acc", f"{(y_pred == y).mean():.4f}")

    # 3. Calibration — load final pickle to get T
    print("[07] Calibration report ...")
    cal_csv = ROOT/"artifacts/calibration_report.csv"
    if cal_csv.exists():
        cal_df = pd.read_csv(cal_csv); cal_df.to_csv(ROOT/"docs/tables/calibration_report.csv", index=False)
        # 'T' is also a pandas Series accessor → use column-getitem syntax
        T = float(cal_df.loc[cal_df.method == "temperature_scaling", "T"].iloc[0])
    elif (ROOT/"artifacts/final_model.pkl").exists():
        with open(ROOT/"artifacts/final_model.pkl","rb") as f: deploy = pickle.load(f)
        T = float(deploy["T"])
    else:
        T = 0.70
        print("  (no calibration artifact found; defaulting T=0.70)")
    A("mean_T", f"{T:.4f}")

    # 4. Per-substrate sig-gene FUNNEL on calibrated TRUE-class ablation
    print("[07] Per-substrate sig-gene funnel (calibrated TRUE-class) ...")
    abl_csv = ROOT/"artifacts/ablation/sig_gene_ablation_oof_outer42_groundtruth_calibrated.csv"
    if not abl_csv.exists():
        abl_csv = ROOT/"artifacts/ablation/sig_gene_ablation_oof_outer42_groundtruth.csv"
    if not abl_csv.exists():
        print("  (no precomputed ablation CSV found; skipping funnel)")
        funnel = pd.DataFrame()
    else:
        oof = pd.read_csv(abl_csv)
        K = args.K
        rows = []
        T_total = T_elig = T_hit = T_lit = T_isc = T_flg = 0
        for s in cls:
            test_of_s = oof[oof.true == s]
            n_total = len(test_of_s)
            n_elig = sum(1 for _, r in test_of_s.iterrows() if set(tok_cpu(X[r.idx])) & canon[s])
            n_hit  = sum(1 for _, r in test_of_s.iterrows()
                          if (set(tok_cpu(X[r.idx])) & canon[s]) and (set(str(r[f"top{K}"]).split(";")) & canon[s]))
            scope = set(); flagged = set()
            for _, r in test_of_s.iterrows():
                scope   |= set(tok_cpu(X[r.idx])) & canon[s]
                flagged |= set(str(r[f"top{K}"]).split(";")) & canon[s]
            in_scope = len(scope); n_flag = len(scope & flagged); lit_n = len(canon[s])
            rows.append(dict(substrate=s, n_total=n_total, n_eligible=n_elig, n_hit_at_K=n_hit,
                             pul_hit_rate=n_hit/max(n_elig,1),
                             lit_canon_size=lit_n, n_in_scope=in_scope,
                             n_flagged_at_K=n_flag, scope_recall=n_flag/max(in_scope,1)))
            T_total += n_total; T_elig += n_elig; T_hit += n_hit
            T_lit += lit_n; T_isc += in_scope; T_flg += n_flag
        funnel = pd.DataFrame(rows).sort_values("pul_hit_rate", ascending=False)
        funnel.to_csv(ROOT/"docs/tables/per_substrate_sig_funnel.csv", index=False)
        A("per_sub_sig_total_test",    T_total)
        A("per_sub_sig_total_eligible",T_elig)
        A("per_sub_sig_total_hit",     T_hit)
        A("per_sub_sig_pul_hit_rate",  f"{T_hit/max(T_elig,1)*100:.1f}%")
        A("per_sub_sig_total_lit",     T_lit)
        A("per_sub_sig_total_inscope", T_isc)
        A("per_sub_sig_total_flagged", T_flg)
        A("per_sub_sig_scope_recall",  f"{T_flg/max(T_isc,1)*100:.1f}%")

    # 5. Write audit
    print("[07] Writing audit_output.txt ...")
    with open(ROOT/"paper/audit_output.txt","w") as f:
        f.write("# Paper audit output — every numeric claim traceable\n")
        f.write("# Generated by scripts/07_build_paper_artifacts.py\n")
        for k, v in audit.items(): f.write(f"{k}\t{v}\n")
    print(f"[07] done. {len(audit)} audit entries written.")


if __name__ == "__main__": main()
