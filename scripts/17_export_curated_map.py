#!/usr/bin/env python3
"""Export the curated enzyme-to-substrate table with the label merge made explicit.

The literature table is annotated with fine-grained substrate names -- ``starch``,
``glycogen``, ``cellulose``, ``xyloglucan``, ``sialic-acid`` and so on. The model predicts
twelve classes, so those names are collapsed by ``src/lit_validation/alias_map.py`` before
anything is scored against them. That collapse is the single most domain-sensitive decision
in the pipeline: it decides what counts as a correct explanation.

This writes it out in full, so a specialist can check it row by row without reading any
code:

  curated_family_substrate_map.csv   one row per (source label, CAZy family) in the source
                                     table, with the class it was merged into, the enzyme
                                     name, the EC number, and whether the model can
                                     actually read that family
  substrate_alias_summary.csv        one row per source label: what it merged into, how
                                     many families it contributed, how many of those the
                                     model could ever name (a family absent from all 1,030
                                     training loci is not in its vocabulary, so it can
                                     never be reported), or why the label was dropped
  curated_map.xlsx                   both of the above as one workbook, if openpyxl is
                                     installed

    python3 scripts/17_export_curated_map.py            # -> paper/tables/
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.lit_validation.alias_map import SUBSTRATE_ALIAS  # noqa: E402
from src.lit_validation.canon import _split_lit_substr    # noqa: E402

TSV = ROOT / "data" / "Literature_Data_fam_substrate_mapping.tsv"
OUT = ROOT / "paper" / "tables"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lit = pd.read_csv(TSV, sep="\t")
    lit.columns = [c.strip() for c in lit.columns]

    # source label -> our class
    to_class: dict[str, str] = {}
    for our, aliases in SUBSTRATE_ALIAS.items():
        for a in aliases:
            to_class[a] = our

    vocab = set(joblib.load(ROOT / "artifacts" / "final_model_v2.pkl")
                ["pipeline"].named_steps["cv"].get_feature_names_out())

    rows = []
    for _, r in lit.iterrows():
        fam = str(r["Family"]).strip()
        for part in _split_lit_substr(r["Substrate_high_level"]):
            rows.append({
                "source_substrate": part,
                "merged_substrate": to_class.get(part, ""),
                "modelled": bool(part in to_class),
                "cazy_family": fam,
                "enzyme_name": str(r.get("Name", "")).strip(),
                "ec_number": str(r.get("EC_Number", "")).strip(),
                "family_in_model_vocabulary": fam in vocab,
            })
    df = pd.DataFrame(rows).drop_duplicates()
    df = df.sort_values(["merged_substrate", "source_substrate", "cazy_family"],
                        key=lambda c: c.where(c != "", "zzz") if c.name == "merged_substrate" else c)
    p1 = args.out / "curated_family_substrate_map.csv"
    df.to_csv(p1, index=False)

    # one row per source label
    # count DISTINCT families, not rows: one family appears on several rows of the
    # source table, once per enzyme name / EC number it covers.
    g = (df.groupby(["source_substrate", "merged_substrate", "modelled"], dropna=False)
           .agg(distinct_families=("cazy_family", "nunique"),
                families_the_model_could_ever_report=(
                    "cazy_family",
                    lambda c: df.loc[c.index][df.loc[c.index].family_in_model_vocabulary]
                                .cazy_family.nunique()),
                source_rows=("cazy_family", "size"))
           .reset_index())
    g["status"] = g.apply(
        lambda r: f"merged into '{r.merged_substrate}'" if r.modelled
        else "not modelled (no counterpart among the 12 classes)", axis=1)
    g = g.sort_values(["modelled", "merged_substrate", "source_substrate"],
                      ascending=[False, True, True])
    p2 = args.out / "substrate_alias_summary.csv"
    g.drop(columns=["modelled"]).to_csv(p2, index=False)

    modelled = df[df.modelled]
    pairs = modelled[["merged_substrate", "cazy_family"]].drop_duplicates()
    print(f"[17] source table: {len(lit):,} rows, "
          f"{df.source_substrate.nunique()} distinct substrate labels")
    print(f"[17] {g.modelled.sum()} labels merge into {modelled.merged_substrate.nunique()} "
          f"classes; {(~g.modelled).sum()} are not modelled")
    print(f"[17] after the merge: {len(pairs)} distinct (class, CAZy family) pairs")
    print(f"[17] of those families, {pairs.cazy_family.isin(vocab).sum()} are in the "
          f"model's vocabulary\n")
    for our in sorted(modelled.merged_substrate.unique()):
        srcs = sorted(modelled[modelled.merged_substrate == our].source_substrate.unique())
        n = modelled[modelled.merged_substrate == our].cazy_family.nunique()
        print(f"  {our:<16s} {n:>3d} families  <- {', '.join(srcs)}")
    dropped = sorted(g[~g.modelled].source_substrate.unique())
    print(f"\n  not modelled ({len(dropped)}): {', '.join(dropped)}")

    print(f"\n[17] wrote {p1.relative_to(ROOT)}")
    print(f"[17] wrote {p2.relative_to(ROOT)}")
    try:
        p3 = args.out / "curated_map.xlsx"
        with pd.ExcelWriter(p3) as xl:
            df.to_excel(xl, sheet_name="family_substrate_map", index=False)
            g.drop(columns=["modelled"]).to_excel(xl, sheet_name="label_merge_summary",
                                                  index=False)
        print(f"[17] wrote {p3.relative_to(ROOT)}")
    except Exception as e:      # openpyxl not installed -- the CSVs are the deliverable
        print(f"[17] (no .xlsx: {type(e).__name__})")


if __name__ == "__main__":
    main()
