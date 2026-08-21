#!/usr/bin/env python3
"""Run inference on one or many new PULs using the deployed calibrated model.

Input formats supported
-----------------------
1. ``--seq "GH13,CBM6|null"``
     A single PUL already in the trained token-string format (annotations
     comma-separated; multi-domain genes ``|``-separated within a gene).

2. ``--in-csv path/to/cgcs.csv --col sig_gene_seq``
     A CSV with one PUL per row in the same token-string format.

3. ``--cgc-standard path/to/cgc_standard.out``
     A dbCAN-style cgc_standard.out (8-column TSV: CGC#, Gene Type,
     Contig ID, Protein ID, Gene Start, Gene Stop, Direction, Protein
     Family). Use this if you ran dbCAN and want to predict substrates
     directly on the dbCAN output without manually converting.

     The parser at ``src/preprocessing/cgc_loader.py`` groups genes by
     (Contig, CGC#) and applies these transformations per gene_type:

         null         → "null"
         TF / STP     → '+' → '|'   (so the tokenizer splits domains apart)
         CAZyme/other → as-is
         TC           → as-is by default; tok_cpu_v2 reads it at the 3-level
                        family, so no pre-truncation is needed. See --tc-mode.

     Both input paths are held to produce identical predictions by
     tests/verify_input_formats_agree.py.

Output
------
For each PUL the returned record has: ``predicted`` substrate, ``confidence``
(the calibrated probability of ``predicted``), full 12-class ``probabilities``,
Dirichlet-uniform ``p_values``, ``is_significant`` flag (p<0.05), top-K
``signature_genes`` via leave-one-token-out ablation on the CALIBRATED
probabilities (same protocol as the slides and paper §3.7), plus
``oov_proportion`` and ``refuse_to_predict``.

``refuse_to_predict`` is purely an informational caveat — the inference runs
identically regardless of OOV. See README §"What 'unknown' tokens mean" for
how to interpret it.

Examples
--------
    # single PUL
    python scripts/06_inference.py --seq "GH13,CBM6|null"  --pretty

    # bulk via CSV of already-tokenized PULs
    python scripts/06_inference.py --in-csv data/new_puls.csv --col sig_gene_seq --out predictions.csv

    # bulk directly from dbCAN cgc_standard.out
    python scripts/06_inference.py --cgc-standard data/example_cgc_standard.out --out predictions.csv

    # bulk from dbCAN but use the legacy 3-part TC convention for reproducibility
    python scripts/06_inference.py --cgc-standard data/example_cgc_standard.out --tc-mode truncate --out predictions.csv
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.inference import load_predictor
from src.preprocessing.cgc_loader import cgc_to_pul_strings


def _json_default(o):
    """Make numpy scalars / arrays JSON-serializable."""
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_):    return bool(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _flatten_row(cgc_id, r):
    """Common per-row flattening for CSV output (in-csv and cgc-standard paths)."""
    return {
        "cgc_id":            cgc_id,
        "pul_string":        r.get("_pul_string", ""),
        "predicted":         r["predicted"],
        "confidence":        r["confidence"],
        "is_significant":    r["is_significant"],
        "oov_proportion":    r["oov_proportion"],
        "refuse_to_predict": r["refuse_to_predict"],
        "signature_genes":   ";".join(
            f"{g['token']}:{g['delta']:+.4f}{'*' if g.get('is_lit_canonical') else ''}"
            for g in r["signature_genes"] if "token" in g
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seq",
                    help="Inference on a single PUL token-string.")
    g.add_argument("--in-csv",
                    help="Inference on a CSV column of PUL token-strings.")
    g.add_argument("--cgc-standard",
                    help="Inference on a dbCAN cgc_standard.out (8-col TSV).")
    ap.add_argument("--col", default="sig_gene_seq",
                     help="CSV column with PUL sequences (for --in-csv only).")
    ap.add_argument("--tc-mode", default="full",
                     choices=["full", "truncate", "both"],
                     help="How to render TC numbers (for --cgc-standard only). "
                          "Default 'full' leaves them alone; the deployed tokenizer "
                          "reads them at the family level. 'both' is retained for the "
                          "older tokenizer and double-counts transporters under this one. "
                          "Default 'both' emits 5-part|3-part for max vocab overlap.")
    ap.add_argument("--model", default=str(ROOT/"artifacts/final_model.pkl"))
    ap.add_argument("--lit",   default=str(ROOT/"data/Literature_Data_fam_substrate_mapping.tsv"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out",   default=None)
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout.")
    args = ap.parse_args()

    pred = load_predictor(args.model, args.lit)

    # ---- single PUL token-string ----
    if args.seq:
        out = pred.predict(args.seq, top_k=args.top_k)
        if args.pretty: print(json.dumps(out, indent=2, default=_json_default))
        else: print(json.dumps(out, default=_json_default))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2, default=_json_default)
        return

    # ---- bulk via cgc_standard.out (dbCAN output) ----
    if args.cgc_standard:
        cgc_map = cgc_to_pul_strings(args.cgc_standard, tc_mode=args.tc_mode)
        if not cgc_map:
            sys.exit(f"[06-inference] no CGCs parsed from {args.cgc_standard}; "
                      "check the file is a valid dbCAN cgc_standard.out")
        print(f"[06-inference] parsed {len(cgc_map)} CGCs from {args.cgc_standard} "
              f"(tc_mode={args.tc_mode!r})")
        rows = []
        for cgc_id, pul_string in cgc_map.items():
            r = pred.predict(pul_string, top_k=args.top_k)
            r["_pul_string"] = pul_string
            rows.append(_flatten_row(cgc_id, r))
        out_df = pd.DataFrame(rows)
        if args.out:
            out_df.to_csv(args.out, index=False)
            print(f"[06-inference] wrote {args.out}")
        else:
            print(out_df.to_string(index=False))
        return

    # ---- bulk via CSV of token-strings ----
    df = pd.read_csv(args.in_csv)
    if args.col not in df.columns:
        sys.exit(f"[06-inference] column {args.col!r} not in {args.in_csv}; "
                  f"columns: {list(df.columns)}")
    rows = []
    for i, seq in enumerate(df[args.col].astype(str).values):
        r = pred.predict(seq, top_k=args.top_k)
        r["_pul_string"] = seq
        rows.append(_flatten_row(f"row_{i}", r))
    out_df = pd.DataFrame(rows)
    if args.out:
        out_df.to_csv(args.out, index=False)
        print(f"[06-inference] wrote {args.out}")
    else:
        print(out_df.to_string(index=False))


if __name__ == "__main__": main()
