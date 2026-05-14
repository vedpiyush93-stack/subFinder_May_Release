#!/usr/bin/env python3
"""Run inference on one or many new PULs using the deployed calibrated model.

Input formats supported:
  --seq "GH13,CBM6|null"                     a single PUL token-string
  --in-csv path/to/cgcs.csv  --col sig_gene_seq  a CSV with one PUL per row

For each PUL, returns: predicted substrate, calibrated probabilities for all 12
classes, Dirichlet-uniform p-values, significance flag (p<0.05), top-5 signature
genes via leave-one-token-out ablation against the predicted class (computed on
the CALIBRATED probabilities — same protocol as the slides and paper §3.7).

If the PUL has >10% out-of-vocabulary tokens, the predictor sets
``refuse_to_predict=True`` purely as a "needs manual review" caveat — the
inference itself runs identically (same substrate, same calibrated
probabilities, same p-values, same signature genes). The flag and the
``oov_proportion`` field are just two extra columns to help downstream
tooling decide how much to trust the result.

Examples:
    python scripts/06_inference.py --seq "GH13,CBM6|null"  --pretty
    python scripts/06_inference.py --in-csv data/new_puls.csv --col sig_gene_seq --out predictions.csv
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.inference import load_predictor


def _json_default(o):
    """Make numpy scalars / arrays JSON-serializable."""
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_):    return bool(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seq",     help="Inference on a single PUL token-string.")
    g.add_argument("--in-csv",  help="Inference on a CSV column of PUL token-strings.")
    ap.add_argument("--col",   default="sig_gene_seq", help="CSV column with PUL sequences.")
    ap.add_argument("--model", default=str(ROOT/"artifacts/final_model.pkl"))
    ap.add_argument("--lit",   default=str(ROOT/"data/Literature_Data_fam_substrate_mapping.tsv"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out",   default=None)
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout.")
    args = ap.parse_args()

    pred = load_predictor(args.model, args.lit)
    if args.seq:
        out = pred.predict(args.seq, top_k=args.top_k)
        if args.pretty: print(json.dumps(out, indent=2, default=_json_default))
        else: print(json.dumps(out, default=_json_default))
        if args.out:
            with open(args.out, "w") as f: json.dump(out, f, indent=2, default=_json_default)
        return

    df = pd.read_csv(args.in_csv)
    if args.col not in df.columns:
        sys.exit(f"[06-inference] column {args.col!r} not in {args.in_csv}; columns: {list(df.columns)}")
    rows = []
    for i, seq in enumerate(df[args.col].astype(str).values):
        r = pred.predict(seq, top_k=args.top_k)
        rows.append({
            "row":             i,
            "predicted":       r["predicted"],
            "confidence":      r["confidence"],
            "is_significant":  r["is_significant"],
            "oov_proportion":  r["oov_proportion"],
            "refuse_to_predict": r["refuse_to_predict"],
            "signature_genes": ";".join(f"{g['token']}:{g['delta']:+.4f}{'*' if g.get('is_lit_canonical') else ''}"
                                          for g in r["signature_genes"] if "token" in g),
        })
    out_df = pd.DataFrame(rows)
    if args.out: out_df.to_csv(args.out, index=False); print(f"[06-inference] wrote {args.out}")
    else: print(out_df.to_string(index=False))


if __name__ == "__main__": main()
