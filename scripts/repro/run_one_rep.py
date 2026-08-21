#!/usr/bin/env python3
"""Run ONE reproducibility rep of the full 29-config 5×5 RSKF benchmark.

Reuses the SAME data splits (seeds 42-46 × 5 inner folds) — these are fixed by
the embedding cache. What varies per rep is the model-init seed, threaded
through ``REPRO_REP_SEED`` env var that ``src.shallow.architectures`` and
``src.deep.architectures`` honor.

Per rep, all artifacts (per-fold classifier weights, preds, meta.json) are
written under ``reproducibility/rep_<N>/predictions/``. Then 04_benchmark.py +
05_calibrate_best.py are run with that out-dir to produce the rep's own
leaderboard and final_model.

For FT configs, the wrapper-decompressed ``.npy`` sidecars are deleted between
configs to keep disk footprint at ~2.4 GB (not 60 GB).

Usage:
    REPRO_REP_SEED=1000 python scripts/repro/run_one_rep.py --rep 1
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
REPRO = ROOT / "reproducibility"

# 29 configs split by FT flavor they need (for ordered disk-frugal training)
SHALLOW_FT_CBOW = ["ftCbow_MM__ET500_sqrt", "ftCbow__BRF100"]      # uses cbow_shallow
SHALLOW_FT_SG   = ["ftSg__BRF100"]                                  # uses sg_shallow
SHALLOW_OTHER   = ["cpu__ET500_log2", "cv__BRF100",
                   "w2vCbow__BRF100", "w2vSg__BRF100",
                   "d2vDm__BRF100", "d2vDbow__BRF100"]              # no FT
DL_FT_CBOW = ["ftCbow__LSTM", "ftCbow__LSTMattn",
              "ftCbow__JustAttn", "ftCbow__Trans"]                  # uses cbow_dl
DL_FT_SG   = ["ftSg__LSTM",   "ftSg__LSTMattn",
              "ftSg__JustAttn",  "ftSg__Trans"]                     # uses sg_dl
DL_OTHER   = ["w2vCbow__LSTM", "w2vCbow__LSTMattn",
              "w2vCbow__JustAttn", "w2vCbow__Trans",
              "w2vSg__LSTM",   "w2vSg__LSTMattn",
              "w2vSg__JustAttn",  "w2vSg__Trans"]                   # no FT
# Doc2Vec DL configs were dropped in May 2026: a sequence model has no sequence
# to run over when the embedding emits one vector per document. Doc2Vec is still
# benchmarked in the shallow configs, where a document vector is the right input.


def _run(cmd: list[str], log_path: Path):
    """Run a subprocess, streaming stdout/stderr to a log file."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    with log_path.open("ab") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        print(f"  ⚠ command exited with {rc} — see {log_path}", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep",      type=int, required=True, help="rep id (1..5)")
    ap.add_argument("--rep-seed", type=int, default=None,  help="model-init seed (default 1000*rep)")
    ap.add_argument("--skip-shallow", action="store_true")
    ap.add_argument("--skip-dl",      action="store_true")
    args = ap.parse_args()

    rep_id   = args.rep
    rep_seed = args.rep_seed if args.rep_seed is not None else 1000 * rep_id

    out_dir = REPRO / f"rep_{rep_id}"
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "seed.txt").write_text(f"{rep_seed}\n")

    env = os.environ.copy()
    env["REPRO_REP_SEED"] = str(rep_seed)
    py = str(ROOT / ".venv/bin/python")
    if not Path(py).exists():
        py = "python3"
    log_root = out_dir / "logs"
    print(f"\n=== rep_{rep_id}  REPRO_REP_SEED={rep_seed} ===\n", flush=True)
    t_rep0 = time.time()

    # -------- shallow phases --------
    if not args.skip_shallow:
        for label, configs in [
            ("shallow_other",   SHALLOW_OTHER),
            ("shallow_ftCbow",  SHALLOW_FT_CBOW),
            ("shallow_ftSg",    SHALLOW_FT_SG),
        ]:
            t0 = time.time()
            print(f"\n--- rep_{rep_id} phase: {label} ({len(configs)} configs)", flush=True)
            subprocess.run([py, str(ROOT/"scripts/02_train_shallow.py"),
                             "--retrain", "--only", *configs,
                             "--out-dir", str(out_dir/"predictions")],
                            env=env, check=False,
                            stdout=open(log_root/f"{label}.log", "ab"),
                            stderr=subprocess.STDOUT)
            print(f"--- {label} done ({time.time()-t0:.0f}s)", flush=True)

    # -------- DL phases --------
    if not args.skip_dl:
        for label, configs in [
            ("dl_other",  DL_OTHER),
            ("dl_ftCbow", DL_FT_CBOW),
            ("dl_ftSg",   DL_FT_SG),
        ]:
            t0 = time.time()
            print(f"\n--- rep_{rep_id} phase: {label} ({len(configs)} configs)", flush=True)
            subprocess.run([py, str(ROOT/"scripts/03_train_deep.py"),
                             "--retrain", "--only", *configs,
                             "--emb-dir", str(ROOT/"artifacts/embeddings"),
                             "--out-dir",   str(out_dir/"predictions")],
                            env=env, check=False,
                            stdout=open(log_root/f"{label}.log", "ab"),
                            stderr=subprocess.STDOUT)
            print(f"--- {label} done ({time.time()-t0:.0f}s)", flush=True)

    # -------- benchmark + calibration on this rep's artifacts --------
    print(f"\n--- rep_{rep_id} 04_benchmark", flush=True)
    subprocess.run([py, str(ROOT/"scripts/04_benchmark.py"),
                     "--predictions-dir", str(out_dir/"predictions"),
                     "--out",             str(out_dir/"leaderboard.csv")],
                    env=env, check=False,
                    stdout=open(log_root/"bench.log", "wb"),
                    stderr=subprocess.STDOUT)

    print(f"\n--- rep_{rep_id} 05_calibrate", flush=True)
    subprocess.run([py, str(ROOT/"scripts/05_calibrate_best.py"),
                     "--predictions-dir", str(out_dir/"predictions"),
                     "--out",             str(out_dir/"final_model.pkl"),
                     "--out-csv",         str(out_dir/"calibration_report.csv")],
                    env=env, check=False,
                    stdout=open(log_root/"calibrate.log", "wb"),
                    stderr=subprocess.STDOUT)

    json.dump({
        "rep_id": rep_id,
        "rep_seed": rep_seed,
        "wall_sec": time.time() - t_rep0,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, open(out_dir/"rep_meta.json","w"), indent=2)
    print(f"\n=== rep_{rep_id} done in {(time.time()-t_rep0)/60:.1f} min ===\n", flush=True)


if __name__ == "__main__":
    main()
