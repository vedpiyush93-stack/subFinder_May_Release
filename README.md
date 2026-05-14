<div align="center">

# subFinder — calibrated classical-ML for PUL substrate prediction

**Predict the polysaccharide substrate of a bacterial PUL from its gene-token sequence.**

<sub>One classical-ML pipeline, twenty deep baselines, leak-free 5×5 RSKF benchmark, calibrated deployment.</sub>

<br>

[![Paper PDF](https://img.shields.io/badge/paper-PDF-1a3a5c?style=for-the-badge)](paper/main.pdf)
[![Supplement](https://img.shields.io/badge/supplement-PDF-1a3a5c?style=for-the-badge)](paper/supplement.pdf)
[![Interactive Deck](https://img.shields.io/badge/interactive%20deck-HTML-27ae60?style=for-the-badge)](docs/deck.html)
[![Static Deck](https://img.shields.io/badge/static%20deck-PPTX-7f8c8d?style=for-the-badge)](presentations/deck.pptx)

<br>

**Headline:** `cpu__ET500_log2` (CountVec_cpu × OvR ExtraTrees-500) reaches **0.9058 ± 0.0172** mean test accuracy under the full 5-repeat × 5-fold RSKF protocol (n = 25 trials), beating the published Balanced Random Forest baseline by **+6.16 pp** (paired t = 15.50, p ≈ 5 × 10⁻¹⁴) and the strongest published deep architecture by **+8.10 pp**.

</div>

---

## TL;DR — what you'll find here

| What | Where | Status |
|---|---|---|
| Best-model weights for inference | [`artifacts/final_model.pkl`](artifacts/final_model.pkl) | shipped (~180 MB) |
| Per-trial classifier weights, all 29 configs × 25 trials | [`artifacts/predictions/`](artifacts/predictions/) | shipped (~8.5 GB) |
| Per-fold word embeddings (255 GB) | **regenerate locally** via `scripts/01_train_embeddings.py --retrain` (only needed for non-top-model configs) | see [§ 5 Tier B](#tier-b--embedding-cache-you-regenerate-locally-if-you-need-it) |
| Compiled paper + supplement | [`paper/main.pdf`](paper/main.pdf), [`paper/supplement.pdf`](paper/supplement.pdf) | shipped |
| Interactive deck (Plotly) | [`docs/deck.html`](docs/deck.html) | shipped |
| Static deck (PowerPoint) | [`presentations/deck.pptx`](presentations/deck.pptx) | shipped |
| Audit trail — every numeric claim | [`paper/audit_output.txt`](paper/audit_output.txt) | shipped |

```bash
# Quick start — predict the substrate of one PUL in ~5 seconds
python3 scripts/06_inference.py --seq "GH13,CBM6|PfkB,GH97_4,null" --pretty
```

---

## 1. Repository layout

```
subFinder_May_Release/
│
├── README.md                ← you are here (GitHub-Pages-ready)
├── requirements.txt
├── data/
│   ├── Train_data.csv                              ← 1,030 labeled PULs
│   └── Literature_Data_fam_substrate_mapping.tsv   ← curated CAZy ↔ substrate DB (753 entries)
│
├── src/                     ← clean, modular, no external paths
│   ├── preprocessing/        tokenizers + featurizers (sparse + embedding-based)
│   ├── embeddings/           6 gensim architectures (FastText, Word2Vec, Doc2Vec × 2 variants each)
│   ├── shallow/              3 shallow classifier architectures (ExtraTrees ×2, Balanced RF)
│   ├── deep/                 4 paper-verbatim deep architectures (LSTM, LSTM+attn, attention-only, transformer)
│   ├── calibration/          temperature scaling + inner-CV leak-free fit
│   ├── ablation/             leave-one-token-out Δ-prob signature-gene attribution
│   ├── lit_validation/       curated alias map (75 lit → 12 model) + canonical-set builders
│   ├── inference/            PULPredictor wrapper for new-PUL inference
│   └── splits.py             5×5 RSKF splitter
│
├── scripts/                 ← nine CLI drivers, each with --reuse / --retrain flags
│   ├── 01_train_embeddings.py        train (or reuse) per-fold word embeddings
│   ├── 02_train_shallow.py           train (or reuse) shallow classifiers
│   ├── 03_train_deep.py              train (or reuse) deep classifiers
│   ├── 04_benchmark.py               aggregate 29-config 5×5 RSKF leaderboard
│   ├── 05_calibrate_best.py          fit temperature scaling on top model (leak-free)
│   ├── 06_inference.py               new-PUL inference using the deployed calibrated model
│   ├── 07_build_paper_artifacts.py   regenerate every paper table/figure/audit number
│   ├── 08_build_static_deck.py       regenerate static .pptx deck (21 slides)
│   ├── 09_build_interactive_deck.py  regenerate interactive Plotly HTML deck (21 slides)
│   └── make_release_tarball.sh       bundle heavy artifacts into 3 tarballs for distribution
│
├── notebooks/               ← the master "paper-artifact feeder" notebook
│   └── build_paper_artifacts.ipynb   one-shot notebook (38 cells) that regenerates every CSV
│                                     under paper/tables/, every audit line in paper/audit_output.txt,
│                                     and key figures. See §6 below for the cell map.
│
├── artifacts/               ← outputs of the training scripts (mostly shipped pre-computed)
│   ├── predictions/         29 configs × 25 trials of (classifier.joblib|keras + probs_*.npz + meta.json)
│   │                          shipped:    probs_*.npz + meta.json for all 29 configs × 25 trials (~30 MB)
│   │                          shipped:    classifier.joblib for cpu__ET500_log2 SEED-42 ONLY (5 folds, ~220 MB)
│   │                          NOT shipped: classifier.joblib for the other 4 seeds of cpu__ET500_log2 (20 files)
│   │                          NOT shipped: classifier.joblib/keras for the other 28 configs (all seeds)
│   │                                       (see §5 for the `subfinder_classifiers.tar.gz` download)
│   ├── calibration/         per-fold temperature-scaling outputs (npz)
│   ├── ablation/            leave-one-token-out Δ-prob CSVs (argmax-class and TRUE-class, raw and calibrated)
│   ├── embeddings_cache/    [NOT shipped] 255 GB of per-fold gensim word-embedding models
│   │                                    Regenerate locally via scripts/01_train_embeddings.py --retrain
│   │                                    Only needed for the 7 BRF+embedding configs and the 20 DL configs
│   │                                    Top model (cpu__ET500_log2) uses NO embeddings — see §5 Tier B
│   ├── final_model.pkl      deployed cpu__ET500_log2 pipeline + temperature T (used by 06_inference.py)
│   │                        ~180 MB; NOT shipped in git; regenerated by `python scripts/05_calibrate_best.py`
│   ├── leaderboard.csv      29-row sorted leaderboard
│   ├── calibration_report.csv  4-method calibration comparison
│   └── per_fold_metrics.csv full 725-row per-trial metrics CSV
│
├── paper/                   compiled paper + every table that feeds it
│   ├── main.pdf, supplement.pdf, audit_output.txt
│   └── tables/              12 CSV tables produced by notebooks/build_paper_artifacts.ipynb
│
├── presentations/           static .pptx deck (also regenerable via scripts/08)
├── docs/                    GitHub-Pages content
│   ├── deck.html            interactive Plotly deck (regenerable via scripts/09)
│   ├── figures/             PNG figures (from notebook + decks)
│   └── tables/              CSV tables (subset that needs to be web-accessible)
│
└── tests/                   leakage assertions + smoke tests (run via pytest)
```

### What's in git vs. what you download separately

| Artifact | Size | In git? | Where to get it if not |
|---|---|---|---|
| All source modules (`src/`, `scripts/`, `notebooks/`) | ~250 KB | ✅ yes | — |
| Data (1,030 PULs + lit DB) | ~160 KB | ✅ yes | — |
| Paper PDFs + supplement + audit | ~600 KB | ✅ yes | — |
| `presentations/deck.pptx` + `docs/deck.html` + figures | ~3 MB | ✅ yes | — |
| `artifacts/predictions/*/r*_f*/probs_*.npz + meta.json` | ~30 MB | ✅ yes | — |
| `artifacts/predictions/cpu__ET500_log2/r42_f*/classifier.joblib` (5 files × 45 MB) | ~220 MB | ⚠️ yes-with-LFS, otherwise excluded | `subfinder_classifiers.tar.gz` (release) |
| `artifacts/final_model.pkl` | 180 MB | ❌ **no** (>100 MB hard limit) | rerun `python scripts/05_calibrate_best.py` |
| `artifacts/predictions/{28 other configs + 4 other seeds of top model}/.../classifier.{joblib,keras}` | ~7 GB | ❌ no | `subfinder_classifiers.tar.gz` (release) |
| `artifacts/embeddings_cache/` (255 GB raw) | huge | ❌ no | regenerate locally via `scripts/01_train_embeddings.py --retrain` (~6 h). NOT needed for the top-model reproduction. See §5 Tier B. |

**GitHub file-size sanity (per `find . -size +50M`):**
- 1 file > 100 MB (`artifacts/final_model.pkl`) — will be rejected; **don't commit**, regenerate locally.
- 0 files in the 50–100 MB warning band.
- 25 files in the 40–50 MB band (the `classifier.joblib` for `cpu__ET500_log2`) — these are fine per-file but bring the in-git repo to ~700 MB, which exceeds GitHub's 1 GB **soft** cap. Three options:
  - **Default**: commit them anyway. GitHub will warn but accept.
  - **git-lfs**: `git lfs track "artifacts/predictions/cpu__ET500_log2/r*_f*/classifier.joblib"` — bumps the per-file cap to 2 GB.
  - **Exclude them**: uncomment the relevant line in `.gitignore`, distribute via the release tarball.

The `.gitignore` is set up so that nothing > 100 MB is ever tracked.

---

## 2. Five-minute quickstart

### 2a. Predict the substrate of a new PUL

```bash
# Single PUL
python3 scripts/06_inference.py --seq "GH13,CBM6|PfkB,GH97_4,null" --pretty

# Many PULs from a CSV
python3 scripts/06_inference.py --in-csv data/new_puls.csv --col sig_gene_seq --out predictions.csv
```

Each prediction returns:
- **predicted substrate** (argmax)
- **calibrated probability vector** over all 12 classes
- **Dirichlet-uniform p-value** per class (significant ⇔ p < 0.05)
- **top-5 signature genes** by leave-one-token-out Δ-prob, with `is_lit_canonical` flag
- **OOV proportion** + `refuse_to_predict` flag (true when > 10 % of tokens are unknown to the train vocab)

### 2b. Reproduce the paper from cached artifacts (≈ 1 min wall time)

```bash
python3 scripts/04_benchmark.py            # 29-config leaderboard
python3 scripts/05_calibrate_best.py        # leak-free calibration, four-method comparison
python3 scripts/07_build_paper_artifacts.py # every table + audit_output.txt
```

### 2c. Retrain everything from scratch (≈ 12 h on M4 Max)

```bash
python3 scripts/01_train_embeddings.py --retrain  # 6 × 25 = 150 word-embedding models (~6 h)
python3 scripts/02_train_shallow.py    --retrain  # 9 × 25 = 225 shallow classifier fits (~30 min)
python3 scripts/03_train_deep.py       --retrain  # 20 × 25 = 500 deep classifier fits (~6 h)
python3 scripts/04_benchmark.py
python3 scripts/05_calibrate_best.py
python3 scripts/07_build_paper_artifacts.py
```

---

## 3. The model — what the top config actually is

The winning configuration is the only one in the >0.90 mean-accuracy band of the 29-config 5×5 RSKF benchmark.

| Shorthand          | `cpu__ET500_log2`                                              |
|--------------------|----------------------------------------------------------------|
| **Featurizer**     | `CountVectorizer` with `tok_cpu` tokenizer (splits on `,`, `\|`, `_`) |
| **Vocabulary**     | ~488 tokens per fold (fit on outer-train only)                 |
| **Classifier**     | `OneVsRestClassifier(ExtraTreesClassifier(n=500, max_features='log2', class_weight='balanced', bootstrap=False))` |
| **Mean accuracy**  | **0.9058 ± 0.0172** (n=25 trials)                              |
| **Macro F1**       | 0.892                                                          |
| **Classwise acc**  | 0.882                                                          |
| **Training time**  | ≈ 3 s/fold (≈ 75 s for a complete 25-fold config)             |
| **Calibration**    | per-class binary logit / T / sigmoid / renormalize, T ≈ 0.70 |

The win is **almost entirely from the classifier swap** (BRF → OvR ExtraTrees) — the tokenizer/featurizer changes contribute only ~1-2 pp on top. We trace the gain to two design choices in `BalancedRandomForestClassifier` that hurt on a small, moderately-imbalanced 12-class dataset: (i) bootstrap-balanced sampling discards majority-class signal per tree; (ii) the 100-tree ensemble is too small to recover the variance.

---

## 4. Calibration — why temperature scaling

The deployed model is **temperature-calibrated**. Three calibration methods were compared **leak-free** on the held-out 5-fold outer folds (script `05_calibrate_best.py`):

| Method                                | OOF Accuracy | ECE (10-bin) | Argmax preserved? |
|---------------------------------------|--------------|--------------|--------------------|
| Uncalibrated (raw OvR-ExtraTrees)     | 0.9029       | 0.094         | —                 |
| **Temperature scaling (T ≈ 0.70)**    | **0.9029**   | **0.029**     | **yes** (monotonic per-class) |
| Isotonic CV5 (`CalibratedClassifierCV`) | 0.8903       | 0.040         | no — can re-rank classes |
| Sigmoid CV5 (`CalibratedClassifierCV`) | 0.9019       | 0.153         | no                |

**Why temperature scaling is the right deployment choice:**
1. It's monotonic per-class (`logit / T / sigmoid` is order-preserving for fixed class), so the argmax — and therefore the headline accuracy — is mathematically guaranteed unchanged.
2. It halves the 10-bin ECE (0.094 → 0.029) with one scalar parameter, which is robust.
3. The fit is leak-free: inside each outer-train fold, we do an inner 5-fold CV, accumulate inner-OOF probabilities, and minimize multi-class NLL of those probabilities → T. **No outer-test row is ever used to fit T**, asserted in the script:

```python
assert len(set(te) & set(tr_outer)) == 0, "outer test ∩ outer train must be empty"
```

The mean T across the 5 outer folds is ~0.70 — *less than one*. This corresponds to **sharpening** the model's output (the opposite of the canonical neural-network overconfidence regime) because OvR(ExtraTrees) outputs are diffuse rather than peaked.

**Why isotonic is reported but not deployed:** it can reach a lower ECE on some distributions, but it's fit per-OvR-binary-classifier and is allowed to re-rank the 12 classes — costing ~1.3 pp accuracy on this dataset.

---

## 5. Heavy artifacts — what we ship vs. what you regenerate

We split artifacts into two tiers based on size and how often a reviewer actually needs them:

- **Tier A (shipped as 2 Google Drive zips):** the deployable top model + the per-trial weights for all 29 configs. Tiny relative to the heavy stuff, and these are what the paper-claim numbers come from.
- **Tier B (regenerate locally, NOT shipped):** the per-fold gensim word-embedding cache. ~255 GB raw, can't compress meaningfully, and only matters if you want to retrain the 7 embedding-based shallow configs or the 20 DL configs. **The top model itself uses NO embeddings**, so the headline results are fully reproducible without ever generating the cache.

### Tier A — Google Drive zips (download → unzip)

Both zips live in a single Google Drive folder:

**📦 [subFinder release artifacts (Google Drive)](https://drive.google.com/drive/folders/1UkVjswMtFwk5AE-VBeRFMJA7Wn56p39P?usp=sharing)**

Download the two `.zip` files from that folder into the repo root, then unzip them as described below.

#### Zip 1 — `subfinder_final_model.zip` (~48 MB)

**Contents:** the 180 MB deployable `final_model.pkl` (calibrated `cpu__ET500_log2` pipeline trained on all 1,030 rows) + the per-fold temperature-scaling `.npz` files.

**Needed for:** `scripts/06_inference.py` (any inference call), `scripts/07_build_paper_artifacts.py` (calibration audit lines).

```bash
cd /path/to/subFinder_May_Release
# After downloading subfinder_final_model.zip from the Drive folder above:
unzip -q subfinder_final_model.zip                # extracts into ./artifacts/
rm subfinder_final_model.zip
python3 scripts/06_inference.py --seq "GH13,CBM6|null" --pretty    # verify
```

#### Zip 2 — `subfinder_classifier_weights.zip` (~7.7 GB)

**Contents:** the per-trial classifier weights — `classifier.joblib` for the 9 shallow configs and `classifier.keras` for the 20 DL configs, all 25 trials each (725 fits total).

**Needed for:** any cross-fold ablation, the leakage-audit test, comparing rep_1 vs. a fresh retrain. *Most reviewers won't need this* — the lightweight `probs_*.npz + meta.json` shipped in git is sufficient for `04_benchmark.py` and the paper-artifact pipeline.

```bash
cd /path/to/subFinder_May_Release
# After downloading subfinder_classifier_weights.zip from the Drive folder above:
unzip -q subfinder_classifier_weights.zip         # extracts into ./artifacts/predictions/
rm subfinder_classifier_weights.zip
```

#### Checksums

```text
subfinder_final_model.zip         48 MB     SHA-256: 654a5ad1e2c613f0df96f365e52f9ec3f38039ff7dd4ca93fb5f75a5a6cf96d5
subfinder_classifier_weights.zip  7.7 GB    SHA-256: 3784f21cb4317bb61e56bbe349b7db1cea2da8e1859c5daf36a189dee1aabc16
```

Verify post-download with:
```bash
shasum -a 256 subfinder_final_model.zip       # must match the SHA above
shasum -a 256 subfinder_classifier_weights.zip
```

### Tier B — Embedding cache (you regenerate locally if you need it)

**Why we don't ship it:** the per-fold gensim cache is ~255 GB raw. Float-array models compress only ~5% (10 GB → 9.4 GB tested), so a zip would still be ~240 GB — too big to host conveniently, and most reviewers don't need it.

**When you DO need it:** if you want to retrain any of these:
- the 7 `*__BRF100` shallow configs (paper baseline + 6 embedding-featurizer variants)
- the 20 DL configs (4 architectures × 5 embeddings)
- the `ftCbow_MM__ET500_sqrt` (our second shallow, uses FastText mean+max)

**When you DO NOT need it:** if you only care about the top model (`cpu__ET500_log2`). That config is CountVectorizer-only — no embeddings anywhere in its training path. **Everything in §2a "predict a new PUL" and §2b "reproduce the paper from cached artifacts" works without touching the cache.**

#### How to regenerate the cache (~6 h on M4 Max)

```bash
python3 scripts/01_train_embeddings.py --retrain
```

This script:
1. Walks the 25 outer (seed, fold) splits of 5-repeat × 5-fold RSKF (`seeds ∈ {42,43,44,45,46}`).
2. For each split, builds two training corpora:
   - **Shallow corpus** = `unsupervised ∪ outer_train` (the 824 outer-train rows + any unsupervised data you've supplied).
   - **DL corpus**      = `unsupervised ∪ inner_train` (the 618 inner-train rows, excluding the 206 inner-val rows so EarlyStopping is clean).
3. Trains all 6 gensim architectures (FastText CBOW, FastText skip-gram, Word2Vec CBOW, Word2Vec skip-gram, Doc2Vec DM, Doc2Vec DBOW) on each corpus.
4. Saves per-fold to `artifacts/embeddings_cache/r<seed>_f<fold>/`:

   | File | Contents |
   |---|---|
   | `splits.npz` | `outer_tr`, `te`, `tr_inner`, `val` index arrays — single source of truth for the leak audit |
   | `<arch>_shallow.npz` | token vectors trained on `unsupervised ∪ outer_train`; carries `train_indices` (the labeled-row indices used) |
   | `<arch>_dl.npz` | token vectors trained on `unsupervised ∪ inner_train`; carries `train_indices` (excludes val for EarlyStopping) |
   | `<arch>_<shallow|dl>_model/` | raw gensim model pickle (so you can re-extract or re-train incrementally) |

**Why two embedding flavors per fold (`_shallow` vs. `_dl`):** the shallow configs see all 824 outer-train rows at fit time; the DL configs reserve 206 for validation. Re-fitting one embedding for both regimes would leak val rows into the DL training corpus.

#### How downstream scripts pick up from the cache

```
01_train_embeddings  ──>  artifacts/embeddings_cache/r{seed}_f{fold}/*_shallow.npz   <─┐
                                                          *_dl.npz        <─┐         │
                                                                            │         │
02_train_shallow      ─reads─►  *_shallow.npz   ──>  9 shallow configs × 25 trials ──┘
                                                          │
                                                          └──>  artifacts/predictions/<config>/r<seed>_f<fold>/{classifier.joblib, probs_*.npz, meta.json}

03_train_deep         ─reads─►  *_dl.npz        ──>  20 DL configs × 25 trials
                                                          │
                                                          └──>  artifacts/predictions/<config>/r<seed>_f<fold>/{classifier.keras, probs_*.npz, history.json, meta.json}

04_benchmark          ─reads─►  meta.json (across all 29 × 25)  ──>  artifacts/leaderboard.csv

05_calibrate_best     ─reads─►  artifacts/predictions/cpu__ET500_log2/r42_f*/probs_*.npz  ──> fits T per outer fold
                                                                                           └──> artifacts/final_model.pkl
06_inference          ─reads─►  artifacts/final_model.pkl  ──> per-PUL prediction

07_build_paper_artifacts ─reads─►  predictions/ + calibration/  ──> paper/tables/ + paper/audit_output.txt
08_build_static_deck     ─reads─►  predictions/ + ablation/ + calibration/  ──> presentations/deck.pptx
09_build_interactive_deck─reads─►  predictions/ + ablation/ + calibration/  ──> docs/deck.html
```

So the dependency chain for someone who wants to fully retrain non-top-model configs is:

```
01_train_embeddings (--retrain, ~6h)  →  02_train_shallow (--retrain, ~30min) + 03_train_deep (--retrain, ~6h)  →  04 →  07 / 08 / 09
```

And for someone who only cares about the top model + paper claims:

```
download final_model.zip + (optionally) classifier_weights.zip  →  04 (uses precomputed metas, 5 s)  →  05 (5 min)  →  07 / 08 / 09 (~1 min each)
```

Two paths, same audit_output.

#### What guarantees leak-freedom

Every `<arch>_*.npz` carries the explicit `train_indices` array of labeled rows used for fitting. `tests/leak_audit.py` (run via `pytest tests/`) asserts:

```python
assert set(npz["train_indices"]) & set(splits["te"]) == set()
```

across every (seed, fold, architecture, regime). The assertion fires before any classifier downstream touches the cache — so leakage shows up immediately, not buried in a metric.

---

## 6. The feeder notebook — `notebooks/build_paper_artifacts.ipynb`

This is the **master end-to-end notebook** that regenerates every paper table, figure, audit number, and downstream artifact in a single executable document. The Python scripts under `scripts/07–09` are CLI-friendly entry points that call into the same logic; the notebook is the human-readable canonical source.

```bash
# Run it end-to-end without opening Jupyter (~1 min on cached artifacts):
jupyter nbconvert --to notebook --execute --inplace notebooks/build_paper_artifacts.ipynb \
    --ExecutePreprocessor.timeout=900
```

**What it produces** (every output path is relative to the repo root):

| Output file | Source cell(s) | What it contains |
|---|---|---|
| `paper/tables/table1_leaderboard.csv` | §A — Headline leaderboard | top-N 5×5 RSKF leaderboard, mean ± std |
| `paper/tables/table2_per_substrate.csv` | §B — Per-substrate breakdown | per-substrate P/R/F1 for top model |
| `paper/tables/table3_calibration.csv` | §C — Calibration | 4-method calibration metrics (uncal, T, isotonic, sigmoid) |
| `paper/tables/table4_signature_genes_per_substrate.csv` | §D — Sig genes via feature_importance | top-K most important features per substrate (raw OvR-Gini) |
| `paper/tables/table5_literature_validation.csv` | §E — Lit validation prep | alias-collapse audit of the 75 → 12 mapping |
| `paper/tables/table6_example_predictions.csv` | §F — Example PULs | 6 cherry-picked correctly-classified test PULs |
| `paper/tables/table7_sig_gene_validation_aggregate.csv` | §F — Population sig-gene val | per-PUL any-hit @K (argmax-gated) |
| `paper/tables/table8_scoped_lit_recall_per_pair.csv` | §G — Scope coverage | gene-scope recall (argmax-gated) |
| `paper/tables/table11_lit_gene_scope_coverage.csv` | §G — Per-substrate coverage | per-substrate breakdown of scope coverage |
| `paper/tables/table12_per_substrate_sig_pr.csv` | §H — Per-substrate funnel (TRUE-class, calibrated) | new: count funnels per substrate |
| `paper/tables/sig_gene_ablation_oof_outer42.csv` | §F (precomputed) | per-PUL leave-one-token-out Δ-prob ranks |
| `paper/audit_output.txt` | §G — Audit trail | every numeric claim with a stable key (40+ entries) |
| `paper/Fig/fig_*.png` | scattered | 2 paper figures (leaderboard + reliability) |

**Cell map (38 cells total, grouped into 8 logical sections):**

- **§A — Headline benchmark (cells 1–8).** Loads `artifacts/per_fold_metrics.csv`, computes 29-config leaderboard, paired t-test + Wilcoxon vs. baseline, generates Table 1 + the leaderboard figure.
- **§B — Per-substrate breakdown (cells 9–11).** Loads top-model OOF probabilities (seed 42), computes precision/recall/F1 per class.
- **§C — Calibration (cells 12–14).** Loads `artifacts/calibration/oof_outer42_best_of_both.npz` (proper inner-CV T) and the cv5-isotonic / cv5-sigmoid alternatives; emits the four-method comparison table.
- **§D — Sig genes via feature importance (cells 15–18).** Per-OvR-binary `feature_importances_` across the 5 inner folds; top-10 unfiltered per substrate + the lit-filtered top-3.
- **§E — Lit validation prep (cells 19–21).** Builds the 75 → 12 alias-collapsed canonical-CAZy map; produces Supplement S3 (alias-citation table).
- **§F — Per-PUL sig genes by ablation (cells 22–28).** Precomputes the leave-one-token-out Δ-prob CSV; aggregates into Table 7 (any-hit @K).
- **§G — Scope-coverage validation (cells 29–31).** Per-gene scope recall; per-substrate breakdown for Supplement S5.
- **§H — Audit trail (cells 32–37).** Writes `paper/audit_output.txt`. Every audit line corresponds to a `\textbf{...}` claim somewhere in the paper or supplement.

**Re-running the notebook is idempotent** — it reads from `artifacts/` (cached classifier weights + calibration) and writes to `paper/tables/` + `paper/audit_output.txt`. It does NOT train any model. If you want to retrain the underlying classifiers, run `scripts/02_train_shallow.py --retrain` (and `03_train_deep.py --retrain`) first, then re-execute the notebook.

### Pipeline ordering (most reviewers will only run the bottom three)

```
                                                                ┌──> presentations/deck.pptx
data + artifacts/predictions/ → notebooks/build_paper_artifacts ┼──> docs/deck.html
                                          │                     ├──> paper/tables/*.csv
                                          │                     └──> paper/audit_output.txt
                                          │                                │
                                          │                                ▼
                                          │                          paper/main.pdf
                                          │                          paper/supplement.pdf
                                          │                          (compiled separately via tectonic)
                                          ▼
                              scripts/07_build_paper_artifacts.py  ←  same logic, CLI-friendly subset
                              scripts/08_build_static_deck.py     ←  reads same tables, builds pptx
                              scripts/09_build_interactive_deck.py ←  reads same tables, builds html
```

So if all you want is to regenerate the deck after a code change:

```bash
python3 scripts/08_build_static_deck.py        # 25 sec
python3 scripts/09_build_interactive_deck.py   # 8 sec
```

If you want to regenerate the paper tables + audit:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/build_paper_artifacts.ipynb
# OR (subset, CLI-friendly):
python3 scripts/07_build_paper_artifacts.py
```

If you want to compile the PDFs:

```bash
cd paper && tectonic -X compile main.tex && tectonic -X compile supplement.tex
```

---

## 7. The signature-gene story — what we claim, how we validate

For every PUL we report **per-PUL signature genes**: tokens whose removal causes the largest drop in the predicted-class probability under leave-one-token-out ablation. Formally:

```
Δ_s(t)  =  P_calibrated(s | T)  −  P_calibrated(s | T \ {t})
```

where `s` is a target class and `T` is the PUL's token set. We use **the temperature-calibrated probability** so the sig genes reflect the deployed model, not the raw OvR output.

Two attribution flavors are reported throughout the paper and deck:

| Flavor          | Target class `s`                 | What it measures                              |
|-----------------|----------------------------------|------------------------------------------------|
| **argmax-class** (deployment view) | `argmax_c P(c)` | "What did the model think mattered for its prediction?" |
| **TRUE-class** (clean attribution test) | the ground-truth substrate | "Did the model attribute correctly when given the right answer?" |

### Literature validation

We validate signature genes against a curated CAZy ↔ substrate mapping at [`data/Literature_Data_fam_substrate_mapping.tsv`](data/Literature_Data_fam_substrate_mapping.tsv). The DB uses 75 fine-grained substrate names; our model output space is 12 classes. The alias-collapse map below is the single source of truth for how those 75 names roll up — every non-trivial group has a primary-literature citation.

#### Substrate alias map (with primary-literature citations)

| Our class       | Lit-name aliases absorbed                                                                          | Citations                                                                                                                                                                |
|-----------------|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `beta-glucan`   | cellulose, cellooligosaccharide, xyloglucan, beta-glycan                                            | Burton 2006 *Science* 311:1940 (β-D-glucan backbone); Eklof & Brumer 2010 *Plant Physiol* 153:456 (xyloglucan GH16)                                                       |
| `alpha-glucan`  | starch, glycogen, sucrose, raffinose, trehalose, palatinose, glucooligosaccharide                   | Stam 2006 *Protein Eng. Des. Sel.* 19:555 (GH13 clan); Janecek 2014 *Cell. Mol. Life Sci.* 71:1149 (α-glucan repertoire)                                                  |
| `arabinogalactan` | arabinogalactan protein, arabinan                                                                  | Tan 2013 *Plant Cell* 25:270 (type-II AGP); Showalter 2010 *Plant Physiol* 153:485 (arabinan side-chain)                                                                  |
| `host glycan`   | human-milk-polysaccharide, sialic-acid, fucose                                                      | Marcobal 2011 *Cell Host & Microbe* 10:507 (mucin/HMO/sialic group); Tailford 2015 *Frontiers in Genetics* 6:81 (mucin backbone)                                          |
| `chitin`        | chitosan, chitooligosaccharide                                                                      | Hartl 2012 *Appl. Microbiol. Biotechnol.* 93:533 (chitosan = deacetylated chitin); Adrangi & Faramarzi 2013 *Biotechnol. Adv.* 31:1786                                    |
| `galactan`      | alpha-galactan, beta-galactan                                                                       | CAZy DB (Lombard 2014 *NAR* 42:D490) — α/β anomericity sub-classes                                                                                                       |
| `alginate`, `pectin`, `xylan`, `alpha-mannan`, `beta-mannan`, `fructan` | (exact — no aliasing required)                                                                  | CAZy DB                                                                                                                                                                  |

After alias collapse the DB yields **394 distinct (substrate, canonical-CAZy) pairs** across our 12 classes. Of those, **173 pairs** are *in-scope*: the canonical CAZy actually appears in at least one of our 1,030 PULs. Substrates with no counterpart in our 12 (lignin, agar, fucoidan, peptidoglycan, polyphenol, etc.) are dropped — the model cannot predict them.

Multi-substrate lit rows like `"cellulose, chitin"` are split on commas/`and` so the CAZy family is credited to **every** substrate component (e.g. to both `beta-glucan` via cellulose AND `chitin` via chitin).

#### Headline validation numbers (TRUE-class, calibrated)

| Metric (K=3) | Value | What it measures |
|---|---|---|
| Per-PUL any-hit | **768/837 = 91.8 %** | Of PULs with ≥1 lit-canonical CAZy present, fraction whose top-3 by Δ_true contains a canonical |
| Gene-scope coverage | **109/173 = 63.0 %** | Of in-scope canonical CAZy families, fraction surfaced as a top-3 sig gene anywhere |

Per-substrate breakdown of both funnels lives in [`docs/tables/per_substrate_sig_funnel.csv`](docs/tables/per_substrate_sig_funnel.csv) and Supplementary Table S10 of the paper.

---

## 8. Auditing every number

`paper/audit_output.txt` is the machine-readable source of truth — every numeric claim in the paper, supplement, and deck appears on exactly one line there with a stable key:

```
top1_acc                                                0.9058
top1_acc_std                                            0.0172
top1_n                                                  25
gap_ours_vs_paper_baseline                              0.0616
gap_ours_vs_best_paper_dl                               0.0810
mean_T                                                  0.6996
lit_db_substrate_family_pairs_after_alias_collapse      394
per_sub_sig_GT_total_hit_at_K                           768
per_sub_sig_GT_total_eligible                           837
per_sub_sig_GT_pul_hit_rate                             91.8%
per_sub_sig_GT_scope_recall                             63.0%
```

The audit is rewritten end-to-end every time you run `scripts/07_build_paper_artifacts.py`.

---

## 9. The interactive deck

[![Open interactive deck](https://img.shields.io/badge/open-interactive%20deck-27ae60?style=for-the-badge)](docs/deck.html)

21 self-contained slides with Plotly charts (hover tooltips, click-legend filtering, keyboard arrow nav). On a GitHub Pages-enabled fork, the deck is live at `https://<username>.github.io/<repo>/deck.html`.

---

## 10. Citation

If you use this code or model:

```bibtex
@misc{subfinder2026,
  title  = {subFinder: Calibrated classical-ML for polysaccharide utilization locus substrate prediction},
  author = {<authors>},
  year   = {2026},
  url    = {<repo-url>}
}
```

---

<div align="center">
<sub>Built with attention to detail. PRs and reviewer comments welcome.</sub>
</div>
