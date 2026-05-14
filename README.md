<div align="center">

# subFinder

**Predict the polysaccharide substrate of a bacterial PUL from its gene-token sequence.**

<sub>A leak-free 5×5 RSKF benchmark of 29 model configurations. One calibrated classical-ML pipeline that beats every published deep baseline by ≥8 pp on the same data.</sub>

<br>

[![Paper PDF](https://img.shields.io/badge/paper-PDF-1a3a5c?style=for-the-badge)](paper/main.pdf)
[![Supplement](https://img.shields.io/badge/supplement-PDF-1a3a5c?style=for-the-badge)](paper/supplement.pdf)
[![Static Deck](https://img.shields.io/badge/static%20deck-PPTX-7f8c8d?style=for-the-badge)](docs/deck.pptx)
[![Drive Release](https://img.shields.io/badge/heavy%20artifacts-Drive-4285f4?style=for-the-badge)](https://drive.google.com/drive/folders/1UkVjswMtFwk5AE-VBeRFMJA7Wn56p39P?usp=sharing)

**Headline:** `cpu__ET500_log2` (CountVec_cpu × OvR ExtraTrees‑500) reaches **0.9058 ± 0.0172** mean test accuracy across 25 trials, beats the published Balanced Random Forest baseline by **+6.16 pp** (paired t = 15.50, p ≈ 5 × 10⁻¹⁴) and the strongest published deep architecture by **+8.10 pp**. The deployed model is temperature‑calibrated (T ≈ 0.70) with leak‑free inner‑CV fitting.

</div>

---

## What's included, what isn't, and what you have to do

The repo ships everything **except the heaviest files that can't fit on GitHub**. Read this once and you'll know exactly what to download (or not).

| Tier | What you have | What you can do | Disk | Time to set up |
|:--:|---|---|---:|---:|
| **0** | Just `git clone` (the repo as you see it) | Recompute the paper's leaderboard, calibration, sig-gene metrics, and audit numbers from cached prediction probs already in the repo. | 100 MB | < 1 min |
| **1** | + `subfinder_final_model.zip` (48 MB, 1 click) | Predict substrates for *your own* PULs. Same script, same model the paper deploys. | 250 MB | 5 min |
| **2** | + `subfinder_classifier_weights.zip` (7.7 GB, 1 click) | Re-do cross-fold ablations, run the leak audit, retrain the top model (`cpu__ET500_log2`) without touching embeddings. | 8 GB | 30 min |
| **3** | + regenerate the **embeddings cache locally** | Retrain *every* config including the 7 BRF+embedding shallow configs and the 20 DL configs. | 255 GB | 6–12 h on M4 Max |

> 📦 **All downloads live in one Drive folder:** [subFinder release artifacts](https://drive.google.com/drive/folders/1UkVjswMtFwk5AE-VBeRFMJA7Wn56p39P?usp=sharing)
>
> ⚠️ **The 255 GB embeddings cache is not on Drive — it's too big to host.** It only exists locally on the author's machine. You regenerate it with one command if (and only if) you reach Tier 3. The top model's accuracy claim is fully reproducible at Tier 0, so most readers stop at Tier 0 or Tier 1.

### Which tier matches you?

| You are… | Stop at | Go to |
|---|:--:|---|
| 🧪 **A practitioner** — "I have a PUL, give me a substrate prediction" | Tier 1 | [Path A](#path-a--predict-the-substrate-of-your-pul) |
| 🔍 **A reviewer** — "verify every paper number, no training" | Tier 0 | [Path B](#path-b--reproduce-every-paper-number-no-training) |
| 🔬 **A researcher (light)** — "do ablations or retrain the top model" | Tier 2 | [Path C](#path-c--retrain-or-extend) C.1–C.2 |
| 🧬 **A researcher (full)** — "retrain DL configs / try new embeddings" | Tier 3 | [Path C.3](#path-c3--retrain-the-embedding-using-configs-12-h-on-m4-max) |

Pick one and stop reading. The other paths don't matter to you.

---

## Path A — Predict the substrate of *your* PUL

**Audience:** you have a PUL gene-token sequence and want subFinder's prediction.
**Time:** ~5 minutes (mostly the 48 MB download).

```bash
# 1. Clone
git clone https://github.com/vedpiyush93-stack/subFinder_May_Release.git
cd subFinder_May_Release

# 2. Install deps (clean conda env recommended)
pip install -r requirements.txt

# 3. Download the deployed model from the Drive folder ↓
#    https://drive.google.com/drive/folders/1UkVjswMtFwk5AE-VBeRFMJA7Wn56p39P
#    Grab subfinder_final_model.zip (48 MB) and put it in the repo root.

unzip -q subfinder_final_model.zip            # extracts into ./artifacts/
rm subfinder_final_model.zip

# 4. Predict
python3 scripts/06_inference.py \
    --seq "GH13,CBM6|PfkB,GH97_4|null" \
    --pretty
```

You'll see a block like:

```
predicted substrate : alpha-glucan
calibrated probs    : { alpha-glucan: 0.612, beta-glucan: 0.084, ... }
top-5 sig genes     : GH13 (Δ=0.214), GH97_4 (Δ=0.118), CBM6 (Δ=0.067), ...
OOV proportion      : 0.00   (no tokens unknown to the train vocab)
refuse_to_predict   : False
```

**Bulk mode (many PULs):**

```bash
python3 scripts/06_inference.py --in-csv data/new_puls.csv --col sig_gene_seq --out predictions.csv
```

#### How to read the output

| Field | What it means |
|---|---|
| `predicted substrate` | argmax over the 12 classes (`alpha-glucan`, `beta-glucan`, `pectin`, `xylan`, …) |
| `calibrated probs` | full probability vector after temperature scaling — these are well-calibrated (ECE ≈ 0.03) |
| `top-5 sig genes` | the 5 tokens whose **removal** drops the predicted-class probability the most (leave-one-token-out Δ-prob) |
| `OOV proportion` | fraction of your PUL's tokens that are **not in the training vocabulary** — [jump to the explanation ↓](#unknown-tokens) |
| `refuse_to_predict` | **Purely informational — a "needs manual review" caveat.** `True` when `OOV > 0.10`. **The inference is identical regardless of this flag** — you still get the substrate, calibrated probabilities, p-values, *and* signature genes for every PUL. The flag and the `OOV proportion` are just two extra fields you can use to decide how much to trust the result. |

---

## Path B — Reproduce every paper number, no training

**Audience:** you're reviewing the paper and want to verify every numeric claim.
**Time:** ~10 minutes total. **No GPU. No model retraining. No Drive download needed for the headline numbers.**

The repo ships **2,675 lightweight prediction files** (`probs_*.npz` + `meta.json`, ~30 MB) for all **29 configs × 25 trials**. Every leaderboard / calibration / sig-gene number in the paper recomputes from these.

```bash
# 1. Clone + install
git clone https://github.com/vedpiyush93-stack/subFinder_May_Release.git
cd subFinder_May_Release
pip install -r requirements.txt

# 2. Regenerate the 29-config leaderboard from cached probs (~5 s)
python3 scripts/04_benchmark.py
#  →  artifacts/leaderboard.csv  (29 rows, sorted by mean accuracy)

# 3. Regenerate the 4-method calibration comparison (~30 s)
python3 scripts/05_calibrate_best.py
#  →  artifacts/calibration_report.csv   (uncal / T / isotonic-cv5 / sigmoid-cv5)

# 4. Regenerate every paper table + the audit_output.txt (~45 s)
python3 scripts/07_build_paper_artifacts.py
#  →  paper/tables/*.csv  (12 tables)
#  →  paper/audit_output.txt  (every numeric claim, key → value)

# 5. (optional) Run the master notebook for the slow/fancy bits ↓
jupyter nbconvert --to notebook --execute --inplace notebooks/build_paper_artifacts.ipynb
```

After step 4, every `\textbf{...}` claim in [`paper/main.pdf`](paper/main.pdf) / [`paper/supplement.pdf`](paper/supplement.pdf) corresponds to one stable key in `paper/audit_output.txt`:

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

#### Optional: also verify the deployed (calibrated) model

If you want to additionally run inference + confirm the calibration-audit lines, [download `subfinder_final_model.zip` from the Drive folder](https://drive.google.com/drive/folders/1UkVjswMtFwk5AE-VBeRFMJA7Wn56p39P?usp=sharing) (48 MB), `unzip` in the repo root, then `python3 scripts/06_inference.py --seq "GH13,CBM6|null" --pretty`.

---

## Path C — Retrain or extend

**Audience:** you want to retrain a config, run your own ablations, or modify the architecture.
**Time:** 30 min – 12 h depending on what you retrain.

Heavy artifacts ship via Drive (the GitHub 100 MB hard limit prevents shipping them in-repo):

| Drive file | Size | Needed for |
|---|---|---|
| [`subfinder_final_model.zip`](https://drive.google.com/drive/folders/1UkVjswMtFwk5AE-VBeRFMJA7Wn56p39P?usp=sharing) | 48 MB | inference (Path A), calibration audit (Path B optional) |
| [`subfinder_classifier_weights.zip`](https://drive.google.com/drive/folders/1UkVjswMtFwk5AE-VBeRFMJA7Wn56p39P?usp=sharing) | 7.7 GB | leak audit, cross-fold ablation, comparing rep_1 vs. a fresh retrain |

```bash
# Download both zips from the Drive folder into the repo root, then:
unzip -q subfinder_final_model.zip            # → artifacts/final_model.pkl + calibration/*.npz
unzip -q subfinder_classifier_weights.zip     # → artifacts/predictions/*/r*_f*/classifier.{joblib,keras}
```

Verify SHA-256 before unzipping (paranoid mode):

```text
subfinder_final_model.zip         654a5ad1e2c613f0df96f365e52f9ec3f38039ff7dd4ca93fb5f75a5a6cf96d5
subfinder_classifier_weights.zip  3784f21cb4317bb61e56bbe349b7db1cea2da8e1859c5daf36a189dee1aabc16
```

### Path C.1 — Run the leakage audit (5 s)

```bash
pytest tests/leak_audit.py -v
# asserts that for every (seed, fold) in 5×5 RSKF, outer_test ∩ outer_train = ∅
# asserts that every cached embedding's train_indices excludes outer_test
```

### Path C.2 — Retrain the top model from scratch (no embeddings needed!)

The winning config `cpu__ET500_log2` uses **only CountVectorizer features** — no word embeddings anywhere. So the whole 5×5 retrain (125 fits) finishes in ~10 minutes:

```bash
python3 scripts/02_train_shallow.py --configs cpu__ET500_log2 --retrain
python3 scripts/05_calibrate_best.py     # re-fit temperature scaling
python3 scripts/04_benchmark.py          # confirm leaderboard regenerates the same headline
```

### Path C.3 — Retrain the embedding-using configs (~12 h on M4 Max)

The 7 BRF+embedding shallow configs and the 20 DL configs need per-fold gensim word embeddings. We **don't ship** these — the raw cache is ~255 GB and barely compresses. Regenerate locally:

```bash
# (~6 h) — trains 6 architectures × 25 folds × 2 corpora (shallow vs. DL)
python3 scripts/01_train_embeddings.py --retrain

# (~30 min)
python3 scripts/02_train_shallow.py --retrain

# (~6 h)
python3 scripts/03_train_deep.py --retrain

# (~5 s)
python3 scripts/04_benchmark.py
```

**Why two embedding flavors per fold (`*_shallow.npz` and `*_dl.npz`):** shallow configs see all 824 outer-train rows; DL configs reserve 206 inner-val rows for EarlyStopping. Re-fitting one embedding for both would leak val rows into the DL training corpus. The leak audit (Path C.1) checks this.

---

<a id="unknown-tokens"></a>
## What "unknown" tokens mean (the OOV concept)

The model's vocabulary is finite. It's built by `CountVectorizer(tokenizer=tok_cpu).fit(outer_train)` — **fit per fold on outer-train only**, so it's leak-free. For seed 42 the typical fold-vocab size is ~488 tokens.

When you give the deployed model a new PUL, some of its tokens may not be in this vocabulary. Those are **OOV** ("out-of-vocabulary"). The inference pipeline **does not change** based on OOV — every PUL goes through the same path and gets the same outputs (substrate, calibrated probabilities, p-values, signature genes). The inference output simply reports two extra fields:

| Field | Meaning |
|---|---|
| `oov_proportion` | `(# OOV tokens) / (# total tokens in your PUL)` |
| `refuse_to_predict` | `True` iff `oov_proportion > 0.10` — **a "needs manual review" caveat, not a gate.** Treat the PUL like any other; just review the prediction before trusting it. |

**Why this matters:** the deployed model is robust to *some* OOV, but accuracy collapses past 10 %. We confirmed this empirically on the seed‑42 OOF test set (1,030 PULs across 5 folds):

| OOV bucket | n PULs | share of test set | mean OOV | accuracy |
|---|---:|---:|---:|---:|
| **0%**     | 920 | 89.3% | 0.0%  | **0.914** |
| 0–5%       |  20 |  1.9% | 3.4%  | **1.000** |
| 5–10%      |  33 |  3.2% | 6.8%  | **1.000** |
| 10–25%     |  39 |  3.8% | 15.1% | **0.641** |
| ≥25%       |  18 |  1.7% | 35.0% | **0.611** |

**Plain-English read on these numbers:** as the share of unknown tokens in a PUL goes up, the model gets the answer wrong more often. Across all 1,030 test PULs the trend is statistically significant (we ran the standard correlation test for a yes/no outcome against a continuous predictor; the chance of seeing a relationship this strong by accident is roughly 4 in a hundred million).

So: ~94 % of test PULs have ≤10 % OOV and the model is reliable on them (accuracy 0.91 – 1.00). Past 10 % OOV accuracy collapses to ~0.62 — which is exactly why `predict_one` raises `refuse_to_predict=True` when `oov_proportion > 0.10`. **The prediction is computed identically** — same substrate, same calibrated probabilities, same p-values, same signature genes — the flag is just an explicit caveat that downstream tooling (or a human reviewer) can use to triage which PULs deserve a closer look.

**Per-prediction OOV figure:** [`docs/figures/fig8c_oov_vs_accuracy.png`](docs/figures/fig8c_oov_vs_accuracy.png) (also rendered on slide 12c of the decks).

---

## The model in one paragraph

The winning config is `cpu__ET500_log2`:

```
features  : CountVectorizer(tokenizer=tok_cpu, lowercase=False)
            tok_cpu splits on  ','  '|'  '_'   (3 separators)
            yields ~488 tokens per fold (fit per fold on outer-train only)

classifier: OneVsRestClassifier(ExtraTreesClassifier(
              n_estimators=500, max_features='log2',
              class_weight='balanced', bootstrap=False))

calibration: temperature scaling — one scalar T per outer fold,
             fit by inner-5-fold OOF NLL minimization (leak-free).
             Mean T ≈ 0.70 (sharpens the diffuse OvR outputs).
```

The win is almost entirely from the classifier swap (BRF → OvR ExtraTrees). Two design choices in `BalancedRandomForestClassifier` hurt on a small, moderately-imbalanced 12-class dataset: (i) bootstrap-balanced sampling discards majority-class signal per tree; (ii) the 100-tree ensemble is too small to recover the variance. OvR ExtraTrees-500 with `class_weight='balanced'` fixes both.

Full per-substrate / per-fold metrics: [`paper/tables/`](paper/tables/) and Supplement Tables S2–S10.

---

## Calibration — why temperature scaling

Three calibration methods were compared **leak-free** on the held-out 5-fold outer folds (`scripts/05_calibrate_best.py`):

| Method | OOF Accuracy | ECE (10-bin) | Argmax preserved? |
|---|---|---|---|
| Uncalibrated (raw OvR-ExtraTrees) | 0.9029 | 0.094 | — |
| **Temperature scaling (T ≈ 0.70)** | **0.9029** | **0.029** | **yes** (monotonic per-class) |
| Isotonic CV5 (`CalibratedClassifierCV`) | 0.8903 | 0.040 | no — can re-rank classes |
| Sigmoid CV5 (`CalibratedClassifierCV`) | 0.9019 | 0.153 | no |

Temperature scaling halves the ECE with one scalar parameter, and `logit / T / sigmoid` is monotonic per class — so the argmax (and therefore the headline accuracy) is mathematically guaranteed unchanged. Isotonic can reach a lower ECE on some distributions, but it's allowed to re-rank the 12 classes — costing ~1.3 pp accuracy on this dataset.

The leak-free guarantee:

```python
# scripts/05_calibrate_best.py
assert len(set(outer_test) & set(outer_train)) == 0, "must be disjoint"
# T is fit on inner-OOF probs of outer_train; no outer_test row ever touches T
```

---

## Signature genes — what we claim, how we validate

For every PUL we report **per-PUL signature genes**: tokens whose removal causes the largest drop in the predicted-class probability under leave-one-token-out ablation. We use the **temperature-calibrated** probability so sig genes reflect the deployed model, not the raw OvR output.

```
Δ_s(t)  =  P_calibrated(s | T)  −  P_calibrated(s | T \ {t})
```

Two attribution flavors:

| Flavor | Target class `s` | What it measures |
|---|---|---|
| **argmax-class** (deployment view) | `argmax_c P(c)` | "What did the model think mattered for its prediction?" |
| **TRUE-class** (clean attribution test) | the ground-truth substrate | "Did the model attribute correctly when given the right answer?" |

### Literature validation

The curated DB at [`data/Literature_Data_fam_substrate_mapping.tsv`](data/Literature_Data_fam_substrate_mapping.tsv) lists 75 fine-grained substrate names; the model output space is 12 classes. The 75 → 12 alias-collapse map (see [`src/lit_validation/alias_map.py`](src/lit_validation/alias_map.py)) is the single source of truth for how those names roll up; every non-trivial group has a primary-literature citation in the supplement.

After alias collapse the DB yields **394 distinct (substrate, canonical-CAZy) pairs** across the 12 classes. Of those, **173 pairs** are *in-scope* (the canonical CAZy appears in at least one of the 1,030 PULs).

**Headline validation numbers (TRUE-class, calibrated):**

| Metric (K=3) | Value | Meaning |
|---|---|---|
| Per-PUL any-hit | **768/837 = 91.8 %** | Of PULs with ≥1 lit-canonical CAZy present, fraction whose top-3 by Δ_true contains a canonical |
| Gene-scope coverage | **109/173 = 63.0 %** | Of in-scope canonical CAZy families, fraction surfaced as a top-3 sig gene anywhere |

Per-substrate breakdown: [`paper/tables/table12_per_substrate_sig_pr.csv`](paper/tables/table12_per_substrate_sig_pr.csv) and Supplement Table S10.

---

## Repository layout

```
subFinder_May_Release/
├── README.md
├── requirements.txt
├── data/                    1,030 labeled PULs + curated CAZy ↔ substrate DB
├── src/                     library code (preprocessing, embeddings, shallow, deep, calibration, ablation, lit_validation, inference)
├── scripts/                 9 CLI drivers (01_train_embeddings → 09_build_interactive_deck)
├── notebooks/               build_paper_artifacts.ipynb (master end-to-end feeder)
├── artifacts/
│   ├── predictions/         29 configs × 25 trials × {meta.json, probs_test.npz, probs_train.npz}   [shipped, ~30 MB]
│   ├── calibration/         per-fold T + 4-method comparison CSV                                    [shipped]
│   ├── ablation/            leave-one-token-out Δ-prob (argmax + TRUE class, raw + calibrated)      [shipped]
│   ├── leaderboard.csv      29-row sorted leaderboard                                               [shipped]
│   ├── per_fold_metrics.csv full 725-row per-trial metrics CSV                                     [shipped]
│   ├── final_model.pkl      ← in Drive zip (subfinder_final_model.zip)                              [NOT in git]
│   ├── predictions/*/r*_f*/classifier.{joblib,keras}  ← in Drive zip (classifier_weights.zip)       [NOT in git]
│   └── embeddings_cache/    ← regenerate locally via scripts/01 (~255 GB, only for non-top configs) [NOT in git]
├── paper/                   compiled paper + supplement + audit_output.txt + 12 source tables
├── docs/                    deck.pptx + deck.html + figures/ + tables/
├── presentations/           build_README.md + symlinked deck.pptx (mirror of docs/)
└── tests/                   leak_audit.py (pytest)
```

---

## Decks

Two views of the same content, **21 slides each**:

- **Static PowerPoint:** [`docs/deck.pptx`](docs/deck.pptx) — clickable in the browser file viewer; download to open in Keynote/PowerPoint.
- **Interactive Plotly HTML:** [`docs/deck.html`](docs/deck.html) — Plotly charts with hover tooltips, click-legend filtering, and keyboard arrow navigation. **GitHub renders this as raw HTML inside the repo viewer.** To see the rendered deck: clone the repo and open `docs/deck.html` in your browser:

  ```bash
  git clone https://github.com/vedpiyush93-stack/subFinder_May_Release.git
  cd subFinder_May_Release
  open docs/deck.html        # macOS
  # or: xdg-open docs/deck.html (linux), start docs/deck.html (windows)
  ```

Regenerate both decks anytime from the cached artifacts:

```bash
python3 scripts/08_build_static_deck.py        # ~30 s  → docs/deck.pptx
python3 scripts/09_build_interactive_deck.py   # ~10 s  → docs/deck.html
```

---

## Citation

If you use this code or model:

```bibtex
@misc{subfinder2026,
  title  = {subFinder: Calibrated classical-ML for polysaccharide utilization locus substrate prediction},
  author = {<authors>},
  year   = {2026},
  url    = {https://github.com/vedpiyush93-stack/subFinder_May_Release}
}
```

---

<div align="center">
<sub>Built with attention to detail. Issues and PRs welcome.</sub>
</div>
