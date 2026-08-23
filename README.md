<div align="center">

# subFinder

**Give it the gene labels of a polysaccharide utilization locus.
It returns which sugar the locus is for, how sure it is, whether that beats chance,
and which genes decided the call.**

[![Try it in the browser](https://img.shields.io/badge/try%20it-web%20app-0e5c6b?style=for-the-badge)](https://huggingface.co/spaces/vpcfc/subfinder)
[![Held-out accuracy](https://img.shields.io/badge/held--out%20accuracy-0.9177%20%C2%B1%200.0146-17635f?style=for-the-badge)](#reproduce-every-number-in-the-paper)
[![License](https://img.shields.io/badge/license-Apache%202.0-5c6a70?style=for-the-badge)](LICENSE)

</div>

---

## What it does

A PUL is a cluster of genes a bacterium uses to detect, import and break down one
polysaccharide. Tools like dbCAN tell you a PUL is *there* and list its gene labels — CAZyme
families, transporter identifiers, regulators. They do not tell you which sugar it is for.

subFinder does. Feed it the labels, get back one of **12** substrates:

```bash
python3 scripts/06_inference.py --model artifacts/final_model_v2.pkl \
  --seq "1.B.14.12.1,GntR,PL6|PL6_1,PL17_2|PL17,2.A.1.14.25,null" --pretty
```

```jsonc
{
  "predicted": "alginate",
  "confidence": 0.9957643534153193,      // calibrated — read it at face value
  "p_value_winner_adjusted": 1.06e-110,  // Bonferroni-corrected over all 12 substrates
  "is_significant": true,                // beats chance AND >= 2 genes were readable
  "n_informative_tokens": 7,
  "signature_genes": [
    { "token": "PL17", "delta": 0.0676, "is_lit_canonical": true  },
    { "token": "PL6",  "delta": 0.0180, "is_lit_canonical": true  },
    { "token": "GntR", "delta": 0.0121, "is_lit_canonical": false }
  ]
  // ...plus all 12 probabilities, all 12 p-values, vote counts and OOV fraction
}
```

Three things come back with every prediction, because a bare substrate call is not
something a user can act on:

|  |  |
|---|---|
| **a probability that means what it says** | Loci called at 0.95 or above are correct **98.5 %** of the time. Measured on held-out data, not asserted. |
| **a *p*-value** | Each substrate's 500-tree panel is tested against a coin toss, Bonferroni-corrected across the 12. Below the bar the tool reports *no call* rather than guessing. |
| **the genes that drove it** | Leave-one-token-out ablation. For **774 of the 837** loci where the question can be asked (**92.5 %**), at least one is an enzyme family the literature independently documents for that substrate. |

<sub><b>Trained on</b> 1,030 experimentally characterised loci · <b>benchmarked</b> across 26 configurations and 650 train/test runs · <b>ahead of</b> the published Balanced-Random-Forest baseline by 7.75 pp and of the best of 16 deep sequence models by 12.54 pp.</sub>

---

## Try it without installing anything

### → **[huggingface.co/spaces/vpcfc/subfinder](https://huggingface.co/spaces/vpcfc/subfinder)**

Paste loci, or upload a CSV, TSV, plain text file, or a dbCAN `cgc_standard.out` table
straight from the annotation run. Up to 2,000 loci per go; a thousand take about two
seconds. Results download as CSV or as a self-contained HTML page.

The app runs the same bundle and the same code as the CLI and returns the same numbers to
the last decimal — [enforced by a test](#the-web-app-is-this-repo), not assumed.

---

## Install

```bash
git clone https://github.com/vedpiyush93-stack/subFinder_May_Release.git
cd subFinder_May_Release
pip install -r requirements.txt
```

No Git LFS and nothing to download. The model, the labelled data, the unsupervised corpus
and every per-fold prediction are all in the clone. Prediction needs no GPU and takes
milliseconds.

## Predict

| Input | Flag | Example |
|---|---|---|
| One locus | `--seq` | `--seq "GH13,CBM48,SusC,null"` |
| Many loci | `--in-csv FILE --col sig_gene_seq` | `--in-csv my_puls.csv --out preds.csv` |
| dbCAN output | `--cgc-standard FILE` | `--cgc-standard data/example_cgc_standard.out` |

The token string and the dbCAN table give **identical** predictions, and that is enforced
rather than assumed: `pytest tests/verify_input_formats_agree.py` round-trips real loci
through the CGC reader and requires the same tokens and the same probabilities.

### Reading the output

| Field | Meaning |
|---|---|
| `predicted` | the highest-probability substrate |
| `probabilities` | all 12, calibrated, summing to 1 |
| `p_values` | one per substrate, from its forest's vote count — the binomial chance of that many trees voting yes if they were coin flips |
| `p_value_winner_adjusted` | the winner's *p*-value, Bonferroni-corrected over the 12 |
| `is_significant` | clears *p* < 0.05/12 **and** at least 2 genes were readable. For 500 trees that means at least **280** voting yes |
| `insufficient_evidence` | `True` iff fewer than 2 readable genes — the verdict is **withheld, not negative** |
| `signature_genes` | the genes whose removal drops that substrate's probability most |
| `n_informative_tokens` | how many genes the model could actually read (in vocabulary, not `null`) |
| `oov_proportion` | fraction of this locus's tokens unseen in training |

**Read significance first.** If the winner does not clear the bar the locus has no confident
call, and the top two or three should be read together — the truth is within the top two
**96.1 %** of the time and the top three **97.3 %**.

> **Tokenizer.** `tok_cpu_v2` splits on `,` `|` `_` and then drops the bare subfamily index,
> so `GH43_34` becomes `GH43`. Transporters are read at their 3-level TC family, so
> `1.B.14.6.1` and `1.B.14` are the same feature. Keep the `null`s in — how many a locus has
> is itself evidence. On the CGC path leave `--tc-mode` at its default `full`; `both` emits
> both TC forms, which the tokenizer collapses into one token and so double-counts every
> transporter.

---

## Reproduce every number in the paper

Every figure, table and number in the manuscript is generated; none is typed by hand. No
training and no GPU — it all recomputes from the prediction matrices already in the clone.

```bash
export REPRO_REP_SEED=1000                       # required; artifacts/ is rep_1

python3 scripts/04_benchmark.py                  #  5 s   leaderboard.csv, 26 configs
python3 scripts/04b_rebuild_per_fold_metrics.py  #  5 s   per_fold_metrics.csv, 650 rows
python3 scripts/05_calibrate_best.py             # 30 s   calibration_report.csv, 4 methods
python3 scripts/07_build_paper_artifacts.py      # 45 s   paper/audit_output.txt
python3 scripts/07c_build_paper_figures.py       # 40 s   12 figures + paper/generated/*.tex

cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main && cd ..

python3 scripts/07b_verify_paper_numbers.py      # must pass
```

That last line is the guarantee. `07b` re-derives each value from the artifacts and then
checks it actually appears in the rendered PDF, so a number cannot drift between the code,
the macro and the page. It currently checks **135** values across the main paper and the
supplement, and fails loudly on any that has gone stale.

`07c_build_paper_figures.py` is deterministic: re-running it reproduces all 12 figures
pixel-for-pixel and all six generated `.tex` files byte-for-byte.

To hand the manuscript to someone who edits in Word:

```bash
python3 scripts/14_paper_to_docx.py              # -> paper/docx/*.docx
```

<details>
<summary><b>Retraining, ablations, and the leakage audit</b></summary>

```bash
pytest tests/leak_audit.py -v          # splits disjoint; embeddings never see a label

export REPRO_REP_SEED=1000             # NOT optional. It defaults to 42, which silently
                                       # trains a different model — 0.9058, not 0.9066

python3 scripts/02_train_shallow.py --only cpu__ET500_log2 --retrain   #  2 min
python3 scripts/13_train_tc2_refinement.py                             # 90 s, the deployed model
python3 scripts/02_train_shallow.py --retrain                          # 20 min, all shallow
python3 scripts/03_train_deep.py    --retrain                          #  1.5 h, all 16 deep
python3 scripts/01_train_embeddings.py --retrain                       # 21 min, all six embeddings
```

The 9 shallow configurations reproduce **bit-identically** anywhere. The 16 deep ones
reproduce bit-identically on the same machine and backend; across different hardware expect
drift around 1e-3 from floating-point reduction order.

The 625 per-trial classifier weights (6.4 GB) are **not** shipped, and nothing downstream
reads them. The leaderboard, figures, tables and paper numbers all read `probs_test.npz`,
`probs_train.npz` and `meta.json`, which are tracked.

</details>

<details>
<summary><b>Why transporters are read at three levels</b></summary>

TCDB numbers are `class.subclass.family.subfamily.protein`. **Level 3 is the family** —
`1.B.14` is the Outer Membrane Receptor family, the TonB-dependent SusC-like receptors that
define a PUL. An earlier version truncated to level 2, which is merely "β-barrel porin" and
collapsed 596 distinct families into 26 tokens (`2.A` alone swallowed 106).

That variant reported far better out-of-vocabulary coverage, which looked like better
generalization but was an artifact of having almost no vocabulary left. On accuracy the two
were indistinguishable — 0.9151 ± 0.0189 at two levels against 0.9150 ± 0.0167 at three.
Given a tie, the version that preserves 596 biological families wins, and three levels is
also the format the unsupervised corpus already uses, which is what closes the vocabulary
gap in the first place.

Reading transporters this way is worth 1.11 points on its own: 0.9066 → **0.9177**, with the
deployed vocabulary shrinking from 517 tokens to 305.

</details>

---

## The web app is this repo

`spaces/subfinder/` **is** the deployed Hugging Face Space — the same directory, pushed
as-is. It is self-contained on purpose: it carries its own copy of the model bundle and of
the inference modules it needs, so the running app never imports from the research tree and
cannot be broken by a refactor here.

```
spaces/subfinder/
├── app.py                 Gradio interface
├── engine.py              prediction + batched leave-one-token-out ablation
├── render.py theme.py sortjs.py    the results view
├── final_model_v2.pkl     the deployed bundle, byte-identical to artifacts/final_model_v2.pkl
├── src/                   trimmed copy: tokenizer, temperature, ablation, CGC reader, literature table
├── examples/              downloadable example inputs in every accepted format
└── requirements.txt       pinned to the versions the model was pickled with
```

Deploy or update it with:

```bash
python3 scripts/15_deploy_space.py --repo vpcfc/subfinder
```

<details>
<summary><b>Why it is packaged this way</b></summary>

**The copies are checked, not trusted.** `pytest tests/verify_space_parity.py` loads both
bundles and compares the temperature, the class list, the tree count and the vocabulary, then
runs edge-case loci — one informative gene, none, unknown labels, a close two-way call —
through the CLI predictor and through the app engine, and requires the substrate, the
probability, the *p*-value and the significance verdict to agree to within 1e-12.

**Pinned Python and scikit-learn.** The model is pickled with scikit-learn 1.8.0, which needs
Python 3.11 or newer, and the Space image defaults to 3.10 — so the Space's own `README.md`
pins 3.12.12. An estimator unpickled under a different minor version either warns or fails,
and a silent partial load would be worse than either.

**The GPU probe.** `app.py` defines one trivial `@spaces.GPU` function that nothing calls.
ZeroGPU refuses to start a Space with no GPU-decorated function, but this model is CPU-only
scikit-learn. Decorating the real prediction would request a GPU on every click and exhaust
the daily quota for no speedup at all.

</details>

---

## Tests

```bash
pytest                                  # ~2 min
```

| Suite | Asserts |
|---|---|
| `leak_audit.py` | outer test ∩ outer train = ∅ for every split; embeddings never see a label or a fold |
| `verify_input_formats_agree.py` | token string ≡ dbCAN CGC table — same tokens, same probabilities |
| `verify_evidence_guard.py` | a call is withheld below two readable genes, and `null` never surfaces as a signature gene |
| `verify_space_parity.py` | the deployed app and the CLI return the same numbers |
| `verify_reduced_embedding_files.py` | the compacted FastText matches the full gensim model |

The format test earns its keep: a previous CGC default double-counted transporters and moved
predictions by up to 0.251 without raising anything.

---

## Repository layout

```
data/       1,030 labelled PULs · curated CAZy-substrate table · unsupervised corpus (359,763 PULs)
src/        library — preprocessing, embeddings, shallow, deep, calibration, ablation, inference
scripts/    numbered drivers, 01 through 15
artifacts/  model bundles, per-fold predictions, leaderboard, calibration, ablation deltas
spaces/     the deployed web app
tests/      the five suites above
```

> `paper/`, `presentations/` and `unravel/` are gitignored — drafts under active revision,
> not part of the released code. Everything needed to reproduce the paper's numbers is
> tracked here; the manuscript sources are shared with collaborators directly.

<details>
<summary><b>The model in one block</b></summary>

```
tokenizer    tok_cpu_v2 — split on ',' '|' '_'; drop bare subfamily indices;
             transporters at the 3-level TC family. 305 tokens, ~11.5 per locus.

features     CountVectorizer(tokenizer=tok_cpu_v2, lowercase=False)

classifier   OneVsRestClassifier(ExtraTreesClassifier(
               n_estimators=500, max_features='log2',
               class_weight='balanced', bootstrap=False))

calibration  temperature scaling on the per-class logits — one scalar, fit against ECE
             on held-out folds. T = 0.6759. ECE 0.063 -> 0.016.

evaluation   5x5 repeated stratified K-fold: 25 fits, 5,150 held-out predictions.
```

The win is almost entirely the classifier swap. Two Balanced-Random-Forest design choices
hurt on a small 12-class problem: bootstrap-balanced sampling discards majority-class signal
per tree, and 100 trees is too few to recover the variance. OvR ExtraTrees-500 with
`class_weight='balanced'` fixes both.

</details>

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The web app ships
under the same licence.

The curated CAZy-family-to-substrate mapping in
`data/Literature_Data_fam_substrate_mapping.tsv` is compiled from the primary literature
cited in the manuscript; the underlying facts belong to those publications.

## Citation

```bibtex
@misc{subfinder2026,
  title = {subFinder: predicting the substrate of a polysaccharide utilization
           locus with a calibrated, explainable model},
  year  = {2026},
  url   = {https://github.com/vedpiyush93-stack/subFinder_May_Release}
}
```
