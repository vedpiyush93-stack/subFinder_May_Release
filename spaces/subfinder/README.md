---
title: subFinder
emoji: 🧬
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
hardware: zero-a10g
# scikit-learn 1.8.0 -- the version this model was pickled with -- requires
# Python >=3.11, and the Space image defaults to 3.10. ZeroGPU supports 3.12.12.
python_version: 3.12.12
pinned: false
license: apache-2.0
short_description: Predict what polysaccharide a PUL breaks down
---

# subFinder

Give it the gene labels of a polysaccharide utilization locus (PUL) and it returns the
substrate that locus most likely acts on, together with what you need in order to act on
the prediction: a calibrated probability, a p-value saying whether the call beats chance,
and the genes that produced it — for all twelve substrates, not only the winner.

## Input

**Typed or pasted.** One locus per line, gene labels separated by commas, a pipe between
alternative annotations of the same gene. An optional name before a tab is carried through
to the results.

```
alginate_demo	1.B.14.12.1,GntR,PL6|PL6_1,PL17_2|PL17,2.A.1.14.25,null
```

**Uploaded.** A dbCAN `cgc_standard.out` table, read exactly as the command-line tool reads
it, or a plain text file in the format above. Up to 2,000 loci per run.

## Output

One row per locus × substrate, so a locus gives twelve rows.

| column | meaning |
|---|---|
| `probability` | calibrated; held-out loci called at 0.9 or above are correct 98.5% of the time |
| `p_value` | binomial test of that substrate's vote count against a coin toss, Bonferroni-corrected over the twelve |
| `significant` | the p-value clears 0.05/12 **and** at least two genes in the locus are ones the model recognises |
| `votes` | yes-votes out of 500 in that substrate's forest |
| `signature_genes` | genes whose removal moves that substrate's probability most |
| `documented` | those of them the curated literature table lists for that substrate |

Everything shown is downloadable as CSV.

## On-screen results

A summary strip, a card per locus (colour-coded by how much to trust the call), then
every row in one table you can **sort by any column**, filter by text, or narrow to
rows that beat chance. Sorting happens in the browser, so it never re-runs the model.
Sort on **documented** to surface the calls with the strongest independent literature
support.

Long runs report progress by stage — loading the model, predicting substrates, computing
p-values, finding signature genes — with a percentage, so a large file never looks stalled.

## Downloads

**CSV** for downstream analysis: one row per locus × substrate, full precision. **HTML** is
the page above saved to a file; sorting and filtering still work offline.

## Controls

- **Signature genes per substrate** (1–5, default 3).
- **Explain substrates above this probability** (default 0.05). Below it the probability and
  p-value are still reported but the genes are not: gene attributions for a substrate that
  was never in contention are unstable and easy to over-read.
- **Explain all twelve regardless** — off by default, available if you want it.

## What it will not do

A call is withheld when fewer than two genes in the locus are ones the model recognises.
The p-value asks whether a substrate's panel out-voted a coin toss; it does not ask whether
the winner beats the runner-up, so a locus split 0.52 against 0.48 is both clearly
informative and entirely undecided. Read the top two rows when they are close.

## Model

`cpuV2__ET500_log2`: bag-of-token counts over gene labels, one-vs-rest ensembles of 500
extremely randomized trees, temperature-calibrated. Held-out accuracy 0.9177 ± 0.0146 over
5 repeats of 5-fold cross-validation on 1,030 experimentally characterised loci.
