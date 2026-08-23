# Deploying this Space

This folder is the Space. It is self-contained — model, tokenizer, curated literature table
and app — and imports nothing from the research repository at run time.

## The normal path

From the repository root:

```bash
huggingface-cli login                                    # once, with a write token
python3 scripts/15_deploy_space.py --repo <user>/subfinder
```

That script re-checks parity with the command-line tool before it uploads anything, and
refuses to push if the check fails. Useful flags:

| Flag | Effect |
|---|---|
| `--sync` | re-copy the model bundle and the shared inference modules from the research tree first |
| `--dry-run` | run every check, upload nothing |
| `--make-public` | flip the Space from private to public (never happens implicitly) |

## Creating the Space the first time

```bash
huggingface-cli repo create subfinder --type space --space_sdk gradio
```

Then set the hardware to **ZeroGPU** in the Space settings and push. The metadata at the top
of `README.md` carries the rest of the configuration.

## Three things that will break the build if changed carelessly

**`python_version: 3.12.12`** in `README.md`. The model is pickled with scikit-learn 1.8.0,
which requires Python 3.11 or newer; the Space image defaults to 3.10 and the build fails on
import. ZeroGPU supports 3.12.12.

**`requirements.txt`** pins the exact versions the model was pickled with, scikit-learn most
of all. An estimator unpickled under a different minor version either warns or fails, and a
silent partial load would be worse than either. If you rebuild the model, regenerate these
pins from the environment that built it.

**The `@spaces.GPU` probe in `app.py`.** ZeroGPU refuses to start a Space with no
GPU-decorated function, but this model is CPU-only scikit-learn and never needs one. The
probe exists solely to satisfy that check; nothing calls it, so no GPU is requested and no
daily quota is consumed. Do not delete it, and do not move the decorator onto the real
prediction — that would request a GPU on every click and exhaust the free five minutes a day
within a few dozen predictions, for no speedup whatsoever.

## Checking it still agrees with the command line

```bash
pytest tests/verify_space_parity.py -v -s
```

Loads both model bundles and compares the temperature, class list, tree count and vocabulary;
hashes every shared module; then runs ten loci — including a two-way split, a locus with one
readable gene, and one with none — through the CLI predictor and through this app's engine in
separate interpreters, and requires all twelve probabilities, all twelve *p*-values and the
significance verdict to agree to within 1e-12.

`scripts/15_deploy_space.py` runs this for you. Run it by hand after touching `engine.py`,
the tokenizer, or the model bundle.
