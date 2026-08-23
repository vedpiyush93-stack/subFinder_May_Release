"""Prediction engine for the subFinder Space.

Wraps the released model so the UI never touches sklearn directly. Everything
here mirrors the command-line tool: the same tokenizer, the same temperature,
the same vote-count p-values and the same rule about which genes may be shown.
"""
from __future__ import annotations
import io
import re
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binom

from src.preprocessing.tokenizers import tok_cpu_v2
from src.preprocessing.cgc_loader import cgc_to_pul_strings
from src.calibration.temperature import apply_temperature
from src.ablation.leave_one_token_out import ablate_pul_for_class
from src.lit_validation.canon import build_canon
from src.lit_validation.alias_map import SUBSTRATE_ALIAS

ROOT = Path(__file__).parent
NOT_A_GENE = frozenset({"null", ""})
ALPHA = 0.05

_bundle = joblib.load(ROOT / "final_model_v2.pkl")
PIPE = _bundle["pipeline"]
T = float(_bundle["T"])
CLASSES = [str(c) for c in _bundle["classes"]]
K = len(CLASSES)
_ests = PIPE.named_steps["vr"].estimators_
NTREES = int(_ests[0].n_estimators)
VOCAB = set(PIPE.named_steps["cv"].get_feature_names_out())
CANON = build_canon(ROOT / "Literature_Data_fam_substrate_mapping.tsv", SUBSTRATE_ALIAS)

# a substrate is significant when its own panel out-votes a coin toss, after
# correcting for having asked the question twelve times
VOTE_THRESHOLD = int(binom.isf(ALPHA / K, NTREES, 0.5)) + 1
MIN_INFORMATIVE = 2


def informative_tokens(seq: str) -> int:
    """Genes the model can actually read: in-vocabulary, and not the null placeholder."""
    return sum(1 for t in tok_cpu_v2(seq) if t not in NOT_A_GENE and t in VOCAB)


def _vote_fractions(seqs: list[str]) -> np.ndarray:
    Z = PIPE.named_steps["cv"].transform(seqs)
    return np.column_stack([e.predict_proba(Z)[:, 1] for e in _ests])


def signature_genes(seq: str, substrate: str, k: int = 3) -> list[tuple[str, float, bool]]:
    """Genes whose removal moves this substrate's probability most.

    `null` and bare subfamily indices are dropped: both are real features, but
    neither names a gene a user could look up.
    """
    try:
        deltas = ablate_pul_for_class(PIPE, seq, substrate, top_k=k + 6, apply_temp=T)
    except Exception:
        return []
    out = []
    for tok, d in deltas:
        if tok in NOT_A_GENE or tok.isdigit():
            continue
        out.append((tok, float(d), tok in CANON.get(substrate, set())))
        if len(out) == k:
            break
    return out


def predict_frame_staged(items, *, gene_k=3, min_prob_for_detail=0.05,
                        detail_everything=False, report=None):
    """Same as predict_frame, but reports progress as it goes.

    `report(fraction, message)` is called at each stage. Signature-gene attribution
    dominates the cost -- it re-runs the model once per token per substrate -- so it
    is reported per locus rather than as one long silent block.
    """
    def say(f, m):
        if report: report(f, m)

    ids  = [i for i, _ in items]
    seqs = [s for _, s in items]
    if not seqs:
        return pd.DataFrame()

    say(0.05, "Reading loci")
    n_inf_all = [informative_tokens(s) for s in seqs]

    say(0.15, f"Predicting substrates for {len(seqs):,} loci")
    V = _vote_fractions(seqs)
    P = apply_temperature(PIPE.predict_proba(seqs), T)

    say(0.35, "Computing p-values from the vote counts")
    counts = np.rint(V * NTREES).astype(int)
    pvals = np.minimum(1.0, K * binom.sf(counts - 1, NTREES, 0.5))

    say(0.45, f"Finding signature genes across {len(seqs):,} loci")
    wanted = [[int(ci) for ci in range(K)
               if detail_everything or P[r, ci] >= min_prob_for_detail]
              for r in range(len(seqs))]
    # one batched pass instead of one model call per (locus, substrate)
    genes_by_locus = signature_genes_batch(seqs, wanted, gene_k)
    say(0.90, "Assembling results")

    rows = []
    for r, (locus, seq) in enumerate(zip(ids, seqs)):
        n_inf = n_inf_all[r]
        enough = n_inf >= MIN_INFORMATIVE
        order = np.argsort(-P[r])
        for rank, ci in enumerate(order, start=1):
            sub = CLASSES[int(ci)]
            prob = float(P[r, ci])
            show_detail = detail_everything or prob >= min_prob_for_detail
            genes = genes_by_locus[r].get(int(ci), []) if show_detail else []
            rows.append({
                "locus": locus, "rank": rank, "substrate": sub,
                "probability": prob,
                "p_value": float(pvals[r, ci]),
                "significant": bool(pvals[r, ci] < ALPHA) and enough,
                "votes": f"{counts[r, ci]}/{NTREES}",
                "signature_genes": ", ".join(g for g, _, _ in genes) if show_detail else "",
                "documented": ", ".join(g for g, _, lit in genes if lit) if show_detail else "",
                "is_winner": rank == 1,
                "readable_genes": n_inf,
                "verdict": ("reported" if enough else
                            f"withheld: fewer than {MIN_INFORMATIVE} readable genes"),
            })
    say(0.97, "Formatting results")
    return pd.DataFrame(rows)


def predict_frame(items: list[tuple[str, str]], *, gene_k: int = 3,
                  min_prob_for_detail: float = 0.05,
                  detail_everything: bool = False) -> pd.DataFrame:
    """One row per (locus x substrate), long format.

    `min_prob_for_detail` suppresses signature genes below a probability the user
    chooses. Low-probability rows are still returned with their probability and
    p-value; what is withheld is the gene-level explanation, which is unstable
    when the substrate was never really in contention.
    """
    ids = [i for i, _ in items]
    seqs = [s for _, s in items]
    if not seqs:
        return pd.DataFrame()

    V = _vote_fractions(seqs)
    P = apply_temperature(PIPE.predict_proba(seqs), T)
    counts = np.rint(V * NTREES).astype(int)
    pvals = np.minimum(1.0, K * binom.sf(counts - 1, NTREES, 0.5))

    wanted = [[int(ci) for ci in range(K)
               if detail_everything or P[r, ci] >= min_prob_for_detail]
              for r in range(len(seqs))]
    genes_by_locus = signature_genes_batch(seqs, wanted, gene_k)

    rows = []
    for r, (locus, seq) in enumerate(zip(ids, seqs)):
        n_inf = informative_tokens(seq)
        enough = n_inf >= MIN_INFORMATIVE
        order = np.argsort(-P[r])
        for rank, ci in enumerate(order, start=1):
            sub = CLASSES[int(ci)]
            prob = float(P[r, ci])
            show_detail = detail_everything or prob >= min_prob_for_detail
            genes = genes_by_locus[r].get(int(ci), []) if show_detail else []
            sig = bool(pvals[r, ci] < ALPHA) and enough
            rows.append({
                "locus": locus,
                "rank": rank,
                "substrate": sub,
                # full precision: the CSV is data, and rounding here would make the
                # Space disagree with the command-line tool in the fourth decimal.
                # The table formats for display instead.
                "probability": prob,
                "p_value": float(pvals[r, ci]),
                "significant": sig,
                "votes": f"{counts[r, ci]}/{NTREES}",
                "signature_genes": ", ".join(g for g, _, _ in genes) if show_detail else "",
                "documented": ", ".join(g for g, _, lit in genes if lit) if show_detail else "",
                "is_winner": rank == 1,
                "readable_genes": n_inf,
                "verdict": ("reported" if enough else
                            f"withheld: fewer than {MIN_INFORMATIVE} readable genes"),
            })
    return pd.DataFrame(rows)


def parse_typed(text: str) -> list[tuple[str, str]]:
    """One locus per line. `name<TAB>sequence` or just the sequence."""
    items = []
    for n, raw in enumerate((text or "").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if "\t" in line:
            name, seq = line.split("\t", 1)
        else:
            name, seq = f"locus_{n}", line
        seq = seq.strip()
        if seq:
            items.append((name.strip(), seq))
    return items


# columns we will accept for the locus name and the gene string, in priority order
_ID_COLS  = ("locus", "locus_id", "id", "name", "cgc", "cgc_id", "pul", "pul_id")
_SEQ_COLS = ("sequence", "sig_gene_seq", "genes", "gene_string", "pul_string",
             "annotation", "protein_family", "seq")
_HEADER_HINTS = _ID_COLS + _SEQ_COLS


def parse_table(path: str) -> list[tuple[str, str]]:
    """Read a CSV or TSV of loci from a file."""
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        return parse_table_stream(fh)


def parse_table_text(text: str) -> list[tuple[str, str]]:
    """Read a CSV or TSV of loci that arrived as pasted text."""
    return parse_table_stream(io.StringIO(text))


def parse_table_stream(fh) -> list[tuple[str, str]]:
    """Read a CSV or TSV of loci.

    A comma-delimited file is genuinely ambiguous here, because the gene string is
    itself comma-separated. So the file must be a real CSV: the gene string in one
    quoted field, or the file tab-separated. We identify the columns by name where
    a header exists, and refuse rather than guess when it does not.
    """
    import csv as _csv
    sample = fh.read(8192); fh.seek(0)
    try:
        dialect = _csv.Sniffer().sniff(sample, delimiters=",\t;")
    except _csv.Error:
        dialect = _csv.excel_tab if "\t" in sample else _csv.excel
    rows = [r for r in _csv.reader(fh, dialect) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("there is nothing to read")

    header = [c.strip().lower() for c in rows[0]]
    has_header = any(h in _HEADER_HINTS for h in header)
    if has_header:
        seq_i = next((i for i, h in enumerate(header) if h in _SEQ_COLS), None)
        id_i  = next((i for i, h in enumerate(header) if h in _ID_COLS), None)
        if seq_i is None:
            # a header is present but names no gene column; take the widest column,
            # which is the gene string in every real file we have seen
            body = rows[1:]
            seq_i = max(range(len(header)),
                        key=lambda c: max((len(r[c]) for r in body if c < len(r)), default=0))
        body = rows[1:]
    else:
        body = rows
        seq_i = max(range(len(rows[0])),
                    key=lambda c: max((len(r[c]) for r in rows if c < len(r)), default=0))
        id_i = 0 if len(rows[0]) > 1 and seq_i != 0 else None

    items = []
    for n, r in enumerate(body, start=1):
        if seq_i >= len(r):
            continue
        seq = r[seq_i].strip()
        if not seq:
            continue
        name = (r[id_i].strip() if id_i is not None and id_i < len(r) and r[id_i].strip()
                else f"locus_{n}")
        items.append((name, seq))
    if not items:
        raise ValueError("no gene strings found. Expected a column of comma-separated "
                         "gene labels, e.g. \"GH13,CBM48,1.B.14.6.1,null\"")
    return items


def parse_cgc(path: str) -> list[tuple[str, str]]:
    """A dbCAN cgc_standard.out table, read exactly as the command-line tool reads it."""
    return list(cgc_to_pul_strings(path).items())


def looks_like_cgc_text(text: str) -> bool:
    """Is this pasted text a dbCAN CGC table rather than one locus per line?

    People paste what they have. A CGC table read as one-locus-per-line turns every
    gene row into its own locus and the header into a locus called ``CGC#``, none of
    which carries a readable gene -- so the tool answers "no call" a dozen times over
    instead of saying it was handed the wrong shape.
    """
    head = text.lstrip().splitlines()[:1]
    if not head:
        return False
    first = head[0]
    return "CGC#" in first or ("Gene Type" in first and "Protein Family" in first)


def parse_cgc_text(text: str) -> list[tuple[str, str]]:
    """Same as parse_cgc, for a table that arrived as pasted text."""
    f = tempfile.NamedTemporaryFile(prefix="subfinder_cgc_", suffix=".out",
                                    delete=False, mode="w", encoding="utf-8")
    f.write(text)
    f.close()
    return parse_cgc(f.name)


def looks_like_table_text(text: str) -> bool:
    """Does this text open with a header naming a locus or gene-string column?"""
    first = next((l for l in (text or "").splitlines() if l.strip()), "")
    cells = [c.strip().strip('"').lower() for c in re.split(r"[,\t;]", first)]
    return sum(c in _HEADER_HINTS for c in cells) >= 1 and len(cells) >= 2


def parse_any(text: str) -> tuple[list[tuple[str, str]], str]:
    """Read whichever accepted layout this text is, and say which it was.

    The text box and the file upload both come through here, so the two cannot
    drift apart: whatever you can upload, you can paste, and it is read the same way.
    """
    if looks_like_cgc_text(text):
        return parse_cgc_text(text), "dbCAN CGC table"
    if looks_like_table_text(text):
        return parse_table_text(text), "table with a header row"
    return parse_typed(text), "one locus per line"


def to_csv(df: pd.DataFrame) -> str:
    f = tempfile.NamedTemporaryFile(prefix="subfinder_", suffix=".csv",
                                    delete=False, mode="w", newline="")
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


def to_html_file(html_body: str, title: str = "subFinder results") -> str:
    """A standalone HTML report: the same view, saved, openable offline."""
    from theme import CSS
    from sortjs import SORT_JS
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{title}</title>{SORT_JS}<style>{CSS}\n"
           "body{margin:0;padding:30px 24px 56px;"
           "background:radial-gradient(1100px 460px at 50% -180px,#dce9e9 0%,transparent 70%),"
           "#eaf0f0;font-family:'Inter',-apple-system,sans-serif;color:#101a1d}"
           ".gradio-container{max-width:1280px;margin:0 auto}</style></head>"
           f"<body><div class='gradio-container'>{html_body}</div></body></html>")
    f = tempfile.NamedTemporaryFile(prefix="subfinder_report_", suffix=".html",
                                    delete=False, mode="w", encoding="utf-8")
    f.write(doc); f.close()
    return f.name


# --------------------------------------------------------------- batched ablation
def signature_genes_batch(seqs: list[str], wanted: list[list[int]],
                          gene_k: int = 3) -> list[dict[int, list]]:
    """Signature genes for many loci and many substrates in one pass.

    The per-locus helper re-runs the classifier once for every (locus, substrate)
    pair. That is wasteful twice over: `predict_proba` already returns all twelve
    columns, so scoring a second substrate for the same locus recomputes an
    identical prediction just to read a different column; and every call pays the
    fixed cost of pushing rows through twelve 500-tree forests.

    Here every ablated row for every locus is stacked into one matrix and predicted
    once, then sliced back out. Same arithmetic, one call.

    Args:
        seqs:   the locus strings.
        wanted: for each locus, the substrate indices to attribute.
        gene_k: genes to return per (locus, substrate).
    Returns:
        One dict per locus: {substrate_index: [(token, delta, is_documented), ...]}
    """
    from scipy import sparse
    cv = PIPE.named_steps["cv"]
    clf = PIPE.named_steps["vr"]
    inv = {v: k for k, v in cv.vocabulary_.items()}

    Xv = cv.transform(seqs).tocsr()
    P_full = apply_temperature(clf.predict_proba(Xv), T)

    # one ablated copy per (locus, present token), built directly from the CSR
    # arrays -- going through lil per row costs more than the prediction does
    rows_i, cols_i, vals_i, owner = [], [], [], []
    out_row = 0
    for i in range(Xv.shape[0]):
        lo, hi = Xv.indptr[i], Xv.indptr[i + 1]
        idx, dat = Xv.indices[lo:hi], Xv.data[lo:hi]
        for drop in range(len(idx)):
            keep = np.arange(len(idx)) != drop
            rows_i.append(np.full(keep.sum(), out_row, dtype=np.int32))
            cols_i.append(idx[keep])
            vals_i.append(dat[keep])
            owner.append((i, int(idx[drop])))
            out_row += 1

    result: list[dict[int, list]] = [{} for _ in seqs]
    if out_row == 0:
        return result

    big = sparse.coo_matrix(
        (np.concatenate(vals_i), (np.concatenate(rows_i), np.concatenate(cols_i))),
        shape=(out_row, Xv.shape[1])).tocsr()
    P_ab = apply_temperature(clf.predict_proba(big), T)

    by_locus: dict[int, list] = {}
    for r, (i, col) in enumerate(owner):
        by_locus.setdefault(i, []).append((r, col))

    for i, entries in by_locus.items():
        for ci in wanted[i]:
            deltas = []
            for r, col in entries:
                tok = inv[col]
                if tok in NOT_A_GENE or tok.isdigit():
                    continue
                deltas.append((tok, float(P_full[i, ci] - P_ab[r, ci])))
            deltas.sort(key=lambda x: -x[1])
            sub = CLASSES[ci]
            result[i][ci] = [(t, d, t in CANON.get(sub, set()))
                             for t, d in deltas[:gene_k]]
    return result
