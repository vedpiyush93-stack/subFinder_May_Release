"""HTML rendering for results. Kept apart from the engine so presentation
choices never leak into what the model reports."""
from __future__ import annotations
import html
import pandas as pd

from engine import ALPHA, CANON, MIN_INFORMATIVE

# Substrates below this, outside the top three, are folded away by default:
# at four decimal places they are a row of zeros, and twelve rows per locus
# buries the two or three that carry the answer.
TAIL_AT = 0.005


def _fmt_p(p: float) -> str:
    if p >= 0.01:  return f"{p:.3f}"
    if p == 0:     return "&lt;1e-300"
    return f"{p:.1e}".replace("e-0", "e-")


def _genes_html(cell: str, documented: str) -> str:
    if not cell:
        return '<span class="sf-empty">not shown</span>'
    lit = {g.strip() for g in (documented or "").split(",") if g.strip()}
    return "".join(
        f'<span class="sf-gene{" lit" if g.strip() in lit else ""}">{html.escape(g.strip())}</span>'
        for g in cell.split(",") if g.strip())


def verdict_card(sub: pd.DataFrame) -> str:
    """The headline for one locus: what it is, how sure, and whether to believe it."""
    win = sub[sub.is_winner].iloc[0]
    withheld = not str(win.verdict).startswith("reported")
    runner = sub[sub["rank"] == 2].iloc[0] if (sub["rank"] == 2).any() else None
    margin = float(win.probability) - float(runner.probability) if runner is not None else 1.0

    if withheld:
        cls, state = "stop", "call withheld"
        note = (f"Only {int(win.readable_genes)} gene the model recognises. "
                f"Below {MIN_INFORMATIVE} we report the numbers but withhold the "
                f"call, because one gene is not enough to be confident.")
    elif not bool(win.significant):
        cls, state = "stop", "no call"
        note = ("No substrate's panel out-voted a coin toss. Treat this locus as "
                "unresolved rather than as a weak hit.")
    elif margin < 0.30 and runner is not None:
        cls, state = "warn", "close call"
        note = (f"Above chance, but {html.escape(str(runner.substrate))} is not far "
                f"behind at {runner.probability:.3f}. The test says this locus is "
                f"informative; it does not say the winner is the right one of the "
                f"two. Read both rows before acting.")
    else:
        cls, state = "ok", "clear call"
        note = ("Above chance, and the runner-up is far behind. The genes listed below "
                "are the ones whose removal moves this call most.")

    return f"""
<div class="sf-verdict {cls}">
  <div><div class="sf-k">Locus</div>
       <div class="sf-name">{html.escape(str(win.locus))}</div></div>
  <div><div class="sf-k">Most likely substrate</div>
       <div class="sf-name">{html.escape(str(win.substrate))}</div></div>
  <div><div class="sf-k">Probability</div>
       <div class="sf-num">{win.probability:.3f}</div></div>
  <div><div class="sf-k">Votes</div>
       <div class="sf-num" style="font-size:1.35rem">{html.escape(str(win.votes))}</div></div>
  <div><div class="sf-k">p-value</div>
       <div class="sf-num" style="font-size:1.35rem">{_fmt_p(float(win.p_value))}</div></div>
  <div class="sf-spacer"></div>
  <div><div class="sf-k">Verdict</div>
       <span class="sf-state {cls}">{state}</span></div>
  <div class="sf-note">{note}</div>
</div>"""


def table(sub: pd.DataFrame) -> str:
    """Every substrate for one locus, with the long tail folded away.

    Ten of the twelve substrates are usually at or near zero. Printing them all
    makes the reader scan past nine dead rows to reach the sortable table, so
    anything below `TAIL_AT` past the top three is hidden behind one line the
    user can open. Nothing is dropped -- the rows are in the page, the master
    table and the CSV regardless.
    """
    # teal means "trust this". A locus whose call was withheld, or that beat nothing,
    # must not get it on its top row just for ranking first.
    endorsed = bool(sub[sub.is_winner].iloc[0].significant)
    rows, tail_n = [], 0
    for _, r in sub.iterrows():
        winner = bool(r.is_winner)
        prob = float(r.probability)
        in_tail = (not winner) and int(r["rank"]) > 3 and prob < TAIL_AT
        tail_n += in_tail
        faded = (not winner) and prob < 0.01
        bar = max(1, round(prob * 100))
        pill = ('<span class="sf-pill yes">yes</span>' if bool(r.significant)
                else '<span class="sf-pill no">no</span>')
        top = ("win" if endorsed else "winoff") if winner else ("dim" if faded else "")
        cls = " ".join(c for c in (top, "sf-tail" if in_tail else "") if c)
        rows.append(f"""
<tr class="{cls}"{' hidden' if in_tail else ''}>
  <td class="sub">{html.escape(str(r.substrate))}</td>
  <td class="n">{prob:.4f}</td>
  <td><span class="sf-track"><span
      class="sf-bar{'' if prob >= .05 else ' lo'}" style="width:{bar}%"></span></span></td>
  <td class="n">{html.escape(str(r.votes))}</td>
  <td class="n">{_fmt_p(float(r.p_value))}</td>
  <td>{pill}</td>
  <td>{_genes_html(str(r.signature_genes), str(r.documented))}</td>
</tr>""")

    more = ""
    if tail_n:
        lo = f"{TAIL_AT:g}"
        more = (f'<tfoot><tr class="sf-tailrow"><td colspan="7">'
                f'<button type="button" class="sf-more" data-open="0" '
                f'data-more="show the {tail_n} substrates below {lo}" '
                f'data-less="hide them again" onclick="sfTail(this)">'
                f'show the {tail_n} substrates below {lo}</button></td></tr></tfoot>')

    return f"""
<div class="sf-scroll"><table class="sf-tbl">
<colgroup><col style="width:200px"><col style="width:104px"><col style="width:132px">
<col style="width:96px"><col style="width:116px"><col style="width:118px"><col></colgroup>
<thead><tr>
  <th>Substrate</th><th>Probability</th><th></th><th>Votes</th>
  <th>p-value</th><th>Beats chance</th><th>Signature genes</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>{more}</table></div>"""


IDLE_HTML = """
<div class="sf-idle">
  <h3>Results appear here</h3>
  <p>Every locus gets a card &mdash; the substrate, how confident the model is, and the genes
  that produced the call &mdash; followed by one sortable table holding every locus against
  every substrate. The coloured edge of each card is the first thing to read:</p>
  <div class="sf-idle-grid">
    <div class="sf-idle-cell ok"><b>Clear call</b><span>Beats chance and the runner-up is far
      behind. Take the substrate at face value.</span></div>
    <div class="sf-idle-cell warn"><b>Close call</b><span>Beats chance, but the second substrate
      is nearly as likely. The locus is informative; which of the two it is, is not settled.</span></div>
    <div class="sf-idle-cell stop"><b>No call</b><span>Nothing out-voted a coin toss, or fewer
      than two genes were recognised. The tool is declining to guess.</span></div>
  </div>
</div>"""


def results_html(df: pd.DataFrame, max_detail: int = 8) -> str:
    """The on-tool view: a summary strip, a card per locus, then every row sortable.

    Cards are capped because they are tall; the table below always carries every
    row, and the CSV carries them too, so nothing is ever only-partially shown.
    """
    if df.empty:
        return IDLE_HTML
    blocks = [summary_strip(df)]
    loci = list(dict.fromkeys(df.locus))
    for locus in loci[:max_detail]:
        sub = df[df.locus == locus].sort_values("rank")
        blocks.append(f'<div class="sf-card" style="margin-bottom:16px">'
                      f'{verdict_card(sub)}{table(sub)}</div>')
    if len(loci) > max_detail:
        blocks.append(f'<p class="sf-foot">Detailed cards are shown for the first '
                      f'{max_detail} of {len(loci)} loci. Every locus appears in the '
                      f'sortable table below and in the CSV.</p>')
    blocks.append(master_table(df))
    blocks.append(
        '<p class="sf-foot">'
        '<span class="sf-gene lit">teal gene</span> the curated literature table lists this '
        'family for that substrate &nbsp;&middot;&nbsp; '
        '<span class="sf-gene">grey gene</span> it does not, which is not evidence against '
        'the call &nbsp;&middot;&nbsp; <b>Beats chance</b> means that substrate\'s own panel '
        'of 500 trees out-voted a coin toss after correcting for testing all twelve '
        '(p &lt; 0.05/12) <i>and</i> the locus had at least two genes the model recognises.'
        '</p>')
    return "".join(blocks)


# ---------------------------------------------------------------- summary strip
def summary_strip(df: pd.DataFrame) -> str:
    """Numbers a user checks before reading anything else."""
    loci = df.locus.nunique()
    win = df[df.is_winner]
    n_sig = int(win.significant.sum())
    n_held = int((~win.verdict.astype(str).str.startswith("reported")).sum())
    n_doc = int((win.documented.fillna("") != "").sum())
    med = float(win.probability.median())
    cards = [
        ("loci", f"{loci:,}", "submitted"),
        ("with a call", f"{n_sig:,}", f"{100*n_sig/max(loci,1):.0f}% beat chance"),
        ("withheld", f"{n_held:,}", "too few readable genes"),
        ("median confidence", f"{med:.2f}", "of the winning substrate"),
        ("literature-backed", f"{n_doc:,}", "winner has a documented gene"),
    ]
    inner = "".join(
        f'<div class="sf-stat"><div class="sf-stat-k">{k}</div>'
        f'<div class="sf-stat-v">{v}</div><div class="sf-stat-s">{s_}</div></div>'
        for k, v, s_ in cards)
    return f'<div class="sf-strip">{inner}</div>'


# ---------------------------------------------------------------- sortable table
def master_table(df: pd.DataFrame, table_id: str = "sfMaster") -> str:
    """Every row, every locus, sortable on any column.

    The CSV is for downstream work; this is for looking. Sorting happens in the
    browser so it is instant and does not re-run the model.
    """
    cols = [
        ("Locus",      "text", "locus"),
        ("Substrate",  "text", "substrate"),
        ("Rank",       "num",  "rank"),
        ("Probability","num",  "probability"),
        ("Votes",      "num",  "votes"),
        ("p-value",    "num",  "p_value"),
        ("Beats chance","num", "significant"),
        ("Signature genes", "text", "signature_genes"),
        ("Documented", "num",  "documented"),
    ]
    head = "".join(
        f'<th data-type="{t}" onclick="sfSort(this)" title="click to sort">{lab}</th>'
        for lab, t, _ in cols)

    rows = []
    for _, r in df.iterrows():
        genes = str(r.signature_genes or "")
        doc = {g.strip() for g in str(r.documented or "").split(",") if g.strip()}
        n_doc = len(doc)
        gene_html = (_genes_html(genes, str(r.documented or ""))
                     if genes else '<span class="sf-empty">not shown</span>')
        sig = bool(r.significant)
        vote_n = int(str(r.votes).split("/")[0])
        top = ("win" if sig else "winoff") if bool(r.is_winner) else ""
        rows.append(
            f'<tr data-sig="{1 if sig else 0}" class="{top}">'
            f'<td data-v="{html.escape(str(r.locus))}" class="mono">{html.escape(str(r.locus))}</td>'
            f'<td data-v="{html.escape(str(r.substrate))}" class="sub">{html.escape(str(r.substrate))}</td>'
            f'<td data-v="{int(r["rank"])}" class="n">{int(r["rank"])}</td>'
            f'<td data-v="{float(r.probability):.6f}" class="n">{float(r.probability):.4f}</td>'
            f'<td data-v="{vote_n}" class="n">{html.escape(str(r.votes))}</td>'
            f'<td data-v="{float(r.p_value):.3e}" class="n">{_fmt_p(float(r.p_value))}</td>'
            f'<td data-v="{1 if sig else 0}">'
            f'{"<span class=\"sf-pill yes\">yes</span>" if sig else "<span class=\"sf-pill no\">no</span>"}</td>'
            f'<td data-v="{html.escape(genes)}">{gene_html}</td>'
            f'<td data-v="{n_doc}" class="n">{n_doc if genes else ""}</td>'
            f'</tr>')

    n = len(df)
    return f"""
<div class="sf-card">
  <div class="sf-tablebar">
    <p class="sf-label" style="margin:0">All results &mdash; click any column to sort</p>
    <div class="sf-tools">
      <input class="sf-search" type="text" placeholder="filter rows&hellip;"
             data-target="{table_id}" data-count="{table_id}Count" oninput="sfFilter(this)">
      <label class="sf-check"><input type="checkbox" data-target="{table_id}"
             data-count="{table_id}Count" onchange="sfOnlySig(this)"> only rows that beat chance</label>
      <span class="sf-count" id="{table_id}Count">{n} of {n} rows</span>
    </div>
  </div>
  <div class="sf-scroll sf-tallscroll">
    <table class="sf-tbl sf-sortable" id="{table_id}">
      <thead><tr>{head}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</div>"""
