"""subFinder — predict the substrate of a polysaccharide utilization locus."""
from __future__ import annotations
import pathlib

import gradio as gr
import pandas as pd

# ZeroGPU refuses to start a Space with no GPU-decorated function ("No @spaces.GPU
# function detected during startup"). This model is CPU-only scikit-learn and never
# needs a GPU, but ZeroGPU is the only tier a free account may run a Python app on.
# The probe below exists solely to satisfy that check. Nothing in the interface calls
# it, so no GPU is ever requested and no daily GPU quota is consumed. Decorating the
# real prediction instead would request a GPU on every click and exhaust the free
# 5 minutes/day within a few dozen predictions, for no speedup whatsoever.
try:
    import spaces
    if hasattr(spaces, "GPU"):          # the real package, not a directory shadowing it

        @spaces.GPU(duration=1)
        def _zerogpu_startup_probe() -> str:
            return "subfinder runs on cpu"
except Exception:                       # absent locally, or shadowed by a local path entry
    pass

import engine
import render
from theme import CSS, THEME, FORCE_LIGHT
from sortjs import SORT_JS

EXAMPLE = "1.B.14.12.1,GntR,PL6|PL6_1,PL17_2|PL17,2.A.1.14.25,null"
EXAMPLES = "\n".join([
    f"alginate_demo\t{EXAMPLE}",
    "starch_demo\tGH13_1,GH13,CBM48,SusC,1.B.14.6.1,GH31,null",
    "xylan_demo\tGH10,GH43_4,CBM6,1.B.14.2.1,AraC,null,GH67",
])
# Raised from 300 after batching the ablation: 100 loci now take ~1.6 s
# rather than ~70 s, so the cap is about keeping one request bounded,
# not about the model being slow.
BATCH_CAP = 2000
EX = pathlib.Path(__file__).parent / "examples"

HELP_HTML = """
<div class="sf-help">

<h4>1 &nbsp;What one locus looks like</h4>
<p>Whichever route you take, a locus is a <b>comma-separated list of gene labels</b>. Use a
pipe for two annotations of the same gene, and <code>null</code> for a gene the annotation
pipeline could not label.</p>
<pre>1.B.14.12.1,GntR,PL6|PL6_1,PL17_2|PL17,2.A.1.14.25,null</pre>
<p>That is six genes: a transporter, a regulator, two lyases (each with an alternative
annotation), another transporter, and one unannotated gene. Keep the <code>null</code>s in;
the model uses them as evidence. Gene order does not matter.</p>

<h4>2 &nbsp;Four ways to submit</h4>

<p><b>A &nbsp;Type or paste.</b> One locus per line. Put a name before a <b>tab</b> to carry
it into the results; otherwise loci are numbered for you.</p>
<pre>alginate_demo&#9;1.B.14.12.1,GntR,PL6|PL6_1,PL17_2|PL17,2.A.1.14.25,null
my_locus_2&#9;GH13_1,GH13,CBM48,SusC,1.B.14.6.1,GH31,null</pre>

<p><b>B &nbsp;A plain text file</b> (<code>.txt</code>) &mdash; the same layout, uploaded
rather than pasted.</p>

<p><b>C &nbsp;A CSV or TSV.</b> This is the one to get right, because the gene string
contains commas itself. <b>In a CSV the gene string must be quoted:</b></p>
<pre>locus_id,sequence
my_locus_1,"1.B.14.12.1,GntR,PL6|PL6_1,PL17_2|PL17,2.A.1.14.25,null"
my_locus_2,"GH13_1,GH13,CBM48,SusC,1.B.14.6.1,GH31,null"</pre>
<p>A TSV needs no quoting, because tabs separate the columns:</p>
<pre>locus&#9;sig_gene_seq
my_locus_1&#9;1.B.14.12.1,GntR,PL6|PL6_1,PL17_2|PL17,2.A.1.14.25,null</pre>
<p>Column names are matched case-insensitively. Accepted for the name:
<code>locus</code>, <code>locus_id</code>, <code>id</code>, <code>name</code>,
<code>cgc</code>, <code>pul</code>. For the gene string: <code>sequence</code>,
<code>sig_gene_seq</code>, <code>genes</code>, <code>gene_string</code>,
<code>pul_string</code>, <code>annotation</code>, <code>seq</code>. With no header, the
widest column is taken as the gene string and the first as the name. Extra columns are
ignored, so your own metadata can stay in the file.</p>

<p><b>D &nbsp;A dbCAN CGC table</b> (<code>cgc_standard.out</code>) &mdash; one row per gene,
exactly as dbCAN writes it. Upload unchanged; each CGC becomes one locus. Detected
automatically from the header.</p>

<p>Up to 2,000 loci per run; a thousand take about two seconds.</p>

<h4>3 &nbsp;Mistakes that cost people time</h4>
<table>
<tr><th>what you see</th><th>what happened</th></tr>
<tr><td>a locus predicted from apparently nothing</td><td>An <b>unquoted CSV</b>. Every comma split the gene string into its own column, so almost nothing reached the model. Quote the gene string, or use a TSV.</td></tr>
<tr><td>one extra locus at the top</td><td>A header row the tool did not recognise, so it was read as data. Name the column <code>sequence</code> or <code>sig_gene_seq</code>.</td></tr>
<tr><td>your names replaced by locus_1, locus_2</td><td>No identifier column, or a name separated from the sequence by spaces instead of a tab.</td></tr>
<tr><td>everything withheld</td><td>The labels are not ones the model knows. It expects CAZy families (<code>GH13</code>), TC numbers (<code>1.B.14.6.1</code>) and regulator families &mdash; not protein IDs, locus tags or gene names.</td></tr>
</table>

<h4>4 &nbsp;Reading the answer</h4>
<p>Each locus gets a card whose coloured edge tells you how far to trust it before you read
a single number:</p>
<table>
<tr><td style="color:#17635f"><b>teal</b></td><td>above chance, runner-up far behind</td></tr>
<tr><td style="color:#b8791f"><b>amber</b></td><td>above chance, but the second substrate is close &mdash; read both rows</td></tr>
<tr><td style="color:#a8434b"><b>red</b></td><td>no call: nothing beat chance, or fewer than two genes were recognised</td></tr>
</table>
<p>Below the cards, every row for every locus in one table. Click any column to sort, type in
the box to filter, or tick <i>only rows that beat chance</i>. All of it happens in your
browser, so it is instant and never re-runs the model.</p>

<h4>5 &nbsp;What the columns mean</h4>
<table>
<tr><th>column</th><th>meaning</th></tr>
<tr><td>probability</td><td>Calibrated. Held-out loci called at 0.95 or above are correct 98.5% of the time, so the number can be read at face value.</td></tr>
<tr><td>votes</td><td>How many of that substrate's 500 trees voted yes &mdash; the raw evidence behind the probability.</td></tr>
<tr><td>p-value</td><td>The chance of that many yes votes from a coin toss, corrected for testing all twelve substrates.</td></tr>
<tr><td>beats chance</td><td>Yes when the p-value clears 0.05/12 <i>and</i> at least two genes were recognised.</td></tr>
<tr><td>signature genes</td><td>Genes whose removal moves that substrate's probability most. Teal means the curated literature table lists that family for that substrate; grey means it does not, which is not evidence against the call.</td></tr>
<tr><td>documented</td><td>How many of the shown genes are literature-backed. Sort on it to surface the best-supported calls.</td></tr>
</table>

<h4>6 &nbsp;The dials</h4>
<p><b>Signature genes per substrate</b> &mdash; how many to list, one to five.</p>
<p><b>Explain substrates above this probability</b> &mdash; below it, probability and p-value
are still reported but the genes are not, because attributions for a substrate that was never
in contention are unstable. Set it to 0 to explain everything.</p>
<p><b>Explain all twelve regardless</b> &mdash; the same thing as a switch.</p>

<h4>7 &nbsp;Taking it with you</h4>
<p><b>CSV</b> for analysis: one row per locus &times; substrate, full precision, every column
above. <b>HTML</b> is this page saved to a file &mdash; sorting and filtering still work
offline.</p>

<h4>What it will not tell you</h4>
<p>The test asks whether a substrate beats a coin toss, not whether the winner beats the
runner-up. A locus split 0.52 against 0.48 is both clearly informative and entirely
undecided &mdash; read both rows. And on uncurated loci, anything not from a characterised
PUL, expect roughly a quarter to get a call. That is the tool declining to guess rather than
failing.</p>

</div>
"""
CHEATSHEET = """
<div class="sf-cheat">
  <p class="sf-label" style="margin:0 0 9px">Labels the model reads</p>
  <div class="sf-cheat-grid">
    <div><code>GH13</code> <code>PL6</code> <code>CE2</code> <code>CBM48</code>
         <span>CAZy families. A subfamily suffix is fine &mdash; <code>GH43_4</code> is read
         as <code>GH43</code>.</span></div>
    <div><code>1.B.14.6.1</code> <code>2.A.1.14.25</code>
         <span>TC numbers for transporters, read to three levels.</span></div>
    <div><code>SusC</code> <code>SusD</code> <code>AraC</code> <code>GntR</code>
         <span>Sus components and regulator families, by name.</span></div>
    <div><code>null</code>
         <span>A gene the pipeline could not annotate. Keep these &mdash; how many a locus has
         is itself evidence.</span></div>
  </div>
  <p class="sf-cheat-warn">Protein IDs, locus tags and gene names are not labels the model
  knows; a locus made only of those gets no call.</p>
</div>"""

HERO = """
<div id="sf-hero">
  <h1>subFinder</h1>
  <p class="sf-sub">Give it the gene labels of a polysaccharide utilization locus and it
  returns the substrate that locus most likely acts on, how confident that is, whether the
  call beats chance, and which genes produced it &mdash; for all twelve substrates, not just
  the winner.</p>
  <div class="sf-chips">
    <span class="sf-chip"><b>0.9177 &plusmn; 0.0146</b> held-out accuracy</span>
    <span class="sf-chip"><b>12</b> substrate classes</span>
    <span class="sf-chip"><b>500</b> trees per substrate</span>
    <span class="sf-chip"><b>1,030</b> experimentally characterised loci</span>
  </div>
</div>"""


def _run(items, gene_k, min_prob, show_all, progress):
    if not items:
        return (render.results_html(pd.DataFrame()),
                gr.DownloadButton(interactive=False),
                gr.DownloadButton(interactive=False),
                gr.update(value="Nothing to read in that input.", visible=True))
    trimmed = items[:BATCH_CAP]

    stages = []
    def report(frac, msg):
        stages.append(msg)
        progress(frac, desc=msg)

    progress(0.02, desc="Loading the model")
    df = engine.predict_frame_staged(
        trimmed, gene_k=int(gene_k), min_prob_for_detail=float(min_prob),
        detail_everything=bool(show_all), report=report)

    html = render.results_html(df)
    progress(0.99, desc="Writing downloads")
    csv_path = engine.to_csv(df)
    html_path = engine.to_html_file(html)

    n = len(trimmed)
    msg = f"**{n:,} {'locus' if n == 1 else 'loci'} predicted.** "
    win = df[df.is_winner]
    msg += (f"{int(win.significant.sum()):,} beat chance; "
            f"{int((~win.verdict.astype(str).str.startswith('reported')).sum()):,} withheld.")
    if len(items) > BATCH_CAP:
        msg += f" Input had {len(items):,}; the first {BATCH_CAP} were used."
    return (html, gr.DownloadButton(value=csv_path, interactive=True),
        gr.DownloadButton(value=html_path, interactive=True),
        gr.update(value=msg, visible=True))


def run_typed(text, gene_k, min_prob, show_all, progress=gr.Progress()):
    return _run(engine.parse_typed(text), gene_k, min_prob, show_all, progress)


def run_file(file, gene_k, min_prob, show_all, progress=gr.Progress()):
    if file is None:
        return (render.results_html(pd.DataFrame()),
                gr.DownloadButton(interactive=False),
                gr.DownloadButton(interactive=False),
                gr.update(value="Choose a file first.", visible=True))
    path = file.name if hasattr(file, "name") else str(file)
    try:
        if _looks_like_cgc(path):
            items = engine.parse_cgc(path)
            kind = "dbCAN CGC table"
        elif path.lower().endswith((".csv", ".tsv")):
            items = engine.parse_table(path)
            kind = "table"
        else:
            items = engine.parse_typed(open(path, encoding="utf-8", errors="replace").read())
            kind = "one locus per line"
    except Exception as e:
        return (render.results_html(pd.DataFrame()),
                gr.DownloadButton(interactive=False),
                gr.DownloadButton(interactive=False),
                gr.update(value=f"**Could not read that file.** {e}\n\n"
                                f"See *How to use this* below for the three accepted "
                                f"layouts.", visible=True))
    return _run(items, gene_k, min_prob, show_all, progress)


def _looks_like_cgc(path: str) -> bool:
    """A dbCAN cgc_standard.out has a CGC# column; a plain list of loci does not."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    return "CGC#" in head or ("Gene Type" in head and "Protein Family" in head)


with gr.Blocks(theme=THEME, css=CSS, head=SORT_JS, js=FORCE_LIGHT,
               title="subFinder", analytics_enabled=False) as demo:
    gr.HTML(HERO)

    with gr.Row(equal_height=False, elem_id="sf-console"):
        with gr.Column(scale=3, elem_id="sf-inputs"):
            with gr.Tabs():
                with gr.Tab("Type or paste"):
                    text = gr.Textbox(
                        label="", value=EXAMPLES, lines=5, max_lines=18,
                        placeholder="One locus per line. Optionally name it first, "
                                    "separated by a tab.",
                        info="One locus per line: comma-separated gene labels, "
                             "pipes for alternative annotations of the same gene. "
                             "An optional name before a tab is carried into the results.")
                    go_text = gr.Button("Predict", variant="primary", size="lg")
                    gr.HTML(CHEATSHEET)
                with gr.Tab("Upload a file"):
                    upload = gr.File(
                        label="", elem_id="sf-upload", file_types=[".out", ".tsv", ".txt", ".csv"],
                        file_count="single")
                    gr.Markdown(
                        "Accepts a **CSV** or **TSV** (gene string quoted, see the template "
                        "below), a dbCAN **`cgc_standard.out`** table, or a plain text file "
                        f"with one locus per line. Up to {BATCH_CAP:,} loci per run.")
                    go_file = gr.Button("Predict", variant="primary", size="lg")
                    with gr.Accordion("Need an example? Download one and copy its shape",
                                      open=False):
                        with gr.Row():
                            gr.DownloadButton("Template CSV", value=str(EX/"TEMPLATE_upload_me.csv"),
                                              size="sm", elem_classes="sf-dl")
                            gr.DownloadButton("100 loci, CSV", value=str(EX/"example_100_loci.csv"),
                                              size="sm", elem_classes="sf-dl")
                            gr.DownloadButton("100 loci, TSV-style text",
                                              value=str(EX/"example_100_loci.txt"),
                                              size="sm", elem_classes="sf-dl")
                        with gr.Row():
                            gr.DownloadButton("dbCAN CGC table",
                                              value=str(EX/"example_100_loci_cgc_standard.out"),
                                              size="sm", elem_classes="sf-dl")
                            gr.DownloadButton("that CGC table as CSV",
                                              value=str(EX/"example_100_loci_from_cgc.csv"),
                                              size="sm", elem_classes="sf-dl")
                            gr.DownloadButton("100 uncurated loci",
                                              value=str(EX/"example_unsupervised_100.csv"),
                                              size="sm", elem_classes="sf-dl")

        with gr.Column(scale=2, elem_id="sf-dials"):
            gr.HTML('<p class="sf-label">What to show</p>')
            gene_k = gr.Slider(1, 5, value=3, step=1,
                               label="Signature genes per substrate")
            min_prob = gr.Slider(
                0.0, 0.5, value=0.05, step=0.01,
                label="Explain substrates above this probability",
                info="Below it, the probability and p-value are still reported but the "
                     "genes are not. Gene attributions for a substrate that was never in "
                     "contention are unstable and easy to over-read.")
            show_all = gr.Checkbox(
                value=False, label="Explain all twelve regardless",
                info="Slower, and most of what it adds is noise. Here if you want it.")
            status = gr.Markdown(visible=False, elem_classes="sf-status")
            gr.HTML('<p class="sf-label" style="margin:20px 0 8px">Take it with you</p>')
            with gr.Row():
                csv = gr.DownloadButton("Results as CSV", size="sm",
                                        interactive=False, elem_classes="sf-dl")
                report = gr.DownloadButton("Results as HTML", size="sm",
                                           interactive=False, elem_classes="sf-dl")
            gr.HTML('<p class="sf-hint">The CSV carries every locus &times; substrate row at '
                    'full precision. The HTML is this page saved to a file &mdash; sorting '
                    'and filtering still work offline.</p>')

    with gr.Accordion("How to use this \u2014 read once, then never again",
                      open=False):
        gr.HTML(HELP_HTML)

    out = gr.HTML(render.results_html(pd.DataFrame()))

    gr.HTML(
        '<p class="sf-foot">Every row is one substrate for one locus, so a locus contributes '
        'twelve rows. <code>probability</code> is calibrated: across held-out data, loci called '
        'at 0.95 or above are correct 98.5&#37; of the time. <code>p_value</code> is a binomial '
        'test on that substrate\'s vote count against a coin toss, Bonferroni-corrected across '
        'the twelve. A call is withheld when fewer than two genes in the locus are ones the '
        'model recognises.</p>')

    args = [gene_k, min_prob, show_all]
    go_text.click(run_typed, [text] + args, [out, csv, report, status])
    go_file.click(run_file, [upload] + args, [out, csv, report, status])

if __name__ == "__main__":
    demo.launch()
