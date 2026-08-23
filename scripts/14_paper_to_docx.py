#!/usr/bin/env python3
"""Convert the manuscript and supplement to .docx for collaborators who edit in Word.

The paper is written against the OUP submission class, which pandoc does not know:
``\\journaltitle``, ``\\authormark``, ``\\abstract{...}`` as an argument rather than an
environment, and a handful of other commands that only that class defines. Handing
main.tex to pandoc unmodified silently drops the title block and the abstract.

So this script rewrites the source into a form pandoc reads correctly -- flattening the
``\\input``s, swapping the document class, and turning the OUP metadata commands into
standard ones -- and then converts. The rewrite happens on a copy in a temporary
directory; nothing under paper/ is touched.

    python3 scripts/14_paper_to_docx.py            # -> paper/docx/*.docx

What survives the trip: section structure, every table, every figure with its caption,
italics and bold, math (as Word equations), and the citations, rendered author-date from
reference.bib. What does not: the journal's two-column layout and its exact typography,
neither of which matters for editing.
"""
from __future__ import annotations
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
OUT = PAPER / "docx"

# Commands the OUP class defines that carry no meaning in a Word draft. Dropped whole,
# argument included.
DROP_WITH_ARG = [
    "journaltitle", "DOI", "copyrightyear", "pubyear", "access", "appnotes",
    "authormark", "theoremstyle", "graphicspath", "WarningFilter", "hypersetup",
]
# Commands that are pure typesetting control; dropped, no argument to consume.
DROP_BARE = [
    r"\\maketitle", r"\\hbadness=\d+", r"\\vbadness=\d+",
    r"\\setlength\{\\emergencystretch\}\{[^}]*\}",
    r"\\usepackage\{silence\}", r"\\newtheorem\{[^}]*\}(\[[^]]*\])?\{[^}]*\}(\[[^]]*\])?",
]


def strip_command(text: str, name: str) -> str:
    """Remove ``\\name{...}`` including a brace-balanced argument."""
    out, i = [], 0
    pat = "\\" + name
    while True:
        j = text.find(pat, i)
        if j == -1:
            out.append(text[i:])
            return "".join(out)
        # not a prefix of a longer command name
        after = j + len(pat)
        if after < len(text) and (text[after].isalpha()):
            out.append(text[i:after])
            i = after
            continue
        out.append(text[i:j])
        k = _skip_optional(text, after)
        if k < len(text) and text[k] == "{":
            depth, k = 1, k + 1
            while k < len(text) and depth:
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                k += 1
        i = k


def flatten_inputs(text: str, base: Path) -> str:
    """Inline ``\\input{...}`` so the macro definitions and tables reach pandoc."""
    def sub(m: re.Match) -> str:
        p = base / m.group(1)
        if not p.suffix:
            p = p.with_suffix(".tex")
        if not p.exists():
            raise SystemExit(f"\\input target missing: {p}")
        return flatten_inputs(p.read_text(encoding="utf-8"), base)
    return re.sub(r"\\input\{([^}]*)\}", sub, text)


def _skip_optional(text: str, k: int) -> int:
    """Step past whitespace and any ``[...]`` optional argument starting at k."""
    while k < len(text) and text[k] in " \t\n":
        k += 1
    if k < len(text) and text[k] == "[":
        depth = 1
        k += 1
        while k < len(text) and depth:
            if text[k] == "[":
                depth += 1
            elif text[k] == "]":
                depth -= 1
            k += 1
        while k < len(text) and text[k] in " \t\n":
            k += 1
    return k


def extract_braced(text: str, name: str) -> str | None:
    """Return the brace-balanced argument of ``\\name{...}``, or None.

    Handles ``\\title[running head]{full title}``: the optional argument has to be
    stepped over, not scanned character by character, or the first letter inside the
    brackets aborts the search and the document silently loses its title.
    """
    j = text.find("\\" + name)
    if j == -1:
        return None
    k = _skip_optional(text, j + len(name) + 1)
    if k >= len(text) or text[k] != "{":
        return None
    depth, start, k = 1, k + 1, k + 1
    while k < len(text) and depth:
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
        k += 1
    return text[start:k - 1]


def _number_captions(text: str, *, supplement: bool) -> str:
    """Number the floats, then resolve every \\ref to the number it points at.

    pandoc drops \\ref entirely, so the Word draft arrived saying "Figure  shows"
    and "Table  gives" with a blank where the number belongs. Numbering the captions
    here means the mapping is already in hand, so the references can be filled in
    from the same pass.
    """
    prefix = "S" if supplement else ""
    counts = {"figure": 0, "table": 0}
    ref = {}                       # label -> "Figure 3" / "Table S1"
    out, i, env, pending = [], 0, None, None
    while i < len(text):
        for name in ("figure", "table"):
            if text.startswith(f"\\begin{{{name}}}", i):
                env, pending = name, None
                break
        if text.startswith("\\caption{", i) and env:
            counts[env] += 1
            pending = f"{env.capitalize()} {prefix}{counts[env]}"
            out.append("\\caption{" + pending + ". ")
            i += len("\\caption{")
            env = None
            continue
        if text.startswith("\\label{", i) and pending:
            j = text.index("}", i)
            ref[text[i + len("\\label{"):j]] = pending
            out.append(text[i:j + 1])
            i = j + 1
            continue
        out.append(text[i])
        i += 1
    text = "".join(out)

    def fill(m: re.Match) -> str:
        target = ref.get(m.group(1))
        if target is None:
            return m.group(0)
        # the prose already writes "Figure~\ref{...}", so emit the bare number
        return target.split()[-1]
    return re.sub(r"\\ref\{([^}]*)\}", fill, text)


def prepare(src: Path, work: Path) -> Path:
    text = flatten_inputs(src.read_text(encoding="utf-8"), src.parent)

    title = extract_braced(text, "title") or src.stem
    abstract = extract_braced(text, "abstract")
    keywords = extract_braced(text, "keywords")

    for name in DROP_WITH_ARG:
        text = strip_command(text, name)
    for pat in DROP_BARE:
        text = re.sub(pat, "", text)
    text = strip_command(text, "title")
    text = strip_command(text, "abstract")
    text = strip_command(text, "keywords")

    text = text.replace(
        "\\documentclass[unnumsec,webpdf,contemporary,large,namedate]{oup-authoring-template}",
        "\\documentclass[11pt]{article}\n\\usepackage{graphicx}\n\\usepackage{booktabs}")

    # The starred float environments mean "span both columns", which has no meaning in a
    # single-column Word document. pandoc does not read them at all: a table* is dropped
    # silently, taking the leaderboard with it. Unstar them.
    for env in ("table", "figure"):
        text = text.replace(f"\\begin{{{env}*}}", f"\\begin{{{env}}}")
        text = text.replace(f"\\end{{{env}*}}", f"\\end{{{env}}}")

    # \centering inside a float makes pandoc wrap the figure in a two-column table
    # whose columns are 2.75in wide, so every plot came out at less than half the
    # text width. Pandoc centres figures on its own, so the command is redundant here.
    text = re.sub(r"\\centering\s*", "", text)

    # Number the captions. The journal class does this at typesetting time, but pandoc
    # emits captions as plain paragraphs, so a Word draft arrives with six unlabelled
    # figures and nobody can say "see Figure 3" in a comment.
    text = _number_captions(text, supplement=("supplement" in src.name))

    # booktabs partial rules draw nothing in Word, and pandoc spills their arguments
    # into the header row as literal text ("(lr) 2-4").
    text = re.sub(r"\\cmidrule(\([a-z]{1,2}\))?\{[\d-]+\}", "", text)

    # The abstract and keywords come back as an ordinary environment and a paragraph, so
    # they land in the Word file where a reader expects them.
    head = ["\\title{" + title + "}", "\\author{}", "\\date{}"]
    body_open = "\\begin{document}\n\\maketitle\n"
    if abstract:
        body_open += "\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n"
    if keywords:
        body_open += "\n\\textbf{Keywords:} " + keywords + "\n"
    text = text.replace("\\begin{document}", "\n".join(head) + "\n" + body_open, 1)

    dst = work / src.name
    dst.write_text(text, encoding="utf-8")
    return dst


def convert(src: Path, bib: Path, resource: Path, out: Path) -> None:
    cmd = [
        "pandoc", str(src),
        "--from", "latex+raw_tex",
        "--to", "docx",
        "--citeproc",
        "--bibliography", str(bib),
        "--resource-path", f"{resource}:{resource / 'Fig'}",
        "--number-sections",
        # citeproc emits the bibliography with no heading; name it so the Word
        # file has an obvious References section rather than a bare list.
        "--metadata", "reference-section-title=References",
        "--wrap", "none",
        "--output", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"pandoc failed on {src.name}")
    # pandoc warns rather than fails on an unresolved citation; surface those.
    for line in r.stderr.splitlines():
        if "citation" in line.lower() or "not found" in line.lower():
            print(f"  [pandoc] {line.strip()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["main", "supplement"], help="convert just one")
    args = ap.parse_args()

    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc not found. Install it: brew install pandoc")

    OUT.mkdir(parents=True, exist_ok=True)
    targets = [t for t in ("main", "supplement") if args.only in (None, t)]

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in targets:
            src = PAPER / f"{name}.tex"
            prepared = prepare(src, work)
            dst = OUT / f"subFinder_{name}.docx"
            convert(prepared, PAPER / "reference.bib", PAPER, dst)
            print(f"[14] {src.name} -> {dst.relative_to(ROOT)}  "
                  f"({dst.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
