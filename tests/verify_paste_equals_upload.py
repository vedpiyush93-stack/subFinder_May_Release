"""Prove the text box and the file upload are the same door.

Run with:  pytest tests/verify_paste_equals_upload.py -v -s

The app offers two ways in. They used to be read by different code: the upload branched on
the file extension and the text box assumed one locus per line, so pasting a dbCAN table --
the obvious thing to do with a dbCAN table -- turned every gene row into its own locus and
the header into a locus called ``CGC#``. Nothing raised; the user just got a column of
"no call" and no reason.

Both now go through ``engine.parse_any``, which decides on content rather than on a filename.
These tests pin that: every accepted layout is recognised, a batch behaves like a batch, and
pasting a file's contents produces the same loci, the same predictions and the same
downloadable CSV as uploading that file.
"""
from __future__ import annotations
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "spaces" / "subfinder"
EX = SPACE / "examples"

pytestmark = pytest.mark.skipif(
    not (SPACE / "engine.py").exists(),
    reason="the Space bundle is not present in this checkout")


@pytest.fixture(scope="module")
def engine():
    sys.path.insert(0, str(SPACE))
    spec = importlib.util.spec_from_file_location("space_engine", SPACE / "engine.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Every file the app ships as an example, and the layout it should be recognised as.
BATCH_FILES = [
    ("example_100_loci.csv", "table with a header row", 100),
    ("example_100_loci.txt", "one locus per line", 100),
    ("example_100_loci_from_cgc.csv", "table with a header row", 100),
    ("example_100_loci_cgc_standard.out", "dbCAN CGC table", 100),
    ("example_unsupervised_100.csv", "table with a header row", 100),
    ("TEMPLATE_upload_me.csv", "table with a header row", 7),
]


@pytest.mark.parametrize("fname,kind,n", BATCH_FILES)
def test_every_example_is_recognised_from_its_content(engine, fname, kind, n):
    """Detection is on content: no extension is consulted, and none is needed."""
    text = (EX / fname).read_text(encoding="utf-8", errors="replace")
    items, got_kind = engine.parse_any(text)
    assert got_kind == kind, f"{fname} read as {got_kind!r}"
    assert len(items) == n, f"{fname} gave {len(items)} loci, expected {n}"
    assert all(name and seq for name, seq in items), f"{fname} produced an empty field"


@pytest.mark.parametrize("fname,kind,n", BATCH_FILES)
def test_paste_matches_upload(engine, fname, kind, n):
    """Pasting a file's contents and uploading the file give the same loci."""
    path = EX / fname
    text = path.read_text(encoding="utf-8", errors="replace")
    pasted, _ = engine.parse_any(text)
    with open(path, encoding="utf-8", errors="replace") as fh:
        uploaded, _ = engine.parse_any(fh.read())
    assert pasted == uploaded


def test_batch_predictions_and_csv_are_identical(engine):
    """A 100-locus batch predicts the same and downloads the same, either way in."""
    path = EX / "example_100_loci_cgc_standard.out"
    text = path.read_text(encoding="utf-8", errors="replace")

    pasted, kind_p = engine.parse_any(text)
    with open(path, encoding="utf-8", errors="replace") as fh:
        uploaded, kind_u = engine.parse_any(fh.read())
    assert kind_p == kind_u == "dbCAN CGC table"
    assert len(pasted) == 100

    df_p = engine.predict_frame(pasted)
    df_u = engine.predict_frame(uploaded)
    assert len(df_p) == 100 * 12, "one row per locus x substrate"
    assert df_p.equals(df_u), "the two routes produced different predictions"

    # the artifacts a user actually takes away
    h_p = hashlib.sha256(Path(engine.to_csv(df_p)).read_bytes()).hexdigest()
    h_u = hashlib.sha256(Path(engine.to_csv(df_u)).read_bytes()).hexdigest()
    assert h_p == h_u, "the downloaded CSV differs between pasting and uploading"
    print(f"\n  100 loci, both routes: {len(df_p)} rows, identical CSV ({h_p[:12]}…)")


def test_same_loci_expressed_three_ways_agree(engine):
    """The same batch as a CGC table, as a CSV, and as one-locus-per-line."""
    cgc = engine.parse_any((EX / "example_100_loci_cgc_standard.out")
                           .read_text(encoding="utf-8", errors="replace"))[0]
    csv = engine.parse_any((EX / "example_100_loci_from_cgc.csv")
                           .read_text(encoding="utf-8", errors="replace"))[0]
    assert len(cgc) == len(csv) == 100
    # same loci, same gene strings -- the CSV is that CGC table already converted
    assert dict(cgc) == dict(csv), "the CGC table and its CSV form disagree"

    # and one locus per line, built from those same strings
    lines = "\n".join(f"{n}\t{s}" for n, s in cgc)
    plain = engine.parse_any(lines)[0]
    assert plain == cgc
    print(f"\n  CGC table == its CSV == one-per-line, all {len(cgc)} loci")


def test_a_pasted_cgc_table_is_not_read_line_by_line(engine):
    """The regression this file exists for."""
    text = (EX / "example_100_loci_cgc_standard.out").read_text(encoding="utf-8")
    items, kind = engine.parse_any(text)
    assert kind == "dbCAN CGC table"
    assert not any(name.startswith("CGC#") for name, _ in items), \
        "the header row was read as a locus"
    assert len(items) < len(text.splitlines()) / 2, \
        "looks like one locus per gene row rather than per cluster"
    readable = [engine.informative_tokens(s) for _, s in items]
    assert sum(r >= 2 for r in readable) > len(items) * 0.5, \
        "most loci have nothing the model can read -- the table was misparsed"
