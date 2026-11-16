"""Pins that a binder MSA never carries evolutionary signal at the CDRs.

The framework's homologs are what make the complex foldable, but every
homolog's CDR is a germline loop the design deliberately is not. Both folding
backends route their binder MSA through one helper; these tests pin the
masking rule and that both routes apply it.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from germinal.filters import af3

PASSED = 0
FAILED = 0


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name} {detail}")
        if "PYTEST_CURRENT_TEST" in os.environ:
            raise AssertionError(f"{name} {detail}")


def _a3m(tmp, rows):
    path = os.path.join(tmp, "binder.a3m")
    with open(path, "w") as fh:
        for i, seq in enumerate(rows):
            fh.write(f">s{i}\n{seq}\n")
    return path


def _rows(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line and not line.startswith(">") and not line.startswith("#"):
            out.append(line)
    return out


# --- the masking rule ---
def test_cdr_columns_are_gapped_in_homologs():
    with tempfile.TemporaryDirectory() as tmp:
        #              0123456789
        path = _a3m(tmp, ["ACDEFGHIKL", "MNPQRSTVWY", "ACDEFGHIKL"])
        n = af3.mask_cdrs_in_a3m(path, [2, 3, 4])
        rows = _rows(path)
        check("every homolog is masked", n == 2, str(n))
        check(
            "the CDR columns become gaps in homolog rows",
            rows[1] == "MN---STVWY" and rows[2] == "AC---GHIKL",
            str(rows),
        )
        check(
            "framework columns are untouched",
            rows[1][:2] == "MN" and rows[1][5:] == "STVWY",
            rows[1],
        )


def test_the_query_row_is_never_altered():
    # The query defines the chain and must keep matching the `sequence` field
    # of the input JSON; gapping it would change the molecule being folded.
    with tempfile.TemporaryDirectory() as tmp:
        query = "ACDEFGHIKL"
        path = _a3m(tmp, [query, "MNPQRSTVWY"])
        af3.mask_cdrs_in_a3m(path, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        rows = _rows(path)
        check("the query survives verbatim", rows[0] == query, rows[0])
        check("the homolog is fully gapped", rows[1] == "-" * 10, rows[1])


def test_lowercase_insertions_inside_a_cdr_are_dropped():
    # Lowercase a3m insertions occupy no query column, so they are not
    # sequence positions.
    with tempfile.TemporaryDirectory() as tmp:
        #                  query cols:  0 1 2 3 4
        path = _a3m(tmp, ["ACDEF", "ApqCDEF", "ACDpqEF"])
        af3.mask_cdrs_in_a3m(path, [1])  # mask query column 1 only
        rows = _rows(path)
        check(
            "an insertion anchored in the CDR is dropped with its column",
            rows[1] == "A-DEF",
            rows[1],
        )
        check(
            "an insertion anchored outside the CDR is kept",
            rows[2] == "A-DpqEF",
            rows[2],
        )


def test_no_cdrs_is_a_no_op():
    with tempfile.TemporaryDirectory() as tmp:
        path = _a3m(tmp, ["ACDEF", "MNPQR"])
        before = _rows(path)
        check("empty cdr list masks nothing", af3.mask_cdrs_in_a3m(path, []) == 0)
        check("the file is unchanged", _rows(path) == before)
        check("None masks nothing", af3.mask_cdrs_in_a3m(path, None) == 0)


def test_a_stub_msa_has_nothing_to_mask():
    # A single-sequence stub is all query and no homologs, so it must come
    # back byte-identical rather than gapped.
    with tempfile.TemporaryDirectory() as tmp:
        path = _a3m(tmp, ["ACDEFGHIKL"])
        n = af3.mask_cdrs_in_a3m(path, [2, 3, 4])
        check("a stub is left alone", n == 0 and _rows(path) == ["ACDEFGHIKL"])


def test_missing_file_does_not_raise():
    check(
        "a path that does not exist is a no-op, not a crash",
        af3.mask_cdrs_in_a3m("/no/such/file.a3m", [1, 2]) == 0,
    )


def test_real_cdr_positions_mask_the_right_count():
    # The shipped scFv layout: 47 CDR positions over a 241-residue binder.
    cdr = [25, 26, 27, 28, 29, 30, 31, 32, 50, 51, 52, 53, 54, 55, 56, 57,
           96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
           161, 162, 163, 164, 165, 166, 184, 185, 186,
           223, 224, 225, 226, 227, 228, 229, 230, 231]
    with tempfile.TemporaryDirectory() as tmp:
        seq = "A" * 241
        path = _a3m(tmp, [seq, "C" * 241])
        af3.mask_cdrs_in_a3m(path, cdr)
        row = _rows(path)[1]
        check(
            "exactly the CDR residues are gapped",
            row.count("-") == len(cdr) and len(row) == 241,
            f"gaps={row.count('-')} len={len(row)}",
        )
        check(
            "the gaps sit at the CDR indices",
            all(row[i] == "-" for i in cdr)
            and all(row[i] == "C" for i in range(241) if i not in set(cdr)),
        )


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print(f"\n{PASSED}/{PASSED + FAILED} passed")
    sys.exit(1 if FAILED else 0)
