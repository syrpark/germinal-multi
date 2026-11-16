"""Pins that binder-CDR masking FAILS LOUDLY rather than silently skipping.

mask_cdrs_in_a3m stays a no-op on empty input, which is the right contract
for a low-level rewrite. This covers the policy wrapper, mask_binder_cdrs,
which must raise both when nothing is masked and when the wrong columns are:
a stale cdr_positions is strictly worse than not masking, and as invisible.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from germinal.filters import af3
from germinal.filters.af3 import CdrMaskError


def _a3m(tmp, rows, name="binder.a3m"):
    path = os.path.join(tmp, name)
    with open(path, "w") as fh:
        for i, seq in enumerate(rows):
            fh.write(f">s{i}\n{seq}\n")
    return path


def _rows(path):
    return [
        l.strip() for l in open(path)
        if l.strip() and not l.startswith(">") and not l.startswith("#")
    ]


def _raises(fn, what):
    try:
        fn()
    except CdrMaskError:
        return True
    except Exception as e:
        raise AssertionError(f"{what}: wrong exception {type(e).__name__}: {e}")
    raise AssertionError(f"{what}: did not raise -- this is a silent no-op")


def test_empty_cdr_positions_raises():
    with tempfile.TemporaryDirectory() as tmp:
        p = _a3m(tmp, ["ACDEFGHIKL", "MNPQRSTVWY"])
        _raises(lambda: af3.mask_binder_cdrs(p, [], "d"), "empty cdr_positions")
        _raises(lambda: af3.mask_binder_cdrs(p, None, "d"), "None cdr_positions")
    print("PASS: empty/None cdr_positions raises instead of skipping")


def test_missing_binder_msa_raises():
    _raises(lambda: af3.mask_binder_cdrs(None, [1, 2], "d"), "no path")
    _raises(lambda: af3.mask_binder_cdrs("", [1, 2], "d"), "empty path")
    _raises(lambda: af3.mask_binder_cdrs("/no/such.a3m", [1, 2], "d"), "missing file")
    print("PASS: a missing binder MSA raises instead of skipping")


def test_positions_outside_the_binder_raise():
    """The stale-config case: cdr_lengths describe a different binder."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _a3m(tmp, ["A" * 10, "C" * 10])
        _raises(lambda: af3.mask_binder_cdrs(p, [2, 3, 10], "d"), "pos == len")
        _raises(lambda: af3.mask_binder_cdrs(p, [2, 3, 240], "d"), "pos > len")
    print("PASS: CDR positions outside the binder raise")


def test_query_row_must_be_the_binder_being_folded():
    with tempfile.TemporaryDirectory() as tmp:
        p = _a3m(tmp, ["ACDEFGHIKL", "MNPQRSTVWY"])
        _raises(
            lambda: af3.mask_binder_cdrs(p, [2, 3], "d", binder_seq="WWWWWWWWWW"),
            "query != binder_seq",
        )
        # the matching case must go through
        n = af3.mask_binder_cdrs(p, [2, 3], "d", binder_seq="ACDEFGHIKL")
        assert n == 1, n
    print("PASS: query row is validated against the binder being folded")


def test_valid_mask_applies_and_leaves_the_query_alone():
    with tempfile.TemporaryDirectory() as tmp:
        p = _a3m(tmp, ["ACDEFGHIKL", "MNPQRSTVWY", "WWWWWWWWWW"])
        n = af3.mask_binder_cdrs(p, [2, 3], "d", binder_seq="ACDEFGHIKL")
        rows = _rows(p)
        assert n == 2, n
        assert rows[0] == "ACDEFGHIKL", rows[0]
        assert rows[1] == "MN--RSTVWY", rows[1]
        assert rows[2] == "WW--WWWWWW", rows[2]
    print("PASS: CDR columns gapped in homologs, query untouched")


def test_masking_is_idempotent():
    """A cached a3m is re-masked on every cache hit; that must be a no-change."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _a3m(tmp, ["ACDEFGHIKL", "MNPQRSTVWY"])
        af3.mask_binder_cdrs(p, [2, 3], "d", binder_seq="ACDEFGHIKL")
        first = _rows(p)
        af3.mask_binder_cdrs(p, [2, 3], "d", binder_seq="ACDEFGHIKL")
        assert _rows(p) == first, (_rows(p), first)
    print("PASS: re-masking a cached MSA changes nothing")


def test_a_stub_warns_but_does_not_raise():
    """Query-only is legitimate; it must be visible, not fatal."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _a3m(tmp, ["ACDEFGHIKL"])
        n = af3.mask_binder_cdrs(p, [2, 3], "d", binder_seq="ACDEFGHIKL")
        assert n == 0, n
        assert _rows(p) == ["ACDEFGHIKL"], _rows(p)
    print("PASS: a single-sequence stub warns and returns 0")


def test_insertions_do_not_shift_the_query_length_check():
    """Lowercase insertions occupy no query column, so they must not count."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _a3m(tmp, ["ACDEFGHIKL", "MNpqPQRSTVWY"])
        n = af3.mask_binder_cdrs(p, [9], "d", binder_seq="ACDEFGHIKL")
        assert n == 1, n
    print("PASS: insertions are excluded from the binder-length check")


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("\nAll strict CDR-mask tests passed.")
