"""Tests for ``utils.idx_from_ranges``, which resolves residue specifications.

The returned indices are positions within a SINGLE chain, so a specification
spanning two chains is ambiguous and raises. These tests pin that, and pin the
shapes the pipeline actually uses -- notably 'B30-B33', where the chain letter
appears on both endpoints of a range.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from germinal.utils.utils import idx_from_ranges  # noqa: E402


# --- Shapes the pipeline actually produces ---

def test_range_with_chain_on_both_endpoints():
    # The form target configs use. The old replace() removed both letters; a
    # naive per-endpoint parse would try int("B33") and raise.
    assert idx_from_ranges("B30-B33", "B") == [29, 30, 31, 32]


def test_range_without_chain_prefix():
    assert idx_from_ranges("30-33", "B") == [29, 30, 31, 32]


def test_single_residue_with_and_without_prefix():
    assert idx_from_ranges("B45", "B") == [44]
    assert idx_from_ranges("45", "B") == [44]


def test_comma_separated_list():
    assert idx_from_ranges("B1,B5,B9", "B") == [0, 4, 8]


def test_empty_specification_yields_nothing():
    # score_interface returns "" when no residue is within its 4.0 A cutoff --
    # a routine outcome for a dissociated prediction, not an error.
    assert idx_from_ranges("", "B") == []
    assert idx_from_ranges("  ", "B") == []


def test_offset_applies_to_both_branches():
    # The offset used to be applied only to parts carrying a chain letter, so
    # the same positions resolved differently depending on how they were written.
    assert idx_from_ranges("B30-B33", "B", offset=5) == [34, 35, 36, 37]
    assert idx_from_ranges("30-33", "B", offset=5) == [34, 35, 36, 37]
    assert idx_from_ranges("B45", "B", offset=5) == [49]
    assert idx_from_ranges("45", "B", offset=5) == [49]


# --- Multi-chain targets ---

def test_multi_chain_target_accepts_either_chain_alone():
    # create_starting_structure supports target_chain="A,C", so both are valid
    # accepted chains -- as long as one specification does not mix them.
    assert idx_from_ranges("A43-A50", "A,C") == list(range(42, 50))
    assert idx_from_ranges("C10", "A,C") == [9]


def test_unaccepted_chain_raises_instead_of_mapping_onto_the_first():
    # The original bug: 'C133' against target_chain="A" silently became index
    # 133 of chain A.
    try:
        idx_from_ranges("C133", "A")
    except ValueError as exc:
        assert "C" in str(exc)
        return
    raise AssertionError("a foreign chain letter was silently accepted")


def test_specification_spanning_two_chains_raises():
    # Ambiguous: the function cannot know how many residues precede chain C.
    try:
        idx_from_ranges("A43,C12", "A,C")
    except ValueError as exc:
        assert "spans chains" in str(exc)
        return
    raise AssertionError("a multi-chain specification was silently resolved")


def test_chain_accepts_a_list_as_well_as_a_comma_string():
    assert idx_from_ranges("C10", ["A", "C"]) == idx_from_ranges("C10", "A,C")


# --- chain=None, used by interface_cdrs ---

def test_chain_none_accepts_whichever_single_chain_is_used():
    # interface_cdrs passes chain=None, so a non-B binder chain must not
    # start raising.
    assert idx_from_ranges("B5", None) == [4]
    assert idx_from_ranges("H5", None) == [4]
    assert idx_from_ranges("H5,H9", None) == [4, 8]


def test_chain_none_still_rejects_a_mixed_specification():
    try:
        idx_from_ranges("B5,H9", None)
    except ValueError:
        return
    raise AssertionError("chain=None must still reject a multi-chain mix")


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
