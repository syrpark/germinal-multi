"""Pins that binder MSAs are cached per FRAMEWORK, not per binder sequence.

CDR columns are gapped right after the search and redesign only substitutes
within them, so every candidate in a run shares one homolog block. The query
row may not be shared: it must equal the input JSON's `sequence`, so each
reuse writes its own on top.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from germinal.filters import af3

# The shipped scFv layout: 47 CDR positions over a 242-residue binder.
CDRS = [25, 26, 27, 28, 29, 30, 31, 32, 50, 51, 52, 53, 54, 55, 56, 57,
        96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
        161, 162, 163, 164, 165, 166, 184, 185, 186,
        223, 224, 225, 226, 227, 228, 229, 230, 231]
N = 242


def _binder(cdr_fill, fw_fill="A"):
    return "".join(cdr_fill if i in set(CDRS) else fw_fill for i in range(N))


def test_same_framework_different_cdrs_is_one_key():
    a = _binder("C")
    b = _binder("W")
    assert a != b, "fixture broken: the two binders must differ"
    ka = af3.binder_framework_key(a, CDRS)
    kb = af3.binder_framework_key(b, CDRS)
    assert ka == kb, f"{ka} != {kb} -- a shared framework must share a key"
    print("PASS: binders differing only at CDRs share one cache key")


def test_a_framework_mutation_changes_the_key():
    a = _binder("C")
    mutated = "M" + a[1:]          # position 0 is framework
    assert af3.binder_framework_key(a, CDRS) != af3.binder_framework_key(
        mutated, CDRS
    ), "a framework mutation must miss the cache"
    print("PASS: a framework mutation changes the key")


def test_different_cdr_layouts_do_not_collide():
    seq = _binder("C")
    other = [p + 1 for p in CDRS]
    assert af3.binder_framework_key(seq, CDRS) != af3.binder_framework_key(
        seq, other
    ), "a different CDR slicing must not reuse this entry"
    print("PASS: a different CDR layout gets a different key")


def test_reuse_swaps_in_the_new_query_row():
    """The whole point: homologs reused, query row replaced."""
    first, second = _binder("C"), _binder("W")
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        os.makedirs(cache)
        src = os.path.join(tmp, "first.a3m")
        with open(src, "w") as fh:
            fh.write(f">query\n{first}\n>h1\n{'D' * N}\n>h2\n{'E' * N}\n")

        key = af3.binder_framework_key(first, CDRS)
        af3.store_framework_msa(cache, key, src)

        dest = os.path.join(tmp, "second.a3m")
        assert af3.load_cached_framework_msa(cache, key, second, dest) is True

        rows = [l.strip() for l in open(dest)
                if l.strip() and not l.startswith(">")]
        assert rows[0] == second, "query row must be the NEW binder"
        assert rows[1] == "D" * N and rows[2] == "E" * N, "homologs preserved"

        # and the result survives the strict mask, which checks the query
        af3.mask_binder_cdrs(dest, CDRS, "d", binder_seq=second)
    print("PASS: cached homologs reused under a fresh query row")


def test_a_length_mismatch_declines_the_cache():
    """A wrong-length entry must cost a search, never a bad fold."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        os.makedirs(cache)
        src = os.path.join(tmp, "s.a3m")
        with open(src, "w") as fh:
            fh.write(f">q\n{'A' * 10}\n>h1\n{'C' * 10}\n")
        af3.store_framework_msa(cache, "deadbeef", src)
        dest = os.path.join(tmp, "d.a3m")
        assert af3.load_cached_framework_msa(
            cache, "deadbeef", "A" * 99, dest
        ) is False
    print("PASS: a length mismatch declines the cache instead of folding wrong")


def test_a_miss_is_reported_not_raised():
    with tempfile.TemporaryDirectory() as tmp:
        assert af3.load_cached_framework_msa(tmp, "nope", "AAA", os.path.join(tmp, "x.a3m")) is False
        assert af3.load_cached_framework_msa(None, "nope", "AAA", os.path.join(tmp, "x.a3m")) is False
    print("PASS: an empty or absent cache is a miss, not a crash")


def test_the_cache_collapses_a_whole_run():
    """A whole run's worth of designs: 60 binder sequences, one framework."""
    binders = [_binder(c) for c in "CDEFGHIKLMNPQRSTVWY"] * 3 + [_binder("C")] * 3
    assert len(binders) == 60
    keys = {af3.binder_framework_key(b, CDRS) for b in binders}
    assert len(keys) == 1, f"expected 1 search, got {len(keys)}"
    print(f"PASS: {len(binders)} binder sequences -> {len(keys)} colabfold_search")


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("\nAll framework-cache tests passed.")


def test_both_legs_of_one_design_search_the_binder_once():
    """The af3 regression: the binder MSA was keyed on input_json_data["name"].

    That name embeds the target leg, so the key must come from the binder
    sequence and its framework instead.
    """
    import germinal.filters.af3 as af3

    searches = []

    def fake_search(seq, name, out_dir, use_metagenomic_db=False):
        searches.append((seq, name))
        rel = os.path.join("msas", f"{name}.a3m")
        full = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as fh:
            fh.write(f">query\n{seq}\n>homolog\n{'A' * len(seq)}\n")
        return rel

    real = af3.call_generate_colabfold_msa_with_timeout
    af3.call_generate_colabfold_msa_with_timeout = fake_search
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "msa_cache")
            os.makedirs(cache)
            binder = "QVQLVESGGGLVQ"
            cdrs = [3, 4]
            # Leg 0 and leg 1 of ONE design: different output dirs, as the
            # two target legs genuinely have, same binder.
            paths = []
            for leg in range(2):
                out = os.path.join(tmp, f"leg{leg}")
                os.makedirs(out)
                rel, reused = af3.resolve_binder_msa(
                    binder_seq=binder, output_dir=out,
                    msa_method="colabfold_remote", msa_cache_dir=cache,
                    cdr_positions=cdrs,
                )
                paths.append((rel, reused))
            assert len(searches) == 1, (
                f"two legs of one design ran {len(searches)} binder searches: "
                f"{[n for _, n in searches]}"
            )
            assert paths[1][1] is True, "the second leg was not served from cache"
            assert paths[0][0] == paths[1][0], f"legs disagree on filename: {paths}"
            assert "binder_" in paths[0][0], (
                f"MSA name {paths[0][0]!r} is not keyed on the binder"
            )

            # A different design sharing the framework must also reuse.
            mutated = "QVQWWESGGGLVQ"  # differs ONLY at CDR positions 3,4
            out = os.path.join(tmp, "other_design")
            os.makedirs(out)
            af3.resolve_binder_msa(
                binder_seq=mutated, output_dir=out,
                msa_method="colabfold_remote", msa_cache_dir=cache,
                cdr_positions=cdrs,
            )
            assert len(searches) == 1, (
                f"a second design sharing the framework re-searched: "
                f"{len(searches)} searches total"
            )
            print("PASS: one binder search serves both legs and both designs")
    finally:
        af3.call_generate_colabfold_msa_with_timeout = real
