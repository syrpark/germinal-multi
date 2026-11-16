"""Tests for the run-artifact write path and the run-level target MSA cache.

Covers three fixes:
  * _concat_and_write_locked no longer appends headerless rows onto a CSV it
    cannot parse -- it quarantines the file and starts a clean one.
  * final_config.yaml / run_summary.txt go through a locked, atomic write
    like every CSV in the same shared results root.
  * target MSAs are cached per run, keyed by sequence, instead of per AF3
    input directory where the cache could never hit across designs.
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from germinal.utils.io import IO, RunLayout
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
        # Needed because the only other exit-code guard is the __main__
        # block, which pytest never runs.
        if "PYTEST_CURRENT_TEST" in os.environ:
            raise AssertionError(f"{name} {detail}")


def _io(tmp):
    return IO(RunLayout.create(Path(tmp) / "run"))


# --- _concat_and_write_locked on an unparseable CSV ---
def test_unreadable_csv_is_quarantined_not_appended_to():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        csv_path = io.layout.root / "broken.csv"
        # The overlong row must come second: a wider FIRST row is silently
        # read as an index instead of raising ParserError.
        csv_path.write_text("a,b\n1,2\n3,4,5\n")

        row = pd.DataFrame([{"a": 10, "b": 20}])
        io._concat_and_write_locked(csv_path, row)

        out = pd.read_csv(csv_path)
        check(
            "unreadable csv is replaced by a clean, parseable file",
            list(out.columns) == ["a", "b"] and len(out) == 1,
            f"got columns={list(out.columns)} rows={len(out)}",
        )
        quarantined = list(io.layout.root.glob("broken.csv.corrupt.*"))
        check(
            "the unparseable file is moved aside, not destroyed",
            len(quarantined) == 1
            and quarantined[0].read_text() == "a,b\n1,2\n3,4,5\n",
        )


def test_readable_csv_still_concatenates():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        csv_path = io.layout.root / "fine.csv"
        pd.DataFrame([{"a": 1, "b": 2}]).to_csv(csv_path, index=False)
        io._concat_and_write_locked(csv_path, pd.DataFrame([{"a": 3, "b": 4}]))
        out = pd.read_csv(csv_path)
        check(
            "normal path still appends a row",
            len(out) == 2 and out["a"].tolist() == [1, 3],
            f"got {out.to_dict('list')}",
        )
        check(
            "no quarantine file is produced on the normal path",
            not list(io.layout.root.glob("fine.csv.corrupt.*")),
        )


def test_new_columns_widen_rather_than_misalign():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        csv_path = io.layout.root / "widen.csv"
        pd.DataFrame([{"a": 1}]).to_csv(csv_path, index=False)
        io._concat_and_write_locked(csv_path, pd.DataFrame([{"a": 2, "new": 9}]))
        out = pd.read_csv(csv_path)
        check(
            "a row with a new column widens the file",
            set(out.columns) == {"a", "new"} and len(out) == 2,
            f"got columns={list(out.columns)}",
        )


# --- locked text writes ---
def test_run_summary_is_written_and_leaves_no_temp_file():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        io.save_run_summary("done: 3 accepted")
        path = io.layout.root / "run_summary.txt"
        check("run_summary.txt is written", path.read_text() == "done: 3 accepted")
        check(
            "no .tmp.<pid> file is left behind",
            not list(io.layout.root.glob("run_summary.txt.tmp.*")),
        )


def test_run_summary_rewrite_replaces_rather_than_appends():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        io.save_run_summary("first")
        io.save_run_summary("second")
        check(
            "a second write replaces the first",
            (io.layout.root / "run_summary.txt").read_text() == "second",
        )


def test_save_run_config_roundtrips():
    import yaml

    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        io.save_run_config(
            {"seed": 1},
            {"target_chain": "A"},
            {"ipsae": {"operator": ">=", "value": 0.5}},
            {"ipsae": {"operator": ">=", "value": 0.55}},
        )
        loaded = yaml.safe_load(io.layout.final_config.read_text())
        check(
            "final_config.yaml roundtrips through the locked write",
            loaded == {
                "run_settings": {"seed": 1},
                "target_settings": {"target_chain": "A"},
                "initial_filters": {"ipsae": {"operator": ">=", "value": 0.5}},
                "final_filters": {"ipsae": {"operator": ">=", "value": 0.55}},
            },
            f"got {loaded}",
        )
        check(
            "the thresholds that decide acceptance are in the snapshot",
            "0.55" in io.layout.final_config.read_text(),
        )


# --- run-level target MSA cache ---
SEQ_A = "MKTAYIAKQRQISFVKSHFSRQ"
SEQ_B = "GAVLIPFMWCSTYNQDEKRHGG"


def test_cache_path_is_keyed_by_sequence():
    a1 = af3._target_msa_cache_path("/cache", SEQ_A)
    a2 = af3._target_msa_cache_path("/cache", SEQ_A)
    b = af3._target_msa_cache_path("/cache", SEQ_B)
    check("same sequence maps to the same cache entry", a1 == a2)
    check("different sequences map to different cache entries", a1 != b)
    check("cache entries live under the cache dir", a1.startswith("/cache/"))


def test_cache_disabled_when_no_dir_configured():
    check(
        "no cache dir means no cache path",
        af3._target_msa_cache_path(None, SEQ_A) is None,
    )
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "msas", "target_0.a3m")
        check(
            "load is a no-op when caching is disabled",
            af3._load_cached_target_msa(None, SEQ_A, dest) is False
            and not os.path.exists(dest),
        )


def test_store_then_load_across_separate_input_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "msa_cache")
        # design 1: generates its MSA, publishes it
        d1 = os.path.join(tmp, "design1", "msas", "target_0.a3m")
        os.makedirs(os.path.dirname(d1))
        with open(d1, "w") as fh:
            fh.write(">t\nMKTA\n")
        af3._store_target_msa(cache, SEQ_A, d1)

        # design 2: a different input dir entirely -- the old per-dir check
        # could never hit here
        d2 = os.path.join(tmp, "design2", "msas", "target_0.a3m")
        hit = af3._load_cached_target_msa(cache, SEQ_A, d2)
        check("a later design hits the cache from its own input dir", hit is True)
        check(
            "the cached MSA is copied in verbatim",
            os.path.exists(d2) and open(d2).read() == ">t\nMKTA\n",
        )
        check(
            "no lock or temp file is left in the cache",
            sorted(os.listdir(cache)) == [os.path.basename(
                af3._target_msa_cache_path(cache, SEQ_A))]
            or all(not f.endswith(".tmp") for f in os.listdir(cache)),
        )


def test_a_different_target_misses():
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        src = os.path.join(tmp, "a.a3m")
        with open(src, "w") as fh:
            fh.write(">t\nMKTA\n")
        af3._store_target_msa(cache, SEQ_A, src)
        dest = os.path.join(tmp, "out", "target_0.a3m")
        check(
            "a different target sequence does not reuse the cached MSA",
            af3._load_cached_target_msa(cache, SEQ_B, dest) is False
            and not os.path.exists(dest),
        )


def test_empty_msa_is_not_cached_or_served():
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        src = os.path.join(tmp, "empty.a3m")
        open(src, "w").close()
        af3._store_target_msa(cache, SEQ_A, src)
        dest = os.path.join(tmp, "out", "target_0.a3m")
        check(
            "an empty MSA is never published to the cache",
            af3._load_cached_target_msa(cache, SEQ_A, dest) is False,
        )


def test_store_is_idempotent_and_does_not_clobber():
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "cache")
        first = os.path.join(tmp, "first.a3m")
        with open(first, "w") as fh:
            fh.write("FIRST")
        af3._store_target_msa(cache, SEQ_A, first)
        second = os.path.join(tmp, "second.a3m")
        with open(second, "w") as fh:
            fh.write("SECOND")
        af3._store_target_msa(cache, SEQ_A, second)
        check(
            "a concurrent second publish does not overwrite the first",
            open(af3._target_msa_cache_path(cache, SEQ_A)).read() == "FIRST",
        )


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print(f"\n{PASSED}/{PASSED + FAILED} passed")
    sys.exit(1 if FAILED else 0)
