"""Pins the end-of-run report on work left in ``inflight/``.

A run that ends at a wall-clock limit routinely parks designs that passed
hallucination but never filled a fold batch. That work is real, already paid
for, and absent from every CSV, because an in-flight record is deliberately
not a row. So the summary has to name it, say it is not a failure, and say
that a relaunch finishes it -- silence there reads as "nothing left to do".
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_germinal_multi as runner

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


class _IO:
    """Just the one method the note reads."""

    def __init__(self, records):
        self._records = records

    def read_inflight(self):
        return list(self._records)


def _rec(name, phase):
    return {"design_name": name, "phase": phase}


def test_an_empty_inflight_says_so_rather_than_nothing():
    # The clean case still gets a line: absence of a warning is not a
    # statement, and "is it empty or did it not look" costs a dig otherwise.
    note = runner._unfinished_work_note(_IO([]))
    check("a clean exit says inflight/ is empty", "empty" in note, note)
    check("and does not cry unfinished work",
          "UNFINISHED WORK" not in note, note)


def test_parked_designs_are_counted_and_named():
    note = runner._unfinished_work_note(_IO([
        _rec("d_s1", "hallucinated"),
        _rec("d_s2", "hallucinated"),
        _rec("d_s3", "candidates_ready"),
    ]))
    check("the total is stated", "3 design(s) left in inflight/" in note, note)
    check("each phase is broken out",
          "2 at 'hallucinated'" in note and "1 at 'candidates_ready'" in note,
          note)
    check("designs are named, so they can be checked by hand",
          "d_s1" in note and "d_s3" in note, note)
    check("and what each phase is owed is spelled out",
          "folded and filtered" in note and "scored" in note, note)


def test_the_note_says_these_are_not_failures_and_how_to_finish_them():
    # The whole point: someone reading a log that ends in "0 accepted" must
    # not conclude the parked designs were rejected, or redo them by hand.
    note = runner._unfinished_work_note(_IO([_rec("d_s1", "hallucinated")]))
    check("it says they are not failures", "NOT failures" in note, note)
    check("it says a relaunch adopts them", "adopts them" in note, note)
    check("it says they cost no attempt",
          "no attempt" in note, note)


def test_designs_that_never_folded_name_the_knob_that_parked_them():
    # The cause is a batch that never filled, and the fix is one setting. The
    # run that hit it is the run whose log gets read, so it belongs here.
    note = runner._unfinished_work_note(_IO([_rec("d_s1", "hallucinated")]))
    check("the unfolded count is called out",
          "1 of them never reached a fold" in note, note)
    check("max_hallucination_passing is named",
          "max_hallucination_passing" in note, note)
    check("and so is the attempt cap that also closes a batch",
          "max_hallucination_attempts" in note, note)


def test_no_fold_advice_when_everything_already_folded():
    # candidates_ready designs have folded twice already; telling their reader
    # to lower a hallucination batch setting would be noise.
    note = runner._unfinished_work_note(_IO([_rec("d_s1", "candidates_ready")]))
    check("no batch-size advice when nothing is parked pre-fold",
          "max_hallucination_passing" not in note, note)


def test_a_long_list_is_truncated_not_dumped():
    note = runner._unfinished_work_note(_IO([
        _rec(f"d_s{i}", "hallucinated") for i in range(9)
    ]))
    check("nine parked designs are counted", "9 design(s)" in note, note)
    check("but the names are truncated", "..." in note, note)
    check("to four", note.count("d_s") <= 5, note)


def test_an_unknown_phase_does_not_crash_the_summary():
    # read_inflight skips foreign schemas, but a phase this build does not
    # know must not take the whole end-of-run report down with it.
    note = runner._unfinished_work_note(_IO([_rec("d_s1", "some_new_phase")]))
    check("an unknown phase still reports", "1 design(s)" in note, note)
    check("named as-is", "some_new_phase" in note, note)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
