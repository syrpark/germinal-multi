"""Pins what "this seed is already done" means once designs fold in batches.

Resume has no checkpoint file: it is check_existing_seed plus the
deterministic seed list. A working directory appears in phase B while the row
may not be written until phase D, so only the master CSV and the terminal
output directories count as proof that a seed is finished.
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from germinal.utils.io import IO, RunLayout

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


def _io(tmp):
    return IO(RunLayout.create(Path(tmp) / "run"))


def _row(io, design_name):
    pd.DataFrame([{"design_name": design_name, "status": "x"}]).to_csv(
        io.layout.all_csv, index=False
    )


def test_a_recorded_design_is_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _row(io, "egfr_scfv_s488766")
        check("a seed with a master row is done", io.check_existing_seed(488766))


def test_a_bare_working_directory_is_not_proof_of_completion():
    # Phase B creates the working directory, so a crash before phase D
    # writes the row must leave the seed available.
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        os.makedirs(io.layout.trajectories / "structures" / "egfr_scfv_s488766")
        check(
            "a working directory alone does NOT mark the seed done",
            io.check_existing_seed(488766) is False,
        )


def test_a_redesign_candidates_workdir_is_not_proof_either():
    # This directory exists before any row does, and its name carries
    # _s<seed>, so treating it as terminal would mark a seed done early.
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        os.makedirs(
            io.layout.redesign_candidates / "structures" / "egfr_scfv_s488766"
        )
        check(
            "a redesign_candidates working directory does NOT mark the seed done",
            io.check_existing_seed(488766) is False,
        )
        # and a recycle fold sitting in it is still not a result
        (io.layout.redesign_candidates / "structures" / "egfr_scfv_s488766"
         / "egfr_scfv_s488766_recycle1_seq0_target0_protenix.pdb").write_text("ATOM\n")
        check(
            "a recycle prediction does NOT mark the seed done",
            io.check_existing_seed(488766) is False,
        )


def test_saved_results_still_count_without_a_csv():
    # Terminal directories hold only saved results, so a trace there is
    # proof. redesign_candidates is a working directory and is excluded.
    for leg in ("accepted", "filter_failed"):
        with tempfile.TemporaryDirectory() as tmp:
            io = _io(tmp)
            out = getattr(io.layout, leg) / "structures"
            os.makedirs(out, exist_ok=True)
            (out / "egfr_scfv_s488766_abmpnn_0_0.pdb").write_text("ATOM\n")
            check(
                f"a saved structure in {leg}/ marks the seed done",
                io.check_existing_seed(488766),
            )


def test_an_unused_seed_is_available():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _row(io, "egfr_scfv_s488766")
        check("an unrelated seed is free", io.check_existing_seed(123456) is False)


def test_seed_matching_is_delimited_not_substring():
    # `seed in fname` reported 8876 as used because "..._s488766_..." contains
    # "8876"; every false positive burns a trajectory slot.
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _row(io, "egfr_scfv_s488766")
        check("a substring of another seed is not a match",
              io.check_existing_seed(8876) is False)
        check("the full delimited seed still matches",
              io.check_existing_seed(488766))


def test_candidate_rows_mark_their_parent_seed_done():
    # Phase D writes _abmpnn_N rows, not a row for the parent. The seed lives
    # in the candidate name, so resume must still see it.
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _row(io, "egfr_scfv_s488766_abmpnn_3")
        check(
            "a candidate row marks the parent seed done",
            io.check_existing_seed(488766),
        )


def test_an_unreadable_csv_warns_and_falls_back():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        io.layout.all_csv.write_text("a,b\n1,2\n3,4,5\n")
        out = io.layout.accepted / "structures"
        os.makedirs(out, exist_ok=True)
        (out / "egfr_scfv_s488766_abmpnn_0_0.pdb").write_text("ATOM\n")
        check(
            "a corrupt master CSV still finds saved results",
            io.check_existing_seed(488766),
        )
        check(
            "and does not claim an unsaved seed is done",
            io.check_existing_seed(999999) is False,
        )


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print(f"\n{PASSED}/{PASSED + FAILED} passed")
    sys.exit(1 if FAILED else 0)
