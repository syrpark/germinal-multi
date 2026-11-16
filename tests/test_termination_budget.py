"""Pins what each termination budget counts.

``all_trajectories.csv`` holds one row per candidate, not per trajectory.
``max_hallucinated_trajectories`` is a per-hallucination budget, so counting
rows would charge it once per ``_abmpnn_N`` fan-out. ``max_passing_designs``
counts rows on purpose: each accepted candidate is its own design.
"""

import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from germinal.utils.io import IO, RunLayout  # noqa: E402


def _io(tmp, all_rows, accepted_rows=()):
    """A DesignIO over a throwaway run directory holding the given rows."""
    io = IO(RunLayout.create(os.path.join(tmp, "run")))
    pd.DataFrame({"design_name": list(all_rows)}).to_csv(io.layout.all_csv, index=False)
    if accepted_rows:
        pd.DataFrame({"design_name": list(accepted_rows)}).to_csv(
            io.layout.accepted / "designs.csv", index=False
        )
    return io


SETTINGS = {
    "max_passing_designs": 1000,
    "max_hallucinated_trajectories": 3,
    "max_trajectories": 1000,
}


def test_candidates_of_one_trajectory_count_once():
    # Three hallucinations, each fanned out to four candidates: 12 rows, but
    # only 3 hallucinations. Counting rows would have stopped at the first.
    rows = [
        f"t_s{seed}_abmpnn_{n}" for seed in (1, 2) for n in range(1, 5)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp, rows)
        stop, reason = io.check_termination_conditions(SETTINGS, n_trajectories=2)
        assert not stop, (
            f"stopped after 2 hallucinations against a budget of 3: {reason}. "
            f"The 8 candidate rows are being counted as 8 hallucinations."
        )


def test_the_budget_still_fires_on_distinct_hallucinations():
    # Guards against the fix turning the budget off altogether.
    rows = [f"t_s{seed}_abmpnn_1" for seed in (1, 2, 3)]
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp, rows)
        stop, reason = io.check_termination_conditions(SETTINGS, n_trajectories=3)
        assert stop and "hallucinated" in reason.lower(), (
            f"3 distinct hallucinations against a budget of 3 did not stop: "
            f"{(stop, reason)}"
        )


def test_bare_trajectory_rows_are_counted_too():
    # A trajectory that failed before redesign has no _abmpnn suffix; it still
    # consumed a hallucination.
    rows = ["t_s1", "t_s2", "t_s3"]
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp, rows)
        stop, _ = io.check_termination_conditions(SETTINGS, n_trajectories=3)
        assert stop, "failed trajectories must still charge the budget"


def test_a_suffix_is_only_stripped_from_the_end():
    # Paranoia about the regex: a design whose own name contains the token
    # must not be collapsed into another.
    rows = ["t_abmpnn_x_s1_abmpnn_1", "t_abmpnn_x_s2_abmpnn_1"]
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp, rows)
        stop, _ = io.check_termination_conditions(
            {**SETTINGS, "max_hallucinated_trajectories": 2}, n_trajectories=2
        )
        assert stop, "two distinct hallucinations were collapsed into one"


def test_accepted_designs_count_every_candidate():
    # The other direction: each accepted _abmpnn_N IS its own accepted design.
    accepted = [f"t_s1_abmpnn_{n}" for n in range(1, 5)]
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp, ["t_s1"], accepted_rows=accepted)
        stop, reason = io.check_termination_conditions(
            {**SETTINGS, "max_passing_designs": 4}, n_trajectories=1
        )
        assert stop and "accepted" in reason.lower(), (
            f"4 accepted candidates against a budget of 4 did not stop: "
            f"{(stop, reason)}"
        )


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
