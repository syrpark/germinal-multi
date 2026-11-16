"""Pins when an all-failed fold batch is fatal to the RUN vs to one design.

The recycle stage folds 1 candidate x 2 targets, and both legs share the
binder sequence, so they share one MSA. One flaky search therefore fails 2 of
2 jobs: `len(errors) == len(jobs)` fires and AllJobsFailed would abort an
otherwise healthy run hours in.

"Every job failed" means systemic breakage only if the jobs could have failed
INDEPENDENTLY. Legs of one design cannot. The rule is keyed on the number of
distinct binder sequences in the batch, not on the batch size.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from germinal.filters.folding import FoldJob, AllJobsFailed, sequential


def _job(name, binder, target_index=0):
    return FoldJob(
        name=name,
        binder_sequence=binder,
        target_sequence="TARGETSEQ",
        target_index=target_index,
    )


def _always_fails(exc):
    def fold_one(job, ctx):
        raise exc

    return sequential(fold_one)


def test_one_design_all_legs_failed_is_not_fatal():
    """2 legs of 1 design, both failed -> the CAUSE propagates, not AllJobsFailed."""
    boom = RuntimeError("colabfold_search timed out after 600s")
    backend = _always_fails(boom)
    jobs = [_job("d_t0", "BINDER_A", 0), _job("d_t1", "BINDER_A", 1)]

    try:
        backend(jobs, ctx=None)
    except AllJobsFailed:
        raise AssertionError(
            "one design's legs failing must not abort the run"
        )
    except RuntimeError as e:
        assert "timed out" in str(e)
    else:
        raise AssertionError("expected the underlying cause to propagate")
    print("PASS: single-design all-legs failure propagates the cause")


def test_multiple_designs_all_failed_is_fatal():
    """Distinct binders all failing IS systemic -- still AllJobsFailed."""
    backend = _always_fails(RuntimeError("backend is broken"))
    jobs = [_job("d0", "BINDER_A"), _job("d1", "BINDER_B"), _job("d2", "BINDER_C")]

    try:
        backend(jobs, ctx=None)
    except AllJobsFailed as e:
        assert "3 distinct binder" in str(e), str(e)
    else:
        raise AssertionError("3 independent designs failing must stay fatal")
    print("PASS: multi-design all-failed still raises AllJobsFailed")


def test_partial_failure_returns_none_and_does_not_raise():
    """A real negative result: some legs fail, the batch survives."""

    def fold_one(job, ctx):
        if job.binder_sequence == "BINDER_B":
            raise RuntimeError("one bad design")
        return ("pdb", {}, 0.5)

    backend = sequential(fold_one)
    jobs = [_job("d0", "BINDER_A"), _job("d1", "BINDER_B")]

    results = backend(jobs, ctx=None)
    assert results[0] is not None
    assert results[1] is None
    print("PASS: partial failure returns None without raising")


def test_empty_batch_does_not_raise():
    backend = _always_fails(RuntimeError("never called"))
    assert backend([], ctx=None) == []
    print("PASS: empty batch is a no-op")


if __name__ == "__main__":
    test_one_design_all_legs_failed_is_not_fatal()
    test_multiple_designs_all_failed_is_fatal()
    test_partial_failure_returns_none_and_does_not_raise()
    test_empty_batch_does_not_raise()
    print("\nAll folding-failure-scope tests passed.")
