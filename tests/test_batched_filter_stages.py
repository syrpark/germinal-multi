"""Pins that the initial and final filter stages fold in ONE call per batch.

The AF3 container start dominates a single prediction, so folding one design
at a time paid it once per design and once per candidate. These tests assert
the production path reaches fold_designs_against_targets, not merely that it
works.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from germinal.filters import filter_utils, folding

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


class _Layout:
    def __init__(self, root):
        self.trajectories = Path(root)
        # Filter-model predictions and the _batch_fold scratch live here now;
        # trajectories/ keeps only the AF2 hallucination PDBs.
        self.redesign_candidates = Path(root)


class _IO:
    def __init__(self, root):
        self.layout = _Layout(root)


def _settings():
    return {
        "num_fold_seeds_initial": 3,
        "num_fold_seeds_final": 5,
        "filter_folder": {
            "backend": "af3",
            "num_samples": 5,
            "num_cycles": 10,
            "sample_select_mode": "best",
            "seed_select_mode": "worst",
            "msa_method": "colabfold_remote",
            "msa_scope": "target_only",
        },
    }


def _targets():
    return {
        "target_name": ["wt", "mut"],
        "target_chain": "A",
        "binder_chain": "B",
        "target_pdb_path": ["/wt.pdb", "/mut.pdb"],
    }


def _capture(monkey_results=None):
    """Stub folding._run and record the jobs it is handed."""
    seen = {}

    def fake_run(stage, jobs, ctx):
        seen["stage"] = stage
        seen["jobs"] = list(jobs)
        seen["n_jobs"] = len(jobs)
        seen["seeds"] = len(ctx.seeds)
        seen["want_all"] = ctx.want_all_samples
        return [(f"{j.name}.pdb", {}, {"ipsae": 0.5}) for j in jobs]

    return seen, fake_run


def _call_fold(designs, is_final):
    seen, fake_run = _capture()
    real_run, real_seqs = folding._run, filter_utils.resolve_target_sequences
    folding._run = fake_run
    filter_utils.resolve_target_sequences = lambda ts: ["TTTT", "TTTA"]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = filter_utils.fold_designs_for_filter(
                designs=designs,
                run_settings=_settings(),
                target_settings=_targets(),
                io=_IO(tmp),
                is_final=is_final,
            )
        return seen, out
    finally:
        folding._run = real_run
        filter_utils.resolve_target_sequences = real_seqs


# --- fold_designs_for_filter batches ---
def test_many_designs_fold_in_one_call():
    designs = [("dA", "AAA"), ("dB", "BBB"), ("dC", "CCC")]
    seen, out = _call_fold(designs, is_final=False)
    check(
        "three designs x two targets is six jobs in ONE call",
        seen["n_jobs"] == 6,
        f"got {seen['n_jobs']}",
    )
    check("the filter backend is used", seen["stage"] == "filter", seen["stage"])
    check(
        "the initial stage uses num_fold_seeds_initial",
        seen["seeds"] == 3,
        str(seen["seeds"]),
    )
    check(
        "results regroup by design name, targets in order",
        sorted(out) == ["dA", "dB", "dC"] and len(out["dA"]) == 2,
        str({k: len(v) for k, v in out.items()}),
    )


def test_final_stage_budget_and_ensemble():
    seen, _ = _call_fold([("dA", "AAA"), ("dB", "BBB")], is_final=True)
    check(
        "the final stage uses num_fold_seeds_final",
        seen["seeds"] == 5,
        str(seen["seeds"]),
    )
    check("the final stage asks for every sample", seen["want_all"] is True)


def test_no_designs_is_not_a_fold():
    seen, out = _call_fold([], is_final=False)
    check("an empty batch folds nothing", "jobs" not in seen and out == {})

def test_mismatched_fold_count_is_rejected():
    # Results are matched to targets by position, so a short list would score
    # a design against the wrong antigen rather than raising.
    import germinal.filters.filter_utils as fu

    class _Traj:
        design_name = "d0"

    with tempfile.TemporaryDirectory() as tmp:
        try:
            fu.multi_run_filters(
                _Traj(), _settings(), _targets(), {}, _IO(tmp),
                "AAAA", ["/a.pdb", "/b.pdb"],
                is_final=False,
                prediction_results=[("only-one.pdb", {}, {})],
            )
            check("a short fold list is rejected", False, "no exception")
        except ValueError as e:
            check(
                "a short fold list is rejected, naming the counts",
                "1" in str(e) and "2" in str(e),
                str(e),
            )
        except Exception as e:
            check("a short fold list raises ValueError", False, repr(e))


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print(f"\n{PASSED}/{PASSED + FAILED} passed")
    sys.exit(1 if FAILED else 0)
