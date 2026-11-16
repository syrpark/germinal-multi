"""fold_designs_against_targets: job construction and result regrouping.

Batches designs as well as targets, so the container start is paid once per
BATCH rather than once per design. The backend is stubbed -- this covers the
job list and the regrouping, not the fold itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from germinal.filters import folding

PASSED = 0
FAILED = 0


def check(label, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label} {detail}")


RUN_SETTINGS = {
    "num_fold_seeds_initial": 3,
    "num_fold_seeds_final": 5,
    "filter_folder": {"backend": "af3", "num_samples": 5, "num_cycles": 10,
        "sample_select_mode": "best",
        "seed_select_mode": "worst",
        "msa_method": "colabfold_remote",
        "msa_scope": "target_only",
    },
}


def _stub(monkey_jobs):
    def _run(stage, jobs, ctx):
        monkey_jobs.append((stage, list(jobs), ctx))
        return [f"result::{j.name}" for j in jobs]
    return _run


def test_jobs_cover_the_cross_product_and_regroup_by_design():
    seen = []
    orig = folding._run
    folding._run = _stub(seen)
    try:
        out = folding.fold_designs_against_targets(
            designs=[("dA", "SEQA"), ("dB", "SEQB"), ("dC", "SEQC")],
            target_sequences=["T0", "T1"],
            target_names=["wt", "mut"],
            target_chain="A",
            binder_chain="B",
            output_dir="/tmp/x",
            run_settings=RUN_SETTINGS,
        )
    finally:
        folding._run = orig

    stage, jobs, ctx = seen[0]
    check("one batched call, not one per design", len(seen) == 1, f"got {len(seen)}")
    check("stage is the filter stage", stage == "filter", stage)
    check("3 designs x 2 targets = 6 jobs", len(jobs) == 6, f"got {len(jobs)}")
    check(
        "job names are design_target",
        [j.name for j in jobs] == [
            "dA_wt", "dA_mut", "dB_wt", "dB_mut", "dC_wt", "dC_mut"
        ],
        f"got {[j.name for j in jobs]}",
    )
    check(
        "target_index is per design, not global",
        [j.target_index for j in jobs] == [0, 1, 0, 1, 0, 1],
        f"got {[j.target_index for j in jobs]}",
    )
    check(
        "each design keeps its own sequence",
        [j.binder_sequence for j in jobs] == [
            "SEQA", "SEQA", "SEQB", "SEQB", "SEQC", "SEQC"
        ],
    )
    check("regrouped by design name", set(out) == {"dA", "dB", "dC"}, f"got {set(out)}")
    check(
        "each design gets its targets in order",
        out["dB"] == ["result::dB_wt", "result::dB_mut"],
        f"got {out['dB']}",
    )
    # Seeds: the initial budget unless is_final.
    check("initial seed budget by default", len(ctx.seeds) == 3, f"got {len(ctx.seeds)}")
    check("want_all_samples off unless final", ctx.want_all_samples is False)


def test_final_uses_the_final_seed_budget():
    seen = []
    orig = folding._run
    folding._run = _stub(seen)
    try:
        folding.fold_designs_against_targets(
            designs=[("d", "S")],
            target_sequences=["T0", "T1"],
            target_names=["wt", "mut"],
            target_chain="A",
            binder_chain="B",
            output_dir="/tmp/x",
            run_settings=RUN_SETTINGS,
            is_final=True,
        )
    finally:
        folding._run = orig
    _, _, ctx = seen[0]
    check("final seed budget", len(ctx.seeds) == 5, f"got {len(ctx.seeds)}")
    check("all samples requested for the relax ensemble", ctx.want_all_samples is True)


def test_duplicate_design_names_raise():
    """Results are matched back by name, so a duplicate would silently drop one."""
    raised = False
    try:
        folding.fold_designs_against_targets(
            designs=[("d", "S1"), ("d", "S2")],
            target_sequences=["T0"],
            target_names=["wt"],
            target_chain="A",
            binder_chain="B",
            output_dir="/tmp/x",
            run_settings=RUN_SETTINGS,
        )
    except ValueError as e:
        raised = "unique design names" in str(e)
    check("duplicate design names are rejected", raised)


if __name__ == "__main__":
    test_jobs_cover_the_cross_product_and_regroup_by_design()
    test_final_uses_the_final_seed_budget()
    test_duplicate_design_names_raise()
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
