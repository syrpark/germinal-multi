"""Tests for the folding router: which backend runs, with which seeds.

The two stages fold independently and may use different models, so: no caller
branches on the backend, the seed count comes from the stage, and an af3-only
install never imports protenix. The real backends are swapped for recording
fakes -- what is under test is the routing, not the folding.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from germinal.filters import folding

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


def _folder(backend, **over):
    """One instance of the configs/folding group, as Hydra would compose it."""
    f = {
        "backend": backend,
        "num_samples": 5,
        "num_cycles": 10,
        "sample_select_mode": "best",
        "seed_select_mode": "worst",
        "msa_method": "colabfold_remote",
        "msa_scope": "target_only",
        # Both stages must fold with MSAs; validate_backends enforces it.
        "use_msa": True,
    }
    if backend == "protenix":
        f.update(
            num_steps=200,
            conda_env="protenix",
            model_name="protenix_base_default_v1.0.0",
        )
    f.update(over)
    return f


BASE_SETTINGS = {
    "msa_mode": "target",
    "num_fold_seeds_recycle": 1,
    "num_fold_seeds_initial": 3,
    "num_fold_seeds_final": 5,
    # Both default to af3: tests that swap in a fake backend only register
    # one, and the stage they are not exercising must still resolve.
    "recycle_folder": _folder("af3"),
    "filter_folder": _folder("af3"),
}


class _Recorder:
    """Stands in for a backend; records exactly how it was called."""

    def __init__(self, label):
        self.label = label
        self.calls = []

    def __call__(self, jobs, ctx):
        self.calls.append((list(jobs), ctx))
        out = []
        for job in jobs:
            base = (f"{self.label}_{job.name}.pdb", {"plddt": 0.9}, {"ipsae": 0.7})
            out.append(base + ([base],) if ctx.want_all_samples else base)
        return out


def _install(**recorders):
    """Swap the registry for recording fakes; returns a restore callable."""
    saved = dict(folding._BACKENDS)
    folding._BACKENDS.clear()
    for name, rec in recorders.items():
        folding.register_backend(name, rec)

    def restore():
        folding._BACKENDS.clear()
        folding._BACKENDS.update(saved)

    return restore


def _settings(**over):
    s = dict(BASE_SETTINGS)
    s.update(over)
    return s


# --- backend resolution ---
def test_each_stage_reads_its_own_key():
    s = _settings(recycle_folder=_folder("protenix"), filter_folder=_folder("af3"))
    check(
        "the recycle stage reads recycle_folder",
        folding.resolve_backend_name(s, "recycle") == "protenix",
    )
    check(
        "the filter stage reads filter_folder",
        folding.resolve_backend_name(s, "filter") == "af3",
    )


def test_the_two_stages_are_independent():
    s = _settings(recycle_folder=_folder("af3"), filter_folder=_folder("protenix"))
    check(
        "the stages can name different backends",
        folding.resolve_backend_name(s, "recycle") == "af3"
        and folding.resolve_backend_name(s, "filter") == "protenix",
    )


def test_legacy_flat_keys_are_rejected_not_silently_ignored():
    # The pre-split key scheme must fail at startup rather than degrade
    # quietly: under it the two backends ran different sample budgets.
    s = dict(BASE_SETTINGS)
    del s["recycle_folder"]
    s["structure_model"] = "protenix"
    try:
        folding.resolve_backend_name(s, "recycle")
        check("a pre-split config raises rather than resolving", False,
              "no exception")
    except ValueError as e:
        check(
            "a pre-split config raises, naming the key to set",
            "recycle_folder" in str(e) and "folding@" in str(e),
            str(e),
        )


def test_default_is_af3():
    check(
        "with nothing configured, both stages default to af3",
        folding.resolve_backend_name(_settings(), "recycle") == "af3"
        and folding.resolve_backend_name(_settings(), "filter") == "af3",
    )


def test_unknown_backend_is_rejected_by_name():
    try:
        folding.resolve_backend_name(_settings(recycle_folder=_folder("example")), "recycle")
        check("an unknown backend raises", False, "no exception")
    except ValueError as e:
        check(
            "an unknown backend raises naming the key and the valid options",
            "recycle_folder" in str(e) and "af3" in str(e),
            str(e),
        )


def test_unknown_stage_is_rejected():
    try:
        folding.resolve_backend_name(_settings(), "final")
        check("an unknown stage raises", False, "no exception")
    except ValueError:
        check("an unknown stage raises", True)


def test_case_is_not_significant():
    check(
        "backend names are matched case-insensitively",
        folding.resolve_backend_name(
            _settings(recycle_folder=_folder("protenix")), "recycle"
        ) == "protenix",
    )


# --- startup validation ---
def test_validate_accepts_a_mixed_configuration():
    try:
        folding.validate_backends(
            _settings(
                recycle_folder=_folder("protenix"),
                filter_folder=_folder("af3"),
            )
        )
        check("a protenix recycle / af3 filter run validates", True)
    except Exception as e:
        check("a protenix recycle / af3 filter run validates", False, repr(e))


def test_validate_rejects_a_bad_backend_before_any_gpu_time():
    try:
        folding.validate_backends(_settings(filter_folder=_folder("nope")))
        check("validate_backends rejects an unknown backend", False, "no exception")
    except ValueError:
        check("validate_backends rejects an unknown backend", True)


def test_validate_rejects_decoupled_select_mode_on_protenix():
    # "decoupled" is af3-only, so it must be rejected at startup rather than
    # at the first design's final filter.
    try:
        folding.validate_backends(
            _settings(
                filter_folder=_folder("protenix", decoupled=True),
            )
        )
        check("decoupled + protenix is rejected", False, "no exception")
    except ValueError as e:
        check(
            "decoupled + protenix is rejected, naming both",
            "decoupled" in str(e) and "protenix" in str(e),
            str(e),
        )


def test_decoupled_is_fine_on_af3():
    try:
        folding.validate_backends(
            _settings(filter_folder=_folder("af3", decoupled=True))
        )
        check("decoupled + af3 validates", True)
    except Exception as e:
        check("decoupled + af3 validates", False, repr(e))


def test_legacy_flat_select_mode_is_rejected_not_half_honoured():
    # The removed key reduced samples and seeds together, so reading it would
    # keep flat-pool behaviour under a config that looks migrated.
    folder = _folder("af3")
    del folder["sample_select_mode"]
    del folder["seed_select_mode"]
    folder["select_mode"] = "worst"
    try:
        folding.resolve_selection(_settings(filter_folder=folder), "filter")
        check("a config still carrying select_mode is rejected", False, "no exception")
    except ValueError as e:
        check(
            "a config still carrying select_mode is rejected, naming both new keys",
            "sample_select_mode" in str(e) and "seed_select_mode" in str(e),
            str(e),
        )


def test_a_missing_or_bogus_axis_is_rejected():
    for axis in ("sample_select_mode", "seed_select_mode"):
        folder = _folder("af3")
        del folder[axis]
        try:
            folding.resolve_selection(_settings(filter_folder=folder), "filter")
            check(f"a missing {axis} is rejected", False, "no exception")
        except ValueError as e:
            check(f"a missing {axis} is rejected", axis in str(e), str(e))

        folder = _folder("af3")
        folder[axis] = "median"
        try:
            folding.resolve_selection(_settings(filter_folder=folder), "filter")
            check(f"a bogus {axis} is rejected", False, "no exception")
        except ValueError as e:
            check(f"a bogus {axis} is rejected", axis in str(e), str(e))


def test_both_axes_reach_the_context():
    # The whole point: the two decisions must arrive at the backend
    # separately, not collapsed back into one mode.
    seen = {}

    def fold(jobs, ctx):
        seen["sample"] = ctx.sample_select_mode
        seen["seed"] = ctx.seed_select_mode
        return [(f"{j.name}.pdb", {}, {"ipsae": 0.5}) for j in jobs]

    folding.register_backend("probe_axes", fold)
    folder = _folder("af3")
    folder.update(backend="probe_axes",
                  sample_select_mode="best", seed_select_mode="worst")
    folding.fold_candidates(
        binder_sequences=["AAA"], target_sequences=["TTT"],
        target_chain="A", binder_chain="B", output_dir="/tmp/x",
        design_name_prefix="d",
        run_settings=_settings(recycle_folder=folder),
    )
    check(
        "sample=best and seed=worst arrive as two separate fields",
        seen == {"sample": "best", "seed": "worst"},
        str(seen),
    )


# --- dispatch ---
def test_recycle_folds_go_to_the_recycle_backend_only():
    af3r, pxr = _Recorder("af3"), _Recorder("protenix")
    restore = _install(af3=af3r, protenix=pxr)
    try:
        folding.fold_candidates(
            binder_sequences=["AAA", "BBB"],
            target_sequences=["TTT"],
            target_chain="A",
            binder_chain="B",
            output_dir="/tmp/x",
            design_name_prefix="d_recycle1",
            run_settings=_settings(
                recycle_folder=_folder("protenix"), filter_folder=_folder("af3")
            ),
        )
    finally:
        restore()
    check(
        "the recycle fold went to protenix",
        len(pxr.calls) == 1 and not af3r.calls,
        f"af3={len(af3r.calls)} protenix={len(pxr.calls)}",
    )


def test_filter_folds_go_to_the_filter_backend_only():
    af3r, pxr = _Recorder("af3"), _Recorder("protenix")
    restore = _install(af3=af3r, protenix=pxr)
    try:
        folding.fold_against_targets(
            binder_sequence="AAA",
            target_sequences=["TTT"],
            target_names=["t0"],
            target_chain="A",
            binder_chain="B",
            output_dir="/tmp/x",
            design_name="d",
            run_settings=_settings(
                recycle_folder=_folder("protenix"), filter_folder=_folder("af3")
            ),
        )
    finally:
        restore()
    check(
        "the filter fold went to af3",
        len(af3r.calls) == 1 and not pxr.calls,
        f"af3={len(af3r.calls)} protenix={len(pxr.calls)}",
    )


def test_result_ordering_is_sequence_major():
    rec = _Recorder("af3")
    restore = _install(af3=rec)
    try:
        out = folding.fold_candidates(
            binder_sequences=["AAA", "BBB"],
            target_sequences=["T0", "T1", "T2"],
            target_chain="A",
            binder_chain="B",
            output_dir="/tmp/x",
            design_name_prefix="d",
            run_settings=_settings(),
        )
    finally:
        restore()
    check(
        "one result per (sequence, target) pair",
        len(out) == 6,
        f"got {len(out)}",
    )


# --- seeds come from the stage, not the backend ---
def test_recycle_uses_the_recycle_seed_count():
    rec = _Recorder("af3")
    restore = _install(af3=rec)
    try:
        folding.fold_candidates(
            binder_sequences=["AAA"],
            target_sequences=["TTT"],
            target_chain="A",
            binder_chain="B",
            output_dir="/tmp/x",
            design_name_prefix="d",
            run_settings=_settings(num_fold_seeds_recycle=2),
        )
    finally:
        restore()
    seeds = rec.calls[0][1].seeds
    check(
        "the recycle stage draws num_fold_seeds_recycle seeds, not "
        "num_fold_seeds_initial",
        len(seeds) == 2,
        f"got {len(seeds)} (initial is {BASE_SETTINGS['num_fold_seeds_initial']})",
    )


def test_filter_seed_count_depends_on_is_final():
    rec = _Recorder("af3")
    restore = _install(af3=rec)
    try:
        for is_final in (False, True):
            folding.fold_against_targets(
                binder_sequence="AAA",
                target_sequences=["TTT"],
                target_names=["t0"],
                target_chain="A",
                binder_chain="B",
                output_dir="/tmp/x",
                design_name="d",
                run_settings=_settings(),
                is_final=is_final,
            )
    finally:
        restore()
    initial_seeds = len(rec.calls[0][1].seeds)
    final_seeds = len(rec.calls[1][1].seeds)
    check(
        "the initial filter draws num_fold_seeds_initial",
        initial_seeds == BASE_SETTINGS["num_fold_seeds_initial"],
        f"got {initial_seeds}",
    )
    check(
        "the final filter draws num_fold_seeds_final",
        final_seeds == BASE_SETTINGS["num_fold_seeds_final"],
        f"got {final_seeds}",
    )
    check(
        "is_final is passed through as want_all_samples, so the final stage "
        "gets every sample for the relax ensemble",
        rec.calls[0][1].want_all_samples is False
        and rec.calls[1][1].want_all_samples is True,
    )


# --- the extension point: adding a third folder ---
def test_a_new_backend_needs_one_function():
    # The whole point of the registry: a new folder is one fold() and one
    # register_backend() call, with no change to the router or any caller.
    saved = dict(folding._BACKENDS)
    seen = []

    def fold_example(jobs, ctx):
        seen.extend(job.name for job in jobs)
        return [(f"example_{j.name}.pdb", {"plddt": 0.5}, {"ipsae": 0.4}) for j in jobs]

    try:
        folding.register_backend("example", fold_example)
        check("a registered backend becomes selectable", "example" in folding.available_backends())
        out = folding.fold_candidates(
            binder_sequences=["AAA"],
            target_sequences=["T0", "T1"],
            target_chain="A",
            binder_chain="B",
            output_dir="/tmp/x",
            design_name_prefix="d",
            run_settings=_settings(recycle_folder=_folder("example")),
        )
        check("the new backend received both jobs", len(seen) == 2, str(seen))
        check("and its results come straight back", len(out) == 2 and out[0][0].startswith("example_"))
    finally:
        folding._BACKENDS.clear()
        folding._BACKENDS.update(saved)


def test_sequential_wraps_a_one_at_a_time_folder():
    # Most folders handle one complex per call; sequential() is supposed to
    # supply the loop, the ordering and the error handling for them.
    calls = []

    def fold_one(job, ctx):
        calls.append(job.name)
        if job.name.endswith("target1"):
            raise RuntimeError("this one blew up")
        return (f"{job.name}.pdb", {}, {"ipsae": 0.5})

    backend = folding.sequential(fold_one)
    jobs = [
        folding.FoldJob(f"d_seq0_target{i}", "AAA", "TTT", i) for i in range(3)
    ]
    ctx = folding.FoldContext(
        target_chain="A", binder_chain="B", output_dir="/tmp/x",
        seeds=[1], run_settings=_settings(),
        num_samples=5, num_cycles=10,
        msa_method="colabfold_remote", msa_scope="target_only",
        sample_select_mode="best", seed_select_mode="worst",
        folder=_folder("protenix"),
    )
    out = backend(jobs, ctx)
    check("every job is attempted", len(calls) == 3, str(calls))
    check("one result per job, in order", len(out) == 3)
    check(
        "a job that raises becomes None without taking the batch with it",
        out[0] is not None and out[1] is None and out[2] is not None,
        str(out),
    )


def test_misaligned_backend_results_are_caught():
    # Results are matched to jobs by position, so a backend returning the
    # wrong count would silently misattribute structures to targets.
    saved = dict(folding._BACKENDS)
    try:
        folding.register_backend("short", lambda jobs, ctx: [])
        folding.fold_candidates(
            binder_sequences=["AAA"],
            target_sequences=["T0", "T1"],
            target_chain="A",
            binder_chain="B",
            output_dir="/tmp/x",
            design_name_prefix="d",
            run_settings=_settings(recycle_folder=_folder("short")),
        )
        check("a backend returning too few results raises", False, "no exception")
    except RuntimeError as e:
        check(
            "a backend returning too few results raises rather than "
            "misaligning structures to targets",
            "by position" in str(e),
            str(e),
        )
    finally:
        folding._BACKENDS.clear()
        folding._BACKENDS.update(saved)


def test_filter_stage_raises_on_a_missing_target():
    # The recycle path tolerates a None slot; the filter path must not, since
    # every target is needed to score the design.
    saved = dict(folding._BACKENDS)
    try:
        folding.register_backend("flaky", lambda jobs, ctx: [None] * len(jobs))
        folding.fold_against_targets(
            binder_sequence="AAA",
            target_sequences=["T0"],
            target_names=["t0"],
            target_chain="A",
            binder_chain="B",
            output_dir="/tmp/x",
            design_name="d",
            run_settings=_settings(filter_folder=_folder("flaky")),
        )
        check("a missing target raises on the filter path", False, "no exception")
    except FileNotFoundError as e:
        check(
            "a missing target raises on the filter path, naming the backend",
            "flaky" in str(e),
            str(e),
        )
    finally:
        folding._BACKENDS.clear()
        folding._BACKENDS.update(saved)


def test_recycle_stage_tolerates_a_missing_candidate():
    saved = dict(folding._BACKENDS)
    try:
        folding.register_backend("flaky", lambda jobs, ctx: [None] * len(jobs))
        out = folding.fold_candidates(
            binder_sequences=["AAA"],
            target_sequences=["T0"],
            target_chain="A",
            binder_chain="B",
            output_dir="/tmp/x",
            design_name_prefix="d",
            run_settings=_settings(recycle_folder=_folder("flaky")),
        )
        check(
            "a missing candidate is a None slot on the recycle path, not an error",
            out == [None],
            str(out),
        )
    finally:
        folding._BACKENDS.clear()
        folding._BACKENDS.update(saved)


def test_both_shipped_backends_are_registered():
    check(
        "af3 and protenix are both registered out of the box",
        set(folding.available_backends()) >= {"af3", "protenix"},
        str(folding.available_backends()),
    )


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print(f"\n{PASSED}/{PASSED + FAILED} passed")
    sys.exit(1 if FAILED else 0)


def test_build_context_is_the_only_reader_of_the_stage_config():
    # The four shared knobs arrive on the context, never dug out of `folder`
    # by each backend.
    folder = _folder("af3")
    folder.update(num_samples=7, num_cycles=3,
                  sample_select_mode="worst", seed_select_mode="best")
    ctx = folding.build_context(
        _settings(filter_folder=folder), "filter",
        target_chain="A", binder_chain="B", output_dir="/tmp/x",
        seeds=[11, 22, 33],
    )
    check(
        "num_samples / num_cycles / both axes are read from the stage config",
        (ctx.num_samples, ctx.num_cycles,
         ctx.sample_select_mode, ctx.seed_select_mode) == (7, 3, "worst", "best"),
        f"{ctx.num_samples} {ctx.num_cycles} "
        f"{ctx.sample_select_mode} {ctx.seed_select_mode}",
    )
    check("how many seeds is len(ctx.seeds)", len(ctx.seeds) == 3, str(ctx.seeds))


def test_both_backends_are_handed_the_same_four():
    # af3 and protenix must take num_samples, seeds and both axes
    # identically, or the stages run different budgets.
    import inspect
    from germinal.filters import protenix as px

    px_params = inspect.signature(px.run_protenix).parameters
    for name in ("num_samples", "num_cycles",
                 "sample_select_mode", "seed_select_mode"):
        check(
            f"protenix.run_protenix takes {name} explicitly",
            name in px_params,
            str(sorted(px_params)),
        )
        check(
            f"protenix.run_protenix does not default {name}",
            px_params[name].default is inspect.Parameter.empty,
            f"default={px_params[name].default!r}",
        )

    from germinal.filters import af3 as af3_mod
    af3_params = inspect.signature(af3_mod._run_af3).parameters
    for name in ("num_samples", "num_cycles",
                 "sample_select_mode", "seed_select_mode"):
        check(
            f"af3._run_af3 takes {name} explicitly",
            name in af3_params,
            str(sorted(af3_params)),
        )
        check(
            f"af3._run_af3 does not default {name}",
            af3_params[name].default is inspect.Parameter.empty,
            f"default={af3_params[name].default!r}",
        )


# --- protenix batching at the filter stages ---
def _px_ctx(stage, want_all=False, **over):
    kw = dict(
        target_chain="A", binder_chain="B", output_dir="/tmp/px",
        run_settings={}, num_samples=5, seeds=[1, 2], sample_select_mode="best",
        seed_select_mode="worst", msa_method="colabfold_remote",
        msa_scope="none", num_cycles=10, folder={"backend": "protenix"},
        stage=stage, want_all_samples=want_all,
    )
    kw.update(over)
    return folding.FoldContext(**kw)


def _px_jobs(n):
    return [folding.FoldJob(name=f"d{i}", binder_sequence=f"QV{i}",
                            target_sequence="TT", target_index=0)
            for i in range(n)]


def _px_fakes(batch_result=None, single_result=None):
    """Swap both protenix entry points; returns (restore, calls)."""
    from germinal.filters import protenix
    calls = {"batch": 0, "single": 0, "names": None}

    def fake_batch(designs, **kw):
        calls["batch"] += 1
        calls["names"] = [d[0] for d in designs]
        return [batch_result for _ in designs]

    def fake_single(**kw):
        calls["single"] += 1
        return single_result

    saved = (protenix.run_protenix_batch, protenix.run_protenix)
    protenix.run_protenix_batch, protenix.run_protenix = fake_batch, fake_single

    def restore():
        protenix.run_protenix_batch, protenix.run_protenix = saved
    return restore, calls


ONE = ("p.pdb", {"plddt": 0.8}, {"ipsae": 0.5})


def test_the_filter_stages_fold_protenix_in_one_call():
    restore, calls = _px_fakes(batch_result=ONE, single_result=ONE)
    try:
        out = folding.fold_protenix(_px_jobs(4), _px_ctx("filter"))
        check("four filter jobs are one batch call", calls["batch"] == 1, calls)
        check("and no per-complex calls", calls["single"] == 0, calls)
        check("one result per job", len(out) == 4, len(out))
        check("in job order", calls["names"] == ["d0", "d1", "d2", "d3"], calls)
    finally:
        restore()


def test_the_recycle_stage_stays_sequential():
    # It folds one candidate at a time, so batching would stage for nothing.
    restore, calls = _px_fakes(batch_result=ONE, single_result=ONE)
    try:
        folding.fold_protenix(_px_jobs(4), _px_ctx("recycle"))
        check("recycle never batches", calls["batch"] == 0, calls)
        check("it calls one complex at a time", calls["single"] == 4, calls)
    finally:
        restore()


def test_a_single_job_is_not_worth_batching():
    restore, calls = _px_fakes(batch_result=ONE, single_result=ONE)
    try:
        folding.fold_protenix(_px_jobs(1), _px_ctx("filter"))
        check("one job goes down the sequential path", calls["single"] == 1, calls)
    finally:
        restore()


def test_a_batched_final_fold_keeps_the_four_tuple():
    # is_final asks for every sample; extract returns (chosen, all_samples)
    # and the router must flatten it the way the af3 backend does.
    restore, _ = _px_fakes(batch_result=(ONE, ["s0", "s1"]), single_result=ONE)
    try:
        out = folding.fold_protenix(_px_jobs(3), _px_ctx("filter", want_all=True))
        check("each result is a 4-tuple", all(len(r) == 4 for r in out),
              [len(r) for r in out])
        check("the samples ride in the last slot",
              out[0][3] == ["s0", "s1"], out[0][3])
    finally:
        restore()


def test_a_failed_design_is_none_not_an_exception():
    restore, _ = _px_fakes(batch_result=None, single_result=ONE)
    try:
        out = folding.fold_protenix(_px_jobs(3), _px_ctx("filter"))
        check("a failure is None in its slot", out == [None, None, None], out)
        out = folding.fold_protenix(_px_jobs(3), _px_ctx("filter", want_all=True))
        check("and stays None through the want_all flatten",
              out == [None, None, None], out)
    finally:
        restore()


def test_an_unset_stage_does_not_batch():
    # A hand-built context defaults to "", which must take the safe path.
    restore, calls = _px_fakes(batch_result=ONE, single_result=ONE)
    try:
        folding.fold_protenix(_px_jobs(3), _px_ctx(""))
        check("an unset stage stays sequential", calls["batch"] == 0, calls)
    finally:
        restore()


# --- A batch that produced nothing at all ---
# All-None is how a batch backend reports failure, so the escalation lives in
# the entry point for every backend rather than in each one. af3 used to raise
# and protenix used to return Nones for this one condition.


def _dead_backend(name, results):
    """Register a backend returning `results`; returns a restore()."""
    saved = dict(folding._BACKENDS)
    folding.register_backend(name, lambda jobs, ctx: list(results(jobs)))

    def restore():
        folding._BACKENDS.clear()
        folding._BACKENDS.update(saved)
    return restore


def _fold_designs(designs, n_targets, backend):
    return folding.fold_designs_against_targets(
        designs=designs,
        target_sequences=[f"T{i}" for i in range(n_targets)],
        target_names=[f"t{i}" for i in range(n_targets)],
        target_chain="A",
        binder_chain="B",
        output_dir="/tmp/x",
        run_settings=_settings(filter_folder=_folder(backend)),
    )


def test_a_dead_batch_of_independent_designs_is_escalated():
    # Distinct binder sequences cannot all fail for design reasons, so this is
    # the environment fault that previously only warned on the protenix path.
    restore = _dead_backend("dead", lambda jobs: [None] * len(jobs))
    try:
        _fold_designs([("d0", "QVA"), ("d1", "QVB"), ("d2", "QVC")], 1, "dead")
        check("a wholly dead batch raises", False, "no exception")
    except folding.AllJobsFailed as e:
        check("a wholly dead batch of independent designs raises",
              "3 distinct binder sequences" in str(e), str(e))
        check("and names the stage and backend so the artifact says which fold",
              "filter stage (dead)" in str(e), str(e))
    finally:
        restore()


def test_a_dead_batch_of_one_designs_legs_is_not_escalated():
    # The carve-out sequential() already makes. Batches of a single design are
    # routine whenever few hallucinations pass, and one design failing to fold
    # must not look like a broken environment.
    restore = _dead_backend("dead", lambda jobs: [None] * len(jobs))
    try:
        out = _fold_designs([("d0", "QVA")], 3, "dead")
        check("one design's legs all failing does not raise",
              out == {"d0": [None, None, None]}, out)
    finally:
        restore()


def test_two_designs_sharing_a_sequence_are_still_one_failure():
    # Same binder, two names (a design and its rescored twin): still one
    # sequence, so still not independent evidence of an environment fault.
    restore = _dead_backend("dead", lambda jobs: [None] * len(jobs))
    try:
        out = _fold_designs([("d0", "QVA"), ("d0_rescore", "QVA")], 1, "dead")
        check("a shared binder sequence stays a per-design fault",
              out == {"d0": [None], "d0_rescore": [None]}, out)
    finally:
        restore()


def test_a_short_result_list_is_caught_before_the_dead_batch_check():
    # all() over an empty list is True, so the order matters: a backend that
    # returns nothing must report the positional mismatch, not a fold failure.
    restore = _dead_backend("short", lambda jobs: [])
    try:
        _fold_designs([("d0", "QVA"), ("d1", "QVB")], 1, "short")
        check("a short result list raises", False, "no exception")
    except folding.AllJobsFailed as e:
        check("a short result list is not mistaken for a dead batch",
              False, f"got AllJobsFailed instead of RuntimeError: {e}")
    except RuntimeError as e:
        check("a short result list reports the positional mismatch",
              "by position" in str(e), str(e))
    finally:
        restore()


def test_af3_producing_nothing_becomes_all_none_not_a_raise():
    # _run_af3_batch_inference raises when no design produced output. That
    # bypassed the entry point's decision and crashed the run even for a lone
    # design, which _run treats as a per-design fault.
    from germinal.filters import af3
    saved = af3._run_af3_batch_inference

    def boom(**kw):
        raise FileNotFoundError("AF3 produced no output for any of the 2 batched job(s).")

    af3._run_af3_batch_inference = boom
    try:
        jobs = [folding.FoldJob(name=f"d{i}", binder_sequence=f"QV{i}",
                                target_sequence="TT", target_index=0)
                for i in range(2)]
        out = folding.fold_af3(jobs, _px_ctx("filter"))
        check("af3 reports produced-nothing as all-None",
              out == [None, None], out)
    finally:
        af3._run_af3_batch_inference = saved
