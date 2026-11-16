"""Tests for how a rejected design is sorted and attributed.

Covers the routing of a final-filter rejection into ``filter_failed/`` and the
attribution of that rejection to a filter in ``failure_counts.csv``.

No GPU, no AF3 and no network: the structure file is a one-atom stand-in and
the filter dictionaries are shaped like ``multi_run_filters(is_final=True)``.
"""

import os
import shutil
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from germinal.filters import filter_utils  # noqa: E402
from germinal.utils import utils  # noqa: E402
from germinal.utils.io import IO, RunLayout, Trajectory  # noqa: E402

STUB_PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n"


class _Run:
    """A throwaway results directory with its IO wrapper."""

    def __init__(self):
        self.tmp = tempfile.mkdtemp(prefix="germinal_test_")
        self.layout = RunLayout.create(os.path.join(self.tmp, "run"))
        self.io = IO(self.layout)
        self.struct = os.path.join(self.tmp, "stub.pdb")
        with open(self.struct, "w") as fh:
            fh.write(STUB_PDB)
        self.struct2 = os.path.join(self.tmp, "stub2.pdb")
        shutil.copy(self.struct, self.struct2)

    def rejected(self, name):
        """A trajectory shaped like one that just failed the final filters."""
        t = Trajectory(name, "test", "8,8,13,6,3,9", "A43,A98,A133")
        t.update_filtering_metrics(
            {
                "plddt": 0.71,
                "ipsae": 0.30,
                "plddt_filter": False,
                "ipsae_filter": False,
            }
        )
        t.update_other_metrics({"failure_stage": "initial_filter"})
        t.set_save_location("filter_failed")
        return t

    def rows(self, csv_path):
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return 0
        return len(pd.read_csv(csv_path))

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def test_single_target_rejection_lands_in_filter_failed():
    r = _Run()
    try:
        t = r.rejected("design_single")
        # Structures are always indexed per target, so a single-target design
        # writes design_single_0.pdb. A bare-string final_struct still works.
        t.set_final_struct(r.struct)
        assert r.io.save_trajectory(t) == 0

        assert (
            r.layout.filter_failed / "structures" / "design_single_0.pdb"
        ).exists()
        df = pd.read_csv(r.layout.filter_failed / "designs.csv")
        assert len(df) == 1
        # The stage label must survive into the CSV, or tallying by it is
        # impossible -- it used to be blank for every death but one.
        assert df.iloc[0]["failure_stage"] == "initial_filter"
    finally:
        r.cleanup()


def test_dual_target_rejection_copies_both_legs():
    r = _Run()
    try:
        t = r.rejected("design_dual")
        t.set_final_struct([r.struct, r.struct2])
        assert r.io.save_trajectory(t) == 0

        for leg in (0, 1):
            assert (
                r.layout.filter_failed / "structures" / f"design_dual_{leg}.pdb"
            ).exists(), f"missing leg {leg}"
        assert r.rows(r.layout.filter_failed / "designs.csv") == 1
    finally:
        r.cleanup()


def test_accepted_and_rejected_stay_separate():
    r = _Run()
    try:
        bad = r.rejected("design_bad")
        bad.set_final_struct([r.struct, r.struct2])
        r.io.save_trajectory(bad)

        good = r.rejected("design_good")
        good.set_save_location("accepted")
        good.set_final_struct([r.struct, r.struct2])
        r.io.save_trajectory(good)

        assert r.rows(r.layout.accepted / "designs.csv") == 1
        assert r.rows(r.layout.filter_failed / "designs.csv") == 1
        # redesign_candidates was the dumping ground for rejections; nothing
        # should route there any more.
        assert r.rows(r.layout.redesign_candidates / "designs.csv") == 0
        # Both still aggregate into the run-wide CSV.
        assert r.rows(r.layout.all_csv) == 2
    finally:
        r.cleanup()


def test_only_failing_filters_are_charged():
    r = _Run()
    try:
        filter_results = {
            "plddt_filter": False,
            "ipsae_filter": False,
            "iptm_filter": True,
            "interface_hbonds_filter": False,
        }
        filter_metrics = {
            "plddt": 0.71,
            "ipsae": 0.30,
            "iptm": 0.80,
            "interface_hbonds": 2,
        }
        failed = filter_utils.select_failed_filters(filter_results, filter_metrics)
        assert set(failed) == {
            "plddt_filter",
            "ipsae_filter",
            "interface_hbonds_filter",
        }

        r.io.update_failures(failed)
        df = pd.read_csv(r.layout.failure_csv)
        for key in failed:
            assert int(df.iloc[0][key]) == 1, f"{key} not recorded"
        # A filter the design passed must never appear.
        assert "iptm_filter" not in df.columns

        # A second rejection accumulates rather than overwriting.
        r.io.update_failures(failed)
        assert int(pd.read_csv(r.layout.failure_csv).iloc[0]["ipsae_filter"]) == 2
    finally:
        r.cleanup()


def test_unmeasured_metric_is_not_blamed_on_its_filter():
    """A metric that was never measured still rejects the design, but charging
    it to that filter would report a filter failure that did not happen."""
    failed = filter_utils.select_failed_filters(
        {"surface_hydrophobicity_filter": False, "ipsae_filter": False},
        {"surface_hydrophobicity": None, "ipsae": 0.1},
    )
    assert "surface_hydrophobicity_filter" not in failed
    assert "_unmeasured_metric" in failed
    assert failed.get("ipsae_filter") == 1


def test_dual_run_charges_only_the_rejected_legs_measured_filters():
    """One early-rejected leg must not read as a dozen filter failures.

    When one leg is early-rejected and the other is not, the aggregate still
    carries the measured leg's value, so judging by it charges every filter
    for a rejection whose real cause was one confidence metric.
    """
    interface = [
        "clashes",
        "sc_rmsd",
        "binder_near_hotspot",
        "percent_interface_cdr",
        "interface_shape_comp",
        "interface_hbonds",
        "surface_hydrophobicity",
        "interface_hydrophobicity",
    ]
    confidence = {"iptm": 0.65, "ptm": 0.77, "ipsae": 0.253947}

    filter_results = {f"{m}_filter": False for m in interface}
    filter_results.update({f"{m}_filter": False for m in confidence})
    filter_results["plddt_filter"] = True

    filter_metrics = {
        "failed_target": "egfr_mut",
        "failure_stage": "early_confidence_filter",
        "plddt": 0.90,
        "plddt.egfr_mut": 0.90,
    }
    # Aggregates look passing because they come from the *other* leg.
    filter_metrics.update(
        {
            "clashes": 0.0,
            "sc_rmsd": 2.71,
            "binder_near_hotspot": True,
            "percent_interface_cdr": 0.9286,
            "interface_shape_comp": 0.584,
            "interface_hbonds": 9.8,
            "surface_hydrophobicity": 0.394,
            "interface_hydrophobicity": 58.466,
        }
    )
    # The rejected leg never had them computed.
    filter_metrics.update({f"{m}.egfr_mut": float("nan") for m in interface})
    # The confidence metrics were measured on the rejected leg.
    for m, v in confidence.items():
        filter_metrics[m] = v
        filter_metrics[f"{m}.egfr_mut"] = v

    failed = filter_utils.select_failed_filters(filter_results, filter_metrics)

    assert set(failed) == {
        "iptm_filter",
        "ptm_filter",
        "ipsae_filter",
        "_short_circuit_early_confidence",
    }, f"over-attributed: {sorted(failed)}"
    for m in interface:
        assert f"{m}_filter" not in failed, f"{m} was never measured on the rejected leg"


def test_nan_counts_as_unmeasured():
    """Per-leg metrics arrive as NaN, not None, so a None-only test misses them."""
    failed = filter_utils.select_failed_filters(
        {"interface_hbonds_filter": False},
        {
            "failed_target": "leg_b",
            "failure_stage": "early_confidence_filter",
            "interface_hbonds": 9.8,
            "interface_hbonds.leg_b": float("nan"),
        },
    )
    assert "interface_hbonds_filter" not in failed
    assert "_short_circuit_early_confidence" in failed


def test_failed_leg_index_resolves_per_leg_columns():
    """Per-leg columns are keyed .leg<i>; failed_leg is what reaches them.

    Without failed_leg the lookup falls back to the aggregate, which carries
    the measured leg's value.
    """
    failed = filter_utils.select_failed_filters(
        {"ipsae_filter": False, "interface_hbonds_filter": False},
        {
            "failed_target": "egfr_mut",
            "failed_leg": 1,
            "failure_stage": "early_confidence_filter",
            "min_ipsae": 0.2,
            "ipsae.leg0": 0.61,
            "ipsae.leg1": 0.2,
            "min_interface_hbonds": 9.8,
            "interface_hbonds.leg0": 9.8,
            "interface_hbonds.leg1": float("nan"),
        },
    )
    assert failed.get("ipsae_filter") == 1, "measured on the failed leg"
    assert "interface_hbonds_filter" not in failed, (
        "never measured on the failed leg; the aggregate's 9.8 is leg0's"
    )
    assert "_short_circuit_early_confidence" in failed


def test_target_named_per_leg_columns_still_read():
    """Older CSVs are keyed by target name, not leg index."""
    failed = filter_utils.select_failed_filters(
        {"interface_hbonds_filter": False},
        {
            "failed_target": "egfr_mut",
            "failure_stage": "early_confidence_filter",
            "interface_hbonds": 9.8,
            "interface_hbonds.egfr_mut": float("nan"),
        },
    )
    assert "interface_hbonds_filter" not in failed


def test_single_target_behaviour_is_unchanged():
    """With no failed_target the aggregate is still the test, as before."""
    failed = filter_utils.select_failed_filters(
        {"ipsae_filter": False, "interface_hbonds_filter": False},
        {"ipsae": 0.2, "interface_hbonds": None},
    )
    assert failed.get("ipsae_filter") == 1
    assert "interface_hbonds_filter" not in failed
    assert "_unmeasured_metric" in failed


def test_failure_stage_defaults_to_the_reporting_site():
    """A rejection with no stage in its filter data gets the caller's label.

    A blank failure_stage is indistinguishable from a design that passed
    when tallying by that column.
    """
    assert utils.resolve_failure_stage({}, "final_filter") == "final_filter"
    assert utils.resolve_failure_stage({}, "initial_filter") == "initial_filter"
    # A null is as blank as a missing key.
    assert (
        utils.resolve_failure_stage({"failure_stage": None}, "final_filter")
        == "final_filter"
    )


def test_nan_stage_is_treated_as_absent():
    """NaN is truthy, so `stage or default` would keep the NaN.

    Only reachable once the metrics have been round-tripped through a
    DataFrame, where a missing value arrives as NaN rather than absent.
    """
    assert (
        utils.resolve_failure_stage(
            {"failure_stage": float("nan")}, "final_filter"
        )
        == "final_filter"
    )
    # An empty string is blank too.
    assert (
        utils.resolve_failure_stage({"failure_stage": ""}, "final_filter")
        == "final_filter"
    )


def test_early_confidence_stage_is_not_overwritten():
    """The filter code's own stage is more specific than the call site's."""
    assert (
        utils.resolve_failure_stage(
            {"failure_stage": "early_confidence_filter"}, "final_filter"
        )
        == "early_confidence_filter"
    )


def test_accepted_design_gets_no_failure_stage():
    """failure_stage is a rejection label; an accepted design must not carry one."""
    r = _Run()
    try:
        t = r.rejected("design_accepted")
        # Mirror the entry points: the key is only added when not accepted.
        t.other_metrics.pop("failure_stage", None)
        t.set_save_location("accepted")
        t.set_final_struct([r.struct, r.struct2])
        r.io.save_trajectory(t)

        df = pd.read_csv(r.layout.accepted / "designs.csv")
        assert len(df) == 1
        stage = df.iloc[0].get("failure_stage")
        assert stage is None or pd.isna(stage), f"accepted row carries stage {stage!r}"
    finally:
        r.cleanup()


def test_invalid_save_location_is_rejected():
    r = _Run()
    try:
        t = r.rejected("design_bogus")
        t.set_save_location("not_a_category")
        t.set_final_struct(r.struct)
        try:
            r.io.save_trajectory(t)
        except ValueError:
            return
        raise AssertionError("an unknown save_location should raise ValueError")
    finally:
        r.cleanup()


def _abmpnn(seq, score, seqid):
    return {"seq": seq, "score": score, "seqid": seqid}


def test_codesign_false_gives_abmpnn_only():
    """include_codesign=False is the recycling path, whose loop already
    folded and scored the baseline as one of its own candidates."""
    cands = utils.build_final_candidates(
        "d1",
        "CODESIGNSEQ",
        [_abmpnn("AAA", 0.7, 0.4), _abmpnn("BBB", 0.8, 0.5)],
        include_codesign=False,
    )
    assert [c["name"] for c in cands] == ["d1_abmpnn_1", "d1_abmpnn_2"]
    assert [c["seq"] for c in cands] == ["AAA", "BBB"]
    assert all(c["source"] == "abmpnn" for c in cands)


def test_codesign_candidate_is_prepended_as_abmpnn_0():
    """The co-designed sequence leads the list and keeps the AbMPNN numbering."""
    cands = utils.build_final_candidates(
        "d1",
        "CODESIGNSEQ",
        [_abmpnn("AAA", 0.7, 0.4), _abmpnn("BBB", 0.8, 0.5)],
        include_codesign=True,
    )
    assert [c["name"] for c in cands] == [
        "d1_abmpnn_0",
        "d1_abmpnn_1",
        "d1_abmpnn_2",
    ]
    assert cands[0]["seq"] == "CODESIGNSEQ"
    assert cands[0]["source"] == "codesign"
    # Adding it must not renumber or reorder the proposals.
    assert [c["seq"] for c in cands[1:]] == ["AAA", "BBB"]


def test_codesign_candidate_has_no_mpnn_score():
    """score/seqid are None, not 0.0, which would read as a perfect score."""
    cands = utils.build_final_candidates(
        "d1", "CODESIGNSEQ", [_abmpnn("AAA", 0.7, 0.4)], include_codesign=True
    )
    assert cands[0]["score"] is None
    assert cands[0]["seqid"] is None
    assert cands[1]["score"] == 0.7
    assert cands[1]["seqid"] == 0.4


def test_codesign_candidate_survives_an_empty_redesign():
    """If AbMPNN returned nothing, the co-designed sequence is still filtered."""
    cands = utils.build_final_candidates(
        "d1", "CODESIGNSEQ", [], include_codesign=True
    )
    assert len(cands) == 1
    assert cands[0]["name"] == "d1_abmpnn_0"
    assert (
        utils.build_final_candidates(
            "d1", "CODESIGNSEQ", [], include_codesign=False
        )
        == []
    )


def test_include_codesign_has_no_default():
    """It must be stated at every call site.

    The co-designed sequence always competes, so a default here would let a
    new call site silently drop it from filtering.
    """
    import inspect

    sig = inspect.signature(utils.build_final_candidates)
    param = sig.parameters["include_codesign"]
    assert param.default is inspect.Parameter.empty, (
        f"include_codesign has default {param.default!r}; it must be explicit"
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
