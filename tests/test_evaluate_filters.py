"""Tests for ``filter_utils.evaluate_filters``, the gate every design passes.

Covers all six operators against pass/fail/boundary values; a missing metric
and a ``None`` metric, which must fail rather than pass; NaN; an unknown
operator; and the conjunction that makes one failure reject the design. Also
pins the shipped filter configs against the real metric names, so a renamed
metric cannot silently turn every gate into a reject.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from germinal.filters import filter_utils  # noqa: E402


def _one(metric, value, threshold, operator):
    """Evaluate a single filter and return (all_passed, this_filter_passed)."""
    passed, results = filter_utils.evaluate_filters(
        {metric: {"value": threshold, "operator": operator}}, {metric: value}
    )
    return passed, results[f"{metric}_filter"]


# --- The operator dispatch, exhaustively ---

# (operator, threshold, value, expected), three per operator: passing,
# failing, and exactly on the boundary, which is what separates < from <=.
_OPERATOR_CASES = [
    ("<",  5.0, 4.0,  True),
    ("<",  5.0, 6.0,  False),
    ("<",  5.0, 5.0,  False),   # boundary excluded
    ("<=", 5.0, 4.0,  True),
    ("<=", 5.0, 6.0,  False),
    ("<=", 5.0, 5.0,  True),    # boundary included
    (">",  5.0, 6.0,  True),
    (">",  5.0, 4.0,  False),
    (">",  5.0, 5.0,  False),   # boundary excluded
    (">=", 5.0, 6.0,  True),
    (">=", 5.0, 4.0,  False),
    (">=", 5.0, 5.0,  True),    # boundary included
    ("==", 5.0, 5.0,  True),
    ("==", 5.0, 4.0,  False),
    ("=",  5.0, 5.0,  True),    # alias for ==
    ("=",  5.0, 4.0,  False),
]


def test_every_operator_in_both_directions():
    for operator, threshold, value, expected in _OPERATOR_CASES:
        _, got = _one("metric", value, threshold, operator)
        assert got is expected or got == expected, (
            f"{value} {operator} {threshold} -> {got}, expected {expected}"
        )


def test_boundary_distinguishes_strict_from_inclusive():
    # The pair of assertions that catches < swapped for <= and > for >=.
    assert _one("m", 5.0, 5.0, "<")[1] is False
    assert _one("m", 5.0, 5.0, "<=")[1] is True
    assert _one("m", 5.0, 5.0, ">")[1] is False
    assert _one("m", 5.0, 5.0, ">=")[1] is True


def test_operators_are_not_inverted():
    # A direction swap in the dispatch would make each of these flip.
    assert _one("m", 1.0, 10.0, "<")[1] is True    # 1 < 10
    assert _one("m", 10.0, 1.0, "<")[1] is False   # 10 < 1
    assert _one("m", 10.0, 1.0, ">")[1] is True    # 10 > 1
    assert _one("m", 1.0, 10.0, ">")[1] is False   # 1 > 10


def test_booleans_compare_with_equality():
    # binder_near_hotspot ships as `value: true, operator: '=='`.
    assert _one("binder_near_hotspot", True, True, "==")[1] is True
    assert _one("binder_near_hotspot", False, True, "==")[1] is False


# --- Unmeasured metrics must reject, never pass ---

def test_missing_metric_raises_rather_than_failing_every_design():
    # A filter naming an absent metric is a config error, not a design
    # property, so it raises rather than rejecting every design.
    try:
        filter_utils.evaluate_filters(
            {"absent_metric": {"value": 1.0, "operator": ">="}},
            {"something_else": 5.0},
        )
        raise AssertionError("an absent metric must raise, not reject silently")
    except KeyError as e:
        assert "absent_metric" in str(e)
        # The message must name what IS available, or the next person has to
        # go read build_filter_metrics to find the right spelling.
        assert "something_else" in str(e)


def test_none_metric_fails_rather_than_raising():
    # None is what an early-rejected target and a failed sc_rmsd now report.
    # Comparing it would raise TypeError; it must fail instead.
    passed, results = filter_utils.evaluate_filters(
        {"sc_rmsd": {"value": 8.0, "operator": "<="}}, {"sc_rmsd": None}
    )
    assert results["sc_rmsd_filter"] is False
    assert passed is False


def test_nan_metric_fails_every_operator():
    # NaN compares False against everything, so it rejects. Accidental
    # rather than explicit, so pin it against a future "handle NaN" change.
    nan = float("nan")
    for operator in ("<", "<=", ">", ">=", "==", "="):
        _, got = _one("metric", nan, 5.0, operator)
        assert got is False or got == False, f"NaN {operator} 5.0 -> {got}"


def test_unknown_operator_fails_rather_than_passing():
    passed, results = filter_utils.evaluate_filters(
        {"metric": {"value": 5.0, "operator": "=~"}}, {"metric": 5.0}
    )
    assert results["metric_filter"] is False
    assert passed is False


# --- Conjunction: every filter must pass ---

def test_all_must_pass():
    filter_set = {
        "a": {"value": 1.0, "operator": ">="},
        "b": {"value": 1.0, "operator": ">="},
        "c": {"value": 1.0, "operator": ">="},
    }
    passed, results = filter_utils.evaluate_filters(
        filter_set, {"a": 2.0, "b": 2.0, "c": 2.0}
    )
    assert passed is True
    assert all(results.values())

    # One failure rejects the design.
    passed, results = filter_utils.evaluate_filters(
        filter_set, {"a": 2.0, "b": 0.0, "c": 2.0}
    )
    assert passed is False
    assert results["b_filter"] is False
    assert results["a_filter"] is True and results["c_filter"] is True


def test_empty_filter_set_accepts_everything():
    # An empty filter set means ungated, which is intended -- but it is also
    # what a mis-pathed Hydra config resolves to, so the two look alike.
    passed, results = filter_utils.evaluate_filters({}, {"anything": 1.0})
    assert results == {}
    assert passed is True


# --- The shipped configs, against the real metric names ---

def _load_shipped(path):
    import yaml

    with open(path) as handle:
        loaded = yaml.safe_load(handle)
    # Drop Hydra's @package directive if PyYAML surfaced it as a key.
    return {k: v for k, v in loaded.items() if isinstance(v, dict) and "value" in v}


def test_shipped_filter_keys_are_real_metric_names():
    # build_filter_metrics is the authority for what a design row contains,
    # so this is the test that catches a metric rename.
    import inspect

    source = inspect.getsource(filter_utils.build_filter_metrics)
    root = os.path.join(os.path.dirname(__file__), "..", "configs", "filter")
    checked = 0
    for sub in ("initial", "final"):
        directory = os.path.join(root, sub)
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".yaml"):
                continue
            for metric in _load_shipped(os.path.join(directory, name)):
                # Keys are "<stage>.<metric>": the stage must match the
                # directory, or the file applies another stage's thresholds.
                stage, _, bare = metric.partition(".")
                assert stage == sub, (
                    f"{sub}/{name}: filter key '{metric}' names the {stage!r} "
                    f"stage but lives in {sub}/"
                )
                assert bare, f"{sub}/{name}: filter key '{metric}' has no metric"
                assert f'"{bare}"' in source, (
                    f"{sub}/{name}: filter '{bare}' is not emitted by "
                    f"build_filter_metrics, so it would reject every design"
                )
                checked += 1
    assert checked > 0, "no shipped filter keys were checked"


def test_shipped_configs_accept_a_good_design_and_reject_a_bad_one():
    # End-to-end through the real thresholds: a design comfortably inside every
    # gate must pass, and nudging one metric to the wrong side must reject.
    path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "filter", "final", "scfv_multi.yaml"
    )
    filter_set = _load_shipped(path)
    assert filter_set, "shipped final filter set is empty"

    good, bad_key = {}, None
    for metric, config in filter_set.items():
        threshold, operator = config["value"], config["operator"]
        if operator in ("==", "="):
            good[metric] = threshold
        elif operator in ("<", "<="):
            good[metric] = threshold - 1 if threshold > 1 else threshold / 2
        else:
            good[metric] = threshold + 1
        if bad_key is None and operator not in ("==", "="):
            bad_key = metric

    passed, _ = filter_utils.evaluate_filters(filter_set, good)
    assert passed is True, "a design inside every shipped threshold was rejected"

    bad = dict(good)
    operator = filter_set[bad_key]["operator"]
    threshold = filter_set[bad_key]["value"]
    bad[bad_key] = threshold + 1 if operator in ("<", "<=") else threshold - 1
    passed, results = filter_utils.evaluate_filters(filter_set, bad)
    assert passed is False, f"a design failing {bad_key} was accepted"
    assert results[f"{bad_key}_filter"] is False


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


# --- backends report the same metric in different shapes ---
def test_a_metric_reaches_the_csv_whatever_shape_it_arrives_in():
    # af3 reports numpy scalars; protenix reports 1-element lists for ptm,
    # iptm and aggregate_score. float([0.8]) raises, which crashed the filter
    # stage for every design that got past the early check.
    from germinal.filters.filter_utils import _as_float
    import numpy as np

    for value, want in [
        (np.float64(0.87), 0.87),
        (np.float32(0.5), 0.5),
        (0.8, 0.8),
        ([0.8], 0.8),
        ((0.8,), 0.8),
        (np.array([0.8]), 0.8),
    ]:
        got = _as_float(value)
        assert isinstance(got, float), f"{value!r} gave {type(got).__name__}"
        assert abs(got - want) < 1e-6, f"{value!r} gave {got}, wanted {want}"
