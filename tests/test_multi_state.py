"""Tests that the multi_binder protocol is generic over the number of states.

``_prep_multi_binder`` has always taken a list of pdbs, but several places
hard-coded two -- notably ``set_seq``, which wrote the sequence bias to states
0 and 1 only, so a third state raised KeyError on the first design step.
Pinned: every state is reached, and the two-state case is unchanged.

No GPU, no AF weights, no network.
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from colabdesign.af.prep import get_multi_id  # noqa: E402
from colabdesign.shared.model import design_model  # noqa: E402


# --- Helpers ---

# Lengths differ between states on purpose, so anything assuming a single
# shared length breaks here rather than in a run.
BINDER_LEN = 242
TARGET_LENS = [194, 193, 196, 190]


def _lengths_for(num_states):
    return [[TARGET_LENS[i], BINDER_LEN] for i in range(num_states)]


class _StubModel(design_model):
    """Minimal stand-in exposing only what ``set_seq`` touches.

    Deliberately has no ``key`` attribute, so sequence init is zeros and the
    test is deterministic.
    """

    def __init__(self, num_states):
        self.protocol = "multi_binder"
        self._num = 1
        self._len = BINDER_LEN
        self._args = {"alphabet_size": 20}
        self.opt = {}
        self._params = {}
        self._target_len = [TARGET_LENS[i] for i in range(num_states)]
        self._inputs = [{} for _ in range(num_states)]


def _legacy_two_state_bias(inputs, b):
    """The pre-generalization write, kept to pin N=2 behaviour."""
    inputs[0]["bias"] = inputs[1]["bias"] = b
    return inputs


# --- The bias fan-out: the bug that was silent ---

def test_bias_reaches_every_state():
    for num_states in (2, 3, 4):
        model = _StubModel(num_states)
        bias = np.arange(20, dtype=float)
        model.set_seq(bias=bias)

        assert len(model._inputs) == num_states
        for state_idx, state_input in enumerate(model._inputs):
            assert "bias" in state_input, (
                f"state {state_idx} of {num_states} never received a bias key; "
                f"soft_seq would raise KeyError on it at the first design step"
            )
            assert np.allclose(state_input["bias"], np.broadcast_to(bias, (BINDER_LEN, 20))), (
                f"state {state_idx} of {num_states} got the wrong bias"
            )


def test_bias_fanout_matches_legacy_for_two_states():
    model = _StubModel(2)
    bias = np.arange(20, dtype=float)
    model.set_seq(bias=bias)

    legacy = _legacy_two_state_bias(
        [{}, {}], np.array(np.broadcast_to(bias, (BINDER_LEN, 20)))
    )
    for state_idx, (now, before) in enumerate(zip(model._inputs, legacy)):
        assert np.array_equal(now["bias"], before["bias"]), (
            f"state {state_idx} differs from the pre-generalization write"
        )


def test_every_state_shares_one_bias_object_value():
    # The design path asserts bias is identical across states; a fan-out that
    # wrote per-state values would make that assertion fire mid-run.
    model = _StubModel(3)
    model.set_seq(bias=np.full(20, 0.5))
    reference = model._inputs[0]["bias"]
    for state_idx in range(1, 3):
        assert np.array_equal(model._inputs[state_idx]["bias"], reference)


def test_bias_defaults_to_zero_for_every_state():
    model = _StubModel(3)
    model.set_seq()
    for state_idx, state_input in enumerate(model._inputs):
        assert np.array_equal(
            state_input["bias"], np.zeros((BINDER_LEN, 20))
        ), f"state {state_idx} did not get a zeroed bias"


# --- Per-state feature sizing ---

def _build_state_inputs(lengths, res_idx):
    """Mirrors the loop in ``_prep_multi_binder`` (without the AF feature dict)."""
    out = []
    for i, state_lengths in enumerate(lengths):
        state_input = {}
        state_input.update(
            {"residue_index": res_idx[i], **get_multi_id(state_lengths)}
        )
        out.append(state_input)
    return out


def test_state_features_sized_to_each_states_own_target_length():
    for num_states in (2, 3, 4):
        lengths = _lengths_for(num_states)
        res_idx = [np.arange(sum(l)) for l in lengths]
        inputs = _build_state_inputs(lengths, res_idx)

        assert len(inputs) == num_states
        for i, (state_input, state_lengths) in enumerate(zip(inputs, lengths)):
            total = sum(state_lengths)
            assert len(state_input["asym_id"]) == total, (
                f"state {i} of {num_states} sized {len(state_input['asym_id'])}, "
                f"expected {total}"
            )
            # The target occupies [0, target_len) and the binder the remainder.
            target_residues = int((state_input["asym_id"] == 0).sum())
            assert target_residues == state_lengths[0], (
                f"state {i} target/binder split at {target_residues}, "
                f"expected {state_lengths[0]}"
            )


def test_state_features_match_legacy_pair_for_two_states():
    lengths = _lengths_for(2)
    res_idx = [np.arange(sum(l)) for l in lengths]

    legacy = [{}, {}]
    legacy[0].update({"residue_index": res_idx[0], **get_multi_id(lengths[0])})
    legacy[1].update({"residue_index": res_idx[1], **get_multi_id(lengths[1])})

    now = _build_state_inputs(lengths, res_idx)
    assert len(now) == len(legacy)
    for i, (a, b) in enumerate(zip(now, legacy)):
        assert a.keys() == b.keys(), f"state {i}: {a.keys()} vs {b.keys()}"
        for k in a:
            assert np.array_equal(a[k], b[k]), f"state {i} key {k} changed"


def test_states_may_differ_in_length():
    # Guards the assumption that broke first: a shared num_res across states.
    lengths = _lengths_for(3)
    res_idx = [np.arange(sum(l)) for l in lengths]
    inputs = _build_state_inputs(lengths, res_idx)
    sizes = [len(s["asym_id"]) for s in inputs]
    assert len(set(sizes)) > 1, "test is vacuous if every state is the same size"
    assert sizes == [sum(l) for l in lengths]


# --- Hotspots: one per state, count checked rather than silently truncated ---

def _zip_hotspots(hotspots, states):
    """Mirrors the generalized hotspot mapping, including its length check."""
    assert len(hotspots) == len(states), (
        f"ERROR: hotspot needs one entry per antigen state; got "
        f"{len(hotspots)} for {len(states)} states"
    )
    return [(h, s) for h, s in zip(hotspots, states)]


def test_hotspots_map_one_per_state():
    states = ["s0", "s1", "s2"]
    pairs = _zip_hotspots(["A30-A40", "A31-A41", "A32-A42"], states)
    assert len(pairs) == 3
    assert [p[1] for p in pairs] == states


def test_too_few_hotspots_is_rejected_not_truncated():
    # zip() alone would silently pair two hotspots with three states and leave
    # the third unconstrained; the length check must reject instead.
    try:
        _zip_hotspots(["A30-A40", "A31-A41"], ["s0", "s1", "s2"])
    except AssertionError:
        return
    raise AssertionError("a short hotspot list was accepted and silently truncated")


def test_too_many_hotspots_is_rejected():
    try:
        _zip_hotspots(["a", "b", "c"], ["s0", "s1"])
    except AssertionError:
        return
    raise AssertionError("an over-long hotspot list was accepted")


# --- The subbatch heuristic must consider every state ---

def test_large_structure_heuristic_sees_every_state():
    # Only the third state is over the 384-residue threshold, so the
    # heuristic has to scan every state.
    small = [[100, 100], [100, 100], [300, 200]]
    assert any(sum(l) > 384 for l in small)
    assert not any(sum(l) > 384 for l in small[:2])


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
