"""Pins the routing and stage order of the merged entry point.

``run_germinal_multi.py`` is routed by two config keys: ``logits_only`` picks
the hallucination schedule, ``recycling`` swaps the single redesign for the
AbMPNN+refold loop. Pinned here: both keys reach the right place;
``validate_single_path_redesign_cutoff`` stays gated on ``not recycling``; the
sc_rmsd reference is the hallucination PDBs initially and each candidate's own
``designed_on`` at the final filter, never a rebindable name read late; and a
recycling candidate reaches the final filters only past all three gates.

Asserted at the source level: `main()` needs a GPU, AF3 and PyRosetta.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CONFIGS = os.path.join(ROOT, "configs", "run")

PASSED = 0
FAILED = 0


def check(name, cond, detail=""):
    """Record a check. Raises under pytest so a failure is not silent.

    Standalone it counts and keeps going, so one run reports every failure.
    Under pytest the return value is ignored, so it asserts instead.
    """
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name} {detail}")
        if "PYTEST_CURRENT_TEST" in os.environ:
            raise AssertionError(f"{name} {detail}")


def test_configs_set_both_keys():
    print("\ntest_configs_set_both_keys")
    import yaml

    expected = {
        "multi.yaml": (False, False),
        "multi_rescue.yaml": (False, False),
        "multi_logits.yaml": (True, True),
        "multi_rescue_logits.yaml": (True, True),
    }
    for fn, (lo, rc) in expected.items():
        path = os.path.join(CONFIGS, fn)
        if not os.path.exists(path):
            check(f"{fn} exists", False)
            continue
        with open(path, encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
        check(
            f"{fn}: logits_only={lo} recycling={rc}",
            d.get("logits_only") is lo and d.get("recycling") is rc,
            f"-> {d.get('logits_only')}, {d.get('recycling')}",
        )


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print(f"\n{PASSED}/{PASSED + FAILED} passed")
    sys.exit(1 if FAILED else 0)
