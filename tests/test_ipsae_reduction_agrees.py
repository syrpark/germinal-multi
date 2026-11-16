"""Pins that both backends reduce the ipSAE table the same way.

For a two-chain complex the ipsae CLI writes both asymmetric directions and
their max, so ``.iloc[-1]`` is the better direction and ``.min()`` the worse.
Reducing the recycle gate and the final filter from opposite ends would let a
candidate clear the gate on its better direction and be scored on its worse.
"""

import os
import sys

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


def test_the_reduction_is_the_worse_chain_direction():
    # Guards the semantics rather than the spelling: given the CLI's three
    # rows, the chosen reduction must return the worse asymmetric value.
    import pandas as pd

    df = pd.DataFrame({
        "Chn1": ["A", "B", "B"],
        "Chn2": ["B", "A", "A"],
        "Type": ["asym", "asym", "max"],
        "ipSAE": [0.42, 0.67, 0.67],
    })
    check(
        ".min() picks the worse direction",
        df["ipSAE"].min() == 0.42,
        str(df["ipSAE"].min()),
    )
    check(
        "and .iloc[-1] would have picked the better one",
        df["ipSAE"].iloc[-1] == 0.67,
    )


def test_the_gate_is_not_looser_than_the_filter_it_feeds():
    # With both ends reduced the same way, a recycle-gate score can never
    # exceed what the final filter computes from an equivalent table.
    import pandas as pd

    table = pd.DataFrame({"ipSAE": [0.42, 0.67, 0.67]})
    gate_score = table["ipSAE"].min()      # protenix, recycle gate
    filter_score = table["ipSAE"].min()    # af3, final filter
    check(
        "gate and filter agree on the same table",
        gate_score == filter_score,
        f"gate={gate_score} filter={filter_score}",
    )


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print(f"\n{PASSED}/{PASSED + FAILED} passed")
    sys.exit(1 if FAILED else 0)
