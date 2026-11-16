"""Pins that a hallucination batch closes on passers OR attempts.

Phase A collects passers and phases B-D fold them in one call, so whatever
bounds phase A decides how long the run goes before writing any record. Both
bounds are needed: an arm that passes nothing never closes on passers alone,
and a lucky arm folds later than it needed to on attempts alone.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "configs" / "config.yaml"

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

def test_both_bounds_are_configured():
    text = CONFIG.read_text()
    for k in ("max_hallucination_passing", "max_hallucination_attempts"):
        check(f"{k} has a default", f"{k}:" in text)

def test_the_bounds_are_simulated_correctly():
    # The loop condition as a property, over both orders of tripping.
    def run(outcomes, max_passing, max_attempts):
        passers, attempts = 0, 0
        for ok in outcomes:
            if passers >= max_passing or attempts >= max_attempts:
                break
            attempts += 1
            if ok:
                passers += 1
        return passers, attempts

    # 10 passers arrive before 100 attempts: stop at the 10th passer.
    p, a = run([True] * 50, max_passing=10, max_attempts=100)
    check("a passing arm closes on passers", (p, a) == (10, 10), f"{p},{a}")

    # Nothing passes: stop at the attempt bound, not the trajectory budget.
    p, a = run([False] * 500, max_passing=10, max_attempts=100)
    check("a failing arm closes on attempts", (p, a) == (0, 100), f"{p},{a}")

    # Mixed, 1 in 4 passing: 40 attempts for 10 passers.
    p, a = run([False, False, False, True] * 100, max_passing=10, max_attempts=100)
    check("a mixed arm closes on passers", (p, a) == (10, 40), f"{p},{a}")


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print(f"\n{PASSED}/{PASSED + FAILED} passed")
    sys.exit(1 if FAILED else 0)
