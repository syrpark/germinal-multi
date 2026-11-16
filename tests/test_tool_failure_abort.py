"""A broken tool must stop the run, not be recorded as thousands of rejections.

When AbMPNN cannot start -- wrong submodule, missing abmpnn.pt, no GPU the
child can open -- every design fails for the same reason. Each one is
individually indistinguishable from a design that simply was not good enough,
so without a consecutive-failure count the run ends with a 0% acceptance rate
and no error. These pin the counter's contract: it fires only on an unbroken
streak, the caller's reset breaks the streak, and 0 disables it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_germinal_multi import _tool_failure


def test_a_streak_shorter_than_the_limit_only_counts():
    count = 0
    for expected in (1, 2, 3):
        count = _tool_failure(count, 10, "proteinmpnn_failed")
        assert count == expected


def test_the_run_aborts_when_the_streak_reaches_the_limit():
    count = 2
    with pytest.raises(RuntimeError) as exc:
        _tool_failure(count, 3, "proteinmpnn_failed")
    message = str(exc.value)
    # The message has to name the cause and what to check, since this is the
    # only thing the user sees.
    assert "proteinmpnn_failed" in message
    assert "max_consecutive_tool_failures" in message


def test_zero_disables_the_check():
    count = 0
    for _ in range(50):
        count = _tool_failure(count, 0, "proteinmpnn_failed")
    assert count == 50


def test_recycle_reports_tool_failure_as_a_flag():
    """The caller must not have to parse truncation_reason to know."""
    from germinal.design import recycle as recycle_mod
    import inspect

    src = inspect.getsource(recycle_mod.recycle)
    assert '"tool_failure": False' in src
    assert 'info["tool_failure"] = redesign_reason in' in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
