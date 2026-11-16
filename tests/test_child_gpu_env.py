"""The environment handed to GPU child processes.

Germinal shells out to four GPU programs -- AbMPNN, AF3, Protenix, mmseqs --
while holding its own JAX context. On an Exclusive_Process cluster a card
admits one process, so those children have to be pinned to a different card
than the parent, which `child_env` does from GERMINAL_CHILD_CUDA_DEVICE.

The property that matters most here is the *absence* of an effect: with the
variable unset, a call routed through this helper must behave exactly as it
did before, on every cluster that never had this problem. That is why unset
returns None rather than a copy of os.environ -- subprocess treats env=None
identically to an omitted env, so the child inherits the live environment
even if a C extension has called setenv() since interpreter startup and
os.environ never saw it.
"""

import os

import pytest

from germinal.utils.gpu import child_env

VAR = "GERMINAL_CHILD_CUDA_DEVICE"


@pytest.fixture(autouse=True)
def _clear_var(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)


def test_unset_inherits_rather_than_copying():
    assert child_env() is None, (
        "with no pinning and no blocklist the helper must return None, so the "
        "child inherits exactly what it inherits today; returning a dict here "
        "would silently drop any variable set via C-level setenv()"
    )


def test_pin_overrides_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv(VAR, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env = child_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_pin_preserves_the_rest_of_the_environment(monkeypatch):
    monkeypatch.setenv(VAR, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("GERMINAL_TEST_CANARY", "keep-me")
    env = child_env()
    assert env["GERMINAL_TEST_CANARY"] == "keep-me"
    # Everything except the pinned variable matches the parent.
    differing = {
        k for k in set(env) | set(os.environ) if env.get(k) != os.environ.get(k)
    }
    assert differing == {"CUDA_VISIBLE_DEVICES"}


def test_blocklist_still_strips_without_a_pin(monkeypatch):
    """AF3's container needs this even when nothing is pinned.

    _container_env() delegates here, so the blocklist has to work on its own:
    the deprecated XLA variable crashes the container's jaxlib, and that is
    true on a Default-mode cluster that never sets GERMINAL_CHILD_CUDA_DEVICE.
    """
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    env = child_env(("XLA_PYTHON_CLIENT_MEM_FRACTION",))
    assert env is not None, "a blocklist alone must still produce an env"
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in env


def test_af3_container_env_drops_the_xla_variable(monkeypatch):
    """The real caller, not just the helper -- guards the af3.py refactor."""
    from germinal.filters.af3 import _container_env

    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    monkeypatch.setenv(VAR, "1")
    env = _container_env()
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_child_actually_sees_the_pin(monkeypatch):
    """End to end through subprocess, since that is the only thing that counts."""
    import subprocess
    import sys

    monkeypatch.setenv(VAR, "7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    out = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"],
        capture_output=True,
        text=True,
        env=child_env(),
    )
    assert out.stdout.strip() == "7", out.stderr
