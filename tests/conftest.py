"""Let the suite run without PyRosetta.

`germinal.filters.pyrosetta_utils` imports pyrosetta at module scope, and
`filter_utils` imports that module, so three test modules could not even be
collected without a PyRosetta install -- an academic-license download of 1.7 GB
that none of them actually exercise. They import `filter_utils` for its filter
evaluation and design classification logic, not for Rosetta scoring.

So: if pyrosetta is importable, nothing here happens and the tests run against
the real thing. If it is missing, a stub stands in for it, and the stub is
announced loudly rather than quietly papering over a broken environment.

The stub is deliberately dumb. Any attribute resolves to a dummy that raises on
*call*, so a test that strays into a real Rosetta code path fails with a clear
message instead of silently passing against a fake.
"""

import importlib.util
import sys
import types


def _pyrosetta_is_installed() -> bool:
    try:
        return importlib.util.find_spec("pyrosetta") is not None
    except (ImportError, ValueError):
        return False


class _Unavailable:
    """Stands in for a Rosetta symbol. Importable, but not usable."""

    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self._name} needs a real PyRosetta install; this is the stub from "
            "tests/conftest.py. A test reaching this is exercising a Rosetta code "
            "path and must be skipped without PyRosetta, not run against a fake."
        )

    def __getattr__(self, attr: str) -> "_Unavailable":
        return _Unavailable(f"{self._name}.{attr}")


class _StubModule(types.ModuleType):
    def __getattr__(self, attr: str) -> _Unavailable:
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _Unavailable(f"{self.__name__}.{attr}")


class _StubFinder:
    """Fabricates any `pyrosetta...` submodule on demand.

    pyrosetta_utils imports ~15 deep paths (pyrosetta.rosetta.protocols.relax
    and friends); a finder covers all of them without listing any.
    """

    _PREFIX = "pyrosetta"

    def find_module(self, fullname, path=None):  # pragma: no cover - legacy API
        return self if self._handles(fullname) else None

    def _handles(self, fullname: str) -> bool:
        return fullname == self._PREFIX or fullname.startswith(self._PREFIX + ".")

    def find_spec(self, fullname, path=None, target=None):
        if not self._handles(fullname):
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        module = _StubModule(spec.name)
        module.__path__ = []  # mark as a package so submodule imports resolve
        return module

    def exec_module(self, module):
        return None


if not _pyrosetta_is_installed():
    print(
        "\n[conftest] PyRosetta is not installed: substituting a stub so the "
        "suite can be collected.\n"
        "[conftest] Rosetta code paths are NOT covered by this run. Install "
        "PyRosetta (environment_setup.md) for the full suite.\n",
        file=sys.stderr,
    )
    sys.meta_path.insert(0, _StubFinder())
