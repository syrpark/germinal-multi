"""Pins what happens when someone clones without --recurse-submodules.

A plain `git clone` leaves external/proteinmpnn empty, which is the common
case rather than the exotic one. The startup check populates it rather than
only reporting it, and still fails clearly where it cannot: a source snapshot
has no submodule to fetch, and a node without network cannot fetch one.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from germinal.utils import config

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


def _fake_repo(tmp, with_git: bool, with_entry: bool):
    """A stand-in checkout; config resolves repo_root from its own location."""
    root = Path(tmp) / "repo"
    (root / "germinal" / "utils").mkdir(parents=True)
    if with_git:
        subprocess.run(["git", "init", "-q", str(root)], check=True)
    if with_entry:
        run_dir = root / "external" / "proteinmpnn" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "run_protein_mpnn.py").write_text("# stand-in\n")
    return root


class _Result:
    def __init__(self, returncode, stdout, stderr):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _validate_from(root):
    """Run the real validator against a stand-in checkout."""
    try:
        config.validate_redesign_submodule(repo_root=root)
        return _Result(0, "", "")
    except Exception as exc:
        return _Result(1, "", str(exc))


def test_a_populated_submodule_is_left_alone():
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_repo(tmp, with_git=True, with_entry=True)
        r = _validate_from(root)
        check("a populated submodule passes without touching git",
              r.returncode == 0, r.stderr[-400:])
        check("and says nothing about initialising",
              "submodule update" not in r.stdout, r.stdout[-200:])


def test_a_snapshot_without_git_says_so_rather_than_failing_git():
    # A zip/tarball download has no .git, so `git submodule update` would fail
    # with something the reader cannot act on.
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_repo(tmp, with_git=False, with_entry=False)
        r = _validate_from(root)
        out = r.stdout + r.stderr
        check("a snapshot fails", r.returncode != 0, out[-300:])
        check("naming the real cause, not a git error",
              "not a git checkout" in out, out[-400:])
        check("and telling the reader to clone instead",
              "--recurse-submodules" in out, out[-400:])


def test_a_git_checkout_with_no_submodule_registered_fails_clearly():
    # `git init` with no .gitmodules: the command succeeds and populates
    # nothing, which must not be reported as success.
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_repo(tmp, with_git=True, with_entry=False)
        r = _validate_from(root)
        out = r.stdout + r.stderr
        check("an empty-but-registered submodule is not called success",
              r.returncode != 0, out[-300:])
        check("and the message says to fix it by hand",
              "submodule update --init" in out, out[-400:])


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{PASSED}/{PASSED + FAILED} passed")
    sys.exit(1 if FAILED else 0)
