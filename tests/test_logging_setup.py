"""The logging setup, guarded against the one way it fails silently.

`configure()` attaches its handler to the ``germinal`` logger, so only loggers
under that name reach it. A module that names its logger ``__name__`` is fine
-- it is ``germinal.something``. A *script* that does is not: executed as a
script, ``__name__`` is ``"__main__"``, which sits outside that tree, and every
record it logs is dropped with no error. That is exactly what happened to
run_germinal_multi.py's 33 log calls, and nothing in the suite caught it,
because logging that goes nowhere raises nothing.
"""

import logging

from germinal.utils import log as glog


def test_configure_attaches_exactly_one_handler():
    glog.configure(level="INFO")
    glog.configure(level="INFO")
    handlers = logging.getLogger(glog.ROOT).handlers
    assert len(handlers) == 1, (
        f"configure() left {len(handlers)} handlers; calling it twice must "
        f"replace rather than append, or every line prints twice"
    )


def test_a_module_logger_reaches_the_handler():
    glog.configure(level="INFO")
    child = logging.getLogger("germinal.filters.af3")
    assert child.isEnabledFor(logging.INFO)
    # The handler lives on the parent, found by propagation.
    assert logging.getLogger(glog.ROOT).handlers


def test_level_gates_progress_but_not_warnings():
    glog.configure(level="WARNING")
    child = logging.getLogger("germinal.filters.af3")
    assert not child.isEnabledFor(logging.INFO), "WARNING must suppress progress"
    assert child.isEnabledFor(logging.WARNING)
    glog.configure(level="INFO")


def test_configure_survives_hydra_disabling_existing_loggers():
    """configs/config.yaml selects Hydra's `job_logging: disabled` profile.

    That calls dictConfig with disable_existing_loggers, and it runs after every
    module-level get_logger() call, so it marks the whole germinal tree
    .disabled = True. A disabled logger emits nothing however good its handler,
    and nothing raises -- the pipeline just runs silently. configure() must undo
    it.
    """
    import logging.config

    mod = logging.getLogger("germinal.filters.af3")
    logging.config.dictConfig({"version": 1, "disable_existing_loggers": True})
    assert mod.disabled, "precondition: dictConfig should have disabled it"
    glog.configure(level="INFO")
    assert not mod.disabled, (
        "configure() left a germinal logger disabled, so every record it takes "
        "is dropped with no error -- this is how the entire pipeline went "
        "silent while 256 tests passed"
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
