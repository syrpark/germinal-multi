"""One logger for the pipeline, so output can be turned down or up.

Every module does::

    from germinal.utils.log import get_logger
    log = get_logger(__name__)

and the entry point calls :func:`configure` once. The default format is the
bare message, so a SLURM ``.out`` file reads exactly as it did when this was
``print()``; pass ``timestamps=True`` to prefix each line with the time and
level, which is what you want when diagnosing a long run.

Levels in use: ``info`` for progress, ``warning`` for a recoverable problem
worth seeing, ``error`` for one that ends a design. Anything that ends the run
raises rather than logging.
"""

import logging
import sys

#: Parent of every module logger, so one call configures them all.
ROOT = "germinal"

_PLAIN = "%(message)s"
_DETAILED = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a module, as ``get_logger(__name__)``."""
    return logging.getLogger(name)


def configure(level: str = "INFO", timestamps: bool = False) -> None:
    """Point the pipeline's logging at stdout. Call once, at startup.

    Args:
        level: any logging level name -- "DEBUG", "INFO", "WARNING", ...
        timestamps: prefix each line with time, level and module.

    Replaces its own handler on a repeat call rather than adding a second one,
    so calling it twice does not double every line.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_DETAILED if timestamps else _PLAIN))

    root = logging.getLogger(ROOT)
    root.handlers[:] = [handler]
    root.setLevel(level.upper())

    # Undo disable_existing_loggers. configs/config.yaml selects Hydra's
    # `job_logging: disabled` profile, which calls dictConfig with
    # disable_existing_loggers set -- and that runs AFTER every module's
    # module-level get_logger() call, so it marks the whole germinal tree
    # .disabled = True. A disabled logger emits nothing no matter what handler
    # it can reach, so without this the pipeline runs silently and raises
    # nothing to say so.
    root.disabled = False
    for name, logger in logging.root.manager.loggerDict.items():
        if (name == ROOT or name.startswith(ROOT + ".")) and isinstance(
            logger, logging.Logger
        ):
            logger.disabled = False
    # Do not hand records to the root logger as well: libraries configure that,
    # and a basicConfig() anywhere would otherwise duplicate every line.
    root.propagate = False
