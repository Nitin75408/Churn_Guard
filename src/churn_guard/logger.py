"""Central logging setup.

Usage — one line at the top of every module::

    from churn_guard.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Loaded %d rows", len(df))

Passing ``__name__`` means each message is stamped with the module it came from,
so a log line reads ``churn_guard.data.ingest`` and you know instantly where to
look.

Two destinations, deliberately different
----------------------------------------
* **Console** — ``INFO`` and above. What you watch while a job runs. Kept quiet
  so real progress is not buried under debug noise.
* **File** (``logs/churn_guard.log``) — ``DEBUG`` and above, everything. What you
  read *after* something failed unattended at 3am.

The file rotates at 5 MB and keeps 3 old copies, so a long-running service can
never fill the disk with logs.

Lazy formatting
---------------
Write ``logger.info("Loaded %d rows", n)`` rather than
``logger.info(f"Loaded {n} rows")``. With the ``%s`` form the string is only
built if that level is actually enabled — so ``DEBUG`` calls cost nothing when
you are running at ``INFO``.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from churn_guard.config import load_config

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Guard so repeated imports don't attach duplicate handlers, which is what
# causes every message to print two or three times.
_configured = False


def _configure_root_logger() -> None:
    """Attach console and rotating-file handlers to the ``churn_guard`` logger.

    Runs once per process. Configures the package logger rather than Python's
    root logger, so noisy third-party libraries keep their own settings and we
    don't accidentally swallow or amplify their output.
    """
    global _configured
    if _configured:
        return

    cfg = load_config()
    log_dir: Path = cfg.paths.logs
    log_file = log_dir / cfg.logging.filename

    package_logger = logging.getLogger("churn_guard")
    # Set the logger itself to the most permissive level in use; each handler
    # then filters down. If the logger sat at INFO, DEBUG records would be
    # dropped before any handler ever saw them.
    package_logger.setLevel(logging.DEBUG)
    # Don't hand records to the root logger as well — that is the other common
    # cause of duplicated output.
    package_logger.propagate = False

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(getattr(logging, str(cfg.logging.level).upper(), logging.INFO))
    console.setFormatter(formatter)
    package_logger.addHandler(console)

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=int(cfg.logging.max_bytes),
        backupCount=int(cfg.logging.backup_count),
        encoding="utf-8",
    )
    file_handler.setLevel(
        getattr(logging, str(cfg.logging.file_level).upper(), logging.DEBUG)
    )
    file_handler.setFormatter(formatter)
    package_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for ``name``, configuring handlers on first use.

    Args:
        name: Almost always ``__name__``.

    Returns:
        A logger writing to both console and the rotating log file.
    """
    _configure_root_logger()

    # Names outside our package would not inherit our handlers, so normalise
    # them onto the package logger.
    if not name.startswith("churn_guard"):
        name = f"churn_guard.{name}"
    return logging.getLogger(name)
