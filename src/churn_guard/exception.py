"""Project-specific exception types.

Why bother, when Python already has ``ValueError`` and friends?

Because a bare ``ValueError: could not convert string to float`` tells you *what*
broke but not *where in the system*. With a small hierarchy, the exception type
itself names the failing stage:

    >>> raise DataValidationError("Column 'TotalCharges' has 11 blank strings")
    DataValidationError: Column 'TotalCharges' has 11 blank strings

Callers can then react selectively — retry on ingestion failures, halt on
validation failures — because they can catch a specific type. Catching plain
``Exception`` forces you to treat every failure identically.

Chaining: always ``raise ... from err``
---------------------------------------
When re-raising inside an ``except`` block, use ``from``::

    try:
        df = pd.read_csv(path)
    except FileNotFoundError as err:
        raise DataIngestionError(f"Raw data missing at {path}") from err

``from err`` keeps the original error attached, so the traceback shows both the
root cause and your friendlier message. Without it Python prints a confusing
"During handling of the above exception, another exception occurred".

A note on a common tutorial pattern
-----------------------------------
Several popular ML courses teach a custom exception that calls ``sys.exc_info()``
to splice the filename and line number into the message by hand. That is
unnecessary — Python 3 tracebacks already carry file and line — and it tends to
*replace* the real traceback with a string, losing information. A plain subclass
plus ``raise ... from err`` is simpler and strictly more useful.
"""

from __future__ import annotations


class ChurnGuardError(Exception):
    """Base class for every error raised by this project.

    Catching this catches all of ours while letting genuine bugs
    (``TypeError``, ``AttributeError``) propagate as they should.
    """


class ConfigError(ChurnGuardError):
    """Configuration is missing, malformed, or missing a required key."""


class DataIngestionError(ChurnGuardError):
    """Raw data could not be downloaded, found, or read."""


class DataValidationError(ChurnGuardError):
    """Data loaded, but failed a schema or quality check.

    Raised for wrong row/column counts, unexpected column names, an
    out-of-range target, or missingness beyond an accepted threshold. This is
    the guard that stops a silently corrupted upstream file from producing a
    confidently wrong model.
    """


class ModelTrainingError(ChurnGuardError):
    """Training or hyperparameter search failed."""


class ModelNotFoundError(ChurnGuardError):
    """A saved model artifact was expected on disk but is not there.

    Typically means the API was started before the training pipeline had run.
    """


class PredictionError(ChurnGuardError):
    """Inference failed — malformed input, or features the model never saw."""
