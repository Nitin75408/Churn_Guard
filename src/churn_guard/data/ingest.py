"""Download, load and validate the raw Telco churn dataset.

Run it directly::

    uv run python -m churn_guard.data.ingest

Two principles drive this module.

**Acquisition is code, not a browser download.** If the dataset arrives by a
manual click, nobody else — including you in three months — can reproduce your
starting point. A script means the raw data is regenerable from the repo alone,
which is why ``data/`` is safe to leave out of git.

**Validate on arrival — the data contract.** The file is fetched from a URL we
do not control. If it changes upstream, we want a loud failure *now*, not a
model that trains happily on shifted data and quietly makes worse decisions for
months. Checking shape, columns and target values costs milliseconds and has
saved countless production models.

In a larger system these checks would be a schema library (``pandera``) or a
data-quality suite (Great Expectations) wired into an orchestrator. The logic is
the same; only the packaging differs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from churn_guard.config import ConfigSection, load_config
from churn_guard.exception import DataIngestionError, DataValidationError
from churn_guard.logger import get_logger

logger = get_logger(__name__)

_DOWNLOAD_TIMEOUT_SECONDS = 60


def download_raw_data(cfg: ConfigSection, force: bool = False) -> Path:
    """Fetch the raw CSV into ``data/raw/`` if it is not already there.

    Args:
        cfg: Project configuration.
        force: Re-download even when a local copy exists.

    Returns:
        Path to the downloaded CSV.

    Raises:
        DataIngestionError: If the download fails or returns an empty body.
    """
    destination: Path = cfg.paths.data_raw / cfg.data.raw_filename

    if destination.exists() and not force:
        size_kb = destination.stat().st_size / 1024
        logger.info("Raw data already present (%.0f KB), skipping download.", size_kb)
        return destination

    url = cfg.data.source_url
    logger.info("Downloading raw data from %s", url)

    try:
        response = requests.get(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as err:
        raise DataIngestionError(
            f"Could not download the dataset from {url}. Check your internet "
            f"connection, or download it manually from Kaggle "
            f"(blastchar/telco-customer-churn) and save it as {destination}."
        ) from err

    if not response.content:
        raise DataIngestionError(f"{url} returned an empty response body.")

    # Write only after a fully successful response, so an interrupted download
    # can never leave a half-written file that looks valid to the next run.
    destination.write_bytes(response.content)
    logger.info(
        "Saved %.0f KB to %s", len(response.content) / 1024, destination.name
    )
    return destination


def validate_raw_data(df: pd.DataFrame, cfg: ConfigSection) -> None:
    """Assert the dataframe matches the expected contract.

    Hard failures raise; suspicious-but-survivable findings are logged as
    warnings so they surface in EDA rather than halting the pipeline.

    Raises:
        DataValidationError: On wrong shape, missing key columns, duplicate
            IDs, or unexpected target values.
    """
    logger.info("Validating raw data against the expected contract...")

    expected_rows = int(cfg.data.expected_rows)
    expected_cols = int(cfg.data.expected_columns)
    if df.shape != (expected_rows, expected_cols):
        raise DataValidationError(
            f"Shape mismatch: expected ({expected_rows}, {expected_cols}), "
            f"got {df.shape}. The upstream file may have changed — inspect it "
            f"before trusting any model trained on it."
        )

    id_col = cfg.data.id_column
    target_col = cfg.data.target_column

    missing_columns = [c for c in (id_col, target_col) if c not in df.columns]
    if missing_columns:
        raise DataValidationError(
            f"Required column(s) absent: {missing_columns}. "
            f"Found: {list(df.columns)}"
        )

    duplicate_ids = int(df[id_col].duplicated().sum())
    if duplicate_ids:
        raise DataValidationError(
            f"{duplicate_ids} duplicate values in '{id_col}'. Each row must be "
            f"one distinct customer, otherwise the same person can land in both "
            f"train and test and inflate your scores."
        )

    observed_targets = set(df[target_col].dropna().unique())
    allowed_targets = {"Yes", "No"}
    if not observed_targets.issubset(allowed_targets):
        raise DataValidationError(
            f"Unexpected values in target '{target_col}': "
            f"{observed_targets - allowed_targets}. Expected only {allowed_targets}."
        )

    # --- Soft checks: logged, not fatal -----------------------------------
    null_counts = df.isna().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if columns_with_nulls.empty:
        logger.info("No true NaN values found.")
    else:
        logger.warning("Columns containing NaN: %s", columns_with_nulls.to_dict())

    # Blank strings masquerading as numbers are the classic trap in this
    # dataset: TotalCharges is stored as text, and customers with tenure 0
    # have " " rather than 0. pandas reads the column as object, isna() sees
    # nothing wrong, and any naive .astype(float) explodes later.
    # Selected by exclusion rather than select_dtypes(include="object"): pandas 3
    # splits text into both 'object' and 'str' dtypes, and asking for one silently
    # misses the other.
    non_numeric_columns = df.columns.difference(
        df.select_dtypes(include="number").columns
    )
    for column in non_numeric_columns:
        blank_count = int((df[column].astype(str).str.strip() == "").sum())
        if blank_count:
            logger.warning(
                "Column '%s' holds %d blank strings — these are missing values "
                "in disguise and isna() will not see them.",
                column,
                blank_count,
            )

    churn_rate = (df[target_col] == cfg.data.positive_label).mean()
    logger.info(
        "Target balance: %.2f%% churn, %.2f%% retained.",
        churn_rate * 100,
        (1 - churn_rate) * 100,
    )

    logger.info("Validation passed: %d rows x %d columns.", *df.shape)


def load_raw_data(cfg: ConfigSection | None = None, validate: bool = True) -> pd.DataFrame:
    """Download if needed, read the CSV, and validate it.

    Args:
        cfg: Project configuration; loaded from disk when omitted.
        validate: Run the data contract checks.

    Returns:
        The raw dataframe, entirely unmodified — no cleaning happens here.
        ``data/raw`` is read-only by convention, and cleaning belongs in the
        pipeline where it can be fitted on training data alone.
    """
    cfg = cfg or load_config()
    path = download_raw_data(cfg)

    try:
        df = pd.read_csv(path)
    except Exception as err:
        raise DataIngestionError(f"Failed to read CSV at {path}.") from err

    logger.info("Loaded raw data: %d rows x %d columns.", *df.shape)

    if validate:
        validate_raw_data(df, cfg)

    return df


def main() -> None:
    """CLI entry point: fetch and validate the raw dataset."""
    cfg = load_config()
    df = load_raw_data(cfg)
    logger.info("Ingestion complete. Columns: %s", list(df.columns))


if __name__ == "__main__":
    main()
