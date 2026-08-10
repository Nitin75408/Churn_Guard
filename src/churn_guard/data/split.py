"""Split the raw data into train / validation / test — before anything else.

Run it::

    uv run python -m churn_guard.data.split

This runs immediately after ingestion and before EDA, cleaning, imputation,
scaling or balancing. That ordering is the single most important rule in the
project: any statistic learned from rows that later appear in the test set
makes the test score a measurement of memorisation rather than of skill, and
no amount of later care can repair it.

Three splits, three distinct jobs
---------------------------------
=========== ======= ==================================================
Split       Share   Used for
=========== ======= ==================================================
train       70%     fitting models and preprocessing statistics
validation  15%     comparing models, tuning, choosing the threshold
test        15%     one final honest measurement, once, at the end
=========== ======= ==================================================

Why validation and test are separate: every time you look at a set and change
something in response, you leak a little information into your choices. After
fifty tuning decisions guided by the validation set, that set no longer gives an
unbiased estimate — you have partly fitted *yourself* to it. The test set stays
sealed so one genuinely untouched measurement remains.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_guard.config import ConfigSection, load_config
from churn_guard.data.ingest import load_raw_data
from churn_guard.exception import DataValidationError
from churn_guard.logger import get_logger

logger = get_logger(__name__)

SPLIT_NAMES = ("train", "val", "test")
MANIFEST_FILENAME = "split_manifest.json"


def _fingerprint(ids: pd.Series) -> str:
    """Short stable hash of a split's customer IDs.

    Recorded in the manifest so we can prove later that the sealed test set is
    byte-for-byte the same one created here. If a rerun changes the seed or the
    upstream data, the fingerprint changes and every result becomes suspect.
    """
    joined = "|".join(sorted(ids.astype(str)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def split_data(
    df: pd.DataFrame, cfg: ConfigSection
) -> dict[str, pd.DataFrame]:
    """Partition ``df`` into train/val/test with stratification.

    Done in two stages because scikit-learn splits in two at a time:

    1. hold out ``test`` from everything
    2. split the remainder into ``train`` and ``val``

    The second stage uses an adjusted fraction — to end up with 15% of the
    *original* data as validation while drawing from the remaining 85%, we ask
    for ``0.15 / 0.85`` of what is left.

    Args:
        df: Raw dataframe, uncleaned.
        cfg: Project configuration.

    Returns:
        Mapping of split name to dataframe.

    Raises:
        DataValidationError: If any customer ends up in more than one split.
    """
    target = cfg.data.target_column
    seed = int(cfg.project.random_seed)
    test_size = float(cfg.split.test_size)
    val_size = float(cfg.split.val_size)

    # Stratify keeps the churn rate identical across splits. Without it, random
    # chance can hand the test set an unrepresentative churn rate, and your
    # final number becomes noise rather than measurement.
    stratify_on = df[target] if cfg.split.stratify else None

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_on,
        shuffle=True,
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    stratify_on = train_val_df[target] if cfg.split.stratify else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=stratify_on,
        shuffle=True,
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}
    _verify_no_overlap(splits, cfg)
    return splits


def _verify_no_overlap(splits: dict[str, pd.DataFrame], cfg: ConfigSection) -> None:
    """Assert that no customer appears in more than one split.

    ``train_test_split`` will not duplicate rows, so this should never fire —
    which is exactly why it is worth asserting. It is the guard that catches a
    future refactor (adding oversampling in the wrong place, say) that silently
    reintroduces the leakage we just eliminated.
    """
    id_col = cfg.data.id_column
    names = list(splits)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = set(splits[first][id_col]) & set(splits[second][id_col])
            if shared:
                raise DataValidationError(
                    f"{len(shared)} customer(s) appear in both '{first}' and "
                    f"'{second}'. This is data leakage — the test score would "
                    f"measure memorisation, not skill."
                )
    logger.info("Overlap check passed: all splits are disjoint.")


def save_splits(
    splits: dict[str, pd.DataFrame], cfg: ConfigSection
) -> dict[str, Path]:
    """Write each split to ``data/interim/`` and record a manifest.

    They land in ``interim`` rather than ``processed`` because they are still
    completely raw — split, but not cleaned. Cleaning happens inside the
    pipeline, fitted on train alone.
    """
    output_dir: Path = cfg.paths.data_interim
    target = cfg.data.target_column
    positive = cfg.data.positive_label
    id_col = cfg.data.id_column

    total = sum(len(part) for part in splits.values())
    paths: dict[str, Path] = {}
    manifest: dict[str, object] = {
        "random_seed": int(cfg.project.random_seed),
        "stratified": bool(cfg.split.stratify),
        "total_rows": total,
        "splits": {},
    }

    for name in SPLIT_NAMES:
        part = splits[name]
        destination = output_dir / f"{name}.csv"
        part.to_csv(destination, index=False)
        paths[name] = destination

        churn_rate = float((part[target] == positive).mean())
        manifest["splits"][name] = {
            "rows": int(len(part)),
            "share": round(len(part) / total, 4),
            "churn_rate": round(churn_rate, 4),
            "id_fingerprint": _fingerprint(part[id_col]),
        }

        logger.info(
            "%-5s -> %5d rows (%4.1f%%)  churn %.2f%%  [%s]",
            name,
            len(part),
            100 * len(part) / total,
            100 * churn_rate,
            destination.name,
        )

    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Split manifest written to %s", manifest_path.name)

    return paths


def load_split(name: str, cfg: ConfigSection | None = None) -> pd.DataFrame:
    """Read one split back from disk.

    Args:
        name: One of ``train``, ``val``, ``test``.

    Raises:
        DataValidationError: If ``name`` is not a known split, or the file is
            absent because the split step has not been run.
    """
    cfg = cfg or load_config()
    if name not in SPLIT_NAMES:
        raise DataValidationError(
            f"Unknown split '{name}'. Expected one of {SPLIT_NAMES}."
        )

    path: Path = cfg.paths.data_interim / f"{name}.csv"
    if not path.is_file():
        raise DataValidationError(
            f"{path} not found. Run 'uv run python -m churn_guard.data.split' first."
        )
    return pd.read_csv(path)


def main() -> None:
    """CLI entry point: ingest, split, and persist."""
    cfg = load_config()
    df = load_raw_data(cfg)

    logger.info("Splitting BEFORE any cleaning or EDA — leakage prevention.")
    splits = split_data(df, cfg)
    save_splits(splits, cfg)

    logger.warning(
        "test.csv is now SEALED. Do not read it during EDA, feature "
        "engineering, model selection or tuning — only for the final "
        "evaluation on Day 3."
    )


if __name__ == "__main__":
    main()
