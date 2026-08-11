"""Data contract and split integrity.

The tests here guard the two properties that everything downstream assumes: the
raw data matches the schema we validated against, and the splits are disjoint.
The second one is the important one — it is a regression test for the leakage
that inflated PR-AUC from 0.59 to 0.95 when we demonstrated it deliberately.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from churn_guard.config import load_config
from churn_guard.data.ingest import validate_raw_data
from churn_guard.data.split import MANIFEST_FILENAME, _fingerprint, split_data
from churn_guard.exception import DataValidationError


class TestSplitIntegrity:
    """Regression tests for data leakage between splits."""

    def test_no_customer_appears_in_two_splits(self, raw_splits, cfg):
        """The single most important test in the suite.

        If this fails, every metric the project reports is meaningless: the model
        would be scored on customers it trained on.
        """
        id_column = cfg.data.id_column
        names = list(raw_splits)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                shared = set(raw_splits[first][id_column]) & set(
                    raw_splits[second][id_column]
                )
                assert not shared, (
                    f"{len(shared)} customers appear in both {first} and {second}"
                )

    def test_splits_reconstruct_the_full_dataset(self, raw_splits):
        total = sum(len(part) for part in raw_splits.values())
        assert total == 7043

    def test_no_duplicate_ids_within_a_split(self, raw_splits, cfg):
        for name, part in raw_splits.items():
            duplicates = part[cfg.data.id_column].duplicated().sum()
            assert duplicates == 0, f"{name} has {duplicates} duplicate IDs"

    def test_stratification_preserved_churn_rate(self, raw_splits, cfg):
        """Every split should mirror the overall 26.5% churn rate.

        A split whose churn rate drifts from the population makes its metrics a
        measurement of the split rather than of the model.
        """
        rates = {
            name: (part[cfg.data.target_column] == cfg.data.positive_label).mean()
            for name, part in raw_splits.items()
        }
        spread = max(rates.values()) - min(rates.values())
        assert spread < 0.02, f"churn rate varies too much across splits: {rates}"

    def test_test_set_matches_its_recorded_fingerprint(self, raw_splits, cfg):
        """The seal from Day 1 still holds.

        Guards against a silent reshuffle — a changed seed or shifted upstream
        data — which would invalidate the reported test score without any
        obvious symptom.
        """
        manifest_path = cfg.paths.data_interim / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest["splits"]["test"]["id_fingerprint"]
        actual = _fingerprint(raw_splits["test"][cfg.data.id_column])
        assert actual == expected

    def test_split_is_deterministic(self, cfg):
        """Re-splitting with the same seed must reproduce the same partition."""
        frame = pd.concat(
            [pd.read_csv(cfg.paths.data_interim / f"{n}.csv") for n in ("train", "val", "test")],
            ignore_index=True,
        )
        first = split_data(frame, cfg)
        second = split_data(frame, cfg)
        for name in ("train", "val", "test"):
            assert list(first[name][cfg.data.id_column]) == list(
                second[name][cfg.data.id_column]
            )


class TestDataContract:
    """The validation that runs on every ingest."""

    def _valid_frame(self, cfg) -> pd.DataFrame:
        return pd.concat(
            [pd.read_csv(cfg.paths.data_interim / f"{n}.csv") for n in ("train", "val", "test")],
            ignore_index=True,
        )

    def test_accepts_the_real_dataset(self, cfg):
        validate_raw_data(self._valid_frame(cfg), cfg)

    def test_rejects_wrong_row_count(self, cfg):
        with pytest.raises(DataValidationError, match="Shape mismatch"):
            validate_raw_data(self._valid_frame(cfg).head(100), cfg)

    def test_rejects_duplicate_customer_ids(self, cfg):
        frame = self._valid_frame(cfg)
        frame.loc[1, cfg.data.id_column] = frame.loc[0, cfg.data.id_column]
        with pytest.raises(DataValidationError, match="duplicate"):
            validate_raw_data(frame, cfg)

    def test_rejects_unexpected_target_values(self, cfg):
        frame = self._valid_frame(cfg)
        frame.loc[0, cfg.data.target_column] = "Maybe"
        with pytest.raises(DataValidationError, match="Unexpected values"):
            validate_raw_data(frame, cfg)


class TestConfig:
    def test_paths_are_absolute(self, cfg):
        """Relative paths break as soon as code runs from another directory."""
        assert cfg.paths.data_raw.is_absolute()
        assert cfg.paths.models.is_absolute()

    def test_mistyped_key_names_the_valid_options(self, cfg):
        from churn_guard.exception import ConfigError

        with pytest.raises(ConfigError, match="Available keys"):
            _ = cfg.split.tst_size

    def test_missing_config_file_is_a_clear_error(self):
        from churn_guard.exception import ConfigError

        with pytest.raises(ConfigError, match="not found"):
            load_config("/nonexistent/config.yaml")
