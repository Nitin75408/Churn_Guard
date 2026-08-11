"""Shared pytest fixtures.

Fixtures are session-scoped because loading the splits and fitting the pipeline
costs a second or two each. Per-test setup would make the suite slow enough that
people stop running it, and a test suite nobody runs is worse than none — it
gives false confidence.
"""

from __future__ import annotations

import pytest

from churn_guard.config import load_config
from churn_guard.data.split import load_split
from churn_guard.features.build import build_feature_pipeline, load_xy


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def train_xy(cfg):
    return load_xy("train", cfg)


@pytest.fixture(scope="session")
def val_xy(cfg):
    return load_xy("val", cfg)


@pytest.fixture(scope="session")
def raw_splits(cfg):
    """All three splits as raw dataframes, for integrity checks."""
    return {name: load_split(name, cfg) for name in ("train", "val", "test")}


@pytest.fixture(scope="session")
def fitted_pipeline(train_xy):
    """Feature pipeline fitted on the training split only.

    Fitted here, once, precisely so tests cannot accidentally refit it on
    validation data and mask the leakage they are meant to detect.
    """
    X_train, y_train = train_xy
    pipeline = build_feature_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


@pytest.fixture(scope="session")
def model_available(cfg) -> bool:
    return (cfg.paths.models / "model.joblib").is_file()
