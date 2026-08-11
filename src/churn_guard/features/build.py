"""Feature engineering and preprocessing, as scikit-learn transformers.

Every decision recorded in ``docs/eda-findings.md`` is implemented here, and
nowhere else. Wrapping them as transformers rather than writing them as loose
dataframe code buys two things that matter more than convenience:

**Leakage becomes structurally impossible.** Stateful steps (scaling, imputing,
encoding) learn their statistics inside ``Pipeline.fit``, which only ever sees
the training split. There is no code path where a test-set statistic can reach
the model, so it cannot happen by accident during a refactor.

**Train/serve skew becomes impossible.** The fitted pipeline is one object,
saved to one file. The API on Day 3 loads that object and calls ``predict`` on
raw customer records. It does not reimplement a single transformation, so the
serving path cannot drift from the training path.

Stateless vs stateful
---------------------
:class:`DomainFeatureBuilder` is **stateless** — it works a row at a time and
learns nothing, so its ``fit`` is a no-op. It still belongs in the pipeline so
that serving applies it identically.

``StandardScaler``, ``SimpleImputer`` and ``OneHotEncoder`` are **stateful** —
they learn column statistics. Those are the steps leakage hides in, and the
reason the pipeline exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_guard.config import ConfigSection, load_config
from churn_guard.data.split import load_split
from churn_guard.logger import get_logger

logger = get_logger(__name__)

# The six optional add-on services (EDA §6, §7).
ADDON_SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Values meaning "this does not apply to you", not a customer choice (EDA §7).
NOT_APPLICABLE_VALUES = ["No internet service", "No phone service"]

# Bucket edges from the tenure analysis (EDA §3). The final bin is open-ended:
# the training data tops out at 72 months, but a real customer could exceed it
# and must not produce NaN at serving time.
TENURE_BINS = [-np.inf, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49mo+"]

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    # engineered
    "num_addon_services",
    "avg_charges_per_month",
    "has_internet",
    "is_automatic_payment",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    # engineered
    "tenure_bucket",
]


class DomainFeatureBuilder(TransformerMixin, BaseEstimator):
    # Mixin before BaseEstimator — scikit-learn resolves estimator type through
    # __sklearn_tags__ along the MRO, and the reverse order breaks that.
    """Clean the known data defects and add the engineered features.

    Stateless: ``fit`` learns nothing and exists only to satisfy the scikit-learn
    interface. Everything here is computed from a single row, so it can be
    applied to train, test, or one live customer with identical results.

    Implements EDA decisions 1-7:

    1. drop ``customerID`` — an identifier lets the model memorise individuals
    2. ``TotalCharges`` to numeric, blanks filled with 0
    3. collapse "No internet/phone service" to "No"
    4. ``tenure_bucket`` — the tenure/churn relationship is not linear
    5. ``num_addon_services`` and ``has_internet`` — monotonic once paired
    6. ``is_automatic_payment`` — manual vs automatic is the real signal
    7. ``avg_charges_per_month`` — less collinear than raw ``TotalCharges``
    """

    def __init__(self, id_column: str = "customerID") -> None:
        self.id_column = id_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DomainFeatureBuilder:
        """No-op. Nothing is learned from the data.

        Feature names are recorded so scikit-learn can validate column order at
        transform time and fail loudly on a mismatch rather than silently
        misaligning columns.
        """
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # (1) Identifier carries no signal and invites memorisation.
        df = df.drop(columns=[self.id_column], errors="ignore")

        # (2) TotalCharges arrives as text because 11 rows hold a blank string.
        # Blank exactly when tenure == 0, i.e. never billed, so the true value
        # is 0. Median imputation would claim these new customers spent ~$1,400.
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

        # (3) "No internet service" duplicates information already in
        # InternetService, six times over. Collapsing removes six redundant
        # one-hot columns without losing anything.
        service_like = ADDON_SERVICE_COLUMNS + ["MultipleLines"]
        for column in service_like:
            if column in df.columns:
                df[column] = df[column].replace(NOT_APPLICABLE_VALUES, "No")

        # (4) Churn falls steeply then flattens; a linear term cannot express
        # that shape. Buckets let linear models see the curvature.
        df["tenure_bucket"] = pd.cut(
            df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
        ).astype(str)

        # (5) Each add-on raises switching cost. Ambiguous on its own because
        # customers without internet cannot have any, so has_internet ships
        # alongside it — see the Simpson's paradox note in the findings.
        df["num_addon_services"] = (
            df[ADDON_SERVICE_COLUMNS].eq("Yes").sum(axis=1).astype(int)
        )
        df["has_internet"] = (df["InternetService"] != "No").astype(int)

        # (6) The two automatic methods churn at ~15%, the two manual ones at
        # 19-46%. Removing a recurring decision point is what matters.
        df["is_automatic_payment"] = (
            df["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)
        )

        # (7) TotalCharges correlates 0.9996 with tenure x MonthlyCharges. The
        # ratio is closer to independent. clip(lower=1) guards tenure == 0.
        df["avg_charges_per_month"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

        # Return columns in exactly the order get_feature_names_out declares.
        # With set_output(transform="pandas"), scikit-learn relabels output
        # columns *by position* from that method — so if the two disagree, every
        # column is silently renamed to its neighbour and string data lands in
        # the numeric branch. Selecting explicitly also drops anything not
        # declared, so a stray upstream column cannot slip through.
        return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(NUMERIC_FEATURES + CATEGORICAL_FEATURES, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Route numeric and categorical columns to their own treatments.

    A ``ColumnTransformer`` applies different pipelines to different columns —
    scaling makes no sense for a category, one-hot encoding makes none for a
    price. Both branches are stateful, so both are fitted on training data only.
    """
    numeric_pipeline = Pipeline(
        steps=[
            # Defensive: the engineered columns cannot be null by construction,
            # but a malformed API payload could still arrive with a gap.
            ("impute", SimpleImputer(strategy="median")),
            # Required by regularised linear models: without it, penalties fall
            # unevenly because TotalCharges spans thousands while SeniorCitizen
            # is 0/1. Tree models ignore scale, and are unharmed by it.
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    # Critical for serving: if a live request carries a category
                    # never seen in training, encode it as all-zeros instead of
                    # crashing the API.
                    handle_unknown="ignore",
                    # Drop the first level of binary columns only. Removes the
                    # redundant column for Yes/No pairs while keeping every
                    # level of multi-class columns visible for interpretation.
                    drop="if_binary",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        # Anything not listed is dropped, so a stray column added upstream can
        # never leak into the model unnoticed.
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline() -> Pipeline:
    """The full preprocessing pipeline, model excluded.

    Returns:
        A pipeline that takes a raw customer dataframe and returns a numeric
        feature matrix, ready for any estimator to be appended.
    """
    pipeline = Pipeline(
        steps=[
            ("domain", DomainFeatureBuilder()),
            ("prepare", build_preprocessor()),
        ]
    )
    # Pandas output keeps feature names attached through every step, which is
    # what makes SHAP plots and coefficient tables readable on Day 3.
    return pipeline.set_output(transform="pandas")


def load_xy(
    split: str, cfg: ConfigSection | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Load one split as ``(X, y)`` with the target encoded 0/1.

    Args:
        split: ``train``, ``val`` or ``test``.

    Returns:
        Raw features (untransformed — the pipeline handles that) and the binary
        target, where 1 means the customer churned.
    """
    cfg = cfg or load_config()
    df = load_split(split, cfg)
    target = cfg.data.target_column
    y = (df[target] == cfg.data.positive_label).astype(int)
    X = df.drop(columns=[target])
    return X, y
