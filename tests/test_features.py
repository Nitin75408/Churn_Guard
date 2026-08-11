"""Feature engineering and preprocessing.

Several of these are regression tests for bugs that actually occurred while
building this project, which is the most useful kind of test to write:

* ``test_transform_output_matches_declared_feature_names`` — when the two
  disagreed, ``set_output(transform="pandas")`` relabelled every column by
  position and routed string data into the numeric branch.
* ``test_scaler_is_fitted_on_train_only`` — the leakage check, expressed as an
  assertion rather than as a habit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_guard.features.build import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    DomainFeatureBuilder,
)


@pytest.fixture
def sample_customer() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customerID": "0000-TEST",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 0,
                "PhoneService": "Yes",
                "MultipleLines": "No phone service",
                "InternetService": "No",
                "OnlineSecurity": "No internet service",
                "OnlineBackup": "No internet service",
                "DeviceProtection": "No internet service",
                "TechSupport": "No internet service",
                "StreamingTV": "No internet service",
                "StreamingMovies": "No internet service",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 25.0,
                "TotalCharges": " ",
            }
        ]
    )


class TestDomainFeatureBuilder:
    def test_transform_output_matches_declared_feature_names(self, sample_customer):
        """Column order must match get_feature_names_out exactly.

        scikit-learn relabels pandas output positionally from that method. When
        the two disagree it does not raise — it silently renames every column,
        and the failure surfaces much later as a type error deep in the
        ColumnTransformer.
        """
        builder = DomainFeatureBuilder()
        out = builder.fit_transform(sample_customer)
        assert list(out.columns) == list(builder.get_feature_names_out())
        assert list(out.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES

    def test_identifier_is_dropped(self, sample_customer):
        out = DomainFeatureBuilder().fit_transform(sample_customer)
        assert "customerID" not in out.columns

    def test_blank_total_charges_becomes_zero_not_median(self, sample_customer):
        """Blank TotalCharges means tenure 0 — never billed, so the value is 0.

        Median imputation would claim a brand-new customer had already spent
        about $1,400, which inverts the strongest signal in the dataset.
        """
        out = DomainFeatureBuilder().fit_transform(sample_customer)
        assert out["TotalCharges"].iloc[0] == 0.0

    def test_zero_tenure_does_not_divide_by_zero(self, sample_customer):
        out = DomainFeatureBuilder().fit_transform(sample_customer)
        value = out["avg_charges_per_month"].iloc[0]
        assert np.isfinite(value)

    def test_not_applicable_levels_collapse_to_no(self, sample_customer):
        """"No internet service" duplicates information already in
        InternetService, six times over."""
        out = DomainFeatureBuilder().fit_transform(sample_customer)
        for column in ("OnlineSecurity", "TechSupport", "StreamingTV", "MultipleLines"):
            assert out[column].iloc[0] == "No"

    def test_tenure_bucket_handles_values_beyond_training_range(self, sample_customer):
        """Training tops out at 72 months; a real customer can exceed it.

        Closed bins would produce NaN for such a customer and fail at serving
        time — the exact scenario an open-ended final bin exists to prevent.
        """
        frame = sample_customer.copy()
        frame.loc[0, "tenure"] = 500
        out = DomainFeatureBuilder().fit_transform(frame)
        assert out["tenure_bucket"].iloc[0] == "49mo+"
        assert out["tenure_bucket"].notna().all()

    def test_addon_count_and_internet_flag(self, sample_customer):
        frame = sample_customer.copy()
        frame.loc[0, ["InternetService", "OnlineSecurity", "TechSupport"]] = [
            "Fiber optic", "Yes", "Yes",
        ]
        out = DomainFeatureBuilder().fit_transform(frame)
        assert out["num_addon_services"].iloc[0] == 2
        assert out["has_internet"].iloc[0] == 1

    def test_automatic_payment_flag(self, sample_customer):
        frame = sample_customer.copy()
        frame.loc[0, "PaymentMethod"] = "Bank transfer (automatic)"
        assert DomainFeatureBuilder().fit_transform(frame)["is_automatic_payment"].iloc[0] == 1
        assert DomainFeatureBuilder().fit_transform(sample_customer)[
            "is_automatic_payment"
        ].iloc[0] == 0

    def test_is_stateless(self, sample_customer, train_xy):
        """Fitting on different data must not change the output.

        This is what makes the builder safe: it works a row at a time, so it
        cannot carry information between splits.
        """
        X_train, _ = train_xy
        from_sample = DomainFeatureBuilder().fit(sample_customer).transform(sample_customer)
        from_train = DomainFeatureBuilder().fit(X_train).transform(sample_customer)
        pd.testing.assert_frame_equal(from_sample, from_train)


class TestPipelineLeakage:
    def test_scaler_statistics_come_from_the_training_split_alone(
        self, fitted_pipeline, train_xy
    ):
        """The leakage check that cannot be fooled.

        Compares the scaler's learned mean and scale against the training data's
        own statistics. If the pipeline were fitted on anything beyond the
        training split — the whole dataset, train plus validation — these numbers
        would differ, and no amount of later care could hide it.

        An earlier version of this test instead asserted that transformed
        *training* features average ~0 while *validation* features do not. That
        version was too weak to be useful: StandardScaler zeroes the mean of
        whatever it was fitted on **as a whole**, so fitting on train+val leaves
        validation alone at -0.011 — indistinguishable from the honest -0.014.
        It caught only "fitted on validation only", which nobody does by
        accident, and missed "fitted on everything", which is the mistake people
        actually make. Verified by mutation: fitting on train+val makes this
        version fail and the old one pass.
        """
        X_train, _ = train_xy
        domain = fitted_pipeline.named_steps["domain"]
        scaler = (
            fitted_pipeline.named_steps["prepare"]
            .named_transformers_["numeric"]
            .named_steps["scale"]
        )

        engineered = domain.transform(X_train)[NUMERIC_FEATURES].astype(float)
        np.testing.assert_allclose(
            scaler.mean_,
            engineered.mean().to_numpy(),
            rtol=1e-9,
            err_msg="Scaler means differ from the training split — it was fitted "
            "on data beyond train. This is leakage.",
        )
        np.testing.assert_allclose(
            scaler.scale_,
            engineered.std(ddof=0).to_numpy(),
            rtol=1e-9,
            err_msg="Scaler scales differ from the training split — leakage.",
        )

    def test_transformed_training_features_are_centred(self, fitted_pipeline, train_xy):
        """Weaker companion to the check above, kept as a smoke test that the
        scaler was fitted at all rather than passed through."""
        X_train, _ = train_xy
        transformed = fitted_pipeline.transform(X_train)
        for column in ("tenure", "MonthlyCharges", "TotalCharges"):
            assert abs(transformed[column].mean()) < 1e-10

    def test_transform_is_deterministic(self, fitted_pipeline, val_xy):
        X_val, _ = val_xy
        pd.testing.assert_frame_equal(
            fitted_pipeline.transform(X_val), fitted_pipeline.transform(X_val)
        )

    def test_unseen_category_does_not_crash(self, fitted_pipeline, sample_customer):
        """handle_unknown="ignore" is what stops a novel category 500ing the API."""
        frame = sample_customer.copy()
        frame.loc[0, "Contract"] = "Three year"  # never present in training
        out = fitted_pipeline.transform(frame)
        assert len(out) == 1
        assert np.isfinite(out.to_numpy().astype(float)).all()

    def test_produces_the_expected_feature_count(self, fitted_pipeline, val_xy):
        X_val, _ = val_xy
        assert fitted_pipeline.transform(X_val).shape[1] == 35

    def test_output_is_all_numeric(self, fitted_pipeline, val_xy):
        X_val, _ = val_xy
        out = fitted_pipeline.transform(X_val)
        assert all(np.issubdtype(dtype, np.number) for dtype in out.dtypes)
