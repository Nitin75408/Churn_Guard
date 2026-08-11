"""Cost model, metrics and the business-rule baseline.

The threshold formula is worth testing because it is derived, not tuned, and an
error in it would quietly change every contact decision the system makes while
leaving all the ML metrics untouched.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import cross_val_score

from churn_guard.models.baselines import BusinessRuleClassifier
from churn_guard.models.metrics import Costs, evaluate, expected_value, precision_at_k


@pytest.fixture
def costs() -> Costs:
    return Costs(clv=780.0, offer_cost=78.0, acceptance_rate=0.30, capacity=500)


class TestCostModel:
    def test_true_positive_value(self, costs):
        # 0.30 x 780 - 78
        assert costs.value_true_positive == pytest.approx(156.0)

    def test_false_positive_cost_is_the_wasted_offer(self, costs):
        assert costs.value_false_positive == pytest.approx(-78.0)

    def test_optimal_threshold_formula(self, costs):
        """offer_cost / (acceptance_rate x CLV) = 78 / 234."""
        assert costs.optimal_threshold == pytest.approx(1 / 3, abs=1e-9)

    def test_threshold_moves_inversely_with_acceptance_rate(self):
        """A less effective offer justifies contacting fewer people.

        This is the relationship the sensitivity analysis explores; if it ever
        inverted, the system would contact more customers precisely when
        intervening works least.
        """
        low = Costs(clv=780, offer_cost=78, acceptance_rate=0.15, capacity=500)
        high = Costs(clv=780, offer_cost=78, acceptance_rate=0.50, capacity=500)
        assert low.optimal_threshold > high.optimal_threshold

    def test_threshold_is_below_the_naive_default(self, costs):
        """0.5 assumes both errors cost the same, which they do not here."""
        assert costs.optimal_threshold < 0.5


class TestExpectedValue:
    def test_contacting_nobody_scores_zero(self, costs):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.zeros(4, dtype=int)
        assert expected_value(y_true, y_pred, costs) == 0.0

    def test_perfect_targeting(self, costs):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert expected_value(y_true, y_pred, costs) == pytest.approx(2 * 156.0)

    def test_contacting_everyone_pays_for_the_false_positives(self, costs):
        y_true = np.array([1, 0, 0, 0])
        y_pred = np.ones(4, dtype=int)
        assert expected_value(y_true, y_pred, costs) == pytest.approx(156.0 - 3 * 78.0)


class TestPrecisionAtK:
    def test_perfect_ranking(self):
        y_true = np.array([1, 1, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        assert precision_at_k(y_true, y_prob, k=2) == 1.0

    def test_inverted_ranking(self):
        y_true = np.array([1, 1, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.7])
        assert precision_at_k(y_true, y_prob, k=2) == 0.0

    def test_k_larger_than_population_is_clamped(self):
        y_true = np.array([1, 0])
        y_prob = np.array([0.9, 0.1])
        assert precision_at_k(y_true, y_prob, k=100) == 0.5


class TestEvaluate:
    def test_pr_auc_of_random_scores_approaches_the_base_rate(self):
        """No-skill PR-AUC is class prevalence, not 0.5.

        Comparing PR-AUC against 0.5 badly misjudges a model on imbalanced data,
        so the property is worth pinning down.
        """
        rng = np.random.RandomState(0)
        y_true = (rng.rand(4000) < 0.25).astype(int)
        y_prob = rng.rand(4000)
        metrics = evaluate("random", y_true, y_prob, Costs(780, 78, 0.3, 500))
        assert metrics.pr_auc == pytest.approx(0.25, abs=0.05)
        assert metrics.roc_auc == pytest.approx(0.5, abs=0.05)

    def test_capacity_scales_to_the_split_size(self, costs):
        """Top-500-of-7043 must become top-75-of-1057 when scoring a split."""
        y_true = np.zeros(1057, dtype=int)
        y_true[:280] = 1
        y_prob = np.linspace(0, 1, 1057)
        metrics = evaluate("m", y_true, y_prob, costs, total_population=7043)
        assert metrics.k == 75


class TestBusinessRuleBaseline:
    def test_flags_new_month_to_month_customers(self, train_xy):
        X_train, y_train = train_xy
        model = BusinessRuleClassifier(tenure_threshold=6).fit(X_train, y_train)
        flagged = model.predict(X_train).astype(bool)
        subset = X_train[flagged]
        assert (subset["Contract"] == "Month-to-month").all()
        assert (subset["tenure"] < 6).all()

    def test_flagged_branch_has_the_higher_churn_rate(self, train_xy):
        X_train, y_train = train_xy
        model = BusinessRuleClassifier().fit(X_train, y_train)
        assert model.rate_flagged_ > model.rate_unflagged_

    def test_works_with_sklearn_cross_validation(self, train_xy):
        """Regression test for the estimator-type bug.

        With ``BaseEstimator`` before ``ClassifierMixin``, scikit-learn failed to
        recognise this as a classifier, handed the scorer both probability
        columns, and silently produced nan instead of raising.
        """
        X_train, y_train = train_xy
        scores = cross_val_score(
            BusinessRuleClassifier(), X_train, y_train, cv=3, scoring="average_precision"
        )
        assert not np.isnan(scores).any(), "scoring returned nan — check the MRO"
        assert (scores > 0.25).all()
