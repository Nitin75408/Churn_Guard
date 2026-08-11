"""Baseline models — the bar every later model has to clear.

Run it::

    uv run python -m churn_guard.models.baselines

Three baselines, in increasing order of sophistication:

``B0`` — majority class
    Predicts "nobody churns". Establishes the floor and demonstrates why
    accuracy is the wrong headline metric: it scores ~73.5% while catching zero
    churners.

``B1`` — the existing business rule
    ``Contract == month-to-month AND tenure < 6``. **This is the important one.**
    Beating random is meaningless; beating what the business already does without
    any ML is the number that justifies the project. A model that cannot clear
    this should not ship, however good its AUC looks in isolation.

``B2`` — logistic regression
    Simple, fast, interpretable, and on tabular problems of this size often
    within a couple of points of a tuned gradient booster. It is a genuine
    candidate, not a formality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from churn_guard.config import ConfigSection, load_config
from churn_guard.features.build import build_feature_pipeline, load_xy
from churn_guard.logger import get_logger
from churn_guard.models.metrics import Costs, Metrics, evaluate, format_table

logger = get_logger(__name__)


class BusinessRuleClassifier(ClassifierMixin, BaseEstimator):
    # Mixin first, BaseEstimator second. scikit-learn resolves an estimator's
    # type through __sklearn_tags__ along the MRO; with the order reversed it
    # does not register as a classifier, and scorers then hand the full
    # two-column predict_proba output to a metric expecting one column.
    """The heuristic the retention team uses today, as an estimator.

    Flags a customer when ``Contract == "Month-to-month"`` and
    ``tenure < tenure_threshold``.

    ``predict_proba`` returns the empirical churn rate of each branch, measured
    on the training data, rather than a hard 0/1. A hard rule produces only two
    distinct scores and therefore cannot rank customers at all — which would
    make its PR-AUC artificially terrible and the comparison unfair. Giving it
    calibrated branch rates is the strongest honest version of the rule.

    That it still cannot rank *within* a branch is precisely the limitation a
    model is meant to fix: with capacity for 500 calls and 2,708 month-to-month
    customers, "flagged" is not an answer to "which 500?".
    """

    def __init__(self, tenure_threshold: int = 6) -> None:
        self.tenure_threshold = tenure_threshold

    def _rule(self, X: pd.DataFrame) -> np.ndarray:
        return (
            (X["Contract"] == "Month-to-month") & (X["tenure"] < self.tenure_threshold)
        ).to_numpy()

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BusinessRuleClassifier":
        self.classes_ = np.array([0, 1])
        y = np.asarray(y)
        flagged = self._rule(X)
        # Fall back to the overall rate if a branch is empty, which can happen
        # in a small cross-validation fold.
        overall = float(y.mean())
        self.rate_flagged_ = float(y[flagged].mean()) if flagged.any() else overall
        self.rate_unflagged_ = float(y[~flagged].mean()) if (~flagged).any() else overall
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        flagged = self._rule(X)
        p = np.where(flagged, self.rate_flagged_, self.rate_unflagged_)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._rule(X).astype(int)


def build_baselines(cfg: ConfigSection) -> dict[str, BaseEstimator]:
    """The three baseline estimators, each ready to fit on raw data."""
    seed = int(cfg.project.random_seed)

    return {
        "B0 majority class": Pipeline(
            [
                ("features", build_feature_pipeline()),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "B1 business rule": BusinessRuleClassifier(tenure_threshold=6),
        "B2 logistic regression": Pipeline(
            [
                ("features", build_feature_pipeline()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        # Reweights the loss so the 26.5% minority is not
                        # ignored. Preferred over resampling: no duplicated
                        # rows, so no risk of the leakage we measured on Day 1.
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def cross_validate_model(
    name: str, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, cfg: ConfigSection
) -> tuple[float, float]:
    """Repeated stratified K-fold CV on the training split.

    Why cross-validate at all when we have a validation split? A single split of
    1,057 rows gives a noisy estimate — differences of a point or two between
    models can be pure luck. CV averages over 5 folds and, more usefully, reports
    the spread, so we can tell a real improvement from noise.

    Returns:
        Mean and standard deviation of PR-AUC across folds.
    """
    folds = StratifiedKFold(
        n_splits=int(cfg.cross_validation.n_splits),
        shuffle=bool(cfg.cross_validation.shuffle),
        random_state=int(cfg.project.random_seed),
    )
    results = cross_validate(
        model, X, y, cv=folds, scoring="average_precision", n_jobs=-1
    )
    scores = results["test_score"]
    logger.info(
        "%-24s CV PR-AUC = %.4f +/- %.4f", name, scores.mean(), scores.std()
    )
    return float(scores.mean()), float(scores.std())


def main() -> None:
    cfg = load_config()
    costs = Costs.from_config(cfg)

    X_train, y_train = load_xy("train", cfg)
    X_val, y_val = load_xy("val", cfg)
    population = len(X_train) + len(X_val) + 1057  # full customer base

    print(f"\n{'=' * 92}")
    print("  BASELINES")
    print(f"{'=' * 92}")
    print(f"  Decision threshold: {costs.optimal_threshold:.4f}  "
          f"(= offer_cost / (acceptance_rate x CLV) = "
          f"{costs.offer_cost:.0f} / ({costs.acceptance_rate} x {costs.clv:.0f}))")
    print(f"  Value of a true positive : +${costs.value_true_positive:,.0f}")
    print(f"  Cost  of a false positive: -${costs.offer_cost:,.0f}")

    models = build_baselines(cfg)

    print(f"\n  Cross-validation on the training split ({cfg.cross_validation.n_splits} folds):")
    cv_results: dict[str, tuple[float, float]] = {}
    for name, model in models.items():
        cv_results[name] = cross_validate_model(name, model, X_train, y_train, cfg)

    rows: list[Metrics] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        rows.append(
            evaluate(name, y_val.to_numpy(), y_prob, costs, total_population=population)
        )

    print(f"\n  Held-out validation set ({len(y_val):,} customers, "
          f"threshold {costs.optimal_threshold:.3f}):\n")
    print("  " + format_table(rows).replace("\n", "\n  "))

    print(f"\n  P@k = precision among the top {rows[0].k} highest-risk customers")
    print("  $/1000 = net retention value per 1,000 customers scored\n")

    best = max(rows, key=lambda r: r.pr_auc)
    logger.info("Best baseline by PR-AUC: %s (%.4f)", best.model, best.pr_auc)
    logger.warning(
        "Every model from here must beat B1 (the existing business rule) to "
        "justify deploying ML at all."
    )


if __name__ == "__main__":
    main()
