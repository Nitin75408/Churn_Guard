"""Evaluation metrics — statistical and financial.

Two layers, reported side by side:

**ML metrics** answer "does the model rank customers well?" — PR-AUC as primary
(the classes are imbalanced at 2.77:1, where accuracy and to a lesser extent
ROC-AUC flatter a model for getting the majority right), ROC-AUC and
threshold-dependent precision/recall alongside.

**Business metrics** answer "does this make money?" — expected net value from the
matrix in ``docs/problem-definition.md``, plus Precision@K, which is what the
capacity-limited retention team actually experiences.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_guard.config import ConfigSection


@dataclass
class Costs:
    """Business economics, read from ``configs/config.yaml``."""

    clv: float
    offer_cost: float
    acceptance_rate: float
    capacity: int

    @classmethod
    def from_config(cls, cfg: ConfigSection) -> Costs:
        return cls(
            clv=float(cfg.costs.customer_lifetime_value),
            offer_cost=float(cfg.costs.retention_offer_cost),
            acceptance_rate=float(cfg.costs.offer_acceptance_rate),
            capacity=int(cfg.costs.monthly_contact_capacity),
        )

    @property
    def value_true_positive(self) -> float:
        """Net gain from contacting a customer who would have churned.

        Expected retained value minus the offer cost. Negative would mean the
        retention programme destroys value and should not run at all.
        """
        return self.acceptance_rate * self.clv - self.offer_cost

    @property
    def value_false_positive(self) -> float:
        """Loss from contacting a customer who was never going to leave."""
        return -self.offer_cost

    @property
    def optimal_threshold(self) -> float:
        """Probability above which contacting is worth it.

        Derived from ``p × (acceptance_rate × CLV) > offer_cost``. Independent of
        the model — it is a property of the economics, and the model only has to
        supply calibrated probabilities.
        """
        return self.offer_cost / (self.acceptance_rate * self.clv)


@dataclass
class Metrics:
    """One row of results for one model."""

    model: str
    # Threshold-independent — measure ranking quality.
    roc_auc: float
    pr_auc: float
    brier: float
    # Threshold-dependent.
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    # Business.
    precision_at_k: float
    k: int
    value_per_1000: float

    def as_dict(self) -> dict[str, float | str | int]:
        return asdict(self)


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Share of genuine churners among the k highest-risk customers.

    This is the metric the retention team lives with. They cannot call everyone
    above a threshold — they can make k calls, so what matters is the hit rate
    among the model's top k, not its behaviour across the full ranking.
    """
    if k <= 0:
        return 0.0
    k = min(k, len(y_true))
    top_k = np.argsort(y_prob)[::-1][:k]
    return float(np.mean(y_true[top_k]))


def expected_value(y_true: np.ndarray, y_pred: np.ndarray, costs: Costs) -> float:
    """Net value of acting on these predictions, versus contacting nobody.

    Only the two "contact" cells carry value: a correct contact earns
    ``acceptance_rate × CLV − offer_cost``, a wasted one costs ``offer_cost``.
    Customers we do not contact score zero by construction, since doing nothing
    is the baseline everything is measured against.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tp * costs.value_true_positive + fp * costs.value_false_positive)


def evaluate(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    costs: Costs,
    threshold: float | None = None,
    total_population: int | None = None,
) -> Metrics:
    """Score a set of predicted probabilities.

    Args:
        y_true: Binary ground truth, 1 = churned.
        y_prob: Predicted probability of churn.
        costs: Business economics.
        threshold: Decision cut-off. Defaults to the cost-optimal value rather
            than 0.5 — 0.5 is only correct when both errors cost the same.
        total_population: Size of the full customer base, used to scale the
            monthly contact capacity down to this split. Defaults to the length
            of ``y_true``.

    Returns:
        Statistical and business metrics for one model.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    threshold = costs.optimal_threshold if threshold is None else threshold
    y_pred = (y_prob >= threshold).astype(int)

    # The retention team's capacity is defined against the whole customer base,
    # so scale it proportionally when scoring a 15% split — otherwise "top 500"
    # would mean half of a 1,057-row validation set.
    population = total_population or len(y_true)
    k = max(1, round(costs.capacity * len(y_true) / population))

    n = len(y_true)
    return Metrics(
        model=model_name,
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        # Brier score: mean squared error of the probabilities themselves.
        # Lower is better. A model can rank perfectly and still be badly
        # calibrated, which matters because we feed probabilities into a
        # money calculation.
        brier=float(brier_score_loss(y_true, y_prob)),
        threshold=float(threshold),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        accuracy=float((y_pred == y_true).mean()),
        precision_at_k=precision_at_k(y_true, y_prob, k),
        k=k,
        value_per_1000=expected_value(y_true, y_pred, costs) / n * 1000,
    )


def format_table(rows: list[Metrics]) -> str:
    """Render results as a fixed-width comparison table."""
    header = (
        f"{'model':<26}{'PR-AUC':>9}{'ROC-AUC':>9}{'Brier':>8}"
        f"{'prec':>7}{'recall':>8}{'F1':>7}{'P@k':>7}{'$/1000':>10}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.model:<26}{row.pr_auc:>9.4f}{row.roc_auc:>9.4f}{row.brier:>8.4f}"
            f"{row.precision:>7.3f}{row.recall:>8.3f}{row.f1:>7.3f}"
            f"{row.precision_at_k:>7.3f}{row.value_per_1000:>10.0f}"
        )
    return "\n".join(lines)
