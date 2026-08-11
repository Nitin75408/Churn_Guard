"""Explainability and fairness checks for the selected model.

Run it::

    uv run python -m churn_guard.models.explain

Three questions, answered separately:

**What drives churn overall?** Global SHAP importance, cross-checked against the
model's own coefficients. Two independent routes to the same answer is a useful
sanity check — if they disagree, something is wrong.

**Why is *this* customer flagged?** Per-customer SHAP attributions. This is what a
retention agent actually needs: not "contract type matters in general" but "this
person is at 71% because they are month-to-month with 3 months tenure".

**Does the model treat groups differently?** Sliced metrics by gender, senior
status and family status. A model can look excellent on average while failing a
subgroup badly, and the average will never show it.

A note on units
---------------
SHAP values for a logistic regression are in **log-odds**, not probability. They
sum exactly to ``model output - base value``, which makes them additive and
therefore fairly attributable. Probability is a squashed version of that sum, so
probabilities do not add up the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, roc_auc_score

from churn_guard.config import ConfigSection, load_config
from churn_guard.data.split import load_split
from churn_guard.features.build import load_xy
from churn_guard.logger import get_logger
from churn_guard.models.evaluate import load_model
from churn_guard.models.metrics import Costs

logger = get_logger(__name__)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

# Columns to check for unequal treatment. gender was deliberately kept in the
# dataset for exactly this purpose despite carrying almost no predictive signal.
FAIRNESS_SLICES = ["gender", "SeniorCitizen", "Partner", "Dependents"]


def coefficient_table(pipeline, top_n: int = 15) -> pd.DataFrame:
    """Logistic regression coefficients as odds ratios.

    Features were standard-scaled, so coefficients are directly comparable: each
    is the change in log-odds per one standard deviation of that feature. The
    odds ratio ``exp(coef)`` is the more readable form — 1.5 means "50% higher
    odds of churn per standard deviation".
    """
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["features"].get_feature_names_out()
    coefficients = model.coef_[0]

    table = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False)
    return table.head(top_n).reset_index(drop=True)


def compute_shap(pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame):
    """SHAP values for the classifier stage, on transformed features.

    The pipeline is split deliberately: preprocessing runs first so SHAP sees the
    35 model-ready columns it can attribute to, rather than the 20 raw ones the
    model never sees.

    ``LinearExplainer`` is exact for a linear model — no sampling, no
    approximation — which makes it both faster and more trustworthy than the
    model-agnostic explainers.
    """
    features = pipeline.named_steps["features"]
    model = pipeline.named_steps["model"]

    background = features.transform(X_background)
    explain = features.transform(X_explain)

    explainer = shap.LinearExplainer(model, background)
    return explainer(explain), explain


def plot_global_importance(shap_values, cfg: ConfigSection) -> None:
    """Beeswarm and mean-|SHAP| bar chart."""
    fig = plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP — feature effects across the test set", fontsize=12)
    path = cfg.paths.figures / "09_shap_beeswarm.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved figure: %s", path.name)

    fig = plt.figure(figsize=(8, 6))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("SHAP — mean absolute impact", fontsize=12)
    path = cfg.paths.figures / "10_shap_importance.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved figure: %s", path.name)


def explain_customers(
    shap_values,
    X_raw: pd.DataFrame,
    probabilities: np.ndarray,
    cfg: ConfigSection,
    n_examples: int = 2,
) -> list[dict]:
    """Per-customer explanations for the highest- and lowest-risk cases."""
    order = np.argsort(probabilities)
    picks = list(order[-n_examples:][::-1]) + list(order[:n_examples])
    labels = ["highest risk"] * n_examples + ["lowest risk"] * n_examples

    explanations: list[dict] = []
    for rank, (index, label) in enumerate(zip(picks, labels)):
        index = int(index)
        values = shap_values[index]

        fig = plt.figure(figsize=(9, 5))
        shap.plots.waterfall(values, max_display=12, show=False)
        plt.title(
            f"{label} — churn probability {probabilities[index]:.1%}", fontsize=11
        )
        path = cfg.paths.figures / f"11_shap_customer_{rank + 1}_{label.replace(' ', '_')}.png"
        fig.savefig(path)
        plt.close(fig)

        contributions = pd.Series(values.values, index=values.feature_names)
        top = contributions.reindex(contributions.abs().sort_values(ascending=False).index)[:6]

        explanations.append(
            {
                "label": label,
                "customer_id": str(X_raw.iloc[index][cfg.data.id_column]),
                "probability": float(probabilities[index]),
                "top_drivers": {k: round(float(v), 4) for k, v in top.items()},
            }
        )
    logger.info("Saved %d per-customer waterfall plots.", len(picks))
    return explanations


def fairness_report(
    X_raw: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Sliced performance, to surface unequal treatment the average hides.

    Three quantities, which are genuinely different questions:

    * **selection rate** — what share of the group gets contacted (demographic parity)
    * **recall** — of the group's real churners, how many we catch (equal opportunity)
    * **precision** — when we flag someone in this group, how often we are right

    These cannot all be equalised at once when groups have different underlying
    churn rates. That is a mathematical fact, not a modelling failure, so the
    honest approach is to report all three and say which one the business is
    choosing to prioritise.
    """
    y_pred = (probabilities >= threshold).astype(int)
    records = []

    for column in FAIRNESS_SLICES:
        for value in sorted(X_raw[column].unique(), key=str):
            mask = (X_raw[column] == value).to_numpy()
            group_true = y_true[mask]
            group_pred = y_pred[mask]
            group_prob = probabilities[mask]

            if group_true.sum() < 5 or len(group_true) < 30:
                continue  # too small for a meaningful estimate

            true_positive = int(((group_pred == 1) & (group_true == 1)).sum())
            records.append(
                {
                    "slice": f"{column}={value}",
                    "n": int(mask.sum()),
                    "base_rate": float(group_true.mean()),
                    "selection_rate": float(group_pred.mean()),
                    "recall": true_positive / max(int(group_true.sum()), 1),
                    "precision": true_positive / max(int(group_pred.sum()), 1),
                    "roc_auc": float(roc_auc_score(group_true, group_prob)),
                    "pr_auc": float(average_precision_score(group_true, group_prob)),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    cfg = load_config()
    costs = Costs.from_config(cfg)

    results_path = Path(cfg.paths.reports) / "final_evaluation.json"
    threshold = (
        json.loads(results_path.read_text())["operating_threshold"]
        if results_path.is_file()
        else costs.optimal_threshold
    )

    pipeline = load_model(cfg)
    X_train, _ = load_xy("train", cfg)
    X_test, y_test = load_xy("test", cfg)
    y_true = y_test.to_numpy()
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Raw frame retained so fairness slices can use the original columns.
    X_test_raw = load_split("test", cfg)

    # ---------------------------------------------------------------- 1 ----
    print(f"\n{'=' * 88}")
    print("  1. GLOBAL — what drives churn, per the model's own coefficients")
    print(f"{'=' * 88}\n")
    coefficients = coefficient_table(pipeline)
    print(f"  {'feature':<34}{'coef':>9}{'odds ratio':>13}   direction")
    print("  " + "-" * 74)
    for _, row in coefficients.iterrows():
        direction = "increases churn" if row["coefficient"] > 0 else "reduces churn"
        print(f"  {row['feature']:<34}{row['coefficient']:>+9.3f}"
              f"{row['odds_ratio']:>13.3f}   {direction}")

    # ---------------------------------------------------------------- 2 ----
    logger.info("Computing SHAP values...")
    shap_values, _ = compute_shap(pipeline, X_train, X_test)
    plot_global_importance(shap_values, cfg)

    mean_abs = pd.Series(
        np.abs(shap_values.values).mean(axis=0), index=shap_values.feature_names
    ).sort_values(ascending=False)

    print(f"\n{'=' * 88}")
    print("  2. GLOBAL — SHAP mean absolute impact (independent cross-check)")
    print(f"{'=' * 88}\n")
    for feature, value in mean_abs.head(12).items():
        bar = "#" * int(value * 120)
        print(f"  {feature:<34}{value:>8.4f}  {bar}")

    # ---------------------------------------------------------------- 3 ----
    print(f"\n{'=' * 88}")
    print("  3. LOCAL — why these specific customers?")
    print(f"{'=' * 88}")
    explanations = explain_customers(shap_values, X_test_raw, probabilities, cfg)
    for item in explanations:
        print(f"\n  {item['label']} — {item['customer_id']} "
              f"(churn probability {item['probability']:.1%})")
        for feature, contribution in item["top_drivers"].items():
            arrow = "^" if contribution > 0 else "v"
            print(f"      {arrow} {feature:<32}{contribution:>+8.3f} log-odds")

    # ---------------------------------------------------------------- 4 ----
    print(f"\n{'=' * 88}")
    print(f"  4. FAIRNESS — sliced performance at threshold {threshold:.3f}")
    print(f"{'=' * 88}\n")
    fairness = fairness_report(X_test_raw, y_true, probabilities, threshold)
    print(f"  {'slice':<26}{'n':>6}{'base':>8}{'flagged':>9}"
          f"{'recall':>9}{'prec':>8}{'ROC-AUC':>10}")
    print("  " + "-" * 76)
    for _, row in fairness.iterrows():
        print(f"  {row['slice']:<26}{row['n']:>6.0f}{row['base_rate']:>8.3f}"
              f"{row['selection_rate']:>9.3f}{row['recall']:>9.3f}"
              f"{row['precision']:>8.3f}{row['roc_auc']:>10.3f}")

    spread = fairness["roc_auc"].max() - fairness["roc_auc"].min()
    print(f"\n  Widest ROC-AUC gap between slices: {spread:.3f}")
    if spread > 0.10:
        logger.warning("Performance differs materially across slices — investigate.")
    else:
        logger.info("No large performance gap across the slices checked.")

    output = {
        "threshold": threshold,
        "top_coefficients": coefficients.to_dict(orient="records"),
        "shap_mean_abs": mean_abs.head(15).to_dict(),
        "customer_explanations": explanations,
        "fairness": fairness.to_dict(orient="records"),
    }
    path = Path(cfg.paths.reports) / "explainability.json"
    path.write_text(json.dumps(output, indent=2, default=float), encoding="utf-8")
    logger.info("Wrote %s", path.name)


if __name__ == "__main__":
    main()
