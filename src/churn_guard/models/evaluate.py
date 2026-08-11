"""Choose the operating threshold, then evaluate once on the sealed test set.

Run it::

    uv run python -m churn_guard.models.evaluate

Order of operations, and why it matters
---------------------------------------
1. verify the test set's fingerprint still matches the Day 1 manifest
2. sweep thresholds **on validation**, pick the one maximising expected value
3. run a sensitivity analysis over the shakiest cost assumption
4. only then load the test set, score it once, and report whatever it says

The threshold is a parameter. Fitting it on the test set would make the final
number optimistic in exactly the way the held-out set exists to prevent — the
same mistake as tuning hyperparameters on test, just less obvious because a
threshold does not feel like "training".
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)

from churn_guard.config import ConfigSection, load_config
from churn_guard.data.split import MANIFEST_FILENAME, _fingerprint, load_split
from churn_guard.exception import DataValidationError, ModelNotFoundError
from churn_guard.features.build import load_xy
from churn_guard.logger import get_logger
from churn_guard.models.metrics import Costs, evaluate, expected_value, format_table

logger = get_logger(__name__)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def verify_test_seal(cfg: ConfigSection) -> None:
    """Confirm the test split is byte-identical to the one created on Day 1.

    Re-hashes the test set's customer IDs and compares against the manifest
    written before any modelling began. A mismatch means the seed changed, the
    upstream data shifted, or the split was rerun — in any of which cases the
    "held-out" set may have been seen during development and the final number is
    not trustworthy.

    Raises:
        DataValidationError: If the manifest is missing or the hash differs.
    """
    manifest_path: Path = cfg.paths.data_interim / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise DataValidationError(
            f"{manifest_path} not found — cannot prove the test set is unchanged."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["splits"]["test"]["id_fingerprint"]
    actual = _fingerprint(load_split("test", cfg)[cfg.data.id_column])

    if expected != actual:
        raise DataValidationError(
            f"Test set fingerprint mismatch: manifest says {expected}, found "
            f"{actual}. The held-out set is not the one sealed on Day 1, so any "
            f"score computed on it is unreliable."
        )
    logger.info("Test seal verified: fingerprint %s unchanged since Day 1.", actual)


def load_model(cfg: ConfigSection):
    """Load the fitted pipeline saved by the training run."""
    path: Path = cfg.paths.models / "model.joblib"
    if not path.is_file():
        raise ModelNotFoundError(
            f"No model at {path}. Run 'uv run python -m churn_guard.models.train' first."
        )
    return joblib.load(path)


def sweep_thresholds(
    y_true: np.ndarray, y_prob: np.ndarray, costs: Costs
) -> pd.DataFrame:
    """Expected value and confusion counts across every candidate threshold."""
    grid = np.linspace(0.01, 0.95, 190)
    records = []
    for threshold in grid:
        y_pred = (y_prob >= threshold).astype(int)
        contacted = int(y_pred.sum())
        caught = int(((y_pred == 1) & (y_true == 1)).sum())
        records.append(
            {
                "threshold": threshold,
                "contacted": contacted,
                "contacted_share": contacted / len(y_true),
                "caught": caught,
                "recall": caught / max(int(y_true.sum()), 1),
                "precision": caught / max(contacted, 1),
                "value_per_1000": expected_value(y_true, y_pred, costs) / len(y_true) * 1000,
            }
        )
    return pd.DataFrame(records)


def sensitivity_analysis(
    y_true: np.ndarray, y_prob: np.ndarray, cfg: ConfigSection
) -> pd.DataFrame:
    """How the optimal threshold moves as the acceptance rate assumption moves.

    The 30% offer-acceptance rate is the least evidence-backed number in the
    cost model — it came from a business assumption, not from this dataset. The
    theoretical optimum ``offer_cost / (acceptance_rate x CLV)`` is inversely
    proportional to it, so an honest report shows the range rather than a single
    confident figure.
    """
    base = Costs.from_config(cfg)
    records = []
    for rate in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        costs = Costs(
            clv=base.clv,
            offer_cost=base.offer_cost,
            acceptance_rate=rate,
            capacity=base.capacity,
        )
        sweep = sweep_thresholds(y_true, y_prob, costs)
        best = sweep.loc[sweep["value_per_1000"].idxmax()]
        records.append(
            {
                "acceptance_rate": rate,
                "theoretical_threshold": costs.optimal_threshold,
                "empirical_threshold": float(best["threshold"]),
                "recall": float(best["recall"]),
                "value_per_1000": float(best["value_per_1000"]),
                "programme_viable": costs.value_true_positive > 0,
            }
        )
    return pd.DataFrame(records)


def plot_threshold_sweep(
    sweep: pd.DataFrame, theoretical: float, empirical: float, cfg: ConfigSection
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(sweep["threshold"], sweep["value_per_1000"], color="#4C78A8", lw=2)
    axes[0].axvline(theoretical, color="#E45756", ls="--",
                    label=f"theoretical {theoretical:.3f}")
    axes[0].axvline(empirical, color="#54A24B", ls=":",
                    label=f"empirical {empirical:.3f}")
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].set_xlabel("decision threshold")
    axes[0].set_ylabel("net value per 1,000 customers ($)")
    axes[0].set_title("Expected value vs threshold")
    axes[0].legend(fontsize=8)

    axes[1].plot(sweep["threshold"], sweep["precision"], label="precision", color="#4C78A8")
    axes[1].plot(sweep["threshold"], sweep["recall"], label="recall", color="#E45756")
    axes[1].plot(sweep["threshold"], sweep["contacted_share"],
                 label="share contacted", color="#B279A2", ls="--")
    axes[1].axvline(empirical, color="#54A24B", ls=":")
    axes[1].set_xlabel("decision threshold")
    axes[1].set_title("Precision / recall / contact volume")
    axes[1].legend(fontsize=8)

    fig.suptitle("Choosing the operating threshold (validation set)", fontsize=13)
    path = cfg.paths.figures / "07_threshold_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved figure: %s", path.name)


def plot_final_evaluation(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, cfg: ConfigSection
) -> None:
    """ROC, precision-recall, confusion matrix and calibration on the test set."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0, 0].plot(fpr, tpr, color="#4C78A8", lw=2)
    axes[0, 0].plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    axes[0, 0].set_xlabel("false positive rate")
    axes[0, 0].set_ylabel("true positive rate")
    axes[0, 0].set_title("ROC curve")

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    base_rate = float(y_true.mean())
    axes[0, 1].plot(recall, precision, color="#E45756", lw=2)
    axes[0, 1].axhline(base_rate, ls="--", color="grey", lw=1,
                       label=f"no skill = {base_rate:.3f}")
    axes[0, 1].set_xlabel("recall")
    axes[0, 1].set_ylabel("precision")
    axes[0, 1].set_title("Precision-Recall curve")
    axes[0, 1].legend(fontsize=8)

    y_pred = (y_prob >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["stayed", "churned"],
        cmap="Blues", colorbar=False, ax=axes[1, 0],
    )
    axes[1, 0].set_title(f"Confusion matrix @ {threshold:.3f}")

    # Calibration: of customers the model scored around 0.7, did roughly 70%
    # actually churn? Ranking can be perfect while probabilities are skewed,
    # and skewed probabilities corrupt every value calculation downstream.
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    axes[1, 1].plot(prob_pred, prob_true, marker="o", color="#54A24B")
    axes[1, 1].plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect")
    axes[1, 1].set_xlabel("predicted probability")
    axes[1, 1].set_ylabel("observed churn rate")
    axes[1, 1].set_title("Calibration curve")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Final evaluation — sealed test set", fontsize=14)
    path = cfg.paths.figures / "08_final_evaluation.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved figure: %s", path.name)


def main() -> None:
    cfg = load_config()
    costs = Costs.from_config(cfg)

    verify_test_seal(cfg)
    model = load_model(cfg)

    X_val, y_val = load_xy("val", cfg)
    y_val_true = y_val.to_numpy()
    y_val_prob = model.predict_proba(X_val)[:, 1]

    # ---------------------------------------------------------------- 1 ----
    print(f"\n{'=' * 92}")
    print("  1. THRESHOLD SELECTION  (validation set — test still sealed)")
    print(f"{'=' * 92}")

    sweep = sweep_thresholds(y_val_true, y_val_prob, costs)
    best_row = sweep.loc[sweep["value_per_1000"].idxmax()]
    empirical = float(best_row["threshold"])
    theoretical = costs.optimal_threshold

    print(f"\n  Theoretical optimum : {theoretical:.4f}   "
          f"(offer_cost / (acceptance_rate x CLV))")
    print(f"  Empirical optimum   : {empirical:.4f}   "
          f"(maximises value on validation)")
    print(f"  Difference          : {abs(empirical - theoretical):.4f}")

    default_value = float(
        sweep.iloc[(sweep["threshold"] - 0.5).abs().idxmin()]["value_per_1000"]
    )
    print(f"\n  Value at threshold 0.500 (sklearn default): ${default_value:>9,.0f} per 1,000")
    print(f"  Value at threshold {empirical:.3f} (cost-optimal)  : "
          f"${best_row['value_per_1000']:>9,.0f} per 1,000")
    print(f"  Gain from choosing the threshold properly : "
          f"${best_row['value_per_1000'] - default_value:>9,.0f} per 1,000")

    plot_threshold_sweep(sweep, theoretical, empirical, cfg)

    # ---------------------------------------------------------------- 2 ----
    print(f"\n{'=' * 92}")
    print("  2. SENSITIVITY — how much does the acceptance-rate assumption matter?")
    print(f"{'=' * 92}\n")
    sensitivity = sensitivity_analysis(y_val_true, y_val_prob, cfg)
    print(f"  {'accept':>8}{'theory':>10}{'empirical':>12}{'recall':>9}"
          f"{'$/1000':>11}   viable")
    print("  " + "-" * 62)
    for _, row in sensitivity.iterrows():
        print(f"  {row['acceptance_rate']:>8.0%}{row['theoretical_threshold']:>10.3f}"
              f"{row['empirical_threshold']:>12.3f}{row['recall']:>9.3f}"
              f"{row['value_per_1000']:>11,.0f}   "
              f"{'yes' if row['programme_viable'] else 'NO'}")

    # ---------------------------------------------------------------- 3 ----
    print(f"\n{'=' * 92}")
    print("  3. CAPACITY — the retention team cannot call everyone")
    print(f"{'=' * 92}")
    flagged_share = float(best_row["contacted_share"])
    capacity_share = costs.capacity / 7043
    print(f"\n  At threshold {empirical:.3f} the model flags "
          f"{flagged_share:.1%} of customers.")
    print(f"  The team can contact {costs.capacity}/month = {capacity_share:.1%}.")
    if flagged_share > capacity_share:
        print("\n  Capacity binds. The threshold is not the operative constraint —")
        print(f"  ranking is. Contact the top {capacity_share:.1%} by score, which is")
        print("  what Precision@K measures.")

    # ---------------------------------------------------------------- 4 ----
    print(f"\n{'=' * 92}")
    print("  4. FINAL EVALUATION — opening the sealed test set, once")
    print(f"{'=' * 92}")

    X_test, y_test = load_xy("test", cfg)
    y_test_true = y_test.to_numpy()
    y_test_prob = model.predict_proba(X_test)[:, 1]
    population = len(X_val) + len(X_test) + 4929

    rows = [
        evaluate("validation", y_val_true, y_val_prob, costs,
                 threshold=empirical, total_population=population),
        evaluate("TEST (sealed)", y_test_true, y_test_prob, costs,
                 threshold=empirical, total_population=population),
    ]
    print()
    print("  " + format_table(rows).replace("\n", "\n  "))

    plot_final_evaluation(y_test_true, y_test_prob, empirical, cfg)

    # ---------------------------------------------------------------- 5 ----
    print(f"\n{'=' * 92}")
    print("  5. ACCEPTANCE CRITERIA  (fixed on Day 1, before any results)")
    print(f"{'=' * 92}\n")
    test_metrics = rows[1]
    targets = {
        "PR-AUC": (test_metrics.pr_auc, float(cfg.evaluation.targets.average_precision)),
        "ROC-AUC": (test_metrics.roc_auc, float(cfg.evaluation.targets.roc_auc)),
        "Recall": (test_metrics.recall, float(cfg.evaluation.targets.recall)),
    }
    for label, (achieved, target) in targets.items():
        status = "PASS" if achieved >= target else "MISS"
        print(f"  {label:<10} target >= {target:.3f}   achieved {achieved:.3f}   {status}")

    results = {
        "operating_threshold": empirical,
        "theoretical_threshold": theoretical,
        "validation": rows[0].as_dict(),
        "test": test_metrics.as_dict(),
        "sensitivity": sensitivity.to_dict(orient="records"),
        "acceptance_criteria": {
            label: {"target": target, "achieved": achieved, "passed": achieved >= target}
            for label, (achieved, target) in targets.items()
        },
    }
    results_path = Path(cfg.paths.reports) / "final_evaluation.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", results_path.name)
    logger.warning("Test set has now been used. Any further tuning invalidates it.")


if __name__ == "__main__":
    main()
