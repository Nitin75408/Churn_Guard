"""Measure what data leakage actually costs, on this dataset.

Run it::

    uv run python -m churn_guard.experiments.leakage

Why this is committed rather than a one-off notebook cell
---------------------------------------------------------
The project claims that balancing classes before splitting inflates PR-AUC by
about 60%. A claim like that is only worth making if anyone can re-run it, so
the experiment lives in the repository beside the code it warns about.

The mistake being measured is a common one: "the classes are imbalanced, so let
me balance them first". Done before the split, the duplicated minority rows land
on both sides of it, and the model is scored partly on customers it memorised.

The comparison is deliberately fair — same features, same model, same seed, same
split proportions. The only difference is the order of two operations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from churn_guard.config import ConfigSection, load_config
from churn_guard.data.ingest import load_raw_data
from churn_guard.logger import get_logger

logger = get_logger(__name__)

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]


def prepare(df: pd.DataFrame, target: str, positive: str) -> tuple[pd.DataFrame, pd.Series]:
    """Minimal feature prep, identical for both arms of the experiment."""
    data = df.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)
    y = (data[target] == positive).astype(int)
    X = pd.concat(
        [data[NUMERIC], pd.get_dummies(data[CATEGORICAL], drop_first=True)], axis=1
    )
    return X, y


def balance(
    X: pd.DataFrame, y: pd.Series, rng: np.random.RandomState
) -> tuple[pd.DataFrame, pd.Series]:
    """Naive random oversampling — duplicate minority rows until the classes match."""
    minority = y[y == 1].index
    needed = int((y == 0).sum() - (y == 1).sum())
    extra = rng.choice(minority, size=needed, replace=True)
    index = np.concatenate([X.index.to_numpy(), extra])
    return X.loc[index], y.loc[index]


def fit_and_score(X_tr, y_tr, X_te, y_te, seed: int) -> tuple[float, float]:
    model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(X_tr, y_tr)
    probabilities = model.predict_proba(X_te)[:, 1]
    return (
        float(roc_auc_score(y_te, probabilities)),
        float(average_precision_score(y_te, probabilities)),
    )


def run(cfg: ConfigSection | None = None) -> dict:
    cfg = cfg or load_config()
    seed = int(cfg.project.random_seed)
    rng = np.random.RandomState(seed)

    df = load_raw_data(cfg, validate=False)
    X, y = prepare(df, cfg.data.target_column, cfg.data.positive_label)

    # --- correct: split first, then balance the training half only ------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    Xb_train, yb_train = balance(X_train, y_train, rng)
    honest_roc, honest_pr = fit_and_score(Xb_train, yb_train, X_test, y_test, seed)

    # --- wrong: balance everything, then split --------------------------------
    Xb, yb = balance(X, y, rng)
    Xl_train, Xl_test, yl_train, yl_test = train_test_split(
        Xb, yb, test_size=0.2, stratify=yb, random_state=seed
    )
    leaky_roc, leaky_pr = fit_and_score(Xl_train, yl_train, Xl_test, yl_test, seed)

    shared = len(set(Xl_train.index) & set(Xl_test.index))
    contaminated = float(Xl_test.index.isin(Xl_train.index).mean())

    results = {
        "customers_in_both_splits": int(shared),
        "share_of_test_seen_in_training": round(contaminated, 4),
        "leaky": {"roc_auc": round(leaky_roc, 4), "pr_auc": round(leaky_pr, 4)},
        "honest": {"roc_auc": round(honest_roc, 4), "pr_auc": round(honest_pr, 4)},
        "roc_auc_inflation_pct": round(100 * (leaky_roc / honest_roc - 1), 1),
        "pr_auc_inflation_pct": round(100 * (leaky_pr / honest_pr - 1), 1),
    }

    print(f"\n{'=' * 74}\n  WHAT LEAKAGE COSTS\n{'=' * 74}")
    print(f"\n  Customers on both sides of the split : {shared:,}")
    print(f"  Share of 'unseen' test already seen  : {contaminated:.1%}")
    print(f"\n  {'':<44}{'ROC-AUC':>12}{'PR-AUC':>12}")
    print("  " + "-" * 68)
    print(f"  {'WRONG: balance then split (reported)':<44}{leaky_roc:>12.4f}{leaky_pr:>12.4f}")
    print(f"  {'RIGHT: split then balance train (truth)':<44}{honest_roc:>12.4f}{honest_pr:>12.4f}")
    print("  " + "-" * 68)
    print(f"  ROC-AUC overstated by {results['roc_auc_inflation_pct']:>5.1f}%")
    print(f"  PR-AUC  overstated by {results['pr_auc_inflation_pct']:>5.1f}%")
    print(f"\n{'=' * 74}\n")

    path = Path(cfg.paths.reports) / "leakage_experiment.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path.name)
    return results


if __name__ == "__main__":
    run()
