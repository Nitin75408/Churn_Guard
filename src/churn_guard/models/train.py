"""Train, tune and compare candidate models, tracking everything in MLflow.

Run it::

    uv run python -m churn_guard.models.train

Then inspect the results::

    uv run mlflow ui --backend-store-uri mlruns

Search strategy
---------------
``RandomizedSearchCV`` rather than ``GridSearchCV``. For an equal budget, random
sampling explores many more distinct values of whichever hyperparameter actually
matters, where a grid spends most of its fits re-testing the same value of the
important parameter against variations of unimportant ones.

Scoring is **average precision** (PR-AUC), matching the primary metric declared
in ``docs/problem-definition.md`` before any results were seen. Tuning against
one metric and reporting another is a quiet way to overfit your own judgement.

What gets logged
----------------
Every run records its parameters, cross-validated and validation metrics, the
full search history as an artifact, and the fitted pipeline itself — which
includes preprocessing, so the artifact is directly servable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from churn_guard.config import PROJECT_ROOT, ConfigSection, load_config
from churn_guard.exception import ModelTrainingError
from churn_guard.features.build import build_feature_pipeline, load_xy
from churn_guard.logger import get_logger
from churn_guard.models.baselines import BusinessRuleClassifier
from churn_guard.models.metrics import Costs, Metrics, evaluate, format_table

logger = get_logger(__name__)


# Complexity ordering, lowest first. Used by the one-standard-error rule to
# break ties: a linear model is cheaper to serve, easier to explain to a
# regulator, and has far fewer ways to fail than an ensemble of 200 trees.
COMPLEXITY_RANK = {
    "logistic_regression": 1,
    "hist_gradient_boosting": 2,
    "random_forest": 3,
}


def select_best_model(
    searches: dict[str, RandomizedSearchCV], n_folds: int
) -> tuple[str, str]:
    """Choose a model with the one-standard-error rule.

    Taking the highest cross-validated score at face value treats differences
    smaller than the noise as real. Here the three families land within 0.002 of
    each other while fold-to-fold standard deviation is around 0.022 — the
    ranking would reshuffle under a different seed.

    The rule: find the best mean score, compute the standard error of that
    estimate (``std / sqrt(n_folds)``), and among every model within one
    standard error of the best, keep the simplest.

    Returns:
        The selected model name and a human-readable justification.
    """
    scores = {name: float(s.best_score_) for name, s in searches.items()}
    top_name = max(scores, key=scores.__getitem__)
    top_search = searches[top_name]
    top_std = float(top_search.cv_results_["std_test_score"][int(top_search.best_index_)])

    standard_error = top_std / np.sqrt(n_folds)
    cutoff = scores[top_name] - standard_error

    within_noise = [name for name, score in scores.items() if score >= cutoff]
    selected = min(within_noise, key=lambda n: COMPLEXITY_RANK.get(n, 99))

    if selected == top_name:
        reason = f"highest CV score ({scores[top_name]:.4f}) and no simpler model ties it"
    else:
        reason = (
            f"{top_name} scored highest ({scores[top_name]:.4f}) but "
            f"{selected} ({scores[selected]:.4f}) is within one standard error "
            f"({standard_error:.4f}) and is simpler"
        )
    return selected, reason


def model_specs(seed: int) -> dict[str, tuple[BaseEstimator, dict[str, Any]]]:
    """Candidate estimators and the distributions to sample their params from.

    Parameter names are prefixed ``model__`` because the estimator is the step
    named ``model`` inside the pipeline. Tuning the whole pipeline rather than a
    bare estimator is what keeps preprocessing fitted inside each CV fold — tune
    a bare model on pre-transformed data and every fold's validation slice has
    already seen the scaler.

    Distributions, not lists: ``loguniform(0.001, 100)`` samples across orders of
    magnitude, which is how regularisation strength actually behaves. A list
    would only ever try the values you happened to guess.
    """
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, random_state=seed),
            {
                # Regularisation strength spans several orders of magnitude, so
                # sample log-uniformly rather than linearly.
                "model__C": loguniform(1e-3, 1e2),
                "model__class_weight": ["balanced", None],
                "model__solver": ["lbfgs", "liblinear"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {
                "model__n_estimators": randint(200, 800),
                "model__max_depth": [None, 6, 10, 16, 24],
                # The main defence against overfitting in a forest: larger leaves
                # mean each prediction is an average over more customers.
                "model__min_samples_leaf": randint(1, 40),
                "model__max_features": ["sqrt", "log2", 0.3, 0.5],
                "model__class_weight": ["balanced", "balanced_subsample", None],
            },
        ),
        "hist_gradient_boosting": (
            # scikit-learn's histogram-based booster — the same algorithm family
            # as LightGBM/XGBoost, without the extra dependency.
            HistGradientBoostingClassifier(random_state=seed, early_stopping=False),
            {
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__max_iter": randint(100, 500),
                "model__max_leaf_nodes": randint(8, 64),
                "model__min_samples_leaf": randint(5, 60),
                "model__l2_regularization": loguniform(1e-4, 10),
                "model__class_weight": ["balanced", None],
            },
        ),
    }


def tune_model(
    name: str,
    estimator: BaseEstimator,
    param_distributions: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: ConfigSection,
) -> RandomizedSearchCV:
    """Randomized hyperparameter search over the full pipeline."""
    pipeline = Pipeline(
        [("features", build_feature_pipeline()), ("model", estimator)]
    )

    folds = StratifiedKFold(
        n_splits=int(cfg.cross_validation.n_splits),
        shuffle=bool(cfg.cross_validation.shuffle),
        random_state=int(cfg.project.random_seed),
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=int(cfg.tuning.n_iter),
        scoring=str(cfg.evaluation.primary_metric),
        cv=folds,
        n_jobs=int(cfg.tuning.n_jobs),
        random_state=int(cfg.project.random_seed),
        # Keep training scores so we can see the train/validation gap, which is
        # what tells us whether a configuration is overfitting.
        return_train_score=True,
        refit=True,
        verbose=int(cfg.tuning.verbose),
    )

    logger.info("Tuning %s — %d candidates x %d folds...",
                name, int(cfg.tuning.n_iter), folds.get_n_splits())
    try:
        search.fit(X_train, y_train)
    except Exception as err:
        raise ModelTrainingError(f"Hyperparameter search failed for {name}.") from err

    best_index = int(search.best_index_)
    train_score = float(search.cv_results_["mean_train_score"][best_index])
    logger.info(
        "%-24s CV PR-AUC = %.4f +/- %.4f  (train %.4f, gap %+.4f)",
        name,
        search.best_score_,
        search.cv_results_["std_test_score"][best_index],
        train_score,
        train_score - search.best_score_,
    )
    return search


def log_to_mlflow(
    name: str,
    search: RandomizedSearchCV,
    metrics: Metrics,
    X_val: pd.DataFrame,
    cfg: ConfigSection,
) -> str:
    """Record one model's run: params, metrics, search history, artifact.

    Returns:
        The run id, so the selected model can later be promoted to the registry.
    """
    best_index = int(search.best_index_)

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags(
            {
                "model_family": name,
                "primary_metric": str(cfg.evaluation.primary_metric),
                "seed": str(cfg.project.random_seed),
            }
        )

        # Strip the pipeline prefix so the UI shows "C" rather than "model__C".
        mlflow.log_params(
            {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        )
        mlflow.log_param("n_iter", int(cfg.tuning.n_iter))
        mlflow.log_param("cv_folds", int(cfg.cross_validation.n_splits))

        mlflow.log_metrics(
            {
                "cv_pr_auc": float(search.best_score_),
                "cv_pr_auc_std": float(search.cv_results_["std_test_score"][best_index]),
                "cv_train_pr_auc": float(search.cv_results_["mean_train_score"][best_index]),
                "val_pr_auc": metrics.pr_auc,
                "val_roc_auc": metrics.roc_auc,
                "val_brier": metrics.brier,
                "val_precision": metrics.precision,
                "val_recall": metrics.recall,
                "val_f1": metrics.f1,
                "val_precision_at_k": metrics.precision_at_k,
                "val_value_per_1000": metrics.value_per_1000,
            }
        )

        # The full search history, so a later question like "did we ever try
        # max_depth=6?" has an answer.
        history = pd.DataFrame(search.cv_results_)
        history_path = Path(cfg.paths.reports) / f"cv_results_{name}.csv"
        history.to_csv(history_path, index=False)
        mlflow.log_artifact(str(history_path))

        # Logging the pipeline, not the bare estimator, means the artifact
        # accepts raw customer records. The signature records the expected
        # input schema so a malformed payload is rejected at serving time.
        #
        # Integer columns are declared as float64 in the signature. MLflow's
        # schema enforcement rejects a double where it expected a long, and a
        # JSON payload carrying `"tenure": 12.0` is entirely normal — declaring
        # the wider type accepts both.
        signature_example = X_val.copy()
        integer_columns = signature_example.select_dtypes(include="integer").columns
        signature_example[integer_columns] = signature_example[integer_columns].astype(
            "float64"
        )
        signature = mlflow.models.infer_signature(
            signature_example, search.best_estimator_.predict_proba(X_val)
        )
        mlflow.sklearn.log_model(
            search.best_estimator_,
            name="model",
            signature=signature,
            input_example=signature_example.head(3),
            # MLflow 3 serialises with skops rather than pickle. skops refuses
            # to reconstruct any type not on this list, which is what stops a
            # malicious model file from executing code on load. Our own
            # transformer has to be declared explicitly.
            skops_trusted_types=[
                "churn_guard.features.build.DomainFeatureBuilder",
                "numpy.dtype",
            ],
        )
        return run.info.run_id


def setup_mlflow(cfg: ConfigSection) -> None:
    """Point MLflow at the project's SQLite store and experiment.

    Absolute paths throughout, so runs land in the same database whether the
    script is launched from the project root, a notebook, or a container.
    """
    database = PROJECT_ROOT / str(cfg.mlflow.tracking_db)
    mlflow.set_tracking_uri(f"sqlite:///{database}")

    artifacts = PROJECT_ROOT / str(cfg.mlflow.artifact_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    name = str(cfg.mlflow.experiment_name)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(name)
    logger.info("MLflow tracking to sqlite:///%s", database.name)


def main() -> None:
    cfg = load_config()
    costs = Costs.from_config(cfg)
    seed = int(cfg.project.random_seed)

    setup_mlflow(cfg)

    X_train, y_train = load_xy("train", cfg)
    X_val, y_val = load_xy("val", cfg)
    population = len(X_train) + 2 * len(X_val)

    print(f"\n{'=' * 92}")
    print("  MODEL SELECTION")
    print(f"{'=' * 92}")
    print(f"  Tuning metric : {cfg.evaluation.primary_metric} (PR-AUC)")
    print(f"  Search        : randomized, {cfg.tuning.n_iter} candidates x "
          f"{cfg.cross_validation.n_splits} folds")
    print(f"  Threshold     : {costs.optimal_threshold:.4f} (cost-derived)\n")

    rows: list[Metrics] = []
    searches: dict[str, RandomizedSearchCV] = {}
    run_ids: dict[str, str] = {}

    # The business rule is carried through as the bar to beat, not as a
    # candidate — it has nothing to tune.
    rule = BusinessRuleClassifier().fit(X_train, y_train)
    rows.append(
        evaluate(
            "B1 business rule",
            y_val.to_numpy(),
            rule.predict_proba(X_val)[:, 1],
            costs,
            total_population=population,
        )
    )

    for name, (estimator, distributions) in model_specs(seed).items():
        search = tune_model(name, estimator, distributions, X_train, y_train, cfg)
        searches[name] = search

        y_prob = search.best_estimator_.predict_proba(X_val)[:, 1]
        metrics = evaluate(
            name, y_val.to_numpy(), y_prob, costs, total_population=population
        )
        rows.append(metrics)
        run_ids[name] = log_to_mlflow(name, search, metrics, X_val, cfg)

    print(f"\n  Held-out validation set ({len(y_val):,} customers):\n")
    print("  " + format_table(rows).replace("\n", "\n  "))

    # Selection is on cross-validated score rather than the single validation
    # number — CV averages over five folds and is the less noisy estimate — and
    # applies the one-standard-error rule so a difference smaller than the noise
    # does not buy us a more complex model. The validation set is the sanity
    # check that CV was not itself misleading.
    best_name, reason = select_best_model(
        searches, int(cfg.cross_validation.n_splits)
    )
    best_search = searches[best_name]
    best_metrics = next(r for r in rows if r.model == best_name)

    print(f"\n  Selected: {best_name}  (CV PR-AUC {best_search.best_score_:.4f})")
    print(f"  Reason  : {reason}")
    print("  Best parameters:")
    for key, value in sorted(best_search.best_params_.items()):
        print(f"    {key.replace('model__', ''):<24} {value}")

    model_path = Path(cfg.paths.models) / "model.joblib"
    joblib.dump(best_search.best_estimator_, model_path)

    # Save a small transformed sample beside the model for SHAP to use as its
    # background distribution at serving time. Without this the API would have
    # to load data/interim/train.csv, which means shipping the training set
    # inside the production image and breaking if that file ever moves. A
    # serving artifact should depend only on the model directory.
    background = (
        best_search.best_estimator_
        .named_steps["features"]
        .transform(X_train.sample(n=min(200, len(X_train)), random_state=seed))
    )
    background_path = Path(cfg.paths.models) / "shap_background.parquet"
    background.to_parquet(background_path, index=False)
    logger.info("Saved SHAP background sample (%d rows) to %s",
                len(background), background_path.name)

    # Promote the winner to the model registry. The registry is what separates
    # "a file on someone's laptop" from a versioned artifact the serving layer
    # can resolve by name and stage (Staging -> Production -> Archived).
    registered = mlflow.register_model(
        model_uri=f"runs:/{run_ids[best_name]}/model",
        name=str(cfg.mlflow.registered_model_name),
    )
    logger.info(
        "Registered %s version %s", registered.name, registered.version
    )

    metadata = {
        "model_family": best_name,
        "selection_reason": reason,
        "mlflow_run_id": run_ids[best_name],
        "registered_model_version": registered.version,
        "best_params": {
            k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
            for k, v in best_search.best_params_.items()
        },
        "cv_pr_auc": float(best_search.best_score_),
        "validation": best_metrics.as_dict(),
        "decision_threshold": costs.optimal_threshold,
        "random_seed": seed,
        "sklearn_feature_count": int(
            best_search.best_estimator_.named_steps["features"]
            .transform(X_val.head(1))
            .shape[1]
        ),
    }
    metadata_path = Path(cfg.paths.models) / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Saved model to %s", model_path)
    logger.info("Saved metadata to %s", metadata_path)
    logger.info(
        "Inspect runs with: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db"
    )


if __name__ == "__main__":
    main()
