"""Prediction service — model loading, scoring and explanation.

Kept separate from the FastAPI layer in ``main.py`` so the logic can be unit
tested without spinning up a web server, and so the same code could be reused by
a batch scoring job or a Streamlit app without dragging HTTP along with it.

The central guarantee: this module loads the **fitted pipeline** and calls
``predict_proba`` on raw customer records. It reimplements no cleaning, no
encoding, no scaling. Train/serve skew — where the API computes a feature
slightly differently from training and accuracy quietly rots — is impossible by
construction rather than by discipline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from churn_guard.config import ConfigSection, load_config
from churn_guard.exception import ModelNotFoundError, PredictionError
from churn_guard.logger import get_logger
from churn_guard.models.metrics import Costs

logger = get_logger(__name__)

# Risk tiers drive the action, per docs/problem-definition.md section 3: high =
# proactive call plus offer, medium = automated email, low = no action.
HIGH_RISK_MULTIPLIER = 1.5


class ChurnService:
    """Loads the model once and serves predictions.

    Loading is done at startup rather than per request. Deserialising a pipeline
    takes tens of milliseconds — trivial once, but it would dominate latency if
    repeated on every call.
    """

    def __init__(self, cfg: ConfigSection | None = None) -> None:
        self.cfg = cfg or load_config()
        self.costs = Costs.from_config(self.cfg)
        self.pipeline = None
        self.metadata: dict = {}
        self.test_metrics: dict = {}
        self.threshold: float = self.costs.optimal_threshold
        self._explainer = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Load the pipeline, metadata, threshold and SHAP explainer."""
        import joblib

        model_path: Path = self.cfg.paths.models / "model.joblib"
        if not model_path.is_file():
            raise ModelNotFoundError(
                f"No model at {model_path}. Run "
                f"'uv run python -m churn_guard.models.train' before starting the API."
            )
        self.pipeline = joblib.load(model_path)
        logger.info("Loaded model from %s", model_path.name)

        metadata_path: Path = self.cfg.paths.models / "model_metadata.json"
        if metadata_path.is_file():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        # Prefer the threshold chosen empirically during evaluation; fall back to
        # the value the cost model implies if evaluation has not been run.
        # Also pick up the sealed-test metrics, which are the honest performance
        # figures to advertise — model_metadata.json holds validation numbers,
        # and serving those as "test" would overstate the model.
        evaluation_path: Path = Path(self.cfg.paths.reports) / "final_evaluation.json"
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            self.threshold = float(evaluation["operating_threshold"])
            self.test_metrics = evaluation.get("test", {})
        logger.info("Decision threshold: %.4f", self.threshold)

        self._build_explainer()

    def _build_explainer(self) -> None:
        """Prepare a SHAP explainer, if the model family supports a fast one.

        Explanations are best-effort. A retention agent wants to know *why* a
        customer is flagged, but a failure to explain must never take down
        scoring — the prediction is the critical path, the explanation is not.
        """
        try:
            import shap

            from churn_guard.data.split import load_split

            train = load_split("train", self.cfg)
            background = self.pipeline.named_steps["features"].transform(
                train.drop(columns=[self.cfg.data.target_column])
            )
            self._feature_names = list(background.columns)
            self._explainer = shap.LinearExplainer(
                self.pipeline.named_steps["model"], background
            )
            logger.info("SHAP explainer ready (%d features).", len(self._feature_names))
        except Exception as err:  # noqa: BLE001 - explanations are optional
            self._explainer = None
            logger.warning("Explanations unavailable: %s", err)

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None

    @property
    def explanations_available(self) -> bool:
        return self._explainer is not None

    # --------------------------------------------------------------- predict
    def predict_frame(self, customers: pd.DataFrame) -> np.ndarray:
        """Churn probabilities for a dataframe of raw customer records."""
        if self.pipeline is None:
            raise ModelNotFoundError("Model is not loaded.")
        try:
            return self.pipeline.predict_proba(customers)[:, 1]
        except Exception as err:
            raise PredictionError(
                "Scoring failed. The payload validated but the pipeline could not "
                "process it — check that every expected column is present."
            ) from err

    def explain_frame(self, customers: pd.DataFrame) -> list[list[tuple[str, float]]]:
        """Top SHAP drivers per row, largest absolute contribution first."""
        if self._explainer is None:
            return [[] for _ in range(len(customers))]
        try:
            transformed = self.pipeline.named_steps["features"].transform(customers)
            values = self._explainer(transformed).values
            results = []
            for row in values:
                series = pd.Series(row, index=self._feature_names)
                ordered = series.reindex(series.abs().sort_values(ascending=False).index)
                results.append([(k, float(v)) for k, v in ordered.head(5).items()])
            return results
        except Exception as err:  # noqa: BLE001 - never fail a prediction over this
            logger.warning("Explanation failed: %s", err)
            return [[] for _ in range(len(customers))]

    # ----------------------------------------------------------------- logic
    def risk_tier(self, probability: float) -> str:
        """Map a probability to an action tier."""
        if probability >= self.threshold * HIGH_RISK_MULTIPLIER:
            return "high"
        if probability >= self.threshold:
            return "medium"
        return "low"

    def expected_value_of_contact(self, probability: float) -> float:
        """Expected $ gain from contacting this customer.

        ``p x (acceptance_rate x CLV) - offer_cost``. Positive means the contact
        pays for itself in expectation, which is precisely the condition the
        decision threshold encodes.
        """
        return probability * self.costs.acceptance_rate * self.costs.clv - self.costs.offer_cost

    def score(self, customers: pd.DataFrame) -> list[dict]:
        """Full scoring: probability, tier, decision, value and drivers."""
        probabilities = self.predict_frame(customers)
        explanations = self.explain_frame(customers)

        results = []
        for probability, drivers in zip(probabilities, explanations):
            probability = float(probability)
            results.append(
                {
                    "churn_probability": round(probability, 4),
                    "risk_tier": self.risk_tier(probability),
                    "should_contact": bool(probability >= self.threshold),
                    "decision_threshold": round(self.threshold, 4),
                    "expected_value_of_contact": round(
                        self.expected_value_of_contact(probability), 2
                    ),
                    "top_drivers": [
                        {
                            "feature": feature,
                            "contribution": round(contribution, 4),
                            "direction": (
                                "increases risk" if contribution > 0 else "reduces risk"
                            ),
                        }
                        for feature, contribution in drivers
                    ],
                }
            )
        return results
