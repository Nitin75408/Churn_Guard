"""API contract tests.

Run against FastAPI's ``TestClient``, which exercises the real application —
routing, validation, serialisation — without binding a port.

Tests needing a trained model are skipped when ``models/model.joblib`` is
absent, so the suite still runs on a fresh clone. CI trains before testing, so
they do execute there.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn_guard.api.main import app

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.7,
    "TotalCharges": 151.65,
}

LOYAL_CUSTOMER = {
    **VALID_CUSTOMER,
    "tenure": 70,
    "Contract": "Two year",
    "InternetService": "DSL",
    "PaymentMethod": "Credit card (automatic)",
    "TechSupport": "Yes",
    "OnlineSecurity": "Yes",
    "TotalCharges": 4949.0,
}


@pytest.fixture(scope="module")
def client():
    # The context manager form triggers lifespan, which loads the model.
    with TestClient(app) as test_client:
        yield test_client


needs_model = pytest.mark.skipif(
    not (__import__("churn_guard.config", fromlist=["load_config"]).load_config().paths.models
         / "model.joblib").is_file(),
    reason="no trained model — run churn_guard.models.train first",
)


class TestValidation:
    """Rejection happens at the boundary, before the model sees anything."""

    def test_rejects_wrong_type(self, client):
        payload = {**VALID_CUSTOMER, "tenure": "two"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert any(e["loc"][-1] == "tenure" for e in response.json()["detail"])

    def test_rejects_value_outside_the_permitted_set(self, client):
        payload = {**VALID_CUSTOMER, "InternetService": "Fibre"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_rejects_unknown_field(self, client):
        """extra="forbid" catches typos that would otherwise be dropped silently,
        leaving the model to score a customer with a default value."""
        response = client.post("/predict", json={**VALID_CUSTOMER, "Tenure": 5})
        assert response.status_code == 422

    def test_rejects_missing_field(self, client):
        payload = {k: v for k, v in VALID_CUSTOMER.items() if k != "Contract"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_rejects_negative_tenure(self, client):
        response = client.post("/predict", json={**VALID_CUSTOMER, "tenure": -5})
        assert response.status_code == 422

    def test_rejects_empty_batch(self, client):
        assert client.post("/predict/batch", json={"customers": []}).status_code == 422


class TestHealth:
    def test_health_always_responds(self, client):
        """Never 500s, even without a model — an orchestrator needs a signal it
        can act on, not a crash."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] in {"ok", "degraded"}


@needs_model
class TestPrediction:
    def test_returns_a_wellformed_prediction(self, client):
        body = client.post("/predict", json=VALID_CUSTOMER).json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["risk_tier"] in {"high", "medium", "low"}
        assert isinstance(body["should_contact"], bool)

    def test_high_risk_profile_scores_above_low_risk_profile(self, client):
        risky = client.post("/predict", json=VALID_CUSTOMER).json()
        loyal = client.post("/predict", json=LOYAL_CUSTOMER).json()
        assert risky["churn_probability"] > loyal["churn_probability"]

    def test_contact_decision_follows_the_threshold(self, client):
        body = client.post("/predict", json=VALID_CUSTOMER).json()
        assert body["should_contact"] == (
            body["churn_probability"] >= body["decision_threshold"]
        )

    def test_threshold_is_not_the_naive_default(self, client):
        body = client.post("/predict", json=VALID_CUSTOMER).json()
        assert body["decision_threshold"] != 0.5

    def test_expected_value_is_positive_when_contact_is_advised(self, client):
        body = client.post("/predict", json=VALID_CUSTOMER).json()
        if body["should_contact"]:
            assert body["expected_value_of_contact"] > 0

    def test_explanations_are_returned(self, client):
        drivers = client.post("/predict", json=VALID_CUSTOMER).json()["top_drivers"]
        assert drivers
        for driver in drivers:
            assert driver["direction"] in {"increases risk", "reduces risk"}
            assert (driver["contribution"] > 0) == (driver["direction"] == "increases risk")

    def test_identical_requests_give_identical_answers(self, client):
        first = client.post("/predict", json=VALID_CUSTOMER).json()
        second = client.post("/predict", json=VALID_CUSTOMER).json()
        assert first["churn_probability"] == second["churn_probability"]


@needs_model
class TestBatch:
    def test_results_are_ranked_by_descending_risk(self, client):
        payload = {"customers": [LOYAL_CUSTOMER, VALID_CUSTOMER, LOYAL_CUSTOMER]}
        body = client.post("/predict/batch", json=payload).json()
        probabilities = [p["churn_probability"] for p in body["predictions"]]
        assert probabilities == sorted(probabilities, reverse=True)
        assert [p["rank"] for p in body["predictions"]] == [1, 2, 3]

    def test_input_index_maps_results_back_to_the_request(self, client):
        """Regression test: the batch response is re-sorted, so without this the
        caller cannot tell which customer a row refers to."""
        payload = {"customers": [LOYAL_CUSTOMER, VALID_CUSTOMER]}
        body = client.post("/predict/batch", json=payload).json()
        top = body["predictions"][0]
        assert top["input_index"] == 1  # the risky customer was second in
        assert sorted(p["input_index"] for p in body["predictions"]) == [0, 1]

    def test_contact_count_matches_the_flags(self, client):
        payload = {"customers": [VALID_CUSTOMER, LOYAL_CUSTOMER]}
        body = client.post("/predict/batch", json=payload).json()
        assert body["contact_recommended"] == sum(
            1 for p in body["predictions"] if p["should_contact"]
        )

    def test_batch_agrees_with_single_prediction(self, client):
        """Batching must not change the answer."""
        single = client.post("/predict", json=VALID_CUSTOMER).json()
        batch = client.post("/predict/batch", json={"customers": [VALID_CUSTOMER]}).json()
        assert batch["predictions"][0]["churn_probability"] == single["churn_probability"]


@needs_model
class TestModelInfo:
    def test_reports_provenance(self, client):
        body = client.get("/model/info").json()
        assert body["model_family"]
        assert body["trained_features"] == 35

    def test_reports_sealed_test_metrics_not_validation(self, client):
        """The published number must be the one measured once on held-out data."""
        import json
        from pathlib import Path

        from churn_guard.config import load_config

        cfg = load_config()
        results = json.loads(
            (Path(cfg.paths.reports) / "final_evaluation.json").read_text()
        )
        body = client.get("/model/info").json()
        assert body["test_metrics"]["pr_auc"] == pytest.approx(
            results["test"]["pr_auc"]
        )
