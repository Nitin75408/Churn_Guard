"""Request and response contracts for the prediction API.

These Pydantic models are the boundary between the outside world and the model.
Anything that fails validation here is rejected with a 422 and never reaches the
pipeline — a malformed request should produce a precise error message, not a
confident prediction from garbage input.

Categorical fields are typed as ``Literal`` rather than ``str``. That does three
useful things at once: it rejects unknown values, it documents the permitted set
in the auto-generated OpenAPI schema, and it turns the ``/docs`` page into
dropdowns instead of free text.

Note the interaction with ``handle_unknown="ignore"`` on the encoder. That
setting exists so an unseen category degrades gracefully rather than crashing.
These ``Literal`` types mean it should never be needed in practice — defence in
depth, with the strict check at the edge and the soft fallback deeper in.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
YesNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    """One customer's raw attributes, exactly as they appear in the source data.

    Deliberately mirrors the raw schema rather than the engineered features. The
    caller should not have to know that ``num_addon_services`` exists — that is
    the pipeline's job, and keeping the contract raw means feature engineering
    can change without breaking every client.
    """

    model_config = ConfigDict(
        # Reject unknown fields. A typo like "Tenure" would otherwise be silently
        # dropped and the model would score a customer with a default tenure.
        extra="forbid",
        json_schema_extra={
            "example": {
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
        },
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(description="1 if the customer is 65 or older")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=200, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoInternet
    OnlineBackup: YesNoInternet
    DeviceProtection: YesNoInternet
    TechSupport: YesNoInternet
    StreamingTV: YesNoInternet
    StreamingMovies: YesNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100000)


class Driver(BaseModel):
    """One feature's contribution to this prediction."""

    feature: str
    contribution: float = Field(
        description="SHAP value in log-odds. Positive raises churn risk."
    )
    direction: Literal["increases risk", "reduces risk"]


class PredictionResponse(BaseModel):
    """Risk score plus everything needed to act on it."""

    churn_probability: float = Field(ge=0, le=1)
    risk_tier: Literal["high", "medium", "low"]
    # The recommendation, not just the score. A probability alone leaves the
    # caller to reinvent the cost logic; this applies the threshold the
    # economics imply, so every consumer acts consistently.
    should_contact: bool
    decision_threshold: float
    expected_value_of_contact: float = Field(
        description="Expected $ gain from contacting this customer, net of the offer cost"
    )
    top_drivers: list[Driver]


class BatchPredictionRequest(BaseModel):
    """Several customers in one call.

    Batching matters operationally: the retention team scores its whole base
    nightly, and one request of 1,000 customers is far cheaper than 1,000
    requests.
    """

    customers: list[CustomerFeatures] = Field(min_length=1, max_length=1000)


class RankedPrediction(PredictionResponse):
    """A batch prediction with its position in the risk ranking."""

    rank: int = Field(ge=1, description="1 = highest risk in this batch")
    # Because the batch response is re-sorted by risk, the caller otherwise has
    # no way to tell which of its customers a row refers to. Carrying the
    # original position back makes the response joinable to the request.
    input_index: int = Field(
        ge=0, description="Zero-based position of this customer in the request"
    )


class BatchPredictionResponse(BaseModel):
    """Batch results, sorted by risk.

    Sorted because capacity binds: the team can only contact a few hundred
    people, so the ordering is the actionable output, not the raw scores.
    """

    count: int
    contact_recommended: int
    predictions: list[RankedPrediction]


class HealthResponse(BaseModel):
    """Liveness and readiness signal for deployment tooling."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_family: str | None = None
    model_version: str | None = None
    explanations_available: bool


class ModelInfoResponse(BaseModel):
    """Provenance: what is running, and how good it was measured to be."""

    model_config = ConfigDict(protected_namespaces=())

    model_family: str
    registered_version: str | None
    mlflow_run_id: str | None
    decision_threshold: float
    trained_features: int
    test_metrics: dict[str, float]
