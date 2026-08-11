"""FastAPI application serving churn predictions.

Run it::

    uv run uvicorn churn_guard.api.main:app --reload

Then open http://127.0.0.1:8000/docs for an interactive console generated from
the type hints — no separate API documentation to write or keep in sync.

Endpoints
---------
``GET  /health``        liveness and readiness, for deployment tooling
``GET  /model/info``    what model is running and how it scored
``POST /predict``       score one customer
``POST /predict/batch`` score up to 1,000, returned ranked by risk
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from churn_guard.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from churn_guard.api.service import ChurnService
from churn_guard.exception import ChurnGuardError, ModelNotFoundError
from churn_guard.logger import get_logger

logger = get_logger(__name__)

service = ChurnService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, not per request.

    A failure here is deliberately allowed to be non-fatal: the process still
    starts and ``/health`` reports ``degraded``. An orchestrator can then see a
    clear unhealthy signal rather than a crash loop that tells it nothing.
    """
    try:
        service.load()
        logger.info("API ready.")
    except ModelNotFoundError as err:
        logger.error("Startup without a model: %s", err)
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="ChurnGuard API",
    description=(
        "Customer churn risk scoring with cost-based decisioning and per-customer "
        "SHAP explanations. Serves the exact pipeline produced by training, so "
        "preprocessing cannot drift between training and serving."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status and latency for every request.

    Latency per endpoint is the first thing you want when a service is reported
    slow, and it has to be collected before you need it.
    """
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
    return response


@app.exception_handler(ChurnGuardError)
async def churn_guard_exception_handler(request: Request, exc: ChurnGuardError):
    """Return our own errors as clean JSON instead of a 500 with a stack trace.

    Internal tracebacks leak implementation detail to callers and are useless to
    them; the detail belongs in the logs, which is where it goes.
    """
    logger.error("%s on %s: %s", type(exc).__name__, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


def _require_model() -> None:
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run the training pipeline, then restart the API.",
        )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Readiness probe.

    Reports ``degraded`` rather than failing when the model is missing, so a
    load balancer can route traffic away while the process stays inspectable.
    """
    return HealthResponse(
        status="ok" if service.is_ready else "degraded",
        model_loaded=service.is_ready,
        model_family=service.metadata.get("model_family"),
        model_version=str(service.metadata.get("registered_model_version") or "") or None,
        explanations_available=service.explanations_available,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
async def model_info() -> ModelInfoResponse:
    """Provenance of the running model, for auditing and debugging."""
    _require_model()
    # Sealed-test metrics, not validation. The validation set was used to choose
    # the model and the threshold, so its scores are optimistic; the test set was
    # opened exactly once and is the number worth publishing.
    reported = service.test_metrics or service.metadata.get("validation", {})
    return ModelInfoResponse(
        model_family=service.metadata.get("model_family", "unknown"),
        registered_version=str(service.metadata.get("registered_model_version") or "") or None,
        mlflow_run_id=service.metadata.get("mlflow_run_id"),
        decision_threshold=service.threshold,
        trained_features=int(service.metadata.get("sklearn_feature_count", 0)),
        test_metrics={
            key: float(value)
            for key, value in reported.items()
            if isinstance(value, (int, float))
        },
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Score a single customer.

    The request has already been validated by Pydantic, so anything reaching
    this function has the right fields with permitted values.
    """
    _require_model()
    frame = pd.DataFrame([customer.model_dump()])
    return PredictionResponse(**service.score(frame)[0])


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score up to 1,000 customers, returned ranked by risk.

    Ranking is the point. The retention team has capacity for a few hundred
    calls, so what they need is an ordered worklist, not a pile of scores.
    """
    _require_model()
    frame = pd.DataFrame([c.model_dump() for c in request.customers])
    scored = service.score(frame)

    ranked = sorted(scored, key=lambda r: r["churn_probability"], reverse=True)
    predictions = [{**row, "rank": i + 1} for i, row in enumerate(ranked)]

    return BatchPredictionResponse(
        count=len(predictions),
        contact_recommended=sum(1 for r in predictions if r["should_contact"]),
        predictions=predictions,
    )
