"""
FastAPI service for fraud detection inference.
Features: JWT Auth, Rate Limiting, SHAP Explainability, Prometheus Metrics, Drift Detection
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from auth import (
    Token,
    User,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    require_scope,
)
from drift import DriftDetector, get_drift_detector
from metrics import (
    AUTH_EVENTS,
    DRIFT_SCORE,
    PREDICTION_LATENCY,
    MetricsMiddleware,
    get_metrics,
    record_prediction,
    record_transaction_amount,
    set_model_info,
)
from predict import FraudDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# ===================
# Pydantic Models
# ===================
class LoginRequest(BaseModel):
    username: str
    password: str


class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    V1: float = Field(..., description="PCA component V1")
    V2: float = Field(..., description="PCA component V2")
    V3: float = Field(..., description="PCA component V3")
    V4: float = Field(..., description="PCA component V4")
    V5: float = Field(..., description="PCA component V5")
    V6: float = Field(..., description="PCA component V6")
    V7: float = Field(..., description="PCA component V7")
    V8: float = Field(..., description="PCA component V8")
    V9: float = Field(..., description="PCA component V9")
    V10: float = Field(..., description="PCA component V10")
    V11: float = Field(..., description="PCA component V11")
    V12: float = Field(..., description="PCA component V12")
    V13: float = Field(..., description="PCA component V13")
    V14: float = Field(..., description="PCA component V14")
    V15: float = Field(..., description="PCA component V15")
    V16: float = Field(..., description="PCA component V16")
    V17: float = Field(..., description="PCA component V17")
    V18: float = Field(..., description="PCA component V18")
    V19: float = Field(..., description="PCA component V19")
    V20: float = Field(..., description="PCA component V20")
    V21: float = Field(..., description="PCA component V21")
    V22: float = Field(..., description="PCA component V22")
    V23: float = Field(..., description="PCA component V23")
    V24: float = Field(..., description="PCA component V24")
    V25: float = Field(..., description="PCA component V25")
    V26: float = Field(..., description="PCA component V26")
    V27: float = Field(..., description="PCA component V27")
    V28: float = Field(..., description="PCA component V28")
    Amount: float = Field(..., description="Transaction amount in dollars")


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    threshold: float
    latency_ms: float


class ExplanationFeature(BaseModel):
    feature: str
    contribution: float
    value: Optional[float] = None
    direction: str


class ExplainedPredictionResponse(BaseModel):
    prediction: PredictionResponse
    explanation: dict


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_transactions: int
    flagged_count: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    threshold: float
    training_metrics: dict
    feature_importance: dict


# ===================
# Application Setup
# ===================
detector: Optional[FraudDetector] = None
drift_detector: Optional[DriftDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global detector, drift_detector

    # Startup
    logger.info("Loading fraud detection model...")
    try:
        detector = FraudDetector("models/fraud_model.joblib")
        logger.info("Model loaded successfully")

        # Set model info for metrics
        set_model_info(
            model_name="fraud_xgboost",
            version="1.0.0",
            threshold=detector.threshold,
            metrics=detector.metrics,
        )

        # Initialize drift detector
        drift_detector = get_drift_detector()
        drift_detector.feature_columns = detector.feature_cols

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        detector = None

    yield

    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Fraud Detection API",
    description="""
    Real-time credit card fraud detection using XGBoost.

    ## Features
    - **Real-time predictions** with <100ms latency
    - **SHAP explainability** for regulatory compliance
    - **JWT authentication** for secure access
    - **Rate limiting** to prevent abuse
    - **Prometheus metrics** for monitoring
    - **Drift detection** for model health

    ## Authentication
    Use `/auth/login` to get a JWT token, then include it in the Authorization header:
    `Authorization: Bearer <token>`

    ## Demo Credentials
    - admin / admin123 (full access)
    - analyst / analyst123 (read only)
    - api_user / api123 (read + write)
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Add middlewares
app.add_middleware(MetricsMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Rate limit error handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return Response(
        content='{"detail": "Rate limit exceeded. Please slow down."}',
        status_code=429,
        media_type="application/json",
    )


# ===================
# Auth Endpoints
# ===================
@app.post("/auth/login", response_model=Token, tags=["Authentication"])
@limiter.limit("10/minute")
async def login(request: Request, login_req: LoginRequest):
    """
    Authenticate and get JWT tokens.

    Demo credentials:
    - admin / admin123
    - analyst / analyst123
    - api_user / api123
    """
    user = authenticate_user(login_req.username, login_req.password)
    if not user:
        AUTH_EVENTS.labels(event_type="login", success="false").inc()
        raise HTTPException(status_code=401, detail="Invalid credentials")

    AUTH_EVENTS.labels(event_type="login", success="true").inc()

    access_token = create_access_token(data={"sub": user.username, "scopes": user.scopes})
    refresh_token = create_refresh_token(data={"sub": user.username, "scopes": user.scopes})

    return Token(access_token=access_token, refresh_token=refresh_token)


@app.post("/auth/refresh", response_model=Token, tags=["Authentication"])
async def refresh_token(refresh_token: str):
    """Get new access token using refresh token."""
    try:
        token_data = decode_token(refresh_token)
        access_token = create_access_token(
            data={"sub": token_data.username, "scopes": token_data.scopes}
        )
        new_refresh = create_refresh_token(
            data={"sub": token_data.username, "scopes": token_data.scopes}
        )
        return Token(access_token=access_token, refresh_token=new_refresh)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid refresh token")


@app.get("/auth/me", tags=["Authentication"])
async def get_current_user_info(user: User = Depends(get_current_user)):
    """Get current user information."""
    return {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "scopes": user.scopes,
    }


# ===================
# Health & Metrics
# ===================
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if detector else "unhealthy", model_loaded=detector is not None
    )


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    content, content_type = get_metrics()
    return Response(content=content, media_type=content_type)


@app.get("/model/info", response_model=MetricsResponse, tags=["Model"])
async def get_model_info(user: User = Depends(require_scope("read"))):
    """Get model information and training metrics."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return MetricsResponse(
        threshold=detector.threshold,
        training_metrics=detector.metrics,
        feature_importance=detector.get_feature_importance(),
    )


# ===================
# Prediction Endpoints
# ===================
@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
@limiter.limit("100/minute")
async def predict_single(
    request: Request, transaction: Transaction, user: User = Depends(require_scope("read"))
):
    """
    Predict fraud probability for a single transaction.

    Returns:
    - fraud_probability: 0-1 score
    - is_fraud: boolean based on threshold
    - risk_level: LOW/MEDIUM/HIGH
    - latency_ms: processing time
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    with PREDICTION_LATENCY.time():
        result = detector.predict(transaction.model_dump())
    latency = (time.time() - start) * 1000

    # Record metrics
    record_prediction(result)
    record_transaction_amount(transaction.Amount)

    # Add to drift detector
    if drift_detector:
        drift_detector.add_sample(transaction.model_dump())

    return PredictionResponse(
        fraud_probability=result["fraud_probability"],
        is_fraud=result["is_fraud"],
        risk_level=result["risk_level"],
        threshold=result["threshold"],
        latency_ms=round(latency, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
@limiter.limit("20/minute")
async def predict_batch(
    request: Request, transactions: List[Transaction], user: User = Depends(require_scope("write"))
):
    """
    Batch prediction for multiple transactions.
    Limited to 100 transactions per request.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 transactions per batch")

    start = time.time()
    results = detector.predict_batch([t.model_dump() for t in transactions])
    latency = (time.time() - start) * 1000

    # Record metrics
    for result, txn in zip(results, transactions):
        record_prediction(result)
        record_transaction_amount(txn.Amount)

    predictions = [
        PredictionResponse(
            fraud_probability=r["fraud_probability"],
            is_fraud=r["is_fraud"],
            risk_level=r["risk_level"],
            threshold=r["threshold"],
            latency_ms=0,
        )
        for r in results
    ]

    return BatchPredictionResponse(
        predictions=predictions,
        total_transactions=len(transactions),
        flagged_count=sum(1 for r in results if r["is_fraud"]),
        latency_ms=round(latency, 2),
    )


@app.post("/predict/explain", response_model=ExplainedPredictionResponse, tags=["Predictions"])
@limiter.limit("30/minute")
async def predict_with_explanation(
    request: Request,
    transaction: Transaction,
    top_k: int = 10,
    user: User = Depends(require_scope("read")),
):
    """
    Predict with SHAP explanation.

    Returns prediction plus top contributing features explaining why
    the transaction was flagged or approved.

    Use this for:
    - Regulatory compliance (explainable AI)
    - Customer disputes
    - Analyst review
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    result = detector.explain(transaction.model_dump(), top_k=top_k)
    latency = (time.time() - start) * 1000

    # Record metrics
    record_prediction(result["prediction"])

    prediction_response = PredictionResponse(
        fraud_probability=result["prediction"]["fraud_probability"],
        is_fraud=result["prediction"]["is_fraud"],
        risk_level=result["prediction"]["risk_level"],
        threshold=result["prediction"]["threshold"],
        latency_ms=round(latency, 2),
    )

    return ExplainedPredictionResponse(
        prediction=prediction_response, explanation=result["explanation"]
    )


# ===================
# Drift Detection
# ===================
@app.get("/drift/status", tags=["Monitoring"])
async def get_drift_status(user: User = Depends(require_scope("read"))):
    """
    Get current feature drift status.

    Monitors distribution shift between training and production data.
    High drift scores may indicate model degradation.
    """
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")

    report = drift_detector.detect_drift(min_samples=50)

    if report is None:
        return {
            "status": "insufficient_data",
            "message": f"Need more samples. Current: {len(drift_detector.production_buffer)}",
            "min_required": 50,
        }

    # Update drift metrics
    for feature_result in report.feature_results:
        DRIFT_SCORE.labels(feature=feature_result.feature).set(feature_result.drift_score)

    return report.to_dict()


@app.get("/drift/stats", tags=["Monitoring"])
async def get_production_stats(user: User = Depends(require_scope("read"))):
    """Get statistics of recent production data."""
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")

    return {
        "sample_count": len(drift_detector.production_buffer),
        "window_size": drift_detector.window_size,
        "feature_stats": drift_detector.get_feature_stats(),
    }


# ===================
# Run Server
# ===================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104 - intended for Docker
