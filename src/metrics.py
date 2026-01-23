"""
Prometheus metrics for fraud detection API.
"""

import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

# ===================
# Counters
# ===================
PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total", "Total number of fraud predictions made", ["result", "risk_level"]
)

REQUESTS_TOTAL = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

AUTH_EVENTS = Counter("auth_events_total", "Authentication events", ["event_type", "success"])

# ===================
# Histograms
# ===================
PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds",
    "Fraud prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0],
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_PROBABILITY = Histogram(
    "fraud_prediction_probability",
    "Distribution of fraud probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

TRANSACTION_AMOUNT = Histogram(
    "transaction_amount_dollars",
    "Distribution of transaction amounts",
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
)

# ===================
# Gauges
# ===================
MODEL_THRESHOLD = Gauge("fraud_model_threshold", "Current fraud detection threshold")

FRAUD_RATE = Gauge("fraud_detection_rate", "Rolling fraud detection rate (flagged/total)")

DRIFT_SCORE = Gauge(
    "feature_drift_score", "Feature drift score (0=no drift, 1=high drift)", ["feature"]
)

ACTIVE_CONNECTIONS = Gauge("active_connections", "Number of active API connections")

# ===================
# Info
# ===================
MODEL_INFO = Info("fraud_model", "Information about the loaded fraud detection model")


def set_model_info(model_name: str, version: str, threshold: float, metrics: dict):
    """Set model information metrics."""
    MODEL_INFO.info(
        {
            "name": model_name,
            "version": version,
            "threshold": str(threshold),
            "recall": str(metrics.get("recall", 0)),
            "precision": str(metrics.get("precision", 0)),
            "pr_auc": str(metrics.get("pr_auc", 0)),
        }
    )
    MODEL_THRESHOLD.set(threshold)


def record_prediction(result: dict):
    """Record metrics for a prediction."""
    is_fraud = "fraud" if result["is_fraud"] else "legitimate"
    risk_level = result["risk_level"].lower()

    PREDICTIONS_TOTAL.labels(result=is_fraud, risk_level=risk_level).inc()
    PREDICTION_PROBABILITY.observe(result["fraud_probability"])


def record_transaction_amount(amount: float):
    """Record transaction amount."""
    TRANSACTION_AMOUNT.observe(amount)


class MetricsMiddleware:
    """Middleware to track request metrics."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        ACTIVE_CONNECTIONS.inc()
        start_time = time.time()

        # Capture response status
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            ACTIVE_CONNECTIONS.dec()

            # Record metrics
            method = scope.get("method", "UNKNOWN")
            path = scope.get("path", "unknown")
            latency = time.time() - start_time

            # Normalize path for metrics (remove IDs)
            endpoint = self._normalize_path(path)

            REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()

            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)

    def _normalize_path(self, path: str) -> str:
        """Normalize path for metric labels."""
        # Remove trailing slashes
        path = path.rstrip("/")

        # Map common patterns
        if path.startswith("/predict"):
            return "/predict"
        if path.startswith("/auth"):
            return "/auth"

        return path or "/"


def get_metrics():
    """Generate Prometheus metrics output."""
    return generate_latest(), CONTENT_TYPE_LATEST
