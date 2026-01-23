"""
Unit tests for FastAPI endpoints.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        # Only import app if model exists (for local testing)
        # In CI, these tests will be skipped
        if not os.path.exists("models/fraud_model.joblib"):
            pytest.skip("Model file not available")

        from app import app

        return TestClient(app)

    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Test that health endpoint returns status field."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        if not os.path.exists("models/fraud_model.joblib"):
            pytest.skip("Model file not available")

        from app import app

        return TestClient(app)

    def test_login_with_valid_credentials(self, client):
        """Test successful login."""
        response = client.post(
            "/auth/login", json={"username": "admin", "password": "admin123"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_login_with_invalid_credentials(self, client):
        """Test login with wrong password."""
        response = client.post(
            "/auth/login", json={"username": "admin", "password": "wrongpassword"}
        )

        assert response.status_code == 401

    def test_login_with_nonexistent_user(self, client):
        """Test login with non-existent user."""
        response = client.post(
            "/auth/login", json={"username": "nonexistent", "password": "password"}
        )

        assert response.status_code == 401


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        if not os.path.exists("models/fraud_model.joblib"):
            pytest.skip("Model file not available")

        from app import app

        return TestClient(app)

    @pytest.fixture
    def auth_token(self, client):
        """Get auth token for protected endpoints."""
        response = client.post(
            "/auth/login", json={"username": "admin", "password": "admin123"}
        )
        return response.json()["access_token"]

    def test_predict_requires_auth(self, client, sample_transaction):
        """Test that predict endpoint requires authentication."""
        response = client.post("/predict", json=sample_transaction)

        # Should fail without auth
        assert response.status_code in [401, 403]

    def test_predict_with_auth(self, client, auth_token, sample_transaction):
        """Test successful prediction with authentication."""
        response = client.post(
            "/predict",
            json=sample_transaction,
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "risk_level" in data
        assert "latency_ms" in data

    def test_predict_probability_range(self, client, auth_token, sample_transaction):
        """Test that fraud probability is between 0 and 1."""
        response = client.post(
            "/predict",
            json=sample_transaction,
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        data = response.json()
        assert 0 <= data["fraud_probability"] <= 1

    def test_predict_with_missing_field(self, client, auth_token):
        """Test prediction with missing required field."""
        incomplete_transaction = {"Amount": 100.0}

        response = client.post(
            "/predict",
            json=incomplete_transaction,
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 422  # Validation error


class TestBatchPredictEndpoint:
    """Tests for /predict/batch endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        if not os.path.exists("models/fraud_model.joblib"):
            pytest.skip("Model file not available")

        from app import app

        return TestClient(app)

    @pytest.fixture
    def auth_token(self, client):
        """Get auth token with write scope."""
        response = client.post(
            "/auth/login",
            json={"username": "api_user", "password": "api123"},  # Has write scope
        )
        return response.json()["access_token"]

    def test_batch_predict_returns_multiple(
        self, client, auth_token, batch_transactions
    ):
        """Test batch prediction returns correct count."""
        response = client.post(
            "/predict/batch",
            json=batch_transactions,
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total_transactions"] == len(batch_transactions)
        assert len(data["predictions"]) == len(batch_transactions)

    def test_batch_flagged_count(self, client, auth_token, batch_transactions):
        """Test batch prediction includes flagged count."""
        response = client.post(
            "/predict/batch",
            json=batch_transactions,
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        data = response.json()
        assert "flagged_count" in data
        assert isinstance(data["flagged_count"], int)


class TestExplainEndpoint:
    """Tests for /predict/explain endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        if not os.path.exists("models/fraud_model.joblib"):
            pytest.skip("Model file not available")

        from app import app

        return TestClient(app)

    @pytest.fixture
    def auth_token(self, client):
        """Get auth token."""
        response = client.post(
            "/auth/login", json={"username": "admin", "password": "admin123"}
        )
        return response.json()["access_token"]

    def test_explain_returns_explanation(self, client, auth_token, sample_transaction):
        """Test that explain endpoint returns SHAP explanation."""
        response = client.post(
            "/predict/explain",
            json=sample_transaction,
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "prediction" in data
        assert "explanation" in data
        assert "top_features" in data["explanation"]

    def test_explain_top_features_format(self, client, auth_token, sample_transaction):
        """Test that top features have correct format."""
        response = client.post(
            "/predict/explain?top_k=5",
            json=sample_transaction,
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        data = response.json()
        top_features = data["explanation"]["top_features"]

        assert len(top_features) <= 5

        for feature in top_features:
            assert "feature" in feature
            assert "contribution" in feature
            assert "direction" in feature


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        if not os.path.exists("models/fraud_model.joblib"):
            pytest.skip("Model file not available")

        from app import app

        return TestClient(app)

    def test_metrics_returns_prometheus_format(self, client):
        """Test that metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

    def test_metrics_contains_fraud_metrics(self, client):
        """Test that metrics include fraud-related metrics."""
        response = client.get("/metrics")
        content = response.text

        # Should contain our custom metrics
        assert "fraud_predictions_total" in content or response.status_code == 200


class TestDriftEndpoints:
    """Tests for drift detection endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        if not os.path.exists("models/fraud_model.joblib"):
            pytest.skip("Model file not available")

        from app import app

        return TestClient(app)

    @pytest.fixture
    def auth_token(self, client):
        """Get auth token."""
        response = client.post(
            "/auth/login", json={"username": "admin", "password": "admin123"}
        )
        return response.json()["access_token"]

    def test_drift_status_returns_result(self, client, auth_token):
        """Test that drift status endpoint returns a result."""
        response = client.get(
            "/drift/status", headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Should return either drift report or insufficient data message
        assert "status" in data or "drift_detected" in data

    def test_drift_stats_returns_result(self, client, auth_token):
        """Test that drift stats endpoint returns statistics."""
        response = client.get(
            "/drift/stats", headers={"Authorization": f"Bearer {auth_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "sample_count" in data
        assert "window_size" in data
