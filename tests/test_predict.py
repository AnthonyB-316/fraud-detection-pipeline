"""
Unit tests for prediction module.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestFraudDetectorPreprocess:
    """Tests for FraudDetector preprocessing."""

    def test_preprocess_adds_engineered_features(self, sample_transaction):
        """Test that preprocessing adds required engineered features."""
        # Import here to avoid model loading issues in CI
        from predict import FraudDetector

        # Skip if model not available
        if not os.path.exists("models/fraud_model.joblib"):
            pytest.skip("Model file not available")

        detector = FraudDetector()
        result = detector.preprocess(sample_transaction)

        # Should have all feature columns
        assert len(result.columns) == len(detector.feature_cols)

    def test_preprocess_returns_dataframe(self, sample_transaction):
        """Test that preprocess returns a pandas DataFrame."""
        import pandas as pd

        from predict import FraudDetector

        if not os.path.exists("models/fraud_model.joblib"):
            pytest.skip("Model file not available")

        detector = FraudDetector()
        result = detector.preprocess(sample_transaction)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1


class TestFraudDetectorPredict:
    """Tests for FraudDetector.predict method."""

    @pytest.fixture
    def detector(self):
        """Load fraud detector if model exists."""
        from predict import FraudDetector

        if not os.path.exists("models/fraud_model.joblib"):
            pytest.skip("Model file not available")

        return FraudDetector()

    def test_predict_returns_required_keys(self, detector, sample_transaction):
        """Test that predict returns all required keys."""
        result = detector.predict(sample_transaction)

        required_keys = ["fraud_probability", "is_fraud", "threshold", "risk_level"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_fraud_probability_is_bounded(self, detector, sample_transaction):
        """Test that fraud_probability is between 0 and 1."""
        result = detector.predict(sample_transaction)

        assert 0 <= result["fraud_probability"] <= 1

    def test_is_fraud_is_boolean(self, detector, sample_transaction):
        """Test that is_fraud is a boolean."""
        result = detector.predict(sample_transaction)

        assert isinstance(result["is_fraud"], bool)

    def test_risk_level_is_valid(self, detector, sample_transaction):
        """Test that risk_level is one of the expected values."""
        result = detector.predict(sample_transaction)

        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_is_fraud_matches_threshold(self, detector, sample_transaction):
        """Test that is_fraud flag matches probability vs threshold."""
        result = detector.predict(sample_transaction)

        expected_is_fraud = result["fraud_probability"] >= result["threshold"]
        assert result["is_fraud"] == expected_is_fraud


class TestFraudDetectorBatch:
    """Tests for FraudDetector.predict_batch method."""

    @pytest.fixture
    def detector(self):
        """Load fraud detector if model exists."""
        from predict import FraudDetector

        if not os.path.exists("models/fraud_model.joblib"):
            pytest.skip("Model file not available")

        return FraudDetector()

    def test_batch_returns_correct_count(self, detector, batch_transactions):
        """Test that batch prediction returns correct number of results."""
        results = detector.predict_batch(batch_transactions)

        assert len(results) == len(batch_transactions)

    def test_batch_results_have_required_keys(self, detector, batch_transactions):
        """Test that each batch result has required keys."""
        results = detector.predict_batch(batch_transactions)

        required_keys = ["fraud_probability", "is_fraud", "threshold", "risk_level"]
        for result in results:
            for key in required_keys:
                assert key in result, f"Missing key: {key}"

    def test_empty_batch_returns_empty_list(self, detector):
        """Test that empty batch returns empty list."""
        results = detector.predict_batch([])

        assert results == []


class TestFraudDetectorExplain:
    """Tests for FraudDetector.explain method (SHAP)."""

    @pytest.fixture
    def detector(self):
        """Load fraud detector if model exists."""
        from predict import FraudDetector

        if not os.path.exists("models/fraud_model.joblib"):
            pytest.skip("Model file not available")

        return FraudDetector()

    def test_explain_returns_prediction(self, detector, sample_transaction):
        """Test that explain returns prediction."""
        result = detector.explain(sample_transaction)

        assert "prediction" in result
        assert "fraud_probability" in result["prediction"]

    def test_explain_returns_explanation(self, detector, sample_transaction):
        """Test that explain returns explanation with SHAP values."""
        result = detector.explain(sample_transaction)

        assert "explanation" in result
        assert "top_features" in result["explanation"]
        assert "base_value" in result["explanation"]

    def test_explain_top_features_sorted_by_importance(self, detector, sample_transaction):
        """Test that top features are sorted by absolute contribution."""
        result = detector.explain(sample_transaction, top_k=5)

        top_features = result["explanation"]["top_features"]
        contributions = [abs(f["contribution"]) for f in top_features]

        # Should be sorted in descending order
        assert contributions == sorted(contributions, reverse=True)

    def test_explain_top_k_limits_features(self, detector, sample_transaction):
        """Test that top_k parameter limits returned features."""
        result = detector.explain(sample_transaction, top_k=3)

        assert len(result["explanation"]["top_features"]) == 3


class TestRiskLevelClassification:
    """Tests for risk level classification logic."""

    def test_low_risk_classification(self):
        """Test LOW risk for probability < 0.3."""
        from predict import FraudDetector

        if not os.path.exists("models/fraud_model.joblib"):
            pytest.skip("Model file not available")

        detector = FraudDetector()

        # Test the internal method
        assert detector._get_risk_level(0.1) == "LOW"
        assert detector._get_risk_level(0.29) == "LOW"

    def test_medium_risk_classification(self):
        """Test MEDIUM risk for 0.3 <= probability < 0.6."""
        from predict import FraudDetector

        if not os.path.exists("models/fraud_model.joblib"):
            pytest.skip("Model file not available")

        detector = FraudDetector()

        assert detector._get_risk_level(0.3) == "MEDIUM"
        assert detector._get_risk_level(0.5) == "MEDIUM"
        assert detector._get_risk_level(0.59) == "MEDIUM"

    def test_high_risk_classification(self):
        """Test HIGH risk for probability >= 0.6."""
        from predict import FraudDetector

        if not os.path.exists("models/fraud_model.joblib"):
            pytest.skip("Model file not available")

        detector = FraudDetector()

        assert detector._get_risk_level(0.6) == "HIGH"
        assert detector._get_risk_level(0.9) == "HIGH"
        assert detector._get_risk_level(1.0) == "HIGH"
