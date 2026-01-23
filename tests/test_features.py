"""
Unit tests for feature engineering module.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import engineer_features, get_feature_columns, prepare_train_test


class TestEngineerFeatures:
    """Tests for engineer_features function."""

    def test_engineer_features_adds_expected_columns(self, sample_dataframe):
        """Test that engineer_features adds all expected derived columns."""
        result = engineer_features(sample_dataframe)

        expected_columns = [
            "Amount_Scaled",
            "Hour",
            "Hour_Sin",
            "Hour_Cos",
            "Amount_Zscore",
            "High_Amount",
            "V1_V2_Interaction",
            "V1_V3_Interaction",
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_hour_features_are_bounded(self, sample_dataframe):
        """Test that hour-derived features are properly bounded."""
        result = engineer_features(sample_dataframe)

        # Sin and Cos should be between -1 and 1
        assert result["Hour_Sin"].between(-1, 1).all()
        assert result["Hour_Cos"].between(-1, 1).all()

        # Hour should be between 0 and 24
        assert result["Hour"].between(0, 24).all()

    def test_high_amount_is_binary(self, sample_dataframe):
        """Test that High_Amount is binary (0 or 1)."""
        result = engineer_features(sample_dataframe)

        assert result["High_Amount"].isin([0, 1]).all()

    def test_interaction_features_calculated_correctly(self, sample_dataframe):
        """Test that interaction features are calculated correctly."""
        result = engineer_features(sample_dataframe)

        expected_v1_v2 = sample_dataframe["V1"] * sample_dataframe["V2"]
        expected_v1_v3 = sample_dataframe["V1"] * sample_dataframe["V3"]

        np.testing.assert_array_almost_equal(
            result["V1_V2_Interaction"].values, expected_v1_v2.values
        )
        np.testing.assert_array_almost_equal(
            result["V1_V3_Interaction"].values, expected_v1_v3.values
        )

    def test_original_columns_preserved(self, sample_dataframe):
        """Test that original columns are preserved."""
        original_cols = sample_dataframe.columns.tolist()
        result = engineer_features(sample_dataframe)

        for col in original_cols:
            assert col in result.columns, f"Original column lost: {col}"


class TestGetFeatureColumns:
    """Tests for get_feature_columns function."""

    def test_returns_correct_count(self, sample_dataframe):
        """Test that correct number of features are returned."""
        df = engineer_features(sample_dataframe)
        features = get_feature_columns(df)

        # 28 V columns + 7 engineered = 35
        assert len(features) == 35

    def test_all_v_columns_included(self, sample_dataframe):
        """Test that all V columns are included."""
        df = engineer_features(sample_dataframe)
        features = get_feature_columns(df)

        for i in range(1, 29):
            assert f"V{i}" in features

    def test_engineered_columns_included(self, sample_dataframe):
        """Test that engineered columns are included."""
        df = engineer_features(sample_dataframe)
        features = get_feature_columns(df)

        expected_engineered = [
            "Amount_Scaled",
            "Hour_Sin",
            "Hour_Cos",
            "Amount_Zscore",
            "High_Amount",
            "V1_V2_Interaction",
            "V1_V3_Interaction",
        ]

        for col in expected_engineered:
            assert col in features, f"Missing engineered feature: {col}"


class TestPrepareTrainTest:
    """Tests for prepare_train_test function."""

    def test_smote_increases_minority_class(self, sample_dataframe):
        """Test that SMOTE increases the fraud (minority) class."""
        df = engineer_features(sample_dataframe)
        X_train, X_test, y_train, y_test, _ = prepare_train_test(df)

        # After SMOTE, fraud rate should be higher than original
        train_fraud_rate = y_train.mean()

        # SMOTE with 0.5 sampling strategy means fraud:non-fraud = 1:2
        assert train_fraud_rate > 0.1, "SMOTE should increase minority class proportion"

    def test_test_set_unchanged(self, sample_dataframe):
        """Test that test set is not modified by SMOTE."""
        df = engineer_features(sample_dataframe)
        original_fraud_rate = df["Class"].mean()

        _, X_test, _, y_test, _ = prepare_train_test(df)

        # Test set fraud rate should be close to original
        test_fraud_rate = y_test.mean()

        # Allow some variance due to stratified split
        assert abs(test_fraud_rate - original_fraud_rate) < 0.1

    def test_returns_correct_shapes(self, sample_dataframe):
        """Test that returned arrays have correct shapes."""
        df = engineer_features(sample_dataframe)
        X_train, X_test, y_train, y_test, features = prepare_train_test(
            df, test_size=0.2
        )

        # Test set should be ~20% of data
        assert len(X_test) == pytest.approx(len(df) * 0.2, rel=0.1)

        # X and y should have same number of samples
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

        # X should have correct number of features
        assert X_train.shape[1] == len(features)
        assert X_test.shape[1] == len(features)


class TestDataTypes:
    """Tests for data type handling."""

    def test_handles_float_columns(self):
        """Test that float columns are handled correctly."""
        df = pd.DataFrame(
            {
                "Time": [0.0, 100.0, 200.0],
                "Amount": [10.5, 20.5, 30.5],
                "Class": [0, 0, 1],
                **{
                    f"V{i}": [float(i), float(i + 1), float(i + 2)]
                    for i in range(1, 29)
                },
            }
        )

        result = engineer_features(df)
        assert result["Amount_Scaled"].dtype in [np.float64, np.float32]

    def test_handles_zero_values(self):
        """Test that zero values don't cause issues."""
        df = pd.DataFrame(
            {
                "Time": [0, 0, 0],
                "Amount": [0, 0, 0],
                "Class": [0, 0, 0],
                **{f"V{i}": [0.0, 0.0, 0.0] for i in range(1, 29)},
            }
        )

        # Should not raise any errors
        result = engineer_features(df)
        assert not result.isnull().any().any(), "No null values should be created"
