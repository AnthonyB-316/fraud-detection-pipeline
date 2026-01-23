"""
Pytest fixtures for fraud detection pipeline tests.
"""
import pytest
import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))


@pytest.fixture
def sample_transaction():
    """Single sample transaction for testing."""
    return {
        'Time': 0,
        'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155,
        'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698,
        'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600, 'V12': -0.617801,
        'V13': -0.991390, 'V14': -0.311169, 'V15': 1.468177, 'V16': -0.470401,
        'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412,
        'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928,
        'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053,
        'Amount': 149.62
    }


@pytest.fixture
def fraudulent_transaction():
    """Transaction that should be flagged as fraud (high-risk features)."""
    return {
        'Time': 50000,
        'V1': -5.0, 'V2': 3.5, 'V3': -8.0, 'V4': 6.0,
        'V5': -3.0, 'V6': -2.5, 'V7': -5.0, 'V8': 1.0,
        'V9': -3.0, 'V10': -5.0, 'V11': 4.0, 'V12': -8.0,
        'V13': 1.0, 'V14': -12.0, 'V15': 1.0, 'V16': -6.0,
        'V17': -8.0, 'V18': -3.0, 'V19': 2.0, 'V20': 0.5,
        'V21': 0.5, 'V22': 0.5, 'V23': -0.5, 'V24': 0.5,
        'V25': 0.5, 'V26': -0.5, 'V27': 0.5, 'V28': 0.5,
        'Amount': 999.99
    }


@pytest.fixture
def batch_transactions(sample_transaction, fraudulent_transaction):
    """Batch of transactions for testing."""
    return [sample_transaction, fraudulent_transaction]


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame with credit card transactions."""
    np.random.seed(42)
    n_samples = 100

    data = {
        'Time': np.random.uniform(0, 172800, n_samples),
        'Amount': np.random.exponential(100, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    }

    # Add V1-V28 features
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)

    return pd.DataFrame(data)


@pytest.fixture
def feature_columns():
    """List of feature columns used in the model."""
    v_cols = [f'V{i}' for i in range(1, 29)]
    engineered = [
        'Amount_Scaled',
        'Hour_Sin',
        'Hour_Cos',
        'Amount_Zscore',
        'High_Amount',
        'V1_V2_Interaction',
        'V1_V3_Interaction'
    ]
    return v_cols + engineered


@pytest.fixture
def auth_headers():
    """
    Authentication headers for API tests.
    Note: In real tests, you would get these by calling /auth/login.
    """
    return {"Authorization": "Bearer test_token"}


@pytest.fixture
def api_base_url():
    """Base URL for API tests."""
    return "http://localhost:8000"
