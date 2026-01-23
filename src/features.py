"""
Feature engineering for fraud detection pipeline.
Handles class imbalance and creates derived features.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load the credit card fraud dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} transactions")
    print(f"Fraud rate: {df['Class'].mean():.4%}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw transaction data.

    The Kaggle dataset has V1-V28 (PCA components) + Time + Amount.
    We'll add:
    - Normalized amount
    - Time-based features (hour of day patterns)
    - Amount anomaly score
    """
    df = df.copy()

    scaler = StandardScaler()
    df["Amount_Scaled"] = scaler.fit_transform(df[["Amount"]])

    df["Hour"] = (df["Time"] / 3600) % 24
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    amount_mean = df["Amount"].mean()
    amount_std = df["Amount"].std()
    df["Amount_Zscore"] = (df["Amount"] - amount_mean) / amount_std

    df["High_Amount"] = (df["Amount_Zscore"] > 2).astype(int)

    df["V1_V2_Interaction"] = df["V1"] * df["V2"]
    df["V1_V3_Interaction"] = df["V1"] * df["V3"]

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature columns for training."""
    v_cols = [f"V{i}" for i in range(1, 29)]
    engineered = [
        "Amount_Scaled",
        "Hour_Sin",
        "Hour_Cos",
        "Amount_Zscore",
        "High_Amount",
        "V1_V2_Interaction",
        "V1_V3_Interaction",
    ]
    return v_cols + engineered


def prepare_train_test(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """
    Split data and apply SMOTE to handle class imbalance.
    SMOTE only on training data to prevent data leakage.
    """
    from sklearn.model_selection import train_test_split

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(
        f"Before SMOTE - Train: {len(X_train):,} | Fraud: {y_train.sum():,} ({y_train.mean():.4%})"
    )

    smote = SMOTE(random_state=random_state, sampling_strategy=0.5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(
        f"After SMOTE - Train: {len(X_train_resampled):,} | Fraud: {y_train_resampled.sum():,} ({y_train_resampled.mean():.4%})"
    )
    print(f"Test set: {len(X_test):,} | Fraud: {y_test.sum():,} ({y_test.mean():.4%})")

    return X_train_resampled, X_test, y_train_resampled, y_test, feature_cols


if __name__ == "__main__":
    df = load_data("data/creditcard.csv")
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, features = prepare_train_test(df)
    print(f"\nFeatures: {len(features)}")
