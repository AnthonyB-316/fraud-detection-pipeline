"""
Prediction module for fraud detection with SHAP explainability.
Loads trained model and makes predictions on new transactions.
"""
import joblib
import pandas as pd
import numpy as np
import shap
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = "models/fraud_model.joblib"


class FraudDetector:
    """
    Fraud detection inference class with SHAP explainability.

    Features:
    - Single and batch predictions
    - Risk level classification (LOW/MEDIUM/HIGH)
    - SHAP-based explanations for model decisions
    """

    def __init__(self, model_path: str = MODEL_PATH):
        """Load trained model and initialize SHAP explainer."""
        checkpoint = joblib.load(model_path)
        self.model = checkpoint['model']
        self.threshold = checkpoint['threshold']
        self.feature_cols = checkpoint['feature_cols']
        self.metrics = checkpoint['metrics']

        # Initialize SHAP explainer (lazy loading)
        self._explainer: Optional[shap.TreeExplainer] = None

        logger.info(f"Model loaded. Threshold: {self.threshold:.4f}")

    @property
    def explainer(self) -> shap.TreeExplainer:
        """Lazy-load SHAP explainer."""
        if self._explainer is None:
            logger.info("Initializing SHAP TreeExplainer...")
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def preprocess(self, transaction: Dict) -> pd.DataFrame:
        """
        Preprocess a single transaction for prediction.
        Expects dict with V1-V28, Amount, Time.
        """
        df = pd.DataFrame([transaction])

        # Engineer features (same as training)
        # Note: Using approximate training statistics for scaling
        amount_mean = 88.35
        amount_std = 250.12

        df['Amount_Scaled'] = (df['Amount'] - amount_mean) / amount_std
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Amount_Zscore'] = (df['Amount'] - amount_mean) / amount_std
        df['High_Amount'] = (df['Amount_Zscore'] > 2).astype(int)
        df['V1_V2_Interaction'] = df['V1'] * df['V2']
        df['V1_V3_Interaction'] = df['V1'] * df['V3']

        return df[self.feature_cols]

    def preprocess_batch(self, transactions: List[Dict]) -> pd.DataFrame:
        """Preprocess multiple transactions."""
        df = pd.DataFrame(transactions)

        amount_mean = 88.35
        amount_std = 250.12

        df['Amount_Scaled'] = (df['Amount'] - amount_mean) / amount_std
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Amount_Zscore'] = (df['Amount'] - amount_mean) / amount_std
        df['High_Amount'] = (df['Amount_Zscore'] > 2).astype(int)
        df['V1_V2_Interaction'] = df['V1'] * df['V2']
        df['V1_V3_Interaction'] = df['V1'] * df['V3']

        return df[self.feature_cols]

    def _get_risk_level(self, proba: float) -> str:
        """Classify risk level based on probability."""
        if proba < 0.3:
            return 'LOW'
        elif proba < 0.6:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def predict(self, transaction: Dict) -> Dict:
        """
        Predict fraud probability for a single transaction.

        Returns:
            {
                'fraud_probability': float,
                'is_fraud': bool,
                'threshold': float,
                'risk_level': str
            }
        """
        X = self.preprocess(transaction)
        proba = self.model.predict_proba(X)[0, 1]
        is_fraud = proba >= self.threshold

        return {
            'fraud_probability': float(proba),
            'is_fraud': bool(is_fraud),
            'threshold': self.threshold,
            'risk_level': self._get_risk_level(proba)
        }

    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Predict fraud for multiple transactions efficiently."""
        if not transactions:
            return []

        X = self.preprocess_batch(transactions)
        probas = self.model.predict_proba(X)[:, 1]

        results = []
        for proba in probas:
            is_fraud = proba >= self.threshold
            results.append({
                'fraud_probability': float(proba),
                'is_fraud': bool(is_fraud),
                'threshold': self.threshold,
                'risk_level': self._get_risk_level(proba)
            })

        return results

    def explain(self, transaction: Dict, top_k: int = 10) -> Dict:
        """
        Explain prediction using SHAP values.

        Returns:
            {
                'prediction': {...},
                'explanation': {
                    'base_value': float,  # Expected model output
                    'top_features': [
                        {'feature': str, 'contribution': float, 'value': float},
                        ...
                    ],
                    'shap_values': {...}  # All SHAP values
                }
            }
        """
        # Get prediction first
        prediction = self.predict(transaction)

        # Compute SHAP values
        X = self.preprocess(transaction)
        shap_values = self.explainer.shap_values(X)

        # For binary classification, get SHAP values for fraud class (index 1)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]

        # Get feature contributions
        feature_contributions = []
        for i, (feature, shap_val) in enumerate(zip(self.feature_cols, shap_vals)):
            feature_contributions.append({
                'feature': feature,
                'contribution': float(shap_val),
                'value': float(X.iloc[0, i]),
                'abs_contribution': abs(float(shap_val))
            })

        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)

        # Get top K features
        top_features = [
            {
                'feature': f['feature'],
                'contribution': f['contribution'],
                'value': f['value'],
                'direction': 'increases fraud risk' if f['contribution'] > 0 else 'decreases fraud risk'
            }
            for f in feature_contributions[:top_k]
        ]

        # Base value (expected value)
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                base_value = float(self.explainer.expected_value[1])
            else:
                base_value = float(self.explainer.expected_value)
        else:
            base_value = 0.0

        return {
            'prediction': prediction,
            'explanation': {
                'base_value': base_value,
                'output_value': float(sum(shap_vals) + base_value),
                'top_features': top_features,
                'feature_count': len(self.feature_cols),
                'all_shap_values': {
                    feat: float(val) for feat, val in zip(self.feature_cols, shap_vals)
                }
            }
        }

    def explain_batch(self, transactions: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Explain predictions for multiple transactions.
        Note: More expensive than predict_batch due to SHAP computation.
        """
        if not transactions:
            return []

        # Get predictions
        predictions = self.predict_batch(transactions)

        # Compute SHAP values for batch
        X = self.preprocess_batch(transactions)
        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_vals_batch = shap_values[1]
        else:
            shap_vals_batch = shap_values

        # Get base value
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                base_value = float(self.explainer.expected_value[1])
            else:
                base_value = float(self.explainer.expected_value)
        else:
            base_value = 0.0

        results = []
        for i, (pred, shap_vals) in enumerate(zip(predictions, shap_vals_batch)):
            # Get top K features for this sample
            contributions = [
                (feat, float(val), abs(float(val)))
                for feat, val in zip(self.feature_cols, shap_vals)
            ]
            contributions.sort(key=lambda x: x[2], reverse=True)

            top_features = [
                {
                    'feature': feat,
                    'contribution': val,
                    'direction': 'increases fraud risk' if val > 0 else 'decreases fraud risk'
                }
                for feat, val, _ in contributions[:top_k]
            ]

            results.append({
                'prediction': pred,
                'explanation': {
                    'base_value': base_value,
                    'top_features': top_features
                }
            })

        return results

    def get_feature_importance(self) -> Dict[str, float]:
        """Get global feature importance from the model."""
        importance = self.model.feature_importances_
        return {
            feat: float(imp)
            for feat, imp in sorted(
                zip(self.feature_cols, importance),
                key=lambda x: x[1],
                reverse=True
            )
        }


if __name__ == "__main__":
    # Test with a sample transaction
    detector = FraudDetector()

    sample = {
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

    print("=== Basic Prediction ===")
    result = detector.predict(sample)
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"Is Fraud: {result['is_fraud']}")
    print(f"Risk Level: {result['risk_level']}")

    print("\n=== Prediction with Explanation ===")
    explained = detector.explain(sample)
    print(f"\nTop contributing features:")
    for feat in explained['explanation']['top_features'][:5]:
        print(f"  {feat['feature']}: {feat['contribution']:+.4f} ({feat['direction']})")
