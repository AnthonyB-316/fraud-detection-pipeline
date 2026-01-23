"""
HuggingFace Spaces Demo - Fraud Detection Pipeline
Gradio interface for interactive fraud detection demo.
"""
import gradio as gr
import numpy as np
import pandas as pd
import joblib
import os

# Try to load model, otherwise create a demo mode
MODEL_LOADED = False
detector = None

try:
    # For HuggingFace Spaces, model should be in the repo
    if os.path.exists("models/fraud_model.joblib"):
        checkpoint = joblib.load("models/fraud_model.joblib")
        model = checkpoint['model']
        threshold = checkpoint['threshold']
        feature_cols = checkpoint['feature_cols']
        MODEL_LOADED = True
        print("Model loaded successfully!")
except Exception as e:
    print(f"Model not loaded: {e}")
    MODEL_LOADED = False


def engineer_features(transaction: dict) -> pd.DataFrame:
    """Engineer features from raw transaction."""
    df = pd.DataFrame([transaction])

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

    return df


def predict_fraud(amount, time_seconds, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                  v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                  v21, v22, v23, v24, v25, v26, v27, v28):
    """Make fraud prediction."""

    transaction = {
        'Time': time_seconds, 'Amount': amount,
        'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4, 'V5': v5,
        'V6': v6, 'V7': v7, 'V8': v8, 'V9': v9, 'V10': v10,
        'V11': v11, 'V12': v12, 'V13': v13, 'V14': v14, 'V15': v15,
        'V16': v16, 'V17': v17, 'V18': v18, 'V19': v19, 'V20': v20,
        'V21': v21, 'V22': v22, 'V23': v23, 'V24': v24, 'V25': v25,
        'V26': v26, 'V27': v27, 'V28': v28
    }

    if MODEL_LOADED:
        df = engineer_features(transaction)
        X = df[feature_cols]
        proba = model.predict_proba(X)[0, 1]
        is_fraud = proba >= threshold
    else:
        # Demo mode - simulate prediction based on heuristics
        proba = 0.05  # Base probability
        if amount > 500:
            proba += 0.2
        if v14 < -5:
            proba += 0.4
        if v4 > 3:
            proba += 0.15
        proba = min(proba, 0.99)
        is_fraud = proba >= 0.15
        threshold_used = 0.15

    # Determine risk level
    if proba < 0.3:
        risk_level = "LOW"
        risk_color = "green"
    elif proba < 0.6:
        risk_level = "MEDIUM"
        risk_color = "orange"
    else:
        risk_level = "HIGH"
        risk_color = "red"

    # Format output
    result = f"""
## Prediction Result

| Metric | Value |
|--------|-------|
| **Fraud Probability** | {proba:.1%} |
| **Risk Level** | <span style="color:{risk_color}">**{risk_level}**</span> |
| **Flagged as Fraud** | {'Yes' if is_fraud else 'No'} |
| **Threshold** | {threshold if MODEL_LOADED else 0.15:.2f} |

### Interpretation
"""

    if is_fraud:
        result += """
This transaction has been **flagged for review**. Key risk factors may include:
- Unusual transaction amount
- Suspicious PCA component patterns (V14, V4, V12 are typically important)
- Time of transaction
"""
    else:
        result += """
This transaction appears **legitimate**. The fraud probability is below the detection threshold.
"""

    if not MODEL_LOADED:
        result += "\n\n*Demo mode: Using heuristic rules. Deploy with trained model for real predictions.*"

    return result


def load_sample(sample_type):
    """Load sample transaction data."""
    if sample_type == "Legitimate Transaction":
        return [
            149.62, 0,  # Amount, Time
            -1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10,  # V1-V8
            0.36, 0.09, -0.55, -0.62, -0.99, -0.31, 1.47, -0.47,  # V9-V16
            0.21, 0.03, 0.40, 0.25, -0.02, 0.28, -0.11, 0.07,  # V17-V24
            0.13, -0.19, 0.13, -0.02  # V25-V28
        ]
    else:  # Suspicious Transaction
        return [
            999.99, 50000,  # High amount, late time
            -5.0, 3.5, -8.0, 6.0, -3.0, -2.5, -5.0, 1.0,  # Unusual V1-V8
            -3.0, -5.0, 4.0, -8.0, 1.0, -12.0, 1.0, -6.0,  # Unusual V9-V16
            -8.0, -3.0, 2.0, 0.5, 0.5, 0.5, -0.5, 0.5,  # V17-V24
            0.5, -0.5, 0.5, 0.5  # V25-V28
        ]


# Build Gradio Interface
with gr.Blocks(title="Fraud Detection Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Credit Card Fraud Detection

    Real-time fraud detection using XGBoost trained on 284K transactions.

    **Model Performance:** 94% Recall | 85% PR-AUC | <100ms latency

    [GitHub](https://github.com/AnthonyB-316/fraud-detection-pipeline) |
    [API Docs](https://fraud-detection-api.onrender.com/docs)

    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Transaction Details")

            sample_btn = gr.Radio(
                ["Legitimate Transaction", "Suspicious Transaction"],
                label="Load Sample",
                value="Legitimate Transaction"
            )

            amount = gr.Number(label="Amount ($)", value=149.62)
            time_input = gr.Number(label="Time (seconds from start)", value=0)

            gr.Markdown("### PCA Components (V1-V28)")
            gr.Markdown("*These are anonymized features from the payment processor*")

            with gr.Row():
                v1 = gr.Number(label="V1", value=-1.36)
                v2 = gr.Number(label="V2", value=-0.07)
                v3 = gr.Number(label="V3", value=2.54)
                v4 = gr.Number(label="V4", value=1.38)
            with gr.Row():
                v5 = gr.Number(label="V5", value=-0.34)
                v6 = gr.Number(label="V6", value=0.46)
                v7 = gr.Number(label="V7", value=0.24)
                v8 = gr.Number(label="V8", value=0.10)
            with gr.Row():
                v9 = gr.Number(label="V9", value=0.36)
                v10 = gr.Number(label="V10", value=0.09)
                v11 = gr.Number(label="V11", value=-0.55)
                v12 = gr.Number(label="V12", value=-0.62)
            with gr.Row():
                v13 = gr.Number(label="V13", value=-0.99)
                v14 = gr.Number(label="V14", value=-0.31)
                v15 = gr.Number(label="V15", value=1.47)
                v16 = gr.Number(label="V16", value=-0.47)
            with gr.Row():
                v17 = gr.Number(label="V17", value=0.21)
                v18 = gr.Number(label="V18", value=0.03)
                v19 = gr.Number(label="V19", value=0.40)
                v20 = gr.Number(label="V20", value=0.25)
            with gr.Row():
                v21 = gr.Number(label="V21", value=-0.02)
                v22 = gr.Number(label="V22", value=0.28)
                v23 = gr.Number(label="V23", value=-0.11)
                v24 = gr.Number(label="V24", value=0.07)
            with gr.Row():
                v25 = gr.Number(label="V25", value=0.13)
                v26 = gr.Number(label="V26", value=-0.19)
                v27 = gr.Number(label="V27", value=0.13)
                v28 = gr.Number(label="V28", value=-0.02)

            predict_btn = gr.Button("Detect Fraud", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Prediction Result")
            output = gr.Markdown()

    # Event handlers
    all_inputs = [amount, time_input, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                  v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                  v21, v22, v23, v24, v25, v26, v27, v28]

    predict_btn.click(fn=predict_fraud, inputs=all_inputs, outputs=output)

    # Load sample data
    sample_btn.change(
        fn=load_sample,
        inputs=sample_btn,
        outputs=all_inputs
    )

    gr.Markdown("""
    ---
    ### About This Project

    This fraud detection system demonstrates:
    - **XGBoost** classifier optimized for imbalanced data
    - **SMOTE** oversampling to handle 0.17% fraud rate
    - **SHAP** explainability for regulatory compliance
    - **FastAPI** backend with JWT authentication
    - **Docker** containerization for deployment
    - **Prometheus/Grafana** monitoring stack

    Built by [Anthony Buonantuono](https://github.com/AnthonyB-316)
    """)


if __name__ == "__main__":
    demo.launch()
