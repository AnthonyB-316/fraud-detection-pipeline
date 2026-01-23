"""
Fraud Detection Demo - Clean interface for portfolio showcase.
"""

import os
import random

import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load model
MODEL_LOADED = False
model = None
threshold = 0.15
feature_cols = None

try:
    if os.path.exists("models/fraud_model.joblib"):
        checkpoint = joblib.load("models/fraud_model.joblib")
        model = checkpoint["model"]
        threshold = checkpoint["threshold"]
        feature_cols = checkpoint["feature_cols"]
        MODEL_LOADED = True
except Exception as e:
    print(f"Model not loaded: {e}")

# Sample transactions - realistic patterns from the dataset
SAMPLES = {
    "legitimate_small": {
        "name": "Small Purchase - Coffee Shop",
        "amount": 12.50,
        "time": 36000,  # 10am
        "v_pattern": "normal",
        "description": "Typical morning purchase"
    },
    "legitimate_medium": {
        "name": "Online Shopping - Electronics",
        "amount": 299.99,
        "time": 54000,  # 3pm
        "v_pattern": "normal",
        "description": "Regular e-commerce transaction"
    },
    "legitimate_large": {
        "name": "Monthly Rent Payment",
        "amount": 1850.00,
        "time": 32400,  # 9am
        "v_pattern": "normal",
        "description": "Expected recurring payment"
    },
    "suspicious_amount": {
        "name": "Unusual Large Purchase",
        "amount": 4999.99,
        "time": 10800,  # 3am
        "v_pattern": "suspicious",
        "description": "High amount at unusual hour"
    },
    "suspicious_pattern": {
        "name": "Anomalous Transaction",
        "amount": 847.32,
        "time": 14400,  # 4am
        "v_pattern": "very_suspicious",
        "description": "Unusual spending pattern detected"
    },
}


def generate_v_features(pattern: str) -> dict:
    """Generate realistic V1-V28 features based on pattern type."""
    np.random.seed(random.randint(0, 10000))

    if pattern == "normal":
        # Normal transactions cluster around 0 with small variance
        return {f"V{i}": np.random.normal(0, 0.8) for i in range(1, 29)}
    elif pattern == "suspicious":
        # Suspicious: some features deviate significantly
        v = {f"V{i}": np.random.normal(0, 1) for i in range(1, 29)}
        # Key fraud indicators based on feature importance
        v["V14"] = np.random.uniform(-8, -4)  # V14 is highly predictive
        v["V4"] = np.random.uniform(3, 6)
        v["V12"] = np.random.uniform(-6, -3)
        return v
    else:  # very_suspicious
        v = {f"V{i}": np.random.normal(0, 1.2) for i in range(1, 29)}
        v["V14"] = np.random.uniform(-12, -7)
        v["V4"] = np.random.uniform(4, 8)
        v["V12"] = np.random.uniform(-10, -5)
        v["V10"] = np.random.uniform(-6, -3)
        v["V3"] = np.random.uniform(-8, -4)
        return v


def engineer_features(transaction: dict) -> pd.DataFrame:
    """Engineer features from raw transaction."""
    df = pd.DataFrame([transaction])

    amount_mean, amount_std = 88.35, 250.12

    df["Amount_Scaled"] = (df["Amount"] - amount_mean) / amount_std
    df["Hour"] = (df["Time"] / 3600) % 24
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Amount_Zscore"] = (df["Amount"] - amount_mean) / max(amount_std, 0.01)
    df["High_Amount"] = (df["Amount_Zscore"] > 2).astype(int)
    df["V1_V2_Interaction"] = df["V1"] * df["V2"]
    df["V1_V3_Interaction"] = df["V1"] * df["V3"]

    return df


def predict(sample_type: str, custom_amount: float = None):
    """Make prediction on selected sample."""

    if sample_type not in SAMPLES:
        return "Please select a transaction type.", "", ""

    sample = SAMPLES[sample_type]
    amount = custom_amount if custom_amount else sample["amount"]

    # Build transaction
    transaction = {
        "Time": sample["time"],
        "Amount": amount,
        **generate_v_features(sample["v_pattern"])
    }

    # Predict
    if MODEL_LOADED:
        df = engineer_features(transaction)
        X = df[feature_cols]
        proba = float(model.predict_proba(X)[0, 1])
        is_fraud = proba >= threshold
    else:
        # Demo fallback
        base = 0.05
        if sample["v_pattern"] == "suspicious":
            base = 0.45
        elif sample["v_pattern"] == "very_suspicious":
            base = 0.78
        if amount > 1000:
            base += 0.1
        proba = min(base + random.uniform(-0.05, 0.05), 0.99)
        is_fraud = proba >= 0.15

    # Format results
    if proba < 0.3:
        risk_level = "LOW RISK"
        risk_color = "#28a745"
        risk_emoji = "✅"
    elif proba < 0.6:
        risk_level = "MEDIUM RISK"
        risk_color = "#ffc107"
        risk_emoji = "⚠️"
    else:
        risk_level = "HIGH RISK"
        risk_color = "#dc3545"
        risk_emoji = "🚨"

    # Result card
    result_html = f"""
    <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; font-family: system-ui;">
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 48px;">{risk_emoji}</span>
            <h2 style="margin: 10px 0; color: {risk_color};">{risk_level}</h2>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; text-align: center;">
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold;">{proba:.1%}</div>
                <div style="font-size: 12px; opacity: 0.7;">Fraud Probability</div>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold;">${amount:,.2f}</div>
                <div style="font-size: 12px; opacity: 0.7;">Transaction Amount</div>
            </div>
        </div>
        <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 8px;">
            <strong>Decision:</strong> {"🚫 BLOCKED - Manual review required" if is_fraud else "✓ APPROVED - Transaction cleared"}
        </div>
    </div>
    """

    # Transaction details
    hour = int((sample["time"] / 3600) % 24)
    time_str = f"{hour:02d}:00" + (" AM" if hour < 12 else " PM")

    details = f"""
**Transaction:** {sample["name"]}

**Amount:** ${amount:,.2f}

**Time:** {time_str}

**Description:** {sample["description"]}
    """

    # Model info
    model_info = f"""
**Model:** XGBoost Classifier

**Threshold:** {threshold:.2f}

**Status:** {"Production Model" if MODEL_LOADED else "Demo Mode"}

**Features:** 35 (28 PCA + 7 engineered)
    """

    return result_html, details, model_info


def random_transaction():
    """Generate a random transaction for testing."""
    samples = list(SAMPLES.keys())
    return random.choice(samples)


# Build interface
with gr.Blocks(
    title="Fraud Detection",
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
    ),
    css="""
    .gradio-container { max-width: 900px !important; }
    footer { display: none !important; }
    """
) as demo:

    gr.Markdown("""
    # 🔒 Credit Card Fraud Detection

    Real-time ML system trained on 284K transactions. Select a transaction scenario below to see how the model classifies it.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Select Transaction")

            sample_dropdown = gr.Dropdown(
                choices=[
                    ("☕ Small Purchase - Coffee Shop", "legitimate_small"),
                    ("🛒 Online Shopping - Electronics", "legitimate_medium"),
                    ("🏠 Monthly Rent Payment", "legitimate_large"),
                    ("⚠️ Unusual Large Purchase", "suspicious_amount"),
                    ("🚨 Anomalous Transaction", "suspicious_pattern"),
                ],
                value="legitimate_small",
                label="Transaction Type",
                interactive=True
            )

            custom_amount = gr.Number(
                label="Custom Amount (optional)",
                value=None,
                minimum=0,
                maximum=50000,
                info="Override the default amount"
            )

            with gr.Row():
                predict_btn = gr.Button("Analyze Transaction", variant="primary", scale=2)
                random_btn = gr.Button("🎲 Random", scale=1)

            gr.Markdown("### Transaction Details")
            details_output = gr.Markdown()

            gr.Markdown("### Model Info")
            model_output = gr.Markdown()

        with gr.Column(scale=1):
            gr.Markdown("### Analysis Result")
            result_output = gr.HTML()

    gr.Markdown("""
    ---
    ### How It Works

    This model uses **XGBoost** trained on the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
    with SMOTE oversampling to handle the 0.17% fraud rate.

    **Key Features:**
    - 28 PCA-transformed features from transaction data
    - 7 engineered features (time patterns, amount anomalies)
    - 94% recall optimized (catches most fraud, accepts some false positives)
    - SHAP explainability for regulatory compliance

    [📂 View Source Code](https://github.com/AnthonyB-316/fraud-detection-pipeline)
    """)

    # Event handlers
    predict_btn.click(
        fn=predict,
        inputs=[sample_dropdown, custom_amount],
        outputs=[result_output, details_output, model_output]
    )

    random_btn.click(
        fn=random_transaction,
        outputs=[sample_dropdown]
    )

    # Load initial prediction
    demo.load(
        fn=predict,
        inputs=[sample_dropdown, custom_amount],
        outputs=[result_output, details_output, model_output]
    )


if __name__ == "__main__":
    demo.launch()
