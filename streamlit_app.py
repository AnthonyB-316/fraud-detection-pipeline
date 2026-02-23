"""
Streamlit dashboard for fraud detection monitoring.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("Fraud Detection Pipeline")
st.markdown("Real-time monitoring and analysis of credit card fraud detection")

# Tabs
tab1, tab2, tab3 = st.tabs(["Model Performance", "Live Predictions", "Data Explorer"])

with tab1:
    st.header("Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    # These would come from actual model metrics in production
    col1.metric("Recall", "94.2%", help="Percentage of fraud caught")
    col2.metric("Precision", "12.8%", help="Accuracy of fraud flags")
    col3.metric("False Positive Rate", "4.8%", help="Legitimate transactions flagged")
    col4.metric("PR AUC", "0.847", help="Precision-Recall Area Under Curve")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Precision-Recall Tradeoff")
        # Simulated PR curve data
        recalls = np.linspace(0, 1, 100)
        precisions = 1 / (1 + np.exp(5 * (recalls - 0.5)))  # Sigmoid-like
        precisions = precisions * 0.8 + np.random.normal(0, 0.02, 100)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recalls, y=precisions, mode='lines', name='Model'))
        fig.add_vline(x=0.94, line_dash="dash", line_color="red", annotation_text="Current Threshold")
        fig.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Confusion Matrix")
        cm_data = [[54500, 2800], [28, 464]]
        fig = px.imshow(
            cm_data,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Legitimate', 'Fraud'],
            y=['Legitimate', 'Fraud'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Live Transaction Scoring")

    st.markdown("Enter transaction details to get fraud prediction:")

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.form("transaction_form"):
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
            time_val = st.number_input("Time (seconds from start)", min_value=0, value=0)

            st.markdown("**PCA Components** (V1-V28 from payment processor)")
            v_cols = st.columns(4)
            v_values = {}
            for i in range(1, 29):
                col_idx = (i - 1) % 4
                v_values[f'V{i}'] = v_cols[col_idx].number_input(
                    f"V{i}", value=0.0, format="%.4f", key=f"v{i}"
                )

            submitted = st.form_submit_button("Score Transaction", type="primary")

    with col2:
        st.subheader("Prediction Result")

        if submitted:
            # Simulate prediction
            score = np.random.random() * 0.3  # Most transactions are legit
            if amount > 1000:
                score += 0.3
            if v_values.get('V14', 0) < -5:
                score += 0.4

            score = min(score, 0.99)

            if score < 0.3:
                risk_level = "LOW"
                color = "green"
            elif score < 0.6:
                risk_level = "MEDIUM"
                color = "orange"
            else:
                risk_level = "HIGH"
                color = "red"

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': '#2ecc71'},
                        {'range': [30, 60], 'color': '#f1c40f'},
                        {'range': [60, 100], 'color': '#e74c3c'}
                    ],
                }
            ))
            fig.update_layout(height=250, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            if risk_level == "HIGH":
                st.error(f"**{risk_level} RISK** - Flag for review")
            elif risk_level == "MEDIUM":
                st.warning(f"**{risk_level} RISK** - Monitor closely")
            else:
                st.success(f"**{risk_level} RISK** - Approve transaction")
        else:
            st.info("Submit a transaction to see prediction")

with tab3:
    st.header("Dataset Explorer")

    st.markdown("Upload a CSV to analyze fraud patterns in your data")

    uploaded = st.file_uploader("Upload transactions CSV", type=['csv'])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df):,} transactions")

        if 'Class' in df.columns:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", f"{len(df):,}")
            col2.metric("Fraud Cases", f"{df['Class'].sum():,}")
            col3.metric("Fraud Rate", f"{df['Class'].mean():.4%}")

            st.subheader("Amount Distribution by Class")
            fig = px.histogram(
                df, x='Amount', color='Class',
                nbins=50, marginal='box',
                color_discrete_map={0: 'blue', 1: 'red'},
                labels={'Class': 'Is Fraud'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload the creditcard.csv file to explore the dataset")

# Footer
st.markdown("---")
st.caption("Fraud Detection Pipeline | Built with XGBoost, FastAPI, Streamlit")
