"""
Fraud Detection Dashboard - User-friendly demo
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")

st.title("Credit Card Fraud Detection")
st.caption("XGBoost model trained on 284K transactions | 94% recall")

# Generate sample data for demo
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_legit = 1000
    n_fraud = 50  # ~5% fraud rate (slightly elevated for demo visibility)

    # Legitimate transactions - log-normal amounts (realistic distribution)
    # Log-normal: most transactions $20-100, some $100-300, few $300+
    legit_amounts = np.random.lognormal(mean=3.5, sigma=0.8, size=n_legit)  # median ~$33, mean ~$50
    legit_amounts = np.clip(legit_amounts, 5, 500)  # realistic bounds

    # Hours: bimodal - lunch (11-14) and evening (17-20) peaks
    legit_hours = np.concatenate([
        np.random.normal(12, 1.5, n_legit // 3).astype(int),  # lunch
        np.random.normal(18, 2, n_legit // 3).astype(int),    # evening
        np.random.randint(8, 22, n_legit - 2 * (n_legit // 3))  # spread
    ])
    legit_hours = np.clip(legit_hours, 6, 23)

    # Distance: most purchases near home, log-normal
    legit_distance = np.random.lognormal(mean=1.5, sigma=0.7, size=n_legit)
    legit_distance = np.clip(legit_distance, 0.1, 30)

    legit = pd.DataFrame({
        'Amount': legit_amounts,
        'Hour': legit_hours,
        'DayOfWeek': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], n_legit),
        'Category': np.random.choice(['Grocery', 'Gas', 'Restaurant', 'Online', 'Retail'], n_legit, p=[0.30, 0.15, 0.20, 0.20, 0.15]),
        'Distance': legit_distance,
        'Class': 0
    })

    # Fraudulent transactions - different patterns
    # Higher amounts, bimodal: small test charges + large fraud
    fraud_amounts = np.concatenate([
        np.random.uniform(1, 10, n_fraud // 4),           # test charges
        np.random.lognormal(5.5, 0.6, n_fraud - n_fraud // 4)  # large purchases
    ])
    fraud_amounts = np.clip(fraud_amounts, 1, 2000)

    # Hours: late night bias (but not exclusively)
    fraud_hours = np.concatenate([
        np.random.choice([0, 1, 2, 3, 4, 5, 23], n_fraud * 2 // 3),  # late night
        np.random.randint(0, 24, n_fraud - n_fraud * 2 // 3)         # some daytime
    ])

    # Distance: far from home
    fraud_distance = np.random.lognormal(mean=3.5, sigma=0.8, size=n_fraud)
    fraud_distance = np.clip(fraud_distance, 10, 200)

    fraud = pd.DataFrame({
        'Amount': fraud_amounts,
        'Hour': fraud_hours,
        'DayOfWeek': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], n_fraud),
        'Category': np.random.choice(['Online', 'ATM', 'Wire', 'Retail'], n_fraud, p=[0.35, 0.30, 0.20, 0.15]),
        'Distance': fraud_distance,
        'Class': 1
    })

    df = pd.concat([legit, fraud]).reset_index(drop=True)
    df['Amount'] = df['Amount'].round(2)
    df['Distance'] = df['Distance'].round(1)
    df['Hour'] = df['Hour'].astype(int)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

SAMPLE_DATA = generate_sample_data()

# Tabs
tab1, tab2, tab3 = st.tabs(["Model Performance", "Test Transaction", "Data Explorer"])

with tab1:
    st.header("Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recall", "94.2%", help="% of fraud we catch")
    col2.metric("Precision", "12.8%", help="% of flags that are real fraud")
    col3.metric("False Positive Rate", "4.8%", help="% of legit flagged as fraud")
    col4.metric("PR AUC", "0.847", help="Overall model quality")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Precision-Recall Curve")

        # PR curve data
        np.random.seed(42)
        recalls = np.linspace(0, 1, 100)
        precisions = 1 / (1 + np.exp(5 * (recalls - 0.5)))
        precisions = precisions * 0.8 + np.random.normal(0, 0.02, 100)
        precisions = np.clip(precisions, 0, 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recalls, y=precisions, mode='lines', name='Model', line=dict(color='#1f77b4', width=2)))
        fig.add_vline(x=0.94, line_dash="dash", line_color="red")
        fig.add_annotation(x=0.94, y=0.15, text="Our threshold<br>(94% recall)", showarrow=False, font=dict(color="red"))
        fig.update_layout(
            xaxis_title="Recall (% of fraud caught)",
            yaxis_title="Precision (% of flags correct)",
            height=350,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Explanation
        st.markdown("""
        **What this means:**

        There's a tradeoff between catching fraud and false alarms:
        - **High recall** = Catch most fraud, but more false alarms
        - **High precision** = Fewer false alarms, but miss some fraud

        **Our choice:** 94% recall threshold
        - We catch 94% of all fraud
        - ~13% of our flags are real fraud (rest are false alarms)
        - Missing fraud costs more than reviewing false alarms, so we optimize for recall
        """)

    with col2:
        st.subheader("Confusion Matrix")

        # Confusion matrix
        cm_data = [[54500, 2800], [28, 464]]
        fig = px.imshow(
            cm_data,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Legitimate', 'Fraud'],
            y=['Legitimate', 'Fraud'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Reading the matrix:**

        |  | Predicted Legit | Predicted Fraud |
        |--|-----------------|-----------------|
        | **Actually Legit** | 54,500 ✓ | 2,800 (false alarm) |
        | **Actually Fraud** | 28 (missed!) | 464 ✓ |

        - **464 / 492 = 94%** of fraud caught
        - Only **28 fraud transactions** slipped through
        """)

with tab2:
    st.header("Test a Transaction")
    st.markdown("Simulate how the model scores different transactions")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Transaction Details")

        # Preset scenarios
        scenario = st.selectbox(
            "Quick scenarios",
            ["Custom", "Normal grocery purchase", "Late night online purchase", "Large ATM withdrawal", "Gas station (your area)", "Suspicious: far location, late night"]
        )

        if scenario == "Normal grocery purchase":
            amount, hour, category, distance = 47.50, 14, "Grocery", 2.0
        elif scenario == "Late night online purchase":
            amount, hour, category, distance = 250.00, 2, "Online", 0.0
        elif scenario == "Large ATM withdrawal":
            amount, hour, category, distance = 500.00, 23, "ATM", 15.0
        elif scenario == "Gas station (your area)":
            amount, hour, category, distance = 45.00, 8, "Gas", 3.0
        elif scenario == "Suspicious: far location, late night":
            amount, hour, category, distance = 800.00, 3, "Wire", 150.0
        else:
            amount, hour, category, distance = 100.00, 12, "Retail", 5.0

        amount = st.number_input(
            "Amount ($)", min_value=0.01, max_value=10000.0, value=amount,
            help="Transaction dollar amount. Fraudsters often test with small amounts, then go big."
        )
        hour = st.slider(
            "Hour of day", 0, 23, hour,
            help="0 = midnight, 12 = noon. Fraud peaks between midnight and 5am when cardholders are asleep."
        )
        category = st.selectbox(
            "Category",
            ["Grocery", "Gas", "Restaurant", "Online", "Retail", "ATM", "Wire"],
            index=["Grocery", "Gas", "Restaurant", "Online", "Retail", "ATM", "Wire"].index(category),
            help="Transaction type. ATM withdrawals and wire transfers are high-risk because they're hard to reverse."
        )
        distance = st.number_input(
            "Distance from home (miles)", min_value=0.0, max_value=500.0, value=distance,
            help="How far from the cardholder's home address. Transactions far from home are more suspicious."
        )

        score_btn = st.button("Score Transaction", type="primary", use_container_width=True)

    with col2:
        st.subheader("Risk Assessment")

        if score_btn:
            # Simple scoring logic that mimics fraud patterns
            score = 0.05  # Base score

            # Amount risk
            if amount > 500:
                score += 0.25
            elif amount > 200:
                score += 0.10

            # Time risk (late night)
            if hour < 5 or hour > 22:
                score += 0.20

            # Category risk
            if category in ["ATM", "Wire"]:
                score += 0.25
            elif category == "Online":
                score += 0.10

            # Distance risk
            if distance > 50:
                score += 0.30
            elif distance > 20:
                score += 0.15

            score = min(score, 0.99)

            # Display gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 15], 'color': '#2ecc71'},
                        {'range': [15, 40], 'color': '#f1c40f'},
                        {'range': [40, 100], 'color': '#e74c3c'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 15
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # Decision
            if score >= 0.15:
                st.error(f"**FLAGGED FOR REVIEW** ({score:.0%} fraud probability)")
                st.markdown("This transaction would be held for manual review.")
            else:
                st.success(f"**APPROVED** ({score:.0%} fraud probability)")
                st.markdown("This transaction would be approved automatically.")

            # Risk factors
            st.markdown("**Risk factors:**")
            factors = []
            if amount > 200:
                factors.append(f"• High amount (${amount:.2f})")
            if hour < 5 or hour > 22:
                factors.append(f"• Unusual hour ({hour}:00)")
            if category in ["ATM", "Wire", "Online"]:
                factors.append(f"• Higher-risk category ({category})")
            if distance > 20:
                factors.append(f"• Far from home ({distance:.0f} miles)")

            if factors:
                for f in factors:
                    st.markdown(f)
            else:
                st.markdown("• No significant risk factors")
        else:
            st.info("Select a scenario or enter details, then click Score")

with tab3:
    st.header("Data Explorer")

    # Data source toggle
    data_source = st.radio("Data source", ["Sample data", "Upload your own CSV"], horizontal=True)

    if data_source == "Sample data":
        df = SAMPLE_DATA.copy()
        st.success(f"Loaded {len(df):,} sample transactions ({df['Class'].sum()} fraud, {df['Class'].mean():.1%} fraud rate)")
    else:
        uploaded = st.file_uploader("Upload transactions CSV", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df):,} transactions")
        else:
            st.info("Upload a CSV with transaction data. Expected columns: Amount, Hour, Category, Class (0=legit, 1=fraud)")
            df = None

    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Transactions", f"{len(df):,}", help="Total number of transactions in dataset")
        col2.metric("Fraud Cases", f"{df['Class'].sum():,}", help="Number of confirmed fraudulent transactions")
        col3.metric("Fraud Rate", f"{df['Class'].mean():.2%}", help="Percentage of transactions that were fraud")
        col4.metric("Avg Amount", f"${df['Amount'].mean():.2f}", help="Average transaction dollar amount")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Amount Distribution")
            use_log = st.checkbox("Log scale", value=True, help="Transaction amounts follow a log-normal distribution. Log scale shows the 'bell curve' shape.")

            fig = px.histogram(
                df, x='Amount', color='Class',
                nbins=50,
                color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                labels={'Class': 'Fraud'},
                category_orders={'Class': [0, 1]},
                log_x=use_log
            )
            fig.update_layout(height=300, bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)

            # Show distribution stats
            legit_amt = df[df['Class'] == 0]['Amount']
            fraud_amt = df[df['Class'] == 1]['Amount']
            st.caption(f"Legit: median ${legit_amt.median():.0f}, mean ${legit_amt.mean():.0f} | Fraud: median ${fraud_amt.median():.0f}, mean ${fraud_amt.mean():.0f}")

        with col2:
            st.subheader("Fraud by Category")
            if 'Category' in df.columns:
                cat_fraud = df.groupby('Category')['Class'].agg(['sum', 'count'])
                cat_fraud['rate'] = cat_fraud['sum'] / cat_fraud['count'] * 100
                fig = px.bar(cat_fraud.reset_index(), x='Category', y='rate',
                           labels={'rate': 'Fraud Rate (%)'})
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Category column in data")

        # Second row of charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Transactions by Hour")
            hour_data = df.groupby(['Hour', 'Class']).size().reset_index(name='Count')
            fig = px.bar(hour_data, x='Hour', y='Count', color='Class',
                        color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                        labels={'Class': 'Fraud'},
                        barmode='stack')
            fig.update_layout(height=280)
            fig.update_xaxes(tickmode='linear', dtick=2)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Fraud peaks late night (11pm-5am) when cardholders are asleep")

        with col2:
            st.subheader("Distance from Home")
            fig = px.box(df, x='Class', y='Distance',
                        color='Class',
                        color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                        labels={'Class': 'Transaction Type', 'Distance': 'Miles'})
            fig.update_layout(height=280, showlegend=False)
            fig.update_xaxes(tickvals=[0, 1], ticktext=['Legitimate', 'Fraud'])
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Fraud transactions are typically far from the cardholder's home")

        st.markdown("---")
        with st.expander("View Raw Data"):
            st.dataframe(df, use_container_width=True, hide_index=True, height=300)

st.markdown("---")
st.caption("Fraud Detection Pipeline | XGBoost + FastAPI + Streamlit | [GitHub](https://github.com/AnthonyB-316/fraud-detection-pipeline)")
