import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Fraud Operations Dashboard", page_icon="🛡️", layout="wide")

# Custom Dark Theme CSS
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #ffffff;}
    .stMetric {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    explainer = shap.TreeExplainer(model)
    return model, scaler, explainer

@st.cache_data
def load_data():
    df = pd.read_csv('dashboard_data.csv')
    return df

try:
    model, scaler, explainer = load_model_and_scaler()
    df = load_data()
except Exception as e:
    st.error(f"Error loading assets. Ensure model.pkl, scaler.pkl, and dashboard_data.csv are in this folder. Error: {e}")
    st.stop()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("🛡️ FraudOps Portal")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Overview", "Transaction Explorer", "SHAP Explainer"])
st.sidebar.markdown("---")
st.sidebar.info("System Status: **ONLINE**\n\nModel: XGBoost Optimized")

# --- 4. PAGE 1: OVERVIEW ---
if page == "Overview":
    st.title("Global Fraud Overview")
    
    # Calculate KPIs
    total_tx = len(df)
    total_fraud = len(df[df['isFraud'] == 1])
    detection_rate = (total_fraud / total_tx) * 100
    avg_fraud_amt = df[df['isFraud'] == 1]['TransactionAmt'].mean() if total_fraud > 0 else 0
    
    # Render KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions Monitored", f"{total_tx:,}")
    col2.metric("Critical Fraud Alerts", f"{total_fraud:,}", delta_color="inverse")
    col3.metric("Current Detection Rate", f"{detection_rate:.2f}%")
    col4.metric("Avg Fraud Amount", f"${avg_fraud_amt:.2f}")
    
    st.markdown("### Interactive Threat Intelligence")
    c1, c2 = st.columns(2)
    
    with c1:
        # Donut Chart for Risk Tiers
        tier_counts = df['Risk_Tier'].value_counts().reset_index()
        tier_counts.columns = ['Risk Tier', 'Count']
        fig_donut = px.pie(tier_counts, values='Count', names='Risk Tier', hole=0.6, 
                           title='Transaction Volume by Risk Tier',
                           color='Risk Tier',
                           color_discrete_map={'Clear':'#2ecc71', 'Suspicious':'#f1c40f', 'Critical Risk':'#e74c3c'})
        fig_donut.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        # Interactive Scatter: Amount vs Hour
        fig_scatter = px.scatter(df, x='HourOfDay', y='TransactionAmt', color='Risk_Tier',
                                 title='Transaction Amount vs. Hour of Day',
                                 color_discrete_map={'Clear':'#2ecc71', 'Suspicious':'#f1c40f', 'Critical Risk':'#e74c3c'},
                                 hover_data=['TransactionAmt', 'Fraud_Probability'])
        fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        fig_scatter.update_yaxes(type="log") # Log scale for better visibility
        st.plotly_chart(fig_scatter, use_container_width=True)


# --- 5. PAGE 2: TRANSACTION EXPLORER ---
elif page == "Transaction Explorer":
    st.title("Live Transaction Explorer")
    st.markdown("Filter and search through recent transactions to assess live risk scores.")
    
    # Sidebar Filters
    st.sidebar.subheader("Filters")
    selected_tier = st.sidebar.multiselect("Risk Tier", options=df['Risk_Tier'].unique(), default=df['Risk_Tier'].unique())
    min_amt = st.sidebar.slider("Minimum Amount ($)", min_value=0.0, max_value=float(df['TransactionAmt'].max()), value=0.0)
    
    # Apply Filters
    filtered_df = df[(df['Risk_Tier'].isin(selected_tier)) & (df['TransactionAmt'] >= min_amt)]
    
    # Format table for display
    display_df = filtered_df[['TransactionAmt', 'HourOfDay', 'DeviceRisk', 'Risk_Tier', 'Fraud_Probability', 'isFraud']].copy()
    display_df['Fraud_Probability'] = (display_df['Fraud_Probability'] * 100).round(2).astype(str) + "%"
    
    st.dataframe(display_df.sort_values(by='Fraud_Probability', ascending=False), use_container_width=True)


# --- 6. PAGE 3: SHAP EXPLAINER ---
elif page == "SHAP Explainer":
    st.title("Explainable AI (SHAP) Engine")
    st.markdown("Understand exactly **why** the algorithm flagged a transaction.")
    
    # We will use the index of our sample dataframe as a proxy for "TransactionID"
    st.info("Select a transaction record from the dropdown to run the Explainable AI engine.")
    
    # Dropdown for available records, sorted to show Critical Risks first
    record_options = df.sort_values('Fraud_Probability', ascending=False).index.tolist()
    selected_record = st.selectbox("Select Transaction Record Index", options=record_options)
    
    if st.button("Generate AI Explanation"):
        with st.spinner("Analyzing neural pathways..."):
            # Get the raw data for the selected record
            record_data = df.loc[selected_record]
            
            # Reconstruct the feature array required by the model (drop the target/analysis columns)
            # Drop the analysis columns AND the original ID/Time columns
            features = record_data.drop(['isFraud', 'Fraud_Probability', 'Risk_Tier', 'TransactionID', 'TransactionDT'], errors='ignore')
            
            # Force numeric conversion and reshape
            features = pd.to_numeric(features, errors='coerce').fillna(0).values.reshape(1, -1)
            
            # Scale the features
            scaled_features = scaler.transform(features)
            
            # Calculate SHAP values
            shap_values = explainer(scaled_features)
            
            probability = record_data['Fraud_Probability']
            
            st.markdown(f"### Transaction Risk Score: **{probability*100:.1f}%**")
            
            # Plain English Explanation
            if probability >= 0.75:
                st.error("🚨 **CRITICAL RISK:** The model is highly confident this is a fraudulent transaction. Look at the red bars below to see which variables drove this score up.")
            elif probability >= 0.40:
                st.warning("⚠️ **SUSPICIOUS:** This transaction shows irregular patterns. Manual review recommended.")
            else:
                st.success("✅ **CLEAR:** This transaction aligns with legitimate customer behavior.")
            
            # Render SHAP Waterfall Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            # Set matplotlib styling to match dark theme
            plt.style.use('dark_background')
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig)