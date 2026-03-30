import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from data_processor import DataProcessor
from ml_models import GeopoliticalMLModels
from agent_ai import GeopoliticalAIAgent
import warnings
warnings.filterwarnings('ignore')

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="🌍 Global Superpower & Geopolitical Risk Nexus",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; }
    .metric-card { 
        background-color: #1f77b4; 
        padding: 20px; 
        border-radius: 10px; 
        color: white; 
        text-align: center;
    }
    .risk-high { color: #ff0000; font-weight: bold; }
    .risk-medium { color: #ff9500; font-weight: bold; }
    .risk-low { color: #00cc00; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ============ SESSION STATE ============
@st.cache_resource
def load_pipeline():
    """Load data, models, and agent"""
    processor = DataProcessor('global.csv')
    processor.load_data()
    processor.handle_missing_values()
    
    ml_models = GeopoliticalMLModels()
    agent = GeopoliticalAIAgent(processor.df, ml_models, processor)
    
    return processor, ml_models, agent

processor, ml_models, agent = load_pipeline()

# ============ SIDEBAR NAVIGATION ============
st.sidebar.title("🔬 Navigation")
page = st.sidebar.radio("Select Page", [
    "📊 Dashboard Overview",
    "🤖 AI Agent Analysis",
    "📈 ML Model Predictions",
    "🚨 Anomaly Detection",
    "💡 Insights & Recommendations",
    "📋 Executive Summary"
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📌 Dataset Information
- **Timeframe:** Jan 2015 - Mar 2026
- **Trading Days:** 4,051
- **Features:** 115
- **Markets:** Defense, Commodities, Indices
- **Geopolitical Terms:** 15 strategic flashpoints
""")

# ============ PAGE: DASHBOARD OVERVIEW ============
if page == "📊 Dashboard Overview":
    st.title("🌍 Global Superpower & Geopolitical Risk Nexus")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📅 Total Days", f"{processor.df.shape[0]:,}")
    with col2:
        st.metric("🎯 Features", processor.df.shape[1])
    with col3:
        st.metric("📍 Start Date", processor.df.index[0].strftime('%Y-%m-%d'))
    with col4:
        st.metric("🔚 End Date", processor.df.index[-1].strftime('%Y-%m-%d'))
    
    st.markdown("---")
    
    # Financial Assets Overview
    st.subheader("📊 Financial Assets Performance")
    financial_data = processor.get_financial_features()
    
    close_prices = pd.DataFrame()
    for col in financial_data.columns:
        if '_Close' in col:
            asset_name = col.replace('_Close', '')
            close_prices[asset_name] = financial_data[col]
    
    # Normalize for visualization
    close_prices_normalized = close_prices / close_prices.iloc[0] * 100
    
    fig = go.Figure()
    for col in close_prices_normalized.columns:
        fig.add_trace(go.Scatter(
            x=close_prices_normalized.index,
            y=close_prices_normalized[col],
            name=col,
            mode='lines'
        ))
    
    fig.update_layout(
        title="Normalized Financial Asset Performance (Base = 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price Index",
        hovermode='x unified',
        height=500,
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Geopolitical Metrics
    st.subheader("🌐 Geopolitical Attention Metrics (Last 90 Days)")
    geo_data = processor.get_geopolitical_features()
    
    views_cols = [col for col in geo_data.columns if '_Views' in col]
    geo_views = geo_data[views_cols].tail(90)
    
    fig2 = go.Figure()
    for col in geo_views.columns:
        fig2.add_trace(go.Scatter(
            x=geo_views.index,
            y=geo_views[col],
            name=col.replace('_Views', ''),
            mode='lines'
        ))
    
    fig2.update_layout(
        title="Wikipedia Pageviews - Geopolitical Terms",
        xaxis_title="Date",
        yaxis_title="Daily Views",
        hovermode='x unified',
        height=500,
        template='plotly_dark'
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============ PAGE: AI AGENT ANALYSIS ============
elif page == "🤖 AI Agent Analysis":
    st.title("🤖 Agentic AI Analysis Engine")
    
    st.markdown("""
    This intelligent agent performs multi-dimensional analysis of geopolitical factors 
    and their impact on global markets.
    """)
    
    # Trend Analysis
    st.subheader("📈 Geopolitical Trend Analysis (30-Day Lookback)")
    trends = agent.analyze_geopolitical_trends(lookback_days=30)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        high_alerts = len([a for a in trends['alerts'] if 'HIGH' in a])
        st.metric("🚨 High Risk Metrics", high_alerts)
    with col2:
        total_metrics = len(trends['trends'])
        st.metric("📊 Metrics Analyzed", total_metrics)
    with col3:
        st.metric("⏱️ Analysis Period", "30 days")
    
    # Alert Table
    if trends['alerts']:
        st.warning("⚠️ **Active Alerts:**")
        for alert in trends['alerts']:
            st.write(f"• {alert}")
    
    # Market-Geopolitical Linkage
    st.subheader("🔗 Market-Geopolitical Linkage Analysis")
    linkage = agent.detect_market_financial_linkage()
    
    if linkage['correlations']:
        linkage_df = pd.DataFrame([
            {
                'Relationship': k,
                'Correlation': v['correlation'],
                'Strength': v['strength']
            }
            for k, v in linkage['correlations'].items()
        ])
        st.dataframe(linkage_df, use_container_width=True)
    else:
        st.info("No significant correlations detected in current period")
    
    # Market Stress Prediction
    st.subheader("⚡ Market Stress Scenario Prediction")
    stress = agent.predict_market_stress(days_ahead=5)
    
    risk_color = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
    st.metric("Risk Level", f"{risk_color.get(stress['risk_level'], '❓')} {stress['risk_level']}")
    
    for scenario in stress['stress_scenarios']:
        with st.expander(f"📌 {scenario['scenario']}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Probability", f"{scenario['probability']:.0%}")
            with col2:
                st.metric("Impact", scenario['impact'])
            with col3:
                st.write("")
            st.info(f"**Recommendation:** {scenario['recommendation']}")

# ============ PAGE: ML MODEL PREDICTIONS ============
elif page == "📈 ML Model Predictions":
    st.title("📈 Machine Learning Model Performance")
    
    st.subheader("🔧 Model Training & Evaluation")
    
    # Prepare data
    financial_data = processor.get_financial_features()
    close_cols = [col for col in financial_data.columns if '_Close' in col]
    X = financial_data[close_cols].fillna(method='ffill')
    y = X.mean(axis=1)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    col1, col2 = st.columns(2)
    
    # Model 1: Volatility Predictor
    with col1:
        with st.spinner("Training Volatility Predictor..."):
            model1, metrics1 = ml_models.build_volatility_predictor(X_train, y_train, X_test, y_test)
        
        st.metric("✅ Random Forest (Volatility)", "Trained")
        st.write(f"R² Score: {metrics1['r2']:.4f}")
        st.write(f"RMSE: {metrics1['rmse']:.4f}")
        st.write(f"MAE: {metrics1['mae']:.4f}")
    
    # Model 2: Market Shock Detector
    with col2:
        with st.spinner("Training Market Shock Detector..."):
            model2, metrics2 = ml_models.build_market_shock_detector(X_train, y_train, X_test, y_test)
        
        st.metric("✅ Gradient Boosting (Market Shock)", "Trained")
        st.write(f"R² Score: {metrics2['r2']:.4f}")
        st.write(f"RMSE: {metrics2['rmse']:.4f}")
        st.write(f"MAE: {metrics2['mae']:.4f}")
    
    st.markdown("---")
    
    # Model Predictions Visualization
    st.subheader("🎯 Model Predictions vs Actual")
    
    pred_df = pd.DataFrame({
        'Date': X_test.index,
        'Actual': y_test.values,
        'RF Prediction': metrics1['predictions'],
        'GB Prediction': metrics2['predictions']
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Actual'], name='Actual', mode='lines'))
    fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['RF Prediction'], name='RF Prediction', mode='lines', opacity=0.7))
    fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['GB Prediction'], name='GB Prediction', mode='lines', opacity=0.7))
    
    fig.update_layout(
        title="Model Predictions Performance",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        height=500,
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)

# ============ PAGE: ANOMALY DETECTION ============
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Geopolitical Anomaly Detection System")
    
    st.subheader("Detection Results")
    
    # Detect anomalies
    anomalies = agent.anomaly_alert_system()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔍 Anomalies Detected", anomalies['total_anomalies_detected'])
    with col2:
        st.metric("📍 Recent Anomalies", len(anomalies['recent_anomalies']))
    with col3:
        st.metric("⚠️ Action Items", len(anomalies['action_items']))
    
    # Recent Anomalies Table
    if anomalies['recent_anomalies']:
        st.subheader("📋 Recent Anomalous Events")
        anomaly_df = pd.DataFrame(anomalies['recent_anomalies'])
        st.dataframe(anomaly_df, use_container_width=True)
    
    # Action Items
    st.subheader("✅ Recommended Action Items")
    for action in anomalies['action_items']:
        st.write(f"• {action}")
    
    # Anomaly Score Distribution
    st.subheader("📊 Anomaly Score Distribution")
    ml_models.build_anomaly_detector(processor.df.fillna(method='ffill'))
    anomaly_detection = ml_models.detect_geopolitical_anomalies(processor.df)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=anomaly_detection['anomaly_scores'],
        nbinsx=50,
        name='Anomaly Scores'
    ))
    fig.update_layout(
        title="Distribution of Anomaly Scores",
        xaxis_title="Anomaly Score",
        yaxis_title="Frequency",
        height=400,
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)

# ============ PAGE: INSIGHTS & RECOMMENDATIONS ============
elif page == "💡 Insights & Recommendations":
    st.title("💡 Strategic Insights & Recommendations")
    
    st.subheader("🎯 Sector Recommendations")
    recommendations = agent.sector_recommendation_engine()
    
    for sector, data in recommendations.items():
        if sector != 'timestamp':
            rating_color = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡', 'NEUTRAL': '⚪'}
            with st.expander(f"{rating_color.get(data['rating'], '❓')} {sector.upper()} - {data['rating']}"):
                st.write(f"**Analysis:** {data['reasoning']}")
    
    st.markdown("---")
    
    # Memory Log
    st.subheader("📝 Agent Memory Log")
    memory = agent.get_memory_log()
    if memory:
        latest_memory = memory[-1]
        st.write(f"**Timestamp:** {latest_memory['timestamp']}")
        st.write(f"**Analysis Period:** {latest_memory['lookback_days']} days")
        st.metric("Metrics Analyzed", len(latest_memory['trends']))
    
    st.markdown("---")
    
    # Insights Log
    st.subheader("🔬 Generated Insights")
    insights = agent.get_insights_log()
    st.write(f"Total Insights Generated: {len(insights)}")

# ============ PAGE: EXECUTIVE SUMMARY ============
elif page == "📋 Executive Summary":
    st.title("📊 Executive Summary Report")
    
    summary = agent.executive_summary()
    
    st.markdown(f"""
    ### 📈 Report Details
    - **Report Date:** {summary['report_date']}
    - **Data Coverage:** {summary['data_coverage']}
    - **Total Observations:** {summary['total_observations']:,}
    - **Total Features:** {summary['total_features']}
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🧠 Memory Events", summary['memory_events'])
    with col2:
        st.metric("💡 Insights Generated", summary['insights_generated'])
    with col3:
        st.metric("⚠️ Top Risks Identified", len(summary['top_risks']))
    
    st.markdown("---")
    
    # Key Findings
    st.subheader("🔍 Key Findings")
    for finding in summary['key_findings']:
        st.write(f"• {finding}")
    
    # Top Risks
    st.subheader("⛔ Top Identified Risks")
    if summary['top_risks']:
        for risk in summary['top_risks']:
            st.warning(f"• {risk}")
    else:
        st.info("No critical risks identified in current period")
    
    # Opportunities
    st.subheader("🚀 Identified Opportunities")
    if summary['opportunities']:
        for opp in summary['opportunities']:
            st.success(f"• {opp}")
    else:
        st.info("Monitor geopolitical developments for emerging opportunities")
    
    st.markdown("---")
    st.markdown("*Report Generated by Geopolitical AI Agent*")

# ============ FOOTER ============
st.markdown("""
---
### 🔗 Data Sources
- **Markets:** Yahoo Finance API
- **Geopolitical Metrics:** Wikimedia Foundation REST API
- **Analysis Framework:** Machine Learning + Agentic AI

*This application integrates real-time market data with geopolitical attention metrics to provide strategic risk assessment and opportunity identification.*
""")