import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from data_processor import DataProcessor
from ml_models import GeopoliticalMLModels
from agent_ai import GeopoliticalAIAgent
from chatbot import GeopoliticalChatbot
from data_search import DataSearchEngine, ManualDataInput
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
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 100%;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        color: white;
        margin-right: 20%;
        border: 1px solid #4a5568;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #4a5568;
        border-radius: 10px;
        background-color: #1a202c;
    }
    .quick-action-btn {
        margin: 5px 0;
    }
    .search-result-card {
        background-color: #2d3748;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
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
    chatbot = GeopoliticalChatbot(processor, ml_models, agent)
    search_engine = DataSearchEngine(processor)
    manual_input = ManualDataInput(processor)
    
    return processor, ml_models, agent, chatbot, search_engine, manual_input

processor, ml_models, agent, chatbot, search_engine, manual_input = load_pipeline()

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'search_results' not in st.session_state:
    st.session_state.search_results = None

if 'current_search_tab' not in st.session_state:
    st.session_state.current_search_tab = 0

# ============ SIDEBAR NAVIGATION ============
st.sidebar.title("🔬 Navigation")
page = st.sidebar.radio("Select Page", [
    "📊 Dashboard Overview",
    "💬 AI Chatbot",
    "🔍 Data Search & Input",
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

# Quick Stats in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
quick_stats = chatbot.get_quick_stats()
st.sidebar.write(f"📅 Latest Data: {quick_stats['latest_date']}")
st.sidebar.write(f"📈 Records: {quick_stats['total_records']:,}")
st.sidebar.write(f"🎯 Features: {quick_stats['features']}")

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

# ============ PAGE: AI CHATBOT ============
elif page == "💬 AI Chatbot":
    st.title("💬 Geopolitical AI Chatbot")
    
    st.markdown("""
    Welcome to the **Geopolitical Risk Analysis Chatbot**! Ask me anything about:
    - 📊 Market data and prices
    - 🌐 Geopolitical risks and trends
    - 🔮 Predictions and forecasts
    - 💡 Investment recommendations
    """)
    
    # Chat interface layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat history display
        st.markdown("### 💬 Conversation")
        
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            if not st.session_state.chat_history:
                st.info("👋 Start a conversation by typing a message below or using the quick action buttons!")
            
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong><br>{message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Format assistant message with proper line breaks
                    formatted_content = message['content'].replace('\n', '<br>')
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant:</strong><br>{formatted_content}
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Input area with form for better UX
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Type your message...",
                key="chat_input",
                placeholder="Ask about markets, risks, predictions... (Press Enter to send)"
            )
            
            col_send, col_clear, col_export = st.columns([1, 1, 1])
            
            with col_send:
                submit_button = st.form_submit_button("📤 Send", use_container_width=True)
            
            with col_clear:
                pass  # Will add clear button outside form
            
            with col_export:
                pass  # Will add export button outside form
        
        # Process message if submitted
        if submit_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Get chatbot response
            response = chatbot.process_message(user_input)
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            
            # Rerun to update display
            st.rerun()
        
        # Clear and Export buttons outside form
        col_clear, col_export = st.columns(2)
        
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                chatbot.clear_history()
                st.rerun()
        
        with col_export:
            if st.session_state.chat_history:
                chat_export = "\n\n".join([
                    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                    for m in st.session_state.chat_history
                ])
                st.download_button(
                    "📥 Export Chat",
                    chat_export,
                    "chat_history.txt",
                    "text/plain",
                    use_container_width=True
                )
    
    with col2:
        st.markdown("### 🚀 Quick Actions")
        
        # Quick action buttons
        quick_actions = [
            ("📊 Market Summary", "Show me a market summary"),
            ("🚨 Risk Analysis", "What are the current risk levels?"),
            ("🔮 Predictions", "Predict market stress for next 5 days"),
            ("💡 Recommendations", "Give me sector recommendations"),
            ("🚨 Anomalies", "Detect any anomalies"),
            ("📈 Gold Price", "What's the current gold price?"),
            ("🛢️ Oil Analysis", "Analyze oil market trends"),
            ("🌐 NATO Risk", "What's the NATO risk level?"),
            ("📋 Summary", "Generate executive summary")
        ]
        
        for action_name, action_query in quick_actions:
            if st.button(action_name, use_container_width=True, key=f"quick_{action_name}"):
                st.session_state.chat_history.append({'role': 'user', 'content': action_name})
                response = chatbot.process_message(action_query)
                st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        st.markdown("""
        - What's the gold price?
        - Compare gold vs oil
        - NATO risk level?
        - Show SP500 trend
        - Analyze China tensions
        - Defense sector outlook
        - Show last week's data
        - Market volatility report
        """)
        
        st.markdown("---")
        st.markdown("### 📈 Chat Statistics")
        st.metric("Messages", len(st.session_state.chat_history))
        st.metric("User Queries", len([m for m in st.session_state.chat_history if m['role'] == 'user']))

# ============ PAGE: DATA SEARCH & INPUT ============
elif page == "🔍 Data Search & Input":
    st.title("🔍 Data Search & Manual Input")
    
    st.markdown("""
    Search and explore the geopolitical and financial dataset with powerful filtering options.
    You can also create custom what-if scenarios for analysis.
    """)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Search by Date", 
        "🏷️ Search by Asset", 
        "🌐 Geopolitical Search",
        "🔢 Value Search",
        "✏️ Manual Input"
    ])
    
    # TAB 1: Search by Date
    with tab1:
        st.subheader("📅 Search by Date")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📍 Single Date Search")
            search_date = st.date_input(
                "Select Date",
                value=processor.df.index[-1].date(),
                min_value=processor.df.index[0].date(),
                max_value=processor.df.index[-1].date(),
                key="single_date_search"
            )
            
            # Column filter
            show_all_cols = st.checkbox("Show all columns", value=False, key="show_all_single")
            
            if st.button("🔍 Search Date", key="btn_search_date"):
                result = search_engine.search_by_date(search_date)
                if result['success']:
                    if result.get('nearest'):
                        st.warning(f"Exact date not found. Showing nearest date: {result['date'].date()}")
                    else:
                        st.success(f"✅ Data found for {result['date'].date()}")
                    
                    if show_all_cols:
                        # Show all data
                        df_display = pd.DataFrame([result['data']])
                        st.dataframe(df_display.T, use_container_width=True)
                    else:
                        # Display financial data
                        st.markdown("**📊 Financial Assets (Close Prices):**")
                        financial_cols = {k.replace('_Close', ''): f"${v:,.2f}" 
                                        for k, v in result['data'].items() if '_Close' in k}
                        if financial_cols:
                            fin_df = pd.DataFrame([financial_cols])
                            st.dataframe(fin_df, use_container_width=True)
                        
                        # Display geopolitical data
                        st.markdown("**🌐 Geopolitical Metrics (Page Views):**")
                        geo_cols = {k.replace('_Views', ''): f"{v:,.0f}" 
                                   for k, v in result['data'].items() if '_Views' in k}
                        if geo_cols:
                            geo_df = pd.DataFrame([geo_cols])
                            st.dataframe(geo_df, use_container_width=True)
                        
                        # Display shock metrics
                        st.markdown("**⚡ Shock Indicators:**")
                        shock_cols = {k.replace('_Shock', ''): f"{v:.4f}" 
                                     for k, v in result['data'].items() if '_Shock' in k}
                        if shock_cols:
                            shock_df = pd.DataFrame([shock_cols])
                            st.dataframe(shock_df, use_container_width=True)
                else:
                    st.error(f"❌ {result['error']}")
        
        with col2:
            st.markdown("#### 📆 Date Range Search")
            
            date_range_col1, date_range_col2 = st.columns(2)
            with date_range_col1:
                start_date = st.date_input(
                    "Start Date",
                    value=processor.df.index[-30].date(),
                    min_value=processor.df.index[0].date(),
                    max_value=processor.df.index[-1].date(),
                    key="range_start_date"
                )
            with date_range_col2:
                end_date = st.date_input(
                    "End Date",
                    value=processor.df.index[-1].date(),
                    min_value=processor.df.index[0].date(),
                    max_value=processor.df.index[-1].date(),
                    key="range_end_date"
                )
            
            # Column type filter
            col_type = st.selectbox(
                "Select Data Type",
                ["Close Prices", "Volumes", "Geopolitical Views", "Shock Indicators", "All"],
                key="date_range_col_type"
            )
            
            if st.button("🔍 Search Range", key="btn_search_range"):
                result = search_engine.search_by_date_range(start_date, end_date)
                if result['success']:
                    st.success(f"✅ Found {result['records']} records from {start_date} to {end_date}")
                    
                    # Filter columns based on selection
                    if col_type == "Close Prices":
                        display_cols = [col for col in result['data'].columns if '_Close' in col]
                    elif col_type == "Volumes":
                        display_cols = [col for col in result['data'].columns if '_Volume' in col]
                    elif col_type == "Geopolitical Views":
                        display_cols = [col for col in result['data'].columns if '_Views' in col]
                    elif col_type == "Shock Indicators":
                        display_cols = [col for col in result['data'].columns if '_Shock' in col]
                    else:
                        display_cols = result['data'].columns.tolist()
                    
                    if display_cols:
                        display_data = result['data'][display_cols]
                        st.dataframe(display_data, use_container_width=True)
                        
                        # Summary statistics
                        st.markdown("**📈 Summary Statistics:**")
                        st.dataframe(display_data.describe(), use_container_width=True)
                        
                        # Download option
                        csv = display_data.to_csv()
                        st.download_button(
                            "📥 Download Results as CSV",
                            csv,
                            f"search_results_{start_date}_{end_date}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    else:
                        st.warning("No columns found for selected data type")
                else:
                    st.error(f"❌ {result['error']}")
    
    # TAB 2: Search by Asset
    with tab2:
        st.subheader("🏷️ Search by Asset")
        
        available_assets = search_engine.get_available_assets()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_asset = st.selectbox(
                "Select Asset",
                available_assets,
                key="asset_select"
            )
        
        with col2:
            st.markdown(f"**{len(available_assets)}** assets available")
        
        # Optional date filter
        use_date_filter = st.checkbox("Apply Date Filter", key="asset_date_filter")
        
        if use_date_filter:
            col1, col2 = st.columns(2)
            with col1:
                asset_start = st.date_input(
                    "From", 
                    value=processor.df.index[-90].date(),
                    min_value=processor.df.index[0].date(),
                    max_value=processor.df.index[-1].date(),
                    key="asset_start"
                )
            with col2:
                asset_end = st.date_input(
                    "To", 
                    value=processor.df.index[-1].date(),
                    min_value=processor.df.index[0].date(),
                    max_value=processor.df.index[-1].date(),
                    key="asset_end"
                )
        else:
            asset_start, asset_end = None, None
        
        if st.button("🔍 Search Asset", key="btn_search_asset"):
            result = search_engine.search_by_asset(selected_asset, asset_start, asset_end)
            
            if result['success']:
                st.success(f"✅ Found data for **{selected_asset}**")
                st.markdown(f"**Available columns:** {', '.join(result['columns'])}")
                
                # Display recent data
                st.markdown("### 📊 Recent Data (Last 30 Days)")
                st.dataframe(result['data'].tail(30), use_container_width=True)
                
                # Statistics
                st.markdown("### 📈 Statistics")
                close_col = f"{selected_asset}_Close"
                if close_col in result['data'].columns:
                    stats = search_engine.get_statistics(close_col)
                    if stats['success']:
                        stat_cols = st.columns(5)
                        with stat_cols[0]:
                            st.metric("Mean", f"${stats['statistics']['mean']:,.2f}")
                        with stat_cols[1]:
                            st.metric("Std Dev", f"${stats['statistics']['std']:,.2f}")
                        with stat_cols[2]:
                            st.metric("Min", f"${stats['statistics']['min']:,.2f}")
                        with stat_cols[3]:
                            st.metric("Max", f"${stats['statistics']['max']:,.2f}")
                        with stat_cols[4]:
                            st.metric("Median", f"${stats['statistics']['median']:,.2f}")
                
                # Visualization
                st.markdown("### 📉 Price Chart")
                fig = go.Figure()
                
                if f"{selected_asset}_Close" in result['data'].columns:
                    fig.add_trace(go.Scatter(
                        x=result['data'].index,
                        y=result['data'][f"{selected_asset}_Close"],
                        name='Close Price',
                        mode='lines',
                        line=dict(color='#00d4ff', width=2)
                    ))
                
                # Add moving averages
                if len(result['data']) > 20:
                    ma_20 = result['data'][f"{selected_asset}_Close"].rolling(20).mean()
                    fig.add_trace(go.Scatter(
                        x=result['data'].index,
                        y=ma_20,
                        name='20-Day MA',
                        mode='lines',
                        line=dict(color='#ff6b6b', width=1, dash='dash')
                    ))
                
                fig.update_layout(
                    title=f"{selected_asset} Price History",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template='plotly_dark',
                    height=450,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart if available
                if f"{selected_asset}_Volume" in result['data'].columns:
                    st.markdown("### 📊 Volume Chart")
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(
                        x=result['data'].index,
                        y=result['data'][f"{selected_asset}_Volume"],
                        name='Volume',
                        marker_color='#667eea'
                    ))
                    fig_vol.update_layout(
                        title=f"{selected_asset} Trading Volume",
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        template='plotly_dark',
                        height=300
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)
                
                # Download
                csv = result['data'].to_csv()
                st.download_button(
                    "📥 Download Asset Data",
                    csv,
                    f"{selected_asset}_data.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.error(f"❌ {result['error']}")
    
    # TAB 3: Search by Geopolitical Term
    with tab3:
        st.subheader("🌐 Search by Geopolitical Term")
        
        available_terms = search_engine.get_available_geopolitical_terms()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_term = st.selectbox(
                "Select Geopolitical Term",
                available_terms,
                key="geo_term_select"
            )
        
        with col2:
            st.markdown(f"**{len(available_terms)}** terms available")
        
        # Date filter
        use_geo_date_filter = st.checkbox("Apply Date Filter", key="geo_date_filter")
        
        if use_geo_date_filter:
            col1, col2 = st.columns(2)
            with col1:
                geo_start = st.date_input(
                    "From",
                    value=processor.df.index[-180].date(),
                    min_value=processor.df.index[0].date(),
                    max_value=processor.df.index[-1].date(),
                    key="geo_start"
                )
            with col2:
                geo_end = st.date_input(
                    "To",
                    value=processor.df.index[-1].date(),
                    min_value=processor.df.index[0].date(),
                    max_value=processor.df.index[-1].date(),
                    key="geo_end"
                )
        else:
            geo_start, geo_end = None, None
        
        if st.button("🔍 Search Term", key="btn_search_geo"):
            result = search_engine.search_by_geopolitical_term(selected_term, geo_start, geo_end)
            
            if result['success']:
                st.success(f"✅ Found data for **{selected_term}**")
                st.markdown(f"**Metrics available:** {', '.join(result['columns'])}")
                
                # Display data
                st.markdown("### 📊 Recent Data")
                st.dataframe(result['data'].tail(30), use_container_width=True)
                
                # Multi-panel visualization
                st.markdown("### 📈 Metrics Visualization")
                
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    subplot_titles=['📰 Wikipedia Page Views', '📈 Momentum', '⚡ Shock Index'],
                    vertical_spacing=0.08
                )
                
                if f"{selected_term}_Views" in result['data'].columns:
                    fig.add_trace(go.Scatter(
                        x=result['data'].index,
                        y=result['data'][f"{selected_term}_Views"],
                        name='Views',
                        mode='lines',
                        line=dict(color='#00d4ff', width=2)
                    ), row=1, col=1)
                
                if f"{selected_term}_Momentum" in result['data'].columns:
                    momentum_data = result['data'][f"{selected_term}_Momentum"]
                    colors = ['#00cc00' if x >= 0 else '#ff4444' for x in momentum_data]
                    fig.add_trace(go.Bar(
                        x=result['data'].index,
                        y=momentum_data,
                        name='Momentum',
                        marker_color=colors
                    ), row=2, col=1)
                
                if f"{selected_term}_Shock" in result['data'].columns:
                    fig.add_trace(go.Scatter(
                        x=result['data'].index,
                        y=result['data'][f"{selected_term}_Shock"],
                        name='Shock',
                        mode='lines',
                        fill='tozeroy',
                        line=dict(color='#ff6b6b', width=2)
                    ), row=3, col=1)
                
                fig.update_layout(
                    height=700,
                    template='plotly_dark',
                    showlegend=True,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics summary
                st.markdown("### 📊 Statistics Summary")
                for col in result['columns']:
                    if col in result['data'].columns:
                        col_stats = result['data'][col].describe()
                        with st.expander(f"📈 {col}"):
                            stats_cols = st.columns(4)
                            with stats_cols[0]:
                                st.metric("Mean", f"{col_stats['mean']:,.2f}")
                            with stats_cols[1]:
                                st.metric("Std", f"{col_stats['std']:,.2f}")
                            with stats_cols[2]:
                                st.metric("Min", f"{col_stats['min']:,.2f}")
                            with stats_cols[3]:
                                st.metric("Max", f"{col_stats['max']:,.2f}")
                
                # Download
                csv = result['data'].to_csv()
                st.download_button(
                    "📥 Download Geopolitical Data",
                    csv,
                    f"{selected_term}_data.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.error(f"❌ {result['error']}")
    
    # TAB 4: Value Search
    with tab4:
        st.subheader("🔢 Search by Value Threshold")
        
        st.markdown("Find records where specific metrics exceed or fall below thresholds.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            all_columns = list(processor.df.columns)
            search_column = st.selectbox("Select Column", all_columns, key="value_search_col")
        
        with col2:
            operator = st.selectbox("Operator", [">", "<", ">=", "<=", "=="], key="value_search_op")
        
        with col3:
            # Get column statistics for reference
            col_mean = processor.df[search_column].mean()
            threshold_value = st.number_input(
                "Threshold Value",
                value=float(col_mean),
                key="value_search_threshold"
            )
        
        # Show column statistics
        with st.expander("📊 Column Statistics"):
            col_stats = processor.df[search_column].describe()
            stat_cols = st.columns(5)
            with stat_cols[0]:
                st.metric("Mean", f"{col_stats['mean']:,.2f}")
            with stat_cols[1]:
                st.metric("Std", f"{col_stats['std']:,.2f}")
            with stat_cols[2]:
                st.metric("Min", f"{col_stats['min']:,.2f}")
            with stat_cols[3]:
                st.metric("Max", f"{col_stats['max']:,.2f}")
            with stat_cols[4]:
                st.metric("Median", f"{col_stats['50%']:,.2f}")
        
        if st.button("🔍 Search Values", key="btn_search_values"):
            result = search_engine.search_by_value_threshold(search_column, operator, threshold_value)
            
            if result['success']:
                st.success(f"✅ Found **{result['records']}** records matching: {result['query']}")
                
                if result['records'] > 0:
                    # Display results
                    st.markdown("### 📋 Matching Records")
                    st.dataframe(result['data'].head(100), use_container_width=True)
                    
                    # Visualization
                    st.markdown("### 📈 Distribution")
                    fig = go.Figure()
                    
                    # Histogram with threshold line
                    fig.add_trace(go.Histogram(
                        x=processor.df[search_column],
                        name='All Values',
                        opacity=0.7,
                        marker_color='#667eea'
                    ))
                    
                    fig.add_vline(
                        x=threshold_value,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Threshold: {threshold_value:,.2f}"
                    )
                    
                    fig.update_layout(
                        title=f"Distribution of {search_column}",
                        xaxis_title=search_column,
                        yaxis_title="Frequency",
                        template='plotly_dark',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download
                    csv = result['data'].to_csv()
                    st.download_button(
                        "📥 Download Filtered Data",
                        csv,
                        f"filtered_{search_column}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("No records found matching the criteria.")
            else:
                st.error(f"❌ {result['error']}")
        
        st.markdown("---")
        
        # Anomaly Search
        st.subheader("🚨 Anomaly Search")
        
        col1, col2 = st.columns(2)
        
        with col1:
            anomaly_column = st.selectbox(
                "Select Column for Anomaly Detection",
                all_columns,
                key="anomaly_search_col"
            )
        
        with col2:
            std_threshold = st.slider(
                "Standard Deviation Threshold",
                min_value=1.0,
                max_value=4.0,
                value=2.0,
                step=0.5,
                key="anomaly_std_threshold"
            )
        
        if st.button("🔍 Find Anomalies", key="btn_find_anomalies"):
            result = search_engine.search_anomalies(anomaly_column, std_threshold)
            
            if result['success']:
                st.success(f"✅ Found **{result['anomalies_count']}** anomalies (>{std_threshold} std deviations)")
                
                st.markdown(f"""
                **Bounds:**
                - Upper: {result['upper_bound']:,.2f}
                - Lower: {result['lower_bound']:,.2f}
                - Mean: {result['mean']:,.2f}
                - Std: {result['std']:,.2f}
                """)
                
                if result['anomalies_count'] > 0:
                    st.dataframe(result['data'][[anomaly_column]].head(50), use_container_width=True)
            else:
                st.error(f"❌ {result['error']}")
        
        st.markdown("---")
        
        # Correlation Search
        st.subheader("🔗 Correlation Search")
        
        col1, col2 = st.columns(2)
        
        with col1:
            corr_col1 = st.selectbox("First Column", all_columns, key="corr_col1")
        
        with col2:
            corr_col2 = st.selectbox("Second Column", all_columns, index=1, key="corr_col2")
        
        if st.button("🔍 Calculate Correlation", key="btn_calc_corr"):
            result = search_engine.search_correlation(corr_col1, corr_col2)
            
            if result['success']:
                corr_val = result['correlation']
                
                # Determine color based on correlation
                if corr_val > 0.7:
                    color = "🟢"
                    desc = "Strong Positive"
                elif corr_val > 0.4:
                    color = "🟡"
                    desc = "Moderate Positive"
                elif corr_val > -0.4:
                    color = "⚪"
                    desc = "Weak"
                elif corr_val > -0.7:
                    color = "🟠"
                    desc = "Moderate Negative"
                else:
                    color = "🔴"
                    desc = "Strong Negative"
                
                st.markdown(f"""
                ### {color} Correlation: **{corr_val:.4f}**
                **Strength:** {desc}
                """)
                
                # Scatter plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=processor.df[corr_col1],
                    y=processor.df[corr_col2],
                    mode='markers',
                    marker=dict(
                        color='#667eea',
                        opacity=0.5,
                        size=5
                    ),
                    name='Data Points'
                ))
                
                fig.update_layout(
                    title=f"Correlation: {corr_col1} vs {corr_col2}",
                    xaxis_title=corr_col1,
                    yaxis_title=corr_col2,
                    template='plotly_dark',
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"❌ {result['error']}")
    
    # TAB 5: Manual Input
    with tab5:
        st.subheader("✏️ Manual Data Input & What-If Analysis")
        
        st.markdown("""
        Create custom scenarios by modifying data points and analyze potential outcomes.
        This is useful for stress testing and scenario planning.
        """)
        
        st.markdown("---")
        
        # Add Custom Data Point
        st.markdown("### ➕ Add Custom Data Point")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_date = st.date_input(
                "Date",
                value=processor.df.index[-1].date(),
                min_value=processor.df.index[0].date(),
                max_value=processor.df.index[-1].date(),
                key="manual_input_date"
            )
        
        with col2:
            all_columns = list(processor.df.columns)
            input_column = st.selectbox("Column", all_columns, key="manual_input_column")
        
        with col3:
            # Show current value for reference
            try:
                current_val = processor.df.loc[pd.to_datetime(input_date), input_column]
                st.markdown(f"**Current Value:** {current_val:,.2f}")
            except:
                st.markdown("**Current Value:** N/A")
            
            input_value = st.number_input("New Value", value=0.0, key="manual_input_value")
        
        if st.button("➕ Add Data Point", key="btn_add_data_point"):
            result = manual_input.add_custom_data_point(input_date, input_column, input_value)
            if result['success']:
                st.success(f"✅ {result['message']}")
            else:
                st.error(f"❌ {result['error']}")
        
        # Show custom data
        custom_data = manual_input.get_custom_data()
        if not custom_data.empty:
            st.markdown("### 📋 Custom Data Points")
            st.dataframe(custom_data, use_container_width=True)
            
            if st.button("🗑️ Clear Custom Data", key="btn_clear_custom"):
                manual_input.clear_custom_data()
                st.success("Custom data cleared")
                st.rerun()
        
        st.markdown("---")
        
        # What-If Scenario Builder
        st.markdown("### 🔮 What-If Scenario Builder")
        
        scenario_name = st.text_input("Scenario Name", value="My Scenario", key="scenario_name_input")
        
        st.markdown("**Define Changes:**")
        
        num_changes = st.number_input(
            "Number of Changes",
            min_value=1,
            max_value=10,
            value=1,
            key="num_scenario_changes"
        )
        
        changes = []
        for i in range(int(num_changes)):
            st.markdown(f"**Change {i+1}:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                change_date = st.date_input(
                    f"Date",
                    value=processor.df.index[-1].date(),
                    min_value=processor.df.index[0].date(),
                    max_value=processor.df.index[-1].date(),
                    key=f"scenario_date_{i}"
                )
            
            with col2:
                change_col = st.selectbox(
                    f"Column",
                    all_columns,
                    key=f"scenario_col_{i}"
                )
            
            with col3:
                change_val = st.number_input(
                    f"Value",
                    value=0.0,
                    key=f"scenario_val_{i}"
                )
            
            changes.append({
                'date': str(change_date),
                'column': change_col,
                'value': change_val
            })
        
        if st.button("🎯 Create Scenario", key="btn_create_scenario"):
            result = manual_input.create_scenario(scenario_name, changes)
            if result['success']:
                st.success(f"✅ Scenario '{scenario_name}' created with {result['changes_applied']} changes")
            else:
                st.error(f"❌ {result['error']}")
        
        # Show existing scenarios
        scenarios = manual_input.get_scenarios()
        if scenarios:
            st.markdown("### 📁 Existing Scenarios")
            scenarios_df = pd.DataFrame(scenarios)
            st.dataframe(scenarios_df, use_container_width=True)

# ============ PAGE: AI AGENT ANALYSIS ============
elif page == "🤖 AI Agent Analysis":
    st.title("🤖 Agentic AI Analysis Engine")
    
    st.markdown("""
    This intelligent agent performs multi-dimensional analysis of geopolitical factors 
    and their impact on global markets.
    """)
    
    # Trend Analysis
    st.subheader("📈 Geopolitical Trend Analysis")
    
    lookback_days = st.slider("Lookback Period (Days)", 7, 90, 30)
    
    trends = agent.analyze_geopolitical_trends(lookback_days=lookback_days)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        high_alerts = sum(1 for t in trends['trends'].values() if t['status'] == 'HIGH')
        st.metric("🔴 High Risk", high_alerts)
    with col2:
        medium_alerts = sum(1 for t in trends['trends'].values() if t['status'] == 'MEDIUM')
        st.metric("🟠 Medium Risk", medium_alerts)
    with col3:
        low_alerts = sum(1 for t in trends['trends'].values() if t['status'] == 'LOW')
        st.metric("🟢 Low Risk", low_alerts)
    with col4:
        st.metric("📊 Total Metrics", len(trends['trends']))
    
    # Alert Table
    if trends['alerts']:
        st.warning("⚠️ **Active Alerts:**")
        for alert in trends['alerts']:
            st.write(f"• {alert}")
    else:
        st.success("✅ No critical alerts at this time")
    
    # Trend Details
    with st.expander("📊 View All Trend Details"):
        trend_df = pd.DataFrame([
            {
                'Metric': k,
                'Latest': v['latest'],
                'Mean': v['mean'],
                'Std': v['std'],
                'Z-Score': v['z_score'],
                'Status': v['status']
            }
            for k, v in trends['trends'].items()
        ])
        st.dataframe(trend_df, use_container_width=True)
    
    st.markdown("---")
    
    # Market-Geopolitical Linkage
    st.subheader("🔗 Market-Geopolitical Linkage Analysis")
    linkage = agent.detect_market_financial_linkage()
    
    if linkage['correlations']:
        linkage_df = pd.DataFrame([
            {
                'Relationship': k,
                'Correlation': f"{v['correlation']:.4f}",
                'Strength': v['strength']
            }
            for k, v in linkage['correlations'].items()
        ])
        st.dataframe(linkage_df, use_container_width=True)
        
        # Visualization
        if len(linkage['correlations']) > 0:
            fig = go.Figure()
            relationships = list(linkage['correlations'].keys())[:10]
            correlations = [linkage['correlations'][r]['correlation'] for r in relationships]
            colors = ['#00cc00' if c > 0 else '#ff4444' for c in correlations]
            
            fig.add_trace(go.Bar(
                x=correlations,
                y=relationships,
                orientation='h',
                marker_color=colors
            ))
            
            fig.update_layout(
                title="Top Market-Geopolitical Correlations",
                xaxis_title="Correlation Coefficient",
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No significant correlations detected in current period")
    
    st.markdown("---")
    
    # Market Stress Prediction
    st.subheader("⚡ Market Stress Scenario Prediction")
    
    days_ahead = st.slider("Prediction Horizon (Days)", 1, 30, 5)
    stress = agent.predict_market_stress(days_ahead=days_ahead)
    
    risk_color = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
    risk_bg = {'CRITICAL': '#ff0000', 'HIGH': '#ff9500', 'MEDIUM': '#ffcc00', 'LOW': '#00cc00'}
    
    st.markdown(f"""
    <div style="background-color: {risk_bg.get(stress['risk_level'], '#gray')}20; 
                padding: 20px; border-radius: 10px; text-align: center;">
        <h2>{risk_color.get(stress['risk_level'], '❓')} Risk Level: {stress['risk_level']}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    for scenario in stress['stress_scenarios']:
        with st.expander(f"📌 {scenario['scenario']}", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Probability", f"{scenario['probability']:.0%}")
            with col2:
                st.metric("Impact", scenario['impact'])
            with col3:
                st.metric("Horizon", f"{days_ahead} days")
            st.info(f"**💡 Recommendation:** {scenario['recommendation']}")

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
        st.markdown("### 🌲 Random Forest Regressor")
        with st.spinner("Training Volatility Predictor..."):
            model1, metrics1 = ml_models.build_volatility_predictor(X_train, y_train, X_test, y_test)
        
        st.success("✅ Model Trained")
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("R² Score", f"{metrics1['r2']:.4f}")
        with metric_cols[1]:
            st.metric("RMSE", f"{metrics1['rmse']:.4f}")
        with metric_cols[2]:
            st.metric("MAE", f"{metrics1['mae']:.4f}")
    
    # Model 2: Market Shock Detector
    with col2:
        st.markdown("### 🚀 Gradient Boosting Regressor")
        with st.spinner("Training Market Shock Detector..."):
            model2, metrics2 = ml_models.build_market_shock_detector(X_train, y_train, X_test, y_test)
        
        st.success("✅ Model Trained")
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("R² Score", f"{metrics2['r2']:.4f}")
        with metric_cols[1]:
            st.metric("RMSE", f"{metrics2['rmse']:.4f}")
        with metric_cols[2]:
            st.metric("MAE", f"{metrics2['mae']:.4f}")
    
    st.markdown("---")
    
    # Model 3: Neural Network
    st.markdown("### 🧠 Neural Network (MLP Regressor)")
    with st.spinner("Training Neural Network..."):
        model3, metrics3 = ml_models.build_time_series_forecaster(X_train, y_train, X_test, y_test)
    
    st.success("✅ Model Trained")
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("R² Score", f"{metrics3['r2']:.4f}")
    with metric_cols[1]:
        st.metric("RMSE", f"{metrics3['rmse']:.4f}")
    with metric_cols[2]:
        st.metric("MAE", f"{metrics3['mae']:.4f}")
    
    st.markdown("---")
    
    # Model Predictions Visualization
    st.subheader("🎯 Model Predictions vs Actual")
    
    pred_df = pd.DataFrame({
        'Date': X_test.index,
        'Actual': y_test.values,
        'Random Forest': metrics1['predictions'],
        'Gradient Boosting': metrics2['predictions'],
        'Neural Network': metrics3['predictions']
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Actual'],
        name='Actual',
        mode='lines',
        line=dict(color='white', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Random Forest'],
        name='Random Forest',
        mode='lines',
        opacity=0.7,
        line=dict(color='#00d4ff', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Gradient Boosting'],
        name='Gradient Boosting',
        mode='lines',
        opacity=0.7,
        line=dict(color='#ff6b6b', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Neural Network'],
        name='Neural Network',
        mode='lines',
        opacity=0.7,
        line=dict(color='#00cc00', width=1)
    ))
    
    fig.update_layout(
        title="Model Predictions vs Actual Values",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        height=500,
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Comparison Table
    st.subheader("📊 Model Comparison")
    comparison_df = pd.DataFrame({
        'Model': ['Random Forest', 'Gradient Boosting', 'Neural Network'],
        'R² Score': [metrics1['r2'], metrics2['r2'], metrics3['r2']],
        'RMSE': [metrics1['rmse'], metrics2['rmse'], metrics3['rmse']],
        'MAE': [metrics1['mae'], metrics2['mae'], metrics3['mae']]
    })
    st.dataframe(comparison_df, use_container_width=True)
    
    # Feature Importance
    st.subheader("🎯 Feature Importance (Random Forest)")
    importance = ml_models.feature_importance_analysis('volatility_rf', close_cols)
    if importance:
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            x=importance['importance_scores'],
            y=importance['features'],
            orientation='h',
            marker_color='#667eea'
        ))
        fig_imp.update_layout(
            title="Top 10 Most Important Features",
            xaxis_title="Importance Score",
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ============ PAGE: ANOMALY DETECTION ============
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Geopolitical Anomaly Detection System")
    
    st.subheader("🔍 Anomaly Detection Results")
    
    # Build anomaly detector
    with st.spinner("Running anomaly detection..."):
        ml_models.build_anomaly_detector(processor.df.fillna(method='ffill'))
        anomaly_detection = ml_models.detect_geopolitical_anomalies(processor.df)
    
    # Get agent alerts
    anomalies = agent.anomaly_alert_system()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔍 Total Anomalies", anomaly_detection['anomalies_count'])
    with col2:
        st.metric("📍 Recent Anomalies", len(anomalies['recent_anomalies']))
    with col3:
        st.metric("⚠️ Action Items", len(anomalies['action_items']))
    
    st.markdown("---")
    
    # Recent Anomalies Table
    if anomalies['recent_anomalies']:
        st.subheader("📋 Recent Anomalous Events")
        anomaly_df = pd.DataFrame(anomalies['recent_anomalies'])
        st.dataframe(anomaly_df, use_container_width=True)
    else:
        st.info("No recent anomalies detected")
    
    # Action Items
    st.subheader("✅ Recommended Action Items")
    if anomalies['action_items']:
        for action in anomalies['action_items']:
            st.write(f"• {action}")
    else:
        st.success("No immediate actions required")
    
    st.markdown("---")
    
    # Anomaly Score Distribution
    st.subheader("📊 Anomaly Score Distribution")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=anomaly_detection['anomaly_scores'],
        nbinsx=50,
        name='Anomaly Scores',
        marker_color='#667eea'
    ))
    
    # Add threshold line
    threshold = np.percentile(anomaly_detection['anomaly_scores'], 5)
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text="Anomaly Threshold")
    
    fig.update_layout(
        title="Distribution of Anomaly Scores",
        xaxis_title="Anomaly Score (Lower = More Anomalous)",
        yaxis_title="Frequency",
        height=400,
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Timeline of Anomalies
    if anomaly_detection['anomaly_dates']:
        st.subheader("📅 Anomaly Timeline")
        
        anomaly_dates = pd.to_datetime(anomaly_detection['anomaly_dates'])
        monthly_counts = anomaly_dates.to_series().groupby(anomaly_dates.to_period('M')).count()
        
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Bar(
            x=monthly_counts.index.astype(str),
            y=monthly_counts.values,
            marker_color='#ff6b6b'
        ))
        
        fig_timeline.update_layout(
            title="Monthly Anomaly Count",
            xaxis_title="Month",
            yaxis_title="Number of Anomalies",
            template='plotly_dark',
            height=350
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

# ============ PAGE: INSIGHTS & RECOMMENDATIONS ============
elif page == "💡 Insights & Recommendations":
    st.title("💡 Strategic Insights & Recommendations")
    
    st.subheader("🎯 Sector Recommendations")
    recommendations = agent.sector_recommendation_engine()
    
    rating_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡', 'NEUTRAL': '⚪'}
    rating_color = {'BUY': '#00cc00', 'SELL': '#ff4444', 'HOLD': '#ffcc00', 'NEUTRAL': '#888888'}
    
    rec_cols = st.columns(3)
    
    for idx, (sector, data) in enumerate(recommendations.items()):
        if sector != 'timestamp':
            with rec_cols[idx % 3]:
                emoji = rating_emoji.get(data['rating'], '❓')
                color = rating_color.get(data['rating'], '#888888')
                
                st.markdown(f"""
                <div style="background-color: {color}20; padding: 20px; border-radius: 10px; 
                            border-left: 4px solid {color}; margin-bottom: 10px;">
                    <h3>{emoji} {sector.replace('_', ' ').title()}</h3>
                    <h4>Rating: {data['rating']}</h4>
                    <p>{data['reasoning'] if data['reasoning'] else 'No specific signals detected'}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Memory Log
    st.subheader("📝 Agent Memory Log")
    memory = agent.get_memory_log()
    if memory:
        st.write(f"**Total Memory Events:** {len(memory)}")
        
        with st.expander("View Recent Memory"):
            for idx, mem in enumerate(memory[-5:]):
                st.markdown(f"""
                **Event {idx + 1}:**
                - Timestamp: {mem['timestamp']}
                - Lookback Days: {mem['lookback_days']}
                - Metrics Analyzed: {len(mem['trends'])}
                - Alerts: {len(mem['alerts'])}
                """)
    else:
        st.info("No memory events recorded yet")
    
    st.markdown("---")
    
    # Insights Log
    st.subheader("🔬 Generated Insights")
    insights = agent.get_insights_log()
    st.write(f"**Total Insights Generated:** {len(insights)}")
    
    if insights:
        with st.expander("View Recent Insights"):
            for idx, insight in enumerate(insights[-5:]):
                st.json(insight)

# ============ PAGE: EXECUTIVE SUMMARY ============
elif page == "📋 Executive Summary":
    st.title("📊 Executive Summary Report")
    
    summary = agent.executive_summary()
    
    # Report Header
    st.markdown(f"""
    <div style="background-color: #1f77b4; padding: 20px; border-radius: 10px; color: white;">
        <h2>📈 Geopolitical Risk Analysis Report</h2>
        <p><strong>Report Date:</strong> {summary['report_date'][:10]}</p>
        <p><strong>Data Coverage:</strong> {summary['data_coverage']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Observations", f"{summary['total_observations']:,}")
    with col2:
        st.metric("🎯 Features", summary['total_features'])
    with col3:
        st.metric("🧠 Memory Events", summary['memory_events'])
    with col4:
        st.metric("💡 Insights", summary['insights_generated'])
    
    st.markdown("---")
    
    # Key Findings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Key Findings")
        if summary['key_findings']:
            for finding in summary['key_findings']:
                st.write(f"• {finding}")
        else:
            st.info("No key findings to report")
    
    with col2:
        st.markdown("### ⛔ Top Identified Risks")
        if summary['top_risks']:
            for risk in summary['top_risks']:
                st.warning(f"• {risk}")
        else:
            st.success("No critical risks identified")
    
    st.markdown("---")
    
    # Opportunities
    st.markdown("### 🚀 Identified Opportunities")
    if summary['opportunities']:
        opp_cols = st.columns(len(summary['opportunities']))
        for idx, opp in enumerate(summary['opportunities']):
            with opp_cols[idx]:
                st.success(f"✅ {opp}")
    else:
        st.info("Monitor geopolitical developments for emerging opportunities")
    
    st.markdown("---")
    
    # Market Overview
    st.markdown("### 📈 Market Performance Snapshot")
    
    financial_data = processor.get_financial_features()
    close_cols = [col for col in financial_data.columns if '_Close' in col]
    
    performance_data = []
    for col in close_cols[:8]:  # Top 8 assets
        asset = col.replace('_Close', '')
        latest = financial_data[col].iloc[-1]
        week_ago = financial_data[col].iloc[-7]
        month_ago = financial_data[col].iloc[-30]
        
        weekly_change = ((latest - week_ago) / week_ago) * 100
        monthly_change = ((latest - month_ago) / month_ago) * 100
        
        performance_data.append({
            'Asset': asset,
            'Current Price': f"${latest:,.2f}",
            '7-Day Change': f"{weekly_change:+.2f}%",
            '30-Day Change': f"{monthly_change:+.2f}%"
        })
    
    st.dataframe(pd.DataFrame(performance_data), use_container_width=True)
    
    st.markdown("---")
    
    # Download Report
    report_text = f"""
    GEOPOLITICAL RISK ANALYSIS REPORT
    ==================================
    
    Report Date: {summary['report_date'][:10]}
    Data Coverage: {summary['data_coverage']}
    
    KEY METRICS:
    - Total Observations: {summary['total_observations']:,}
    - Total Features: {summary['total_features']}
    - Memory Events: {summary['memory_events']}
    - Insights Generated: {summary['insights_generated']}
    
    KEY FINDINGS:
    {chr(10).join(['- ' + f for f in summary['key_findings']])}
    
    TOP RISKS:
    {chr(10).join(['- ' + r for r in summary['top_risks']]) if summary['top_risks'] else '- No critical risks identified'}
    
    OPPORTUNITIES:
    {chr(10).join(['- ' + o for o in summary['opportunities']]) if summary['opportunities'] else '- Monitor for emerging opportunities'}
    
    ---
    Report Generated by Geopolitical AI Agent
    """
    
    st.download_button(
        "📥 Download Executive Report",
        report_text,
        "executive_summary.txt",
        "text/plain",
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("*Report Generated by Geopolitical AI Agent*")

# ============ FOOTER ============
st.markdown("""
---
### 🔗 Data Sources
- **Markets:** Yahoo Finance API
- **Geopolitical Metrics:** Wikimedia Foundation REST API
- **Analysis Framework:** Machine Learning + Agentic AI

### 🛠️ Features
- 💬 **Interactive AI Chatbot** - Natural language queries
- 🔍 **Advanced Data Search** - Multi-dimensional filtering
- ✏️ **Manual Input** - What-if scenario analysis
- 🤖 **AI Agent** - Autonomous risk analysis
- 📈 **ML Models** - Predictive analytics

*This application integrates real-time market data with geopolitical attention metrics 
to provide strategic risk assessment and opportunity identification.*
""")
