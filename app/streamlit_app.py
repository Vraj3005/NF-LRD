"""
Streamlit Dashboard for NIFTY 50 Market Regime Intelligence Platform.
Author: Senior Full-Stack Quantitative Data Scientist
"""

import os
import sys
import io
import json
import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Add the project root to sys.path to ensure src imports resolve correctly
# when run via streamlit from any directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Set up page config first
st.set_page_config(
    page_title="NIFTY 50 Market Regime Terminal",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Imports
from src.visualization.charts import (
    plot_equity_curves,
    plot_drawdowns,
    plot_regime_distributions,
    plot_monte_carlo_paths,
    plot_monte_carlo_distributions,
    plot_rolling_risk_metrics,
    plot_regime_overlaid_price,
    plot_pca_2d,
    plot_pca_3d,
    plot_correlation_heatmap,
    plot_regime_durations,
    plot_regime_probabilities_chart
)
from src.visualization.dashboard_components import (
    load_all_dashboard_data,
    run_full_pipeline_from_dashboard,
    run_custom_backtest,
    run_custom_monte_carlo,
    render_custom_header,
    render_educational_disclaimer
)
from src.analysis.regime_analysis import analyze_regimes
from src.data.fetch_data import fetch_ticker_data, load_settings
from src.data.clean_data import process_and_save_data
from src.features.feature_pipeline import run_feature_engineering_pipeline
from src.analysis.monte_carlo import MonteCarloSimulator

# ----------------------------------------------------
# STATE INITIALIZATION
# ----------------------------------------------------
if "labeled_data" not in st.session_state:
    data = load_all_dashboard_data()
    if data["labeled_data"] is not None:
        st.session_state.labeled_data = data["labeled_data"]
        st.session_state.transition_df = data["transition_matrix"]
        st.session_state.model_report = data["model_report"]
        st.session_state.regime_risk = data["regime_risk"]
        st.session_state.baseline_risk = data["baseline_risk"]
        st.session_state.stress_test = data["stress_test"]
        st.session_state.model_comparison = data["model_comparison"]
        
        # Default properties
        st.session_state.model_type = "HMM"
        st.session_state.n_regimes = int(data["labeled_data"]['regime_state'].nunique())
        st.session_state.is_custom_trained = False
        st.session_state.custom_probs = None
    else:
        st.session_state.labeled_data = None
        st.session_state.transition_df = None
        st.session_state.model_report = None
        st.session_state.regime_risk = None
        st.session_state.baseline_risk = None
        st.session_state.stress_test = None
        st.session_state.model_comparison = None

# Helper function to generate dynamic HTML report
def generate_html_report(backtest_df: pd.DataFrame, regime_risk_df: pd.DataFrame, baseline_df: pd.DataFrame) -> str:
    bt_html = backtest_df.to_html(index=False, classes="report-table")
    regime_html = regime_risk_df.to_html(index=False, classes="report-table")
    baseline_html = baseline_df.to_html(index=False, classes="report-table")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NIFTY 50 Market Regime & Risk Intelligence Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #0f172a;
                color: #f8fafc;
                margin: 0;
                padding: 40px;
            }}
            .container {{
                max-width: 1100px;
                margin: 0 auto;
                background-color: #1e293b;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 8px 30px rgba(0,0,0,0.5);
                border: 1px solid #334155;
            }}
            h1 {{
                color: #3b82f6;
                border-bottom: 2px solid #334155;
                padding-bottom: 15px;
                font-size: 2.2rem;
                margin-top: 0;
            }}
            h2 {{
                color: #60a5fa;
                margin-top: 30px;
                border-bottom: 1px solid #334155;
                padding-bottom: 8px;
            }}
            .report-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 0.95rem;
            }}
            .report-table th, .report-table td {{
                border: 1px solid #334155;
                padding: 10px 12px;
                text-align: left;
            }}
            .report-table th {{
                background-color: #1e1b4b;
                color: #38bdf8;
                font-weight: 600;
            }}
            .report-table tr:nth-child(even) {{
                background-color: #0f172a;
            }}
            .footer {{
                margin-top: 50px;
                text-align: center;
                font-size: 0.8rem;
                color: #64748b;
                border-top: 1px solid #334155;
                padding-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>NIFTY 50 Latent Market Regime Discovery & Risk Report</h1>
            <p><strong>System Generated On:</strong> {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>1. Baseline Asset Performance (NIFTY 50)</h2>
            {baseline_html}
            
            <h2>2. Quantitative Strategy Backtest Comparison</h2>
            {bt_html}
            
            <h2>3. Regime-Wise Risk Intelligence Breakdown</h2>
            {regime_html}
            
            <h2>4. Academic & Research Disclaimer</h2>
            <p style="color: #fca5a5; font-size: 0.85rem; line-height: 1.5; background-color: rgba(239,68,68,0.1); padding: 12px; border-left: 4px solid #ef4444; border-radius: 4px;">
                This report is compiled strictly for portfolio and interview demonstration purposes. Simulated returns are derived from historical indices and account for estimated trading fees and slippage models. Actual trading involves significant loss risks. Past performance is never a guarantee of future returns.
            </p>
            
            <div class="footer">
                <p>NIFTY 50 Latent Market Regime Discovery & Risk Intelligence Platform &copy; 2026</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

def main():
    # Initialize page state if not already set
    if "page" not in st.session_state:
        st.session_state.page = "Home / Project Overview"

    # ----------------------------------------------------
    # PREMIUM UI STYLE INJECTION (2026 SaaS Dashboard Style)
    # ----------------------------------------------------
    st.markdown(
        """
        <style>
            /* Ambient liquid mesh animation for the background */
            @keyframes liquidMesh {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            .stApp {
                background-color: #030712 !important;
                background-image: radial-gradient(at 0% 0%, rgba(30, 58, 138, 0.18) 0, transparent 60%),
                                  radial-gradient(at 50% 0%, rgba(99, 102, 241, 0.12) 0, transparent 60%),
                                  radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.05) 0, transparent 60%) !important;
                background-size: 200% 200% !important;
                animation: liquidMesh 25s ease infinite !important;
            }
            
            /* High-end minimalist typography & text scaling */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@600;700;800&display=swap');
            
            /* Glassmorphic metrics styling */
            div[data-testid="stMetricValue"] {
                font-size: 1.85rem !important;
                font-weight: 700 !important;
                font-family: 'Outfit', sans-serif !important;
                background: linear-gradient(135deg, #ffffff 30%, #9ca3af 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.5px;
            }
            div[data-testid="stMetricLabel"] {
                font-size: 0.78rem !important;
                font-weight: 600 !important;
                color: #6b7280 !important;
                text-transform: uppercase !important;
                letter-spacing: 1px !important;
            }
            div[data-testid="stMetric"] {
                background: rgba(10, 15, 30, 0.6) !important;
                border: 1px solid rgba(255, 255, 255, 0.06) !important;
                border-radius: 14px !important;
                padding: 18px 22px !important;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25) !important;
                backdrop-filter: blur(12px) !important;
                transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
            }
            div[data-testid="stMetric"]:hover {
                border-color: rgba(99, 102, 241, 0.3) !important;
                box-shadow: 0 8px 32px 0 rgba(99, 102, 241, 0.08) !important;
                transform: translateY(-3px) !important;
            }
            
            /* Custom Glassmorphic container utility */
            .glass-card {
                background: rgba(10, 15, 30, 0.6) !important;
                border: 1px solid rgba(255, 255, 255, 0.06) !important;
                border-radius: 14px !important;
                padding: 20px !important;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25) !important;
                backdrop-filter: blur(12px) !important;
                margin-bottom: 20px;
                transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
            }
            .glass-card:hover {
                border-color: rgba(99, 102, 241, 0.25) !important;
                box-shadow: 0 8px 32px 0 rgba(99, 102, 241, 0.06) !important;
            }
            
            /* Sidebar customizations */
            section[data-testid="stSidebar"] {
                background-color: #050811 !important;
                border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
            }
            
            /* Target Streamlit secondary buttons in sidebar (Inactive Pages) */
            div[data-testid="stSidebar"] button[kind="secondary"] {
                background: rgba(255, 255, 255, 0.02) !important;
                border: 1px solid rgba(255, 255, 255, 0.04) !important;
                color: #9ca3af !important;
                text-align: left !important;
                padding: 12px 18px !important;
                border-radius: 10px !important;
                transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
                font-size: 0.92rem !important;
                font-weight: 500 !important;
                margin-bottom: 8px !important;
                display: block !important;
                width: 100% !important;
            }
            div[data-testid="stSidebar"] button[kind="secondary"]:hover {
                background: rgba(99, 102, 241, 0.08) !important;
                border-color: rgba(99, 102, 241, 0.3) !important;
                color: #818cf8 !important;
                transform: translateX(4px) !important;
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.05) !important;
            }
            
            /* Target Streamlit primary buttons in sidebar (Active Page) */
            div[data-testid="stSidebar"] button[kind="primary"] {
                background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%) !important;
                border: 1px solid rgba(99, 102, 241, 0.4) !important;
                color: #ffffff !important;
                text-align: left !important;
                padding: 12px 18px !important;
                border-radius: 10px !important;
                font-size: 0.92rem !important;
                font-weight: 600 !important;
                box-shadow: 0 4px 20px rgba(79, 70, 229, 0.25) !important;
                transform: translateX(4px) !important;
                border-left: 4px solid #818cf8 !important;
                margin-bottom: 8px !important;
                display: block !important;
                width: 100% !important;
            }
            
            /* Stylized dynamic page tabs */
            button[data-baseweb="tab"] {
                font-family: 'Inter', sans-serif !important;
                font-weight: 500 !important;
                font-size: 0.95rem !important;
                color: #9ca3af !important;
                border-bottom: 2px solid transparent !important;
                padding: 10px 16px !important;
                background-color: transparent !important;
                transition: all 0.3s ease !important;
            }
            button[data-baseweb="tab"][aria-selected="true"] {
                color: #818cf8 !important;
                border-bottom-color: #818cf8 !important;
                font-weight: 600 !important;
            }
            
            /* Clean modern scrollbar */
            ::-webkit-scrollbar {
                width: 5px;
                height: 5px;
            }
            ::-webkit-scrollbar-track {
                background: #030712;
            }
            ::-webkit-scrollbar-thumb {
                background: #1f2937;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #374151;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Render beautiful custom header
    render_custom_header()
    
    # ----------------------------------------------------
    # SIDEBAR: Navigation Button Tabs
    # ----------------------------------------------------
    st.sidebar.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h3 style="color: #60a5fa; margin: 0; font-family: 'Outfit', sans-serif;">Navigation</h3>
            <span style="color: #6b7280; font-size: 0.85rem;">Platform Operations Control</span>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    pages = [
        ("Home / Project Overview", "Home / Project Overview"),
        ("Data Explorer", "Data Explorer"),
        ("Feature Analytics", "Feature Analytics"),
        ("Regime Discovery", "Regime Discovery"),
        ("Risk Intelligence", "Risk Intelligence"),
        ("Monte Carlo Simulator", "Monte Carlo Simulator"),
        ("Strategy Backtesting", "Strategy Backtesting"),
        ("Final Report & Exports", "Final Report & Exports")
    ]
    
    # Render navigation buttons
    for label, page_val in pages:
        is_active = (st.session_state.page == page_val)
        btn_type = "primary" if is_active else "secondary"
        if st.sidebar.button(label, key=f"nav_{page_val}", type=btn_type, use_container_width=True):
            st.session_state.page = page_val
            st.rerun()
            
    page = st.session_state.page
    st.sidebar.markdown("---")
    
    # ----------------------------------------------------
    # SIDEBAR: Control Sliders and Inputs
    # ----------------------------------------------------
    st.sidebar.markdown(
        """
        <div style="margin-bottom: 10px;">
            <span style="color: #6b7280; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Global Controls</span>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    tc_input = st.sidebar.slider(
        "Transaction Cost (bps)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help="One-way transaction fee applied on portfolio reallocation."
    )
    slip_input = st.sidebar.slider(
        "Execution Slippage (bps)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
        help="Estimated execution slippage penalty."
    )
    
    st.sidebar.markdown("---")
    
    mc_sims = st.sidebar.number_input(
        "Monte Carlo Paths",
        min_value=100,
        max_value=10000,
        value=5000,
        step=100,
        help="Number of stochastic returns paths to simulate."
    )
    mc_seed = st.sidebar.number_input(
        "Random Seed",
        value=42,
        step=1,
        help="Random seed for path reproducibility."
    )
    st.sidebar.markdown("---")

    
    # Load configuration settings
    settings = load_settings()
    
    # Safety Check: If no data loaded, show pipeline initialization prompt
    if st.session_state.labeled_data is None:
        st.warning("Market regime and backtest reports could not be found in models/reports/.")
        st.info("The system requires a completed run of the modeling and backtesting pipeline to populate dashboard elements.")
        
        if st.button("Initialize and Run First-Time Pipeline", use_container_width=True):
            with st.spinner("Running full pipeline (this will download data, build indicators, train HMM model, and execute backtests)..."):
                success, logs = run_full_pipeline_from_dashboard()
                if success:
                    st.success("Pipeline initialized successfully!")
                    st.cache_data.clear()
                    st.session_state.clear()
                    st.rerun()
                else:
                    st.error("Pipeline initialization failed.")
                    st.text_area("Detailed Error Logs", logs, height=300)
        
        render_educational_disclaimer()
        return

    # Extract dynamic states metadata for Home display
    labeled_data = st.session_state.labeled_data
    n_regimes = st.session_state.n_regimes
    
    last_row = labeled_data.iloc[-1]
    curr_regime = last_row["regime_label"]
    curr_state = int(last_row["regime_state"])
    curr_date = last_row["date"].strftime('%Y-%m-%d')
    
    # Map current state to risk score
    risk_mapping = {
        "Bullish Low Volatility": 15,
        "Recovery Regime": 35,
        "Sideways Low Volatility": 45,
        "Bullish High Volatility": 55,
        "Distribution / Risk-Off Regime": 75,
        "Bearish High Volatility": 95
    }
    risk_score = risk_mapping.get(curr_regime, 50)
    
    # Load standard features matrix for PCA/Analytics
    features_parquet = settings.get("features", {}).get("feature_parquet_path", "data/processed/features.parquet")
    features_df = None
    if os.path.exists(features_parquet):
        features_df = pd.read_parquet(features_parquet)

    # ----------------------------------------------------
    # PAGE 1: HOME / PROJECT OVERVIEW
    # ----------------------------------------------------
    if page == "Home / Project Overview":
        st.markdown("### Latent Market Regime Platform Dashboard")
        st.markdown(
            "This platform implements a mathematical framework to partition historical NIFTY 50 timeseries into "
            "uncorrelated latent regime states. By parameterizing regimes using dynamic indicators rather than "
            "arbitrary static boundaries, we optimize tactical asset allocations, stress-test portfolios, and calculate forward simulations."
        )
        
        # Real-time state metrics
        col_home1, col_home2, col_home3 = st.columns(3)
        with col_home1:
            st.metric(
                label="Latest Available Trading Date",
                value=curr_date
            )
        with col_home2:
            st.metric(
                label="Current Detected Regime",
                value=curr_regime
            )
        with col_home3:
            st.metric(
                label="Current Systemic Risk Score",
                value=f"{risk_score} / 100",
                help="Heuristic risk index based on current regime's crash probability and transition risk to bearish states."
            )
            
        st.progress(risk_score / 100.0)
        
        st.markdown("---")
        
        col_arch_l, col_arch_r = st.columns([3, 2])
        with col_arch_l:
            st.markdown("#### System Architecture Flowchart")
            st.graphviz_chart('''
            digraph G {
                rankdir=LR;
                node [shape=box, style=filled, color="#1e293b", fontcolor=white, fontname="Helvetica", fontsize=10];
                edge [color="#3b82f6", fontname="Helvetica", fontsize=8];
                
                Ingestion -> Clean [label="yfinance / CSV"];
                Clean -> FeatureEng [label="Raw OHLCV"];
                FeatureEng -> ModelTrain [label="40+ Indicators"];
                ModelTrain -> RegimeState [label="Unsupervised ML"];
                RegimeState -> RiskDecomp [label="Regimes"];
                RegimeState -> Backtest [label="Tactical weights"];
                RegimeState -> MonteCarlo [label="Transitions"];
                
                Ingestion [fillcolor="#0f172a", label="Data Ingestion\n(yfinance / CSV)"];
                Clean [fillcolor="#0f172a", label="Data Validation\n(Logical Checks)"];
                FeatureEng [fillcolor="#1e1b4b", label="Feature Engineering\n(Trend, Vol, Mom, Stat)"];
                ModelTrain [fillcolor="#311042", label="Latent Regime Models\n(HMM / GMM / MSR)"];
                RegimeState [fillcolor="#1c1917", label="Discovered Regimes\n(Bullish, Bearish, etc)"];
                RiskDecomp [fillcolor="#3f0f15", label="Risk Intelligence\n(VaR/CVaR, Stress)"];
                Backtest [fillcolor="#0c2310", label="Tactical Backtester\n(Cost & Slippage)"];
                MonteCarlo [fillcolor="#3b2307", label="Monte Carlo Simulator\n(Stochastic Paths)"];
            }
            ''')
        with col_arch_r:
            st.markdown(
                """
                <div class="glass-card" style="margin-top: 15px;">
                    <h4 style="margin-top:0; color:#60a5fa;">Platform Components</h4>
                    <ol style="color:#9ca3af; font-size:0.9rem; padding-left:20px; line-height:1.6; margin-bottom:0;">
                        <li><strong>Data Ingestion Engine</strong>: Fetches daily indices or loads manual files, running validation gates.</li>
                        <li><strong>Feature Engineering Pipeline</strong>: Formulates 40+ features covering returns, Parkinson/Garman-Klass volatility, momentum oscillators, statistical complexity, and global macro covariates.</li>
                        <li><strong>Latent Model Engine</strong>: Trains Gaussian HMM, GMM, or Markov Switching models to decode regimes.</li>
                        <li><strong>Risk Intelligence</strong>: Decomposes returns per state, computing Value at Risk and stress indices.</li>
                        <li><strong>Monte Carlo Path Simulator</strong>: Generates future market projections using HMM transitions.</li>
                        <li><strong>Tactical Backtester</strong>: Assesses regime-shifting allocation strategies with zero-lookahead bias.</li>
                    </ol>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        render_educational_disclaimer()

    # ----------------------------------------------------
    # PAGE 2: DATA EXPLORER
    # ----------------------------------------------------
    elif page == "Data Explorer":
        st.markdown("### Data Ingestion and Cleaned Explorer")
        st.markdown(
            "Fetch raw index prices from Yahoo Finance or upload custom NIFTY 50 trading datasets. "
            "The system validates inputs to ensure logical consistency (High >= Close/Open >= Low)."
        )
        
        tab_ing1, tab_ing2 = st.tabs(["yfinance Fetch Engine", "CSV Manual Uploader"])
        
        with tab_ing1:
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input("Start Date:", value=pd.to_datetime("2015-01-01"))
            with col_date2:
                end_date = st.date_input("End Date:", value=pd.to_datetime("today"))
                
            if st.button("Fetch and Re-engineer Data via yfinance", use_container_width=True):
                with st.spinner("Downloading ^NSEI and ^INDIAVIX..."):
                    nifty_df = fetch_ticker_data("^NSEI", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                    vix_df = fetch_ticker_data("^INDIAVIX", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                    
                    if nifty_df is not None:
                        # Process files using backend clean and feature functions
                        raw_dir = settings.get("data", {}).get("raw_dir", "data/raw")
                        os.makedirs(raw_dir, exist_ok=True)
                        n_path = os.path.join(raw_dir, "nifty_raw.csv")
                        nifty_df.to_csv(n_path, index=False)
                        
                        v_path = None
                        if vix_df is not None:
                            v_path = os.path.join(raw_dir, "vix_raw.csv")
                            vix_df.to_csv(v_path, index=False)
                            
                        with st.spinner("Cleaning raw datasets and engineering technical indicators..."):
                            process_and_save_data(n_path, v_path)
                            run_feature_engineering_pipeline()
                            st.success("Ingestion pipeline completed successfully! Resetting states.")
                            st.cache_data.clear()
                            st.rerun()
                    else:
                        st.error("Failed to download NIFTY 50 data.")
                        
        with tab_ing2:
            st.markdown("Upload a custom NIFTY 50 daily OHLCV dataset in CSV format.")
            uploaded_file = st.file_uploader("Choose Nifty CSV File", type="csv")
            
            if uploaded_file is not None:
                nifty_raw = pd.read_csv(uploaded_file)
                st.write("Manual File Preview:")
                st.dataframe(nifty_raw.head(3), use_container_width=True)
                
                if st.button("Upload and Process Manual CSV", use_container_width=True):
                    raw_dir = settings.get("data", {}).get("raw_dir", "data/raw")
                    os.makedirs(raw_dir, exist_ok=True)
                    n_path = os.path.join(raw_dir, "nifty_raw.csv")
                    nifty_raw.to_csv(n_path, index=False)
                    
                    with st.spinner("Cleaning manual CSV and generating feature matrices..."):
                        process_and_save_data(n_path, None)
                        run_feature_engineering_pipeline()
                        st.success("Uploaded manual CSV successfully processed!")
                        st.cache_data.clear()
                        st.rerun()
                        
        st.markdown("---")
        
        st.markdown("#### Cleaned Dataset & Verification Summary")
        if features_df is not None:
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Total Valid Days", len(features_df))
            with col_info2:
                st.metric("Clean Data Columns", len(features_df.columns))
            with col_info3:
                # Missing rows check
                missing_count = features_df.isnull().sum().sum()
                st.metric("Missing Values", missing_count)
                
            st.dataframe(features_df.head(5), use_container_width=True)
            
            # Interactive price and return plots
            col_plt1, col_plt2 = st.columns(2)
            with col_plt1:
                fig_p = px.line(features_df, x='date', y='raw_close', title='Cleaned NIFTY 50 Close Price', template='plotly_dark')
                fig_p.update_layout(yaxis_title="Index Level", xaxis_title="")
                st.plotly_chart(fig_p, use_container_width=True)
            with col_plt2:
                # Reconstruct return simple
                features_df['ret_simple'] = features_df['raw_close'].pct_change(fill_method=None).fillna(0.0)
                fig_r = px.line(features_df, x='date', y='ret_simple', title='NIFTY 50 Daily Simple Returns', template='plotly_dark')
                fig_r.update_layout(yaxis_title="Daily Return", xaxis_title="")
                st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.warning("Cleaned features parquet not found. Please re-run ingestion.")

    # ----------------------------------------------------
    # PAGE 3: FEATURE ANALYTICS
    # ----------------------------------------------------
    elif page == "Feature Analytics":
        st.markdown("### Engineered Feature Analytics & PCA Projection")
        st.markdown(
            "Analyze relationships across technical features. "
            "The Principal Component Analysis (PCA) applies dimensionality reduction on standardized features, "
            "projecting the high-dimensional matrix down to visualize regime separation."
        )
        
        if features_df is not None:
            tab_f1, tab_f2, tab_f3 = st.tabs(["Correlation Heatmap", "Principal Component Analysis (PCA)", "Technical Indicator Charts"])
            
            with tab_f1:
                corr_fig = plot_correlation_heatmap(labeled_data)
                st.plotly_chart(corr_fig, use_container_width=True)
                
            with tab_f2:
                pca_dim = st.radio("Select PCA Dimension:", ["2D Scatter Plot", "3D Interactive Scatter Plot"], horizontal=True)
                col_pca_l, col_pca_r = st.columns([3, 1])
                with col_pca_l:
                    if pca_dim == "2D Scatter Plot":
                        pca_fig = plot_pca_2d(labeled_data)
                    else:
                        pca_fig = plot_pca_3d(labeled_data)
                    st.plotly_chart(pca_fig, use_container_width=True)
                with col_pca_r:
                    st.markdown(
                        """
                        <div class="glass-card" style="margin-top: 15px;">
                            <h4 style="margin-top:0; color:#60a5fa;">Dimensionality Reduction Notes</h4>
                            <p style="color:#9ca3af; font-size:0.9rem; line-height:1.6; margin-bottom:0;">
                                Applying PCA standardizes the engineered indicators (volatilities, oscillator strengths, slopes, statistical complexity) and projects them into orthogonal directions. Clear spatial clustering of regime colors demonstrates that the unsupervised ML model discovers highly distinct mathematical spaces within raw market time-series.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
            with tab_f3:
                # Volatility select chart
                col_feat1, col_feat2 = st.columns(2)
                with col_feat1:
                    vol_indicator = st.selectbox(
                        "Choose Volatility Feature to Plot:",
                        options=[c for c in labeled_data.columns if "vol" in c or "atr" in c or "drawdown" in c] or ["atr_14"],
                        index=0
                    )
                    
                    fig_v = px.line(labeled_data, x='date', y=vol_indicator, title=f'{vol_indicator.replace("_", " ").title()} Series', template='plotly_dark')
                    fig_v.update_layout(yaxis_title="Indicator Level", xaxis_title="")
                    st.plotly_chart(fig_v, use_container_width=True)
                    
                with col_feat2:
                    mom_indicator = st.selectbox(
                        "Choose Momentum / Trend Feature to Plot:",
                        options=[c for c in labeled_data.columns if "rsi" in c or "macd" in c or "slope" in c or "hurst" in c or "entropy" in c] or ["rsi_14"],
                        index=0
                    )
                    
                    fig_m = px.line(labeled_data, x='date', y=mom_indicator, title=f'{mom_indicator.replace("_", " ").title()} Series', template='plotly_dark')
                    fig_m.update_layout(yaxis_title="Indicator Level", xaxis_title="")
                    st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.warning("Feature matrix not found.")

    # ----------------------------------------------------
    # PAGE 4: REGIME DISCOVERY ENGINE
    # ----------------------------------------------------
    elif page == "Regime Discovery":
        st.markdown("### Latent Market Regime Discovery Engine")
        st.markdown(
            "Train and fit unsupervised models dynamically on NIFTY 50 feature matrices. "
            "Choose between temporal modeling (HMM), static cluster analysis (GMM), or return regime switching (MSR)."
        )
        
        if features_df is not None:
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            with col_ctrl1:
                model_choice = st.selectbox("Select Model Architecture:", ["HMM (Gaussian Hidden Markov)", "GMM (Gaussian Mixture Model)", "MSR (Markov Switching Regression)"])
            with col_ctrl2:
                n_states_choice = st.slider("Target Hidden States (K):", min_value=2, max_value=6, value=st.session_state.n_regimes)
            with col_ctrl3:
                # Add spacing
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                train_clicked = st.button("Train Model on Feature Matrix", use_container_width=True)
                
            st.markdown("---")
            
            # Perform training if clicked
            if train_clicked:
                with st.spinner(f"Fitting {model_choice} with {n_states_choice} states..."):
                    # Feature columns
                    feature_cols = [c for c in features_df.columns if c != 'date' and not c.startswith('raw_')]
                    X = features_df[feature_cols].values
                    dates = features_df['date']
                    
                    model_type = "HMM"
                    if "GMM" in model_choice:
                        model_type = "GMM"
                    elif "MSR" in model_choice:
                        model_type = "MSR"
                        
                    try:
                        states = None
                        probs = None
                        
                        if model_type == "HMM":
                            from src.models.hmm_model import GaussianHMM
                            model = GaussianHMM(n_components=n_states_choice, covariance_type="diag", random_state=42)
                            model.fit(X)
                            states = model.predict(X)
                            probs = model.predict_proba(X)
                            
                        elif model_type == "GMM":
                            from src.models.gmm_model import GMMRegimeModel
                            model = GMMRegimeModel(n_components=n_states_choice, covariance_type="diag", random_state=42)
                            model.fit(X)
                            states = model.predict(X)
                            probs = model.predict_proba(X)
                            
                        else:  # MSR
                            from src.models.markov_switching import MarkovSwitchingModel
                            model = MarkovSwitchingModel(n_components=n_states_choice)
                            # MSR fits directly on returns
                            log_ret = np.log(features_df['raw_close'] / features_df['raw_close'].shift(1)).fillna(0.0).values
                            model.fit(log_ret)
                            states = model.predict(log_ret)
                            probs = model.predict_proba(log_ret)
                            
                        # Run regime labeling and analysis
                        df_regime, summary_df, trans_df = analyze_regimes(features_df, states, n_states_choice)
                        
                        # Store in session state
                        st.session_state.labeled_data = df_regime
                        st.session_state.transition_df = trans_df
                        st.session_state.regime_risk = summary_df.rename(columns={
                            "regime_state": "Regime_State",
                            "regime_label": "Regime_Label",
                            "daily_count": "Total_Days",
                            "annualized_volatility": "Annualized_Volatility",
                            "max_drawdown": "Max_Drawdown"
                        }) # Adapt to naming conventions
                        
                        st.session_state.model_type = model_type
                        st.session_state.n_regimes = n_states_choice
                        st.session_state.is_custom_trained = True
                        st.session_state.custom_probs = probs
                        
                        st.success(f"{model_choice} model successfully trained and loaded!")
                        
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
                        
            # Under the hood, if not trained, load cache
            # Visualisations block
            st.markdown("#### Model Outputs & State Overlay")
            
            # Regime overlaid Close price chart
            overlaid_fig = plot_regime_overlaid_price(labeled_data)
            st.plotly_chart(overlaid_fig, use_container_width=True)
            
            col_reg1, col_reg2 = st.columns(2)
            
            with col_reg1:
                st.markdown("#### Transition Probability Heatmap")
                if st.session_state.transition_df is not None:
                    # Plot transition probability matrix
                    t_df = st.session_state.transition_df
                    fig_t = px.imshow(
                        t_df.values,
                        x=t_df.columns,
                        y=t_df.index,
                        color_continuous_scale='Greens',
                        text_auto=".3f",
                        title='Empirical Transitions Probability Heatmap',
                        template='plotly_dark'
                    )
                    fig_t.update_layout(coloraxis_showscale=False, height=350, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_t, use_container_width=True)
                else:
                    st.info("No transition matrix available for the selected model configuration.")
                    
            with col_reg2:
                st.markdown("#### Regime Durations Breakdown")
                dur_fig = plot_regime_durations(labeled_data)
                st.plotly_chart(dur_fig, use_container_width=True)
                
            # If custom trained and probs available, display stacked area chart
            if st.session_state.custom_probs is not None:
                st.markdown("---")
                st.markdown("#### Smoothed Regime Posterior Probabilities")
                state_labels = list(labeled_data['regime_label'].unique())
                prob_fig = plot_regime_probabilities_chart(
                    labeled_data['date'], 
                    st.session_state.custom_probs, 
                    state_labels
                )
                st.plotly_chart(prob_fig, use_container_width=True)
                
        else:
            st.warning("Engineered feature matrix not found. Please ingest data first.")

    # ----------------------------------------------------
    # PAGE 5: RISK INTELLIGENCE
    # ----------------------------------------------------
    elif page == "Risk Intelligence":
        st.markdown("### Quantitative Risk Intelligence Terminal")
        st.markdown("Calculates risk ratios, drawdown windows, and historical Value at Risk (VaR/CVaR) metrics per regime.")
        
        regime_risk_df = st.session_state.regime_risk
        
        if regime_risk_df is not None:
            st.markdown("#### Regime-Wise Risk Intelligence Decomposition")
            
            # Select columns to display
            display_cols = [c for c in ["Regime_State", "Regime_Label", "Total_Days", "Annualized_Volatility", 
                                       "Daily_VaR_95", "Daily_CVaR_95", "Max_Drawdown", "Average_Daily_Return", "Median_Daily_Return", "Transition_Risk_to_Bearish"] if c in regime_risk_df.columns]
            
            st.dataframe(
                regime_risk_df[display_cols].style.format({
                    "Annualized_Volatility": "{:.2%}",
                    "Daily_VaR_95": "{:.2%}",
                    "Daily_CVaR_95": "{:.2%}",
                    "Max_Drawdown": "{:.2%}",
                    "Average_Daily_Return": "{:.4%}",
                    "Median_Daily_Return": "{:.4%}",
                    "Transition_Risk_to_Bearish": "{:.2%}"
                }),
                use_container_width=True
            )
            
            st.markdown("---")
            
            col_risk_l, col_risk_r = st.columns(2)
            
            with col_risk_l:
                st.markdown("#### Rolling Performance Curves (Dual Axis)")
                # Calculate rolling Sharpe and volatility on returns
                rolling_ret = labeled_data.set_index('date')['ret_simple']
                rolling_fig = plot_rolling_risk_metrics(rolling_ret, window=63)
                st.plotly_chart(rolling_fig, use_container_width=True)
                
            with col_risk_r:
                st.markdown("#### Predefined Crisis Stress Testing Comparison")
                # Stress test windows comparison
                if st.session_state.stress_test is not None:
                    stress_df = st.session_state.stress_test
                    st.dataframe(
                        stress_df.style.format({
                            "Total_Return": "{:.2%}",
                            "Max_Drawdown": "{:.2%}"
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("Stress testing report is empty.")
        else:
            st.warning("Regime risk report not found.")

    # ----------------------------------------------------
    # PAGE 6: MONTE CARLO PROJECTIONS
    # ----------------------------------------------------
    elif page == "Monte Carlo Simulator":
        st.markdown("### Stochastic Return Projections Simulator")
        st.markdown(
            "Simulate future market trajectories starting from the final historical date. "
            "Choose between global random draws, regime-conditioned returns, or Markov-chain transitions."
        )
        
        col_mc1, col_mc2, col_mc3 = st.columns(3)
        with col_mc1:
            mc_method = st.selectbox(
                "Simulation Methodology:",
                ["Markov-Chain Transition Simulation", "Historical Bootstrap (Unconditional)", "Regime-Conditioned Bootstrap"]
            )
        with col_mc2:
            mc_horizon = st.slider("Simulation Horizon (trading days):", min_value=10, max_value=252, value=126)
        with col_mc3:
            dd_threshold = st.slider("Drawdown Loss Alert Threshold (%):", min_value=5.0, max_value=50.0, value=15.0)
            
        # If regime-conditioned bootstrap chosen, show select state box
        conditioned_state = 0
        if mc_method == "Regime-Conditioned Bootstrap":
            state_label_map = labeled_data[['regime_state', 'regime_label']].drop_duplicates().set_index('regime_state')['regime_label'].to_dict()
            conditioned_state = st.selectbox(
                "Select Regime State to Condition On:",
                options=list(state_label_map.keys()),
                format_func=lambda x: f"State {x}: {state_label_map[x]}"
            )
            
        st.markdown("---")
        
        # Run simulation on-the-fly
        with st.spinner("Generating stochastic projections..."):
            simulator = MonteCarloSimulator(labeled_data, random_seed=int(mc_seed))
            
            paths = None
            if mc_method == "Historical Bootstrap (Unconditional)":
                paths = simulator.simulate_bootstrap(int(mc_sims), mc_horizon)
            elif mc_method == "Regime-Conditioned Bootstrap":
                paths = simulator.simulate_regime_conditioned(int(mc_sims), mc_horizon, conditioned_state)
            else:
                # Markov Chain
                # Check for custom transition matrix or fallback to computed
                states = labeled_data['regime_state'].values
                from src.analysis.regime_analysis import compute_transition_matrix
                transition_matrix = compute_transition_matrix(states, n_regimes)
                start_state = int(states[-1])
                paths = simulator.simulate_markov_chain(
                    n_sims=int(mc_sims),
                    horizon=mc_horizon,
                    start_state=start_state,
                    transition_matrix=transition_matrix
                )
                
            # Perform path calculations
            terminal_prices = paths[-1, :]
            terminal_returns = terminal_prices - 1.0
            
            # expected & percentiles
            expected_ret = float(np.mean(terminal_returns))
            median_ret = float(np.median(terminal_returns))
            p5 = float(np.percentile(terminal_returns, 5))
            p95 = float(np.percentile(terminal_returns, 95))
            prob_loss = float((terminal_returns < 0.0).sum() / int(mc_sims))
            
            # calculate max drawdown per path
            running_max = np.maximum.accumulate(paths, axis=0)
            drawdowns = (paths - running_max) / (running_max + 1e-15)
            max_dd_per_path = drawdowns.min(axis=0) # shape: [mc_sims]
            
            prob_dd_exceeded = float((max_dd_per_path < (-dd_threshold / 100.0)).sum() / int(mc_sims))
            avg_max_dd = float(np.mean(max_dd_per_path))
            
        col_mcp1, col_mcp2 = st.columns([2, 1])
        with col_mcp1:
            st.markdown("#### Projected Path Ribbons (Fan Chart)")
            fan_fig = plot_monte_carlo_paths(paths)
            st.plotly_chart(fan_fig, use_container_width=True)
            
            st.markdown("#### Terminal Returns Histogram & Tails")
            term_fig = plot_monte_carlo_distributions(paths)
            st.plotly_chart(term_fig, use_container_width=True)
            
        with col_mcp2:
            st.markdown("#### Projections Risk metrics")
            
            st.markdown(
                f"""
                <div class="metric-card" style="margin-bottom:15px;">
                    <div class="metric-label">Expected (Mean) Return</div>
                    <div class="metric-value">{expected_ret*100:.2f}%</div>
                </div>
                <div class="metric-card" style="margin-bottom:15px;">
                    <div class="metric-label">Median Return</div>
                    <div class="metric-value">{median_ret*100:.2f}%</div>
                </div>
                <div class="metric-card" style="margin-bottom:15px;">
                    <div class="metric-label">Probability of Capital Loss</div>
                    <div class="metric-value" style="color:#ef4444;">{prob_loss*100:.2f}%</div>
                </div>
                <div class="metric-card" style="margin-bottom:15px;">
                    <div class="metric-label">Probability of Drawdown &gt; {dd_threshold}%</div>
                    <div class="metric-value" style="color:#f97316;">{prob_dd_exceeded*100:.2f}%</div>
                </div>
                <div class="metric-card" style="margin-bottom:15px;">
                    <div class="metric-label">Average Path Max Drawdown</div>
                    <div class="metric-value">{avg_max_dd*100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.info(
                f"**Worst 5% Tail Return Bound**: {p5*100:.2f}%  \n"
                f"**Best 5% Tail Return Bound**: {p95*100:.2f}%"
            )

    # ----------------------------------------------------
    # PAGE 7: STRATEGY BACKTESTING
    # ----------------------------------------------------
    elif page == "Strategy Backtesting":
        st.markdown("### Tactical Asset Allocation Backtesting Terminal")
        st.markdown(
            "Backtest model-driven position weights. The engine ensures **zero lookahead bias** "
            "by shifting HMM state signals by 1 day, accounting for costs and execution slippage."
        )
        
        # Dynamic backtest calculations
        with st.spinner("Backtesting strategies dynamically..."):
            equity_curves_df, backtest_summary_df = run_custom_backtest(
                labeled_data,
                tc_input,
                slip_input
            )
            
        col_bt_l, col_bt_r = st.columns([2, 1])
        
        with col_bt_l:
            st.markdown("#### Cumulative Strategy Performance")
            eq_fig = plot_equity_curves(equity_curves_df)
            st.plotly_chart(eq_fig, use_container_width=True)
            
            st.markdown("#### Drawdown Comparison Profiles")
            dd_fig = plot_drawdowns(equity_curves_df)
            st.plotly_chart(dd_fig, use_container_width=True)
            
        with col_bt_r:
            st.markdown("#### Strategy Metrics Comparison Table")
            st.dataframe(
                backtest_summary_df.style.highlight_max(subset=["CAGR", "Sharpe_Ratio", "Sortino_Ratio", "Calmar_Ratio"], color="#065f46")
                                          .highlight_min(subset=["Max_Drawdown"], color="#111827")
                                          .format({
                                              "CAGR": "{:.2%}",
                                              "Annualized_Volatility": "{:.2%}",
                                              "Sharpe_Ratio": "{:.3f}",
                                              "Sortino_Ratio": "{:.3f}",
                                              "Calmar_Ratio": "{:.3f}",
                                              "Max_Drawdown": "{:.2%}",
                                              "Average_Drawdown": "{:.2%}",
                                              "Drawdown_Duration_Days": "{:.1f}",
                                              "Win_Rate": "{:.2%}",
                                              "Profit_Factor": "{:.3f}",
                                              "Total_Turnover": "{:.2f}",
                                              "Exposure_Days_Pct": "{:.2%}"
                                          }),
                use_container_width=True
            )
            
            # Details expander
            with st.expander("Tactical Signal Allocation Rules"):
                st.markdown(
                    """
                    *   **Buy & Hold**: Long NIFTY 50 (1.0 weight).
                    *   **EMA Crossover**: 100% position weight if EMA 50 > EMA 200, else 0% (sits in cash).
                    *   **Regime-Aware Strategy**:
                        *   *Bullish Low Volatility*: **100%** weight.
                        *   *Recovery / Bullish High Vol*: **50%** weight.
                        *   *Sideways Low Vol*: **25%** weight.
                        *   *Bearish High Vol / Distribution / Risk-Off*: **0%** weight.
                    *   **Hybrid Regime + Trend**: Long 100% only if the EMA 50/200 crossover is bullish AND the current regime is Bullish Low Volatility. Else goes to cash.
                    """
                )

    # ----------------------------------------------------
    # PAGE 8: FINAL REPORT & EXPORTS
    # ----------------------------------------------------
    elif page == "Final Report & Exports":
        st.markdown("### Export Center & Career Presentation Summary")
        st.markdown("Export portfolio metrics to share with users and recruiters.")
        
        # Calculate dynamic backtests matching current sliders for reports
        with st.spinner("Compiling latest reports..."):
            equity_df, backtest_df = run_custom_backtest(labeled_data, tc_input, slip_input)
            regime_risk_df = st.session_state.regime_risk
            baseline_df = st.session_state.baseline_risk
            
        col_rep1, col_rep2 = st.columns(2)
        
        with col_rep1:
            st.markdown("#### CSV Data Downloads")
            
            # Backtest summary
            bt_csv = backtest_df.to_csv(index=False)
            st.download_button(
                label="Download Strategy Backtest Summary CSV",
                data=bt_csv,
                file_name="nifty_backtest_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Regime risk statistics
            risk_csv = regime_risk_df.to_csv(index=False) if regime_risk_df is not None else ""
            st.download_button(
                label="Download Regime Risk Decomposition CSV",
                data=risk_csv,
                file_name="nifty_regime_risk_report.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with col_rep2:
            st.markdown("#### Interactive HTML Exports")
            
            # Generate HTML
            html_report = generate_html_report(backtest_df, regime_risk_df, baseline_df)
            st.download_button(
                label="Download Consolidated HTML Research Report",
                data=html_report,
                file_name="nifty50_regime_risk_report.html",
                mime="text/html",
                use_container_width=True
            )
            
        st.markdown("---")
            
    # Render global educational disclaimer
    render_educational_disclaimer()

if __name__ == "__main__":
    main()
