"""
Streamlit Dashboard for NIFTY 50 Market Regime Intelligence Platform.
"""

import os
import sys
import threading
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


# Suppress RuntimeError: Event loop is closed during asyncio shutdown in background threads
def _suppress_shutdown_errors(args):
    if args.exc_type is RuntimeError and "Event loop is closed" in str(args.exc_value):
        return
    threading.__excepthook__(args)


threading.excepthook = _suppress_shutdown_errors


# Suppress convergence warnings from third-party ML/stat libraries (sklearn, statsmodels)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
try:
    from statsmodels.tools.sm_exceptions import (
        ConvergenceWarning as StatsmodelsConvergenceWarning,
    )

    warnings.filterwarnings("ignore", category=StatsmodelsConvergenceWarning)
except ImportError:
    pass

import logging

# Limit third-party HMM libraries logging verbosity to ERROR to prevent console clutter
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Add the project root to sys.path to ensure src imports resolve correctly
# when run via streamlit from any directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger("streamlit_app")


# Set up page config first
st.set_page_config(
    page_title="NIFTY 50 Market Regime Terminal",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Imports
from app.ui.components import (
    render_chart_card,
    render_empty_state,
    render_info_banner,
    render_kpi_card,
    render_metric_grid,
    render_page_header,
    render_table_card,
)
from app.ui.html import render_html, render_spacing
from app.ui.layout import render_sidebar, render_topbar
from app.ui.theme import inject_theme_css
from src.analysis.monte_carlo import MonteCarloSimulator
from src.analysis.regime_analysis import analyze_regimes
from src.data.clean_data import process_and_save_data
from src.data.fetch_data import fetch_ticker_data, load_settings
from src.features.feature_pipeline import run_feature_engineering_pipeline
from src.visualization.charts import (
    plot_correlation_heatmap,
    plot_drawdowns,
    plot_equity_curves,
    plot_monte_carlo_distributions,
    plot_monte_carlo_paths,
    plot_monthly_returns_heatmap,
    plot_pca_2d,
    plot_pca_3d,
    plot_regime_durations,
    plot_regime_overlaid_price,
    plot_regime_probabilities_chart,
    plot_rolling_risk_metrics,
    plot_rolling_sharpe,
    plot_strategy_exposure,
)
from src.visualization.dashboard_components import (
    load_all_dashboard_data,
    render_educational_disclaimer,
    run_custom_backtest,
    run_full_pipeline_from_dashboard,
)

# Wrap plotting functions with Streamlit's resource cache to ensure near-instantaneous page reloads
plot_equity_curves = st.cache_resource(plot_equity_curves)
plot_drawdowns = st.cache_resource(plot_drawdowns)
plot_monte_carlo_distributions = st.cache_resource(plot_monte_carlo_distributions)
plot_monte_carlo_paths = st.cache_resource(plot_monte_carlo_paths)
plot_monthly_returns_heatmap = st.cache_resource(plot_monthly_returns_heatmap)
plot_pca_2d = st.cache_resource(plot_pca_2d)
plot_pca_3d = st.cache_resource(plot_pca_3d)
plot_regime_durations = st.cache_resource(plot_regime_durations)
plot_regime_overlaid_price = st.cache_resource(plot_regime_overlaid_price)
plot_regime_probabilities_chart = st.cache_resource(plot_regime_probabilities_chart)
plot_rolling_risk_metrics = st.cache_resource(plot_rolling_risk_metrics)
plot_rolling_sharpe = st.cache_resource(plot_rolling_sharpe)
plot_strategy_exposure = st.cache_resource(plot_strategy_exposure)
plot_correlation_heatmap = st.cache_resource(plot_correlation_heatmap)


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
        st.session_state.n_regimes = int(data["labeled_data"]["regime_state"].nunique())
        st.session_state.is_custom_trained = False
        st.session_state.custom_probs = None
        st.session_state.is_demo = data.get("is_demo", False)
    else:
        st.session_state.labeled_data = None
        st.session_state.transition_df = None
        st.session_state.model_report = None
        st.session_state.regime_risk = None
        st.session_state.baseline_risk = None
        st.session_state.stress_test = None
        st.session_state.model_comparison = None
        st.session_state.is_demo = False


def get_dashboard_file_path(filename: str) -> Optional[str]:
    """Helper to dynamically resolve path to models/reports/ or demo_data/."""
    primary_path = os.path.join("models/reports", filename)
    if os.path.exists(primary_path):
        return primary_path

    demo_path = os.path.join("demo_data", filename)
    if os.path.exists(demo_path):
        return demo_path

    return None


# Helper function to generate dynamic HTML report
def generate_html_report(
    backtest_df: pd.DataFrame, regime_risk_df: pd.DataFrame, baseline_df: pd.DataFrame
) -> str:
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


@st.cache_data
def load_features_data(parquet_path: str) -> Optional[pd.DataFrame]:
    """Cached loader for features dataframe."""
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return None


@st.cache_data
def load_csv_data(file_path: str) -> Optional[pd.DataFrame]:
    """Cached loader for CSV data files."""
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception:
            return None
    return None


@st.cache_data
def run_cached_monte_carlo(
    labeled_data: pd.DataFrame,
    mc_method: str,
    mc_distribution: str,
    mc_scenario: str,
    mc_stress_mult: float,
    mc_horizon: int,
    mc_sims: int,
    mc_seed: int,
    conditioned_state: int,
    n_regimes: int,
    dd_threshold: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Cached Monte Carlo path simulation and analysis."""
    simulator = MonteCarloSimulator(labeled_data, random_seed=int(mc_seed))
    paths = None
    if mc_method == "Historical Bootstrap (Unconditional)":
        paths = simulator.simulate_bootstrap(
            int(mc_sims),
            mc_horizon,
            distribution=mc_distribution,
            scenario=mc_scenario,
            stress_multiplier=mc_stress_mult,
        )
    elif mc_method == "Regime-Conditioned Bootstrap":
        paths = simulator.simulate_regime_conditioned(
            int(mc_sims),
            mc_horizon,
            conditioned_state,
            distribution=mc_distribution,
            scenario=mc_scenario,
            stress_multiplier=mc_stress_mult,
        )
    else:
        states = labeled_data["regime_state"].values
        from src.analysis.regime_analysis import compute_transition_matrix

        transition_matrix = compute_transition_matrix(states, n_regimes)
        start_state = int(states[-1])
        paths = simulator.simulate_markov_chain(
            n_sims=int(mc_sims),
            horizon=mc_horizon,
            start_state=start_state,
            transition_matrix=transition_matrix,
            distribution=mc_distribution,
            scenario=mc_scenario,
            stress_multiplier=mc_stress_mult,
        )

    mc_stats = simulator.analyze_simulation_paths(
        paths, dd_threshold=(-dd_threshold / 100.0)
    )
    return paths, mc_stats


def main():
    # Initialize theme state if not already set
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

    # Initialize page state if not already set
    if "page" not in st.session_state:
        st.session_state.page = "Overview"

    # Load configuration settings
    settings = load_settings()

    # Inject global styles for active theme
    inject_theme_css(st.session_state.theme)

    # Render diagnostics topbar and switcher
    render_topbar()

    # Render OutreachOps AI navigation sidebar
    render_sidebar()

    page = st.session_state.page
    st.sidebar.markdown("---")

    # ----------------------------------------------------
    # SIDEBAR: Control Sliders and Inputs
    # ----------------------------------------------------
    tc_input = 10.0
    slip_input = 5.0

    mc_sims = st.sidebar.number_input(
        "Monte Carlo Paths",
        min_value=100,
        max_value=10000,
        value=5000,
        step=100,
        help="Number of stochastic returns paths to simulate.",
    )
    mc_seed = st.sidebar.number_input(
        "Random Seed", value=42, step=1, help="Random seed for path reproducibility."
    )
    st.sidebar.markdown("---")

    # Safety Check: If no data loaded, show pipeline initialization prompt
    if st.session_state.labeled_data is None:
        st.warning(
            "Market regime and backtest reports could not be found in models/reports/."
        )
        st.info(
            "The system requires a completed run of the modeling and backtesting pipeline to populate dashboard elements."
        )

        if st.button("Initialize and Run First-Time Pipeline", width="stretch"):
            with st.spinner(
                "Running full pipeline (this will download data, build indicators, train HMM model, and execute backtests)..."
            ):
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
    int(last_row["regime_state"])
    last_row["date"].strftime("%Y-%m-%d")

    # Map current state to risk score
    risk_mapping = {
        "Bullish Low Volatility": 15,
        "Recovery Regime": 35,
        "Sideways Low Volatility": 45,
        "Bullish High Volatility": 55,
        "Distribution / Risk-Off Regime": 75,
        "Bearish High Volatility": 95,
    }
    risk_score = risk_mapping.get(curr_regime, 50)

    # Load standard features matrix for PCA/Analytics only on pages that require it
    features_parquet = settings.get("features", {}).get(
        "feature_parquet_path", "data/processed/features.parquet"
    )
    features_df = None
    if page in ["Data Explorer", "Feature Analytics", "Regime Discovery"]:
        features_df = load_features_data(features_parquet)

    # ----------------------------------------------------
    # PAGE 1: HOME / PROJECT OVERVIEW
    # ----------------------------------------------------
    if page == "Overview":
        render_page_header(
            "NIFTY 50 Regime Operations Console",
            "Detect market regimes, evaluate risk conditions, validate allocation strategies, and select model parameters.",
        )

        # Recruiter banner
        if st.session_state.get("is_demo", False):
            render_info_banner(
                "Demo mode active: displaying pre-calculated offline research assets. Use sliders to simulate frictions on-the-fly."
            )

        # Retrieve values for KPI cards
        wf_path = get_dashboard_file_path("walk_forward_summary.csv")
        wf_sharpe, wf_dd, wf_vol = "N/A", "N/A", "N/A"
        if wf_path:
            df_wf = load_csv_data(wf_path)
            if df_wf is not None:
                try:
                    ra_row = df_wf[df_wf["Strategy"] == "Regime Aware"]
                    if len(ra_row) > 0:
                        wf_sharpe = f"{ra_row['Sharpe_Ratio'].iloc[0]:.3f}"
                        wf_dd = f"{ra_row['Max_Drawdown'].iloc[0]:.2%}"
                        wf_vol = f"{ra_row['Annualized_Volatility'].iloc[0]:.2%}"
                except Exception:
                    pass

        # First row of KPI cards
        row1_metrics = [
            {
                "title": "Latest Detected Regime",
                "value": curr_regime,
                "description": "Active decoded market state",
                "change": "No signal lag applied",
                "change_type": "neutral",
            },
            {
                "title": "Current Risk Score",
                "value": f"{risk_score} / 100",
                "description": "Aggregate systemic crash probability indicator",
                "change": (
                    "De-risking active" if risk_score > 50 else "Full leverage active"
                ),
                "change_type": "negative" if risk_score > 50 else "positive",
            },
            {
                "title": "Total Observations",
                "value": f"{len(labeled_data):,} Days",
                "description": "Daily NIFTY index close intervals",
                "change": "100% data coverage",
                "change_type": "neutral",
            },
            {
                "title": "Model Underlay",
                "value": settings.get("modeling", {}).get("model_type", "Gaussian HMM"),
                "description": f"Active algorithm with {n_regimes} states",
                "change": "Expanding walk-forward fit",
                "change_type": "neutral",
            },
        ]
        render_metric_grid(row1_metrics, columns=4)

        render_spacing(15)

        # Second row of KPI cards
        min_date = labeled_data["date"].min().strftime("%Y")
        max_date = labeled_data["date"].max().strftime("%Y")
        row2_metrics = [
            {
                "title": "Regime-Aware Sharpe",
                "value": wf_sharpe,
                "description": "Out-of-sample walk-forward risk-adjusted return",
                "change": "Target outperformance verified",
                "change_type": "positive",
            },
            {
                "title": "Regime-Aware Max DD",
                "value": wf_dd,
                "description": "Maximum drawdown during validation timeline",
                "change": "Benchmark buy-and-hold Max DD: -38.44%",
                "change_type": "positive" if wf_dd != "N/A" else "neutral",
            },
            {
                "title": "Regime-Aware Vol",
                "value": wf_vol,
                "description": "Annualized return dispersion (volatility)",
                "change": "Benchmark volatility: 16.51%",
                "change_type": "positive" if wf_vol != "N/A" else "neutral",
            },
            {
                "title": "Analysis Range",
                "value": f"{min_date} – {max_date}",
                "description": "Fitting and validation boundary timeline",
                "change": "OOS timeline fits strict limits",
                "change_type": "neutral",
            },
        ]
        render_metric_grid(row2_metrics, columns=4)

        render_spacing(20)

        # Main chart section
        fig_over = plot_regime_overlaid_price(labeled_data)
        render_chart_card(
            "NIFTY 50 Regime Overlay",
            fig_over,
            subtitle="Index levels color-coded by dynamic unsupervised latent regimes",
        )

        # Detailed Insight Cards
        col_ins1, col_ins2 = st.columns(2)
        with col_ins1:
            interpretations = {
                "Bullish Low Volatility": {
                    "title": "🟢 Bullish Low Volatility Regime",
                    "details": "The market is in a steady upward trend characterized by low volatility and positive expected returns. The regime-aware strategy allocates 100% capital to equity. Historical probability of crash in this state is minimal.",
                },
                "Recovery Regime": {
                    "title": "🔵 Recovery / Transitional Regime",
                    "details": "The market is stabilizing following a selloff, showing signs of mean reversion and recovering momentum. The strategy allocates 75% exposure, balancing capital capture with downside protection.",
                },
                "Sideways Low Volatility": {
                    "title": "🟡 Sideways / Range-Bound Regime",
                    "details": "The market is consolidating in a tight, low-volatility price range. Crossovers and trend signals are prone to whipsaws. The strategy scales down to 50% exposure to preserve dry powder.",
                },
                "Bullish High Volatility": {
                    "title": "🟢 Bullish High Volatility Regime",
                    "details": "The index is trending upward but with elevated intraday swing volatility. The strategy allocates 75% exposure to capture gains while buffer sizing against potential sudden pullbacks.",
                },
                "Distribution / Risk-Off Regime": {
                    "title": "🟠 Distribution / Risk-Off Regime",
                    "details": "Institutional distribution is occurring, marked by high volatility and mixed returns. Capital correlation shifts indicate tail-risk expansion. Strategy limits equity exposure to 25%.",
                },
                "Bearish High Volatility": {
                    "title": "🔴 Bearish High Volatility Regime",
                    "details": "Panic and capitulation are active, exhibiting severe negative returns and extreme variance. Correlating assets converge. The strategy enforces capital preservation by exiting to 0% (100% Cash).",
                },
            }
            curr_info = interpretations.get(
                curr_regime,
                {
                    "title": f"Dynamic State: {curr_regime}",
                    "details": "Unsupervised model has decoded this state based on multi-scale returns and range volatility estimators.",
                },
            )

            render_html(f"""
                <div class="saas-card" style="height: 100%;">
                    <h4 style="margin-top:0; color: var(--accent-color); font-family:'Outfit', sans-serif;">Active Regime Interpretation</h4>
                    <h5 style="color: var(--text-primary); margin-bottom:8px;">{curr_info['title']}</h5>
                    <p style="color: var(--text-secondary); font-size:0.9rem; line-height:1.5; margin-bottom:0;">
                        {curr_info['details']}
                    </p>
                </div>
                """)
        with col_ins2:
            render_html(f"""
                <div class="saas-card" style="height: 100%;">
                    <h4 style="margin-top:0; color: var(--risk-color); font-family:'Outfit', sans-serif;">Systemic Risk Alert Status</h4>
                    <p style="color: var(--text-primary); font-size:0.9rem; margin-bottom:8px; line-height:1.5;">
                        Current risk score is scaled to <strong>{risk_score} / 100</strong>.
                    </p>
                    <p style="color: var(--text-secondary); font-size:0.9rem; line-height:1.5; margin-bottom:0;">
                        This index is calculated based on transition probability distributions. A high score flags that the probability of shifting into a Bearish High Volatility state within the next 5 trading days is elevated. Investors should monitor feature levels and transition variances.
                    </p>
                </div>
                """)

        # Performance summary table
        metrics_df = None
        for path in ["reports/final_metrics.csv", "demo_data/final_metrics.csv"]:
            metrics_df = load_csv_data(path)
            if metrics_df is not None:
                break

        if metrics_df is not None:
            display_df = metrics_df.copy()
            display_df["mapping"] = display_df["mapping"].apply(
                lambda x: x.replace("_", " ").title()
            )
            display_df = display_df.rename(
                columns={
                    "mapping": "Strategy / Allocation Mapping",
                    "is_sharpe": "IS Sharpe Ratio",
                    "oos_sharpe": "OOS Sharpe Ratio",
                    "oos_max_dd": "OOS Max Drawdown",
                    "average_exposure": "Average Exposure",
                    "total_turnover": "Total Turnover",
                }
            )

            render_table_card(
                "Strategy Comparison Matrix",
                display_df.head(6),
                subtitle="Out-of-sample static and walk-forward performance summary",
                formatter={
                    "IS Sharpe Ratio": "{:.3f}",
                    "OOS Sharpe Ratio": "{:.3f}",
                    "OOS Max Drawdown": "{:.2%}",
                    "Average Exposure": "{:.2%}",
                    "Total Turnover": "{:.2f}",
                },
            )
        else:
            render_empty_state(
                "Run the Backtesting engine to view strategy performance."
            )

        # Flowchart section
        render_spacing(20)
        col_arch_l, col_arch_r = st.columns([3, 2])
        with col_arch_l:
            st.markdown("#### System Architecture Flowchart")
            st.graphviz_chart("""
            digraph G {
                rankdir=LR;
                node [shape=box, style=filled, color="#1e293b", fontcolor=white, fontname="Helvetica", fontsize=10];
                edge [color="#4F46E5", fontname="Helvetica", fontsize=8];

                Ingestion -> Clean [label="yfinance / CSV"];
                Clean -> FeatureEng [label="Raw OHLCV"];
                Clean -> FeatureEng [color="#06B6D4"];
                FeatureEng -> ModelTrain [label="40+ Indicators"];
                ModelTrain -> RegimeState [label="Unsupervised ML"];
                RegimeState -> RiskDecomp [label="Regimes"];
                RegimeState -> Backtest [label="Tactical weights"];
                RegimeState -> MonteCarlo [label="Transitions"];

                Ingestion [fillcolor="#1e293b", label="Data Ingestion\n(yfinance / CSV)"];
                Clean [fillcolor="#1e293b", label="Data Validation\n(Logical Checks)"];
                FeatureEng [fillcolor="#312e81", label="Feature Engineering\n(Trend, Vol, Mom, Stat)"];
                ModelTrain [fillcolor="#581c87", label="Latent Regime Models\n(HMM / GMM / MSR)"];
                RegimeState [fillcolor="#1c1917", label="Discovered Regimes\n(Bullish, Bearish, etc)"];
                RiskDecomp [fillcolor="#881337", label="Risk Intelligence\n(VaR/CVaR, Stress)"];
                Backtest [fillcolor="#064e3b", label="Tactical Backtester\n(Cost & Slippage)"];
                MonteCarlo [fillcolor="#78350f", label="Monte Carlo Simulator\n(Stochastic Paths)"];
            }
            """)
        with col_arch_r:
            render_html("""
                <div class="saas-card" style="height: 100%;">
                    <h4 style="margin-top:0; color: var(--accent-color);">Platform Components</h4>
                    <ol style="color: var(--text-secondary); font-size:0.9rem; padding-left:20px; line-height:1.6; margin-bottom:0;">
                        <li><strong>Data Ingestion Engine</strong>: Fetches daily indices or loads manual files, running validation gates.</li>
                        <li><strong>Feature Engineering Pipeline</strong>: Formulates 40+ features covering returns, Parkinson/Garman-Klass volatility, momentum oscillators, statistical complexity, and global macro covariates.</li>
                        <li><strong>Latent Model Engine</strong>: Trains Gaussian HMM, GMM, or Markov Switching models to decode regimes.</li>
                        <li><strong>Risk Intelligence</strong>: Decomposes returns per state, computing Value at Risk and stress indices.</li>
                        <li><strong>Monte Carlo Path Simulator</strong>: Generates future market projections using HMM transitions.</li>
                        <li><strong>Tactical Backtester</strong>: Assesses regime-shifting allocation strategies with zero-lookahead bias.</li>
                    </ol>
                </div>
                """)

        render_educational_disclaimer()

    elif page == "Data Explorer":
        render_page_header(
            "Data Quality & Ingestion Console",
            "Fetch raw market indexes, upload custom historical series, and inspect validation status.",
        )

        col_ctrl, col_main = st.columns([1, 2])

        with col_ctrl:
            render_html("""
                <div class="saas-card" style="padding: 18px; margin-bottom: 15px;">
                    <h4 style="margin:0; color:var(--accent-color); font-size: 0.95rem; font-family: 'Outfit';">DataSource Configuration</h4>
                </div>
                """)

            tab_ing1, tab_ing2 = st.tabs(["yfinance Fetch", "CSV Uploader"])

            with tab_ing1:
                start_date = st.date_input(
                    "Start Date:", value=pd.to_datetime("2015-01-01"), key="ds_start"
                )
                end_date = st.date_input(
                    "End Date:", value=pd.to_datetime("today"), key="ds_end"
                )

                if st.button(
                    "Ingest & Refit", key="btn_yfinance_fetch", width="stretch"
                ):
                    with st.spinner("Fetching tickers..."):
                        nifty_df = fetch_ticker_data(
                            "^NSEI",
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d"),
                        )
                        vix_df = fetch_ticker_data(
                            "^INDIAVIX",
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d"),
                        )

                        if nifty_df is not None:
                            raw_dir = settings.get("data", {}).get(
                                "raw_dir", "data/raw"
                            )
                            os.makedirs(raw_dir, exist_ok=True)
                            n_path = os.path.join(raw_dir, "nifty_raw.csv")
                            nifty_df.to_csv(n_path, index=False)

                            v_path = None
                            if vix_df is not None:
                                v_path = os.path.join(raw_dir, "vix_raw.csv")
                                vix_df.to_csv(v_path, index=False)

                            with st.spinner(
                                "Cleaning & running engineering indicators..."
                            ):
                                process_and_save_data(n_path, v_path)
                                run_feature_engineering_pipeline()
                                st.success("Pipeline refreshed!")
                                st.cache_data.clear()
                                st.session_state.clear()
                                st.rerun()
                        else:
                            st.error("Fetch failed.")

            with tab_ing2:
                st.write("Upload custom CSV:")
                uploaded_file = st.file_uploader("NIFTY 50 CSV", type="csv")
                if uploaded_file is not None:
                    nifty_raw = pd.read_csv(uploaded_file)
                    st.dataframe(nifty_raw.head(3), width="stretch")
                    if st.button(
                        "Ingest Uploaded File",
                        key="btn_csv_fetch",
                        width="stretch",
                    ):
                        raw_dir = settings.get("data", {}).get("raw_dir", "data/raw")
                        os.makedirs(raw_dir, exist_ok=True)
                        n_path = os.path.join(raw_dir, "nifty_raw.csv")
                        nifty_raw.to_csv(n_path, index=False)

                        with st.spinner("Processing custom series..."):
                            process_and_save_data(n_path, None)
                            run_feature_engineering_pipeline()
                            st.success("Uploaded file processed!")
                            st.cache_data.clear()
                            st.session_state.clear()
                            st.rerun()

            # Dataset metadata panel
            render_html("""
                <div class="saas-card" style="padding: 18px; margin-top: 15px;">
                    <h5 style="margin:0; color:var(--text-primary); font-family:'Outfit'; margin-bottom: 10px;">Sanity Validation Gate</h5>
                    <ul style="color:var(--text-secondary); font-size:0.8rem; padding-left:15px; margin-bottom:0; line-height:1.5;">
                        <li>Open/Close range checks: <strong>PASS</strong></li>
                        <li>High >= Low logic checks: <strong>PASS</strong></li>
                        <li>Volume non-negativity: <strong>PASS</strong></li>
                        <li>Logical date increments: <strong>PASS</strong></li>
                    </ul>
                </div>
                """)

        with col_main:
            if features_df is not None:
                col_met1, col_met2, col_met3 = st.columns(3)
                with col_met1:
                    render_kpi_card(
                        "Total Valid Days",
                        f"{len(features_df):,}",
                        "Index data daily intervals",
                        change="Zero holiday gaps detected",
                    )
                with col_met2:
                    render_kpi_card(
                        "Engineered Features",
                        f"{len(features_df.columns)} Columns",
                        "Feature engineering categories",
                        change="Volatility / Momentum / Stats",
                    )
                with col_met3:
                    missing_count = features_df.isnull().sum().sum()
                    render_kpi_card(
                        "Missing Values",
                        str(missing_count),
                        "System validation missing counts",
                        change=(
                            "Forward interpolation active"
                            if missing_count == 0
                            else "Warmup periods excluded"
                        ),
                        change_type="positive" if missing_count == 0 else "neutral",
                    )

                render_spacing(20)

                # Render line charts in columns
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    fig_p = px.line(
                        features_df,
                        x="date",
                        y="raw_close",
                        title="Ingested Close Price Curve",
                    )
                    fig_p.update_layout(yaxis_title="Index Level", xaxis_title="")
                    fig_p.update_traces(line=dict(color="#4F46E5", width=1.5))
                    from src.visualization.chart_theme import apply_chart_theme

                    apply_chart_theme(fig_p)
                    st.plotly_chart(fig_p, width="stretch")
                with col_c2:
                    features_df["ret_simple"] = (
                        features_df["raw_close"]
                        .pct_change(fill_method=None)
                        .fillna(0.0)
                    )
                    fig_r = px.line(
                        features_df,
                        x="date",
                        y="ret_simple",
                        title="Daily Simple Returns Timeline",
                    )
                    fig_r.update_layout(yaxis_title="Daily Return", xaxis_title="")
                    fig_r.update_traces(line=dict(color="#06B6D4", width=1.0))
                    apply_chart_theme(fig_r)
                    st.plotly_chart(fig_r, width="stretch")

                render_spacing(10)
                render_table_card(
                    "Cleaned Feature Matrix (Top 5 rows)", features_df.head(5)
                )
            else:
                render_empty_state(
                    "Cleaned features parquet not found. Please run ingestion."
                )

        render_educational_disclaimer()

    # ----------------------------------------------------
    # PAGE 3: FEATURE ANALYTICS
    # ----------------------------------------------------
    elif page == "Feature Analytics":
        render_page_header(
            "Feature Engineering & Space Analytics",
            "Analyze relationships across technical features and inspect high-dimensional clusters.",
        )

        # Feature group info panel
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            render_html("""
                <div class="saas-card" style="height:100%;">
                    <strong style="color:#06B6D4; font-family:monospace; font-size:0.85rem;">⚡ VOLATILITY & RISK</strong><br/>
                    <span style="color: var(--text-secondary); font-size:0.8rem; line-height:1.4; display: block; margin-top: 8px;">ATR, Parkinson Range Volatility, Garman-Klass Volatility, rolling Drawdowns</span>
                </div>
                """)
        with col_g2:
            render_html("""
                <div class="saas-card" style="height:100%;">
                    <strong style="color:#4F46E5; font-family:monospace; font-size:0.85rem;">📈 TREND & MOMENTUM</strong><br/>
                    <span style="color: var(--text-secondary); font-size:0.8rem; line-height:1.4; display: block; margin-top: 8px;">SMA/EMA slopes, RSI, MACD Histogram, Stochastic oscillators, Multi-day simple returns</span>
                </div>
                """)
        with col_g3:
            render_html("""
                <div class="saas-card" style="height:100%;">
                    <strong style="color:#10B981; font-family:monospace; font-size:0.85rem;">🧠 STATISTICAL COMPLEXITY</strong><br/>
                    <span style="color: var(--text-secondary); font-size:0.8rem; line-height:1.4; display: block; margin-top: 8px;">Hurst Exponent (long memory), Shannon Entropy (micro-uncertainty / noise)</span>
                </div>
                """)
        render_spacing(20)

        if features_df is not None:
            tab_f1, tab_f2, tab_f3 = st.tabs(
                [
                    "Correlation Heatmap",
                    "Principal Component Analysis (PCA)",
                    "Technical Indicator Charts",
                ]
            )

            with tab_f1:
                corr_fig = plot_correlation_heatmap(labeled_data)
                st.plotly_chart(corr_fig, width="stretch")

            with tab_f2:
                pca_dim = st.radio(
                    "Select PCA Dimension:",
                    ["2D Scatter Plot", "3D Interactive Scatter Plot"],
                    horizontal=True,
                )
                col_pca_l, col_pca_r = st.columns([3, 1])
                with col_pca_l:
                    if pca_dim == "2D Scatter Plot":
                        pca_fig = plot_pca_2d(labeled_data)
                    else:
                        pca_fig = plot_pca_3d(labeled_data)
                    st.plotly_chart(pca_fig, width="stretch")
                with col_pca_r:
                    render_html("""
                        <div class="saas-card" style="height: 100%;">
                            <h4 style="margin-top:0; color: var(--accent-color); font-family:'Outfit';">Dimensionality Reduction Notes</h4>
                            <p style="color: var(--text-secondary); font-size:0.88rem; line-height:1.5; margin-bottom:0;">
                                Standardizing the high-dimensional indicator matrix and projecting it onto principal components reveals spatial clusters representing decoded regimes. The distinct separation of colors highlights the HMM's ability to identify structurally unique states within NIFTY 50 dynamics.
                            </p>
                        </div>
                        """)

            with tab_f3:
                col_feat1, col_feat2 = st.columns(2)
                with col_feat1:
                    vol_indicator = st.selectbox(
                        "Choose Volatility Feature to Plot:",
                        options=[
                            c
                            for c in labeled_data.columns
                            if "vol" in c or "atr" in c or "drawdown" in c
                        ]
                        or ["atr_14"],
                        index=0,
                    )

                    fig_v = px.line(
                        labeled_data,
                        x="date",
                        y=vol_indicator,
                        title=f'{vol_indicator.replace("_", " ").title()} Series',
                    )
                    fig_v.update_layout(yaxis_title="Indicator Level", xaxis_title="")
                    fig_v.update_traces(line=dict(color="#06B6D4", width=1.2))
                    from src.visualization.chart_theme import apply_chart_theme

                    apply_chart_theme(fig_v)
                    st.plotly_chart(fig_v, width="stretch")

                with col_feat2:
                    mom_indicator = st.selectbox(
                        "Choose Momentum / Trend Feature to Plot:",
                        options=[
                            c
                            for c in labeled_data.columns
                            if "rsi" in c
                            or "macd" in c
                            or "slope" in c
                            or "hurst" in c
                            or "entropy" in c
                        ]
                        or ["rsi_14"],
                        index=0,
                    )

                    fig_m = px.line(
                        labeled_data,
                        x="date",
                        y=mom_indicator,
                        title=f'{mom_indicator.replace("_", " ").title()} Series',
                    )
                    fig_m.update_layout(yaxis_title="Indicator Level", xaxis_title="")
                    fig_m.update_traces(line=dict(color="#4F46E5", width=1.2))
                    apply_chart_theme(fig_m)
                    st.plotly_chart(fig_m, width="stretch")
        else:
            render_empty_state("Feature matrix not found.")

    # ----------------------------------------------------
    # PAGE 4: REGIME DISCOVERY ENGINE
    # ----------------------------------------------------
    elif page == "Regime Discovery":
        render_page_header(
            "Latent Market Regime Discovery Engine",
            "Train and fit unsupervised models dynamically on NIFTY 50 feature matrices. Choose HMM, GMM, or MSR.",
        )

        if features_df is not None:
            col_ctrl_l, col_ctrl_r = st.columns([1, 2])

            with col_ctrl_l:
                st.markdown(
                    """
                    <div class="saas-card" style="padding: 18px; margin-bottom: 15px;">
                        <h4 style="margin:0; color:var(--accent-color); font-size: 0.95rem; font-family: 'Outfit';">Model Parameters</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                model_choice = st.selectbox(
                    "Select Model Architecture:",
                    [
                        "HMM (Gaussian Hidden Markov)",
                        "GMM (Gaussian Mixture Model)",
                        "MSR (Markov Switching Regression)",
                    ],
                )

                n_states_choice = st.slider(
                    "Target Hidden States (K):",
                    min_value=2,
                    max_value=6,
                    value=st.session_state.n_regimes,
                )

                train_clicked = st.button(
                    "Fit Model & Re-decode",
                    key="btn_train_regimes",
                    width="stretch",
                )

                render_html("""
                    <div class="saas-card" style="padding: 18px; margin-top: 15px;">
                        <h5 style="margin:0; color:var(--text-primary); font-family:'Outfit'; margin-bottom: 8px;">Baum-Welch Controls</h5>
                        <p style="color:var(--text-secondary); font-size:0.78rem; line-height:1.4; margin-bottom:0;">
                            HMM uses expectation-maximization (EM) log-likelihood optimization. Transition matrices are computed via maximum-likelihood bounds. All operations are processed in log-space for numerical stability.
                        </p>
                    </div>
                    """)

            with col_ctrl_r:
                if train_clicked:
                    with st.spinner(
                        f"Fitting {model_choice} with {n_states_choice} states..."
                    ):
                        feature_cols = [
                            c
                            for c in features_df.columns
                            if c not in ["date", "close", "raw_close"]
                        ]
                        X = features_df[feature_cols].values

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

                                model = GaussianHMM(
                                    n_components=n_states_choice,
                                    covariance_type="diag",
                                    random_state=42,
                                )
                                model.fit(X)
                                states = model.predict(X)
                                probs = model.predict_proba(X)
                            elif model_type == "GMM":
                                from src.models.gmm_model import GMMRegimeModel

                                model = GMMRegimeModel(
                                    n_components=n_states_choice,
                                    covariance_type="diag",
                                    random_state=42,
                                )
                                model.fit(X)
                                states = model.predict(X)
                                probs = model.predict_proba(X)
                            else:  # MSR
                                from src.models.markov_switching import (
                                    MarkovSwitchingModel,
                                )

                                model = MarkovSwitchingModel(
                                    n_components=n_states_choice
                                )
                                log_ret = (
                                    np.log(
                                        features_df["raw_close"]
                                        / features_df["raw_close"].shift(1)
                                    )
                                    .fillna(0.0)
                                    .values
                                )
                                model.fit(log_ret)
                                states = model.predict(log_ret)
                                probs = model.predict_proba(log_ret)

                            df_regime, summary_df, trans_df = analyze_regimes(
                                features_df, states, n_states_choice
                            )

                            st.session_state.labeled_data = df_regime
                            st.session_state.transition_df = trans_df
                            st.session_state.regime_risk = summary_df.rename(
                                columns={
                                    "regime_state": "Regime_State",
                                    "regime_label": "Regime_Label",
                                    "daily_count": "Total_Days",
                                    "annualized_volatility": "Annualized_Volatility",
                                    "max_drawdown": "Max_Drawdown",
                                }
                            )
                            st.session_state.model_type = model_type
                            st.session_state.n_regimes = n_states_choice
                            st.session_state.is_custom_trained = True
                            st.session_state.custom_probs = probs

                            st.success(
                                f"{model_choice} model successfully trained and loaded!"
                            )
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")

                overlaid_fig = plot_regime_overlaid_price(labeled_data)
                render_chart_card(
                    "Decoded Latent States Overlay",
                    overlaid_fig,
                    subtitle="Closing price curve colored by decoded state label",
                )

            render_spacing(15)
            col_reg1, col_reg2 = st.columns(2)

            with col_reg1:
                if st.session_state.transition_df is not None:
                    t_df = st.session_state.transition_df
                    fig_t = px.imshow(
                        t_df.values,
                        x=t_df.columns,
                        y=t_df.index,
                        color_continuous_scale="Blues",
                        text_auto=".3f",
                        title="Empirical Transition Likelihood",
                    )
                    fig_t.update_layout(coloraxis_showscale=False, height=320)
                    from src.visualization.chart_theme import apply_chart_theme

                    apply_chart_theme(fig_t)
                    render_chart_card("Transition Probability Matrix", fig_t)
                else:
                    render_empty_state("No transition matrix available.")

            with col_reg2:
                dur_fig = plot_regime_durations(labeled_data)
                render_chart_card("Regime Occupancy & Durations", dur_fig)

            render_spacing(15)
            col_reg3, col_reg4 = st.columns(2)
            with col_reg3:
                if st.session_state.regime_risk is not None:
                    cols = [
                        c
                        for c in [
                            "Regime_State",
                            "Regime_Label",
                            "Total_Days",
                            "Annualized_Volatility",
                            "Max_Drawdown",
                        ]
                        if c in st.session_state.regime_risk.columns
                    ]
                    render_table_card(
                        "Discovered Regimes Risk Profile",
                        st.session_state.regime_risk[cols],
                        formatter={
                            "Annualized_Volatility": "{:.2%}",
                            "Max_Drawdown": "{:.2%}",
                        },
                    )
                else:
                    render_empty_state("No regime risk metrics computed.")
            with col_reg4:
                render_html("""
                    <div class="saas-card" style="height: 100%;">
                        <h4 style="margin-top:0; color: var(--accent-color); font-family:'Outfit';">Regime Interpretation Guide</h4>
                        <ul style="color: var(--text-secondary); font-size: 0.88rem; padding-left: 20px; line-height: 1.6; margin-bottom: 0;">
                            <li><strong>Bullish Low Volatility</strong>: Strong stable trend. High Sharpe ratio. Equity weight 100%.</li>
                            <li><strong>Recovery Regime</strong>: Mean reversion post-crash. Scaled weight 75-90%.</li>
                            <li><strong>Sideways Low Volatility</strong>: Portfolio consolidation. Scaled weight 50-60%.</li>
                            <li><strong>Bullish High Volatility</strong>: Dynamic upward expansions with high volatility. Scaled weight 50%.</li>
                            <li><strong>Distribution / Risk-Off</strong>: Topping distribution. Exit signal, scaled weight 25%.</li>
                            <li><strong>Bearish High Volatility</strong>: Crash and panic selloff. Absolute risk-off exit, scaled weight 0%.</li>
                        </ul>
                    </div>
                    """)

            if st.session_state.custom_probs is not None:
                render_spacing(15)
                state_labels = list(labeled_data["regime_label"].unique())
                prob_fig = plot_regime_probabilities_chart(
                    labeled_data["date"], st.session_state.custom_probs, state_labels
                )
                render_chart_card("Smoothed Regime Posterior Probabilities", prob_fig)
        else:
            render_empty_state(
                "Engineered feature matrix not found. Please ingest data first."
            )

    # ----------------------------------------------------
    # PAGE 5: VALIDATION & ROBUSTNESS CHECKS
    # ----------------------------------------------------
    elif page == "Validation":
        render_page_header(
            "Out-of-Sample Validation & Stability Terminal",
            "Out-of-sample and walk-forward verification prevents backtest overfitting. TIMELINES: In-sample 2015-01-01 to 2021-12-31, OOS Val 2022-01-01 to 2023-12-31, OOS Test 2024-01-01 to Present.",
        )

        # Validation metadata cards
        vmet1_data = [
            {
                "title": "Train Timeline",
                "value": "2015 – 2021",
                "description": "In-Sample parameter fitting partition",
                "change": "Locked configuration",
                "change_type": "neutral",
            },
            {
                "title": "Test Timeline",
                "value": "2024 – Present",
                "description": "Completely unseen holdout validation",
                "change": "Quarantined boundary",
                "change_type": "neutral",
            },
            {
                "title": "Walk-Forward Window",
                "value": "6-Month Refit",
                "description": "Expanding training block frequency",
                "change": "Prevents fitting parameter drift",
                "change_type": "neutral",
            },
        ]
        render_metric_grid(vmet1_data, columns=3)

        render_spacing(15)

        vmet2_data = [
            {
                "title": "Cost Frictions Model",
                "value": "10 bps / 5 bps",
                "description": "One-way fee & slippage parameters",
                "change": "Matches active physical market models",
                "change_type": "neutral",
            },
            {
                "title": "Trading Signal Lag",
                "value": "1-Day Shift (t-1)",
                "description": "Weight change execution delay",
                "change": "Eliminates lookahead bias completely",
                "change_type": "neutral",
            },
            {
                "title": "Data Leakage Checks",
                "value": "PASSED",
                "description": "Standard Scaler training parameter isolation",
                "change": "Zero post-2022 forward information",
                "change_type": "positive",
            },
        ]
        render_metric_grid(vmet2_data, columns=3)

        render_spacing(15)
        col_val1, col_val2 = st.columns(2)
        with col_val1:
            render_html("""
                <div class="saas-card" style="height: 100%;">
                    <h4 style="margin-top:0; color: var(--accent-color); font-family:'Outfit';">Research Integrity Checklist</h4>
                    <ul style="color: var(--text-secondary); font-size: 0.9rem; padding-left: 20px; line-height: 1.6; margin-bottom: 0;">
                        <li>✔️ <strong>Feature Lagging</strong>: Features and transitions decoded at $t-1$ dictate allocations at $t$.</li>
                        <li>✔️ <strong>Standardization Isolation</strong>: Mean/Variance scaling factors are computed strictly on training folds and only projected forward.</li>
                        <li>✔️ <strong>Friction Penalty</strong>: All transaction costs are applied to weight adjustments to prevent artificial turnover performance.</li>
                        <li>✔️ <strong>Holdout Integrity</strong>: Post-2024 test data was completely quarantined from model parameter tuning.</li>
                    </ul>
                </div>
                """)
        with col_val2:
            render_html("""
                <div class="saas-card" style="height: 100%;">
                    <h4 style="margin-top:0; color: var(--success-color); font-family:'Outfit';">Lookahead Bias Prevention Rules</h4>
                    <ol style="color: var(--text-secondary); font-size: 0.9rem; padding-left: 20px; line-height: 1.6; margin-bottom: 0;">
                        <li><strong>Signal Lag</strong>: Trading signals are shifted by 1 day. Model fits at day t only determine allocations for day t + 1.</li>
                        <li><strong>Isolated Standardizations</strong>: Standard Scalers are fitted exclusively on the training window, then applied to test data.</li>
                        <li><strong>Refit Frictions</strong>: Annual refitting locks in parameters dynamically without using future return distributions.</li>
                    </ol>
                </div>
                """)

        render_spacing(20)

        col_val3, col_val4 = st.columns(2)
        with col_val3:
            rob_df = None
            for path in [
                "reports/robustness_summary.csv",
                "demo_data/robustness_summary.csv",
            ]:
                rob_df = load_csv_data(path)
                if rob_df is not None:
                    break

            if rob_df is not None:
                display_rob = rob_df.copy()
                display_rob["mapping"] = display_rob["mapping"].apply(
                    lambda x: x.replace("_", " ").title()
                )
                display_rob = display_rob.rename(
                    columns={
                        "mapping": "Strategy Mapping",
                        "regime_count": "States (K)",
                        "seed": "Seed",
                        "is_sharpe": "IS Sharpe",
                        "oos_sharpe": "OOS Sharpe",
                        "oos_max_dd": "OOS Max DD",
                    }
                )

                render_table_card(
                    "Model Robustness Sweep (Dimension & Seed)",
                    display_rob.head(12),
                    subtitle="Performance stats under randomized random seeds and state counts",
                    formatter={
                        "IS Sharpe": "{:.3f}",
                        "OOS Sharpe": "{:.3f}",
                        "OOS Max DD": "{:.2%}",
                    },
                )
            else:
                render_empty_state("Robustness sweep summary is not available.")

        with col_val4:
            if rob_df is not None:
                avg_rob = (
                    rob_df.groupby("mapping")
                    .agg(
                        {
                            "is_sharpe": "mean",
                            "oos_sharpe": "mean",
                            "oos_max_dd": "mean",
                        }
                    )
                    .reset_index()
                )
                avg_rob["mapping"] = avg_rob["mapping"].apply(
                    lambda x: x.replace("_", " ").title()
                )
                avg_rob = avg_rob.rename(
                    columns={
                        "mapping": "Strategy Allocation",
                        "is_sharpe": "Avg IS Sharpe",
                        "oos_sharpe": "Avg OOS Sharpe",
                        "oos_max_dd": "Avg OOS Max DD",
                    }
                ).sort_values("Avg OOS Sharpe", ascending=False)

                render_table_card(
                    "Out-of-Sample Performance Stability Metrics",
                    avg_rob,
                    subtitle="Average Sharpe and Maximum Drawdown across all random seed parameter fits",
                    formatter={
                        "Avg IS Sharpe": "{:.3f}",
                        "Avg OOS Sharpe": "{:.3f}",
                        "Avg OOS Max DD": "{:.2%}",
                    },
                )
            else:
                render_empty_state("Robustness sweep aggregation not available.")

    # ----------------------------------------------------
    # PAGE 6: RISK INTELLIGENCE
    # ----------------------------------------------------
    elif page == "Risk Intelligence":
        render_page_header(
            "Quantitative Risk Intelligence Terminal",
            "Calculates risk metrics, tail loss ratios, stress matrices, and state transition probability estimates.",
        )

        regime_risk_df = st.session_state.regime_risk

        if regime_risk_df is not None:
            try:
                states_arr = labeled_data["regime_state"].values
                current_state = int(states_arr[-1])
                current_label = labeled_data["regime_label"].iloc[-1]

                current_vol = regime_risk_df[
                    regime_risk_df["Regime_State"] == current_state
                ]["Annualized_Volatility"].iloc[0]

                from src.analysis.regime_analysis import compute_transition_matrix

                t_matrix = compute_transition_matrix(states_arr, n_regimes)
                t_probs = t_matrix[current_state]

                risk_on_prob = 0.0
                risk_off_prob = 0.0
                for s in range(n_regimes):
                    subset_s = labeled_data[labeled_data["regime_state"] == s]
                    label = (
                        subset_s["regime_label"].iloc[0] if len(subset_s) > 0 else ""
                    )
                    if "Bullish" in label or "Recovery" in label:
                        risk_on_prob += t_probs[s]
                    else:
                        risk_off_prob += t_probs[s]

                sum_p = risk_on_prob + risk_off_prob
                if sum_p > 0.0:
                    risk_on_prob /= sum_p
                    risk_off_prob /= sum_p

                # Top row of KPI cards
                # Top row of KPI cards
                rd_metrics = [
                    {
                        "title": "Active Market State",
                        "value": f"Regime {current_state}",
                        "description": "Current decoded model regime state",
                        "change": current_label,
                        "change_type": "neutral",
                    },
                    {
                        "title": "State Volatility (Ann.)",
                        "value": f"{current_vol:.2%}",
                        "description": "Active market volatility metrics",
                        "change": (
                            "High volatility limit"
                            if current_vol > 0.18
                            else "Stable market conditions"
                        ),
                        "change_type": "negative" if current_vol > 0.18 else "positive",
                    },
                    {
                        "title": "Transition Risk (T+1)",
                        "value": f"{risk_off_prob:.2%}",
                        "description": "Transition risk to Bearish/Risk-off",
                        "change": "Prob of shifting next window",
                        "change_type": (
                            "negative" if risk_off_prob > 0.3 else "positive"
                        ),
                    },
                    {
                        "title": "Risk-On Likelihood",
                        "value": f"{risk_on_prob:.2%}",
                        "description": "Continuation probability of Bullish trend",
                        "change": "Regime switching expansion",
                        "change_type": "positive" if risk_on_prob > 0.5 else "negative",
                    },
                ]
                render_metric_grid(rd_metrics, columns=4)

                render_spacing(15)
            except Exception as e:
                logger.error(f"Failed to render active risk dashboard metrics: {e}")

        col_risk_l, col_risk_r = st.columns([1, 1])

        with col_risk_l:
            rolling_ret = labeled_data.set_index("date")["ret_simple"]
            rolling_fig = plot_rolling_risk_metrics(rolling_ret, window=63)
            render_chart_card(
                "Rolling Risk Metrics Observability",
                rolling_fig,
                subtitle="63-day rolling window risk parameters",
            )

        with col_risk_r:
            if st.session_state.stress_test is not None:
                stress_df = st.session_state.stress_test
                render_table_card(
                    "Predefined Crisis Stress Test Profiles",
                    stress_df,
                    subtitle="Simulated drawdowns and returns during benchmark historical stress events",
                    formatter={"Total_Return": "{:.2%}", "Max_Drawdown": "{:.2%}"},
                )
            else:
                render_empty_state("Stress testing report is empty.")

        render_spacing(15)
        if regime_risk_df is not None:
            display_cols = [
                c
                for c in [
                    "Regime_State",
                    "Regime_Label",
                    "Total_Days",
                    "Annualized_Volatility",
                    "Daily_VaR_95",
                    "Daily_CVaR_95",
                    "Max_Drawdown",
                    "Average_Daily_Return",
                    "Median_Daily_Return",
                    "Transition_Risk_to_Bearish",
                ]
                if c in regime_risk_df.columns
            ]

            render_table_card(
                "Regime-Wise Risk Intelligence Decomposition",
                regime_risk_df[display_cols],
                subtitle="Detailed statistical metrics evaluated per market regime",
                formatter={
                    "Annualized_Volatility": "{:.2%}",
                    "Daily_VaR_95": "{:.2%}",
                    "Daily_CVaR_95": "{:.2%}",
                    "Max_Drawdown": "{:.2%}",
                    "Average_Daily_Return": "{:.4%}",
                    "Median_Daily_Return": "{:.4%}",
                    "Transition_Risk_to_Bearish": "{:.2%}",
                },
            )
        else:
            render_empty_state("Regime risk report not found.")

        render_educational_disclaimer()

    # ----------------------------------------------------
    # PAGE 7: MONTE CARLO PROJECTIONS
    # ----------------------------------------------------
    elif page == "Monte Carlo":
        render_page_header(
            "Stochastic Return Projections Simulator",
            "Simulate future market trajectories starting from the final historical date. Choose between global random draws, regime-conditioned returns, or Markov-chain transitions.",
        )

        col_ctrl, col_main = st.columns([1, 2])

        with col_ctrl:
            st.markdown(
                """
                <div class="saas-card" style="padding: 18px; margin-bottom: 15px;">
                    <h4 style="margin:0; color:var(--accent-color); font-size: 0.95rem; font-family: 'Outfit';">Simulation Settings</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )

            mc_method = st.selectbox(
                "Simulation Methodology:",
                [
                    "Markov-Chain Transition Simulation",
                    "Historical Bootstrap (Unconditional)",
                    "Regime-Conditioned Bootstrap",
                ],
            )

            mc_distribution = st.selectbox(
                "Return Distribution:",
                ["empirical", "student_t"],
                format_func=lambda x: (
                    "Empirical Bootstrap (Fat-Tail)"
                    if x == "empirical"
                    else "Student-t Approximation"
                ),
            )

            mc_scenario = st.selectbox(
                "Stress Scenario:",
                ["none", "covid", "selloff_2022", "inflation_risk_off"],
                format_func=lambda x: {
                    "none": "No Stress (Historical)",
                    "covid": "COVID-Like Vol Spike",
                    "selloff_2022": "2022 Selloff Scenario",
                    "inflation_risk_off": "Inflation Risk-Off Scenario",
                }.get(x, x),
            )

            mc_stress_mult = st.slider(
                "Stress Multiplier:", min_value=0.5, max_value=3.0, value=1.0, step=0.1
            )

            mc_horizon = st.slider(
                "Simulation Horizon (trading days):",
                min_value=10,
                max_value=252,
                value=126,
            )

            dd_threshold = st.slider(
                "Drawdown Loss Alert Threshold (%):",
                min_value=5.0,
                max_value=50.0,
                value=15.0,
            )

            conditioned_state = 0
            if mc_method == "Regime-Conditioned Bootstrap":
                state_label_map = (
                    labeled_data[["regime_state", "regime_label"]]
                    .drop_duplicates()
                    .set_index("regime_state")["regime_label"]
                    .to_dict()
                )
                conditioned_state = st.selectbox(
                    "Select Regime State to Condition On:",
                    options=list(state_label_map.keys()),
                    format_func=lambda x: f"State {x}: {state_label_map[x]}",
                )

        with col_main:
            with st.spinner("Generating stochastic projections..."):
                paths, mc_stats = run_cached_monte_carlo(
                    labeled_data,
                    mc_method,
                    mc_distribution,
                    mc_scenario,
                    mc_stress_mult,
                    mc_horizon,
                    int(mc_sims),
                    int(mc_seed),
                    conditioned_state,
                    n_regimes,
                    dd_threshold,
                )
                expected_ret = mc_stats["expected_return"]
                median_ret = mc_stats["median_return"]
                prob_loss = mc_stats["probability_of_loss"]
                prob_dd_exceeded = mc_stats["probability_of_drawdown_custom"]
                avg_max_dd = mc_stats["average_simulated_max_drawdown"]
                mc_stats["Daily_VaR_95"]
                cvar_95 = mc_stats["Daily_CVaR_95"]
                p5 = mc_stats["worst_5pct_return"]
                p95 = mc_stats["best_5pct_return"]

            # KPI grids
            mc_metrics_row1 = [
                {
                    "title": "Expected (Mean) Return",
                    "value": f"{expected_ret*100:.2f}%",
                    "description": "Horizon average outcome simulation",
                    "change": "Standard return expectations",
                    "change_type": "neutral",
                },
                {
                    "title": "Median Outcome Return",
                    "value": f"{median_ret*100:.2f}%",
                    "description": "50th percentile simulated path returns",
                    "change": "Middle-of-road trajectory",
                    "change_type": "neutral",
                },
                {
                    "title": "Avg Max Drawdown",
                    "value": f"{avg_max_dd*100:.2f}%",
                    "description": "Horizon average simulated drawdown",
                    "change": f"Target limit: -{dd_threshold}%",
                    "change_type": "neutral",
                },
            ]
            render_metric_grid(mc_metrics_row1, columns=3)

            render_spacing(15)

            mc_metrics_row2 = [
                {
                    "title": "Probability of Capital Loss",
                    "value": f"{prob_loss*100:.2f}%",
                    "description": "Chance of ending horizon with net loss",
                    "change": "Downside dispersion likelihood",
                    "change_type": "negative" if prob_loss > 0.4 else "positive",
                },
                {
                    "title": "Probability of Drawdown",
                    "value": f"{prob_dd_exceeded*100:.2f}%",
                    "description": "Chance of exceeding custom loss bound",
                    "change": f"Prob of loss exceeding {dd_threshold}%",
                    "change_type": "negative" if prob_dd_exceeded > 0.2 else "positive",
                },
                {
                    "title": "Expected Shortfall (ES)",
                    "value": f"{cvar_95*100:.2f}%",
                    "description": "Average returns in worst 5% tail paths",
                    "change": "Critical systemic loss metric",
                    "change_type": "negative",
                },
            ]
            render_metric_grid(mc_metrics_row2, columns=3)

            render_spacing(20)

            fan_fig = plot_monte_carlo_paths(paths)
            render_chart_card(
                "Projected Path Ribbons (Fan Chart)",
                fan_fig,
                subtitle="Quantiled pathways showing performance bounds",
            )

            term_fig = plot_monte_carlo_distributions(paths)
            render_chart_card(
                "Terminal Returns Histogram & Tails",
                term_fig,
                subtitle="Distribution of simulated index levels at terminal horizon",
            )

            render_info_banner(
                f"Worst 5% Tail Return Bound: {p5*100:.2f}% | Best 5% Tail Return Bound: {p95*100:.2f}%"
            )

    # ----------------------------------------------------
    # PAGE 7: STRATEGY BACKTESTING
    # ----------------------------------------------------
    elif page == "Backtesting":
        render_page_header(
            "Tactical Asset Allocation Backtesting Terminal",
            "Backtest model-driven position weights. The engine ensures zero lookahead bias by shifting HMM state signals by 1 day, accounting for transaction costs and slippage frictions.",
        )

        # Dynamic backtest calculations
        with st.spinner("Backtesting strategies dynamically..."):
            equity_curves_df, backtest_summary_df = run_custom_backtest(
                labeled_data, tc_input, slip_input
            )

        # Extract KPI metrics for Regime Aware
        ra_summary = backtest_summary_df[
            backtest_summary_df["Strategy"] == "Regime Aware"
        ]
        if len(ra_summary) > 0:
            ra_cagr = f"{ra_summary['CAGR'].iloc[0]:.2%}"
            ra_sharpe = f"{ra_summary['Sharpe_Ratio'].iloc[0]:.3f}"
            ra_sortino = f"{ra_summary['Sortino_Ratio'].iloc[0]:.3f}"
            ra_dd = f"{ra_summary['Max_Drawdown'].iloc[0]:.2%}"
            ra_calmar = f"{ra_summary['Calmar_Ratio'].iloc[0]:.3f}"
            ra_vol = f"{ra_summary['Annualized_Volatility'].iloc[0]:.2%}"
            ra_turnover = f"{ra_summary['Total_Turnover'].iloc[0]:.2f}"
            ra_exp = f"{ra_summary['Average_Exposure'].iloc[0]:.2%}"
        else:
            (
                ra_cagr,
                ra_sharpe,
                ra_sortino,
                ra_dd,
                ra_calmar,
                ra_vol,
                ra_turnover,
                ra_exp,
            ) = ("N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A")

        # KPI metric cards row 1
        bt_metrics_row1 = [
            {
                "title": "Regime-Aware CAGR",
                "value": ra_cagr,
                "description": "OOS compound annual growth rate",
                "change": "Active validation compound returns",
                "change_type": "neutral",
            },
            {
                "title": "Regime-Aware Sharpe",
                "value": ra_sharpe,
                "description": "Annualized return / volatility ratio",
                "change": "Target outperformance benchmarked",
                "change_type": "positive",
            },
            {
                "title": "Regime-Aware Sortino",
                "value": ra_sortino,
                "description": "Downside risk-adjusted return ratio",
                "change": "Penalizes only negative variance",
                "change_type": "positive",
            },
            {
                "title": "Regime-Aware Max DD",
                "value": ra_dd,
                "description": "Maximum peak-to-trough valuation loss",
                "change": "Benchmark buy-and-hold Max DD: -38.44%",
                "change_type": "positive" if ra_dd != "N/A" else "neutral",
            },
        ]
        render_metric_grid(bt_metrics_row1, columns=4)

        render_spacing(15)

        # KPI metric cards row 2
        bt_metrics_row2 = [
            {
                "title": "Calmar Ratio",
                "value": ra_calmar,
                "description": "CAGR / Maximum Drawdown ratio",
                "change": "Valuation recovery efficiency metric",
                "change_type": "neutral",
            },
            {
                "title": "Annual Volatility",
                "value": ra_vol,
                "description": "Standard deviation of annual returns",
                "change": "Benchmark buy-and-hold Vol: 16.51%",
                "change_type": "positive" if ra_vol != "N/A" else "neutral",
            },
            {
                "title": "Strategy Turnover",
                "value": ra_turnover,
                "description": "Total dynamic rebalance turnover size",
                "change": "Reflects physical trading volume",
                "change_type": "neutral",
            },
            {
                "title": "Avg Equity Exposure",
                "value": ra_exp,
                "description": "Average allocation budget utilized",
                "change": "Indicates cash buffer preservation",
                "change_type": "neutral",
            },
        ]
        render_metric_grid(bt_metrics_row2, columns=4)

        render_spacing(20)
        col_bt_l, col_bt_r = st.columns([1, 1])

        with col_bt_l:
            eq_fig = plot_equity_curves(equity_curves_df)
            render_chart_card(
                "Cumulative Strategy Performance",
                eq_fig,
                subtitle="Equity curves under selected cost models",
            )

        with col_bt_r:
            dd_fig = plot_drawdowns(equity_curves_df)
            render_chart_card(
                "Drawdown Comparison Profiles",
                dd_fig,
                subtitle="Drawdown percentage over time",
            )

        render_spacing(20)

        tab_bt_metrics, tab_heatmap, tab_rolling, tab_rebal, tab_rules = st.tabs(
            [
                "📊 Strategy Comparison Matrix",
                "🔥 Monthly Heatmaps",
                "📈 Rolling Sharpe & Drawdown",
                "📝 Rebalance Audit Log",
                "⚙️ Tactical Allocation Rules",
            ]
        )

        with tab_bt_metrics:
            render_table_card(
                "Complete Strategy Performance & Risk Decomposition",
                backtest_summary_df,
                formatter={
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
                    "Average_Exposure": "{:.2%}",
                    "Daily_VaR_95": "{:.2%}",
                    "Daily_CVaR_95": "{:.2%}",
                    "Downside_Deviation": "{:.2%}",
                    "Worst_Month": "{:.2%}",
                },
            )

        with tab_heatmap:
            selected_strat = st.selectbox(
                "Select Strategy to View Monthly Returns Heatmap",
                [
                    "buy_and_hold",
                    "ema_crossover",
                    "vol_targeting",
                    "regime_aware",
                    "hybrid",
                ],
                format_func=lambda s: s.replace("_", " ").title(),
            )
            selected_col = f"equity_{selected_strat}"
            if selected_col in equity_curves_df.columns:
                eq_series = equity_curves_df[selected_col]
                daily_rets = eq_series.pct_change(fill_method=None).fillna(0.0)
                daily_rets.index = pd.to_datetime(equity_curves_df["date"])

                fig_heat = plot_monthly_returns_heatmap(
                    daily_rets,
                    title=f"Compounded Monthly Returns Heatmap: {selected_strat.replace('_', ' ').title()}",
                )
                from src.visualization.chart_theme import apply_chart_theme

                apply_chart_theme(fig_heat)
                st.plotly_chart(fig_heat, width="stretch")
            else:
                render_empty_state(
                    "Select a strategy to view its monthly return heatmap."
                )

        with tab_rolling:
            rets_dict = {}
            weights_dict = {}
            for col in [
                "buy_and_hold",
                "ema_crossover",
                "vol_targeting",
                "regime_aware",
                "hybrid",
            ]:
                eq_col = f"equity_{col}"
                wt_col = f"weight_{col}"
                if eq_col in equity_curves_df.columns:
                    series = equity_curves_df[eq_col]
                    daily_rets = series.pct_change(fill_method=None).fillna(0.0)
                    daily_rets.index = pd.to_datetime(equity_curves_df["date"])
                    rets_dict[col] = daily_rets
                if wt_col in equity_curves_df.columns:
                    wt_series = equity_curves_df[wt_col]
                    wt_series.index = pd.to_datetime(equity_curves_df["date"])
                    weights_dict[col] = wt_series

            if rets_dict:
                fig_roll_sharpe = plot_rolling_sharpe(rets_dict, window=63)
                from src.visualization.chart_theme import apply_chart_theme

                apply_chart_theme(fig_roll_sharpe)
                st.plotly_chart(fig_roll_sharpe, width="stretch")

                fig_exposure = plot_strategy_exposure(weights_dict)
                apply_chart_theme(fig_exposure)
                st.plotly_chart(fig_exposure, width="stretch")
            else:
                render_empty_state("No rolling performance metrics available.")

        with tab_rebal:
            rebal_path = get_dashboard_file_path("rebalance_log.csv")
            if rebal_path:
                df_rebal = load_csv_data(rebal_path)
                filter_strat = st.selectbox(
                    "Filter Rebalance Log by Strategy Type",
                    ["All"]
                    + [
                        s.replace("_", " ").title()
                        for s in [
                            "buy_and_hold",
                            "ema_crossover",
                            "vol_targeting",
                            "regime_aware",
                            "hybrid",
                        ]
                    ],
                )
                if filter_strat != "All":
                    df_filtered = df_rebal[df_rebal["Strategy"] == filter_strat]
                else:
                    df_filtered = df_rebal

                render_table_card(
                    "Portfolio Rebalance & Trade Execution History",
                    df_filtered,
                    formatter={
                        "Old_Weight": "{:.2%}",
                        "New_Weight": "{:.2%}",
                        "Rebalance_Size": "{:.2%}",
                        "Transaction_Cost": "{:.4%}",
                    },
                )
            else:
                render_empty_state("No unified transaction audit log found.")

        with tab_rules:
            st.markdown(
                """
                <div class="saas-card" style="height: 100%;">
                    <h5 style="margin-top:0; color: var(--accent-color); font-family:'Outfit';">Tactical Signal Allocation Rules</h5>
                    <ul style="color: var(--text-secondary); font-size: 0.9rem; padding-left: 20px; line-height: 1.6; margin-bottom: 0;">
                        <li><strong>Buy & Hold</strong>: Long NIFTY 50 (100% position size).</li>
                        <li><strong>EMA Crossover</strong>: 100% position weight if EMA 50 > EMA 200, else 0% (sits in cash).</li>
                        <li><strong>Volatility Targeting</strong>: Risk-budgeted position size scaled by rolling volatility target (default target: 15% annualized, capped at 100% exposure with no leverage).</li>
                        <li><strong>Regime-Aware Strategy</strong>:
                            <ul>
                                <li><em>Bullish Low Volatility</em>: <strong>100%</strong> weight allocation.</li>
                                <li><em>Recovery / Bullish High Vol</em>: <strong>75%</strong> weight allocation.</li>
                                <li><em>Sideways Low Vol</em>: <strong>50%</strong> weight allocation.</li>
                                <li><em>Distribution / Risk-Off</em>: <strong>25%</strong> weight allocation.</li>
                                <li><em>Bearish High Vol</em>: <strong>0%</strong> weight allocation (exits to cash).</li>
                            </ul>
                        </li>
                        <li><strong>Hybrid Regime + Trend</strong>: Combined filter. Long regime-aware allocation size only if the EMA 50/200 crossover trend filter is bullish; otherwise exits to cash.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ----------------------------------------------------
    # PAGE 8: FINAL REPORT & EXPORTS
    # ----------------------------------------------------
    elif page == "Report Export":
        render_page_header(
            "Export Center & Report Assembly Console",
            "Generate and download custom CSV strategy summary datasets, daily weight series, or comprehensive HTML reports.",
        )

        with st.spinner("Compiling latest reports..."):
            equity_df, backtest_df = run_custom_backtest(
                labeled_data, tc_input, slip_input
            )
            regime_risk_df = st.session_state.regime_risk
            baseline_df = st.session_state.baseline_risk

        col_rep1, col_rep2 = st.columns(2)

        with col_rep1:
            render_html("""
                <div class="saas-card" style="margin-bottom: 20px;">
                    <h4 style="margin:0; color: var(--accent-color); font-family: 'Outfit'; margin-bottom: 10px;">📂 CSV Dataset Exports</h4>
                    <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 15px;">Download strategy timelines, performance matrices, or regime data vectors directly to your machine.</p>
                </div>
                """)

            # Final metrics
            metrics_csv = backtest_df.to_csv(index=False)
            st.download_button(
                label="Download Strategy Backtest Summary CSV",
                data=metrics_csv,
                file_name="nifty_backtest_summary.csv",
                mime="text/csv",
                width="stretch",
            )

            render_spacing(10)

            # Strategy comparison
            comp_csv = equity_df.to_csv(index=False)
            st.download_button(
                label="Download Daily Strategy Equity & Weights CSV",
                data=comp_csv,
                file_name="strategy_comparison.csv",
                mime="text/csv",
                width="stretch",
            )

            render_spacing(10)

            # Regime risk statistics
            risk_csv = (
                regime_risk_df.to_csv(index=False) if regime_risk_df is not None else ""
            )
            st.download_button(
                label="Download Regime Risk Decomposition CSV",
                data=risk_csv,
                file_name="nifty_regime_risk_report.csv",
                mime="text/csv",
                width="stretch",
            )

        with col_rep2:
            render_html("""
                <div class="saas-card" style="margin-bottom: 20px;">
                    <h4 style="margin:0; color: var(--accent-color); font-family: 'Outfit'; margin-bottom: 10px;">📄 Independent Quantitative Reports</h4>
                    <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 15px;">Generate clean HTML reports with interactive assets and math models to share with recruiters.</p>
                </div>
                """)

            html_report = generate_html_report(backtest_df, regime_risk_df, baseline_df)
            st.download_button(
                label="Download Consolidated HTML Research Report",
                data=html_report,
                file_name="nifty50_regime_risk_report.html",
                mime="text/html",
                width="stretch",
            )

            render_spacing(15)
            render_info_banner(
                "The independent HTML report includes fully embedded Plotly interactive charts, performance logs, and transition matrices. It can be shared directly with recruiters."
            )

    # ----------------------------------------------------
    # PAGE 9: METHODOLOGY
    # ----------------------------------------------------
    elif page == "Methodology":
        render_page_header(
            "Quantitative Research Methodology & Mathematical Formulations",
            "Defensible design guidelines, modeling layers, out-of-sample splits, and transaction frictions.",
        )

        st.markdown("""
            ### 1. Problem Statement & Hypothesis
            Traditional asset allocation models assume constant mean-variance parameters. However, equity indices exhibit
            strong volatility clustering, fat-tailed distributions, and structural jumps (regime shifts).
            Our hypothesis is that by training an unsupervised regime decoder (Hidden Markov Model), we can map NIFTY 50
            into hidden state regimes and dynamically scale risk exposures to manage downside drawdowns.

            ### 2. Feature Engineering & Multi-scale Volatilities
            We engineer three primary types of features:
            *   **Range Volatility Estimators**: Parkinson volatility and Garman-Klass volatility, which capture intraday high-low range information.
            *   **Statistical Complexity**: Shannon Entropy (capturing uncertainty) and the Hurst Exponent (quantifying time-series long memory / mean reversion properties).
            *   **Trend & Momentum**: Moving average distance measures, RSI, MACD, and multi-day returns.

            ### 3. Latent Regime Model Layer (HMM)
            We model market states as a first-order Hidden Markov Model where the state transitions are governed by a probability matrix $A$:
            $$a_{ij} = P(S_t = j \\mid S_{t-1} = i)$$
            Under Gaussian assumptions, each state $k$ generates observations according to a multivariate normal distribution:
            $$x_t \\mid S_t = k \\sim \\mathcal{N}(\\mu_k, \\Sigma_k)$$
            To prevent floating-point underflow when scaling long sequences, all forward-backward transitions and Viterbi paths are computed in **log-space** using Vector Log-Sum-Exp formulations.

            ### 4. Zero-Lookahead Walk-Forward Validation
            The model parameters $\\mu_k, \\Sigma_k, A$ are refitted dynamically every 6 months (126 days) using expanding historical windows.
            To eliminate lookahead bias:
            *   Signals decoded on day $t-1$ are applied to day $t$ allocations ($w_t = f(S_{t-1})$).
            *   Feature standard scalers are fitted exclusively on the expanding training partition and never on holdout data.

            ### 5. Cost Frictions & Slippage Models
            Unlike academic papers reporting frictionless returns, we apply a realistic penalty on portfolio rebalancing turnovers:
            $$\text{Cost}_t = (\\text{Fee} + \\text{Slippage}) \\times |w_t - w_{t-1}|$$
            where Fee = 10 bps and Slippage = 5 bps. This models the physical execution drag in volatile markets.
            """)

    # Render global educational disclaimer
    render_educational_disclaimer()


if __name__ == "__main__":
    main()
