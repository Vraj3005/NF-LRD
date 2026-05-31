"""
UI components and state management functions for Streamlit application.
Provides data loaders, custom simulation executors, dynamic backtest runners,
and stylized HTML rendering helpers.
"""

import os
import sys
import subprocess
import json
import logging
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Tuple, Optional, List

# Add the project root to sys.path to ensure src imports resolve correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.fetch_data import load_settings
from src.backtesting.backtester import VectorizedBacktester
from src.analysis.monte_carlo import MonteCarloSimulator
from src.analysis.regime_analysis import compute_transition_matrix

logger = logging.getLogger("dashboard_components")

@st.cache_data
def load_all_dashboard_data(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Cached function to load all report files, parquet datasets, and configuration metadata.
    Returns a dictionary of dataframes and json models, or empty structures if files are missing.
    """
    data = {}
    try:
        settings = load_settings(config_path)
    except Exception as e:
        logger.error(f"Failed to load settings from {config_path}: {e}")
        settings = {}
        
    proc_dir = settings.get("data", {}).get("processed_dir", "data/processed")
    reports_dir = os.path.join(proc_dir, "../../models/reports")
    
    # Files to load
    paths = {
        "labeled_data": os.path.join(reports_dir, "regime_labeled_data.parquet"),
        "model_report": os.path.join(reports_dir, "selected_model_report.json"),
        "transition_matrix": os.path.join(reports_dir, "transition_matrix.csv"),
        "regime_risk": os.path.join(reports_dir, "regime_risk_report.csv"),
        "baseline_risk": os.path.join(reports_dir, "risk_report.csv"),
        "backtest_summary": os.path.join(reports_dir, "backtest_summary.csv"),
        "equity_curves": os.path.join(reports_dir, "strategy_equity_curves.parquet"),
        "stress_test": os.path.join(reports_dir, "stress_test_report.csv"),
        "monte_carlo_summary": os.path.join(reports_dir, "monte_carlo_summary.csv"),
        "model_comparison": os.path.join(reports_dir, "model_comparison.csv")
    }
    
    # Load each file, default to None or empty if missing
    for key, path in paths.items():
        if not os.path.exists(path):
            data[key] = None
            logger.warning(f"Dashboard data file not found: {path}")
            continue
            
        try:
            if path.endswith(".parquet"):
                df = pd.read_parquet(path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                data[key] = df
            elif path.endswith(".csv"):
                df = pd.read_csv(path)
                if key == "transition_matrix":
                    # Keep index
                    pass
                data[key] = df
            elif path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data[key] = json.load(f)
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            data[key] = None
            
    return data

def run_pipeline_step(script_name: str, args: List[str] = []) -> Tuple[bool, str]:
    """Runs a pipeline step python script in a subprocess and returns status and logs."""
    # Find Python executable in local virtualenv (.venv) or default to sys.executable
    venv_python = os.path.join(".venv", "Scripts", "python.exe")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable
    
    cmd = [python_exe, script_name] + args
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, f"Error: Exit Code {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return False, f"Exception occurred while running {script_name}: {str(e)}"

def run_full_pipeline_from_dashboard() -> Tuple[bool, str]:
    """Runs the full ingestion, engineering, modeling and backtesting pipeline."""
    steps = [
        ("run_pipeline.py", []),
        ("run_modeling.py", []),
        ("run_backtesting.py", [])
    ]
    
    full_log = []
    for script, args in steps:
        full_log.append(f"=== Starting Step: {script} ===")
        success, logs = run_pipeline_step(script, args)
        full_log.append(logs)
        if not success:
            full_log.append(f"\nPipeline failed during step: {script}")
            return False, "\n".join(full_log)
        full_log.append(f"=== Completed Step: {script} successfully ===\n")
        
    return True, "\n".join(full_log)

def run_custom_backtest(
    df: pd.DataFrame, 
    tc_bps: float, 
    slippage_bps: float,
    config_path: str = "config/settings.yaml"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs a customized backtest dynamically on the provided dataframe."""
    backtester = VectorizedBacktester(transaction_cost_bps=tc_bps, slippage_bps=slippage_bps)
    equity_curves, summary = backtester.backtest_strategies(df, config_path)
    return equity_curves, summary

def run_custom_monte_carlo(
    df: pd.DataFrame,
    n_sims: int,
    horizon_days: int,
    seed: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Runs a customized regime-aware Markov chain transition Monte Carlo simulation on-the-fly."""
    simulator = MonteCarloSimulator(df, random_seed=seed)
    
    n_components = int(df['regime_state'].nunique())
    states = df['regime_state'].values
    
    # Compute transition matrix
    transition_matrix = compute_transition_matrix(states, n_components)
    start_state = int(states[-1])
    
    # Run simulation
    paths = simulator.simulate_markov_chain(
        n_sims=n_sims,
        horizon=horizon_days,
        start_state=start_state,
        transition_matrix=transition_matrix
    )
    
    # Run analysis
    summary = simulator.analyze_simulation_paths(paths)
    return paths, summary

def render_custom_header():
    """Renders a premium, dark-themed glowing header for the dashboard."""
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@700;800&display=swap" rel="stylesheet">
        <style>
            .header-container {
                background: linear-gradient(135deg, #090d16 0%, #111827 100%);
                padding: 30px;
                border-radius: 15px;
                border: 1px solid #1f2937;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.25);
                margin-bottom: 25px;
            }
            .header-title {
                color: transparent;
                background-image: linear-gradient(to right, #60a5fa, #3b82f6, #1d4ed8);
                -webkit-background-clip: text;
                background-clip: text;
                margin: 0;
                font-family: 'Outfit', sans-serif;
                font-weight: 800;
                font-size: 2.5rem;
                letter-spacing: -0.5px;
            }
            .header-subtitle {
                color: #9ca3af;
                margin: 8px 0 0 0;
                font-family: 'Inter', sans-serif;
                font-size: 1.05rem;
                font-weight: 400;
                letter-spacing: 0.2px;
            }
            .metric-card {
                background: #1f2937;
                border: 1px solid #374151;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            }
            .metric-label {
                color: #9ca3af;
                font-size: 0.85rem;
                font-weight: 600;
                text-transform: uppercase;
                margin-bottom: 5px;
            }
            .metric-value {
                color: #f3f4f6;
                font-size: 1.6rem;
                font-weight: 700;
            }
        </style>
        <div class="header-container">
            <h1 class="header-title">NIFTY 50 Latent Market Regime Discovery & Risk Intelligence</h1>
            <p class="header-subtitle">
                Advanced Unsupervised Machine Learning (HMM), Regime-conditioned Risk Analysis, and Dynamic Tactical Asset Allocation Backtesting
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_educational_disclaimer():
    """Renders a prominent research/educational use disclaimer at the bottom of the page."""
    st.markdown("---")
    st.markdown(
        """
        <div style="background-color: rgba(239, 68, 68, 0.08); border-left: 4px solid #ef4444; padding: 15px; border-radius: 4px; margin-top: 20px;">
            <h4 style="color: #f87171; margin-top: 0; margin-bottom: 5px; font-family: 'Inter', sans-serif; font-weight: 600;">Academic & Research Disclaimer</h4>
            <p style="color: #fca5a5; margin: 0; font-size: 0.9rem; font-family: 'Inter', sans-serif; line-height: 1.4;">
                This platform is built strictly for educational, academic, and portfolio demonstration purposes. All strategies, model predictions, backtest results, and simulated outcomes are simulated and do not constitute professional financial advice. Live trading involves substantial risk, and past performance is never a guarantee of future results. Always verify quantitative models and consult with registered financial advisors before allocating real capital.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
