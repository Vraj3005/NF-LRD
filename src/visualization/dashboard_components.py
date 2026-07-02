"""
UI components and state management functions for Streamlit application.
Provides data loaders, custom simulation executors, dynamic backtest runners,
and stylized HTML rendering helpers.
"""

import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Add the project root to sys.path to ensure src imports resolve correctly
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analysis.monte_carlo import MonteCarloSimulator
from src.analysis.regime_analysis import compute_transition_matrix
from src.backtesting.backtester import VectorizedBacktester
from src.data.fetch_data import load_settings

logger = logging.getLogger("dashboard_components")


@st.cache_data
def load_all_dashboard_data(
    config_path: str = "config/settings.yaml",
) -> Dict[str, Any]:
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
        "model_comparison": os.path.join(reports_dir, "model_comparison.csv"),
    }

    # Load each file, default to demo_data/ if missing, or None if both are missing
    is_demo = False
    for key, path in paths.items():
        load_path = path
        if not os.path.exists(load_path):
            filename = os.path.basename(path)
            demo_path = os.path.join("demo_data", filename)
            if os.path.exists(demo_path):
                load_path = demo_path
                is_demo = True
            else:
                data[key] = None
                logger.warning(
                    f"Dashboard data file not found in both primary and demo paths: {path}"
                )
                continue

        try:
            if load_path.endswith(".parquet"):
                df = pd.read_parquet(load_path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                data[key] = df
            elif load_path.endswith(".csv"):
                df = pd.read_csv(load_path)
                data[key] = df
            elif load_path.endswith(".json"):
                with open(load_path, "r", encoding="utf-8") as f:
                    data[key] = json.load(f)
        except Exception as e:
            logger.error(f"Error reading file {load_path}: {e}")
            data[key] = None

    data["is_demo"] = is_demo
    return data


def run_pipeline_step(script_name: str, args: List[str] = []) -> Tuple[bool, str]:
    """Runs a pipeline step python script in a subprocess and returns status and logs."""
    # Find Python executable in local virtualenv (.venv) or default to sys.executable
    venv_python = os.path.join(".venv", "Scripts", "python.exe")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    cmd = [python_exe, script_name] + args
    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return True, result.stdout
        else:
            return (
                False,
                f"Error: Exit Code {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
            )
    except Exception as e:
        return False, f"Exception occurred while running {script_name}: {str(e)}"


def run_full_pipeline_from_dashboard() -> Tuple[bool, str]:
    """Runs the full ingestion, engineering, modeling and backtesting pipeline."""
    steps = [
        ("run_pipeline.py", []),
        ("run_modeling.py", []),
        ("run_backtesting.py", []),
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


@st.cache_data
def run_custom_backtest(
    df: pd.DataFrame,
    tc_bps: float,
    slippage_bps: float,
    config_path: str = "config/settings.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs a customized backtest dynamically on the provided dataframe."""
    backtester = VectorizedBacktester(
        transaction_cost_bps=tc_bps, slippage_bps=slippage_bps
    )
    equity_curves, summary = backtester.backtest_strategies(df, config_path)
    return equity_curves, summary


def run_custom_monte_carlo(
    df: pd.DataFrame, n_sims: int, horizon_days: int, seed: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Runs a customized regime-aware Markov chain transition Monte Carlo simulation on-the-fly."""
    simulator = MonteCarloSimulator(df, random_seed=seed)

    n_components = int(df["regime_state"].nunique())
    states = df["regime_state"].values

    # Compute transition matrix
    transition_matrix = compute_transition_matrix(states, n_components)
    start_state = int(states[-1])

    # Run simulation
    paths = simulator.simulate_markov_chain(
        n_sims=n_sims,
        horizon=horizon_days,
        start_state=start_state,
        transition_matrix=transition_matrix,
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
        unsafe_allow_html=True,
    )


def render_educational_disclaimer():
    """Renders a prominent research/educational use disclaimer at the bottom of the page."""
    pass


def render_status_pill(label: str, status_type: str = "success"):
    """
    Renders a premium diagnostics status pill.
    status_type: success (green), info (blue), warning (orange), danger/risk (red), inactive (gray)
    """
    colors = {
        "success": {
            "bg": "rgba(16, 185, 129, 0.12)",
            "border": "#10b981",
            "text": "#10b981",
        },
        "info": {
            "bg": "rgba(59, 130, 246, 0.12)",
            "border": "#3b82f6",
            "text": "#3b82f6",
        },
        "warning": {
            "bg": "rgba(245, 158, 11, 0.12)",
            "border": "#f59e0b",
            "text": "#f59e0b",
        },
        "danger": {
            "bg": "rgba(244, 63, 94, 0.12)",
            "border": "#f43f5e",
            "text": "#f43f5e",
        },
        "risk": {
            "bg": "rgba(244, 63, 94, 0.12)",
            "border": "#f43f5e",
            "text": "#f43f5e",
        },
        "inactive": {
            "bg": "rgba(148, 163, 184, 0.12)",
            "border": "#94a3b8",
            "text": "#94a3b8",
        },
    }
    c = colors.get(status_type, colors["inactive"])
    st.markdown(
        f"""
        <div style="
            display: inline-block;
            padding: 4px 12px;
            font-size: 0.78rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background-color: {c['bg']};
            border: 1px solid {c['border']};
            color: {c['text']};
            border-radius: 20px;
            margin: 2px 4px;
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str, value: str, change: str = None, change_type: str = "neutral"
):
    """
    Renders a premium glassmorphic KPI metric card.
    """
    change_colors = {"positive": "#10b981", "negative": "#f43f5e", "neutral": "#94a3b8"}
    color = change_colors.get(change_type, change_colors["neutral"])

    change_html = (
        f'<div style="color: {color}; font-size: 0.8rem; font-weight: 600; margin-top: 4px;">{change}</div>'
        if change
        else ""
    )

    st.markdown(
        f"""
        <div class="glass-card" style="padding: 20px; border-radius: 14px; margin-bottom: 15px;">
            <div style="color: var(--text-secondary); font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">{label}</div>
            <div style="color: var(--text-primary); font-size: 1.85rem; font-weight: 700; font-family: 'Outfit', sans-serif; letter-spacing: -0.5px;">{value}</div>
            {change_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_glass_panel(content: str, border_color: str = "rgba(255,255,255,0.06)"):
    """
    Wraps arbitrary HTML content inside a premium glassmorphic panel.
    """
    st.markdown(
        f"""
        <div class="glass-card" style="border-color: {border_color};">
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str = None):
    """
    Renders a stylized monospace section header.
    """
    subtitle_html = (
        f'<div style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 2px;">{subtitle}</div>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div style="margin-top: 25px; margin-bottom: 15px; border-left: 3px solid var(--accent-color); padding-left: 12px;">
            <div style="color: var(--text-primary); font-size: 1.15rem; font-weight: 700; font-family: 'Outfit', sans-serif; letter-spacing: -0.2px;">{title}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_badge(model_name: str):
    """Renders a technical model badge."""
    st.markdown(
        f"""
        <div style="
            display: inline-block;
            padding: 2px 8px;
            font-size: 0.72rem;
            font-weight: 600;
            font-family: monospace;
            background-color: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            color: #60a5fa;
            border-radius: 4px;
        ">
            {model_name}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_badge(risk_label: str):
    """Renders a color-coded risk category badge."""
    colors = {
        "low": {
            "border": "rgba(16, 185, 129, 0.4)",
            "color": "#10b981",
            "bg": "rgba(16, 185, 129, 0.08)",
        },
        "medium": {
            "border": "rgba(245, 158, 11, 0.4)",
            "color": "#f59e0b",
            "bg": "rgba(245, 158, 11, 0.08)",
        },
        "high": {
            "border": "rgba(244, 63, 94, 0.4)",
            "color": "#f43f5e",
            "bg": "rgba(244, 63, 94, 0.08)",
        },
    }

    risk_lower = risk_label.lower()
    r = colors["low"]
    if "high" in risk_lower or "bearish" in risk_lower or "distribution" in risk_lower:
        r = colors["high"]
    elif "medium" in risk_lower or "sideways" in risk_lower or "recovery" in risk_lower:
        r = colors["medium"]

    st.markdown(
        f"""
        <div style="
            display: inline-block;
            padding: 3px 10px;
            font-size: 0.75rem;
            font-weight: 600;
            background-color: {r['bg']};
            border: 1px solid {r['border']};
            color: {r['color']};
            border-radius: 6px;
        ">
            {risk_label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str = None):
    """Renders a clean page header."""
    subtitle_html = (
        f'<p style="color: var(--text-secondary); font-size: 1rem; margin-top: 4px; margin-bottom: 0;">{subtitle}</p>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div style="margin-bottom: 25px;">
            <h2 style="color: var(--text-primary); font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.8rem; margin: 0; letter-spacing: -0.5px;">{title}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(message: str = "No data loaded yet."):
    """Renders a clean glassmorphic empty state."""
    st.markdown(
        f"""
        <div class="glass-card" style="padding: 40px 20px; text-align: center; border-style: dashed; border-color: var(--border-color);">
            <div style="font-size: 2.2rem; margin-bottom: 12px;">📊</div>
            <div style="color: var(--text-secondary); font-size: 0.95rem; font-family: 'Inter', sans-serif;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error_state(message: str):
    """Renders a stylized error card matching the console theme."""
    st.markdown(
        f"""
        <div style="
            background-color: rgba(244, 63, 94, 0.06);
            border: 1px solid rgba(244, 63, 94, 0.25);
            border-left: 4px solid #f43f5e;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        ">
            <div style="color: #fca3b7; font-size: 0.9rem; font-family: 'Inter', sans-serif; font-weight: 500;">
                <span style="font-weight: 700; margin-right: 5px;">⚠️ ERROR:</span> {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_mode_banner():
    """Renders a distinct recruiter banner indicating the platform is operating on demo data."""
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, rgba(59, 130, 246, 0.12) 0%, rgba(99, 102, 241, 0.12) 100%);
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 10px;
            padding: 12px 20px;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.05);
        ">
            <span style="font-size: 1.2rem; margin-right: 12px;">💡</span>
            <div style="color: var(--text-primary); font-size: 0.88rem; font-family: 'Inter', sans-serif; line-height: 1.4;">
                <strong style="color: var(--accent-color);">Recruiter Demo Mode Active:</strong> Ingestion and fitting tasks are skipped. The dashboard is populated with cached walk-forward backtest results and historical data curves for immediate exploration.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
