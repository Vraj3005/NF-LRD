#!/usr/bin/env python
import argparse
import logging
import os

import numpy as np
import pandas as pd

from src.analysis.monte_carlo import MonteCarloSimulator
from src.analysis.regime_analysis import compute_transition_matrix
from src.analysis.risk_metrics import (
    calculate_downside_deviation,
    calculate_max_drawdown,
    calculate_portfolio_risk_report,
    calculate_var_cvar,
)
from src.backtesting.backtester import VectorizedBacktester
from src.data.fetch_data import load_settings

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_backtesting")


def parse_args():
    parser = argparse.ArgumentParser(
        description="NIFTY 50 Latent Market Regime Discovery - Risk and Backtesting Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration settings YAML file.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=5000,
        help="Number of Monte Carlo paths to simulate.",
    )
    return parser.parse_args()


def generate_regime_wise_risk_report(
    df: pd.DataFrame, n_components: int
) -> pd.DataFrame:
    """Calculates detailed return, volatility, VaR, CVaR and drawdown metrics per regime state."""
    logger.info("Generating regime-wise risk intelligence breakdown...")

    # Calculate ret_simple directly from raw_close to prevent using scaled returns features
    if "raw_close" in df.columns:
        df["ret_simple"] = df["raw_close"].pct_change(fill_method=None)
    elif "close" in df.columns:
        df["ret_simple"] = df["close"].pct_change(fill_method=None)
    df["ret_simple"] = df["ret_simple"].fillna(0.0)

    states = df["regime_state"].values
    trans_matrix = compute_transition_matrix(states, n_components)

    # Let's check which states are high-volatility/bearish states to measure transition risk
    # Transition risk = probability of transitioning from this state to Bearish High Vol or Distribution state next
    # Identify high vol/bearish indices from the labels
    bearish_states = []
    for s in range(n_components):
        label = (
            df[df["regime_state"] == s]["regime_label"].iloc[0]
            if s in df["regime_state"].values
            else ""
        )
        if "Bearish" in label or "Distribution" in label or "Risk-Off" in label:
            bearish_states.append(s)

    regime_reports = []

    for s in range(n_components):
        subset = df[df["regime_state"] == s]
        ret = subset["ret_simple"]
        label = subset["regime_label"].iloc[0] if len(subset) > 0 else f"State {s}"

        if len(ret) == 0:
            continue

        # Standard statistics
        mean_ret = float(ret.mean())
        median_ret = float(ret.median())
        vol = float(ret.std() * np.sqrt(252.0))
        downside_vol = calculate_downside_deviation(ret)

        # Historical VaR/CVaR (95%)
        var_95, cvar_95 = calculate_var_cvar(ret, 0.95)
        # Max drawdown
        max_dd = calculate_max_drawdown(ret)

        # Probabilities
        prob_pos = float((ret > 0.0).sum() / len(ret))
        # Prob of daily crash (return < -1.5%)
        prob_crash = float((ret < -0.015).sum() / len(ret))

        # Transition risk: sum of transition probabilities to bearish states
        trans_risk = (
            float(np.sum(trans_matrix[s, bearish_states])) if bearish_states else 0.0
        )

        # Get count of days spent in state
        days_count = len(ret)

        regime_reports.append(
            {
                "Regime_State": s,
                "Regime_Label": label,
                "Total_Days": days_count,
                "Average_Daily_Return": mean_ret,
                "Median_Daily_Return": median_ret,
                "Annualized_Volatility": vol,
                "Annualized_Downside_Volatility": downside_vol,
                "Daily_VaR_95": var_95,
                "Daily_CVaR_95": cvar_95,
                "Max_Drawdown": max_dd,
                "Probability_Positive_Day": prob_pos,
                "Probability_Crash_Day_1.5pct": prob_crash,
                "Transition_Risk_to_Bearish": trans_risk,
            }
        )

    return pd.DataFrame(regime_reports)


def main():
    args = parse_args()
    logger.info("Initializing Risk Intelligence Platform...")

    settings = load_settings(args.config)
    proc_dir = settings["data"].get("processed_dir", "data/processed")
    reports_dir = os.path.join(proc_dir, "../../models/reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Load regime labeled data
    labeled_parquet_path = os.path.join(reports_dir, "regime_labeled_data.parquet")
    if not os.path.exists(labeled_parquet_path):
        logger.error(
            f"Regime-labeled data Parquet not found at: {labeled_parquet_path}. Please run run_modeling.py first."
        )
        return

    logger.info(f"Loading regime labeled data: {labeled_parquet_path}")
    labeled_df = pd.read_parquet(labeled_parquet_path)
    labeled_df["date"] = pd.to_datetime(labeled_df["date"])

    n_components = labeled_df["regime_state"].nunique()

    # Calculate ret_simple directly from raw_close to prevent using scaled returns features
    if "raw_close" in labeled_df.columns:
        labeled_df["ret_simple"] = labeled_df["raw_close"].pct_change(fill_method=None)
    elif "close" in labeled_df.columns:
        labeled_df["ret_simple"] = labeled_df["close"].pct_change(fill_method=None)
    labeled_df["ret_simple"] = labeled_df["ret_simple"].fillna(0.0)

    # --- 1. Compute Base Asset Risk Report ---
    logger.info("Calculating standard baseline risk metrics for NIFTY 50...")
    nifty_returns = labeled_df["ret_simple"]
    nifty_risk = calculate_portfolio_risk_report(nifty_returns)

    risk_report_df = pd.DataFrame([nifty_risk])
    risk_report_path = os.path.join(reports_dir, "risk_report.csv")
    risk_report_df.to_csv(risk_report_path, index=False)
    logger.info(
        f"NIFTY 50 baseline risk report saved to: {os.path.abspath(risk_report_path)}"
    )

    # --- 2. Compute Regime-Wise Risk Report ---
    regime_risk_df = generate_regime_wise_risk_report(labeled_df, n_components)
    regime_risk_path = os.path.join(reports_dir, "regime_risk_report.csv")
    regime_risk_df.to_csv(regime_risk_path, index=False)
    logger.info(
        f"Regime-wise risk intelligence breakdown saved to: {os.path.abspath(regime_risk_path)}"
    )

    # --- 3. Execute Vectorized Portfolio Backtesting ---
    # Setup backtester with default 10 bps fee and 5 bps slippage
    backtester = VectorizedBacktester(transaction_cost_bps=10.0, slippage_bps=5.0)

    equity_curves_df, backtest_summary_df = backtester.backtest_strategies(
        labeled_df, args.config
    )

    # Save backtest summaries
    summary_path = os.path.join(reports_dir, "backtest_summary.csv")
    backtest_summary_df.to_csv(summary_path, index=False)
    logger.info(
        f"Portfolio strategies backtesting summary saved to: {os.path.abspath(summary_path)}"
    )

    # Save equity curves
    curves_parquet_path = os.path.join(reports_dir, "strategy_equity_curves.parquet")
    equity_curves_df.to_parquet(curves_parquet_path, index=False)
    logger.info(
        f"Daily portfolio equity curves saved to: {os.path.abspath(curves_parquet_path)}"
    )

    # --- 4. Execute Historical Stress Testing ---
    stress_report_df = backtester.run_stress_testing(labeled_df, equity_curves_df)
    stress_report_path = os.path.join(reports_dir, "stress_test_report.csv")
    stress_report_df.to_csv(stress_report_path, index=False)
    logger.info(
        f"Historical stress test comparison saved to: {os.path.abspath(stress_report_path)}"
    )

    # --- 5. Execute Markov Transition Monte Carlo Simulation ---
    logger.info("Initializing Monte Carlo simulation engine...")
    simulator = MonteCarloSimulator(labeled_df, random_seed=42)

    # Retrieve model transitions matrix
    trans_matrix_path = os.path.join(reports_dir, "transition_matrix.csv")
    if os.path.exists(trans_matrix_path):
        # Read transition matrix and clean index/headers
        trans_mat_df = pd.read_csv(trans_matrix_path, index_col=0)
        transition_matrix = trans_mat_df.values
    else:
        # Recompute transition matrix on-the-fly
        states = labeled_df["regime_state"].values
        transition_matrix = compute_transition_matrix(states, n_components)

    # Start state is the final observed historical regime state in dataset
    start_state = int(labeled_df["regime_state"].iloc[-1])

    mc_summaries = []

    # Simulate across horizons: 1 month (21d), 3 months (63d), 6 months (126d), 1 year (252d)
    horizons = {"1m": 21, "3m": 63, "6m": 126, "1y": 252}

    for label, steps in horizons.items():
        logger.info(
            f"Simulating {args.simulations} paths over {label} horizon ({steps} days)..."
        )
        # Run Markov-chain transition Monte Carlo paths
        paths = simulator.simulate_markov_chain(
            n_sims=args.simulations,
            horizon=steps,
            start_state=start_state,
            transition_matrix=transition_matrix,
        )

        # Analyze simulation paths
        path_metrics = simulator.analyze_simulation_paths(paths)
        path_metrics["horizon_label"] = label
        mc_summaries.append(path_metrics)

    mc_summary_df = pd.DataFrame(mc_summaries)

    # Re-order columns for readability
    mc_cols = [
        "horizon_label",
        "horizon_days",
        "paths_simulated",
        "expected_return",
        "median_return",
        "worst_5pct_return",
        "best_5pct_return",
        "probability_of_loss",
        "probability_of_drawdown_10pct",
        "probability_of_drawdown_20pct",
        "average_simulated_max_drawdown",
    ]
    mc_summary_df = mc_summary_df[mc_cols]

    mc_summary_path = os.path.join(reports_dir, "monte_carlo_summary.csv")
    mc_summary_df.to_csv(mc_summary_path, index=False)
    logger.info(
        f"Markov-chain Monte Carlo path projections saved to: {os.path.abspath(mc_summary_path)}"
    )
    logger.info("Risk and Backtesting modeling run completed successfully!")


if __name__ == "__main__":
    main()
