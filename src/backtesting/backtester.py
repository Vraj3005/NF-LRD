"""
Vectorized Backtesting Engine for Latent Market Regime Platform.
Simulates portfolio returns for asset strategies, accounting for position size scaling,
transaction fees, execution slippage, rebalance logs, and performance metrics.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis.risk_metrics import (
    calculate_portfolio_risk_report,
)
from src.backtesting.strategy_rules import generate_strategy_weights

logger = logging.getLogger(__name__)


def generate_monthly_returns_heatmap(ret_series: pd.Series) -> pd.DataFrame:
    """Calculates monthly compound returns and compiles them into a year-by-month heatmap matrix."""
    if len(ret_series) == 0:
        return pd.DataFrame()
    series = ret_series.copy()
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    try:
        monthly_ret = series.resample("ME").apply(lambda r: (1.0 + r).prod() - 1.0)
    except ValueError:
        monthly_ret = series.resample("M").apply(lambda r: (1.0 + r).prod() - 1.0)

    years = sorted(list(set(monthly_ret.index.year)))
    df_heatmap = pd.DataFrame(index=years)
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    for m_idx, m_name in enumerate(months, 1):
        m_mask = monthly_ret.index.month == m_idx
        m_vals = monthly_ret[m_mask]
        df_heatmap[m_name] = pd.Series(m_vals.values, index=m_vals.index.year)

    df_heatmap = df_heatmap.fillna(0.0)

    try:
        ytd_ret = series.resample("YE" if hasattr(pd, "DatetimeIndex") else "A").apply(
            lambda r: (1.0 + r).prod() - 1.0
        )
    except ValueError:
        try:
            ytd_ret = series.resample("A").apply(lambda r: (1.0 + r).prod() - 1.0)
        except ValueError:
            ytd_ret = series.resample("Y").apply(lambda r: (1.0 + r).prod() - 1.0)

    df_heatmap["YTD"] = pd.Series(ytd_ret.values, index=ytd_ret.index.year).fillna(0.0)
    return df_heatmap


def generate_rebalance_log(
    date_series: pd.Series,
    weights_series: pd.Series,
    total_cost_rate: float,
    strategy_name: str,
) -> pd.DataFrame:
    """Compiles rebalance logs identifying changes in target allocations."""
    w_diff = weights_series.diff().fillna(0.0)
    rebalance_mask = w_diff != 0.0

    if len(weights_series) > 0 and weights_series.iloc[0] != 0.0:
        rebalance_mask.iloc[0] = True
        w_diff.iloc[0] = weights_series.iloc[0]

    dates = date_series[rebalance_mask]
    old_w = weights_series.shift(1).fillna(0.0)[rebalance_mask]
    new_w = weights_series[rebalance_mask]
    sizes = w_diff.abs()[rebalance_mask]
    costs = sizes * total_cost_rate

    # Ensure date string format
    date_strings = dates.dt.strftime("%Y-%m-%d") if hasattr(dates, "dt") else dates

    log_df = pd.DataFrame(
        {
            "Date": date_strings,
            "Strategy": strategy_name,
            "Old_Weight": old_w,
            "New_Weight": new_w,
            "Rebalance_Size": sizes,
            "Transaction_Cost": costs,
        }
    )
    return log_df.reset_index(drop=True)


def calculate_rolling_drawdown(ret_series: pd.Series, window: int = 252) -> pd.Series:
    """Calculates rolling peak-to-trough drawdown over a sliding window."""
    cum_ret = (1.0 + ret_series).cumprod()
    roll_peak = cum_ret.rolling(window=window, min_periods=1).max()
    drawdowns = (cum_ret - roll_peak) / (roll_peak + 1e-15)
    return drawdowns


def compute_rolling_sharpe(ret_series: pd.Series, window: int = 252) -> pd.Series:
    """Computes rolling Sharpe ratio over a moving window."""
    rolling_mean = ret_series.rolling(window=window).mean()
    rolling_std = ret_series.rolling(window=window).std()
    return (rolling_mean / (rolling_std + 1e-15)) * np.sqrt(252.0)


class VectorizedBacktester:
    """
    Vectorized backtesting engine.
    Simulates portfolio returns for asset strategies, accounting for position size scaling,
    transaction fees, execution slippage, and portfolio turnover.
    """

    def __init__(
        self,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        cash_yield: float = 0.0,
    ):
        """
        Args:
            transaction_cost_bps: Fee rate in basis points (1 bp = 0.0001).
            slippage_bps: Trade execution slippage in basis points.
            cash_yield: Annual interest yield assumption on cash (defaults to 0.0).
        """
        self.fee_rate = transaction_cost_bps / 10000.0
        self.slip_rate = slippage_bps / 10000.0
        self.total_cost_rate = self.fee_rate + self.slip_rate
        self.cash_yield = cash_yield

    def generate_walk_forward_regimes(
        self,
        df: pd.DataFrame,
        model_cls: Any,
        n_components: int,
        initial_train_days: int = 1260,
        refit_interval_days: int = 63,
    ) -> pd.Series:
        """
        Generates lookahead-free walk-forward decoded market regimes.
        Retrains the model periodically on expanding training sets and decodes out-of-sample.
        Handles state label switching dynamically at each step.
        """
        from src.analysis.regime_analysis import assign_regime_labels

        feature_cols = [
            c
            for c in df.columns
            if c not in ["date", "regime_label", "regime_state"]
            and not c.startswith("raw_")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        X = df[feature_cols].values
        valid_mask = ~np.isnan(X).any(axis=1)
        valid_indices = np.where(valid_mask)[0]

        X_valid = X[valid_mask]

        if "ret_simple" not in df.columns:
            if "raw_close" in df.columns:
                asset_ret = (
                    df["raw_close"].pct_change(fill_method=None).fillna(0.0).values
                )
            else:
                asset_ret = df["close"].pct_change(fill_method=None).fillna(0.0).values
        else:
            asset_ret = df["ret_simple"].fillna(0.0).values

        ret_valid = asset_ret[valid_mask]
        wf_labels = ["Distribution / Risk-Off Regime"] * len(df)

        if len(X_valid) <= initial_train_days:
            logger.warning(
                "Not enough data for walk-forward. Falling back to global fit."
            )
            model = model_cls(
                n_components=n_components, covariance_type="diag", random_state=42
            )
            model.fit(X_valid)
            states = model.predict(X_valid)

            regime_stats = []
            for i in range(n_components):
                state_mask = states == i
                state_ret = ret_valid[state_mask]
                mean_ret = state_ret.mean() if len(state_ret) > 0 else 0.0
                std_ret = state_ret.std() if len(state_ret) > 0 else 0.0
                ann_ret = mean_ret * 252.0
                ann_vol = std_ret * np.sqrt(252.0)
                sharpe = (ann_ret / ann_vol) if ann_vol > 0.0 else 0.0
                regime_stats.append(
                    {
                        "regime_state": i,
                        "annualized_return": ann_ret,
                        "annualized_volatility": ann_vol,
                        "sharpe_ratio": sharpe,
                    }
                )
            labels = assign_regime_labels(regime_stats)
            for idx, state_idx in enumerate(states):
                wf_labels[valid_indices[idx]] = labels[state_idx]
            return pd.Series(wf_labels, index=df.index)

        # Expanding window logic
        model = model_cls(
            n_components=n_components, covariance_type="diag", random_state=42
        )
        model.fit(X_valid[:initial_train_days])
        initial_states = model.predict(X_valid[:initial_train_days])

        regime_stats = []
        ret_train = ret_valid[:initial_train_days]
        for i in range(n_components):
            state_mask = initial_states == i
            state_ret = ret_train[state_mask]
            mean_ret = state_ret.mean() if len(state_ret) > 0 else 0.0
            std_ret = state_ret.std() if len(state_ret) > 0 else 0.0
            ann_ret = mean_ret * 252.0
            ann_vol = std_ret * np.sqrt(252.0)
            sharpe = (ann_ret / ann_vol) if ann_vol > 0.0 else 0.0
            regime_stats.append(
                {
                    "regime_state": i,
                    "annualized_return": ann_ret,
                    "annualized_volatility": ann_vol,
                    "sharpe_ratio": sharpe,
                }
            )
        labels = assign_regime_labels(regime_stats)
        mapping = {i: labels[i] for i in range(n_components)}

        for idx, state_idx in enumerate(initial_states):
            wf_labels[valid_indices[idx]] = mapping.get(
                state_idx, "Distribution / Risk-Off Regime"
            )

        t = initial_train_days
        while t < len(X_valid):
            oos_end = min(t + refit_interval_days, len(X_valid))
            model = model_cls(
                n_components=n_components, covariance_type="diag", random_state=42
            )
            model.fit(X_valid[:t])
            train_states = model.predict(X_valid[:t])

            regime_stats = []
            ret_train = ret_valid[:t]
            for i in range(n_components):
                state_mask = train_states == i
                state_ret = ret_train[state_mask]
                mean_ret = state_ret.mean() if len(state_ret) > 0 else 0.0
                std_ret = state_ret.std() if len(state_ret) > 0 else 0.0
                ann_ret = mean_ret * 252.0
                ann_vol = std_ret * np.sqrt(252.0)
                sharpe = (ann_ret / ann_vol) if ann_vol > 0.0 else 0.0
                regime_stats.append(
                    {
                        "regime_state": i,
                        "annualized_return": ann_ret,
                        "annualized_volatility": ann_vol,
                        "sharpe_ratio": sharpe,
                    }
                )
            labels = assign_regime_labels(regime_stats)
            mapping = {i: labels[i] for i in range(n_components)}

            oos_states = model.predict(X_valid[t:oos_end])
            for offset, state_idx in enumerate(oos_states):
                global_idx = valid_indices[t + offset]
                wf_labels[global_idx] = mapping.get(
                    state_idx, "Distribution / Risk-Off Regime"
                )

            t = oos_end

        return pd.Series(wf_labels, index=df.index)

    def backtest_strategies(
        self,
        df: pd.DataFrame,
        config_path: str = "config/settings.yaml",
        walk_forward: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs backtests for Buy & Hold, EMA Crossover, Vol Targeting, Regime-Aware, and Hybrid strategies.
        """
        logger.info("Starting vectorized portfolio backtesting...")

        if walk_forward:
            logger.info("Running Walk-Forward HMM Regime Decoding...")
            n_components = 3
            try:
                import json

                report_path = "models/reports/selected_model_report.json"
                if os.path.exists(report_path):
                    with open(report_path, "r", encoding="utf-8") as f:
                        report = json.load(f)
                    n_components = report.get("best_n_components", 3)
            except Exception:
                pass

            from src.models.hmm_model import GaussianHMM

            wf_labels = self.generate_walk_forward_regimes(
                df, model_cls=GaussianHMM, n_components=n_components
            )
            df = df.copy()
            df["regime_label"] = wf_labels

        # 1. Calculate daily asset returns (prefer raw_close first)
        if "raw_close" in df.columns:
            asset_ret = df["raw_close"].pct_change(fill_method=None).fillna(0.0)
        elif "close" in df.columns:
            asset_ret = df["close"].pct_change(fill_method=None).fillna(0.0)
        elif "ret_simple" in df.columns:
            asset_ret = df["ret_simple"].fillna(0.0)
        else:
            raise ValueError(
                "No price or return column found to calculate asset returns."
            )

        # Get target weights
        weights_df = generate_strategy_weights(df)

        equity_curves = pd.DataFrame(index=df.index)
        equity_curves["date"] = df["date"]

        strategy_returns = {}
        strategy_metrics = []
        all_rebal_logs = []

        strategies = [
            "buy_and_hold",
            "ema_crossover",
            "vol_targeting",
            "regime_aware",
            "hybrid",
        ]
        cash_rate_daily = self.cash_yield / 252.0

        for strat in strategies:
            weight_col = f"weight_{strat}"
            weights = weights_df[weight_col].values

            # Rebalance size
            weight_diff = np.abs(np.diff(weights, prepend=0.0))
            costs = weight_diff * self.total_cost_rate
            cash_fraction = np.maximum(0.0, 1.0 - np.abs(weights))

            # Daily net returns
            cash_return = cash_fraction * cash_rate_daily
            strat_ret = (weights * asset_ret.values) + cash_return - costs

            # Record returns
            ret_series = pd.Series(strat_ret, index=df.index)
            strategy_returns[strat] = ret_series

            # Cumulative returns starting at 1.0 with margin liquidation rule
            cum_ret = (1.0 + ret_series.values).cumprod()
            zero_crossings = np.where(cum_ret <= 0.0)[0]
            if len(zero_crossings) > 0:
                cum_ret[zero_crossings[0] :] = 0.0

            equity_curves[f"equity_{strat}"] = cum_ret
            equity_curves[f"weight_{strat}"] = weights

            # Calculate risk statistics
            risk_report = calculate_portfolio_risk_report(ret_series)

            # Extra statistics
            total_turnover = float(np.sum(weight_diff))
            exposure_fraction = float(np.mean(np.abs(weights)))

            metrics = {
                "Strategy": strat.replace("_", " ").title(),
                "CAGR": risk_report["CAGR"],
                "Annualized_Volatility": risk_report["Annualized_Volatility"],
                "Sharpe_Ratio": risk_report["Sharpe_Ratio"],
                "Sortino_Ratio": risk_report["Sortino_Ratio"],
                "Calmar_Ratio": risk_report["Calmar_Ratio"],
                "Max_Drawdown": risk_report["Max_Drawdown"],
                "Average_Drawdown": risk_report["Average_Drawdown"],
                "Drawdown_Duration_Days": risk_report["Drawdown_Duration_Days"],
                "Win_Rate": risk_report["Win_Rate"],
                "Profit_Factor": risk_report["Profit_Factor"],
                "Daily_VaR_95": risk_report["Daily_VaR_95"],
                "Daily_CVaR_95": risk_report["Daily_CVaR_95"],
                "Downside_Deviation": risk_report["Downside_Deviation"],
                "Worst_Month": risk_report["Worst_Month"],
                "Total_Turnover": total_turnover,
                "Average_Exposure": exposure_fraction,
            }
            strategy_metrics.append(metrics)

            # Generate monthly returns heatmap
            try:
                heatmap_df = generate_monthly_returns_heatmap(ret_series)
                heatmap_path = f"models/reports/monthly_heatmap_{strat}.csv"
                os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
                heatmap_df.to_csv(heatmap_path)
            except Exception as e:
                logger.error(f"Failed to generate monthly heatmap for {strat}: {e}")

            # Generate rebalance logs
            try:
                rebal_log = generate_rebalance_log(
                    df["date"],
                    pd.Series(weights, index=df.index),
                    self.total_cost_rate,
                    strat.replace("_", " ").title(),
                )
                all_rebal_logs.append(rebal_log)
            except Exception as e:
                logger.error(f"Failed to compile rebalance log for {strat}: {e}")

        # Save unified rebalance log
        if all_rebal_logs:
            try:
                unified_log = (
                    pd.concat(all_rebal_logs, axis=0)
                    .sort_values("Date")
                    .reset_index(drop=True)
                )
                rebal_path = "models/reports/rebalance_log.csv"
                os.makedirs(os.path.dirname(rebal_path), exist_ok=True)
                unified_log.to_csv(rebal_path, index=False)
                logger.info(
                    f"Unified audit rebalance log saved to: {os.path.abspath(rebal_path)}"
                )
            except Exception as e:
                logger.error(f"Failed to save unified rebalance log: {e}")

        # Save rolling metrics (252-day moving window)
        try:
            rolling_metrics = pd.DataFrame(index=df.index)
            rolling_metrics["date"] = df["date"]
            for strat in strategies:
                rolling_metrics[f"rolling_sharpe_{strat}"] = compute_rolling_sharpe(
                    strategy_returns[strat], window=252
                )
                rolling_metrics[f"rolling_drawdown_{strat}"] = (
                    calculate_rolling_drawdown(strategy_returns[strat], window=252)
                )

            rolling_path = "models/reports/rolling_metrics.parquet"
            rolling_metrics.to_parquet(rolling_path, index=False)
            logger.info(
                f"Rolling backtest metrics saved to: {os.path.abspath(rolling_path)}"
            )
        except Exception as e:
            logger.error(f"Failed to save rolling metrics: {e}")

        summary_df = pd.DataFrame(strategy_metrics)
        return equity_curves, summary_df

    def run_stress_testing(
        self,
        df: pd.DataFrame,
        equity_curves: pd.DataFrame,
        custom_windows: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Runs stress tests across predefined crisis windows and custom date segments.
        """
        logger.info("Initializing historical stress testing...")

        stress_periods = [
            {
                "name": "COVID Crash (Spring 2020)",
                "start": "2020-02-01",
                "end": "2020-05-31",
            },
            {
                "name": "2022 Volatility & Growth Selloff",
                "start": "2022-01-01",
                "end": "2022-06-30",
            },
        ]

        if custom_windows is not None:
            stress_periods.extend(custom_windows)

        stress_results = []
        strategies = [
            "buy_and_hold",
            "ema_crossover",
            "vol_targeting",
            "regime_aware",
            "hybrid",
        ]

        df_align = df.copy()
        df_align["date"] = pd.to_datetime(df_align["date"])

        for period in stress_periods:
            p_name = period["name"]
            p_start = pd.to_datetime(period["start"])
            p_end = pd.to_datetime(period["end"])

            mask = (df_align["date"] >= p_start) & (df_align["date"] <= p_end)
            period_data = df_align[mask]

            if period_data.empty:
                logger.warning(
                    f"No historical data found inside stress period: {p_name} ({p_start} to {p_end}). Skipping."
                )
                continue

            logger.info(
                f"Evaluating stress window: {p_name} from {p_start.strftime('%Y-%m-%d')} to {p_end.strftime('%Y-%m-%d')}..."
            )

            if "raw_close" in period_data.columns:
                asset_ret = (
                    period_data["raw_close"].pct_change(fill_method=None).fillna(0.0)
                )
            elif "close" in period_data.columns:
                asset_ret = (
                    period_data["close"].pct_change(fill_method=None).fillna(0.0)
                )
            elif "ret_simple" in period_data.columns:
                asset_ret = period_data["ret_simple"].fillna(0.0)
            else:
                raise ValueError(
                    "No price or return column found to calculate asset returns."
                )

            weights_df = generate_strategy_weights(df)

            for strat in strategies:
                weight_col = f"weight_{strat}"
                weights = weights_df[weight_col].values[mask]

                weight_diff = np.abs(np.diff(weights, prepend=0.0))
                costs = weight_diff * self.total_cost_rate
                cash_fraction = np.maximum(0.0, 1.0 - np.abs(weights))

                cash_return = cash_fraction * (self.cash_yield / 252.0)
                strat_ret = (weights * asset_ret.values) + cash_return - costs
                ret_series = pd.Series(strat_ret, index=period_data.index)

                total_return = float((1.0 + ret_series).prod() - 1.0)

                cum_ret = (1.0 + ret_series).cumprod()
                peak_ret = cum_ret.cummax()
                drawdowns = (cum_ret - peak_ret) / (peak_ret + 1e-15)
                max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

                stress_results.append(
                    {
                        "Stress_Period": p_name,
                        "Start_Date": p_start.strftime("%Y-%m-%d"),
                        "End_Date": p_end.strftime("%Y-%m-%d"),
                        "Strategy": strat.replace("_", " ").title(),
                        "Total_Return": total_return,
                        "Max_Drawdown": max_dd,
                    }
                )

        return pd.DataFrame(stress_results)
