import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from src.backtesting.strategy_rules import generate_strategy_weights
from src.analysis.risk_metrics import calculate_portfolio_risk_report

logger = logging.getLogger(__name__)

class VectorizedBacktester:
    """
    Vectorized backtesting engine.
    Simulates portfolio returns for asset strategies, accounting for position size scaling,
    transaction fees, execution slippage, and portfolio turnover.
    """
    def __init__(
        self,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0
    ):
        """
        Args:
            transaction_cost_bps: Fee rate in basis points (1 bp = 0.0001).
            slippage_bps: Trade execution slippage in basis points.
        """
        self.fee_rate = transaction_cost_bps / 10000.0
        self.slip_rate = slippage_bps / 10000.0
        self.total_cost_rate = self.fee_rate + self.slip_rate

    def backtest_strategies(
        self, 
        df: pd.DataFrame, 
        config_path: str = "config/settings.yaml"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs backtests for Buy & Hold, EMA Crossover, Regime-Aware, and Hybrid strategies.

        Args:
            df: pd.DataFrame containing daily asset close prices, returns, and HMM regime labels.
            config_path: Settings config path.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (equity_curves_df, summary_df)
        """
        logger.info("Starting vectorized portfolio backtesting...")
        
        # 1. Calculate daily asset returns
        if 'ret_simple' not in df.columns:
            if 'raw_close' in df.columns:
                asset_ret = df['raw_close'].pct_change(fill_method=None).fillna(0.0)
            else:
                asset_ret = df['close'].pct_change(fill_method=None).fillna(0.0)
        else:
            asset_ret = df['ret_simple'].fillna(0.0)
            
        # Get target weights
        weights_df = generate_strategy_weights(df)
        
        equity_curves = pd.DataFrame(index=df.index)
        equity_curves['date'] = df['date']
        
        # We will save returns and weights to calculate metrics
        strategy_returns = {}
        strategy_metrics = []
        
        strategies = ["buy_and_hold", "ema_crossover", "regime_aware", "hybrid"]
        
        for strat in strategies:
            weight_col = f"weight_{strat}"
            weights = weights_df[weight_col].values
            
            # Daily change in position size to measure trade turnover
            weight_diff = np.abs(np.diff(weights, prepend=0.0))
            
            # Daily cost: cost rate * weight change
            costs = weight_diff * self.total_cost_rate
            
            # Modeled risk-free yield on unallocated cash (5.5% annualized)
            cash_rate_daily = 0.055 / 252.0
            # Modeled cost of borrowing for leverage (6.5% annualized)
            borrow_rate_daily = 0.065 / 252.0
            
            # Cash and borrowing fractions
            cash_fraction = np.maximum(0.0, 1.0 - np.abs(weights))
            borrow_fraction = np.maximum(0.0, np.abs(weights) - 1.0)
            
            cash_return = cash_fraction * cash_rate_daily
            borrow_cost = borrow_fraction * borrow_rate_daily
            
            # Daily strategy returns: weight * asset return + cash yield - borrow cost - cost
            strat_ret = (weights * asset_ret.values) + cash_return - borrow_cost - costs
            
            # Save returns series
            ret_series = pd.Series(strat_ret, index=df.index)
            strategy_returns[strat] = ret_series
            
            # Cumulative returns starting at 1.0
            equity_curves[f'equity_{strat}'] = (1.0 + ret_series).cumprod()
            equity_curves[f'weight_{strat}'] = weights
            
            # Calculate standard performance metrics
            risk_report = calculate_portfolio_risk_report(ret_series)
            
            # Calculate extra backtest-specific statistics:
            # Total turnover: sum of absolute weight changes
            total_turnover = float(np.sum(weight_diff))
            # Exposure days fraction: pct of days where position size is non-zero
            exposure_fraction = float((weights != 0.0).sum() / len(weights))
            
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
                "Total_Turnover": total_turnover,
                "Exposure_Days_Pct": exposure_fraction
            }
            strategy_metrics.append(metrics)
            
        summary_df = pd.DataFrame(strategy_metrics)
        return equity_curves, summary_df

    def run_stress_testing(
        self,
        df: pd.DataFrame,
        equity_curves: pd.DataFrame,
        custom_windows: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Runs stress tests across predefined crisis windows and custom date segments.
        Computes total return and max drawdown for all strategies in those windows.
        """
        logger.info("Initializing historical stress testing...")
        
        # Predefined stress periods
        stress_periods = [
            {
                "name": "COVID Crash (Spring 2020)",
                "start": "2020-02-01",
                "end": "2020-05-31"
            },
            {
                "name": "2022 Volatility & Growth Selloff",
                "start": "2022-01-01",
                "end": "2022-06-30"
            }
        ]
        
        if custom_windows is not None:
            stress_periods.extend(custom_windows)
            
        stress_results = []
        strategies = ["buy_and_hold", "ema_crossover", "regime_aware", "hybrid"]
        
        # Align index on Date
        df_align = df.copy()
        df_align['date'] = pd.to_datetime(df_align['date'])
        
        for period in stress_periods:
            p_name = period["name"]
            p_start = pd.to_datetime(period["start"])
            p_end = pd.to_datetime(period["end"])
            
            # Locate indices in date range
            mask = (df_align['date'] >= p_start) & (df_align['date'] <= p_end)
            period_data = df_align[mask]
            
            if period_data.empty:
                logger.warning(f"No historical data found inside stress period: {p_name} ({p_start} to {p_end}). Skipping.")
                continue
                
            logger.info(f"Evaluating stress window: {p_name} from {p_start.strftime('%Y-%m-%d')} to {p_end.strftime('%Y-%m-%d')}...")
            
            # Get asset returns during the window
            if 'ret_simple' not in period_data.columns:
                if 'raw_close' in period_data.columns:
                    asset_ret = period_data['raw_close'].pct_change(fill_method=None).fillna(0.0)
                else:
                    asset_ret = period_data['close'].pct_change(fill_method=None).fillna(0.0)
            else:
                asset_ret = period_data['ret_simple'].fillna(0.0)
                
            weights_df = generate_strategy_weights(df)
            
            for strat in strategies:
                weight_col = f"weight_{strat}"
                # Get weights matching this period
                weights = weights_df[weight_col].values[mask]
                
                # Calculate daily change in position size to measure trade turnover
                weight_diff = np.abs(np.diff(weights, prepend=0.0))
                costs = weight_diff * self.total_cost_rate
                
                # Modeled risk-free yield on unallocated cash (5.5% annualized)
                cash_rate_daily = 0.055 / 252.0
                # Modeled cost of borrowing for leverage (6.5% annualized)
                borrow_rate_daily = 0.065 / 252.0
                
                # Cash and borrowing fractions
                cash_fraction = np.maximum(0.0, 1.0 - np.abs(weights))
                borrow_fraction = np.maximum(0.0, np.abs(weights) - 1.0)
                
                cash_return = cash_fraction * cash_rate_daily
                borrow_cost = borrow_fraction * borrow_rate_daily
                
                # Daily strategy returns: weight * asset return + cash yield - borrow cost - cost
                strat_ret = (weights * asset_ret.values) + cash_return - borrow_cost - costs
                ret_series = pd.Series(strat_ret, index=period_data.index)
                
                # Metrics for this period
                total_return = float((1.0 + ret_series).prod() - 1.0)
                
                # Max Drawdown
                cum_ret = (1.0 + ret_series).cumprod()
                peak_ret = cum_ret.cummax()
                drawdowns = (cum_ret - peak_ret) / (peak_ret + 1e-15)
                max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0
                
                stress_results.append({
                    "Stress_Period": p_name,
                    "Start_Date": p_start.strftime('%Y-%m-%d'),
                    "End_Date": p_end.strftime('%Y-%m-%d'),
                    "Strategy": strat.replace("_", " ").title(),
                    "Total_Return": total_return,
                    "Max_Drawdown": max_dd
                })
                
        return pd.DataFrame(stress_results)
