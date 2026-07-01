import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

def calculate_cagr(returns: pd.Series) -> float:
    """Calculates the Compound Annual Growth Rate (CAGR)."""
    if len(returns) == 0:
        return 0.0
    cum_ret = (1.0 + returns).prod()
    n_years = len(returns) / 252.0
    if n_years > 0 and cum_ret > 0:
        return float((cum_ret) ** (1.0 / n_years) - 1.0)
    return 0.0

def calculate_annualized_volatility(returns: pd.Series) -> float:
    """Calculates annualized standard deviation of returns."""
    if len(returns) == 0:
        return 0.0
    return float(returns.std() * np.sqrt(252.0))

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculates the annualized Sharpe Ratio."""
    vol = calculate_annualized_volatility(returns)
    if vol > 0.0:
        # Subtract daily risk-free rate
        excess_returns = returns - (risk_free_rate / 252.0)
        ann_excess_ret = excess_returns.mean() * 252.0
        return float(ann_excess_ret / vol)
    return 0.0

def calculate_downside_deviation(returns: pd.Series, target_return: float = 0.0) -> float:
    """Calculates the downside deviation below a target return (annualized)."""
    if len(returns) == 0:
        return 0.0
    downside_diff = returns - (target_return / 252.0)
    downside_diff = downside_diff.clip(upper=0.0)
    variance = (downside_diff ** 2).mean()
    return float(np.sqrt(variance) * np.sqrt(252.0))

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculates the Sortino Ratio."""
    downside_dev = calculate_downside_deviation(returns, risk_free_rate)
    if downside_dev > 0.0:
        excess_returns = returns - (risk_free_rate / 252.0)
        ann_excess_ret = excess_returns.mean() * 252.0
        return float(ann_excess_ret / downside_dev)
    return 0.0

def calculate_drawdown_series(returns: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes cumulative returns, peak returns, and drawdown series.
    Returns: (cum_returns, peak_returns, drawdown_series)
    """
    cum_returns = (1.0 + returns).cumprod()
    peak_returns = cum_returns.cummax()
    drawdowns = (cum_returns - peak_returns) / (peak_returns + 1e-15)
    return cum_returns, peak_returns, drawdowns

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculates maximum peak-to-trough drawdown."""
    if len(returns) == 0:
        return 0.0
    _, _, drawdowns = calculate_drawdown_series(returns)
    max_dd = drawdowns.min()
    return float(max_dd) if not np.isnan(max_dd) else 0.0

def calculate_average_drawdown(returns: pd.Series) -> float:
    """Calculates the average drawdown when the asset is below peak levels."""
    if len(returns) == 0:
        return 0.0
    _, _, drawdowns = calculate_drawdown_series(returns)
    negative_drawdowns = drawdowns[drawdowns < 0.0]
    if len(negative_drawdowns) > 0:
        return float(negative_drawdowns.mean())
    return 0.0

def calculate_drawdown_duration(returns: pd.Series) -> int:
    """Calculates the maximum drawdown duration in trading days."""
    if len(returns) == 0:
        return 0
    _, _, drawdowns = calculate_drawdown_series(returns)
    
    # Identify duration of consecutive days spent in drawdown (drawdown < 0)
    is_in_dd = (drawdowns < 0.0).astype(int)
    # Group consecutive drawdown days using cumulative sum of flips
    flips = (is_in_dd == 0).cumsum()
    durations = is_in_dd.groupby(flips).sum()
    
    max_dur = durations.max()
    return int(max_dur) if not np.isnan(max_dur) else 0

def calculate_calmar_ratio(returns: pd.Series) -> float:
    """Calculates the Calmar Ratio (CAGR / Max Drawdown absolute value)."""
    cagr = calculate_cagr(returns)
    max_dd = calculate_max_drawdown(returns)
    if max_dd < 0.0:
        return float(cagr / abs(max_dd))
    return 0.0

def calculate_win_rate(returns: pd.Series) -> float:
    """Calculates percentage of trading days with positive returns."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0.0).sum() / len(returns))

def calculate_profit_factor(returns: pd.Series) -> float:
    """Calculates the profit factor (gross gains / gross losses)."""
    gains = returns[returns > 0.0].sum()
    losses = abs(returns[returns < 0.0].sum())
    if losses > 0.0:
        return float(gains / losses)
    return 999.0 if gains > 0.0 else 0.0

def calculate_var_cvar(returns: pd.Series, alpha: float = 0.95) -> Tuple[float, float]:
    """
    Computes historical Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    
    Args:
        returns: pd.Series of daily returns.
        alpha: Confidence level (e.g. 0.95 or 0.99).
        
    Returns:
        Tuple[float, float]: (VaR, CVaR) as daily fraction values.
    """
    if len(returns) == 0:
        return 0.0, 0.0
    sorted_returns = returns.sort_values()
    percentile_idx = int((1.0 - alpha) * len(sorted_returns))
    
    # Daily VaR: maximum expected loss at (1-alpha) confidence
    # We take the negative value to represent loss as positive, or keep sign.
    # Standard practice: VaR is represented as positive loss, but let's keep daily return sign
    # (negative) to avoid sign confusion, or return absolute loss. Let's return absolute loss.
    var_val = abs(sorted_returns.iloc[percentile_idx])
    
    # CVaR is the average return of the worst (1-alpha) percentile outcomes
    cvar_val = abs(sorted_returns.iloc[:percentile_idx + 1].mean())
    
    return float(var_val), float(cvar_val)

def compute_rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    """Computes rolling Sharpe ratio over a moving window."""
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    return (rolling_mean / (rolling_std + 1e-15)) * np.sqrt(252.0)

def compute_rolling_volatility(returns: pd.Series, window: int = 63) -> pd.Series:
    """Computes rolling annualized volatility."""
    return returns.rolling(window=window).std() * np.sqrt(252.0)

def compute_rolling_beta(
    returns: pd.Series, 
    benchmark_returns: pd.Series, 
    window: int = 63
) -> pd.Series:
    """
    Computes the rolling beta of returns relative to benchmark returns.
    Beta = Cov(Asset, Benchmark) / Var(Benchmark)
    """
    # Align indices
    df = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if df.empty:
        return pd.Series(0.0, index=returns.index)
        
    asset_col = df.columns[0]
    bench_col = df.columns[1]
    
    covariance = df[asset_col].rolling(window=window).cov(df[bench_col])
    benchmark_variance = df[bench_col].rolling(window=window).var()
    
    return covariance / (benchmark_variance + 1e-15)

def calculate_portfolio_risk_report(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
    """Generates a complete dictionary of risk performance metrics for a return series."""
    cagr = calculate_cagr(returns)
    vol = calculate_annualized_volatility(returns)
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd = calculate_max_drawdown(returns)
    calmar = calculate_calmar_ratio(returns)
    win_rate = calculate_win_rate(returns)
    profit_factor = calculate_profit_factor(returns)
    avg_dd = calculate_average_drawdown(returns)
    dd_dur = calculate_drawdown_duration(returns)
    
    var_95, cvar_95 = calculate_var_cvar(returns, 0.95)
    var_99, cvar_99 = calculate_var_cvar(returns, 0.99)
    
    report = {
        "CAGR": cagr,
        "Annualized_Volatility": vol,
        "Sharpe_Ratio": sharpe,
        "Sortino_Ratio": sortino,
        "Calmar_Ratio": calmar,
        "Max_Drawdown": max_dd,
        "Average_Drawdown": avg_dd,
        "Drawdown_Duration_Days": float(dd_dur),
        "Win_Rate": win_rate,
        "Profit_Factor": profit_factor,
        "Daily_VaR_95": var_95,
        "Daily_CVaR_95": cvar_95,
        "Daily_VaR_99": var_99,
        "Daily_CVaR_99": cvar_99
    }
    
    if benchmark_returns is not None:
        # Calculate full beta
        df = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if not df.empty:
            cov = df.cov().iloc[0, 1]
            var = df.iloc[:, 1].var()
            report["Beta"] = float(cov / (var + 1e-15))
        else:
            report["Beta"] = 1.0
            
    return report
