import numpy as np
import pandas as pd

from src.analysis.risk_metrics import (
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_drawdown_duration,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var_cvar,
)


def test_risk_metrics_constant_returns():
    """Tests risk metrics on a constant positive returns series."""
    # 1% return daily for 252 days = (1.01)^252 - 1 = 11.27
    daily_ret = 0.01
    returns = pd.Series([daily_ret] * 252)

    # Volatility should be near zero (numerical precision)
    vol = calculate_annualized_volatility(returns)
    assert vol == 0.0

    # Sharpe ratio should be 0 (since volatility is 0, handled safely as 0.0)
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe == 0.0

    # Max drawdown should be 0.0 (constantly rising equity curve)
    max_dd = calculate_max_drawdown(returns)
    assert max_dd == 0.0

    # CAGR calculation
    cagr = calculate_cagr(returns)
    expected_cagr = (1.0 + daily_ret) ** 252 - 1.0
    assert np.isclose(cagr, expected_cagr)


def test_max_drawdown_calculation():
    """Tests that max drawdown matches manual peak-to-trough drop."""
    # Construct returns that lead to a specific price path:
    # Day 0: 100
    # Day 1: 110 (+10%)
    # Day 2: 99 (-10%)
    # Day 3: 88 (-11.11%)
    # Day 4: 121 (+37.5%)
    # Peak is 110, Trough is 88. Max drawdown = (88 - 110) / 110 = -22 / 110 = -0.20 (-20%)
    returns = pd.Series([0.10, -0.10, -0.11111111, 0.375])

    max_dd = calculate_max_drawdown(returns)
    assert np.isclose(max_dd, -0.20, atol=1e-4)


def test_drawdown_duration():
    """Tests drawdown duration in trading days."""
    # Day 0: 100
    # Day 1: 110 (Peak)
    # Day 2: 99 (Drawdown day 1)
    # Day 3: 99 (Drawdown day 2)
    # Day 4: 121 (Recovery, new peak)
    # Consecutive days spent below peak = 2 days (days 2 and 3)
    returns = pd.Series([0.10, -0.10, 0.0, 0.22222222])
    duration = calculate_drawdown_duration(returns)
    assert duration == 2


def test_sharpe_and_sortino_ratio():
    """Tests Sharpe and Sortino ratio calculations."""
    # Generate 100 days of random returns with positive mean
    np.random.seed(42)
    daily_rets = np.random.normal(0.001, 0.01, 252)
    returns = pd.Series(daily_rets)

    # Calculate Sharpe
    sharpe = calculate_sharpe_ratio(returns)
    ann_mean = returns.mean() * 252.0
    ann_std = returns.std() * np.sqrt(252.0)
    expected_sharpe = ann_mean / ann_std
    assert np.isclose(sharpe, expected_sharpe)

    # Sortino should be larger than Sharpe if negative returns are smaller than positive returns
    # Let's verify Sortino computes successfully
    sortino = calculate_sortino_ratio(returns)
    assert isinstance(sortino, float)
    assert not np.isnan(sortino)


def test_var_and_cvar():
    """Tests Value at Risk and Conditional Value at Risk percentile selections."""
    # 100 returns sorted: -50, -49, ... , 48, 49
    rets_arr = np.arange(-50, 50).astype(float) / 100.0
    returns = pd.Series(rets_arr)

    # Alpha = 0.95 (worst 5% tail index is 5th element)
    # Sorted indices: 0 to 99. Index of 5% is: int(0.05 * 100) = 5
    # The 5th sorted return is: -50 + 5 = -45 / 100 = -0.45
    # Daily VaR 95% = 0.45 (represented as positive loss)
    # CVaR 95% is mean of elements 0 to 5: [-0.50, -0.49, -0.48, -0.47, -0.46, -0.45]
    # Mean = -0.475
    # CVaR 95% = 0.475
    var_95, cvar_95 = calculate_var_cvar(returns, alpha=0.95)

    assert np.isclose(var_95, 0.45)
    assert np.isclose(cvar_95, 0.475)
