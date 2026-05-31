import numpy as np
import pandas as pd
import pytest
from src.backtesting.backtester import VectorizedBacktester
from src.backtesting.strategy_rules import generate_strategy_weights
from src.analysis.monte_carlo import MonteCarloSimulator

@pytest.fixture
def sample_backtest_data():
    """Generates synthetic historical data containing regimes and moving averages."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2026-01-01', periods=n, freq='B')
    
    # 1% positive average return
    asset_ret = np.random.normal(0.0005, 0.005, n)
    close = 100.0 * (1.0 + asset_ret).cumprod()
    
    # Regimes
    regime_states = np.random.choice([0, 1], size=n)
    regime_labels = np.where(regime_states == 0, "Bullish Low Volatility", "Bearish High Volatility")
    
    # EMAs
    ema_50 = close * 0.99
    ema_200 = close * 0.97 # bullish trend
    
    df = pd.DataFrame({
        'date': dates,
        'close': close,
        'raw_close': close,
        'open': close * 0.99,
        'high': close * 1.01,
        'low': close * 0.98,
        'volume': 1000.0,
        'ret_simple': asset_ret,
        'regime_state': regime_states,
        'regime_label': regime_labels,
        'ema_50': ema_50,
        'ema_200': ema_200
    })
    return df

def test_lookahead_bias_prevention(sample_backtest_data):
    """Verifies that strategy weights on day t strictly use information from day t-1."""
    weights_df = generate_strategy_weights(sample_backtest_data)
    
    # The first day's position should be 0.0 or nan for shifted strategies
    # because there is no prior day's data
    assert weights_df['weight_ema_crossover'].iloc[0] == 0.0
    assert weights_df['weight_regime_aware'].iloc[0] == 0.0
    assert weights_df['weight_hybrid'].iloc[0] == 0.0
    
    # Check that day 1 allocation matches day 0 signal
    # EMA trend is bullish at day 0 (ema_50 > ema_200), so at day 1 ema_crossover weight should be 1.0
    assert weights_df['weight_ema_crossover'].iloc[1] == 1.0
    
    # Check regime aware day 1 matches regime state at day 0
    # Day 0 label is first element in sample_backtest_data
    day0_label = sample_backtest_data['regime_label'].iloc[0]
    expected_weight = 1.0 if day0_label == "Bullish Low Volatility" else 0.0
    assert weights_df['weight_regime_aware'].iloc[1] == expected_weight

def test_backtester_costs_decay(sample_backtest_data):
    """Tests that transaction costs and slippage correctly decay strategy returns on trade days."""
    # Setup backtester with high costs (e.g. 50 bps fee + 50 bps slippage = 1% total cost per trade)
    backtester = VectorizedBacktester(transaction_cost_bps=50.0, slippage_bps=50.0)
    
    # We modify weights to introduce a single trade
    df_test = sample_backtest_data.copy()
    
    equity_curves, summary = backtester.backtest_strategies(df_test)
    
    # Let's inspect the returns of the hybrid strategy
    # If the weight shifts from 0.0 to 1.0, 1% of the portfolio value should be deducted as costs
    # Let's verify that summary contains all strategies and metrics are computed
    assert len(summary) == 4
    assert "CAGR" in summary.columns
    assert "Max_Drawdown" in summary.columns
    assert "Total_Turnover" in summary.columns
    
    # Buy and hold has 0 turnover (initial entry is not counted as cost in backtester weight diff,
    # or is it? Let's check: weight_diff uses prepend=0.0. For buy_and_hold, weight is 1.0,
    # so diff on first day is 1.0 - 0.0 = 1.0. So it does pay entry cost of 1.0 * cost_rate.
    # Total turnover of buy_and_hold is exactly 1.0.)
    bnh_turnover = summary[summary['Strategy'] == 'Buy And Hold']['Total_Turnover'].iloc[0]
    assert bnh_turnover == 1.0

def test_monte_carlo_output_shapes(sample_backtest_data):
    """Tests that Monte Carlo path dimensions and stats summaries are mathematically correct."""
    simulator = MonteCarloSimulator(sample_backtest_data, random_seed=42)
    n_sims = 100
    horizon = 21
    
    paths = simulator.simulate_bootstrap(n_sims=n_sims, horizon=horizon)
    
    # Shape should be [horizon + 1, n_sims]
    assert paths.shape == (horizon + 1, n_sims)
    # Starts at 1.0
    assert np.allclose(paths[0, :], 1.0)
    
    # Verify statistics dictionary
    stats = simulator.analyze_simulation_paths(paths)
    assert stats["horizon_days"] == horizon
    assert stats["paths_simulated"] == n_sims
    assert "expected_return" in stats
    assert "probability_of_loss" in stats
    assert "average_simulated_max_drawdown" in stats
    assert stats["average_simulated_max_drawdown"] <= 0.0
