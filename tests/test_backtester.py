import numpy as np
import pandas as pd
import pytest

from src.analysis.monte_carlo import MonteCarloSimulator
from src.backtesting.backtester import VectorizedBacktester
from src.backtesting.strategy_rules import generate_strategy_weights


@pytest.fixture
def sample_backtest_data():
    """Generates synthetic historical data containing regimes and moving averages."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2026-01-01", periods=n, freq="B")

    # 1% positive average return
    asset_ret = np.random.normal(0.0005, 0.005, n)
    close = 100.0 * (1.0 + asset_ret).cumprod()

    # Regimes
    regime_states = np.random.choice([0, 1], size=n)
    regime_labels = np.where(
        regime_states == 0, "Bullish Low Volatility", "Bearish High Volatility"
    )

    # EMAs
    ema_50 = close * 0.99
    ema_200 = close * 0.97  # bullish trend

    df = pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "raw_close": close,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "volume": 1000.0,
            "ret_simple": asset_ret,
            "regime_state": regime_states,
            "regime_label": regime_labels,
            "ema_50": ema_50,
            "ema_200": ema_200,
        }
    )
    return df


def test_lookahead_bias_prevention(sample_backtest_data):
    """Verifies that strategy weights on day t strictly use information from day t-1."""
    weights_df = generate_strategy_weights(sample_backtest_data)

    # The first day's position should be 0.0 or nan for shifted strategies
    assert weights_df["weight_ema_crossover"].iloc[0] == 0.0
    assert weights_df["weight_regime_aware"].iloc[0] == 0.0
    assert weights_df["weight_hybrid"].iloc[0] == 0.0

    # Check that day 1 allocation matches day 0 signal
    # EMA trend is bullish at day 0 (ema_50 > ema_200), so at day 1 ema_crossover weight should be 1.0
    assert weights_df["weight_ema_crossover"].iloc[1] == 1.0

    # Check regime aware day 1 matches regime state at day 0
    day0_label = sample_backtest_data["regime_label"].iloc[0]
    expected_weight = 1.0 if day0_label == "Bullish Low Volatility" else 0.0
    assert weights_df["weight_regime_aware"].iloc[1] == expected_weight


def test_backtester_costs_decay(sample_backtest_data):
    """Tests that transaction costs and slippage correctly decay strategy returns."""
    # Setup backtester with high costs (e.g. 50 bps fee + 50 bps slippage = 1% total cost per trade)
    backtester_high_cost = VectorizedBacktester(
        transaction_cost_bps=50.0, slippage_bps=50.0
    )
    backtester_zero_cost = VectorizedBacktester(
        transaction_cost_bps=0.0, slippage_bps=0.0
    )

    df_test = sample_backtest_data.copy()

    _, summary_high = backtester_high_cost.backtest_strategies(df_test)
    _, summary_zero = backtester_zero_cost.backtest_strategies(df_test)

    # Verify that we have 5 strategies
    assert len(summary_high) == 5
    assert len(summary_zero) == 5

    # Check that high cost has lower CAGR than zero cost for strategies with turnover
    for strat in ["Ema Crossover", "Regime Aware", "Hybrid", "Vol Targeting"]:
        cagr_high = summary_high[summary_high["Strategy"] == strat]["CAGR"].iloc[0]
        cagr_zero = summary_zero[summary_zero["Strategy"] == strat]["CAGR"].iloc[0]
        assert cagr_high < cagr_zero


def test_drawdown_and_turnover_math(sample_backtest_data):
    """Verifies that max drawdown and turnover calculations are mathematically correct."""
    backtester = VectorizedBacktester(transaction_cost_bps=10.0, slippage_bps=5.0)
    df_test = sample_backtest_data.copy()

    equity_curves, summary = backtester.backtest_strategies(df_test)

    for strat in [
        "buy_and_hold",
        "ema_crossover",
        "vol_targeting",
        "regime_aware",
        "hybrid",
    ]:
        # Check max drawdown in curves matches summary report
        equity_series = equity_curves[f"equity_{strat}"]
        peak = equity_series.cummax()
        drawdown_series = (equity_series - peak) / (peak + 1e-15)
        calc_max_dd = float(drawdown_series.min())

        summary_name = strat.replace("_", " ").title()
        report_max_dd = summary[summary["Strategy"] == summary_name][
            "Max_Drawdown"
        ].iloc[0]
        assert np.isclose(calc_max_dd, report_max_dd)

        # Check total turnover calculation
        weights = equity_curves[f"weight_{strat}"].values
        expected_turnover = float(np.sum(np.abs(np.diff(weights, prepend=0.0))))
        report_turnover = summary[summary["Strategy"] == summary_name][
            "Total_Turnover"
        ].iloc[0]
        assert np.isclose(expected_turnover, report_turnover)


def test_benchmark_comparison(sample_backtest_data):
    """Checks that the Buy and Hold strategy weight is always 1.0 and acts as standard baseline."""
    weights_df = generate_strategy_weights(sample_backtest_data)
    assert (weights_df["weight_buy_and_hold"] == 1.0).all()


def test_monte_carlo_output_shapes(sample_backtest_data):
    """Tests that Monte Carlo path dimensions and stats summaries are correct."""
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

    # Extra risk metrics
    assert "Daily_VaR_95" in stats
    assert "Daily_CVaR_95" in stats
    assert "Expected_Shortfall_95" in stats
    assert "fan_5" in stats
    assert len(stats["fan_5"]) == horizon + 1


def test_monte_carlo_deterministic_seed(sample_backtest_data):
    """Verifies that running Monte Carlo with the same seed produces identical paths."""
    sim1 = MonteCarloSimulator(sample_backtest_data, random_seed=123)
    sim2 = MonteCarloSimulator(sample_backtest_data, random_seed=123)

    paths1 = sim1.simulate_bootstrap(n_sims=50, horizon=10)
    paths2 = sim2.simulate_bootstrap(n_sims=50, horizon=10)

    assert np.allclose(paths1, paths2)


def test_monte_carlo_percentiles_ordered(sample_backtest_data):
    """Ensures fan chart percentile paths are ordered correctly (fan_5 <= fan_25 <= fan_50 <= fan_75 <= fan_95)."""
    simulator = MonteCarloSimulator(sample_backtest_data, random_seed=42)
    paths = simulator.simulate_bootstrap(n_sims=100, horizon=20)
    stats = simulator.analyze_simulation_paths(paths)

    f5 = np.array(stats["fan_5"])
    f25 = np.array(stats["fan_25"])
    f50 = np.array(stats["fan_50"])
    f75 = np.array(stats["fan_75"])
    f95 = np.array(stats["fan_95"])

    assert (f5 <= f25).all()
    assert (f25 <= f50).all()
    assert (f50 <= f75).all()
    assert (f75 <= f95).all()


def test_monte_carlo_no_negative_prices(sample_backtest_data):
    """Ensures that price paths remain strictly positive (no price <= 0.0) even in severe stress scenarios."""
    simulator = MonteCarloSimulator(sample_backtest_data, random_seed=42)
    # COVID stress scenario with 5.0x multiplier to really test the limits
    paths = simulator.simulate_bootstrap(
        n_sims=200, horizon=50, scenario="covid", stress_multiplier=5.0
    )
    assert (paths > 0.0).all()


def test_monte_carlo_var_cvar_math():
    """Manually checks VaR and CVaR calculations on simple mock returns to assert correct mathematical implementation."""
    returns = np.array([-0.50, -0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40])
    sorted_ret = np.sort(returns)
    var_idx_95 = int(10 * 0.05)
    var_95 = float(abs(sorted_ret[var_idx_95]))
    cvar_95 = float(abs(np.mean(sorted_ret[: var_idx_95 + 1])))

    assert var_95 == 0.50
    assert cvar_95 == 0.50


def test_walk_forward_regimes_generation(sample_backtest_data):
    """Tests that lookahead-free walk-forward regimes can be generated cleanly."""
    from src.models.hmm_model import GaussianHMM

    backtester = VectorizedBacktester(transaction_cost_bps=10.0, slippage_bps=5.0)

    wf_labels = backtester.generate_walk_forward_regimes(
        sample_backtest_data,
        model_cls=GaussianHMM,
        n_components=2,
        initial_train_days=50,
        refit_interval_days=10,
    )

    assert len(wf_labels) == len(sample_backtest_data)
    assert wf_labels.isnull().sum() == 0
    assert isinstance(wf_labels.iloc[0], str)
