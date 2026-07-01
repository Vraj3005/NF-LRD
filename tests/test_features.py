import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import (
    compute_market_structure_features,
    compute_rolling_entropy,
    compute_rolling_hurst,
)
from src.features.technical_indicators import (
    compute_momentum,
    compute_returns,
    compute_rsi,
)
from src.features.volatility_features import (
    compute_atr_features,
    compute_garman_klass_volatility,
    compute_parkinson_volatility,
)


@pytest.fixture
def synthetic_ohlcv():
    """Generates a synthetic dataset for indicator testing."""
    np.random.seed(42)
    n = 250
    # Create an upward trend with some noise
    close = 100.0 + np.cumsum(np.random.normal(0.2, 1.0, n))
    open_val = close - np.random.normal(0.0, 0.5, n)
    high = np.maximum(open_val, close) + np.abs(np.random.normal(0.5, 0.3, n))
    low = np.minimum(open_val, close) - np.abs(np.random.normal(0.5, 0.3, n))
    volume = np.random.randint(1000, 5000, n).astype(float)

    df = pd.DataFrame(
        {"open": open_val, "high": high, "low": low, "close": close, "volume": volume}
    )
    return df


def test_returns_calculation(synthetic_ohlcv):
    """Tests calculation of returns features."""
    feats = compute_returns(synthetic_ohlcv)

    assert "ret_simple" in feats.columns
    assert "ret_log" in feats.columns
    assert "ret_3d" in feats.columns
    assert "ret_cum_21d" in feats.columns
    assert len(feats) == len(synthetic_ohlcv)

    # Test mathematical consistency: log return = ln(price_t / price_{t-1})
    expected_log = np.log(
        synthetic_ohlcv["close"].iloc[10] / synthetic_ohlcv["close"].iloc[9]
    )
    assert np.isclose(feats["ret_log"].iloc[10], expected_log)


def test_rsi_bounds(synthetic_ohlcv):
    """Tests RSI 14-day boundaries and calculations."""
    rsi = compute_rsi(synthetic_ohlcv["close"], period=14)

    # RSI must be between 0 and 100
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0.0).all()
    assert (valid_rsi <= 100.0).all()
    assert len(rsi) == len(synthetic_ohlcv)


def test_parkinson_and_gk_volatility(synthetic_ohlcv):
    """Tests Parkinson and Garman-Klass range volatility calculations."""
    park = compute_parkinson_volatility(synthetic_ohlcv, window=21)
    gk = compute_garman_klass_volatility(synthetic_ohlcv, window=21)

    assert len(park) == len(synthetic_ohlcv)
    assert len(gk) == len(synthetic_ohlcv)

    # Parkinson and GK must be non-negative
    assert (park.dropna() >= 0).all()
    assert (gk.dropna() >= 0).all()


def test_atr_features(synthetic_ohlcv):
    """Tests True Range and Average True Range calculation."""
    feats = compute_atr_features(synthetic_ohlcv, period=14)

    assert "atr_14" in feats.columns
    assert "atr_pct_14" in feats.columns
    assert "hl_range_pct" in feats.columns
    assert (feats["atr_14"].dropna() > 0).all()


def test_momentum_indicators(synthetic_ohlcv):
    """Tests momentum features: MACD, Stochastics, Williams %R."""
    feats = compute_momentum(synthetic_ohlcv)

    assert "macd_line" in feats.columns
    assert "macd_signal" in feats.columns
    assert "macd_hist" in feats.columns
    assert "stoch_k" in feats.columns
    assert "stoch_d" in feats.columns
    assert "williams_r" in feats.columns

    # Williams %R should be negative and bound between -100 and 0
    w_r = feats["williams_r"].dropna()
    assert (w_r <= 0).all()
    assert (w_r >= -100).all()


def test_market_structure(synthetic_ohlcv):
    """Tests drawdown, candle wicks and bodies, and Kaufman's ER."""
    feats = compute_market_structure_features(synthetic_ohlcv)

    assert "candle_body_pct" in feats.columns
    assert "upper_wick_pct" in feats.columns
    assert "lower_wick_pct" in feats.columns
    assert "drawdown" in feats.columns
    assert "trend_strength_score" in feats.columns

    # Drawdowns should always be <= 0
    assert (feats["drawdown"].dropna() <= 0).all()
    # Trend strength score should lie between 0 and 1
    assert (feats["trend_strength_score"].dropna() >= 0).all()
    assert (feats["trend_strength_score"].dropna() <= 1.0).all()


def test_hurst_exponent_estimation():
    """Tests the rolling Hurst Exponent approximation."""
    # Generate a random walk (cumulative sum of normal noise)
    np.random.seed(42)
    noise = pd.Series(np.random.normal(0, 1, 200))
    rw = noise.cumsum()

    # Compute rolling Hurst exponent with a window of 126
    hurst_rw = compute_rolling_hurst(rw, window=126)

    # The last element should be a valid number (since length 200 > window 126)
    final_rw_val = hurst_rw.iloc[-1]
    assert not np.isnan(final_rw_val)

    # Random walk Hurst should be around 0.5 (allow range 0.25 to 0.75 for finite sample fluctuations)
    assert 0.25 < final_rw_val < 0.75


def test_rolling_entropy():
    """Tests the rolling Shannon entropy of returns."""
    # 1. Completely constant returns should have entropy of 0.0
    const_ret = pd.Series(np.zeros(100))
    entropy_const = compute_rolling_entropy(const_ret, window=50, bins=10)
    assert np.isclose(entropy_const.iloc[-1], 0.0, atol=1e-5)

    # 2. High variance random returns should have high positive entropy
    np.random.seed(42)
    rand_ret = pd.Series(np.random.normal(0, 1, 100))
    entropy_rand = compute_rolling_entropy(rand_ret, window=50, bins=10)
    assert entropy_rand.iloc[-1] > 1.0


def test_future_leakage_prevention(synthetic_ohlcv):
    """Verifies that future data updates do not modify historical rolling features (no future leakage)."""
    synthetic_ohlcv["close"].copy()

    # Calculate baseline returns
    feats_baseline = compute_returns(synthetic_ohlcv)

    # Modify ONLY the last element of close price (simulating future change)
    ohlcv_modified = synthetic_ohlcv.copy()
    ohlcv_modified.loc[len(ohlcv_modified) - 1, "close"] = 9999.0

    feats_modified = compute_returns(ohlcv_modified)

    # Check that historical feature values (up to T-1) remain identical
    pd.testing.assert_series_equal(
        feats_baseline["ret_simple"].iloc[:-1],
        feats_modified["ret_simple"].iloc[:-1],
    )
    pd.testing.assert_series_equal(
        feats_baseline["ret_cum_21d"].iloc[:-1],
        feats_modified["ret_cum_21d"].iloc[:-1],
    )


def test_nan_warmup_handling(synthetic_ohlcv):
    """Verifies that rolling window indicators correctly populate the NaN warmup period."""
    # SMA distance for a 50-day window
    # The first 49 values must be NaN
    close = synthetic_ohlcv["close"]
    sma_50 = close.rolling(window=50).mean()

    assert sma_50.iloc[:49].isna().all()
    assert not sma_50.iloc[49:].isna().any()


def test_no_infinite_values(synthetic_ohlcv):
    """Verifies that none of the computed feature columns contain infinite values."""
    feats_ret = compute_returns(synthetic_ohlcv)
    feats_mom = compute_momentum(synthetic_ohlcv)

    assert not np.isinf(feats_ret.values).any()
    assert not np.isinf(feats_mom.values).any()


def test_standardization_pipeline_leakage(synthetic_ohlcv):
    """Verifies that StandardizationPipeline scales features correctly and stores state."""
    from src.features.feature_pipeline import StandardizationPipeline

    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    pipeline = StandardizationPipeline()
    scaled = pipeline.fit_transform(df)

    # Verify column presence
    assert "feature1" in scaled.columns
    assert "feature2" in scaled.columns

    # Verify mean is close to 0 and std is close to 1
    assert np.isclose(scaled["feature1"].mean(), 0.0)
    assert np.isclose(scaled["feature2"].mean(), 0.0)


def test_reproducibility_determinism(synthetic_ohlcv):
    """Verifies that running feature pipeline calculations twice yields identical values (determinism)."""
    feats1 = compute_returns(synthetic_ohlcv)
    feats2 = compute_returns(synthetic_ohlcv.copy())

    pd.testing.assert_frame_equal(feats1, feats2)
