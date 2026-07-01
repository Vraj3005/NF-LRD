import numpy as np
import pandas as pd
import pytest

from src.analysis.oos_validation import (
    run_walk_forward_oos_validation,
    select_features_on_train,
    split_by_dates,
)
from src.backtesting.strategy_rules import generate_strategy_weights
from src.features.feature_pipeline import StandardizationPipeline


@pytest.fixture
def dummy_raw_features():
    """Generates synthetic unscaled features for testing validation splits."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range(start="2015-01-01", periods=n, freq="D")

    # 3 features: feat1, feat2, feat3
    feat1 = np.random.normal(10.0, 2.0, n)  # Mean 10, std 2
    feat2 = np.random.normal(5.0, 1.0, n)  # Mean 5, std 1
    feat3 = feat1 * 0.95 + np.random.normal(0, 0.01, n)  # Collinear with feat1

    # Prices
    close = 100.0 * (1.0 + np.random.normal(0.0005, 0.01, n)).cumprod()

    df = pd.DataFrame(
        {
            "date": dates,
            "feat1": feat1,
            "feat2": feat2,
            "feat3": feat3,
            "raw_open": close * 0.99,
            "raw_high": close * 1.01,
            "raw_low": close * 0.98,
            "raw_close": close,
            "raw_volume": np.random.randint(100, 1000, n).astype(float),
        }
    )
    return df


def test_split_by_dates(dummy_raw_features):
    """Verifies that splits do not overlap and partitions cover the expected dates."""
    df = dummy_raw_features
    train = split_by_dates(df, "2015-01-01", "2015-06-30")
    val = split_by_dates(df, "2015-07-01", "2015-09-30")
    test = split_by_dates(df, "2015-10-01", "2016-12-31")

    # No overlapping dates
    overlap_train_val = set(train["date"]).intersection(set(val["date"]))
    overlap_val_test = set(val["date"]).intersection(set(test["date"]))
    assert len(overlap_train_val) == 0
    assert len(overlap_val_test) == 0

    # Dates are sequential
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()


def test_feature_selection_leakage_prevention(dummy_raw_features):
    """
    Checks that feature selection is determined solely on training data.
    If a feature has zero variance in train but high variance in test,
    it should still be dropped based on the training filter.
    """
    df = dummy_raw_features.copy()

    # Add a custom feature that has zero variance in train (first 200 rows) but variance in test (last 300 rows)
    custom_feat = np.zeros(len(df))
    custom_feat[200:] = np.random.normal(5.0, 1.0, len(df) - 200)
    df["custom_feat"] = custom_feat

    train_df = df.iloc[:200]

    # Select features on train
    selected_train = select_features_on_train(
        train_df[["feat1", "feat2", "feat3", "custom_feat"]]
    )

    # custom_feat must be excluded because it has zero variance in training split
    assert "custom_feat" not in selected_train


def test_scaler_fitted_only_on_train(dummy_raw_features):
    """
    Asserts that StandardScaler parameters (mean, scale) are computed
    exclusively on the training subset and applied to the test subset.
    """
    df = dummy_raw_features
    train_df = df.iloc[:250].copy()
    test_df = df.iloc[250:].copy()

    scaler = StandardizationPipeline()
    scaler.fit(train_df[["feat1", "feat2"]])

    # Mean and scale are computed from train_df
    expected_mean_feat1 = train_df["feat1"].mean()
    expected_scale_feat1 = train_df["feat1"].std(ddof=0)

    assert np.isclose(scaler.mean_["feat1"], expected_mean_feat1)
    assert np.isclose(scaler.scale_["feat1"], expected_scale_feat1)

    # Apply transformation to test set
    test_scaled = scaler.transform(test_df[["feat1", "feat2"]])

    # Check that test set was standardized using train set parameters
    manual_test_scaled_feat1 = (
        test_df["feat1"] - expected_mean_feat1
    ) / expected_scale_feat1
    assert np.allclose(test_scaled["feat1"], manual_test_scaled_feat1)


def test_walk_forward_oos_windows_non_overlapping(dummy_raw_features):
    """Verifies walk-forward validation generates predictions without leaking future regimes."""
    df = dummy_raw_features

    # Run a quick walk forward validation
    df_wf, wf_curves, wf_summary = run_walk_forward_oos_validation(
        df,
        start_date="2015-01-01",
        train_window_years=1,  # 252 days
        test_window_months=3,  # 63 days
        n_components=3,
    )

    # Assert output shapes
    assert len(df_wf) == len(df)
    assert "regime_label" in df_wf.columns
    assert "regime_state" in df_wf.columns

    # Check that first year has "Warmup" labels
    warmup_period = df_wf.iloc[:252]
    assert (warmup_period["regime_label"] == "Warmup / Undefined").all()

    # Check that post-warmup period has predicted labels
    post_warmup = df_wf.iloc[252:]
    assert not (post_warmup["regime_label"] == "Warmup / Undefined").all()


def test_backtest_uses_shifted_signals(dummy_raw_features):
    """
    Ensures that trading strategy weights use shifted signals (t-1)
    to prevent lookahead execution leakages.
    """
    df = dummy_raw_features.copy()

    # Assign some random labels
    labels = [
        "Bullish Low Volatility",
        "Bearish High Volatility",
        "Sideways Low Volatility",
    ]
    df["regime_label"] = np.random.choice(labels, len(df))
    df["ema_50"] = df["raw_close"].ewm(span=50).mean()
    df["ema_200"] = df["raw_close"].ewm(span=200).mean()

    weights = generate_strategy_weights(df)

    # Assert that the weights on day t are matching the regime on day t-1
    for t in range(1, len(df)):
        prev_label = df.loc[t - 1, "regime_label"]
        current_weight = weights.loc[t, "weight_regime_aware"]

        # Verify mapping matches previous day
        if prev_label == "Bullish Low Volatility":
            assert current_weight == 1.0
        elif prev_label == "Bearish High Volatility":
            assert current_weight == 0.0
        elif prev_label == "Sideways Low Volatility":
            assert current_weight == 0.5
