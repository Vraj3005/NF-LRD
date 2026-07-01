from typing import List

import numpy as np
import pandas as pd


def compute_rolling_volatilities(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Computes rolling standard deviations of log returns (close-to-close volatility)
    across multiple lookback windows.

    Args:
        df: pd.DataFrame containing a 'close' column.
        windows: List of window lengths.

    Returns:
        pd.DataFrame: Close-to-close volatility columns.
    """
    feats = pd.DataFrame(index=df.index)
    log_ret = np.log(df["close"] / df["close"].shift(1))

    for w in windows:
        # Daily rolling standard deviation of log returns
        feats[f"vol_close_{w}d"] = log_ret.rolling(window=w).std()

    return feats


def compute_parkinson_volatility(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Computes rolling Parkinson Volatility.
    Uses the high/low range to estimate volatility.

    Formula:
    Daily_Var = (ln(High / Low)^2) / (4 * ln(2))
    Parkinson_Vol = sqrt( Rolling_Mean(Daily_Var, window) )
    """
    # Prevent division by zero if High == Low by adding epsilon
    hl_ratio = df["high"] / (df["low"] + 1e-10)

    # Clip ratio to avoid extreme values in log
    hl_ratio = hl_ratio.clip(lower=1.00001)

    daily_var = (np.log(hl_ratio) ** 2) / (4.0 * np.log(2.0))
    return np.sqrt(daily_var.rolling(window=window).mean())


def compute_garman_klass_volatility(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Computes rolling Garman-Klass Volatility.
    Incorporates open, high, low, and close prices.

    Formula:
    Daily_Var = 0.5 * ln(High/Low)^2 - (2*ln(2) - 1) * ln(Close/Open)^2
    GK_Vol = sqrt( Rolling_Mean(Daily_Var, window) )
    """
    hl_ratio = df["high"] / (df["low"] + 1e-10)
    hl_ratio = hl_ratio.clip(lower=1.00001)

    co_ratio = df["close"] / (df["open"] + 1e-10)
    co_ratio = co_ratio.clip(lower=1e-5)

    term1 = 0.5 * (np.log(hl_ratio) ** 2)
    term2 = (2.0 * np.log(2.0) - 1.0) * (np.log(co_ratio) ** 2)

    daily_var = term1 - term2
    # GK variance can occasionally go slightly negative due to float inaccuracies or extreme jumps,
    # so we clip it to 0 before taking the square root.
    daily_var = daily_var.clip(lower=0.0)

    return np.sqrt(daily_var.rolling(window=window).mean())


def compute_atr_features(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Computes Average True Range (ATR), ATR percentage, and High-Low range percentage.

    Args:
        df: pd.DataFrame containing high, low, close.
        period: Wilder's smoothing lookback period (default 14).

    Returns:
        pd.DataFrame: ATR and range percentage features.
    """
    feats = pd.DataFrame(index=df.index)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    # 1. True Range (TR)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 2. Average True Range (ATR) using Wilder's smoothing (EWM)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    feats[f"atr_{period}"] = atr

    # 3. ATR Percentage (normalized by close price)
    feats[f"atr_pct_{period}"] = atr / (close + 1e-10)

    # 4. High-Low range percentage
    feats["hl_range_pct"] = (high - low) / (close + 1e-10)

    return feats


def compute_ewma_volatility(df: pd.DataFrame, span: int = 21) -> pd.Series:
    """
    Computes Exponentially Weighted Moving Average (EWMA) volatility of log returns.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    # Daily EWMA standard deviation
    ewma_var = log_ret.pow(2).ewm(span=span, adjust=False).mean()
    return np.sqrt(ewma_var)


def compute_volatility_percentile(
    vol_series: pd.Series, window: int = 252
) -> pd.Series:
    """
    Computes the rolling rank/percentile of the current volatility compared to its historical distribution.
    Matches volatility regime percentile.
    """
    return vol_series.rolling(window=window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
    )


def compute_realized_vol_ratio(
    df: pd.DataFrame, short_w: int = 5, long_w: int = 21
) -> pd.Series:
    """
    Computes the ratio of short-term realized volatility to long-term realized volatility.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol_short = log_ret.rolling(window=short_w).std()
    vol_long = log_ret.rolling(window=long_w).std()
    return vol_short / (vol_long + 1e-10)
