import numpy as np
import pandas as pd
from typing import List, Union

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes simple, log, multi-period, and rolling cumulative returns.

    Args:
        df: pd.DataFrame containing a 'close' column.

    Returns:
        pd.DataFrame: New DataFrame containing the returns features.
    """
    feats = pd.DataFrame(index=df.index)
    close = df['close']
    
    # 1. Simple Daily Return
    feats['ret_simple'] = close.pct_change(fill_method=None)
    
    # 2. Log Return
    feats['ret_log'] = np.log(close / close.shift(1))
    
    # 3. Multi-period returns
    for days in [3, 5, 10, 21]:
        feats[f'ret_{days}d'] = close.pct_change(periods=days, fill_method=None)
        
    # 4. Rolling cumulative return (21-day cumulative log returns sum converted to simple return)
    feats['ret_cum_21d'] = np.exp(feats['ret_log'].rolling(window=21).sum()) - 1
    
    return feats

def compute_rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Calculates the rolling linear regression slope of a series.
    Vectorized implementation using rolling sums for high performance.
    
    y = beta * x + alpha
    beta = (N*sum(xy) - sum(x)*sum(y)) / (N*sum(x^2) - (sum(x))^2)
    """
    N = window
    x = np.arange(1, N + 1)
    sum_x = x.sum()
    sum_x_sq = (x ** 2).sum()
    
    # Calculate rolling sums
    sum_y = series.rolling(window=window).sum()
    
    # Vectorized calculation of sum(x * y)
    # We construct a custom rolling calculation for x * y
    xy_series = series.copy()
    # Apply weights to get sum(x * y)
    # y_i * i where i goes from 1 to N
    # We can do this with a rolling window dot product or raw rolling sum of weighted terms.
    # To do this cleanly, we can use a rolling window apply, or vectorized rolling sum.
    # Since rolling apply can be slow, let's use a fast trick:
    # rolling sum of y_t * t minus adjustments, or a simple rolling apply with numba/raw numpy.
    # Let's write a clean, robust, vectorized rolling window multiplier:
    weights = np.arange(1, window + 1)
    
    def get_slope(y_window):
        if len(y_window) < window:
            return np.nan
        # dot product of weights and y_window
        cov = np.dot(weights, y_window)
        beta = (N * cov - sum_x * y_window.sum()) / (N * sum_x_sq - sum_x**2)
        return beta
        
    return series.rolling(window=window).apply(get_slope, raw=True)

def compute_trends(
    df: pd.DataFrame, 
    sma_windows: List[int], 
    ema_windows: List[int]
) -> pd.DataFrame:
    """
    Computes moving averages (SMA and EMA), distances to them,
    crossovers, and slope measures.

    Args:
        df: pd.DataFrame containing a 'close' column.
        sma_windows: List of SMA lookback window lengths.
        ema_windows: List of EMA lookback window lengths.

    Returns:
        pd.DataFrame: Trend features.
    """
    feats = pd.DataFrame(index=df.index)
    close = df['close']
    
    # 1. Simple Moving Averages and distances
    for w in sma_windows:
        sma = close.rolling(window=w).mean()
        feats[f'sma_{w}'] = sma
        feats[f'dist_sma_{w}'] = (close - sma) / sma
        # Calculate SMA slope (5-day rolling slope)
        feats[f'slope_sma_{w}'] = compute_rolling_slope(sma, window=5)

    # 2. Exponential Moving Averages and distances
    for w in ema_windows:
        ema = close.ewm(span=w, adjust=False).mean()
        feats[f'ema_{w}'] = ema
        feats[f'dist_ema_{w}'] = (close - ema) / ema
        # Calculate EMA slope (5-day rolling slope)
        feats[f'slope_ema_{w}'] = compute_rolling_slope(ema, window=5)

    # 3. Trend Crossover Signals
    # Short-term Crossover: EMA 9 vs EMA 21
    if 9 in ema_windows and 21 in ema_windows:
        feats['cross_ema_9_21'] = np.where(feats['ema_9'] > feats['ema_21'], 1.0, -1.0)
    else:
        # Fallback to computing them on-the-fly
        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        feats['cross_ema_9_21'] = np.where(ema_9 > ema_21, 1.0, -1.0)
        
    # Long-term Crossover (Golden Cross): SMA 50 vs SMA 200
    if 50 in sma_windows and 200 in sma_windows:
        feats['cross_sma_50_200'] = np.where(feats['sma_50'] > feats['sma_200'], 1.0, -1.0)
    else:
        sma_50 = close.rolling(window=50).mean()
        sma_200 = close.rolling(window=200).mean()
        feats['cross_sma_50_200'] = np.where(sma_50 > sma_200, 1.0, -1.0)

    # Keep only distance, slope and crossover columns (raw MAs are not scale-invariant)
    cols_to_keep = [c for c in feats.columns if 'dist' in c or 'slope' in c or 'cross' in c]
    return feats[cols_to_keep]

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes Wilder's Relative Strength Index (RSI) 14.
    """
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    
    # Wilder's smoothing using EWM
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    
    rs = roll_up / (roll_down + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def compute_momentum(
    df: pd.DataFrame,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    stoch_k: int = 14,
    stoch_d: int = 3,
    williams_r_period: int = 14,
    roc_period: int = 10
) -> pd.DataFrame:
    """
    Computes momentum features: RSI, MACD, ROC, Stochastic, and Williams %R.

    Args:
        df: pd.DataFrame containing OHLCV.
        ... configuration periods.

    Returns:
        pd.DataFrame: Momentum features.
    """
    feats = pd.DataFrame(index=df.index)
    close = df['close']
    high = df['high']
    low = df['low']
    
    # 1. RSI 14
    feats[f'rsi_{rsi_period}'] = compute_rsi(close, rsi_period)
    
    # 2. MACD
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    feats['macd_line'] = ema_fast - ema_slow
    feats['macd_signal'] = feats['macd_line'].ewm(span=macd_signal, adjust=False).mean()
    feats['macd_hist'] = feats['macd_line'] - feats['macd_signal']
    
    # 3. Rate of Change (ROC)
    feats[f'roc_{roc_period}'] = close.pct_change(periods=roc_period, fill_method=None)
    
    # 4. Stochastic Oscillator
    low_min = low.rolling(window=stoch_k).min()
    high_max = high.rolling(window=stoch_k).max()
    denom = high_max - low_min + 1e-10
    
    feats['stoch_k'] = ((close - low_min) / denom) * 100.0
    feats['stoch_d'] = feats['stoch_k'].rolling(window=stoch_d).mean()
    
    # 5. Williams %R
    feats['williams_r'] = ((high_max - close) / denom) * -100.0
    
    return feats
