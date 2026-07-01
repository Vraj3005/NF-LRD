"""
Strategy rules and allocation weight generators for the portfolio backtesting engine.
Enforces no leverage by default and shifts signals by 1 day to prevent lookahead leakages.
"""

import numpy as np
import pandas as pd


def get_regime_weight_mapping(regime_label: str) -> float:
    """
    Returns the target portfolio allocation weight based on the regime label.
    Capped at 1.0 (no leverage) by default for capital defense:
    - Bullish Low Volatility: 1.0
    - Bullish High Volatility: 0.75
    - Recovery Regime: 0.75
    - Sideways Low Volatility: 0.50
    - Distribution / Risk-Off Regime: 0.25
    - Bearish High Volatility: 0.0
    """
    mapping = {
        "Bullish Low Volatility": 1.0,
        "Bullish High Volatility": 0.75,
        "Recovery Regime": 0.75,
        "Sideways Low Volatility": 0.50,
        "Distribution / Risk-Off Regime": 0.25,
        "Bearish High Volatility": 0.0,
    }
    return mapping.get(regime_label, 0.0)


def generate_strategy_weights(
    df: pd.DataFrame, target_vol: float = 0.15, vol_lookback: int = 21
) -> pd.DataFrame:
    """
    Generates historical daily target weights for the five compared strategies.
    Enforces shift(1) on all signals to prevent lookahead bias.

    Strategies:
    - weight_buy_and_hold: 100% long at all times.
    - weight_ema_crossover: 100% long if EMA 50 > EMA 200, 0% else.
    - weight_vol_targeting: Target vol exposure = target_vol / rolling_vol, capped at 1.0.
    - weight_regime_aware: Allocates based on the regime weight mapping.
    - weight_hybrid: Combines regime allocation with the EMA trend filter (Regime * Crossover).
    """
    weights = pd.DataFrame(index=df.index)

    # 1. Buy and Hold
    weights["weight_buy_and_hold"] = 1.0

    # 2. Trend Crossover (EMA 50 / EMA 200)
    if "ema_50" in df.columns and "ema_200" in df.columns:
        ema_50 = df["ema_50"]
        ema_200 = df["ema_200"]
    else:
        close = df["raw_close"] if "raw_close" in df.columns else df["close"]
        ema_50 = close.ewm(span=50, adjust=False).mean()
        ema_200 = close.ewm(span=200, adjust=False).mean()

    ema_bullish = (ema_50 > ema_200).astype(float)
    weights["weight_ema_crossover"] = ema_bullish.shift(1).fillna(0.0)

    # 3. Volatility Targeting
    close_col = "raw_close" if "raw_close" in df.columns else "close"
    daily_ret = df[close_col].pct_change(fill_method=None).fillna(0.0)
    rolling_vol = daily_ret.rolling(window=vol_lookback).std() * np.sqrt(252.0)
    rolling_vol = rolling_vol.fillna(target_vol)

    vol_target_weight = np.minimum(1.0, target_vol / (rolling_vol + 1e-15))
    weights["weight_vol_targeting"] = vol_target_weight.shift(1).fillna(1.0)

    # 4. Regime-Aware allocation
    if "regime_label" in df.columns:
        regime_labels_shifted = df["regime_label"].shift(1)
        weights["weight_regime_aware"] = regime_labels_shifted.apply(
            get_regime_weight_mapping
        ).fillna(0.0)
    else:
        weights["weight_regime_aware"] = 0.0

    # 5. Hybrid Regime + Trend Crossover
    weights["weight_hybrid"] = (
        weights["weight_regime_aware"] * weights["weight_ema_crossover"]
    )

    return weights
