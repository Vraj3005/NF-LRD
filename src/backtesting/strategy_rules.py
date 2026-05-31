import numpy as np
import pandas as pd
from typing import Dict, Any

def get_regime_weight_mapping(regime_label: str) -> float:
    """
    Returns the target portfolio allocation weight based on the regime label.
    Optimized for dynamic leverage in high-conviction bull markets and capital defense:
    - Bullish Low Volatility: 2.5 (Aggressive leveraged long)
    - Bullish High Volatility: 1.5 (Leveraged long)
    - Recovery Regime: 1.5 (Leveraged long)
    - Sideways Low Volatility: 0.8 (Moderate exposure)
    - Bearish High Volatility: 0.0 (Cash defense)
    - Distribution / Risk-Off Regime: 0.0 (Cash defense)
    """
    mapping = {
        "Bullish Low Volatility": 2.5,
        "Bullish High Volatility": 1.5,
        "Recovery Regime": 1.5,
        "Sideways Low Volatility": 0.8,
        "Bearish High Volatility": 0.0,
        "Distribution / Risk-Off Regime": 0.0
    }
    return mapping.get(regime_label, 0.0)

def generate_strategy_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates historical daily target weights for the four compared strategies.
    Enforces shift(1) on all signals to prevent lookahead bias.
    
    Strategies:
    - strat_buy_and_hold: 100% long at all times.
    - strat_ema_crossover: 100% long if EMA 50 > EMA 200, 0% else.
    - strat_regime_aware: Allocates based on the HMM regime weight mapping.
    - strat_hybrid: Leveraged long (2.0x) if regime is favorable AND EMA 50 > EMA 200, 0% else.
    """
    weights = pd.DataFrame(index=df.index)
    
    # 1. Buy and Hold
    weights['weight_buy_and_hold'] = 1.0
    
    # 2. Trend Crossover (EMA 50 / EMA 200)
    if 'ema_50' in df.columns and 'ema_200' in df.columns:
        ema_50 = df['ema_50']
        ema_200 = df['ema_200']
    else:
        close = df['raw_close'] if 'raw_close' in df.columns else df['close']
        ema_50 = close.ewm(span=50, adjust=False).mean()
        ema_200 = close.ewm(span=200, adjust=False).mean()
        
    ema_bullish = (ema_50 > ema_200).astype(float)
    weights['weight_ema_crossover'] = ema_bullish.shift(1).fillna(0.0)
    
    # 3. Regime-Aware allocation
    regime_labels_shifted = df['regime_label'].shift(1)
    weights['weight_regime_aware'] = regime_labels_shifted.apply(get_regime_weight_mapping).fillna(0.0)
    
    # 4. Hybrid Regime + Trend
    favorable_regime = regime_labels_shifted.isin([
        "Bullish Low Volatility",
        "Bullish High Volatility",
        "Recovery Regime",
        "Sideways Low Volatility"
    ]).astype(float)
    
    hybrid_signal = (ema_bullish.shift(1) == 1.0) & (favorable_regime == 1.0)
    weights['weight_hybrid'] = np.where(hybrid_signal, 2.0, 0.0)
    
    return weights
