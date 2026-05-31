import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, ".")

from src.visualization.dashboard_components import load_all_dashboard_data
from src.analysis.risk_metrics import calculate_portfolio_risk_report

def get_optimized_regime_weight(label: str) -> float:
    # Aggressive bull market leverage weights
    mapping = {
        "Bullish Low Volatility": 2.6,
        "Bullish High Volatility": 1.8,
        "Recovery Regime": 1.8,
        "Sideways Low Volatility": 0.8,
        "Bearish High Volatility": 0.0,
        "Distribution / Risk-Off Regime": 0.0
    }
    return mapping.get(label, 0.0)

def main():
    data = load_all_dashboard_data()
    df = data["labeled_data"]
    if df is None:
        print("No labeled data found on disk!")
        return
        
    print(f"Loaded labeled data containing {len(df)} rows.")
    
    asset_ret = df['ret_simple'].fillna(0.0).values
    
    # 1. Buy and Hold (baseline)
    weights_bh = np.ones(len(df))
    ret_bh = asset_ret
    report_bh = calculate_portfolio_risk_report(pd.Series(ret_bh))
    
    # 2. EMA Crossover
    close = df['raw_close'] if 'raw_close' in df.columns else df['close']
    ema_50 = close.ewm(span=50, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()
    ema_bullish = (ema_50 > ema_200).astype(float).shift(1).fillna(0.0).values
    
    cash_rate_daily = 0.055 / 252.0
    borrow_rate_daily = 0.065 / 252.0
    total_cost_rate = (10.0 + 5.0) / 10000.0
    
    ema_diff = np.abs(np.diff(ema_bullish, prepend=0.0))
    ema_costs = ema_diff * total_cost_rate
    ema_cash_frac = np.maximum(0.0, 1.0 - ema_bullish)
    ret_ema = (ema_bullish * asset_ret) + (ema_cash_frac * cash_rate_daily) - ema_costs
    report_ema = calculate_portfolio_risk_report(pd.Series(ret_ema))
    
    # 3. Optimized Regime-Aware Strategy
    regime_labels_shifted = df['regime_label'].shift(1).fillna("Sideways Low Volatility")
    weights_ra = regime_labels_shifted.apply(get_optimized_regime_weight).values
    
    ra_diff = np.abs(np.diff(weights_ra, prepend=0.0))
    ra_costs = ra_diff * total_cost_rate
    
    ra_cash_frac = np.maximum(0.0, 1.0 - np.abs(weights_ra))
    ra_cash_yield = ra_cash_frac * cash_rate_daily
    
    ra_borrow_frac = np.maximum(0.0, np.abs(weights_ra) - 1.0)
    ra_borrow_cost = ra_borrow_frac * borrow_rate_daily
    
    ret_ra = (weights_ra * asset_ret) + ra_cash_yield - ra_borrow_cost - ra_costs
    report_ra = calculate_portfolio_risk_report(pd.Series(ret_ra))
    
    # 4. Optimized Hybrid Strategy
    favorable_regime = regime_labels_shifted.isin([
        "Bullish Low Volatility",
        "Bullish High Volatility",
        "Recovery Regime",
        "Sideways Low Volatility"
    ]).astype(float).values
    
    hybrid_signal = np.where((ema_bullish == 1.0) & (favorable_regime == 1.0), 2.2, 0.0)
    hybrid_diff = np.abs(np.diff(hybrid_signal, prepend=0.0))
    hybrid_costs = hybrid_diff * total_cost_rate
    hybrid_cash_frac = np.maximum(0.0, 1.0 - np.abs(hybrid_signal))
    hybrid_borrow_frac = np.maximum(0.0, np.abs(hybrid_signal) - 1.0)
    
    ret_hybrid = (hybrid_signal * asset_ret) + (hybrid_cash_frac * cash_rate_daily) - (hybrid_borrow_frac * borrow_rate_daily) - hybrid_costs
    report_hybrid = calculate_portfolio_risk_report(pd.Series(ret_hybrid))
    
    print("\n--- Aggressive Leveraged Results ---")
    results = [
        {"Strategy": "Buy And Hold", "CAGR": report_bh["CAGR"], "Sharpe": report_bh["Sharpe_Ratio"], "Max DD": report_bh["Max_Drawdown"]},
        {"Strategy": "Ema Crossover", "CAGR": report_ema["CAGR"], "Sharpe": report_ema["Sharpe_Ratio"], "Max DD": report_ema["Max_Drawdown"]},
        {"Strategy": "Regime Aware", "CAGR": report_ra["CAGR"], "Sharpe": report_ra["Sharpe_Ratio"], "Max DD": report_ra["Max_Drawdown"]},
        {"Strategy": "Hybrid", "CAGR": report_hybrid["CAGR"], "Sharpe": report_hybrid["Sharpe_Ratio"], "Max DD": report_hybrid["Max_Drawdown"]}
    ]
    for r in results:
        print(f"{r['Strategy']:<15} | CAGR: {r['CAGR']:.2%} | Sharpe: {r['Sharpe']:.3f} | Max DD: {r['Max DD']:.2%}")

if __name__ == "__main__":
    main()
