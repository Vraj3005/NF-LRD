import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy.stats import skew, kurtosis

logger = logging.getLogger(__name__)

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculates maximum drawdown of a returns series using a cumulative equity curve."""
    if len(returns) == 0:
        return 0.0
    # Add 1.0 to convert to multipliers
    cum_returns = (1.0 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / (running_max + 1e-10)
    max_dd = drawdowns.min()
    return float(max_dd) if not np.isnan(max_dd) else 0.0

def calculate_sortino_ratio(returns: pd.Series, annualized_mean: float) -> float:
    """Calculates Sortino ratio (annualized return over annualized downside deviation)."""
    if len(returns) == 0:
        return 0.0
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0.0] = 0.0
    downside_std = downside_returns.std() * np.sqrt(252)
    if downside_std > 0.0:
        return float(annualized_mean / downside_std)
    return 0.0

def compute_transition_matrix(states: np.ndarray, n_components: int) -> np.ndarray:
    """
    Computes empirical transition probability matrix from a sequence of states.
    P[i, j] is the probability of transitioning from state i to state j.
    """
    matrix = np.zeros((n_components, n_components))
    for t in range(len(states) - 1):
        i = states[t]
        j = states[t+1]
        matrix[i, j] += 1.0
        
    # Normalize rows to sum to 1.0
    row_sums = matrix.sum(axis=1)
    for i in range(n_components):
        if row_sums[i] > 0.0:
            matrix[i] = matrix[i] / row_sums[i]
        else:
            matrix[i] = np.zeros(n_components)
            matrix[i, i] = 1.0  # Absorb state if never exited
            
    return matrix

def assign_regime_labels(regime_stats: List[Dict[str, Any]]) -> List[str]:
    """
    Heuristically assigns human-readable labels to regimes based on their returns and volatilities.
    Supported labels:
    - Bullish Low Volatility
    - Bullish High Volatility
    - Bearish High Volatility
    - Sideways Low Volatility
    - Recovery Regime
    - Distribution / Risk-Off Regime
    """
    n_components = len(regime_stats)
    
    # Sort indices by annualized return
    sorted_by_ret = sorted(range(n_components), key=lambda i: regime_stats[i]["annualized_return"])
    
    # Sort indices by volatility
    sorted_by_vol = sorted(range(n_components), key=lambda i: regime_stats[i]["annualized_volatility"])
    
    labels = [""] * n_components
    
    # Simple heuristic mappings based on component counts
    if n_components == 2:
        # Bearish / High Vol vs Bullish / Low Vol
        idx_bear = sorted_by_ret[0]
        idx_bull = sorted_by_ret[1]
        labels[idx_bear] = "Bearish High Volatility" if regime_stats[idx_bear]["annualized_return"] < 0 else "Sideways Low Volatility"
        labels[idx_bull] = "Bullish Low Volatility"
        
    elif n_components == 3:
        # High, Mid, Low returns
        idx_low = sorted_by_ret[0]
        idx_mid = sorted_by_ret[1]
        idx_high = sorted_by_ret[2]
        
        # Low Return
        if regime_stats[idx_low]["annualized_return"] < -0.05:
            labels[idx_low] = "Bearish High Volatility"
        else:
            labels[idx_low] = "Distribution / Risk-Off Regime"
            
        # Mid Return
        if abs(regime_stats[idx_mid]["annualized_return"]) < 0.05:
            labels[idx_mid] = "Sideways Low Volatility"
        else:
            labels[idx_mid] = "Recovery Regime"
            
        # High Return
        # Check if high return has high vol (highest volatility of all 3 states)
        if sorted_by_vol.index(idx_high) == 2:
            labels[idx_high] = "Bullish High Volatility"
        else:
            labels[idx_high] = "Bullish Low Volatility"
            
    else:
        # General case (4 to 6 components): Use ranking rules
        assigned = set()
        
        # 1. Bearish High Volatility is the state with the lowest returns
        lowest_ret_idx = sorted_by_ret[0]
        labels[lowest_ret_idx] = "Bearish High Volatility"
        assigned.add(lowest_ret_idx)
        
        # 2. Bullish Low Volatility is the state with the highest Sharpe ratio
        sharpe_ranks = sorted(range(n_components), key=lambda i: regime_stats[i]["sharpe_ratio"], reverse=True)
        best_sharpe_idx = next(idx for idx in sharpe_ranks if idx not in assigned)
        labels[best_sharpe_idx] = "Bullish Low Volatility"
        assigned.add(best_sharpe_idx)
        
        # 3. Bullish High Volatility is the state with highest returns among remaining
        highest_ret_remain = next(idx for idx in reversed(sorted_by_ret) if idx not in assigned)
        labels[highest_ret_remain] = "Bullish High Volatility"
        assigned.add(highest_ret_remain)
        
        # 4. Sideways Low Volatility is the state with lowest volatility among remaining
        lowest_vol_remain = next(idx for idx in sorted_by_vol if idx not in assigned)
        labels[lowest_vol_remain] = "Sideways Low Volatility"
        assigned.add(lowest_vol_remain)
        
        # 5. Recovery or Distribution for the rest
        remaining = [idx for idx in range(n_components) if idx not in assigned]
        for idx in remaining:
            if regime_stats[idx]["annualized_return"] > 0:
                labels[idx] = "Recovery Regime"
            else:
                labels[idx] = "Distribution / Risk-Off Regime"
                
    return labels

def analyze_regimes(
    df: pd.DataFrame, 
    states: np.ndarray, 
    n_components: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates detailed statistics for each discovered market regime.
    Assumes df has 'raw_close' and 'ret_log' (or we compute returns from close).
    """
    # Align states sequence
    df_regime = df.copy()
    df_regime['regime_state'] = states
    
    # Calculate daily simple return if not present
    if 'ret_simple' not in df_regime.columns and 'raw_close' in df_regime.columns:
        df_regime['ret_simple'] = df_regime['raw_close'].pct_change(fill_method=None)
    elif 'ret_simple' not in df_regime.columns:
        # Fallback to reconstructing return from scaled feature or close
        df_regime['ret_simple'] = df_regime['close'].pct_change(fill_method=None)
        
    df_regime['ret_simple'] = df_regime['ret_simple'].fillna(0.0)
    
    # Calculate state transitions
    trans_matrix = compute_transition_matrix(states, n_components)
    
    # Run consecutive runs to compute average state durations
    consecutive_runs = {i: [] for i in range(n_components)}
    current_state = states[0]
    current_run = 1
    for t in range(1, len(states)):
        if states[t] == current_state:
            current_run += 1
        else:
            consecutive_runs[current_state].append(current_run)
            current_state = states[t]
            current_run = 1
    consecutive_runs[current_state].append(current_run)
    
    regime_summaries = []
    
    for i in range(n_components):
        subset = df_regime[df_regime['regime_state'] == i]
        ret = subset['ret_simple']
        
        # Calculate daily metrics
        mean_ret = ret.mean()
        std_ret = ret.std()
        
        # Annualized metrics
        ann_ret = mean_ret * 252.0
        ann_vol = std_ret * np.sqrt(252.0)
        
        # Sharpe ratio
        sharpe = (ann_ret / ann_vol) if ann_vol > 0.0 else 0.0
        # Sortino
        sortino = calculate_sortino_ratio(ret, ann_ret)
        # Max drawdown
        max_dd = calculate_max_drawdown(ret)
        
        # Win rate
        win_rate = (ret > 0.0).sum() / len(ret) if len(ret) > 0 else 0.0
        # Durations
        durations = consecutive_runs[i]
        avg_dur = float(np.mean(durations)) if len(durations) > 0 else 0.0
        
        regime_summaries.append({
            "regime_state": i,
            "daily_count": len(subset),
            "percentage_days": float(len(subset) / len(df_regime)),
            "average_daily_return": float(mean_ret),
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "average_duration_days": float(avg_dur),
            "worst_day_return": float(ret.min()) if len(ret) > 0 else 0.0,
            "best_day_return": float(ret.max()) if len(ret) > 0 else 0.0,
            "skewness": float(skew(ret)) if len(ret) > 2 else 0.0,
            "kurtosis": float(kurtosis(ret)) if len(ret) > 2 else 0.0,
            "prob_staying": float(trans_matrix[i, i])
        })
        
    # Heuristically label regimes
    labels = assign_regime_labels(regime_summaries)
    for idx, label in enumerate(labels):
        regime_summaries[idx]["regime_label"] = label
        
    summary_df = pd.DataFrame(regime_summaries)
    
    # Map labels back to dataframe
    label_map = {item["regime_state"]: item["regime_label"] for item in regime_summaries}
    df_regime["regime_label"] = df_regime["regime_state"].map(label_map)
    
    # Create structured transition matrix dataframe with labels
    trans_columns = [f"To: {labels[j]}" for j in range(n_components)]
    trans_index = [f"From: {labels[i]}" for i in range(n_components)]
    trans_df = pd.DataFrame(trans_matrix, columns=trans_columns, index=trans_index)
    
    return df_regime, summary_df, trans_df

def run_walk_forward_validation(
    features_df: pd.DataFrame,
    model_cls: Any,
    n_components: int,
    train_ratio: float = 0.70
) -> Dict[str, Any]:
    """
    Splits features into 70% training and 30% testing.
    Fits the model on train, predicts on test, and compares statistics
    to analyze out-of-sample regime stability.
    """
    logger.info("Initializing Walk-forward Validation (70/30 split)...")
    
    # Extract feature columns
    feature_cols = [c for c in features_df.columns if c != 'date' and not c.startswith('raw_')]
    
    X = features_df[feature_cols].values
    split_idx = int(len(X) * train_ratio)
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    
    # Fit model on training set
    model = model_cls(n_components=n_components, covariance_type="diag", random_state=42)
    model.fit(X_train)
    
    # Predict states
    train_states = model.predict(X_train)
    test_states = model.predict(X_test)
    
    # Check alignment: calculate mean returns in train vs test for each state
    # We map returns from the dataframe
    if 'raw_close' in features_df.columns:
        ret_series = features_df['raw_close'].pct_change(fill_method=None).fillna(0.0)
    else:
        ret_series = features_df['close'].pct_change(fill_method=None).fillna(0.0)
        
    ret_values = ret_series.values
    ret_train = ret_values[:split_idx]
    ret_test = ret_values[split_idx:]
    
    stability_report = []
    
    for i in range(n_components):
        train_mask = (train_states == i)
        test_mask = (test_states == i)
        
        train_count = train_mask.sum()
        test_count = test_mask.sum()
        
        train_mean = ret_train[train_mask].mean() * 252.0 if train_count > 0 else 0.0
        train_vol = ret_train[train_mask].std() * np.sqrt(252.0) if train_count > 0 else 0.0
        
        test_mean = ret_test[test_mask].mean() * 252.0 if test_count > 0 else 0.0
        test_vol = ret_test[test_mask].std() * np.sqrt(252.0) if test_count > 0 else 0.0
        
        stability_report.append({
            "regime_state": i,
            "train_days": int(train_count),
            "test_days": int(test_count),
            "train_annualized_return": float(train_mean),
            "test_annualized_return": float(test_mean),
            "train_annualized_vol": float(train_vol),
            "test_annualized_vol": float(test_vol),
            "return_stability_delta": float(abs(train_mean - test_mean)),
            "vol_stability_delta": float(abs(train_vol - test_vol))
        })
        
    # Overall stability score: average of return and volatility deltas (lower is more stable)
    avg_ret_delta = np.mean([item["return_stability_delta"] for item in stability_report])
    avg_vol_delta = np.mean([item["vol_stability_delta"] for item in stability_report])
    
    return {
        "regime_stability": stability_report,
        "average_return_delta": float(avg_ret_delta),
        "average_volatility_delta": float(avg_vol_delta),
        "is_stable_oos": bool(avg_ret_delta < 0.15 and avg_vol_delta < 0.10)
    }
