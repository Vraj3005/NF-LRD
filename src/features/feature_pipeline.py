import os
import yaml
import logging
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler

from src.features.technical_indicators import compute_returns, compute_trends, compute_momentum
from src.features.volatility_features import compute_rolling_volatilities, compute_parkinson_volatility, compute_garman_klass_volatility, compute_atr_features
from src.data.fetch_data import load_settings

logger = logging.getLogger(__name__)

def compute_market_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes candlestick structure and trend strength metrics.
    Includes: wicks percentage, gap percentage, rolling drawdown,
    and Kaufman's Efficiency Ratio for trend strength.
    """
    feats = pd.DataFrame(index=df.index)
    open_val = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    
    eps = 1e-10
    candle_range = high - low + eps
    
    # 1. Candlestick geometry
    feats['candle_body_pct'] = (close - open_val).abs() / candle_range
    feats['upper_wick_pct'] = (high - np.maximum(open_val, close)) / candle_range
    feats['lower_wick_pct'] = (np.minimum(open_val, close) - low) / candle_range
    
    # 2. Gap percentage
    feats['gap_pct'] = (open_val - close.shift(1)) / (close.shift(1) + eps)
    
    # 3. Drawdowns (63-day rolling peak)
    rolling_peak = close.rolling(window=63, min_periods=1).max()
    feats['drawdown'] = (close - rolling_peak) / (rolling_peak + eps)
    feats['max_drawdown_63d'] = feats['drawdown'].rolling(window=63, min_periods=1).min()
    
    # 4. Kaufman's Efficiency Ratio (ER) as Trend Strength Score (14-day window)
    net_change = (close - close.shift(14)).abs()
    sum_abs_changes = close.diff().abs().rolling(window=14).sum()
    feats['trend_strength_score'] = net_change / (sum_abs_changes + eps)
    
    return feats

def compute_rolling_hurst(series: pd.Series, window: int = 126) -> pd.Series:
    """
    Approximates the rolling Hurst Exponent using a log-log regression of standard deviations
    of lag differences. A Hurst Exponent H > 0.5 indicates trending, H < 0.5 indicates
    mean-reverting, and H = 0.5 indicates a random walk.
    """
    lags = [2, 4, 8, 16]
    log_lags = np.log(lags)
    
    def get_hurst(window_arr):
        if len(window_arr) < window:
            return np.nan
        # Convert array back to pd.Series for shifting
        s = pd.Series(window_arr)
        stdevs = []
        for lag in lags:
            diff = s - s.shift(lag)
            stdevs.append(diff.std(ddof=1))
            
        if any(np.isnan(stdevs)) or any(std <= 0 for std in stdevs):
            return np.nan
            
        # Slope of log(std) vs log(lag)
        slope, _ = np.polyfit(log_lags, np.log(stdevs), 1)
        return slope
        
    return series.rolling(window=window).apply(get_hurst, raw=True)

def compute_rolling_entropy(series: pd.Series, window: int = 63, bins: int = 10) -> pd.Series:
    """
    Calculates the rolling Shannon entropy of returns to measure complexity/uncertainty.
    """
    def get_entropy(window_arr):
        if len(window_arr) < window:
            return np.nan
        # Filter out NaNs
        window_arr = window_arr[~np.isnan(window_arr)]
        if len(window_arr) == 0:
            return np.nan
        counts, _ = np.histogram(window_arr, bins=bins)
        probs = counts / (counts.sum() + 1e-10)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))
        
    return series.rolling(window=window).apply(get_entropy, raw=True)

def compute_complexity_features(df: pd.DataFrame, stats_w: int, hurst_w: int, entropy_w: int, bins: int) -> pd.DataFrame:
    """
    Computes statistical and complexity features: rolling skewness, kurtosis,
    Sharpe ratio, Sortino ratio, rolling Hurst exponent, and rolling entropy.
    """
    feats = pd.DataFrame(index=df.index)
    close = df['close']
    log_ret = np.log(close / close.shift(1))
    
    # 1. Rolling Skewness and Kurtosis
    feats['skewness'] = log_ret.rolling(window=stats_w).skew()
    feats['kurtosis'] = log_ret.rolling(window=stats_w).kurt()
    
    # 2. Z-Score of Returns
    mean_ret = log_ret.rolling(window=stats_w).mean()
    std_ret = log_ret.rolling(window=stats_w).std()
    feats['z_score_ret'] = (log_ret - mean_ret) / (std_ret + 1e-10)
    
    # 3. Rolling Sharpe Ratio (annualized, assuming risk-free rate = 0 for simplicity)
    feats['sharpe_ratio'] = (mean_ret / (std_ret + 1e-10)) * np.sqrt(252)
    
    # 4. Rolling Sortino Ratio (annualized)
    downside_returns = log_ret.copy()
    downside_returns[downside_returns > 0] = 0.0
    downside_std = downside_returns.rolling(window=stats_w).std()
    feats['sortino_ratio'] = (mean_ret / (downside_std + 1e-10)) * np.sqrt(252)
    
    # 5. Hurst Exponent Approximation
    feats['hurst_exponent'] = compute_rolling_hurst(close, window=hurst_w)
    
    # 6. Shannon Entropy of returns
    feats['entropy'] = compute_rolling_entropy(log_ret, window=entropy_w, bins=bins)
    
    return feats

def fetch_optional_external_features(
    start_date: str, 
    end_date: str, 
    external_tickers: Dict[str, str]
) -> pd.DataFrame:
    """
    Downloads optional external market drivers on a best-effort basis.
    """
    logger.info("Attempting to download external features (S&P 500, USD/INR, Crude Oil)...")
    external_dfs = []
    
    for name, ticker in external_tickers.items():
        try:
            logger.info(f"Fetching {name} data using ticker {ticker}...")
            # We fetch daily Close
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                logger.warning(f"No data returned for external ticker {ticker}.")
                continue
                
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
                
            # Keep date and close, rename close to ext_{name}_close
            df['Date'] = pd.to_datetime(df['Date'])
            df_subset = df[['Date', 'Close']].copy().rename(columns={'Date': 'date', 'Close': f'ext_{name}'})
            
            # Convert to float and clean commas
            df_subset[f'ext_{name}'] = pd.to_numeric(
                df_subset[f'ext_{name}'].astype(str).str.replace(',', ''), 
                errors='coerce'
            )
            
            # Calculate daily log returns for the asset (scale-invariant)
            df_subset[f'ext_{name}_ret'] = np.log(df_subset[f'ext_{name}'] / df_subset[f'ext_{name}'].shift(1))
            
            # Keep only date and return columns
            df_subset = df_subset[['date', f'ext_{name}_ret']]
            external_dfs.append(df_subset)
            logger.info(f"Successfully processed return for {name}.")
            
        except Exception as e:
            logger.warning(f"Skipping external asset {name} due to fetch error: {e}")
            
    if not external_dfs:
        return pd.DataFrame()
        
    # Merge all external features
    merged_ext = external_dfs[0]
    for next_df in external_dfs[1:]:
        merged_ext = pd.merge(merged_ext, next_df, on='date', how='outer')
        
    return merged_ext

def select_features(
    df: pd.DataFrame, 
    var_thresh: float, 
    corr_thresh: float
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Applies low-variance and high-correlation filters to select features.
    
    Args:
        df: pd.DataFrame with all features (index is datetime or date).
        var_thresh: float, variance threshold.
        corr_thresh: float, correlation threshold (absolute value).
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: (selected_dataframe, list of dropped columns)
    """
    logger.info("Applying feature selection filters...")
    dropped_cols = []
    
    # 1. Low Variance Filter
    variances = df.var()
    low_var_cols = variances[variances < var_thresh].index.tolist()
    if low_var_cols:
        logger.info(f"Dropping {len(low_var_cols)} features due to low variance (< {var_thresh}): {low_var_cols}")
        df = df.drop(columns=low_var_cols)
        dropped_cols.extend(low_var_cols)
        
    # 2. Correlation Filter (dropping collinear variables)
    corr_matrix = df.corr().abs()
    
    # Find index pairs where correlation exceeds threshold (excluding self-correlation)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    collinear_pairs = []
    for col in upper_tri.columns:
        high_corr_indices = upper_tri.index[upper_tri[col] > corr_thresh].tolist()
        for idx in high_corr_indices:
            collinear_pairs.append((idx, col))
            
    if collinear_pairs:
        # We calculate the average correlation of each feature with all other features
        mean_corrs = corr_matrix.mean()
        to_drop = set()
        
        for featA, featB in collinear_pairs:
            if featA in to_drop or featB in to_drop:
                continue
            # Drop the one with higher average correlation to reduce collinearity overall
            if mean_corrs[featA] > mean_corrs[featB]:
                to_drop.add(featA)
            else:
                to_drop.add(featB)
                
        to_drop_list = list(to_drop)
        logger.info(f"Dropping {len(to_drop_list)} collinear features (> {corr_thresh} correlation): {to_drop_list}")
        df = df.drop(columns=to_drop_list)
        dropped_cols.extend(to_drop_list)
        
    return df, dropped_cols

def run_feature_engineering_pipeline(config_path: str = "config/settings.yaml") -> str:
    """
    Loads cleaned parquet data, builds features, pulls external data,
    filters collinearity/low variance, standardizes, saves the scaler,
    saves the final matrix and feature metadata report.
    """
    settings = load_settings(config_path)
    proc_dir = settings["data"].get("processed_dir", "data/processed")
    cleaned_parquet = os.path.join(proc_dir, "nifty_cleaned.parquet")
    
    if not os.path.exists(cleaned_parquet):
        raise FileNotFoundError(f"Cleaned dataset not found. Please run data pipeline first. Path: {cleaned_parquet}")
        
    logger.info(f"Loading cleaned Nifty 50 dataset: {cleaned_parquet}")
    df = pd.read_parquet(cleaned_parquet)
    df['date'] = pd.to_datetime(df['date'])
    
    # Store dates and prices separately for mapping later
    nifty_meta = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # Fetch parameters
    f_settings = settings.get("features", {})
    vol_windows = f_settings.get("volatility_windows", [5, 10, 21, 63])
    sma_windows = f_settings.get("sma_windows", [20, 50, 100, 200])
    ema_windows = f_settings.get("ema_windows", [9, 21, 50, 200])
    stats_w = f_settings.get("stats_window", 21)
    hurst_w = f_settings.get("hurst_window", 126)
    entropy_w = f_settings.get("entropy_window", 63)
    entropy_bins = f_settings.get("entropy_bins", 10)
    
    # Dictionary to track feature names by their logical categories
    feature_groups: Dict[str, List[str]] = {
        "returns": [],
        "volatility": [],
        "trend": [],
        "momentum": [],
        "market_structure": [],
        "statistical_complexity": [],
        "external": []
    }
    
    # --- 1. Compute Returns ---
    logger.info("Computing returns features...")
    df_ret = compute_returns(df)
    feature_groups["returns"] = df_ret.columns.tolist()
    
    # --- 2. Compute Volatility ---
    logger.info("Computing volatility features...")
    df_vol = compute_rolling_volatilities(df, vol_windows)
    df_vol['vol_parkinson_21d'] = compute_parkinson_volatility(df, window=21)
    df_vol['vol_garman_klass_21d'] = compute_garman_klass_volatility(df, window=21)
    df_atr = compute_atr_features(df, period=14)
    df_vol_all = pd.concat([df_vol, df_atr], axis=1)
    feature_groups["volatility"] = df_vol_all.columns.tolist()
    
    # --- 3. Compute Trends ---
    logger.info("Computing trend features...")
    df_trend = compute_trends(df, sma_windows, ema_windows)
    feature_groups["trend"] = df_trend.columns.tolist()
    
    # --- 4. Compute Momentum ---
    logger.info("Computing momentum features...")
    df_mom = compute_momentum(
        df,
        rsi_period=f_settings.get("rsi_period", 14),
        macd_fast=f_settings.get("macd_fast", 12),
        macd_slow=f_settings.get("macd_slow", 26),
        macd_signal=f_settings.get("macd_signal", 9),
        stoch_k=f_settings.get("stochastic_k", 14),
        stoch_d=f_settings.get("stochastic_d", 3),
        williams_r_period=f_settings.get("williams_r_period", 14),
        roc_period=f_settings.get("roc_period", 10)
    )
    feature_groups["momentum"] = df_mom.columns.tolist()
    
    # --- 5. Compute Market Structure ---
    logger.info("Computing market structure features...")
    df_struct = compute_market_structure_features(df)
    feature_groups["market_structure"] = df_struct.columns.tolist()
    
    # --- 6. Compute Complexity & Statistical features ---
    logger.info("Computing complexity and statistical features...")
    df_complex = compute_complexity_features(df, stats_w, hurst_w, entropy_w, entropy_bins)
    feature_groups["statistical_complexity"] = df_complex.columns.tolist()
    
    # Combine Nifty-based features
    features_df = pd.concat([df_ret, df_vol_all, df_trend, df_mom, df_struct, df_complex], axis=1)
    
    # Set date as alignment key
    features_df['date'] = df['date']
    
    # --- 7. Ingest External Features (Optional & Fail-safe) ---
    ext_tickers = f_settings.get("external_tickers", {})
    if ext_tickers:
        start_dt = df['date'].min().strftime('%Y-%m-%d')
        end_dt = df['date'].max().strftime('%Y-%m-%d')
        ext_df = fetch_optional_external_features(start_dt, end_dt, ext_tickers)
        if not ext_df.empty:
            features_df = pd.merge(features_df, ext_df, on='date', how='left')
            # Categorize external features
            ext_cols = [c for c in ext_df.columns if c != 'date']
            feature_groups["external"] = ext_cols
            # Fill missing values for external metrics if calendars differ
            features_df[ext_cols] = features_df[ext_cols].ffill().bfill()
            
    # Include India VIX if it is already present in cleaned data
    if 'vix_close' in df.columns:
        features_df['vix_close_ret'] = np.log(df['vix_close'] / df['vix_close'].shift(1))
        features_df['vix_close_ret'] = features_df['vix_close_ret'].fillna(0.0)
        feature_groups["external"].append('vix_close_ret')

    # Drop the date column temporarily for validation and pruning
    align_dates = features_df['date']
    features_only = features_df.drop(columns=['date'])
    
    # Drop rows containing NaNs due to lookback periods (e.g. SMA 200 needs 200 rows)
    # This prevents scaler fitting and training crashes
    valid_mask = features_only.notna().all(axis=1)
    logger.info(f"Filtering NaN rows due to indicators lookback: Keeping {valid_mask.sum()} rows of {len(features_only)} total.")
    
    features_clean = features_only[valid_mask].copy()
    align_dates_clean = align_dates[valid_mask].reset_index(drop=True)
    nifty_meta_clean = nifty_meta[valid_mask].reset_index(drop=True)
    
    # Apply Feature Selection (low variance, correlation)
    var_thresh = float(f_settings.get("variance_threshold", 1e-4))
    corr_thresh = float(f_settings.get("correlation_threshold", 0.90))
    features_selected, dropped = select_features(features_clean, var_thresh, corr_thresh)
    
    # Re-group final selected feature names into categories for metadata tracking
    selected_cols = set(features_selected.columns)
    final_feature_groups: Dict[str, List[str]] = {}
    for group, cols in feature_groups.items():
        matched_cols = [c for c in cols if c in selected_cols]
        if matched_cols:
            final_feature_groups[group] = matched_cols
            
    # Standardize Features using StandardScaler
    logger.info("Fitting StandardScaler on selected features...")
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(features_selected)
    
    # Save the fitted scaler
    scaler_path = f_settings.get("scaler_joblib_path", "models/saved/scaler.joblib")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Fitted StandardScaler saved to: {os.path.abspath(scaler_path)}")
    
    # Re-assemble the final dataframe with date and target indexes
    scaled_df = pd.DataFrame(scaled_values, columns=features_selected.columns)
    scaled_df.insert(0, 'date', align_dates_clean)
    
    # Add back non-scaled variables for convenience in analysis/backtesting later
    # (like raw close, high, low, volume for charting and transaction simulation)
    meta_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in meta_cols:
        scaled_df[f'raw_{col}'] = nifty_meta_clean[col]
        
    # Save features Parquet
    parquet_path = f_settings.get("feature_parquet_path", "data/processed/features.parquet")
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    scaled_df.to_parquet(parquet_path, index=False)
    logger.info(f"Standardized feature matrix saved as Parquet: {os.path.abspath(parquet_path)}")
    
    # Write feature metadata yaml file
    metadata_path = f_settings.get("metadata_yaml_path", "data/processed/feature_metadata.yaml")
    metadata_content = {
        "dataset_info": {
            "total_records": len(scaled_df),
            "start_date": align_dates_clean.min().strftime('%Y-%m-%d'),
            "end_date": align_dates_clean.max().strftime('%Y-%m-%d')
        },
        "dropped_features": dropped,
        "selected_features_count": len(features_selected.columns),
        "feature_categories": final_feature_groups
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(metadata_content, f, default_flow_style=False)
        
    logger.info(f"Feature metadata report saved to: {os.path.abspath(metadata_path)}")
    return os.path.abspath(parquet_path)
