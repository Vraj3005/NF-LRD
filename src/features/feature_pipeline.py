import logging
import os

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from src.data.fetch_data import load_settings
from src.features.technical_indicators import (
    compute_momentum,
    compute_returns,
    compute_trends,
)
from src.features.volatility_features import (
    compute_atr_features,
    compute_ewma_volatility,
    compute_garman_klass_volatility,
    compute_parkinson_volatility,
    compute_realized_vol_ratio,
    compute_rolling_volatilities,
    compute_volatility_percentile,
)

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """Registry to keep track of available features and their logical group labels."""

    GROUPS = {
        "returns": "Returns Indicators",
        "volatility": "Volatility & Range Measures",
        "trend": "Trend Distances & Crossover States",
        "momentum": "Momentum Oscillators",
        "market_structure": "Market Structure & Wicks/Gaps",
        "statistical_complexity": "Statistical Moments & Information Entropy",
        "external": "External Macro Drivers",
    }

    def __init__(self):
        self._registry = {}

    def register(self, feature_name: str, group_name: str):
        if group_name not in self.GROUPS:
            raise ValueError(
                f"Invalid group '{group_name}'. Supported groups: {list(self.GROUPS.keys())}"
            )
        self._registry[feature_name] = group_name

    def get_group(self, feature_name: str) -> str:
        return self._registry.get(feature_name, "unknown")

    def get_features_by_group(self, group_name: str) -> list[str]:
        return [feat for feat, grp in self._registry.items() if grp == group_name]

    def get_registry(self) -> dict[str, str]:
        return self._registry.copy()


# Global Feature Registry instance
feature_registry = FeatureRegistry()


class StandardizationPipeline:
    """Standardization pipeline wrapper that fits only on training data to prevent lookahead leakage."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.columns_ = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        # Exclude date and raw columns from scaling
        cols_to_scale = [
            c for c in X.columns if c != "date" and not c.startswith("raw_")
        ]
        self.columns_ = cols_to_scale

        # Compute mean and standard deviation manually to ensure float64 precision and stability
        sub_df = X[cols_to_scale].replace([np.inf, -np.inf], np.nan)
        self.mean_ = sub_df.mean()
        self.scale_ = sub_df.std(ddof=0)
        self.scale_[self.scale_ < 1e-10] = 1.0  # Avoid division by zero
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("StandardizationPipeline is not fitted.")

        X_clean = X.copy()
        # Apply scaling
        for col in self.columns_:
            if col in X_clean.columns:
                mean_val = self.mean_[col]
                scale_val = self.scale_[col]
                # Clip extreme infinite outliers to NaN first
                col_data = X_clean[col].replace([np.inf, -np.inf], np.nan)
                X_clean[col] = (col_data - mean_val) / scale_val

        return X_clean

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


def compute_market_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes candlestick structure and trend strength metrics.
    Includes: wicks percentage, gap percentage, rolling drawdown,
    and Kaufman's Efficiency Ratio for trend strength.
    """
    feats = pd.DataFrame(index=df.index)
    open_val = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]

    eps = 1e-10
    candle_range = high - low + eps

    # 1. Candlestick geometry
    feats["candle_body_pct"] = (close - open_val).abs() / candle_range
    feats["upper_wick_pct"] = (high - np.maximum(open_val, close)) / candle_range
    feats["lower_wick_pct"] = (np.minimum(open_val, close) - low) / candle_range

    # 2. Gap percentage
    feats["gap_pct"] = (open_val - close.shift(1)) / (close.shift(1) + eps)

    # 3. Drawdowns (63-day rolling peak)
    rolling_peak = close.rolling(window=63, min_periods=1).max()
    feats["drawdown"] = (close - rolling_peak) / (rolling_peak + eps)
    feats["max_drawdown_63d"] = (
        feats["drawdown"].rolling(window=63, min_periods=1).min()
    )

    # 4. Kaufman's Efficiency Ratio (ER) as Trend Strength Score (14-day window)
    net_change = (close - close.shift(14)).abs()
    sum_abs_changes = close.diff().abs().rolling(window=14).sum()
    feats["trend_strength_score"] = net_change / (sum_abs_changes + eps)

    # 5. Rolling high-low max/min range distance
    rolling_max = high.rolling(window=21).max()
    rolling_min = low.rolling(window=21).min()
    feats["max_min_dist_21d"] = (rolling_max - rolling_min) / (close + eps)

    return feats


def compute_rolling_hurst(series: pd.Series, window: int = 126) -> pd.Series:
    """
    Approximates the rolling Hurst Exponent using a log-log regression of standard deviations
    of lag differences.
    """
    lags = [2, 4, 8, 16]
    log_lags = np.log(lags)

    def get_hurst(window_arr):
        if len(window_arr) < window:
            return np.nan
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


def compute_rolling_entropy(
    series: pd.Series, window: int = 63, bins: int = 10
) -> pd.Series:
    """
    Calculates the rolling Shannon entropy of returns to measure complexity/uncertainty.
    """

    def get_entropy(window_arr):
        if len(window_arr) < window:
            return np.nan
        window_arr = window_arr[~np.isnan(window_arr)]
        if len(window_arr) == 0:
            return np.nan
        counts, _ = np.histogram(window_arr, bins=bins)
        probs = counts / (counts.sum() + 1e-10)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    return series.rolling(window=window).apply(get_entropy, raw=True)


def compute_complexity_features(
    df: pd.DataFrame, stats_w: int, hurst_w: int, entropy_w: int, bins: int
) -> pd.DataFrame:
    """
    Computes statistical and complexity features: rolling skewness, kurtosis,
    Sharpe ratio, Sortino ratio, rolling Hurst exponent, and rolling entropy.
    """
    feats = pd.DataFrame(index=df.index)
    close = df["close"]
    log_ret = np.log(close / close.shift(1))

    # 1. Rolling Skewness and Kurtosis
    feats["skewness"] = log_ret.rolling(window=stats_w).skew()
    feats["kurtosis"] = log_ret.rolling(window=stats_w).kurt()

    # 2. Z-Score of Returns
    mean_ret = log_ret.rolling(window=stats_w).mean()
    std_ret = log_ret.rolling(window=stats_w).std()
    feats["z_score_ret"] = (log_ret - mean_ret) / (std_ret + 1e-10)

    # 3. Rolling Sharpe Ratio
    feats["sharpe_ratio"] = (mean_ret / (std_ret + 1e-10)) * np.sqrt(252)

    # 4. Rolling Sortino Ratio
    downside_returns = log_ret.copy()
    downside_returns[downside_returns > 0] = 0.0
    downside_std = downside_returns.rolling(window=stats_w).std()
    feats["sortino_ratio"] = (mean_ret / (downside_std + 1e-10)) * np.sqrt(252)

    # 5. Hurst Exponent
    feats["hurst_exponent"] = compute_rolling_hurst(close, window=hurst_w)

    # 6. Shannon Entropy
    feats["entropy"] = compute_rolling_entropy(log_ret, window=entropy_w, bins=bins)

    return feats


def fetch_optional_external_features(
    start_date: str, end_date: str, external_tickers: dict[str, str]
) -> pd.DataFrame:
    """
    Downloads optional external market drivers on a best-effort basis.
    """
    logger.info(
        "Attempting to download external features (S&P 500, USD/INR, Crude Oil)..."
    )
    external_dfs = []

    for name, ticker in external_tickers.items():
        try:
            logger.info(f"Fetching {name} data using ticker {ticker}...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                logger.warning(f"No data returned for external ticker {ticker}.")
                continue

            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df["Date"] = pd.to_datetime(df["Date"])
            df_subset = (
                df[["Date", "Close"]]
                .copy()
                .rename(columns={"Date": "date", "Close": f"ext_{name}"})
            )

            df_subset[f"ext_{name}"] = pd.to_numeric(
                df_subset[f"ext_{name}"].astype(str).str.replace(",", ""),
                errors="coerce",
            )

            df_subset[f"ext_{name}_ret"] = np.log(
                df_subset[f"ext_{name}"] / df_subset[f"ext_{name}"].shift(1)
            )

            df_subset = df_subset[["date", f"ext_{name}_ret"]]
            external_dfs.append(df_subset)
            logger.info(f"Successfully processed return for {name}.")

        except Exception as e:
            logger.warning(f"Skipping external asset {name} due to fetch error: {e}")

    if not external_dfs:
        return pd.DataFrame()

    merged_ext = external_dfs[0]
    for next_df in external_dfs[1:]:
        merged_ext = pd.merge(merged_ext, next_df, on="date", how="outer")

    return merged_ext


def select_features(
    df: pd.DataFrame, var_thresh: float, corr_thresh: float
) -> tuple[pd.DataFrame, list[str]]:
    """
    Applies low-variance and high-correlation filters to select features.
    """
    logger.info("Applying feature selection filters...")
    dropped_cols = []

    # 1. Low Variance Filter
    variances = df.var()
    low_var_cols = variances[variances < var_thresh].index.tolist()
    if low_var_cols:
        logger.info(
            f"Dropping {len(low_var_cols)} features due to low variance (< {var_thresh}): {low_var_cols}"
        )
        df = df.drop(columns=low_var_cols)
        dropped_cols.extend(low_var_cols)

    # 2. Correlation Filter (dropping collinear variables)
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    collinear_pairs = []
    for col in upper_tri.columns:
        high_corr_indices = upper_tri.index[upper_tri[col] > corr_thresh].tolist()
        for idx in high_corr_indices:
            collinear_pairs.append((idx, col))

    if collinear_pairs:
        mean_corrs = corr_matrix.mean()
        to_drop = set()

        for featA, featB in collinear_pairs:
            if featA in to_drop or featB in to_drop:
                continue
            if mean_corrs[featA] > mean_corrs[featB]:
                to_drop.add(featA)
            else:
                to_drop.add(featB)

        to_drop_list = list(to_drop)
        logger.info(
            f"Dropping {len(to_drop_list)} collinear features (> {corr_thresh} correlation): {to_drop_list}"
        )
        df = df.drop(columns=to_drop_list)
        dropped_cols.extend(to_drop_list)

    return df, dropped_cols


def run_feature_engineering_pipeline(config_path: str = "config/settings.yaml") -> str:
    """
    Orchestrates the feature extraction pipeline. Computes, registers, and scales
    reproducible financial features.
    """
    settings = load_settings(config_path)
    proc_dir = settings["data"].get("processed_dir", "data/processed")
    cleaned_parquet = os.path.join(proc_dir, "nifty_cleaned.parquet")

    if not os.path.exists(cleaned_parquet):
        raise FileNotFoundError(f"Cleaned dataset not found. Path: {cleaned_parquet}")

    logger.info(f"Loading cleaned Nifty 50 dataset: {cleaned_parquet}")
    df = pd.read_parquet(cleaned_parquet)
    df["date"] = pd.to_datetime(df["date"])

    nifty_meta = df[["date", "open", "high", "low", "close", "volume"]].copy()

    f_settings = settings.get("features", {})
    vol_windows = f_settings.get("volatility_windows", [5, 10, 21, 63])
    sma_windows = f_settings.get("sma_windows", [20, 50, 100, 200])
    ema_windows = f_settings.get("ema_windows", [9, 21, 50, 200])
    stats_w = f_settings.get("stats_window", 21)
    hurst_w = f_settings.get("hurst_window", 126)
    entropy_w = f_settings.get("entropy_window", 63)
    entropy_bins = f_settings.get("entropy_bins", 10)
    missing_value_policy = f_settings.get("missing_value_policy", "keep")

    # --- 1. Compute Returns ---
    df_ret = compute_returns(df)
    for col in df_ret.columns:
        feature_registry.register(col, "returns")

    # --- 2. Compute Volatility ---
    df_vol = compute_rolling_volatilities(df, vol_windows)
    df_vol["vol_parkinson_21d"] = compute_parkinson_volatility(df, window=21)
    df_vol["vol_garman_klass_21d"] = compute_garman_klass_volatility(df, window=21)
    df_vol["vol_ewma_21"] = compute_ewma_volatility(df, span=21)
    df_vol["vol_regime_pct_21d"] = compute_volatility_percentile(
        df_vol["vol_close_21d"], window=252
    )

    df_atr = compute_atr_features(df, period=14)
    df_vol_all = pd.concat([df_vol, df_atr], axis=1)
    for col in df_vol_all.columns:
        feature_registry.register(col, "volatility")

    # --- 3. Compute Trends ---
    df_trend = compute_trends(df, sma_windows, ema_windows)
    for col in df_trend.columns:
        feature_registry.register(col, "trend")

    # --- 4. Compute Momentum ---
    df_mom = compute_momentum(
        df,
        rsi_period=f_settings.get("rsi_period", 14),
        macd_fast=f_settings.get("macd_fast", 12),
        macd_slow=f_settings.get("macd_slow", 26),
        macd_signal=f_settings.get("macd_signal", 9),
        stoch_k=f_settings.get("stochastic_k", 14),
        stoch_d=f_settings.get("stochastic_d", 3),
        williams_r_period=f_settings.get("williams_r_period", 14),
        roc_period=f_settings.get("roc_period", 10),
    )
    for col in df_mom.columns:
        feature_registry.register(col, "momentum")

    # --- 5. Compute Market Structure ---
    df_struct = compute_market_structure_features(df)
    df_struct["vol_ratio_5_21"] = compute_realized_vol_ratio(df, short_w=5, long_w=21)
    for col in df_struct.columns:
        feature_registry.register(col, "market_structure")

    # --- 6. Compute Complexity & Statistical features ---
    df_complex = compute_complexity_features(
        df, stats_w, hurst_w, entropy_w, entropy_bins
    )
    for col in df_complex.columns:
        feature_registry.register(col, "statistical_complexity")

    # Combine Nifty-based features
    features_df = pd.concat(
        [df_ret, df_vol_all, df_trend, df_mom, df_struct, df_complex], axis=1
    )
    features_df["date"] = df["date"]

    # --- 7. Ingest External Features (Optional & Fail-safe) ---
    ext_tickers = f_settings.get("external_tickers", {})
    ext_cols = []
    if ext_tickers:
        start_dt = df["date"].min().strftime("%Y-%m-%d")
        end_dt = df["date"].max().strftime("%Y-%m-%d")
        ext_df = fetch_optional_external_features(start_dt, end_dt, ext_tickers)
        if not ext_df.empty:
            features_df = pd.merge(features_df, ext_df, on="date", how="left")
            ext_cols = [c for c in ext_df.columns if c != "date"]
            for col in ext_cols:
                feature_registry.register(col, "external")
            features_df[ext_cols] = features_df[ext_cols].ffill().bfill()

    # Include India VIX if it is already present in cleaned data
    if "vix_close" in df.columns:
        vix_ret_col = "vix_close_ret"
        features_df[vix_ret_col] = np.log(df["vix_close"] / df["vix_close"].shift(1))
        features_df[vix_ret_col] = features_df[vix_ret_col].fillna(0.0)
        feature_registry.register(vix_ret_col, "external")
        ext_cols.append(vix_ret_col)

    # Separate Date and raw metrics
    align_dates = features_df["date"]
    features_only = features_df.drop(columns=["date"])

    # Handle Missing Values based on policy
    # We always need to scale the features. Scaling cannot handle NaNs during fitting.
    # Therefore, we fit the scaler on non-NaN rows, but we record NaNs in the final parquet if policy is 'keep'!
    valid_mask = features_only.notna().all(axis=1)
    features_clean = features_only[valid_mask].copy()

    # Feature Selection (low variance, correlation pruning)
    var_thresh = float(f_settings.get("variance_threshold", 1e-4))
    corr_thresh = float(f_settings.get("correlation_threshold", 0.90))
    prune_correlation = f_settings.get("prune_correlation", True)

    if prune_correlation:
        features_selected, dropped = select_features(
            features_clean, var_thresh, corr_thresh
        )
    else:
        # Keep all features (no correlation pruning, just low variance)
        variances = features_clean.var()
        low_var_cols = variances[variances < var_thresh].index.tolist()
        features_selected = features_clean.drop(columns=low_var_cols)
        dropped = low_var_cols

    # Standardize Features using StandardizationPipeline
    logger.info("Fitting StandardizationPipeline on selected features...")
    scaler = StandardizationPipeline()
    scaler.fit(features_selected)

    # Save scaler object to models/saved/
    scaler_path = f_settings.get("scaler_joblib_path", "models/saved/scaler.joblib")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(
        f"Fitted StandardizationPipeline saved to: {os.path.abspath(scaler_path)}"
    )

    # Construct feature matrix output
    # If missing_value_policy is "keep", we keep the entire length (including NaN warmup rows)
    meta_cols = ["open", "high", "low", "close", "volume"]
    if missing_value_policy == "keep":
        # Standardize the entire feature matrix (rows with NaNs will keep their NaNs after transformation)
        scaled_df = scaler.transform(features_only[features_selected.columns])
        scaled_df.insert(0, "date", align_dates)
        # Add back non-scaled variables matching the original dates length
        for col in meta_cols:
            scaled_df[f"raw_{col}"] = nifty_meta[col]
        kept_rows = len(scaled_df)

        # Build raw unscaled dataframe
        raw_df = features_only[features_selected.columns].copy()
        raw_df.insert(0, "date", align_dates)
        for col in meta_cols:
            raw_df[f"raw_{col}"] = nifty_meta[col]
    else:
        # Default behavior: drop NaN warmup rows
        scaled_values_clean = scaler.transform(features_selected)
        scaled_df = scaled_values_clean
        scaled_df.insert(
            0, "date", align_dates_clean=align_dates[valid_mask].reset_index(drop=True)
        )
        nifty_meta_clean = nifty_meta[valid_mask].reset_index(drop=True)
        for col in meta_cols:
            scaled_df[f"raw_{col}"] = nifty_meta_clean[col]
        kept_rows = len(scaled_df)

        # Build raw unscaled dataframe
        raw_df = features_selected.copy()
        raw_df.insert(
            0, "date", align_dates_clean=align_dates[valid_mask].reset_index(drop=True)
        )
        for col in meta_cols:
            raw_df[f"raw_{col}"] = nifty_meta_clean[col]

    logger.info(
        f"Feature matrix row policy '{missing_value_policy}' output shape: {scaled_df.shape}"
    )

    # Re-group final selected feature names into categories for metadata tracking
    selected_cols = set(features_selected.columns)
    final_feature_groups = {}
    for feat, group in feature_registry.get_registry().items():
        if feat in selected_cols:
            final_feature_groups.setdefault(group, []).append(feat)

    # Save feature parquet
    parquet_path = f_settings.get(
        "feature_parquet_path", "data/processed/features.parquet"
    )
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    scaled_df.to_parquet(parquet_path, index=False)
    logger.info(
        f"Standardized feature matrix saved to: {os.path.abspath(parquet_path)}"
    )

    # Save unscaled raw feature parquet for leakage-free validation
    raw_parquet_path = parquet_path.replace(".parquet", "_raw.parquet")
    raw_df.to_parquet(raw_parquet_path, index=False)
    logger.info(
        f"Unscaled raw feature matrix saved to: {os.path.abspath(raw_parquet_path)}"
    )

    # Save feature metadata json or yaml
    metadata_path = f_settings.get(
        "metadata_yaml_path", "data/processed/feature_metadata.yaml"
    )
    metadata_content = {
        "dataset_info": {
            "total_records": kept_rows,
            "start_date": align_dates.min().strftime("%Y-%m-%d"),
            "end_date": align_dates.max().strftime("%Y-%m-%d"),
        },
        "dropped_features": dropped,
        "selected_features_count": len(features_selected.columns),
        "feature_categories": final_feature_groups,
    }

    if YAML_AVAILABLE:
        with open(metadata_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(metadata_content, f, default_flow_style=False)
        logger.info(
            f"Feature metadata report saved to: {os.path.abspath(metadata_path)}"
        )
    else:
        json_path = metadata_path.replace(".yaml", ".json")
        import json

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata_content, f, indent=4)
        logger.info(
            f"Feature metadata report saved as JSON: {os.path.abspath(json_path)}"
        )

    return os.path.abspath(parquet_path)
