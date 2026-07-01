"""
Out-of-Sample and Walk-Forward Validation Engine for Latent Market Regime Platform.
Ensures zero-lookahead feature scaling, dynamic feature selection, and labeling alignment.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.analysis.regime_analysis import assign_regime_labels
from src.backtesting.backtester import VectorizedBacktester
from src.data.fetch_data import load_settings
from src.features.feature_pipeline import StandardizationPipeline
from src.models.hmm_model import GaussianHMM

logger = logging.getLogger(__name__)


def select_features_on_train(
    df: pd.DataFrame, var_thresh: float = 1e-4, corr_thresh: float = 0.90
) -> List[str]:
    """
    Applies low-variance and high-correlation filters solely on the training dataframe
    to select features without out-of-sample data leakages.
    """
    # 1. Low Variance Filter
    variances = df.var()
    low_var_cols = variances[variances < var_thresh].index.tolist()
    clean_cols = [c for c in df.columns if c not in low_var_cols]

    if not clean_cols:
        return list(df.columns)

    df_clean = df[clean_cols]

    # 2. Correlation Filter (collinear pruning)
    corr_matrix = df_clean.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    collinear_pairs = []
    for col in upper_tri.columns:
        high_corr_indices = upper_tri.index[upper_tri[col] > corr_thresh].tolist()
        for idx in high_corr_indices:
            collinear_pairs.append((idx, col))

    to_drop = set()
    if collinear_pairs:
        mean_corrs = corr_matrix.mean()
        for featA, featB in collinear_pairs:
            if featA in to_drop or featB in to_drop:
                continue
            if mean_corrs[featA] > mean_corrs[featB]:
                to_drop.add(featA)
            else:
                to_drop.add(featB)

    selected_cols = [c for c in clean_cols if c not in to_drop]
    return selected_cols


def align_regime_labels_dynamically(
    train_df: pd.DataFrame, train_states: np.ndarray, n_components: int
) -> Dict[int, str]:
    """
    Computes annualized return and volatility statistics of training states to assign
    consistent human-readable regime labels dynamically.
    """
    # Calculate simple return from raw_close (or close as fallback)
    if "raw_close" in train_df.columns:
        ret_series = train_df["raw_close"].pct_change(fill_method=None).fillna(0.0)
    else:
        ret_series = train_df["close"].pct_change(fill_method=None).fillna(0.0)

    ret_values = ret_series.values
    regime_stats = []

    for i in range(n_components):
        mask = train_states == i
        if mask.sum() == 0:
            regime_stats.append(
                {
                    "regime_state": i,
                    "annualized_return": 0.0,
                    "annualized_volatility": 0.0,
                }
            )
            continue

        r = ret_values[mask]
        ann_ret = float(r.mean() * 252.0)
        ann_vol = float(r.std() * np.sqrt(252.0))
        regime_stats.append(
            {
                "regime_state": i,
                "annualized_return": ann_ret,
                "annualized_volatility": ann_vol,
            }
        )

    labels = assign_regime_labels(regime_stats)
    return {i: labels[i] for i in range(n_components)}


def split_by_dates(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Helper to partition dataframe by date bounds."""
    mask = (df["date"] >= pd.to_datetime(start_date)) & (
        df["date"] <= pd.to_datetime(end_date)
    )
    return df[mask].copy()


def run_oos_partition_validation(
    df_raw: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
    n_components: int = 3,
    config_path: str = "config/settings.yaml",
) -> Dict[str, Any]:
    """
    Performs train/validation/test partitions with zero-leakage pipeline fits.
    Each subset is standardized and feature-selected based purely on training params.
    """
    logger.info("Initializing Out-of-Sample Partition Validation...")

    # 1. Create split copies
    train_df = split_by_dates(df_raw, train_start, train_end)
    val_df = split_by_dates(df_raw, val_start, val_end)
    test_df = split_by_dates(df_raw, test_start, test_end)

    if train_df.empty:
        raise ValueError(
            f"Training split is empty for dates: {train_start} to {train_end}"
        )

    # Get active feature list (columns excluding date/raw columns)
    active_cols = [
        c for c in df_raw.columns if c != "date" and not c.startswith("raw_")
    ]

    # 2. Dynamic Feature Selection on Train slice only
    train_clean_features = train_df[active_cols].dropna()
    selected_features = select_features_on_train(train_clean_features)
    logger.info(
        f"Selected {len(selected_features)} features based on training subset filters."
    )

    # 3. Fit Standardization Pipeline on Train slice only
    scaler = StandardizationPipeline()
    scaler.fit(train_df[selected_features])

    # 4. Transform datasets
    train_scaled = scaler.transform(train_df[selected_features])
    val_scaled = scaler.transform(val_df[selected_features])
    test_scaled = scaler.transform(test_df[selected_features])

    # Fit HMM on train only
    model = GaussianHMM(
        n_components=n_components, covariance_type="diag", random_state=42, n_iter=200
    )

    # Clean train NaNs
    valid_mask_train = ~train_scaled.isna().any(axis=1)
    X_train = train_scaled[valid_mask_train].values

    model.fit(X_train)

    # Predict states
    train_states_clean = model.predict(X_train)
    train_states = np.full(len(train_df), -1, dtype=int)
    train_states[valid_mask_train] = train_states_clean

    # Map dynamic aligned labels based on train returns/vols
    label_mapping = align_regime_labels_dynamically(
        train_df, train_states, n_components
    )
    label_mapping[-1] = "Warmup / Undefined"

    train_df["regime_state"] = train_states
    train_df["regime_label"] = train_df["regime_state"].map(label_mapping)

    # Validation predictions (using validation-scaled features)
    valid_mask_val = ~val_scaled.isna().any(axis=1)
    val_states = np.full(len(val_df), -1, dtype=int)
    if len(val_df) > 0 and valid_mask_val.any():
        val_states_clean = model.predict(val_scaled[valid_mask_val].values)
        val_states[valid_mask_val] = val_states_clean
    val_df["regime_state"] = val_states
    val_df["regime_label"] = val_df["regime_state"].map(label_mapping)

    # Test predictions (using test-scaled features)
    valid_mask_test = ~test_scaled.isna().any(axis=1)
    test_states = np.full(len(test_df), -1, dtype=int)
    if len(test_df) > 0 and valid_mask_test.any():
        test_states_clean = model.predict(test_scaled[valid_mask_test].values)
        test_states[valid_mask_test] = test_states_clean
    test_df["regime_state"] = test_states
    test_df["regime_label"] = test_df["regime_state"].map(label_mapping)

    # 5. Execute backtests on each partition separately using VectorizedBacktester
    settings = load_settings(config_path)
    bt_settings = settings.get("backtest", {})
    tc = bt_settings.get("transaction_cost_bps", 10.0)
    slip = bt_settings.get("slippage_bps", 5.0)
    backtester = VectorizedBacktester(transaction_cost_bps=tc, slippage_bps=slip)

    train_curves, train_summary = backtester.backtest_strategies(
        train_df, config_path=config_path, walk_forward=False
    )
    val_curves, val_summary = backtester.backtest_strategies(
        val_df, config_path=config_path, walk_forward=False
    )
    test_curves, test_summary = backtester.backtest_strategies(
        test_df, config_path=config_path, walk_forward=False
    )

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_summary": train_summary,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "train_curves": train_curves,
        "val_curves": val_curves,
        "test_curves": test_curves,
        "selected_features": selected_features,
        "label_mapping": label_mapping,
    }


def run_walk_forward_oos_validation(
    df_raw: pd.DataFrame,
    start_date: str,
    train_window_years: int = 4,
    test_window_months: int = 6,
    n_components: int = 3,
    config_path: str = "config/settings.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs a zero-lookahead expanding walk-forward validation.
    Fits feature selection, scaling, model parameters, and regime labeling
    strictly on the training window and projects onto the test window.
    Concatenates out-of-sample predictions.
    """
    logger.info("Initializing Zero-Lookahead Expanding Walk-Forward OOS Validation...")

    # Filter dates
    df_sorted = df_raw.sort_values("date").reset_index(drop=True)
    df_sorted = df_sorted[df_sorted["date"] >= pd.to_datetime(start_date)].reset_index(
        drop=True
    )

    train_days = int(train_window_years * 252)
    test_days = int(test_window_months * 21)

    total_len = len(df_sorted)
    if total_len <= train_days:
        raise ValueError(
            f"Dataset length {total_len} is smaller than training window {train_days} days."
        )

    oos_labels = np.full(total_len, "Warmup / Undefined", dtype=object)
    oos_states = np.full(total_len, -1, dtype=int)

    active_cols = [
        c for c in df_sorted.columns if c != "date" and not c.startswith("raw_")
    ]

    t = train_days
    while t < total_len:
        oos_end = min(t + test_days, total_len)

        # Slices
        train_slice = df_sorted.iloc[:t].copy()
        test_slice = df_sorted.iloc[t:oos_end].copy()

        # Dynamic feature selection on train_slice
        train_features_clean = train_slice[active_cols].dropna()
        selected_features = select_features_on_train(train_features_clean)

        # Fit scaler
        scaler = StandardizationPipeline()
        scaler.fit(train_slice[selected_features])

        # Transform
        train_scaled = scaler.transform(train_slice[selected_features])
        test_scaled = scaler.transform(test_slice[selected_features])

        # Fit HMM model
        valid_mask_train = ~train_scaled.isna().any(axis=1)
        X_train = train_scaled[valid_mask_train].values

        model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            random_state=42,
            n_iter=200,
        )
        model.fit(X_train)

        # Predict train states & dynamic label alignment
        train_states_clean = model.predict(X_train)
        train_states = np.full(len(train_slice), -1, dtype=int)
        train_states[valid_mask_train] = train_states_clean

        label_mapping = align_regime_labels_dynamically(
            train_slice, train_states, n_components
        )
        label_mapping[-1] = "Warmup / Undefined"

        # Predict test states (OOS)
        valid_mask_test = ~test_scaled.isna().any(axis=1)
        test_states = np.full(len(test_slice), -1, dtype=int)
        if len(test_slice) > 0 and valid_mask_test.any():
            test_states_clean = model.predict(test_scaled[valid_mask_test].values)
            test_states[valid_mask_test] = test_states_clean

        # Map labels
        for offset in range(len(test_slice)):
            idx = t + offset
            state_val = test_states[offset]
            oos_states[idx] = state_val
            oos_labels[idx] = label_mapping.get(state_val, "Warmup / Undefined")

        t = oos_end

    df_wf = df_sorted.copy()
    df_wf["regime_state"] = oos_states
    df_wf["regime_label"] = oos_labels

    # Backtest OOS walk-forward predictions
    settings = load_settings(config_path)
    bt_settings = settings.get("backtest", {})
    tc = bt_settings.get("transaction_cost_bps", 10.0)
    slip = bt_settings.get("slippage_bps", 5.0)
    backtester = VectorizedBacktester(transaction_cost_bps=tc, slippage_bps=slip)
    wf_curves, wf_summary = backtester.backtest_strategies(
        df_wf, config_path=config_path, walk_forward=False
    )

    return df_wf, wf_curves, wf_summary
