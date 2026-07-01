import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd

from src.data.fetch_data import load_settings
from src.data.validation import validate_market_data

logger = logging.getLogger(__name__)


def normalize_columns(
    df: pd.DataFrame, config_path: str = "config/settings.yaml"
) -> pd.DataFrame:
    """
    Normalizes dataframe column names to standard lowercase names using settings mapping.
    Also cleans commas or special characters from numeric values and converts them to floats.

    Args:
        df: pd.DataFrame raw loaded market data.
        config_path: Path to YAML configuration settings.

    Returns:
        pd.DataFrame: DataFrame with columns normalized and cleaned.
    """
    settings = load_settings(config_path)
    mappings = settings.get("csv_mappings", {})

    df_clean = df.copy()

    # Strip whitespace from string headers if any
    df_clean.columns = [str(col).strip() for col in df_clean.columns]

    column_rename_map = {}

    # Identify and rename target columns based on mappings list
    target_columns = ["date", "open", "high", "low", "close", "volume", "turnover"]
    for target in target_columns:
        mapping_list = mappings.get(f"{target}_columns", [])
        # Find which column in df_clean matches one of the values in the mapping list
        found_col = None
        for col in df_clean.columns:
            if col in mapping_list:
                found_col = col
                break
        if found_col:
            column_rename_map[found_col] = target

    df_clean = df_clean.rename(columns=column_rename_map)

    # Check if 'date' column was successfully matched
    if "date" not in df_clean.columns:
        # If 'Date' or 'date' was not matched, check if index is DatetimeIndex
        if isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean = df_clean.reset_index().rename(
                columns={"index": "date", "Date": "date"}
            )
        elif "date" not in df_clean.columns and df_clean.columns.size > 0:
            # Last resort: try to find column that looks like a date or close column
            logger.warning(
                "Could not find mapped date column, trying to guess by column type/name."
            )
            for col in df_clean.columns:
                if (
                    "date" in col.lower()
                    or "time" in col.lower()
                    or "record" in col.lower()
                ):
                    df_clean = df_clean.rename(columns={col: "date"})
                    break

    # If date is still not found, raise a ValueError
    if "date" not in df_clean.columns:
        raise ValueError(
            f"Could not locate a valid 'date' column in DataFrame. Available columns: {list(df_clean.columns)}"
        )

    # Convert Date column to datetime
    df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")

    # Drop rows where Date could not be parsed
    nat_dates = df_clean["date"].isna().sum()
    if nat_dates > 0:
        logger.warning(f"Dropping {nat_dates} rows where Date could not be parsed.")
        df_clean = df_clean.dropna(subset=["date"])

    # Standardize column types: clean numeric values (remove commas, spaces, currency symbols)
    numeric_cols = [
        col
        for col in ["open", "high", "low", "close", "volume", "turnover"]
        if col in df_clean.columns
    ]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            # Remove commas, rupees symbol or currency characters, and whitespace
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.strip()
            )
        # Convert to numeric, force invalid to NaN
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Drop rows with NaN in critical price columns (open, high, low, close)
    critical_cols = [
        col for col in ["open", "high", "low", "close"] if col in df_clean.columns
    ]
    if critical_cols:
        nan_rows = df_clean[critical_cols].isna().any(axis=1).sum()
        if nan_rows > 0:
            logger.warning(
                f"Dropping {nan_rows} rows with NaN in critical price columns: {critical_cols}"
            )
            df_clean = df_clean.dropna(subset=critical_cols)

    return df_clean


def merge_and_align_vix(
    nifty_df: pd.DataFrame, vix_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Left-joins VIX data on NIFTY data and forward-fills missing VIX readings.

    Args:
        nifty_df: Normalized NIFTY DataFrame.
        vix_df: Optional Normalized VIX DataFrame.

    Returns:
        pd.DataFrame: Unified DataFrame with both assets.
    """
    if vix_df is None or vix_df.empty:
        logger.warning(
            "No VIX data provided for alignment. Proceeding with NIFTY data only."
        )
        return nifty_df

    logger.info("Merging and aligning Nifty 50 and India VIX data...")

    # Keep only date and close columns from VIX, rename close to vix_close
    vix_clean = vix_df[["date", "close"]].copy().rename(columns={"close": "vix_close"})

    # Sort both on date
    nifty_sorted = nifty_df.sort_values("date").reset_index(drop=True)
    vix_sorted = vix_clean.sort_values("date").reset_index(drop=True)

    # Perform outer join or left join? We do left join since NIFTY is the primary asset
    merged = pd.merge(nifty_sorted, vix_sorted, on="date", how="left")

    # Forward fill missing VIX values (e.g. if VIX has different holidays)
    missing_vix_before = merged["vix_close"].isna().sum()
    if missing_vix_before > 0:
        merged["vix_close"] = merged["vix_close"].ffill().bfill()
        missing_vix_after = merged["vix_close"].isna().sum()
        logger.info(
            f"Forward-filled VIX: {missing_vix_before - missing_vix_after} rows filled."
        )

    return merged


def process_and_save_data(
    nifty_raw_path: str,
    vix_raw_path: Optional[str] = None,
    config_path: str = "config/settings.yaml",
    source: str = "yfinance",
    strict: bool = True,
) -> str:
    """
    Reads raw CSV file(s), normalizes columns, aligns VIX if available,
    validates, and saves to the processed folder as a Parquet file.
    Saves validation_report.json and dataset_metadata.json.

    Args:
        nifty_raw_path: Path to raw NIFTY CSV.
        vix_raw_path: Optional path to raw VIX CSV.
        config_path: Path to configuration YAML.
        source: Source of download ('yfinance', 'local_cache', 'manual_csv_upload').
        strict: If True, fails with ValueError on validation failure.

    Returns:
        str: Absolute path of the saved Parquet file.
    """
    settings = load_settings(config_path)
    processed_dir = settings["data"].get("processed_dir", "data/processed")
    os.makedirs(processed_dir, exist_ok=True)

    logger.info(f"Loading raw Nifty 50 data from {nifty_raw_path}...")
    if not os.path.exists(nifty_raw_path):
        raise FileNotFoundError(f"Raw Nifty data file not found at: {nifty_raw_path}")

    raw_nifty = pd.read_csv(nifty_raw_path)
    nifty_normalized = normalize_columns(raw_nifty, config_path)

    # Process VIX if provided
    vix_normalized = None
    if vix_raw_path and os.path.exists(vix_raw_path):
        logger.info(f"Loading raw India VIX data from {vix_raw_path}...")
        raw_vix = pd.read_csv(vix_raw_path)
        vix_normalized = normalize_columns(raw_vix, config_path)

    # Merge datasets
    final_df = merge_and_align_vix(nifty_normalized, vix_normalized)

    # Sort by date
    final_df = final_df.sort_values("date").reset_index(drop=True)

    # Run validation
    has_vix = "vix_close" in final_df.columns
    is_valid, report = validate_market_data(final_df, check_vix=has_vix)

    # Save validation report
    val_report_path = os.path.join(processed_dir, "validation_report.json")
    if hasattr(report, "model_dump"):
        report_dict = report.model_dump()
    else:
        report_dict = report.dict()
    with open(val_report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)
    logger.info(f"Validation report saved to: {val_report_path}")

    if not is_valid:
        logger.warning(f"Validation issues discovered. Details: {report.details}.")
        if strict:
            raise ValueError(
                f"Data validation failed in strict mode. Inconsistencies: {report.details}."
            )
    else:
        logger.info("Validation checks PASSED successfully.")

    # Compute SHA-256 hash of the cleaned dataframe representation
    csv_bytes = final_df.to_csv(index=False).encode("utf-8")
    dataset_hash = hashlib.sha256(csv_bytes).hexdigest()

    # Generate and save dataset metadata JSON
    metadata = {
        "ticker": settings["data"].get("nifty_ticker", "^NSEI"),
        "start_date": (
            final_df["date"].min().strftime("%Y-%m-%d") if not final_df.empty else None
        ),
        "end_date": (
            final_df["date"].max().strftime("%Y-%m-%d") if not final_df.empty else None
        ),
        "downloaded_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "row_count": len(final_df),
        "missing_value_count": int(report.missing_values_count),
        "dataset_hash": dataset_hash,
    }

    metadata_path = os.path.join(processed_dir, "dataset_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Dataset metadata saved to: {metadata_path}")

    # Save to Parquet
    output_path = os.path.join(processed_dir, "nifty_cleaned.parquet")
    final_df.to_parquet(output_path, index=False)
    logger.info(f"Cleaned dataset saved as Parquet: {os.path.abspath(output_path)}")

    return os.path.abspath(output_path)
