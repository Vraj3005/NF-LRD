import logging
from typing import Any, Dict, Tuple

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ValidationReport(BaseModel):
    """Pydantic model representing the validation results."""

    is_valid: bool = Field(
        ..., description="Whether the dataset passed all strict validation rules"
    )
    total_rows: int = Field(..., description="Total number of rows in the dataset")
    missing_dates_count: int = Field(
        ..., description="Number of missing dates (business days)"
    )
    duplicate_rows_count: int = Field(
        ..., description="Number of duplicate date rows found"
    )
    non_numeric_count: int = Field(
        ..., description="Number of non-numeric cells in OHLCV"
    )
    invalid_ohlc_relation_count: int = Field(
        ..., description="Number of rows violating high/low/open/close boundaries"
    )
    invalid_price_count: int = Field(
        ..., description="Number of rows with close or open <= 0"
    )
    unsorted_dates_count: int = Field(
        0, description="Number of rows out of chronological order"
    )
    future_dates_count: int = Field(0, description="Number of rows with future dates")
    negative_volume_count: int = Field(
        0, description="Number of rows with negative volume"
    )
    impossible_returns_count: int = Field(
        0, description="Number of rows with impossible returns (> 30% or < -30%)"
    )
    missing_values_count: int = Field(
        0, description="Total number of missing/NaN values in required columns"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed error counts and logs per check"
    )


def validate_market_data(
    df: pd.DataFrame, check_vix: bool = False
) -> Tuple[bool, ValidationReport]:
    """
    Validates a market data DataFrame for logical inconsistencies, duplicate rows,
    missing dates, bad price relations, and non-numeric values.

    Args:
        df: pd.DataFrame with normalized column names: 'date', 'open', 'high', 'low', 'close', 'volume'
            (and optionally 'turnover' and/or 'vix_close').
        check_vix: bool, whether VIX validation checks should be run on 'vix_close'.

    Returns:
        Tuple[bool, ValidationReport]: (is_valid, report)
    """
    logger.info("Starting market data validation...")

    is_valid = True
    details: Dict[str, Any] = {}

    # 0. Check for empty dataframe
    if df.empty:
        logger.error("DataFrame is empty.")
        report = ValidationReport(
            is_valid=False,
            total_rows=0,
            missing_dates_count=0,
            duplicate_rows_count=0,
            non_numeric_count=0,
            invalid_ohlc_relation_count=0,
            invalid_price_count=0,
            details={"error": "Empty DataFrame"},
        )
        return False, report

    total_rows = len(df)

    # 1. Date Format & Column Checks
    required_cols = ["date", "open", "high", "low", "close", "volume"]
    if check_vix:
        required_cols.append("vix_close")

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in DataFrame: {missing_cols}")
        details["missing_columns"] = missing_cols
        report = ValidationReport(
            is_valid=False,
            total_rows=total_rows,
            missing_dates_count=0,
            duplicate_rows_count=0,
            non_numeric_count=0,
            invalid_ohlc_relation_count=0,
            invalid_price_count=0,
            details=details,
        )
        return False, report

    # Ensure 'date' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        try:
            df_to_check = df.copy()
            df_to_check["date"] = pd.to_datetime(df_to_check["date"])
        except Exception as e:
            logger.error(f"Date column is not parseable to datetime: {e}")
            details["date_parsing_error"] = str(e)
            report = ValidationReport(
                is_valid=False,
                total_rows=total_rows,
                missing_dates_count=0,
                duplicate_rows_count=0,
                non_numeric_count=0,
                invalid_ohlc_relation_count=0,
                invalid_price_count=0,
                details=details,
            )
            return False, report
    else:
        df_to_check = df

    # 2. Duplicate rows on Date
    duplicates = df_to_check[df_to_check.duplicated(subset=["date"], keep=False)]
    duplicate_count = duplicates["date"].nunique()
    if duplicate_count > 0:
        is_valid = False
        logger.warning(f"Found {duplicate_count} duplicate dates in data.")
        details["duplicate_dates"] = (
            duplicates["date"].dt.strftime("%Y-%m-%d").unique().tolist()
        )
    else:
        duplicate_count = 0

    # 3. Non-numeric OHLCV check
    non_numeric_count = 0
    numeric_cols = ["open", "high", "low", "close", "volume"]
    if check_vix and "vix_close" in df_to_check.columns:
        numeric_cols.append("vix_close")

    for col in numeric_cols:
        non_numeric_mask = pd.to_numeric(df_to_check[col], errors="coerce").isna()
        if non_numeric_mask.any():
            is_valid = False
            invalid_dates = (
                df_to_check.loc[non_numeric_mask, "date"]
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )
            non_numeric_count += len(invalid_dates)
            logger.warning(
                f"Non-numeric values found in column '{col}' for dates: {invalid_dates}"
            )
            details[f"non_numeric_{col}"] = invalid_dates

    # If non-numeric cells found, we coerce them to continue logical checks without crashing
    df_clean = df_to_check.copy()
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # 4. Inconsistent Close/Open values (<= 0)
    invalid_price_mask = (df_clean["close"] <= 0) | (df_clean["open"] <= 0)
    invalid_price_count = invalid_price_mask.sum()
    if invalid_price_count > 0:
        is_valid = False
        invalid_dates = (
            df_clean.loc[invalid_price_mask, "date"].dt.strftime("%Y-%m-%d").tolist()
        )
        logger.warning(f"Invalid open or close price (<= 0) on dates: {invalid_dates}")
        details["invalid_prices"] = invalid_dates

    # 5. High < Low
    high_low_mask = df_clean["high"] < df_clean["low"]
    high_low_count = high_low_mask.sum()
    if high_low_count > 0:
        is_valid = False
        invalid_dates = (
            df_clean.loc[high_low_mask, "date"].dt.strftime("%Y-%m-%d").tolist()
        )
        logger.warning(f"High price less than low price on dates: {invalid_dates}")
        details["high_less_than_low"] = invalid_dates

    # 6. Open or Close outside High-Low range (adding small float epsilon)
    eps = 1e-5
    outside_hl_mask = (
        (df_clean["open"] > df_clean["high"] + eps)
        | (df_clean["open"] < df_clean["low"] - eps)
        | (df_clean["close"] > df_clean["high"] + eps)
        | (df_clean["close"] < df_clean["low"] - eps)
    )
    outside_hl_count = outside_hl_mask.sum()
    if outside_hl_count > 0:
        is_valid = False
        invalid_dates = (
            df_clean.loc[outside_hl_mask, "date"].dt.strftime("%Y-%m-%d").tolist()
        )
        logger.warning(
            f"Open or Close prices outside [Low, High] range on dates: {invalid_dates}"
        )
        details["price_outside_high_low"] = invalid_dates

    total_invalid_ohlc_relations = int(high_low_count + outside_hl_count)

    # 7. Check for future dates
    future_mask = df_clean["date"] > pd.Timestamp.now()
    future_dates_count = int(future_mask.sum())
    if future_dates_count > 0:
        is_valid = False
        invalid_dates = (
            df_clean.loc[future_mask, "date"].dt.strftime("%Y-%m-%d").tolist()
        )
        logger.warning(f"Future dates found: {invalid_dates}")
        details["future_dates"] = invalid_dates

    # 8. Check for negative volume
    negative_volume_mask = df_clean["volume"] < 0
    negative_volume_count = int(negative_volume_mask.sum())
    if negative_volume_count > 0:
        is_valid = False
        invalid_dates = (
            df_clean.loc[negative_volume_mask, "date"].dt.strftime("%Y-%m-%d").tolist()
        )
        logger.warning(f"Negative volume found on dates: {invalid_dates}")
        details["negative_volume"] = invalid_dates

    # 9. Check for impossible returns (|daily change| > 30%)
    impossible_returns_count = 0
    if total_rows > 1:
        daily_returns = df_clean["close"].pct_change()
        impossible_returns_mask = (daily_returns.abs() > 0.30) & (daily_returns.notna())
        impossible_returns_count = int(impossible_returns_mask.sum())
        if impossible_returns_count > 0:
            is_valid = False
            invalid_dates = (
                df_clean.loc[impossible_returns_mask, "date"]
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )
            logger.warning(
                f"Impossible daily returns (>30%) found on dates: {invalid_dates}"
            )
            details["impossible_returns"] = invalid_dates

    # 10. Check for missing values in required columns
    missing_values_count = int(df_clean[required_cols].isna().sum().sum())
    if missing_values_count > 0:
        is_valid = False
        missing_summary = df_clean[required_cols].isna().sum().to_dict()
        logger.warning(f"Missing values found in required columns: {missing_summary}")
        details["missing_values"] = missing_summary

    # 11. Check chronological sorting
    unsorted_dates_count = 0
    if not df_clean["date"].is_monotonic_increasing:
        is_valid = False
        logger.warning("Dates are not strictly sorted in ascending order.")
        unsorted_dates_count = int((df_clean["date"].diff() < pd.Timedelta(0)).sum())
        details["unsorted_dates"] = "Dataset is not sorted chronologically"

    # 12. Check for missing business days (Gaps in trading history)
    missing_dates_count = 0
    if total_rows > 1:
        # Create a complete business day range between min and max dates
        min_date = df_clean["date"].min()
        max_date = df_clean["date"].max()
        expected_business_days = pd.date_range(start=min_date, end=max_date, freq="B")

        actual_dates = set(df_clean["date"].dt.normalize())
        expected_dates = set(expected_business_days.normalize())

        missing_dates = expected_dates - actual_dates

        missing_dates_count = len(missing_dates)
        if missing_dates_count > 0:
            logger.info(
                f"Found {missing_dates_count} missing business days (may include national holidays)."
            )
            details["missing_business_days"] = sorted(
                [d.strftime("%Y-%m-%d") for d in missing_dates]
            )

    report = ValidationReport(
        is_valid=is_valid,
        total_rows=total_rows,
        missing_dates_count=missing_dates_count,
        duplicate_rows_count=int(duplicate_count),
        non_numeric_count=non_numeric_count,
        invalid_ohlc_relation_count=total_invalid_ohlc_relations,
        invalid_price_count=int(invalid_price_count),
        unsorted_dates_count=unsorted_dates_count,
        future_dates_count=future_dates_count,
        negative_volume_count=negative_volume_count,
        impossible_returns_count=impossible_returns_count,
        missing_values_count=missing_values_count,
        details=details,
    )

    logger.info(f"Validation completed. Dataset Valid: {is_valid}")
    return is_valid, report
