import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.clean_data import (
    merge_and_align_vix,
    normalize_columns,
)
from src.data.fetch_data import fetch_ticker_data
from src.data.validation import validate_market_data


@pytest.fixture
def sample_valid_df():
    """Generates a valid market dataframe."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04"]
            ),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [105.0, 106.0, 107.0, 108.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [103.0, 102.0, 104.0, 105.0],
            "volume": [1000, 1500, 1200, 1600],
        }
    )


@pytest.fixture
def dummy_settings_yaml():
    """Returns a temporary settings config string."""
    return """
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  sample_dir: "data/sample"
  nifty_ticker: "^NSEI"
  vix_ticker: "^INDIAVIX"
  default_start_date: "2026-01-01"
  default_end_date: null

csv_mappings:
  date_columns:
    - "Date"
    - "Record Date"
  open_columns:
    - "Open"
    - "Open Price"
  high_columns:
    - "High"
    - "High Price"
  low_columns:
    - "Low"
    - "Low Price"
  close_columns:
    - "Close"
    - "Close Price"
    - "Index Value"
  volume_columns:
    - "Volume"
    - "Shares Traded"
  turnover_columns:
    - "Turnover"
    - "Turnover (Rs. Cr)"
"""


def test_validation_valid_data(sample_valid_df):
    """Tests validation with structurally correct financial data."""
    is_valid, report = validate_market_data(sample_valid_df)
    assert is_valid is True
    assert report.is_valid is True
    assert report.total_rows == 4
    assert report.duplicate_rows_count == 0
    assert report.invalid_ohlc_relation_count == 0


def test_validation_invalid_ohlc(sample_valid_df):
    """Tests validator flags rows where High < Low."""
    df_invalid = sample_valid_df.copy()
    # High < Low on row 1
    df_invalid.loc[1, "high"] = 98.0
    df_invalid.loc[1, "low"] = 100.0

    is_valid, report = validate_market_data(df_invalid)
    assert is_valid is False
    assert report.is_valid is False
    assert report.invalid_ohlc_relation_count > 0
    assert "high_less_than_low" in report.details


def test_validation_price_outside_range(sample_valid_df):
    """Tests validator flags rows where Close is outside [Low, High] range."""
    df_invalid = sample_valid_df.copy()
    # Close > High on row 2
    df_invalid.loc[2, "close"] = 110.0

    is_valid, report = validate_market_data(df_invalid)
    assert is_valid is False
    assert report.is_valid is False
    assert "price_outside_high_low" in report.details


def test_validation_duplicate_dates(sample_valid_df):
    """Tests validator flags duplicate date entries."""
    df_invalid = sample_valid_df.copy()
    # Duplicate second date
    df_invalid.loc[2, "date"] = df_invalid.loc[1, "date"]

    is_valid, report = validate_market_data(df_invalid)
    assert is_valid is False
    assert report.duplicate_rows_count == 1
    assert "duplicate_dates" in report.details


def test_validation_negative_prices(sample_valid_df):
    """Tests validator flags zero/negative prices."""
    df_invalid = sample_valid_df.copy()
    df_invalid.loc[1, "close"] = -5.0

    is_valid, report = validate_market_data(df_invalid)
    assert is_valid is False
    assert report.invalid_price_count == 1


def test_column_normalization(dummy_settings_yaml):
    """Tests column normalization using the mapping defined in configuration."""
    raw_df = pd.DataFrame(
        {
            "Date": ["2026-05-01", "2026-05-02"],
            "Open Price": ["100.0", "1,010.0"],  # Test commas too
            "High Price": [105.0, 106.0],
            "Low Price": [99.0, 100.0],
            "Close Price": [103.0, 102.0],
            "Shares Traded": [1000, 1500],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(dummy_settings_yaml)
        temp_config_path = f.name

    try:
        normalized = normalize_columns(raw_df, config_path=temp_config_path)
        assert "date" in normalized.columns
        assert "open" in normalized.columns
        assert "volume" in normalized.columns
        assert normalized.loc[1, "open"] == 1010.0  # Verify comma removed
    finally:
        os.remove(temp_config_path)


def test_vix_alignment(sample_valid_df):
    """Tests merge and alignment of Nifty 50 and India VIX data."""
    vix_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-05-01", "2026-05-03", "2026-05-04"]
            ),  # 2026-05-02 is missing
            "close": [15.0, 16.0, 14.0],
        }
    )

    merged = merge_and_align_vix(sample_valid_df, vix_df)
    assert "vix_close" in merged.columns
    assert len(merged) == len(sample_valid_df)
    # Check that missing date 2026-05-02 was filled (forward-filled from 2026-05-01 = 15.0)
    assert merged.loc[1, "vix_close"] == 15.0


@patch("yfinance.download")
def test_fetch_ticker_data(mock_download):
    """Tests fetch_ticker_data behaves correctly with yfinance download calls."""
    # Setup mock dataframe
    mock_df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.0],
            "Adj Close": [100.0],
            "Volume": [100],
        },
        index=pd.DatetimeIndex(["2026-05-01"]),
    )
    mock_df.index.name = "Date"

    mock_download.return_value = mock_df

    res = fetch_ticker_data("^NSEI", "2026-05-01", "2026-05-02")
    assert res is not None
    assert "Date" in res.columns
    assert "Open" in res.columns


def test_validation_unsorted_dates(sample_valid_df):
    """Tests that unsorted dates fail validation."""
    df_invalid = sample_valid_df.copy()
    # Swap date index 1 and 2 to make it unsorted
    temp_date = df_invalid.loc[1, "date"]
    df_invalid.loc[1, "date"] = df_invalid.loc[2, "date"]
    df_invalid.loc[2, "date"] = temp_date

    is_valid, report = validate_market_data(df_invalid)
    assert is_valid is False
    assert report.is_valid is False
    assert report.unsorted_dates_count > 0
    assert "unsorted_dates" in report.details


def test_validation_future_dates(sample_valid_df):
    """Tests that future dates fail validation."""
    df_invalid = sample_valid_df.copy()
    # Set date in future
    df_invalid.loc[1, "date"] = pd.Timestamp.now() + pd.Timedelta(days=5)

    is_valid, report = validate_market_data(df_invalid)
    assert is_valid is False
    assert report.is_valid is False
    assert report.future_dates_count == 1
    assert "future_dates" in report.details


def test_validation_negative_volume(sample_valid_df):
    """Tests that negative volumes fail validation."""
    df_invalid = sample_valid_df.copy()
    # Set negative volume
    df_invalid.loc[1, "volume"] = -100

    is_valid, report = validate_market_data(df_invalid)
    assert is_valid is False
    assert report.is_valid is False
    assert report.negative_volume_count == 1
    assert "negative_volume" in report.details


def test_validation_impossible_returns(sample_valid_df):
    """Tests that daily price changes exceeding +-30% fail validation."""
    df_invalid = sample_valid_df.copy()
    # Set close price to jump by 40%
    df_invalid.loc[1, "close"] = df_invalid.loc[0, "close"] * 1.40

    is_valid, report = validate_market_data(df_invalid)
    assert is_valid is False
    assert report.is_valid is False
    assert report.impossible_returns_count == 1
    assert "impossible_returns" in report.details


def test_validation_missing_values(sample_valid_df):
    """Tests that NaN or missing required values fail validation."""
    df_invalid = sample_valid_df.copy()
    # Insert NaN close price
    df_invalid.loc[1, "close"] = None

    is_valid, report = validate_market_data(df_invalid)
    assert is_valid is False
    assert report.is_valid is False
    assert report.missing_values_count == 1
    assert "missing_values" in report.details


def test_metadata_hash_determinism(sample_valid_df):
    """Verifies that identical dataframes generate identical SHA-256 hashes."""
    import hashlib

    # Compute hash 1
    csv_bytes1 = sample_valid_df.to_csv(index=False).encode("utf-8")
    hash1 = hashlib.sha256(csv_bytes1).hexdigest()

    # Compute hash 2
    csv_bytes2 = sample_valid_df.copy().to_csv(index=False).encode("utf-8")
    hash2 = hashlib.sha256(csv_bytes2).hexdigest()

    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) == 64
