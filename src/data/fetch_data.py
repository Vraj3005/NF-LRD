import os
import yaml
import logging
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

def load_settings(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Loads configuration settings from a YAML file."""
    if not os.path.exists(config_path):
        # Fallback to defaults if settings file is not found
        logger.warning(f"Configuration file {config_path} not found. Using default dictionary configuration.")
        return {
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "sample_dir": "data/sample",
                "nifty_ticker": "^NSEI",
                "vix_ticker": "^INDIAVIX",
                "default_start_date": "2010-01-01",
                "default_end_date": None
            }
        }
    
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def fetch_ticker_data(
    ticker: str, 
    start_date: str, 
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Downloads historical daily data for a given ticker from yfinance.

    Args:
        ticker: The Yahoo Finance ticker symbol (e.g. '^NSEI').
        start_date: Start date string (YYYY-MM-DD).
        end_date: Optional end date string (YYYY-MM-DD). If None, fetches up to today.

    Returns:
        Optional[pd.DataFrame]: DataFrame of historical data or None if download fails.
    """
    logger.info(f"Downloading data for ticker {ticker} from {start_date} to {end_date or 'today'}...")
    try:
        # Fetching data using yfinance
        df = yf.download(
            tickers=ticker, 
            start=start_date, 
            end=end_date, 
            progress=False, 
            auto_adjust=False  # We want both Close and Adj Close
        )
        
        if df.empty:
            logger.warning(f"No data returned for ticker {ticker}.")
            return None
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # yfinance columns might be MultiIndex if downloaded with multiple tickers,
        # but here we download singly. Let's flatten columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        logger.info(f"Successfully downloaded {len(df)} rows for {ticker}.")
        return df
    except Exception as e:
        logger.error(f"Error downloading data for ticker {ticker}: {e}")
        return None

def save_raw_data(df: pd.DataFrame, filename: str, raw_dir: str) -> str:
    """
    Saves the fetched raw DataFrame into the raw data directory as a CSV.

    Args:
        df: pd.DataFrame of historical data.
        filename: Target file name (e.g., 'nifty_raw.csv').
        raw_dir: Directory where raw files are saved.

    Returns:
        str: Absolute path of the saved file.
    """
    os.makedirs(raw_dir, exist_ok=True)
    file_path = os.path.join(raw_dir, filename)
    df.to_csv(file_path, index=False)
    logger.info(f"Saved raw data to {file_path}")
    return os.path.abspath(file_path)

def execute_fetch_pipeline(
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None,
    config_path: str = "config/settings.yaml"
) -> Dict[str, str]:
    """
    Orchestrates the fetching process for both Nifty 50 and India VIX.

    Args:
        start_date: Start date string (YYYY-MM-DD). If None, uses value from settings.yaml.
        end_date: End date string (YYYY-MM-DD). If None, uses value from settings.yaml.
        config_path: Path to configuration file.

    Returns:
        Dict[str, str]: Map of ticker names to absolute saved file paths.
    """
    settings = load_settings(config_path)
    
    # Resolve dates
    start = start_date or settings["data"].get("default_start_date", "2010-01-01")
    end = end_date or settings["data"].get("default_end_date")
    
    if not end:
        end = datetime.today().strftime('%Y-%m-%d')
        
    raw_dir = settings["data"].get("raw_dir", "data/raw")
    nifty_ticker = settings["data"].get("nifty_ticker", "^NSEI")
    vix_ticker = settings["data"].get("vix_ticker", "^INDIAVIX")
    
    saved_files = {}
    
    # Fetch NIFTY 50
    nifty_df = fetch_ticker_data(nifty_ticker, start, end)
    if nifty_df is not None:
        saved_path = save_raw_data(nifty_df, "nifty_raw.csv", raw_dir)
        saved_files["nifty"] = saved_path
    else:
        logger.error("Failed to download NIFTY 50 data.")
        
    # Fetch India VIX (optional/best-effort)
    vix_df = fetch_ticker_data(vix_ticker, start, end)
    if vix_df is not None:
        saved_path = save_raw_data(vix_df, "vix_raw.csv", raw_dir)
        saved_files["vix"] = saved_path
    else:
        logger.warning("Failed or skipped downloading India VIX.")
        
    return saved_files
