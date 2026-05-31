#!/usr/bin/env python
import os
import argparse
import logging
import shutil
from src.data.fetch_data import execute_fetch_pipeline, load_settings
from src.data.clean_data import process_and_save_data
from src.features.feature_pipeline import run_feature_engineering_pipeline

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_pipeline")

def parse_args():
    parser = argparse.ArgumentParser(
        description="NIFTY 50 Latent Market Regime Discovery - Data Pipeline Runner"
    )
    parser.add_argument(
        "--start-date", 
        type=str, 
        default=None, 
        help="Start date for data download (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end-date", 
        type=str, 
        default=None, 
        help="End date for data download (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--nifty-csv", 
        type=str, 
        default=None, 
        help="Path to a manual NIFTY 50 CSV file. Skips downloading from Yahoo Finance if provided."
    )
    parser.add_argument(
        "--vix-csv", 
        type=str, 
        default=None, 
        help="Path to a manual India VIX CSV file. Skips downloading from Yahoo Finance if provided."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/settings.yaml", 
        help="Path to configuration settings YAML file."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Initializing Data Ingestion and Validation Pipeline...")
    
    settings = load_settings(args.config)
    raw_dir = settings["data"].get("raw_dir", "data/raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    nifty_raw_path = ""
    vix_raw_path = ""
    
    # Check if manual CSV files are provided
    if args.nifty_csv:
        if not os.path.exists(args.nifty_csv):
            logger.error(f"Manual Nifty CSV file not found at: {args.nifty_csv}")
            return
        
        # Copy manual CSV file to raw directory
        nifty_raw_path = os.path.join(raw_dir, "nifty_raw.csv")
        logger.info(f"Using manual Nifty 50 CSV: {args.nifty_csv}. Copying to {nifty_raw_path}...")
        shutil.copy(args.nifty_csv, nifty_raw_path)
        
        # If manual VIX is provided
        if args.vix_csv:
            if not os.path.exists(args.vix_csv):
                logger.error(f"Manual VIX CSV file not found at: {args.vix_csv}")
                return
            vix_raw_path = os.path.join(raw_dir, "vix_raw.csv")
            logger.info(f"Using manual VIX CSV: {args.vix_csv}. Copying to {vix_raw_path}...")
            shutil.copy(args.vix_csv, vix_raw_path)
    else:
        # No manual CSV provided, fetch via yfinance
        logger.info("No manual CSV provided. Fetching data from Yahoo Finance...")
        fetched_files = execute_fetch_pipeline(
            start_date=args.start_date,
            end_date=args.end_date,
            config_path=args.config
        )
        
        nifty_raw_path = fetched_files.get("nifty", "")
        vix_raw_path = fetched_files.get("vix", "")
        
    if not nifty_raw_path or not os.path.exists(nifty_raw_path):
        logger.error("Nifty 50 raw data could not be acquired. Aborting pipeline.")
        return

    # Run processing and saving pipeline
    try:
        processed_path = process_and_save_data(
            nifty_raw_path=nifty_raw_path,
            vix_raw_path=vix_raw_path if vix_raw_path else None,
            config_path=args.config
        )
        logger.info(f"Data cleaning executed successfully! Cleaned Parquet file location: {processed_path}")
        
        # Run feature engineering pipeline
        logger.info("Initializing Feature Engineering Pipeline...")
        feature_parquet_path = run_feature_engineering_pipeline(config_path=args.config)
        logger.info(f"Feature engineering executed successfully! Features Parquet location: {feature_parquet_path}")
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
