#!/usr/bin/env python
"""
Health check script to verify that all precomputed demo artifacts exist,
are non-empty, and load without corruption.
"""

import json
import logging
import os
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("check_demo_artifacts")

REQUIRED_ARTIFACTS = {
    "regime_labeled_data.parquet": "parquet",
    "selected_model_report.json": "json",
    "transition_matrix.csv": "csv",
    "regime_risk_report.csv": "csv",
    "risk_report.csv": "csv",
    "backtest_summary.csv": "csv",
    "strategy_equity_curves.parquet": "parquet",
    "stress_test_report.csv": "csv",
    "monte_carlo_summary.csv": "csv",
    "model_comparison.csv": "csv",
}


def main():
    demo_dir = "demo_data"

    if not os.path.exists(demo_dir):
        logger.error(f"Demo directory '{demo_dir}' does not exist.")
        sys.exit(1)

    missing_files = []
    corruption_errors = []

    logger.info(f"Auditing demo artifacts in '{demo_dir}'...")

    for filename, file_type in REQUIRED_ARTIFACTS.items():
        path = os.path.join(demo_dir, filename)
        if not os.path.exists(path):
            logger.error(f"Missing required artifact: {path}")
            missing_files.append(filename)
            continue

        # File exists, check size
        size = os.path.getsize(path)
        if size == 0:
            logger.error(f"Empty artifact file: {path}")
            corruption_errors.append(f"{filename} (empty file)")
            continue

        # Try loading
        try:
            if file_type == "parquet":
                df = pd.read_parquet(path)
                logger.info(
                    f"✓ Loaded Parquet '{filename}': {df.shape[0]} rows, {df.shape[1]} columns."
                )
            elif file_type == "csv":
                df = pd.read_csv(path)
                logger.info(
                    f"✓ Loaded CSV '{filename}': {df.shape[0]} rows, {df.shape[1]} columns."
                )
            elif file_type == "json":
                with open(path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                logger.info(
                    f"✓ Loaded JSON '{filename}': {len(content)} key-value fields."
                )
        except Exception as e:
            logger.error(f"✗ Failed to load/parse '{filename}': {e}")
            corruption_errors.append(f"{filename} (error: {e})")

    if missing_files or corruption_errors:
        logger.error("Audit FAILED.")
        if missing_files:
            logger.error(f"Missing files ({len(missing_files)}): {missing_files}")
        if corruption_errors:
            logger.error(
                f"Corrupt files ({len(corruption_errors)}): {corruption_errors}"
            )
        sys.exit(1)

    logger.info(
        "All demo data artifacts verified successfully! Ready for Streamlit Cloud deployment."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
