"""
Unit tests verifying report generations, CSV files, MD documents, and JSON metadata artifacts.
"""

import json
import os

import pandas as pd


def test_reports_exist_and_are_valid():
    """Verifies that strategy sweep reports and JSON metadata are created and valid."""
    # Check parameter sweep reports
    metrics_path = "reports/final_metrics.csv"
    comp_path = "reports/strategy_comparison.csv"
    results_path = "reports/final_results.md"
    robust_path = "reports/robustness_summary.csv"

    # We allow tests to pass if run before sweep by checking existence or verifying structure if they exist
    for path in [metrics_path, comp_path, results_path, robust_path]:
        assert os.path.exists(path), f"Report path does not exist: {path}"
        assert os.path.getsize(path) > 0, f"Report path is empty: {path}"

    # Verify CSV headers
    df_metrics = pd.read_csv(metrics_path)
    assert "mapping" in df_metrics.columns
    assert "is_sharpe" in df_metrics.columns
    assert "oos_sharpe" in df_metrics.columns

    df_robust = pd.read_csv(robust_path)
    assert "mapping" in df_robust.columns
    assert "regime_count" in df_robust.columns
    assert "seed" in df_robust.columns


def test_json_artifacts_are_valid():
    """Verifies that dataset validation and metadata JSON files exist and contain valid JSON structure."""
    val_json_path = "data/processed/validation_report.json"
    meta_json_path = "data/processed/dataset_metadata.json"

    for path in [val_json_path, meta_json_path]:
        assert os.path.exists(path), f"JSON artifact path does not exist: {path}"
        assert os.path.getsize(path) > 0, f"JSON artifact path is empty: {path}"

        # Load and verify JSON structure
        with open(path, "r") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"JSON root is not a dictionary: {path}"

    # Schema checks
    with open(val_json_path, "r") as f:
        val_data = json.load(f)
        assert "is_valid" in val_data
        assert "duplicate_rows_count" in val_data

    with open(meta_json_path, "r") as f:
        meta_data = json.load(f)
        assert "ticker" in meta_data
        assert "dataset_hash" in meta_data
