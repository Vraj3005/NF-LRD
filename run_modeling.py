#!/usr/bin/env python
import argparse
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

from src.analysis.regime_analysis import analyze_regimes, run_walk_forward_validation
from src.data.fetch_data import load_settings
from src.models.gmm_model import GMMRegimeModel
from src.models.hmm_model import GaussianHMM
from src.models.markov_switching import MarkovSwitchingModel
from src.models.model_selection import run_model_selection_sweep

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_modeling")


def parse_args():
    parser = argparse.ArgumentParser(
        description="NIFTY 50 Latent Market Regime Discovery - Modeling Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration settings YAML file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing Latent Market Regime Discovery Modeling Pipeline...")

    settings = load_settings(args.config)
    proc_dir = settings["data"].get("processed_dir", "data/processed")
    features_parquet = settings.get("features", {}).get(
        "feature_parquet_path", "data/processed/features.parquet"
    )

    if not os.path.exists(features_parquet):
        logger.error(
            f"Features Parquet not found at: {features_parquet}. Please run the feature pipeline first."
        )
        return

    logger.info(f"Loading feature matrix from {features_parquet}...")
    features_df = pd.read_parquet(features_parquet)

    # 1. Run parameter sweep and model selection
    best_model_type, best_k, best_metrics = run_model_selection_sweep(
        features_df, args.config
    )
    logger.info(f"Model Selection Selected: {best_model_type} with {best_k} regimes.")

    # Extract only feature columns for training
    feature_cols = [
        c for c in features_df.columns if c != "date" and not c.startswith("raw_")
    ]
    X = features_df[feature_cols].values

    # Filter NaN warmup rows for training
    valid_mask = ~np.isnan(X).any(axis=1)
    X_clean = X[valid_mask]

    # 2. Train final selected model on full dataset
    logger.info(
        f"Training final {best_model_type} model on all data with {best_k} components..."
    )

    final_model = None
    states_clean = None

    if best_model_type == "HMM":
        final_model = GaussianHMM(
            n_components=best_k, covariance_type="diag", random_state=42, n_iter=200
        )
        final_model.fit(X_clean)
        states_clean = final_model.predict(X_clean)
    elif best_model_type == "GMM":
        final_model = GMMRegimeModel(
            n_components=best_k, covariance_type="diag", random_state=42
        )
        final_model.fit(X_clean)
        states_clean = final_model.predict(X_clean)
    elif best_model_type == "MSR":
        if "raw_close" in features_df.columns:
            returns = (
                features_df["raw_close"].pct_change(fill_method=None).fillna(0.0).values
            )
        else:
            returns = X[:, 0]
        returns_clean = returns[valid_mask]
        final_model = MarkovSwitchingModel(n_components=best_k, random_state=42)
        final_model.fit(returns_clean)
        states_clean = final_model.predict(returns_clean)
    else:
        from src.models.kmeans_model import KMeansRegimeModel

        final_model = KMeansRegimeModel(n_components=best_k, random_state=42)
        final_model.fit(X_clean)
        states_clean = final_model.predict(X_clean)

    # Reconstruct state sequence mapping -1 for NaN warmup rows
    states = np.full(len(X), -1, dtype=int)
    states[valid_mask] = states_clean

    # Save the trained model binary
    models_dir = os.path.join(proc_dir, "../../models/saved")
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "regime_model.joblib")
    joblib.dump(final_model, model_save_path)
    logger.info(
        f"Trained model binary serialized to: {os.path.abspath(model_save_path)}"
    )

    # 3. Analyze discovered regimes
    logger.info("Analyzing discovered regimes and assigning human-readable labels...")
    labeled_df, summary_df, trans_df = analyze_regimes(features_df, states, best_k)

    # Save output dataframes
    reports_dir = os.path.join(proc_dir, "../../models/reports")
    os.makedirs(reports_dir, exist_ok=True)

    labeled_parquet_path = os.path.join(reports_dir, "regime_labeled_data.parquet")
    labeled_df.to_parquet(labeled_parquet_path, index=False)
    logger.info(
        f"Regime-labeled dataset saved to: {os.path.abspath(labeled_parquet_path)}"
    )

    # Save summaries
    summary_csv_path = os.path.join(reports_dir, "regime_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(
        f"Regime statistical summary saved to: {os.path.abspath(summary_csv_path)}"
    )

    # Save transition probabilities
    trans_csv_path = os.path.join(reports_dir, "transition_matrix.csv")
    trans_df.to_csv(trans_csv_path)
    logger.info(
        f"Transition probability matrix saved to: {os.path.abspath(trans_csv_path)}"
    )

    # 4. Perform Walk-forward Validation
    if best_model_type == "HMM":
        model_cls = GaussianHMM
    else:
        model_cls = GMMRegimeModel

    validation_report = run_walk_forward_validation(features_df, model_cls, best_k)

    # Append validation results to selected_model_report
    report_json_path = os.path.join(reports_dir, "selected_model_report.json")
    with open(report_json_path, "r", encoding="utf-8") as f:
        report_content = json.load(f)

    report_content["walk_forward_validation"] = validation_report

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_content, f, indent=4)

    logger.info(
        f"Updated selected model report with out-of-sample stability metrics: {os.path.abspath(report_json_path)}"
    )
    logger.info("Market Regime Discovery modeling run completed successfully!")


if __name__ == "__main__":
    main()
