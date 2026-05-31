import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from src.models.hmm_model import GaussianHMM
from src.models.gmm_model import GMMRegimeModel
from src.data.fetch_data import load_settings

logger = logging.getLogger(__name__)

def calculate_hmm_aic_bic(
    log_likelihood: float, 
    n_samples: int, 
    n_features: int, 
    n_components: int,
    covariance_type: str = "diag"
) -> Tuple[float, float]:
    """
    Computes AIC and BIC for a Hidden Markov Model.
    
    Number of parameters P:
    - startprob: components - 1
    - transmat: components * (components - 1)
    - means: components * features
    - covariances: components * features (if diagonal)
    """
    M = n_components
    D = n_features
    
    if covariance_type == "diag":
        n_params = (M - 1) + M * (M - 1) + M * D + M * D
    else:
        # Full covariance parameters per component: D * (D + 1) / 2
        n_params = (M - 1) + M * (M - 1) + M * D + M * (D * (D + 1) // 2)
        
    aic = 2.0 * n_params - 2.0 * log_likelihood
    bic = n_params * np.log(n_samples) - 2.0 * log_likelihood
    
    return float(aic), float(bic)

def run_model_selection_sweep(
    features_df: pd.DataFrame,
    config_path: str = "config/settings.yaml"
) -> Tuple[str, int, Dict[str, Any]]:
    """
    Sweeps components from 2 to 6 for HMM and GMM.
    Selects the optimal model automatically (minimizing HMM BIC) or respects manual override.

    Args:
        features_df: Standardized feature matrix (first column is 'date').
        config_path: Settings configuration path.

    Returns:
        Tuple[str, int, Dict[str, Any]]: (best_model_type, best_n_components, metrics_dict)
    """
    settings = load_settings(config_path)
    reports_dir = os.path.join(settings["data"].get("processed_dir", "data/processed"), "../../models/reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Extract only feature columns (exclude 'date' and raw_* meta columns)
    feature_cols = [c for c in features_df.columns if c != 'date' and not c.startswith('raw_')]
    X = features_df[feature_cols].values
    n_samples, n_features = X.shape
    
    logger.info(f"Starting model selection sweep on {n_samples} samples with {n_features} features...")
    
    results = []
    
    # Sweep from 2 to 6 components
    for k in range(2, 7):
        # 1. Evaluate GMM
        try:
            gmm = GMMRegimeModel(n_components=k, covariance_type="diag", random_state=42)
            gmm.fit(X)
            gmm_metrics = gmm.evaluate(X)
            results.append({
                "model_type": "GMM",
                "n_components": k,
                "log_likelihood": gmm_metrics["log_likelihood"],
                "aic": gmm_metrics["aic"],
                "bic": gmm_metrics["bic"],
                "silhouette_score": gmm_metrics["silhouette_score"],
                "avg_duration": 1.0  # GMM has no transition memory, daily assignment is independent
            })
        except Exception as e:
            logger.error(f"GMM fitting failed for k={k}: {e}")
            
        # 2. Evaluate HMM
        try:
            hmm = GaussianHMM(n_components=k, covariance_type="diag", random_state=42, n_iter=150)
            hmm.fit(X)
            
            # Predict states to compute average duration
            states = hmm.predict(X)
            
            # Calculate average duration of consecutive regimes
            # Find lengths of consecutive runs
            consecutive_runs = []
            current_run = 1
            for i in range(1, len(states)):
                if states[i] == states[i-1]:
                    current_run += 1
                else:
                    consecutive_runs.append(current_run)
                    current_run = 1
            consecutive_runs.append(current_run)
            avg_duration = float(np.mean(consecutive_runs))
            
            log_lik = hmm.score(X)
            aic, bic = calculate_hmm_aic_bic(log_lik, n_samples, n_features, k, "diag")
            
            # Approximate silhouette score for HMM states
            # Use states decoded by Viterbi as clustering labels
            if len(np.unique(states)) > 1:
                from sklearn.metrics import silhouette_score
                # Sample for speed if too large
                if len(X) > 5000:
                    indices = np.random.choice(len(X), size=3000, replace=False)
                    sil = float(silhouette_score(X[indices], states[indices], random_state=42))
                else:
                    sil = float(silhouette_score(X, states, random_state=42))
            else:
                sil = 0.0
                
            results.append({
                "model_type": "HMM",
                "n_components": k,
                "log_likelihood": log_lik,
                "aic": aic,
                "bic": bic,
                "silhouette_score": sil,
                "avg_duration": avg_duration
            })
        except Exception as e:
            logger.error(f"HMM fitting failed for k={k}: {e}")
            
    # Save comparison to CSV
    comparison_df = pd.DataFrame(results)
    comparison_path = os.path.join(reports_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Model comparison matrix saved to: {os.path.abspath(comparison_path)}")
    
    # Perform Automatic Selection: Minimize HMM BIC (standard practice for parsimony)
    hmm_results = [r for r in results if r["model_type"] == "HMM"]
    
    if hmm_results:
        # Find model with lowest BIC
        best_auto_hmm = min(hmm_results, key=lambda x: x["bic"])
        best_model_type = "HMM"
        best_n_components = best_auto_hmm["n_components"]
        best_metrics = best_auto_hmm
    else:
        # Fallback to GMM if HMM completely failed
        best_gmm = min(results, key=lambda x: x["bic"])
        best_model_type = "GMM"
        best_n_components = best_gmm["n_components"]
        best_metrics = best_gmm

    # Check for User Configuration Override
    override_model = settings.get("features", {}).get("override_model", "HMM")
    override_k = settings.get("features", {}).get("override_regimes", None)
    
    if override_k is not None:
        logger.info(f"Manual override detected in config: Using {override_model} with {override_k} regimes.")
        # Try to locate the requested override in results
        matched_run = [r for r in results if r["model_type"] == override_model and r["n_components"] == int(override_k)]
        if matched_run:
            best_model_type = override_model
            best_n_components = int(override_k)
            best_metrics = matched_run[0]
        else:
            best_model_type = override_model
            best_n_components = int(override_k)
            best_metrics = {"info": "Config override, metrics not calculated in sweep"}
            
    # Save final selected model report JSON
    report_path = os.path.join(reports_dir, "selected_model_report.json")
    report_content = {
        "selected_model_type": best_model_type,
        "n_components": best_n_components,
        "selection_rationale": "Minimized BIC parameter penalty (HMM baseline sweep)" if override_k is None else "User config override",
        "evaluation_metrics": best_metrics
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_content, f, indent=4)
        
    logger.info(f"Selected model report saved to: {os.path.abspath(report_path)}")
    
    return best_model_type, best_n_components, best_metrics
