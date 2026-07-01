import json
import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.data.fetch_data import load_settings
from src.models.gmm_model import GMMRegimeModel
from src.models.hmm_model import GaussianHMM
from src.models.kmeans_model import KMeansRegimeModel
from src.models.markov_switching import MarkovSwitchingModel

logger = logging.getLogger(__name__)


def calculate_hmm_aic_bic(
    log_likelihood: float,
    n_samples: int,
    n_features: int,
    n_components: int,
    covariance_type: str = "diag",
) -> Tuple[float, float]:
    """
    Computes AIC and BIC for a Hidden Markov Model.
    """
    M = n_components
    D = n_features

    if covariance_type == "diag":
        n_params = (M - 1) + M * (M - 1) + M * D + M * D
    else:
        n_params = (M - 1) + M * (M - 1) + M * D + M * (D * (D + 1) // 2)

    aic = 2.0 * n_params - 2.0 * log_likelihood
    bic = n_params * np.log(n_samples) - 2.0 * log_likelihood

    return float(aic), float(bic)


def label_regimes(df: pd.DataFrame, predicted_states: np.ndarray) -> Dict[int, str]:
    """
    Assigns descriptive regime names to numeric states dynamically based on their statistics.
    Stats: annualized return, annualized volatility, max drawdown, skew, mean VIX (if available).
    """
    df_states = df.copy()
    df_states["state"] = predicted_states

    stats = []
    for state in np.unique(predicted_states):
        subset = df_states[df_states["state"] == state]
        if subset.empty:
            continue

        ret_col = "ret_simple" if "ret_simple" in subset.columns else None
        if not ret_col:
            if "raw_close" in subset.columns:
                subset = subset.copy()
                subset["ret_simple"] = subset["raw_close"].pct_change(fill_method=None)
                ret_col = "ret_simple"
            else:
                ret_cols = [c for c in subset.columns if "ret" in c or "return" in c]
                if ret_cols:
                    ret_col = ret_cols[0]

        if ret_col:
            returns = subset[ret_col].dropna()
        else:
            returns = pd.Series([0.0])

        ann_ret = float(returns.mean() * 252) if not returns.empty else 0.0
        ann_vol = (
            float(returns.std() * np.sqrt(252))
            if (not returns.empty and len(returns) > 1)
            else 1e-5
        )
        skew = (
            float(returns.skew()) if (not returns.empty and len(returns) > 2) else 0.0
        )

        if "raw_close" in subset.columns:
            prices = subset["raw_close"]
            peaks = prices.cummax()
            drawdowns = (prices - peaks) / (peaks + 1e-10)
            max_dd = float(drawdowns.min())
        else:
            max_dd = 0.0

        mean_vix = (
            float(subset["raw_vix_close"].mean())
            if "raw_vix_close" in subset.columns
            else 0.0
        )

        stats.append(
            {
                "state": int(state),
                "ann_ret": ann_ret,
                "ann_vol": ann_vol,
                "skew": skew,
                "max_dd": max_dd,
                "mean_vix": mean_vix,
                "count": len(subset),
            }
        )

    # Sort states by Sharpe-like ratio: return / volatility
    stats_sorted = sorted(
        stats, key=lambda x: x["ann_ret"] / (x["ann_vol"] + 1e-10), reverse=True
    )

    labels = {}
    n_states = len(stats_sorted)

    if n_states == 2:
        labels[stats_sorted[0]["state"]] = "Bullish Low Volatility"
        labels[stats_sorted[1]["state"]] = "Bearish High Volatility"
    elif n_states == 3:
        labels[stats_sorted[0]["state"]] = "Bullish Low Volatility"
        labels[stats_sorted[1]["state"]] = "Sideways / Range-Bound"
        labels[stats_sorted[2]["state"]] = "Bearish High Volatility"
    elif n_states == 4:
        labels[stats_sorted[0]["state"]] = "Bullish Low Volatility"
        labels[stats_sorted[1]["state"]] = "Recovery Regime"
        labels[stats_sorted[2]["state"]] = "Distribution / Risk-Off"
        labels[stats_sorted[3]["state"]] = "Bearish High Volatility"
    elif n_states >= 5:
        labels[stats_sorted[0]["state"]] = "Bullish Low Volatility"
        labels[stats_sorted[1]["state"]] = "Bullish High Volatility"
        labels[stats_sorted[2]["state"]] = "Recovery Regime"
        labels[stats_sorted[3]["state"]] = "Sideways / Range-Bound"
        labels[stats_sorted[4]["state"]] = "Bearish High Volatility"
        for i in range(5, n_states):
            labels[stats_sorted[i]["state"]] = "Distribution / Risk-Off"

    return labels


def check_seed_sensitivity(X: np.ndarray, n_components: int) -> float:
    """
    Fits the HMM with 5 different random seeds and returns the average Adjusted Rand Index (ARI)
    against the baseline (seed=42) to evaluate optimization stability.
    """
    try:
        base_model = GaussianHMM(n_components=n_components, random_state=42, n_iter=50)
        base_model.fit(X)
        base_states = base_model.predict(X)
    except Exception:
        return 0.0

    seeds = [100, 2026, 999, 123, 777]
    aris = []
    for s in seeds:
        try:
            m = GaussianHMM(n_components=n_components, random_state=s, n_iter=50)
            m.fit(X)
            states = m.predict(X)
            aris.append(adjusted_rand_score(base_states, states))
        except Exception:
            pass

    return float(np.mean(aris)) if aris else 1.0


def check_regime_stability(
    predicted_states: np.ndarray, transition_matrix: np.ndarray
) -> dict:
    """
    Performs stability checks including transition matrix sanity, occupancy levels,
    persistence metrics, and issues warnings for under-occupied regimes.
    """
    n_samples = len(predicted_states)
    unique_states, counts = np.unique(predicted_states, return_counts=True)
    occupancy = counts / n_samples

    warnings = []
    for state, occ, count in zip(unique_states, occupancy, counts):
        if occ < 0.05 or count < 30:
            warnings.append(
                f"Regime {state} is under-occupied: {occ:.1%} occupancy ({count} days). "
                "Risk of overfitting or regime instability."
            )
            logger.warning(warnings[-1])

    persistence = np.diagonal(transition_matrix)
    trans_row_sums = transition_matrix.sum(axis=1)
    is_trans_sane = np.allclose(trans_row_sums, 1.0, atol=1e-5)

    return {
        "occupancy": {int(s): float(occ) for s, occ in zip(unique_states, occupancy)},
        "persistence": {int(s): float(p) for s, p in enumerate(persistence)},
        "transition_matrix_sane": bool(is_trans_sane),
        "stability_warnings": warnings,
    }


def run_model_selection_sweep(
    features_df: pd.DataFrame, config_path: str = "config/settings.yaml"
) -> Tuple[str, int, Dict[str, Any]]:
    """
    Sweeps components from 2 to 6 for GMM, HMM, Markov Switching Regression, and KMeans.
    Creates a comprehensive model comparison table and selects the optimal model.
    """
    settings = load_settings(config_path)
    reports_dir = os.path.join(
        settings["data"].get("processed_dir", "data/processed"),
        "../../models/reports",
    )
    os.makedirs(reports_dir, exist_ok=True)

    feature_cols = [
        c for c in features_df.columns if c != "date" and not c.startswith("raw_")
    ]
    X = features_df[feature_cols].values

    # Extract returns column for MSR
    if "raw_close" in features_df.columns:
        returns = (
            features_df["raw_close"].pct_change(fill_method=None).fillna(0.0).values
        )
    else:
        returns = X[:, 0]

    # Filter NaN warmup rows to prevent fitting errors
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    returns = returns[valid_mask]
    n_samples, n_features = X.shape

    logger.info(
        f"Starting model selection sweep on {n_samples} samples with {n_features} features..."
    )

    results = []

    # Sweep from 2 to 6 components
    for k in range(2, 7):
        # 1. Evaluate GMM
        try:
            gmm = GMMRegimeModel(
                n_components=k, covariance_type="diag", random_state=42
            )
            gmm.fit(X)
            gmm_metrics = gmm.evaluate(X)
            results.append(
                {
                    "model_type": "GMM",
                    "n_components": k,
                    "log_likelihood": gmm_metrics["log_likelihood"],
                    "aic": gmm_metrics["aic"],
                    "bic": gmm_metrics["bic"],
                    "silhouette_score": gmm_metrics["silhouette_score"],
                    "avg_duration": 1.0,
                }
            )
        except Exception as e:
            logger.error(f"GMM fitting failed for k={k}: {e}")

        # 2. Evaluate KMeans
        try:
            km = KMeansRegimeModel(n_components=k, random_state=42)
            km.fit(X)
            states_km = km.predict(X)
            sil_km = (
                float(silhouette_score(X[::2], states_km[::2], random_state=42))
                if len(X) > 2
                else 0.0
            )
            results.append(
                {
                    "model_type": "KMeans",
                    "n_components": k,
                    "log_likelihood": -float(km.model.inertia_),
                    "aic": 9999.0,
                    "bic": 9999.0,
                    "silhouette_score": sil_km,
                    "avg_duration": 1.0,
                }
            )
        except Exception as e:
            logger.error(f"KMeans fitting failed for k={k}: {e}")

        # 3. Evaluate Markov Switching Regression (limit to at most 3 components due to statsmodels constraints)
        if k <= 3:
            try:
                msr = MarkovSwitchingModel(n_components=k, random_state=42)
                msr.fit(returns)
                msr_metrics = msr.evaluate()
                results.append(
                    {
                        "model_type": "MSR",
                        "n_components": k,
                        "log_likelihood": msr_metrics["log_likelihood"],
                        "aic": msr_metrics["aic"],
                        "bic": msr_metrics["bic"],
                        "silhouette_score": 0.0,
                        "avg_duration": 1.0,
                    }
                )
            except Exception as e:
                logger.error(f"MSR fitting failed for k={k}: {e}")

        # 4. Evaluate HMM
        try:
            hmm = GaussianHMM(
                n_components=k,
                covariance_type="diag",
                random_state=42,
                n_iter=150,
            )
            hmm.fit(X)

            states = hmm.predict(X)

            consecutive_runs = []
            current_run = 1
            for i in range(1, len(states)):
                if states[i] == states[i - 1]:
                    current_run += 1
                else:
                    consecutive_runs.append(current_run)
                    current_run = 1
            consecutive_runs.append(current_run)
            avg_duration = float(np.mean(consecutive_runs))

            log_lik = hmm.score(X)
            aic, bic = calculate_hmm_aic_bic(log_lik, n_samples, n_features, k, "diag")

            if len(np.unique(states)) > 1:
                if len(X) > 5000:
                    indices = np.random.choice(len(X), size=3000, replace=False)
                    sil = float(
                        silhouette_score(X[indices], states[indices], random_state=42)
                    )
                else:
                    sil = float(silhouette_score(X, states, random_state=42))
            else:
                sil = 0.0

            results.append(
                {
                    "model_type": "HMM",
                    "n_components": k,
                    "log_likelihood": log_lik,
                    "aic": aic,
                    "bic": bic,
                    "silhouette_score": sil,
                    "avg_duration": avg_duration,
                }
            )
        except Exception as e:
            logger.error(f"HMM fitting failed for k={k}: {e}")

    # Save comparison to CSV
    comparison_df = pd.DataFrame(results)
    comparison_path = os.path.join(reports_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Model comparison matrix saved to: {os.path.abspath(comparison_path)}")

    # Automatic optimal model selection (minimizing HMM BIC)
    hmm_results = [r for r in results if r["model_type"] == "HMM"]

    if hmm_results:
        best_auto_hmm = min(hmm_results, key=lambda x: x["bic"])
        best_model_type = "HMM"
        best_n_components = best_auto_hmm["n_components"]
        best_metrics = best_auto_hmm
    else:
        best_gmm = min(
            [r for r in results if r["model_type"] == "GMM"],
            key=lambda x: x["bic"],
        )
        best_model_type = "GMM"
        best_n_components = best_gmm["n_components"]
        best_metrics = best_gmm

    # User Configuration Override
    override_model = settings.get("features", {}).get("override_model", "HMM")
    override_k = settings.get("features", {}).get("override_regimes", None)

    if override_k is not None:
        logger.info(
            f"Manual override detected: Using {override_model} with {override_k} regimes."
        )
        matched_run = [
            r
            for r in results
            if r["model_type"] == override_model
            and r["n_components"] == int(override_k)
        ]
        if matched_run:
            best_model_type = override_model
            best_n_components = int(override_k)
            best_metrics = matched_run[0]
        else:
            best_model_type = override_model
            best_n_components = int(override_k)
            best_metrics = {"info": "Config override, metrics not calculated in sweep"}

    # Run seed sensitivity check on selected n_components if selected model is HMM
    seed_stability = 1.0
    if best_model_type == "HMM":
        seed_stability = check_seed_sensitivity(X, best_n_components)
        logger.info(
            f"HMM Seed Initialization Stability (Mean ARI): {seed_stability:.4f}"
        )

    # Save final selected model report JSON
    report_path = os.path.join(reports_dir, "selected_model_report.json")
    report_content = {
        "selected_model_type": best_model_type,
        "n_components": best_n_components,
        "selection_rationale": (
            "Minimized BIC parameter penalty (HMM baseline sweep)"
            if override_k is None
            else "User config override"
        ),
        "evaluation_metrics": best_metrics,
        "seed_sensitivity_ari": seed_stability,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_content, f, indent=4)

    logger.info(f"Selected model report saved to: {os.path.abspath(report_path)}")

    return best_model_type, best_n_components, best_metrics
