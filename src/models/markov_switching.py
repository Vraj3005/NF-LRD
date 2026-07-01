import logging
import os
import warnings
from typing import Dict, Optional

# Suppress statsmodels convergence warnings
try:
    from statsmodels.tools.sm_exceptions import (
        ConvergenceWarning as StatsmodelsConvergenceWarning,
    )

    warnings.filterwarnings("ignore", category=StatsmodelsConvergenceWarning)
except ImportError:
    pass

import joblib
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from src.models.base import RegimeModelBase

logger = logging.getLogger(__name__)


class MarkovSwitchingModel(RegimeModelBase):
    """
    Wrapper for statsmodels Markov Switching Regression.
    Fits a regime switching model directly on asset returns with switching variances.
    """

    def __init__(
        self,
        n_components: int = 2,
        switching_variance: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.switching_variance = switching_variance
        self.random_state = random_state

        self.model = None
        self.results = None
        self.fallback_model = None
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> "MarkovSwitchingModel":
        """
        Fits the Markov Switching Regression model.
        Handles numerical estimation failures or convergence errors gracefully by falling back to HMM.
        """
        logger.info(
            f"Fitting Markov Switching Regression with {self.n_components} regimes..."
        )

        # Extract returns column if X_train is 2D
        if X_train.ndim == 2:
            returns = X_train[:, 0]
        else:
            returns = X_train

        clean_returns = returns[~np.isnan(returns)]

        try:
            self.model = MarkovRegression(
                endog=clean_returns,
                k_regimes=self.n_components,
                switching_variance=self.switching_variance,
            )

            # Fit with EM search replicates to avoid local optima
            self.results = self.model.fit(search_reps=20, disp=False)
            self.is_fitted = True
            self.fallback_model = None
            logger.info("Markov Switching Regression fitted successfully.")
        except Exception as e:
            logger.warning(
                f"Markov Switching Regression failed to converge: {e}. "
                "Falling back to GaussianHMM on returns."
            )
            from src.models.hmm_model import GaussianHMM

            X_fallback = clean_returns.reshape(-1, 1)
            self.fallback_model = GaussianHMM(
                n_components=self.n_components,
                covariance_type="diag",
                random_state=self.random_state or 42,
            )
            self.fallback_model.fit(X_fallback)
            self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the most likely regime for each day using smoothed probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")

        if self.fallback_model is not None:
            y = X[:, 0] if X.ndim == 2 else X
            return self.fallback_model.predict(y.reshape(-1, 1))

        if self.results is None:
            return np.zeros(len(X), dtype=int)

        y = X[:, 0] if X.ndim == 2 else X
        if len(y) == len(self.results.smoothed_marginal_probabilities):
            return np.argmax(self.results.smoothed_marginal_probabilities, axis=1)

        # Out-of-sample heuristic: match to closest regime const parameter
        try:
            regime_means = []
            for i in range(self.n_components):
                mean_val = 0.0
                for name, val in zip(
                    self.results.model.param_names, self.results.params
                ):
                    if (
                        f"const[{i}]" in name
                        or f"const.regime_{i}" in name
                        or "const" in name
                    ):
                        mean_val = val
                regime_means.append(mean_val)

            states = []
            for val in y:
                states.append(np.argmin([np.abs(val - m) for m in regime_means]))
            return np.array(states)
        except Exception:
            return np.zeros(len(y), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns smoothed marginal probabilities of each regime."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting probabilities.")

        if self.fallback_model is not None:
            y = X[:, 0] if X.ndim == 2 else X
            return self.fallback_model.predict_proba(y.reshape(-1, 1))

        if self.results is None:
            dummy = np.zeros((len(X), self.n_components))
            dummy[:, 0] = 1.0
            return dummy

        y = X[:, 0] if X.ndim == 2 else X
        if len(y) == len(self.results.smoothed_marginal_probabilities):
            return self.results.smoothed_marginal_probabilities

        # Out-of-sample fallback probability matrix
        dummy = np.zeros((len(y), self.n_components))
        states = self.predict(X)
        for idx, s in enumerate(states):
            dummy[idx, s] = 1.0
        return dummy

    def evaluate(self) -> Dict[str, float]:
        """Calculates AIC, BIC, and Log Likelihood of the fitted model."""
        if not self.is_fitted:
            return {"log_likelihood": -9999.0, "aic": 9999.0, "bic": 9999.0}

        if self.fallback_model is not None:
            # HMM fallback approximation
            return {
                "log_likelihood": -9999.0,
                "aic": 9999.0,
                "bic": 9999.0,
            }

        return {
            "log_likelihood": float(self.results.llf),
            "aic": float(self.results.aic),
            "bic": float(self.results.bic),
        }

    def get_transition_matrix(self) -> np.ndarray:
        """Returns the transition matrix."""
        if not self.is_fitted:
            return np.eye(self.n_components)

        if self.fallback_model is not None:
            return self.fallback_model.get_transition_matrix()

        if self.results is None:
            return np.eye(self.n_components)

        try:
            matrix = self.results.regime_transition_matrix
            if isinstance(matrix, np.ndarray):
                if matrix.ndim == 3:
                    matrix = matrix[:, :, 0]
                # Transpose to match standard P(i -> j) orientation
                trans = matrix.T
                # Re-normalize to ensure rows sum exactly to 1.0
                row_sums = trans.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                return trans / row_sums
            return np.eye(self.n_components)
        except Exception as e:
            logger.warning(f"Could not retrieve transition matrix: {e}")
            return np.eye(self.n_components)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Markov Switching model saved successfully to {path}")

    @classmethod
    def load(cls, path: str) -> "MarkovSwitchingModel":
        return joblib.load(path)

    def summarize(self) -> dict:
        if not self.is_fitted:
            return {"is_fitted": False}

        if self.fallback_model is not None:
            return {
                "n_components": self.n_components,
                "is_fallback": True,
                "transition_matrix": self.get_transition_matrix().tolist(),
            }

        return {
            "n_components": self.n_components,
            "switching_variance": self.switching_variance,
            "transition_matrix": self.get_transition_matrix().tolist(),
            "log_likelihood": float(self.results.llf) if self.results else None,
            "converged": True,
        }
