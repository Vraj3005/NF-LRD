import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class MarkovSwitchingModel:
    """
    Wrapper for statsmodels Markov Switching Regression.
    Fits a regime switching model directly on asset returns with switching variances.
    """
    def __init__(
        self,
        n_components: int = 2,
        switching_variance: bool = True
    ):
        self.n_components = n_components
        self.switching_variance = switching_variance
        
        self.model = None
        self.results = None
        self.is_fitted = False

    def fit(self, returns: np.ndarray) -> "MarkovSwitchingModel":
        """
        Fits the Markov Switching Regression model.
        Handles numerical estimation failures or convergence errors gracefully.
        """
        logger.info(f"Fitting Markov Switching Regression with {self.n_components} regimes...")
        
        # Ensure returns is 1D array and drop NaNs
        clean_returns = returns[~np.isnan(returns)]
        
        try:
            # We fit MarkovRegression on returns
            self.model = MarkovRegression(
                endog=clean_returns,
                k_regimes=self.n_components,
                switching_variance=self.switching_variance
            )
            
            # fit with EM search replicates to avoid local optima and increase convergence probability
            self.results = self.model.fit(
                search_reps=20,
                disp=False
            )
            self.is_fitted = True
            logger.info("Markov Switching Regression fitted successfully.")
        except Exception as e:
            logger.error(f"Markov Switching Regression failed to converge/fit: {e}")
            self.is_fitted = False
            self.results = None
            
        return self

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """
        Predicts the most likely regime for each day using smoothed probabilities.
        """
        if not self.is_fitted or self.results is None:
            # Return dummy zeros if fitting failed
            return np.zeros(len(returns), dtype=int)
            
        # Get smoothed probabilities for each state
        # results.smoothed_marginal_probabilities is shape [n_samples, n_components]
        smoothed_probs = self.results.smoothed_marginal_probabilities
        
        # Most likely regime is the argmax of smoothed probabilities per day
        predicted_states = np.argmax(smoothed_probs, axis=1)
        
        return predicted_states

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """
        Returns smoothed marginal probabilities of each regime.
        """
        if not self.is_fitted or self.results is None:
            # Return dummy probabilities if fitting failed
            dummy = np.zeros((len(returns), self.n_components))
            dummy[:, 0] = 1.0
            return dummy
            
        return self.results.smoothed_marginal_probabilities

    def evaluate(self) -> Dict[str, float]:
        """Calculates AIC, BIC, and Log Likelihood of the fitted model."""
        if not self.is_fitted or self.results is None:
            return {"log_likelihood": -9999.0, "aic": 9999.0, "bic": 9999.0}
            
        return {
            "log_likelihood": float(self.results.llf),
            "aic": float(self.results.aic),
            "bic": float(self.results.bic)
        }

    def get_transition_matrix(self) -> np.ndarray:
        """
        Returns the transition matrix.
        Statsmodels saves transition probabilities as:
        results.regime_transition_matrix (shape: [n_regimes, n_regimes])
        """
        if not self.is_fitted or self.results is None:
            return np.eye(self.n_components)
            
        # Statsmodels represents transitions as transition probability parameters,
        # results.regime_transition_matrix is a 3D or 2D array depending on whether it switches.
        # Typically it is shape [k_regimes, k_regimes]. Let's return it or a copy of it.
        try:
            matrix = self.results.regime_transition_matrix
            if isinstance(matrix, np.ndarray):
                # Statsmodels might return transition probability matrix where rows/cols indicate transitions.
                # Let's ensure it is 2D and normalized.
                if matrix.ndim == 3:
                    # In statsmodels it can be [regime_t, regime_t+1, time] if time-varying (not the case here)
                    matrix = matrix[:, :, 0]
                return matrix.T # Transpose if necessary, statsmodels transition matrix has matrix[i, j] as P(j -> i).
                # We want matrix[i, j] as P(i -> j) to match standard HMM, so transpose it.
            return np.eye(self.n_components)
        except Exception as e:
            logger.warning(f"Could not retrieve transition matrix: {e}")
            return np.eye(self.n_components)
