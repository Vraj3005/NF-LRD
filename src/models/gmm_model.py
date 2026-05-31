import logging
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class GMMRegimeModel:
    """
    Gaussian Mixture Model wrapper for static (non-temporal) market regime clustering.
    Provides baseline clustering comparison for HMMs.
    """
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "diag",
        random_state: Optional[int] = None,
        max_iter: int = 100
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter
        
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=1
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "GMMRegimeModel":
        """Fits the GMM model on input features."""
        logger.info(f"Fitting Gaussian Mixture Model with {self.n_components} components...")
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts latent clusters (regimes) for input features."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts cluster assignment posterior probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting probabilities.")
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """
        Evaluates GMM fit quality using: log likelihood, AIC, BIC, and Silhouette Score.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation.")
            
        log_lik = float(self.model.score(X) * len(X))  # sklearn score is average log-likelihood
        aic = float(self.model.aic(X))
        bic = float(self.model.bic(X))
        
        # Predict clusters to compute Silhouette score
        labels = self.predict(X)
        
        # Silhouette score requires at least 2 distinct clusters
        if len(np.unique(labels)) > 1:
            # Silhouette calculation can be slow on very large matrices, so we take a sample if too large
            if len(X) > 10000:
                indices = np.random.choice(len(X), size=5000, replace=False)
                sil = float(silhouette_score(X[indices], labels[indices], random_state=self.random_state))
            else:
                sil = float(silhouette_score(X, labels, random_state=self.random_state))
        else:
            sil = 0.0
            
        return {
            "log_likelihood": log_lik,
            "aic": aic,
            "bic": bic,
            "silhouette_score": sil
        }
