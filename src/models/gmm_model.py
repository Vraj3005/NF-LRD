import logging
import os
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from src.models.base import RegimeModelBase

logger = logging.getLogger(__name__)


class GMMRegimeModel(RegimeModelBase):
    """
    Gaussian Mixture Model wrapper for static (non-temporal) market regime clustering.
    Provides baseline clustering comparison for HMMs.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "diag",
        random_state: Optional[int] = None,
        max_iter: int = 100,
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
            n_init=1,
        )
        self.is_fitted = False
        self.last_predictions = None

    def fit(self, X: np.ndarray) -> "GMMRegimeModel":
        """Fits the GMM model on input features."""
        logger.info(
            f"Fitting Gaussian Mixture Model with {self.n_components} components..."
        )
        self.model.fit(X)
        self.is_fitted = True
        self.last_predictions = self.model.predict(X)
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

    def get_transition_matrix(self) -> np.ndarray:
        """
        GMM is static/non-temporal, so we compute empirical transition probabilities
        based on the sequence of decoded states in training to serve as a baseline.
        """
        if not self.is_fitted or self.last_predictions is None:
            return np.eye(self.n_components)

        states = self.last_predictions
        trans = np.zeros((self.n_components, self.n_components))
        for i in range(len(states) - 1):
            trans[states[i], states[i + 1]] += 1

        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return trans / row_sums

    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """Evaluates GMM fit quality using: log likelihood, AIC, BIC, and Silhouette Score."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation.")

        log_lik = float(self.model.score(X) * len(X))
        aic = float(self.model.aic(X))
        bic = float(self.model.bic(X))

        labels = self.predict(X)

        if len(np.unique(labels)) > 1:
            if len(X) > 10000:
                indices = np.random.choice(len(X), size=5000, replace=False)
                sil = float(
                    silhouette_score(
                        X[indices], labels[indices], random_state=self.random_state
                    )
                )
            else:
                sil = float(silhouette_score(X, labels, random_state=self.random_state))
        else:
            sil = 0.0

        return {
            "log_likelihood": log_lik,
            "aic": aic,
            "bic": bic,
            "silhouette_score": sil,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"GMM model saved successfully to {path}")

    @classmethod
    def load(cls, path: str) -> "GMMRegimeModel":
        return joblib.load(path)

    def summarize(self) -> dict:
        if not self.is_fitted:
            return {"is_fitted": False}

        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "means": (
                self.model.means_.tolist() if self.model.means_ is not None else None
            ),
            "covariances": (
                self.model.covariances_.tolist()
                if self.model.covariances_ is not None
                else None
            ),
            "transition_matrix": self.get_transition_matrix().tolist(),
            "converged": bool(self.model.converged_),
        }
