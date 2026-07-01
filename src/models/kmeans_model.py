import logging
import os
from typing import Optional

import joblib
import numpy as np
from sklearn.cluster import KMeans

from src.models.base import RegimeModelBase

logger = logging.getLogger(__name__)


class KMeansRegimeModel(RegimeModelBase):
    """
    K-Means clustering wrapper for static market regime assignment.
    Provides a simple distance-based benchmark.
    """

    def __init__(self, n_components: int = 3, random_state: Optional[int] = None):
        self.n_components = n_components
        self.random_state = random_state
        self.model = KMeans(
            n_clusters=self.n_components,
            random_state=self.random_state,
            n_init=10,
        )
        self.is_fitted = False
        self.last_predictions = None

    def fit(self, X: np.ndarray) -> "KMeansRegimeModel":
        logger.info(f"Fitting KMeans Model with {self.n_components} clusters...")
        self.model.fit(X)
        self.is_fitted = True
        self.last_predictions = self.model.predict(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        K-Means is a hard clustering model and does not natively output probabilities.
        We approximate soft assignments using inverse distances to cluster centers.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting probabilities.")
        distances = self.model.transform(X)
        eps = 1e-10
        inv_distances = 1.0 / (distances + eps)
        proba = inv_distances / inv_distances.sum(axis=1, keepdims=True)
        return proba

    def get_transition_matrix(self) -> np.ndarray:
        if not self.is_fitted or self.last_predictions is None:
            return np.eye(self.n_components)

        states = self.last_predictions
        trans = np.zeros((self.n_components, self.n_components))
        for i in range(len(states) - 1):
            trans[states[i], states[i + 1]] += 1

        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return trans / row_sums

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"KMeans model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "KMeansRegimeModel":
        return joblib.load(path)

    def summarize(self) -> dict:
        if not self.is_fitted:
            return {"is_fitted": False}
        return {
            "n_components": self.n_components,
            "inertia": float(self.model.inertia_),
            "centroids": (
                self.model.cluster_centers_.tolist()
                if self.model.cluster_centers_ is not None
                else None
            ),
            "transition_matrix": self.get_transition_matrix().tolist(),
        }
