from abc import ABC, abstractmethod

import numpy as np


class RegimeModelBase(ABC):
    """
    Abstract Base Class establishing a common interface for all latent market regime discovery models.
    """

    @abstractmethod
    def fit(self, X_train: np.ndarray):
        """Fits the regime discovery model on training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Decodes the most likely regime sequences for input observations."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculates state posterior or smoothed probabilities for input observations."""
        pass

    @abstractmethod
    def get_transition_matrix(self) -> np.ndarray:
        """Returns the [n_components, n_components] state transition probability matrix."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Saves the fitted model object to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Loads a model object from disk."""
        pass

    @abstractmethod
    def summarize(self) -> dict:
        """Returns a diagnostic summary dictionary of fitted model metrics and configurations."""
        pass
