import logging
import os
import warnings
from typing import Optional, Tuple

# Suppress convergence and optimization warnings from third-party libraries
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

import joblib
import numpy as np
import scipy.stats

from src.models.base import RegimeModelBase

logger = logging.getLogger(__name__)

# Check if hmmlearn is available
try:
    from hmmlearn.hmm import GaussianHMM as LibGaussianHMM

    HMMLEARN_AVAILABLE = True
    logger.info(
        "hmmlearn successfully imported. Using compiled library for GaussianHMM."
    )
except ImportError:
    HMMLEARN_AVAILABLE = False
    logger.warning(
        "hmmlearn NOT available on this system. Falling back to pure Python/NumPy log-space GaussianHMM."
    )


def fast_logsumexp(
    a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False
) -> np.ndarray:
    """
    Vectorized logsumexp implementation using raw NumPy for high performance.
    Correctly handles keepdims and prevents broadcasting errors.
    """
    a_max = np.amax(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if not keepdims:
        if axis is not None:
            out = np.squeeze(out, axis=axis)
        else:
            out = float(np.squeeze(out))
    return out


class CustomGaussianHMM:
    """
    Custom implementation of Gaussian Hidden Markov Model in pure NumPy/SciPy.
    Performs all expectation-maximization (Baum-Welch) and decoding (Viterbi)
    calculations in log-space to prevent numerical underflow.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "diag",
        min_covar: float = 1e-3,
        n_iter: int = 100,
        tol: float = 1e-3,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        # Parameters to fit
        self.startprob_ = None
        self.transmat_ = None
        self.means_ = None
        self.covars_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def _init_params(self, X: np.ndarray):
        """Initializes HMM parameters using a Gaussian Mixture Model clustering baseline."""
        n_samples, n_features = X.shape

        # 1. Initialize transition probabilities and start probabilities to uniform + small noise
        startprob = np.ones(self.n_components) / self.n_components
        self.startprob_ = startprob / startprob.sum()

        transmat = np.ones((self.n_components, self.n_components)) / self.n_components
        # Add slight self-loop bias (persistence)
        transmat += np.eye(self.n_components) * 0.5
        self.transmat_ = transmat / transmat.sum(axis=1)[:, None]

        # 2. Use sklearn GMM to initialize means and covariances
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=1,
        )
        gmm.fit(X)
        self.means_ = gmm.means_
        self.covars_ = gmm.covariances_

        # Ensure covariance elements do not drop below min_covar
        if self.covariance_type == "diag":
            self.covars_ = np.maximum(self.covars_, self.min_covar)
        else:
            # For full covariance, add small epsilon to diagonal
            for i in range(self.n_components):
                diag_idx = np.diag_indices(n_features)
                self.covars_[i][diag_idx] = np.maximum(
                    self.covars_[i][diag_idx], self.min_covar
                )

    def _compute_log_emission_probs(self, X: np.ndarray) -> np.ndarray:
        """Computes log likelihood log b_i(x_t) for each state and timestep."""
        n_samples, n_features = X.shape
        log_emission = np.zeros((n_samples, self.n_components))

        for i in range(self.n_components):
            mean = self.means_[i]
            covar = self.covars_[i]

            if self.covariance_type == "diag":
                diff = X - mean
                log_density = -0.5 * (
                    n_features * np.log(2.0 * np.pi)
                    + np.log(covar).sum()
                    + (diff**2 / covar).sum(axis=1)
                )
                log_emission[:, i] = log_density
            else:
                log_density = scipy.stats.multivariate_normal.logpdf(
                    X, mean=mean, cov=covar
                )
                log_emission[:, i] = log_density

        return log_emission

    def _forward(self, log_emission: np.ndarray) -> Tuple[np.ndarray, float]:
        """Runs log-space Forward algorithm. Returns log_alpha and total log likelihood."""
        n_samples, n_states = log_emission.shape
        log_alpha = np.zeros((n_samples, n_states))

        log_startprob = np.log(self.startprob_ + 1e-15)
        log_transmat = np.log(self.transmat_ + 1e-15)

        log_alpha[0] = log_startprob + log_emission[0]

        for t in range(1, n_samples):
            v = log_alpha[t - 1][:, np.newaxis] + log_transmat
            log_alpha[t] = log_emission[t] + fast_logsumexp(v, axis=0)

        log_likelihood = fast_logsumexp(log_alpha[-1])
        return log_alpha, log_likelihood

    def _backward(self, log_emission: np.ndarray) -> np.ndarray:
        """Runs log-space Backward algorithm. Returns log_beta."""
        n_samples, n_states = log_emission.shape
        log_beta = np.zeros((n_samples, n_states))

        log_transmat = np.log(self.transmat_ + 1e-15)

        log_beta[-1] = 0.0

        for t in range(n_samples - 2, -1, -1):
            v = log_transmat + (log_emission[t + 1] + log_beta[t + 1])[np.newaxis, :]
            log_beta[t] = fast_logsumexp(v, axis=1)

        return log_beta

    def fit(
        self, X: np.ndarray, lengths: Optional[np.ndarray] = None
    ) -> "CustomGaussianHMM":
        """Fits the HMM model parameters using the Baum-Welch (EM) algorithm."""
        n_samples, n_features = X.shape
        self._init_params(X)

        prev_log_lik = -np.inf

        for iteration in range(self.n_iter):
            log_emission = self._compute_log_emission_probs(X)
            log_alpha, log_likelihood = self._forward(log_emission)
            log_beta = self._backward(log_emission)

            if self.verbose:
                logger.info(
                    f"Iteration {iteration:3d}: log likelihood = {log_likelihood:.4f}"
                )

            if np.abs(log_likelihood - prev_log_lik) < self.tol:
                break
            prev_log_lik = log_likelihood

            log_gamma = log_alpha + log_beta - log_likelihood
            gamma = np.exp(log_gamma)

            log_transmat = np.log(self.transmat_ + 1e-15)
            log_xi = np.zeros((n_samples - 1, self.n_components, self.n_components))

            for t in range(n_samples - 1):
                log_xi[t] = (
                    log_alpha[t][:, np.newaxis]
                    + log_transmat
                    + (log_emission[t + 1] + log_beta[t + 1])[np.newaxis, :]
                    - log_likelihood
                )

            self.startprob_ = gamma[0] / gamma[0].sum()

            sum_xi = np.exp(fast_logsumexp(log_xi, axis=0))
            self.transmat_ = sum_xi / (gamma[:-1].sum(axis=0)[:, None] + 1e-10)
            self.transmat_ /= self.transmat_.sum(axis=1)[:, None]

            for i in range(self.n_components):
                weight = gamma[:, i]
                sum_weight = weight.sum() + 1e-10
                self.means_[i] = (X * weight[:, None]).sum(axis=0) / sum_weight

                diff = X - self.means_[i]
                if self.covariance_type == "diag":
                    self.covars_[i] = (weight[:, None] * (diff**2)).sum(
                        axis=0
                    ) / sum_weight
                    self.covars_[i] = np.maximum(self.covars_[i], self.min_covar)
                else:
                    weighted_diff = diff * weight[:, None]
                    self.covars_[i] = np.dot(weighted_diff.T, diff) / sum_weight
                    diag_idx = np.diag_indices(n_features)
                    self.covars_[i][diag_idx] = np.maximum(
                        self.covars_[i][diag_idx], self.min_covar
                    )

        return self

    def predict(
        self, X: np.ndarray, lengths: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Decodes the most likely sequence of hidden states using the Viterbi algorithm."""
        n_samples, _ = X.shape
        log_emission = self._compute_log_emission_probs(X)

        log_startprob = np.log(self.startprob_ + 1e-15)
        log_transmat = np.log(self.transmat_ + 1e-15)

        log_V = np.zeros((n_samples, self.n_components))
        backpointer = np.zeros((n_samples, self.n_components), dtype=int)

        log_V[0] = log_startprob + log_emission[0]

        for t in range(1, n_samples):
            for j in range(self.n_components):
                probs = log_V[t - 1] + log_transmat[:, j]
                backpointer[t, j] = np.argmax(probs)
                log_V[t, j] = log_emission[t, j] + probs[backpointer[t, j]]

        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(log_V[-1])

        for t in range(n_samples - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]

        return states

    def predict_proba(
        self, X: np.ndarray, lengths: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculates the posterior state probabilities (smoothed probabilities)."""
        log_emission = self._compute_log_emission_probs(X)
        log_alpha, log_likelihood = self._forward(log_emission)
        log_beta = self._backward(log_emission)

        log_gamma = log_alpha + log_beta - log_likelihood
        return np.exp(log_gamma)

    def score(self, X: np.ndarray, lengths: Optional[np.ndarray] = None) -> float:
        """Computes the log likelihood of the input sequence."""
        log_emission = self._compute_log_emission_probs(X)
        _, log_likelihood = self._forward(log_emission)
        return float(log_likelihood)


# Unified Wrapper for GaussianHMM implementing the RegimeModelBase interface
class GaussianHMM(RegimeModelBase):
    """
    Unified HMM model class implementing the RegimeModelBase interface.
    Under the hood, instantiates hmmlearn.GaussianHMM if available,
    otherwise falls back to CustomGaussianHMM.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "diag",
        min_covar: float = 1e-3,
        n_iter: int = 100,
        tol: float = 1e-3,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.is_fitted = False

        if HMMLEARN_AVAILABLE:
            self.model = LibGaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                min_covar=min_covar,
                n_iter=n_iter,
                tol=tol,
                random_state=random_state,
                verbose=verbose,
            )
        else:
            self.model = CustomGaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                min_covar=min_covar,
                n_iter=n_iter,
                tol=tol,
                random_state=random_state,
                verbose=verbose,
            )

    def fit(self, X_train: np.ndarray) -> "GaussianHMM":
        self.model.fit(X_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting probabilities.")
        return self.model.predict_proba(X)

    def get_transition_matrix(self) -> np.ndarray:
        if not self.is_fitted:
            return np.eye(self.n_components)
        return self.model.transmat_

    def score(self, X: np.ndarray) -> float:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring.")
        return self.model.score(X)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"HMM model saved successfully to {path}")

    @classmethod
    def load(cls, path: str) -> "GaussianHMM":
        return joblib.load(path)

    def summarize(self) -> dict:
        if not self.is_fitted:
            return {"is_fitted": False}

        converged = True
        if HMMLEARN_AVAILABLE:
            converged = bool(getattr(self.model, "monitor_", None).converged)

        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "means": (
                self.model.means_.tolist() if self.model.means_ is not None else None
            ),
            "covariances": (
                self.model.covars_.tolist() if self.model.covars_ is not None else None
            ),
            "transition_matrix": self.get_transition_matrix().tolist(),
            "converged": converged,
        }
