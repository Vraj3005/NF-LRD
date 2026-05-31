import logging
import numpy as np
import scipy.stats
from scipy.special import logsumexp
import joblib
from typing import Optional, Union, List

logger = logging.getLogger(__name__)

# Check if hmmlearn is available
try:
    from hmmlearn.hmm import GaussianHMM as LibGaussianHMM
    HMMLEARN_AVAILABLE = True
    logger.info("hmmlearn successfully imported. Using compiled library for GaussianHMM.")
except ImportError:
    HMMLEARN_AVAILABLE = False
    logger.warning("hmmlearn NOT available on this system. Falling back to pure Python/NumPy log-space GaussianHMM.")

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
        verbose: bool = False
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
            n_init=1
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
                self.covars_[i][diag_idx] = np.maximum(self.covars_[i][diag_idx], self.min_covar)

    def _compute_log_emission_probs(self, X: np.ndarray) -> np.ndarray:
        """Computes log likelihood log b_i(x_t) for each state and timestep."""
        n_samples, n_features = X.shape
        log_emission = np.zeros((n_samples, self.n_components))
        
        for i in range(self.n_components):
            mean = self.means_[i]
            covar = self.covars_[i]
            
            if self.covariance_type == "diag":
                # covar is 1D shape [n_features]
                # Log density for diagonal covariance Gaussian:
                diff = X - mean
                log_density = -0.5 * (
                    n_features * np.log(2.0 * np.pi) +
                    np.log(covar).sum() +
                    (diff ** 2 / covar).sum(axis=1)
                )
                log_emission[:, i] = log_density
            else:
                # covar is 2D shape [n_features, n_features]
                log_density = scipy.stats.multivariate_normal.logpdf(X, mean=mean, cov=covar)
                log_emission[:, i] = log_density
                
        return log_emission

    def _forward(self, log_emission: np.ndarray) -> Tuple[np.ndarray, float]:
        """Runs log-space Forward algorithm. Returns log_alpha and total log likelihood."""
        n_samples, n_states = log_emission.shape
        log_alpha = np.zeros((n_samples, n_states))
        
        log_startprob = np.log(self.startprob_ + 1e-15)
        log_transmat = np.log(self.transmat_ + 1e-15)
        
        # t = 0
        log_alpha[0] = log_startprob + log_emission[0]
        
        # t > 0
        for t in range(1, n_samples):
            for j in range(n_states):
                # logsumexp(log_alpha[t-1, i] + log_transmat[i, j])
                log_alpha[t, j] = log_emission[t, j] + logsumexp(log_alpha[t-1] + log_transmat[:, j])
                
        # Total log likelihood of observations
        log_likelihood = logsumexp(log_alpha[-1])
        return log_alpha, log_likelihood

    def _backward(self, log_emission: np.ndarray) -> np.ndarray:
        """Runs log-space Backward algorithm. Returns log_beta."""
        n_samples, n_states = log_emission.shape
        log_beta = np.zeros((n_samples, n_states))
        
        log_transmat = np.log(self.transmat_ + 1e-15)
        
        # t = T-1 (already initialized to 0 = log(1))
        log_beta[-1] = 0.0
        
        # t < T-1
        for t in range(n_samples - 2, -1, -1):
            for i in range(n_states):
                # logsumexp(log_transmat[i, j] + log_emission[t+1, j] + log_beta[t+1, j])
                log_beta[t, i] = logsumexp(log_transmat[i, :] + log_emission[t+1] + log_beta[t+1])
                
        return log_beta

    def fit(self, X: np.ndarray, lengths: Optional[np.ndarray] = None) -> "CustomGaussianHMM":
        """
        Fits the HMM model parameters using the Baum-Welch (EM) algorithm.

        Args:
            X: np.ndarray of shape [n_samples, n_features] containing features.
            lengths: Optional array of sub-sequence lengths (ignored, assuming single sequence).
        """
        n_samples, n_features = X.shape
        self._init_params(X)
        
        prev_log_lik = -np.inf
        
        for iteration in range(self.n_iter):
            # E-step: Forward-Backward
            log_emission = self._compute_log_emission_probs(X)
            log_alpha, log_likelihood = self._forward(log_emission)
            log_beta = self._backward(log_emission)
            
            if self.verbose:
                logger.info(f"Iteration {iteration:3d}: log likelihood = {log_likelihood:.4f}")
                
            # Convergence check
            if np.abs(log_likelihood - prev_log_lik) < self.tol:
                break
            prev_log_lik = log_likelihood
            
            # Compute posteriors (gammas) and joint posteriors (xi) in log-space
            log_gamma = log_alpha + log_beta - log_likelihood
            gamma = np.exp(log_gamma)
            
            log_transmat = np.log(self.transmat_ + 1e-15)
            log_xi = np.zeros((n_samples - 1, self.n_components, self.n_components))
            
            for t in range(n_samples - 1):
                for i in range(self.n_components):
                    for j in range(self.n_components):
                        log_xi[t, i, j] = (
                            log_alpha[t, i] + 
                            log_transmat[i, j] + 
                            log_emission[t+1, j] + 
                            log_beta[t+1, j] - 
                            log_likelihood
                        )
                        
            # M-step: Parameter updates
            # 1. Update startprob_
            self.startprob_ = gamma[0] / gamma[0].sum()
            
            # 2. Update transmat_
            sum_xi = np.exp(logsumexp(log_xi, axis=0))
            self.transmat_ = sum_xi / (gamma[:-1].sum(axis=0)[:, None] + 1e-10)
            self.transmat_ /= self.transmat_.sum(axis=1)[:, None] # Re-normalize
            
            # 3. Update means_
            for i in range(self.n_components):
                weight = gamma[:, i]
                sum_weight = weight.sum() + 1e-10
                self.means_[i] = (X * weight[:, None]).sum(axis=0) / sum_weight
                
                # 4. Update covars_
                diff = X - self.means_[i]
                if self.covariance_type == "diag":
                    self.covars_[i] = (weight[:, None] * (diff ** 2)).sum(axis=0) / sum_weight
                    self.covars_[i] = np.maximum(self.covars_[i], self.min_covar)
                else:
                    weighted_diff = diff * weight[:, None]
                    self.covars_[i] = np.dot(weighted_diff.T, diff) / sum_weight
                    # Add min_covar to diagonal
                    diag_idx = np.diag_indices(n_features)
                    self.covars_[i][diag_idx] = np.maximum(self.covars_[i][diag_idx], self.min_covar)
                    
        return self

    def predict(self, X: np.ndarray, lengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decodes the most likely sequence of hidden states using the Viterbi algorithm.
        """
        n_samples, _ = X.shape
        log_emission = self._compute_log_emission_probs(X)
        
        log_startprob = np.log(self.startprob_ + 1e-15)
        log_transmat = np.log(self.transmat_ + 1e-15)
        
        # log_V[t, j] = log probability of the most likely path to state j at time t
        log_V = np.zeros((n_samples, self.n_components))
        backpointer = np.zeros((n_samples, self.n_components), dtype=int)
        
        # t = 0
        log_V[0] = log_startprob + log_emission[0]
        
        # t > 0
        for t in range(1, n_samples):
            for j in range(self.n_components):
                # max_i (log_V[t-1, i] + log_transmat[i, j])
                probs = log_V[t-1] + log_transmat[:, j]
                backpointer[t, j] = np.argmax(probs)
                log_V[t, j] = log_emission[t, j] + probs[backpointer[t, j]]
                
        # Reconstruct optimal path
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(log_V[-1])
        
        for t in range(n_samples - 2, -1, -1):
            states[t] = backpointer[t+1, states[t+1]]
            
        return states

    def predict_proba(self, X: np.ndarray, lengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculates the posterior state probabilities (smoothed probabilities).
        """
        log_emission = self._compute_log_emission_probs(X)
        log_alpha, log_likelihood = self._forward(log_emission)
        log_beta = self._backward(log_emission)
        
        log_gamma = log_alpha + log_beta - log_likelihood
        return np.exp(log_gamma)

    def score(self, X: np.ndarray, lengths: Optional[np.ndarray] = None) -> float:
        """
        Computes the log likelihood of the input sequence.
        """
        log_emission = self._compute_log_emission_probs(X)
        _, log_likelihood = self._forward(log_emission)
        return log_likelihood

# Unified Wrapper for GaussianHMM
class GaussianHMM:
    """
    Factory wrapper class that instantiates lib-based hmmlearn.GaussianHMM 
    if available, otherwise instantiates CustomGaussianHMM.
    """
    def __new__(
        cls,
        n_components: int = 3,
        covariance_type: str = "diag",
        min_covar: float = 1e-3,
        n_iter: int = 100,
        tol: float = 1e-3,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        if HMMLEARN_AVAILABLE:
            # Map parameters to hmmlearn standard names
            return LibGaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                min_covar=min_covar,
                n_iter=n_iter,
                tol=tol,
                random_state=random_state,
                verbose=verbose
            )
        else:
            return CustomGaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                min_covar=min_covar,
                n_iter=n_iter,
                tol=tol,
                random_state=random_state,
                verbose=verbose
            )
