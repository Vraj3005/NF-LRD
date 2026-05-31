import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """
    Monte Carlo path simulation engine for NIFTY 50 returns.
    Supports standard bootstrap, regime-conditioned, and Markov-chain transition simulations.
    """
    def __init__(
        self,
        historical_df: pd.DataFrame,
        random_seed: Optional[int] = 42
    ):
        """
        Args:
            historical_df: pd.DataFrame containing daily 'ret_simple' (or 'close' returns),
                'regime_state' (integer state IDs), and 'regime_label' (state string names).
            random_seed: Optional seed for reproducibility.
        """
        self.df = historical_df.copy()
        
        # Calculate daily returns if not already present
        if 'ret_simple' not in self.df.columns:
            if 'raw_close' in self.df.columns:
                self.df['ret_simple'] = self.df['raw_close'].pct_change(fill_method=None)
            else:
                self.df['ret_simple'] = self.df['close'].pct_change(fill_method=None)
                
        self.df['ret_simple'] = self.df['ret_simple'].fillna(0.0)
        self.returns = self.df['ret_simple'].values
        
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate_bootstrap(
        self, 
        n_sims: int, 
        horizon: int
    ) -> np.ndarray:
        """
        Standard bootstrap simulation.
        Draws random returns from all historical periods independently.
        Returns array of shape [horizon + 1, n_sims] representing cumulative returns paths starting at 1.0.
        """
        logger.info(f"Running Bootstrap Monte Carlo: {n_sims} paths, horizon {horizon} days...")
        # Draw returns: shape [horizon, n_sims]
        simmed_returns = np.random.choice(self.returns, size=(horizon, n_sims), replace=True)
        
        # Compute cumulative paths starting at 1.0
        paths = np.ones((horizon + 1, n_sims))
        paths[1:] = np.cumprod(1.0 + simmed_returns, axis=0)
        return paths

    def simulate_regime_conditioned(
        self, 
        n_sims: int, 
        horizon: int, 
        regime_state: int
    ) -> np.ndarray:
        """
        Regime-conditioned simulation.
        Draws returns only from the historical distribution of the specified regime state.
        """
        logger.info(f"Running Regime-Conditioned Monte Carlo (Regime {regime_state}): {n_sims} paths, horizon {horizon} days...")
        
        # Filter returns for specific regime
        regime_returns = self.df[self.df['regime_state'] == regime_state]['ret_simple'].values
        if len(regime_returns) == 0:
            logger.warning(f"Regime {regime_state} has no historical days. Falling back to global returns.")
            regime_returns = self.returns
            
        simmed_returns = np.random.choice(regime_returns, size=(horizon, n_sims), replace=True)
        
        paths = np.ones((horizon + 1, n_sims))
        paths[1:] = np.cumprod(1.0 + simmed_returns, axis=0)
        return paths

    def simulate_markov_chain(
        self, 
        n_sims: int, 
        horizon: int, 
        start_state: int, 
        transition_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Markov-chain regime transition simulation.
        1. Simulates the sequence of regimes using the transition probability matrix.
        2. For each day, draws a return from the matching historical regime returns distribution.
        """
        logger.info(f"Running Markov-Chain transition Monte Carlo: {n_sims} paths, horizon {horizon} days...")
        n_states = transition_matrix.shape[0]
        
        # Group historical returns by regime for fast sampling
        regime_returns_dict = {}
        for state in range(n_states):
            ret_vals = self.df[self.df['regime_state'] == state]['ret_simple'].values
            if len(ret_vals) == 0:
                ret_vals = self.returns  # Fallback
            regime_returns_dict[state] = ret_vals
            
        paths = np.ones((horizon + 1, n_sims))
        
        for sim in range(n_sims):
            current_state = start_state
            # Track daily returns for this simulation path
            sim_rets = np.zeros(horizon)
            
            for t in range(horizon):
                # 1. Transition to next state based on current state's probabilities
                transition_probs = transition_matrix[current_state]
                next_state = np.random.choice(n_states, p=transition_probs)
                current_state = next_state
                
                # 2. Draw return from the next state's return list
                sim_rets[t] = np.random.choice(regime_returns_dict[next_state])
                
            paths[1:, sim] = np.cumprod(1.0 + sim_rets)
            
        return paths

    def analyze_simulation_paths(self, paths: np.ndarray) -> Dict[str, Any]:
        """
        Calculates descriptive return and drawdown statistics from simulation paths.
        
        Args:
            paths: np.ndarray of shape [horizon + 1, n_sims]
            
        Returns:
            Dict: Summary statistics of terminal returns and drawdowns.
        """
        horizon_steps, n_sims = paths.shape
        terminal_prices = paths[-1, :]
        terminal_returns = terminal_prices - 1.0
        
        # Calculate expected (mean) and median returns
        expected_ret = float(np.mean(terminal_returns))
        median_ret = float(np.median(terminal_returns))
        
        # Percentiles
        var_5pct = float(np.percentile(terminal_returns, 5))
        pct_95 = float(np.percentile(terminal_returns, 95))
        
        # Probability of loss (terminal return < 0)
        prob_loss = float((terminal_returns < 0.0).sum() / n_sims)
        
        # Drawdowns across simulated paths
        # To compute max drawdown per path, we need running max along axis=0
        running_max = np.maximum.accumulate(paths, axis=0)
        drawdowns = (paths - running_max) / (running_max + 1e-15)
        max_drawdown_per_path = drawdowns.min(axis=0) # Shape: [n_sims]
        
        prob_dd_10 = float((max_drawdown_per_path < -0.10).sum() / n_sims)
        prob_dd_20 = float((max_drawdown_per_path < -0.20).sum() / n_sims)
        avg_max_dd = float(np.mean(max_drawdown_per_path))
        
        return {
            "horizon_days": horizon_steps - 1,
            "paths_simulated": n_sims,
            "expected_return": expected_ret,
            "median_return": median_ret,
            "worst_5pct_return": var_5pct,
            "best_5pct_return": pct_95,
            "probability_of_loss": prob_loss,
            "probability_of_drawdown_10pct": prob_dd_10,
            "probability_of_drawdown_20pct": prob_dd_20,
            "average_simulated_max_drawdown": avg_max_dd
        }
