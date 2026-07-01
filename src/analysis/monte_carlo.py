"""
Monte Carlo Simulation Engine for NIFTY 50 Regime-Aware Platform.
Supports Bootstrap, Regime-Conditioned, and Markov-chain path simulations
using either empirical bootstrap or fitted Student-t distributions.
Provides comprehensive drawdown distributions, tail-risk metrics (VaR/CVaR),
expected shortfall, and stress testing scenarios.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as std_t

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo path simulation engine for NIFTY 50 returns.
    Supports standard bootstrap, regime-conditioned, and Markov-chain transition simulations
    with empirical bootstrap and Student-t distribution settings.
    """

    def __init__(self, historical_df: pd.DataFrame, random_seed: Optional[int] = 42):
        """
        Args:
            historical_df: pd.DataFrame containing daily 'ret_simple',
                'regime_state' (integer state IDs), and 'regime_label' (state string names).
            random_seed: Optional seed for reproducibility.
        """
        self.df = historical_df.copy()

        # Calculate daily returns from raw closing prices to ensure they are unscaled
        if "raw_close" in self.df.columns:
            self.df["ret_simple"] = self.df["raw_close"].pct_change(
                fill_method=None
            )
        elif "close" in self.df.columns:
            self.df["ret_simple"] = self.df["close"].pct_change(fill_method=None)
        elif "ret_simple" not in self.df.columns:
            self.df["ret_simple"] = 0.0

        self.df["ret_simple"] = self.df["ret_simple"].fillna(0.0)
        self.returns = self.df["ret_simple"].values

        self.random_seed = random_seed
        self.rng = (
            np.random.RandomState(random_seed)
            if random_seed is not None
            else np.random.RandomState()
        )

        self.regime_returns: Dict[int, np.ndarray] = {}
        self.regime_t_params: Dict[int, Tuple[float, float, float]] = {}

        # Pre-fit distributions for all available regimes
        self._fit_regime_distributions()

    def _fit_regime_distributions(self):
        """Fits empirical returns and Student-t parameters for each regime."""
        self.regime_returns = {}
        self.regime_t_params = {}

        # Add global returns state (-1)
        self.regime_returns[-1] = self.returns
        try:
            df_t, loc, scale = std_t.fit(self.returns)
            self.regime_t_params[-1] = (df_t, loc, scale)
        except Exception:
            self.regime_t_params[-1] = (
                3.0,
                np.mean(self.returns),
                np.std(self.returns) + 1e-5,
            )

        # Add individual regime states
        if "regime_state" in self.df.columns:
            states = self.df["regime_state"].dropna().unique()
            for state in states:
                ret_vals = self.df[self.df["regime_state"] == state][
                    "ret_simple"
                ].values
                if len(ret_vals) == 0:
                    ret_vals = self.returns
                self.regime_returns[state] = ret_vals

                try:
                    if len(ret_vals) > 4:
                        df_t, loc, scale = std_t.fit(ret_vals)
                    else:
                        df_t, loc, scale = (
                            3.0,
                            np.mean(ret_vals),
                            np.std(ret_vals) + 1e-5,
                        )
                    self.regime_t_params[state] = (df_t, loc, scale)
                except Exception as e:
                    logger.warning(f"Failed to fit Student-t for state {state}: {e}")
                    self.regime_t_params[state] = (
                        3.0,
                        np.mean(ret_vals),
                        np.std(ret_vals) + 1e-5,
                    )

    def _sample_returns(
        self,
        state: int,
        size: Tuple[int, ...],
        distribution: str = "empirical",
        scenario: str = "none",
        stress_multiplier: float = 1.0,
    ) -> np.ndarray:
        """
        Samples returns for a given state, adjusting for the specified stress scenario.
        """
        ret_vals = self.regime_returns.get(state, self.returns)
        if ret_vals is None:
            ret_vals = self.returns

        # 1. Base sampling
        if distribution == "student_t":
            df_t, loc, scale = self.regime_t_params.get(
                state, (3.0, np.mean(ret_vals), np.std(ret_vals) + 1e-5)
            )
            samples = std_t.rvs(
                df=df_t, loc=loc, scale=scale, size=size, random_state=self.rng
            )
        else:
            # Empirical bootstrap
            samples = self.rng.choice(ret_vals, size=size, replace=True)

        # 2. Adjust for stress scenarios
        vol_multiplier = 1.0
        drift = 0.0

        if scenario == "covid":
            # COVID-like high vol spike
            vol_multiplier = 2.5 * stress_multiplier
        elif scenario == "selloff_2022":
            # 2022 Growth selloff: negative drift (-15% annualized) and 1.5x vol
            drift = -0.0006 * stress_multiplier
            vol_multiplier = 1.5 * stress_multiplier
        elif scenario == "inflation_risk_off":
            # Risk-off analog: negative drift (-25% annualized) and 2.0x vol
            drift = -0.0010 * stress_multiplier
            vol_multiplier = 2.0 * stress_multiplier

        # Apply volatility and drift adjustments
        mean_val = np.mean(samples)
        adjusted_samples = (samples - mean_val) * vol_multiplier + mean_val + drift

        # Ensure returns do not drop below -1.0 to prevent negative prices
        adjusted_samples = np.maximum(adjusted_samples, -0.999)
        return adjusted_samples

    def simulate_bootstrap(
        self,
        n_sims: int,
        horizon: int,
        distribution: str = "empirical",
        scenario: str = "none",
        stress_multiplier: float = 1.0,
    ) -> np.ndarray:
        """
        Standard bootstrap simulation.
        Draws random returns from all historical periods independently.
        Returns array of shape [horizon + 1, n_sims] representing cumulative returns paths starting at 1.0.
        """
        logger.info(
            f"Running Bootstrap Monte Carlo: {n_sims} paths, horizon {horizon} days, distribution={distribution}, scenario={scenario}"
        )
        simmed_returns = self._sample_returns(
            state=-1,
            size=(horizon, n_sims),
            distribution=distribution,
            scenario=scenario,
            stress_multiplier=stress_multiplier,
        )

        paths = np.ones((horizon + 1, n_sims))
        paths[1:] = np.cumprod(1.0 + simmed_returns, axis=0)
        return paths

    def simulate_regime_conditioned(
        self,
        n_sims: int,
        horizon: int,
        regime_state: int,
        distribution: str = "empirical",
        scenario: str = "none",
        stress_multiplier: float = 1.0,
    ) -> np.ndarray:
        """
        Regime-conditioned simulation.
        Draws returns only from the historical distribution of the specified regime state.
        """
        logger.info(
            f"Running Regime-Conditioned Monte Carlo (State {regime_state}): {n_sims} paths, horizon {horizon} days, distribution={distribution}, scenario={scenario}"
        )
        simmed_returns = self._sample_returns(
            state=regime_state,
            size=(horizon, n_sims),
            distribution=distribution,
            scenario=scenario,
            stress_multiplier=stress_multiplier,
        )

        paths = np.ones((horizon + 1, n_sims))
        paths[1:] = np.cumprod(1.0 + simmed_returns, axis=0)
        return paths

    def simulate_markov_chain(
        self,
        n_sims: int,
        horizon: int,
        start_state: int,
        transition_matrix: np.ndarray,
        distribution: str = "empirical",
        scenario: str = "none",
        stress_multiplier: float = 1.0,
    ) -> np.ndarray:
        """
        Markov-chain regime transition simulation.
        1. Simulates the sequence of regimes using the transition probability matrix.
        2. For each day, draws a return from the matching historical regime returns distribution.
        """
        logger.info(
            f"Running Markov-Chain transition Monte Carlo: {n_sims} paths, horizon {horizon} days, distribution={distribution}, scenario={scenario}"
        )
        n_states = transition_matrix.shape[0]

        # Stressed Transition matrix adjustment for inflation/risk-off
        matrix_stressed = transition_matrix.copy()
        if scenario == "inflation_risk_off":
            bearish_states = []
            for s in range(n_states):
                ret_vals = self.regime_returns.get(s, self.returns)
                if ret_vals is None:
                    ret_vals = self.returns
                if np.mean(ret_vals) < 0.0:
                    bearish_states.append(s)

            if bearish_states:
                for s in range(n_states):
                    bearish_sum = np.sum(matrix_stressed[s, bearish_states])
                    if bearish_sum < 0.8:
                        non_bearish_states = [
                            x for x in range(n_states) if x not in bearish_states
                        ]
                        matrix_stressed[s, non_bearish_states] *= 0.2
                        matrix_stressed[s, bearish_states] = (
                            1.0 - np.sum(matrix_stressed[s, non_bearish_states])
                        ) / len(bearish_states)

        # Pre-compute cumulative probabilities for fast transition simulation
        # Normalize transition matrix rows to guarantee sum to 1.0
        matrix_stressed_norm = matrix_stressed / np.sum(matrix_stressed, axis=1, keepdims=True)
        cum_transitions = np.cumsum(matrix_stressed_norm, axis=1)

        # Simulate state pathways for all simulations
        state_paths = np.zeros((horizon, n_sims), dtype=int)
        for sim in range(n_sims):
            current_state = start_state
            for t in range(horizon):
                r = self.rng.rand()
                # Fast lookup using searchsorted
                next_state = np.searchsorted(cum_transitions[current_state], r)
                # Cap in case of floating-point inaccuracies
                next_state = min(next_state, n_states - 1)
                state_paths[t, sim] = next_state
                current_state = next_state

        # Vectorized returns assignment based on simulated state occupancy masks
        sim_rets = np.zeros((horizon, n_sims))
        for state in range(n_states):
            mask = (state_paths == state)
            n_samples = int(np.sum(mask))
            if n_samples > 0:
                sampled_rets = self._sample_returns(
                    state=state,
                    size=(n_samples,),
                    distribution=distribution,
                    scenario=scenario,
                    stress_multiplier=stress_multiplier,
                )
                sim_rets[mask] = sampled_rets

        paths = np.ones((horizon + 1, n_sims))
        paths[1:] = np.cumprod(1.0 + sim_rets, axis=0)
        return paths

    def analyze_simulation_paths(
        self, paths: np.ndarray, dd_threshold: float = -0.15
    ) -> Dict[str, Any]:
        """
        Calculates descriptive return and drawdown statistics from simulation paths.

        Args:
            paths: np.ndarray of shape [horizon + 1, n_sims]
            dd_threshold: Float representing a custom drawdown threshold (default: -15%).

        Returns:
            Dict: Summary statistics of terminal returns, drawdowns, and tail-risk VaR/CVaR.
        """
        horizon_steps, n_sims = paths.shape
        terminal_prices = paths[-1, :]
        terminal_returns = terminal_prices - 1.0

        # Calculate expected (mean) and median returns
        expected_ret = float(np.mean(terminal_returns))
        median_ret = float(np.median(terminal_returns))

        # Probability of loss (terminal return < 0)
        prob_loss = float((terminal_returns < 0.0).sum() / n_sims)

        # Drawdowns across simulated paths
        running_max = np.maximum.accumulate(paths, axis=0)
        drawdowns = (paths - running_max) / (running_max + 1e-15)
        max_drawdown_per_path = drawdowns.min(axis=0)  # Shape: [n_sims]

        prob_dd_10 = float((max_drawdown_per_path < -0.10).sum() / n_sims)
        prob_dd_20 = float((max_drawdown_per_path < -0.20).sum() / n_sims)
        prob_dd_custom = float((max_drawdown_per_path < dd_threshold).sum() / n_sims)
        avg_max_dd = float(np.mean(max_drawdown_per_path))

        # Historical Value at Risk (VaR) & Conditional VaR (CVaR) of ending return
        sorted_returns = np.sort(terminal_returns)

        # 95% Percentiles
        var_idx_95 = int(n_sims * 0.05)
        var_95 = float(abs(sorted_returns[var_idx_95]))
        cvar_95 = float(abs(np.mean(sorted_returns[: var_idx_95 + 1])))

        # 99% Percentiles
        var_idx_99 = int(n_sims * 0.01)
        var_99 = float(abs(sorted_returns[var_idx_99]))
        cvar_99 = float(abs(np.mean(sorted_returns[: var_idx_99 + 1])))

        # Fan Chart Percentiles (along the path timeline axis=1)
        fan_5 = np.percentile(paths, 5, axis=1).tolist()
        fan_25 = np.percentile(paths, 25, axis=1).tolist()
        fan_50 = np.percentile(paths, 50, axis=1).tolist()
        fan_75 = np.percentile(paths, 75, axis=1).tolist()
        fan_95 = np.percentile(paths, 95, axis=1).tolist()

        return {
            "horizon_days": horizon_steps - 1,
            "paths_simulated": n_sims,
            "expected_return": expected_ret,
            "median_return": median_ret,
            "worst_5pct_return": -var_95,  # Backward compatible name for worst 5% return
            "best_5pct_return": float(np.percentile(terminal_returns, 95)),
            "probability_of_loss": prob_loss,
            "probability_of_drawdown_10pct": prob_dd_10,
            "probability_of_drawdown_20pct": prob_dd_20,
            "probability_of_drawdown_custom": prob_dd_custom,
            "average_simulated_max_drawdown": avg_max_dd,
            "Daily_VaR_95": var_95,
            "Daily_CVaR_95": cvar_95,
            "Daily_VaR_99": var_99,
            "Daily_CVaR_99": cvar_99,
            "Expected_Shortfall_95": cvar_95,
            "fan_5": fan_5,
            "fan_25": fan_25,
            "fan_50": fan_50,
            "fan_75": fan_75,
            "fan_95": fan_95,
            "max_drawdowns": max_drawdown_per_path.tolist(),
            "terminal_returns": terminal_returns.tolist(),
        }
