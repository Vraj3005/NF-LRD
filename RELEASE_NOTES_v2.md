# Release Notes (v2.0.0-stable)
## NIFTY 50 Latent Market Regime Discovery & Risk Intelligence Platform

This release upgrades the platform to a stable, production-ready, and recruiter-auditable state. It introduces structural enhancements, numerical stability, walk-forward expansion pipelines, and fully reproducible backtests.

---

## 🚀 What's New & Key Changes
*   **Pure NumPy HMM Fallback**: In production environments where compiling compiled C/C++ extensions for packages like `hmmlearn` fails, the system automatically falls back to our pure Python/NumPy log-space Gaussian Hidden Markov Model solver.
*   **Log-Space Calculations**: Rewrote all forward-backward and Viterbi decoding functions in log-space using the Log-Sum-Exp (LSE) formulation. This eliminates numerical underflow issues when multiplying sequential probabilities across 3,000+ trading days.
*   **Standardized API Interfaces**: Decoupled modeling, features, and risk pipelines. All models (`GaussianHMM`, `GaussianMixture`, `MarkovSwitchingRegression`) now adhere to a standard scikit-learn estimator interface (`fit`, `predict`, `predict_proba`).
*   **Interactive Analytics Dashboard**: Upgraded the Streamlit app with premium glassmorphic UI aesthetics, dynamic slider controls for transaction costs and slippage, and an interactive Report Export tab.

---

## 🔬 Methodology Upgrades
*   **Parkinson and Garman-Klass Intraday Volatility**: Replaced standard close-to-close returns variance with extreme-value estimators. Parkinson integrates intraday High-Low ranges, and Garman-Klass incorporates Open and Close price gaps. These estimators are significantly more statistically efficient, allowing the HMM to flag regime changes faster.
*   **Shannon Entropy and Hurst Exponent**: Formulated and engineered rolling complexity features. Shannon Entropy measures daily return random disorder, while the Hurst Exponent classifies whether the index is strongly trending ($H > 0.5$) or mean-reverting ($H < 0.5$).
*   **Diagonal Covariance Structure**: Adopted a `diag` covariance configuration across hidden states to minimize parameters ($O(D)$ parameters per state), eliminating overfitting during high-dimensional indicator sweeps.

---

## 🔬 Validation & Leakage Prevention
*   **Strict Zero-Lookahead Backtesting**: Position weights are computed on day $t-1$ and applied on day $t$ using the shifted signal sequence (`shift(1)`).
*   **Out-of-Sample (OOS) Timeline Division**:
    *   **In-Sample Train**: 2015-01-01 to 2021-12-31
    *   **Out-of-Sample Validation**: 2022-01-01 to 2023-12-31 (used for hyperparameter tuning)
    *   **Out-of-Sample Test**: 2024-01-01 to Present (completely unseen test holdout)
*   **Walk-Forward Training Pipeline**: Implemented an expanding training window validation. The HMM is refitted at 6-month steps (126 trading days) to capture structural breaks and compute realistic out-of-sample decay rates.
*   **Friction and Transaction Adjustment**: Daily returns incorporate 10 bps transaction fees and 5 bps execution slippage applied directly to absolute rebalancing shifts:
    $$\text{Cost}_t = (\text{Fee} + \text{Slippage}) \times |w_t - w_{t-1}|$$

---

## 📊 Final Performance Results
All performance figures are fully reproducible by running the command `make pipeline` or executing the scripts sequentially under the baseline transaction cost profile (10 bps fees, 5 bps slippage):

### 1. Walk-Forward Validation Metrics
*   **Buy & Hold**: CAGR = 9.92%, Volatility = 16.51%, Sharpe = 0.656, Max Drawdown = -38.44%, Calmar = 0.258, Turnover = 1.0x.
*   **EMA Crossover Baseline**: CAGR = 3.80%, Volatility = 12.66%, Sharpe = 0.358, Max Drawdown = -28.68%, Calmar = 0.133, Turnover = 32.0x.
*   **Volatility Targeting**: CAGR = 7.36%, Volatility = 13.51%, Sharpe = 0.593, Max Drawdown = -25.70%, Calmar = 0.286, Turnover = 43.5x.
*   **Regime-Aware Strategy**: CAGR = 6.52%, Volatility = 8.91%, Sharpe = 0.754, Max Drawdown = -19.00%, Calmar = 0.343, Turnover = 107.0x.
*   **Hybrid (Regime + Trend)**: CAGR = 4.86%, Volatility = 7.80%, Sharpe = 0.647, Max Drawdown = -15.29%, Calmar = 0.318, Turnover = 90.0x.

### 2. Out-of-Sample Performance Decay
| Strategy | In-Sample Train (2015-2021) CAGR | In-Sample Train Sharpe | OOS Validation (2022-2023) CAGR | OOS Validation Sharpe | OOS Test (2024-Pres) CAGR | OOS Test Sharpe |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Buy & Hold** | 11.20% | 0.692 | 11.21% | 0.827 | 3.99% | 0.352 |
| **EMA Baseline** | 4.82% | 0.446 | 6.35% | 0.696 | -0.96% | -0.032 |
| **Vol Target** | 9.18% | 0.719 | 10.91% | 0.898 | -0.13% | 0.053 |
| **Regime-Aware** | 10.42% | 0.866 | 7.27% | 0.655 | 2.01% | 0.234 |
| **Hybrid** | 5.75% | 0.588 | 6.15% | 0.739 | -0.75% | -0.023 |

---

## ⚠️ Limitations & Disclaimers
*   **Historical Simulation**: All backtest results are simulated and do not represent live trading performance.
*   **No Financial Advice**: This project is built solely for educational, presentation, and research purposes.
*   **Parameter Instability**: HMM parameters are sensitive to random initialization seeds and sample lengths. High-regime shifts or macroeconomic structural breaks can lead to classification decay.
*   **yfinance Data Quality**: Data is acquired from public Yahoo Finance sources, which are subject to adjustments, splits, and lack tick-level depth or order book spread metrics.
