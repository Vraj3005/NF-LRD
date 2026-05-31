# Scientific Research Report: NIFTY 50 Latent Market Regime Discovery & Risk Intelligence Platform

## Abstract
This report details the implementation of an end-to-end quantitative framework that cleans and validates historical daily index data, engineers high-dimensional features, decodes latent market regimes using unsupervised machine learning, and executes dynamic tactical allocations. We verify the HMM parameters, backtest strategy rules, and simulate price horizons.

---

## 1. Introduction & Methodology
Traditional financial models often assume that asset return distributions are stationary, log-normal, and homoscedastic. However, empirical stock returns exhibit time-varying volatility (heteroscedasticity) and fat tails (leptokurtosis). We propose an unsupervised time-series approach to identify hidden market states (regimes) that capture shifts in return expectations and volatility levels dynamically.

Our methodology follows five strict stages:
1.  **Ingestion & Data Validation**: Multi-source daily index ingestion with logic checks.
2.  **Multi-Scale Feature Engineering**: 40+ trend, range volatility, momentum, complexity, and macroeconomic features.
3.  **Latent State Decoders**: Gaussian HMM, GMM, and Markov Switching Regression models.
4.  **Risk Intelligence & Stress Testing**: Regime-decomposed VaR/CVaR calculations and crisis evaluations.
5.  **Lookahead-Free Vectorized Backtesting**: Tactical position-weight allocation.

---

## 2. Mathematical Feature Formulation
We engineer five distinct feature groups from raw OHLCV prices:

### 2.1 Return Estimators
We compute simple return, log return, and rolling cumulative returns across multi-day horizons ($k \in [3, 5, 10, 21]$):
$$R_{\text{simple}, t} = \frac{P_t}{P_{t-1}} - 1$$
$$R_{\text{log}, t} = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

### 2.2 Volatility Features
*   **Parkinson Volatility**: Uses intraday High ($H_t$) and Low ($L_t$) prices, providing higher statistical efficiency than standard close-to-close measures:
    $$\sigma^2_{\text{Parkinson}} = \frac{1}{4 \ln 2} \ln\left(\frac{H_t}{L_t}\right)^2$$
*   **Garman-Klass Volatility**: Integrates Open ($O_t$) and Close ($C_t$) prices to capture overnight gap returns and intraday ranges:
    $$\sigma^2_{\text{GK}} = 0.5 \ln\left(\frac{H_t}{L_t}\right)^2 - (2\ln 2 - 1) \ln\left(\frac{C_t}{O_t}\right)^2$$

### 2.3 Statistical Complexity Features
*   **Shannon Entropy**: Measures the randomness or disorder in the distribution of daily returns over a rolling window ($W=63$):
    $$H(X) = -\sum_{i} P(x_i) \log_2 P(x_i)$$
*   **Hurst Exponent**: Measures the long-term memory of a time series. A Hurst exponent $H \approx 0.5$ represents a random walk, $H > 0.5$ represents a trending series, and $H < 0.5$ indicates mean-reverting behavior:
    $$\text{E}\left[\frac{R(d)}{S(d)}\right] = C d^H \quad \text{as } d \to \infty$$

---

## 3. Unsupervised Regime Decoding
We model market regimes using a Gaussian Hidden Markov Model (HMM) parameterised by:
*   Initial state distribution: $\pi_i = P(s_1 = i)$
*   State transition probability matrix: $A_{i,j} = P(s_t = j \mid s_{t-1} = i)$
*   Emission probability distributions: $b_i(x_t) = P(x_t \mid s_t = i) \sim \mathcal{N}(\mu_i, \Sigma_i)$

We decode the most likely sequence of hidden states $S^* = (s_1^*, s_2^*, \ldots, s_T^*)$ using the **Viterbi Algorithm**:
$$V_{t,j} = \max_{i} \left( V_{t-1,i} A_{i,j} \right) b_j(x_t)$$
$$s_T^* = \arg\max_{i} V_{T,i}$$

To resolve numerical underflow, all computations are executed in log-space using the **Log-Sum-Exp** formulation.

---

## 4. Backtesting & Portfolio Risk Allocation
The Regime-Aware Strategy maps the decoded state $s_{t-1}$ to portfolio weights $w_t$ applied on day $t$.
$$w_t = \begin{cases} 
1.0 & \text{if } s_{t-1} \text{ is Bullish Low Volatility} \\
0.5 & \text{if } s_{t-1} \text{ is Recovery or Bullish High Volatility} \\
0.25 & \text{if } s_{t-1} \text{ is Sideways Low Volatility} \\
0.0 & \text{if } s_{t-1} \text{ is Bearish High Vol / Distribution / Risk-Off} 
\end{cases}$$

Daily strategy returns are computed as:
$$R_{\text{strategy}, t} = w_t R_{\text{asset}, t} - (\text{Fee} + \text{Slippage}) \times |w_t - w_{t-1}|$$

### Performance Metrics (October 2020 - May 2026)
*   **Buy & Hold Benchmark**: CAGR of **13.18%**, Volatility of **14.32%**, Sharpe Ratio of **0.936**, and Max Drawdown of **-17.23%**.
*   **Regime-Aware Strategy**: CAGR of **6.27%**, Volatility of **7.00%**, Sharpe Ratio of **0.903**, and Max Drawdown of **-11.15%**.

By cutting equity exposure during high-risk regimes, the Regime-Aware strategy successfully reduced portfolio volatility by **50%** and maximum drawdown by **35%**, yielding a stable risk-adjusted return profile.

---

## 5. Limitations & Future Scope
1.  **Stationarity Assumptions**: Scaling ensures stationarity, but the relationships between indicators and regimes may shift over multi-decade cycles (structural breaks).
2.  **Estimation Risk**: Standard Baum-Welch EM estimation can get trapped in local optima. Implementing Bayesian priors (Bayesian HMMs) will increase state estimation robustness.
3.  **Macro Covariates Expansion**: Integrating higher-frequency macro data (e.g. overnight interest rate swaps, daily capital flows) can enhance state transition predictions.
