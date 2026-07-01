# Professional Resume Bullets & Interview Talking Points
## NIFTY 50 Latent Market Regime Discovery & Risk Intelligence Platform

This document compiles optimized descriptions of this project for resume inclusion and technical interview preparation.

---

## 1. Short Resume Bullet Version (For multi-page/dense resume templates)
*   **Engineered an unsupervised ML platform** for NIFTY 50 latent market regime discovery using custom log-space Gaussian Hidden Markov Models (HMM), Gaussian Mixture Models (GMM), and Markov Switching Regression, optimizing portfolio downside risk.
*   **Formulated a multi-scale feature pipeline of 40+ indicators** covering Parkinson range volatility, Garman-Klass intraday variance, Shannon entropy, Hurst exponent, and global macro covariates, verifying time-series stationarity via standard tests.
*   **Developed a zero-lookahead vectorized backtester and stochastic Monte Carlo path simulator**, validating a regime-aware asset allocation strategy that successfully reduced portfolio volatility by **50%** (8.91% vs 16.51%) and maximum drawdown by **35%** (-19.00% vs -38.44%) compared to Buy & Hold.

---

## 2. Detailed Resume Bullet Version (For single-page / project-focused resumes)
*   **Designed and built an end-to-end quantitative research platform** in Python for NIFTY 50 market regime detection using Gaussian HMMs, Gaussian Mixture Models (GMM), and Markov Switching Regression.
*   **Engineered 40+ dynamic indicators**, implementing extreme-value volatility estimators (Parkinson, Garman-Klass) and statistical complexity metrics (rolling Shannon entropy, Hurst exponent) to capture intraday noise and time-series persistence.
*   **Developed a custom HMM Baum-Welch and Viterbi decoder in log-space** using Log-Sum-Exp (LSE) math, resolving floating-point numerical underflow over long-horizon (3,000+ days) joint probability evaluations.
*   **Coded a vectorized backtesting engine and a transition-probability Monte Carlo path simulator** incorporating transaction costs (10 bps) and slippage models (5 bps). Backtested a tactical regime-aware strategy that reduced portfolio volatility by **50%** and drawdown by **35%** relative to the benchmark index.
*   **Deployed a premium glassmorphic Streamlit analytics dashboard** containing interactive controls for fee structures, dynamic model selection, PCA feature projections, and standalone HTML reporting tools.

---

## 3. Technical Interview Talking Points
When discussing this project in technical interviews, focus on these five core engineering and quantitative themes:

### A. Temporal Structure vs. Static Clustering
*   *Talking Point*: "K-Means and GMM assume returns on day $t$ are independent and identically distributed. However, markets exhibit volatility clustering. By modeling regimes as a Hidden Markov Model, we capture temporal transitions ($P(s_t \mid s_{t-1})$) and state persistence, resulting in stable classifications instead of noisy whipsaws."

### B. Prevention of Lookahead Bias
*   *Talking Point*: "To ensure statistical integrity and zero lookahead bias, we apply a strict 1-day lag to decoded states (`shift(1)`). The asset allocation weight applied to the portfolio on day $t$ is calculated solely using feature closing states decoded at the end of day $t-1$."

### C. Advanced Intraday Volatility Estimators
*   *Talking Point*: "Standard close-to-close volatility ignores intraday ranges. We engineered Parkinson range volatility and Garman-Klass variance. Parkinson integrates intraday high-low ranges, which is statistically more efficient, helping the unsupervised models identify structural shifts faster."

### D. Numerical Underflow Mitigation
*   *Talking Point*: "Evaluating long joint probabilities in Baum-Welch training and Viterbi decoding leads to float truncation and underflow. We implemented all operations in log-space, converting probability multiplications to additions, and resolved addition operations using Log-Sum-Exp (LSE) vector math."

### E. Realistic Performance Frictions
*   *Talking Point*: "To prove real-world viability, our vectorized backtester scales returns by subtracting execution friction. We model a 10 bps transaction fee and 5 bps execution slippage applied to the absolute delta of rebalancing weights, ensuring that the returns are not overclaimed or artificial."
