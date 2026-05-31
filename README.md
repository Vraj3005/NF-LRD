# NIFTY 50 Latent Market Regime Discovery & Risk Intelligence Platform

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![CI Status](https://github.com/your-username/NIFTY-50-Regime-Discovery/actions/workflows/lint_test.yml/badge.svg)](https://github.com/your-username/NIFTY-50-Regime-Discovery/actions)

An end-to-end quantitative research and tactical asset allocation platform that ingests historical NIFTY 50 price data, engineers 40+ advanced financial indicators, decodes hidden market regimes using unsupervised machine learning (HMM, GMM, Markov Switching), runs stochastically transitions Monte Carlo paths, and executes lookahead-free vectorized backtests.

---

## 🚀 Live Demo & Visuals
*   **Live Streamlit Dashboard**: [Link to Streamlit Community Cloud App (Placeholder)](https://share.streamlit.io/)
*   **Visual Interface Demo**:
    *   *[Insert Screenshot/GIF Placeholder of Dashboard Overview]*
    *   *[Insert Screenshot/GIF Placeholder of Backtesting Tab]*

---

## 📌 Problem Statement & Relevance
Financial asset returns do not follow stationary Gaussian bell curves. Markets shift dynamically between structural modes:
*   **Bullish low-volatility uptrends** where compound returns accumulate steadily.
*   **High-volatility corrective selloffs** where asset correlations converge to 1.0, destroying portfolio capital.
*   **Mean-reverting consolidation ranges** where standard trend-following systems suffer repeated whipsaws.

### Why Latent Market Regime Discovery Matters
By modeling market dynamics as a latent (hidden) Markov chain, we can identify these structural regime shifts in real-time. This allows portfolio managers to:
1.  **Dynamically scale capital exposure**: Enforce capital protection in high-risk states and maximize capture in bullish states.
2.  **Mitigate tail risks**: Avoid severe drawdowns (like the 2020 COVID crash) without sacrificing long-term returns.
3.  **Perform stress testing**: Run realistic simulations parameterized by historical state variances rather than simple gaussian models.

---

## 🏢 Platform Architecture

The system is built on a modular, decoupled structure:

```
                  ┌────────────────────────────────────────┐
                  │          yfinance Ingestion / CSV      │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │      Data Cleaning & Validation        │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    40+ Technical Feature Engineering   │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │       Unsupervised ML Training         │
                  │        (HMM / GMM / Markov Reg)        │
                  └───────────┬───────────────┬────────────┘
                              │               │
                              ▼               ▼
                 ┌───────────────────┐ ┌─────────────┐
                 │  Monte Carlo      │ │  Vectorised │
                 │  Projections      │ │  Backtester │
                 └───────────────────┘ └─────────────┘
```

---

## ⚙️ Feature Engineering Pipeline
We calculate 40+ features classified into five key groups:
1.  **Return Features**: Simple, log, and cumulative returns over 3d, 5d, 10d, and 21d windows.
2.  **Volatility Estimators**: Rolling standard deviation, Parkinson range volatility, Garman-Klass intraday variance, and Average True Range (ATR).
3.  **Trend Indicators**: Exponential and Simple Moving Average (SMA/EMA) slopes, price distances, and moving average crossover indicators.
4.  **Momentum Oscillators**: RSI 14, MACD lines, MACD Histograms, Williams %R, and Stochastic Oscillators.
5.  **Statistical Complexity**: Rolling Shannon Entropy (randomness measure) and Hurst Exponent (time-series persistence).

---

## 🤖 Latent Regime Modeling
The platform implements three candidate architectures:
*   **Gaussian Hidden Markov Model (HMM)**: A temporal state model utilizing a custom log-space Baum-Welch training and Viterbi decoder. Evaluates transition probability matrices.
*   **Gaussian Mixture Model (GMM)**: A non-temporal baseline clustering algorithm.
*   **Markov Switching Regression (MSR)**: A statsmodels autoregressive regime switcher modeled directly on returns.

Models are evaluated and selected using information criteria (**AIC/BIC**) to prevent parameter overfitting:
$$\text{BIC} = -2\log(\hat{L}) + p\log(N)$$

---

## ⚡ Backtesting & Risk Intelligence
The vectorized backtesting engine implements strict **zero lookahead bias** by shifting the decoded state sequences by 1 day (`shift(1)`).
Strategies evaluated:
*   **Buy & Hold**: Stays 100% long NIFTY 50.
*   **EMA Crossover**: Long index if EMA 50 > EMA 200, else exits to cash.
*   **Regime-Aware Strategy**: Dynamically adjusts position size (100% in Bullish Low Vol, 50% in recovery, 25% in sideways, 0% in risk-off/bearish).
*   **Hybrid Strategy**: Allocates only when both EMA trend is bullish AND the regime is Bullish Low Volatility.

### Performance Summary (2020 - 2026)
*   **Regime-Aware Strategy** reduced annualized volatility by **50%** (7% vs 14.32% for Buy & Hold) and cut maximum drawdown by **35%** (-11.15% vs -17.23%), maintaining a strong Sharpe ratio of **0.903**.
*   **2022 Growth Selloff Stress Test**: Buy & Hold lost **-9.20%** with a max drawdown of **-16.47%**. The Regime-Aware strategy limited losses to **-4.08%** with a drawdown of only **-6.06%**.

---

## 📦 Installation & Setup

### Prerequisite: Python 3.11+
Clone the repository and initialize the virtual environment:

```powershell
# 1. Clone the project
git clone https://github.com/your-username/NIFTY-50-Regime-Discovery.git
cd NIFTY-50-Regime-Discovery

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install required libraries
pip install -r requirements.txt
```

---

## 💻 Local Execution Guide

Execute the CLI pipeline scripts or boot the web dashboard directly:

```powershell
# Run the entire data & modeling pipeline
python run_pipeline.py
python run_modeling.py
python run_backtesting.py

# Execute the test suite
pytest -v

# Run the Streamlit interactive dashboard
streamlit run app/streamlit_app.py
```

---

## ☁️ Deployment on Streamlit Community Cloud
1.  Push the project code to a public GitHub repository.
2.  Login to [Streamlit Share](https://share.streamlit.io/) using your GitHub account.
3.  Click **New App**, select your repository, branch, and set the entrypoint path to `app/streamlit_app.py`.
4.  Click **Deploy**! Streamlit automatically reads `requirements.txt` and compiles the environment.

---

## 💼 Resume & Interview Prep

### Key Resume Bullets
*   **Built an end-to-end unsupervised time-series ML research platform** for NIFTY 50 latent market regime discovery using custom log-space Baum-Welch HMMs, Gaussian Mixture Models, and Markov Switching Regression, optimizing portfolio downside risk.
*   **Engineered 40+ multi-scale financial features** covering Parkinson and Garman-Klass range volatilities, momentum oscillators (RSI, MACD), statistical complexities (Shannon entropy, Hurst exponent), and global macro covariates.
*   **Developed a zero-lookahead vectorized backtester and Monte Carlo path simulator**, executing transition simulations and stress testing that validated a regime-shifting strategy reducing portfolio volatility by **50%** and drawdown by **35%**.
*   **Deployed an interactive Streamlit analytics dashboard** containing dynamic model training, correlation matrices, 2D PCA feature visualizations, and downloadable HTML reporting tools.

### Sample Interview Questions
*   **Q**: *Why use HMM over standard GMM/K-Means?*  
    **A**: Financial markets exhibit volatility clustering. Static algorithms ignore temporal order, whereas HMMs model state transitions ($P(s_t|s_{t-1})$), capturing this persistence.
*   **Q**: *How do you prevent lookahead bias in backtesting?*  
    **A**: We apply a strict 1-day lag (`shift(1)`) on regime signals. The position weight applied on day $t$ is calculated using indicators and states decoded at the close of day $t-1$.

---

## ⚠️ Disclaimer
This platform is built strictly for academic, research, and portfolio demonstration purposes. All backtests and projections are simulated and do not constitute professional financial advice.
