# NF-LRD: Project Rebuild Audit & Engineering Plan
**NIFTY 50 Latent Market Regime Discovery & Risk Intelligence Platform**

---

## 📋 Executive Summary
This audit reviews the `NF-LRD` repository to evaluate its technical depth, mathematical integrity, code quality, and readiness for a senior-level quantitative research portfolio. 

While the project features a strong set of indicators, a clean custom HMM implementation, and an appealing dashboard layout, **several severe structural and mathematical flaws render its performance metrics invalid and its execution inefficient**. 

Most notably, the backtesting engine suffers from a **critical lookahead bias** (regime discovery is trained globally over the entire dataset before backtesting), and the computational pipeline is choked by **non-vectorized algorithms** (nested Python loops in the Baum-Welch training step and rolling regression loops in the Hurst exponent approximation).

This report outlines the discovered issues categorized by severity and proposes a concrete, step-by-step plan to transition this project from a standard academic script to a high-performance, lookahead-free, production-grade quantitative platform.

---

## 🚨 1. Critical Issues (Red Flags)

### 1.1 Lookahead Bias in Regime-Aware Backtesting (Data Leakage)
*   **Location**: `src/backtesting/strategy_rules.py` ([L54-L67](file:///c:/Desktop/NF_LRD/src/backtesting/strategy_rules.py#L54-L67)) & `run_modeling.py` ([L65-L78](file:///c:/Desktop/NF_LRD/run_modeling.py#L65-L78))
*   **The Issue**: The HMM, GMM, and Markov Switching models are fitted on the *entire historical feature matrix* $X$ (representing the full span from 2010 to present). Following this, states are decoded for the entire history using the global model parameters. In `strategy_rules.py`, the code applies `df['regime_label'].shift(1)` to construct strategy weights. 
*   **Why It's Critical**: Shifting the states series by 1 day does **not** remove lookahead bias. The model parameters (means, covariances, and transition matrix) were optimized using future data. Furthermore, the Viterbi algorithm is a global decoder: it calculates the most likely state sequence $S^* = \arg\max_{S} P(S, X | \lambda)$ by backward-recursing from the terminal timestep $T$. Consequently, the decoded state at day $t-1$ actively contains leaked information from day $t$ through the transition and emission variables.
*   **Impact**: The reported strategy returns are heavily inflated. A recruiter or quant researcher reviewing this code will immediately reject it as an "overfit lookahead backtest."
*   **Resolution**: Implement a true **Walk-Forward In-Sample/Out-of-Sample training framework**. The models must be trained on expanding or rolling windows (e.g., train on years 2010-2019, predict out-of-sample states for month $M$, advance the window by 1 month, and retrain). Alternatively, use the HMM's **Forward Filter Probabilities** ($\alpha_t$) to get the probability of being in a state at time $t$ using *only* features available up to time $t$, completely avoiding Viterbi's backward pass.

### 1.2 Computational Bottleneck: Baum-Welch Log Posterior Joint Probability ($\xi_t$)
*   **Location**: `src/models/hmm_model.py` ([L183-L194](file:///c:/Desktop/NF_LRD/src/models/hmm_model.py#L183-L194))
*   **The Issue**: The Baum-Welch (EM) step inside `CustomGaussianHMM` calculates the log joint posterior probabilities ($\xi_t$) using three nested loops:
    ```python
    for t in range(n_samples - 1):
        for i in range(self.n_components):
            for j in range(self.n_components):
                log_xi[t, i, j] = log_alpha[t, i] + log_transmat[i, j] + log_emission[t+1, j] + log_beta[t+1, j] - log_likelihood
    ```
*   **Why It's Critical**: For a dataset of $N=3,500$ rows and $K=4$ states, this loop executes $3,500 \times 4 \times 4 = 56,000$ iterations *per EM step*. Over $150$ iterations, this equates to **8.4 million iterations in pure Python**. This is why running tests and custom training on the Streamlit dashboard takes several minutes and freezes the terminal.
*   **Impact**: Extremely poor developer iteration speed, dashboard timeouts, and high CPU waste.
*   **Resolution**: Vectorize this calculation using NumPy broadcasting. The equation can be written as:
    ```python
    log_xi = (log_alpha[:-1, :, np.newaxis] + 
              log_transmat[np.newaxis, :, :] + 
              log_emission[1:, np.newaxis, :] + 
              log_beta[1:, np.newaxis, :]) - log_likelihood
    ```
    This single-line vectorized operation runs in milliseconds instead of seconds.

### 1.3 Computational Bottleneck: Hurst Exponent Rolling Polyfit
*   **Location**: `src/features/feature_pipeline.py` ([L52-L78](file:///c:/Desktop/NF_LRD/src/features/feature_pipeline.py#L52-L78))
*   **The Issue**: The Hurst exponent is calculated on a rolling window of 126 days using a sub-function `get_hurst` executed inside pandas `.rolling().apply(..., raw=True)`:
    ```python
    def get_hurst(window_arr):
        ...
        slope, _ = np.polyfit(log_lags, np.log(stdevs), 1)
        return slope
    ```
*   **Why It's Critical**: Pandas `.rolling().apply()` is a Python-level loop. Inside it, `np.polyfit` (which solves a least-squares linear regression under the hood) is called for *every single row* in Nifty's history (e.g. 3,500+ times). This creates a massive lag during the feature engineering pipeline.
*   **Impact**: Building features takes a long time, slowing down pipelines and dashboard responsiveness.
*   **Resolution**: Implement a fast vectorized Hurst exponent approximation (e.g., using rescaled range scaling or variance ratio tests without fitting in a loop) or calculate the Hurst exponent on a weekly/monthly basis instead of daily. Alternatively, use Numba (`@njit`) to compile the rolling polyfit logic.

### 1.4 Mathematical Instability in Daily Return Leverage (Bankruptcy Risk)
*   **Location**: `src/backtesting/backtester.py` ([L92](file:///c:/Desktop/NF_LRD/src/backtesting/backtester.py#L92)) & `strategy_rules.py` ([L17](file:///c:/Desktop/NF_LRD/src/backtesting/strategy_rules.py#L17))
*   **The Issue**: The backtester allows leverage up to 2.5x ("Bullish Low Volatility": 2.5). The strategy returns are calculated using a simple linear return formula: `(weights * asset_ret.values)`. 
*   **Why It's Critical**: In a vectorized backtest with high leverage, if NIFTY 50 suffers a daily drop of -45% (which can happen in extreme tail events or when trading is halted), a 2.5x leveraged long will experience a daily simple return of $-45\% \times 2.5 = -112.5\%$. Under a simple vectorized product:
    $$\text{Equity}_{t} = \text{Equity}_{t-1} \times (1 + r_t)$$
    The equity path becomes negative, which represents a bankrupt portfolio. However, the simulation allows it to keep trading, recovering from a negative equity balance. Additionally, there are no borrowing costs simulated on the leveraged portion in a standard way, nor margin maintenance or liquidation rules.
*   **Impact**: Unrealistic backtest returns in extreme market conditions.
*   **Resolution**: Implement a threshold check: if equity hits $\le 0$, the portfolio is liquidated (returns become $-100\%$ permanently). Limit leverage to a conservative 1.5x, or model leverage as a daily margin-rebalanced account that takes into account margin call limits.

---

## 🟡 2. High Priority Issues (Yellow Flags)

### 2.1 Monolithic Streamlit Dashboard Architecture
*   **Location**: `app/streamlit_app.py` ([L1-L1167](file:///c:/Desktop/NF_LRD/app/streamlit_app.py#L1))
*   **The Issue**: `streamlit_app.py` is a single file containing 1,167 lines of code. It mixes page routing, HTML/CSS layout configurations, data caching, inline Plotly graph builders, dynamic pipeline triggering, and raw pandas manipulations.
*   **Impact**: Violates basic clean code principles (Separation of Concerns). Hard to maintain, debug, or write tests for.
*   **Resolution**: Split `streamlit_app.py` into a clean multi-page or modular architecture:
    *   `app/streamlit_app.py` should only handle the sidebar navigation and layout.
    *   Individual views should be moved to separate files under `app/views/` (e.g., `backtesting_view.py`, `regime_discovery_view.py`).
    *   Chart configs should be fully encapsulated inside `src/visualization/charts.py`.

### 2.2 yfinance Ingestion Fragility & Lack of Caching
*   **Location**: `src/data/fetch_data.py` ([L50-L74](file:///c:/Desktop/NF_LRD/src/data/fetch_data.py#L50-L74))
*   **The Issue**: `fetch_ticker_data` calls `yf.download` directly. If Yahoo Finance returns empty data due to API limits, network disconnects, or ticker format changes, the pipeline fails silently or crashes. Furthermore, when users run the pipeline from the dashboard, it makes active network requests every time, which can trigger IP blocks.
*   **Impact**: High fragility during recruitment displays or offline reviews.
*   **Resolution**: 
    1. Wrap yfinance calls with `tenacity` retry logic.
    2. Provide a local fallback dataset (e.g. `data/sample/nifty_sample.csv` and `data/sample/vix_sample.csv`) that loads automatically if the network request fails, ensuring the pipeline *never* crashes.

### 2.3 Soft Dependency Pinning & Lack of Structured Environment
*   **Location**: `requirements.txt` ([L1-L17](file:///c:/Desktop/NF_LRD/requirements.txt#L1-L17)) & `pyproject.toml` ([L1-L22](file:///c:/Desktop/NF_LRD/pyproject.toml#L1-L22))
*   **The Issue**: Requirements are pinned using `package>=version`. This does not lock sub-dependencies, leading to "dependency drift" where a fresh install on a recruiter's machine might fetch newer versions of packages that break the API.
*   **Impact**: Low reproducibility.
*   **Resolution**: Implement a modern virtual environment configuration using `pyproject.toml` with strict pins. We will configure `uv` or `poetry` to lock all dependencies in `requirements.txt` or a `uv.lock`/`poetry.lock` format.

### 2.4 Silhouette Score Performance Bottleneck
*   **Location**: `src/models/model_selection.py` ([L117-L123](file:///c:/Desktop/NF_LRD/src/models/model_selection.py#L117-L123)) & `gmm_model.py` ([L71-L76](file:///c:/Desktop/NF_LRD/src/models/gmm_model.py#L71-L76))
*   **The Issue**: The silhouette score is calculated on the full feature matrix to evaluate clustering. Although there is a sample check (`len(X) > 5000`), calculating it on 3,000 points still takes time and slows down the hyperparameter grid search.
*   **Resolution**: Downsample the silhouette score input even further (e.g., to 1,000 samples) or use metrics like AIC/BIC as the primary decision variables.

---

## 🟢 3. Medium Priority Issues (Minor Flags)

### 3.1 Hardcoded Constants
*   **Location**: `src/backtesting/backtester.py` ([L80-L82](file:///c:/Desktop/NF_LRD/src/backtesting/backtester.py#L80-L82))
*   **The Issue**: Risk-free rates (5.5%) and borrowing costs (6.5%) are hardcoded inside the backtesting script.
*   **Resolution**: Move these constants to `config/settings.yaml` under a `backtest` block.

### 3.2 Overlapping / Redundant Pipeline Execution
*   **Location**: `app/streamlit_app.py` & Runner scripts
*   **The Issue**: Streamlit imports clean/feature modules and runs them inline, while `run_pipeline.py` duplicate imports them.
*   **Resolution**: Clean up imports and route all runs through unified functions in `src/`.

### 3.3 Lack of Model Versioning
*   **Location**: `run_modeling.py` ([L84](file:///c:/Desktop/NF_LRD/run_modeling.py#L84))
*   **The Issue**: Model binaries (`regime_model.joblib`) are overwritten. There is no trace of model metrics or parameters of older runs.
*   **Resolution**: Save model files using a timestamped scheme and maintain a lightweight JSON database for model tracking.

---

## 📐 4. Target Architecture

The refactored project will adhere to a clean separation of concerns:

```
NF-LRD/
│
├── .github/workflows/          # CI/CD configurations
│   └── lint_test.yml           # Automated linter and test suite
│
├── app/                        # Streamlit Web App Components
│   ├── streamlit_app.py        # Dashboard main runner (layout and routing)
│   └── views/                  # UI View components
│       ├── overview.py         # Dashboard welcome page
│       ├── explorer.py         # Data download & cleaning visualizer
│       ├── features.py         # Feature distribution & PCA plots
│       ├── regimes.py          # HMM/GMM model training & transition matrices
│       ├── risk.py             # VaR/CVaR tables & stress analysis
│       ├── simulator.py        # Monte Carlo pathway projections
│       └── backtest.py         # Vectorized backtests & performance metrics
│
├── config/
│   └── settings.yaml           # Central config (features, backtests, paths)
│
├── data/
│   ├── raw/                    # Raw CSVs from yfinance
│   ├── processed/              # Cleaned Parquet & standard feature matrices
│   └── sample/                 # Fallback offline datasets (Nifty/VIX)
│
├── models/
│   ├── reports/                # Model evaluation comparisons (AIC/BIC)
│   └── saved/                  # Trained Model binaries
│
├── src/                        # Platform Backend Modules
│   ├── data/
│   │   ├── fetch_data.py       # Data download with tenacity retries & caching
│   │   ├── clean_data.py       # Normalization and alignment of VIX
│   │   └── validation.py       # Strict logical data checks via Pydantic
│   ├── features/
│   │   ├── technical_indicators.py # Vectorized slopes, EMAs, momentum
│   │   ├── volatility_features.py  # Parkinson, Garman-Klass, ATR
│   │   └── feature_pipeline.py    # Standardized features & fast Hurst exponent
│   ├── models/
│   │   ├── hmm_model.py        # Log-space vectorized HMM (NumPy/SciPy)
│   │   ├── gmm_model.py        # GMM clustering wrapper
│   │   ├── markov_switching.py # Statsmodels MSR wrapper (with fallbacks)
│   │   └── model_selection.py  # AIC/BIC grid sweeps
│   ├── backtesting/
│   │   ├── backtester.py       # Walk-forward backtester & risk engine
│   │   └── strategy_rules.py   # Shifted rules & leverage constraints
│   ├── analysis/
│   │   ├── risk_metrics.py     # CAGR, Sharpe, CVaR, drawdown calculations
│   │   ├── monte_carlo.py      # Markov transition simulator (local random state)
│   │   └── explainability.py   # Linear model feature importance analysis
│   └── visualization/
│       └── charts.py           # Centralized Plotly charts generator
│
├── tests/                      # Automated Test Suite (pytest)
│   ├── test_data_pipeline.py
│   ├── test_features.py
│   ├── test_regime_models.py
│   ├── test_backtester.py
│   └── test_risk_metrics.py
│
├── pyproject.toml              # UV/Poetry environment specification
├── Dockerfile                  # Production deployment container
└── README.md                   # Recruiter-facing portfolio guide
```

---

## 🗺️ 5. Step-by-Step Rebuild Roadmap

### Phase 1: Environment & Config Refactoring
*   **Goal**: Ensure clean, reproducible setups for developers and recruiters.
*   **Tasks**:
    1. Rewrite `pyproject.toml` to list dependencies under `[project]` or install using `uv`.
    2. Centralize backtest parameters (fees, slippage, borrowing rates, risk-free interest rates) in `config/settings.yaml`.

### Phase 2: Ingestion & Feature Engineering Optimization
*   **Goal**: Increase pipeline reliability and eliminate performance bottlenecks.
*   **Tasks**:
    1. Implement tenacity retry logic inside `src/data/fetch_data.py`.
    2. Place offline Nifty and VIX datasets in `data/sample/` as manual fallbacks.
    3. Replace the slow polyfit-based rolling Hurst exponent calculation with an optimized, vectorized rescaled range algorithm.

### Phase 3: Vectorize HMM & Stabilize Model Sweeps
*   **Goal**: Speed up HMM training and prevent statistics models crashes.
*   **Tasks**:
    1. Vectorize the EM log joint posterior calculation in `CustomGaussianHMM` using NumPy broadcasting.
    2. Add fallback checks in `MarkovSwitchingModel` to prevent crash on non-convergence.
    3. Downsample feature inputs for Silhouette score calculations to 1,000 samples to speed up hyperparameter sweeps.

### Phase 4: Walk-Forward Backtester & Leverage Risk Controls
*   **Goal**: Solve lookahead bias and model realistic leverage limits.
*   **Tasks**:
    1. Rewrite the backtesting system to calculate strategy returns using a rolling/expanding walk-forward HMM retraining framework.
    2. Enforce absolute zero-lookahead by training models only on historical windows and applying the model to out-of-sample data.
    3. Implement margin liquidation rules: if equity drops below 0 on a leveraged day, freeze the strategy equity at -100%.

### Phase 5: Streamlit App Modularization
*   **Goal**: Refactor the monolithic dashboard code into clean, modular files.
*   **Tasks**:
    1. Create `app/views/` and split dashboard subpages into individual modules.
    2. Refactor `streamlit_app.py` to act as a routing controller.
    3. Decouple Plotly chart layouts and encapsulate all styling within `src/visualization/charts.py`.

### Phase 6: Code Quality, Testing, and Documentation
*   **Goal**: Verify all systems and update documentation with honest, recruiter-ready bullets.
*   **Tasks**:
    1. Verify that all unit tests in `tests/` pass with the new vectorized code.
    2. Update `README.md` to display honest, lookahead-free backtesting results.
    3. Add a section explaining the Walk-Forward validation methodology to demonstrate high technical competency.
