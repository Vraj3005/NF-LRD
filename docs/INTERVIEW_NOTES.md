# Technical Interview Study Guide & Notes
## NIFTY 50 Latent Market Regime Discovery & Risk Intelligence Platform

This document compiles technical questions and answers designed to prepare a CSE/CS student for quantitative research or machine learning engineer interviews, explaining the mathematical and engineering design choices behind this project.

---

### Q1: What is a Hidden Markov Model (HMM), and why is it preferred over standard static clustering algorithms (like K-Means or GMM) for market regime discovery?
**Answer**:
*   **Static Clustering (K-Means/GMM)**: Treats each data point (trading day) as independent and identically distributed (i.i.d.). It ignores the temporal order of events.
*   **Hidden Markov Model (HMM)**: Incorporates time-series structure by modeling market regimes as a latent (hidden) Markov chain. The state of the market on day $t$ depends on its state on day $t-1$ through a transition probability matrix:
    $$P(s_t = j \mid s_{t-1} = i) = A_{i,j}$$
*   **Why HMM Wins**: Financial markets exhibit **volatility clustering**—high-volatility days cluster together, as do low-volatility periods. HMM captures this temporal persistence, whereas GMM or K-Means produces noisy, fragmented regime classifications.

---

### Q2: How did you implement your HMM? What numerical issues did you encounter, and how did you resolve them?
**Answer**:
*   **Numerical Issue**: The standard Forward-Backward and Viterbi algorithms involve multiplying sequence probabilities over hundreds or thousands of steps. This leads to **numerical underflow**, where values fall below float precision limits and truncate to exactly $0.0$.
*   **Resolution**: We developed a custom log-space Baum-Welch and Viterbi solver. All probability multiplications are converted to additions in log-space:
    $$\log(P \times Q) = \log P + \log Q$$
    To add probabilities in log-space, we used the **Log-Sum-Exp (LSE)** formulation to prevent overflow/underflow:
    $$\log(e^x + e^y) = x + \log(1 + e^{y-x}) \quad \text{for } x \ge y$$
*   **Compilation Fallback**: When deployment on Windows encountered C++ compiler issues for compiling `hmmlearn`, our pure Python/NumPy log-space fallback ensured the platform remained cross-platform and deployable.

---

### Q3: How do you prevent Lookahead Bias in the backtesting engine?
**Answer**:
*   **The Hazard**: Lookahead bias occurs if the trading decision on day $t$ utilizes information that becomes available at or after the close of day $t$. For example, fitting the HMM on the entire dataset and using the decoded state at day $t$ to execute at day $t$'s open is a violation of lookahead constraints.
*   **The Resolution**: We implement a strict **1-day lag** on signals:
    $$w_t = f(s_{t-1})$$
    The weight $w_t$ allocated to the index on day $t$ is computed using the market regime state $s_{t-1}$ decoded at the close of day $t-1$. Daily returns are calculated as:
    $$R_{\text{strategy}, t} = w_t \times R_{\text{asset}, t} - (\text{Transaction Fees} + \text{Slippage}) \times |w_t - w_{t-1}|$$

---

### Q4: Explain the difference between Close-to-Close Volatility, Parkinson Volatility, and Garman-Klass Volatility.
**Answer**:
*   **Close-to-Close Volatility**: Standard historical volatility calculated as the standard deviation of daily close-to-close returns. It completely ignores intraday price movements.
*   **Parkinson Volatility**: An extreme-value volatility estimator that utilizes High ($H_t$) and Low ($L_t$) prices, capturing intraday range information:
    $$\sigma^2_{\text{Parkinson}} = \frac{1}{4 \ln 2} \ln\left(\frac{H_t}{L_t}\right)^2$$
*   **Garman-Klass Volatility**: An extension that incorporates Open ($O_t$), High, Low, and Close ($C_t$) prices to capture overnight gaps and intraday ranges:
    $$\sigma^2_{\text{GK}} = 0.5 \ln\left(\frac{H_t}{L_t}\right)^2 - (2\ln 2 - 1) \ln\left(\frac{C_t}{O_t}\right)^2$$
*   **Utility**: Parkinson and Garman-Klass provide more efficient and less noisy estimates of daily market variance than standard close-to-close returns, helping the HMM detect regime shifts faster.

---

### Q5: What is Stationarity, why does it matter for time-series ML, and how did your pipeline address it?
**Answer**:
*   **Stationarity**: A stationary time series has statistical properties (mean, variance, autocorrelation) that do not change over time.
*   **Why it Matters**: Machine learning models fit on non-stationary data (like raw asset prices, which trend upwards) fail out-of-sample because the model encounters values outside its training distribution bounds.
*   **Platform Address**:
    1. Raw price data is log-differenced into returns: $r_t = \ln(P_t / P_{t-1})$.
    2. Indicators are scaled relative to moving averages (e.g., price distance from SMA).
    3. We apply `StandardScaler` to normalize features (zero mean, unit variance) prior to HMM/GMM fitting.

---

### Q6: How does the Markov-chain Transition Monte Carlo Simulator generate path projections?
**Answer**:
1.  **Retrieve Starting State**: Identify the last historically observed regime state ($s_T$).
2.  **Stochastic Transition Path**: For each step $t \in [1, H]$ in the horizon, transition to state $s_t$ by drawing from the discrete probability distribution in row $s_{t-1}$ of the HMM transition matrix:
    $$s_t \sim A_{s_{t-1}, :}$$
3.  **Empirical Return Sampling**: Draw a return from the empirical historical distribution of returns belonging *specifically* to that simulated state $s_t$:
    $$R_t \sim \text{EmpiricalReturns}(S = s_t)$$
4.  **Compounding**: Compound the daily returns to generate the asset price path. This preserves both the temporal regime persistence (from the transition matrix) and the non-normal empirical fat tails (by bootstrapping historical returns).

---

### Q7: What are Value at Risk (VaR) and Conditional Value at Risk (CVaR)? How did you calculate them?
**Answer**:
*   **Value at Risk (VaR $\alpha$)**: The maximum expected loss at a given confidence level $\alpha$ over a specified horizon. For example, a daily VaR 95% of $-1.5\%$ means there is a 5% probability that the daily loss will exceed $1.5\%$.
*   **Conditional Value at Risk (CVaR $\alpha$ / Expected Shortfall)**: The expected loss given that the loss exceeds the VaR threshold (the average of the worst $(1-\alpha)$ returns).
*   **Implementation**: We implemented historical VaR and CVaR:
    $$\text{VaR}_{\alpha} = -\text{Percentile}(\text{Returns}, (1-\alpha) \times 100)$$
    $$\text{CVaR}_{\alpha} = -\text{Mean}(\{r \in \text{Returns} \mid r \le -\text{VaR}_{\alpha}\})$$

---

### Q8: Describe the Regime-Aware asset allocation strategy and why it outperforms standard trends.
**Answer**:
*   **Allocation Logic**: Position weights are dynamically scaled based on the HMM classified market regimes:
    *   *Bullish Low Volatility*: 100% position size (fully capture upward trend).
    *   *Recovery / Bullish High Vol*: 50% position size (moderate exposure due to higher variance).
    *   *Sideways Low Vol*: 25% position size (limit exposure in choppy ranges).
    *   *Bearish High Vol / Distribution / Risk-Off*: 0% position size (hold cash to protect principal).
*   **Outperformance Rationale**: Standard trend-following strategies (like EMA crossovers) suffer heavy whipsaws in choppy markets or sudden corrections because they react slowly. The HMM identifies shifts in volatility and statistical complexity rapidly, cutting exposure *before* a trend fully breaks, mitigating drawdown.

---

### Q9: Why is the choice of covariance type ('diag' vs 'full') important in Gaussian HMMs?
**Answer**:
*   **Diagonal Covariance ('diag')**: Assumes features are independent conditional on the hidden state. The covariance matrix for each state is diagonal. This significantly reduces the number of parameters to estimate, preventing overfitting, especially with high-dimensional features.
*   **Full Covariance ('full')**: Models dependencies between all pairs of features within each state. While more expressive, it requires estimating $O(D^2)$ parameters per state, which increases the likelihood of overfitting and numeric convergence failures.
*   **Project Decision**: We used `diag` covariance combined with standardizing, which provided the highest out-of-sample stability and numeric convergence.

---

### Q10: How do you evaluate the quality of an unsupervised clustering model when ground-truth labels are absent?
**Answer**:
1.  **Information Criteria (AIC/BIC)**: Measure the trade-off between model fit (likelihood) and complexity (number of parameters). We choose the number of states $K$ that minimizes the Bayesian Information Criterion (BIC) to prevent overfitting:
    $$\text{BIC} = -2\ln(\hat{L}) + p\ln(N)$$
2.  **Silhouette Score**: Measures how well-separated and dense the clusters are in the feature space.
3.  **Walk-Forward Validation**: Fit the model on training data and evaluate OOS stability. If the return and volatility delta between train and test distributions is minimal, the regime structure is stable and did not overfit.
