import numpy as np
import pandas as pd
import pytest
from src.models.hmm_model import GaussianHMM, CustomGaussianHMM
from src.models.gmm_model import GMMRegimeModel
from src.models.markov_switching import MarkovSwitchingModel
from src.analysis.regime_analysis import compute_transition_matrix, assign_regime_labels, analyze_regimes

@pytest.fixture
def dummy_features():
    """Generates synthetic scaled features and close prices for testing models."""
    np.random.seed(42)
    n = 300
    n_feats = 5
    
    # Scale features
    X = np.random.normal(0, 1, (n, n_feats))
    
    # Introduce some state structures: first 100 high returns, next 100 negative, last 100 zero
    X[:100, 0] += 2.0
    X[100:200, 0] -= 2.0
    
    dates = pd.date_range(start='2026-01-01', periods=n, freq='D')
    
    # Create simple returns matching the states structure
    ret = np.random.normal(0.0005, 0.005, n)
    ret[:100] += 0.015    # Bullish state
    ret[100:200] -= 0.02  # Bearish state
    
    close = 100.0 * (1.0 + ret).cumprod()
    
    df = pd.DataFrame({
        'date': dates,
        'close': close,
        'open': close * 0.99,
        'high': close * 1.01,
        'low': close * 0.98,
        'volume': np.random.randint(100, 1000, n).astype(float)
    })
    
    # Set scaled feature columns
    for i in range(n_feats):
        df[f'feat_{i}'] = X[:, i]
        
    return df

def test_gmm_convergence(dummy_features):
    """Tests that GMM fits, predicts, and evaluates successfully."""
    X = dummy_features[['feat_0', 'feat_1', 'feat_2']].values
    
    model = GMMRegimeModel(n_components=3, random_state=42)
    model.fit(X)
    
    assert model.is_fitted is True
    
    states = model.predict(X)
    assert len(states) == len(dummy_features)
    assert set(states).issubset({0, 1, 2})
    
    eval_metrics = model.evaluate(X)
    assert "aic" in eval_metrics
    assert "bic" in eval_metrics
    assert "silhouette_score" in eval_metrics
    assert eval_metrics["silhouette_score"] > -1.0

def test_custom_hmm_execution(dummy_features):
    """Tests that our custom log-space HMM executes, converges, and decodes states."""
    X = dummy_features[['feat_0', 'feat_1', 'feat_2']].values
    
    # Explicitly test CustomGaussianHMM to ensure our log-space code is verified
    model = CustomGaussianHMM(n_components=3, random_state=42, n_iter=20, tol=1e-3)
    model.fit(X)
    
    assert model.startprob_ is not None
    assert model.transmat_ is not None
    assert model.means_ is not None
    assert model.covars_ is not None
    
    # Transition probabilities must sum to 1.0
    assert np.allclose(model.transmat_.sum(axis=1), 1.0)
    # Start probability must sum to 1.0
    assert np.isclose(model.startprob_.sum(), 1.0)
    
    # Predict hidden states (Viterbi decoding)
    states = model.predict(X)
    assert len(states) == len(X)
    assert set(states).issubset({0, 1, 2})
    
    # Posteriors
    post = model.predict_proba(X)
    assert post.shape == (len(X), 3)
    assert np.allclose(post.sum(axis=1), 1.0)
    
    # Scoring
    score = model.score(X)
    assert isinstance(score, float)
    assert not np.isnan(score)

def test_transition_probability_matrix():
    """Tests empirical transition probability matrix calculation from states."""
    states = np.array([0, 0, 1, 1, 1, 2, 2, 0, 1, 2])
    matrix = compute_transition_matrix(states, n_components=3)
    
    assert matrix.shape == (3, 3)
    # Rows must sum to 1.0
    assert np.allclose(matrix.sum(axis=1), 1.0)
    
    # Specific transition verification
    # State 0 transitions: 0->0, 0->1, 0->1. Total transitions from 0 is 3.
    # 0->0 happens 1 time (prob = 1/3)
    # 0->1 happens 2 times (prob = 2/3)
    # 0->2 happens 0 times (prob = 0)
    assert np.isclose(matrix[0, 0], 1.0/3.0)
    assert np.isclose(matrix[0, 1], 2.0/3.0)

def test_regime_labeling_logic():
    """Tests the heuristic auto-labeling mappings for different regime configurations."""
    # Setup mock statistics for 3 states
    regime_stats = [
        # State 0: Negative return, high vol
        {"annualized_return": -0.15, "annualized_volatility": 0.22, "sharpe_ratio": -0.68, "regime_state": 0},
        # State 1: Flat return, low vol
        {"annualized_return": 0.01, "annualized_volatility": 0.08, "sharpe_ratio": 0.12, "regime_state": 1},
        # State 2: High return, low vol
        {"annualized_return": 0.18, "annualized_volatility": 0.12, "sharpe_ratio": 1.5, "regime_state": 2}
    ]
    
    labels = assign_regime_labels(regime_stats)
    assert len(labels) == 3
    
    # State 0 should be Bearish
    assert "Bearish" in labels[0] or "Risk-Off" in labels[0]
    # State 1 should be Sideways
    assert "Sideways" in labels[1]
    # State 2 should be Bullish Low Volatility
    assert "Bullish Low Volatility" == labels[2]

def test_regime_analysis_module(dummy_features):
    """Tests integration of analyze_regimes and return statistics."""
    X = dummy_features[['feat_0', 'feat_1', 'feat_2']].values
    
    model = GMMRegimeModel(n_components=3, random_state=42)
    model.fit(X)
    states = model.predict(X)
    
    labeled_df, summary_df, trans_df = analyze_regimes(dummy_features, states, n_components=3)
    
    assert 'regime_state' in labeled_df.columns
    assert 'regime_label' in labeled_df.columns
    assert len(summary_df) == 3
    assert 'regime_label' in summary_df.columns
    assert trans_df.shape == (3, 3)

def test_markov_switching_baseline(dummy_features):
    """Tests statsmodels Markov Switching Regression wrapper works."""
    ret = dummy_features['close'].pct_change().fillna(0.0).values
    
    model = MarkovSwitchingModel(n_components=2)
    model.fit(ret)
    
    # Since MarkovRegression fit might fail to converge on random data,
    # we verify that the wrapper doesn't crash and returns valid predictions
    states = model.predict(ret)
    assert len(states) == len(ret)
    
    eval_metrics = model.evaluate()
    assert "log_likelihood" in eval_metrics
