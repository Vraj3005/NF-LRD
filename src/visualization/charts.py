import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Tuple, Optional

def plot_equity_curves(equity_df: pd.DataFrame) -> go.Figure:
    """Generates an interactive Plotly chart comparing portfolio equity curves."""
    fig = go.Figure()
    
    # Identify equity columns
    equity_cols = [c for c in equity_df.columns if c.startswith('equity_')]
    
    for col in equity_cols:
        label = col.replace('equity_', '').replace('_', ' ').title()
        fig.add_trace(go.Scatter(
            x=equity_df['date'],
            y=equity_df[col],
            mode='lines',
            name=label,
            line=dict(width=2)
        ))
        
    fig.update_layout(
        title='Strategy Performance Comparison (Cumulative Equity Curves)',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (Base = 1.0)',
        hovermode='x unified',
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_drawdowns(equity_df: pd.DataFrame) -> go.Figure:
    """Generates an interactive Plotly chart comparing drawdown profiles."""
    fig = go.Figure()
    
    equity_cols = [c for c in equity_df.columns if c.startswith('equity_')]
    
    for col in equity_cols:
        label = col.replace('equity_', '').replace('_', ' ').title()
        # Compute drawdown series
        equity_series = equity_df[col]
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / (peak + 1e-15)
        
        fig.add_trace(go.Scatter(
            x=equity_df['date'],
            y=drawdown * 100.0,  # convert to percent
            mode='lines',
            name=label,
            fill='tozeroy' if col == 'equity_hybrid' else None,  # Highlight hybrid drawdown
            line=dict(width=1.5)
        ))
        
    fig.update_layout(
        title='Strategy Drawdown Comparison (%)',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_regime_distributions(labeled_df: pd.DataFrame) -> go.Figure:
    """Generates Plotly violin/box chart showing daily return distributions per regime."""
    # Ensure ret_simple is computed
    if 'ret_simple' not in labeled_df.columns:
        if 'raw_close' in labeled_df.columns:
            labeled_df['ret_simple'] = labeled_df['raw_close'].pct_change(fill_method=None)
        else:
            labeled_df['ret_simple'] = labeled_df['close'].pct_change(fill_method=None)
            
    # Drop first row containing NaN
    df_clean = labeled_df.dropna(subset=['ret_simple', 'regime_label'])
    
    # Scale returns to percent for readability
    df_clean = df_clean.copy()
    df_clean['ret_pct'] = df_clean['ret_simple'] * 100.0
    
    fig = px.violin(
        df_clean,
        x='regime_label',
        y='ret_pct',
        color='regime_label',
        box=True,
        points="outliers",
        title='Daily Return Distribution by Discovered Market Regime',
        labels={'regime_label': 'Market Regime', 'ret_pct': 'Daily Return (%)'},
        template='plotly_dark'
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis={'categoryorder': 'total median ascending'}
    )
    return fig

def plot_monte_carlo_paths(paths: np.ndarray, dates_horizon: Optional[List[Any]] = None) -> go.Figure:
    """
    Generates a Monte Carlo path projection chart showing confidence ribbons (fan chart).
    
    Args:
        paths: np.ndarray shape [horizon + 1, n_sims]
        dates_horizon: Optional list/array of future date steps. If None, uses integer steps.
    """
    horizon_steps, n_sims = paths.shape
    x_steps = dates_horizon if dates_horizon is not None else np.arange(horizon_steps)
    
    # Compute median, percentile boundaries for fan chart
    p5 = np.percentile(paths, 5, axis=1)
    p25 = np.percentile(paths, 25, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p75 = np.percentile(paths, 75, axis=1)
    p95 = np.percentile(paths, 95, axis=1)
    
    fig = go.Figure()
    
    # 5th to 95th Percentile Area (Light Shading)
    fig.add_trace(go.Scatter(
        x=x_steps, y=p95, mode='lines', line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_steps, y=p5, mode='lines', fill='tonexty', 
        fillcolor='rgba(99, 110, 250, 0.1)', line=dict(width=0),
        name='5th - 95th Percentile'
    ))
    
    # 25th to 75th Percentile Area (Medium Shading)
    fig.add_trace(go.Scatter(
        x=x_steps, y=p75, mode='lines', line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_steps, y=p25, mode='lines', fill='tonexty', 
        fillcolor='rgba(99, 110, 250, 0.25)', line=dict(width=0),
        name='25th - 75th Percentile'
    ))
    
    # Median Path (Solid Line)
    fig.add_trace(go.Scatter(
        x=x_steps,
        y=p50,
        mode='lines',
        name='Median Projection',
        line=dict(color='rgb(99, 110, 250)', width=2.5)
    ))
    
    # Plot first 10 paths as visual reference
    for i in range(min(10, n_sims)):
        fig.add_trace(go.Scatter(
            x=x_steps,
            y=paths[:, i],
            mode='lines',
            line=dict(width=0.8, color='rgba(255, 255, 255, 0.25)'),
            name=f'Sim Path {i+1}',
            showlegend=False
        ))
        
    fig.update_layout(
        title=f'Monte Carlo Path Simulation ({n_sims} paths, {horizon_steps - 1} steps)',
        xaxis_title='Steps / Horizon',
        yaxis_title='Relative Portfolio Value (Start = 1.0)',
        hovermode='x unified',
        template='plotly_dark'
    )
    return fig

def plot_monte_carlo_distributions(paths: np.ndarray) -> go.Figure:
    """Generates a histogram chart showing simulated terminal return outcomes."""
    terminal_prices = paths[-1, :]
    terminal_returns = (terminal_prices - 1.0) * 100.0  # converted to percent
    
    fig = go.Figure()
    
    # Histogram of outcomes
    fig.add_trace(go.Histogram(
        x=terminal_returns,
        nbinsx=50,
        marker_color='rgb(99, 110, 250)',
        opacity=0.75,
        name='Terminal Returns'
    ))
    
    # Add vertical lines for median and 5% tail values
    median_val = np.median(terminal_returns)
    tail_5pct = np.percentile(terminal_returns, 5)
    
    fig.add_vline(x=median_val, line_dash="dash", line_color="green", line_width=2, 
                  annotation_text=f"Median: {median_val:.1f}%", annotation_position="top right")
    fig.add_vline(x=tail_5pct, line_dash="dot", line_color="red", line_width=2, 
                  annotation_text=f"Worst 5% tail: {tail_5pct:.1f}%", annotation_position="top left")
                  
    fig.update_layout(
        title='Terminal Returns Distribution at Horizon (%)',
        xaxis_title='Terminal Return (%)',
        yaxis_title='Frequency',
        template='plotly_dark',
        showlegend=False
    )
    return fig

def plot_rolling_risk_metrics(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None, window: int = 63) -> go.Figure:
    """Generates rolling Sharpe Ratio and rolling Volatility charts on dual axes."""
    from src.analysis.risk_metrics import compute_rolling_sharpe, compute_rolling_volatility
    
    rolling_sharpe = compute_rolling_sharpe(returns, window)
    rolling_vol = compute_rolling_volatility(returns, window) * 100.0  # to percent
    
    fig = go.Figure()
    
    # Left Axis: Rolling Sharpe Ratio
    fig.add_trace(go.Scatter(
        x=returns.index,
        y=rolling_sharpe,
        mode='lines',
        name='Rolling Sharpe Ratio (Left Axis)',
        line=dict(color='orange', width=2)
    ))
    
    # Right Axis: Rolling Volatility (%)
    fig.add_trace(go.Scatter(
        x=returns.index,
        y=rolling_vol,
        mode='lines',
        name='Rolling Volatility % (Right Axis)',
        line=dict(color='cyan', width=1.5),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=f'Rolling Performance Metrics (Lookback Window = {window} days)',
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        yaxis2=dict(
            title='Annualized Volatility (%)',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_regime_overlaid_price(labeled_df: pd.DataFrame) -> go.Figure:
    """
    Plots the close price overlaid with shaded background colors representing
    the discovered market regimes. Contiguous blocks are grouped to optimize rendering.
    """
    fig = go.Figure()
    
    # Plot close price
    # Check if we should use 'raw_close' or 'close'
    close_col = 'raw_close' if 'raw_close' in labeled_df.columns else 'close'
    
    fig.add_trace(go.Scatter(
        x=labeled_df['date'],
        y=labeled_df[close_col],
        mode='lines',
        name='NIFTY 50 Close',
        line=dict(color='rgba(255, 255, 255, 0.85)', width=2)
    ))
    
    # Define a modern color palette with low opacity for backgrounds
    color_map = {
        "Bullish Low Volatility": "rgba(16, 185, 129, 0.08)",       # Emerald
        "Bullish High Volatility": "rgba(5, 150, 105, 0.06)",      # Teal-Green
        "Recovery Regime": "rgba(59, 130, 246, 0.06)",             # Blue
        "Sideways Low Volatility": "rgba(245, 158, 11, 0.06)",     # Amber
        "Distribution / Risk-Off Regime": "rgba(249, 115, 22, 0.08)", # Orange
        "Bearish High Volatility": "rgba(239, 68, 68, 0.1)"        # Rose
    }
    
    default_colors = [
        "rgba(245, 158, 11, 0.06)",  # State 0
        "rgba(5, 150, 105, 0.06)",   # State 1
        "rgba(249, 115, 22, 0.08)",  # State 2
        "rgba(59, 130, 246, 0.06)",  # State 3
        "rgba(16, 185, 129, 0.08)",  # State 4
        "rgba(239, 68, 68, 0.1)"     # State 5
    ]
    
    # Find contiguous blocks of regimes
    df = labeled_df.sort_values('date').reset_index(drop=True)
    df['regime_change'] = df['regime_state'] != df['regime_state'].shift(1)
    block_starts = df[df['regime_change']].index.tolist()
    if 0 not in block_starts:
        block_starts = [0] + block_starts
    block_ends = block_starts[1:] + [len(df)]
    
    # Add shapes for contiguous blocks
    for start_idx, end_idx in zip(block_starts, block_ends):
        sub_df = df.iloc[start_idx:end_idx]
        state = sub_df['regime_state'].iloc[0]
        label = sub_df['regime_label'].iloc[0] if 'regime_label' in sub_df.columns else f"State {state}"
        
        color = color_map.get(label, default_colors[state % len(default_colors)])
        
        start_date = sub_df['date'].iloc[0]
        end_date = sub_df['date'].iloc[-1]
        
        fig.add_vrect(
            x0=start_date,
            x1=end_date,
            fillcolor=color,
            opacity=1.0,
            layer="below",
            line_width=0,
            showlegend=False
        )
        
    # Add a dummy scatter trace for each unique regime to render a clean, interactive legend
    unique_regimes = df[['regime_state', 'regime_label']].drop_duplicates().sort_values('regime_state')
    for _, row in unique_regimes.iterrows():
        state = row['regime_state']
        label = row['regime_label']
        # Use higher opacity for the legend markers
        legend_color = color_map.get(label, default_colors[state % len(default_colors)]).replace("0.12", "0.8").replace("0.08", "0.8").replace("0.15", "0.8").replace("0.06", "0.8").replace("0.1", "0.8").replace("0.07", "0.8")
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=legend_color, symbol='square'),
            name=label,
            showlegend=True
        ))
        
    fig.update_layout(
        title='NIFTY 50 Close Price overlaid with Discovered Latent Regimes',
        xaxis_title='Date',
        yaxis_title='Index Level',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_pca_2d(labeled_df: pd.DataFrame) -> go.Figure:
    """
    Fits PCA on all standardized feature columns, projects them to 2D,
    and returns a Plotly scatter plot colored by discovered regimes.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Identify feature columns (numeric, non-date, non-raw, non-regime)
    feature_cols = [c for c in labeled_df.columns if c not in ['date', 'regime_state', 'regime_label', 'ret_simple'] and not c.startswith('raw_')]
    
    # Drop rows containing NaNs in features
    df_clean = labeled_df.dropna(subset=feature_cols)
    if len(df_clean) < 10:
        # Return empty figure if insufficient data
        fig = go.Figure()
        fig.update_layout(title="PCA 2D Projection (Insufficient Data)", template='plotly_dark')
        return fig
        
    X = df_clean[feature_cols].values
    labels = df_clean['regime_label'].values
    
    # Standardize
    X_scaled = StandardScaler().fit_transform(X)
    
    # Fit PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(X_pca, columns=['Principal Component 1', 'Principal Component 2'])
    pca_df['Regime'] = labels
    
    # Define standard colors matching other plots
    color_map = {
        "Bullish Low Volatility": "#10b981",       # Emerald
        "Bullish High Volatility": "#059669",      # Teal-Green
        "Recovery Regime": "#3b82f6",               # Blue
        "Sideways Low Volatility": "#f59e0b",      # Amber
        "Distribution / Risk-Off Regime": "#f97316", # Orange
        "Bearish High Volatility": "#ef4444"       # Rose
    }
    
    fig = px.scatter(
        pca_df,
        x='Principal Component 1',
        y='Principal Component 2',
        color='Regime',
        color_discrete_map=color_map,
        opacity=0.6,
        title='PCA 2D Feature Space Projection (Dimensionality Reduction)',
        template='plotly_dark'
    )
    
    fig.update_layout(
        xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance Explained)',
        yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance Explained)',
        hovermode='closest',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_correlation_heatmap(labeled_df: pd.DataFrame) -> go.Figure:
    """
    Renders a Pearson correlation matrix heatmap for key engineered features.
    """
    key_features = [
        'ret_3d', 'ret_10d', 'atr_14', 'slope_sma_50', 'dist_sma_200',
        'rsi_14', 'macd_hist', 'drawdown', 'trend_strength_score',
        'hurst_exponent', 'entropy', 'vix_close_ret'
    ]
    
    # Fallback to whatever exists in df if key features are missing
    cols = [c for c in key_features if c in labeled_df.columns]
    if not cols:
        cols = [c for c in labeled_df.columns if c not in ['date', 'regime_state', 'regime_label'] and not c.startswith('raw_')][:10]
        
    corr_matrix = labeled_df[cols].corr()
    
    # Capitalize names for formatting
    formatted_labels = [c.replace('_', ' ').title() for c in cols]
    
    fig = px.imshow(
        corr_matrix.values,
        x=formatted_labels,
        y=formatted_labels,
        color_continuous_scale='RdBu_r',
        zmin=-1.0, zmax=1.0,
        title='Key Feature Pearson Correlation Heatmap',
        template='plotly_dark',
        text_auto=".2f"
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_colorbar=dict(title="Correlation")
    )
    return fig

def plot_regime_durations(labeled_df: pd.DataFrame) -> go.Figure:
    """
    Calculates duration of contiguous blocks for each state and displays box/points plots.
    """
    df = labeled_df.sort_values('date').reset_index(drop=True)
    df['regime_change'] = df['regime_state'] != df['regime_state'].shift(1)
    df['block_id'] = df['regime_change'].cumsum()
    
    durations = df.groupby(['block_id', 'regime_label']).size().reset_index(name='duration')
    
    # Define standard colors matching other plots
    color_map = {
        "Bullish Low Volatility": "#10b981",
        "Bullish High Volatility": "#059669",
        "Recovery Regime": "#3b82f6",
        "Sideways Low Volatility": "#f59e0b",
        "Distribution / Risk-Off Regime": "#f97316",
        "Bearish High Volatility": "#ef4444"
    }
    
    fig = px.box(
        durations,
        x='regime_label',
        y='duration',
        color='regime_label',
        color_discrete_map=color_map,
        points="all",
        title='Regime Durations (Trading Days per Contiguous Phase)',
        labels={'regime_label': 'Market Regime', 'duration': 'Days in State'},
        template='plotly_dark'
    )
    
    fig.update_layout(
        showlegend=False,
        yaxis_title='Days in State',
        xaxis_title='',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def plot_regime_probabilities_chart(dates: pd.Series, probs: np.ndarray, state_labels: List[str]) -> go.Figure:
    """
    Generates a stacked area chart of daily regime probabilities over time.
    """
    fig = go.Figure()
    
    for i in range(probs.shape[1]):
        label = state_labels[i] if i < len(state_labels) else f"State {i}"
        fig.add_trace(go.Scatter(
            x=dates,
            y=probs[:, i],
            mode='lines',
            name=label,
            stackgroup='one',
            line=dict(width=0.5)
        ))
        
    fig.update_layout(
        title='Smoothed Regime Posterior Probabilities (Daily Time-Series)',
        xaxis_title='Date',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1.0]),
        hovermode='x unified',
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_pca_3d(labeled_df: pd.DataFrame) -> go.Figure:
    """
    Fits PCA on all standardized feature columns, projects them to 3D,
    and returns a Plotly 3D scatter plot colored by discovered regimes.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Identify feature columns (numeric, non-date, non-raw, non-regime)
    feature_cols = [c for c in labeled_df.columns if c not in ['date', 'regime_state', 'regime_label', 'ret_simple'] and not c.startswith('raw_')]
    
    # Drop rows containing NaNs in features
    df_clean = labeled_df.dropna(subset=feature_cols)
    if len(df_clean) < 10:
        fig = go.Figure()
        fig.update_layout(title="PCA 3D Projection (Insufficient Data)", template='plotly_dark')
        return fig
        
    X = df_clean[feature_cols].values
    labels = df_clean['regime_label'].values
    
    # Standardize
    X_scaled = StandardScaler().fit_transform(X)
    
    # Fit PCA with 3 components
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Regime'] = labels
    
    # Define standard colors matching other plots
    color_map = {
        "Bullish Low Volatility": "#10b981",       # Emerald
        "Bullish High Volatility": "#059669",      # Teal-Green
        "Recovery Regime": "#3b82f6",               # Blue
        "Sideways Low Volatility": "#f59e0b",      # Amber
        "Distribution / Risk-Off Regime": "#f97316", # Orange
        "Bearish High Volatility": "#ef4444"       # Rose
    }
    
    fig = px.scatter_3d(
        pca_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Regime',
        color_discrete_map=color_map,
        opacity=0.7,
        title='3D PCA Feature Space Projection (Dimensionality Reduction)',
        template='plotly_dark'
    )
    
    # Adjust layout to make it look clean and modern (transparent backgrounds, subtle gridlines)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(255,255,255,0.05)",
                showbackground=False,
                title="Principal Component 1"
            ),
            yaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(255,255,255,0.05)",
                showbackground=False,
                title="Principal Component 2"
            ),
            zaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(255,255,255,0.05)",
                showbackground=False,
                title="Principal Component 3"
            )
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig



