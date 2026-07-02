"""
Centralized styling and themes for Plotly figures in the NIFTY 50 Market Regime Terminal.
Provides a premium 'Dark Observability Console' look: near-black, transparent surfaces,
cyan/blue highlights, and clean technical typography.
"""

import plotly.graph_objects as go

# Centralized color palette
COLOR_PALETTE = {
    "bg_dark": "#030712",
    "card_bg": "rgba(10, 15, 30, 0.6)",
    "border": "rgba(255, 255, 255, 0.06)",
    "grid": "rgba(255, 255, 255, 0.05)",
    "text_primary": "#f8fafc",
    "text_secondary": "#94a3b8",
    "accent_blue": "#3b82f6",
    "accent_cyan": "#06b6d4",
    "accent_green": "#10b981",
    "accent_orange": "#f59e0b",
    "accent_red": "#f43f5e",
}

# Regime-specific stable color mapping
REGIME_COLORS = {
    "Bullish Low Volatility": "#10b981",  # Emerald Green
    "Recovery Regime": "#06b6d4",  # Cyan
    "Sideways Low Volatility": "#3b82f6",  # Blue
    "Bullish High Volatility": "#84cc16",  # Lime/Yellow-Green
    "Distribution / Risk-Off Regime": "#f59e0b",  # Amber/Orange
    "Bearish High Volatility": "#f43f5e",  # Red/Rose
}

# Strategy-specific stable color mapping (premium, high-contrast in both themes)
STRATEGY_COLORS = {
    "Buy And Hold": "#94a3b8",        # Slate Gray (Benchmark)
    "Ema Crossover": "#3b82f6",       # Vibrant Blue
    "Vol Targeting": "#a855f7",       # Vibrant Purple
    "Regime Aware": "#f59e0b",        # Vibrant Amber/Orange
    "Hybrid": "#10b981",              # Vibrant Emerald Green
}


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """
    Applies template parameters that dynamically adjust to light/dark themes.
    """
    try:
        import streamlit as st

        theme = st.session_state.get("theme", "light")
    except Exception:
        theme = "light"

    template = "plotly_white" if theme == "light" else "plotly_dark"

    # Theme parameters
    if theme == "light":
        text_primary = "#111827"
        text_secondary = "#4B5563"
        grid_color = "rgba(0, 0, 0, 0.08)"
        border_color = "rgba(0, 0, 0, 0.12)"
        hover_bg = "#FFFFFF"
        hover_border = "rgba(79, 70, 229, 0.4)"
        legend_bg = "rgba(255, 255, 255, 0.95)"
    else:
        text_primary = "#F9FAFB"
        text_secondary = "#A1A1AA"
        grid_color = "rgba(255, 255, 255, 0.05)"
        border_color = "rgba(255, 255, 255, 0.06)"
        hover_bg = "#161618"
        hover_border = "rgba(34, 211, 238, 0.4)"
        legend_bg = "rgba(22, 22, 24, 0.9)"

    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0.0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0.0)",
        font=dict(family="Inter, sans-serif", color=text_primary),
        hoverlabel=dict(
            bgcolor=hover_bg,
            bordercolor=hover_border,
            font=dict(family="Inter, sans-serif", color=text_primary),
        ),
        margin=dict(t=50, b=40, l=40, r=20),
    )

    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        zerolinecolor=border_color,
        tickfont=dict(color=text_secondary, size=11),
        title_font=dict(color=text_primary, size=12),
        linecolor=border_color,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        zerolinecolor=border_color,
        tickfont=dict(color=text_secondary, size=11),
        title_font=dict(color=text_primary, size=12),
        linecolor=border_color,
    )

    # Update 3D scene axes if present
    if fig.layout.scene:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    gridcolor=grid_color,
                    linecolor=border_color,
                    tickfont=dict(color=text_secondary, size=10),
                    title_font=dict(color=text_primary, size=11),
                ),
                yaxis=dict(
                    gridcolor=grid_color,
                    linecolor=border_color,
                    tickfont=dict(color=text_secondary, size=10),
                    title_font=dict(color=text_primary, size=11),
                ),
                zaxis=dict(
                    gridcolor=grid_color,
                    linecolor=border_color,
                    tickfont=dict(color=text_secondary, size=10),
                    title_font=dict(color=text_primary, size=11),
                ),
            )
        )

    # Update legend
    if fig.layout.legend:
        fig.update_layout(
            legend=dict(
                bgcolor=legend_bg,
                bordercolor=border_color,
                borderwidth=1,
                font=dict(color=text_secondary, size=11),
            )
        )

    return fig
