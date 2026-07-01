"""
Strict Centralized SaaS UI Components library for NIFTY 50 Market Regime Terminal.
All custom layouts and HTML components route through this module.
"""

import pandas as pd
import streamlit as st

from app.ui.html import render_html


def render_status_pill(label: str, status_type: str = "info") -> str:
    """
    Returns HTML string representing a status badge.
    """
    theme = st.session_state.get("theme", "light")
    if theme == "light":
        colors = {
            "success": ("#E6F4EA", "#137333", "#10B981"),
            "warning": ("#FEF7E0", "#B06000", "#F59E0B"),
            "danger": ("#FCE8E6", "#C5221F", "#EF4444"),
            "info": ("#E8F0FE", "#1A73E8", "#3B82F6"),
            "neutral": ("#F1F3F4", "#3C4043", "#9CA3AF"),
        }
    else:
        colors = {
            "success": ("rgba(16, 185, 129, 0.1)", "#34D399", "#10B981"),
            "warning": ("rgba(245, 158, 11, 0.1)", "#FBBF24", "#F59E0B"),
            "danger": ("rgba(251, 113, 133, 0.1)", "#FDA4AF", "#FB7185"),
            "info": ("rgba(59, 130, 246, 0.1)", "#93C5FD", "#3B82F6"),
            "neutral": ("rgba(113, 113, 122, 0.1)", "#D4D4D8", "#71717A"),
        }

    bg, fg, border_c = colors.get(status_type, colors["info"])
    return f"""
    <span style="
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        background-color: {bg};
        color: {fg};
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid {border_c}40;
        margin-right: 6px;
        font-family: 'Inter', sans-serif;
    ">
        <span style="
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: {fg};
            margin-right: 6px;
            display: inline-block;
        "></span>
        {label}
    </span>
    """


def render_kpi_card(
    title: str,
    value: str,
    description: str,
    change: str = None,
    change_type: str = "neutral",
    value_size: str = None,
):
    """
    Renders a premium KPI metric card inside the current Streamlit column block.
    Implements auto-scaling font sizes to prevent word breaks and container overflows.
    """
    theme = st.session_state.get("theme", "light")
    change_colors = {
        "positive": "#10B981" if theme == "light" else "#34D399",
        "negative": "#EF4444" if theme == "light" else "#FB7185",
        "neutral": "#6B7280" if theme == "light" else "#A1A1AA",
    }
    color = change_colors.get(change_type, change_colors["neutral"])

    change_html = ""
    if change:
        change_html = f'<div style="color: {color}; font-size: 0.8rem; font-weight: 600; margin-top: var(--space-xs);">{change}</div>'

    # Font size calculation to fit long texts like "Sideways Low Volatility" or "Gaussian HMM"
    if value_size is None:
        val_len = len(str(value))
        if val_len > 18:
            size_css = "font-size: 1.15rem; font-weight: 700;"
        elif val_len > 12:
            size_css = "font-size: 1.4rem; font-weight: 700;"
        else:
            size_css = "font-size: 1.8rem; font-weight: 800;"
    elif value_size == "sm":
        size_css = "font-size: 1.1rem; font-weight: 700;"
    elif value_size == "md":
        size_css = "font-size: 1.4rem; font-weight: 700;"
    else:
        size_css = "font-size: 1.8rem; font-weight: 800;"

    html = f"""
    <div class="saas-card" style="margin-bottom: 0px; height: 100%; min-height: 180px; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;">
        <div style="min-width: 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{title}</div>
            <div style="{size_css} color: var(--text-primary); margin-top: var(--space-sm); font-family: 'Outfit', sans-serif; line-height: 1.2; word-break: normal; overflow-wrap: normal; hyphens: none; max-width: 100%; min-width: 0;">{value}</div>
        </div>
        <div style="margin-top: var(--space-md);">
            {change_html}
            <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: var(--space-sm); line-height: 1.3;">{description}</div>
        </div>
    </div>
    """
    render_html(html)


def render_card(title: str = None, subtitle: str = None) -> str:
    """
    Returns the HTML container start block for general dashboard cards.
    """
    title_html = ""
    if title:
        subtitle_html = (
            f'<div style="font-size: 0.82rem; color: var(--text-secondary); margin-top: var(--space-xs);">{subtitle}</div>'
            if subtitle
            else ""
        )
        title_html = f"""
        <div style="border-bottom: 1px solid var(--border-color); padding-bottom: var(--space-md); margin-bottom: var(--space-lg);">
            <h4 style="margin: 0; font-size: 1rem; color: var(--text-primary);">{title}</h4>
            {subtitle_html}
        </div>
        """
    return f"""
    <div class="saas-card">
        {title_html}
    """


def render_card_end() -> str:
    """
    Returns the closing tags for general dashboard cards.
    """
    return "</div>"


def render_chart_card(title: str, fig, subtitle: str = None):
    """
    Wraps a Plotly chart inside a styled card.
    """
    with st.container(border=True):
        if title:
            subtitle_html = (
                f'<div style="font-size: 0.82rem; color: var(--text-secondary); margin-top: var(--space-xs);">{subtitle}</div>'
                if subtitle
                else ""
            )
            render_html(f"""
                <div style="border-bottom: 1px solid var(--border-color); padding-bottom: var(--space-sm); margin-bottom: var(--space-md);">
                    <h4 style="margin: 0; font-size: 1.0rem; color: var(--text-primary);">{title}</h4>
                    {subtitle_html}
                </div>
                """)
        st.plotly_chart(fig, width="stretch")


def render_table_card(
    title: str, df: pd.DataFrame, subtitle: str = None, formatter=None
):
    """
    Wraps a Pandas dataframe inside a styled card container.
    """
    with st.container(border=True):
        if title:
            subtitle_html = (
                f'<div style="font-size: 0.82rem; color: var(--text-secondary); margin-top: var(--space-xs);">{subtitle}</div>'
                if subtitle
                else ""
            )
            render_html(f"""
                <div style="border-bottom: 1px solid var(--border-color); padding-bottom: var(--space-sm); margin-bottom: var(--space-md);">
                    <h4 style="margin: 0; font-size: 1.0rem; color: var(--text-primary);">{title}</h4>
                    {subtitle_html}
                </div>
                """)
        if formatter:
            st.dataframe(df.style.format(formatter), width="stretch")
        else:
            st.dataframe(df, width="stretch")


def render_page_header(title: str, subtitle: str):
    """
    Renders standardized title and description subtexts.
    """
    html = f"""
    <div style="margin-top: var(--space-sm); margin-bottom: var(--space-xl);">
        <h2 style="margin: 0; font-size: 1.6rem; font-weight: 800; color: var(--text-primary); font-family: 'Outfit', sans-serif; letter-spacing: -0.5px;">{title}</h2>
        <div style="color: var(--text-secondary); font-size: 0.88rem; margin-top: var(--space-xs); line-height: 1.4;">{subtitle}</div>
    </div>
    """
    render_html(html)


def render_section_header(title: str, subtitle: str = None):
    """
    Renders a section header with an accent indicator bar on the left.
    """
    subtitle_html = (
        f'<div style="color: var(--text-secondary); font-size: 0.88rem; margin-top: var(--space-xs);">{subtitle}</div>'
        if subtitle
        else ""
    )
    html = f"""
    <div style="margin-top: var(--space-xl); margin-bottom: var(--space-lg); padding-left: var(--space-md); border-left: 4px solid var(--accent-color);">
        <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: var(--text-primary);">{title}</h3>
        {subtitle_html}
    </div>
    """
    render_html(html)


def render_section(title: str, subtitle: str = None):
    """
    Alias of render_section_header for backward compatibility.
    """
    render_section_header(title, subtitle)


def render_metric_grid(cards_data: list, columns: int = 4):
    """
    Enforces a strict visual grid layout on KPI cards.
    Automatically collapses to fewer columns if long values might wrap awkwardly.
    """
    has_long_values = any(len(str(c.get("value", ""))) > 15 for c in cards_data)
    cols_to_use = min(columns, 2 if has_long_values else columns)

    for i in range(0, len(cards_data), cols_to_use):
        row_data = cards_data[i : i + cols_to_use]
        cols = st.columns(len(row_data))
        for idx, card in enumerate(row_data):
            with cols[idx]:
                render_kpi_card(
                    title=card.get("title", ""),
                    value=card.get("value", ""),
                    description=card.get("description", ""),
                    change=card.get("change"),
                    change_type=card.get("change_type", "neutral"),
                    value_size=card.get("value_size"),
                )


def render_two_column_section(col1_func, col2_func, ratio=[1, 1]):
    """
    Wraps side-by-side elements cleanly inside structural responsive layout columns.
    """
    col1, col2 = st.columns(ratio)
    with col1:
        col1_func()
    with col2:
        col2_func()


def render_empty_state(message: str, action_label: str = None):
    """
    Renders a premium dashed-border empty placeholder state.
    """
    action_html = (
        f'<div style="margin-top: 15px;"><span class="stButton">{action_label}</span></div>'
        if action_label
        else ""
    )
    html = f"""
    <div style="
        border: 2px dashed var(--border-color);
        border-radius: var(--card-radius);
        padding: var(--space-xl);
        text-align: center;
        color: var(--text-secondary);
        background-color: var(--bg-card);
        margin-bottom: var(--card-gap);
    ">
        <div style="font-size: 2.2rem; margin-bottom: 10px;">📂</div>
        <div style="font-size: 0.95rem; font-weight: 600; color: var(--text-primary);">{message}</div>
        <div style="font-size: 0.82rem; color: var(--text-muted); margin-top: 4px;">Initialize dataset or adjust parameter options to build outputs.</div>
        {action_html}
    </div>
    """
    render_html(html)


def render_error_state(message: str):
    """
    Renders a styled error alert block.
    """
    html = f"""
    <div style="
        border-left: 4px solid var(--risk-color);
        background-color: rgba(239, 68, 68, 0.08);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
        color: var(--text-primary);
    ">
        <div style="font-weight: 700; color: var(--risk-color); font-size: 0.9rem;">System Error / Constraint Violation</div>
        <div style="font-size: 0.85rem; margin-top: 6px; line-height: 1.4;">{message}</div>
    </div>
    """
    render_html(html)


def render_info_banner(message: str, banner_type: str = "info"):
    """
    Renders a styled information banner (e.g. system warnings, notifications).
    """
    colors = {
        "info": ("var(--accent-color)", "rgba(79, 70, 229, 0.06)"),
        "warning": ("var(--warning-color)", "rgba(245, 158, 11, 0.06)"),
        "risk": ("var(--risk-color)", "rgba(239, 68, 68, 0.06)"),
    }
    border_c, bg = colors.get(banner_type, colors["info"])
    html = f"""
    <div style="
        border-left: 4px solid {border_c};
        background-color: {bg};
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
        color: var(--text-primary);
    ">
        <div style="font-size: 0.85rem; line-height: 1.4;">{message}</div>
    </div>
    """
    render_html(html)


def render_info_panel(message: str, panel_type: str = "info"):
    """
    High-level dashboard panel info display.
    """
    render_info_banner(message, panel_type)
