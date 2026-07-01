"""
Shell Layout and Header elements for NIFTY 50 Market Regime Terminal.
"""

import streamlit as st

from app.ui.components import render_status_pill
from app.ui.html import render_html


def select_page(page_val: str):
    st.session_state.page = page_val


def toggle_theme():
    current_theme = st.session_state.get("theme", "light")
    st.session_state.theme = "dark" if current_theme == "light" else "light"


def render_sidebar():
    """
    Renders the OutreachOps AI styled sidebar header and page navigation.
    """
    # Logo header at top of sidebar
    render_html(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px; padding: 10px 5px 0 5px;">
            <div style="
                width: 36px;
                height: 36px;
                border-radius: 50%;
                background: linear-gradient(135deg, #4F46E5, #06B6D4);
                display: flex;
                align-items: center;
                justify-content: center;
                color: #FFFFFF;
                font-family: 'Outfit', sans-serif;
                font-weight: 800;
                font-size: 1.15rem;
                box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25);
            ">N</div>
            <div>
                <div style="font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.1rem; color: var(--text-primary); line-height: 1.1;">NF-LRD</div>
                <div style="font-size: 0.72rem; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">Quant Risk Console</div>
            </div>
        </div>
        """,
        sidebar=True,
    )

    # Navigation items
    pages = [
        ("Dashboard / Overview", "Overview"),
        ("Data Explorer", "Data Explorer"),
        ("Feature Analytics", "Feature Analytics"),
        ("Regime Discovery", "Regime Discovery"),
        ("Validation", "Validation"),
        ("Backtesting", "Backtesting"),
        ("Risk Intelligence", "Risk Intelligence"),
        ("Monte Carlo", "Monte Carlo"),
        ("Report Export", "Report Export"),
        ("Methodology", "Methodology"),
    ]

    # Custom Sidebar Navigation CSS overrides to resemble OutreachOps nav buttons
    render_html(
        """
        <style>
            /* Style sidebar nav buttons kind="secondary" (inactive) */
            [data-testid="stSidebar"] .stButton>button[kind="secondary"] {
                background-color: transparent !important;
                border: 1px solid transparent !important;
                color: var(--sidebar-text) !important;
                width: 100% !important;
                text-align: left !important;
                display: flex !important;
                align-items: center !important;
                padding: 10px 16px !important;
                border-radius: 12px !important;
                font-size: 0.92rem !important;
                font-weight: 500 !important;
                margin-bottom: 2px !important;
                justify-content: flex-start !important;
                box-shadow: none !important;
            }
            [data-testid="stSidebar"] .stButton>button[kind="secondary"]:hover {
                background-color: var(--sidebar-active-bg) !important;
                color: var(--sidebar-active-text) !important;
            }

            /* Style sidebar nav buttons kind="primary" (active) */
            [data-testid="stSidebar"] .stButton>button[kind="primary"] {
                background-color: var(--sidebar-active-bg) !important;
                color: var(--sidebar-active-text) !important;
                font-weight: 600 !important;
                border: 1px solid var(--border-color) !important;
                width: 100% !important;
                text-align: left !important;
                display: flex !important;
                align-items: center !important;
                padding: 10px 16px !important;
                border-radius: 12px !important;
                font-size: 0.92rem !important;
                margin-bottom: 2px !important;
                justify-content: flex-start !important;
                box-shadow: var(--card-shadow) !important;
            }

            /* Tighten sidebar vertical spacing */
            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                gap: 0.2rem !important;
            }
        </style>
        """,
        sidebar=True,
    )

    # Loop over and render nav items
    for label, page_val in pages:
        is_active = st.session_state.get("page", "Overview") == page_val
        btn_type = "primary" if is_active else "secondary"
        st.sidebar.button(
            label,
            key=f"nav_{page_val}",
            type=btn_type,
            on_click=select_page,
            args=(page_val,),
        )


def render_topbar():
    """
    Renders top status pill bar and the theme switcher toggle.
    """
    col_l, col_r = st.columns([3, 1])

    with col_l:
        # Build status indicators based on state
        db_pill = (
            render_status_pill("Database Connected", "success")
            if st.session_state.get("labeled_data") is not None
            else render_status_pill("Database Disconnected", "danger")
        )
        model_pill = (
            render_status_pill("Model Fitted", "success")
            if st.session_state.get("model_report") is not None
            else render_status_pill("Model Pending", "warning")
        )
        demo_pill = (
            render_status_pill("Demo Data Active", "info")
            if st.session_state.get("is_demo", False)
            else render_status_pill("Live Pipeline Active", "success")
        )
        backtest_pill = (
            render_status_pill("Backtest Loaded", "success")
            if st.session_state.get("regime_risk") is not None
            else render_status_pill("Backtest Pending", "neutral")
        )

        render_html(f"""
            <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 15px;">
                <span style="font-size: 0.8rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-right: 8px;">Diagnostics:</span>
                {db_pill}
                {model_pill}
                {demo_pill}
                {backtest_pill}
            </div>
            """)

    with col_r:
        # Theme toggle control
        theme = st.session_state.get("theme", "light")
        toggle_label = "🌙 Dark Mode" if theme == "light" else "☀️ Light Mode"

        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            render_html("""
                <div style="text-align: right; font-size: 0.72rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 10px;">
                    NF-LRD Console
                </div>
                """)
        with col_t2:
            st.button(
                toggle_label,
                key="theme_switcher",
                type="secondary",
                on_click=toggle_theme,
            )
