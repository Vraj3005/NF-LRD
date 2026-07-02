"""
Design tokens and theme styles compilation for NIFTY 50 Market Regime Terminal.
"""

import streamlit as st

THEME_TOKENS = {
    "light": {
        "bg_app": "#F7F7F8",
        "bg_main": "#FFFFFF",
        "bg_card": "#FFFFFF",
        "border": "#E5E7EB",
        "text_primary": "#111827",
        "text_secondary": "#6B7280",
        "text_muted": "#9CA3AF",
        "accent": "#4F46E5",
        "success": "#10B981",
        "warning": "#F59E0B",
        "risk": "#EF4444",
        "shadow": "0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03)",
        "sidebar_bg": "#FFFFFF",
        "sidebar_border": "#E5E7EB",
        "sidebar_text": "#374151",
        "sidebar_active_bg": "#F3F4F6",
        "sidebar_active_text": "#111827",
    },
    "dark": {
        "bg_app": "#070707",
        "bg_main": "#111113",
        "bg_card": "#161618",
        "border": "#26262A",
        "text_primary": "#F9FAFB",
        "text_secondary": "#A1A1AA",
        "text_muted": "#71717A",
        "accent": "#22D3EE",
        "success": "#10B981",
        "warning": "#F59E0B",
        "risk": "#FB7185",
        "shadow": "0 4px 20px rgba(0, 0, 0, 0.4)",
        "sidebar_bg": "#111113",
        "sidebar_border": "#26262A",
        "sidebar_text": "#A1A1AA",
        "sidebar_active_bg": "#1C1C1F",
        "sidebar_active_text": "#F9FAFB",
    },
}


def inject_theme_css(theme: str = "light"):
    """
    Inject global theme variables and specific element styling based on the active theme mode.
    """
    tokens = THEME_TOKENS.get(theme, THEME_TOKENS["light"])

    # CSS template
    css = f"""
    <style>
        :root {{
            --bg-app: {tokens["bg_app"]};
            --bg-main: {tokens["bg_main"]};
            --bg-card: {tokens["bg_card"]};
            --border-color: {tokens["border"]};
            --text-primary: {tokens["text_primary"]};
            --text-secondary: {tokens["text_secondary"]};
            --text-muted: {tokens["text_muted"]};
            --accent-color: {tokens["accent"]};
            --success-color: {tokens["success"]};
            --warning-color: {tokens["warning"]};
            --risk-color: {tokens["risk"]};
            --card-shadow: {tokens["shadow"]};

            --sidebar-bg: {tokens["sidebar_bg"]};
            --sidebar-border: {tokens["sidebar_border"]};
            --sidebar-text: {tokens["sidebar_text"]};
            --sidebar-active-bg: {tokens["sidebar_active_bg"]};
            --sidebar-active-text: {tokens["sidebar_active_text"]};

            /* Platform Spacing Tokens */
            --space-xs: 4px;
            --space-sm: 8px;
            --space-md: 16px;
            --space-lg: 24px;
            --space-xl: 32px;
            --card-radius: 20px;
            --card-padding: 24px;
            --card-gap: 20px;
        }}

        /* Global Background and Typography Overrides */
        .stApp, .main, [data-testid="stAppViewContainer"] {{
            background-color: var(--bg-app) !important;
            color: var(--text-primary) !important;
            font-family: 'Inter', sans-serif !important;
        }}

        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Outfit', 'Inter', sans-serif !important;
            color: var(--text-primary) !important;
            font-weight: 700 !important;
        }}

        /* Sidebar Styling Override */
        [data-testid="stSidebar"] {{
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid var(--sidebar-border) !important;
        }}

        [data-testid="stSidebar"] .main {{
            background-color: var(--sidebar-bg) !important;
        }}

        /* Compact and style the sidebar header containing the collapse button */
        [data-testid="stSidebarHeader"] {{
            padding: 8px 12px 0px 12px !important;
            min-height: 36px !important;
            height: 36px !important;
            background-color: transparent !important;
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
        }}

        /* Style the collapse button to look like a premium SaaS widget */
        [data-testid="stSidebarHeader"] button,
        [data-testid="stSidebarCollapseButton"] button,
        button[aria-label="Close sidebar"],
        [data-testid="stSidebar"] button[data-testid="stBaseButton-header"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            width: 30px !important;
            height: 30px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            color: var(--text-secondary) !important;
            transition: all 0.2s ease-in-out !important;
            padding: 0 !important;
        }}

        [data-testid="stSidebarHeader"] button:hover,
        [data-testid="stSidebarCollapseButton"] button:hover,
        button[aria-label="Close sidebar"]:hover,
        [data-testid="stSidebar"] button[data-testid="stBaseButton-header"]:hover {{
            background-color: var(--sidebar-active-bg) !important;
            color: var(--sidebar-active-text) !important;
            border-color: var(--sidebar-active-text) !important;
            transform: scale(1.03);
        }}

        /* Remove default padding from user content area and set side margins */
        [data-testid="stSidebarUserContent"] {{
            padding-top: 0px !important;
            padding-left: 12px !important;
            padding-right: 12px !important;
        }}

        section[data-testid="stSidebar"] > div:first-child {{
            padding-top: 0rem !important;
        }}

        /* Navigation Nav items */
        .nav-container {{
            margin-top: 20px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .nav-item {{
            display: flex;
            align-items: center;
            padding: 10px 16px;
            border-radius: 12px;
            color: var(--sidebar-text);
            text-decoration: none;
            font-weight: 500;
            font-size: 0.92rem;
            transition: all 0.2s ease;
            cursor: pointer;
            border: 1px solid transparent;
        }}

        .nav-item:hover {{
            background-color: var(--sidebar-active-bg);
            color: var(--sidebar-active-text);
        }}

        .nav-item-active {{
            background-color: var(--sidebar-active-bg) !important;
            color: var(--sidebar-active-text) !important;
            font-weight: 600;
            border: 1px solid var(--border-color) !important;
            box-shadow: var(--card-shadow);
        }}

        /* General Streamlit Cards */
        .saas-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--card-radius);
            padding: var(--card-padding);
            margin-bottom: var(--card-gap);
            box-shadow: var(--card-shadow);
            color: var(--text-primary);
        }}

        /* Buttons Overrides */
        .stButton>button {{
            border-radius: 12px !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }}

        .stButton>button[kind="primary"] {{
            background-color: var(--accent-color) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--accent-color) !important;
        }}

        .stButton>button[kind="primary"]:hover {{
            opacity: 0.9;
        }}

        .stButton>button[kind="secondary"] {{
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }}

        .stButton>button[kind="secondary"]:hover {{
            background-color: var(--sidebar-active-bg) !important;
        }}

        /* Streamlit Dataframe wrapper */
        [data-testid="stTable"], [data-testid="stDataFrame"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }}

        /* Clean up standard Streamlit header clutter and reduce top padding */
        [data-testid="stHeader"] {{
            display: none !important;
            height: 0px !important;
            min-height: 0px !important;
        }}

        .block-container {{
            padding-top: 1.5rem !important;
            padding-bottom: 1rem !important;
            max-width: 95% !important;
        }}

        /* Override Streamlit native container border wrapper to look exactly like saas-card */
        [data-testid="stVerticalBlockBorderWrapper"], .stVerticalBlockBorderWrapper {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--card-radius) !important;
            padding: var(--card-padding) !important;
            margin-bottom: var(--card-gap) !important;
            box-shadow: var(--card-shadow) !important;
            color: var(--text-primary) !important;
        }}

        /* Tooltip styling */
        .stTooltipIcon {{
            color: var(--text-secondary) !important;
        }}

        /* Streamlit widget labels, markdown, selectboxes, and tabs contrast fixes for light mode compatibility */
        p, li, ol, ul, label, span:not([style*="color"]) {{
            color: var(--text-primary);
        }}

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span:not([style*="color"]) {{
            color: var(--text-primary) !important;
        }}

        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span {{
            color: var(--text-primary) !important;
        }}

        input, select, textarea, button {{
            color: var(--text-primary) !important;
        }}

        [data-testid="stSelectbox"] div[role="button"] {{
            color: var(--text-primary) !important;
        }}

        [data-testid="stMetricLabel"] {{
            color: var(--text-secondary) !important;
        }}

        [data-testid="stMetricValue"] {{
            color: var(--text-primary) !important;
        }}

        button[data-baseweb="tab"] p {{
            color: var(--text-secondary) !important;
        }}

        button[data-baseweb="tab"][aria-selected="true"] p {{
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }}

        [data-testid="stExpander"] summary p {{
            color: var(--text-primary) !important;
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {{
            color: var(--sidebar-text) !important;
        }}

        /* Dropdown listbox and select item contrast overrides for theme switching */
        ul[role="listbox"], [role="listbox"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
        }}
        
        li[role="option"], [role="option"] {{
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
        }}
        
        li[role="option"]:hover, [role="option"]:hover,
        li[role="option"][aria-selected="true"], [role="option"][aria-selected="true"] {{
            background-color: var(--sidebar-active-bg) !important;
            color: var(--sidebar-active-text) !important;
        }}

        /* Input and form widgets background & text styling */
        div[data-baseweb="select"] > div {{
            background-color: var(--bg-card) !important;
            border-color: var(--border-color) !important;
            color: var(--text-primary) !important;
        }}
        
        div[data-baseweb="input"] input, 
        .stNumberInput input, 
        .stTextInput input, 
        .stTextArea textarea {{
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-color) !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
