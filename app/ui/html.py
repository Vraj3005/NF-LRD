"""
Centralized safe HTML rendering for NIFTY 50 Market Regime Terminal.
"""

import streamlit as st


def render_html(html: str, sidebar: bool = False) -> None:
    """
    Safely renders custom HTML content on the Streamlit page or sidebar.
    """
    # Clean leading whitespace from each line to prevent markdown parser from treating indented HTML as code blocks
    cleaned_html = "\n".join([line.lstrip() for line in html.splitlines()])
    if sidebar:
        st.sidebar.markdown(cleaned_html, unsafe_allow_html=True)
    else:
        st.markdown(cleaned_html, unsafe_allow_html=True)


def render_spacing(height_px: int) -> None:
    """
    Renders standardized vertical spacing.
    """
    render_html(f'<div style="margin-top: {height_px}px;"></div>')
