"""
Smoke tests for the Streamlit dashboard application to ensure startup stability.
"""

import os
import sys


def test_streamlit_imports_and_startup():
    """Verifies that streamlit and all dashboard-level imports compile without errors."""
    # Add project root and app directory to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app_dir = os.path.join(project_root, "app")

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    # 1. Verify core library imports
    import streamlit as st

    assert st is not None

    # 2. Verify dashboard component helper imports
    from src.visualization.dashboard_components import (
        load_all_dashboard_data,
        render_educational_disclaimer,
    )

    assert load_all_dashboard_data is not None
    assert render_educational_disclaimer is not None

    # 3. Verify visualization plotting imports
    from src.visualization.charts import (
        plot_drawdowns,
        plot_equity_curves,
        plot_monthly_returns_heatmap,
        plot_rolling_sharpe,
    )

    assert plot_equity_curves is not None
    assert plot_drawdowns is not None
    assert plot_monthly_returns_heatmap is not None
    assert plot_rolling_sharpe is not None
