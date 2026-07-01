#!/usr/bin/env python
"""
Runner script for Out-of-Sample and Walk-Forward Validation.
Generates metrics reports, degradation summaries, and visual performance charts.
"""

import logging
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.oos_validation import (
    run_oos_partition_validation,
    run_walk_forward_oos_validation,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_validation")


def plot_validation_timeline(
    train_start, train_end, val_start, val_end, test_start, test_end, output_path
):
    """Generates a timeline gantt chart indicating training, validation, and testing periods."""
    logger.info("Generating validation timeline visualization...")
    fig, ax = plt.subplots(figsize=(10, 2.5), facecolor="#050811")
    ax.set_facecolor("#050811")

    t_start = pd.to_datetime(train_start)
    t_end = pd.to_datetime(train_end)
    v_start = pd.to_datetime(val_start)
    v_end = pd.to_datetime(val_end)
    te_start = pd.to_datetime(test_start)
    te_end = pd.to_datetime(test_end)

    # Plot blocks
    ax.barh(
        "Periods",
        (t_end - t_start).days,
        left=t_start,
        height=0.4,
        color="#3b82f6",
        label="In-Sample Training",
    )
    ax.barh(
        "Periods",
        (v_end - v_start).days,
        left=v_start,
        height=0.4,
        color="#f59e0b",
        label="Out-of-Sample Validation",
    )
    ax.barh(
        "Periods",
        (te_end - te_start).days,
        left=te_start,
        height=0.4,
        color="#10b981",
        label="Out-of-Sample Testing",
    )

    # Styling
    ax.set_title(
        "Out-of-Sample Partition Timeline Configuration",
        color="white",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1f2937")
    ax.spines["bottom"].set_color("#1f2937")

    # Legend
    legend = ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.65), ncol=3, frameon=False
    )
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    plt.savefig(
        output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close()
    logger.info(f"Validation timeline saved to: {output_path}")


def plot_oos_equity_curves(res, output_path):
    """Plots equity curves of backtested strategies in Train, Val, and Test subplots."""
    logger.info("Generating OOS equity curves visualization...")
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 10), sharex=False, facecolor="#050811"
    )

    colors = {
        "equity_buy_and_hold": "#94a3b8",  # Slate grey
        "equity_ema_crossover": "#60a5fa",  # Blue
        "equity_vol_targeting": "#a855f7",  # Purple
        "equity_regime_aware": "#f59e0b",  # Amber
        "equity_hybrid": "#10b981",  # Emerald
    }

    labels = {
        "equity_buy_and_hold": "Buy & Hold",
        "equity_ema_crossover": "EMA Crossover",
        "equity_vol_targeting": "Vol Targeting",
        "equity_regime_aware": "Regime Aware",
        "equity_hybrid": "Hybrid Strategy",
    }

    # Helper function to plot subplot
    def plot_axes(ax, curves, title):
        ax.set_facecolor("#050811")
        for col in colors.keys():
            if col in curves.columns:
                ax.plot(
                    curves["date"],
                    curves[col],
                    color=colors[col],
                    label=labels[col],
                    linewidth=1.5,
                )
        ax.set_title(
            title, color="white", fontsize=11, fontweight="bold", loc="left", pad=10
        )
        ax.tick_params(colors="white")
        ax.grid(color="#1f2937", linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#1f2937")
        ax.spines["bottom"].set_color("#1f2937")

    plot_axes(ax1, res["train_curves"], "In-Sample Training Period (2015-2021)")
    plot_axes(ax2, res["val_curves"], "Out-of-Sample Validation Period (2022-2023)")
    plot_axes(ax3, res["test_curves"], "Out-of-Sample Testing Period (2024-Latest)")

    # Title & Legend
    fig.suptitle(
        "Out-of-Sample Performance Comparison Curves",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    handles, labels_list = ax1.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels_list,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(
        output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close()
    logger.info(f"OOS equity curves saved to: {output_path}")


def plot_metrics_comparison(degradation_df, output_path):
    """Generates a bar chart comparing Sharpe and Max Drawdowns across partitions."""
    logger.info("Generating metrics comparisons visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#050811")
    ax1.set_facecolor("#050811")
    ax2.set_facecolor("#050811")

    strategies = degradation_df["Strategy"].tolist()
    x = np.arange(len(strategies))
    width = 0.25

    # 1. Sharpe Ratio plot
    ax1.bar(
        x - width,
        degradation_df["Train_Sharpe"],
        width,
        label="In-Sample (Train)",
        color="#3b82f6",
    )
    ax1.bar(
        x,
        degradation_df["Val_Sharpe"],
        width,
        label="OOS (Validation)",
        color="#f59e0b",
    )
    ax1.bar(
        x + width,
        degradation_df["Test_Sharpe"],
        width,
        label="OOS (Test)",
        color="#10b981",
    )

    ax1.set_title(
        "Sharpe Ratio Comparison", color="white", fontweight="bold", fontsize=11
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, color="white", rotation=15)
    ax1.tick_params(colors="white")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#1f2937")
    ax1.spines["bottom"].set_color("#1f2937")
    ax1.grid(color="#1f2937", linestyle="--", linewidth=0.5, axis="y")

    # 2. Max Drawdown plot
    ax2.bar(
        x - width,
        degradation_df["Train_Max_DD"],
        width,
        label="In-Sample (Train)",
        color="#3b82f6",
    )
    ax2.bar(
        x,
        degradation_df["Val_Max_DD"],
        width,
        label="OOS (Validation)",
        color="#f59e0b",
    )
    ax2.bar(
        x + width,
        degradation_df["Test_Max_DD"],
        width,
        label="OOS (Test)",
        color="#10b981",
    )

    ax2.set_title(
        "Maximum Drawdown Comparison", color="white", fontweight="bold", fontsize=11
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, color="white", rotation=15)
    ax2.tick_params(colors="white")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#1f2937")
    ax2.spines["bottom"].set_color("#1f2937")
    ax2.grid(color="#1f2937", linestyle="--", linewidth=0.5, axis="y")

    # Legend
    handles, labels_list = ax1.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels_list,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.15),
    )
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    plt.savefig(
        output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close()
    logger.info(f"OOS metrics comparison saved to: {output_path}")


def main():
    logger.info("Initializing Out-of-Sample & Walk-Forward Validation Suite...")

    # Paths config
    raw_parquet_path = "data/processed/features_raw.parquet"
    reports_dir = "models/reports"
    os.makedirs(reports_dir, exist_ok=True)

    if not os.path.exists(raw_parquet_path):
        raise FileNotFoundError(
            f"Raw features parquet not found at: {raw_parquet_path}. "
            "Please run 'python run_pipeline.py' to generate raw feature parquet first."
        )

    df_raw = pd.read_parquet(raw_parquet_path)
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    # Date splits config
    train_start = "2015-01-01"
    train_end = "2021-12-31"
    val_start = "2022-01-01"
    val_end = "2023-12-31"
    test_start = "2024-01-01"
    test_end = df_raw["date"].max().strftime("%Y-%m-%d")

    # 1. Run OOS Partition Validation
    res = run_oos_partition_validation(
        df_raw,
        train_start,
        train_end,
        val_start,
        val_end,
        test_start,
        test_end,
        n_components=3,
        config_path="config/settings.yaml",
    )

    # 2. Run expanding walk-forward OOS validation (refitting every 6 months)
    # Using 2010 start if data covers it
    wf_start = (
        "2010-01-01"
        if df_raw["date"].min() <= pd.to_datetime("2010-01-04")
        else df_raw["date"].min().strftime("%Y-%m-%d")
    )
    df_wf, wf_curves, wf_summary = run_walk_forward_oos_validation(
        df_raw,
        start_date=wf_start,
        train_window_years=4,
        test_window_months=6,
        n_components=3,
        config_path="config/settings.yaml",
    )

    # Save walk forward curves & summaries
    wf_summary.to_csv(
        os.path.join(reports_dir, "walk_forward_summary.csv"), index=False
    )
    wf_curves.to_parquet(
        os.path.join(reports_dir, "walk_forward_curves.parquet"), index=False
    )
    logger.info("Saved walk-forward predictions summary and curve Parquet.")

    # 3. Create Performance Degradation Table
    # Reindex summaries to Strategy column
    train_sum = res["train_summary"].set_index("Strategy")
    val_sum = res["val_summary"].set_index("Strategy")
    test_sum = res["test_summary"].set_index("Strategy")

    degradation_rows = []
    for strat in train_sum.index:
        train_sharpe = train_sum.loc[strat, "Sharpe_Ratio"]
        train_cagr = train_sum.loc[strat, "CAGR"]
        train_max_dd = train_sum.loc[strat, "Max_Drawdown"]

        val_sharpe = (
            val_sum.loc[strat, "Sharpe_Ratio"] if strat in val_sum.index else 0.0
        )
        val_cagr = val_sum.loc[strat, "CAGR"] if strat in val_sum.index else 0.0
        val_max_dd = (
            val_sum.loc[strat, "Max_Drawdown"] if strat in val_sum.index else 0.0
        )

        test_sharpe = (
            test_sum.loc[strat, "Sharpe_Ratio"] if strat in test_sum.index else 0.0
        )
        test_cagr = test_sum.loc[strat, "CAGR"] if strat in test_sum.index else 0.0
        test_max_dd = (
            test_sum.loc[strat, "Max_Drawdown"] if strat in test_sum.index else 0.0
        )

        # Calculate degradation percentages
        sharpe_degradation = (
            ((test_sharpe - train_sharpe) / train_sharpe * 100.0)
            if train_sharpe != 0.0
            else 0.0
        )

        degradation_rows.append(
            {
                "Strategy": strat,
                "Train_CAGR": train_cagr,
                "Train_Sharpe": train_sharpe,
                "Train_Max_DD": train_max_dd,
                "Val_CAGR": val_cagr,
                "Val_Sharpe": val_sharpe,
                "Val_Max_DD": val_max_dd,
                "Test_CAGR": test_cagr,
                "Test_Sharpe": test_sharpe,
                "Test_Max_DD": test_max_dd,
                "Sharpe_Degradation_Pct": sharpe_degradation,
            }
        )

    degradation_df = pd.DataFrame(degradation_rows)
    degradation_path = os.path.join(reports_dir, "oos_degradation_report.csv")
    degradation_df.to_csv(degradation_path, index=False)
    logger.info(f"OOS performance degradation report saved to: {degradation_path}")

    # 4. Generate Visualizations
    plot_validation_timeline(
        train_start,
        train_end,
        val_start,
        val_end,
        test_start,
        test_end,
        os.path.join(reports_dir, "validation_timeline.png"),
    )

    plot_oos_equity_curves(res, os.path.join(reports_dir, "oos_equity_curves.png"))

    plot_metrics_comparison(
        degradation_df, os.path.join(reports_dir, "oos_metrics_comparison.png")
    )

    # Print out-of-sample summary log
    print("\n=========================================================================")
    print("      OUT-OF-SAMPLE PERFORMANCE DEGRADATION SUMMARY (SHARPE RATIO)       ")
    print("=========================================================================")
    for idx, row in degradation_df.iterrows():
        print(f"Strategy: {row['Strategy']}")
        print(f"  In-Sample Train (2015-2021): {row['Train_Sharpe']:.3f}")
        print(f"  Out-of-Sample Val (2022-2023): {row['Val_Sharpe']:.3f}")
        print(f"  Out-of-Sample Test (2024-Pres): {row['Test_Sharpe']:.3f}")
        print(f"  Degradation (Train -> Test): {row['Sharpe_Degradation_Pct']:.1f}%\n")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
