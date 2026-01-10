#!/usr/bin/env python
"""
Visualization functions for zero-shot evaluation results.

This module contains all plotting and visualization functions extracted from zs.py
to reduce code size and improve maintainability.

Functions:
    - create_comprehensive_comparison_plots: Main comparison bar charts
    - create_ua_shift_analysis_plots: UA (Uncertainty-Accuracy) analysis
    - create_pavpu_comparison_plots: PAvPU scatter/hexbin plots
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # Use non-interactive backend


__all__ = [
    "create_comprehensive_comparison_plots",
    "create_ua_shift_analysis_plots",
    "create_pavpu_comparison_plots",
]
def create_comprehensive_comparison_plots(
    sam2_results: dict[str, tuple[float, float, float]],
    bndl_results: dict[str, tuple[float, float, float]],
    bndl_statistics: dict[str, Any],
    output_path: Path,
    uctta_results: dict[str, tuple[float, float, float]] | None = None,
    uctta_statistics: dict[str, Any] | None = None,
    aue_version: str | None = None,
    bndl_baseline_results: dict[str, tuple[float, float, float]] | None = None,
) -> None:
    """Create comprehensive comparison plots between SAM-2, BNDL baseline, and BNDL_AUE

    Args:
        aue_version: AUE版本标识，用于生成带版本后缀的文件夹名
        bndl_baseline_results: Optional BNDL baseline results for comparison
    """

    print("\nGenerating comprehensive comparison plots...")

    # Prepare data - use datasets that exist in SAM-2 and BNDL_AUE results
    datasets = [d for d in sam2_results if d in bndl_results]
    if not datasets:
        print("No common results to plot!")
        return

    # Extract metrics (only J&F needed)
    sam2_jf = [sam2_results[d][0] for d in datasets]
    bndl_aue_jf = [bndl_results[d][0] for d in datasets]

    # Extract BNDL baseline metrics if available (only J&F needed)
    bndl_baseline_jf = None
    if bndl_baseline_results:
        # Only use datasets that exist in BNDL baseline
        baseline_datasets = [d for d in datasets if d in bndl_baseline_results]
        if baseline_datasets:
            # Align with main datasets list
            bndl_baseline_jf = [bndl_baseline_results.get(d, (0, 0, 0))[0] for d in datasets]

    # Calculate improvements (only J&F)
    jf_improvements = [bndl_aue_jf[i] - sam2_jf[i] for i in range(len(datasets))]

    # Calculate improvements (BNDL baseline vs SAM-2) if baseline available
    jf_improvements_baseline_vs_sam = None
    if bndl_baseline_jf:
        jf_improvements_baseline_vs_sam = [bndl_baseline_jf[i] - sam2_jf[i] for i in range(len(datasets))]

    # 计算去除 MOSE 的宏平均 (排除 MOSE_train 和 MOSE_val)
    averages_excl_mose = None
    non_mose_idx = [i for i, d in enumerate(datasets) if not d.startswith("MOSE")]
    if non_mose_idx:
        avg_sam2_jf = float(np.mean([sam2_jf[i] for i in non_mose_idx]))
        avg_bndl_aue_jf = float(np.mean([bndl_aue_jf[i] for i in non_mose_idx]))

        averages_excl_mose = {
            "sam2": {"jf": avg_sam2_jf},
            "bndl_aue": {"jf": avg_bndl_aue_jf},
            "improvements": {
                "jf": avg_bndl_aue_jf - avg_sam2_jf,
            },
            "datasets_count": len(non_mose_idx),
        }

        # Add BNDL baseline averages if available
        if bndl_baseline_jf:
            avg_bndl_baseline_jf = float(np.mean([bndl_baseline_jf[i] for i in non_mose_idx]))
            averages_excl_mose["bndl_baseline"] = {
                "jf": avg_bndl_baseline_jf,
            }
            averages_excl_mose["improvements_baseline_vs_sam"] = {
                "jf": avg_bndl_baseline_jf - avg_sam2_jf,
            }

    # Set up plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Compact layout: only Improvement Summary + Summary Statistics
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 1, hspace=0.5, wspace=0.4)

    # Main title
    title = "SAM-2 vs BNDL vs BNDL_AUE Zero-shot Evaluation (Compact)"
    if not bndl_baseline_jf:
        title = "SAM-2 vs BNDL_AUE Zero-shot Evaluation (Compact)"
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.95)

    # Improvement Summary (top) - 横坐标为improvement，纵坐标为数据集
    ax4 = fig.add_subplot(gs[0, 0])
    y_imp = np.arange(len(datasets))

    # 使用水平条形图（barh），只显示 J&F
    ax4.barh(y_imp, jf_improvements, alpha=0.85, color="steelblue", label="J&F")

    ax4.set_title("Improvement Summary (BNDL_AUE vs SAM-2)", fontweight="bold", fontsize=14)
    ax4.set_xlabel("Improvement (BNDL_AUE - SAM-2)", fontsize=11)
    ax4.set_ylabel("Dataset", fontsize=11)
    ax4.set_yticks(y_imp)
    ax4.set_yticklabels(datasets, fontsize=10)
    ax4.legend(fontsize=10)
    ax4.axvline(x=0, color="black", linestyle="-", alpha=0.3)
    ax4.grid(True, alpha=0.3, axis="x")
    # 反转y轴顺序，使第一个数据集在顶部
    ax4.invert_yaxis()

    # Summary Statistics (bottom) - 横排为数据集，纵向为方法，只显示ΔJ&F
    ax9 = fig.add_subplot(gs[1, 0])
    ax9.axis("off")

    # Create transposed summary table (methods as rows, datasets as columns)
    # 表头：Method + 各数据集
    col_labels = ["Method"] + datasets

    # 添加平均列（不含MOSE）
    if averages_excl_mose:
        col_labels.append("AVG (excl MOSE)")

    # 表数据：根据是否有BNDL baseline决定行数
    summary_data = []

    # 第一行：SAM-2 J&F
    row_sam2 = ["SAM-2 J&F"]
    for i in range(len(datasets)):
        row_sam2.append(f"{sam2_jf[i]:.2f}")
    if averages_excl_mose:
        row_sam2.append(f"{averages_excl_mose['sam2']['jf']:.2f}")
    summary_data.append(row_sam2)

    # 第二行：BNDL baseline J&F (if available)
    if bndl_baseline_jf:
        row_bndl_baseline = ["BNDL J&F"]
        for i in range(len(datasets)):
            row_bndl_baseline.append(f"{bndl_baseline_jf[i]:.2f}")
        if averages_excl_mose and "bndl_baseline" in averages_excl_mose:
            row_bndl_baseline.append(f"{averages_excl_mose['bndl_baseline']['jf']:.2f}")
        summary_data.append(row_bndl_baseline)

    # BNDL_AUE J&F
    row_bndl_aue = ["BNDL_AUE J&F"]
    for i in range(len(datasets)):
        row_bndl_aue.append(f"{bndl_aue_jf[i]:.2f}")
    if averages_excl_mose:
        row_bndl_aue.append(f"{averages_excl_mose['bndl_aue']['jf']:.2f}")
    summary_data.append(row_bndl_aue)

    # ΔJ&F (BNDL baseline vs SAM-2) if available
    if bndl_baseline_jf and jf_improvements_baseline_vs_sam:
        row_delta_baseline_sam = ["ΔJ&F (BNDL-SAM)"]
        for i in range(len(datasets)):
            row_delta_baseline_sam.append(f"{jf_improvements_baseline_vs_sam[i]:+.2f}")
        if averages_excl_mose and "improvements_baseline_vs_sam" in averages_excl_mose:
            row_delta_baseline_sam.append(f"{averages_excl_mose['improvements_baseline_vs_sam']['jf']:+.2f}")
        summary_data.append(row_delta_baseline_sam)

    # ΔJ&F (BNDL_AUE vs SAM-2)
    row_delta = ["ΔJ&F (AUE-SAM)"]
    for i in range(len(datasets)):
        row_delta.append(f"{jf_improvements[i]:+.2f}")
    if averages_excl_mose:
        row_delta.append(f"{averages_excl_mose['improvements']['jf']:+.2f}")
    summary_data.append(row_delta)

    table = ax9.table(cellText=summary_data, colLabels=col_labels, cellLoc="center", loc="center", bbox=None)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    # Color code improvements in the ΔJ&F rows
    # Find the row indices for improvement rows
    # Order: SAM-2, BNDL (if exists), BNDL_AUE, ΔJ&F (BNDL-SAM) (if exists), ΔJ&F (AUE-SAM)
    if bndl_baseline_jf:
        # Find indices: BNDL-SAM, AUE-SAM
        delta_baseline_sam_idx = None
        delta_aue_sam_idx = None
        for idx, row in enumerate(summary_data):
            if row[0] == "ΔJ&F (BNDL-SAM)":
                delta_baseline_sam_idx = idx
            elif row[0] == "ΔJ&F (AUE-SAM)":
                delta_aue_sam_idx = idx
    else:
        # Only AUE-SAM
        delta_aue_sam_idx = len(summary_data) - 1

    for j in range(1, len(col_labels)):  # Skip first column (Method label)
        try:
            # Color code BNDL baseline vs SAM-2 improvement if available
            if bndl_baseline_jf and delta_baseline_sam_idx is not None:
                val_baseline_sam = float(summary_data[delta_baseline_sam_idx][j])
                row_num_baseline_sam = delta_baseline_sam_idx + 1  # +1 because header is row 0
                if val_baseline_sam > 0:
                    table[(row_num_baseline_sam, j)].set_facecolor("#ADD8E6")  # Light blue
                elif val_baseline_sam < 0:
                    table[(row_num_baseline_sam, j)].set_facecolor("#FFB6C1")  # Light red

            # Color code BNDL_AUE vs SAM-2 improvement
            if delta_aue_sam_idx is not None:
                val_aue_sam = float(summary_data[delta_aue_sam_idx][j])
                row_num_aue_sam = delta_aue_sam_idx + 1
                if val_aue_sam > 0:
                    table[(row_num_aue_sam, j)].set_facecolor("#90EE90")  # Light green
                elif val_aue_sam < 0:
                    table[(row_num_aue_sam, j)].set_facecolor("#FFB6C1")  # Light red
        except (ValueError, IndexError):
            pass

    title = "Summary Statistics (ΔJ&F Focus)"
    if bndl_baseline_jf:
        title = "Summary Statistics: SAM-2 vs BNDL vs BNDL_AUE"
    ax9.set_title(title, fontweight="bold", fontsize=12, pad=20)

    # Save plots with AUE version suffix if provided
    if aue_version:
        plots_dir = output_path / f"comparison_plots_AUE_{aue_version}"
    else:
        plots_dir = output_path / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)

    plot_path = plots_dir / "sam2_vs_bndl_comprehensive_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.savefig(plots_dir / "sam2_vs_bndl_comprehensive_comparison.pdf", bbox_inches="tight")

    print(f"Comprehensive comparison plots saved to: {plot_path}")

    # Save CSV summary (only J&F)
    csv_file = plots_dir / "comparison_summary.csv"
    csv_data = []
    for i, dataset in enumerate(datasets):
        row = {
            "Dataset": dataset,
            "SAM2_JF": sam2_jf[i],
            "BNDL_AUE_JF": bndl_aue_jf[i],
            "JF_Improvement_AUE_SAM": jf_improvements[i],
        }
        if bndl_baseline_jf:
            row.update(
                {
                    "BNDL_JF": bndl_baseline_jf[i],
                    "JF_Improvement_BNDL_SAM": jf_improvements_baseline_vs_sam[i],
                }
            )
        csv_data.append(row)

    # CSV 中追加平均行（不含 MOSE_train/val）
    if averages_excl_mose:
        avg_row = {
            "Dataset": "AVG(excl MOSE)",
            "SAM2_JF": averages_excl_mose["sam2"]["jf"],
            "BNDL_AUE_JF": averages_excl_mose["bndl_aue"]["jf"],
            "JF_Improvement_AUE_SAM": averages_excl_mose["improvements"]["jf"],
        }
        if "bndl_baseline" in averages_excl_mose:
            avg_row.update(
                {
                    "BNDL_JF": averages_excl_mose["bndl_baseline"]["jf"],
                    "JF_Improvement_BNDL_SAM": averages_excl_mose["improvements_baseline_vs_sam"]["jf"],
                }
            )
        csv_data.append(avg_row)

    df_summary = pd.DataFrame(csv_data)
    df_summary.to_csv(csv_file, index=False)
    print(f"Summary CSV saved to: {csv_file}")


def create_ua_shift_analysis_plots(
    bndl_statistics: dict[str, Any],
    sam2_results: dict[str, tuple[float, float, float]],
    bndl_results: dict[str, tuple[float, float, float]],
    output_path: Path,
    uctta_statistics: dict[str, Any] | None = None,
    uctta_results: dict[str, tuple[float, float, float]] | None = None,
    ur_ern_results: dict[str, tuple[float, float, float]] | None = None,
    bndl_pure_statistics: dict[str, Any] | None = None,
    bndl_pure_results: dict[str, tuple[float, float, float]] | None = None,
    source_domain: str = "MOSE_train",
    sam2_root_override: Path | None = None,
    bndl_root_override: Path | None = None,
    uctta_root_override: Path | None = None,
    ur_ern_root_override: Path | None = None,
    bndl_pure_root_override: Path | None = None,
    aue_version: str | None = None,
) -> None:
    """Create UA (Uncertainty-Accuracy) shift analysis plots

    Args:
        bndl_statistics: Dictionary containing BNDL_AUE statistics per dataset
        sam2_results: SAM-2 performance results
        bndl_results: BNDL_AUE performance results
        output_path: Output directory for plots
        uctta_statistics: Optional UCTTA statistics per dataset
        uctta_results: Optional UCTTA performance results
        ur_ern_results: Optional UR-ERN performance results
        bndl_pure_statistics: Optional BNDL (pure) statistics per dataset
        bndl_pure_results: Optional BNDL (pure) performance results
        source_domain: Source domain for shift comparison (default: MOSE_train - fine-tune domain)
    """
    print("\nGenerating UA shift analysis plots using BNDL_AUE...")

    # Define unified color scheme for all methods
    METHOD_COLORS = {
        "SAM-2": "#95A5A6",  # Gray - baseline
        "UCTTA": "#FF6B6B",  # Red/Pink - adaptation method
        "BNDL_AUE": "#4ECDC4",  # Teal/Cyan - our method (BNDL+AUE)
        "BNDL": "#2E86AB",  # Dark blue - BNDL pure
        "UR-ERN": "#95E1D3",  # Light teal/mint - alternative method
    }

    # Extract datasets
    datasets = list(bndl_statistics.keys())
    if not datasets:
        print("No BNDL statistics available for UA analysis!")
        return

    # Extract uncertainty metrics from BNDL statistics
    bndl_uncertainty_data = {}
    for dataset_name in datasets:
        stats = bndl_statistics[dataset_name]
        if stats:
            # Look for pixel_uncertainty in statistics
            unc_keys = [k for k in stats if "pixel_uncertainty" in k.lower()]
            if unc_keys:
                # Average all uncertainty values for this dataset
                unc_values = [stats[k] for k in unc_keys if isinstance(stats[k], (int, float))]
                if unc_values:
                    bndl_uncertainty_data[dataset_name] = float(np.mean(unc_values))

    # Extract uncertainty metrics from UCTTA statistics (if available)
    uctta_uncertainty_data = {}
    if uctta_statistics:
        for dataset_name in datasets:
            if dataset_name in uctta_statistics:
                stats = uctta_statistics[dataset_name]
                if stats and "pixel_uncertainty_mean" in stats:
                    uctta_uncertainty_data[dataset_name] = float(stats["pixel_uncertainty_mean"])

    # Extract uncertainty metrics from UR-ERN statistics (if available, passed as function parameter)
    # Note: UR-ERN statistics need to be passed via function call or loaded from ur_ern_root
    ur_ern_uncertainty_data = {}
    # Will be populated after loading UR-ERN PCC data below

    if not bndl_uncertainty_data:
        print("No uncertainty data found in BNDL statistics!")
        return

    has_uctta = len(uctta_uncertainty_data) > 0
    if has_uctta:
        print(f"Including UCTTA data for {len(uctta_uncertainty_data)} datasets in UA analysis")

    # Get performance metrics for BNDL
    bndl_performance_data = {}
    for dataset in bndl_uncertainty_data:
        if dataset in bndl_results:
            jf, j, f = bndl_results[dataset]
            bndl_performance_data[dataset] = {"jf": jf, "j": j, "f": f}

    # Get performance metrics for UCTTA
    uctta_performance_data = {}
    if has_uctta and uctta_results:
        for dataset in uctta_uncertainty_data:
            if dataset in uctta_results:
                jf, j, f = uctta_results[dataset]
                uctta_performance_data[dataset] = {"jf": jf, "j": j, "f": f}

    if len(bndl_performance_data) < 2:
        print("Not enough data points for UA shift analysis!")
        return

    # Determine figure layout: put all ΔPCC plots together
    # Row 0: Basic UA analysis (U vs Perf, U comparison, Improvement vs U, U distribution)
    # Row 1: All ΔPCC plots side by side (BNDL_AUE, BNDL, UCTTA, UR-ERN) - up to 4 plots
    # Row 2: J&F Performance comparison + Summary table
    fig = plt.figure(figsize=(28, 20))
    gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.4)
    title = "UA Consistency Analysis"

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # 1. BNDL: Uncertainty vs J&F Score scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    bndl_datasets_list = list(bndl_performance_data.keys())
    bndl_x_unc = [bndl_uncertainty_data[d] for d in bndl_datasets_list]
    bndl_y_jf = [bndl_performance_data[d]["jf"] for d in bndl_datasets_list]

    # Color source domain differently
    bndl_colors = ["red" if d == source_domain else "blue" for d in bndl_datasets_list]
    ax1.scatter(bndl_x_unc, bndl_y_jf, c=bndl_colors, s=100, alpha=0.6, edgecolors="black", label="BNDL_AUE")

    # Add dataset labels
    for i, dataset in enumerate(bndl_datasets_list):
        ax1.annotate(dataset, (bndl_x_unc[i], bndl_y_jf[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    # Fit trend line
    if len(bndl_x_unc) > 1:
        z = np.polyfit(bndl_x_unc, bndl_y_jf, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(bndl_x_unc), max(bndl_x_unc), 100)
        ax1.plot(x_line, p(x_line), "b--", alpha=0.5, linewidth=2, label=f"BNDL_AUE Trend: y={z[0]:.2f}x+{z[1]:.2f}")

        # Calculate correlation
        corr = np.corrcoef(bndl_x_unc, bndl_y_jf)[0, 1]
        ax1.text(0.05, 0.95, f"BNDL_AUE Correlation: {corr:.3f}", transform=ax1.transAxes, fontsize=9, bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5), verticalalignment="top")

    ax1.set_xlabel("Pixel Uncertainty (mean)", fontsize=11)
    ax1.set_ylabel("J&F Score", fontsize=11)
    ax1.set_title("BNDL_AUE: Uncertainty vs Performance", fontweight="bold", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # NOTE: ax2 will be created after loading all PCC/uncertainty data to include UR-ERN

    # 3. Improvement vs Uncertainty (BNDL vs SAM-2) - MOVED to gs[0, 2]
    ax3 = fig.add_subplot(gs[0, 2])
    common_datasets = [d for d in bndl_datasets_list if d in sam2_results and d in bndl_results]
    improvements = [bndl_results[d][0] - sam2_results[d][0] for d in common_datasets]
    uncertainties = [bndl_uncertainty_data[d] for d in common_datasets]

    colors_imp = ["red" if d == source_domain else "purple" for d in common_datasets]
    ax3.scatter(uncertainties, improvements, s=100, alpha=0.6, c=colors_imp, edgecolors="black")

    for i, dataset in enumerate(common_datasets):
        ax3.annotate(dataset, (uncertainties[i], improvements[i]), xytext=(5, 5), textcoords="offset points", fontsize=9)

    if len(uncertainties) > 1:
        z_imp = np.polyfit(uncertainties, improvements, 1)
        p_imp = np.poly1d(z_imp)
        x_line_imp = np.linspace(min(uncertainties), max(uncertainties), 100)
        ax3.plot(x_line_imp, p_imp(x_line_imp), "g--", alpha=0.5, linewidth=2)

        corr_imp = np.corrcoef(uncertainties, improvements)[0, 1]
        ax3.text(0.05, 0.95, f"Correlation: {corr_imp:.3f}", transform=ax3.transAxes, fontsize=10, bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5), verticalalignment="top")

    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)
    ax3.set_xlabel("Pixel Uncertainty", fontsize=11)
    ax3.set_ylabel("BNDL_AUE Improvement (ΔJ&F)", fontsize=11)
    ax3.set_title("BNDL_AUE Improvement vs Uncertainty", fontweight="bold", fontsize=12)
    ax3.grid(True, alpha=0.3)

    # 4. Uncertainty distribution across datasets (BNDL) - MOVED to gs[0, 3]
    ax4 = fig.add_subplot(gs[0, 3])
    dataset_names_short = [d[:8] for d in bndl_datasets_list]  # Shorten names for readability
    x_pos = np.arange(len(bndl_datasets_list))

    bars4 = ax4.bar(x_pos, bndl_x_unc, color=bndl_colors, alpha=0.7, edgecolor="black")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(dataset_names_short, rotation=45, ha="right", fontsize=9)
    ax4.set_ylabel("Mean Pixel Uncertainty (BNDL_AUE)", fontsize=11)
    ax4.set_title("BNDL_AUE Uncertainty Distribution Across Datasets", fontweight="bold", fontsize=12)
    ax4.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    # === ROW 1: ALL ΔPCC PLOTS TOGETHER ===
    # Load per-dataset PCC and NLL data for SAM-2, BNDL, UCTTA, UR-ERN

    # === 并行加载所有评估结果文件 ===
    print(f"Loading evaluation results for {len(bndl_datasets_list)} datasets across multiple methods...")

    # 构建所有需要加载的文件路径
    files_to_load = []

    # SAM-2 files
    sam2_root = sam2_root_override if sam2_root_override is not None else (output_path / "sam2_results")
    for dataset_name in bndl_datasets_list:
        path = sam2_root / f"{dataset_name.lower()}_sam2_eval" / f"{dataset_name.lower()}_zeroshot_results.json"
        files_to_load.append(("sam2", dataset_name, path))

    # BNDL files
    bndl_root = bndl_root_override if bndl_root_override is not None else (output_path / "bndl_results")
    for dataset_name in bndl_datasets_list:
        path = bndl_root / f"{dataset_name.lower()}_bndl_eval" / f"{dataset_name.lower()}_zeroshot_results.json"
        files_to_load.append(("bndl", dataset_name, path))

    # UCTTA files
    uctta_root = uctta_root_override if uctta_root_override is not None else (output_path / "sam2_uctta_results")
    for dataset_name in bndl_datasets_list:
        path = uctta_root / f"{dataset_name.lower()}_uctta_eval" / f"{dataset_name.lower()}_uctta_results.json"
        files_to_load.append(("uctta", dataset_name, path))

    # UR-ERN files
    ur_ern_root = ur_ern_root_override if ur_ern_root_override is not None else (output_path / "sam2_ur_ern_results")
    for dataset_name in bndl_datasets_list:
        path = ur_ern_root / f"{dataset_name.lower()}_ur_ern_eval" / f"{dataset_name.lower()}_ur_ern_results.json"
        files_to_load.append(("ur_ern", dataset_name, path))

    # BNDL pure files (optional)
    if bndl_pure_statistics or bndl_pure_root_override:
        bndl_pure_root = bndl_pure_root_override if bndl_pure_root_override is not None else (output_path / "bndl_results")
        for dataset_name in bndl_datasets_list:
            path = bndl_pure_root / f"{dataset_name.lower()}_bndl_eval" / f"{dataset_name.lower()}_zeroshot_results.json"
            files_to_load.append(("bndl_pure", dataset_name, path))

    # 并行加载所有文件
    loaded_data = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_info = {executor.submit(_load_eval_json, path): (method, dataset, path) for method, dataset, path in files_to_load}

        for future in as_completed(future_to_info):
            method, dataset, path = future_to_info[future]
            data = future.result()
            if data is not None:
                if method not in loaded_data:
                    loaded_data[method] = {}
                loaded_data[method][dataset] = data

    print(f"Successfully loaded {sum(len(v) for v in loaded_data.values())} evaluation files")

    # 从加载的数据中提取指标
    sam2_nll_data: dict[str, float] = {}
    for dataset, data in loaded_data.get("sam2", {}).items():
        if isinstance(data, dict) and "NLL" in data:
            nll_info = data["NLL"]
            if isinstance(nll_info, dict) and "metric_mean" in nll_info:
                sam2_nll_data[dataset] = float(nll_info["metric_mean"])

    accuracy_pcc_map: dict[str, float] = {}
    bndl_nll_data: dict[str, float] = {}
    for dataset, data in loaded_data.get("bndl", {}).items():
        if isinstance(data, dict) and "Accuracy" in data:
            acc_info = data["Accuracy"]
            if isinstance(acc_info, dict) and "correlation" in acc_info:
                corr_val = acc_info["correlation"]
                if corr_val != "NaN" and not (isinstance(corr_val, float) and np.isnan(corr_val)):
                    accuracy_pcc_map[dataset] = float(corr_val)
                else:
                    print(f"Warning: BNDL {dataset} has NaN correlation")

        if isinstance(data, dict) and "NLL" in data:
            nll_info = data["NLL"]
            if isinstance(nll_info, dict) and "metric_mean" in nll_info:
                bndl_nll_data[dataset] = float(nll_info["metric_mean"])

    uctta_accuracy_pcc_map: dict[str, float] = {}
    uctta_nll_data: dict[str, float] = {}
    for dataset, data in loaded_data.get("uctta", {}).items():
        if isinstance(data, dict) and "Accuracy" in data:
            acc_info = data["Accuracy"]
            if isinstance(acc_info, dict) and "correlation" in acc_info:
                corr_val = acc_info["correlation"]
                if corr_val != "NaN" and not (isinstance(corr_val, float) and np.isnan(corr_val)):
                    uctta_accuracy_pcc_map[dataset] = float(corr_val)
                else:
                    print(f"Warning: UCTTA {dataset} has NaN correlation")

        if isinstance(data, dict) and "NLL" in data:
            nll_info = data["NLL"]
            if isinstance(nll_info, dict) and "metric_mean" in nll_info:
                uctta_nll_data[dataset] = float(nll_info["metric_mean"])

    ur_ern_accuracy_pcc_map: dict[str, float] = {}
    ur_ern_nll_data: dict[str, float] = {}
    for dataset, data in loaded_data.get("ur_ern", {}).items():
        if isinstance(data, dict) and "Accuracy" in data:
            acc_info = data["Accuracy"]
            if isinstance(acc_info, dict):
                if "correlation" in acc_info:
                    corr_val = acc_info["correlation"]
                    if corr_val != "NaN" and not (isinstance(corr_val, float) and np.isnan(corr_val)):
                        ur_ern_accuracy_pcc_map[dataset] = float(corr_val)
                    else:
                        print(f"Warning: UR-ERN {dataset} has NaN correlation (likely constant uncertainty)")
                if "uncertainty_mean" in acc_info:
                    ur_ern_uncertainty_data[dataset] = float(acc_info["uncertainty_mean"])

        if isinstance(data, dict) and "NLL" in data:
            nll_info = data["NLL"]
            if isinstance(nll_info, dict) and "metric_mean" in nll_info:
                ur_ern_nll_data[dataset] = float(nll_info["metric_mean"])

    has_ur_ern = len(ur_ern_accuracy_pcc_map) > 0
    if has_ur_ern:
        print(f"Including UR-ERN data for {len(ur_ern_accuracy_pcc_map)} datasets in UA analysis")

    bndl_pure_accuracy_pcc_map: dict[str, float] = {}
    bndl_pure_nll_data: dict[str, float] = {}
    for dataset, data in loaded_data.get("bndl_pure", {}).items():
        if isinstance(data, dict) and "Accuracy" in data:
            acc_info = data["Accuracy"]
            if isinstance(acc_info, dict) and "correlation" in acc_info:
                corr_val = acc_info["correlation"]
                if corr_val != "NaN" and not (isinstance(corr_val, float) and np.isnan(corr_val)):
                    bndl_pure_accuracy_pcc_map[dataset] = float(corr_val)
                else:
                    print(f"Warning: BNDL (pure) {dataset} has NaN correlation")

        if isinstance(data, dict) and "NLL" in data:
            nll_info = data["NLL"]
            if isinstance(nll_info, dict) and "metric_mean" in nll_info:
                bndl_pure_nll_data[dataset] = float(nll_info["metric_mean"])

    has_bndl_pure = len(bndl_pure_accuracy_pcc_map) > 0
    if has_bndl_pure:
        print(f"Including BNDL (pure) data for {len(bndl_pure_accuracy_pcc_map)} datasets in UA analysis")

    # Update title now that we know which methods are available
    title_parts = ["UA Consistency Analysis: BNDL_AUE"]
    if has_bndl_pure:
        title_parts.append(" vs BNDL")
    if has_uctta:
        title_parts.append(" vs UCTTA")
    if has_ur_ern:
        title_parts.append(" vs UR-ERN")
    fig.suptitle("".join(title_parts), fontsize=16, fontweight="bold", y=0.98)

    # === NOW CREATE ax2: NLL comparison with all methods ===
    ax2 = fig.add_subplot(gs[0, 1])
    common_datasets = sorted(set(sam2_nll_data.keys()) | set(bndl_nll_data.keys()) | set(uctta_nll_data.keys()) | set(ur_ern_nll_data.keys()) | set(bndl_pure_nll_data.keys()))

    if common_datasets:
        # Multi-method NLL comparison
        x_pos = np.arange(len(common_datasets))

        # Determine number of methods and bar width
        num_methods = 0
        if sam2_nll_data:
            num_methods += 1
        if bndl_nll_data:
            num_methods += 1
        if has_bndl_pure and bndl_pure_nll_data:
            num_methods += 1
        if has_uctta and uctta_nll_data:
            num_methods += 1
        if has_ur_ern and ur_ern_nll_data:
            num_methods += 1

        width = 0.22 if num_methods <= 3 else (0.18 if num_methods == 4 else 0.15)
        offset = 0

        # Plot SAM-2 baseline first
        if sam2_nll_data:
            sam2_nll_vals = [sam2_nll_data.get(d, np.nan) for d in common_datasets]
            ax2.barh(x_pos + offset, sam2_nll_vals, width, label="SAM-2", color=METHOD_COLORS["SAM-2"], alpha=0.8)
            offset += width

        # Plot UCTTA if available
        if has_uctta and uctta_nll_data:
            uctta_nll_vals = [uctta_nll_data.get(d, np.nan) for d in common_datasets]
            ax2.barh(x_pos + offset, uctta_nll_vals, width, label="UCTTA", color=METHOD_COLORS["UCTTA"], alpha=0.8)
            offset += width

        # Plot BNDL_AUE
        if bndl_nll_data:
            bndl_nll_vals = [bndl_nll_data.get(d, np.nan) for d in common_datasets]
            ax2.barh(x_pos + offset, bndl_nll_vals, width, label="BNDL_AUE", color=METHOD_COLORS["BNDL_AUE"], alpha=0.8)
            offset += width

        # Plot BNDL (pure) if available
        if has_bndl_pure and bndl_pure_nll_data:
            bndl_pure_nll_vals = [bndl_pure_nll_data.get(d, np.nan) for d in common_datasets]
            ax2.barh(x_pos + offset, bndl_pure_nll_vals, width, label="BNDL", color=METHOD_COLORS["BNDL"], alpha=0.8)
            offset += width

        # Plot UR-ERN if available
        if has_ur_ern and ur_ern_nll_data:
            ur_ern_nll_vals = [ur_ern_nll_data.get(d, np.nan) for d in common_datasets]
            ax2.barh(x_pos + offset, ur_ern_nll_vals, width, label="UR-ERN", color=METHOD_COLORS["UR-ERN"], alpha=0.8)
            offset += width

        # Center the y-tick labels
        center_offset = (num_methods - 1) * width / 2
        ax2.set_yticks(x_pos + center_offset)
        ax2.set_yticklabels(common_datasets, fontsize=9)
        ax2.set_xlabel("Mean NLL (Negative Log-Likelihood)", fontsize=11)

        # Update title based on available methods
        title_methods = []
        if sam2_nll_data:
            title_methods.append("SAM-2")
        if has_uctta and uctta_nll_data:
            title_methods.append("UCTTA")
        if bndl_nll_data:
            title_methods.append("BNDL_AUE")
        if has_bndl_pure and bndl_pure_nll_data:
            title_methods.append("BNDL")
        if has_ur_ern and ur_ern_nll_data:
            title_methods.append("UR-ERN")
        ax2.set_title(f"NLL: {' vs '.join(title_methods)}", fontweight="bold", fontsize=12)

        ax2.legend(fontsize=9)
        ax2.invert_xaxis()  # Lower NLL is better, so invert for better visualization
    else:
        # Just BNDL NLL ranking
        sorted_datasets = sorted(bndl_nll_data.keys(), key=lambda x: bndl_nll_data[x])
        sorted_nll = [bndl_nll_data[d] for d in sorted_datasets]
        bar_colors = ["red" if d == source_domain else "steelblue" for d in sorted_datasets]

        bars = ax2.barh(range(len(sorted_datasets)), sorted_nll, color=bar_colors, alpha=0.7)
        ax2.set_yticks(range(len(sorted_datasets)))
        ax2.set_yticklabels(sorted_datasets, fontsize=9)
        ax2.set_xlabel("Mean NLL (lower is better)", fontsize=11)
        ax2.set_title("BNDL_AUE NLL Ranking by Dataset", fontweight="bold", fontsize=12)
        ax2.invert_xaxis()  # Lower NLL is better

        # Add value labels
        for i, (_bar, val) in enumerate(zip(bars, sorted_nll, strict=True)):
            ax2.text(val, i, f" {val:.4f}", va="center", fontsize=8)

    ax2.grid(True, alpha=0.3, axis="x")

    # === ROW 1: ALL ΔPCC PLOTS TOGETHER ===
    # Dynamically assign column positions based on available methods
    current_col = 0

    # 5. BNDL_AUE ΔPCC (gs[1, 0]) - sorted by absolute value
    if source_domain in accuracy_pcc_map:
        ax5 = fig.add_subplot(gs[1, current_col])
        source_pcc = accuracy_pcc_map[source_domain]
        target_datasets = [d for d in bndl_datasets_list if d in accuracy_pcc_map and d != source_domain]
        delta_pcc = {d: (accuracy_pcc_map[d] - source_pcc) for d in target_datasets}

        sorted_items = sorted(delta_pcc.items(), key=lambda x: abs(x[1]))
        labels = [k for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        y_pos = np.arange(len(labels))

        bars = ax5.barh(y_pos, values, color=["red" if v < 0 else "green" for v in values], alpha=0.7, edgecolor="black")
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(labels, fontsize=9)
        ax5.set_xlabel(f"ΔPCC relative to {source_domain}", fontsize=11)
        ax5.set_title("BNDL_AUE: UA Shift (ΔPCC)", fontweight="bold", fontsize=12)
        ax5.axvline(x=0.0, color="black", linestyle="--", alpha=0.4)
        ax5.grid(True, alpha=0.3, axis="x")

        for bar in bars:
            width = bar.get_width()
            ax5.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height() / 2, f"{width:+.3f}", va="center", ha="left" if width >= 0 else "right", fontsize=8)
        current_col += 1

    # 5b. BNDL (pure) ΔPCC - sorted by absolute value
    if bndl_pure_accuracy_pcc_map and source_domain in bndl_pure_accuracy_pcc_map and has_bndl_pure and current_col < 4:
        ax5b = fig.add_subplot(gs[1, current_col])
        source_pcc_bp = bndl_pure_accuracy_pcc_map[source_domain]
        target_datasets_bp = [d for d in bndl_datasets_list if d in bndl_pure_accuracy_pcc_map and d != source_domain]
        delta_pcc_bp = {d: (bndl_pure_accuracy_pcc_map[d] - source_pcc_bp) for d in target_datasets_bp}
        sorted_items_bp = sorted(delta_pcc_bp.items(), key=lambda x: abs(x[1]))
        labels_bp = [k for k, _ in sorted_items_bp]
        values_bp = [v for _, v in sorted_items_bp]
        y_pos_bp = np.arange(len(labels_bp))
        bars_bp = ax5b.barh(y_pos_bp, values_bp, color=["red" if v < 0 else "green" for v in values_bp], alpha=0.7, edgecolor="black")
        ax5b.set_yticks(y_pos_bp)
        ax5b.set_yticklabels(labels_bp, fontsize=9)
        ax5b.set_xlabel(f"ΔPCC relative to {source_domain}", fontsize=11)
        ax5b.set_title("BNDL: UA Shift (ΔPCC)", fontweight="bold", fontsize=12)
        ax5b.axvline(x=0.0, color="black", linestyle="--", alpha=0.4)
        ax5b.grid(True, alpha=0.3, axis="x")
        for bar in bars_bp:
            width = bar.get_width()
            ax5b.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height() / 2, f"{width:+.3f}", va="center", ha="left" if width >= 0 else "right", fontsize=8)
        current_col += 1

    # 6. UCTTA ΔPCC - sorted by absolute value
    if uctta_accuracy_pcc_map and source_domain in uctta_accuracy_pcc_map and has_uctta and current_col < 4:
        ax6 = fig.add_subplot(gs[1, current_col])
        source_pcc_u = uctta_accuracy_pcc_map[source_domain]
        target_datasets_u = [d for d in bndl_datasets_list if d in uctta_accuracy_pcc_map and d != source_domain]
        delta_pcc_u = {d: (uctta_accuracy_pcc_map[d] - source_pcc_u) for d in target_datasets_u}
        sorted_items_u = sorted(delta_pcc_u.items(), key=lambda x: abs(x[1]))
        labels_u = [k for k, _ in sorted_items_u]
        values_u = [v for _, v in sorted_items_u]
        y_pos_u = np.arange(len(labels_u))
        bars_u = ax6.barh(y_pos_u, values_u, color=["red" if v < 0 else "green" for v in values_u], alpha=0.7, edgecolor="black")
        ax6.set_yticks(y_pos_u)
        ax6.set_yticklabels(labels_u, fontsize=9)
        ax6.set_xlabel(f"ΔPCC relative to {source_domain}", fontsize=11)
        ax6.set_title("UCTTA: UA Shift (ΔPCC)", fontweight="bold", fontsize=12)
        ax6.axvline(x=0.0, color="black", linestyle="--", alpha=0.4)
        ax6.grid(True, alpha=0.3, axis="x")
        for bar in bars_u:
            width = bar.get_width()
            ax6.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height() / 2, f"{width:+.3f}", va="center", ha="left" if width >= 0 else "right", fontsize=8)
        current_col += 1

    # 7. UR-ERN ΔPCC - sorted by absolute value
    if ur_ern_accuracy_pcc_map and source_domain in ur_ern_accuracy_pcc_map and has_ur_ern and current_col < 4:
        ax7 = fig.add_subplot(gs[1, current_col])
        source_pcc_r = ur_ern_accuracy_pcc_map[source_domain]
        target_datasets_r = [d for d in bndl_datasets_list if d in ur_ern_accuracy_pcc_map and d != source_domain]
        delta_pcc_r = {d: (ur_ern_accuracy_pcc_map[d] - source_pcc_r) for d in target_datasets_r}
        sorted_items_r = sorted(delta_pcc_r.items(), key=lambda x: abs(x[1]))
        labels_r = [k for k, _ in sorted_items_r]
        values_r = [v for _, v in sorted_items_r]
        y_pos_r = np.arange(len(labels_r))
        bars_r = ax7.barh(y_pos_r, values_r, color=["red" if v < 0 else "green" for v in values_r], alpha=0.7, edgecolor="black")
        ax7.set_yticks(y_pos_r)
        ax7.set_yticklabels(labels_r, fontsize=9)
        ax7.set_xlabel(f"ΔPCC relative to {source_domain}", fontsize=11)
        ax7.set_title("UR-ERN: UA Shift (ΔPCC)", fontweight="bold", fontsize=12)
        ax7.axvline(x=0.0, color="black", linestyle="--", alpha=0.4)
        ax7.grid(True, alpha=0.3, axis="x")
        for bar in bars_r:
            width = bar.get_width()
            ax7.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height() / 2, f"{width:+.3f}", va="center", ha="left" if width >= 0 else "right", fontsize=8)
        current_col += 1

    # 8. J&F Performance Comparison - MOVED to Row 2 to avoid overlap with ΔPCC plots
    ax8 = fig.add_subplot(gs[2, 2:])

    # Get all datasets that have results from at least one method
    all_datasets_for_jf = sorted(
        set(sam2_results.keys())
        | set(bndl_results.keys())
        | (set(uctta_results.keys()) if uctta_results else set())
        | (set(bndl_pure_results.keys()) if bndl_pure_results else set())
        | (set(ur_ern_results.keys()) if ur_ern_results else set())
    )

    # Prepare data for J&F comparison
    sam_jf_vals = [sam2_results[d][0] if d in sam2_results else np.nan for d in all_datasets_for_jf]
    bndl_jf_vals = [bndl_results[d][0] if d in bndl_results else np.nan for d in all_datasets_for_jf]

    x_pos = np.arange(len(all_datasets_for_jf))

    # Determine number of methods and bar width
    num_methods = 2  # SAM-2 and BNDL_AUE are always present
    if has_bndl_pure and bndl_pure_results:
        num_methods += 1
    if has_uctta and uctta_results:
        num_methods += 1
    if has_ur_ern and ur_ern_results:
        num_methods += 1

    width = 0.22 if num_methods <= 3 else (0.18 if num_methods == 4 else 0.15)
    offset = 0

    # Plot SAM-2 baseline
    ax8.barh(x_pos + offset, sam_jf_vals, width, label="SAM-2", color=METHOD_COLORS["SAM-2"], alpha=0.8)
    offset += width

    # Plot UCTTA if available
    if has_uctta and uctta_results:
        uctta_jf_vals = [uctta_results[d][0] if d in uctta_results else np.nan for d in all_datasets_for_jf]
        ax8.barh(x_pos + offset, uctta_jf_vals, width, label="UCTTA", color=METHOD_COLORS["UCTTA"], alpha=0.8)
        offset += width

    # Plot BNDL_AUE
    ax8.barh(x_pos + offset, bndl_jf_vals, width, label="BNDL_AUE", color=METHOD_COLORS["BNDL_AUE"], alpha=0.8)
    offset += width

    # Plot BNDL (pure) if available
    if has_bndl_pure and bndl_pure_results:
        bndl_pure_jf_vals = [bndl_pure_results[d][0] if d in bndl_pure_results else np.nan for d in all_datasets_for_jf]
        ax8.barh(x_pos + offset, bndl_pure_jf_vals, width, label="BNDL", color=METHOD_COLORS["BNDL"], alpha=0.8)
        offset += width

    # Plot UR-ERN if available
    if has_ur_ern and ur_ern_results:
        ur_ern_jf_vals = [ur_ern_results[d][0] if d in ur_ern_results else np.nan for d in all_datasets_for_jf]
        ax8.barh(x_pos + offset, ur_ern_jf_vals, width, label="UR-ERN", color=METHOD_COLORS["UR-ERN"], alpha=0.8)
        offset += width

    # Center the y-tick labels
    center_offset = (num_methods - 1) * width / 2
    ax8.set_yticks(x_pos + center_offset)
    ax8.set_yticklabels(all_datasets_for_jf, fontsize=9)
    ax8.set_xlabel("J&F Score", fontsize=11)

    # Update title based on available methods
    title_methods = ["SAM-2"]
    if has_uctta and uctta_results:
        title_methods.append("UCTTA")
    title_methods.append("BNDL_AUE")
    if has_bndl_pure and bndl_pure_results:
        title_methods.append("BNDL")
    if has_ur_ern and ur_ern_results:
        title_methods.append("UR-ERN")
    ax8.set_title(f"J&F Performance: {' vs '.join(title_methods)}", fontweight="bold", fontsize=12)

    ax8.legend(fontsize=9, loc="lower right")
    ax8.grid(True, alpha=0.3, axis="x")
    ax8.set_xlim(0, 100)

    # 9. Summary statistics table (gs[2, 0:2]) - moved to row 2
    ax9 = fig.add_subplot(gs[2, 0:2])
    ax9.axis("off")

    # Create summary table
    table_data = []
    for dataset in bndl_datasets_list:
        unc = bndl_uncertainty_data.get(dataset, 0)
        jf = bndl_performance_data.get(dataset, {}).get("jf", 0)
        if dataset in sam2_results and dataset in bndl_results:
            improvement = bndl_results[dataset][0] - sam2_results[dataset][0]
        else:
            improvement = 0

        table_data.append([dataset, f"{unc:.4f}", f"{jf:.2f}", f"{improvement:+.2f}"])

    table = ax9.table(cellText=table_data, colLabels=["Dataset", "Uncertainty", "BNDL_AUE J&F", "Δ vs SAM-2"], cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Color code improvements
    for i in range(1, len(table_data) + 1):
        try:
            val = float(table_data[i - 1][3])
            if val > 0:
                table[(i, 3)].set_facecolor("#90EE90")  # Light green
            elif val < 0:
                table[(i, 3)].set_facecolor("#FFB6C1")  # Light red
        except (ValueError, IndexError):
            pass

    ax9.set_title("UA Summary Statistics", fontweight="bold", fontsize=12, pad=20)

    # Save plots with AUE version suffix if provided
    if aue_version:
        plots_dir = output_path / f"comparison_plots_AUE_{aue_version}"
    else:
        plots_dir = output_path / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)

    ua_plot_path = plots_dir / "ua_shift_analysis.png"
    plt.savefig(ua_plot_path, dpi=300, bbox_inches="tight")
    plt.savefig(plots_dir / "ua_shift_analysis.pdf", bbox_inches="tight")
    plt.close()

    print(f"UA shift analysis plots saved to: {ua_plot_path}")

    # Save UA PCC summary to CSV for downstream analysis
    if accuracy_pcc_map or bndl_pure_accuracy_pcc_map or uctta_accuracy_pcc_map or ur_ern_accuracy_pcc_map:
        import csv

        csv_path = plots_dir / "ua_pcc_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Dataset",
                    "PCC_U_vs_Acc_BNDL_AUE",
                    f"Delta_vs_{source_domain}_BNDL_AUE",
                    "PCC_U_vs_Acc_BNDL",
                    f"Delta_vs_{source_domain}_BNDL",
                    "PCC_U_vs_Acc_UCTTA",
                    f"Delta_vs_{source_domain}_UCTTA",
                    "PCC_U_vs_Acc_UR_ERN",
                    f"Delta_vs_{source_domain}_UR_ERN",
                ]
            )
            src_val = accuracy_pcc_map.get(source_domain)
            src_val_bp = bndl_pure_accuracy_pcc_map.get(source_domain)
            src_val_u = uctta_accuracy_pcc_map.get(source_domain)
            src_val_r = ur_ern_accuracy_pcc_map.get(source_domain)
            all_ds = sorted(set(list(accuracy_pcc_map.keys()) + list(bndl_pure_accuracy_pcc_map.keys()) + list(uctta_accuracy_pcc_map.keys()) + list(ur_ern_accuracy_pcc_map.keys())))
            for d in all_ds:
                pcc_b = accuracy_pcc_map.get(d)
                pcc_bp = bndl_pure_accuracy_pcc_map.get(d)
                pcc_u = uctta_accuracy_pcc_map.get(d)
                pcc_r = ur_ern_accuracy_pcc_map.get(d)
                delta_b = (pcc_b - src_val) if (pcc_b is not None and src_val is not None and d != source_domain) else (0.0 if d == source_domain and pcc_b is not None else "")
                delta_bp = (pcc_bp - src_val_bp) if (pcc_bp is not None and src_val_bp is not None and d != source_domain) else (0.0 if d == source_domain and pcc_bp is not None else "")
                delta_u = (pcc_u - src_val_u) if (pcc_u is not None and src_val_u is not None and d != source_domain) else (0.0 if d == source_domain and pcc_u is not None else "")
                delta_r = (pcc_r - src_val_r) if (pcc_r is not None and src_val_r is not None and d != source_domain) else (0.0 if d == source_domain and pcc_r is not None else "")
                writer.writerow(
                    [
                        d,
                        (f"{pcc_b:.6f}" if pcc_b is not None else ""),
                        (f"{delta_b:+.6f}" if isinstance(delta_b, float) else delta_b),
                        (f"{pcc_bp:.6f}" if pcc_bp is not None else ""),
                        (f"{delta_bp:+.6f}" if isinstance(delta_bp, float) else delta_bp),
                        (f"{pcc_u:.6f}" if pcc_u is not None else ""),
                        (f"{delta_u:+.6f}" if isinstance(delta_u, float) else delta_u),
                        (f"{pcc_r:.6f}" if pcc_r is not None else ""),
                        (f"{delta_r:+.6f}" if isinstance(delta_r, float) else delta_r),
                    ]
                )
        print(f"UA PCC summary CSV saved to: {csv_path}")


def create_pavpu_comparison_plots(
    output_path: Path,
    bndl_aue_statistics: dict[str, Any] | None = None,
    bndl_statistics: dict[str, Any] | None = None,
    uctta_statistics: dict[str, Any] | None = None,
    ur_ern_statistics: dict[str, Any] | None = None,
    datasets: list[str] | None = None,
):
    """
    Create TRUE PAvPU scatter plots (Uncertainty vs Accuracy, NO thresholds)
    PAvPU (Pseudo-mask Accuracy vs Prediction Uncertainty) shows calibration quality

    Args:
        output_path: Output directory for plots
        bndl_aue_statistics: BNDL+AUE statistics containing raw pixel samples
        bndl_statistics: BNDL statistics containing raw pixel samples
        uctta_statistics: UCTTA statistics containing raw pixel samples
        ur_ern_statistics: UR-ERN statistics containing raw pixel samples
        datasets: List of datasets to include in comparison
    """
    print("\n" + "=" * 80)
    print("Creating TRUE PAvPU Scatter Plots (Uncertainty vs Accuracy, NO thresholds)")
    print("=" * 80)

    # Collect raw PAvPU samples from all methods
    methods_with_pavpu = {}

    # Parse BNDL+AUE statistics (get raw samples)
    if bndl_aue_statistics:
        pavpu_data = _extract_pavpu_samples_from_statistics(bndl_aue_statistics, datasets)
        if pavpu_data:
            methods_with_pavpu["BNDL+AUE"] = pavpu_data

    # Parse BNDL statistics
    if bndl_statistics:
        pavpu_data = _extract_pavpu_samples_from_statistics(bndl_statistics, datasets)
        if pavpu_data:
            methods_with_pavpu["BNDL"] = pavpu_data

    # Parse UCTTA statistics
    if uctta_statistics:
        pavpu_data = _extract_pavpu_samples_from_statistics(uctta_statistics, datasets)
        if pavpu_data:
            methods_with_pavpu["UCTTA"] = pavpu_data

    # Parse UR-ERN statistics
    if ur_ern_statistics:
        pavpu_data = _extract_pavpu_samples_from_statistics(ur_ern_statistics, datasets)
        if pavpu_data:
            methods_with_pavpu["UR-ERN"] = pavpu_data

    # Check if we have any methods with PAvPU data
    if not methods_with_pavpu:
        print("⚠ No PAvPU data found for any method. Skipping PAvPU comparison plots.")
        print("   Make sure to run evaluation with --collect_statistics to gather PAvPU data.")
        return

    print(f"Found PAvPU raw samples for methods: {list(methods_with_pavpu.keys())}")
    for method, data in methods_with_pavpu.items():
        total_samples = sum(len(samples["uncertainty"]) for samples in data.values())
        print(f"  {method}: {len(data)} datasets, {total_samples} total pixel samples")

    # Create TRUE PAvPU scatter/density plots (uncertainty vs accuracy)
    _plot_pavpu_scatter(methods_with_pavpu, output_path)
    _plot_pavpu_hexbin(methods_with_pavpu, output_path)

    print(f"✓ TRUE PAvPU scatter plots saved to: {output_path / 'comparison_plots'}")


def _extract_pavpu_samples_from_statistics(statistics: dict[str, Any], datasets: list[str] | None = None) -> dict:
    """
    Extract raw PAvPU samples (uncertainty and accuracy) from statistics dictionary

    Args:
        statistics: Statistics dictionary containing raw pixel samples
        datasets: Optional list of datasets to filter

    Returns:
        dict with structure {dataset: {'uncertainty': [...], 'accuracy': [...]}}
    """
    pavpu_data = {}

    for key, value in statistics.items():
        # Parse keys like "GTEA_eval_pavpu_uncertainty_samples" or "GTEA_eval_pavpu_accuracy_samples"
        if "_pavpu_uncertainty_samples" in key:
            parts = key.split("_pavpu_uncertainty_samples")
            if len(parts) == 2:
                dataset_prefix = parts[0]
                # Extract dataset name (remove trailing _eval if present)
                dataset_name = dataset_prefix.replace("_eval", "")

                # Filter by dataset list if provided
                if datasets is not None and dataset_name not in datasets:
                    continue

                # Initialize dataset dict if needed
                if dataset_name not in pavpu_data:
                    pavpu_data[dataset_name] = {"uncertainty": [], "accuracy": []}

                # Store uncertainty samples
                if isinstance(value, list):
                    pavpu_data[dataset_name]["uncertainty"] = value

        elif "_pavpu_accuracy_samples" in key:
            parts = key.split("_pavpu_accuracy_samples")
            if len(parts) == 2:
                dataset_prefix = parts[0]
                dataset_name = dataset_prefix.replace("_eval", "")

                # Filter by dataset list if provided
                if datasets is not None and dataset_name not in datasets:
                    continue

                # Initialize dataset dict if needed
                if dataset_name not in pavpu_data:
                    pavpu_data[dataset_name] = {"uncertainty": [], "accuracy": []}

                # Store accuracy samples
                if isinstance(value, list):
                    pavpu_data[dataset_name]["accuracy"] = value

    # Filter out datasets with missing or empty data
    pavpu_data = {dataset: samples for dataset, samples in pavpu_data.items() if samples["uncertainty"] and samples["accuracy"] and len(samples["uncertainty"]) == len(samples["accuracy"])}

    return pavpu_data


def _plot_pavpu_scatter(methods_with_pavpu: dict, output_path: Path):
    """
    Plot TRUE PAvPU scatter plots: Uncertainty (X) vs Accuracy (Y)
    Each point represents a pixel. Shows calibration quality directly.
    """
    print("\n  Creating PAvPU scatter plots...")

    # Get all datasets
    all_datasets = set()
    for method_data in methods_with_pavpu.values():
        all_datasets.update(method_data.keys())
    all_datasets = sorted(all_datasets)

    if not all_datasets:
        print("⚠ No datasets found in PAvPU data")
        return

    # Create subplots: one per dataset
    n_datasets = len(all_datasets)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Color scheme for methods
    method_colors = {
        "BNDL+AUE": "#1f77b4",  # Blue
        "BNDL": "#ff7f0e",  # Orange
        "UR-ERN": "#2ca02c",  # Green
        "UCTTA": "#d62728",  # Red
    }

    for idx, dataset in enumerate(all_datasets):
        ax = axes[idx]

        for method, method_data in methods_with_pavpu.items():
            if dataset not in method_data:
                continue

            # Get raw uncertainty and accuracy samples
            samples = method_data[dataset]
            uncertainty = np.array(samples["uncertainty"])
            accuracy = np.array(samples["accuracy"])

            # Plot scatter (with alpha for overlapping points)
            color = method_colors.get(method, "#333333")
            ax.scatter(uncertainty, accuracy, alpha=0.3, s=1, color=color, label=method, rasterized=True)

        # Ideal calibration line (diagonal)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Perfect calibration")

        # Formatting
        ax.set_xlabel("Pixel Uncertainty", fontsize=10)
        ax.set_ylabel("Pixel Accuracy (0=wrong, 1=correct)", fontsize=10)
        ax.set_title(f"{dataset}", fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="best", markerscale=5)
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.05, 1.05])

    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("TRUE PAvPU: Pixel Uncertainty vs Accuracy (NO thresholds)", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()

    # Save
    plots_dir = output_path / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)

    scatter_path = plots_dir / "pavpu_scatter_comparison.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.savefig(plots_dir / "pavpu_scatter_comparison.pdf", bbox_inches="tight")
    plt.close()

    print(f"  ✓ PAvPU scatter plots saved: {scatter_path}")


def _plot_pavpu_hexbin(methods_with_pavpu: dict, output_path: Path):
    """
    Plot PAvPU hexbin density plots (better for large datasets with many overlapping points)
    Shows density of points in hexagonal bins
    """
    print("\n  Creating PAvPU hexbin density plots...")

    # Get all datasets
    all_datasets = set()
    for method_data in methods_with_pavpu.values():
        all_datasets.update(method_data.keys())
    all_datasets = sorted(all_datasets)

    if not all_datasets:
        print("⚠ No datasets found in PAvPU data")
        return

    # Create subplots: one per dataset (combine all methods in a single hexbin per dataset)
    n_datasets = len(all_datasets)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, dataset in enumerate(all_datasets):
        ax = axes[idx]

        # Combine all methods' data for this dataset
        all_uncertainty = []
        all_accuracy = []

        for _, method_data in methods_with_pavpu.items():
            if dataset not in method_data:
                continue

            samples = method_data[dataset]
            all_uncertainty.extend(samples["uncertainty"])
            all_accuracy.extend(samples["accuracy"])

        if not all_uncertainty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Create hexbin density plot
        hb = ax.hexbin(all_uncertainty, all_accuracy, gridsize=30, cmap="YlOrRd", mincnt=1, reduce_C_function=np.sum, linewidths=0.2)

        # Ideal calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=2, label="Perfect calibration")

        # Formatting
        ax.set_xlabel("Pixel Uncertainty", fontsize=10)
        ax.set_ylabel("Pixel Accuracy (0=wrong, 1=correct)", fontsize=10)
        ax.set_title(f"{dataset}\n({len(all_uncertainty)} pixels, all methods)", fontweight="bold", fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.05, 1.05])
        ax.legend(fontsize=8, loc="upper left")

        # Add colorbar
        plt.colorbar(hb, ax=ax, label="Pixel count")

    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("PAvPU Density (Hexbin): Pixel Uncertainty vs Accuracy (NO thresholds)", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()

    # Save
    plots_dir = output_path / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)

    hexbin_path = plots_dir / "pavpu_hexbin_density.png"
    plt.savefig(hexbin_path, dpi=300, bbox_inches="tight")
    plt.savefig(plots_dir / "pavpu_hexbin_density.pdf", bbox_inches="tight")
    plt.close()

    print(f"  ✓ PAvPU hexbin density plots saved: {hexbin_path}")

