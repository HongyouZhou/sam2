#!/usr/bin/env python
# Compare SAM-2 vs SAM-2+BNDL zero-shot evaluation results
# Runs both versions and generates comprehensive comparison plots

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # Use non-interactive backend

# Import both evaluation scripts
# Import dataset configs
from dataset_configs import DATASET_CONFIGS
from dataset_configs import DATASET_TO_TYPE
from dataset_configs import DATASET_TYPE_CATEGORIES
from dataset_configs import DEFAULT_DATASETS
from zero_shot_multi_dataset import run_single_dataset as run_sam2_dataset
from zero_shot_multi_dataset_sam_bndl import run_single_dataset_with_bndl as run_bndl_dataset

# Import SAM-2 builder
from sam2.build_sam import build_sam2_video_predictor


def run_comparison_evaluation(
    datasets: list[str],
    sam2_cfg: str,
    sam2_checkpoint: str,
    bndl_cfg: str,
    bndl_checkpoint: str,
    output_path: Path,
    device: str = "cuda",
    score_thresh: float = 0.0,
    thresh_grid: list[float] | None = None,
    prompt_method: str = "gt_box",
    first_frame_only: bool = False,
    max_objects: int = 20,
    video_limit: int | None = None,
    num_workers: int | None = None,
    save_vis: bool = False,
    collect_bndl_stats: bool = False,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, tuple[float, float, float]], dict[str, Any]]:
    """Run both SAM-2 and SAM-2+BNDL evaluations and return results"""

    print("=" * 80)
    print("COMPARISON EVALUATION: SAM-2 vs SAM-2+BNDL")
    print("=" * 80)

    # Create output directories
    sam2_output = output_path / "sam2_results"
    bndl_output = output_path / "bndl_results"
    sam2_output.mkdir(parents=True, exist_ok=True)
    bndl_output.mkdir(parents=True, exist_ok=True)

    # Build both predictors with identical Hydra overrides to ensure strict consistency
    hydra_overrides_extra = [
        "++model.multimask_output_in_sam=true",
        "++model.multimask_min_pt_num=1",
        "++model.multimask_max_pt_num=2",
    ]

    # Load SAM-2 predictor (original) with the same overrides
    print("\nLoading SAM-2 checkpoint...")
    sam2_predictor = build_sam2_video_predictor(
        config_file=sam2_cfg,
        ckpt_path=sam2_checkpoint,
        device=device,
        hydra_overrides_extra=hydra_overrides_extra,
    )
    print("SAM-2 loaded successfully!")

    # Load SAM-2+BNDL predictor with the same overrides
    print("\nLoading SAM-2+BNDL checkpoint...")
    bndl_predictor = build_sam2_video_predictor(
        config_file=bndl_cfg,
        ckpt_path=bndl_checkpoint,
        device=device,
        hydra_overrides_extra=hydra_overrides_extra,
    )
    print("SAM-2+BNDL loaded successfully!")

    # Run evaluations
    sam2_results = {}
    bndl_results = {}
    bndl_statistics = {}

    total_start_time = time.time()

    # Use provided grid or a single threshold from score_thresh
    thresholds = thresh_grid if (thresh_grid is not None and len(thresh_grid) > 0) else [score_thresh]

    for dataset_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {dataset_name} dataset")
        print(f"{'=' * 60}")

        # Get video subset if limit is specified
        video_subset = None
        if video_limit is not None:
            config = DATASET_CONFIGS[dataset_name]
            root = Path(config["root"])
            split = config["default_split"]
            
            # Handle both single split and multiple splits
            if isinstance(split, list):
                split = split[0]

            if config["has_split_subdir"]:
                jpeg_dir = root / split / "JPEGImages"
            else:
                jpeg_dir = root / "JPEGImages"

            if jpeg_dir.exists():
                all_videos = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
                video_subset = all_videos[:video_limit]
                print(f"Limited to {len(video_subset)} videos for {dataset_name}")

        try:
            # Per-threshold results buffers
            sam2_per_thresh: list[tuple[float, float, float, float]] = []  # (th, jf, j, f)
            bndl_per_thresh: list[tuple[float, float, float, float]] = []

            for th in thresholds:
                # Run SAM-2 evaluation
                print(f"\n--- Running SAM-2 evaluation for {dataset_name} @ thresh={th} ---")
                sam2_start = time.time()
                j_f_sam2, j_sam2, f_sam2 = run_sam2_dataset(
                    dataset_name=dataset_name,
                    predictor=sam2_predictor,
                    output_path=sam2_output,
                    score_thresh=th,
                    num_workers=num_workers,
                    video_subset=video_subset,
                    save_vis=save_vis,
                    enhanced_vis=True,
                    max_objects=max_objects,
                    prompt_method=prompt_method,
                    first_frame_only=first_frame_only,
                )
                sam2_time = time.time() - sam2_start
                sam2_per_thresh.append((th, j_f_sam2, j_sam2, f_sam2))
                print(f"SAM-2 @ {th:.2f} - J&F: {j_f_sam2:.2f}, J: {j_sam2:.2f}, F: {f_sam2:.2f} (Time: {sam2_time:.2f}s)")

                # Run SAM-2+BNDL evaluation
                print(f"--- Running SAM-2+BNDL evaluation for {dataset_name} @ thresh={th} ---")
                bndl_start = time.time()
                j_f_bndl, j_bndl, f_bndl, dataset_stats = run_bndl_dataset(
                    dataset_name=dataset_name,
                    predictor=bndl_predictor,
                    output_path=bndl_output,
                    score_thresh=th,
                    num_workers=num_workers,
                    video_subset=video_subset,
                    save_bndl_vis=save_vis,
                    prompt_method=prompt_method,
                    first_frame_only=first_frame_only,
                    max_objects=max_objects,
                    collect_statistics=True,  # Force collect statistics for comparison
                    reuse_prompts_root=None,  # Reuse prompts saved by the first run
                )
                bndl_time = time.time() - bndl_start
                bndl_per_thresh.append((th, j_f_bndl, j_bndl, f_bndl))
                if dataset_stats:
                    bndl_statistics[dataset_name] = dataset_stats
                print(f"BNDL  @ {th:.2f} - J&F: {j_f_bndl:.2f}, J: {j_bndl:.2f}, F: {f_bndl:.2f} (Time: {bndl_time:.2f}s)")

            # Print per-threshold summary for this dataset
            print("\nPer-threshold summary (SAM-2):")
            for th, jf, j, f in sam2_per_thresh:
                print(f"  th={th:.2f}: J&F={jf:.2f}, J={j:.2f}, F={f:.2f}")
            print("Per-threshold summary (BNDL):")
            for th, jf, j, f in bndl_per_thresh:
                print(f"  th={th:.2f}: J&F={jf:.2f}, J={j:.2f}, F={f:.2f}")

            # Select best-threshold result per method by J&F
            best_sam2 = max(sam2_per_thresh, key=lambda x: x[1])
            best_bndl = max(bndl_per_thresh, key=lambda x: x[1])
            sam2_results[dataset_name] = (best_sam2[1], best_sam2[2], best_sam2[3])
            bndl_results[dataset_name] = (best_bndl[1], best_bndl[2], best_bndl[3])

            print(
                f"\nBest (SAM-2) th={best_sam2[0]:.2f} -> J&F: {best_sam2[1]:.2f}, J: {best_sam2[2]:.2f}, F: {best_sam2[3]:.2f}"
            )
            print(
                f"Best (BNDL ) th={best_bndl[0]:.2f} -> J&F: {best_bndl[1]:.2f}, J: {best_bndl[2]:.2f}, F: {best_bndl[3]:.2f}"
            )

            # Improvement at respective best thresholds
            j_f_improvement = best_bndl[1] - best_sam2[1]
            j_improvement = best_bndl[2] - best_sam2[2]
            f_improvement = best_bndl[3] - best_sam2[3]
            print(f"Improvement (best vs best) - J&F: {j_f_improvement:+.2f}, J: {j_improvement:+.2f}, F: {f_improvement:+.2f}")

        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            continue

    total_time = time.time() - total_start_time
    print(f"\nTotal evaluation time: {total_time:.2f}s")

    return sam2_results, bndl_results, bndl_statistics


def create_comprehensive_comparison_plots(
    sam2_results: dict[str, tuple[float, float, float]],
    bndl_results: dict[str, tuple[float, float, float]],
    bndl_statistics: dict[str, Any],
    output_path: Path,
) -> None:
    """Create comprehensive comparison plots between SAM-2 and SAM-2+BNDL"""

    print("\nGenerating comprehensive comparison plots...")

    # Prepare data
    datasets = list(sam2_results.keys())
    if not datasets:
        print("No results to plot!")
        return

    # Extract metrics
    sam2_jf = [sam2_results[d][0] for d in datasets]
    sam2_j = [sam2_results[d][1] for d in datasets]
    sam2_f = [sam2_results[d][2] for d in datasets]

    bndl_jf = [bndl_results[d][0] for d in datasets]
    bndl_j = [bndl_results[d][1] for d in datasets]
    bndl_f = [bndl_results[d][2] for d in datasets]

    # Calculate improvements
    jf_improvements = [bndl_jf[i] - sam2_jf[i] for i in range(len(datasets))]
    j_improvements = [bndl_j[i] - sam2_j[i] for i in range(len(datasets))]
    f_improvements = [bndl_f[i] - sam2_f[i] for i in range(len(datasets))]

    # Create DataFrame for easier plotting
    df_data = []
    for i, dataset in enumerate(datasets):
        df_data.extend(
            [
                {"Dataset": dataset, "Method": "SAM-2", "Metric": "J&F", "Score": sam2_jf[i]},
                {"Dataset": dataset, "Method": "SAM-2+BNDL", "Metric": "J&F", "Score": bndl_jf[i]},
                {"Dataset": dataset, "Method": "SAM-2", "Metric": "J (IoU)", "Score": sam2_j[i]},
                {"Dataset": dataset, "Method": "SAM-2+BNDL", "Metric": "J (IoU)", "Score": bndl_j[i]},
                {"Dataset": dataset, "Method": "SAM-2", "Metric": "F (Boundary)", "Score": sam2_f[i]},
                {"Dataset": dataset, "Method": "SAM-2+BNDL", "Metric": "F (Boundary)", "Score": bndl_f[i]},
            ]
        )

    df = pd.DataFrame(df_data)

    # 计算去除 MOSE 的宏平均
    averages_excl_mose = None
    non_mose_idx = [i for i, d in enumerate(datasets) if d != "MOSE"]
    if non_mose_idx:
        avg_sam2_jf = float(np.mean([sam2_jf[i] for i in non_mose_idx]))
        avg_bndl_jf = float(np.mean([bndl_jf[i] for i in non_mose_idx]))
        avg_sam2_j = float(np.mean([sam2_j[i] for i in non_mose_idx]))
        avg_bndl_j = float(np.mean([bndl_j[i] for i in non_mose_idx]))
        avg_sam2_f = float(np.mean([sam2_f[i] for i in non_mose_idx]))
        avg_bndl_f = float(np.mean([bndl_f[i] for i in non_mose_idx]))

        averages_excl_mose = {
            "sam2": {"jf": avg_sam2_jf, "j": avg_sam2_j, "f": avg_sam2_f},
            "bndl": {"jf": avg_bndl_jf, "j": avg_bndl_j, "f": avg_bndl_f},
            "improvements": {
                "jf": avg_bndl_jf - avg_sam2_jf,
                "j": avg_bndl_j - avg_sam2_j,
                "f": avg_bndl_f - avg_sam2_f,
            },
            "datasets_count": len(non_mose_idx),
        }

    # Set up plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create comprehensive figure with better spacing
    fig = plt.figure(figsize=(30, 22))
    gs = fig.add_gridspec(4, 4, hspace=0.6, wspace=0.6)

    # Main title
    fig.suptitle("SAM-2 vs SAM-2+BNDL Zero-shot Evaluation Comparison", fontsize=20, fontweight="bold", y=0.95)

    # 1. J&F Scores Comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, sam2_jf, width, label="SAM-2", color="#FF6B6B", alpha=0.8)
    bars2 = ax1.bar(x + width / 2, bndl_jf, width, label="SAM-2+BNDL", color="#4ECDC4", alpha=0.8)

    ax1.set_title("J&F Scores Comparison", fontweight="bold", fontsize=12)
    ax1.set_ylabel("J&F Score", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.1f}", ha="center", va="bottom", fontsize=8)

    # 2. J (IoU) Scores Comparison (top center)
    ax2 = fig.add_subplot(gs[0, 1])
    bars1 = ax2.bar(x - width / 2, sam2_j, width, label="SAM-2", color="#FF6B6B", alpha=0.8)
    bars2 = ax2.bar(x + width / 2, bndl_j, width, label="SAM-2+BNDL", color="#4ECDC4", alpha=0.8)

    ax2.set_title("J (IoU) Scores Comparison", fontweight="bold", fontsize=12)
    ax2.set_ylabel("J (IoU) Score", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.1f}", ha="center", va="bottom", fontsize=8)

    # 3. F (Boundary) Scores Comparison (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    bars1 = ax3.bar(x - width / 2, sam2_f, width, label="SAM-2", color="#FF6B6B", alpha=0.8)
    bars2 = ax3.bar(x + width / 2, bndl_f, width, label="SAM-2+BNDL", color="#4ECDC4", alpha=0.8)

    ax3.set_title("F (Boundary) Scores Comparison", fontweight="bold", fontsize=12)
    ax3.set_ylabel("F (Boundary) Score", fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.1f}", ha="center", va="bottom", fontsize=8)

    # 4. Improvement Summary (top far right)
    ax4 = fig.add_subplot(gs[0, 3])
    improvement_data = [jf_improvements, j_improvements, f_improvements]
    improvement_labels = ["J&F", "J (IoU)", "F (Boundary)"]

    # Create grouped bar chart for improvements
    x_imp = np.arange(len(datasets))
    width_imp = 0.25

    for i, (data, label) in enumerate(zip(improvement_data, improvement_labels, strict=True)):
        ax4.bar(x_imp + i * width_imp, data, width_imp, label=label, alpha=0.8)

    ax4.set_title("Improvement Summary", fontweight="bold", fontsize=12)
    ax4.set_ylabel("Improvement (BNDL - SAM-2)", fontsize=10)
    ax4.set_xticks(x_imp + width_imp)
    ax4.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax4.legend(fontsize=9)
    ax4.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax4.grid(True, alpha=0.3)

    # 5. Heatmap Comparison (middle row, spanning 2 columns)
    ax5 = fig.add_subplot(gs[1, :2])
    heatmap_data = np.array([sam2_jf, bndl_jf, sam2_j, bndl_j, sam2_f, bndl_f])

    im = ax5.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax5.set_xticks(range(len(datasets)))
    ax5.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax5.set_yticks(range(6))
    ax5.set_yticklabels(["SAM-2 J&F", "BNDL J&F", "SAM-2 J", "BNDL J", "SAM-2 F", "BNDL F"], fontsize=9)
    ax5.set_title("Performance Heatmap Comparison", fontweight="bold", fontsize=12)

    # Add text annotations
    for i in range(6):
        for j in range(len(datasets)):
            ax5.text(j, i, f"{heatmap_data[i, j]:.1f}", ha="center", va="center", color="black", fontweight="bold", fontsize=8)

    plt.colorbar(im, ax=ax5, label="Score", fraction=0.046, pad=0.04)

    # 6. Detailed Metrics Breakdown (middle right, spanning 2 columns)
    ax6 = fig.add_subplot(gs[1, 2:])
    sns.barplot(data=df, x="Dataset", y="Score", hue="Method", ax=ax6)
    ax6.set_title("Detailed Metrics Breakdown", fontweight="bold", fontsize=12)
    ax6.set_ylabel("Score", fontsize=10)
    ax6.set_ylim(0, 100)
    ax6.grid(True, alpha=0.3)

    # Rotate x-axis labels and adjust legend
    for label in ax6.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
        label.set_fontsize(9)
    
    # Adjust legend
    ax6.legend(fontsize=9, loc='upper right')

    # 7. BNDL Lambda Statistics (bottom left)
    if bndl_statistics:
        ax7 = fig.add_subplot(gs[2, 0])

        # Extract BNDL statistics for plotting
        lambda_data = []

        for dataset, stats in bndl_statistics.items():
            if stats:
                # Get lambda values
                lambda_keys = [k for k in stats if "lambda_pixel" in k]
                if lambda_keys:
                    lambda_values = [stats[k] for k in lambda_keys if isinstance(stats[k], int | float)]
                    if lambda_values:
                        lambda_data.extend([(dataset, v) for v in lambda_values])

        if lambda_data:
            lambda_df = pd.DataFrame(lambda_data)
            lambda_df.columns = ["Dataset", "Lambda"]
            sns.boxplot(data=lambda_df, x="Dataset", y="Lambda", ax=ax7)
            ax7.set_title("BNDL Lambda (λ) Distribution", fontweight="bold", fontsize=10)
            ax7.set_ylabel("Lambda Value", fontsize=9)
            for label in ax7.get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment("right")
                label.set_fontsize(8)
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, "No Lambda Data", ha="center", va="center", transform=ax7.transAxes, fontsize=9)
            ax7.set_title("BNDL Lambda (λ)", fontweight="bold", fontsize=10)

    # 8. BNDL K Statistics (bottom center-left)
    if bndl_statistics:
        ax8 = fig.add_subplot(gs[2, 1])

        # Extract BNDL statistics for plotting
        k_data = []

        for dataset, stats in bndl_statistics.items():
            if stats:
                # Get k values
                k_keys = [k for k in stats if "k_pixel" in k]
                if k_keys:
                    k_values = [stats[k] for k in k_keys if isinstance(stats[k], int | float)]
                    if k_values:
                        k_data.extend([(dataset, v) for v in k_values])

        if k_data:
            k_df = pd.DataFrame(k_data)
            k_df.columns = ["Dataset", "K"]
            sns.boxplot(data=k_df, x="Dataset", y="K", ax=ax8)
            ax8.set_title("BNDL K Distribution", fontweight="bold", fontsize=10)
            ax8.set_ylabel("K Value", fontsize=9)
            for label in ax8.get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment("right")
                label.set_fontsize(8)
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, "No K Data", ha="center", va="center", transform=ax8.transAxes, fontsize=9)
            ax8.set_title("BNDL K", fontweight="bold", fontsize=10)

    # 9. Summary Statistics Table (bottom right)
    ax9 = fig.add_subplot(gs[2, 2:])
    ax9.axis("off")

    # Create summary table
    summary_data = []
    # Resolve dataset types to human-readable labels
    dataset_types = []
    for dataset in datasets:
        type_key = DATASET_TO_TYPE.get(dataset)
        type_label = DATASET_TYPE_CATEGORIES.get(type_key, "Unknown") if type_key else "Unknown"
        dataset_types.append(type_label)

    for i, dataset in enumerate(datasets):
        summary_data.append(
            [
                dataset,
                dataset_types[i],
                f"{sam2_jf[i]:.2f}",
                f"{bndl_jf[i]:.2f}",
                f"{jf_improvements[i]:+.2f}",
                f"{sam2_j[i]:.2f}",
                f"{bndl_j[i]:.2f}",
                f"{j_improvements[i]:+.2f}",
                f"{sam2_f[i]:.2f}",
                f"{bndl_f[i]:.2f}",
                f"{f_improvements[i]:+.2f}",
            ]
        )

    # 在表格中追加平均行（不含 MOSE）
    if averages_excl_mose:
        summary_data.append(
            [
                "AVG (no MOSE)",
                "—",
                f"{averages_excl_mose['sam2']['jf']:.2f}",
                f"{averages_excl_mose['bndl']['jf']:.2f}",
                f"{averages_excl_mose['improvements']['jf']:+.2f}",
                f"{averages_excl_mose['sam2']['j']:.2f}",
                f"{averages_excl_mose['bndl']['j']:.2f}",
                f"{averages_excl_mose['improvements']['j']:+.2f}",
                f"{averages_excl_mose['sam2']['f']:.2f}",
                f"{averages_excl_mose['bndl']['f']:.2f}",
                f"{averages_excl_mose['improvements']['f']:+.2f}",
            ]
        )

    table = ax9.table(
        cellText=summary_data, 
        colLabels=["Dataset", "Type", "SAM-2 J&F", "BNDL J&F", "ΔJ&F", "SAM-2 J", "BNDL J", "ΔJ", "SAM-2 F", "BNDL F", "ΔF"], 
        cellLoc="center", 
        loc="center", 
        bbox=None
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Color code improvements
    for i in range(1, len(summary_data) + 1):
        for j in [4, 7, 10]:  # Improvement columns after adding Type
            if j < len(summary_data[0]):
                try:
                    val = float(summary_data[i - 1][j])
                    if val > 0:
                        table[(i, j)].set_facecolor("#90EE90")  # Light green
                    elif val < 0:
                        table[(i, j)].set_facecolor("#FFB6C1")  # Light red
                except (ValueError, IndexError):
                    pass

    ax9.set_title("Summary Statistics", fontweight="bold", fontsize=12, pad=20)

    # Save plots
    plots_dir = output_path / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)

    plot_path = plots_dir / "sam2_vs_bndl_comprehensive_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.savefig(plots_dir / "sam2_vs_bndl_comprehensive_comparison.pdf", bbox_inches="tight")

    print(f"Comprehensive comparison plots saved to: {plot_path}")

    # Save detailed results
    results_file = plots_dir / "detailed_results.json"
    detailed_results = {
        "sam2_results": {k: {"jf": v[0], "j": v[1], "f": v[2]} for k, v in sam2_results.items()},
        "bndl_results": {k: {"jf": v[0], "j": v[1], "f": v[2]} for k, v in bndl_results.items()},
        "improvements": {k: {"jf": bndl_results[k][0] - sam2_results[k][0], "j": bndl_results[k][1] - sam2_results[k][1], "f": bndl_results[k][2] - sam2_results[k][2]} for k in datasets},
        "averages_excl_mose": averages_excl_mose,
        "bndl_statistics": bndl_statistics,
    }

    with open(results_file, "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"Detailed results saved to: {results_file}")

    # Save CSV summary
    csv_file = plots_dir / "comparison_summary.csv"
    csv_data = []
    for i, dataset in enumerate(datasets):
        csv_data.append(
            {
                "Dataset": dataset,
                "SAM2_JF": sam2_jf[i],
                "BNDL_JF": bndl_jf[i],
                "JF_Improvement": jf_improvements[i],
                "SAM2_J": sam2_j[i],
                "BNDL_J": bndl_j[i],
                "J_Improvement": j_improvements[i],
                "SAM2_F": sam2_f[i],
                "BNDL_F": bndl_f[i],
                "F_Improvement": f_improvements[i],
            }
        )

    # CSV 中追加平均行（不含 MOSE）
    if averages_excl_mose:
        csv_data.append(
            {
                "Dataset": "AVG(no MOSE)",
                "SAM2_JF": averages_excl_mose["sam2"]["jf"],
                "BNDL_JF": averages_excl_mose["bndl"]["jf"],
                "JF_Improvement": averages_excl_mose["improvements"]["jf"],
                "SAM2_J": averages_excl_mose["sam2"]["j"],
                "BNDL_J": averages_excl_mose["bndl"]["j"],
                "J_Improvement": averages_excl_mose["improvements"]["j"],
                "SAM2_F": averages_excl_mose["sam2"]["f"],
                "BNDL_F": averages_excl_mose["bndl"]["f"],
                "F_Improvement": averages_excl_mose["improvements"]["f"],
            }
        )

    df_summary = pd.DataFrame(csv_data)
    df_summary.to_csv(csv_file, index=False)
    print(f"Summary CSV saved to: {csv_file}")


def parse_args():
    p = argparse.ArgumentParser(description="Compare SAM-2 vs SAM-2+BNDL zero-shot evaluation")

    # Dataset selection
    p.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=list(DATASET_CONFIGS.keys()),
        help="Datasets to evaluate (default: all)",
    )

    # SAM-2 configuration
    p.add_argument(
        "--sam2_cfg",
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM-2 config file",
    )
    p.add_argument(
        "--sam2_checkpoint",
        default="/home/hongyou/dev/ada_samp/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
        help="SAM-2 checkpoint path",
    )

    # SAM-2+BNDL configuration
    p.add_argument(
        "--bndl_cfg",
        default="configs/sam2.1/sam2.1_hiera_b+_bndl.yaml",
        help="SAM-2+BNDL config file",
    )
    p.add_argument(
        "--bndl_checkpoint",
        default="/home/hongyou/dev/ada_samp/logs/sam2/sam2_bndl_007_02_adco/checkpoints/checkpoint_50.pt",
        help="SAM-2+BNDL checkpoint path",
    )

    # Evaluation parameters
    p.add_argument("--device", default="cuda", help="Device to use")
    p.add_argument("--score_thresh", type=float, default=0.0, help="Mask logit threshold (used if --thresh_grid is not set)")
    p.add_argument(
        "--thresh_grid",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of mask logit thresholds to sweep; best J&F per method will be reported",
    )
    p.add_argument(
        "--prompt_method",
        type=str,
        default="gt_box",
        choices=["gt_box", "three_clicks"],
        help="Prompting strategy: gt_box (default) or three_clicks",
    )
    p.add_argument("--num_workers", type=int, default=None, help="Number of evaluation processes")
    p.add_argument("--output_path", default="./outputs/comparison_sam2_vs_bndl_007_02_adco_no_reuse", help="Root output directory")
    p.add_argument("--first_frame_only", action="store_true", help="Evaluate only the first frame per video")

    # Subset options
    p.add_argument("--video_limit", type=int, default=None, help="Limit number of videos per dataset")
    p.add_argument("--max_objects", type=int, default=256, help="Maximum number of objects per video")

    # Visualization options
    p.add_argument("--save_vis", action="store_true", default=True, help="Save visualizations")
    p.add_argument("--collect_bndl_stats", action="store_true", default=True, help="Collect BNDL statistics")

    return p.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Starting comparison evaluation...")
    print(f"Output directory: {output_path}")
    print(f"Datasets: {args.datasets}")
    print(f"SAM-2 config: {args.sam2_cfg}")
    print(f"SAM-2 checkpoint: {args.sam2_checkpoint}")
    print(f"BNDL config: {args.bndl_cfg}")
    print(f"BNDL checkpoint: {args.bndl_checkpoint}")

    # Run comparison evaluation
    sam2_results, bndl_results, bndl_statistics = run_comparison_evaluation(
        datasets=args.datasets,
        sam2_cfg=args.sam2_cfg,
        sam2_checkpoint=args.sam2_checkpoint,
        bndl_cfg=args.bndl_cfg,
        bndl_checkpoint=args.bndl_checkpoint,
        output_path=output_path,
        device=args.device,
        score_thresh=args.score_thresh,
        thresh_grid=args.thresh_grid,
        prompt_method=args.prompt_method,
        first_frame_only=args.first_frame_only,
        max_objects=args.max_objects,
        video_limit=args.video_limit,
        num_workers=args.num_workers,
        save_vis=args.save_vis,
        collect_bndl_stats=args.collect_bndl_stats,
    )

    # Create comparison plots
    create_comprehensive_comparison_plots(sam2_results, bndl_results, bndl_statistics, output_path)

    # Print final summary
    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Dataset':<12} {'SAM-2 J&F':<10} {'BNDL J&F':<10} {'Improvement':<12}")
    print("-" * 80)

    for dataset in sam2_results:
        sam2_jf = sam2_results[dataset][0]
        bndl_jf = bndl_results[dataset][0]
        improvement = bndl_jf - sam2_jf
        print(f"{dataset:<12} {sam2_jf:<10.2f} {bndl_jf:<10.2f} {improvement:+12.2f}")

    valid_for_avg = [k for k in sam2_results if k != "MOSE"]
    print("-" * 80)
    if valid_for_avg:
        sam2_mean_jf = float(np.mean([sam2_results[k][0] for k in valid_for_avg]))
        bndl_mean_jf = float(np.mean([bndl_results[k][0] for k in valid_for_avg]))
        print(f"{'AVG (no MOSE)':<12} {sam2_mean_jf:<10.2f} {bndl_mean_jf:<10.2f} {bndl_mean_jf - sam2_mean_jf:+12.2f}")
    else:
        print("AVG (no MOSE): N/A (no eligible datasets)")

    print(f"\nAll comparison results saved to: {output_path}")
    print("Check the comparison_plots/ directory for detailed visualizations!")


if __name__ == "__main__":
    main()
