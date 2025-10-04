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
from dataset_configs import (
    DATASET_CONFIGS,
    DATASET_TO_TYPE,
    DATASET_TYPE_CATEGORIES,
    DEFAULT_DATASETS,
)
from sam2.build_sam import build_sam2_video_predictor
from zero_shot_multi_dataset import run_single_dataset as run_sam2_dataset
from zero_shot_multi_dataset_sam_bndl import (
    run_single_dataset_with_bndl as run_bndl_dataset,
)
from zero_shot_multi_dataset_uctta import (
    run_single_dataset_with_uctta as run_uctta_dataset,
)

matplotlib.use("Agg")  # Use non-interactive backend


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
    uctta_steps: int = 2,
    uctta_lr: float = 3e-4,
    run_sam: bool = True,
    run_uctta: bool = False,
    run_bndl: bool = True,
    # click protocol passed to baseline SAM run so prompts are saved
    click_protocol: str = "3click",
    min_click_dist: float = 12.0,
    seed: int = 0,
    # Full UCTTA parameters
    uctta_enable_bn: bool = True,
    uctta_fisher_reg: bool = True,
    uctta_fisher_alpha: float = 2000.0,
    uctta_entropy_th: float = 0.4,
    uctta_selection_p: float = 0.1,
) -> tuple[
    dict[str, tuple[float, float, float]],  # sam2_results
    dict[str, tuple[float, float, float]],  # bndl_results
    dict[str, Any],                          # bndl_statistics
    dict[str, tuple[float, float, float]] | None,  # uctta_results
    dict[str, dict[str, Any]],              # ua_data_per_dataset
    dict[str, Any],                          # uctta_statistics
]:
    """Run both SAM-2 and SAM-2+BNDL evaluations and return results"""

    print("=" * 80)
    print("COMPARISON EVALUATION: SAM-2 vs SAM-2+BNDL")
    print("=" * 80)

    # Create output directories
    sam2_output = output_path / "sam2_results"
    bndl_output = output_path / "bndl_results"
    uctta_output = output_path / "sam2_uctta_results"
    sam2_output.mkdir(parents=True, exist_ok=True)
    bndl_output.mkdir(parents=True, exist_ok=True)
    if run_uctta:
        uctta_output.mkdir(parents=True, exist_ok=True)

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
    uctta_results: dict[str, tuple[float, float, float]] | dict[str, list[tuple[float, float, float, float]]] | None = {} if run_uctta else None
    bndl_statistics = {}
    uctta_statistics = {}  # Store UCTTA statistics per dataset
    ua_data_per_dataset = {}  # Store UA data for each dataset

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
                # Run SAM-2 evaluation (optional)
                if run_sam:
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
                        click_protocol=click_protocol,
                        min_click_dist=float(min_click_dist),
                        seed=int(seed),
                    )
                    sam2_time = time.time() - sam2_start
                    sam2_per_thresh.append((th, j_f_sam2, j_sam2, f_sam2))
                    print(f"SAM-2 @ {th:.2f} - J&F: {j_f_sam2:.2f}, J: {j_sam2:.2f}, F: {f_sam2:.2f} (Time: {sam2_time:.2f}s)")

                # Run SAM-2 + UCTTA evaluation (optional)
                if run_uctta:
                    print(f"--- Running SAM-2+UCTTA evaluation for {dataset_name} @ thresh={th} ---")
                    uctta_start = time.time()
                    j_f_uctta, j_uctta, f_uctta, uctta_stats = run_uctta_dataset(
                        dataset_name=dataset_name,
                        predictor=sam2_predictor,
                        output_path=uctta_output,
                        score_thresh=th,
                        num_workers=num_workers,
                        video_subset=video_subset,
                        prompt_method=prompt_method,
                        first_frame_only=first_frame_only,
                        max_objects=max_objects,
                        uctta_steps=uctta_steps,
                        uctta_lr=uctta_lr,
                        reuse_prompts_root=sam2_output if run_sam else None,
                        click_protocol=click_protocol,
                        min_click_dist=min_click_dist,
                        seed=seed,
                        # Full UCTTA parameters
                        enable_bn_adapt=uctta_enable_bn,
                        use_fisher_reg=uctta_fisher_reg,
                        fisher_alpha=uctta_fisher_alpha,
                        entropy_threshold=uctta_entropy_th,
                        selection_p=uctta_selection_p,
                    )
                    uctta_time = time.time() - uctta_start
                    print(f"SAM-2+UCTTA @ {th:.2f} - J&F: {j_f_uctta:.2f}, J: {j_uctta:.2f}, F: {f_uctta:.2f} (Time: {uctta_time:.2f}s)")
                    
                    # Store UCTTA statistics (only need one threshold's stats since uncertainty is independent of threshold)
                    if uctta_stats and dataset_name not in uctta_statistics:
                        uctta_statistics[dataset_name] = uctta_stats
                    
                    # store best per threshold locally; final best selection below
                    assert uctta_results is not None
                    if dataset_name not in uctta_results:
                        uctta_results[dataset_name] = []  # type: ignore[assignment]
                    (uctta_results[dataset_name]).append((th, j_f_uctta, j_uctta, f_uctta))  # type: ignore[index]

                # Run SAM-2+BNDL evaluation
                if run_bndl:
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
                        reuse_prompts_root=sam2_output,  # Reuse prompts saved by the SAM run
                        click_protocol=click_protocol,
                        min_click_dist=min_click_dist,
                        seed=seed,
                    )
                    bndl_time = time.time() - bndl_start
                    bndl_per_thresh.append((th, j_f_bndl, j_bndl, f_bndl))
                    if dataset_stats:
                        bndl_statistics[dataset_name] = dataset_stats
                    print(f"BNDL  @ {th:.2f} - J&F: {j_f_bndl:.2f}, J: {j_bndl:.2f}, F: {f_bndl:.2f} (Time: {bndl_time:.2f}s)")

            # Print per-threshold summary for this dataset
            if run_sam:
                print("\nPer-threshold summary (SAM-2):")
                for th, jf, j, f in sam2_per_thresh:
                    print(f"  th={th:.2f}: J&F={jf:.2f}, J={j:.2f}, F={f:.2f}")
            if run_uctta and isinstance(uctta_results, dict) and dataset_name in uctta_results:
                print("Per-threshold summary (SAM-2+UCTTA):")
                for th, jf, j, f in uctta_results[dataset_name]:  # type: ignore[index]
                    print(f"  th={th:.2f}: J&F={jf:.2f}, J={j:.2f}, F={f:.2f}")
            if run_bndl:
                print("Per-threshold summary (BNDL):")
                for th, jf, j, f in bndl_per_thresh:
                    print(f"  th={th:.2f}: J&F={jf:.2f}, J={j:.2f}, F={f:.2f}")

            # Select best-threshold result per method by J&F
            if run_sam and sam2_per_thresh:
                best_sam2 = max(sam2_per_thresh, key=lambda x: x[1])
                sam2_results[dataset_name] = (best_sam2[1], best_sam2[2], best_sam2[3])
            if run_bndl and bndl_per_thresh:
                best_bndl = max(bndl_per_thresh, key=lambda x: x[1])
                bndl_results[dataset_name] = (best_bndl[1], best_bndl[2], best_bndl[3])
            if run_uctta and isinstance(uctta_results, dict) and dataset_name in uctta_results and len(uctta_results[dataset_name]) > 0:  # type: ignore[index]
                best_uctta = max((uctta_results[dataset_name]), key=lambda x: x[1])  # type: ignore[index]
                # Replace list with best tuple
                uctta_results[dataset_name] = (best_uctta[1], best_uctta[2], best_uctta[3])  # type: ignore[index]

            if run_sam and sam2_per_thresh:
                print(f"\nBest (SAM-2) th={best_sam2[0]:.2f} -> J&F: {best_sam2[1]:.2f}, J: {best_sam2[2]:.2f}, F: {best_sam2[3]:.2f}")
            if run_bndl and bndl_per_thresh:
                print(f"Best (BNDL ) th={best_bndl[0]:.2f} -> J&F: {best_bndl[1]:.2f}, J: {best_bndl[2]:.2f}, F: {best_bndl[3]:.2f}")
            if run_uctta and isinstance(uctta_results, dict) and dataset_name in uctta_results and isinstance(uctta_results[dataset_name], tuple):
                print(f"Best (UCTTA) -> J&F: {uctta_results[dataset_name][0]:.2f}, J: {uctta_results[dataset_name][1]:.2f}, F: {uctta_results[dataset_name][2]:.2f}")

            # Improvement at respective best thresholds
            if run_sam and run_bndl and sam2_per_thresh and bndl_per_thresh:
                j_f_improvement = best_bndl[1] - best_sam2[1]
                j_improvement = best_bndl[2] - best_sam2[2]
                f_improvement = best_bndl[3] - best_sam2[3]
                print(f"Improvement (best vs best) - J&F: {j_f_improvement:+.2f}, J: {j_improvement:+.2f}, F: {f_improvement:+.2f}")

        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            continue

    total_time = time.time() - total_start_time
    print(f"\nTotal evaluation time: {total_time:.2f}s")

    return sam2_results, bndl_results, bndl_statistics, (uctta_results if isinstance(uctta_results, dict) else None), ua_data_per_dataset, uctta_statistics


def save_detailed_results(
    output_path: Path,
    sam2_results: dict[str, tuple[float, float, float]] | None = None,
    bndl_results: dict[str, tuple[float, float, float]] | None = None,
    bndl_statistics: dict[str, Any] | None = None,
    uctta_results: dict[str, tuple[float, float, float]] | None = None,
    uctta_statistics: dict[str, Any] | None = None,
    ua_data: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Save detailed results to JSON file
    
    Args:
        output_path: Root output directory
        sam2_results: SAM-2 results {dataset: (J&F, J, F)}
        bndl_results: BNDL results {dataset: (J&F, J, F)}
        bndl_statistics: BNDL statistics per dataset
        uctta_results: UCTTA results {dataset: (J&F, J, F)}
        uctta_statistics: UCTTA statistics per dataset
        ua_data: UA analysis data per dataset
        
    Returns:
        Path to saved JSON file
    """
    plots_dir = output_path / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)
    results_file = plots_dir / "detailed_results.json"
    
    # Build detailed results dict
    # Always include all keys for consistent JSON structure
    detailed_results = {
        "sam2_results": {k: {"jf": v[0], "j": v[1], "f": v[2]} for k, v in sam2_results.items()} if sam2_results else {},
        "uctta_results": {k: {"jf": v[0], "j": v[1], "f": v[2]} for k, v in uctta_results.items()} if uctta_results else {},
        "bndl_results": {k: {"jf": v[0], "j": v[1], "f": v[2]} for k, v in bndl_results.items()} if bndl_results else {},
        "bndl_statistics": bndl_statistics if bndl_statistics else {},
        "uctta_statistics": uctta_statistics if uctta_statistics else {},
        "ua_data": ua_data if ua_data else {},
    }
    
    # Calculate improvements if both SAM and BNDL results exist
    if sam2_results and bndl_results:
        common_datasets = [d for d in sam2_results if d in bndl_results]
        detailed_results["improvements"] = {
            k: {
                "jf": bndl_results[k][0] - sam2_results[k][0],
                "j": bndl_results[k][1] - sam2_results[k][1],
                "f": bndl_results[k][2] - sam2_results[k][2],
            }
            for k in common_datasets
        }
        
        # Calculate averages excluding MOSE
        non_mose_datasets = [d for d in common_datasets if d != "MOSE"]
        if non_mose_datasets:
            import numpy as np
            avg_sam2_jf = float(np.mean([sam2_results[d][0] for d in non_mose_datasets]))
            avg_bndl_jf = float(np.mean([bndl_results[d][0] for d in non_mose_datasets]))
            avg_sam2_j = float(np.mean([sam2_results[d][1] for d in non_mose_datasets]))
            avg_bndl_j = float(np.mean([bndl_results[d][1] for d in non_mose_datasets]))
            avg_sam2_f = float(np.mean([sam2_results[d][2] for d in non_mose_datasets]))
            avg_bndl_f = float(np.mean([bndl_results[d][2] for d in non_mose_datasets]))
            
            detailed_results["averages_excl_mose"] = {
                "sam2": {"jf": avg_sam2_jf, "j": avg_sam2_j, "f": avg_sam2_f},
                "bndl": {"jf": avg_bndl_jf, "j": avg_bndl_j, "f": avg_bndl_f},
                "improvements": {
                    "jf": avg_bndl_jf - avg_sam2_jf,
                    "j": avg_bndl_j - avg_sam2_j,
                    "f": avg_bndl_f - avg_sam2_f,
                },
                "datasets_count": len(non_mose_datasets),
            }
    
    # Save to file
    with open(results_file, "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"Detailed results saved to: {results_file}")
    return results_file


def create_comprehensive_comparison_plots(
    sam2_results: dict[str, tuple[float, float, float]],
    bndl_results: dict[str, tuple[float, float, float]],
    bndl_statistics: dict[str, Any],
    output_path: Path,
    uctta_results: dict[str, tuple[float, float, float]] | None = None,
    uctta_statistics: dict[str, Any] | None = None,
) -> None:
    """Create comprehensive comparison plots between SAM-2 and SAM-2+BNDL"""

    print("\nGenerating comprehensive comparison plots...")

    # Prepare data - only use datasets that exist in BOTH results
    datasets = [d for d in sam2_results if d in bndl_results]
    if not datasets:
        print("No common results to plot!")
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
        df_data.extend([
            {"Dataset": dataset, "Method": "SAM-2", "Metric": "J&F", "Score": sam2_jf[i]},
            {"Dataset": dataset, "Method": "SAM-2+BNDL", "Metric": "J&F", "Score": bndl_jf[i]},
            {"Dataset": dataset, "Method": "SAM-2", "Metric": "J (IoU)", "Score": sam2_j[i]},
            {"Dataset": dataset, "Method": "SAM-2+BNDL", "Metric": "J (IoU)", "Score": bndl_j[i]},
            {"Dataset": dataset, "Method": "SAM-2", "Metric": "F (Boundary)", "Score": sam2_f[i]},
            {"Dataset": dataset, "Method": "SAM-2+BNDL", "Metric": "F (Boundary)", "Score": bndl_f[i]},
        ])

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
    ax6.legend(fontsize=9, loc="upper right")

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
        summary_data.append([
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
        ])

    # 在表格中追加平均行（不含 MOSE）
    if averages_excl_mose:
        summary_data.append([
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
        ])

    table = ax9.table(
        cellText=summary_data, colLabels=["Dataset", "Type", "SAM-2 J&F", "BNDL J&F", "ΔJ&F", "SAM-2 J", "BNDL J", "ΔJ", "SAM-2 F", "BNDL F", "ΔF"], cellLoc="center", loc="center", bbox=None
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

    # Save CSV summary
    csv_file = plots_dir / "comparison_summary.csv"
    csv_data = []
    for i, dataset in enumerate(datasets):
        csv_data.append({
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
        })

    # CSV 中追加平均行（不含 MOSE）
    if averages_excl_mose:
        csv_data.append({
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
        })

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
    source_domain: str = "MOSE",
) -> None:
    """Create UA (Uncertainty-Accuracy) shift analysis plots
    
    Args:
        bndl_statistics: Dictionary containing BNDL statistics per dataset
        sam2_results: SAM-2 performance results  
        bndl_results: BNDL performance results
        output_path: Output directory for plots
        uctta_statistics: Optional UCTTA statistics per dataset
        uctta_results: Optional UCTTA performance results
        source_domain: Source domain for shift comparison (default: MOSE)
    """
    print("\nGenerating UA shift analysis plots...")
    
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
                if stats and 'pixel_uncertainty_mean' in stats:
                    uctta_uncertainty_data[dataset_name] = float(stats['pixel_uncertainty_mean'])
    
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
    
    # Determine figure layout based on whether UCTTA data is available
    if has_uctta:
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4)
        title = "UA Consistency Analysis: BNDL vs UCTTA"
    else:
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.4)
        title = "UA Consistency Analysis: BNDL across Domains"
    
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    
    # 1. BNDL: Uncertainty vs J&F Score scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    bndl_datasets_list = list(bndl_performance_data.keys())
    bndl_x_unc = [bndl_uncertainty_data[d] for d in bndl_datasets_list]
    bndl_y_jf = [bndl_performance_data[d]["jf"] for d in bndl_datasets_list]
    
    # Color source domain differently
    bndl_colors = ['red' if d == source_domain else 'blue' for d in bndl_datasets_list]
    ax1.scatter(bndl_x_unc, bndl_y_jf, c=bndl_colors, s=100, alpha=0.6, edgecolors='black', label='BNDL')
    
    # Add dataset labels
    for i, dataset in enumerate(bndl_datasets_list):
        ax1.annotate(dataset, (bndl_x_unc[i], bndl_y_jf[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Fit trend line
    if len(bndl_x_unc) > 1:
        z = np.polyfit(bndl_x_unc, bndl_y_jf, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(bndl_x_unc), max(bndl_x_unc), 100)
        ax1.plot(x_line, p(x_line), "b--", alpha=0.5, linewidth=2, 
                label=f'BNDL Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        
        # Calculate correlation
        corr = np.corrcoef(bndl_x_unc, bndl_y_jf)[0, 1]
        ax1.text(0.05, 0.95, f'BNDL Correlation: {corr:.3f}', 
                transform=ax1.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
                verticalalignment='top')
    
    ax1.set_xlabel('Pixel Uncertainty (mean)', fontsize=11)
    ax1.set_ylabel('J&F Score', fontsize=11)
    ax1.set_title('BNDL: Uncertainty vs Performance', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Uncertainty comparison: BNDL (and optionally UCTTA)
    ax2 = fig.add_subplot(gs[0, 1])
    
    if has_uctta:
        # Side-by-side comparison of BNDL and UCTTA uncertainty
        common_datasets = sorted([d for d in bndl_uncertainty_data if d in uctta_uncertainty_data])
        x_pos = np.arange(len(common_datasets))
        width = 0.35
        
        bndl_unc_vals = [bndl_uncertainty_data[d] for d in common_datasets]
        uctta_unc_vals = [uctta_uncertainty_data[d] for d in common_datasets]
        
        ax2.barh(x_pos - width / 2, bndl_unc_vals, width, label='BNDL', color='#4ECDC4', alpha=0.8)
        ax2.barh(x_pos + width / 2, uctta_unc_vals, width, label='UCTTA', color='#FF6B6B', alpha=0.8)
        
        ax2.set_yticks(x_pos)
        ax2.set_yticklabels(common_datasets, fontsize=9)
        ax2.set_xlabel('Mean Pixel Uncertainty', fontsize=11)
        ax2.set_title('Uncertainty Comparison: BNDL vs UCTTA', fontweight='bold', fontsize=12)
        ax2.legend()
    else:
        # Just BNDL uncertainty ranking
        sorted_datasets = sorted(bndl_uncertainty_data.keys(), key=lambda x: bndl_uncertainty_data[x])
        sorted_unc = [bndl_uncertainty_data[d] for d in sorted_datasets]
        bar_colors = ['red' if d == source_domain else 'steelblue' for d in sorted_datasets]
        
        bars = ax2.barh(range(len(sorted_datasets)), sorted_unc, color=bar_colors, alpha=0.7)
        ax2.set_yticks(range(len(sorted_datasets)))
        ax2.set_yticklabels(sorted_datasets, fontsize=9)
        ax2.set_xlabel('Mean Pixel Uncertainty (BNDL)', fontsize=11)
        ax2.set_title('BNDL Uncertainty Ranking by Dataset', fontweight='bold', fontsize=12)
        
        # Add value labels
        for i, (_bar, val) in enumerate(zip(bars, sorted_unc, strict=True)):
            ax2.text(val, i, f' {val:.4f}', va='center', fontsize=8)
    
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. UA Shift from source domain (BNDL)
    if source_domain in bndl_uncertainty_data and source_domain in bndl_performance_data:
        ax3 = fig.add_subplot(gs[0, 2])
        
        source_unc = bndl_uncertainty_data[source_domain]
        source_jf = bndl_performance_data[source_domain]["jf"]
        
        target_datasets = [d for d in bndl_datasets_list if d != source_domain]
        ua_shifts_unc = [bndl_uncertainty_data[d] - source_unc for d in target_datasets]
        ua_shifts_perf = [bndl_performance_data[d]["jf"] - source_jf for d in target_datasets]
        
        ax3.scatter(ua_shifts_unc, ua_shifts_perf, s=100, alpha=0.6, 
                   c='green', edgecolors='black')
        
        for i, dataset in enumerate(target_datasets):
            ax3.annotate(dataset, (ua_shifts_unc[i], ua_shifts_perf[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax3.set_xlabel(f'Δ Uncertainty (vs {source_domain})', fontsize=11)
        ax3.set_ylabel(f'Δ J&F Score (vs {source_domain})', fontsize=11)
        ax3.set_title(f'UA Shift from Source Domain ({source_domain})', 
                     fontweight='bold', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Add quadrant labels
        ax3.text(0.95, 0.95, 'Higher Unc\nBetter Perf', transform=ax3.transAxes,
                ha='right', va='top', fontsize=8, style='italic', alpha=0.5)
        ax3.text(0.05, 0.05, 'Lower Unc\nWorse Perf', transform=ax3.transAxes,
                ha='left', va='bottom', fontsize=8, style='italic', alpha=0.5)
    
    # 4. Improvement vs Uncertainty (BNDL vs SAM-2)
    ax4 = fig.add_subplot(gs[1, 0])
    common_datasets = [d for d in bndl_datasets_list if d in sam2_results and d in bndl_results]
    improvements = [bndl_results[d][0] - sam2_results[d][0] for d in common_datasets]
    uncertainties = [bndl_uncertainty_data[d] for d in common_datasets]
    
    colors_imp = ['red' if d == source_domain else 'purple' for d in common_datasets]
    ax4.scatter(uncertainties, improvements, s=100, alpha=0.6, 
               c=colors_imp, edgecolors='black')
    
    for i, dataset in enumerate(common_datasets):
        ax4.annotate(dataset, (uncertainties[i], improvements[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    if len(uncertainties) > 1:
        z_imp = np.polyfit(uncertainties, improvements, 1)
        p_imp = np.poly1d(z_imp)
        x_line_imp = np.linspace(min(uncertainties), max(uncertainties), 100)
        ax4.plot(x_line_imp, p_imp(x_line_imp), "g--", alpha=0.5, linewidth=2)
        
        corr_imp = np.corrcoef(uncertainties, improvements)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {corr_imp:.3f}', 
                transform=ax4.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
                verticalalignment='top')
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax4.set_xlabel('Pixel Uncertainty', fontsize=11)
    ax4.set_ylabel('BNDL Improvement (ΔJ&F)', fontsize=11)
    ax4.set_title('BNDL Improvement vs Uncertainty', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 5. Uncertainty distribution across datasets (BNDL)
    ax5 = fig.add_subplot(gs[1, 1])
    dataset_names_short = [d[:8] for d in bndl_datasets_list]  # Shorten names for readability
    x_pos = np.arange(len(bndl_datasets_list))
    
    bars5 = ax5.bar(x_pos, bndl_x_unc, color=bndl_colors, alpha=0.7, edgecolor='black')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(dataset_names_short, rotation=45, ha='right', fontsize=9)
    ax5.set_ylabel('Mean Pixel Uncertainty (BNDL)', fontsize=11)
    ax5.set_title('BNDL Uncertainty Distribution Across Datasets', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars5:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 6. Summary statistics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create summary table
    table_data = []
    for dataset in bndl_datasets_list:
        unc = bndl_uncertainty_data.get(dataset, 0)
        jf = bndl_performance_data.get(dataset, {}).get("jf", 0)
        if dataset in sam2_results and dataset in bndl_results:
            improvement = bndl_results[dataset][0] - sam2_results[dataset][0]
        else:
            improvement = 0
        
        table_data.append([
            dataset,
            f"{unc:.4f}",
            f"{jf:.2f}",
            f"{improvement:+.2f}"
        ])
    
    table = ax6.table(
        cellText=table_data,
        colLabels=["Dataset", "Uncertainty", "BNDL J&F", "Δ vs SAM-2"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
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
    
    ax6.set_title('UA Summary Statistics', fontweight='bold', fontsize=12, pad=20)
    
    # Save plots
    plots_dir = output_path / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)
    
    ua_plot_path = plots_dir / "ua_shift_analysis.png"
    plt.savefig(ua_plot_path, dpi=300, bbox_inches="tight")
    plt.savefig(plots_dir / "ua_shift_analysis.pdf", bbox_inches="tight")
    plt.close()
    
    print(f"UA shift analysis plots saved to: {ua_plot_path}")


def load_results_from_detailed_json(
    json_path: Path,
) -> tuple[
    dict[str, tuple[float, float, float]],  # sam2_results
    dict[str, tuple[float, float, float]],  # bndl_results
    dict[str, Any],                          # bndl_statistics
    dict[str, tuple[float, float, float]] | None,  # uctta_results
    dict[str, dict[str, Any]],              # ua_data_per_dataset
    dict[str, Any],                          # uctta_statistics
]:
    """Load previously saved results to avoid re-running experiments.

    Expects a file produced by this script at: output_path/comparison_plots/detailed_results.json
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Detailed results JSON not found: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    sam2_map = {}
    bndl_map = {}
    stats_map = data.get("bndl_statistics", {})
    uctta_map = None
    ua_data_map = data.get("ua_data", {})
    uctta_stats_map = data.get("uctta_statistics", {})

    # Convert nested dicts to expected tuple format
    for k, v in data.get("sam2_results", {}).items():
        # v: {"jf": x, "j": y, "f": z}
        jf = float(v.get("jf", 0.0))
        j = float(v.get("j", 0.0))
        f = float(v.get("f", 0.0))
        sam2_map[k] = (jf, j, f)
    for k, v in data.get("bndl_results", {}).items():
        jf = float(v.get("jf", 0.0))
        j = float(v.get("j", 0.0))
        f = float(v.get("f", 0.0))
        bndl_map[k] = (jf, j, f)

    # UCTTA (optional)
    if "uctta_results" in data:
        uctta_map = {}
        for k, v in data.get("uctta_results", {}).items():
            jf = float(v.get("jf", 0.0))
            j = float(v.get("j", 0.0))
            f = float(v.get("f", 0.0))
            uctta_map[k] = (jf, j, f)

    return sam2_map, bndl_map, stats_map, uctta_map, ua_data_map, uctta_stats_map


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
        default="/home/hongyou/dev/ada_samp/logs/sam2/sam2_bndl_011_02/checkpoints/checkpoint.pt",
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
    p.add_argument("--output_path", default="./outputs/comparison_sam2_vs_bndl_011_01", help="Root output directory")
    p.add_argument("--first_frame_only", action="store_true", help="Evaluate only the first frame per video")

    # Subset options
    p.add_argument("--video_limit", type=int, default=None, help="Limit number of videos per dataset")
    p.add_argument("--max_objects", type=int, default=256, help="Maximum number of objects per video")

    # Visualization options
    p.add_argument("--save_vis", action="store_true", default=True, help="Save visualizations")
    p.add_argument("--collect_bndl_stats", action="store_true", default=True, help="Collect BNDL statistics")
    # Click protocol options
    p.add_argument("--click_protocol", type=str, default="3click", choices=["1click", "3click", "5click"], help="Interaction protocol for first frame")
    p.add_argument("--min_click_dist", type=float, default=12.0, help="Minimum distance between clicks for 5-click protocol")
    p.add_argument("--seed", type=int, default=0, help="Random seed for 'random' point initialization")

    # Cached results options
    p.add_argument("--load_detailed_json", type=str, default=None, help="Path to a previously saved detailed_results.json to render plots and summaries without re-running")
    # Method toggles
    p.add_argument("--run_sam", action="store_true", default=False, help="Run baseline SAM-2 branch")
    p.add_argument("--run_uctta", action="store_true", default=False, help="Run SAM-2 + UCTTA branch")
    p.add_argument("--run_bndl", action="store_true", default=False, help="Run SAM-2 + BNDL branch")
    # UCTTA options
    p.add_argument("--uctta_steps", type=int, default=2, help="UCTTA adaptation steps per frame/batch")
    p.add_argument("--uctta_lr", type=float, default=3e-4, help="UCTTA learning rate")
    p.add_argument("--uctta_enable_bn", action="store_true", default=True, help="Enable BN/LN adaptation (full UCTTA)")
    p.add_argument("--uctta_fisher_reg", action="store_true", default=True, help="Use Fisher regularization")
    p.add_argument("--uctta_fisher_alpha", type=float, default=2000.0, help="Fisher regularization strength")
    p.add_argument("--uctta_entropy_th", type=float, default=0.4, help="Entropy threshold for sample selection")
    p.add_argument("--uctta_selection_p", type=float, default=0.1, help="Fraction of samples to select")

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

    # Optionally load previous results to avoid re-running
    if args.load_detailed_json is not None:
        detailed_path = Path(args.load_detailed_json)
        print(f"Loading cached results from: {detailed_path}")
        sam2_results, bndl_results, bndl_statistics, uctta_results, ua_data, uctta_statistics = load_results_from_detailed_json(detailed_path)
    else:
        # Run comparison evaluation
        sam2_results, bndl_results, bndl_statistics, uctta_results, ua_data, uctta_statistics = run_comparison_evaluation(
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
            uctta_steps=args.uctta_steps,
            uctta_lr=args.uctta_lr,
            run_sam=args.run_sam,
            run_uctta=args.run_uctta,
            run_bndl=args.run_bndl,
            click_protocol=args.click_protocol,
            min_click_dist=args.min_click_dist,
            seed=args.seed,
            # Full UCTTA parameters
            uctta_enable_bn=args.uctta_enable_bn,
            uctta_fisher_reg=args.uctta_fisher_reg,
            uctta_fisher_alpha=args.uctta_fisher_alpha,
            uctta_entropy_th=args.uctta_entropy_th,
            uctta_selection_p=args.uctta_selection_p,
        )

    # Always save detailed results JSON (needed for parallel_compare.py)
    save_detailed_results(
        output_path=output_path,
        sam2_results=sam2_results,
        bndl_results=bndl_results,
        bndl_statistics=bndl_statistics,
        uctta_results=uctta_results,
        uctta_statistics=uctta_statistics,
        ua_data=ua_data,
    )
    
    # Create comparison plots (optional, commented out for parallel execution)
    # if sam2_results and bndl_results:
    #     create_comprehensive_comparison_plots(
    #         sam2_results, bndl_results, bndl_statistics, 
    #         output_path, uctta_results, uctta_statistics
    #     )
    
    # Create UA shift analysis plots (optional, commented out for parallel execution)
    # if bndl_statistics and sam2_results and bndl_results:
    #     create_ua_shift_analysis_plots(
    #         bndl_statistics, 
    #         sam2_results, 
    #         bndl_results, 
    #         output_path,
    #         uctta_statistics=uctta_statistics,
    #         uctta_results=uctta_results
    #     )


if __name__ == "__main__":
    main()
