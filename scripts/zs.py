#!/usr/bin/env python
# Compare SAM-2 vs BNDL vs BNDL_AUE zero-shot evaluation results
# Runs multiple versions and generates comprehensive comparison plots

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from zero_shot_multi_dataset_ur_ern import (
    run_single_dataset_with_ur_ern as run_ur_ern_dataset,
)

matplotlib.use("Agg")  # Use non-interactive backend


def _load_eval_json(file_path: Path) -> dict | None:
    """线程安全的 JSON 文件加载（用于评估结果文件）"""
    if not file_path.exists():
        return None
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return None


def run_comparison_evaluation(
    datasets: list[str],
    sam2_cfg: str,
    sam2_checkpoint: str,
    bndl_aue_cfg: str,
    bndl_aue_checkpoint: str,
    bndl_cfg: str | None,
    bndl_checkpoint: str | None,
    output_path: Path,
    ur_ern_cfg: str | None = None,
    ur_ern_checkpoint: str | None = None,
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
    run_bndl_aue: bool = True,
    run_bndl: bool = True,
    run_ur_ern: bool = False,
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
    # Downsampling parameters
    downsample_max_samples: int = 100000,
) -> tuple[
    dict[str, tuple[float, float, float]],  # sam2_results
    dict[str, tuple[float, float, float]],  # bndl_aue_results
    dict[str, Any],                          # bndl_aue_statistics
    dict[str, tuple[float, float, float]],  # bndl_results
    dict[str, Any],                          # bndl_statistics
    dict[str, tuple[float, float, float]] | None,  # uctta_results
    dict[str, tuple[float, float, float]] | None,  # ur_ern_results
    dict[str, dict[str, Any]],              # ua_data_per_dataset
    dict[str, Any],                          # uctta_statistics
    dict[str, Any],                          # ur_ern_statistics
]:
    """Run SAM-2, BNDL, and BNDL_AUE evaluations and return results"""

    print("=" * 80)
    print("COMPARISON EVALUATION: SAM-2 vs BNDL vs BNDL_AUE")
    print("=" * 80)

    # Create output directories
    sam2_output = output_path / "sam2_results"
    bndl_aue_output = output_path / "bndl_aue_results"
    bndl_output = output_path / "bndl_results"
    uctta_output = output_path / "sam2_uctta_results"
    ur_ern_output = output_path / "sam2_ur_ern_results"
    if run_sam:
        sam2_output.mkdir(parents=True, exist_ok=True)
    if run_bndl_aue:
        bndl_aue_output.mkdir(parents=True, exist_ok=True)
    if run_bndl:
        bndl_output.mkdir(parents=True, exist_ok=True)
    if run_uctta:
        uctta_output.mkdir(parents=True, exist_ok=True)
    if run_ur_ern:
        ur_ern_output.mkdir(parents=True, exist_ok=True)

    # Build both predictors with identical Hydra overrides to ensure strict consistency
    from shared_evaluation_utils import build_predictor_with_overrides

    # Load SAM-2 predictor (original) with the same overrides (optional)
    sam2_predictor = None
    if run_sam or run_uctta:  # UCTTA needs SAM-2 predictor
        print("\nLoading SAM-2 checkpoint...")
        sam2_predictor = build_predictor_with_overrides(
            cfg_file=sam2_cfg,
            ckpt=sam2_checkpoint,
            device=device,
        )
        print("SAM-2 loaded successfully!")

    # Load BNDL predictor with the same overrides (optional)
    bndl_predictor = None
    if run_bndl:
        if bndl_cfg is None or bndl_checkpoint is None:
            raise ValueError("BNDL requires both bndl_cfg and bndl_checkpoint to be specified")
        print("\nLoading BNDL checkpoint...")
        bndl_predictor = build_predictor_with_overrides(
            cfg_file=bndl_cfg,
            ckpt=bndl_checkpoint,
            device=device,
        )
        print("BNDL loaded successfully!")

    # Load BNDL_AUE predictor with the same overrides (optional)
    bndl_aue_predictor = None
    if run_bndl_aue:
        print("\nLoading BNDL_AUE checkpoint...")
        bndl_aue_predictor = build_predictor_with_overrides(
            cfg_file=bndl_aue_cfg,
            ckpt=bndl_aue_checkpoint,
            device=device,
        )
        print("BNDL_AUE loaded successfully!")

    # Load SAM-2+UR-ERN predictor with the same overrides (if needed)
    ur_ern_predictor = None
    if run_ur_ern:
        if ur_ern_cfg is None or ur_ern_checkpoint is None:
            raise ValueError("UR-ERN requires both ur_ern_cfg and ur_ern_checkpoint to be specified")
        print("\nLoading SAM-2+UR-ERN checkpoint...")
        ur_ern_predictor = build_predictor_with_overrides(
            cfg_file=ur_ern_cfg,
            ckpt=ur_ern_checkpoint,
            device=device,
        )
        print("SAM-2+UR-ERN loaded successfully!")

    # Run evaluations
    sam2_results = {}
    bndl_aue_results = {}
    bndl_results = {}
    uctta_results: dict[str, tuple[float, float, float]] | dict[str, list[tuple[float, float, float, float]]] | None = {} if run_uctta else None
    ur_ern_results: dict[str, tuple[float, float, float]] | None = {} if run_ur_ern else None
    bndl_aue_statistics = {}
    bndl_statistics = {}
    uctta_statistics = {}  # Store UCTTA statistics per dataset
    ur_ern_statistics = {}  # Store UR-ERN statistics per dataset
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

        # Per-threshold results buffers
        sam2_per_thresh: list[tuple[float, float, float, float]] = []  # (th, jf, j, f)
        bndl_aue_per_thresh: list[tuple[float, float, float, float]] = []
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
                    downsample_max_samples=downsample_max_samples,
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

            # Run BNDL evaluation
            if run_bndl:
                print(f"--- Running BNDL evaluation for {dataset_name} @ thresh={th} ---")
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
                    collect_statistics=collect_bndl_stats,  # Use parameter instead of forcing True
                    reuse_prompts_root=sam2_output if run_sam else None,  # Only reuse prompts if SAM ran
                    click_protocol=click_protocol,
                    min_click_dist=min_click_dist,
                    seed=seed,
                    downsample_max_samples=downsample_max_samples,
                )
                bndl_time = time.time() - bndl_start
                bndl_per_thresh.append((th, j_f_bndl, j_bndl, f_bndl))
                if dataset_stats:
                    bndl_statistics[dataset_name] = dataset_stats
                print(f"BNDL @ {th:.2f} - J&F: {j_f_bndl:.2f}, J: {j_bndl:.2f}, F: {f_bndl:.2f} (Time: {bndl_time:.2f}s)")

            # Run BNDL_AUE evaluation
            if run_bndl_aue:
                print(f"--- Running BNDL_AUE evaluation for {dataset_name} @ thresh={th} ---")
                bndl_aue_start = time.time()
                j_f_bndl_aue, j_bndl_aue, f_bndl_aue, dataset_stats_aue = run_bndl_dataset(
                    dataset_name=dataset_name,
                    predictor=bndl_aue_predictor,
                    output_path=bndl_aue_output,
                    score_thresh=th,
                    num_workers=num_workers,
                    video_subset=video_subset,
                    save_bndl_vis=save_vis,
                    prompt_method=prompt_method,
                    first_frame_only=first_frame_only,
                    max_objects=max_objects,
                    collect_statistics=collect_bndl_stats,  # Use parameter instead of forcing True
                    reuse_prompts_root=sam2_output if run_sam else None,  # Only reuse prompts if SAM ran
                    click_protocol=click_protocol,
                    min_click_dist=min_click_dist,
                    seed=seed,
                    downsample_max_samples=downsample_max_samples,
                )
                bndl_aue_time = time.time() - bndl_aue_start
                bndl_aue_per_thresh.append((th, j_f_bndl_aue, j_bndl_aue, f_bndl_aue))
                if dataset_stats_aue:
                    bndl_aue_statistics[dataset_name] = dataset_stats_aue
                print(f"BNDL_AUE @ {th:.2f} - J&F: {j_f_bndl_aue:.2f}, J: {j_bndl_aue:.2f}, F: {f_bndl_aue:.2f} (Time: {bndl_aue_time:.2f}s)")

            # Run SAM-2+UR-ERN evaluation
            if run_ur_ern and ur_ern_predictor is not None:
                print(f"--- Running SAM-2+UR-ERN evaluation for {dataset_name} @ thresh={th} ---")
                ur_ern_start = time.time()
                j_f_ur_ern, j_ur_ern, f_ur_ern, dataset_stats = run_ur_ern_dataset(
                    dataset_name=dataset_name,
                    predictor=ur_ern_predictor,
                    output_path=ur_ern_output,
                    score_thresh=th,
                    num_workers=num_workers,
                    video_subset=video_subset,
                    save_ur_ern_vis=save_vis,
                    prompt_method=prompt_method,
                    first_frame_only=first_frame_only,
                    max_objects=max_objects,
                    collect_statistics=True,  # Force collect statistics for comparison
                    reuse_prompts_root=sam2_output if run_sam else None,  # Only reuse prompts if SAM ran
                    click_protocol=click_protocol,
                    min_click_dist=min_click_dist,
                    seed=seed,
                    downsample_max_samples=downsample_max_samples,
                )
                ur_ern_time = time.time() - ur_ern_start
                if dataset_name not in ur_ern_results:
                    ur_ern_results[dataset_name] = []
                ur_ern_results[dataset_name].append((th, j_f_ur_ern, j_ur_ern, f_ur_ern))
                if dataset_stats:
                    ur_ern_statistics[dataset_name] = dataset_stats
                print(f"UR-ERN @ {th:.2f} - J&F: {j_f_ur_ern:.2f}, J: {j_ur_ern:.2f}, F: {f_ur_ern:.2f} (Time: {ur_ern_time:.2f}s)")

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
        if run_bndl_aue:
            print("Per-threshold summary (BNDL_AUE):")
            for th, jf, j, f in bndl_aue_per_thresh:
                print(f"  th={th:.2f}: J&F={jf:.2f}, J={j:.2f}, F={f:.2f}")
        if run_ur_ern and isinstance(ur_ern_results, dict) and dataset_name in ur_ern_results:
            print("Per-threshold summary (UR-ERN):")
            for th, jf, j, f in ur_ern_results[dataset_name]:
                print(f"  th={th:.2f}: J&F={jf:.2f}, J={j:.2f}, F={f:.2f}")

        # Select best-threshold result per method by J&F
        if run_sam and sam2_per_thresh:
            best_sam2 = max(sam2_per_thresh, key=lambda x: x[1])
            sam2_results[dataset_name] = (best_sam2[1], best_sam2[2], best_sam2[3])
        if run_bndl_aue and bndl_aue_per_thresh:
            best_bndl_aue = max(bndl_aue_per_thresh, key=lambda x: x[1])
            bndl_aue_results[dataset_name] = (best_bndl_aue[1], best_bndl_aue[2], best_bndl_aue[3])
        if run_bndl and bndl_per_thresh:
            best_bndl = max(bndl_per_thresh, key=lambda x: x[1])
            bndl_results[dataset_name] = (best_bndl[1], best_bndl[2], best_bndl[3])
        if run_uctta and isinstance(uctta_results, dict) and dataset_name in uctta_results and len(uctta_results[dataset_name]) > 0:  # type: ignore[index]
            best_uctta = max((uctta_results[dataset_name]), key=lambda x: x[1])  # type: ignore[index]
            # Replace list with best tuple
            uctta_results[dataset_name] = (best_uctta[1], best_uctta[2], best_uctta[3])  # type: ignore[index]
        if run_ur_ern and isinstance(ur_ern_results, dict) and dataset_name in ur_ern_results and len(ur_ern_results[dataset_name]) > 0:
            best_ur_ern = max(ur_ern_results[dataset_name], key=lambda x: x[1])
            # Replace list with best tuple
            ur_ern_results[dataset_name] = (best_ur_ern[1], best_ur_ern[2], best_ur_ern[3])

        if run_sam and sam2_per_thresh:
            print(f"\nBest (SAM-2) th={best_sam2[0]:.2f} -> J&F: {best_sam2[1]:.2f}, J: {best_sam2[2]:.2f}, F: {best_sam2[3]:.2f}")
        if run_bndl and bndl_per_thresh:
            print(f"Best (BNDL) th={best_bndl[0]:.2f} -> J&F: {best_bndl[1]:.2f}, J: {best_bndl[2]:.2f}, F: {best_bndl[3]:.2f}")
        if run_bndl_aue and bndl_aue_per_thresh:
            print(f"Best (BNDL_AUE) th={best_bndl_aue[0]:.2f} -> J&F: {best_bndl_aue[1]:.2f}, J: {best_bndl_aue[2]:.2f}, F: {best_bndl_aue[3]:.2f}")
        if run_uctta and isinstance(uctta_results, dict) and dataset_name in uctta_results and isinstance(uctta_results[dataset_name], tuple):
            print(f"Best (UCTTA) -> J&F: {uctta_results[dataset_name][0]:.2f}, J: {uctta_results[dataset_name][1]:.2f}, F: {uctta_results[dataset_name][2]:.2f}")
        if run_ur_ern and isinstance(ur_ern_results, dict) and dataset_name in ur_ern_results and isinstance(ur_ern_results[dataset_name], tuple):
            print(f"Best (UR-ERN) -> J&F: {ur_ern_results[dataset_name][0]:.2f}, J: {ur_ern_results[dataset_name][1]:.2f}, F: {ur_ern_results[dataset_name][2]:.2f}")

        # Improvement at respective best thresholds
        if run_sam and run_bndl and sam2_per_thresh and bndl_per_thresh:
            j_f_improvement = best_bndl[1] - best_sam2[1]
            j_improvement = best_bndl[2] - best_sam2[2]
            f_improvement = best_bndl[3] - best_sam2[3]
            print(f"Improvement (best vs best) - J&F: {j_f_improvement:+.2f}, J: {j_improvement:+.2f}, F: {f_improvement:+.2f}")
        
        # Cleanup GPU memory after successful dataset completion
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"✓ GPU memory cleared after completing {dataset_name}")

    total_time = time.time() - total_start_time
    print(f"\nTotal evaluation time: {total_time:.2f}s")

    return (sam2_results, bndl_aue_results, bndl_aue_statistics, bndl_results, bndl_statistics,
            (uctta_results if isinstance(uctta_results, dict) else None),
            (ur_ern_results if isinstance(ur_ern_results, dict) else None),
            ua_data_per_dataset, uctta_statistics, ur_ern_statistics)


def save_detailed_results(
    output_path: Path,
    sam2_results: dict[str, tuple[float, float, float]] | None = None,
    bndl_aue_results: dict[str, tuple[float, float, float]] | None = None,
    bndl_aue_statistics: dict[str, Any] | None = None,
    bndl_results: dict[str, tuple[float, float, float]] | None = None,
    bndl_statistics: dict[str, Any] | None = None,
    uctta_results: dict[str, tuple[float, float, float]] | None = None,
    ur_ern_results: dict[str, tuple[float, float, float]] | None = None,
    uctta_statistics: dict[str, Any] | None = None,
    ur_ern_statistics: dict[str, Any] | None = None,
    ua_data: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Save detailed results to JSON file

    Args:
        output_path: Root output directory
        sam2_results: SAM-2 results {dataset: (J&F, J, F)}
        bndl_results: BNDL results {dataset: (J&F, J, F)}
        bndl_statistics: BNDL statistics per dataset
        bndl_aue_results: BNDL_AUE results {dataset: (J&F, J, F)}
        bndl_aue_statistics: BNDL_AUE statistics per dataset
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
        "ur_ern_results": {k: {"jf": v[0], "j": v[1], "f": v[2]} for k, v in ur_ern_results.items()} if ur_ern_results else {},
        "bndl_aue_results": {k: {"jf": v[0], "j": v[1], "f": v[2]} for k, v in bndl_aue_results.items()} if bndl_aue_results else {},
        "bndl_aue_statistics": bndl_aue_statistics if bndl_aue_statistics else {},
        "bndl_results": {k: {"jf": v[0], "j": v[1], "f": v[2]} for k, v in bndl_results.items()} if bndl_results else {},
        "bndl_statistics": bndl_statistics if bndl_statistics else {},
        "uctta_statistics": uctta_statistics if uctta_statistics else {},
        "ur_ern_statistics": ur_ern_statistics if ur_ern_statistics else {},
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
        
        # Calculate averages excluding MOSE (both train and val)
        non_mose_datasets = [d for d in common_datasets if not d.startswith("MOSE")]
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
    aue_version: str | None = None,
) -> None:
    """Create comprehensive comparison plots between SAM-2 and BNDL
    
    Args:
        aue_version: AUE版本标识，用于生成带版本后缀的文件夹名
    """

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
            {"Dataset": dataset, "Method": "BNDL_AUE", "Metric": "J&F", "Score": bndl_jf[i]},
            {"Dataset": dataset, "Method": "SAM-2", "Metric": "J (IoU)", "Score": sam2_j[i]},
            {"Dataset": dataset, "Method": "BNDL_AUE", "Metric": "J (IoU)", "Score": bndl_j[i]},
            {"Dataset": dataset, "Method": "SAM-2", "Metric": "F (Boundary)", "Score": sam2_f[i]},
            {"Dataset": dataset, "Method": "BNDL_AUE", "Metric": "F (Boundary)", "Score": bndl_f[i]},
        ])

    df = pd.DataFrame(df_data)

    # 计算去除 MOSE 的宏平均 (排除 MOSE_train 和 MOSE_val)
    averages_excl_mose = None
    non_mose_idx = [i for i, d in enumerate(datasets) if not d.startswith("MOSE")]
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

    # Compact layout: only Improvement Summary + Summary Statistics
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 1, hspace=0.5, wspace=0.4)

    # Main title
    fig.suptitle("SAM-2 vs BNDL_AUE Zero-shot Evaluation (Compact)", fontsize=18, fontweight="bold", y=0.95)

    # Improvement Summary (top) - 横坐标为improvement，纵坐标为数据集
    ax4 = fig.add_subplot(gs[0, 0])
    improvement_data = [jf_improvements, j_improvements, f_improvements]
    improvement_labels = ["J&F", "J (IoU)", "F (Boundary)"]
    y_imp = np.arange(len(datasets))
    height_imp = 0.25
    
    # 使用水平条形图（barh）
    for i, (data, label) in enumerate(zip(improvement_data, improvement_labels, strict=True)):
        ax4.barh(y_imp + i * height_imp, data, height_imp, label=label, alpha=0.85)
    
    ax4.set_title("Improvement Summary", fontweight="bold", fontsize=14)
    ax4.set_xlabel("Improvement (BNDL_AUE - SAM-2)", fontsize=11)
    ax4.set_ylabel("Dataset", fontsize=11)
    ax4.set_yticks(y_imp + height_imp)
    ax4.set_yticklabels(datasets, fontsize=10)
    ax4.legend(fontsize=10)
    ax4.axvline(x=0, color="black", linestyle="-", alpha=0.3)
    ax4.grid(True, alpha=0.3, axis='x')
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
    
    # 表数据：三行（SAM-2, BNDL_AUE, Improvement）
    summary_data = []
    
    # 第一行：SAM-2 J&F
    row_sam2 = ["SAM-2 J&F"]
    for i in range(len(datasets)):
        row_sam2.append(f"{sam2_jf[i]:.2f}")
    if averages_excl_mose:
        row_sam2.append(f"{averages_excl_mose['sam2']['jf']:.2f}")
    summary_data.append(row_sam2)
    
    # 第二行：BNDL_AUE J&F
    row_bndl = ["BNDL_AUE J&F"]
    for i in range(len(datasets)):
        row_bndl.append(f"{bndl_jf[i]:.2f}")
    if averages_excl_mose:
        row_bndl.append(f"{averages_excl_mose['bndl']['jf']:.2f}")
    summary_data.append(row_bndl)
    
    # 第三行：ΔJ&F (Improvement)
    row_delta = ["ΔJ&F"]
    for i in range(len(datasets)):
        row_delta.append(f"{jf_improvements[i]:+.2f}")
    if averages_excl_mose:
        row_delta.append(f"{averages_excl_mose['improvements']['jf']:+.2f}")
    summary_data.append(row_delta)

    table = ax9.table(
        cellText=summary_data, 
        colLabels=col_labels, 
        cellLoc="center", 
        loc="center", 
        bbox=None
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    # Color code improvements in the ΔJ&F row
    for j in range(1, len(col_labels)):  # Skip first column (Method label)
        try:
            val = float(summary_data[2][j])  # Row 2 is ΔJ&F
            if val > 0:
                table[(3, j)].set_facecolor("#90EE90")  # Light green (row 3 because header is row 0)
            elif val < 0:
                table[(3, j)].set_facecolor("#FFB6C1")  # Light red
        except (ValueError, IndexError):
            pass

    ax9.set_title("Summary Statistics (ΔJ&F Focus)", fontweight="bold", fontsize=12, pad=20)

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

    # CSV 中追加平均行（不含 MOSE_train/val）
    if averages_excl_mose:
        csv_data.append({
            "Dataset": "AVG(excl MOSE)",
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
        'SAM-2': '#95A5A6',   # Gray - baseline
        'UCTTA': '#FF6B6B',   # Red/Pink - adaptation method
        'BNDL_AUE': '#4ECDC4',  # Teal/Cyan - our method (BNDL+AUE)
        'BNDL': '#2E86AB',    # Dark blue - BNDL pure
        'UR-ERN': '#95E1D3',  # Light teal/mint - alternative method
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
                if stats and 'pixel_uncertainty_mean' in stats:
                    uctta_uncertainty_data[dataset_name] = float(stats['pixel_uncertainty_mean'])
    
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
    bndl_colors = ['red' if d == source_domain else 'blue' for d in bndl_datasets_list]
    ax1.scatter(bndl_x_unc, bndl_y_jf, c=bndl_colors, s=100, alpha=0.6, edgecolors='black', label='BNDL_AUE')
    
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
                label=f'BNDL_AUE Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        
        # Calculate correlation
        corr = np.corrcoef(bndl_x_unc, bndl_y_jf)[0, 1]
        ax1.text(0.05, 0.95, f'BNDL_AUE Correlation: {corr:.3f}', 
                transform=ax1.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
                verticalalignment='top')
    
    ax1.set_xlabel('Pixel Uncertainty (mean)', fontsize=11)
    ax1.set_ylabel('J&F Score', fontsize=11)
    ax1.set_title('BNDL_AUE: Uncertainty vs Performance', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # NOTE: ax2 will be created after loading all PCC/uncertainty data to include UR-ERN
    
    # 3. Improvement vs Uncertainty (BNDL vs SAM-2) - MOVED to gs[0, 2]
    ax3 = fig.add_subplot(gs[0, 2])
    common_datasets = [d for d in bndl_datasets_list if d in sam2_results and d in bndl_results]
    improvements = [bndl_results[d][0] - sam2_results[d][0] for d in common_datasets]
    uncertainties = [bndl_uncertainty_data[d] for d in common_datasets]
    
    colors_imp = ['red' if d == source_domain else 'purple' for d in common_datasets]
    ax3.scatter(uncertainties, improvements, s=100, alpha=0.6, 
               c=colors_imp, edgecolors='black')
    
    for i, dataset in enumerate(common_datasets):
        ax3.annotate(dataset, (uncertainties[i], improvements[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    if len(uncertainties) > 1:
        z_imp = np.polyfit(uncertainties, improvements, 1)
        p_imp = np.poly1d(z_imp)
        x_line_imp = np.linspace(min(uncertainties), max(uncertainties), 100)
        ax3.plot(x_line_imp, p_imp(x_line_imp), "g--", alpha=0.5, linewidth=2)
        
        corr_imp = np.corrcoef(uncertainties, improvements)[0, 1]
        ax3.text(0.05, 0.95, f'Correlation: {corr_imp:.3f}', 
                transform=ax3.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
                verticalalignment='top')
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax3.set_xlabel('Pixel Uncertainty', fontsize=11)
    ax3.set_ylabel('BNDL_AUE Improvement (ΔJ&F)', fontsize=11)
    ax3.set_title('BNDL_AUE Improvement vs Uncertainty', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Uncertainty distribution across datasets (BNDL) - MOVED to gs[0, 3]
    ax4 = fig.add_subplot(gs[0, 3])
    dataset_names_short = [d[:8] for d in bndl_datasets_list]  # Shorten names for readability
    x_pos = np.arange(len(bndl_datasets_list))
    
    bars4 = ax4.bar(x_pos, bndl_x_unc, color=bndl_colors, alpha=0.7, edgecolor='black')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(dataset_names_short, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Mean Pixel Uncertainty (BNDL_AUE)', fontsize=11)
    ax4.set_title('BNDL_AUE Uncertainty Distribution Across Datasets', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
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
        files_to_load.append(('sam2', dataset_name, path))

    # BNDL files  
    bndl_root = bndl_root_override if bndl_root_override is not None else (output_path / "bndl_results")
    for dataset_name in bndl_datasets_list:
        path = bndl_root / f"{dataset_name.lower()}_bndl_eval" / f"{dataset_name.lower()}_zeroshot_results.json"
        files_to_load.append(('bndl', dataset_name, path))

    # UCTTA files
    uctta_root = uctta_root_override if uctta_root_override is not None else (output_path / "sam2_uctta_results")
    for dataset_name in bndl_datasets_list:
        path = uctta_root / f"{dataset_name.lower()}_uctta_eval" / f"{dataset_name.lower()}_uctta_results.json"
        files_to_load.append(('uctta', dataset_name, path))

    # UR-ERN files
    ur_ern_root = ur_ern_root_override if ur_ern_root_override is not None else (output_path / "sam2_ur_ern_results")
    for dataset_name in bndl_datasets_list:
        path = ur_ern_root / f"{dataset_name.lower()}_ur_ern_eval" / f"{dataset_name.lower()}_ur_ern_results.json"
        files_to_load.append(('ur_ern', dataset_name, path))

    # BNDL pure files (optional)
    if bndl_pure_statistics or bndl_pure_root_override:
        bndl_pure_root = bndl_pure_root_override if bndl_pure_root_override is not None else (output_path / "bndl_results")
        for dataset_name in bndl_datasets_list:
            path = bndl_pure_root / f"{dataset_name.lower()}_bndl_eval" / f"{dataset_name.lower()}_zeroshot_results.json"
            files_to_load.append(('bndl_pure', dataset_name, path))

    # 并行加载所有文件
    loaded_data = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_info = {
            executor.submit(_load_eval_json, path): (method, dataset, path)
            for method, dataset, path in files_to_load
        }
        
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
    for dataset, data in loaded_data.get('sam2', {}).items():
        if isinstance(data, dict) and "NLL" in data:
            nll_info = data["NLL"]
            if isinstance(nll_info, dict) and "metric_mean" in nll_info:
                sam2_nll_data[dataset] = float(nll_info["metric_mean"])

    accuracy_pcc_map: dict[str, float] = {}
    bndl_nll_data: dict[str, float] = {}
    for dataset, data in loaded_data.get('bndl', {}).items():
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
    for dataset, data in loaded_data.get('uctta', {}).items():
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
    for dataset, data in loaded_data.get('ur_ern', {}).items():
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
    for dataset, data in loaded_data.get('bndl_pure', {}).items():
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
            ax2.barh(x_pos + offset, sam2_nll_vals, width, label='SAM-2', color=METHOD_COLORS['SAM-2'], alpha=0.8)
            offset += width
        
        # Plot UCTTA if available
        if has_uctta and uctta_nll_data:
            uctta_nll_vals = [uctta_nll_data.get(d, np.nan) for d in common_datasets]
            ax2.barh(x_pos + offset, uctta_nll_vals, width, label='UCTTA', color=METHOD_COLORS['UCTTA'], alpha=0.8)
            offset += width
        
        # Plot BNDL_AUE
        if bndl_nll_data:
            bndl_nll_vals = [bndl_nll_data.get(d, np.nan) for d in common_datasets]
            ax2.barh(x_pos + offset, bndl_nll_vals, width, label='BNDL_AUE', 
                    color=METHOD_COLORS['BNDL_AUE'], alpha=0.8)
            offset += width
        
        # Plot BNDL (pure) if available
        if has_bndl_pure and bndl_pure_nll_data:
            bndl_pure_nll_vals = [bndl_pure_nll_data.get(d, np.nan) for d in common_datasets]
            ax2.barh(x_pos + offset, bndl_pure_nll_vals, width, label='BNDL', color=METHOD_COLORS['BNDL'], alpha=0.8)
            offset += width
        
        # Plot UR-ERN if available
        if has_ur_ern and ur_ern_nll_data:
            ur_ern_nll_vals = [ur_ern_nll_data.get(d, np.nan) for d in common_datasets]
            ax2.barh(x_pos + offset, ur_ern_nll_vals, width, label='UR-ERN', color=METHOD_COLORS['UR-ERN'], alpha=0.8)
            offset += width
        
        # Center the y-tick labels
        center_offset = (num_methods - 1) * width / 2
        ax2.set_yticks(x_pos + center_offset)
        ax2.set_yticklabels(common_datasets, fontsize=9)
        ax2.set_xlabel('Mean NLL (Negative Log-Likelihood)', fontsize=11)
        
        # Update title based on available methods
        title_methods = []
        if sam2_nll_data:
            title_methods.append('SAM-2')
        if has_uctta and uctta_nll_data:
            title_methods.append('UCTTA')
        if bndl_nll_data:
            title_methods.append('BNDL_AUE')
        if has_bndl_pure and bndl_pure_nll_data:
            title_methods.append('BNDL')
        if has_ur_ern and ur_ern_nll_data:
            title_methods.append('UR-ERN')
        ax2.set_title(f'NLL: {" vs ".join(title_methods)}', fontweight='bold', fontsize=12)
        
        ax2.legend(fontsize=9)
        ax2.invert_xaxis()  # Lower NLL is better, so invert for better visualization
    else:
        # Just BNDL NLL ranking
        sorted_datasets = sorted(bndl_nll_data.keys(), key=lambda x: bndl_nll_data[x])
        sorted_nll = [bndl_nll_data[d] for d in sorted_datasets]
        bar_colors = ['red' if d == source_domain else 'steelblue' for d in sorted_datasets]
        
        bars = ax2.barh(range(len(sorted_datasets)), sorted_nll, color=bar_colors, alpha=0.7)
        ax2.set_yticks(range(len(sorted_datasets)))
        ax2.set_yticklabels(sorted_datasets, fontsize=9)
        ax2.set_xlabel('Mean NLL (lower is better)', fontsize=11)
        ax2.set_title('BNDL_AUE NLL Ranking by Dataset', fontweight='bold', fontsize=12)
        ax2.invert_xaxis()  # Lower NLL is better
        
        # Add value labels
        for i, (_bar, val) in enumerate(zip(bars, sorted_nll, strict=True)):
            ax2.text(val, i, f' {val:.4f}', va='center', fontsize=8)
    
    ax2.grid(True, alpha=0.3, axis='x')
    
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
        
        bars = ax5.barh(y_pos, values, color=['red' if v < 0 else 'green' for v in values], alpha=0.7, edgecolor='black')
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(labels, fontsize=9)
        ax5.set_xlabel(f"ΔPCC relative to {source_domain}", fontsize=11)
        ax5.set_title("BNDL_AUE: UA Shift (ΔPCC)", fontweight='bold', fontsize=12)
        ax5.axvline(x=0.0, color='black', linestyle='--', alpha=0.4)
        ax5.grid(True, alpha=0.3, axis='x')
        
        for bar in bars:
            width = bar.get_width()
            ax5.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                     f"{width:+.3f}", va='center', ha='left' if width >= 0 else 'right', fontsize=8)
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
        bars_bp = ax5b.barh(y_pos_bp, values_bp, color=['red' if v < 0 else 'green' for v in values_bp], alpha=0.7, edgecolor='black')
        ax5b.set_yticks(y_pos_bp)
        ax5b.set_yticklabels(labels_bp, fontsize=9)
        ax5b.set_xlabel(f"ΔPCC relative to {source_domain}", fontsize=11)
        ax5b.set_title("BNDL: UA Shift (ΔPCC)", fontweight='bold', fontsize=12)
        ax5b.axvline(x=0.0, color='black', linestyle='--', alpha=0.4)
        ax5b.grid(True, alpha=0.3, axis='x')
        for bar in bars_bp:
            width = bar.get_width()
            ax5b.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                      f"{width:+.3f}", va='center', ha='left' if width >= 0 else 'right', fontsize=8)
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
        bars_u = ax6.barh(y_pos_u, values_u, color=['red' if v < 0 else 'green' for v in values_u], alpha=0.7, edgecolor='black')
        ax6.set_yticks(y_pos_u)
        ax6.set_yticklabels(labels_u, fontsize=9)
        ax6.set_xlabel(f"ΔPCC relative to {source_domain}", fontsize=11)
        ax6.set_title("UCTTA: UA Shift (ΔPCC)", fontweight='bold', fontsize=12)
        ax6.axvline(x=0.0, color='black', linestyle='--', alpha=0.4)
        ax6.grid(True, alpha=0.3, axis='x')
        for bar in bars_u:
            width = bar.get_width()
            ax6.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                      f"{width:+.3f}", va='center', ha='left' if width >= 0 else 'right', fontsize=8)
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
        bars_r = ax7.barh(y_pos_r, values_r, color=['red' if v < 0 else 'green' for v in values_r], alpha=0.7, edgecolor='black')
        ax7.set_yticks(y_pos_r)
        ax7.set_yticklabels(labels_r, fontsize=9)
        ax7.set_xlabel(f"ΔPCC relative to {source_domain}", fontsize=11)
        ax7.set_title("UR-ERN: UA Shift (ΔPCC)", fontweight='bold', fontsize=12)
        ax7.axvline(x=0.0, color='black', linestyle='--', alpha=0.4)
        ax7.grid(True, alpha=0.3, axis='x')
        for bar in bars_r:
            width = bar.get_width()
            ax7.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                      f"{width:+.3f}", va='center', ha='left' if width >= 0 else 'right', fontsize=8)
        current_col += 1
    
    # 8. J&F Performance Comparison - MOVED to Row 2 to avoid overlap with ΔPCC plots
    ax8 = fig.add_subplot(gs[2, 2:])
    
    # Get all datasets that have results from at least one method
    all_datasets_for_jf = sorted(set(sam2_results.keys()) | set(bndl_results.keys()) | 
                                  (set(uctta_results.keys()) if uctta_results else set()) |
                                  (set(bndl_pure_results.keys()) if bndl_pure_results else set()) |
                                  (set(ur_ern_results.keys()) if ur_ern_results else set()))
    
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
    ax8.barh(x_pos + offset, sam_jf_vals, width, label='SAM-2', color=METHOD_COLORS['SAM-2'], alpha=0.8)
    offset += width
    
    # Plot UCTTA if available
    if has_uctta and uctta_results:
        uctta_jf_vals = [uctta_results[d][0] if d in uctta_results else np.nan for d in all_datasets_for_jf]
        ax8.barh(x_pos + offset, uctta_jf_vals, width, label='UCTTA', color=METHOD_COLORS['UCTTA'], alpha=0.8)
        offset += width
    
    # Plot BNDL_AUE
    ax8.barh(x_pos + offset, bndl_jf_vals, width, label='BNDL_AUE', 
            color=METHOD_COLORS['BNDL_AUE'], alpha=0.8)
    offset += width
    
    # Plot BNDL (pure) if available
    if has_bndl_pure and bndl_pure_results:
        bndl_pure_jf_vals = [bndl_pure_results[d][0] if d in bndl_pure_results else np.nan for d in all_datasets_for_jf]
        ax8.barh(x_pos + offset, bndl_pure_jf_vals, width, label='BNDL', color=METHOD_COLORS['BNDL'], alpha=0.8)
        offset += width
    
    # Plot UR-ERN if available  
    if has_ur_ern and ur_ern_results:
        ur_ern_jf_vals = [ur_ern_results[d][0] if d in ur_ern_results else np.nan for d in all_datasets_for_jf]
        ax8.barh(x_pos + offset, ur_ern_jf_vals, width, label='UR-ERN', color=METHOD_COLORS['UR-ERN'], alpha=0.8)
        offset += width
    
    # Center the y-tick labels
    center_offset = (num_methods - 1) * width / 2
    ax8.set_yticks(x_pos + center_offset)
    ax8.set_yticklabels(all_datasets_for_jf, fontsize=9)
    ax8.set_xlabel('J&F Score', fontsize=11)
    
    # Update title based on available methods
    title_methods = ['SAM-2']
    if has_uctta and uctta_results:
        title_methods.append('UCTTA')
    title_methods.append('BNDL_AUE')
    if has_bndl_pure and bndl_pure_results:
        title_methods.append('BNDL')
    if has_ur_ern and ur_ern_results:
        title_methods.append('UR-ERN')
    ax8.set_title(f'J&F Performance: {" vs ".join(title_methods)}', fontweight='bold', fontsize=12)
    
    ax8.legend(fontsize=9, loc='lower right')
    ax8.grid(True, alpha=0.3, axis='x')
    ax8.set_xlim(0, 100)
    
    # 9. Summary statistics table (gs[2, 0:2]) - moved to row 2
    ax9 = fig.add_subplot(gs[2, 0:2])
    ax9.axis('off')
    
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
    
    table = ax9.table(
        cellText=table_data,
        colLabels=["Dataset", "Uncertainty", "BNDL_AUE J&F", "Δ vs SAM-2"],
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
    
    ax9.set_title('UA Summary Statistics', fontweight='bold', fontsize=12, pad=20)
    
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
            writer.writerow([
                "Dataset",
                "PCC_U_vs_Acc_BNDL_AUE", f"Delta_vs_{source_domain}_BNDL_AUE",
                "PCC_U_vs_Acc_BNDL", f"Delta_vs_{source_domain}_BNDL",
                "PCC_U_vs_Acc_UCTTA", f"Delta_vs_{source_domain}_UCTTA",
                "PCC_U_vs_Acc_UR_ERN", f"Delta_vs_{source_domain}_UR_ERN",
            ])
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
                writer.writerow([d,
                                 (f"{pcc_b:.6f}" if pcc_b is not None else ""),
                                 (f"{delta_b:+.6f}" if isinstance(delta_b, float) else delta_b),
                                 (f"{pcc_bp:.6f}" if pcc_bp is not None else ""),
                                 (f"{delta_bp:+.6f}" if isinstance(delta_bp, float) else delta_bp),
                                 (f"{pcc_u:.6f}" if pcc_u is not None else ""),
                                 (f"{delta_u:+.6f}" if isinstance(delta_u, float) else delta_u),
                                 (f"{pcc_r:.6f}" if pcc_r is not None else ""),
                                 (f"{delta_r:+.6f}" if isinstance(delta_r, float) else delta_r),
                                 ])
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
        total_samples = sum(len(samples['uncertainty']) for samples in data.values())
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
                    pavpu_data[dataset_name] = {'uncertainty': [], 'accuracy': []}
                
                # Store uncertainty samples
                if isinstance(value, list):
                    pavpu_data[dataset_name]['uncertainty'] = value
                
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
                    pavpu_data[dataset_name] = {'uncertainty': [], 'accuracy': []}
                
                # Store accuracy samples
                if isinstance(value, list):
                    pavpu_data[dataset_name]['accuracy'] = value
    
    # Filter out datasets with missing or empty data
    pavpu_data = {
        dataset: samples 
        for dataset, samples in pavpu_data.items() 
        if samples['uncertainty'] and samples['accuracy'] and 
           len(samples['uncertainty']) == len(samples['accuracy'])
    }
    
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
        "BNDL": "#ff7f0e",      # Orange
        "UR-ERN": "#2ca02c",    # Green
        "UCTTA": "#d62728",     # Red
    }
    
    for idx, dataset in enumerate(all_datasets):
        ax = axes[idx]
        
        for method, method_data in methods_with_pavpu.items():
            if dataset not in method_data:
                continue
            
            # Get raw uncertainty and accuracy samples
            samples = method_data[dataset]
            uncertainty = np.array(samples['uncertainty'])
            accuracy = np.array(samples['accuracy'])
            
            # Plot scatter (with alpha for overlapping points)
            color = method_colors.get(method, "#333333")
            ax.scatter(uncertainty, accuracy, alpha=0.3, s=1, 
                      color=color, label=method, rasterized=True)
        
        # Ideal calibration line (diagonal)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Perfect calibration')
        
        # Formatting
        ax.set_xlabel("Pixel Uncertainty", fontsize=10)
        ax.set_ylabel("Pixel Accuracy (0=wrong, 1=correct)", fontsize=10)
        ax.set_title(f"{dataset}", fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc='best', markerscale=5)
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.05, 1.05])
    
    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("TRUE PAvPU: Pixel Uncertainty vs Accuracy (NO thresholds)", 
                 fontsize=14, fontweight='bold', y=1.0)
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
            all_uncertainty.extend(samples['uncertainty'])
            all_accuracy.extend(samples['accuracy'])
        
        if not all_uncertainty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create hexbin density plot
        hb = ax.hexbin(all_uncertainty, all_accuracy, gridsize=30, cmap='YlOrRd', 
                      mincnt=1, reduce_C_function=np.sum, linewidths=0.2)
        
        # Ideal calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Perfect calibration')
        
        # Formatting
        ax.set_xlabel("Pixel Uncertainty", fontsize=10)
        ax.set_ylabel("Pixel Accuracy (0=wrong, 1=correct)", fontsize=10)
        ax.set_title(f"{dataset}\n({len(all_uncertainty)} pixels, all methods)", 
                    fontweight='bold', fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.05, 1.05])
        ax.legend(fontsize=8, loc='upper left')
        
        # Add colorbar
        plt.colorbar(hb, ax=ax, label='Pixel count')
    
    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("PAvPU Density (Hexbin): Pixel Uncertainty vs Accuracy (NO thresholds)", 
                 fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    # Save
    plots_dir = output_path / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)
    
    hexbin_path = plots_dir / "pavpu_hexbin_density.png"
    plt.savefig(hexbin_path, dpi=300, bbox_inches="tight")
    plt.savefig(plots_dir / "pavpu_hexbin_density.pdf", bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ PAvPU hexbin density plots saved: {hexbin_path}")


def load_results_from_detailed_json(
    json_path: Path,
) -> tuple[
    dict[str, tuple[float, float, float]],  # sam2_results
    dict[str, tuple[float, float, float]],  # bndl_aue_results
    dict[str, Any],                          # bndl_aue_statistics
    dict[str, tuple[float, float, float]],  # bndl_results
    dict[str, Any],                          # bndl_statistics
    dict[str, tuple[float, float, float]] | None,  # uctta_results
    dict[str, tuple[float, float, float]] | None,  # ur_ern_results
    dict[str, dict[str, Any]],              # ua_data_per_dataset
    dict[str, Any],                          # uctta_statistics
    dict[str, Any],                          # ur_ern_statistics
]:
    """Load previously saved results to avoid re-running experiments.

    Expects a file produced by this script at: output_path/comparison_plots/detailed_results.json
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Detailed results JSON not found: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    sam2_map = {}
    bndl_aue_map = {}
    bndl_map = {}
    bndl_aue_stats_map = data.get("bndl_aue_statistics", {})
    stats_map = data.get("bndl_statistics", {})
    uctta_map = None
    ur_ern_map = None
    ua_data_map = data.get("ua_data", {})
    uctta_stats_map = data.get("uctta_statistics", {})
    ur_ern_stats_map = data.get("ur_ern_statistics", {})

    # Convert nested dicts to expected tuple format
    for k, v in data.get("sam2_results", {}).items():
        # v: {"jf": x, "j": y, "f": z}
        jf = float(v.get("jf", 0.0))
        j = float(v.get("j", 0.0))
        f = float(v.get("f", 0.0))
        sam2_map[k] = (jf, j, f)
    for k, v in data.get("bndl_aue_results", {}).items():
        jf = float(v.get("jf", 0.0))
        j = float(v.get("j", 0.0))
        f = float(v.get("f", 0.0))
        bndl_aue_map[k] = (jf, j, f)
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

    # UR-ERN (optional)
    if "ur_ern_results" in data:
        ur_ern_map = {}
        for k, v in data.get("ur_ern_results", {}).items():
            jf = float(v.get("jf", 0.0))
            j = float(v.get("j", 0.0))
            f = float(v.get("f", 0.0))
            ur_ern_map[k] = (jf, j, f)

    return sam2_map, bndl_aue_map, bndl_aue_stats_map, bndl_map, stats_map, uctta_map, ur_ern_map, ua_data_map, uctta_stats_map, ur_ern_stats_map


def parse_args():
    p = argparse.ArgumentParser(description="Compare SAM-2 vs BNDL vs BNDL_AUE zero-shot evaluation")

    # Dataset selection
    p.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=list(DATASET_CONFIGS.keys()),
        help="Datasets to evaluate (default: all including MOSE_train/val)",
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

    # BNDL configuration (optional, only needed if --run_bndl is set)
    p.add_argument(
        "--bndl_cfg",
        default="configs/sam2.1/sam2.1_hiera_b+_bndl.yaml",
        help="BNDL config file",
    )
    p.add_argument(
        "--bndl_checkpoint",
        default=None,
        help="BNDL checkpoint path (required only if --run_bndl is set)",
    )

    # BNDL_AUE configuration
    p.add_argument(
        "--bndl_aue_cfg",
        default="configs/sam2.1/sam2.1_hiera_b+_bndl_aue.yaml",
        help="BNDL_AUE config file",
    )
    p.add_argument(
        "--bndl_aue_checkpoint",
        default="/home/hongyou/dev/ada_samp/logs/sam2/sam2_bndl_012_02/checkpoints/checkpoint.pt",
        help="BNDL_AUE checkpoint path",
    )

    # SAM-2+UR-ERN configuration
    p.add_argument(
        "--ur_ern_cfg",
        default="configs/sam2.1/sam2.1_hiera_b+_ur_ern.yaml",
        help="SAM-2+UR-ERN config file",
    )
    p.add_argument(
        "--ur_ern_checkpoint",
        default="/home/hongyou/dev/ada_samp/logs/sam2/sam2_ur_ern_001_01/checkpoints/checkpoint.pt",
        help="SAM-2+UR-ERN checkpoint path",
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
    p.add_argument("--save_vis", action="store_true", default=False, help="Save visualizations")
    p.add_argument("--collect_bndl_stats", action="store_true", default=True, help="Collect BNDL statistics")
    # Click protocol options
    p.add_argument("--click_protocol", type=str, default="3click", 
                   choices=["1click", "3click", "5click"], 
                   help="Interaction protocol for first frame")
    p.add_argument("--min_click_dist", type=float, default=12.0, help="Minimum distance between clicks for 5-click protocol")
    p.add_argument("--seed", type=int, default=0, help="Random seed for 'random' point initialization")
    
    # Downsampling parameters
    p.add_argument("--downsample_max_samples", type=int, default=100000, 
                   help="Maximum number of samples to keep after downsampling (default: 100000)")
    
    # AUE version suffix for comparison plots folder naming
    p.add_argument("--aue_version", type=str, default=None,
                   help="AUE version suffix for comparison plots folder (e.g., '016_02' for comparison_plots_AUE_016_02)")

    # Cached results options
    p.add_argument("--load_detailed_json", type=str, default=None, 
                   help="Path to a previously saved detailed_results.json to render plots and summaries without re-running")
    p.add_argument("--plot_only", action="store_true", default=False, 
                   help="Only generate plots from existing detailed_results.json (requires --load_detailed_json or existing results in output_path)")
    # Method toggles
    p.add_argument("--run_sam", action="store_true", default=False, help="Run baseline SAM-2 branch")
    p.add_argument("--run_uctta", action="store_true", default=False, help="Run SAM-2 + UCTTA branch")
    p.add_argument("--run_bndl", action="store_true", default=False, help="Run BNDL branch")
    p.add_argument("--run_bndl_aue", action="store_true", default=False, help="Run BNDL_AUE branch")
    p.add_argument("--run_ur_ern", action="store_true", default=False, help="Run SAM-2 + UR-ERN branch")
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
    
    # If plot_only mode, try to load existing results
    if args.plot_only:
        print("\n🎨 Plot-only mode enabled - will only generate visualizations from existing results")
        
        # Try to find detailed_results.json
        if args.load_detailed_json is not None:
            detailed_path = Path(args.load_detailed_json)
        else:
            # Auto-detect from output_path
            detailed_path = output_path / "comparison_plots" / "detailed_results.json"
            if not detailed_path.exists():
                print(f"❌ Error: No detailed_results.json found at {detailed_path}")
                print("   Please specify --load_detailed_json or run evaluation first without --plot_only")
                return
        
        print(f"📂 Loading results from: {detailed_path}")
        (sam2_results, bndl_aue_results, bndl_aue_statistics, bndl_results, bndl_statistics,
         uctta_results, ur_ern_results, ua_data, uctta_statistics, ur_ern_statistics) = (
            load_results_from_detailed_json(detailed_path)
        )
        print("✓ Results loaded successfully")
        
    # Optionally load previous results to avoid re-running
    elif args.load_detailed_json is not None:
        detailed_path = Path(args.load_detailed_json)
        print(f"Loading cached results from: {detailed_path}")
        (sam2_results, bndl_aue_results, bndl_aue_statistics, bndl_results, bndl_statistics,
         uctta_results, ur_ern_results, ua_data, uctta_statistics, ur_ern_statistics) = (
            load_results_from_detailed_json(detailed_path)
        )
    else:
        # Run comparison evaluation
        print(f"Datasets: {args.datasets}")
        print(f"SAM-2 config: {args.sam2_cfg}")
        print(f"SAM-2 checkpoint: {args.sam2_checkpoint}")
        print(f"BNDL config: {args.bndl_cfg}")
        print(f"BNDL checkpoint: {args.bndl_checkpoint}")
        print(f"BNDL_AUE config: {args.bndl_aue_cfg}")
        print(f"BNDL_AUE checkpoint: {args.bndl_aue_checkpoint}")
        sam2_results, bndl_aue_results, bndl_aue_statistics, bndl_results, bndl_statistics, uctta_results, ur_ern_results, ua_data, uctta_statistics, ur_ern_statistics = run_comparison_evaluation(
            datasets=args.datasets,
            sam2_cfg=args.sam2_cfg,
            sam2_checkpoint=args.sam2_checkpoint,
            bndl_aue_cfg=args.bndl_aue_cfg,
            bndl_aue_checkpoint=args.bndl_aue_checkpoint,
            bndl_cfg=args.bndl_cfg,
            bndl_checkpoint=args.bndl_checkpoint,
            output_path=output_path,
            ur_ern_cfg=args.ur_ern_cfg,
            ur_ern_checkpoint=args.ur_ern_checkpoint,
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
            run_bndl_aue=args.run_bndl_aue,
            run_bndl=args.run_bndl,
            run_ur_ern=args.run_ur_ern,
            click_protocol=args.click_protocol,
            min_click_dist=args.min_click_dist,
            seed=args.seed,
            # Full UCTTA parameters
            uctta_enable_bn=args.uctta_enable_bn,
            uctta_fisher_reg=args.uctta_fisher_reg,
            uctta_fisher_alpha=args.uctta_fisher_alpha,
            uctta_entropy_th=args.uctta_entropy_th,
            uctta_selection_p=args.uctta_selection_p,
            downsample_max_samples=args.downsample_max_samples,
        )

    # Save detailed results JSON (needed for parallel_compare.py) - skip if plot_only mode
    if not args.plot_only:
        save_detailed_results(
            output_path=output_path,
            sam2_results=sam2_results,
            bndl_aue_results=bndl_aue_results,
            bndl_aue_statistics=bndl_aue_statistics,
            bndl_results=bndl_results,
            bndl_statistics=bndl_statistics,
            uctta_results=uctta_results,
            ur_ern_results=ur_ern_results,
            uctta_statistics=uctta_statistics,
            ur_ern_statistics=ur_ern_statistics,
            ua_data=ua_data,
        )
    
    # Generate plots if requested or if plot_only mode
    if args.plot_only or (not args.load_detailed_json and any([args.run_sam, args.run_bndl, args.run_bndl_aue])):
        print("\n" + "=" * 80)
        print("📊 Generating visualization plots...")
        print("=" * 80)
        
        # Create comparison plots
        if sam2_results and bndl_aue_results:
            print("\n🎨 Creating comprehensive comparison plots...")
            try:
                # 从命令行参数获取AUE版本（parallel_compare.py会传递）
                aue_version = getattr(args, 'aue_version', None)
                if aue_version:
                    print(f"📌 使用AUE版本: {aue_version}")
                
                create_comprehensive_comparison_plots(
                    sam2_results=sam2_results,
                    bndl_results=bndl_aue_results,
                    bndl_statistics=bndl_aue_statistics,
                    output_path=output_path,
                    uctta_results=uctta_results,
                    uctta_statistics=uctta_statistics,
                    aue_version=aue_version,
                )
                print("✓ Comprehensive comparison plots generated")
            except Exception as e:
                print(f"⚠️  Warning: Failed to generate comparison plots: {e}")
                import traceback
                traceback.print_exc()
        
        # Create UA shift analysis plots
        if bndl_aue_statistics and sam2_results and bndl_aue_results:
            print("\n🎨 Creating UA shift analysis plots...")
            try:
                # Build root paths for loading per-dataset results
                sam2_root = output_path / "sam2_results" if (output_path / "sam2_results").exists() else None
                bndl_aue_root = output_path / "bndl_aue_results" if (output_path / "bndl_aue_results").exists() else None
                bndl_root = output_path / "bndl_results" if (output_path / "bndl_results").exists() else None
                uctta_root = output_path / "sam2_uctta_results" if (output_path / "sam2_uctta_results").exists() else None
                ur_ern_root = output_path / "sam2_ur_ern_results" if (output_path / "sam2_ur_ern_results").exists() else None
                
                create_ua_shift_analysis_plots(
                    bndl_statistics=bndl_aue_statistics,
                    sam2_results=sam2_results,
                    bndl_results=bndl_aue_results,
                    output_path=output_path,
                    uctta_statistics=uctta_statistics,
                    uctta_results=uctta_results,
                    ur_ern_results=ur_ern_results,
                    bndl_pure_statistics=bndl_statistics,
                    bndl_pure_results=bndl_results,
                    sam2_root_override=sam2_root,
                    bndl_root_override=bndl_aue_root,
                    uctta_root_override=uctta_root,
                    ur_ern_root_override=ur_ern_root,
                    bndl_pure_root_override=bndl_root,
                )
                print("✓ UA shift analysis plots generated")
            except Exception as e:
                print(f"⚠️  Warning: Failed to generate UA shift plots: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("✓ All visualization plots generated successfully!")
        print("=" * 80)


if __name__ == "__main__":
    main()
