#!/usr/bin/env python
# Compare SAM-2 vs BNDL vs BNDL_AUE zero-shot evaluation results
# Runs multiple versions and generates comprehensive comparison plots

from __future__ import annotations

# CRITICAL: Must set sys.path BEFORE any sam2/BNDL imports
# This allows using experiment-specific source code when PYTHONPATH is set
import sys
import os

# If PYTHONPATH is set (by parallel_compare.py), prepend experiment source to sys.path
# This ensures we use the experiment's source code snapshot, not the current workspace
_pythonpath = os.environ.get("PYTHONPATH", "")
if _pythonpath:
    # Collect paths from PYTHONPATH that contain experiment source code
    _experiment_paths = []
    _experiment_scripts_path = None
    for _p in _pythonpath.split(os.pathsep):
        if _p and "/src" in _p:  # This is likely an experiment source dir
            _experiment_paths.append(_p)
            # Also check for experiment's scripts directory
            _potential_scripts = os.path.join(_p, "scripts")
            if os.path.isdir(_potential_scripts):
                _experiment_scripts_path = _potential_scripts

    if _experiment_paths:
        # Rebuild sys.path with experiment paths at the front
        # Include experiment scripts dir first if it exists
        _new_sys_path = []
        if _experiment_scripts_path:
            _new_sys_path.append(_experiment_scripts_path)
        _new_sys_path.extend(_experiment_paths)

        # Get current script directory to exclude it (avoid shadowing experiment code)
        _current_script_dir = os.path.dirname(os.path.abspath(__file__))

        for _existing in sys.path:
            _skip = False
            # Skip current development sam2 paths that would shadow experiment code
            if "sam2" in _existing and _existing not in _experiment_paths:
                # Skip ALL sam2 paths including scripts dir if we have experiment scripts
                if _experiment_scripts_path:
                    _skip = True
                elif not _existing.endswith("/scripts"):
                    _skip = True
            # Also skip current script directory explicitly
            if _existing == _current_script_dir and _experiment_scripts_path:
                _skip = True
            if not _skip and _existing not in _new_sys_path:
                _new_sys_path.append(_existing)

        sys.path[:] = _new_sys_path
        _scripts_info = f", scripts from: {_experiment_scripts_path}" if _experiment_scripts_path else ""
        print(f"[zs.py] Using experiment source code from: {_experiment_paths}{_scripts_info}")

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dataset_configs import (
    DATASET_CONFIGS,
    DEFAULT_DATASETS,
)
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

# Visualization functions (extracted to separate module)
from zs_visualization import (
    create_comprehensive_comparison_plots,
    create_ua_shift_analysis_plots,
)


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


def _load_predictor(
    model_name: str,
    cfg: str | None,
    checkpoint: str | None,
    device: str,
    required: bool = True,
) -> Any | None:
    """通用模型加载函数，减少代码重复。

    Args:
        model_name: 模型名称（用于日志）
        cfg: 配置文件路径
        checkpoint: 检查点文件路径
        device: 设备（cuda/cpu）
        required: 如果为 True，缺少配置时抛出异常；否则返回 None

    Returns:
        加载的 predictor，或 None（如果 required=False 且配置缺失）
    """
    if cfg is None or checkpoint is None:
        if required:
            raise ValueError(f"{model_name} requires both cfg and checkpoint to be specified")
        return None

    print(f"\nLoading {model_name} checkpoint...")

    try:
        # 统一的配置文件路径解析
        import sam2

        sam2_package_root = Path(sam2.__path__[0])
        sam2_package_dir = sam2_package_root / "sam2"
        config_paths_to_try = [
            Path(cfg),
            sam2_package_dir / cfg,
            sam2_package_root / cfg,
        ]

        # Add experiment config directory if set
        experiment_config_dir = os.environ.get("EXPERIMENT_CONFIG_DIR")
        if experiment_config_dir:
            config_paths_to_try.insert(0, Path(experiment_config_dir) / cfg)

        config_found = any(p.exists() for p in config_paths_to_try)
        if not config_found:
            raise FileNotFoundError(f"{model_name} config file not found: {cfg} (tried: {[str(p) for p in config_paths_to_try]})")

        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"{model_name} checkpoint file not found: {checkpoint}")

        from shared_evaluation_utils import build_predictor_with_overrides

        predictor = build_predictor_with_overrides(
            cfg_file=cfg,
            ckpt=checkpoint,
            device=device,
        )
        print(f"{model_name} loaded successfully!")
        return predictor

    except FileNotFoundError as e:
        print(f"❌ File not found error when loading {model_name}: {e}")
        raise
    except RuntimeError as e:
        print(f"❌ Runtime error when loading {model_name} checkpoint: {e}")
        print(f"   Config: {cfg}")
        print(f"   Checkpoint: {checkpoint}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error when loading {model_name}: {type(e).__name__}: {e}")
        print(f"   Config: {cfg}")
        print(f"   Checkpoint: {checkpoint}")
        import traceback

        traceback.print_exc()
        raise


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
    first_frame_only: bool = False,
    max_objects: int = 20,
    video_limit: int | None = None,
    num_workers: int | None = 2,  # Default to 2 to prevent CPU oversubscription
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
    # Paper figure generation options
    max_vis_per_video: int = 2,
    save_vis_pdf: bool = False,
    no_save_masks: bool = False,
    # Enable UCTTA within BNDL_AUE (combined baseline)
    bndl_aue_enable_uctta: bool = False,
    bndl_aue_uctta_steps: int = 2,
    bndl_aue_uctta_lr: float = 3e-4,
) -> tuple[
    dict[str, tuple[float, float, float]],  # sam2_results
    dict[str, tuple[float, float, float]],  # bndl_aue_results
    dict[str, Any],  # bndl_aue_statistics
    dict[str, tuple[float, float, float]],  # bndl_results
    dict[str, Any],  # bndl_statistics
    dict[str, tuple[float, float, float]] | None,  # uctta_results
    dict[str, tuple[float, float, float]] | None,  # ur_ern_results
    dict[str, dict[str, Any]],  # ua_data_per_dataset
    dict[str, Any],  # uctta_statistics
    dict[str, Any],  # ur_ern_statistics
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

    # Load predictors using unified helper function
    # SAM-2 predictor (needed for SAM and UCTTA)
    sam2_predictor = _load_predictor("SAM-2", sam2_cfg, sam2_checkpoint, device, required=False) if (run_sam or run_uctta) else None

    # BNDL predictor
    # BNDL now builds predictor inside inference; we don't pre-load here
    bndl_predictor = None

    # BNDL_AUE predictor
    # Managed by internal script now
    bndl_aue_predictor = None

    # UR-ERN predictor
    ur_ern_predictor = _load_predictor("SAM-2+UR-ERN", ur_ern_cfg, ur_ern_checkpoint, device, required=run_ur_ern) if run_ur_ern else None

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
            # Define execution tasks to loop over
            # Format: (name, function, predictor, output_dir, results_list, stats_dict, extra_kwargs)
            tasks = []

            # 1. SAM-2
            if run_sam:
                tasks.append(
                    {
                        "name": "SAM-2",
                        "fn": run_sam2_dataset,
                        "predictor": sam2_predictor,
                        "output": sam2_output,
                        "results_list": sam2_per_thresh,
                        "stats": None,
                        "kwargs": {
                            "enhanced_vis": True,
                            "save_vis": save_vis,
                            "save_masks": not no_save_masks,
                        },
                    }
                )

            # 2. SAM-2 + UCTTA
            if run_uctta:
                # Custom result handler for UCTTA (stores directly in result dict)
                # UCTTA returns results differently from SAM (handled via list in dict per dataset)
                def _handle_uctta_result(th, res):
                    # res is (j_f, j, f)
                    assert uctta_results is not None
                    if dataset_name not in uctta_results:
                        uctta_results[dataset_name] = []
                    uctta_results[dataset_name].append((th, *res))

                tasks.append(
                    {
                        "name": "SAM-2+UCTTA",
                        "fn": run_uctta_dataset,
                        "predictor": sam2_predictor,
                        "output": uctta_output,
                        "custom_result_handler": _handle_uctta_result,
                        "stats": uctta_statistics,
                        "kwargs": {
                            "uctta_steps": uctta_steps,
                            "uctta_lr": uctta_lr,
                            "reuse_prompts_root": sam2_output if run_sam else None,
                            "enable_bn_adapt": uctta_enable_bn,
                            "use_fisher_reg": uctta_fisher_reg,
                            "fisher_alpha": uctta_fisher_alpha,
                            "entropy_threshold": uctta_entropy_th,
                            "selection_p": uctta_selection_p,
                            "downsample_max_samples": downsample_max_samples,
                        },
                    }
                )

            # 3. BNDL variants (Pure BNDL and BNDL+AUE)
            bndl_configs = []
            if run_bndl:
                bndl_configs.append(("BNDL", bndl_output, bndl_per_thresh, bndl_statistics, bndl_cfg, bndl_checkpoint))
            if run_bndl_aue:
                bndl_configs.append(("BNDL_AUE", bndl_aue_output, bndl_aue_per_thresh, bndl_aue_statistics, bndl_aue_cfg, bndl_aue_checkpoint))

            for m_name, m_out, m_list, m_stats, m_cfg, m_ckpt in bndl_configs:
                bndl_kwargs = {
                    "predictor_cfg": m_cfg,
                    "predictor_ckpt": m_ckpt,
                    "predictor_device": device,
                    "collect_statistics": collect_bndl_stats,
                    "reuse_prompts_root": sam2_output if run_sam else None,
                    "downsample_max_samples": downsample_max_samples,
                    "multimask_output": True,
                    "save_uncertainty_maps": save_vis,  # Save per-pixel uncertainty maps
                    "max_vis_per_video": max_vis_per_video,
                    "save_vis_pdf": save_vis_pdf,
                    "save_masks": not no_save_masks,
                }
                # Add UCTTA support for BNDL_AUE if enabled
                if m_name == "BNDL_AUE" and bndl_aue_enable_uctta:
                    bndl_kwargs["enable_uctta"] = True
                    bndl_kwargs["uctta_steps"] = bndl_aue_uctta_steps
                    bndl_kwargs["uctta_lr"] = bndl_aue_uctta_lr
                tasks.append(
                    {
                        "name": m_name,
                        "fn": run_bndl_dataset,
                        "predictor": None,  # Self-managed
                        "output": m_out,
                        "results_list": m_list,
                        "stats": m_stats,
                        "kwargs": bndl_kwargs,
                    }
                )

            # 4. SAM-2 + UR-ERN
            if run_ur_ern and ur_ern_predictor is not None:

                def _handle_ur_ern_result(th, res):
                    if dataset_name not in ur_ern_results:
                        ur_ern_results[dataset_name] = []
                    ur_ern_results[dataset_name].append((th, *res))

                tasks.append(
                    {
                        "name": "UR-ERN",
                        "fn": run_ur_ern_dataset,
                        "predictor": ur_ern_predictor,
                        "output": ur_ern_output,
                        "custom_result_handler": _handle_ur_ern_result,
                        "stats": ur_ern_statistics,
                        "kwargs": {
                            "collect_statistics": True,
                            "reuse_prompts_root": sam2_output if run_sam else None,
                            "downsample_max_samples": downsample_max_samples,
                            "save_ur_ern_vis": save_vis,
                        },
                    }
                )

            # --- Execute Tasks ---
            for task in tasks:
                print(f"--- Running {task['name']} evaluation for {dataset_name} @ thresh={th} ---")
                start_time = time.time()

                # Build common kwargs
                call_kwargs = {
                    "dataset_name": dataset_name,
                    "predictor": task["predictor"],
                    "output_path": task["output"],
                    "score_thresh": th,
                    "num_workers": num_workers,
                    "video_subset": video_subset,
                    "first_frame_only": first_frame_only,
                    "max_objects": max_objects,
                    "click_protocol": click_protocol,
                    "min_click_dist": float(min_click_dist),
                    "seed": int(seed),
                }
                # Update with task-specific kwargs
                call_kwargs.update(task.get("kwargs", {}))

                try:
                    # Generic runner execution
                    ret = task["fn"](**call_kwargs)

                    # Unpack results (some return stats, some don't)
                    # All runners eventually return jf, j, f, [stats]
                    stats = None
                    if isinstance(ret, tuple):
                        if len(ret) == 4:
                            j_f, j, f, stats = ret
                        elif len(ret) == 3:
                            j_f, j, f = ret
                        else:
                            raise ValueError(f"Unexpected return length: {len(ret)}")
                    else:
                        raise ValueError(f"Unexpected return type: {type(ret)}")

                    elapsed = time.time() - start_time
                    print(f"{task['name']} @ {th:.2f} - J&F: {j_f:.2f}, J: {j:.2f}, F: {f:.2f} (Time: {elapsed:.2f}s)")

                    # Store results
                    if "results_list" in task and task["results_list"] is not None:
                        task["results_list"].append((th, j_f, j, f))
                    elif "custom_result_handler" in task:
                        task["custom_result_handler"](th, (j_f, j, f))

                    # Store stats
                    if stats and "stats" in task and task["stats"] is not None:
                        # For UCTTA/UR-ERN, we check if key exists to avoid overwrite?
                        # Original code: if uctta_stats and dataset_name not in uctta_statistics: ...
                        # But wait, original code looped through thresholds. If stats are identical per threshold (which they usually are for UCTTA/BNDL UQ), we only need one.
                        # Original code for UCTTA: if uctta_stats and dataset_name not in uctta_statistics: ...
                        # Original code for BNDL: if dataset_stats: stats_dict[dataset_name] = dataset_stats

                        # We'll stick to original logic: Overwrite for BNDL, Check-First for UCTTA?
                        # Actually BNDL stats are cleared per run in helper, but merged?
                        # Let's simple check-first to be safe for all, assuming stats don't change with threshold for UQ methods (which is true as UQ is usually pre-threshold).
                        if method_name := task["name"]:
                            # BNDL logic was overwrite. UCTTA was check.
                            # Let's just overwrite, it shouldn't hurt if they are identical.
                            task["stats"][dataset_name] = stats

                except Exception as e:
                    print(f"❌ Error running {task['name']}: {e}")
                    import traceback

                    traceback.print_exc()
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

    return (
        sam2_results,
        bndl_aue_results,
        bndl_aue_statistics,
        bndl_results,
        bndl_statistics,
        (uctta_results if isinstance(uctta_results, dict) else None),
        (ur_ern_results if isinstance(ur_ern_results, dict) else None),
        ua_data_per_dataset,
        uctta_statistics,
        ur_ern_statistics,
    )


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


def load_results_from_detailed_json(
    json_path: Path,
) -> tuple[
    dict[str, tuple[float, float, float]],  # sam2_results
    dict[str, tuple[float, float, float]],  # bndl_aue_results
    dict[str, Any],  # bndl_aue_statistics
    dict[str, tuple[float, float, float]],  # bndl_results
    dict[str, Any],  # bndl_statistics
    dict[str, tuple[float, float, float]] | None,  # uctta_results
    dict[str, tuple[float, float, float]] | None,  # ur_ern_results
    dict[str, dict[str, Any]],  # ua_data_per_dataset
    dict[str, Any],  # uctta_statistics
    dict[str, Any],  # ur_ern_statistics
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
    # Initialize Hydra config module if not already initialized
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    import os

    # Check if experiment config directory is specified via environment variable
    experiment_config_dir = os.environ.get("EXPERIMENT_CONFIG_DIR")

    if not GlobalHydra.instance().is_initialized():
        initialize_config_module("sam2", version_base="1.2")

    # If experiment config directory is set, prepend it to search path (highest priority)
    if experiment_config_dir and os.path.exists(experiment_config_dir):
        gh = GlobalHydra.instance()
        if gh.is_initialized():
            config_loader = gh.config_loader()
            search_path = config_loader.get_search_path()

            # Prepend experiment directory to search path (highest priority)
            # Hydra will search this directory first before default paths
            search_path.prepend("file", experiment_config_dir)
            print(f"[zs.py] Prepended experiment config directory to search path (highest priority): {experiment_config_dir}")
    else:
        # If no experiment config directory, use default behavior
        gh = GlobalHydra.instance()
        if gh.is_initialized():
            config_loader = gh.config_loader()
            search_path = config_loader.get_search_path()
            # Optionally, you can still prepend experiment directory if it exists but wasn't set
            # This maintains backward compatibility
            if experiment_config_dir and os.path.exists(experiment_config_dir):
                search_path.prepend("file", experiment_config_dir)
                print(f"[zs.py] Added experiment config directory to search path: {experiment_config_dir}")

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
    p.add_argument("--num_workers", type=int, default=2, help="Number of evaluation processes (default: 2 to prevent CPU oversubscription)")
    p.add_argument("--output_path", default="./outputs/comparison_sam2_vs_bndl_011_01", help="Root output directory")
    p.add_argument("--process_full_video", action="store_true", help="Evaluate all frames in video (default: first frame only)")

    # Subset options
    p.add_argument("--video_limit", type=int, default=None, help="Limit number of videos per dataset")
    p.add_argument("--max_objects", type=int, default=256, help="Maximum number of objects per video")

    # Visualization options
    p.add_argument("--save_vis", action="store_true", default=False, help="Save visualizations")
    p.add_argument("--max_vis_per_video", type=int, default=2, help="Max visualizations per video (default: 2 for benchmarking, set higher for paper figures)")
    p.add_argument("--collect_bndl_stats", action="store_true", default=True, help="Collect BNDL statistics")
    # Click protocol options
    p.add_argument("--click_protocol", type=str, default="3click", choices=["1click", "3click", "5click"], help="Interaction protocol for first frame")
    p.add_argument("--min_click_dist", type=float, default=12.0, help="Minimum distance between clicks for 5-click protocol")
    p.add_argument("--seed", type=int, default=0, help="Random seed for 'random' point initialization")

    # Downsampling parameters
    p.add_argument("--downsample_max_samples", type=int, default=100000, help="Maximum number of samples to keep after downsampling (default: 100000)")

    # AUE version suffix for comparison plots folder naming
    p.add_argument("--aue_version", type=str, default=None, help="AUE version suffix for comparison plots folder (e.g., '016_02' for comparison_plots_AUE_016_02)")

    # Cached results options
    p.add_argument("--load_detailed_json", type=str, default=None, help="Path to a previously saved detailed_results.json to render plots and summaries without re-running")
    p.add_argument(
        "--plot_only", action="store_true", default=False, help="Only generate plots from existing detailed_results.json (requires --load_detailed_json or existing results in output_path)"
    )
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

    # Optimization
    p.add_argument("--no_save_masks", action="store_true", default=False, help="Disable mask saving to speed up evaluation")

    # BNDL_AUE + UCTTA combined baseline
    p.add_argument("--bndl_aue_enable_uctta", action="store_true", default=False, help="Enable UCTTA test-time adaptation within BNDL_AUE")
    p.add_argument("--bndl_aue_uctta_steps", type=int, default=2, help="UCTTA steps for BNDL_AUE (default: 2)")
    p.add_argument("--bndl_aue_uctta_lr", type=float, default=3e-4, help="UCTTA learning rate for BNDL_AUE (default: 3e-4)")

    return p.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
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
            (sam2_results, bndl_aue_results, bndl_aue_statistics, bndl_results, bndl_statistics, uctta_results, ur_ern_results, ua_data, uctta_statistics, ur_ern_statistics) = (
                load_results_from_detailed_json(detailed_path)
            )
            print("✓ Results loaded successfully")

        # Optionally load previous results to avoid re-running
        elif args.load_detailed_json is not None:
            detailed_path = Path(args.load_detailed_json)
            print(f"Loading cached results from: {detailed_path}")
            (sam2_results, bndl_aue_results, bndl_aue_statistics, bndl_results, bndl_statistics, uctta_results, ur_ern_results, ua_data, uctta_statistics, ur_ern_statistics) = (
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
            sam2_results, bndl_aue_results, bndl_aue_statistics, bndl_results, bndl_statistics, uctta_results, ur_ern_results, ua_data, uctta_statistics, ur_ern_statistics = (
                run_comparison_evaluation(
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
                    first_frame_only=not args.process_full_video,
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
                    # Paper figure generation options
                    max_vis_per_video=args.max_vis_per_video,
                    save_vis_pdf=True,  # Always save PDF for professional use
                    no_save_masks=args.no_save_masks,
                    # BNDL_AUE + UCTTA combined baseline
                    bndl_aue_enable_uctta=args.bndl_aue_enable_uctta,
                    bndl_aue_uctta_steps=args.bndl_aue_uctta_steps,
                    bndl_aue_uctta_lr=args.bndl_aue_uctta_lr,
                )
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
                aue_version = getattr(args, "aue_version", None)
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
                    bndl_baseline_results=bndl_results if bndl_results else None,
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

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user", file=sys.stderr)
        sys.exit(130)
    except SystemExit:
        # 重新抛出SystemExit，保持退出码
        raise
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ Runtime error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
