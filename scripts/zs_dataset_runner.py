#!/usr/bin/env python
"""
Shared dataset evaluation runner for zero-shot evaluation.

Consolidates the duplicated `run_single_dataset_with_*` wrapper functions
that exist in each method-specific script.

Usage:
    from zs_dataset_runner import run_single_dataset_generic

    jf, j, f, stats = run_single_dataset_generic(
        dataset_name="GTEA",
        predictor=predictor,
        output_path=Path("/tmp/output"),
        inference_fn=my_inference_function,
        method_name="BNDL_AUE",
        ...
    )
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from dataset_configs import DATASET_CONFIGS
from evaluation_pipeline import run_benchmark_evaluation


# Type alias for inference function signature
InferenceFn = Callable[..., dict[str, Any] | None]


def run_single_dataset_generic(
    dataset_name: str,
    predictor: Any,
    output_path: Path,
    inference_fn: InferenceFn,
    method_name: str = "SAM",
    # Dataset options
    split: str | list[str] | None = None,
    video_subset: list[str] | None = None,
    # Evaluation options
    num_workers: int | None = None,
    first_frame_only: bool = False,
    # Common inference parameters
    score_thresh: float = 0.0,
    max_objects: int | None = None,
    reuse_prompts_root: Path | None = None,
    click_protocol: str = "3click",
    min_click_dist: float = 12.0,
    seed: int = 0,
    collect_statistics: bool = True,
    downsample_max_samples: int = 100000,
    # Method-specific kwargs passed to inference_fn
    **inference_kwargs,
) -> tuple[float, float, float, dict[str, Any] | None]:
    """
    Generic single-dataset evaluation runner.

    This consolidates the common logic in:
    - zero_shot_multi_dataset_sam_bndl.run_single_dataset_with_bndl
    - zero_shot_multi_dataset_uctta.run_single_dataset_with_uctta
    - zero_shot_multi_dataset_ur_ern.run_single_dataset_with_ur_ern

    Args:
        dataset_name: Name of dataset (must be in DATASET_CONFIGS)
        predictor: SAM2 video predictor instance
        output_path: Root output directory
        inference_fn: Method-specific inference function
        method_name: Name of method (for logging)
        split: Dataset split to use
        video_subset: Optional list of videos to process
        num_workers: Workers for benchmark evaluation
        first_frame_only: Only process first frame
        score_thresh: Mask threshold
        max_objects: Max objects per video
        reuse_prompts_root: Root for prompt reuse
        click_protocol: Click protocol
        min_click_dist: Min click distance
        seed: Random seed
        collect_statistics: Whether to collect stats
        downsample_max_samples: Max samples for stats
        **inference_kwargs: Additional args for inference_fn

    Returns:
        Tuple of (jf_score, j_score, f_score, statistics_dict)
    """
    # Get dataset config
    config = DATASET_CONFIGS[dataset_name]
    if split is None:
        split = config["default_split"]

    if isinstance(split, list):
        split = split[0]

    assert isinstance(split, str)

    # Resolve paths
    root = Path(config["root"])
    if config["has_split_subdir"]:
        jpeg_dir = root / split / "JPEGImages"
        ann_dir = root / split / "Annotations"
    else:
        jpeg_dir = root / "JPEGImages"
        ann_dir = root / "Annotations"

    if not jpeg_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"JPEGImages or Annotations not found for {dataset_name}: {jpeg_dir}, {ann_dir}")

    # Create output directory
    out_dir = output_path / f"{dataset_name.lower()}_pred"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Running {dataset_name} dataset evaluation (SAM-2 + {method_name})")
    print(f"{'=' * 60}")

    # Handle file_list_txt if present
    if "file_list_txt" in config:
        file_list_path = Path(config["file_list_txt"])
        if file_list_path.exists():
            with open(file_list_path, "r") as f:
                names = [line.strip() for line in f if line.strip()]
            video_subset = [v for v in (video_subset or names) if v in names]

    # Execute method-specific inference
    # Note: first_frame_only and min_click_dist are not passed to inference_fn
    # because inference_with_bndl was simplified to single-frame mode only
    t0 = time.time()
    statistics = inference_fn(
        predictor,
        jpeg_dir,
        ann_dir,
        out_dir,
        score_thresh=score_thresh,
        video_names=video_subset,
        max_objects=max_objects,
        collect_statistics=collect_statistics,
        dataset_name=dataset_name,
        reuse_prompts_root=reuse_prompts_root,
        click_protocol=click_protocol,
        seed=seed,
        downsample_max_samples=downsample_max_samples,
        **inference_kwargs,
    )
    t_infer = time.time() - t0

    # Run benchmark evaluation
    t1 = time.time()
    try:
        j_f_val, j_val, f_val = run_benchmark_evaluation(
            gt_dir=ann_dir,
            pred_dir=out_dir,
            dataset_config=config,
            video_subset=video_subset,
            first_frame_only=first_frame_only,
            num_workers=num_workers,
            output_path=output_path,
            use_symlinks=True,
            dataset_name=dataset_name,
        )
    except Exception as e:
        print(f"Error during evaluation of {dataset_name}: {e}")
        import traceback

        traceback.print_exc()
        return 0.0, 0.0, 0.0, None

    t_eval = time.time() - t1

    print(f"Inference time ({method_name}): {t_infer:.2f}s")
    print(f"Evaluation time: {t_eval:.2f}s")

    return j_f_val, j_val, f_val, statistics


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("Dataset Runner Module Tests")
    print("=" * 60)

    print("\n1. Module imports:")
    print("  ✓ run_single_dataset_generic")

    print("\n✓ All imports successful!")
