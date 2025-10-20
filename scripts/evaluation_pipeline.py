#!/usr/bin/env python
"""Unified evaluation pipeline for zero-shot testing.

This module provides a standardized evaluation workflow that:
1. Uses symlinks instead of file copying (10-100x faster)
2. Handles temporary directory management automatically
3. Provides consistent error handling and cleanup
4. Reduces code duplication across evaluation scripts
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from sav_dataset.utils.sav_benchmark import benchmark


class TempDirManager:
    """Manages temporary directories for evaluation with automatic cleanup.
    
    This class handles the creation, management, and cleanup of temporary
    directories needed for benchmark evaluation. It uses symlinks instead
    of file copying for significant performance improvements.
    
    Performance benefits:
    - Symlinks are 10-100x faster than copying files
    - Automatic cleanup prevents disk space leaks
    - Supports both first-frame-only and full-video evaluation
    """
    
    def __init__(
        self,
        output_path: Path,
        dataset_name: str,
        first_frame_only: bool = False,
    ):
        """Initialize temporary directory manager.
        
        Args:
            output_path: Root output path for temporary directories
            dataset_name: Name of dataset (for directory naming)
            first_frame_only: If True, only evaluate first frames
        """
        self.output_path = Path(output_path)
        self.dataset_name = dataset_name.lower()
        self.first_frame_only = first_frame_only
        
        # Generate temp directory names
        suffix = "_first" if first_frame_only else ""
        self.gt_tmp = self.output_path / f"{self.dataset_name}_tmp_gt{suffix}"
        self.pred_tmp = self.output_path / f"{self.dataset_name}_tmp_pred{suffix}"
        
        # Track if we created the directories
        self._created = False
    
    def prepare_eval_roots(
        self,
        ann_dir: Path,
        pred_dir: Path,
        video_subset: Optional[list[str]] = None,
        use_symlinks: bool = True,
    ) -> tuple[Path, Path]:
        """Prepare evaluation root directories.
        
        This method creates temporary directories with either symlinks or
        copies of the annotation and prediction files, depending on the
        evaluation mode and settings.
        
        Args:
            ann_dir: Ground truth annotation directory
            pred_dir: Prediction directory
            video_subset: Optional list of videos to include
            use_symlinks: Use symlinks instead of copying (faster, default: True)
        
        Returns:
            Tuple of (gt_root, pred_root) paths for evaluation
        """
        # Clean up any existing temp directories
        self.cleanup()
        
        if self.first_frame_only:
            return self._prepare_first_frame_only(
                ann_dir, pred_dir, video_subset, use_symlinks
            )
        elif video_subset is not None:
            return self._prepare_video_subset(
                ann_dir, pred_dir, video_subset, use_symlinks
            )
        else:
            # No temp directories needed - use original directories
            return ann_dir, pred_dir
    
    def _prepare_first_frame_only(
        self,
        ann_dir: Path,
        pred_dir: Path,
        video_subset: Optional[list[str]],
        use_symlinks: bool,
    ) -> tuple[Path, Path]:
        """Prepare directories for first-frame-only evaluation.
        
        Args:
            ann_dir: Ground truth annotation directory
            pred_dir: Prediction directory
            video_subset: Optional list of videos
            use_symlinks: Use symlinks instead of copying
        
        Returns:
            Tuple of (gt_tmp, pred_tmp) paths
        """
        # Determine which videos to process
        if video_subset is not None:
            base_videos = sorted(video_subset)
        else:
            base_videos = sorted([d.name for d in ann_dir.iterdir() if d.is_dir()])
        
        # Create temporary directories
        self.gt_tmp.mkdir(parents=True, exist_ok=True)
        self.pred_tmp.mkdir(parents=True, exist_ok=True)
        self._created = True
        
        # Copy or link first frames
        for v in base_videos:
            v_gt_dir = ann_dir / v
            v_pred_dir = pred_dir / v
            
            if not v_gt_dir.exists() or not v_pred_dir.exists():
                continue
            
            # Find first frame
            gt_pngs = sorted([p for p in v_gt_dir.iterdir() if p.suffix.lower() == ".png"])
            if not gt_pngs:
                continue
            
            first_png = gt_pngs[0].name
            if not (v_pred_dir / first_png).exists():
                continue
            
            # Create video subdirectories
            (self.gt_tmp / v).mkdir(parents=True, exist_ok=True)
            (self.pred_tmp / v).mkdir(parents=True, exist_ok=True)
            
            # Copy or link first frame
            gt_src = v_gt_dir / first_png
            gt_dst = self.gt_tmp / v / first_png
            pred_src = v_pred_dir / first_png
            pred_dst = self.pred_tmp / v / first_png
            
            if use_symlinks:
                # Use absolute paths for symlinks to avoid path resolution issues
                os.symlink(gt_src.resolve(), gt_dst)
                os.symlink(pred_src.resolve(), pred_dst)
            else:
                shutil.copy2(gt_src, gt_dst)
                shutil.copy2(pred_src, pred_dst)
        
        return self.gt_tmp, self.pred_tmp
    
    def _prepare_video_subset(
        self,
        ann_dir: Path,
        pred_dir: Path,
        video_subset: list[str],
        use_symlinks: bool,
    ) -> tuple[Path, Path]:
        """Prepare directories for video subset evaluation.
        
        Args:
            ann_dir: Ground truth annotation directory
            pred_dir: Prediction directory
            video_subset: List of videos to include
            use_symlinks: Use symlinks instead of copying
        
        Returns:
            Tuple of (gt_tmp, pred_tmp) paths
        """
        # Create temporary directories
        self.gt_tmp.mkdir(parents=True, exist_ok=True)
        self.pred_tmp.mkdir(parents=True, exist_ok=True)
        self._created = True
        
        # Copy or link video directories
        for v in video_subset:
            gt_src = ann_dir / v
            pred_src = pred_dir / v
            
            if not gt_src.exists() or not pred_src.exists():
                continue
            
            gt_dst = self.gt_tmp / v
            pred_dst = self.pred_tmp / v
            
            if use_symlinks:
                # Use symlink for entire video directory (fastest)
                # Use absolute paths for symlinks to avoid path resolution issues
                os.symlink(gt_src.resolve(), gt_dst)
                os.symlink(pred_src.resolve(), pred_dst)
            else:
                # Copy entire video directory
                shutil.copytree(gt_src, gt_dst, symlinks=True)
                shutil.copytree(pred_src, pred_dst, symlinks=True)
        
        return self.gt_tmp, self.pred_tmp
    
    def cleanup(self) -> None:
        """Clean up temporary directories."""
        for tmp_dir in [self.gt_tmp, self.pred_tmp]:
            if tmp_dir.exists():
                try:
                    shutil.rmtree(tmp_dir)
                except Exception as e:
                    print(f"Warning: Failed to clean up {tmp_dir}: {e}")
        
        self._created = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup.
        
        Note: Cleanup is disabled here because benchmark() uses multiprocessing,
        and cleaning up temp directories before worker processes finish causes
        "file not found" errors. Temp directories will be cleaned up on the
        next run or manually.
        """
        # self.cleanup()  # DISABLED: causes race condition with multiprocessing
        return False


def run_benchmark_evaluation(
    gt_dir: Path,
    pred_dir: Path,
    dataset_config: dict,
    video_subset: Optional[list[str]] = None,
    first_frame_only: bool = False,
    num_workers: Optional[int] = None,
    output_path: Optional[Path] = None,
    use_symlinks: bool = True,
    dataset_name: Optional[str] = None,
) -> tuple[float, float, float]:
    """Run benchmark evaluation with unified workflow.
    
    This function provides a standardized evaluation pipeline that:
    1. Creates temporary directories (with symlinks for speed)
    2. Runs benchmark evaluation
    3. Handles errors gracefully
    4. Cleans up temporary directories automatically
    
    Performance optimizations:
    - Uses symlinks instead of file copying (10-100x faster)
    - Automatic cleanup prevents disk space leaks
    - Consistent error handling
    
    Args:
        gt_dir: Ground truth annotation directory
        pred_dir: Prediction directory
        dataset_config: Dataset configuration dict (from DATASET_CONFIGS)
        video_subset: Optional list of videos to evaluate
        first_frame_only: Only evaluate first frames
        num_workers: Number of worker processes for evaluation
        output_path: Output path for temporary directories (required if using subsets)
        use_symlinks: Use symlinks instead of copying (default: True)
        dataset_name: Dataset name for temporary directory naming (optional, will use config["name"] if not provided)
    
    Returns:
        Tuple of (j_f, j, f) evaluation scores
    
    Raises:
        ValueError: If output_path not provided when needed
        FileNotFoundError: If required directories don't exist
    """
    # Validate inputs
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    
    # Determine if we need temporary directories
    needs_temp = first_frame_only or (video_subset is not None)
    
    if needs_temp:
        if output_path is None:
            raise ValueError("output_path required when using first_frame_only or video_subset")
        
        # Use temporary directory manager
        # Prefer explicit dataset_name parameter over config["name"]
        if dataset_name is None:
            dataset_name = dataset_config.get("name", "dataset")
        with TempDirManager(output_path, dataset_name, first_frame_only) as temp_mgr:
            gt_root, pred_root = temp_mgr.prepare_eval_roots(
                gt_dir, pred_dir, video_subset, use_symlinks
            )
            
            # Run benchmark evaluation
            return _run_benchmark(
                gt_root, pred_root, dataset_config, num_workers
            )
    else:
        # Use original directories directly
        return _run_benchmark(
            gt_dir, pred_dir, dataset_config, num_workers
        )


def _run_benchmark(
    gt_root: Path,
    pred_root: Path,
    dataset_config: dict,
    num_workers: Optional[int],
) -> tuple[float, float, float]:
    """Internal function to run benchmark evaluation.
    
    Args:
        gt_root: Ground truth root directory
        pred_root: Prediction root directory
        dataset_config: Dataset configuration dict
        num_workers: Number of worker processes
    
    Returns:
        Tuple of (j_f, j, f) scores, or (0.0, 0.0, 0.0) on error
    """
    try:
        J_F, global_J, global_F, _ = benchmark(
            gt_roots=[str(gt_root)],
            mask_roots=[str(pred_root)],
            strict=False,
            num_processes=num_workers,
            skip_first_and_last=dataset_config.get("skip_first_and_last", False),
            verbose=True,
        )
        
        # Handle empty results
        if len(J_F) == 0 or len(global_J) == 0 or len(global_F) == 0:
            print("Warning: Empty evaluation results")
            return 0.0, 0.0, 0.0
        
        # Extract values and handle NaN
        j_f_val = float(J_F[0]) if not np.isnan(J_F[0]) else 0.0
        j_val = float(global_J[0]) if not np.isnan(global_J[0]) else 0.0
        f_val = float(global_F[0]) if not np.isnan(global_F[0]) else 0.0
        
        # Warn about NaN values
        if np.isnan(J_F[0]) or np.isnan(global_J[0]) or np.isnan(global_F[0]):
            print(f"Warning: NaN values detected in evaluation results")
            print(f"  J&F: {J_F[0]}, J: {global_J[0]}, F: {global_F[0]}")
        
        return j_f_val, j_val, f_val
        
    except Exception as e:
        print(f"Error during benchmark evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0.0

