#!/usr/bin/env python
"""Shared utility functions for zero-shot evaluation scripts.

This module consolidates common functionality used across multiple
zero-shot evaluation scripts to reduce code duplication and improve maintainability.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor
from tools.vos_inference import DAVIS_PALETTE, save_masks_to_dir


# ============================================================================
# Video and Mask Loading Utilities
# ============================================================================

def load_first_frame_mask(
    ann_dir: Path, vid: str, frame_names: list[str]
) -> Optional[np.ndarray]:
    """Load the first frame annotation mask for a video.
    
    Args:
        ann_dir: Directory containing annotations
        vid: Video name
        frame_names: List of frame names (sorted)
    
    Returns:
        First frame mask as numpy array, or None if not found
    """
    first_mask_path = ann_dir / vid / f"{frame_names[0]}.png"
    if not first_mask_path.exists():
        print(f"Warning: First frame annotation not found: {first_mask_path}")
        return None
    return np.array(Image.open(first_mask_path))


def get_video_subset(jpeg_dir: Path, limit: Optional[int]) -> Optional[list[str]]:
    """Get a limited subset of videos from a directory.
    
    Args:
        jpeg_dir: Directory containing video frames
        limit: Maximum number of videos to return, or None for all
    
    Returns:
        List of video names (limited), or None if no limit specified
    """
    if limit is None or not jpeg_dir.exists():
        return None
    all_videos = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    return all_videos[:limit]


def threshold_mask_logits(
    mask_logits: torch.Tensor, score_thresh: float
) -> np.ndarray:
    """Apply threshold to mask logits and convert to boolean numpy array.
    
    Args:
        mask_logits: Mask logits tensor [H, W] or [1, H, W]
        score_thresh: Threshold value
    
    Returns:
        Boolean numpy array
    """
    if mask_logits.ndim == 3:
        mask_logits = mask_logits.squeeze(0)
    return (mask_logits > score_thresh).detach().cpu().numpy().astype(bool)


def select_top_objects_by_area(
    mask_np: np.ndarray, max_objects: int
) -> list[int]:
    """Select top N objects from mask by area (largest first).
    
    Args:
        mask_np: Mask array with object IDs
        max_objects: Maximum number of objects to return
    
    Returns:
        List of object IDs (sorted by area, descending)
    """
    all_ids = [oid for oid in np.unique(mask_np) if oid > 0]
    if (max_objects is None) or (len(all_ids) <= max_objects):
        return all_ids
    
    # Calculate areas and sort
    areas = {oid: int((mask_np == oid).sum()) for oid in all_ids}
    sorted_objs = sorted(areas.items(), key=lambda x: x[1], reverse=True)
    return [oid for oid, _ in sorted_objs[:max_objects]]


# ============================================================================
# GPU Memory Management
# ============================================================================

def cleanup_gpu_memory(predictor=None, state: dict | None = None) -> None:
    """Clean up GPU memory by resetting state and emptying cache.
    
    Args:
        predictor: SAM2 predictor instance (optional)
        state: Predictor state to reset (optional)
    """
    # Reset predictor state if provided
    if predictor is not None and state is not None:
        try:
            predictor.reset_state(state)
        except Exception as e:
            print(f"Warning: Failed to reset predictor state: {e}")
    
    # Force garbage collection
    gc.collect()
    
    # Empty CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Predictor Building Utilities
# ============================================================================

def build_predictor_with_defaults(
    config_file: str,
    checkpoint_path: str,
    device: str = "cuda",
    enable_multimask: bool = True,
    multimask_min_pts: int = 1,
    multimask_max_pts: int = 2,
    multimask_for_tracking: bool = False,
    extra_overrides: list[str] | None = None,
) -> object:
    """Build SAM2 video predictor with standard configuration.
    
    This function consolidates the common predictor building logic used
    across all zero-shot evaluation scripts.
    
    Args:
        config_file: Path to SAM2 config file
        checkpoint_path: Path to checkpoint file
        device: Device to use ('cuda' or 'cpu')
        enable_multimask: Enable multimask output
        multimask_min_pts: Minimum points to trigger multimask
        multimask_max_pts: Maximum points to trigger multimask
        multimask_for_tracking: Enable multimask during tracking
        extra_overrides: Additional Hydra overrides
    
    Returns:
        SAM2 video predictor instance
    """
    hydra_overrides_extra = []
    
    if enable_multimask:
        hydra_overrides_extra += [
            "++model.multimask_output_in_sam=true",
            f"++model.multimask_min_pt_num={multimask_min_pts}",
            f"++model.multimask_max_pt_num={multimask_max_pts}",
        ]
        if multimask_for_tracking:
            hydra_overrides_extra.append("++model.multimask_output_for_tracking=true")
    
    # Add any extra overrides
    if extra_overrides:
        hydra_overrides_extra.extend(extra_overrides)
    
    predictor = build_sam2_video_predictor(
        config_file=config_file,
        ckpt_path=checkpoint_path,
        device=device,
        hydra_overrides_extra=hydra_overrides_extra,
    )
    
    return predictor


# ============================================================================
# Mask Processing Utilities
# ============================================================================

def resize_mask_logits(
    mask_logits: torch.Tensor, target_size: tuple[int, int]
) -> torch.Tensor:
    """Resize mask logits to target size using bilinear interpolation.
    
    Args:
        mask_logits: Mask logits tensor [H, W] or [1, H, W] or [K, H, W]
        target_size: Target (H, W) size
    
    Returns:
        Resized mask logits tensor
    """
    if tuple(mask_logits.shape[-2:]) == target_size:
        return mask_logits
    
    # Add batch and channel dimensions if needed
    needs_unsqueeze = mask_logits.ndim == 2
    if needs_unsqueeze:
        mask_logits = mask_logits.unsqueeze(0).unsqueeze(0)
    elif mask_logits.ndim == 3:
        mask_logits = mask_logits.unsqueeze(0)
    
    # Resize
    resized = F.interpolate(
        mask_logits,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    
    # Remove added dimensions
    if needs_unsqueeze:
        resized = resized.squeeze(0).squeeze(0)
    else:
        resized = resized.squeeze(0)
    
    return resized


def select_single_mask_from_multimask(
    mask_logits: torch.Tensor, mask_index: int = 0
) -> torch.Tensor:
    """Select a single mask from multimask output.
    
    SAM-2's multimask output produces K masks. This function selects
    one mask (typically mask 0, which is the single-mask token output).
    
    Args:
        mask_logits: Mask logits [K, H, W] or [1, H, W] or [H, W]
        mask_index: Index of mask to select (default: 0)
    
    Returns:
        Single mask logits [H, W]
    """
    if mask_logits.ndim == 3:
        if mask_logits.shape[0] == 1:
            # Single mask: squeeze it
            return mask_logits.squeeze(0)
        elif mask_logits.shape[0] > 1:
            # Multiple masks: select specified index
            return mask_logits[mask_index]
    
    # Already 2D or other format
    return mask_logits if mask_logits.ndim == 2 else mask_logits.squeeze()

    # Already 2D or other format
    return mask_logits if mask_logits.ndim == 2 else mask_logits.squeeze()


def save_single_mask_helper(f_idx, seg, vid, frame_names, out_dir, H, W, per_obj_png_file=False):
    """Helper function for parallel mask saving.
    
    Moved from local scope to module level for better pickling and code organization.
    """
    save_masks_to_dir(
        output_mask_dir=str(out_dir),
        video_name=vid,
        frame_name=frame_names[f_idx],
        per_obj_output_mask=seg,
        height=H,
        width=W,
        per_obj_png_file=per_obj_png_file,
        output_palette=DAVIS_PALETTE,
    )
    return f_idx
