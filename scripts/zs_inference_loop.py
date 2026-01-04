#!/usr/bin/env python
"""
Unified inference loop for zero-shot video segmentation evaluation.

This module extracts the common video iteration logic shared across:
- zero_shot_multi_dataset_sam_bndl.py
- zero_shot_multi_dataset_uctta.py
- zero_shot_multi_dataset_ur_ern.py

Each method provides callbacks for method-specific processing while using
the same core loop structure.

Usage:
    from zs_inference_loop import run_video_inference_loop
    
    statistics = run_video_inference_loop(
        predictor=predictor,
        jpeg_dir=jpeg_dir,
        ann_dir=ann_dir,
        out_dir=out_dir,
        video_names=video_names,
        method_name="BNDL_AUE",
        on_frame_processed=my_frame_callback,
        ...
    )
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Shared utilities
from zero_shot_utils import (
    load_first_frame_mask,
    threshold_mask_logits,
    select_top_objects_by_area,
    cleanup_gpu_memory,
    select_single_mask_from_multimask,
)
from checkpoint_manager import CheckpointManager
from prompt_generation import generate_click_prompts
from prompt_loader import load_reused_prompts, apply_reused_prompts
from tools.vos_inference import DAVIS_PALETTE, save_masks_to_dir


# Type aliases for callbacks
FrameCallback = Callable[
    [int, list[int], torch.Tensor, dict, Any],  # f_idx, obj_ids, logits, state, predictor
    Optional[dict[str, Any]]  # Optional per-frame statistics
]


def run_video_inference_loop(
    predictor: Any,
    jpeg_dir: Path,
    ann_dir: Path,
    out_dir: Path,
    video_names: list[str] | None = None,
    method_name: str = "SAM",
    # Common parameters
    score_thresh: float = 0.0,
    max_objects: int | None = None,
    first_frame_only: bool = False,
    collect_statistics: bool = True,
    # Prompt parameters
    reuse_prompts_root: Path | None = None,
    dataset_name: str | None = None,
    click_protocol: str = "3click",
    min_click_dist: float = 12.0,
    seed: int | None = 0,
    # Checkpointing
    checkpoint_interval: int = 10,
    downsample_max_samples: int = 100000,
    # Callbacks for method-specific behavior
    on_video_start: Callable[[Any, dict, str], None] | None = None,
    on_frame_processed: FrameCallback | None = None,
    on_video_complete: Callable[[str, dict], None] | None = None,
    # State init parameters
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
) -> dict[str, Any]:
    """
    Generic video inference loop for zero-shot evaluation.
    
    This function handles the common workflow:
    1. Iterate over videos
    2. Initialize predictor state
    3. Load first frame GT and apply prompts
    4. Propagate through video frames
    5. Save masks and collect statistics
    6. Manage checkpoints
    
    Args:
        predictor: SAM2 video predictor instance
        jpeg_dir: Directory containing video frames (JPEGs)
        ann_dir: Directory containing annotations (PNGs)
        out_dir: Output directory for predictions
        video_names: Optional list of video names to process
        method_name: Name of the method (for logging)
        score_thresh: Mask threshold
        max_objects: Maximum objects per video
        first_frame_only: Only process first frame
        collect_statistics: Whether to collect statistics
        reuse_prompts_root: Root for reusing prompts from SAM run
        dataset_name: Dataset name for logging
        click_protocol: Click protocol (1click, 3click, 5click)
        min_click_dist: Minimum distance between clicks
        seed: Random seed
        checkpoint_interval: Videos between checkpoints
        downsample_max_samples: Max samples for downsampling
        on_video_start: Callback(predictor, state, vid) at video start
        on_frame_processed: Callback(f_idx, obj_ids, logits, state, predictor)
                           for per-frame processing, returns optional stats dict
        on_video_complete: Callback(vid, video_stats) at video completion
        offload_video_to_cpu: Offload video to CPU
        offload_state_to_cpu: Offload state to CPU
    
    Returns:
        Dictionary of collected statistics (empty if collect_statistics=False)
    """
    # Get video list
    if video_names is None:
        video_names = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    else:
        video_names = sorted(set(video_names))
    
    print(f"{method_name} inference on {len(video_names)} videos")
    
    # Initialize statistics collection
    all_statistics: dict[str, Any] = {} if collect_statistics else {}
    
    # Setup checkpoint manager
    checkpoint_mgr = CheckpointManager(
        output_dir=out_dir.parent,
        dataset_name=dataset_name or method_name.lower(),
        checkpoint_type="eval",
        interval=checkpoint_interval,
    ) if collect_statistics else None
    
    # Optional seeding
    if seed is not None:
        np.random.seed(int(seed))
    
    for v_idx, vid in enumerate(video_names, 1):
        # Progress logging (format matches parallel_compare.py expectations)
        print(f"\n{'=' * 60}")
        print(f"📹 Processing video [{v_idx:03}/{len(video_names)}]: {vid}")
        print(f"   Progress: {v_idx}/{len(video_names)} ({100.0 * v_idx / len(video_names):.1f}%)")
        print(f"{'=' * 60}")
        
        video_dir = jpeg_dir / vid
        frame_names = sorted(
            [p.stem for p in video_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]],
            key=lambda x: int(x),
        )
        
        # Check if already processed (resumability)
        completion_marker = out_dir / vid / "query_prompts.json"
        if completion_marker.exists():
            print(f"Skipping video {vid} - already processed")
            continue
        
        # Initialize predictor state
        max_frames_to_load = 1 if first_frame_only else None
        state = predictor.init_state(
            str(video_dir),
            max_frames=max_frames_to_load,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        H, W = state["video_height"], state["video_width"]
        
        # Load first frame GT
        first_mask_np = load_first_frame_mask(ann_dir, vid, frame_names)
        if first_mask_np is None:
            continue
        
        obj_ids = select_top_objects_by_area(first_mask_np, max_objects or 10**9)
        if len(obj_ids) == 0:
            print(f"Warning: No objects found in first frame of video {vid}")
            continue
        
        print(f"Processing {len(obj_ids)} objects in video {vid}: {obj_ids}")
        
        # Load reused prompts
        prompts_json = load_reused_prompts(reuse_prompts_root, dataset_name, vid)
        if prompts_json:
            print(f"Loaded reused prompts for video {vid}")
        
        # Call video start callback
        if on_video_start is not None:
            on_video_start(predictor, state, vid)
        
        # Apply prompts for each object
        obj_points: dict[int, list[tuple[int, int, int]]] = {}
        for obj_id in obj_ids:
            gt_bool = first_mask_np == obj_id
            if not np.any(gt_bool):
                continue
            
            # Clear GPU memory for many objects
            if torch.cuda.is_available() and len(obj_ids) > 10:
                torch.cuda.empty_cache()
            
            # Try reused prompts first
            prompt_applied = False
            if prompts_json and obj_id in prompts_json:
                prompt_applied = apply_reused_prompts(predictor, state, obj_id, prompts_json[obj_id])
                if prompt_applied:
                    clicks = prompts_json[obj_id].get("clicks", [])
                    obj_points[obj_id] = [
                        (int(c["xy"][0]), int(c["xy"][1]), int(c.get("label", 1)))
                        for c in clicks if "xy" in c
                    ]
            
            if not prompt_applied:
                # Generate new prompts
                used_pts, used_labels = generate_click_prompts(
                    predictor, state,
                    frame_idx=0,
                    obj_id=obj_id,
                    gt_bool=gt_bool,
                    first_frame_mask_np=first_mask_np,
                    score_thresh=score_thresh,
                    click_protocol=click_protocol,
                    min_click_dist=float(min_click_dist),
                )
                obj_points[obj_id] = [
                    (int(x), int(y), int(label))
                    for (x, y), label in zip(used_pts, used_labels, strict=True)
                ]
        
        # Propagate through video
        max_frames = 0 if first_frame_only else None
        video_segments: dict[int, dict[int, np.ndarray]] = {}
        video_statistics: dict[str, Any] = {}
        
        for f_idx, out_obj_ids, out_logits in predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=max_frames
        ):
            # Safety check for first_frame_only
            if first_frame_only and f_idx > 0:
                del out_logits
                break
            
            # Process masks
            seg: dict[int, np.ndarray] = {}
            for i, oid in enumerate(out_obj_ids):
                mask_logits = out_logits[i]
                
                # Handle multimask output
                mask_logits = select_single_mask_from_multimask(mask_logits)
                
                # Resize if needed
                if tuple(mask_logits.shape[-2:]) != (H, W):
                    mask_logits = F.interpolate(
                        mask_logits.unsqueeze(0).unsqueeze(0),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                
                seg[oid] = threshold_mask_logits(mask_logits, score_thresh)
            
            video_segments[f_idx] = seg
            
            # Clear GPU memory for many objects
            if torch.cuda.is_available() and len(out_obj_ids) > 20:
                torch.cuda.empty_cache()
            
            # Call frame callback for method-specific processing
            if on_frame_processed is not None and collect_statistics:
                frame_stats = on_frame_processed(
                    f_idx, list(out_obj_ids), out_logits, state, predictor
                )
                if frame_stats:
                    video_statistics.update(frame_stats)
            
            # Clean up logits
            del out_logits
        
        # Save masks (parallel I/O)
        _save_video_masks_parallel(
            video_segments, vid, frame_names, out_dir, H, W
        )
        video_segments.clear()
        
        # Save prompts
        _save_prompts_json(out_dir, vid, obj_points, click_protocol)
        
        # Call video complete callback
        if on_video_complete is not None:
            on_video_complete(vid, video_statistics)
        
        # Merge video statistics
        if collect_statistics and video_statistics:
            all_statistics.update(video_statistics)
        
        # Checkpoint if needed
        if checkpoint_mgr and checkpoint_mgr.should_checkpoint(v_idx):
            checkpoint_file = checkpoint_mgr.save_checkpoint(all_statistics, v_idx)
            print(f"💾 Saved checkpoint ({v_idx}/{len(video_names)}) to {checkpoint_file.name}")
            all_statistics.clear()
            CheckpointManager.force_memory_cleanup()
        
        # Cleanup GPU memory
        cleanup_gpu_memory(predictor, state)
        print(f"✓ Video {vid} completed ({v_idx}/{len(video_names)})")
    
    # Save final statistics
    if checkpoint_mgr and all_statistics:
        checkpoint_file = checkpoint_mgr.save_checkpoint(all_statistics, len(video_names))
        print(f"💾 Saved final checkpoint to {checkpoint_file.name}")
        all_statistics.clear()
    
    # Merge all checkpoints
    if checkpoint_mgr:
        merged = checkpoint_mgr.merge_checkpoints()
        return merged
    
    return all_statistics


def _save_video_masks_parallel(
    video_segments: dict[int, dict[int, np.ndarray]],
    vid: str,
    frame_names: list[str],
    out_dir: Path,
    H: int,
    W: int,
    max_workers: int = 4,
) -> None:
    """Save video masks using parallel I/O."""
    def _save_single(f_idx: int, seg: dict[int, np.ndarray]) -> int:
        save_masks_to_dir(
            output_mask_dir=str(out_dir),
            video_name=vid,
            frame_name=frame_names[f_idx],
            per_obj_output_mask=seg,
            height=H,
            width=W,
            per_obj_png_file=False,
            output_palette=DAVIS_PALETTE,
        )
        return f_idx
    
    n_workers = min(max_workers, len(video_segments))
    if n_workers > 0:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_save_single, f_idx, seg)
                for f_idx, seg in video_segments.items()
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Warning: Failed to save mask: {e}")


def _save_prompts_json(
    out_dir: Path,
    vid: str,
    obj_points: dict[int, list[tuple[int, int, int]]],
    click_protocol: str,
) -> None:
    """Save prompts to JSON file."""
    import json
    
    prompt_file = out_dir / vid / "query_prompts.json"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    prompt_data = {
        "click_protocol": click_protocol,
        "objects": {
            str(obj_id): {
                "clicks": [
                    {"xy": [x, y], "label": label}
                    for x, y, label in points
                ]
            }
            for obj_id, points in obj_points.items()
        }
    }
    
    with open(prompt_file, "w") as f:
        json.dump(prompt_data, f, indent=2)


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("Inference Loop Module Tests")
    print("=" * 60)
    
    print("\n1. Module imports:")
    print("  ✓ run_video_inference_loop")
    print("  ✓ _save_video_masks_parallel")
    print("  ✓ _save_prompts_json")
    
    print("\n✓ All imports successful!")
