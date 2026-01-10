#!/usr/bin/env python
"""
Zero-shot evaluation using SAM2ImagePredictor for better alignment with training.

Key differences from zero_shot_multi_dataset_sam_bndl.py:
1. Uses SAM2ImagePredictor instead of SAM2VideoPredictor
2. Selects best mask using iou_predictions (like training)
3. Better alignment with training dice loss calculation

Training logic (loss_fns.py L222-224):
    best_iou_inds = torch.argmax(ious, dim=-1)
    batch_inds = torch.arange(src_masks.size(0), device=src_masks.device)
    best_mask = src_masks[batch_inds, best_iou_inds]  # [N, H, W]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import matplotlib

matplotlib.use("Agg")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------  SAM-2 Image Predictor ----------
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ----------  Shared utilities ----------
from zero_shot_utils import (
    load_first_frame_mask,
    save_single_mask_helper,
    select_top_objects_by_area,
)
from bndl_eval_utils import (
    finalize_bndl_evaluation,
    setup_bndl_collection,
)
from checkpoint_manager import CheckpointManager
from downsampling_utils import downsample_statistics_pavpu

# ---------- Dataset Configurations ----------
from dataset_configs import DATASET_CONFIGS, DEFAULT_DATASETS

# ---------- Unified click prompt generator ----------
from prompt_loader import load_reused_prompts, save_prompts_to_json
from prompt_utils import sample_error_click, sample_pos_neg


def iterative_predict_with_clicks(
    predictor: SAM2ImagePredictor,
    gt_bool: np.ndarray,
    first_frame_mask_np: np.ndarray,
    obj_id: int,
    score_thresh: float,
    num_correction_clicks: int = 2,
    multimask_output: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[int], float]:
    """
    Iteratively generate clicks and predict masks, strictly matching training flow.

    Training flow (sam2.py _iter_correct_pt_sampling):
    1. Initial positive click → predict (no mask_input) → mask
    2. Sample correction click from error region → predict (with mask_input) → new mask
    3. Repeat for num_correction_clicks
    4. Return final mask (from last predict)

    This function unifies "prompt generation" and "inference" into a single iterative process.

    Args:
        predictor: SAM2ImagePredictor with image already set
        gt_bool: Boolean mask of ground truth for this object [H, W]
        first_frame_mask_np: Full first frame mask with all objects [H, W]
        obj_id: Object ID
        score_thresh: Threshold for mask logits
        num_correction_clicks: Number of correction clicks after initial positive click.
            Total clicks = 1 (initial) + num_correction_clicks.
            E.g., num_correction_clicks=2 means 3 total clicks (1 initial + 2 corrections).
        multimask_output: Whether to use multimask output

    Returns:
        (best_mask, best_mask_logits, used_pts, used_labels, best_iou):
            - best_mask: Final binary mask [H, W]
            - best_mask_logits: Final mask logits [H, W]
            - used_pts: List of (x, y) coordinates
            - used_labels: List of labels (1=pos, 0=neg)
            - best_iou: IoU prediction of final mask
    """
    used_pts: list[tuple[int, int]] = []
    used_labels: list[int] = []
    total_clicks = 1 + num_correction_clicks  # 1 initial + N corrections

    # First positive click using SAM2's sample_pos_neg
    try:
        pos_xy, neg_xy = sample_pos_neg(gt_bool, full_mask=first_frame_mask_np, current_obj_id=obj_id)
        cx, cy = int(pos_xy[0]), int(pos_xy[1])
    except Exception:
        ys, xs = np.where(gt_bool)
        cx = int(xs.mean()) if xs.size else 0
        cy = int(ys.mean()) if ys.size else 0

    used_pts.append((cx, cy))
    used_labels.append(1)

    # Track previous low_res_mask for iterative refinement (matches training!)
    prev_low_res_mask: np.ndarray | None = None

    # Final outputs (will be updated in each iteration)
    best_mask: np.ndarray | None = None
    best_mask_logits: np.ndarray | None = None
    best_iou: float = 0.0

    # Iterative prediction and click sampling (matches training exactly!)
    for click_step in range(total_clicks):
        # Prepare prompts as arrays (using clicks 0..click_step)
        point_coords = np.array(used_pts[: click_step + 1], dtype=np.float32)
        point_labels = np.array(used_labels[: click_step + 1], dtype=np.int32)

        # Run prediction with current prompts
        # Training code (sam2.py L743): mask_inputs = low_res_masks
        masks, iou_predictions, low_res_masks = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=prev_low_res_mask,  # Feed previous mask logits!
            multimask_output=multimask_output,
            return_logits=True,
        )

        # Select best mask by IoU (aligns with training!)
        if multimask_output and masks.shape[0] > 1:
            best_idx = int(np.argmax(iou_predictions))
        else:
            best_idx = 0

        best_mask_logits = masks[best_idx]  # [H, W]
        best_low_res = low_res_masks[best_idx]  # [256, 256]
        best_iou = float(iou_predictions[best_idx])

        # Update prev_low_res_mask for next iteration (shape: 1xHxW)
        prev_low_res_mask = best_low_res[np.newaxis, ...]

        # Check if we need to sample more clicks
        if click_step + 1 >= total_clicks:
            # This was the last click, we're done
            break

        # Sample next error click based on current prediction
        pred_bool = (best_mask_logits > score_thresh).astype(bool)
        nx, ny_label = sample_error_click(gt_bool, pred_bool)
        nx_pt = (int(nx[0]), int(nx[1]))

        used_pts.append(nx_pt)
        used_labels.append(int(ny_label))

    # Compute final binary mask (bool type required by save_masks_to_dir)
    assert best_mask_logits is not None
    best_mask = (best_mask_logits > score_thresh).astype(bool)

    return best_mask, best_mask_logits, used_pts, used_labels, best_iou


def build_image_predictor(
    cfg_file: str,
    ckpt: str,
    device: str = "cuda",
    multimask: bool = True,
) -> SAM2ImagePredictor:
    """Build SAM2ImagePredictor with consistent overrides."""
    hydra_overrides_extra = [
        # Set multimask_output_in_sam based on parameter
        f"++model.multimask_output_in_sam={str(multimask).lower()}",
    ]

    # Build base model with SAM2 (not video predictor)
    model = build_sam2(
        config_file=cfg_file,
        ckpt_path=ckpt,
        device=device,
        hydra_overrides_extra=hydra_overrides_extra,
    )

    # Wrap in SAM2ImagePredictor
    predictor = SAM2ImagePredictor(
        sam_model=model,
        mask_threshold=0.0,  # Return logits by default
    )

    return predictor


def select_best_mask_by_iou(
    masks: np.ndarray | torch.Tensor,
    iou_predictions: np.ndarray | torch.Tensor,
    return_all: bool = False,
) -> tuple[np.ndarray | torch.Tensor, int]:
    """
    Select the best mask based on predicted IoU scores.

    This aligns with training logic in loss_fns.py L222-224:
        best_iou_inds = torch.argmax(ious, dim=-1)
        best_mask = src_masks[batch_inds, best_iou_inds]

    Args:
        masks: Shape [K, H, W] or [N, K, H, W] - K masks per sample
        iou_predictions: Shape [K] or [N, K] - IoU prediction for each mask
        return_all: If True, also return all IoU scores

    Returns:
        best_mask: Shape [H, W] or [N, H, W]
        best_idx: Index of selected mask
    """
    if isinstance(masks, torch.Tensor):
        if masks.dim() == 3:  # [K, H, W]
            best_idx = int(torch.argmax(iou_predictions).item())
            return masks[best_idx], best_idx
        else:  # [N, K, H, W]
            best_iou_inds = torch.argmax(iou_predictions, dim=-1)
            batch_inds = torch.arange(masks.size(0), device=masks.device)
            return masks[batch_inds, best_iou_inds], best_iou_inds
    else:  # numpy
        if masks.ndim == 3:  # [K, H, W]
            best_idx = int(np.argmax(iou_predictions))
            return masks[best_idx], best_idx
        else:  # [N, K, H, W]
            best_iou_inds = np.argmax(iou_predictions, axis=-1)
            batch_inds = np.arange(masks.shape[0])
            return masks[batch_inds, best_iou_inds], best_iou_inds


def get_bndl_outputs_from_predictor(predictor: SAM2ImagePredictor) -> dict | None:
    """
    Extract BNDL outputs from SAM2ImagePredictor's last prediction.

    The BNDL outputs are stored in the mask decoder's forward pass.
    We need to access them through the model's internal state.
    """
    if not hasattr(predictor, "model"):
        return None

    model = predictor.model

    # Check if BNDL is enabled
    if not hasattr(model, "sam_mask_decoder"):
        return None

    mask_decoder = model.sam_mask_decoder

    # Check for BNDL projector
    if not hasattr(mask_decoder, "pixel_bndl_projector"):
        return None

    # The BNDL outputs should be in the last aux_outputs
    # We need to modify _predict to capture aux_outputs
    # For now, return None - we'll need to modify the predictor
    return None


@torch.inference_mode()
def inference_single_image_with_bndl(
    predictor: SAM2ImagePredictor,
    image_path: Path,
    gt_mask_path: Path | None,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    score_thresh: float = 0.0,
    multimask_output: bool = True,
    return_logits: bool = True,
) -> dict[str, Any]:
    """
    Run single image inference with BNDL using SAM2ImagePredictor.

    Args:
        predictor: SAM2ImagePredictor instance
        image_path: Path to input image
        gt_mask_path: Optional path to ground truth mask
        point_coords: Nx2 array of point coordinates
        point_labels: N array of point labels (1=foreground, 0=background)
        score_thresh: Threshold for mask binarization
        multimask_output: If True, return multiple masks and select best by IoU
        return_logits: If True, return raw logits instead of binary masks

    Returns:
        dict with:
            - mask: Best selected mask [H, W]
            - mask_logits: Logits of best mask [H, W]
            - iou_prediction: IoU score of selected mask
            - all_masks: All K masks [K, H, W] if multimask_output
            - all_ious: All K IoU predictions [K]
            - selected_idx: Index of selected mask
            - bndl_outputs: Optional BNDL outputs if available
    """
    # Load and set image
    image = np.array(Image.open(image_path).convert("RGB"))
    H, W = image.shape[:2]
    predictor.set_image(image)

    # Run prediction
    masks, iou_predictions, low_res_masks = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=multimask_output,
        return_logits=return_logits,
    )

    # Select best mask by IoU prediction (aligns with training)
    if multimask_output and masks.shape[0] > 1:
        best_mask, best_idx = select_best_mask_by_iou(masks, iou_predictions)
        best_iou = iou_predictions[best_idx]
        best_low_res = low_res_masks[best_idx]
    else:
        best_mask = masks[0] if masks.ndim == 3 else masks
        best_idx = 0
        best_iou = iou_predictions[0] if iou_predictions.ndim == 1 else iou_predictions
        best_low_res = low_res_masks[0] if low_res_masks.ndim == 3 else low_res_masks

    # Apply threshold if returning logits
    if return_logits:
        binary_mask = (best_mask > score_thresh).astype(np.uint8)
    else:
        binary_mask = best_mask.astype(np.uint8)

    result = {
        "mask": binary_mask,
        "mask_logits": best_mask,
        "iou_prediction": float(best_iou),
        "all_masks": masks,
        "all_ious": iou_predictions,
        "all_low_res": low_res_masks,
        "selected_idx": best_idx,
        "image_size": (H, W),
    }

    # Try to get BNDL outputs
    bndl_outputs = get_bndl_outputs_from_predictor(predictor)
    if bndl_outputs is not None:
        result["bndl_outputs"] = bndl_outputs

    return result


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def inference_with_bndl(
    predictor: SAM2ImagePredictor,
    jpeg_dir: Path,
    ann_dir: Path,
    out_dir: Path,
    score_thresh: float = 0.0,
    video_names: list[str] | None = None,
    save_bndl_vis: bool = False,
    vis_dir: Path | None = None,
    dataset_name: str = "unknown",
    collect_statistics: bool = True,
    eval_dir: Path | None = None,
    max_objects: int | None = None,
    reuse_prompts_root: Path | None = None,
    click_protocol: str = "3click",
    seed: int | None = 0,
    downsample_max_samples: int = 100000,
    max_vis_per_video: int = 2,
    save_vis_pdf: bool = True,
    bndl_sample_num: int = 20,
    save_masks: bool = True,
    multimask_output: bool = True,
):
    """
    Single-frame inference with BNDL using SAM2ImagePredictor.

    Strictly matches training click protocol:
    - Iterative prediction with mask_input feedback
    - Best mask selected by argmax(iou_predictions)
    """
    if video_names is None:
        video_names = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    else:
        video_names = sorted(set(video_names))

    print(f"BNDL ({click_protocol}) inference on {len(video_names)} videos")
    print(f"  multimask_output={multimask_output} (selects best mask by IoU prediction)")

    if seed is not None:
        np.random.seed(int(seed))

    if save_bndl_vis and vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    dataset_statistics, stats_checkpoint_mgr, eval_checkpoint_mgr, dataset_evaluator = setup_bndl_collection(collect_statistics, out_dir, dataset_name, eval_dir)
    total_frames_processed = 0

    for v_idx, vid in enumerate(video_names, 1):
        print(f"\n{'=' * 60}")
        print(f"📹 Processing [{v_idx:03}/{len(video_names)}]: {vid}")
        print(f"{'=' * 60}")

        video_dir = jpeg_dir / vid
        frame_names = sorted(
            [p.stem for p in video_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]],
            key=lambda x: int(x),
        )
        if not frame_names:
            print(f"Warning: No frames found for {vid}")
            continue

        # Skip if already processed
        if (out_dir / vid / "query_prompts.json").exists():
            print(f"Skipping {vid} - already processed")
            continue

        # Load first frame GT mask
        first_mask = load_first_frame_mask(ann_dir, vid, frame_names)
        if first_mask is None:
            continue

        H, W = first_mask.shape
        all_obj_ids = [oid for oid in np.unique(first_mask) if oid > 0]

        if len(all_obj_ids) == 0:
            print(f"Warning: No objects found in first frame of video {vid}")
            continue

        # Apply object limit
        obj_ids = select_top_objects_by_area(first_mask, max_objects if max_objects else 10**9)
        if max_objects and len(all_obj_ids) > len(obj_ids):
            print(f"Limited to {len(obj_ids)} largest objects (from {len(all_obj_ids)} total)")

        # Load reused prompts
        prompts_json = load_reused_prompts(reuse_prompts_root, dataset_name, vid)
        if prompts_json:
            print(f"Loaded reused prompts for video {vid}")

        print(f"Processing {len(obj_ids)} objects in video {vid}: {obj_ids}")

        video_statistics = {} if collect_statistics else None
        video_segments = {}
        obj_points: dict[int, list[tuple[int, int, int]]] = {}

        # Single-frame processing: only process first frame
        frame_name = frame_names[0]
        image_path = video_dir / f"{frame_name}.jpg"
        if not image_path.exists():
            image_path = video_dir / f"{frame_name}.jpeg"
        if not image_path.exists():
            print(f"Warning: Image not found for {vid}/{frame_name}")
            continue

        # Load image and set it
        image = np.array(Image.open(image_path).convert("RGB"))
        predictor.set_image(image)

        # Load GT mask for this frame (use first_mask directly)
        gt_mask_full = first_mask

        seg = {}
        num_correction = int(click_protocol.replace("click", "")) - 1 if click_protocol else 2

        for obj_id in obj_ids:
            gt_bool = gt_mask_full == obj_id
            if not np.any(gt_bool):
                continue

            # Check for cached prompts
            if prompts_json and obj_id in prompts_json:
                clicks = prompts_json[obj_id].get("clicks", [])
                if clicks:
                    obj_points[obj_id] = [(int(c["xy"][0]), int(c["xy"][1]), int(c.get("label", 1))) for c in clicks]
                    # Run iterative prediction with cached prompts
                    point_coords = np.array([[c["xy"][0], c["xy"][1]] for c in clicks], dtype=np.float32)
                    point_labels = np.array([c.get("label", 1) for c in clicks], dtype=np.int32)

                    prev_low_res_mask: np.ndarray | None = None
                    for click_step in range(len(clicks)):
                        step_coords = point_coords[: click_step + 1]
                        step_labels = point_labels[: click_step + 1]
                        masks, iou_predictions, low_res_masks = predictor.predict(
                            point_coords=step_coords,
                            point_labels=step_labels,
                            mask_input=prev_low_res_mask,
                            multimask_output=multimask_output,
                            return_logits=True,
                        )
                        best_idx = int(np.argmax(iou_predictions)) if multimask_output and masks.shape[0] > 1 else 0
                        prev_low_res_mask = low_res_masks[best_idx][np.newaxis, ...]

                    best_mask = masks[best_idx]
                    best_iou = float(iou_predictions[best_idx])
                    binary_mask = (best_mask > score_thresh).astype(bool)
                    continue_to_next = False
                else:
                    continue_to_next = True
            else:
                continue_to_next = True

            if continue_to_next:
                # Generate new prompts using iterative_predict_with_clicks
                binary_mask, best_mask, used_pts, used_labels, best_iou = iterative_predict_with_clicks(
                    predictor=predictor,
                    gt_bool=gt_bool,
                    first_frame_mask_np=first_mask,
                    obj_id=obj_id,
                    score_thresh=score_thresh,
                    num_correction_clicks=num_correction,
                    multimask_output=multimask_output,
                )
                obj_points[obj_id] = [(int(x), int(y), int(label)) for (x, y), label in zip(used_pts, used_labels, strict=True)]
                best_idx = -1

            seg[obj_id] = binary_mask

            # Debug: Check for all-foreground anomaly
            fg_ratio = binary_mask.sum() / binary_mask.size
            if fg_ratio > 0.9:
                print(f"⚠️ [ANOMALY] vid={vid}, obj={obj_id}, fg_ratio={fg_ratio:.4f}, logits: min={best_mask.min():.2f}, max={best_mask.max():.2f}")

            # Collect statistics if enabled
            if collect_statistics and video_statistics is not None:
                frame_key = f"{dataset_name}_{vid}_obj{obj_id}_f0"
                video_statistics[f"{frame_key}_iou_pred"] = float(best_iou)
                video_statistics[f"{frame_key}_selected_mask_idx"] = int(best_idx) if best_idx >= 0 else 0
                total_frames_processed += 1

        video_segments[0] = seg

        # Save masks
        from concurrent.futures import ThreadPoolExecutor, as_completed

        per_obj_png_file = "sav" in (dataset_name or "").lower()
        max_io_workers = min(4, len(video_segments))

        if save_masks and max_io_workers > 0:
            with ThreadPoolExecutor(max_workers=max_io_workers) as io_executor:
                futures = [io_executor.submit(save_single_mask_helper, f_idx, seg, vid, frame_names, out_dir, H, W, per_obj_png_file) for f_idx, seg in video_segments.items()]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.warning(f"Failed to save mask: {e}")

        video_segments.clear()

        # Save query prompts
        save_prompts_to_json(out_dir, vid, obj_points, click_protocol)

        # Merge video statistics
        if collect_statistics and video_statistics and dataset_statistics is not None:
            dataset_statistics.update(video_statistics)
            print(f"Collected statistics for video {vid}: {len(video_statistics)} metrics")

            # Checkpoint if needed
            if stats_checkpoint_mgr and stats_checkpoint_mgr.should_checkpoint(v_idx):
                _, n_orig, n_down = downsample_statistics_pavpu(dataset_statistics, max_samples=downsample_max_samples)
                if n_down < n_orig:
                    print(f"  💾 Downsampled: {n_orig:,} → {n_down:,} samples ({n_down / n_orig * 100:.1f}%)")

                checkpoint_file = stats_checkpoint_mgr.save_checkpoint(dataset_statistics, v_idx)
                print(f"💾 Saved checkpoint ({v_idx}/{len(video_names)} videos) to {checkpoint_file.name}")

                dataset_statistics.clear()
                CheckpointManager.force_memory_cleanup()

        # Reset predictor
        predictor.reset_predictor()

        # Cleanup GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✓ Video {vid} completed ({v_idx}/{len(video_names)})")
        # Progress output for parallel_compare.py progress monitor
        print(f"Progress: {v_idx}/{len(video_names)} ({v_idx / len(video_names) * 100:.1f}%)")

    # Save remaining statistics
    if stats_checkpoint_mgr and dataset_statistics and len(dataset_statistics) > 0:
        _, n_orig, n_down = downsample_statistics_pavpu(dataset_statistics, max_samples=downsample_max_samples)
        if n_down < n_orig:
            print(f"  💾 Downsampled: {n_orig:,} → {n_down:,} samples")

        checkpoint_file = stats_checkpoint_mgr.save_checkpoint(dataset_statistics, len(video_names))
        print(f"💾 Saved final checkpoint to {checkpoint_file.name}")
        dataset_statistics.clear()
        CheckpointManager.force_memory_cleanup()

    # Merge checkpoints
    if stats_checkpoint_mgr:
        merged_stats = stats_checkpoint_mgr.merge_checkpoints()
        if dataset_statistics is None:
            dataset_statistics = {}
        dataset_statistics.update(merged_stats)

    # Finalize evaluation
    if collect_statistics:
        finalize_bndl_evaluation(dataset_evaluator, dataset_name)

    if collect_statistics and dataset_statistics:
        print(f"\nBNDL Statistics Summary for {dataset_name}:")
        print(f"Total frames processed: {total_frames_processed}")
        print(f"Total statistics collected: {len(dataset_statistics)}")

    return dataset_statistics if collect_statistics else None


def run_single_dataset_with_bndl(
    dataset_name: str,
    predictor: SAM2ImagePredictor | None,
    output_path: Path,
    split: str | list[str] | None = None,
    score_thresh: float = 0.0,
    num_workers: int | None = None,
    video_subset: list[str] | None = None,
    save_bndl_vis: bool = False,
    first_frame_only: bool = True,
    max_objects: int | None = None,
    collect_statistics: bool = False,
    reuse_prompts_root: Path | None = None,
    click_protocol: str = "3click",
    min_click_dist: float = 12.0,
    seed: int = 0,
    downsample_max_samples: int = 100000,
    max_vis_per_video: int = 2,
    save_vis_pdf: bool = True,
    bndl_sample_num: int = 20,
    save_masks: bool = True,
    multimask_output: bool = True,
    # Predictor build parameters (used if predictor is None)
    predictor_cfg: str | None = None,
    predictor_ckpt: str | None = None,
    predictor_device: str = "cuda",
) -> tuple[float, float, float, dict]:
    """Run evaluation on single dataset using SAM2ImagePredictor."""
    from zs_dataset_runner import run_single_dataset_generic

    # Build predictor if not provided
    if predictor is None and predictor_cfg and predictor_ckpt:
        predictor = build_image_predictor(
            cfg_file=predictor_cfg,
            ckpt=predictor_ckpt,
            device=predictor_device,
            multimask=multimask_output,
        )

    bndl_vis_dir = output_path / f"{dataset_name.lower()}_bndl_vis" if save_bndl_vis else None
    bndl_eval_dir = output_path / f"{dataset_name.lower()}_bndl_eval" if collect_statistics else None

    jf, j, f, stats = run_single_dataset_generic(
        dataset_name=dataset_name,
        predictor=predictor,
        output_path=output_path,
        inference_fn=inference_with_bndl,
        method_name="BNDL",
        split=split,
        video_subset=video_subset,
        num_workers=num_workers,
        first_frame_only=first_frame_only,
        score_thresh=score_thresh,
        max_objects=max_objects,
        reuse_prompts_root=reuse_prompts_root,
        click_protocol=click_protocol,
        seed=seed,
        collect_statistics=collect_statistics,
        downsample_max_samples=downsample_max_samples,
        # BNDL-specific kwargs
        save_bndl_vis=save_bndl_vis,
        vis_dir=bndl_vis_dir,
        eval_dir=bndl_eval_dir,
        max_vis_per_video=max_vis_per_video,
        save_vis_pdf=save_vis_pdf,
        bndl_sample_num=bndl_sample_num,
        save_masks=save_masks,
        multimask_output=multimask_output,
    )

    if stats:
        stats_file = output_path / f"{dataset_name.lower()}_bndl_statistics.json"
        with open(stats_file, "w") as f_out:
            json.dump(stats, f_out, indent=2)
        print(f"BNDL statistics saved to: {stats_file}")

    return jf, j, f, (stats or {})


def parse_args():
    parser = argparse.ArgumentParser(description="BNDL Zero-shot Evaluation (SAM2ImagePredictor)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to SAM2 config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/bndl_eval",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.0,
        help="Score threshold for mask binarization",
    )
    parser.add_argument(
        "--no-multimask",
        action="store_true",
        help="Disable multimask output (not recommended)",
    )
    parser.add_argument(
        "--first-frame-only",
        action="store_true",
        default=True,
        help="Only evaluate first frame (default for image-based)",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Maximum objects per video",
    )
    parser.add_argument(
        "--video-limit",
        type=int,
        default=None,
        help="Limit number of videos per dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--click-protocol",
        type=str,
        default="3click",
        help="Click protocol (1click, 3click, etc.)",
    )
    parser.add_argument(
        "--collect-stats",
        action="store_true",
        help="Collect BNDL statistics",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save visualizations",
    )
    parser.add_argument(
        "--no-save-masks",
        action="store_true",
        help="Do not save predicted masks",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BNDL Zero-shot Evaluation (using SAM2ImagePredictor)")
    print("=" * 80)
    print(f"Key feature: Selects best mask by IoU prediction (aligns with training)")
    print(f"multimask_output: {not args.no_multimask}")
    print()

    # Build predictor
    predictor = build_image_predictor(
        cfg_file=args.cfg,
        ckpt=args.checkpoint,
        device=args.device,
        multimask=not args.no_multimask,
    )

    results = {}
    all_stats = {}

    for dataset_name in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {dataset_name}")
        print(f"{'=' * 60}")

        # Get video subset if limited
        video_subset = None
        if args.video_limit:
            config = DATASET_CONFIGS.get(dataset_name)
            if config:
                root = Path(config["root"])
                split = config.get("default_split", "val")
                if isinstance(split, list):
                    split = split[0]
                if config.get("has_split_subdir"):
                    jpeg_dir = root / split / "JPEGImages"
                else:
                    jpeg_dir = root / "JPEGImages"
                if jpeg_dir.exists():
                    all_videos = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
                    video_subset = all_videos[: args.video_limit]
                    print(f"Limited to {len(video_subset)} videos")

        jf, j, f, stats = run_single_dataset_with_bndl(
            dataset_name=dataset_name,
            predictor=predictor,
            output_path=output_path,
            score_thresh=args.score_thresh,
            video_subset=video_subset,
            first_frame_only=args.first_frame_only,
            max_objects=args.max_objects,
            collect_statistics=args.collect_stats,
            click_protocol=args.click_protocol,
            seed=args.seed,
            multimask_output=not args.no_multimask,
            save_bndl_vis=args.save_vis,
            save_masks=not args.no_save_masks,
        )

        results[dataset_name] = (jf, j, f)
        if stats:
            all_stats[dataset_name] = stats

        print(f"{dataset_name}: J&F={jf:.2f}, J={j:.2f}, F={f:.2f}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for dataset, (jf, j, f) in results.items():
        print(f"  {dataset}: J&F={jf:.2f}, J={j:.2f}, F={f:.2f}")

    if results:
        avg_jf = np.mean([r[0] for r in results.values()])
        avg_j = np.mean([r[1] for r in results.values()])
        avg_f = np.mean([r[2] for r in results.values()])
        print(f"\n  AVERAGE: J&F={avg_jf:.2f}, J={avg_j:.2f}, F={avg_f:.2f}")

    # Save results
    results_file = output_path / "bndl_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "results": {k: {"jf": v[0], "j": v[1], "f": v[2]} for k, v in results.items()},
                "config": {
                    "multimask_output": not args.no_multimask,
                    "click_protocol": args.click_protocol,
                    "score_thresh": args.score_thresh,
                },
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
