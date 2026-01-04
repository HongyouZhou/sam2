#!/usr/bin/env python
# Multi-dataset Zero-shot evaluation of SAM-2 with BNDL
# Supports TrashCan, GTEA, PIDRay, plittersdorf, Hypersim, DRAM, and CITYSCAPES datasets with UQ analysis

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")  # Use non-interactive backend to avoid Qt issues
import logging
import os
import time

import matplotlib.pyplot as plt
import seaborn as sns

# ----------  Tools -----------
from tools.vos_inference import (
    DAVIS_PALETTE,
    save_masks_to_dir,
)

# ----------  BNDL uncertainty functions ----------
from BNDL.BNDL_upload.ViT_Sparse.utils.bndl import pixel_uncertain_sampling, pixel_entropy_uncertainty

# ----------  Dataset Evaluator from SAM2 training ----------
from training.utils.dataset_evaluator import DistributedDatasetEvaluator

# ----------  SAM-2 -----------
from sam2.build_sam import build_sam2_video_predictor

# ----------  Shared utilities (NEW) ----------
from zero_shot_utils import (
    load_first_frame_mask,
    threshold_mask_logits,
    select_top_objects_by_area,
    select_top_objects_by_area,
    cleanup_gpu_memory,
    save_single_mask_helper,
)
from bndl_eval_utils import (
    calculate_pavpu_for_bndl,
    create_bndl_visualization_refactored,
    extract_evaluator_checkpoint_data,
    finalize_bndl_evaluation,
    log_bndl_statistics,
    merge_evaluator_checkpoints,
    setup_bndl_collection,
)
from checkpoint_manager import CheckpointManager, StatisticsCheckpointManager
from evaluation_pipeline import run_benchmark_evaluation
from downsampling_utils import downsample_statistics_pavpu

# ----------  Import refactored visualization modules (lazy import inside function) ----------
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training", "utils"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Dataset Configurations ----------
from dataset_configs import DATASET_CONFIGS, DEFAULT_DATASETS

# ---------- Unified click prompt generator ----------
from prompt_generation import generate_click_prompts
from prompt_loader import load_reused_prompts, apply_reused_prompts, save_prompts_to_json

# Distinct colors for different objects
OBJECT_COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
]

# Point colors for positive/negative/error clicks
POINT_COLORS = [
    (255, 0, 0),  # Red for positive
    (0, 0, 255),  # Blue for negative
    (255, 255, 0),  # Yellow for error-based
]


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def inference_with_bndl(
    predictor,
    jpeg_dir: Path,
    ann_dir: Path,
    out_dir: Path,
    score_thresh: float = 0.0,
    video_names: list[str] | None = None,
    save_bndl_vis: bool = False,  # Default False for faster evaluation
    vis_dir: Path | None = None,
    dataset_name: str = "unknown",
    collect_statistics: bool = True,
    eval_dir: Path | None = None,
    max_objects: int | None = None,
    first_frame_only: bool = False,
    reuse_prompts_root: Path | None = None,
    # Protocol controls (no GT box fallback)
    click_protocol: str = "3click",
    min_click_dist: float = 12.0,
    seed: int | None = 0,
    downsample_max_samples: int = 100000,
    # Paper figure generation options
    max_vis_per_video: int = 2,  # Max visualizations per video (2 for benchmarking, set higher for paper figures)
    save_vis_pdf: bool = True,  # Save PDF versions for paper (300 DPI) - default True for professional use
    # Optimization options
    bndl_sample_num: int = 20,  # Number of samples for BNDL uncertainty
    save_masks: bool = True,  # Save predicted masks to disk
):
    """
    3-click interactive inference with BNDL UQ analysis:
    1) Random positive point inside GT
    2) Random negative point near GT boundary
    3) Error-based point from prediction vs GT difference
    """
    if video_names is None:
        video_names = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    else:
        video_names = sorted(set(video_names))

    # 打印格式要与其他方法一致，以便 parallel_compare.py 的进度监控器能正确捕获
    print(f"BNDL ({click_protocol}) inference on {len(video_names)} videos")

    # Initial cleanup
    cleanup_gpu_memory(predictor)

    # Optional seeding
    if seed is not None:
        np.random.seed(int(seed))

    # Create BNDL visualization directory
    if save_bndl_vis and vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Initialize statistics collection infrastructure (extracted to helper function)
    dataset_statistics, stats_checkpoint_mgr, eval_checkpoint_mgr, dataset_evaluator = setup_bndl_collection(collect_statistics, out_dir, dataset_name, eval_dir)
    total_frames_processed = 0

    # 🧪 EXPERIMENTAL: Try to detect and extract predictor config for potential reload
    predictor_cfg_file = None
    predictor_ckpt = None
    try:
        # Try to extract config from predictor for potential reload
        if hasattr(predictor, "_cfg_file"):
            predictor_cfg_file = predictor._cfg_file
        if hasattr(predictor, "_ckpt_path"):
            predictor_ckpt = predictor._ckpt_path
    except Exception:
        pass

    for v_idx, vid in enumerate(video_names, 1):
        # 🧪 EXPERIMENTAL: Force reinitialize SAM model for each video (slow but thorough)
        # This is a workaround for the issue where BNDL internal states accumulate across videos
        # causing all-foreground predictions after the first video
        print(f"\n🔄 [EXPERIMENT] Reinitializing SAM model for video {v_idx}...")

        # Get device from current predictor
        device = next(predictor.parameters()).device if hasattr(predictor, "parameters") else torch.device("cuda")

        # Clear old predictor from GPU
        del predictor
        torch.cuda.empty_cache()
        import gc

        gc.collect()

        # Rebuild predictor from saved checkpoint path
        # NOTE: This requires that the predictor was built with the same checkpoint path
        # The parent function (e.g., main()) should pass the predictor_config or rebuild here
        from shared_evaluation_utils import build_predictor_with_overrides
        import os

        # Try to find checkpoint from multiple sources (in order of priority):
        # 1. Environment variables set by train_and_zs.sh/parallel_compare.py
        # 2. Variables extracted from the original predictor
        # 3. Inferred from checkpoint path + experiment config directory

        # Priority 1: BNDL_AUE specific env vars (set by train_and_zs.sh)
        # These are passed via --bndl_aue_cfg and --bndl_aue_checkpoint
        cfg_file = os.environ.get("BNDL_AUE_CFG") or os.environ.get("SAM2_CFG_FILE") or predictor_cfg_file
        ckpt_path = os.environ.get("BNDL_AUE_CKPT") or os.environ.get("SAM2_CKPT_PATH") or predictor_ckpt

        predictor = build_predictor_with_overrides(
            cfg_file=cfg_file,
            ckpt=ckpt_path,
            device=str(device),
            multimask=False,
        )
        print(f"✓ SAM model reinitialized successfully from {ckpt_path}")

        if hasattr(predictor, "model"):
            # Reset any internal state in the model
            predictor.model.eval()  # Re-ensure eval mode

            # Reset SAM2 base debug counter (_sam_heads_debug_counter in sam2_base.py)
            if hasattr(predictor.model, "_sam_heads_debug_counter"):
                predictor.model._sam_heads_debug_counter = 0

            # Reset BNDL debug counters (these are just for logging but should be reset per video)
            if hasattr(predictor.model, "sam_mask_decoder"):
                mask_decoder = predictor.model.sam_mask_decoder
                if hasattr(mask_decoder, "pixel_bndl_projector"):
                    bndl = mask_decoder.pixel_bndl_projector
                    if hasattr(bndl, "_eval_debug_counter"):
                        bndl._eval_debug_counter = 0
                    if hasattr(bndl, "_sparse_eval_debug_counter"):
                        bndl._sparse_eval_debug_counter = 0

        # NOTE: Removed per-video cleanup_gpu_memory() to reduce CUDA sync overhead
        # Memory is managed at video completion instead

        print(f"\n{'=' * 60}")
        print(f"📹 Processing video [{v_idx:03}/{len(video_names)}]: {vid}")
        print(f"   Progress: {v_idx}/{len(video_names)} ({100.0 * v_idx / len(video_names):.1f}%)")
        print(f"{'=' * 60}")
        video_dir = jpeg_dir / vid
        frame_names = sorted([p.stem for p in video_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]], key=lambda x: int(x))

        # Check if video is already processed (resumability)
        # We use query_prompts.json as a completion marker as it's saved last
        if (out_dir / vid / "query_prompts.json").exists():
            print(f"Skipping video {vid} - already processed (found query_prompts.json)")
            continue

        # Initialize predictor state
        max_frames_to_load = 1 if first_frame_only else None
        state = predictor.init_state(
            str(video_dir),
            max_frames=max_frames_to_load,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        H, W = state["video_height"], state["video_width"]

        # Read first frame GT to determine object IDs
        first_mask = load_first_frame_mask(ann_dir, vid, frame_names)
        if first_mask is None:
            continue

        all_obj_ids = [oid for oid in np.unique(first_mask) if oid > 0]

        if len(all_obj_ids) == 0:
            print(f"Warning: No objects found in first frame of video {vid}")
            continue

        # Apply object limit if specified (select largest areas)
        obj_ids = select_top_objects_by_area(first_mask, max_objects if max_objects else 10**9)
        if max_objects and len(all_obj_ids) > len(obj_ids):
            print(f"Limited to {len(obj_ids)} largest objects in video {vid} (from {len(all_obj_ids)} total)")

        # Load reused prompts if available
        prompts_json = load_reused_prompts(reuse_prompts_root, dataset_name, vid)
        if prompts_json:
            print(f"Loaded reused prompts for video {vid}")

        print(f"Processing {len(obj_ids)} objects in video {vid}: {obj_ids}")

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(int(seed))

        obj_points: dict[int, list[tuple[int, int, int]]] = {}  # Now stores (x, y, label)

        for obj_id in obj_ids:
            gt_bool = first_mask == obj_id
            if not np.any(gt_bool):
                continue

            # NOTE: Removed per-object cuda.empty_cache() to reduce CUDA sync overhead
            # Memory is managed at frame level for videos with many objects

            # Try reused prompts first, fall back to generation
            prompt_applied = False
            if prompts_json and obj_id in prompts_json:
                prompt_applied = apply_reused_prompts(predictor, state, obj_id, prompts_json[obj_id])
                if prompt_applied:
                    # Extract points AND labels from prompt spec for saving
                    clicks = prompts_json[obj_id].get("clicks", [])
                    obj_points[obj_id] = [(int(c["xy"][0]), int(c["xy"][1]), int(c.get("label", 1))) for c in clicks if "xy" in c]

            if not prompt_applied:
                # Generate new prompts using click protocol
                used_pts, used_labels = generate_click_prompts(
                    predictor,
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    gt_bool=gt_bool,
                    first_frame_mask_np=first_mask,
                    score_thresh=score_thresh,
                    click_protocol=click_protocol,
                    min_click_dist=float(min_click_dist or 12.0),
                )
                obj_points[obj_id] = [(int(x), int(y), int(label)) for (x, y), label in zip(used_pts, used_labels, strict=True)]

        print(f"Query points for video {vid}: {obj_points}")

        # Propagate through entire video with BNDL UQ analysis
        # When first_frame_only=True, only process the first frame (frame 0)
        # In SAM2: end_frame_idx = start_frame_idx + max_frame_num_to_track
        #          processing_order = range(start_frame_idx, end_frame_idx + 1)
        # To only process frame 0: we need end_frame_idx = 0, so max_frame_num_to_track = 0
        max_frames = 0 if first_frame_only else None
        video_segments = {}
        bndl_vis_count = 0
        video_statistics = {} if collect_statistics else None

        for f_idx, out_obj_ids, out_logits in predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=max_frames):
            # Double safety: break if first_frame_only is requested and we have processed one frame
            if first_frame_only and f_idx > 0:
                print(f"Breaking video propagation after frame {f_idx} (first_frame_only=True)")
                del out_logits
                break

            seg = {}
            for i, oid in enumerate(out_obj_ids):
                mask_logits = out_logits[i]

                # Handle multimask output (K>1): select mask 0 (singlemask output token)
                # This matches SAM-2's default behavior when multimask_output=False
                if mask_logits.ndim == 3:
                    if mask_logits.shape[0] == 1:
                        # Single mask: squeeze it
                        mask_logits = mask_logits.squeeze(0)
                    elif mask_logits.shape[0] > 1:
                        # Multiple masks: use mask 0 (singlemask token, SAM-2 default)
                        mask_logits = mask_logits[0]

                if tuple(mask_logits.shape[-2:]) != (H, W):
                    import torch.nn.functional as F

                    mask_logits = F.interpolate(
                        mask_logits.unsqueeze(0).unsqueeze(0),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                binary_mask = threshold_mask_logits(mask_logits, score_thresh)
                seg[oid] = binary_mask

                # DEBUG: Check for all-foreground anomaly in propagate output
                fg_ratio = binary_mask.sum() / binary_mask.size
                if fg_ratio > 0.9:
                    logits_np = mask_logits.cpu().numpy() if hasattr(mask_logits, "cpu") else mask_logits
                    print(f"⚠️ [PROPAGATE ANOMALY] vid={vid}, obj={oid}, fg_ratio={fg_ratio:.4f}")
                    print(f"   logits: min={logits_np.min():.2f}, max={logits_np.max():.2f}, mean={logits_np.mean():.2f}")

            video_segments[f_idx] = seg

            # Clear GPU memory only for videos with many objects (threshold raised to reduce sync overhead)
            if torch.cuda.is_available() and len(out_obj_ids) > 20:
                torch.cuda.empty_cache()

            # Collect BNDL statistics if enabled (with memory optimization)
            if collect_statistics:
                # Map object IDs to their index in out_logits to fetch per-object logits reliably
                id_to_idx = {oid: i for i, oid in enumerate(out_obj_ids)}
                # Limit to 3 objects per frame for stats collection (memory already managed by checkpoints)
                max_obj_stats = 3
                logger.info(f"Processing {len(out_obj_ids[:max_obj_stats])} objects for stats collection")

                # Pre-load GT mask once per frame to avoid redundant I/O
                gt_mask_full = None
                current_mask_path = ann_dir / vid / f"{frame_names[f_idx]}.png"
                if current_mask_path.exists():
                    try:
                        gt_mask_full = np.array(Image.open(current_mask_path))
                    except Exception as e:
                        logger.warning(f"Failed to load GT mask for frame {f_idx}: {e}")

                for obj_id in out_obj_ids[:max_obj_stats]:
                    # Convert obj_id to internal obj_idx using predictor's mapping
                    obj_idx = predictor._obj_id_to_idx(state, obj_id)
                    logger.info(f"Object ID {obj_id} -> internal index {obj_idx}")
                    bndl_outputs = predictor.get_bndl_outputs(state, f_idx, obj_idx)
                    logger.info(f"get_bndl_outputs(frame={f_idx}, obj_idx={obj_idx}): returned {'data' if bndl_outputs is not None else 'None'}")

                    if bndl_outputs is not None:
                        # Calculate PAvPU if we have ground truth AND statistics collection is enabled
                        if collect_statistics and gt_mask_full is not None:
                            # Extract binary mask for current object
                            gt_mask = (gt_mask_full == obj_id).astype(np.float32)
                            # Convert to tensor format for PAvPU calculation
                            gt_tensor = torch.from_numpy(gt_mask).float().unsqueeze(0)  # [1, H, W]
                            # Move to the same device as BNDL outputs
                            if "pixel_logits_raw" in bndl_outputs:
                                gt_tensor = gt_tensor.to(bndl_outputs["pixel_logits_raw"].device)
                            elif "wei_lambda" in bndl_outputs:
                                gt_tensor = gt_tensor.to(bndl_outputs["wei_lambda"].device)
                            bndl_outputs = calculate_pavpu_for_bndl(bndl_outputs, None, gt_tensor, "eval", predictor, sample_num=bndl_sample_num)
                            del gt_tensor, gt_mask

                        # Log statistics if enabled
                        if collect_statistics:
                            video_statistics = log_bndl_statistics(bndl_outputs, f_idx, "eval", f"{dataset_name}_{vid}_obj{obj_id}", video_statistics)
                            total_frames_processed += 1

                        # Add to dataset evaluator (memory managed by checkpoints)
                        if dataset_evaluator is not None:
                            _idx = id_to_idx.get(obj_id)
                            if _idx is not None and _idx < len(out_logits):
                                pred_logits = out_logits[_idx]

                                if gt_mask_full is not None and "pixel_uncertainty" in bndl_outputs:
                                    # Extract binary mask for current object only
                                    current_gt_mask = (gt_mask_full == obj_id).astype(np.float32)
                                    current_gt_tensor = torch.from_numpy(current_gt_mask).unsqueeze(0)

                                    # Move to same device
                                    if "pixel_logits_raw" in bndl_outputs:
                                        current_gt_tensor = current_gt_tensor.to(bndl_outputs["pixel_logits_raw"].device)
                                    elif "wei_lambda" in bndl_outputs:
                                        current_gt_tensor = current_gt_tensor.to(bndl_outputs["wei_lambda"].device)

                                    dataset_evaluator.add_batch_data(
                                        uncertainty=bndl_outputs["pixel_uncertainty"],
                                        pred_logits=pred_logits.unsqueeze(0),
                                        gt_masks=current_gt_tensor,
                                    )
                                    logger.info(f"Added frame {f_idx} obj {obj_id} to dataset evaluator")
                                    del current_gt_mask, current_gt_tensor

                        # Clean up bndl_outputs after processing
                        del bndl_outputs

            # Generate BNDL visualizations for selected frames (reduced for memory)
            if save_bndl_vis and vis_dir is not None and bndl_vis_count < max_vis_per_video:  # Limit visualizations per video
                try:
                    # Extract BNDL outputs from the predictor state
                    # This requires accessing the internal state or outputs
                    # For now, we'll create a mock batch for visualization
                    if hasattr(predictor, "get_bndl_outputs") and out_obj_ids:
                        # Use the first object's ID for visualization
                        first_obj_id = out_obj_ids[0]
                        first_obj_idx = predictor._obj_id_to_idx(state, first_obj_id)
                        bndl_outputs = predictor.get_bndl_outputs(state, f_idx, first_obj_idx)
                        if bndl_outputs is not None:
                            # Calculate PAvPU for visualization (same as in statistics collection)
                            current_mask_path = ann_dir / vid / f"{frame_names[f_idx]}.png"
                            if current_mask_path.exists():
                                gt_mask_full = np.array(Image.open(current_mask_path))
                                # Extract binary mask for the first object (consistent with evaluation)
                                gt_mask = (gt_mask_full == first_obj_id).astype(np.float32)
                                # Convert to tensor format for PAvPU calculation and move to same device as BNDL outputs
                                gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0)  # [1, H, W]
                                # Move to the same device as BNDL outputs
                                if "pixel_logits_raw" in bndl_outputs:
                                    gt_tensor = gt_tensor.to(bndl_outputs["pixel_logits_raw"].device)
                                elif "wei_lambda" in bndl_outputs:
                                    gt_tensor = gt_tensor.to(bndl_outputs["wei_lambda"].device)
                                bndl_outputs = calculate_pavpu_for_bndl(bndl_outputs, None, gt_tensor, "eval", predictor, sample_num=bndl_sample_num)

                                # Clean up ground truth tensor immediately
                                del gt_tensor, gt_mask

                            # Create visualization
                            vis_path = vis_dir / vid
                            vis_path.mkdir(parents=True, exist_ok=True)

                            # Build a batch carrying the real frame image instead of random noise
                            frame_base = frame_names[f_idx]
                            img_path = video_dir / f"{frame_base}.jpg"
                            if not img_path.exists():
                                alt = video_dir / f"{frame_base}.jpeg"
                                if alt.exists():
                                    img_path = alt
                            try:
                                img = Image.open(img_path).convert("RGB")
                                img = img.resize((W, H))
                                img_np = np.array(img).astype(np.float32) / 255.0  # [H, W, 3] in [0,1]
                                img_chw = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                            except Exception:
                                # Fallback to zeros if the frame cannot be read
                                img_chw = torch.zeros(1, 3, H, W, dtype=torch.float32)

                            mock_batch = type(
                                "MockBatch",
                                (),
                                {
                                    "img_batch": img_chw,
                                    "masks": torch.from_numpy(seg[out_obj_ids[0]]).unsqueeze(0) if out_obj_ids else torch.zeros(1, H, W),
                                },
                            )()

                            # Build prompt_info from obj_points for visualization
                            # obj_points format: {obj_id: [(x, y, label), ...]}
                            prompt_info = None
                            if obj_points and first_obj_id in obj_points:
                                pts = obj_points[first_obj_id]
                                if pts:
                                    # Convert to format expected by visualizer
                                    point_coords = np.array([[p[0], p[1]] for p in pts])  # [N, 2]
                                    point_labels = np.array([p[2] for p in pts])  # [N]
                                    prompt_info = {
                                        "point_coords": point_coords,
                                        "point_labels": point_labels,
                                    }

                            # Use refactored visualization function
                            create_bndl_visualization_refactored(
                                bndl_outputs,
                                mock_batch,
                                {"masks": out_logits},
                                str(vis_path),
                                f_idx,
                                0,  # step_index
                                0,  # frame_index
                                "full",
                                save_individual=True,
                                save_unified=False,
                                prompt_info=prompt_info,
                                save_pdf=save_vis_pdf,
                            )
                            bndl_vis_count += 1
                except Exception as e:
                    logger.warning(f"Failed to create BNDL visualization for frame {f_idx}: {e}")

            # Clear output logits after all usage to free memory
            del out_logits

            # Clean up memory after processing each frame (removed to match original flow)

            # Removed extra periodic GPU sync/GC to match original flow

        # Save PNG masks in parallel using ThreadPoolExecutor for better I/O performance
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Determine if per-object PNG files are needed (e.g. for SA-V dataset)
        per_obj_png_file = "sav" in (dataset_name or "").lower()

        # Use up to 4 threads for I/O (more threads don't help with disk I/O)
        max_io_workers = min(4, len(video_segments))
        if save_masks and max_io_workers > 0:
            with ThreadPoolExecutor(max_workers=max_io_workers) as io_executor:
                futures = [io_executor.submit(save_single_mask_helper, f_idx, seg, vid, frame_names, out_dir, H, W, per_obj_png_file) for f_idx, seg in video_segments.items()]
                # Wait for all saves to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.warning(f"Failed to save mask: {e}")

        # Clear any remaining frames
        video_segments.clear()

        # Save query prompts in standard format
        save_prompts_to_json(out_dir, vid, obj_points, click_protocol)

        # Merge video statistics into dataset statistics
        if collect_statistics and video_statistics and dataset_statistics is not None:
            dataset_statistics.update(video_statistics)
            print(f"Collected BNDL statistics for video {vid}: {len(video_statistics)} metrics")

            # Incremental save: periodically save statistics and clear memory to prevent OOM
            if stats_checkpoint_mgr and stats_checkpoint_mgr.should_checkpoint(v_idx):
                # 🎯 保存checkpoint前降采样PAvPU样本
                _, n_orig, n_down = downsample_statistics_pavpu(dataset_statistics, max_samples=downsample_max_samples)
                if n_down < n_orig:
                    print(f"  💾 降采样: {n_orig:,} → {n_down:,} PAvPU样本 ({n_down / n_orig * 100:.1f}%)")

                checkpoint_file = stats_checkpoint_mgr.save_checkpoint(dataset_statistics, v_idx)
                print(f"💾 Saved statistics checkpoint ({v_idx}/{len(video_names)} videos) to {checkpoint_file.name}")

                # Clear statistics from memory
                dataset_statistics.clear()
                CheckpointManager.force_memory_cleanup()

                # Also checkpoint dataset_evaluator to prevent OOM from pixel-level data accumulation
                if eval_checkpoint_mgr and dataset_evaluator is not None and len(dataset_evaluator) > 0:
                    eval_checkpoint_data = extract_evaluator_checkpoint_data(dataset_evaluator)
                    eval_checkpoint_file = eval_checkpoint_mgr.save_checkpoint(eval_checkpoint_data, v_idx)
                    print(f"💾 Saved evaluator checkpoint ({v_idx}/{len(video_names)} videos) to {eval_checkpoint_file.name}")

                    # Clear evaluator data using built-in reset method
                    dataset_evaluator.reset()
                    CheckpointManager.force_memory_cleanup()

        # Critical: Reset predictor state to free memory for this video
        cleanup_gpu_memory(predictor, state)
        print(f"✓ Video {vid} completed ({v_idx}/{len(video_names)})")

    # Save any remaining statistics after the last video
    if stats_checkpoint_mgr and dataset_statistics and len(dataset_statistics) > 0:
        # 🎯 保存最终checkpoint前降采样PAvPU样本
        _, n_orig, n_down = downsample_statistics_pavpu(dataset_statistics, max_samples=downsample_max_samples)
        if n_down < n_orig:
            print(f"  💾 降采样: {n_orig:,} → {n_down:,} PAvPU样本 ({n_down / n_orig * 100:.1f}%)")

        checkpoint_file = stats_checkpoint_mgr.save_checkpoint(dataset_statistics, len(video_names))
        print(f"💾 Saved final statistics checkpoint to {checkpoint_file.name}")
        dataset_statistics.clear()
        CheckpointManager.force_memory_cleanup()

    # Merge evaluator checkpoint files back into dataset_evaluator
    merge_evaluator_checkpoints(eval_checkpoint_mgr, dataset_evaluator, downsample_max_samples)

    # Generate dataset evaluation plots
    if collect_statistics:
        finalize_bndl_evaluation(dataset_evaluator, dataset_name)

    # Merge checkpoint files back into final statistics
    if stats_checkpoint_mgr:
        merged_stats = stats_checkpoint_mgr.merge_checkpoints()
        if dataset_statistics is None:
            dataset_statistics = {}
        dataset_statistics.update(merged_stats)

        # 🎯 关键优化: 合并后再次降采样（防止多个checkpoint累积太多样本）
        _, n_orig, n_down = downsample_statistics_pavpu(dataset_statistics, max_samples=downsample_max_samples)
        if n_down < n_orig:
            print(f"  💾 合并后降采样: {n_orig:,} → {n_down:,} PAvPU样本 ({n_down / n_orig * 100:.1f}%)")

    if collect_statistics and dataset_statistics:
        print(f"\nBNDL Statistics Summary for {dataset_name}:")
        print(f"Total frames processed: {total_frames_processed}")
        print(f"Total statistics collected: {len(dataset_statistics)}")

        # Calculate average statistics
        avg_stats = {}
        # Create a copy of items to avoid "dictionary changed size during iteration" error
        statistics_items = list(dataset_statistics.items())
        for key, values in statistics_items:
            if isinstance(values, int | float):
                avg_stats[key] = values
            elif isinstance(values, list) and len(values) > 0 and not key.endswith("_pavpu_uncertainty_samples") and not key.endswith("_pavpu_accuracy_samples"):
                avg_stats[key] = sum(values) / len(values)

        if avg_stats:
            print("Average BNDL Statistics:")
            for key, value in avg_stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    return dataset_statistics if collect_statistics else None


def run_single_dataset_with_bndl(
    dataset_name: str,
    predictor,
    output_path: Path,
    split: str | list[str] | None = None,
    score_thresh: float = 0.0,
    num_workers: int | None = None,
    video_subset: list[str] | None = None,
    save_bndl_vis: bool = False,  # Default False for faster evaluation
    first_frame_only: bool = False,
    max_objects: int | None = None,
    collect_statistics: bool = False,
    reuse_prompts_root: Path | None = None,
    click_protocol: str = "3click",
    min_click_dist: float = 12.0,
    seed: int = 0,
    downsample_max_samples: int = 100000,
    # Paper figure generation options
    max_vis_per_video: int = 2,  # Max visualizations per video (2 for benchmarking, set higher for paper figures)
    save_vis_pdf: bool = True,  # Save PDF versions for paper (300 DPI) - default True for professional use
    # Optimization options
    bndl_sample_num: int = 20,
    save_masks: bool = True,
) -> tuple[float, float, float, dict]:
    """Run evaluation on a single dataset with BNDL UQ analysis and return metrics.

    This is a thin wrapper around zs_dataset_runner.run_single_dataset_generic
    that passes BNDL-specific parameters.
    """
    from zs_dataset_runner import run_single_dataset_generic

    # BNDL needs vis_dir and eval_dir computed
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
        min_click_dist=min_click_dist,
        seed=seed,
        collect_statistics=collect_statistics,
        downsample_max_samples=downsample_max_samples,
        # BNDL-specific kwargs
        save_bndl_vis=save_bndl_vis,
        vis_dir=bndl_vis_dir,
        eval_dir=bndl_eval_dir,
        # Paper figure generation options
        max_vis_per_video=max_vis_per_video,
        save_vis_pdf=save_vis_pdf,
        # Optimization options
        bndl_sample_num=bndl_sample_num,
        save_masks=save_masks,
    )

    # BNDL-specific: Save statistics to file
    if stats:
        stats_file = output_path / f"{dataset_name.lower()}_bndl_statistics.json"
        with open(stats_file, "w") as f_out:
            json.dump(stats, f_out, indent=2)
        print(f"BNDL statistics saved to: {stats_file}")

    if save_bndl_vis and bndl_vis_dir is not None:
        print(f"BNDL UQ visualizations saved to: {bndl_vis_dir}")

    return jf, j, f, (stats or {})


def create_comparison_plots_with_bndl(results: dict[str, tuple[float, float, float]], output_path: Path, all_statistics: dict | None = None):
    """Create comparison plots for all datasets with BNDL UQ information and dataset correlation analysis"""

    # Prepare data for plotting
    datasets = list(results.keys())
    j_f_scores = [results[d][0] for d in datasets]
    j_scores = [results[d][1] for d in datasets]
    f_scores = [results[d][2] for d in datasets]

    # Create DataFrame for easier plotting
    df_data = []
    for dataset in datasets:
        j_f, j, f = results[dataset]
        df_data.extend(
            [
                {"Dataset": dataset, "Metric": "J&F", "Score": j_f},
                {"Dataset": dataset, "Metric": "J (IoU)", "Score": j},
                {"Dataset": dataset, "Metric": "F (Boundary)", "Score": f},
            ]
        )

    df = pd.DataFrame(df_data)

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    title = "SAM-2 + BNDL Zero-shot 3-Click Evaluation Results"
    if all_statistics:
        title += f" (with {len(all_statistics)} datasets analyzed)"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # 1. Bar plot comparing J&F scores
    ax1 = axes[0, 0]
    bars = ax1.bar(datasets, j_f_scores, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"])
    ax1.set_title("J&F Scores Comparison", fontweight="bold")
    ax1.set_ylabel("J&F Score")
    ax1.set_ylim(0, 100)

    # Add value labels on bars
    for bar, score in zip(bars, j_f_scores, strict=True):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{score:.1f}", ha="center", va="bottom", fontweight="bold")

    # 2. Grouped bar plot for J and F scores
    ax2 = axes[0, 1]
    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax2.bar(x - width / 2, j_scores, width, label="J (IoU)", color="#FF6B6B", alpha=0.8)
    bars2 = ax2.bar(x + width / 2, f_scores, width, label="F (Boundary)", color="#4ECDC4", alpha=0.8)

    ax2.set_title("J and F Scores Comparison", fontweight="bold")
    ax2.set_ylabel("Score")
    ax2.set_xlabel("Dataset")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    ax2.set_ylim(0, 100)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{height:.1f}", ha="center", va="bottom", fontsize=9)

    # 3. Heatmap of all metrics
    ax3 = axes[1, 0]
    heatmap_data = np.array([j_f_scores, j_scores, f_scores])
    im = ax3.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax3.set_xticks(range(len(datasets)))
    ax3.set_xticklabels(datasets)
    ax3.set_yticks(range(3))
    ax3.set_yticklabels(["J&F", "J (IoU)", "F (Boundary)"])
    ax3.set_title("Performance Heatmap", fontweight="bold")

    # Add text annotations
    for i in range(3):
        for j in range(len(datasets)):
            ax3.text(j, i, f"{heatmap_data[i, j]:.1f}", ha="center", va="center", color="black", fontweight="bold")

    plt.colorbar(im, ax=ax3, label="Score")

    # 4. Stacked bar chart showing metric breakdown
    ax4 = axes[1, 1]
    sns.barplot(data=df, x="Dataset", y="Score", hue="Metric", ax=ax4)
    ax4.set_title("Detailed Metrics Breakdown", fontweight="bold")
    ax4.set_ylabel("Score")
    ax4.set_ylim(0, 100)

    # Rotate x-axis labels if needed
    for ax in axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    plt.tight_layout()

    # Save plots
    plots_dir = output_path / "comparison_plots_bndl"
    plots_dir.mkdir(exist_ok=True)

    plot_path = plots_dir / "dataset_comparison_bndl.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.savefig(plots_dir / "dataset_comparison_bndl.pdf", bbox_inches="tight")

    print(f"Comparison plots saved to: {plot_path}")

    # Save results table
    results_table = plots_dir / "results_table_bndl.csv"
    with open(results_table, "w") as file_handle:
        file_handle.write("Dataset,J&F,J (IoU),F (Boundary)\n")
        for dataset in datasets:
            j_f, j, f = results[dataset]
            file_handle.write(f"{dataset},{j_f:.2f},{j:.2f},{f:.2f}\n")

    print(f"Results table saved to: {results_table}")


def parse_args():
    p = argparse.ArgumentParser(description="Multi-dataset Zero-shot SAM-2 + BNDL evaluation with UQ analysis")

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
        default="configs/sam2.1/sam2.1_hiera_b+_bndl.yaml",
        help="SAM-2 config file",
    )
    p.add_argument(
        "--sam2_checkpoint",
        default="/home/hongyou/dev/ada_samp/logs/sam2/sam2_bndl_003_06/checkpoints/checkpoint.pt",
        help="SAM-2 checkpoint path",
    )

    # Evaluation parameters
    p.add_argument("--device", default="cuda", help="Device to use")
    p.add_argument("--score_thresh", type=float, default=0.0, help="Mask logit threshold")
    p.add_argument("--num_workers", type=int, default=None, help="Number of evaluation processes")
    p.add_argument("--output_path", default="./outputs/zs_04_09_sam_bndl", help="Root output directory")
    p.add_argument("--first_frame_only", action="store_true", help="Evaluate only the first frame per video by copying only the first PNG")

    # Multimask configuration for video predictor (safe via Hydra overrides)
    p.add_argument("--enable_multimask", action="store_true", default=True, help="Enable multimask output with predicted-IoU selection on the interaction frame")
    p.add_argument("--multimask_min_pts", type=int, default=1, help="Minimum number of points to trigger multimask (box counts as 2)")
    p.add_argument("--multimask_max_pts", type=int, default=2, help="Maximum number of points to trigger multimask (box counts as 2)")
    p.add_argument("--multimask_for_tracking", action="store_true", default=False, help="Also enable multimask during tracking frames (not just the first click)")

    # BNDL UQ visualization options
    p.add_argument("--save_bndl_vis", action="store_true", default=False, help="Generate BNDL UQ visualizations (disabled by default for speed)")
    p.add_argument("--video_limit", type=int, default=None, help="Limit number of videos per dataset (for quick testing)")
    p.add_argument("--max_objects", type=int, default=20, help="Maximum number of objects to process per video (default: 20)")
    p.add_argument("--collect_statistics", action="store_true", default=False, help="Collect BNDL statistics (uses extra GPU memory)")

    # Reuse prompts from first model outputs
    p.add_argument("--reuse_prompts_root", type=str, default=None, help="Root dir of first-run outputs to reuse prompts (expects {dataset}_pred/*/query_prompts.json)")

    # Downsampling parameters
    p.add_argument("--downsample_max_samples", type=int, default=100000, help="Maximum number of samples to keep after downsampling (default: 100000)")

    # Optimization parameters
    p.add_argument("--bndl_sample_num", type=int, default=20, help="Number of samples for BNDL uncertainty sampling (default: 20)")
    p.add_argument("--no_save_masks", action="store_true", help="Disable saving predicted masks to disk for faster evaluation")

    return p.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load SAM-2 predictor
    print("Loading SAM-2 checkpoint...")
    from shared_evaluation_utils import build_predictor_with_overrides

    predictor = build_predictor_with_overrides(
        cfg_file=args.sam2_cfg,
        ckpt=args.sam2_checkpoint,
        device=args.device,
        multimask=args.enable_multimask,
        min_pts=args.multimask_min_pts,
        max_pts=args.multimask_max_pts,
        for_tracking=args.multimask_for_tracking,
    )
    print("SAM-2 loaded successfully!")

    # Run evaluation on each dataset with BNDL UQ analysis
    results = {}
    all_statistics = {}
    total_start_time = time.time()

    for dataset_name in args.datasets:
        try:
            # Get video subset if limit is specified
            video_subset = None
            if args.video_limit is not None:
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
                    video_subset = all_videos[: args.video_limit]
                    print(f"Limited to {len(video_subset)} videos for {dataset_name}")

            # Run evaluation with BNDL UQ analysis
            j_f, j, f, dataset_statistics = run_single_dataset_with_bndl(
                dataset_name=dataset_name,
                predictor=predictor,
                output_path=output_path,
                score_thresh=args.score_thresh,
                num_workers=args.num_workers,
                video_subset=video_subset,
                save_bndl_vis=args.save_bndl_vis,
                first_frame_only=args.first_frame_only,
                max_objects=args.max_objects,
                collect_statistics=args.collect_statistics,
                reuse_prompts_root=Path(args.reuse_prompts_root) if args.reuse_prompts_root else None,
                downsample_max_samples=args.downsample_max_samples,
                bndl_sample_num=args.bndl_sample_num,
                save_masks=not args.no_save_masks,
            )

            results[dataset_name] = (j_f, j, f)
            if dataset_statistics:
                all_statistics[dataset_name] = dataset_statistics
            print(f"{dataset_name} Results - J&F: {j_f:.2f}, J: {j:.2f}, F: {f:.2f}")

        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            continue
        finally:
            pass

    total_time = time.time() - total_start_time

    # Print summary
    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY WITH BNDL UQ ANALYSIS")
    print(f"{'=' * 80}")
    print(f"{'Dataset':<12} {'J&F':<8} {'J (IoU)':<8} {'F (Boundary)':<12}")
    print("-" * 80)

    for dataset_name, (j_f, j, f) in results.items():
        print(f"{dataset_name:<12} {j_f:<8.2f} {j:<8.2f} {f:<12.2f}")

    print(f"\nTotal evaluation time: {total_time:.2f}s")

    # Print BNDL statistics summary
    if all_statistics:
        print(f"\n{'=' * 80}")
        print("BNDL STATISTICS SUMMARY")
        print(f"{'=' * 80}")

        # Create a copy of items to avoid "dictionary changed size during iteration" error
        statistics_items = list(all_statistics.items())
        for dataset_name, stats in statistics_items:
            print(f"\n{dataset_name} BNDL Statistics:")
            if stats:
                # Calculate averages for key metrics
                # Create a copy of stats items to avoid iteration error
                stats_items = list(stats.items())
                lambda_pixel_values = [v for k, v in stats_items if "lambda_pixel" in k]
                k_pixel_values = [v for k, v in stats_items if "k_pixel" in k]
                uncertainty_values = [v for k, v in stats_items if "pixel_uncertainty" in k]
                pavpu_values = [v for k, v in stats_items if "pavpu" in k]

                if lambda_pixel_values:
                    print(f"  Average Lambda (pixel): {np.mean(lambda_pixel_values):.4f} ± {np.std(lambda_pixel_values):.4f}")
                if k_pixel_values:
                    print(f"  Average K (pixel): {np.mean(k_pixel_values):.4f} ± {np.std(k_pixel_values):.4f}")
                if uncertainty_values:
                    print(f"  Average Uncertainty: {np.mean(uncertainty_values):.4f} ± {np.std(uncertainty_values):.4f}")
                if pavpu_values:
                    print(f"  Average PAvPU: {np.mean(pavpu_values):.4f} ± {np.std(pavpu_values):.4f}")
                print(f"  Total metrics collected: {len(stats)}")
            else:
                print("  No statistics collected")

        # Save combined statistics
        combined_stats_file = output_path / "all_datasets_bndl_statistics.json"
        with open(combined_stats_file, "w") as f:
            json.dump(all_statistics, f, indent=2)
        print(f"\nCombined BNDL statistics saved to: {combined_stats_file}")

    # Create comparison plots
    if len(results) > 1:
        print("\nGenerating comparison plots with BNDL UQ information...")
        create_comparison_plots_with_bndl(results, output_path, all_statistics)

    print(f"\nAll outputs saved to: {output_path}")
    print("BNDL UQ visualizations and dataset correlation analysis completed!")
    print("Check individual dataset evaluation folders for correlation plots and detailed analysis.")


if __name__ == "__main__":
    main()
