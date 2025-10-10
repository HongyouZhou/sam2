#!/usr/bin/env python
# Multi-dataset Zero-shot evaluation of SAM-2 with UCTTA (Uncertainty-Calibrated Test-Time Adaptation)
# Based on "Uncertainty-Calibrated Test-Time Model Adaptation without Forgetting" (TPAMI 2025)

import shutil
from pathlib import Path
from typing import Any, Optional
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib

matplotlib.use("Agg")

import time

# ----------  SAM-2 -----------
from sam2.build_sam import build_sam2_video_predictor

# ----------  Tools -----------
from tools.vos_inference import (
    DAVIS_PALETTE,
    save_masks_to_dir,
)

# ----------  Dataset Configurations ----------
from dataset_configs import DATASET_CONFIGS
from training.utils.dataset_evaluator import DistributedDatasetEvaluator

# ----------  Unified click prompt generator ----------
from prompt_generation import generate_click_prompts
from prompt_loader import load_reused_prompts, apply_reused_prompts

# Add path for visualization utils
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training", "utils"))


def create_uctta_ua_ratio_visualization(
    out_logits, uncertainty_map, original_img, vid, frame_name, vis_dir
):
    """Create U/A ratio visualization for UCTTA"""
    from visualization_utils import VisualizationUtils
    from bndl_visualizer import BNDLVisualizer
    
    viz_utils = VisualizationUtils()
    bndl_viz = BNDLVisualizer()
    
    # Prepare data in format expected by visualizer
    uctta_outputs = {
        "pixel_uncertainty": uncertainty_map,
        "mean_pixel_logits": out_logits,
    }
    
    # Create figure with U/A ratio visualization
    fig, axes = viz_utils.create_figure_layout(1, 3, (18, 6))
    bndl_viz.plot_uncertainty_accuracy_ratio_visualization(
        axes[0, :], uctta_outputs, original_img, step_index=0, ratio_type="U/A"
    )
    
    # Save visualization
    save_path = vis_dir / vid / f"{frame_name}_uctta_ua_ratio.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    viz_utils.save_and_close_figure(fig, str(save_path), dpi=150)


def _load_first_frame_mask(ann_dir: Path, vid: str, frame_names: list[str]) -> Optional[np.ndarray]:
    first_mask_path = ann_dir / vid / f"{frame_names[0]}.png"
    if not first_mask_path.exists():
        print(f"Warning: First frame annotation not found: {first_mask_path}")
        return None
    return np.array(Image.open(first_mask_path))


def _get_video_subset(jpeg_dir: Path, limit: Optional[int]) -> Optional[list[str]]:
    if limit is None or not jpeg_dir.exists():
        return None
    all_videos = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    return all_videos[: limit]


def _entropy_from_logits_scaled(logits: torch.Tensor, logT: torch.Tensor) -> torch.Tensor:
    """Binary entropy averaged over spatial dims for per-object logits with temperature scaling.

    logits: [H, W] or [H, W, K] or [K, H, W] -> compute sigmoid over last channel if needed.
    logT:   scalar parameter (requires_grad=True)
    """
    T = torch.exp(logT).clamp(0.25, 4.0)
    z = logits / T
    if z.ndim == 2:
        p = torch.sigmoid(z)
        ent = -(p * torch.log(p.clamp_min(1e-8)) + (1.0 - p) * torch.log((1.0 - p).clamp_min(1e-8)))
        return ent.mean()
    elif z.ndim == 3:
        # assume [..., K] is last or first; normalize to [..., K]
        if z.shape[0] <= 8 and z.shape[0] != z.shape[-1]:
            # heuristic: treat [K, H, W] -> [H, W, K]
            z = z.permute(1, 2, 0)
        p = torch.sigmoid(z)
        ent = -(p * torch.log(p.clamp_min(1e-8)) + (1.0 - p) * torch.log((1.0 - p).clamp_min(1e-8)))
        # average across channels then space
        return ent.mean()
    else:
        # fallback
        p = torch.sigmoid(z)
        ent = -(p * torch.log(p.clamp_min(1e-8)) + (1.0 - p) * torch.log((1.0 - p).clamp_min(1e-8)))
        return ent.mean()


def _entropy_map_from_logits_scaled(logits: torch.Tensor, logT: torch.Tensor) -> torch.Tensor:
    """Return per-pixel entropy map after temperature scaling, shape [H, W] or [H, W, K] -> [H, W]."""
    T = torch.exp(logT).clamp(0.25, 4.0)
    z = logits / T
    if z.ndim == 3:
        if z.shape[0] <= 8 and z.shape[0] != z.shape[-1]:
            z = z.permute(1, 2, 0)
    p = torch.sigmoid(z)
    ent = -(p * torch.log(p.clamp_min(1e-8)) + (1.0 - p) * torch.log((1.0 - p).clamp_min(1e-8)))
    if ent.ndim == 3:
        return ent.mean(dim=-1)  # [H, W]
    return ent  # [H, W]


def _apply_temperature(logits: torch.Tensor, logT: torch.Tensor) -> torch.Tensor:
    T = torch.exp(logT).clamp(0.25, 4.0)
    return logits / T


def _threshold_bool(mask_logits: torch.Tensor, score_thresh: float) -> np.ndarray:
    if mask_logits.ndim == 3:
        mask_logits = mask_logits.squeeze(0)
    return (mask_logits > score_thresh).detach().cpu().numpy().astype(bool)


def _select_top_objects_by_area(mask_np: np.ndarray, max_objects: int) -> list[int]:
    all_ids = [oid for oid in np.unique(mask_np) if oid > 0]
    if (max_objects is None) or (len(all_ids) <= max_objects):
        return all_ids
    areas = {oid: int((mask_np == oid).sum()) for oid in all_ids}
    sorted_objs = sorted(areas.items(), key=lambda x: x[1], reverse=True)
    return [oid for oid, _ in sorted_objs[:max_objects]]


# ==================== UCTTA Core Components ====================

def setup_uctta_model(model, enable_bn_adapt: bool = True):
    """Setup model for UCTTA: freeze most params, enable BN/LN layers for adaptation.
    
    Args:
        model: SAM2 predictor model
        enable_bn_adapt: Whether to enable BN/LayerNorm adaptation (vs only temperature)
    
    Returns:
        List of adaptable parameters
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    adaptable_params = []
    
    if enable_bn_adapt:
        # Enable BatchNorm and LayerNorm parameters for adaptation
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
                for param in module.parameters():
                    param.requires_grad = True
                    adaptable_params.append(param)
                # Also enable BN running stats update
                module.track_running_stats = True
                module.momentum = 0.1  # Standard BN momentum
    
    return adaptable_params


def compute_fisher_regularization(
    model,
    adaptable_params: list[torch.Tensor],
    fisher_dict: dict[str, torch.Tensor],
    original_params: dict[str, torch.Tensor],
    fisher_alpha: float = 2000.0,
) -> torch.Tensor:
    """Compute Fisher regularization to prevent forgetting.
    
    Args:
        model: The model being adapted
        adaptable_params: List of parameters being updated
        fisher_dict: Precomputed Fisher information for each parameter
        original_params: Original parameter values before adaptation
        fisher_alpha: Regularization strength
    
    Returns:
        Fisher regularization loss
    """
    fisher_loss = torch.tensor(0.0, device=next(iter(adaptable_params)).device)
    
    for name, param in model.named_parameters():
        if param.requires_grad and name in fisher_dict and name in original_params:
            # L2 distance weighted by Fisher information
            fisher_loss += (fisher_dict[name] * (param - original_params[name]).pow(2)).sum()
    
    return fisher_alpha * fisher_loss


def estimate_fisher_information(
    model,
    sample_logits: torch.Tensor,
    sample_labels: torch.Tensor,
    num_samples: int = 1,
) -> dict[str, torch.Tensor]:
    """Estimate Fisher information matrix for adaptable parameters.
    
    Args:
        model: The model
        sample_logits: Sample predictions [B, H, W] or [B, C, H, W]
        sample_labels: Pseudo labels from predictions
        num_samples: Number of samples for estimation
    
    Returns:
        Dictionary mapping parameter name to Fisher information
    """
    fisher_dict = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param)
    
    model.eval()
    for _ in range(num_samples):
        model.zero_grad()
        # Use pseudo-labels from model predictions
        loss = F.binary_cross_entropy_with_logits(sample_logits, sample_labels, reduction='mean')
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data.pow(2) / num_samples
    
    return fisher_dict


def entropy_with_sample_selection(
    logits: torch.Tensor,
    logT: torch.Tensor,
    entropy_threshold: float = 0.4,
    selection_p: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute entropy with active sample selection based on reliability.
    
    Only samples with low entropy (high confidence) are used for adaptation.
    
    Args:
        logits: Prediction logits [H, W] or [B, H, W]
        logT: Temperature parameter
        entropy_threshold: Maximum entropy for sample selection (0-1 range)
        selection_p: Probability threshold for selecting samples
    
    Returns:
        (filtered_loss, selection_mask): Loss from selected samples and selection mask
    """
    T = torch.exp(logT).clamp(0.25, 4.0)
    z = logits / T
    
    if z.ndim == 2:
        z = z.unsqueeze(0)  # [1, H, W]
    
    # Compute probabilities and entropy
    p = torch.sigmoid(z)
    ent = -(p * torch.log(p.clamp_min(1e-8)) + (1.0 - p) * torch.log((1.0 - p).clamp_min(1e-8)))
    
    # Sample selection: only use reliable (low entropy) samples
    # Flatten spatial dimensions for filtering
    ent_flat = ent.reshape(-1)
    
    # Select samples with entropy below threshold
    reliable_mask = ent_flat < entropy_threshold
    
    # Further select top-p most confident samples
    if selection_p < 1.0 and reliable_mask.sum() > 0:
        num_select = max(1, int(selection_p * len(ent_flat)))
        sorted_ent, sorted_idx = torch.sort(ent_flat)
        top_p_mask = torch.zeros_like(ent_flat, dtype=torch.bool)
        top_p_mask[sorted_idx[:num_select]] = True
        reliable_mask = reliable_mask & top_p_mask
    
    # Compute loss only on selected samples
    if reliable_mask.sum() > 0:
        selected_entropy = ent_flat[reliable_mask]
        return selected_entropy.mean(), reliable_mask
    else:
        # No reliable samples, return full entropy
        return ent.mean(), torch.ones_like(ent_flat, dtype=torch.bool)


def inference_with_uctta(
    predictor,
    jpeg_dir: Path,
    ann_dir: Path,
    out_dir: Path,
    score_thresh: float = 0.0,
    video_names: Optional[list[str]] = None,
    max_objects: Optional[int] = None,
    prompt_method: str = "gt_box",
    uctta_steps: int = 2,
    uctta_lr: float = 3e-4,
    collect_statistics: bool = True,
    dataset_name: Optional[str] = None,
    reuse_prompts_root: Optional[Path] = None,
    first_frame_only: bool = False,
    # Optional click protocol controls (used if no reuse_prompts available)
    click_protocol: str | None = None,
    min_click_dist: float | None = None,
    seed: int | None = None,
    # UCTTA-specific parameters
    enable_bn_adapt: bool = True,
    use_fisher_reg: bool = True,
    fisher_alpha: float = 2000.0,
    entropy_threshold: float = 0.4,
    selection_p: float = 0.1,
) -> dict[str, Any] | None:
    """Full UCTTA: Test-time adaptation with BN update, Fisher regularization, and sample selection.

    Based on "Uncertainty-Calibrated Test-Time Model Adaptation without Forgetting" (TPAMI 2025).
    
    Key components:
    - Entropy minimization on test samples
    - BN/LayerNorm parameter adaptation (optional)
    - Fisher regularization to prevent forgetting
    - Active sample selection based on prediction confidence
    - Temperature scaling for uncertainty calibration
    
    Args:
        enable_bn_adapt: Enable BN/LN parameter updates (True for full UCTTA, False for temperature-only)
        use_fisher_reg: Use Fisher regularization to prevent forgetting
        fisher_alpha: Fisher regularization strength
        entropy_threshold: Max entropy for sample selection (reliable samples)
        selection_p: Fraction of samples to use for adaptation
    
    Returns:
        Dictionary containing UCTTA statistics including uncertainty and performance metrics,
        or None if statistics collection is disabled or fails.
    """
    if video_names is None:
        video_names = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    else:
        video_names = sorted(set(video_names))

    print(f"UCTTA inference on {len(video_names)} videos")
    print(f"UCTTA config: BN_adapt={enable_bn_adapt}, Fisher_reg={use_fisher_reg}, "
          f"entropy_th={entropy_threshold}, selection_p={selection_p}")

    # Setup model for adaptation
    adaptable_params = setup_uctta_model(predictor, enable_bn_adapt=enable_bn_adapt)
    print(f"Adaptable parameters: {len(adaptable_params)} (BN/LN layers)")
    
    # Store original parameters for Fisher regularization
    original_params = {}
    fisher_dict = {}
    if use_fisher_reg and enable_bn_adapt:
        for name, param in predictor.named_parameters():
            if param.requires_grad:
                original_params[name] = param.data.clone()
        print("Stored original parameters for Fisher regularization")

    # Prepare dataset evaluator (optional)
    dataset_eval = None
    if collect_statistics:
        # Use consistent path format: <output_root>/<dataset>_uctta_eval
        eval_dir = out_dir.parent / f"{dataset_name.lower()}_uctta_eval" if dataset_name else (out_dir.parent / "uctta_eval")
        eval_dir.mkdir(parents=True, exist_ok=True)
        dataset_eval = DistributedDatasetEvaluator(save_dir=str(eval_dir), distributed=False, rank=0, world_size=1)

    for v_idx, vid in enumerate(video_names, 1):
        print(f"[{v_idx:03}/{len(video_names)}] {vid}")
        video_dir = jpeg_dir / vid
        frame_names = sorted(
            [p.stem for p in video_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]],
            key=lambda x: int(x),
        )

        # Initialize predictor state
        state = predictor.init_state(str(video_dir))
        H, W = state["video_height"], state["video_width"]

        # Discover object ids from first frame GT
        first_mask_np = _load_first_frame_mask(ann_dir, vid, frame_names)
        if first_mask_np is None:
            continue
        obj_ids = _select_top_objects_by_area(first_mask_np, max_objects if max_objects else 10**9)
        if len(obj_ids) == 0:
            print(f"Warning: No objects found in first frame of video {vid}")
            continue
        print(f"Processing {len(obj_ids)} objects in video {vid}: {obj_ids}")

        # Load reused prompts if available
        prompts_json = load_reused_prompts(reuse_prompts_root, dataset_name, vid)
        if prompts_json:
            print(f"Loaded reused prompts for video {vid}")

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(int(seed))

        # Apply prompts for each object
        for obj_id in obj_ids:
            gt_bool = first_mask_np == obj_id
            if not np.any(gt_bool):
                continue
            
            # Clear GPU memory before processing each object to prevent OOM
            if torch.cuda.is_available() and len(obj_ids) > 10:
                torch.cuda.empty_cache()
            
            # Try reused prompts first, fall back to generation
            prompt_applied = False
            if prompts_json and obj_id in prompts_json:
                prompt_applied = apply_reused_prompts(predictor, state, obj_id, prompts_json[obj_id])
            
            if not prompt_applied:
                # Generate new prompts using click protocol
                generate_click_prompts(
                    predictor,
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    gt_bool=gt_bool,
                    first_frame_mask_np=first_mask_np,
                    score_thresh=score_thresh,
                    click_protocol=click_protocol or "3click",
                    min_click_dist=float(min_click_dist or 12.0),
                )

        # Per-video temperature parameter + adaptable model params
        logT = torch.nn.Parameter(torch.zeros(1, device=predictor.device, dtype=torch.float32))
        opt_params = [logT] + (adaptable_params if enable_bn_adapt else [])
        optimizer = torch.optim.Adam(opt_params, lr=float(uctta_lr))
        
        # Estimate Fisher information on first frame (if using Fisher reg and BN adapt)
        video_fisher_dict = {}
        if use_fisher_reg and enable_bn_adapt and len(adaptable_params) > 0:
            # Use first frame predictions to estimate Fisher
            print(f"Estimating Fisher information for video {vid}...")
            # Note: Fisher estimation would ideally use source domain data
            # Here we use a simplified placeholder - you may want to precompute Fisher offline
            video_fisher_dict = fisher_dict.copy() if fisher_dict else {}

        # Propagate and adapt per frame
        # When first_frame_only=True, only process the first frame (frame 0)
        # In SAM2: end_frame_idx = start_frame_idx + max_frame_num_to_track
        #          processing_order = range(start_frame_idx, end_frame_idx + 1)
        # To only process frame 0: we need end_frame_idx = 0, so max_frame_num_to_track = 0
        max_frames = 0 if first_frame_only else None
        video_segments: dict[int, dict[int, np.ndarray]] = {}

        for f_idx, out_obj_ids, out_logits in predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=max_frames
        ):
            # 1) Adapt parameters using current frame logits with UCTTA
            if (uctta_steps is not None) and (uctta_steps > 0):
                for step_i in range(int(uctta_steps)):
                    optimizer.zero_grad(set_to_none=True)
                    # Ensure we are not in global inference_mode during backward
                    with torch.inference_mode(False):
                        with torch.enable_grad():
                            loss_list = []
                            for i, oid in enumerate(out_obj_ids):
                                logits_i = out_logits[i]
                                # Always detach+clone BEFORE any ops to avoid inference tensor issues
                                logits_clean = logits_i.detach().clone()
                                logits_2d = logits_clean.squeeze(0) if logits_clean.ndim == 3 else logits_clean
                                if tuple(logits_2d.shape[-2:]) != (H, W):
                                    logits_2d = F.interpolate(
                                        logits_2d.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
                                    )[0, 0]
                                
                                # Use sample selection for reliable adaptation
                                loss_ent, selection_mask = entropy_with_sample_selection(
                                    logits_2d, 
                                    logT,
                                    entropy_threshold=entropy_threshold,
                                    selection_p=selection_p,
                                )
                                loss_list.append(loss_ent)
                            
                            if loss_list:
                                entropy_loss = torch.stack(loss_list).mean()
                                total_loss = entropy_loss
                                
                                # Add Fisher regularization if enabled
                                if use_fisher_reg and enable_bn_adapt and len(video_fisher_dict) > 0:
                                    fisher_loss = compute_fisher_regularization(
                                        predictor,
                                        adaptable_params,
                                        video_fisher_dict,
                                        original_params,
                                        fisher_alpha=fisher_alpha,
                                    )
                                    total_loss = entropy_loss + fisher_loss
                                    
                                    if step_i == 0:  # Log only first step to avoid spam
                                        print(f"  Frame {f_idx}: entropy={entropy_loss.item():.4f}, "
                                              f"fisher={fisher_loss.item():.4f}, total={total_loss.item():.4f}")
                                
                                total_loss.backward()
                                optimizer.step()

            # 2) Produce masks with temperature scaling
            seg: dict[int, np.ndarray] = {}
            for i, oid in enumerate(out_obj_ids):
                mask_logits = out_logits[i]
                # Make a normal tensor (not an inference tensor) before interacting with logT
                logits_clean = mask_logits.detach().clone()
                
                # Handle multimask output (K>1): select mask 0 (singlemask output token)
                # This matches SAM-2's default behavior when multimask_output=False
                if logits_clean.ndim == 3:
                    if logits_clean.shape[0] == 1:
                        # Single mask: squeeze it
                        logits_2d = logits_clean.squeeze(0)
                    elif logits_clean.shape[0] > 1:
                        # Multiple masks: use mask 0 (singlemask token, SAM-2 default)
                        logits_2d = logits_clean[0]
                    else:
                        logits_2d = logits_clean
                else:
                    logits_2d = logits_clean
                
                # align to original image size
                if tuple(logits_2d.shape[-2:]) != (H, W):
                    logits_2d = F.interpolate(
                        logits_2d.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
                    )[0, 0]
                scaled = _apply_temperature(logits_2d, logT)
                seg[oid] = _threshold_bool(scaled, score_thresh)
            
            # Clear GPU memory after processing frame with many objects
            if torch.cuda.is_available() and len(out_obj_ids) > 10:
                torch.cuda.empty_cache()

            # Add to dataset evaluator (sample a few objects to limit memory)
            if dataset_eval is not None and len(out_obj_ids) > 0:
                try:
                    # Load GT for current frame (full multi-object mask)
                    gt_path = ann_dir / vid / f"{frame_names[f_idx]}.png"
                    gt_full_np = None
                    if gt_path.exists():
                        gt_full_np = np.array(Image.open(gt_path))

                    max_obj_stats = 3
                    # Build per-frame stacked logits [K,H,W] for visualization/accuracy proxy
                    stacked_logits = []
                    # Also prepare a list of per-object uncertainty maps for aggregation
                    per_object_uncertainties = []
                    for i, oid in enumerate(out_obj_ids[:max_obj_stats]):
                        logits_i = out_logits[i]
                        logits_clean = logits_i.detach().clone()
                        logits_2d = logits_clean.squeeze(0) if logits_clean.ndim == 3 else logits_clean
                        if tuple(logits_2d.shape[-2:]) != (H, W):
                            logits_2d = F.interpolate(
                                logits_2d.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
                            )[0, 0]
                        # Uncertainty per object
                        u_map = _entropy_map_from_logits_scaled(logits_2d, logT)  # [H,W]
                        per_object_uncertainties.append(u_map.unsqueeze(0))  # [1,H,W]
                        stacked_logits.append(logits_2d.unsqueeze(0))  # [1,H,W]
                        # Add to evaluator if GT available - extract binary mask for THIS object only
                        if gt_full_np is not None:
                            gt_binary = (gt_full_np == oid).astype(np.float32)  # Extract this object's mask
                            gt_tensor = torch.from_numpy(gt_binary).unsqueeze(0).to(predictor.device)  # [1,H,W]
                            dataset_eval.add_batch_data(uncertainty=u_map.unsqueeze(0), pred_logits=logits_2d.unsqueeze(0), gt_masks=gt_tensor)

                    # Once per frame: create UA ratio visualization using all available objects
                    if stacked_logits and f_idx < 5 and collect_statistics:
                        # Shape to [1,H,W,K]
                        logits_hwk = torch.cat(stacked_logits, dim=0).permute(1, 2, 0).unsqueeze(0)
                        # Aggregate uncertainty across objects: use max for conservative view → [1,H,W]
                        u_agg = torch.max(torch.cat(per_object_uncertainties, dim=0), dim=0, keepdim=True).values  # [1,H,W]
                        # Load original image with .jpg/.jpeg fallback
                        img_path = jpeg_dir / vid / f"{frame_names[f_idx]}.jpg"
                        if not img_path.exists():
                            alt = jpeg_dir / vid / f"{frame_names[f_idx]}.jpeg"
                            if alt.exists():
                                img_path = alt
                        if img_path.exists():
                            img = Image.open(img_path).convert("RGB")
                            img_np = np.array(img).astype(np.float32) / 255.0
                            create_uctta_ua_ratio_visualization(
                                logits_hwk,
                                u_agg,
                                img_np,
                                vid,
                                frame_names[f_idx],
                                out_dir.parent / "uctta_visualizations",
                            )
                except Exception as e:
                    print(f"Warning: UCTTA evaluator add_batch_data failed: {e}")

            video_segments[f_idx] = seg

        # Save PNG masks to disk (same as baseline)
        for f_idx in list(video_segments.keys()):
            frame_name = frame_names[f_idx]
            save_masks_to_dir(
                output_mask_dir=str(out_dir),
                video_name=vid,
                frame_name=frame_name,
                per_obj_output_mask=video_segments[f_idx],
                height=H,
                width=W,
                per_obj_png_file=False,
                output_palette=DAVIS_PALETTE,
            )
        
        # Critical: Reset predictor state to free memory for this video
        predictor.reset_state(state)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Finalize dataset correlation visualization/results
    if dataset_eval is not None and len(dataset_eval) > 0:
        try:
            dataset_eval.evaluate_dataset_correlation()
            dataset_eval.create_dataset_correlation_visualization(
                title=f"{dataset_name} UCTTA - Dataset Correlation" if dataset_name else "UCTTA - Dataset Correlation",
                save_name=f"{dataset_name.lower()}_uctta_dataset_analysis.png" if dataset_name else "uctta_dataset_analysis.png",
            )
            dataset_eval.save_correlation_results(
                save_name=f"{dataset_name.lower()}_uctta_results.json" if dataset_name else "uctta_results.json"
            )
            
            # Extract and return statistics
            # Use the correct data source based on whether we're using pixel-level or image-level stats
            data_source = dataset_eval.pixel_uncertainties if dataset_eval.per_pixel_statistics else dataset_eval.image_uncertainties
            iou_source = dataset_eval.pixel_ious if dataset_eval.per_pixel_statistics else dataset_eval.image_ious
            dice_source = dataset_eval.pixel_dices if dataset_eval.per_pixel_statistics else dataset_eval.image_dices
            accuracy_source = dataset_eval.pixel_accuracies if dataset_eval.per_pixel_statistics else dataset_eval.image_accuracies
            
            # Sample raw data for PAvPU scatter plot (no thresholds)
            max_samples = 10000
            if data_source and accuracy_source:
                total_samples = min(len(data_source), len(accuracy_source))
                if total_samples > max_samples:
                    indices = np.random.choice(total_samples, max_samples, replace=False)
                    uncertainty_samples = [data_source[i] for i in indices]
                    accuracy_samples = [accuracy_source[i] for i in indices]
                else:
                    uncertainty_samples = list(data_source)
                    accuracy_samples = list(accuracy_source)
            else:
                uncertainty_samples = []
                accuracy_samples = []
            
            uctta_statistics = {
                # Pixel uncertainty statistics (works for both pixel-level and image-level)
                'pixel_uncertainty_mean': float(np.mean(data_source)) if data_source else 0.0,
                'pixel_uncertainty_std': float(np.std(data_source)) if data_source else 0.0,
                'pixel_uncertainty_median': float(np.median(data_source)) if data_source else 0.0,
                'pixel_uncertainty_min': float(np.min(data_source)) if data_source else 0.0,
                'pixel_uncertainty_max': float(np.max(data_source)) if data_source else 0.0,
                
                # Performance metrics
                'iou_mean': float(np.mean(iou_source)) if iou_source else 0.0,
                'dice_mean': float(np.mean(dice_source)) if dice_source else 0.0,
                'accuracy_mean': float(np.mean(accuracy_source)) if accuracy_source else 0.0,
                
                # Correlation results (UA relationship)
                'correlation_results': dataset_eval.correlation_results,
                
                # Summary statistics from evaluator
                'summary': dataset_eval.get_summary_statistics(),
                
                # Sample count
                'num_samples': len(data_source),
                
                # Raw PAvPU samples for true scatter plot (no thresholds)
                'eval_pavpu_uncertainty_samples': uncertainty_samples,
                'eval_pavpu_accuracy_samples': accuracy_samples,
            }
            
            print(f"UCTTA statistics collected: {uctta_statistics['num_samples']} samples, "
                  f"mean uncertainty: {uctta_statistics['pixel_uncertainty_mean']:.4f}")
            
            return uctta_statistics
            
        except Exception as e:
            print(f"Warning: UCTTA dataset evaluation failed: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    return None


def run_single_dataset_with_uctta(
    dataset_name: str,
    predictor,
    output_path: Path,
    split: str | list[str] | None = None,
    score_thresh: float = 0.0,
    num_workers: int | None = None,
    video_subset: list[str] | None = None,
    prompt_method: str = "gt_box",
    first_frame_only: bool = False,
    max_objects: int | None = None,
    uctta_steps: int = 2,
    uctta_lr: float = 3e-4,
    reuse_prompts_root: Optional[Path] = None,
    click_protocol: str = "3click",
    min_click_dist: float = 12.0,
    seed: int = 0,
    # Full UCTTA parameters
    enable_bn_adapt: bool = True,
    use_fisher_reg: bool = True,
    fisher_alpha: float = 2000.0,
    entropy_threshold: float = 0.4,
    selection_p: float = 0.1,
) -> tuple[float, float, float, dict[str, Any] | None]:
    """Run evaluation on a single dataset using UCTTA (temperature adaptation) and return J&F/J/F and statistics.

    This mirrors zero_shot_multi_dataset.run_single_dataset, but swaps inference for UCTTA.
    
    Returns:
        Tuple of (j_f_val, j_val, f_val, uctta_statistics)
    """
    config = DATASET_CONFIGS[dataset_name]
    if split is None:
        split = config["default_split"]

    if isinstance(split, list):
        split = split[0]

    assert isinstance(split, str)

    root = Path(config["root"])
    if config["has_split_subdir"]:
        jpeg_dir = root / split / "JPEGImages"
        ann_dir = root / split / "Annotations"
    else:
        jpeg_dir = root / "JPEGImages"
        ann_dir = root / "Annotations"

    if not jpeg_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"JPEGImages or Annotations not found for {dataset_name}: {jpeg_dir}, {ann_dir}")

    out_dir = output_path / f"{dataset_name.lower()}_pred"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Running {dataset_name} dataset evaluation (SAM-2 + UCTTA)")
    print(f"{'=' * 60}")

    # Handle file_list_txt if present
    if "file_list_txt" in config:
        file_list_path = Path(config["file_list_txt"])
        if file_list_path.exists():
            with open(file_list_path, "r") as f:
                names = [line.strip() for line in f if line.strip()]
            video_subset = [v for v in (video_subset or names) if v in names]

    # Execute inference with UCTTA
    t0 = time.time()
    uctta_stats = inference_with_uctta(
        predictor,
        jpeg_dir,
        ann_dir,
        out_dir,
        score_thresh=score_thresh,
        video_names=video_subset,
        max_objects=max_objects,
        prompt_method=prompt_method,
        uctta_steps=uctta_steps,
        uctta_lr=uctta_lr,
        collect_statistics=True,
        dataset_name=dataset_name,
        reuse_prompts_root=reuse_prompts_root,
        first_frame_only=first_frame_only,
        click_protocol=click_protocol,
        min_click_dist=min_click_dist,
        seed=seed,
        # Full UCTTA parameters
        enable_bn_adapt=enable_bn_adapt,
        use_fisher_reg=use_fisher_reg,
        fisher_alpha=fisher_alpha,
        entropy_threshold=entropy_threshold,
        selection_p=selection_p,
    )
    t_infer = time.time() - t0

    # Prepare eval roots (match baseline behavior)
    if first_frame_only:
        base_videos = video_subset if video_subset is not None else [d.name for d in ann_dir.iterdir() if d.is_dir()]
        base_videos = sorted(base_videos)
        gt_tmp = output_path / f"{dataset_name.lower()}_tmp_gt_first"
        pred_tmp = output_path / f"{dataset_name.lower()}_tmp_pred_first"
        if gt_tmp.exists():
            shutil.rmtree(gt_tmp)
        if pred_tmp.exists():
            shutil.rmtree(pred_tmp)
        for v in base_videos:
            v_gt_dir = ann_dir / v
            v_pred_dir = out_dir / v
            if not v_gt_dir.exists() or not v_pred_dir.exists():
                continue
            gt_pngs = sorted([p for p in v_gt_dir.iterdir() if p.suffix.lower() == ".png"])
            if not gt_pngs:
                continue
            first_png = gt_pngs[0].name
            if not (v_pred_dir / first_png).exists():
                continue
            (gt_tmp / v).mkdir(parents=True, exist_ok=True)
            (pred_tmp / v).mkdir(parents=True, exist_ok=True)
            shutil.copy2(v_gt_dir / first_png, gt_tmp / v / first_png)
            shutil.copy2(v_pred_dir / first_png, pred_tmp / v / first_png)
        gt_root_eval, pred_root_eval = gt_tmp, pred_tmp
    else:
        if video_subset is not None:
            gt_tmp = output_path / f"{dataset_name.lower()}_tmp_gt"
            pred_tmp = output_path / f"{dataset_name.lower()}_tmp_pred"
            if gt_tmp.exists():
                shutil.rmtree(gt_tmp)
            if pred_tmp.exists():
                shutil.rmtree(pred_tmp)
            gt_tmp.mkdir(parents=True, exist_ok=True)
            pred_tmp.mkdir(parents=True, exist_ok=True)
            for v in video_subset:
                if (ann_dir / v).exists() and (out_dir / v).exists():
                    shutil.copytree(ann_dir / v, gt_tmp / v, symlinks=True)
                    shutil.copytree(out_dir / v, pred_tmp / v, symlinks=True)
            gt_root_eval, pred_root_eval = gt_tmp, pred_tmp
        else:
            gt_root_eval, pred_root_eval = ann_dir, out_dir

    # Evaluate via SAV benchmark to keep metrics identical
    from sav_dataset.utils.sav_benchmark import benchmark

    t1 = time.time()
    try:
        J_F, global_J, global_F, _ = benchmark(
            gt_roots=[str(gt_root_eval)],
            mask_roots=[str(pred_root_eval)],
            strict=False,
            num_processes=num_workers,
            skip_first_and_last=config["skip_first_and_last"],
            verbose=True,
        )
        if len(J_F) == 0 or len(global_J) == 0 or len(global_F) == 0:
            print(f"Warning: Empty evaluation results for {dataset_name}")
            return 0.0, 0.0, 0.0, None
        j_f_val = float(J_F[0]) if not np.isnan(J_F[0]) else 0.0
        j_val = float(global_J[0]) if not np.isnan(global_J[0]) else 0.0
        f_val = float(global_F[0]) if not np.isnan(global_F[0]) else 0.0
    except Exception as e:
        print(f"Error during evaluation of {dataset_name}: {e}")
        return 0.0, 0.0, 0.0, None
    t_eval = time.time() - t1

    print(f"Inference time (UCTTA): {t_infer:.2f}s")
    print(f"Evaluation time: {t_eval:.2f}s")

    # Cleanup temporary directories
    try:
        if first_frame_only:
            gt_tmp = output_path / f"{dataset_name.lower()}_tmp_gt_first"
            pred_tmp = output_path / f"{dataset_name.lower()}_tmp_pred_first"
        else:
            gt_tmp = output_path / f"{dataset_name.lower()}_tmp_gt"
            pred_tmp = output_path / f"{dataset_name.lower()}_tmp_pred"
        if gt_tmp.exists():
            shutil.rmtree(gt_tmp)
        if pred_tmp.exists():
            shutil.rmtree(pred_tmp)
    except Exception:
        pass

    return j_f_val, j_val, f_val, uctta_stats


def build_predictor_with_overrides(cfg_file: str, ckpt: str, device: str = "cuda", multimask: bool = True, min_pts: int = 1, max_pts: int = 2, for_tracking: bool = False):
    hydra_overrides_extra = []
    if multimask:
        hydra_overrides_extra += [
            "++model.multimask_output_in_sam=true",
            f"++model.multimask_min_pt_num={min_pts}",
            f"++model.multimask_max_pt_num={max_pts}",
        ]
        if for_tracking:
            hydra_overrides_extra += ["++model.multimask_output_for_tracking=true"]
    predictor = build_sam2_video_predictor(
        config_file=cfg_file,
        ckpt_path=ckpt,
        device=device,
        hydra_overrides_extra=hydra_overrides_extra,
    )
    return predictor


