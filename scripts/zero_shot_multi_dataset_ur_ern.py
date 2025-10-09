#!/usr/bin/env python
# Multi-dataset Zero-shot evaluation of SAM-2 with UR-ERN (Uncertainty Regularized Evidential Regression)
# Based on "Uncertainty Regularized Evidential Regression" (AAAI 2024)

import shutil
from pathlib import Path
from typing import Any, Optional
import os
import sys

import numpy as np
import torch
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


def create_ur_ern_ua_ratio_visualization(
    out_logits, uncertainty_map, original_img, vid, frame_name, vis_dir
):
    """Create U/A ratio visualization for UR-ERN"""
    from visualization_utils import VisualizationUtils
    from bndl_visualizer import BNDLVisualizer
    
    viz_utils = VisualizationUtils()
    bndl_viz = BNDLVisualizer()
    
    # Prepare data in format expected by visualizer
    ur_ern_outputs = {
        "pixel_uncertainty": uncertainty_map,
        "mean_pixel_logits": out_logits,
    }
    
    # Create figure with U/A ratio visualization
    fig, axes = viz_utils.create_figure_layout(1, 3, (18, 6))
    bndl_viz.plot_uncertainty_accuracy_ratio_visualization(
        axes[0, :], ur_ern_outputs, original_img, step_index=0, ratio_type="U/A"
    )
    
    # Save visualization
    save_path = vis_dir / vid / f"{frame_name}_ur_ern_ua_ratio.png"
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


def extract_ur_ern_outputs(outputs):
    """Extract UR-ERN outputs from model outputs"""
    if "multistep_aux_outputs" in outputs:
        aux_list = outputs["multistep_aux_outputs"]
        # Use the last valid UR-ERN output (highest resolution)
        for i in reversed(range(len(aux_list))):
            if aux_list[i] is not None and isinstance(aux_list[i], dict):
                ur_ern_outputs = aux_list[i].get("ur_ern", None)
                if ur_ern_outputs is not None:
                    return ur_ern_outputs, i
    return None, None


def _squeeze_to_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Squeeze tensor to 2D shape [H, W]."""
    if tensor.ndim == 4:
        return tensor.squeeze(0).squeeze(0)
    if tensor.ndim == 3:
        return tensor.squeeze(0)
    return tensor


def _ensure_tensor(value) -> torch.Tensor:
    """Convert value to tensor if needed."""
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)


def extract_ur_ern_uncertainty(ur_ern_outputs):
    """Extract uncertainty from UR-ERN outputs.
    
    Returns None if outputs are invalid or missing required parameters.
    Raises exception for unexpected errors to make issues visible.
    """
    if not isinstance(ur_ern_outputs, dict):
        return None
    
    # Extract and validate NIG parameters
    param_names = ["nig_mu", "nig_v", "nig_alpha", "nig_beta"]
    params = {name: ur_ern_outputs.get(name) for name in param_names}
    
    if any(p is None for p in params.values()):
        return None
    
    # Convert to tensors and squeeze to 2D
    nig_mu = _squeeze_to_2d(_ensure_tensor(params["nig_mu"]))
    nig_v = _squeeze_to_2d(_ensure_tensor(params["nig_v"]))
    nig_alpha = _squeeze_to_2d(_ensure_tensor(params["nig_alpha"]))
    nig_beta = _squeeze_to_2d(_ensure_tensor(params["nig_beta"]))
    
    # Validate shapes match
    if not (nig_mu.shape == nig_v.shape == nig_alpha.shape == nig_beta.shape):
        return None
    
    # Calculate total uncertainty: Var[y|x] = β(1+ν)/(ν(α-1))
    eps = 1e-6
    alpha_safe = torch.clamp(nig_alpha, min=1.0 + 1e-3)
    v_safe = torch.clamp(nig_v, min=eps)
    beta_safe = torch.clamp(nig_beta, min=eps)
    
    variance = beta_safe * (1.0 + v_safe) / (v_safe * (alpha_safe - 1.0))
    
    # Handle numerical issues
    variance = torch.nan_to_num(variance, nan=1e3, posinf=1e3, neginf=1e3)
    
    return torch.sqrt(variance)


def extract_ur_ern_aleatoric_epistemic(ur_ern_outputs):
    """Extract aleatoric and epistemic uncertainty from UR-ERN outputs"""
    if not isinstance(ur_ern_outputs, dict):
        return None, None
    
    # Extract NIG parameters
    nig_mu = ur_ern_outputs.get("nig_mu")
    nig_v = ur_ern_outputs.get("nig_v")
    nig_alpha = ur_ern_outputs.get("nig_alpha")
    nig_beta = ur_ern_outputs.get("nig_beta")
    
    if any(x is None for x in [nig_mu, nig_v, nig_alpha, nig_beta]):
        return None, None
    
    # Convert to tensors if needed
    if not isinstance(nig_mu, torch.Tensor):
        nig_mu = torch.tensor(nig_mu)
    if not isinstance(nig_v, torch.Tensor):
        nig_v = torch.tensor(nig_v)
    if not isinstance(nig_alpha, torch.Tensor):
        nig_alpha = torch.tensor(nig_alpha)
    if not isinstance(nig_beta, torch.Tensor):
        nig_beta = torch.tensor(nig_beta)
    
    # Ensure correct shape [H, W]
    if nig_mu.ndim == 4:
        nig_mu = nig_mu.squeeze(0).squeeze(0)
    if nig_v.ndim == 4:
        nig_v = nig_v.squeeze(0).squeeze(0)
    if nig_alpha.ndim == 4:
        nig_alpha = nig_alpha.squeeze(0).squeeze(0)
    if nig_beta.ndim == 4:
        nig_beta = nig_beta.squeeze(0).squeeze(0)
    
    # Calculate aleatoric uncertainty: β/(α-1)
    alpha_clamped = torch.clamp(nig_alpha, min=1.0 + 1e-3)
    aleatoric_var = nig_beta / (alpha_clamped - 1.0)
    aleatoric_unc = torch.sqrt(aleatoric_var)
    
    # Calculate epistemic uncertainty: β/(ν(α-1))
    epistemic_var = nig_beta / (nig_v * (alpha_clamped - 1.0))
    epistemic_unc = torch.sqrt(epistemic_var)
    
    return aleatoric_unc, epistemic_unc


def log_ur_ern_statistics(ur_ern_outputs, step, phase, dataset_name, statistics_dict=None):
    """Log UR-ERN statistics for analysis"""
    if statistics_dict is None:
        statistics_dict = {}
    
    if not isinstance(ur_ern_outputs, dict):
        return statistics_dict
    
    # Extract uncertainty
    uncertainty = extract_ur_ern_uncertainty(ur_ern_outputs)
    if uncertainty is not None:
        uncertainty_np = uncertainty.detach().cpu().numpy()
        
        # Log uncertainty statistics
        key_prefix = f"ur_ern_pixel_uncertainty_{phase}"
        statistics_dict[f"{key_prefix}_mean"] = float(np.mean(uncertainty_np))
        statistics_dict[f"{key_prefix}_std"] = float(np.std(uncertainty_np))
        statistics_dict[f"{key_prefix}_median"] = float(np.median(uncertainty_np))
        statistics_dict[f"{key_prefix}_min"] = float(np.min(uncertainty_np))
        statistics_dict[f"{key_prefix}_max"] = float(np.max(uncertainty_np))
    
    # Extract aleatoric and epistemic uncertainty
    aleatoric_unc, epistemic_unc = extract_ur_ern_aleatoric_epistemic(ur_ern_outputs)
    if aleatoric_unc is not None and epistemic_unc is not None:
        aleatoric_np = aleatoric_unc.detach().cpu().numpy()
        epistemic_np = epistemic_unc.detach().cpu().numpy()
        
        # Log aleatoric uncertainty
        aleatoric_prefix = f"ur_ern_aleatoric_uncertainty_{phase}"
        statistics_dict[f"{aleatoric_prefix}_mean"] = float(np.mean(aleatoric_np))
        statistics_dict[f"{aleatoric_prefix}_std"] = float(np.std(aleatoric_np))
        
        # Log epistemic uncertainty
        epistemic_prefix = f"ur_ern_epistemic_uncertainty_{phase}"
        statistics_dict[f"{epistemic_prefix}_mean"] = float(np.mean(epistemic_np))
        statistics_dict[f"{epistemic_prefix}_std"] = float(np.std(epistemic_np))
    
    return statistics_dict


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def inference_with_ur_ern(
    predictor,
    jpeg_dir: Path,
    ann_dir: Path,
    out_dir: Path,
    score_thresh: float = 0.0,
    video_names: list[str] | None = None,
    save_ur_ern_vis: bool = True,
    vis_dir: Path | None = None,
    dataset_name: str = "unknown",
    collect_statistics: bool = True,
    max_objects: int | None = None,
    prompt_method: str = "gt_box",
    first_frame_only: bool = False,
    reuse_prompts_root: Path | None = None,
    # Protocol controls (no GT box fallback)
    click_protocol: str = "3click",
    min_click_dist: float = 12.0,
    seed: int | None = 0,
):
    """Inference with UR-ERN uncertainty estimation"""
    if video_names is None:
        video_names = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    else:
        video_names = sorted(set(video_names))

    print(f"UR-ERN inference on {len(video_names)} videos")

    # Prepare dataset evaluator (optional)
    dataset_eval = None
    if collect_statistics:
        # Use consistent path format: <output_root>/<dataset>_ur_ern_eval
        eval_dir = out_dir.parent / f"{dataset_name.lower()}_ur_ern_eval" if dataset_name else (out_dir.parent / "ur_ern_eval")
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

        # Propagate and get predictions with UR-ERN uncertainty
        max_frames = 0 if first_frame_only else None
        video_segments: dict[int, dict[int, np.ndarray]] = {}

        for f_idx, out_obj_ids, out_logits in predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=max_frames
        ):
            # Produce masks
            seg: dict[int, np.ndarray] = {}
            for i, oid in enumerate(out_obj_ids):
                mask_logits = out_logits[i]
                if mask_logits.ndim == 3:
                    mask_logits = mask_logits.squeeze(0)
                # align to original image size
                if tuple(mask_logits.shape[-2:]) != (H, W):
                    mask_logits = F.interpolate(
                        mask_logits.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
                    )[0, 0]
                seg[oid] = _threshold_bool(mask_logits, score_thresh)

            # Add to dataset evaluator (sample a few objects to limit memory)
            if dataset_eval is not None and len(out_obj_ids) > 0:
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
                    logits_2d = logits_i.squeeze(0) if logits_i.ndim == 3 else logits_i
                    if tuple(logits_2d.shape[-2:]) != (H, W):
                        logits_2d = F.interpolate(
                            logits_2d.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
                        )[0, 0]
                    
                    # Get uncertainty for this object from UR-ERN outputs
                    obj_idx = predictor._obj_id_to_idx(state, oid)
                    ur_ern_outputs_per_obj = predictor.get_ur_ern_outputs(state, f_idx, obj_idx)
                    
                    uncertainty = None
                    if ur_ern_outputs_per_obj is not None:
                        uncertainty = extract_ur_ern_uncertainty(ur_ern_outputs_per_obj)
                        if uncertainty is not None:
                            per_object_uncertainties.append(uncertainty.unsqueeze(0))  # [1,H,W]
                    
                    stacked_logits.append(logits_2d.unsqueeze(0))  # [1,H,W]
                    # Add to evaluator if GT available - extract binary mask for THIS object only
                    if gt_full_np is not None:
                        gt_binary = (gt_full_np == oid).astype(np.float32)  # Extract this object's mask
                        gt_tensor = torch.from_numpy(gt_binary).unsqueeze(0).to(predictor.device)  # [1,H,W]
                        uncertainty_for_eval = uncertainty.unsqueeze(0) if uncertainty is not None else torch.zeros_like(logits_2d.unsqueeze(0))
                        dataset_eval.add_batch_data(uncertainty=uncertainty_for_eval, pred_logits=logits_2d.unsqueeze(0), gt_masks=gt_tensor)

                # Once per frame: create UA ratio visualization using all available objects
                if stacked_logits and f_idx < 5 and collect_statistics:
                    # Shape to [1,H,W,K]
                    logits_hwk = torch.cat(stacked_logits, dim=0).permute(1, 2, 0).unsqueeze(0)
                    # Aggregate uncertainty across objects: use max for conservative view → [1,H,W]
                    if per_object_uncertainties:
                        u_agg = torch.max(torch.cat(per_object_uncertainties, dim=0), dim=0, keepdim=True).values  # [1,H,W]
                    else:
                        u_agg = torch.zeros(1, H, W)
                    
                    # Load original image with .jpg/.jpeg fallback
                    img_path = jpeg_dir / vid / f"{frame_names[f_idx]}.jpg"
                    if not img_path.exists():
                        alt = jpeg_dir / vid / f"{frame_names[f_idx]}.jpeg"
                        if alt.exists():
                            img_path = alt
                    if img_path.exists():
                        img = Image.open(img_path).convert("RGB")
                        img_np = np.array(img).astype(np.float32) / 255.0
                        create_ur_ern_ua_ratio_visualization(
                            logits_hwk,
                            u_agg,
                            img_np,
                            vid,
                            frame_names[f_idx],
                            out_dir.parent / "ur_ern_visualizations",
                        )

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
        dataset_eval.evaluate_dataset_correlation()
        dataset_eval.create_dataset_correlation_visualization(
            title=f"{dataset_name} UR-ERN - Dataset Correlation" if dataset_name else "UR-ERN - Dataset Correlation",
            save_name=f"{dataset_name.lower()}_ur_ern_dataset_analysis.png" if dataset_name else "ur_ern_dataset_analysis.png",
        )
        dataset_eval.save_correlation_results(
            save_name=f"{dataset_name.lower()}_ur_ern_results.json" if dataset_name else "ur_ern_results.json"
        )
        
        # Extract and return statistics
        ur_ern_statistics = {
            # Pixel uncertainty statistics
            'pixel_uncertainty_mean': float(np.mean([u.item() for u in dataset_eval.image_uncertainties])) if dataset_eval.image_uncertainties else 0.0,
            'pixel_uncertainty_std': float(np.std([u.item() for u in dataset_eval.image_uncertainties])) if dataset_eval.image_uncertainties else 0.0,
            'pixel_uncertainty_median': float(np.median([u.item() for u in dataset_eval.image_uncertainties])) if dataset_eval.image_uncertainties else 0.0,
            'pixel_uncertainty_min': float(np.min([u.item() for u in dataset_eval.image_uncertainties])) if dataset_eval.image_uncertainties else 0.0,
            'pixel_uncertainty_max': float(np.max([u.item() for u in dataset_eval.image_uncertainties])) if dataset_eval.image_uncertainties else 0.0,
            
            # Performance metrics
            'iou_mean': float(np.mean([iou.item() for iou in dataset_eval.image_ious])) if dataset_eval.image_ious else 0.0,
            'dice_mean': float(np.mean([d.item() for d in dataset_eval.image_dices])) if dataset_eval.image_dices else 0.0,
            'accuracy_mean': float(np.mean([a.item() for a in dataset_eval.image_accuracies])) if dataset_eval.image_accuracies else 0.0,
            
            # Correlation results (UA relationship)
            'correlation_results': dataset_eval.correlation_results,
            
            # Summary statistics from evaluator
            'summary': dataset_eval.get_summary_statistics(),
            
            # Sample count
            'num_samples': len(dataset_eval.image_uncertainties),
        }
        
        print(f"UR-ERN statistics collected: {ur_ern_statistics['num_samples']} samples, "
              f"mean uncertainty: {ur_ern_statistics['pixel_uncertainty_mean']:.4f}")
        
        return ur_ern_statistics
    
    return None


def run_single_dataset_with_ur_ern(
    dataset_name: str,
    predictor,
    output_path: Path,
    split: str | list[str] | None = None,
    score_thresh: float = 0.0,
    num_workers: int | None = None,
    video_subset: list[str] | None = None,
    save_ur_ern_vis: bool = True,
    prompt_method: str = "gt_box",
    first_frame_only: bool = False,
    max_objects: int | None = None,
    collect_statistics: bool = False,
    reuse_prompts_root: Path | None = None,
    click_protocol: str = "3click",
    min_click_dist: float = 12.0,
    seed: int = 0,
) -> tuple[float, float, float, dict[str, Any] | None]:
    """Run evaluation on a single dataset using UR-ERN and return J&F/J/F and statistics.

    This mirrors zero_shot_multi_dataset.run_single_dataset, but uses UR-ERN for uncertainty estimation.
    
    Returns:
        Tuple of (j_f_val, j_val, f_val, ur_ern_statistics)
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
    print(f"Running {dataset_name} dataset evaluation (SAM-2 + UR-ERN)")
    print(f"{'=' * 60}")

    # Handle file_list_txt if present
    if "file_list_txt" in config:
        file_list_path = Path(config["file_list_txt"])
        if file_list_path.exists():
            with open(file_list_path, "r") as f:
                names = [line.strip() for line in f if line.strip()]
            video_subset = [v for v in (video_subset or names) if v in names]

    # Execute inference with UR-ERN
    t0 = time.time()
    ur_ern_stats = inference_with_ur_ern(
        predictor,
        jpeg_dir,
        ann_dir,
        out_dir,
        score_thresh=score_thresh,
        video_names=video_subset,
        max_objects=max_objects,
        prompt_method=prompt_method,
        save_ur_ern_vis=save_ur_ern_vis,
        collect_statistics=True,
        dataset_name=dataset_name,
        reuse_prompts_root=reuse_prompts_root,
        first_frame_only=first_frame_only,
        click_protocol=click_protocol,
        min_click_dist=min_click_dist,
        seed=seed,
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
    t_eval = time.time() - t1

    print(f"Inference time (UR-ERN): {t_infer:.2f}s")
    print(f"Evaluation time: {t_eval:.2f}s")

    # Cleanup temporary directories
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

    return j_f_val, j_val, f_val, ur_ern_stats


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


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-dataset zero-shot evaluation with UR-ERN")
    
    # Dataset selection
    parser.add_argument("--datasets", nargs="+", default=["GTEA"], choices=list(DATASET_CONFIGS.keys()))
    
    # Model configuration
    parser.add_argument("--config_file", default="configs/sam2.1/sam2.1_hiera_b+_ur_ern.yaml")
    parser.add_argument("--checkpoint", default="/home/hongyou/dev/ada_samp/logs/sam2/sam2_ur_ern_001_01/checkpoints/checkpoint.pt")
    parser.add_argument("--device", default="cuda")
    
    # Evaluation parameters
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument("--prompt_method", default="gt_box", choices=["gt_box", "three_clicks"])
    parser.add_argument("--first_frame_only", action="store_true")
    parser.add_argument("--max_objects", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=None)
    
    # Visualization and statistics
    parser.add_argument("--save_ur_ern_vis", action="store_true", default=True)
    parser.add_argument("--collect_statistics", action="store_true", default=True)
    
    # Click protocol
    parser.add_argument("--click_protocol", default="3click", choices=["1click", "3click", "5click"])
    parser.add_argument("--min_click_dist", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    
    # Output
    parser.add_argument("--output_path", type=Path, default=Path("./outputs/ur_ern_evaluation"))
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    args.output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("UR-ERN Zero-shot Evaluation")
    print("=" * 80)
    print(f"Datasets: {args.datasets}")
    print(f"Config: {args.config_file}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_path}")
    print("=" * 80)
    
    # Build predictor
    predictor = build_predictor_with_overrides(
        cfg_file=args.config_file,
        ckpt=args.checkpoint,
        device=args.device,
    )
    
    # Run evaluation on each dataset
    all_results = {}
    all_statistics = {}
    
    for dataset_name in args.datasets:
        print(f"\nEvaluating {dataset_name}...")
        
        j_f, j, f, stats = run_single_dataset_with_ur_ern(
            dataset_name=dataset_name,
            predictor=predictor,
            output_path=args.output_path,
            score_thresh=args.score_thresh,
            num_workers=args.num_workers,
            save_ur_ern_vis=args.save_ur_ern_vis,
            prompt_method=args.prompt_method,
            first_frame_only=args.first_frame_only,
            max_objects=args.max_objects,
            collect_statistics=args.collect_statistics,
            click_protocol=args.click_protocol,
            min_click_dist=args.min_click_dist,
            seed=args.seed,
        )
        
        all_results[dataset_name] = (j_f, j, f)
        if stats:
            all_statistics[dataset_name] = stats
        
        print(f"{dataset_name}: J&F={j_f:.2f}, J={j:.2f}, F={f:.2f}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for dataset_name, (j_f, j, f) in all_results.items():
        print(f"{dataset_name:15s}: J&F={j_f:6.2f}, J={j:6.2f}, F={f:6.2f}")
    
    if all_results:
        avg_jf = sum(r[0] for r in all_results.values()) / len(all_results)
        avg_j = sum(r[1] for r in all_results.values()) / len(all_results)
        avg_f = sum(r[2] for r in all_results.values()) / len(all_results)
        print(f"{'AVERAGE':15s}: J&F={avg_jf:6.2f}, J={avg_j:6.2f}, F={avg_f:6.2f}")
    
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()

