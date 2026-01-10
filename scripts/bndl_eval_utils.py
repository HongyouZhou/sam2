#!/usr/bin/env python
# Utilities for BNDL evaluation and PAvPU calculation
# Extracted from zero_shot_multi_dataset_sam_bndl.py

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# Add training utils to path for imports if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training", "utils"))

# from BNDL.BNDL_upload.ViT_Sparse.utils.bndl import (
#     uncertainty_sample_parallel,  # Renamed from pixel_uncertain_sampling
#     entropy_uncertainty,  # Renamed from pixel_entropy_uncertainty
# )
# Note: We implement our own version below for compatibility
from training.utils.dataset_evaluator import DistributedDatasetEvaluator
from tools.vos_inference import DAVIS_PALETTE, save_masks_to_dir

from checkpoint_manager import CheckpointManager, StatisticsCheckpointManager
from downsampling_utils import downsample_statistics_pavpu

from zero_shot_utils import cleanup_gpu_memory

logger = logging.getLogger(__name__)


def extract_pixel_bndl_model_simple(model):
    """Simplified BNDL model extraction."""
    if hasattr(model, "module"):
        model = model.module

    # Check common paths
    for attr in ["sam_mask_decoder", "mask_decoder"]:
        if hasattr(model, attr):
            mask_decoder = getattr(model, attr)
            if hasattr(mask_decoder, "pixel_bndl"):
                return mask_decoder.pixel_bndl

    return None


def extract_pixel_features(bndl_outputs):
    """Extract pixel features needed for uncertainty sampling.

    CRITICAL: Must return pixel_feat (raw features BEFORE BNDL processing),
    not z_out (which is already processed by BNDL's first stage).
    Using z_out would cause double-processing when passed to BNDL again,
    leading to anomalous outputs (all-foreground predictions).
    """
    try:
        # Priority 1: pixel_feat - the raw features before BNDL processing
        # This is what BNDL.forward() expects as input
        if "pixel_feat" in bndl_outputs and bndl_outputs["pixel_feat"] is not None:
            return bndl_outputs["pixel_feat"]

        # Priority 2: upscaled_embedding - alternative name in some code paths
        if "upscaled_embedding" in bndl_outputs and bndl_outputs["upscaled_embedding"] is not None:
            return bndl_outputs["upscaled_embedding"]

        # WARNING: z_out is ALREADY PROCESSED by BNDL - using it would cause double-processing!
        # Only use as last resort with a warning
        if "z_out" in bndl_outputs and bndl_outputs["z_out"] is not None:
            logger.warning("extract_pixel_features: falling back to z_out (already processed by BNDL) - this may cause anomalous outputs!")
            return bndl_outputs["z_out"]

        return None
    except Exception as e:
        logger.warning(f"Failed to extract pixel features: {e}")
        return None


def extract_mask_tokens_from_bndl_outputs(bndl_outputs, batch, mask_decoder):
    """Extract mask_tokens_out from BNDL outputs.

    Since BNDL now directly uses mask_tokens_out (256-dim) via internal linear_add_w projection,
    we simply return the stored mask_tokens_out from aux_outputs.
    """
    try:
        if "mask_tokens_out" in bndl_outputs and bndl_outputs["mask_tokens_out"] is not None:
            return bndl_outputs["mask_tokens_out"]

        # Fallback: try to get from mask_decoder's learned embeddings
        upscaled_shape = bndl_outputs.get("upscaled_shape")
        if upscaled_shape is None:
            return None

        b, c, h, w = upscaled_shape
        num_mask_tokens = mask_decoder.num_mask_tokens

        if hasattr(mask_decoder, "mask_tokens"):
            # Use learned mask token embeddings as fallback
            mask_tokens_out = mask_decoder.mask_tokens.weight.unsqueeze(0).expand(b, -1, -1)
            return mask_tokens_out

        return None

    except Exception as e:
        logger.warning(f"Failed to extract mask_tokens_out from BNDL outputs: {e}")
        return None


def pixel_uncertain_sampling(pixel_bndl_model, pixel_feat, external_pre_out_w=None, sample_num=20):
    """
    Optimized sampling that computes both uncertainty (p-value) and entropy in one pass.
    Updated to work with refactored BNDL that uses mask_tokens_out instead of hyper_in.

    Args:
        pixel_bndl_model: BNDL model
        pixel_feat: [B, H, W, C] pixel features
        external_pre_out_w: [B, K, 256] mask_tokens_out (256-dim)
        sample_num: number of samples
    """
    device = pixel_feat.device
    B, H, W, C = pixel_feat.shape

    # Get factor parameters from model (simplified for refactored BNDL)
    factor_z = getattr(pixel_bndl_model, "default_factor_z", 0.0)
    factor_w = getattr(pixel_bndl_model, "default_factor_w", 0.0)

    # Use mask_tokens_out directly
    mask_tokens_out = external_pre_out_w

    if mask_tokens_out is None:
        raise ValueError("mask_tokens_out (external_pre_out_w) is required for BNDL sampling")

    # Pre-allocate tensor for samples
    with torch.no_grad():
        sample_0, *_ = pixel_bndl_model(
            pixel_feat,
            mask_tokens_out,
            force_sample=True,
            factor_z=factor_z,
            factor_w=factor_w,
        )
        K = sample_0.shape[-1]

    sampled_logits = torch.zeros(B, H, W, K, sample_num, device=device, dtype=sample_0.dtype)
    sampled_logits[..., 0] = sample_0

    # Run remaining samples
    with torch.no_grad():
        for i in range(1, sample_num):
            s_out, *_ = pixel_bndl_model(
                pixel_feat,
                mask_tokens_out,
                force_sample=True,
                factor_z=factor_z,
                factor_w=factor_w,
            )
            sampled_logits[..., i] = s_out

    # 1. Compute Mean Logits
    mean_pixel_logits = sampled_logits.mean(dim=-1)

    # 2. Compute Entropy (Standardized)
    probs = torch.sigmoid(sampled_logits)
    mean_probs = probs.mean(dim=-1)  # [B, H, W, K]

    # Binary entropy: -p log p - (1-p) log (1-p)
    eps = 1e-6
    mean_probs_clamped = torch.clamp(mean_probs, eps, 1.0 - eps)
    entropy_map = -(mean_probs_clamped * torch.log(mean_probs_clamped) + (1.0 - mean_probs_clamped) * torch.log(1.0 - mean_probs_clamped))

    # Normalize by log(2)
    import math

    entropy_norm = torch.clamp(entropy_map / math.log(2.0), 0.0, 1.0)

    # 3. Compute P-Value Uncertainty (Two-Sample Test logic)
    pixel_uncertainty = torch.zeros(B, H, W, device=device)
    if K >= 2:
        # Only relevant if we have multiple masks
        pixel_probs = torch.sigmoid(sampled_logits)
        prob_mean = pixel_probs.mean(dim=-1)  # [B, H, W, K]
        values, indices = torch.topk(prob_mean, min(2, K), dim=-1)

        indices = torch.clamp(indices, 0, K - 1)
        indices_expanded = indices.unsqueeze(-1).expand(B, H, W, 2, sample_num)
        top_logits = torch.gather(sampled_logits, 3, indices_expanded)

        aa = top_logits[:, :, :, 0, :].reshape(-1, sample_num)
        bb = top_logits[:, :, :, 1, :].reshape(-1, sample_num)

        d = aa - bb
        mean_d = d.mean(dim=-1)
        std_d = d.std(dim=-1, unbiased=True).clamp_min(eps)
        t_stat = mean_d / (std_d / (float(sample_num) ** 0.5) + eps)
        z = t_stat.abs() / 1.4142135623730951
        phi = 0.5 * (1.0 + torch.erf(z))
        pixel_uncertainty = (2.0 * (1.0 - phi)).view(B, H, W)

    return pixel_uncertainty, mean_pixel_logits, entropy_norm


def prepare_targets_for_pavpu(targets, bndl_outputs):
    """Prepare ground truth targets in the correct format for PAvPU calculation."""
    try:
        if targets is None:
            return None

        if isinstance(targets, torch.Tensor):
            target_tensor = targets
        elif isinstance(targets, (list, tuple)):
            if len(targets) > 0:
                target_tensor = targets[0]
            else:
                return None
        elif hasattr(targets, "masks"):
            target_tensor = targets.masks
        else:
            return None

        if len(target_tensor.shape) == 4:
            if target_tensor.shape[0] < target_tensor.shape[1] and target_tensor.shape[2] == target_tensor.shape[3]:
                target_tensor = target_tensor.permute(1, 2, 3, 0)
            elif target_tensor.shape[1] > target_tensor.shape[0] and target_tensor.shape[1] > target_tensor.shape[2]:
                target_tensor = target_tensor.permute(0, 2, 3, 1)

        elif len(target_tensor.shape) == 3:
            target_tensor = target_tensor.unsqueeze(-1)

        elif len(target_tensor.shape) == 5:
            target_tensor = target_tensor[:, 0, :, :, :].permute(0, 2, 3, 1)
        else:
            return None

        if torch.isnan(target_tensor).any():
            target_tensor = torch.nan_to_num(target_tensor, nan=0.0)

        target_tensor = torch.clamp(target_tensor, 0.0, 1.0)

        if "pixel_logits_raw" in bndl_outputs and bndl_outputs["pixel_logits_raw"] is not None:
            target_tensor = target_tensor.to(bndl_outputs["pixel_logits_raw"].device)
        elif "wei_lambda_pos" in bndl_outputs and bndl_outputs["wei_lambda_pos"] is not None:
            target_tensor = target_tensor.to(bndl_outputs["wei_lambda_pos"].device)
        elif "wei_lambda" in bndl_outputs and bndl_outputs["wei_lambda"] is not None:
            target_tensor = target_tensor.to(bndl_outputs["wei_lambda"].device)

        return target_tensor

    except Exception as e:
        logger.warning(f"Failed to prepare targets for PAvPU: {e}")
        return None


def calculate_pavpu_for_bndl(bndl_outputs, batch, targets, phase, model, sample_num=20):
    """Store raw pixel-level uncertainty and accuracy for true PAvPU analysis."""
    try:
        pixel_bndl_model = extract_pixel_bndl_model_simple(model)
        if pixel_bndl_model is None:
            return bndl_outputs

        pixel_feat = extract_pixel_features(bndl_outputs)
        if pixel_feat is None:
            return bndl_outputs

        external_pre_out_w = None
        if hasattr(model, "module"):
            mask_decoder = getattr(model.module, "sam_mask_decoder", None) or getattr(model.module, "mask_decoder", None)
        else:
            mask_decoder = getattr(model, "sam_mask_decoder", None) or getattr(model, "mask_decoder", None)

        # Always extract mask_tokens_out since we use global sparse mode
        if mask_decoder:
            external_pre_out_w = extract_mask_tokens_from_bndl_outputs(bndl_outputs, batch, mask_decoder)

        pixel_uncertainty_pval, mean_pixel_logits, entropy_norm = pixel_uncertain_sampling(
            pixel_bndl_model,
            pixel_feat,
            external_pre_out_w=external_pre_out_w,
            sample_num=sample_num,
        )

        pixel_uncertainty = pixel_uncertainty_pval

        output_entropy = entropy_norm.detach()
        if output_entropy.ndim == 4:
            # If per-channel entropy is returned, average it or take max?
            # Original code took mean implicitly if `pixel_entropy_uncertainty` returned [B, H, W]
            # My implementation returns [B, H, W, K].
            # Let's average across K for the summary map
            output_entropy = output_entropy.mean(dim=-1)

        bndl_outputs["pixel_entropy"] = output_entropy

        pixel_targets = prepare_targets_for_pavpu(targets, bndl_outputs)

        if pixel_targets is not None:
            pixel_predictions = bndl_outputs.get("pixel_logits_raw", mean_pixel_logits)
            if pixel_predictions is not None:
                if len(pixel_predictions.shape) == 4:
                    if pixel_predictions.shape[-1] != pixel_predictions.shape[-2]:
                        pixel_predictions = pixel_predictions.permute(0, 2, 3, 1)

                if pixel_predictions.shape != pixel_targets.shape:
                    if len(pixel_targets.shape) == 4 and len(pixel_predictions.shape) == 4:
                        B_pred, H_pred, W_pred, K_pred = pixel_predictions.shape
                        B_targ, H_targ, W_targ, K_targ = pixel_targets.shape

                        if H_pred != H_targ or W_pred != W_targ:
                            pixel_predictions = F.interpolate(
                                pixel_predictions.permute(0, 3, 1, 2),
                                size=(H_targ, W_targ),
                                mode="bilinear",
                                align_corners=False,
                            ).permute(0, 2, 3, 1)

                            if pixel_uncertainty is not None and pixel_uncertainty.shape[-2:] != (H_targ, W_targ):
                                if len(pixel_uncertainty.shape) == 3:
                                    pixel_uncertainty = F.interpolate(
                                        pixel_uncertainty.unsqueeze(1),
                                        size=(H_targ, W_targ),
                                        mode="bilinear",
                                        align_corners=False,
                                    ).squeeze(1)
                                elif len(pixel_uncertainty.shape) == 4:
                                    pixel_uncertainty = F.interpolate(
                                        pixel_uncertainty.permute(0, 3, 1, 2),
                                        size=(H_targ, W_targ),
                                        mode="bilinear",
                                        align_corners=False,
                                    ).permute(0, 2, 3, 1)

                        if B_pred != B_targ:
                            min_batch = min(B_pred, B_targ)
                            pixel_predictions = pixel_predictions[:min_batch]
                            pixel_targets = pixel_targets[:min_batch]

                        if K_pred != K_targ:
                            if K_pred > K_targ and K_targ == 1:
                                pixel_predictions = pixel_predictions[..., 0:1]
                            elif K_targ > K_pred:
                                pixel_targets = pixel_targets[..., :K_pred]
                            else:
                                min_k = min(K_pred, K_targ)
                                pixel_predictions = pixel_predictions[..., :min_k]
                                pixel_targets = pixel_targets[..., :min_k]

                        if pixel_predictions.shape != pixel_targets.shape:
                            return bndl_outputs
                    else:
                        return bndl_outputs

                bndl_outputs["pixel_uncertainty"] = pixel_uncertainty.detach()
                bndl_outputs["mean_pixel_logits"] = mean_pixel_logits.detach()

    except Exception as e:
        import traceback

        logger.warning(f"PAvPU calculation traceback: {traceback.format_exc()}")

    return bndl_outputs


def log_bndl_statistics(bndl_outputs, step, phase, dataset_name, statistics_dict=None):
    """Log BNDL statistics including pixel-level uncertainty and PAvPU."""
    if bndl_outputs is None:
        return statistics_dict or {}

    if statistics_dict is None:
        statistics_dict = {}

    # Optimized: Keep as tensor, avoid .item() sync
    key_prefix = f"{dataset_name}_{phase}"

    if bndl_outputs.get("wei_lambda_pos") is not None and bndl_outputs.get("kappa_pos") is not None:
        lambda_pos_mean = bndl_outputs["wei_lambda_pos"].mean().detach()
        k_pos_mean = bndl_outputs["kappa_pos"].mean().detach()
        statistics_dict[f"{key_prefix}_lambda_pixel_pos"] = lambda_pos_mean
        statistics_dict[f"{key_prefix}_k_pixel_pos"] = k_pos_mean

        lambda_sum = bndl_outputs["wei_lambda_pos"]
        k_sum = bndl_outputs["kappa_pos"]

        if bndl_outputs.get("wei_lambda_neg") is not None and bndl_outputs.get("kappa_neg") is not None:
            lambda_neg_mean = bndl_outputs["wei_lambda_neg"].mean().detach()
            k_neg_mean = bndl_outputs["kappa_neg"].mean().detach()
            statistics_dict[f"{key_prefix}_lambda_pixel_neg"] = lambda_neg_mean
            statistics_dict[f"{key_prefix}_k_pixel_neg"] = k_neg_mean
            lambda_sum = lambda_sum + bndl_outputs["wei_lambda_neg"]
            k_sum = 0.5 * (k_sum + bndl_outputs["kappa_neg"])

        statistics_dict[f"{key_prefix}_lambda_pixel"] = lambda_sum.mean().detach()
        statistics_dict[f"{key_prefix}_k_pixel"] = k_sum.mean().detach()

    elif bndl_outputs.get("wei_lambda") is not None and bndl_outputs.get("kappa") is not None:
        lambda_mean = bndl_outputs["wei_lambda"].mean().detach()  # Keep on device
        k_mean = bndl_outputs["kappa"].mean().detach()  # kappa is already the shape parameter

        # Use lists to accumulate tensors to batch sync later if needed.

        statistics_dict[f"{key_prefix}_lambda_pixel"] = lambda_mean
        statistics_dict[f"{key_prefix}_k_pixel"] = k_mean

    if bndl_outputs.get("pixel_uncertainty") is not None:
        uncertainty_mean = bndl_outputs["pixel_uncertainty"].mean().detach()
        statistics_dict[f"{key_prefix}_pixel_uncertainty"] = uncertainty_mean

    if bndl_outputs.get("pixel_entropy") is not None:
        entropy_mean = bndl_outputs["pixel_entropy"].mean().detach()
        statistics_dict[f"{key_prefix}_pixel_entropy"] = entropy_mean

    if "pavpu_uncertainty_samples" in bndl_outputs and "pavpu_accuracy_samples" in bndl_outputs:
        uncertainty_samples = bndl_outputs["pavpu_uncertainty_samples"]
        accuracy_samples = bndl_outputs["pavpu_accuracy_samples"]

        statistics_dict[f"{key_prefix}_pavpu_uncertainty_samples"] = uncertainty_samples.tolist() if hasattr(uncertainty_samples, "tolist") else list(uncertainty_samples)
        statistics_dict[f"{key_prefix}_pavpu_accuracy_samples"] = accuracy_samples.tolist() if hasattr(accuracy_samples, "tolist") else list(accuracy_samples)

    if bndl_outputs.get("wei_lambda_w_pos") is not None and bndl_outputs.get("kappa_w_pos") is not None:
        lambda_w_pos_mean = bndl_outputs["wei_lambda_w_pos"].mean().detach()
        k_w_pos_mean = bndl_outputs["kappa_w_pos"].mean().detach()
        statistics_dict[f"{key_prefix}_lambda_w_pos"] = lambda_w_pos_mean
        statistics_dict[f"{key_prefix}_k_w_pos"] = k_w_pos_mean

        lambda_w_sum = bndl_outputs["wei_lambda_w_pos"]
        k_w_sum = bndl_outputs["kappa_w_pos"]

        if bndl_outputs.get("wei_lambda_w_neg") is not None and bndl_outputs.get("kappa_w_neg") is not None:
            lambda_w_neg_mean = bndl_outputs["wei_lambda_w_neg"].mean().detach()
            k_w_neg_mean = bndl_outputs["kappa_w_neg"].mean().detach()
            statistics_dict[f"{key_prefix}_lambda_w_neg"] = lambda_w_neg_mean
            statistics_dict[f"{key_prefix}_k_w_neg"] = k_w_neg_mean
            lambda_w_sum = lambda_w_sum + bndl_outputs["wei_lambda_w_neg"]
            k_w_sum = 0.5 * (k_w_sum + bndl_outputs["kappa_w_neg"])

        statistics_dict[f"{key_prefix}_lambda_w"] = lambda_w_sum.mean().detach()
        statistics_dict[f"{key_prefix}_k_w"] = k_w_sum.mean().detach()

    elif bndl_outputs.get("wei_lambda_w") is not None and bndl_outputs.get("kappa_w") is not None:
        lambda_w_mean = bndl_outputs["wei_lambda_w"].mean().detach()
        k_w_mean = bndl_outputs["kappa_w"].mean().detach()
        statistics_dict[f"{key_prefix}_lambda_w"] = lambda_w_mean
        statistics_dict[f"{key_prefix}_k_w"] = k_w_mean

    return statistics_dict


def setup_bndl_collection(
    collect_statistics: bool,
    out_dir: Path,
    dataset_name: str,
    eval_dir: Path | None,
) -> tuple[dict | None, Any | None, Any | None, Any | None]:
    """Initialize statistics collection infrastructure."""
    if not collect_statistics:
        return None, None, None, None

    dataset_statistics = {}

    stats_checkpoint_mgr = StatisticsCheckpointManager(
        output_dir=out_dir.parent,
        dataset_name=dataset_name,
        interval=10,
    )

    eval_checkpoint_mgr = CheckpointManager(
        output_dir=out_dir.parent,
        dataset_name=dataset_name,
        checkpoint_type="eval",
        interval=10,
    )

    dataset_evaluator = None
    try:
        eval_save_dir = eval_dir if eval_dir else (out_dir.parent / f"{dataset_name.lower()}_bndl_eval" if dataset_name else out_dir.parent / "bndl_eval")
        dataset_evaluator = DistributedDatasetEvaluator(
            save_dir=str(eval_save_dir),
            distributed=False,
            rank=0,
            world_size=1,
            foreground_dilation=15,  # Match validation evaluator for consistency
        )
    except Exception as e:
        logger.error(f"Failed to initialize dataset evaluator: {e}")

    return dataset_statistics, stats_checkpoint_mgr, eval_checkpoint_mgr, dataset_evaluator


def extract_evaluator_checkpoint_data(dataset_evaluator: Any) -> dict:
    """Extract checkpoint data from dataset evaluator.

    Updated to use the simplified DistributedDatasetEvaluator API which uses:
    - self.data: dict with image-level metrics (uncertainty, accuracy, correctness, nll)
    - self.pixel_samples: dict with pixel-level samples (uncertainty, is_correct)
    """
    if dataset_evaluator is None:
        return {
            "pixel_uncertainties": [],
            "pixel_accuracies": [],
            "pixel_nlls": [],
        }

    # Extract image-level data
    result = {
        "pixel_uncertainties": list(dataset_evaluator.data.get("uncertainty", [])),
        "pixel_accuracies": list(dataset_evaluator.data.get("accuracy", [])),
        "pixel_nlls": list(dataset_evaluator.data.get("nll", [])),
    }

    # Also include pixel samples if available (for backwards compatibility)
    if hasattr(dataset_evaluator, "pixel_samples") and dataset_evaluator.pixel_samples:
        result["pixel_sample_uncertainties"] = list(dataset_evaluator.pixel_samples.get("uncertainty", []))
        result["pixel_sample_is_correct"] = list(dataset_evaluator.pixel_samples.get("is_correct", []))

    return result


def merge_evaluator_checkpoints(
    eval_checkpoint_mgr: Any,
    dataset_evaluator: Any,
    downsample_max_samples: int,
) -> None:
    """Merge checkpoint files back into dataset evaluator.

    Updated to work with the simplified DistributedDatasetEvaluator API which uses:
    - self.data: dict with image-level metrics (uncertainty, accuracy, correctness, nll)
    - self.pixel_samples: dict with pixel-level samples (uncertainty, is_correct)
    """
    if not eval_checkpoint_mgr or not dataset_evaluator:
        return

    # Simplified key mapping for the new evaluator API
    key_map = {
        "pixel_uncertainties": "uncertainty",
        "pixel_accuracies": "accuracy",
        "pixel_nlls": "nll",
    }

    total_samples = 0

    def _collect_shard(shard_data):
        nonlocal total_samples
        if not shard_data:
            return

        # Add data to evaluator's dict-based storage
        for src, dst in key_map.items():
            if shard_data.get(src):
                dataset_evaluator.data[dst].extend(shard_data[src])
                total_samples += len(shard_data[src])

        # Handle pixel samples if present
        if shard_data.get("pixel_sample_uncertainties"):
            dataset_evaluator.pixel_samples["uncertainty"].extend(shard_data["pixel_sample_uncertainties"])
        if shard_data.get("pixel_sample_is_correct"):
            dataset_evaluator.pixel_samples["is_correct"].extend(shard_data["pixel_sample_is_correct"])

        # Downsample if too much data accumulated
        if total_samples > downsample_max_samples * 2:
            _downsample_data()

    def _downsample_data():
        nonlocal total_samples
        for key in dataset_evaluator.data:
            if len(dataset_evaluator.data[key]) > downsample_max_samples:
                import random

                random.seed(42)
                indices = random.sample(range(len(dataset_evaluator.data[key])), downsample_max_samples)
                dataset_evaluator.data[key] = [dataset_evaluator.data[key][i] for i in indices]
                total_samples = downsample_max_samples
                print(f"  🔄 降采样 {key}: → {downsample_max_samples:,} 样本")

        # Also downsample pixel_samples
        if len(dataset_evaluator.pixel_samples["uncertainty"]) > dataset_evaluator.max_pixel_samples:
            import random

            random.seed(42)
            n = dataset_evaluator.max_pixel_samples
            indices = random.sample(range(len(dataset_evaluator.pixel_samples["uncertainty"])), n)
            dataset_evaluator.pixel_samples["uncertainty"] = [dataset_evaluator.pixel_samples["uncertainty"][i] for i in indices]
            dataset_evaluator.pixel_samples["is_correct"] = [dataset_evaluator.pixel_samples["is_correct"][i] for i in indices]

    # Stream through all checkpoints
    if hasattr(eval_checkpoint_mgr, "merge_checkpoints_streaming"):
        eval_checkpoint_mgr.merge_checkpoints_streaming(_collect_shard)
    else:
        # Fallback if streaming not available
        logger.warning("merge_checkpoints_streaming not available, skipping checkpoint merge")


def finalize_bndl_evaluation(
    dataset_evaluator: Any,
    dataset_name: str,
) -> None:
    """Generate final evaluation plots and save results."""
    if not dataset_evaluator or len(dataset_evaluator) == 0:
        if dataset_evaluator:
            logger.warning(f"No data for evaluation in {dataset_name}")
        return

    try:
        print(f"\nGenerating evaluation analysis for {dataset_name}...")

        # Use the simplified evaluator API
        results = dataset_evaluator.evaluate()
        logger.info(f"Evaluation completed: {len(results)} metrics")

        # Create visualization with the simplified API
        dataset_evaluator.create_visualization(filename=f"{dataset_name.lower()}_zeroshot_correlation.png")

        # Save results with the simplified API
        dataset_evaluator.save_results(filename=f"{dataset_name.lower()}_zeroshot_results.json")

        print(f"Dataset evaluation saved for {dataset_name}")
    except Exception as e:
        logger.warning(f"Dataset evaluation failed: {e}")


def extract_pixel_params(bndl_outputs, batch_idx=0):
    """Extract and process pixel-level parameters"""
    if bndl_outputs.get("wei_lambda_pos") is not None and bndl_outputs.get("kappa_pos") is not None:
        lambda_sum = bndl_outputs["wei_lambda_pos"]
        k_sum = bndl_outputs["kappa_pos"]
        if bndl_outputs.get("wei_lambda_neg") is not None and bndl_outputs.get("kappa_neg") is not None:
            lambda_sum = lambda_sum + bndl_outputs["wei_lambda_neg"]
            k_sum = 0.5 * (k_sum + bndl_outputs["kappa_neg"])
        lambda_vals = lambda_sum.detach().cpu().numpy()
        k_vals = k_sum.detach().cpu().numpy()
    else:
        lambda_vals = bndl_outputs["wei_lambda"].detach().cpu().numpy()
        k_vals = bndl_outputs["kappa"].detach().cpu().numpy()

    # Extract specific batch
    lambda_batch = lambda_vals[batch_idx]
    k_batch = k_vals[batch_idx]

    # Handle channel dimension
    if lambda_batch.shape[-1] > 1:
        lambda_img = lambda_batch.mean(axis=-1)
        k_img = k_batch.mean(axis=-1)
    else:
        lambda_img = lambda_batch.squeeze(-1)
        k_img = k_batch.squeeze(-1)

    return lambda_img, k_img


def extract_original_image(batch, frame_idx=0, batch_idx=0):
    """Extract and process original image, corresponding to the specified frame index"""
    if not hasattr(batch, "img_batch"):
        return None

    try:
        img_batch = batch.img_batch
        if hasattr(img_batch, "cpu"):
            img_batch = img_batch.cpu().numpy()

        if len(img_batch.shape) == 5:  # [T, B, C, H, W]
            T = img_batch.shape[0]
            safe_t = max(0, min(int(frame_idx), T - 1))
            orig_tensor = img_batch[safe_t, batch_idx]
        elif len(img_batch.shape) == 4:  # [B, C, H, W]
            orig_tensor = img_batch[batch_idx]
        else:
            return None

        if len(orig_tensor.shape) == 3 and orig_tensor.shape[0] in [1, 3]:
            original_img = orig_tensor.transpose(1, 2, 0)
        else:
            return None

        if original_img.min() < -1 or original_img.max() > 2:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            if len(original_img.shape) == 3 and original_img.shape[-1] == 3:
                original_img = original_img * std + mean

        original_img = np.clip(original_img, 0, 1)

        if len(original_img.shape) == 2:
            original_img = np.stack([original_img] * 3, axis=-1)
        elif len(original_img.shape) == 3 and original_img.shape[-1] == 1:
            original_img = np.repeat(original_img, 3, axis=-1)

        return original_img

    except Exception as e:
        return None


def upsample_params_to_image_size(lambda_img, k_img, target_shape):
    """Upsample parameter maps to target image size"""
    target_h, target_w = target_shape[:2]

    if lambda_img.shape != (target_h, target_w):
        lambda_img = cv2.resize(lambda_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        k_img = cv2.resize(k_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return lambda_img, k_img


def plot_common_elements_refactored(
    axes, original_img, lambda_img, k_img, step_index, bndl_outputs, has_uncertainty=False, batch=None, outputs_for_vis=None, bndl_viz: Any = None, viz_utils: Any = None
):
    """Plot common visualization elements using refactored modules"""
    if viz_utils is None or bndl_viz is None:
        return
    # First row: original image and parameter heatmaps
    viz_utils.plot_original_image(axes[0, 0], original_img)
    viz_utils.plot_parameter_heatmap(axes[0, 1], lambda_img, f"Lambda (lambda) Step {step_index}", "viridis")
    viz_utils.plot_parameter_heatmap(axes[0, 2], k_img, f"Shape (k) Step {step_index}", "plasma")

    # Second row: parameter overlays or distributions
    if original_img is not None and original_img.shape[:2] == lambda_img.shape:
        if has_uncertainty:
            bndl_viz.plot_parameter_and_uncertainty_overlays(
                axes[1, :],
                original_img,
                lambda_img,
                k_img,
                bndl_outputs,
                step_index,
            )
        else:
            viz_utils.plot_parameter_overlays(axes[1, :], original_img, lambda_img, k_img, step_index)
    else:
        viz_utils.plot_parameter_distributions(axes[1, :], lambda_img, k_img, step_index)

    # Third row: global parameters
    bndl_viz.plot_global_parameters_in_layout(axes[2, :], bndl_outputs, step_index)

    if has_uncertainty:
        # Fourth row: uncertainty visualization
        bndl_viz.plot_uncertainty_visualization(axes[3, :], bndl_outputs, step_index)


def create_bndl_visualization_refactored(
    bndl_outputs,
    batch,
    outputs_for_vis,
    vis_dir,
    data_iter,
    step_index,
    frame_index,
    layout_type="full",
    save_individual=True,
    save_unified=False,
    prompt_info=None,
    save_pdf=False,  # Also save PDF versions for paper (300 DPI)
):
    """Create comprehensive BNDL visualization using refactored modules

    Args:
        bndl_outputs: BNDL model outputs dictionary
        batch: Input batch object with img_batch attribute
        outputs_for_vis: Additional outputs for visualization
        vis_dir: Directory to save visualizations
        data_iter: Data iteration number
        step_index: Step index within iteration
        frame_index: Frame index within video
        layout_type: Layout type ("full" or "basic")
        save_individual: If True, save each component as separate image (default: True for paper figures)
        save_unified: If True, save all components in one combined figure (default: False)
        prompt_info: Dict with 'point_coords' and 'point_labels' for click prompt visualization
        save_pdf: If True, also save PDF versions alongside PNG (300 DPI for paper)
    """
    try:
        try:
            from visualization_utils import VisualizationUtils  # type: ignore
            from bndl_visualizer import BNDLVisualizer  # type: ignore

            viz_utils = VisualizationUtils()
            bndl_viz = BNDLVisualizer()
        except Exception:
            return

        lambda_img, k_img = extract_pixel_params(bndl_outputs)
        original_img = extract_original_image(batch, frame_idx=frame_index)

        if original_img is not None:
            lambda_img, k_img = upsample_params_to_image_size(lambda_img, k_img, original_img.shape)

        has_uncertainty = "pixel_uncertainty" in bndl_outputs and bndl_outputs["pixel_uncertainty"] is not None
        has_pavpu = "pixel_pavpu" in bndl_outputs and bndl_outputs["pixel_pavpu"] is not None

        # Use the unified visualization method - it supports both individual and unified output
        bndl_viz.create_unified_visualization(
            vis_dir=vis_dir,
            data_iter=data_iter,
            step_index=step_index,
            original_img=original_img,
            lambda_img=lambda_img,
            k_img=k_img,
            bndl_outputs=bndl_outputs,
            prompt_info=prompt_info,
            layout_type=layout_type,
            save_individual=save_individual,
            save_unified=save_unified,
            visualize_pavpu_overlay=has_pavpu,
            uncertainty_metric=["entropy"],
            epoch=None,
            save_pdf=save_pdf,
        )

    except Exception as e:
        logger.warning(f"Failed to create BNDL visualization: {e}")
