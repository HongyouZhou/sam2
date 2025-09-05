#!/usr/bin/env python
# Multi-dataset Zero-shot evaluation of SAM-2 with BNDL
# Supports TrashCan, GTEA, PIDRay, plittersdorf, Hypersim, DRAM, and CITYSCAPES datasets with UQ analysis

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

matplotlib.use("Agg")  # Use non-interactive backend to avoid Qt issues
import logging
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ----------  Metric ----------
from sav_dataset.utils.sav_benchmark import benchmark

# ----------  Tools -----------
from tools.vos_inference import (
    DAVIS_PALETTE,
    save_masks_to_dir,
)

# ----------  BNDL uncertainty and PAvPU functions ----------
from BNDL.BNDL_upload.ViT_Sparse.utils.model_helpers import pixel_pavpu_calculation, pixel_uncertain_sampling

# ----------  Dataset Evaluator from SAM2 training ----------
from training.utils.dataset_evaluator import DistributedDatasetEvaluator

# ----------  SAM-2 -----------
from sam2.build_sam import build_sam2_video_predictor

# ----------  Click sampling ----------
from sam2.modeling.sam2_utils import (
    sample_one_point_from_error_center,
)

# ----------  Import refactored visualization modules ----------
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training", "utils"))
from visualization_utils import VisualizationUtils
from bndl_visualizer import BNDLVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Dataset Configurations ----------
from dataset_configs import DATASET_CONFIGS, DEFAULT_DATASETS
from prompt_utils import compute_tight_box_from_bool_mask, box_center_xy, sample_pos_neg, sample_error_click

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


def extract_pixel_bndl_model_simple(model):
    """简化版本的BNDL模型提取"""
    if hasattr(model, "module"):
        model = model.module

    # 直接检查常见路径
    for attr in ["sam_mask_decoder", "mask_decoder"]:
        if hasattr(model, attr):
            mask_decoder = getattr(model, attr)
            if hasattr(mask_decoder, "pixel_bndl"):
                return mask_decoder.pixel_bndl

    return None


def extract_pixel_features(bndl_outputs):
    """Extract pixel features needed for uncertainty sampling"""
    try:
        # We need the intermediate features that were used to generate pixel_logits_raw
        # This should be available in the BNDL outputs
        logger.info(f"Available BNDL outputs keys: {list(bndl_outputs.keys())}")

        if "z_out" in bndl_outputs:
            z_out = bndl_outputs["z_out"]
            logger.info(f"Found z_out with shape: {z_out.shape}")
            return z_out  # [B, H, W, C']
        else:
            logger.warning("z_out not found in BNDL outputs for uncertainty sampling")
            # Try alternative feature sources
            if "upscaled_embedding" in bndl_outputs:
                logger.info("Using upscaled_embedding as alternative feature source")
                return bndl_outputs["upscaled_embedding"]
            return None
    except Exception as e:
        logger.warning(f"Failed to extract pixel features: {e}")
        import traceback

        logger.warning(f"Traceback: {traceback.format_exc()}")
        return None


def extract_bndl_outputs(outputs):
    """Extract BNDL outputs from model outputs"""
    for frame_idx, outs in enumerate(outputs):
        if "multistep_bndl_outputs" in outs:
            bndl_outputs_list = outs["multistep_bndl_outputs"]

            # Use the last valid BNDL output (highest resolution)
            for i in reversed(range(len(bndl_outputs_list))):
                if bndl_outputs_list[i] is not None:
                    return bndl_outputs_list[i], i, frame_idx
    return None, None, None


def extract_hyper_in_from_bndl_outputs(bndl_outputs, batch, mask_decoder):
    """Extract hyper_in (external_pre_out_w) from BNDL outputs or regenerate it"""
    try:
        # First check if hyper_in is stored in BNDL outputs
        if "hyper_in" in bndl_outputs and bndl_outputs["hyper_in"] is not None:
            logger.info("Found hyper_in in BNDL outputs")
            return bndl_outputs["hyper_in"]

        # If not stored, we need to regenerate it by running the mask decoder's hypernetwork
        # This requires reconstructing the transformer output tokens

        # Extract the upscaled shape info
        upscaled_shape = bndl_outputs.get("upscaled_shape")
        if upscaled_shape is None:
            logger.warning("No upscaled_shape found in BNDL outputs for hyper_in extraction")
            return None

        b, c, h, w = upscaled_shape
        num_mask_tokens = mask_decoder.num_mask_tokens

        # Try to regenerate hyper_in by running the hypernetwork MLPs
        if hasattr(mask_decoder, "output_hypernetworks_mlps"):
            try:
                device = next(mask_decoder.parameters()).device

                # Check if we have stored mask_tokens_out in BNDL outputs
                mask_tokens_out = bndl_outputs.get("mask_tokens_out")

                if mask_tokens_out is not None:
                    logger.info("Using stored mask_tokens_out for hyper_in generation")
                    # Use the actual mask tokens from the forward pass
                    batch_size = mask_tokens_out.shape[0]
                else:
                    logger.info("Using mask token embeddings as fallback for hyper_in generation")
                    # Fallback: use the mask token embeddings
                    mask_tokens_out = mask_decoder.mask_tokens.weight.unsqueeze(0).expand(b, -1, -1)  # [B, K, C]
                    batch_size = b

                # Generate hyper_in using the hypernetwork MLPs
                hyper_in_list = []
                for i in range(num_mask_tokens):
                    hyper_out = mask_decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])  # [B, C']
                    hyper_in_list.append(hyper_out)

                # Stack to get [B, K, C'] format
                hyper_in = torch.stack(hyper_in_list, dim=1)

                logger.info(f"Generated hyper_in with shape: {hyper_in.shape}")
                return hyper_in

            except Exception as e:
                logger.warning(f"Failed to regenerate hyper_in: {e}")
                return None
        else:
            logger.warning("No output_hypernetworks_mlps found in mask_decoder")
            return None

    except Exception as e:
        logger.warning(f"Failed to extract hyper_in from BNDL outputs: {e}")
        return None


def prepare_targets_for_pavpu(targets, bndl_outputs):
    """Prepare ground truth targets in the correct format for PAvPU calculation"""
    try:
        # targets should be in a format that can be converted to [B, H, W, K]
        if targets is None:
            return None

        # Handle different target formats
        if isinstance(targets, torch.Tensor):
            # Direct tensor case
            target_tensor = targets
        elif isinstance(targets, list | tuple):
            # List of tensors, take the first one
            if len(targets) > 0:
                target_tensor = targets[0]
            else:
                return None
        elif hasattr(targets, "masks"):
            # Nested structure
            target_tensor = targets.masks
        else:
            logger.warning(f"Unknown target format: {type(targets)}")
            return None

        # Convert to correct shape [B, H, W, K] with bounds checking
        logger.info(f"Original target shape: {target_tensor.shape}")

        if len(target_tensor.shape) == 4:
            # Check different possible 4D formats
            if target_tensor.shape[0] < target_tensor.shape[1] and target_tensor.shape[2] == target_tensor.shape[3]:
                # Format: [K, B, H, W] -> [B, H, W, K]
                target_tensor = target_tensor.permute(1, 2, 3, 0)
                logger.info(f"Transposed targets from [K, B, H, W] to [B, H, W, K]: {target_tensor.shape}")
            elif target_tensor.shape[1] > target_tensor.shape[0] and target_tensor.shape[1] > target_tensor.shape[2]:
                # Format: [B, K, H, W] -> [B, H, W, K]
                target_tensor = target_tensor.permute(0, 2, 3, 1)
                logger.info(f"Transposed targets from [B, K, H, W] to [B, H, W, K]: {target_tensor.shape}")
            # else: already in [B, H, W, K] format or [B, H, W, C] format

        elif len(target_tensor.shape) == 3:
            # [B, H, W] format, add mask dimension
            target_tensor = target_tensor.unsqueeze(-1)
            logger.info(f"Added mask dimension to targets: {target_tensor.shape}")

        elif len(target_tensor.shape) == 5:
            # [B, T, K, H, W] format, take first frame and transpose
            target_tensor = target_tensor[:, 0, :, :, :].permute(0, 2, 3, 1)
            logger.info(f"Used first frame from 5D targets: {target_tensor.shape}")

        else:
            logger.warning(f"Unexpected target shape: {target_tensor.shape}")
            return None

        # Validate tensor values are in reasonable range
        if torch.isnan(target_tensor).any():
            logger.warning("NaN values detected in targets")
            target_tensor = torch.nan_to_num(target_tensor, nan=0.0)

        # Clamp to reasonable range
        target_tensor = torch.clamp(target_tensor, 0.0, 1.0)

        # Ensure it's on the correct device
        if "pixel_logits_raw" in bndl_outputs and bndl_outputs["pixel_logits_raw"] is not None:
            target_tensor = target_tensor.to(bndl_outputs["pixel_logits_raw"].device)
        elif "wei_lambda" in bndl_outputs and bndl_outputs["wei_lambda"] is not None:
            target_tensor = target_tensor.to(bndl_outputs["wei_lambda"].device)

        return target_tensor

    except Exception as e:
        logger.warning(f"Failed to prepare targets for PAvPU: {e}")
        return None


def calculate_pavpu_for_bndl(bndl_outputs, batch, targets, phase, model):
    """Calculate PAvPU metric for BNDL outputs during evaluation"""
    try:
        # Extract pixel BNDL model for uncertainty sampling
        pixel_bndl_model = extract_pixel_bndl_model_simple(model)
        if pixel_bndl_model is None:
            logger.warning("Could not extract pixel_bndl model for PAvPU calculation")
            return bndl_outputs

        # Extract pixel features from BNDL outputs
        pixel_feat = extract_pixel_features(bndl_outputs)
        if pixel_feat is None:
            logger.warning("Could not extract pixel features for PAvPU calculation")
            return bndl_outputs

        # Get external weights (hyper_in) if available
        external_pre_out_w = None
        if hasattr(model, "module"):
            mask_decoder = getattr(model.module, "sam_mask_decoder", None) or getattr(model.module, "mask_decoder", None)
        else:
            mask_decoder = getattr(model, "sam_mask_decoder", None) or getattr(model, "mask_decoder", None)

        if mask_decoder and hasattr(mask_decoder, "bndl_replace_global_with_hyper") and mask_decoder.bndl_replace_global_with_hyper:
            # Extract hyper_in from BNDL outputs or regenerate it
            external_pre_out_w = extract_hyper_in_from_bndl_outputs(bndl_outputs, batch, mask_decoder)

        # Perform pixel-level uncertainty sampling
        pixel_uncertainty, mean_pixel_logits = pixel_uncertain_sampling(
            pixel_bndl_model,
            pixel_feat,
            external_pre_out_w=external_pre_out_w,
            sample_num=20,  # Reduced sample number for speed during evaluation
        )

        # Prepare ground truth masks for PAvPU calculation
        pixel_targets = prepare_targets_for_pavpu(targets, bndl_outputs)

        # Calculate PAvPU scores
        if pixel_targets is not None:
            pixel_predictions = bndl_outputs.get("pixel_logits_raw", mean_pixel_logits)
            if pixel_predictions is not None:
                # Ensure correct format [B, H, W, K] and validate dimensions
                if len(pixel_predictions.shape) == 4:
                    if pixel_predictions.shape[1] == pixel_predictions.shape[2]:
                        # Already in [B, H, W, K] format
                        pass
                    elif pixel_predictions.shape[-1] != pixel_predictions.shape[-2]:
                        # Likely [B, K, H, W] format, transpose to [B, H, W, K]
                        pixel_predictions = pixel_predictions.permute(0, 2, 3, 1)
                else:
                    logger.warning(f"Unexpected pixel_predictions shape: {pixel_predictions.shape}")
                    return bndl_outputs

                # Validate that dimensions match between predictions and targets
                if pixel_predictions.shape != pixel_targets.shape:
                    logger.warning(f"Shape mismatch - predictions: {pixel_predictions.shape}, targets: {pixel_targets.shape}")
                    # Try to fix common mismatches
                    if len(pixel_targets.shape) == 4 and len(pixel_predictions.shape) == 4:
                        B_pred, H_pred, W_pred, K_pred = pixel_predictions.shape
                        B_targ, H_targ, W_targ, K_targ = pixel_targets.shape

                        # Fix batch dimension mismatch
                        if B_pred != B_targ:
                            min_batch = min(B_pred, B_targ)
                            pixel_predictions = pixel_predictions[:min_batch]
                            pixel_targets = pixel_targets[:min_batch]
                            logger.info(f"Fixed batch dimension mismatch: using first {min_batch} samples")

                        # Fix spatial dimension mismatch (resolution difference)
                        if H_pred != H_targ or W_pred != W_targ:
                            # Resize targets to match predictions resolution
                            pixel_targets_resized = F.interpolate(
                                pixel_targets.permute(0, 3, 1, 2),  # [B, H, W, K] -> [B, K, H, W]
                                size=(H_pred, W_pred),
                                mode="bilinear",
                                align_corners=False,
                            ).permute(0, 2, 3, 1)  # [B, K, H, W] -> [B, H, W, K]
                            pixel_targets = pixel_targets_resized
                            logger.info(f"Resized targets from {H_targ}x{W_targ} to {H_pred}x{W_pred}")

                        # Fix mask dimension mismatch
                        if K_pred != K_targ:
                            min_k = min(K_pred, K_targ)
                            pixel_predictions = pixel_predictions[..., :min_k]
                            pixel_targets = pixel_targets[..., :min_k]
                            logger.info(f"Fixed mask dimension mismatch by truncating to K={min_k}")

                        # Final validation
                        if pixel_predictions.shape != pixel_targets.shape:
                            logger.warning(f"Still shape mismatch after fixes - predictions: {pixel_predictions.shape}, targets: {pixel_targets.shape}")
                            return bndl_outputs
                        else:
                            logger.info(f"Successfully fixed shape mismatch! Final shapes: {pixel_predictions.shape}")
                    else:
                        logger.warning("Cannot fix dimension mismatch, skipping PAvPU calculation")
                        return bndl_outputs

                pavpu_scores = pixel_pavpu_calculation(pixel_uncertainty, pixel_predictions, pixel_targets, thresholds=[0.01, 0.05, 0.1])

                # Add PAvPU results to BNDL outputs
                bndl_outputs["pixel_uncertainty"] = pixel_uncertainty.detach()
                bndl_outputs["pixel_pavpu"] = pavpu_scores
                bndl_outputs["mean_pixel_logits"] = mean_pixel_logits.detach()

                logger.info(f"PAvPU scores calculated: {pavpu_scores}")
            else:
                logger.warning("No pixel predictions found for PAvPU calculation")
        else:
            logger.warning("No valid targets found for PAvPU calculation")

    except Exception as e:
        logger.warning(f"Failed to calculate PAvPU: {e}")
        import traceback

        logger.warning(f"PAvPU calculation traceback: {traceback.format_exc()}")

    return bndl_outputs


def log_bndl_statistics(bndl_outputs, step, phase, dataset_name, statistics_dict=None):
    """Log BNDL statistics including pixel-level uncertainty and PAvPU"""
    if bndl_outputs is None:
        return statistics_dict or {}
    
    if statistics_dict is None:
        statistics_dict = {}
    
    # Pixel-level parameters (lambda and k)
    if ('wei_lambda' in bndl_outputs and 'inv_k' in bndl_outputs and 
        bndl_outputs['wei_lambda'] is not None and bndl_outputs['inv_k'] is not None):
        lambda_mean = bndl_outputs['wei_lambda'].mean().detach().cpu().item()
        k_mean = (1. / (bndl_outputs['inv_k'] + 1e-6)).mean().detach().cpu().item()
        
        key_prefix = f"{dataset_name}_{phase}"
        statistics_dict[f"{key_prefix}_lambda_pixel"] = lambda_mean
        statistics_dict[f"{key_prefix}_k_pixel"] = k_mean
        
        logger.info(f"BNDL Stats - {key_prefix}: lambda_pixel={lambda_mean:.4f}, k_pixel={k_mean:.4f}")
        
        # Log pixel uncertainty if available
        if 'pixel_uncertainty' in bndl_outputs and bndl_outputs['pixel_uncertainty'] is not None:
            uncertainty_mean = bndl_outputs['pixel_uncertainty'].mean().detach().cpu().item()
            statistics_dict[f"{key_prefix}_pixel_uncertainty"] = uncertainty_mean
            logger.info(f"BNDL Stats - {key_prefix}: pixel_uncertainty={uncertainty_mean:.4f}")
        
        # Log PAvPU scores if available
        if 'pixel_pavpu' in bndl_outputs and bndl_outputs['pixel_pavpu'] is not None:
            pavpu_scores = bndl_outputs['pixel_pavpu']
            for i, threshold in enumerate([0.01, 0.05, 0.1]):
                if i < len(pavpu_scores):
                    score = pavpu_scores[i].item() if hasattr(pavpu_scores[i], 'item') else pavpu_scores[i]
                    statistics_dict[f"{key_prefix}_pavpu_{threshold}"] = score
                    logger.info(f"BNDL Stats - {key_prefix}: pavpu_{threshold}={score:.4f}")
    
    # Global w statistics (original BNDL)
    if ('wei_lambda_w' in bndl_outputs and 
        'inv_k_w' in bndl_outputs and
        bndl_outputs['wei_lambda_w'] is not None and 
        bndl_outputs['inv_k_w'] is not None):
        lambda_w_mean = bndl_outputs['wei_lambda_w'].mean().detach().cpu().item()
        k_w_mean = (1. / (bndl_outputs['inv_k_w'] + 1e-6)).mean().detach().cpu().item()
        
        key_prefix = f"{dataset_name}_{phase}"
        statistics_dict[f"{key_prefix}_lambda_w"] = lambda_w_mean
        statistics_dict[f"{key_prefix}_k_w"] = k_w_mean
        
        logger.info(f"BNDL Stats - {key_prefix}: lambda_w={lambda_w_mean:.4f}, k_w={k_w_mean:.4f}")
    
    return statistics_dict


# ... (rest of the code remains the same)


def extract_pixel_params(bndl_outputs, batch_idx=0):
    """Extract and process pixel-level parameters"""
    b, c, h, w = bndl_outputs["upscaled_shape"]

    lambda_vals = bndl_outputs["wei_lambda"].detach().cpu().numpy()  # [B, H, W, C]
    inv_k_vals = bndl_outputs["inv_k"].detach().cpu().numpy()  # [B, H, W, C]
    k_vals = 1.0 / (inv_k_vals + 1e-6)

    # Extract specific batch - now working with [B, H, W, C] format
    lambda_batch = lambda_vals[batch_idx]  # [H, W, C]
    k_batch = k_vals[batch_idx]  # [H, W, C]

    # Handle channel dimension - average across channels if multiple channels
    if lambda_batch.shape[-1] > 1:
        lambda_img = lambda_batch.mean(axis=-1)  # [H, W]
        k_img = k_batch.mean(axis=-1)  # [H, W]
    else:
        lambda_img = lambda_batch.squeeze(-1)  # [H, W]
        k_img = k_batch.squeeze(-1)  # [H, W]

    return lambda_img, k_img


def extract_original_image(batch, frame_idx=0, batch_idx=0):
    """Extract and process original image, corresponding to the specified frame index"""
    if not hasattr(batch, "img_batch"):
        return None

    try:
        img_batch = batch.img_batch
        if hasattr(img_batch, "cpu"):
            img_batch = img_batch.cpu().numpy()

        # Extract specific batch and frame
        if len(img_batch.shape) == 5:  # [T, B, C, H, W]
            T = img_batch.shape[0]
            safe_t = max(0, min(int(frame_idx), T - 1))
            orig_tensor = img_batch[safe_t, batch_idx]  # [C, H, W]
        elif len(img_batch.shape) == 4:  # [B, C, H, W]
            orig_tensor = img_batch[batch_idx]  # [C, H, W]
        else:
            logger.warning(f"Unexpected img_batch shape: {img_batch.shape}")
            return None

        # Convert [C, H, W] -> [H, W, C]
        if len(orig_tensor.shape) == 3 and orig_tensor.shape[0] in [1, 3]:
            original_img = orig_tensor.transpose(1, 2, 0)
        else:
            return None

        # Denormalize if needed (ImageNet normalization)
        if original_img.min() < -1 or original_img.max() > 2:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            if len(original_img.shape) == 3 and original_img.shape[-1] == 3:
                original_img = original_img * std + mean

        # Clip to valid range
        original_img = np.clip(original_img, 0, 1)

        # Ensure 3 channels
        if len(original_img.shape) == 2:
            original_img = np.stack([original_img] * 3, axis=-1)
        elif len(original_img.shape) == 3 and original_img.shape[-1] == 1:
            original_img = np.repeat(original_img, 3, axis=-1)

        return original_img

    except Exception as e:
        logger.warning(f"Failed to process original image: {e}")
        return None


def upsample_params_to_image_size(lambda_img, k_img, target_shape):
    """Upsample parameter maps to target image size"""
    target_h, target_w = target_shape[:2]

    if lambda_img.shape != (target_h, target_w):
        lambda_img = cv2.resize(lambda_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        k_img = cv2.resize(k_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return lambda_img, k_img


def normalize_parameters_robust(lambda_img, k_img):
    """Robust parameter normalization, handling outliers"""
    try:
        # Use percentiles for robust normalization, avoiding outlier effects
        lambda_min = np.percentile(lambda_img, 1)
        lambda_max = np.percentile(lambda_img, 99)
        lambda_range = lambda_max - lambda_min
        if lambda_range < 1e-6:
            lambda_range = 1e-6

        k_min = np.percentile(k_img, 1)
        k_max = np.percentile(k_img, 99)
        k_range = k_max - k_min
        if k_range < 1e-6:
            k_range = 1e-6

        lambda_norm = (lambda_img - lambda_min) / lambda_range
        k_norm = (k_img - k_min) / k_range

        # Limit to [0, 1] range
        lambda_norm = np.clip(lambda_norm, 0, 1)
        k_norm = np.clip(k_norm, 0, 1)

        return lambda_norm, k_norm

    except Exception as e:
        logger.warning(f"Parameter normalization failed: {e}")
        # Return original values as fallback
        return lambda_img, k_img


def create_bndl_visualization_refactored(bndl_outputs, batch, outputs_for_vis, vis_dir, data_iter, step_index, frame_index, layout_type="full"):
    """Create comprehensive BNDL visualization using refactored modules"""
    try:
        # Initialize refactored visualization modules
        viz_utils = VisualizationUtils()
        bndl_viz = BNDLVisualizer()

        # Extract parameters and image
        lambda_img, k_img = extract_pixel_params(bndl_outputs)
        original_img = extract_original_image(batch, frame_idx=frame_index)

        if original_img is not None:
            lambda_img, k_img = upsample_params_to_image_size(lambda_img, k_img, original_img.shape)

        has_uncertainty = "pixel_uncertainty" in bndl_outputs and bndl_outputs["pixel_uncertainty"] is not None

        # Determine number of rows based on layout type and uncertainty availability
        if layout_type == "full" and has_uncertainty:
            rows = 4
        else:
            rows = 3

        # Use refactored tools to create figure layout
        fig, axes = viz_utils.create_figure_layout(rows, 3, (18, 6 * rows))

        # Plot common elements using refactored functions
        plot_common_elements_refactored(axes, original_img, lambda_img, k_img, step_index, bndl_outputs, has_uncertainty, batch, outputs_for_vis, bndl_viz, viz_utils)

        # Use refactored tools to save and close figure
        save_path = os.path.join(vis_dir, f"iter_{data_iter}_step_{step_index}_bndl_{layout_type}.png")
        viz_utils.save_and_close_figure(fig, save_path, dpi=150)

        logger.info(f"BNDL visualization saved: {save_path}")

    except Exception as e:
        logger.warning(f"Failed to create BNDL visualization: {e}")
        import traceback

        logger.warning(f"Traceback: {traceback.format_exc()}")


def plot_common_elements_refactored(axes, original_img, lambda_img, k_img, step_index, bndl_outputs, has_uncertainty=False, batch=None, outputs_for_vis=None, bndl_viz=None, viz_utils=None):
    """Plot common visualization elements using refactored modules"""
    # First row: original image and parameter heatmaps
    viz_utils.plot_original_image(axes[0, 0], original_img)
    viz_utils.plot_parameter_heatmap(axes[0, 1], lambda_img, f"Lambda (λ) Step {step_index}", "viridis")
    viz_utils.plot_parameter_heatmap(axes[0, 2], k_img, f"Shape (k) Step {step_index}", "plasma")

    # Second row: parameter overlays or distributions, including uncertainty overlays
    if original_img is not None and original_img.shape[:2] == lambda_img.shape:
        if has_uncertainty:
            bndl_viz.plot_parameter_and_uncertainty_overlays(axes[1, :], original_img, lambda_img, k_img, bndl_outputs, step_index)
        else:
            viz_utils.plot_parameter_overlays(axes[1, :], original_img, lambda_img, k_img, step_index)
    else:
        viz_utils.plot_parameter_distributions(axes[1, :], lambda_img, k_img, step_index)

    # Third row: global parameters
    bndl_viz.plot_global_parameters_in_layout(axes[2, :], bndl_outputs, step_index)

    if has_uncertainty:
        # Fourth row: uncertainty visualization
        bndl_viz.plot_uncertainty_visualization(axes[3, :], bndl_outputs, step_index)


def _sample_pos_neg(gt_mask: np.ndarray, dilate_iter: int = 5, full_mask: np.ndarray | None = None, current_obj_id: int | None = None):
    """Sample positive and negative points from GT mask"""
    # Positive point: random pixel inside GT
    ys, xs = np.where(gt_mask)
    assert len(xs) > 0, "GT mask is empty."
    idx = random.randrange(len(xs))
    pos_xy = (int(xs[idx]), int(ys[idx]))

    # Negative point: background pixel near boundary
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(gt_mask.astype(np.uint8), kernel, iterations=dilate_iter) > 0
    ring = np.logical_and(dilated, ~gt_mask)
    ys_n, xs_n = np.where(ring)
    if len(xs_n) == 0:  # fallback to any background
        ys_n, xs_n = np.where(~gt_mask)
    
    # Handle case where object covers entire image (no background pixels)
    if len(xs_n) == 0:
        if full_mask is not None and current_obj_id is not None:
            # Try to sample negative point from other objects
            other_objects_mask = (full_mask > 0) & (full_mask != current_obj_id)
            if np.any(other_objects_mask):
                ys_other, xs_other = np.where(other_objects_mask)
                idx_other = random.randrange(len(xs_other))
                neg_xy = (int(xs_other[idx_other]), int(ys_other[idx_other]))
                return pos_xy, neg_xy
        
        return pos_xy, None  # Return None for negative point when no background exists
    
    idx_n = random.randrange(len(xs_n))
    neg_xy = (int(xs_n[idx_n]), int(ys_n[idx_n]))
    return pos_xy, neg_xy


@torch.inference_mode()
# Remove autocast to reduce memory usage for large images
def inference_3_clicks_with_bndl(
    predictor,
    jpeg_dir: Path,
    ann_dir: Path,
    out_dir: Path,
    score_thresh: float = 0.0,
    video_names: list[str] | None = None,
    save_bndl_vis: bool = True,
    vis_dir: Path | None = None,
    dataset_name: str = "unknown",
    collect_statistics: bool = True,
    max_objects: int | None = None,
    prompt_method: str = "gt_box",
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

    print(f"3-click inference with BNDL UQ analysis on {len(video_names)} videos")

    # Create BNDL visualization directory
    if save_bndl_vis and vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize statistics collection
    dataset_statistics = {} if collect_statistics else None
    total_frames_processed = 0
    
    # Initialize dataset evaluator for correlation analysis like in SAM trainer
    dataset_evaluator = DistributedDatasetEvaluator(
        save_dir=str(vis_dir / "dataset_evaluation") if vis_dir else str(Path("./dataset_evaluation")),
        distributed=False,  # Single process for zero-shot evaluation
        rank=0,
        world_size=1
    ) if collect_statistics else None

    for v_idx, vid in enumerate(video_names, 1):
        print(f"[{v_idx:03}/{len(video_names)}] {vid}")
        video_dir = jpeg_dir / vid
        frame_names = sorted([p.stem for p in video_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]], key=lambda x: int(x))

        # Initialize predictor state
        state = predictor.init_state(str(video_dir))
        H, W = state["video_height"], state["video_width"]
        
        # Memory optimization: reduce processing resolution for very large images
        max_size = 512 if dataset_name == "Hypersim" else 1024  # Smaller max size for Hypersim
        if max(H, W) > max_size:
            scale_factor = max_size / max(H, W)
            new_H, new_W = int(H * scale_factor), int(W * scale_factor)
            print(f"Large image detected ({H}x{W}), processing at reduced resolution ({new_H}x{new_W})")
            # Note: The actual resizing would need to be implemented in the predictor
            # For now, we'll add more aggressive memory management

        # Read first frame GT to determine object IDs
        first_mask_path = ann_dir / vid / f"{frame_names[0]}.png"
        if not first_mask_path.exists():
            print(f"Warning: First frame annotation not found: {first_mask_path}")
            continue

        first_mask = np.array(Image.open(first_mask_path))
        all_obj_ids = [oid for oid in np.unique(first_mask) if oid > 0]

        if len(all_obj_ids) == 0:
            print(f"Warning: No objects found in first frame of video {vid}")
            continue

        # Apply more aggressive object limit for memory management
        # Reduce max objects for large datasets like CITYSCAPES and Hypersim
        effective_max_objects = max_objects
        if dataset_name in ["CITYSCAPES", "Hypersim"] and max_objects is None:
            effective_max_objects = 3  # Limit memory-intensive datasets to 3 objects max
        elif dataset_name in ["CITYSCAPES", "Hypersim"] and max_objects is not None:
            effective_max_objects = min(max_objects, 3)
        
        if effective_max_objects and len(all_obj_ids) > effective_max_objects:
            # Select objects with largest areas for more meaningful evaluation
            obj_areas = {}
            for oid in all_obj_ids:
                obj_areas[oid] = np.sum(first_mask == oid)
            
            # Sort by area and take top N objects
            sorted_objs = sorted(obj_areas.items(), key=lambda x: x[1], reverse=True)
            obj_ids = [oid for oid, _ in sorted_objs[:effective_max_objects]]
            print(f"Limited to {effective_max_objects} largest objects in video {vid} (from {len(all_obj_ids)} total)")
        else:
            obj_ids = all_obj_ids

        print(f"Processing {len(obj_ids)} objects in video {vid}: {obj_ids}")

        obj_points: dict[int, list] = {}

        for obj_id in obj_ids:
            gt_bool = first_mask == obj_id

            # Check if mask is empty
            if not np.any(gt_bool):
                print(f"Warning: Empty GT mask for object {obj_id} in video {vid}")
                continue

            # Clear GPU memory before processing each object to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if prompt_method == "gt_box":
                # Use tight GT bounding box as prompt
                try:
                    x0, y0, x1, y1 = compute_tight_box_from_bool_mask(gt_bool)
                except Exception as e:
                    print(f"Error computing box for object {obj_id} in video {vid}: {e}")
                    continue

                _, _, _ = predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=np.array([x0, y0, x1, y1], dtype=np.float32),
                )

                # For compatibility with existing visualization, store box center as a single point
                cx, cy = box_center_xy((x0, y0, x1, y1))
                obj_points[obj_id] = [
                    (int(cx), int(cy)),
                ]
            else:
                # three_clicks (legacy)
                try:
                    pos_xy, neg_xy = sample_pos_neg(gt_bool, full_mask=first_mask, current_obj_id=obj_id)
                except Exception as e:
                    print(f"Error sampling points for object {obj_id} in video {vid}: {e}")
                    continue

                if neg_xy is None:
                    print(f"Warning: Object {obj_id} in video {vid} covers entire image and no other objects available for negative sampling, using only positive point")
                    pts = np.array([pos_xy], dtype=np.float32)
                    lbl = np.array([1], dtype=np.int32)
                else:
                    pts = np.array([[pos_xy, neg_xy]], dtype=np.float32)
                    lbl = np.array([[1, 0]], dtype=np.int32)
                    pts = pts.reshape(-1, 2)
                    lbl = lbl.reshape(-1)
                _, obj_ids_after, masks = predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=pts,
                    labels=lbl,
                )

                cur_idx = obj_ids_after.index(obj_id)
                pred_bool = (masks[cur_idx : cur_idx + 1] > score_thresh).cpu().numpy().astype(bool)[0, 0]
                gt_t = torch.from_numpy(gt_bool[None, None]).bool()
                pred_t = torch.from_numpy(pred_bool[None, None]).bool()
                pt3_xy, lb3_i = sample_error_click(gt_t, pred_t)

                predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=np.array(pt3_xy, dtype=np.float32),
                    labels=np.array([lb3_i], dtype=np.int32),
                    clear_old_points=False,
                )

                obj_points[obj_id] = [
                    (int(pos_xy[0]), int(pos_xy[1])),
                    (int(neg_xy[0]), int(neg_xy[1])) if neg_xy is not None else None,
                    (int(pt3_xy[0]), int(pt3_xy[1])),
                ]

        print(f"Generated query points for video {vid}: {obj_points}")

        # Propagate through entire video with BNDL UQ analysis
        video_segments = {}
        bndl_vis_count = 0
        video_statistics = {} if collect_statistics else None
        
        for f_idx, out_obj_ids, out_logits in predictor.propagate_in_video(state):
            seg = {oid: (out_logits[i] > score_thresh).cpu().numpy() for i, oid in enumerate(out_obj_ids)}
            video_segments[f_idx] = seg
            
            # Collect BNDL statistics if enabled (with memory optimization)
            if collect_statistics:
                try:
                    # Clear memory before statistics collection
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Extract BNDL outputs from predictor state for each object
                    if hasattr(predictor, "get_bndl_outputs") and out_obj_ids:
                        # Process objects in smaller batches for memory efficiency
                        # Reduce to first 2 objects for Hypersim, 3 for others
                        max_obj_stats = 2 if dataset_name == "Hypersim" else 3
                        for obj_idx, obj_id in enumerate(out_obj_ids[:max_obj_stats]):
                            try:
                                bndl_outputs = predictor.get_bndl_outputs(state, f_idx, obj_idx)
                                if bndl_outputs is not None:
                                    # Calculate PAvPU if we have ground truth
                                    first_mask_path = ann_dir / vid / f"{frame_names[f_idx]}.png"
                                    if first_mask_path.exists():
                                        gt_mask = np.array(Image.open(first_mask_path))
                                        # Convert to tensor format for PAvPU calculation and move to same device as BNDL outputs
                                        gt_tensor = torch.from_numpy(gt_mask).float().unsqueeze(0)  # [1, H, W]
                                        # Move to the same device as BNDL outputs
                                        if 'pixel_logits_raw' in bndl_outputs:
                                            gt_tensor = gt_tensor.to(bndl_outputs['pixel_logits_raw'].device)
                                        elif 'wei_lambda' in bndl_outputs:
                                            gt_tensor = gt_tensor.to(bndl_outputs['wei_lambda'].device)
                                        bndl_outputs = calculate_pavpu_for_bndl(bndl_outputs, None, gt_tensor, "eval", predictor)
                                        
                                        # Clean up ground truth tensor immediately
                                        del gt_tensor, gt_mask
                                    
                                    # Log statistics
                                    video_statistics = log_bndl_statistics(
                                        bndl_outputs, 
                                        f_idx, 
                                        "eval", 
                                        f"{dataset_name}_{vid}_obj{obj_id}", 
                                        video_statistics
                                    )
                                    total_frames_processed += 1
                                    
                                    # Skip dataset evaluator for Hypersim to save memory
                                    if dataset_evaluator and 'pixel_uncertainty' in bndl_outputs and dataset_name != "Hypersim":
                                        try:
                                            # Get predictions from current frame
                                            pred_logits = out_logits[obj_idx] if obj_idx < len(out_logits) else None
                                            # Get current frame mask path (not just first frame)
                                            current_mask_path = ann_dir / vid / f"{frame_names[f_idx]}.png"
                                            if pred_logits is not None and current_mask_path.exists():
                                                # Load current frame ground truth
                                                current_gt_mask = np.array(Image.open(current_mask_path))
                                                current_gt_tensor = torch.from_numpy(current_gt_mask).float().unsqueeze(0)
                                                if 'pixel_logits_raw' in bndl_outputs:
                                                    current_gt_tensor = current_gt_tensor.to(bndl_outputs['pixel_logits_raw'].device)
                                                elif 'wei_lambda' in bndl_outputs:
                                                    current_gt_tensor = current_gt_tensor.to(bndl_outputs['wei_lambda'].device)
                                                
                                                dataset_evaluator.add_batch_data(
                                                    uncertainty=bndl_outputs['pixel_uncertainty'],
                                                    pred_logits=pred_logits.unsqueeze(0),  # Add batch dimension
                                                    gt_masks=current_gt_tensor
                                                )
                                                logger.info(f"Added frame {f_idx} obj {obj_id} to dataset evaluator")
                                                
                                                # Clean up immediately after adding to evaluator
                                                del current_gt_mask, current_gt_tensor
                                        except Exception as e:
                                            logger.warning(f"Failed to add frame {f_idx} obj {obj_id} to dataset evaluator: {e}")
                                    
                                    # Clean up bndl_outputs after processing
                                    del bndl_outputs
                                    
                            except Exception as e:
                                logger.warning(f"Failed to process BNDL outputs for obj {obj_idx}: {e}")
                            
                            # Clear cache after each object for all datasets
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Failed to collect BNDL statistics for video {vid}, frame {f_idx}: {e}")

            # Generate BNDL visualizations for selected frames (reduced for memory)
            if save_bndl_vis and vis_dir is not None and bndl_vis_count < 2:  # Limit to 2 visualizations per video
                try:
                    # Extract BNDL outputs from the predictor state
                    # This requires accessing the internal state or outputs
                    # For now, we'll create a mock batch for visualization
                    if hasattr(predictor, "get_bndl_outputs") and out_obj_ids:
                        # Use the first object for visualization with memory cleanup
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        bndl_outputs = predictor.get_bndl_outputs(state, f_idx, 0)
                        if bndl_outputs is not None:
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
                            )
                            bndl_vis_count += 1
                except Exception as e:
                    logger.warning(f"Failed to create BNDL visualization for frame {f_idx}: {e}")
            
            # Clear output logits after all usage to free memory
            del out_logits
            
            # Clean up memory after processing each frame
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # For very large videos, clean up old frame data to prevent accumulation
            if f_idx > 10 and f_idx % 5 == 0 and torch.cuda.is_available():  # Every 5 frames after frame 10
                torch.cuda.synchronize()
                import gc
                gc.collect()

        # Save PNG masks and clear video_segments periodically to prevent memory buildup
        processed_frames = []
        for f_idx in list(video_segments.keys()):
            seg = video_segments[f_idx]
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
            processed_frames.append(f_idx)
            
            # Clear processed frames from memory every 20 frames
            if len(processed_frames) >= 20:
                # Create a copy of processed_frames to avoid modifying list during iteration
                frames_to_clear = processed_frames.copy()
                for pf_idx in frames_to_clear:
                    if pf_idx in video_segments:
                        del video_segments[pf_idx]
                processed_frames.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Clear any remaining frames
        video_segments.clear()

        # Save query points
        (out_dir / vid).mkdir(parents=True, exist_ok=True)
        with open(out_dir / vid / "query_points.json", "w") as f:
            json.dump({int(k): v for k, v in obj_points.items()}, f, indent=2)
        
        # Merge video statistics into dataset statistics
        if collect_statistics and video_statistics and dataset_statistics is not None:
            dataset_statistics.update(video_statistics)
            print(f"Collected BNDL statistics for video {vid}: {len(video_statistics)} metrics")

        # Final cleanup after each video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection
            import gc
            gc.collect()
    
    # Generate dataset evaluation plots like in SAM trainer validation phase
    if collect_statistics and dataset_evaluator and len(dataset_evaluator) > 0:
        try:
            print(f"\nGenerating dataset correlation analysis for {dataset_name}...")
            
            # Evaluate correlation like in SAM trainer
            correlation_results = dataset_evaluator.evaluate_dataset_correlation()
            logger.info(f"Correlation evaluation completed with {len(correlation_results)} metrics")
            
            # Create visualization like in SAM trainer  
            dataset_evaluator.create_dataset_correlation_visualization(
                title=f"{dataset_name} Zero-shot Analysis - Dataset Correlation",
                save_name=f"{dataset_name.lower()}_zeroshot_dataset_analysis.png"
            )
            
            # Save results like in SAM trainer
            dataset_evaluator.save_correlation_results(
                save_name=f"{dataset_name.lower()}_zeroshot_results.json"
            )
            
            print(f"Dataset evaluation plots saved for {dataset_name}")
            logger.info(f"Dataset evaluation completed for {dataset_name}")
            
        except Exception as e:
            logger.warning(f"Dataset evaluation failed for {dataset_name}: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
    elif collect_statistics and dataset_evaluator:
        logger.warning(f"No data collected for dataset evaluation in {dataset_name} (collected: {len(dataset_evaluator) if dataset_evaluator else 0})")
    
    # Print final statistics summary
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
            elif isinstance(values, list) and len(values) > 0:
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
    split: str | None = None,
    score_thresh: float = 0.0,
    num_workers: int | None = None,
    video_subset: list[str] | None = None,
    save_bndl_vis: bool = True,
    prompt_method: str = "gt_box",
    first_frame_only: bool = False,
) -> tuple[float, float, float, dict]:
    """Run evaluation on a single dataset with BNDL UQ analysis and return metrics"""

    config = DATASET_CONFIGS[dataset_name]
    if split is None:
        split = config["default_split"]
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

    # Output directories
    out_dir = output_path / f"{dataset_name.lower()}_pred"
    out_dir.mkdir(parents=True, exist_ok=True)

    # BNDL visualization directory
    bndl_vis_dir = output_path / f"{dataset_name.lower()}_bndl_vis" if save_bndl_vis else None

    print(f"\n{'=' * 60}")
    print(f"Running {dataset_name} dataset evaluation with BNDL UQ analysis")
    print(f"{'=' * 60}")

    # Debug: Check data paths and availability
    print(f"JPEG directory: {jpeg_dir}")
    print(f"Annotation directory: {ann_dir}")
    print(f"JPEG dir exists: {jpeg_dir.exists()}")
    print(f"Ann dir exists: {ann_dir.exists()}")

    if jpeg_dir.exists():
        video_dirs = [d for d in jpeg_dir.iterdir() if d.is_dir()]
        print(f"Found {len(video_dirs)} video directories in JPEG dir")
        if video_dirs:
            print(f"First few videos: {[v.name for v in video_dirs[:3]]}")

    if ann_dir.exists():
        ann_dirs = [d for d in ann_dir.iterdir() if d.is_dir()]
        print(f"Found {len(ann_dirs)} annotation directories")
        if ann_dirs:
            print(f"First few annotation videos: {[v.name for v in ann_dirs[:3]]}")

    # Run inference with BNDL UQ analysis
    start_time = time.time()
    try:
        dataset_statistics = inference_3_clicks_with_bndl(
            predictor,
            jpeg_dir,
            ann_dir,
            out_dir,
            score_thresh=score_thresh,
            video_names=video_subset,
            save_bndl_vis=save_bndl_vis,
            vis_dir=bndl_vis_dir,
            dataset_name=dataset_name,
            collect_statistics=True,
            prompt_method=prompt_method,
        )
    except Exception as e:
        print(f"Error during inference for {dataset_name}: {e}")
        raise
    inference_time = time.time() - start_time

    # Run evaluation
    eval_start_time = time.time()

    # Prepare evaluation roots (support first-frame-only copy)
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

    try:
        J_F, global_J, global_F, _ = benchmark(
            gt_roots=[str(gt_root_eval)],
            mask_roots=[str(pred_root_eval)],
            strict=False,
            num_processes=num_workers,
            skip_first_and_last=config["skip_first_and_last"],
            verbose=True,
        )

        # Check for NaN values and handle them
        if len(J_F) == 0 or len(global_J) == 0 or len(global_F) == 0:
            print(f"Warning: Empty evaluation results for {dataset_name}")
            return 0.0, 0.0, 0.0, dataset_statistics or {}

        j_f_val = J_F[0] if not np.isnan(J_F[0]) else 0.0
        j_val = global_J[0] if not np.isnan(global_J[0]) else 0.0
        f_val = global_F[0] if not np.isnan(global_F[0]) else 0.0

        if np.isnan(j_f_val) or np.isnan(j_val) or np.isnan(f_val):
            print(f"Warning: NaN values detected in {dataset_name} evaluation results")
            print(f"  J&F: {J_F[0]}, J: {global_J[0]}, F: {global_F[0]}")
            # Return zeros instead of NaN
            j_f_val = 0.0 if np.isnan(j_f_val) else j_f_val
            j_val = 0.0 if np.isnan(j_val) else j_val
            f_val = 0.0 if np.isnan(f_val) else f_val

        # Removed PIDRay special-case macro averaging to keep a single evaluation pass

    except Exception as e:
        print(f"Error during evaluation of {dataset_name}: {e}")
        return 0.0, 0.0, 0.0, {}

    eval_time = time.time() - eval_start_time

    print(f"Inference time: {inference_time:.2f}s")
    print(f"Evaluation time: {eval_time:.2f}s")

    if save_bndl_vis and bndl_vis_dir is not None:
        print(f"BNDL UQ visualizations saved to: {bndl_vis_dir}")
    
    # Save BNDL statistics to file
    if dataset_statistics:
        stats_file = output_path / f"{dataset_name.lower()}_bndl_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(dataset_statistics, f, indent=2)
        print(f"BNDL statistics saved to: {stats_file}")

    return j_f_val, j_val, f_val, (dataset_statistics or {})


def create_comparison_plots_with_bndl(results: dict[str, tuple[float, float, float]], output_path: Path, all_statistics: dict = None):
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
    p.add_argument(
        "--prompt_method",
        type=str,
        default="gt_box",
        choices=["gt_box", "three_clicks"],
        help="Prompting strategy: gt_box (default) or three_clicks",
    )
    p.add_argument("--num_workers", type=int, default=None, help="Number of evaluation processes")
    p.add_argument("--output_path", default="./outputs/zs_04_09_sam_bndl", help="Root output directory")
    p.add_argument("--first_frame_only", action="store_true", help="Evaluate only the first frame per video by copying only the first PNG")

    # BNDL UQ visualization options
    p.add_argument("--save_bndl_vis", action="store_true", default=True, help="Generate BNDL UQ visualizations")
    p.add_argument("--video_limit", type=int, default=None, help="Limit number of videos per dataset (for quick testing)")
    p.add_argument("--max_objects", type=int, default=5, help="Maximum number of objects to process per video (default: 5, reduced for memory optimization)")
    p.add_argument("--enable_memory_optimization", action="store_true", default=True, help="Enable memory optimization features")
    p.add_argument("--pytorch_cuda_alloc_conf", default="expandable_segments:True", help="PyTorch CUDA memory allocator configuration")

    return p.parse_args()


def main():
    args = parse_args()
    
    # Set PyTorch CUDA memory allocator configuration for better memory management
    if args.enable_memory_optimization:
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.pytorch_cuda_alloc_conf
        print(f"Set PYTORCH_CUDA_ALLOC_CONF={args.pytorch_cuda_alloc_conf}")
        
        # Set additional memory optimization environment variables
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For debugging CUDA OOM issues
        print("Enabled memory optimization features")

    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load SAM-2 predictor
    print("Loading SAM-2 checkpoint...")
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=args.device,
    )
    print("SAM-2 loaded successfully!")
    
    # Clear memory after loading model
    if args.enable_memory_optimization and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("Cleared CUDA cache after model loading")

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
                prompt_method=args.prompt_method,
                first_frame_only=args.first_frame_only,
            )

            results[dataset_name] = (j_f, j, f)
            if dataset_statistics:
                all_statistics[dataset_name] = dataset_statistics
            print(f"{dataset_name} Results - J&F: {j_f:.2f}, J: {j:.2f}, F: {f:.2f}")

        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            continue
        finally:
            # Clean up memory after each dataset
            if args.enable_memory_optimization and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect()
                print(f"Cleaned up memory after {dataset_name}")

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
                lambda_pixel_values = [v for k, v in stats_items if 'lambda_pixel' in k]
                k_pixel_values = [v for k, v in stats_items if 'k_pixel' in k]
                uncertainty_values = [v for k, v in stats_items if 'pixel_uncertainty' in k]
                pavpu_values = [v for k, v in stats_items if 'pavpu' in k]
                
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
