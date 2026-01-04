# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Style augmentation utilities for domain generalization.

Implements:
- Style extraction from images (mean + std per channel)
- AdaIN (Adaptive Instance Normalization) for style transfer
"""

import logging

import torch
import torch.nn as nn


def extract_style_statistics(images: torch.Tensor) -> torch.Tensor:
    """
    Extract style statistics (mean and std) from normalized images.
    
    Args:
        images: [B, 3, H, W] normalized RGB images
    
    Returns:
        styles: [B, 6] tensor (3 channel means + 3 channel stds)
    """
    B, C, H, W = images.shape
    assert C == 3, f"Only RGB images supported, got {C} channels"
    
    # Per-channel mean and std across spatial dimensions
    mean = images.mean(dim=[2, 3])  # [B, 3]
    std = images.std(dim=[2, 3])    # [B, 3]
    
    # SAFETY: Clamp to reasonable ranges (normalized images should have mean ~0.45, std ~0.23)
    mean = mean.clamp(min=-5.0, max=5.0)
    std = std.clamp(min=0.01, max=5.0)
    
    # Concatenate to form 6-dimensional style vector
    styles = torch.cat([mean, std], dim=1)  # [B, 6]
    return styles


def extract_gt_region_style(
    images: torch.Tensor, 
    gt_masks: torch.Tensor,
    min_pixels: int = 100
) -> torch.Tensor:
    """
    Extract style statistics from GT (ground truth) regions (supports multi-object).
    
    For small GT regions (< min_pixels), falls back to global statistics
    for numerical stability.
    
    Args:
        images: [B, 3, H, W] normalized RGB images
        gt_masks: [B, K, H, W] ground truth masks for K objects
        min_pixels: minimum pixel count for stable statistics (default: 100)
    
    Returns:
        styles: [B, K, 6] or [B, 6] style statistics (3 means + 3 stds) per object
                For single-object (K=1), returns [B, 6] for backward compatibility
    """
    B, C, H_img, W_img = images.shape
    assert C == 3, f"Only RGB images supported, got {C} channels"
    
    # Ensure mask is [B, K, H, W]
    if gt_masks.ndim == 3:
        gt_masks = gt_masks.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
    
    B_mask, K, H_mask, W_mask = gt_masks.shape
    assert B == B_mask, f"Batch size mismatch: images {B}, masks {B_mask}"
    
    # Resize mask to image size if needed
    if (H_mask, W_mask) != (H_img, W_img):
        gt_masks = torch.nn.functional.interpolate(
            gt_masks.float(), 
            size=(H_img, W_img), 
            mode='nearest'
        )
    
    # CRITICAL: Binarize masks to prevent numerical instability from soft masks
    # Soft mask values (e.g., 0.001) can cause division instability in mean/std computation
    gt_masks = (gt_masks > 0.5).float()
    
    # Expand images to match mask: [B, 3, H, W] → [B, K, 3, H, W]
    images_expanded = images.unsqueeze(1).expand(-1, K, -1, -1, -1)
    
    # Expand masks to 3 channels: [B, K, H, W] → [B, K, 3, H, W]
    mask_3ch = gt_masks.unsqueeze(2).expand(-1, -1, 3, -1, -1).float()
    
    # Count valid pixels per object per channel: [B, K, 3]
    pixel_count = mask_3ch.sum(dim=[3, 4])
    
    # Compute mean: [B, K, 3]
    masked_sum = (images_expanded * mask_3ch).sum(dim=[3, 4])
    mean = masked_sum / (pixel_count + 1e-8)
    
    # Compute std: [B, K, 3]
    mean_expanded = mean.unsqueeze(-1).unsqueeze(-1)  # [B, K, 3, 1, 1]
    centered = (images_expanded - mean_expanded) * mask_3ch
    variance = (centered ** 2).sum(dim=[3, 4]) / (pixel_count + 1e-8)
    std = torch.sqrt(variance + 1e-8)
    
    # Concatenate: [B, K, 6]
    styles = torch.cat([mean, std], dim=2)
    
    # SAFETY: Clamp style statistics to reasonable ranges
    # For normalized images (mean ~0.45, std ~0.23), style stats should be similar
    # Extreme values indicate numerical instability (e.g., from empty masks)
    styles_mean = styles[:, :, :3].clamp(min=-5.0, max=5.0)
    styles_std = styles[:, :, 3:].clamp(min=0.01, max=5.0)
    styles = torch.cat([styles_mean, styles_std], dim=2)
    
    # Fallback for too-small regions: use global statistics
    too_small = (pixel_count < min_pixels).any(dim=2)  # [B, K]
    if too_small.any():
        # Compute global statistics: [B, 6]
        global_styles = extract_style_statistics(images)
        # Expand to [B, K, 6]
        global_styles_expanded = global_styles.unsqueeze(1).expand(-1, K, -1)
        # Replace small regions
        styles = torch.where(
            too_small.unsqueeze(-1),  # [B, K, 1]
            global_styles_expanded,
            styles
        )
        num_fallback = too_small.sum().item()
        logging.debug(
            f"GT region style extraction: {num_fallback}/{B*K} objects "
            f"fell back to global statistics (GT region < {min_pixels} pixels)"
        )
    
    # For single-object case (K=1), squeeze to [B, 6] for backward compatibility with eb03fdb
    if K == 1:
        styles = styles.squeeze(1)  # [B, 1, 6] -> [B, 6]
    
    return styles  # [B, K, 6] or [B, 6]


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization.
    
    Transfers style statistics to content features by:
    1. Normalizing content features to zero mean and unit variance
    2. Scaling and shifting by target style statistics
    """
    
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, content_feat: torch.Tensor, style_stats: torch.Tensor) -> torch.Tensor:
        """
        Apply AdaIN to content features using style statistics.
        
        Args:
            content_feat: [B, C, H, W] features to be styled
            style_stats: [B, 6] style statistics (3 means + 3 stds from RGB)
        
        Returns:
            styled_feat: [B, C, H, W] features with applied style
        """
        B, C, H, W = content_feat.shape
        
        # Normalize content to zero mean unit variance
        content_mean = content_feat.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        content_std = content_feat.std(dim=[2, 3], keepdim=True) + self.eps  # [B, C, 1, 1]
        normalized = (content_feat - content_mean) / content_std
        
        # Extract target style statistics
        # style_stats: [B, 6] = [mean_r, mean_g, mean_b, std_r, std_g, std_b]
        target_mean = style_stats[:, :3]  # [B, 3]
        target_std = style_stats[:, 3:]   # [B, 3]
        
        # Broadcast style statistics to match feature dimensions
        # For features with C channels, we apply RGB style to first 3 channels
        # and replicate for remaining channels
        if C == 3:
            # Direct mapping for 3-channel features
            style_mean = target_mean.unsqueeze(-1).unsqueeze(-1)  # [B, 3, 1, 1]
            style_std = target_std.unsqueeze(-1).unsqueeze(-1)    # [B, 3, 1, 1]
        else:
            # For C > 3, replicate RGB style across channels
            # Option: linear interpolation or repetition
            repeat_factor = (C + 2) // 3
            style_mean = target_mean.repeat(1, repeat_factor)[:, :C].unsqueeze(-1).unsqueeze(-1)
            style_std = target_std.repeat(1, repeat_factor)[:, :C].unsqueeze(-1).unsqueeze(-1)
        
        # Apply target style
        styled = normalized * style_std + style_mean
        return styled




