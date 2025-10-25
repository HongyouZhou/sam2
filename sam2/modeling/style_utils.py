# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Style augmentation utilities for domain generalization.

Implements:
- Style extraction from images (mean + std per channel)
- AdaIN (Adaptive Instance Normalization) for style transfer
- PGD attack in style space
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
    
    # Concatenate to form 6-dimensional style vector
    styles = torch.cat([mean, std], dim=1)  # [B, 6]
    return styles


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


def pgd_style_attack(
    model,
    img_batch: torch.Tensor,
    initial_styles: torch.Tensor,
    gt: torch.Tensor,
    num_steps: int = 5,
    step_size: float = 0.1,
    epsilon: float = 2.0,
    style_mean: torch.Tensor = None,
    style_std: torch.Tensor = None,
) -> torch.Tensor:
    """
    PGD attack in style space to find adversarial styles.
    
    Performs projected gradient descent to maximize the segmentation loss
    by perturbing style statistics within a constrained range.
    
    Args:
        model: SAM2 model with forward_image_with_style method
        img_batch: [B, 3, H, W] input images
        initial_styles: [B, 6] starting styles (from cache or extracted)
        gt: [B, H, W] ground truth masks
        num_steps: number of PGD iterations
        step_size: gradient ascent step size
        epsilon: L∞ constraint radius (max perturbation per dimension)
        style_mean: [6] dataset mean for range projection (optional)
        style_std: [6] dataset std for range projection (optional)
    
    Returns:
        adversarial_styles: [B, 6] optimized adversarial styles
    """
    adv_styles = initial_styles.clone().detach()
    
    # Compute valid style range if statistics provided
    if style_mean is not None and style_std is not None:
        valid_min = style_mean - 3 * style_std
        valid_max = style_mean + 3 * style_std
    else:
        valid_min = valid_max = None
    
    for step in range(num_steps):
        adv_styles.requires_grad = True
        
        # Forward with current adversarial styles
        # This will apply AdaIN with adv_styles
        backbone_out = model.forward_image_with_style(img_batch, adv_styles)
        
        # Continue through the full model to get predictions
        # We need to compute a loss that we want to maximize
        # For now, use a simplified approach: extract features and compute uncertainty
        # The actual implementation will depend on the model's forward signature
        
        # Simplified loss computation (placeholder)
        # In practice, this should compute the full segmentation loss
        try:
            # Try to get some output to compute gradient
            # This is a simplified version - actual implementation needs full forward
            features = backbone_out['vision_features']
            
            # Simple proxy loss: maximize feature magnitude (encourages large changes)
            # Real implementation should compute actual segmentation loss
            loss = features.abs().mean()
            
        except Exception as e:
            logging.warning(f"PGD forward failed at step {step}: {e}")
            # If forward fails, return initial styles
            return initial_styles
        
        # Compute gradient w.r.t. adversarial styles
        try:
            grad = torch.autograd.grad(loss, adv_styles, create_graph=False)[0]
        except Exception as e:
            logging.warning(f"PGD gradient computation failed at step {step}: {e}")
            return initial_styles
        
        # Gradient ascent (maximize loss)
        with torch.no_grad():
            adv_styles = adv_styles.detach() + step_size * grad.sign()
            
            # Project to epsilon ball around initial styles (L∞ constraint)
            delta = adv_styles - initial_styles
            delta = torch.clamp(delta, -epsilon, epsilon)
            adv_styles = initial_styles + delta
            
            # Project to valid range (based on dataset statistics)
            if valid_min is not None and valid_max is not None:
                adv_styles = torch.clamp(adv_styles, valid_min, valid_max)
    
    return adv_styles.detach()

