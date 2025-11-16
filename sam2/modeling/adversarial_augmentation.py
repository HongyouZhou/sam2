# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Adversarial Augmentation Framework for SAM2

Provides unified interface for both image-level and feature-level augmentations,
including style perturbations and deformations (DG-Font style).

Key components:
- AugmentationResult: Container for augmentation outputs
- AdversarialAugmenter: Unified interface for all augmentation types
- ImageLevelStyleImpl: Image-space style augmentation (existing approach)
- FeatureLevelDeformationImpl: Feature-space deformation (DG-Font style)
- FeatureLevelStyleImpl: Feature-space style augmentation (future)
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from sam2.modeling.sam2_base import SAM2Base


@dataclass
class AugmentationResult:
    """
    Unified container for augmentation results.
    
    Encapsulates outputs from different augmentation modes (image-level vs feature-level)
    and types (style vs deformation), providing a consistent interface.
    
    Attributes:
        features: [B, C, H, W] Augmented backbone features (required)
        high_res_features: Optional list of high-resolution features for SAM decoder
        intermediate_images: Optional intermediate images (e.g., styled images) for debugging
        num_backbone_forwards: Number of backbone forward passes required (for monitoring)
        mode: Augmentation mode ("image_level" or "feature_level")
        aug_type: Augmentation type ("style" or "deformation")
        original_styles: Optional original style statistics (for style augmentation visualization)
        adversarial_styles: Optional adversarial style statistics (for style augmentation visualization)
    """
    features: torch.Tensor  # [B, C, H, W]
    high_res_features: list[torch.Tensor] | None = None
    intermediate_images: torch.Tensor | None = None
    num_backbone_forwards: int = 0
    mode: str = ""
    aug_type: str = ""
    original_styles: torch.Tensor | None = None  # For style augmentation visualization
    adversarial_styles: torch.Tensor | None = None  # For style augmentation visualization
    
    def release_intermediate(self):
        """
        Release intermediate variables to save memory.
        
        Call this after using the augmentation results to free up GPU memory.
        """
        if self.intermediate_images is not None:
            del self.intermediate_images
            self.intermediate_images = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Auto-cleanup on destruction"""
        self.release_intermediate()


class AdversarialAugmenter(nn.Module):
    """
    Unified interface for adversarial augmentations.
    
    Supports multiple augmentation types (style, deformation) and modes (image-level, feature-level).
    Automatically selects the appropriate implementation based on configuration.
    
    Args:
        mode: "image_level" or "feature_level"
        aug_type: "style" or "deformation"
        **kwargs: Additional arguments passed to the specific implementation
    """
    def __init__(
        self,
        mode: str,
        aug_type: str,
        **kwargs
    ):
        super().__init__()
        self.mode = mode
        self.aug_type = aug_type
        
        # Create the appropriate implementation
        self.impl = self._create_impl(**kwargs)
    
    def _create_impl(self, **kwargs):
        """Factory method to create the appropriate implementation"""
        if self.aug_type == "style":
            if self.mode == "image_level":
                return ImageLevelStyleImpl(**kwargs)
            elif self.mode == "feature_level":
                return FeatureLevelStyleImpl(**kwargs)
            else:
                raise ValueError(f"Unknown mode for style augmentation: {self.mode}")
        
        elif self.aug_type == "deformation":
            if self.mode == "feature_level":
                return FeatureLevelDeformationImpl(**kwargs)
            else:
                raise ValueError(f"Deformation only supports feature_level mode, got: {self.mode}")
        
        else:
            raise ValueError(f"Unknown augmentation type: {self.aug_type}")
    
    def apply(
        self,
        img_batch: torch.Tensor,
        clean_features: torch.Tensor,
        clean_high_res: list[torch.Tensor] | None,
        pixel_gt: torch.Tensor,
        model: nn.Module,
    ) -> AugmentationResult:
        """
        Apply augmentation.
        
        Unified interface that works for all augmentation types and modes.
        
        Args:
            img_batch: [B, 3, H, W] Input images
            clean_features: [B, C, H, W] Clean backbone features
            clean_high_res: List of high-resolution features (or None)
            pixel_gt: [B, K, H, W] Ground truth masks
            model: SAM2 model (used for image-level augmentations that need forward_image)
        
        Returns:
            AugmentationResult containing augmented features and metadata
        """
        return self.impl.apply(
            img_batch=img_batch,
            clean_features=clean_features,
            clean_high_res=clean_high_res,
            pixel_gt=pixel_gt,
            model=model,
        )


class ImageLevelStyleImpl(nn.Module):
    """
    Image-level style augmentation implementation.
    
    Applies AdaIN-based style transfer on images and then forwards through backbone.
    Uses PGD-style iterative optimization to find adversarial styles.
    
    Note: This requires an additional backbone forward pass.
    """
    def __init__(
        self,
        pgd_steps: int = 5,
        epsilon: float = 2.0,
        step_size: float = 0.1,
        use_multi_object: bool = False,
        use_gcn: bool = False,
        use_gt_region_style: bool = False,
        enable_background: bool = False,
        use_global_local_mix: bool = False,
        global_epsilon: float = 1.5,
        global_weight: float = 0.7,
        **kwargs
    ):
        super().__init__()
        self.pgd_steps = pgd_steps
        self.epsilon = epsilon
        self.step_size = step_size
        self.use_multi_object = use_multi_object
        self.use_gcn = use_gcn
        self.use_gt_region_style = use_gt_region_style
        self.enable_background = enable_background
        self.use_global_local_mix = use_global_local_mix
        self.global_epsilon = global_epsilon
        self.global_weight = global_weight
    
    def apply(
        self,
        img_batch: torch.Tensor,
        clean_features: torch.Tensor,
        clean_high_res: list[torch.Tensor] | None,
        pixel_gt: torch.Tensor,
        model: "SAM2Base",
    ) -> AugmentationResult:
        """
        Apply image-level style augmentation.
        
        Delegates to existing SAM2Base methods to maintain full compatibility:
        - _prepare_style_adversary_inputs: Extract original styles
        - _pgd_style_attack: Generate adversarial styles with PGD
        - _apply_style_to_images: Apply styles to images
        - forward_image: Forward styled images through backbone
        
        Args:
            img_batch: [B, 3, H, W] Input images
            clean_features: [B, C, H, W] Clean backbone features (passed for GCN)
            clean_high_res: List of high-res features (not used here)
            pixel_gt: [B, K, H, W] Ground truth masks
            model: SAM2Base model (needed for all style methods)
        
        Returns:
            AugmentationResult with styled features and num_backbone_forwards=1
        """
        # 1. Prepare inputs (extract original styles)
        pixel_gt_normalized, original_styles = model._prepare_style_adversary_inputs(
            img_batch, pixel_gt
        )
        
        # 2. Run PGD to find adversarial styles
        # Note: Pass clean_features for GCN (if enabled)
        # Get pixel_bndl_model from model if available
        pixel_bndl_model = None
        if hasattr(model, 'sam_mask_decoder') and hasattr(model.sam_mask_decoder, 'pixel_bndl'):
            pixel_bndl_model = model.sam_mask_decoder.pixel_bndl
        
        adv_styles = model._pgd_find_adversarial_styles(
            img_batch=img_batch,
            pixel_gt=pixel_gt_normalized,
            original_styles=original_styles,
            num_steps=self.pgd_steps,
            step_size=self.step_size,
            epsilon=self.epsilon,
            pixel_bndl_model=pixel_bndl_model,
            backbone_features=clean_features,  # For GCN
        )
        
        # 3. Apply adversarial styles to images
        if self.use_gt_region_style:
            styled_images = model._apply_style_to_images(
                img_batch, adv_styles, gt_mask=pixel_gt_normalized
            )
        else:
            styled_images = model._apply_style_to_images(
                img_batch, adv_styles, gt_mask=None
            )
        
        # 4. Forward styled images through backbone (1 extra forward pass)
        backbone_out = model.forward_image(styled_images, use_checkpoint=True)
        styled_features = backbone_out['backbone_fpn'][-1]
        
        # 5. Extract high-res features if needed
        styled_high_res = None
        if model.use_high_res_features_in_sam:
            styled_high_res = [
                backbone_out['backbone_fpn'][0],
                backbone_out['backbone_fpn'][1]
            ]
        
        # Save styles for visualization if enabled
        enable_vis = getattr(model, '_enable_style_visualization', False)
        orig_styles_vis = original_styles.detach().cpu() if enable_vis else None
        adv_styles_vis = adv_styles.detach().cpu() if enable_vis else None
        
        return AugmentationResult(
            features=styled_features,
            high_res_features=styled_high_res,
            intermediate_images=styled_images,  # For visualization/debugging
            num_backbone_forwards=1,  # Key cost: 1 extra backbone forward
            mode="image_level",
            aug_type="style",
            original_styles=orig_styles_vis,
            adversarial_styles=adv_styles_vis,
        )


class FeatureLevelStyleImpl(nn.Module):
    """
    Feature-level style augmentation implementation (experimental).
    
    This applies style transformations directly on features instead of images,
    avoiding an extra backbone forward pass.
    
    Note: Not implemented yet, reserved for future extension.
    """
    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "Feature-level style augmentation is not implemented yet. "
            "Use mode='image_level' for style augmentation."
        )
    
    def apply(self, img_batch, clean_features, clean_high_res, pixel_gt, model):
        raise NotImplementedError()


class FeatureLevelDeformationImpl(nn.Module):
    """
    Feature-level deformation augmentation (DG-Font style).
    
    Applies spatial deformations to features using deformable convolutions,
    inspired by DG-Font's FDSC (Feature Deformation Skip Connection).
    
    Key advantages:
    - No extra backbone forward pass (uses clean_features directly)
    - Handles multi-object overlaps with soft compositing
    - Optional GCN coordination for multi-object consistency
    
    Args:
        feature_dim: Dimension of backbone features (default: 256)
        epsilon: Deformation strength (not used in PGD yet, placeholder)
        pgd_steps: Number of PGD steps (not used yet, placeholder)
        use_soft_composite: Whether to use soft compositing for overlaps
        temperature: Temperature for soft compositing softmax
        use_gcn: Whether to use GCN for multi-object coordination
        num_deform_groups: Number of deformable convolution groups
        **kwargs: Additional arguments
    """
    def __init__(
        self,
        feature_dim: int = 256,
        epsilon: float = 0.15,
        pgd_steps: int = 3,
        use_soft_composite: bool = True,
        temperature: float = 1.0,
        use_gcn: bool = False,
        gcn_num_layers: int = 2,
        num_deform_groups: int = 4,
        **kwargs
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps
        self.use_soft_composite = use_soft_composite
        self.temperature = temperature
        self.use_gcn = use_gcn
        
        # Deformable convolution module (shared across objects for now)
        self.deform_module = DeformableConvModule(
            feature_dim=feature_dim,
            num_deform_groups=num_deform_groups,
        )
        
        # Soft compositor for handling overlaps
        if self.use_soft_composite:
            self.compositor = SoftCompositor(temperature=temperature)
        
        # Optional GCN for multi-object coordination
        if self.use_gcn:
            # Will be implemented in Phase 5
            logging.warning("GCN coordination for deformation is not implemented yet")
            self.deform_gcn = None
    
    def apply(
        self,
        img_batch: torch.Tensor,
        clean_features: torch.Tensor,
        clean_high_res: list[torch.Tensor] | None,
        pixel_gt: torch.Tensor,
        model: nn.Module,
    ) -> AugmentationResult:
        """
        Apply feature-level deformation.
        
        Args:
            img_batch: [B, 3, H, W] Input images (not used, kept for interface consistency)
            clean_features: [B, C, H, W] Clean backbone features
            clean_high_res: List of high-resolution features (reused as-is)
            pixel_gt: [B, K, H, W] Ground truth masks
            model: SAM2 model (not used, kept for interface consistency)
        
        Returns:
            AugmentationResult with deformed features and num_backbone_forwards=0
        """
        B, C, H_feat, W_feat = clean_features.shape
        K = pixel_gt.shape[1]
        
        # Resize masks to feature resolution
        if pixel_gt.shape[2:] != (H_feat, W_feat):
            pixel_gt_resized = F.interpolate(
                pixel_gt.float(),
                size=(H_feat, W_feat),
                mode='bilinear',
                align_corners=False
            )
        else:
            pixel_gt_resized = pixel_gt.float()
        
        # Per-object deformation
        deformed_list = []
        mask_list = []
        
        for k in range(K):
            mask_k = pixel_gt_resized[:, k:k + 1, :, :]  # [B, 1, H, W]
            
            # Check if mask is non-empty
            if mask_k.sum() > 0:
                # Apply deformation to this object
                deformed_k, offsets_k = self.deform_module(clean_features, mask_k)
                deformed_list.append(deformed_k)
                mask_list.append(mask_k)
            else:
                # Empty mask: use clean features
                deformed_list.append(clean_features)
                mask_list.append(mask_k)
        
        # Compose deformed features
        if self.use_soft_composite and len(deformed_list) > 0:
            final_features = self.compositor(deformed_list, mask_list)
        else:
            # Fallback: simple averaging (not recommended)
            final_features = torch.stack(deformed_list, dim=0).mean(dim=0)
        
        # TODO: GCN coordination (Phase 5)
        if self.use_gcn and self.deform_gcn is not None:
            # Apply GCN refinement
            pass
        
        return AugmentationResult(
            features=final_features,
            high_res_features=clean_high_res,  # Reuse clean high-res features
            intermediate_images=None,
            num_backbone_forwards=0,  # Key advantage: no extra backbone forward!
            mode="feature_level",
            aug_type="deformation",
        )


class DeformableConvModule(nn.Module):
    """
    Per-object deformable convolution module.
    
    Predicts spatial offsets conditioned on features and object mask,
    then applies deformable convolution to warp the features.
    
    Inspired by DG-Font's FDSC (Feature Deformation Skip Connection).
    
    Args:
        feature_dim: Dimension of input features
        num_deform_groups: Number of groups for deformable convolution
    """
    def __init__(self, feature_dim: int = 256, num_deform_groups: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_deform_groups = num_deform_groups
        
        # Offset predictor network
        # Input: [features + mask] -> Output: offset field [2 * kernel_size^2 * groups]
        # For 3x3 kernel: 2 * 9 * num_deform_groups offsets
        self.offset_net = nn.Sequential(
            nn.Conv2d(feature_dim + 1, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2 * 9 * num_deform_groups, kernel_size=3, padding=1),
        )
        
        # Deformable convolution weights
        self.deform_conv_weight = nn.Parameter(
            torch.zeros(feature_dim, feature_dim // num_deform_groups, 3, 3)
        )
        self.deform_conv_bias = nn.Parameter(torch.zeros(feature_dim))
        
        # Initialize weights
        nn.init.kaiming_normal_(
            self.deform_conv_weight, mode='fan_out', nonlinearity='relu'
        )
    
    def forward(
        self,
        features: torch.Tensor,
        object_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply deformable convolution conditioned on object mask.
        
        Args:
            features: [B, C, H, W] Input features
            object_mask: [B, 1, H, W] Object mask
        
        Returns:
            deformed: [B, C, H, W] Deformed features
            offsets: [B, 2*9*groups, H, W] Predicted offsets
        """
        # Concatenate features with mask
        feat_with_mask = torch.cat([features, object_mask], dim=1)  # [B, C+1, H, W]
        
        # Predict offsets
        offsets = self.offset_net(feat_with_mask)  # [B, 2*9*groups, H, W]
        
        # Apply deformable convolution
        from torchvision.ops import deform_conv2d
        
        deformed = deform_conv2d(
            input=features,
            offset=offsets,
            weight=self.deform_conv_weight,
            bias=self.deform_conv_bias,
            stride=1,
            padding=1,
            dilation=1,
        )
        
        return deformed, offsets


class SoftCompositor(nn.Module):
    """
    Soft compositing for handling multi-object overlaps.
    
    Uses softmax weighting to blend features from different objects at overlap regions,
    providing a differentiable solution to the z-order problem.
    
    Args:
        temperature: Temperature for softmax (lower = sharper boundaries)
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        deformed_features_list: list[torch.Tensor],
        mask_list: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compose multiple deformed features using soft weighting.
        
        Args:
            deformed_features_list: List of [B, C, H, W] deformed features for each object
            mask_list: List of [B, 1, H, W] masks for each object
        
        Returns:
            composited: [B, C, H, W] Soft-composited features
        """
        if len(deformed_features_list) == 0:
            raise ValueError("Empty feature list for compositing")
        
        if len(deformed_features_list) == 1:
            # Single object: no need for compositing
            return deformed_features_list[0]
        
        # Stack features and masks
        feat_stack = torch.stack(deformed_features_list, dim=1)  # [B, K, C, H, W]
        mask_stack = torch.stack(mask_list, dim=1)  # [B, K, 1, H, W]
        
        # Compute soft weights using softmax
        # This automatically handles z-order: objects with higher mask values get higher priority
        mask_weights = F.softmax(mask_stack / self.temperature, dim=1)  # [B, K, 1, H, W]
        
        # Weighted sum
        composited = (feat_stack * mask_weights).sum(dim=1)  # [B, C, H, W]
        
        return composited

