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
        deformation_offsets: Optional deformation offset fields [B, K, 2, H, W] for visualization
    """
    features: torch.Tensor  # [B, C, H, W]
    high_res_features: list[torch.Tensor] | None = None
    intermediate_images: torch.Tensor | None = None
    num_backbone_forwards: int = 0
    mode: str = ""
    aug_type: str = ""
    original_styles: torch.Tensor | None = None  # For style augmentation visualization
    adversarial_styles: torch.Tensor | None = None  # For style augmentation visualization
    deformation_offsets: torch.Tensor | None = None  # [B, K, 2, H, W] For deformation visualization
    
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
            if self.mode == "image_level":
                return ImageLevelDeformationImpl(**kwargs)
            elif self.mode == "feature_level":
                return FeatureLevelDeformationImpl(**kwargs)
            else:
                raise ValueError(f"Unknown mode for deformation: {self.mode}")
        
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


class ImageLevelDeformationImpl(nn.Module):
    """
    Image-level deformation augmentation.
    
    Applies spatial deformations directly to images using learned offset fields,
    then forwards warped images through backbone.
    
    Key advantages:
    - Simple GT alignment (use same offset field)
    - Direct visualization of deformation effect
    - Consistent with Style AUE (both image-level)
    
    Args:
        epsilon: Deformation strength in pixels (e.g., 30.0)
        image_channels: Input image channels (default: 3)
        use_soft_composite: Whether to use soft compositing for overlaps
        **kwargs: Additional arguments
    """
    def __init__(
        self,
        epsilon: float = 30.0,
        image_channels: int = 3,
        use_soft_composite: bool = True,
        **kwargs
    ):
        super().__init__()
        self.epsilon = epsilon
        self.use_soft_composite = use_soft_composite
        
        # Offset predictor network (image + mask → offset field)
        self.offset_net = nn.Sequential(
            nn.Conv2d(image_channels + 1, 64, kernel_size=7, padding=3, stride=2),  # 1024→512
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),  # 512→256
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=3, padding=1),  # Output: [B, 2, 256, 256]
            nn.Tanh(),  # Normalize to [-1, 1]
        )
        
        # Initialize with small weights
        # Use kaiming for all layers except the last one
        for i, m in enumerate(self.offset_net):
            if isinstance(m, nn.Conv2d):
                if i == len(self.offset_net) - 2:  # Last conv layer (before Tanh)
                    # Moderate initialization for output layer to enable visible initial deformation
                    # std=0.1 with epsilon=30 gives ~3 pixel initial offset (0.1 * 30 = 3)
                    nn.init.normal_(m.weight, mean=0.0, std=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Soft compositor
        if self.use_soft_composite:
            self.compositor = SoftCompositor(temperature=1.0)
    
    def apply(
        self,
        img_batch: torch.Tensor,
        clean_features: torch.Tensor,
        clean_high_res: list[torch.Tensor] | None,
        pixel_gt: torch.Tensor,
        model: nn.Module,
    ) -> AugmentationResult:
        """Generate offset fields and return warped images for backbone forward."""
        B, _, H_img, W_img = img_batch.shape
        K = pixel_gt.shape[1]
        
        # Generate per-object offset fields
        all_offsets = []
        for k in range(K):
            mask_k = pixel_gt[:, k:k+1, :, :].float()  # [B, 1, H, W]
            
            # Skip empty masks and background
            if mask_k.sum() == 0:
                all_offsets.append(torch.zeros(B, 2, H_img, W_img, device=img_batch.device))
                continue
            
            mask_area = mask_k.sum() / (B * H_img * W_img)
            if (k == K - 1) and (mask_area > 0.5):
                all_offsets.append(torch.zeros(B, 2, H_img, W_img, device=img_batch.device))
                continue
            
            # Predict offset (output is 256x256, need to upsample)
            img_mask_input = torch.cat([img_batch, mask_k], dim=1)  # [B, 4, H, W]
            offset_raw = self.offset_net(img_mask_input)  # [B, 2, 256, 256]
            
            # Debug: check offset magnitudes (first iteration only)
            if k == 0 and not hasattr(self, '_logged_offset_stats'):
                offset_mag = torch.sqrt(offset_raw[:, 0]**2 + offset_raw[:, 1]**2)
                logging.info(f"[DeformAUE Debug] offset_raw stats: "
                           f"mean={offset_raw.mean():.6f}, std={offset_raw.std():.6f}, "
                           f"mag_mean={offset_mag.mean():.6f}, mag_max={offset_mag.max():.6f}")
                self._logged_offset_stats = True
            
            # Upsample to image resolution
            offset_full = F.interpolate(
                offset_raw, size=(H_img, W_img), mode='bilinear', align_corners=False
            )  # [B, 2, H, W]
            
            # Scale by epsilon and mask
            offset_scaled = offset_full * self.epsilon * mask_k
            
            # Debug: check scaled offset magnitudes
            if k == 0 and hasattr(self, '_logged_offset_stats') and not hasattr(self, '_logged_scaled_stats'):
                offset_scaled_mag = torch.sqrt(offset_scaled[:, 0]**2 + offset_scaled[:, 1]**2)
                masked_region = offset_scaled_mag[mask_k.squeeze(1) > 0.5]
                if masked_region.numel() > 0:
                    logging.info(f"[DeformAUE Debug] offset_scaled (in mask) stats: "
                               f"mean={masked_region.mean():.6f}, std={masked_region.std():.6f}, "
                               f"max={masked_region.max():.6f}, epsilon={self.epsilon}")
                self._logged_scaled_stats = True
            
            all_offsets.append(offset_scaled)
        
        # Stack offsets: [B, K, 2, H, W]
        stacked_offsets = torch.stack(all_offsets, dim=1)
        
        # This will be used by compute_aue_loss to warp images
        # Return intermediate result without backbone forward
        return AugmentationResult(
            features=clean_features,  # Will be replaced after warping
            high_res_features=clean_high_res,
            intermediate_images=img_batch,  # Original images
            num_backbone_forwards=0,  # Will forward after combining with style
            mode="image_level",
            aug_type="deformation",
            deformation_offsets=stacked_offsets,  # [B, K, 2, H, W]
        )


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
        epsilon: Deformation strength (maximum offset magnitude, used to constrain offsets)
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
            epsilon=epsilon,  # Pass epsilon for offset constraint
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
        all_offsets_list = []  # Store offsets for visualization
        
        for k in range(K):
            mask_k = pixel_gt_resized[:, k:k + 1, :, :]  # [B, 1, H, W]
            
            # Check if mask is non-empty
            if mask_k.sum() == 0:
                # Empty mask: use clean features
                deformed_list.append(clean_features)
                mask_list.append(mask_k)
                # For empty masks, use zero offsets
                all_offsets_list.append(torch.zeros_like(
                    self.deform_module.offset_net(
                        torch.cat([clean_features, mask_k], dim=1)
                    )
                ))
                continue
            
            # Skip background (last object with area > 50% of feature map)
            mask_area = mask_k.sum() / (B * H_feat * W_feat)
            is_background = (k == K - 1) and (mask_area > 0.5)
            if is_background:
                # Background: use clean features without deformation
                deformed_list.append(clean_features)
                mask_list.append(mask_k)
                # Zero offsets for background
                all_offsets_list.append(torch.zeros_like(
                    self.deform_module.offset_net(
                        torch.cat([clean_features, mask_k], dim=1)
                    )
                ))
                continue
            
            # Apply deformation to foreground object
            deformed_k, offsets_k = self.deform_module(clean_features, mask_k)
            deformed_list.append(deformed_k)
            mask_list.append(mask_k)
            all_offsets_list.append(offsets_k)  # [B, 2*9*groups, H, W]
        
        # Compose deformed features
        if self.use_soft_composite and len(deformed_list) > 0:
            final_features = self.compositor(deformed_list, mask_list)
        else:
            # Fallback: simple averaging (not recommended)
            final_features = torch.stack(deformed_list, dim=0).mean(dim=0)
        
        # Stack offsets for visualization: [B, K, 2*9*groups, H, W]
        # Extract mean offset per object for visualization (simplified to [B, K, 2, H, W])
        stacked_offsets = torch.stack(all_offsets_list, dim=1)  # [B, K, 2*9*groups, H, W]
        
        # Correct reshape considering interleaved (x,y) format from deform_conv2d
        # Format: [Δx₀, Δy₀, Δx₁, Δy₁, ..., Δx₈, Δy₈] repeated for each group
        # Reshape: [B, K, 2*9*groups, H, W] → [B, K, 9*groups, 2, H, W]
        offsets_reshaped = stacked_offsets.view(
            B, K, 9 * self.deform_module.num_deform_groups, 2, H_feat, W_feat
        )
        
        # Average over all sampling points: [B, K, 9*groups, 2, H, W] → [B, K, 2, H, W]
        offsets_mean = offsets_reshaped.mean(dim=2)  # [B, K, 2, H, W]
        
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
            deformation_offsets=offsets_mean,  # [B, K, 2, H, W] for visualization
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
        epsilon: Maximum offset magnitude (constraint for offsets)
    """
    def __init__(self, feature_dim: int = 256, num_deform_groups: int = 4, epsilon: float = 0.15):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_deform_groups = num_deform_groups
        self.epsilon = epsilon
        
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
        
        # Initialize offset_net: use small initialization for last layer to prevent large initial offsets
        # Initialize first layers with kaiming normal (default)
        for i in range(len(self.offset_net) - 1):
            if isinstance(self.offset_net[i], nn.Conv2d):
                nn.init.kaiming_normal_(
                    self.offset_net[i].weight, mode='fan_out', nonlinearity='relu'
                )
        
        # Initialize last layer with small random values to have some initial deformation
        # This ensures non-zero gradients from the start, enabling the network to learn
        # Scale: 0.01 to ensure initial offsets are small but non-zero
        nn.init.normal_(self.offset_net[-1].weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.offset_net[-1].bias)
        
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
        
        # Constrain offsets using epsilon to prevent excessive deformation
        # Use tanh to map to [-1, 1] range, then scale by epsilon
        # This ensures offsets are bounded and training is stable
        offsets = torch.tanh(offsets) * self.epsilon
        
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

