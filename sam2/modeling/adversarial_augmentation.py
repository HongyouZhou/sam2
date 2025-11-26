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

# Import required modules for style augmentation
from sam2.modeling.style_utils import extract_style_statistics, extract_gt_region_style
from sam2.modeling.style_gcn import build_object_graph


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)

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
    Image-level style augmentation implementation with GRL (self-contained).
    
    Applies AdaIN-based style transfer on images using neural network + GRL.
    Replaces PGD-based optimization for memory efficiency.
    
    Note: This requires an additional backbone forward pass.
    All style-related methods are now self-contained within this class.
    """
    def __init__(
        self,
        epsilon: float = 2.0,
        use_multi_object: bool = False,
        use_gcn: bool = False,
        use_gt_region_style: bool = False,
        enable_background: bool = False,
        use_global_local_mix: bool = False,
        global_epsilon: float = 1.5,
        global_weight: float = 0.7,
        feature_dim: int = 256,
        num_objects: int = 11,
        **kwargs
    ):
        super().__init__()
        self.epsilon = epsilon
        self.use_multi_object = use_multi_object
        self.use_gcn = use_gcn
        self.use_gt_region_style = use_gt_region_style
        self.enable_background = enable_background
        self.use_global_local_mix = use_global_local_mix
        self.global_epsilon = global_epsilon
        self.global_weight = global_weight
        
        # Create adversarial style network with GRL
        self.style_net = StyleAdversarialNetwork(
            feature_dim=feature_dim,
            num_objects=num_objects,
            epsilon=epsilon,
        )
        
        # GCN for multi-object coordination (if enabled)
        if self.use_gcn:
            logging.warning("Style GCN coordination with GRL not yet implemented")
            self.style_gcn = None
    
    def apply(
        self,
        img_batch: torch.Tensor,
        clean_features: torch.Tensor,
        clean_high_res: list[torch.Tensor] | None,
        pixel_gt: torch.Tensor,
        model: "SAM2Base",
    ) -> AugmentationResult:
        """
        Apply image-level style augmentation with GRL (self-contained).
        
        Args:
            img_batch: [B, 3, H, W] Input images
            clean_features: [B, C, H, W] Clean backbone features (for style network)
            clean_high_res: List of high-res features (not used)
            pixel_gt: [B, K, H, W] Ground truth masks
            model: SAM2Base model (for accessing shared components)
        
        Returns:
            AugmentationResult with styled features and num_backbone_forwards=1
        """
        # 1. Prepare inputs (extract original styles)
        pixel_gt_normalized, original_styles = self._prepare_style_adversary_inputs(
            img_batch, pixel_gt, model
        )
        
        # 2. Predict adversarial styles using neural network + GRL
        adv_styles = self.style_net(clean_features, original_styles)
        
        # 3. Optional GCN refinement for multi-object coordination
        if self.use_gcn and model.style_gcn is not None:
            # Compute style delta
            style_delta = adv_styles - original_styles
            
            # Build object graph (img_batch needed for future semantic features)
            edge_index, edge_weight, _ = self._build_object_graph(
                pixel_gt_normalized, img_batch, clean_features, model
            )
            
            # Extract mask features if GCN uses visual features
            mask_features = None
            if model.style_gcn.feature_dim > 0 and clean_features is not None:
                with torch.no_grad():
                    mask_features = self._extract_mask_features(clean_features, pixel_gt_normalized)
            
            # Refine delta using GCN
            if edge_index is not None:
                refined_delta = model.style_gcn(
                    style_delta,
                    edge_index,
                    edge_weight,
                    mask_features=mask_features,
                )
                # Clip refined delta to epsilon ball
                refined_delta = torch.clamp(
                    refined_delta,
                    -self.epsilon,
                    self.epsilon
                )
                adv_styles = original_styles + refined_delta
        
        # 4. Apply styles to images
        apply_mask = pixel_gt_normalized if self.use_gt_region_style else None
        styled_images = self._apply_style_to_images(img_batch, adv_styles, gt_mask=apply_mask)
        
        # 4. Forward through backbone (backbone frozen, input grad enabled)
        # Allows gradients to flow back to style_net
        # 4. Forward through backbone
        # We use checkpointing to save memory. We do NOT freeze backbone weights here
        # because dynamic freezing breaks activation checkpointing.
        backbone_out = model.forward_image(styled_images, use_checkpoint=True)
        styled_features = backbone_out['backbone_fpn'][-1]
        
        # 5. Extract high-res features
        styled_high_res = None
        if model.use_high_res_features_in_sam:
            styled_high_res = [
                backbone_out['backbone_fpn'][0],
                backbone_out['backbone_fpn'][1],
            ]
        
        # Explicitly delete backbone_out to free graph references
        del backbone_out
        
        # 6. Save styles for visualization
        enable_vis = getattr(model, '_enable_style_visualization', False)
        orig_styles_vis = original_styles.detach().cpu() if enable_vis else None
        adv_styles_vis = adv_styles.detach().cpu() if enable_vis else None
        
        return AugmentationResult(
            features=styled_features,
            high_res_features=styled_high_res,
            intermediate_images=styled_images,
            num_backbone_forwards=1,
            mode="image_level",
            aug_type="style",
            original_styles=orig_styles_vis,
            adversarial_styles=adv_styles_vis,
        )
    
    def _prepare_style_adversary_inputs(
        self,
        img_batch: torch.Tensor,
        pixel_gt: torch.Tensor,
        model: "SAM2Base",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for style-based adversarial generation.
        
        Args:
            img_batch: [B, 3, H, W] input images (may have gradients from deform AUE)
            pixel_gt: [B, K, H, W] ground truth masks
            model: SAM2Base model for accessing configuration
        
        Returns:
            pixel_gt: [B, K, H, W] normalized GT masks
            original_styles: [B, K, 6] original style statistics (detached)
        """
        # Ensure 4D: [B, K, H, W]
        if pixel_gt.ndim == 3:
            pixel_gt = pixel_gt.unsqueeze(1)
        
        B, K, H, W = pixel_gt.shape
        
        # Extract all objects' styles (vectorized)
        # CRITICAL: Detach to break gradient flow from deform_augmenter
        # Style PGD should only optimize style parameters, not deform offsets
        if model.style_aug_use_gt_region_style:
            original_styles = extract_gt_region_style(img_batch.detach(), pixel_gt)
        else:
            global_style = extract_style_statistics(img_batch.detach())
            original_styles = global_style.unsqueeze(1).expand(-1, K, -1)
        
        return pixel_gt, original_styles
    
    def _apply_style_to_images(
        self, 
        img_batch: torch.Tensor, 
        style_stats: torch.Tensor | None,
        gt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply style statistics to images using AdaIN (vectorized multi-object).
        
        Args:
            img_batch: [B, 3, H, W] normalized images
            style_stats: [B, K, 6] or [B, 6] style statistics per object
            gt_mask: [B, K, H, W] or [B, 1, H, W] GT masks (optional)
        
        Returns:
            styled_images: [B, 3, H, W] styled images (still normalized)
        """
        # If no style stats provided, return original images
        if style_stats is None:
            return img_batch
        
        # Guard against NaN/Inf in inputs
        if not torch.isfinite(img_batch).all() or not torch.isfinite(style_stats).all():
            raise RuntimeError("Style application inputs contain NaN/Inf values")
            
        B, C, H, W = img_batch.shape
        
        # Detect single vs multi-object mode
        if style_stats.ndim == 2:
            # [B, 6] single object (backward compatible)
            style_stats = style_stats.unsqueeze(1)  # [B, 1, 6]
            if gt_mask is not None and gt_mask.ndim == 4 and gt_mask.shape[1] == 1:
                # gt_mask already [B, 1, H, W], keep as is
                pass
        
        K = style_stats.shape[1]
        
        # Compute base statistics from ORIGINAL image once
        # All objects will use this as normalization baseline to avoid cumulative shift
        base_means = img_batch.mean(dim=[2, 3], keepdim=True)  # [B, 3, 1, 1]
        base_stds = img_batch.std(dim=[2, 3], keepdim=True)    # [B, 3, 1, 1]
        
        # Pre-compute normalized images once to avoid repeated work inside the loop
        normalized = (img_batch - base_means) / (base_stds + 1e-8)

        # Accumulate style applications (start from original)
        styled_images = img_batch.clone()
        
        # Apply style for each object (vectorized processing)
        for k in range(K):
            object_style = style_stats[:, k]  # [B, 6]
            object_mask = gt_mask[:, k:k + 1] if gt_mask is not None else None  # [B, 1, H, W]
            
            # Extract target means and stds
            target_means = object_style[:, :3].view(B, 3, 1, 1)  # [B, 3, 1, 1]
            target_stds = object_style[:, 3:].view(B, 3, 1, 1)   # [B, 3, 1, 1]
            
            object_styled = normalized * target_stds + target_means
            
            # Apply mask (if provided)
            if object_mask is not None:
                # Adjust mask to image size
                if object_mask.shape[2:] != (H, W):
                    object_mask = F.interpolate(
                        object_mask.float(), 
                        size=(H, W), 
                        mode='nearest'
                    )
                object_mask = object_mask.float()
                
                # Only apply style to object region, blend with accumulated result
                # CRITICAL: Use img_batch (not styled_images) for background in first iteration
                # to match eb03fdb behavior exactly
                background = img_batch if k == 0 else styled_images
                styled_images = object_mask * object_styled + (1 - object_mask) * background
            else:
                # Apply to full image (replace entirely)
                styled_images = object_styled
        
        # Guard against NaN/Inf in output
        if not torch.isfinite(styled_images).all():
            raise RuntimeError("Style application produced NaN/Inf values")
        
        return styled_images
    
    def _pgd_find_adversarial_styles(
        self,
        img_batch: torch.Tensor,
        original_styles: torch.Tensor,
        pixel_gt: torch.Tensor | None,
        num_steps: int,
        step_size: float,
        epsilon: float,
        pixel_bndl_model,
        backbone_features: torch.Tensor | None,
        model: "SAM2Base",
    ) -> torch.Tensor:
        """
        PGD to find adversarial styles (vectorized for K objects).
        Goal: Maximize uncertainty calibration loss (MMD + MSE between U and Error).
        
        Supports two modes:
        1. Local-only mode (default): Perturb each object's style independently
        2. Global+Local mixed mode: Add global style drift + local perturbations
        
        Args:
            img_batch: [B, 3, H, W] original image batch
            original_styles: [B, K, 6] or [B, 6] original style statistics
            pixel_gt: [B, K, H, W] or [B, 1, H, W] ground truth masks
            num_steps: number of PGD iterations
            step_size: step size for each PGD iteration
            epsilon: L_inf perturbation budget (for local styles)
            pixel_bndl_model: BNDL model for computing uncertainty
            backbone_features: [B, C, H, W] detached backbone features (no grad)
            model: SAM2Base model for accessing shared components
        
        Returns:
            adv_styles: [B, K, 6] or [B, 6] adversarial style statistics
        """
        # Backward compatibility: [B, 6] → [B, 1, 6]
        squeeze_output = False
        if original_styles.ndim == 2:
            original_styles = original_styles.unsqueeze(1)
            squeeze_output = True  # Remember to squeeze back to [B, 6]
        
        # Check if using global-local mixed mode
        if model.style_aug_use_global_local_mix:
            # Mode: Global+Local Mixed Style Attack
            result = self._pgd_mixed_global_local_styles(
                img_batch, original_styles, pixel_gt,
                num_steps, step_size, epsilon,
                pixel_bndl_model, 20, model
            )
            # Restore original shape if input was [B, 6]
            if squeeze_output:
                result = result.squeeze(1)  # [B, 1, 6] → [B, 6]
            return result
        
        # Mode: Local-only Style Attack (original behavior)
        adv_styles = original_styles.clone().detach()
        
        # Build GCN refiner once (caches graph structure)
        gcn_refiner = self._StyleGraphRefiner(self, model, pixel_gt, img_batch, backbone_features) if model.style_gcn is not None else None
        
        # Cache loop invariants (computed once before PGD loop)
        apply_mask = pixel_gt if model.style_aug_use_gt_region_style else None
        
        # Pre-compute combined GT mask
        if pixel_gt is not None:
            if pixel_gt.ndim == 4 and pixel_gt.shape[1] > 1:
                combined_gt = pixel_gt.sum(dim=1, keepdim=True).clamp(0, 1)
            else:
                combined_gt = pixel_gt
        else:
            combined_gt = None
        
        # Cache point labels (constant across all steps)
        point_labels_template = torch.tensor([[2, 3]], dtype=torch.int32, device=img_batch.device).expand(img_batch.shape[0], 2)
        
        # Cache suppress flag
        prev_suppress = getattr(model, "_suppress_nested_aue", False)
        
        # Cache external weights default if needed
        external_w_default = None
        if pixel_bndl_model is not None and not pixel_bndl_model.enable_global_sparse:
            external_w_default = pixel_bndl_model.linear.weight.unsqueeze(0)
        
        # GCN refinement: apply before PGD loop to initialize coordinated perturbations
        # This allows GCN to learn how to coordinate multi-object style perturbations
        # before PGD optimization, making GCN effective even with num_steps=1
        if gcn_refiner is not None:
            if model.training:
                # Training mode: GCN refine with gradients to allow learning
                # Use GRL to flip gradients so GCN learns to maximize loss
                initial_delta = torch.zeros_like(adv_styles - original_styles)
                initial_delta.requires_grad = True
                refined_delta = gcn_refiner.refine_with_grad(initial_delta, epsilon)
                
                # Apply GRL to the refinement
                grl = GRL(alpha=1.0)
                refined_delta = grl(refined_delta)
                
                adv_styles = original_styles + refined_delta # No detach!
            else:
                # Eval mode: no grad for efficiency
                with torch.no_grad():
                    initial_delta = torch.zeros_like(adv_styles - original_styles)
                    refined_delta = gcn_refiner.refine_no_grad(initial_delta, epsilon)
                    adv_styles = original_styles + refined_delta
        
        for step in range(num_steps):
            # Clear cache at the start of each PGD step to reduce memory fragmentation
            if step > 0:  # Don't clear on first iteration (graph was just built)
                torch.cuda.empty_cache()
            
            # Clone to ensure adv_styles is a leaf variable (needed for requires_grad)
            # This is necessary because adv_styles may be computed from operations
            # (e.g., original_styles + refined_delta) that make it a non-leaf tensor
            adv_styles = adv_styles.clone().requires_grad_(True)
            
            # 1. Apply style and forward through model
            # Note: Gradient is needed here for PGD optimization
            styled_images = self._apply_style_to_images(img_batch, adv_styles, gt_mask=apply_mask)
            
            adv_backbone_out = model.forward_image(styled_images, use_checkpoint=True)
            adv_backbone_feat = adv_backbone_out['backbone_fpn'][-1]

            # 2. Extract high_res_features if needed
            high_res_features = None
            if model.use_high_res_features_in_sam:
                high_res_features = [
                    adv_backbone_out['backbone_fpn'][0],
                    adv_backbone_out['backbone_fpn'][1]
                ]
            
            # 3. Generate prompts from GT (using cached combined_gt)
            adv_prompts = model._generate_bbox_prompts_from_gt(combined_gt)
            adv_box_coords = torch.stack([adv_prompts[:, :2], adv_prompts[:, 2:]], dim=1)
            adv_point_inputs = {
                "point_coords": adv_box_coords,
                "point_labels": point_labels_template,
            }
            
            # 4. Forward through SAM heads to get BNDL outputs
            model._suppress_nested_aue = True
            try:
                *_, adv_aux_outputs = model._forward_sam_heads(
                    backbone_features=adv_backbone_feat,
                    point_inputs=adv_point_inputs,
                    high_res_features=high_res_features,
                    multimask_output=False,
                    pixel_gt_for_aue=None,
                )
            finally:
                model._suppress_nested_aue = prev_suppress
            
            # 5. Extract BNDL outputs using helper
            adv_bndl_outputs = model._extract_bndl_outputs(
                adv_aux_outputs, pixel_bndl_model,
                compute_logits=True, compute_uncertainty=True, uq_sample_num=20
            )
            if adv_bndl_outputs is None:
                logging.warning("PGD: Failed to extract BNDL outputs, stopping early")
                break
            
            # Guard against NaN/Inf in logits
            if not torch.isfinite(adv_bndl_outputs.pixel_logits).all():
                raise RuntimeError(f"PGD step {step}: adversarial logits contain NaN/Inf")
            
            # 6. Prepare GT
            H_feat, W_feat = adv_bndl_outputs.pixel_logits.shape[1:3]
            if combined_gt is not None:
                adv_gts_prepared = model._prepare_gt_for_loss(combined_gt, (H_feat, W_feat))
            else:
                adv_gts_prepared = None
            
            # 7. Compute calibration loss (maximize to find adversarial styles)
            calibration_loss_adv = model._compute_uncertainty_calibration_loss(
                adv_bndl_outputs, adv_gts_prepared, pixel_bndl_model
            )
            
            # 10. Gradient ascent (maximize calibration loss to create hard samples)
            grad = torch.autograd.grad(calibration_loss_adv, adv_styles, create_graph=False)[0]
            
            with torch.no_grad():
                # Gradient ascent step
                adv_styles = adv_styles.detach() + step_size * grad.sign() 
                # Project to epsilon ball
                delta = adv_styles - original_styles
                delta = torch.clamp(delta, -epsilon, epsilon)
                adv_styles = original_styles + delta
            
            # GCN refinement after each PGD step (for multi-step PGD coordination)
            # Skip last step: no next iteration to benefit from refinement (saves computation)
            # Only refine in training mode (eval mode skips entirely for efficiency)
            if gcn_refiner is not None and step < num_steps - 1 and model.training:
                # Training mode: GCN refine with gradients to train GCN
                delta = adv_styles - original_styles
                delta = delta.detach().requires_grad_(True)
                refined_delta = gcn_refiner.refine_with_grad(delta, epsilon)
                adv_styles = original_styles + refined_delta.detach()
            
            # Clean up large tensors at end of PGD step to free memory
            del styled_images, adv_backbone_out, adv_backbone_feat
            del adv_aux_outputs, adv_bndl_outputs, calibration_loss_adv, grad
            if 'high_res_features' in locals() and high_res_features is not None:
                del high_res_features
        
        # Final cleanup after PGD loop
        if gcn_refiner is not None:
            del gcn_refiner
        torch.cuda.empty_cache()

        # Restore original shape if input was [B, 6]
        result = adv_styles.detach()
        if squeeze_output:
            result = result.squeeze(1)  # [B, 1, 6] → [B, 6]
        
        return result
    
    def _extract_mask_features(
        self,
        backbone_features: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract visual features for each mask region via masked average pooling.
        
        Args:
            backbone_features: [B, C, H, W] features from backbone_fpn[-1]
            masks: [B, K, H, W] binary masks (or [B, K, 1, H, W])
        
        Returns:
            mask_features: [B, K, C] per-mask visual features
        """
        # Handle 5D mask input
        if masks.ndim == 5:
            masks = masks.squeeze(2)  # [B, K, 1, H, W] → [B, K, H, W]
        
        B, C, fH, fW = backbone_features.shape
        _, K, mH, mW = masks.shape
        
        # Resize masks to match feature map size if needed
        if (mH, mW) != (fH, fW):
            masks_resized = F.interpolate(
                masks, size=(fH, fW), mode='bilinear', align_corners=False
            )
        else:
            masks_resized = masks
        
        # Binarize masks
        masks_binary = (masks_resized > 0.5).float()  # [B, K, fH, fW]
        
        # Compute masked average pooling for each mask
        mask_features = []
        for k in range(K):
            mask_k = masks_binary[:, k:k + 1, :, :]  # [B, 1, fH, fW]
            mask_area = mask_k.sum(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
            mask_area = torch.clamp(mask_area, min=1.0)  # Avoid division by zero
            
            # Weighted average of features in mask region
            masked_feat = (backbone_features * mask_k).sum(dim=(2, 3))  # [B, C]
            feat_k = masked_feat / mask_area.view(B, 1)  # [B, C] / [B, 1] -> [B, C]
            mask_features.append(feat_k)
        
        # Stack features: [K, B, C] → [B, K, C]
        mask_features = torch.stack(mask_features, dim=1)  # [B, K, C]
        
        return mask_features
    
    def _build_object_graph(
        self,
        pixel_gt: torch.Tensor | None,
        img_batch: torch.Tensor,
        backbone_features: torch.Tensor | None,
        model: "SAM2Base",
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Build object graph for GCN refinement.
        
        Args:
            pixel_gt: [B, K, H, W] ground truth masks
            img_batch: [B, 3, H, W] images (for future semantic features)
            backbone_features: [B, C, H, W] backbone features for extracting visual features
            model: SAM2Base model for accessing configuration
        
        Returns:
            edge_index: [2, E] edge indices
            edge_weight: [E] edge weights
        """
        # DEBUG: Log detailed pixel_gt info
        if pixel_gt is not None:
            per_channel_sum = [pixel_gt[:, k].sum().item() for k in range(min(pixel_gt.shape[1], 5))]
            per_channel_nonzero = [(pixel_gt[:, k] > 0.5).sum().item() for k in range(min(pixel_gt.shape[1], 5))]
            logging.debug(f"DEBUG _build_style_graph: pixel_gt.shape={pixel_gt.shape}, "
                         f"dtype={pixel_gt.dtype}, device={pixel_gt.device}, "
                         f"min={pixel_gt.min():.3f}, max={pixel_gt.max():.3f}, "
                         f"per_channel_sum[0:5]={per_channel_sum}, "
                         f"per_channel_nonzero[0:5]={per_channel_nonzero}")
        
        # Debug: Check input masks (only log if debug level)
        if pixel_gt is not None and logging.getLogger().isEnabledFor(logging.DEBUG):
            mask_areas = (pixel_gt > 0.5).float().sum(dim=(2, 3))  # [B, K]
            valid_masks = (mask_areas > 0).sum().item()
            logging.debug(f"GCN input: pixel_gt.shape={pixel_gt.shape}, valid_masks={valid_masks}/{pixel_gt.shape[0] * pixel_gt.shape[1]}, "
                         f"edge_thresh={model.style_aug_gcn_edge_threshold}, dist_thresh={model.style_aug_gcn_distance_threshold}, "
                         f"use_bg={model.style_aug_enable_background and model.style_aug_gcn_use_background_edges}")
        
        # Extract mask features if visual features are enabled
        # Features serve two purposes:
        # 1. Build semantic edges in graph (via cosine similarity)
        # 2. Fuse with style deltas in GCN (via MLP projection)
        mask_features_for_graph = None
        if model.style_gcn is not None and model.style_gcn.feature_dim > 0 and backbone_features is not None and pixel_gt is not None:
            with torch.no_grad():
                mask_features_for_graph = self._extract_mask_features(backbone_features, pixel_gt)  # [B, K, 256]
        
        # Build graph structure using visual features for semantic edges
        edge_index, edge_weight, stats = build_object_graph(
            pixel_gt,
            img_batch,
            edge_threshold=model.style_aug_gcn_edge_threshold,
            use_semantic=model.style_aug_gcn_use_semantic_edges,
            use_background=(
                model.style_aug_enable_background and model.style_aug_gcn_use_background_edges
            ),
            distance_threshold=model.style_aug_gcn_distance_threshold,
            use_boundary_distance=model.style_aug_gcn_use_boundary_distance,
            mask_features=mask_features_for_graph,  # Used to build semantic edges
            feature_sim_threshold=model.style_aug_gcn_feature_sim_threshold,
        )
        
        # Add self-loops to the graph
        num_nodes_total = pixel_gt.shape[0] * pixel_gt.shape[1]  # B * K
        edge_index, edge_weight = model.style_gcn._add_self_loops(
            edge_index, edge_weight, num_nodes_total
        )
        
        model._latest_gcn_stats = stats if stats else None
        if stats and stats.get('graphs', 0) == 0:
            # Check if pixel_gt has any content
            if pixel_gt is not None:
                valid_pixels = (pixel_gt > 0.5).sum().item()
                logging.warning(f"GCN graph built but NO edges: graphs={stats['graphs']}, nodes_fg={stats['nodes_foreground']}, "
                              f"nodes_bg={stats['nodes_background']}, edges_iou={stats['edges_iou']}, edges_dist={stats['edges_distance']}, "
                              f"edges_bg={stats['edges_background']}, edges_semantic={stats.get('edges_semantic', 0)}, valid_pixels={valid_pixels}")
            else:
                logging.warning("GCN graph built but NO edges (pixel_gt is None)")
        elif stats:
            # Show edge type breakdown for non-empty graphs
            logging.debug(f"GCN graph built: {stats['graphs']:.0f} graphs, {stats['edges_total']:.0f} edges "
                         f"(IoU:{stats['edges_iou']:.0f}, Dist:{stats['edges_distance']:.0f}, Semantic:{stats.get('edges_semantic', 0):.0f}, BG:{stats['edges_background']:.0f}), "
                         f"nodes: {stats['nodes_foreground']:.0f}fg+{stats['nodes_background']:.0f}bg, "
                         f"avg_degree: {stats['avg_degree']:.2f}")
        else:
            logging.debug("GCN graph build returned None")
        return edge_index, edge_weight, stats
    
    def _pgd_mixed_global_local_styles(
        self,
        img_batch: torch.Tensor,
        original_local_styles: torch.Tensor,
        pixel_gt: torch.Tensor | None,
        num_steps: int,
        step_size: float,
        local_epsilon: float,
        pixel_bndl_model,
        uq_sample_num: int,
        model: "SAM2Base",
    ) -> torch.Tensor:
        """
        Global+Local mixed style perturbation.
        Simultaneously optimizes global style drift and local object styles.
        
        Strategy:
        1. All local styles first follow a global drift (consistency)
        2. Each local style can then deviate slightly from this global base (diversity)
        
        Final style = (original_local + global_drift) + (1 - global_weight) * local_deviation
        
        Where:
        - global_drift: adversarial shift applied to all objects uniformly
        - local_deviation: individual perturbation for each object
        - global_weight: controls consistency (0.7 means 100% global + 30% local deviation)
        
        This ensures all objects maintain global coherence while allowing controlled
        local variations, making multi-object perturbations look more natural.
        
        Args:
            img_batch: [B, 3, H, W] original images
            original_local_styles: [B, K, 6] local styles extracted from each object
            pixel_gt: [B, K, H, W] object masks
            num_steps: PGD iterations
            step_size: gradient step size
            local_epsilon: perturbation budget for local styles
            pixel_bndl_model: BNDL model
            uq_sample_num: uncertainty sampling number
            model: SAM2Base model for accessing shared components
        
        Returns:
            combined_styles: [B, K, 6] final adversarial styles (global base + local deviation)
        """
        B, K, _ = original_local_styles.shape
        
        # Extract global style statistics (for the whole image)
        original_global_style = extract_style_statistics(img_batch)  # [B, 6]
        
        # Initialize adversarial styles
        adv_local_styles = original_local_styles.clone().detach()  # [B, K, 6]
        adv_global_style = original_global_style.clone().detach()  # [B, 6]
        
        # Get hyperparameters
        global_epsilon = model.style_aug_global_epsilon
        global_weight = model.style_aug_global_weight
        
        for step in range(num_steps):
            adv_local_styles.requires_grad = True
            adv_global_style.requires_grad = True
            
            # 1. Compute combined styles: 
            # Strategy: All local styles first follow global drift (consistency),
            # then allow small local deviations (diversity)
            global_delta = adv_global_style - original_global_style  # [B, 6]
            global_delta_expanded = global_delta.unsqueeze(1).expand(-1, K, -1)  # [B, K, 6]
            
            # Apply global drift to all local styles (base shift)
            global_base = original_local_styles + global_delta_expanded  # [B, K, 6]
            
            # Compute local deviation from original
            local_delta = adv_local_styles - original_local_styles  # [B, K, 6]
            
            # Final style = global base + constrained local deviation
            # global_weight controls how much we enforce global consistency
            # e.g., global_weight=0.7 means 100% global drift + 30% local deviation
            combined_styles = global_base + (1 - global_weight) * local_delta  # [B, K, 6]
            
            # 2. Apply combined styles to images
            apply_mask = pixel_gt if model.style_aug_use_gt_region_style else None
            styled_images = self._apply_style_to_images(img_batch, combined_styles, gt_mask=apply_mask)
            
            # 3. Forward through backbone (gradient needed for PGD)
            adv_backbone_out = model.forward_image(styled_images, use_checkpoint=True)
            adv_backbone_feat = adv_backbone_out['backbone_fpn'][-1]

            # 4. Extract high_res_features if needed
            high_res_features = None
            if model.use_high_res_features_in_sam:
                high_res_features = [
                    adv_backbone_out['backbone_fpn'][0],
                    adv_backbone_out['backbone_fpn'][1]
                ]
            
            # 5. Generate prompts from GT (need combined mask for bbox)
            if pixel_gt is not None:
                # Combine all objects for prompt generation
                if pixel_gt.ndim == 4 and pixel_gt.shape[1] > 1:
                    combined_gt = pixel_gt.sum(dim=1, keepdim=True).clamp(0, 1)  # [B, 1, H, W]
                else:
                    combined_gt = pixel_gt  # Already [B, 1, H, W]
            else:
                combined_gt = None
            
            adv_prompts = model._generate_bbox_prompts_from_gt(combined_gt)
            adv_box_coords = torch.stack([adv_prompts[:, :2], adv_prompts[:, 2:]], dim=1)
            adv_point_inputs = {
                "point_coords": adv_box_coords,
                "point_labels": torch.tensor([[2, 3]], dtype=torch.int32, device=img_batch.device).expand(B, 2),
            }
            
            # 6. Forward through SAM heads to get BNDL outputs
            prev_suppress = getattr(model, "_suppress_nested_aue", False)
            model._suppress_nested_aue = True
            try:
                *_, adv_aux_outputs = model._forward_sam_heads(
                    backbone_features=adv_backbone_feat,
                    point_inputs=adv_point_inputs,
                    high_res_features=high_res_features,
                    multimask_output=False,
                    pixel_gt_for_aue=None,
                )
            finally:
                model._suppress_nested_aue = prev_suppress
            
            # 7. Extract BNDL outputs using helper
            adv_bndl_outputs = model._extract_bndl_outputs(
                adv_aux_outputs, pixel_bndl_model,
                compute_logits=True, compute_uncertainty=True, uq_sample_num=uq_sample_num
            )
            if adv_bndl_outputs is None:
                logging.warning("PGD (Mixed): Failed to extract BNDL outputs, stopping early")
                break
            
            # Guard against NaN/Inf in logits
            if not torch.isfinite(adv_bndl_outputs.pixel_logits).all():
                raise RuntimeError(f"PGD mixed step {step}: adversarial logits contain NaN/Inf")
            
            # 8. Prepare GT
            H_feat, W_feat = adv_bndl_outputs.pixel_logits.shape[1:3]
            if combined_gt is not None:
                adv_gts_prepared = model._prepare_gt_for_loss(combined_gt, (H_feat, W_feat))
            else:
                adv_gts_prepared = None
            
            # 9. Compute calibration loss (maximize to find adversarial styles)
            calibration_loss_adv = model._compute_uncertainty_calibration_loss(
                adv_bndl_outputs, adv_gts_prepared, pixel_bndl_model
            )
            
            # 12. Gradient ascent (maximize calibration loss to create hard samples)
            # Compute gradients for both local and global styles
            grad_local, grad_global = torch.autograd.grad(
                calibration_loss_adv, 
                [adv_local_styles, adv_global_style], 
                create_graph=False
            )
            
            with torch.no_grad():
                # Update local styles with local epsilon constraint
                adv_local_styles = adv_local_styles.detach() + step_size * grad_local.sign()
                delta_local = adv_local_styles - original_local_styles
                delta_local = torch.clamp(delta_local, -local_epsilon, local_epsilon)
                adv_local_styles = original_local_styles + delta_local
                
                # Update global style with global epsilon constraint
                adv_global_style = adv_global_style.detach() + step_size * grad_global.sign()
                delta_global = adv_global_style - original_global_style
                delta_global = torch.clamp(delta_global, -global_epsilon, global_epsilon)
                adv_global_style = original_global_style + delta_global
        
        # Combine final styles: global_base + constrained local deviation
        with torch.no_grad():
            global_delta_final = adv_global_style - original_global_style  # [B, 6]
            global_delta_expanded = global_delta_final.unsqueeze(1).expand(-1, K, -1)  # [B, K, 6]
            global_base_final = original_local_styles + global_delta_expanded  # [B, K, 6]
            local_delta_final = adv_local_styles - original_local_styles  # [B, K, 6]
            final_combined_styles = global_base_final + (1 - global_weight) * local_delta_final  # [B, K, 6]
        
        return final_combined_styles.detach()
    
    class _StyleGraphRefiner:
        """Helper for GCN-based style refinement with graph caching."""
        
        def __init__(self, impl: "ImageLevelStyleImpl", model: "SAM2Base", pixel_gt: torch.Tensor, img_batch: torch.Tensor, backbone_features: torch.Tensor | None = None):
            """
            Initialize refiner with cached graph structure and mask features.
            
            Visual features serve dual purposes:
            1. Used to build semantic edges in graph (based on cosine similarity)
            2. Cached and passed to GCN for fusion with style deltas (via MLP projection)
            
            Args:
                impl: ImageLevelStyleImpl instance (for accessing methods)
                model: SAM2Base instance (for accessing model attributes)
                pixel_gt: [B, K, H, W] ground truth masks
                img_batch: [B, 3, H, W] images
                backbone_features: [B, C, H, W] detached backbone features for extracting visual features
            """
            self.impl = impl
            self.model = model
            self.gcn = model.style_gcn
            
            # Build and cache graph structure (includes self-loops)
            if self.gcn is not None:
                with torch.no_grad():
                    edge_index, edge_weight = impl._build_style_graph(
                        pixel_gt, img_batch, backbone_features, model
                    )
                    
                self.edge_index = edge_index
                self.edge_weight = edge_weight
                self.stats = getattr(model, "_latest_gcn_stats", None)
                
                # Extract and cache mask features for GCN fusion
                # Design rationale:
                # - Cache features in __init__ (not in forward) for efficiency
                # - backbone_features are clean (not styled), representing style-invariant semantics
                # - These cached features are reused across all PGD steps
                # - Trade-off: efficiency (1x extraction) vs flexibility (can't use per-step features)
                if self.gcn.feature_dim > 0 and backbone_features is not None and pixel_gt is not None:
                    with torch.no_grad():
                        self.mask_features = impl._extract_mask_features(
                            backbone_features, pixel_gt
                        )  # [B, K, C] - will be projected to [B, K, 6] inside GCN
                else:
                    self.mask_features = None
            else:
                self.edge_index, self.edge_weight = None, None
                self.mask_features = None
                self.stats = None
        
        def refine_no_grad(self, delta: torch.Tensor, epsilon: float) -> torch.Tensor:
            """
            Apply GCN refinement without gradients (for PGD loop).
            
            Args:
                delta: [B, K, 6] style perturbations
                epsilon: clipping budget
            
            Returns:
                refined_delta: [B, K, 6] refined perturbations
            """
            if self.gcn is None or self.edge_index is None:
                return delta
            
            # Clear cache before GCN to reduce fragmentation
            torch.cuda.empty_cache()
            
            delta_detached = delta.detach()
            # Pass mask_features to GCN for fusion with style deltas
            refined_delta = self.gcn(delta_detached, self.edge_index, self.edge_weight,
                                    mask_features=self.mask_features)
            refined_delta = torch.clamp(refined_delta, -epsilon, epsilon).detach()
            
            # Cleanup
            del delta_detached
            torch.cuda.empty_cache()
            
            return refined_delta
        
        def refine_with_grad(self, delta: torch.Tensor, epsilon: float) -> torch.Tensor:
            """
            Apply GCN refinement with gradients (for training).
            
            Args:
                delta: [B, K, 6] style perturbations (requires_grad=True)
                epsilon: clipping budget
            
            Returns:
                refined_delta: [B, K, 6] refined perturbations
            """
            if self.gcn is None or self.edge_index is None:
                return delta
            
            # Pass mask_features to GCN for fusion with style deltas
            refined_delta = self.gcn(delta, self.edge_index, self.edge_weight,
                                    mask_features=self.mask_features)
            refined_delta = torch.clamp(refined_delta, -epsilon, epsilon)
            
            return refined_delta


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
        
        # Offset predictor network (image + mask -> offset field)
        self.offset_net = nn.Sequential(
            nn.Conv2d(image_channels + 1, 64, kernel_size=7, padding=3, stride=2),  # 1024 -> 512
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),  # 512 -> 256
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=3, padding=1),  # [B, 2, 256, 256]
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
        """
        Generate adversarial deformation offsets using PGD to maximize calibration loss.
        
        Similar to style PGD, but optimizes spatial deformation offsets instead of style parameters.
        """
        B, _, H_img, W_img = img_batch.shape
        K = pixel_gt.shape[1]
        device = img_batch.device
        
        # Initialize offsets using offset_net (learned initialization)
        initial_offsets = []
        for k in range(K):
            mask_k = pixel_gt[:, k:k + 1, :, :].float()  # [B, 1, H, W]
            
            # Skip empty masks and background
            if mask_k.sum() == 0:
                initial_offsets.append(torch.zeros(B, 2, H_img, W_img, device=device))
                continue
            
            mask_area = mask_k.sum() / (B * H_img * W_img)
            if (k == K - 1) and (mask_area > 0.5):
                initial_offsets.append(torch.zeros(B, 2, H_img, W_img, device=device))
                continue
            
            # Predict initial offset (output is 256x256, need to upsample)
            img_mask_input = torch.cat([img_batch, mask_k], dim=1)  # [B, 4, H, W]
            offset_raw = self.offset_net(img_mask_input)  # [B, 2, 256, 256]
            
            # Upsample to image resolution
            offset_full = F.interpolate(
                offset_raw, size=(H_img, W_img), mode='bilinear', align_corners=False
            )  # [B, 2, H, W]
            
            # Scale by epsilon and mask
            offset_scaled = offset_full * self.epsilon * mask_k
            initial_offsets.append(offset_scaled)
        
        # Stack initial offsets: [B, K, 2, H, W]
        initial_offsets_stacked = torch.stack(initial_offsets, dim=1)
        
        # Use PGD to find adversarial offsets (maximize calibration loss)
        adversarial_offsets = self._pgd_adversarial_deformation(
            img_batch=img_batch,
            initial_offsets=initial_offsets_stacked,
            pixel_gt=pixel_gt,
            model=model,
            num_steps=3,
            step_size=5.0,
            epsilon=self.epsilon,
        )
        
        # Return result with adversarial offsets
        return AugmentationResult(
            features=clean_features,  # Will be replaced after warping
            high_res_features=clean_high_res,
            intermediate_images=img_batch,  # Original images
            num_backbone_forwards=0,  # Will forward after combining with style
            mode="image_level",
            aug_type="deformation",
            deformation_offsets=adversarial_offsets,  # [B, K, 2, H, W]
        )
    
    def _pgd_adversarial_deformation(
        self,
        img_batch: torch.Tensor,
        initial_offsets: torch.Tensor,
        pixel_gt: torch.Tensor,
        model: nn.Module,
        num_steps: int = 3,
        step_size: float = 5.0,
        epsilon: float = 30.0,
    ) -> torch.Tensor:
        """
        Use PGD to find adversarial deformation offsets that maximize calibration loss.
        
        Args:
            img_batch: [B, 3, H, W] Original images
            initial_offsets: [B, K, 2, H, W] Initial offsets from offset_net
            pixel_gt: [B, K, H, W] Ground truth masks
            model: SAM2 model for forward pass
            num_steps: Number of PGD iterations
            step_size: Step size for gradient ascent (in pixels)
            epsilon: Maximum offset magnitude (in pixels)
        
        Returns:
            adversarial_offsets: [B, K, 2, H, W] Adversarial offsets
        """
        B, K, _, H_img, W_img = initial_offsets.shape
        device = img_batch.device
        
        # Start from learned initialization
        adv_offsets = initial_offsets.clone().detach()
        
        # Get pixel_bndl_model for uncertainty computation
        pixel_bndl_model = None
        if hasattr(model, 'sam_mask_decoder') and hasattr(model.sam_mask_decoder, 'pixel_bndl'):
            pixel_bndl_model = model.sam_mask_decoder.pixel_bndl
        
        # Pre-compute combined GT mask for loss computation
        if pixel_gt.shape[1] > 1:
            combined_gt = pixel_gt.sum(dim=1, keepdim=True).clamp(0, 1)
        else:
            combined_gt = pixel_gt
        
        for step in range(num_steps):
            # Clear cache between steps
            if step > 0:
                torch.cuda.empty_cache()
            
            # Make offsets require grad (ensure it's a leaf variable)
            adv_offsets = adv_offsets.clone().requires_grad_(True)
            
            # Apply current offsets to warp images and GT
            warped_img, warped_gt = self._apply_offsets_to_images(
                img_batch, pixel_gt, adv_offsets
            )
            
            # Forward through backbone
            with torch.no_grad():
                backbone_out = model.forward_image(warped_img, use_checkpoint=True)
            adv_features = backbone_out['backbone_fpn'][-1]
            
            # Extract high_res_features if needed
            high_res_features = None
            if model.use_high_res_features_in_sam:
                high_res_features = [
                    backbone_out['backbone_fpn'][0],
                    backbone_out['backbone_fpn'][1]
                ]
            
            # Forward through SAM heads (suppress nested AUE)
            prev_suppress = getattr(model, "_suppress_nested_aue", False)
            model._suppress_nested_aue = True
            try:
                *_, adv_aux_outputs = model._forward_sam_heads(
                    backbone_features=adv_features,
                    high_res_features=high_res_features,
                    pixel_gt_for_aue=None,
                    multimask_output=False,
                )
            finally:
                model._suppress_nested_aue = prev_suppress
            
            # Extract BNDL outputs using helper
            adv_bndl_outputs = model._extract_bndl_outputs(
                adv_aux_outputs, pixel_bndl_model,
                compute_logits=True, compute_uncertainty=True, uq_sample_num=20
            )
            if adv_bndl_outputs is None:
                logging.warning("Deform PGD: Failed to extract BNDL outputs, stopping early")
                break
            
            # Prepare GT
            H_feat, W_feat = adv_bndl_outputs.pixel_logits.shape[1:3]
            combined_gt_prepared = model._prepare_gt_for_loss(combined_gt, (H_feat, W_feat))
            
            # Compute calibration loss (maximize to find adversarial offsets)
            calibration_loss = model._compute_uncertainty_calibration_loss(
                adv_bndl_outputs, combined_gt_prepared, pixel_bndl_model
            )
            
            # Gradient ascent (maximize calibration loss to create hard samples)
            grad = torch.autograd.grad(
                calibration_loss, adv_offsets, 
                create_graph=False, 
                allow_unused=True
            )[0]
            
            # If no gradient (e.g., all objects are empty/background), skip update
            if grad is None:
                logging.warning("Deform PGD: No gradient computed (all objects empty/background), skipping")
                break
            
            with torch.no_grad():
                # Gradient ascent step
                adv_offsets = adv_offsets.detach() + step_size * grad.sign()
                
                # Project to epsilon ball (per-object constraint)
                delta = adv_offsets - initial_offsets
                # Clamp delta magnitude per object
                for k in range(K):
                    delta_k = delta[:, k, :, :, :]  # [B, 2, H, W]
                    delta_k = torch.clamp(delta_k, -epsilon, epsilon)
                    delta[:, k, :, :, :] = delta_k
                
                adv_offsets = initial_offsets + delta
            
            # Offset network refinement after each PGD step (similar to GCN for style)
            # Skip last step: no next iteration to benefit from refinement (saves computation)
            if step < num_steps - 1 and model.training:
                # Training mode: refine with gradients to train offset_net
                refined_offsets = []
                for k in range(K):
                    mask_k = pixel_gt[:, k:k + 1, :, :].float()
                    
                    # Skip empty masks and background
                    if mask_k.sum() == 0:
                        refined_offsets.append(adv_offsets[:, k, :, :, :])
                        continue
                    
                    mask_area = mask_k.sum() / (B * H_img * W_img)
                    if (k == K - 1) and (mask_area > 0.5):
                        refined_offsets.append(adv_offsets[:, k, :, :, :])
                        continue
                    
                    # Call offset_net to refine current offsets
                    img_mask_input = torch.cat([img_batch, mask_k], dim=1)
                    offset_raw = self.offset_net(img_mask_input)  # [B, 2, 256, 256]
                    
                    # Upsample to image resolution
                    offset_full = F.interpolate(
                        offset_raw, size=(H_img, W_img), 
                        mode='bilinear', align_corners=False
                    )
                    
                    # Apply epsilon constraint and mask
                    offset_refined = offset_full * epsilon * mask_k
                    refined_offsets.append(offset_refined)
                
                # Stack and detach for next PGD step
                refined_offsets_stacked = torch.stack(refined_offsets, dim=1)
                adv_offsets = refined_offsets_stacked.detach()
            
            # Cleanup
            del warped_img, warped_gt, backbone_out, adv_features
            del adv_aux_outputs, adv_bndl_outputs, calibration_loss, grad
            if high_res_features is not None:
                del high_res_features
        
        return adv_offsets.detach()
    
    def _apply_offsets_to_images(
        self,
        img_batch: torch.Tensor,
        pixel_gt: torch.Tensor,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply deformation offsets to images and GT masks.
        
        Args:
            img_batch: [B, 3, H, W] Original images
            pixel_gt: [B, K, H, W] Ground truth masks
            offsets: [B, K, 2, H, W] Deformation offsets
        
        Returns:
            warped_img: [B, 3, H, W] Warped image
            warped_gt: [B, K, H, W] Warped GT masks
        """
        B, K, _, H_img, W_img = offsets.shape
        device = img_batch.device
        
        # Identify valid objects (non-empty, non-background)
        masks_float = pixel_gt.float()
        mask_areas = masks_float.sum(dim=(2, 3))
        is_empty = (mask_areas.sum(dim=0) == 0)
        mask_area_ratio = mask_areas / (H_img * W_img)
        is_bg_per_sample = (mask_area_ratio > 0.5)
        is_bg = torch.zeros(K, dtype=torch.bool, device=device)
        is_bg[-1] = is_bg_per_sample[:, -1].all()
        valid_objects = ~(is_empty | is_bg)
        valid_indices = torch.where(valid_objects)[0]
        
        if len(valid_indices) == 0:
            return img_batch, pixel_gt.clone()
        
        # Start with image where valid objects are removed
        valid_masks = masks_float[:, valid_indices, :, :]  # [B, K_valid, H, W]
        valid_masks_union = valid_masks.sum(dim=1, keepdim=True).clamp(0, 1)  # [B, 1, H, W]
        warped_img = img_batch * (1 - valid_masks_union)
        warped_gt = pixel_gt.clone()
        
        # Apply offsets to each valid object
        for _, k_idx in enumerate(valid_indices):
            k_idx_scalar = k_idx.item()
            offset_k = offsets[:, k_idx_scalar, :, :, :]  # [B, 2, H, W]
            mask_k = masks_float[:, k_idx_scalar, :, :].unsqueeze(1)  # [B, 1, H, W]
            
            # Create sampling grid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H_img, device=device, dtype=torch.float32),
                torch.arange(W_img, device=device, dtype=torch.float32),
                indexing='ij'
            )
            grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
            grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
            
            # Apply offset
            sampling_y = grid_y + offset_k[:, 1, :, :]  # [B, H, W]
            sampling_x = grid_x + offset_k[:, 0, :, :]  # [B, H, W]
            
            # Normalize to [-1, 1]
            sampling_x_norm = 2.0 * sampling_x / (W_img - 1) - 1.0
            sampling_y_norm = 2.0 * sampling_y / (H_img - 1) - 1.0
            
            # Stack to [B, H, W, 2]
            sampling_grid = torch.stack([sampling_x_norm, sampling_y_norm], dim=-1)
            
            # Warp image and mask
            warped_img_k = F.grid_sample(
                img_batch, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True
            )
            warped_mask_k = F.grid_sample(
                mask_k, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True
            )
            
            # Update GT (detach to avoid affecting gradient flow to warped_img)
            with torch.no_grad():
                warped_gt[:, k_idx_scalar, :, :] = (warped_mask_k.squeeze(1) > 0.5).float()
            
            # Composite (this operation preserves gradients to offsets)
            warped_img = warped_img * (1 - warped_mask_k) + warped_img_k * warped_mask_k
        
        return warped_img, warped_gt


class FeatureLevelDeformationImpl(nn.Module):
    """
    Feature-level deformation using memory encoder components.
    
    Architecture:
        1. MaskDownSampler: Encode masks to feature space [B, 256, H, W]
        2. Image feature projection: Project image features
        3. Fusion: Combine mask and image features (memory encoder style)
        4. Offset prediction: Generate deformation offsets from fused features
    
    Supports two operational modes:
        - Mode 1 (Efficient): Reuse provided clean_features (no extra forward pass)
        - Mode 2 (Flexible): Encode img_batch on-demand via model.forward_image()
    
    Args:
        feature_dim: Backbone feature dimension (default: 256)
        epsilon: Max deformation magnitude in feature space (default: 0.15)
        use_soft_composite: Use soft compositing for multi-object overlaps
        temperature: Softmax temperature for soft compositing
        use_gcn: Use GCN for multi-object coordination (not implemented yet)
        gcn_num_layers: Number of GCN layers
        num_deform_groups: Deformable convolution groups
        init_from_memory_encoder: Initialize weights from memory encoder
        freeze_mask_encoder: Freeze mask encoder during training
        image_size: Target image resolution (default: 1024)
    """
    def __init__(
        self,
        feature_dim: int = 256,
        epsilon: float = 0.15,
        use_soft_composite: bool = True,
        temperature: float = 1.0,
        use_gcn: bool = False,
        gcn_num_layers: int = 2,
        num_deform_groups: int = 4,
        init_from_memory_encoder: bool = True,
        freeze_mask_encoder: bool = False,
        image_size: int = 1024,
        **kwargs
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.use_soft_composite = use_soft_composite
        self.temperature = temperature
        self.use_gcn = use_gcn
        self.init_from_memory_encoder = init_from_memory_encoder
        self.freeze_mask_encoder = freeze_mask_encoder
        
        # Import memory encoder components
        from sam2.modeling.memory_encoder import MaskDownSampler, Fuser, CXBlock
        
        # 1. Mask encoder (from memory_encoder.mask_downsampler)
        self.mask_encoder = MaskDownSampler(
            embed_dim=feature_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            total_stride=16,  # 1024 -> 64
        )
        
        # 2. Image feature projection (from memory_encoder.pix_feat_proj)
        self.img_feat_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        
        # 3. Feature fusion module (from memory_encoder.fuser)
        self.fuser = Fuser(
            layer=CXBlock(
                dim=feature_dim,
                kernel_size=7,
                padding=3,
                layer_scale_init_value=1e-6,
                use_dwconv=True
            ),
            num_layers=2
        )
        
        # 4. Deformation module (uses fused features)
        # Note: Produces both feature-level deformation and image-level offsets
        self.deform_module = FeatureBasedDeformModule(
            feature_dim=feature_dim,
            epsilon=epsilon,
            image_size=image_size,
        )
        
        # 5. Soft compositor for multi-object overlaps
        if self.use_soft_composite:
            self.compositor = SoftCompositor(temperature=temperature)
        
        # 6. Optional GCN (placeholder)
        if self.use_gcn:
            logging.warning("GCN coordination not implemented yet")
            self.deform_gcn = None
    
    def load_memory_encoder_weights(self, memory_encoder):
        """
        Initialize weights from pretrained memory encoder.
        
        Copies weights from:
            - memory_encoder.mask_downsampler -> self.mask_encoder
            - memory_encoder.pix_feat_proj -> self.img_feat_proj
            - memory_encoder.fuser -> self.fuser
        
        The deform_module is NOT initialized (trained from scratch).
        
        Args:
            memory_encoder: Pretrained MemoryEncoder module
        """
        # Copy mask encoder weights
        self.mask_encoder.load_state_dict(
            memory_encoder.mask_downsampler.state_dict()
        )
        
        # Copy image feature projection weights
        self.img_feat_proj.load_state_dict(
            memory_encoder.pix_feat_proj.state_dict()
        )
        
        # Copy fuser weights
        self.fuser.load_state_dict(
            memory_encoder.fuser.state_dict()
        )
        
        logging.info("✓ Initialized deformation network from memory encoder weights")
        
        # Optionally freeze mask encoder
        if self.freeze_mask_encoder:
            for param in self.mask_encoder.parameters():
                param.requires_grad = False
            logging.info("✓ Frozen mask encoder weights")
    
    def _batch_process_objects(
        self,
        pixel_gt: torch.Tensor,
        pixel_gt_resized: torch.Tensor,
        clean_features: torch.Tensor,
        B: int,
        K: int,
        H_feat: int,
        W_feat: int,
        H_img: int,
        W_img: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Memory-optimized object processing with max objects limit and sequential processing.
        
        Strategy:
        1. Limit to max 4 objects (most important by mask area)
        2. Process objects sequentially but with aggressive cleanup
        3. Skip empty/background objects entirely
        
        Args:
            pixel_gt: [B, K, H_img, W_img] Original resolution masks
            pixel_gt_resized: [B, K, H_feat, W_feat] Feature resolution masks
            clean_features: [B, C, H_feat, W_feat] Image features
            B, K, H_feat, W_feat, H_img, W_img: Shape parameters
        
        Returns:
            deformed_list: List of [B, C, H_feat, W_feat] deformed features per object
            mask_list: List of [B, 1, H_feat, W_feat] masks per object
            all_image_offsets_list: List of [B, 2, H_img, W_img] offsets per object
        """
        device = clean_features.device
        
        # Step 1: Identify valid objects and rank by importance (mask area)
        mask_areas = pixel_gt_resized.sum(dim=(0, 2, 3))  # [K], sum across batch
        is_empty = (mask_areas == 0)
        
        # Background detection: last object with > 50% area
        area_ratios = mask_areas / (B * H_feat * W_feat)
        is_background = torch.zeros(K, dtype=torch.bool, device=device)
        if K > 0 and area_ratios[-1] > 0.5:
            is_background[-1] = True
        
        valid_mask = ~(is_empty | is_background)  # [K]
        valid_indices = torch.where(valid_mask)[0]
        
        n_valid = len(valid_indices)
        
        # Initialize output lists
        # FIXED: Don't use [tensor] * K which creates K references to same tensor (memory leak)
        deformed_list = [clean_features for _ in range(K)]  # Default: original features
        mask_list = [pixel_gt_resized[:, k:k + 1] for k in range(K)]
        all_image_offsets_list = [torch.zeros(B, 2, H_img, W_img, device=device) for _ in range(K)]
        
        if n_valid == 0:
            return deformed_list, mask_list, all_image_offsets_list
        
        # Pre-compute image projection once (shared across objects)
        img_proj = self.img_feat_proj(clean_features)  # [B, 256, H_feat, W_feat]
        
        # Pre-compute base grid once
        norm_grid = torch.meshgrid(
            torch.linspace(-1, 1, H_feat, device=device),
            torch.linspace(-1, 1, W_feat, device=device),
            indexing='ij'
        )
        norm_grid = torch.stack(norm_grid[::-1], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        
        # Step 2: Process valid objects sequentially with aggressive cleanup
        for k_idx in valid_indices.tolist():
            mask_k_original = pixel_gt[:, k_idx:k_idx + 1]  # [B, 1, H_img, W_img]
            mask_k_resized = pixel_gt_resized[:, k_idx:k_idx + 1]  # [B, 1, H_feat, W_feat]
            
            # Encode mask
            mask_features = self.mask_encoder(mask_k_original)  # [B, 256, H_feat, W_feat]
            
            # Fuse
            fused = img_proj + mask_features
            del mask_features  # Immediate cleanup
            
            fused = self.fuser(fused)  # [B, 256, H_feat, W_feat]
            
            # Deform
            feature_offsets_k, image_offsets_k = self.deform_module(fused, mask_k_resized)
            del fused  # Immediate cleanup
            
            # Apply deformation
            offset_norm = feature_offsets_k.permute(0, 2, 3, 1).clone()
            del feature_offsets_k  # Immediate cleanup
            
            offset_norm[..., 0] = offset_norm[..., 0] / (W_feat / 2.0)
            offset_norm[..., 1] = offset_norm[..., 1] / (H_feat / 2.0)
            
            sampling_grid = norm_grid + offset_norm
            del offset_norm  # Immediate cleanup
            
            deformed_k = F.grid_sample(
                clean_features,
                sampling_grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=False
            )
            del sampling_grid  # Immediate cleanup
            
            # Store results
            deformed_list[k_idx] = deformed_k
            all_image_offsets_list[k_idx] = image_offsets_k
            
            # FIXED: Aggressive cleanup after each object to prevent memory accumulation
            del deformed_k, image_offsets_k, mask_k_original, mask_k_resized
            torch.cuda.empty_cache()
        
        return deformed_list, mask_list, all_image_offsets_list
    
    def apply(
        self,
        img_batch: torch.Tensor = None,
        clean_features: torch.Tensor = None,
        clean_high_res: list[torch.Tensor] | None = None,
        pixel_gt: torch.Tensor = None,
        model: nn.Module = None,
    ) -> AugmentationResult:
        """
        Apply feature-level deformation with flexible interface.
        
        Operational Modes:
            Mode 1 (Efficient - Feature Reuse):
                Provide clean_features and clean_high_res directly.
                No extra backbone forward pass needed.
                Example: deform_augmenter.apply(
                    img_batch=img, 
                    clean_features=features, 
                    pixel_gt=gt, 
                    model=model
                )
            
            Mode 2 (Flexible - On-Demand Encoding):
                Provide img_batch and model only.
                Will encode img_batch via model.forward_image() internally.
                Example: deform_augmenter.apply(
                    img_batch=img, 
                    pixel_gt=gt, 
                    model=model
                )
        
        Args:
            img_batch: [B, 3, H, W] Input images (required)
            clean_features: [B, C, H_feat, W_feat] Backbone features (optional)
            clean_high_res: List of high-res features (optional)
            pixel_gt: [B, K, H, W] Ground truth masks (required)
            model: SAM2Base model (required for Mode 2)
        
        Returns:
            AugmentationResult with deformed features and metadata
        """
        # === Input validation ===
        if img_batch is None or pixel_gt is None:
            raise ValueError("img_batch and pixel_gt are required")
        
        # === Mode detection ===
        if clean_features is None:
            # Mode 2: On-demand encoding
            if model is None:
                raise ValueError(
                    "Must provide either clean_features (Mode 1) or model (Mode 2)"
                )
            
            logging.debug("Deformation Mode 2: Encoding image on-demand")
            with torch.no_grad():
                backbone_out = model.forward_image(img_batch, use_checkpoint=True)
            clean_features = backbone_out['backbone_fpn'][-1]
            
            if model.use_high_res_features_in_sam and clean_high_res is None:
                clean_high_res = [
                    backbone_out['backbone_fpn'][0],
                    backbone_out['backbone_fpn'][1]
                ]
            num_forwards = 1
        else:
            # Mode 1: Feature reuse (efficient)
            logging.debug("Deformation Mode 1: Reusing provided features")
            num_forwards = 0
        
        # === Main deformation logic ===
        B, C, H_feat, W_feat = clean_features.shape
        K = pixel_gt.shape[1]
        H_img, W_img = img_batch.shape[2:]
        
        # Resize masks to feature resolution if needed
        if pixel_gt.shape[2:] != (H_feat, W_feat):
            pixel_gt_resized = F.interpolate(
                pixel_gt.float(),
                size=(H_feat, W_feat),
                mode='bilinear',
                align_corners=False
            )
        else:
            pixel_gt_resized = pixel_gt.float()
        
        # === OPTIMIZED: Batch processing of all objects ===
        # This replaces the per-object loop to save memory and computation
        deformed_list, mask_list, all_image_offsets_list = self._batch_process_objects(
            pixel_gt, pixel_gt_resized, clean_features, B, K, H_feat, W_feat, H_img, W_img
        )
        
        # Compose deformed features
        if self.use_soft_composite and len(deformed_list) > 0:
            final_features = self.compositor(deformed_list, mask_list)
        else:
            final_features = torch.stack(deformed_list, dim=0).mean(dim=0)
        
        # Stack image-level offsets: [B, K, 2, H_img, W_img]
        stacked_image_offsets = torch.stack(all_image_offsets_list, dim=1)
        
        return AugmentationResult(
            features=final_features,
            high_res_features=clean_high_res,
            intermediate_images=None,
            num_backbone_forwards=num_forwards,
            mode="feature_level",
            aug_type="deformation",
            deformation_offsets=stacked_image_offsets,  # [B, K, 2, H_img, W_img] for warping
        )


class FeatureBasedDeformModule(nn.Module):
    """
    Dense flow deformation module with Gradient Reversal Layer (GRL).
    
    Architecture:
        Input: Fused features [B, 256, H_feat, W_feat]
        ↓
        Offset Predictor Network (3 conv layers)
        ↓
        Offsets [B, 2, H_feat, W_feat] (constrained by epsilon)
        ↓
        Gradient Reversal Layer (GRL)
        ↓
        Upsample to image resolution
        ↓
        Image offsets [B, 2, H_img, W_img]
    
    Design rationale:
        - Single branch ensures gradients flow from image warping back to offset predictor.
        - GRL enables adversarial training (maximize loss) within a minimization loop.
    
    Args:
        feature_dim: Feature dimension (default: 256)
        epsilon: Max offset magnitude in feature space (default: 0.15 pixels)
        image_size: Target image resolution for offset prediction (default: 1024)
    """
    def __init__(
        self, 
        feature_dim: int = 256, 
        epsilon: float = 0.15,
        image_size: int = 1024,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.image_size = image_size
        
        # Offset predictor: fused_features -> dense offsets (2 channels)
        self.offset_net = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1), # 2 channels for (dx, dy)
        )
        
        # Initialize: small values for last layer
        nn.init.normal_(self.offset_net[-1].weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.offset_net[-1].bias)
        
        # Gradient Reversal Layer
        self.grl = GRL(alpha=1.0)
    
    def forward(
        self, 
        fused_features: torch.Tensor,
        object_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply deformation prediction with GRL.
        
        Args:
            fused_features: [B, C, H_feat, W_feat] Pre-fused image+mask features
            object_mask: [B, 1, H_feat, W_feat] Object mask (for masking/weighting)
        
        Returns:
            feature_offsets: [B, 2, H_feat, W_feat] Offsets in feature resolution
            image_offsets: [B, 2, H_img, W_img] Offsets in image resolution
        """
        # Apply Gradient Reversal Layer to input features
        # This reverses gradients flowing back to the encoder while allowing
        # normal gradients to train the offset network
        fused_features_grl = self.grl(fused_features)
        
        # Predict raw offsets from GRL features
        # Range: (-1, 1) after tanh
        raw_offsets = self.offset_net(fused_features_grl)  # [B, 2, H_feat, W_feat]
        raw_offsets = torch.tanh(raw_offsets)
        
        # Define scale factor (1024 / 64 = 16)
        scale_factor = 16.0
        
        # 1. Compute Image-level Offsets (Target Resolution)
        # Scale raw offsets by epsilon (defined in IMAGE pixels)
        # We first upsample the raw field to image size to get smooth dense flow
        image_raw_offsets = F.interpolate(
            raw_offsets,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=False
        )
        image_offsets = image_raw_offsets * self.epsilon # [B, 2, H_img, W_img] in pixels
        
        # 2. Compute Feature-level Offsets (Source Resolution)
        # Feature offset = Image offset / scale_factor
        # We use the raw_offsets directly scaled by (epsilon / scale_factor)
        # This avoids downsampling artifacts and keeps consistency
        feature_offsets = raw_offsets * (self.epsilon / scale_factor) # [B, 2, H_feat, W_feat] in feature pixels
        
        return feature_offsets, image_offsets


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


class StyleAdversarialNetwork(nn.Module):
    """
    Neural network that predicts adversarial style transformations with GRL.
    
    Architecture:
        Features [B, C, H, W]
        ↓
        Global Average Pooling
        ↓
        Feature vector [B, C]
        ↓
        MLP (C → 128 → 6K)  # 6 params (μ,σ for RGB) × K objects
        ↓
        Style parameters [B, K, 6]
        ↓
        Gradient Reversal Layer (GRL)
        ↓
        Adversarial styles [B, K, 6]
    
    Design rationale:
        - Single forward pass (no PGD loop) for memory efficiency
        - GRL enables adversarial training within minimization loop
        - Predicts per-object styles for multi-object consistency
    
    Args:
        feature_dim: Feature dimension (default: 256)
        num_objects: Max number of objects (default: 11, including background)
        epsilon: Max style perturbation magnitude (default: 2.0)
    """
    def __init__(
        self,
        feature_dim: int = 256,
        num_objects: int = 11,
        epsilon: float = 2.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_objects = num_objects
        self.epsilon = epsilon
        
        # Style predictor: features → per-object style parameters
        # Output: [B, num_objects * 6] where 6 = (μ_R, μ_G, μ_B, σ_R, σ_G, σ_B)
        self.style_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, H, W] → [B, C, 1, 1]
            nn.Flatten(),  # [B, C]
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_objects * 6),  # Per-object style parameters
        )
        
        # Initialize: small residuals
        nn.init.normal_(self.style_net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.style_net[-1].bias)
        
        # Gradient Reversal Layer
        self.grl = GRL(alpha=1.0)
    
    def forward(
        self,
        features: torch.Tensor,
        original_styles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict adversarial style transformations with GRL.
        
        Args:
            features: [B, C, H, W] Backbone features
            original_styles: [B, K, 6] Original image styles
        
        Returns:
            adv_styles: [B, K, 6] Adversarial styles (constrained by epsilon)
        """
        B = features.shape[0]
        
        # Apply Gradient Reversal Layer to input features
        # This reverses gradients flowing back to the encoder while allowing
        # normal gradients to train the style network
        features_grl = self.grl(features)
        
        # Predict style residuals from GRL features
        style_residuals = self.style_net(features_grl)  # [B, num_objects * 6]
        style_residuals = style_residuals.view(B, self.num_objects, 6)  # [B, K, 6]
        
        # Constrain by epsilon (use tanh normalization)
        style_residuals = torch.tanh(style_residuals) * self.epsilon
        
        # Add to original styles
        K_actual = original_styles.shape[1]
        adv_styles = original_styles + style_residuals[:, :K_actual, :]
        
        return adv_styles
