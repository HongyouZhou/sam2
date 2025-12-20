# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Adversarial Attack Framework for SAM2

Provides unified interface for both image-level and feature-level attacks,
including style perturbations and deformations (DG-Font style).

Key components:
- AdversarialResult: Container for attack outputs
- AdversarialAttacker: Unified interface for all attack types
- ImageLevelStyleImpl: Image-space style attack (existing approach)
- FeatureLevelDeformationImpl: Feature-space deformation (DG-Font style)
- FeatureLevelStyleImpl: Feature-space style attack (future)
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

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
class AdversarialResult:
    """
    Unified container for adversarial attack results.
    
    Encapsulates outputs from different attack modes (image-level vs feature-level)
    and types (style vs deformation), providing a consistent interface.
    
    Attributes:
        features: [B, C, H, W] Augmented backbone features (required)
        high_res_features: Optional list of high-resolution features for SAM decoder
        intermediate_images: Optional intermediate images (e.g., styled images) for debugging
        num_backbone_forwards: Number of backbone forward passes required (for monitoring)
        mode: Attack mode ("image_level" or "feature_level")
        aug_type: Attack type ("style" or "deformation")
        original_styles: Optional original style statistics (for style attack visualization)
        adversarial_styles: Optional adversarial style statistics (for style attack visualization)
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


class AdversarialAttacker(nn.Module):
    """
    Unified interface for adversarial attacks.
    
    Supports multiple attack types (style, deformation) and modes (image-level, feature-level).
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
                raise ValueError(f"Unknown mode for deformation: {self.mode}")
        
        else:
            raise ValueError(f"Unknown augmentation type: {self.aug_type}")
    
    def predict_params(
        self,
        clean_features: torch.Tensor,
        pixel_gt: torch.Tensor,
        model: nn.Module,
        **kwargs
    ):
        """
        Predict adversarial parameters (e.g., style codes, deformation offsets).
        
        Args:
            clean_features: [B, C, H, W] Clean backbone features
            pixel_gt: [B, K, H, W] Ground truth masks
            model: SAM2 model
            **kwargs: Additional arguments
        
        Returns:
            params: Adversarial parameters (type depends on implementation)
        """
        return self.impl.predict_params(
            clean_features=clean_features,
            pixel_gt=pixel_gt,
            model=model,
            **kwargs
        )
    
    def apply_transform(
        self,
        params: torch.Tensor | dict,
        img_batch: torch.Tensor | None = None,
        clean_features: torch.Tensor | None = None,
        pixel_gt: torch.Tensor | None = None,
        model: nn.Module | None = None,
        **kwargs
    ):
        """
        Apply the transformation using predicted parameters.
        
        Args:
            params: Adversarial parameters predicted by predict_params
            img_batch: [B, 3, H, W] Input images (for image-level aug)
            clean_features: [B, C, H, W] Clean features (for feature-level aug)
            pixel_gt: [B, K, H, W] Ground truth masks
            model: SAM2 model
            **kwargs: Additional arguments
        
        Returns:
            transformed: Transformed images or features
        """
        return self.impl.apply_transform(
            img_batch=img_batch,
            clean_features=clean_features,
            params=params,
            pixel_gt=pixel_gt,
            model=model,
            **kwargs
        )

    def apply(
        self,
        img_batch: torch.Tensor | None,
        clean_features: torch.Tensor | None,
        clean_high_res: list[torch.Tensor] | None,
        pixel_gt: torch.Tensor | None,
        model: nn.Module,
    ) -> AdversarialResult:
        """
        Apply the adversarial attack.
        
        Args:
            img_batch: [B, 3, H, W] Input images (for image-level)
            clean_features: [B, C, H, W] Clean features (for feature-level)
            clean_high_res: List of high-res features
            pixel_gt: [B, K, H, W] Ground truth masks
            model: SAM2 model
        
        Returns:
            AdversarialResult containing augmented features/images
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
    
    def predict_params(
        self,
        clean_features: torch.Tensor,
        pixel_gt: torch.Tensor,
        model: "SAM2Base",
        img_batch: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Predict adversarial style parameters.
        
        Args:
            clean_features: [B, C, H, W] Clean backbone features
            pixel_gt: [B, K, H, W] Ground truth masks
            model: SAM2Base model
            img_batch: [B, 3, H, W] Input images (needed for original style extraction)
            
        Returns:
            adv_styles: [B, K, 6] Adversarial style parameters
        """
        if img_batch is None:
            raise ValueError("ImageLevelStyleImpl.predict_params requires img_batch")
            
        # 1. Prepare inputs (extract original styles)
        pixel_gt_normalized, original_styles = self._prepare_style_adversary_inputs(
            img_batch, pixel_gt, model
        )
        
        # 2. Predict adversarial styles using neural network + GRL
        adv_styles = self.style_net(
            clean_features, 
            original_styles, 
            pixel_gt=pixel_gt_normalized
        )
        
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
        
        return adv_styles

    def apply_transform(
        self,
        img_batch: torch.Tensor,
        params: torch.Tensor,
        model: "SAM2Base",
        pixel_gt: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply style transformation to images.
        
        Args:
            img_batch: [B, 3, H, W] Input images
            params: [B, K, 6] Adversarial style parameters
            model: SAM2Base model (required for accessing config)
            pixel_gt: [B, K, H, W] Ground truth masks (optional, for region-based style)
            
        Returns:
            styled_images: [B, 3, H, W] Styled images
        """
        # Determine if we should use GT region style
        use_gt_region = self.use_gt_region_style
        if model is not None:
            use_gt_region = model.style_adv_use_gt_region_style
            
        # Prepare mask if needed
        apply_mask = None
        if use_gt_region and pixel_gt is not None:
            # Normalize pixel_gt if needed (similar to _prepare_style_adversary_inputs logic)
            if pixel_gt.ndim == 4 and pixel_gt.shape[1] > 1:
                 # Check if we need to handle background channel logic here?
                 # For simplicity, we assume pixel_gt passed here is already appropriate or we use it as is.
                 # But _prepare_style_adversary_inputs does some normalization.
                 # Let's assume pixel_gt is [B, K, H, W] and we use it directly.
                 pass
            apply_mask = pixel_gt
            
        styled_images = self._apply_style_to_images(img_batch, params, gt_mask=apply_mask)
        return styled_images

    def apply(
        self,
        img_batch: torch.Tensor,
        clean_features: torch.Tensor,
        clean_high_res: list[torch.Tensor] | None,
        pixel_gt: torch.Tensor,
        model: "SAM2Base",
    ) -> AdversarialResult:
        """
        Apply image-level style augmentation with GRL (self-contained).
        
        Args:
            img_batch: [B, 3, H, W] Input images
            clean_features: [B, C, H, W] Clean backbone features (for style network)
            clean_high_res: List of high-res features (not used)
            pixel_gt: [B, K, H, W] Ground truth masks
            model: SAM2Base model (for accessing shared components)
        
        Returns:
            AdversarialResult with styled features and num_backbone_forwards=1
        """
        # 1. Predict adversarial styles
        adv_styles = self.predict_params(
            clean_features=clean_features,
            pixel_gt=pixel_gt,
            model=model,
            img_batch=img_batch
        )
        
        # 2. Apply styles to images
        styled_images = self.apply_transform(
            img_batch=img_batch,
            params=adv_styles,
            pixel_gt=pixel_gt,
            model=model
        )
        
        # 3. Forward through backbone
        # We use checkpointing to save memory. We do NOT freeze backbone weights here
        # because dynamic freezing breaks activation checkpointing.
        backbone_out = model.forward_image(styled_images, use_checkpoint=True)
        styled_features = backbone_out['backbone_fpn'][-1]
        
        # 4. Extract high-res features
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
        orig_styles_vis = None
        adv_styles_vis = None
        if enable_vis:
            _, original_styles = self._prepare_style_adversary_inputs(
                img_batch, pixel_gt, model
            )
            orig_styles_vis = original_styles.detach().cpu()
            adv_styles_vis = adv_styles.detach().cpu()
        
        return AdversarialResult(
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
        if model.style_adv_use_gt_region_style:
            original_styles = extract_gt_region_style(img_batch.detach(), pixel_gt)
            # Ensure [B, K, 6] format (extract_gt_region_style returns [B, 6] for K=1)
            if original_styles.ndim == 2:
                # [B, 6] -> [B, 1, 6] for single object
                original_styles = original_styles.unsqueeze(1)
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
        normalized = (img_batch - base_means) / (base_stds + 1e-6)

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
                         f"edge_thresh={model.style_adv_gcn_edge_threshold}, dist_thresh={model.style_adv_gcn_distance_threshold}, "
                         f"use_bg={model.style_adv_enable_background and model.style_adv_gcn_use_background_edges}")
        
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
            edge_threshold=model.style_adv_gcn_edge_threshold,
            use_semantic=model.style_adv_gcn_use_semantic_edges,
            use_background=(
                model.style_adv_enable_background and model.style_adv_gcn_use_background_edges
            ),
            distance_threshold=model.style_adv_gcn_distance_threshold,
            use_boundary_distance=model.style_adv_gcn_use_boundary_distance,
            mask_features=mask_features_for_graph,  # Used to build semantic edges
            feature_sim_threshold=model.style_adv_gcn_feature_sim_threshold,
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
    
    def predict_params(
        self,
        clean_features: torch.Tensor,
        pixel_gt: torch.Tensor,
        model: nn.Module,
        img_batch: torch.Tensor | None = None,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Predict deformation offsets for all objects.
        
        Args:
            clean_features: [B, C, H_feat, W_feat] Clean backbone features
            pixel_gt: [B, K, H_img, W_img] Ground truth masks
            model: SAM2 model
            img_batch: Optional (not used for feature-level deform)
            
        Returns:
            params: Dict containing:
                - feature_offsets: [B, K, 2, H_feat, W_feat]
                - image_offsets: [B, K, 2, H_img, W_img]
                - valid_mask: [K] boolean mask of valid objects
        """
        B, C, H_feat, W_feat = clean_features.shape
        _, K, H_img, W_img = pixel_gt.shape
        device = clean_features.device
        
        # Resize masks to feature resolution
        pixel_gt_resized = F.interpolate(
            pixel_gt.float().flatten(0, 1).unsqueeze(1),
            size=(H_feat, W_feat),
            mode='nearest'
        ).view(B, K, H_feat, W_feat)
        
        # Identify valid objects
        mask_areas = pixel_gt_resized.sum(dim=(0, 2, 3))
        is_empty = (mask_areas == 0)
        
        # Background detection
        area_ratios = mask_areas / (B * H_feat * W_feat)
        is_background = torch.zeros(K, dtype=torch.bool, device=device)
        if K > 0 and area_ratios[-1] > 0.5:
            is_background[-1] = True
            
        valid_mask = ~(is_empty | is_background)
        valid_indices = torch.where(valid_mask)[0]
        
        # Initialize outputs
        feature_offsets_all = torch.zeros(B, K, 2, H_feat, W_feat, device=device)
        image_offsets_all = torch.zeros(B, K, 2, H_img, W_img, device=device)
        
        if len(valid_indices) == 0:
            return {
                "feature_offsets": feature_offsets_all,
                "image_offsets": image_offsets_all,
                "valid_mask": valid_mask
            }
            
        # Pre-compute image projection (keep gradients so backbone can adapt)
        img_proj = self.img_feat_proj(clean_features)
        
        # Process valid objects
        for k_idx in valid_indices.tolist():
            mask_k_original = pixel_gt[:, k_idx:k_idx + 1]
            mask_k_resized = pixel_gt_resized[:, k_idx:k_idx + 1]
            
            # Encode mask
            mask_features = self.mask_encoder(mask_k_original)
            
            # Fuse
            fused = img_proj + mask_features
            fused = self.fuser(fused)
            
            # Predict offsets
            feat_off, img_off = self.deform_module(fused, mask_k_resized)
            
            feature_offsets_all[:, k_idx] = feat_off
            image_offsets_all[:, k_idx] = img_off
            
            # Cleanup
            del mask_features, fused, feat_off, img_off
            
        return {
            "feature_offsets": feature_offsets_all,
            "image_offsets": image_offsets_all,
            "valid_mask": valid_mask
        }
    
    def apply_transform(
        self,
        img_batch: torch.Tensor | None,
        clean_features: torch.Tensor,
        params: dict[str, torch.Tensor],
        pixel_gt: torch.Tensor | None = None,
        model: nn.Module | None = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply deformation to features using predicted offsets.
        
        Args:
            clean_features: [B, C, H_feat, W_feat] Clean features
            params: Dict from predict_params
            
        Returns:
            deformed_features: [B, C, H_feat, W_feat]
        """
        feature_offsets = params["feature_offsets"]
        valid_mask = params["valid_mask"]
        
        B, K, _, H_feat, W_feat = feature_offsets.shape
        device = clean_features.device
        
        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return clean_features
            
        # Pre-compute base grid
        norm_grid = torch.meshgrid(
            torch.linspace(-1, 1, H_feat, device=device),
            torch.linspace(-1, 1, W_feat, device=device),
            indexing='ij'
        )
        norm_grid = torch.stack(norm_grid[::-1], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        deformed_list = []
        mask_list = []
        
        # If pixel_gt is provided, resize it for compositing
        pixel_gt_resized = None
        if pixel_gt is not None:
             pixel_gt_resized = F.interpolate(
                pixel_gt.float().flatten(0, 1).unsqueeze(1),
                size=(H_feat, W_feat),
                mode='nearest'
            ).view(B, K, H_feat, W_feat)
        
        for k in range(K):
            # If object is not valid, use clean features
            if not valid_mask[k]:
                deformed_list.append(clean_features)
                if pixel_gt_resized is not None:
                    mask_list.append(pixel_gt_resized[:, k:k+1])
                continue
                
            # Get offsets
            offset_k = feature_offsets[:, k] # [B, 2, H, W]
            
            # Normalize offsets
            offset_norm = offset_k.permute(0, 2, 3, 1).clone()
            offset_norm[..., 0] = offset_norm[..., 0] / (W_feat / 2.0)
            offset_norm[..., 1] = offset_norm[..., 1] / (H_feat / 2.0)
            
            sampling_grid = norm_grid + offset_norm
            
            # Warp (DO NOT detach backbone features if we want to train the backbone!)
            deformed_k = F.grid_sample(
                clean_features,
                sampling_grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=False
            )
            
            deformed_list.append(deformed_k)
            if pixel_gt_resized is not None:
                mask_list.append(pixel_gt_resized[:, k:k+1])
                
        # Composite
        if self.use_soft_composite and len(deformed_list) > 1:
            return self.compositor(deformed_list, mask_list)
        else:
            # Fallback or single object logic
            if len(deformed_list) > 0:
                return deformed_list[0]
            else:
                return clean_features

    def apply(
        self,
        img_batch: torch.Tensor = None,
        clean_features: torch.Tensor = None,
        clean_high_res: list[torch.Tensor] | None = None,
        pixel_gt: torch.Tensor = None,
        model: nn.Module = None,
    ) -> AdversarialResult:
        """
        Apply feature-level deformation (legacy interface).
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
            # Remove no_grad to allow backbone updates during adversarial training
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
            
        # 1. Predict
        params = self.predict_params(clean_features, pixel_gt, model)
        
        # 2. Apply
        deformed_features = self.apply_transform(
            img_batch, clean_features, params, pixel_gt, model
        )
        
        return AdversarialResult(
            features=deformed_features,
            high_res_features=clean_high_res,
            intermediate_images=img_batch,
            num_backbone_forwards=num_forwards,
            mode="feature_level",
            aug_type="deformation",
            deformation_offsets=params["image_offsets"]
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
        # Placed at OUTPUT to invert gradients for the offset network parameters
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
        # 1. Predict raw offsets (allow gradients to flow to backbone/encoder)
        # Range: (-1, 1) after tanh
        raw_offsets = self.offset_net(fused_features)  # [B, 2, H_feat, W_feat]
        
        # 3. Apply GRL to the offsets (OUTPUT side)
        # This ensures OffsetNet receives inverted gradients (Maximize Loss)
        raw_offsets_adv = self.grl(raw_offsets)
        
        raw_offsets_adv = torch.tanh(raw_offsets_adv)
        
        # Define scale factor (1024 / 64 = 16)
        scale_factor = 16.0
        
        # 1. Compute Image-level Offsets (Target Resolution)
        # Scale raw offsets by epsilon (defined in IMAGE pixels)
        # We first upsample the raw field to image size to get smooth dense flow
        image_raw_offsets = F.interpolate(
            raw_offsets_adv,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=False
        )
        image_offsets = image_raw_offsets * self.epsilon # [B, 2, H_img, W_img] in pixels
        
        # 2. Compute Feature-level Offsets (Source Resolution)
        # Feature offset = Image offset / scale_factor
        # We use the raw_offsets directly scaled by (epsilon / scale_factor)
        # This avoids downsampling artifacts and keeps consistency
        feature_offsets = raw_offsets_adv * (self.epsilon / scale_factor) # [B, 2, H_feat, W_feat] in feature pixels
        
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
        Features + Masks -> Mask-Aware Pooling -> Object Features -> Shared MLP -> Style Params -> GRL
    
    Args:
        feature_dim: Feature dimension (default: 256)
        epsilon: Max style perturbation magnitude (default: 2.0)
    """
    def __init__(
        self,
        feature_dim: int = 256,
        epsilon: float = 2.0,
        **kwargs
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        
        # Shared MLP: [B, K, C] -> [B, K, 6]
        self.object_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )
        
        # Initialize small residuals
        nn.init.normal_(self.object_mlp[-1].weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.object_mlp[-1].bias)
        
        self.grl = GRL(alpha=1.0)
    
    def forward(
        self,
        features: torch.Tensor,
        original_styles: torch.Tensor,
        pixel_gt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, H, W = features.shape
        K_actual = original_styles.shape[1]
        
        if pixel_gt is not None:
            if pixel_gt.shape[-2:] != (H, W):
                masks = F.interpolate(pixel_gt.float(), size=(H, W), mode='nearest')
            else:
                masks = pixel_gt.float()
            
            # Efficient Masked Pooling
            flat_features = features.flatten(2).transpose(1, 2) # [B, N, C]
            flat_masks = masks.flatten(2) # [B, K, N]
            
            mask_sums = flat_masks.sum(dim=2, keepdim=True).clamp(min=1e-6)
            flat_masks_norm = flat_masks / mask_sums
            
            object_features = torch.bmm(flat_masks_norm, flat_features) # [B, K, C]
            
        else:
            # Fallback: Global pooling
            global_feat = features.mean(dim=[2, 3]) # [B, C]
            object_features = global_feat.unsqueeze(1).expand(-1, K_actual, -1)
            
        style_residuals = self.object_mlp(object_features)
        
        # Apply GRL and constrain
        style_residuals_adv = self.grl(style_residuals)
        style_residuals_adv = torch.tanh(style_residuals_adv) * self.epsilon
        
        # Handle shape mismatch if pixel_gt K != original_styles K
        if style_residuals_adv.shape[1] != K_actual:
             if style_residuals_adv.shape[1] > K_actual:
                 style_residuals_adv = style_residuals_adv[:, :K_actual, :]
             else:
                 pad_k = K_actual - style_residuals_adv.shape[1]
                 style_residuals_adv = F.pad(style_residuals_adv, (0, 0, 0, pad_k))

        return original_styles + style_residuals_adv
