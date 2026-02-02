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
    """Pure gradient reversal without scaling.

    Note: alpha scaling was removed as it's redundant with learning rate.
    Control attack strength via LR scheduler and epsilon decay instead.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class GRL(nn.Module):
    """Gradient Reversal Layer - pure negation, no scaling."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return GradientReversalLayer.apply(x)


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

    def __init__(self, mode: str, aug_type: str, **kwargs):
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

    def predict_params(self, clean_features: torch.Tensor, pixel_gt: torch.Tensor, model: nn.Module, **kwargs):
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
        return self.impl.predict_params(clean_features=clean_features, pixel_gt=pixel_gt, model=model, **kwargs)

    def apply_transform(
        self,
        params: torch.Tensor | dict,
        img_batch: torch.Tensor | None = None,
        clean_features: torch.Tensor | None = None,
        pixel_gt: torch.Tensor | None = None,
        model: nn.Module | None = None,
        **kwargs,
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
        return self.impl.apply_transform(img_batch=img_batch, clean_features=clean_features, params=params, pixel_gt=pixel_gt, model=model, **kwargs)

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
        **kwargs,
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
        # If GCN is used, we disable internal GRL and apply it after GCN refinement
        # to ensure both networks receive reversed gradients (collaborative attack).
        self.style_net = StyleAdversarialNetwork(
            feature_dim=feature_dim,
            num_objects=num_objects,
            epsilon=epsilon,
            use_grl=not use_gcn,
        )

        # GCN for multi-object coordination (if enabled)
        # Note: The actual GCN module is created in SAM2Base._build_style_adv_components
        # and accessed via model.style_gcn in predict_params(). This class only stores
        # the flag; GCN refinement with GRL is implemented in predict_params().
        self.style_gcn = None  # Unused: actual GCN is model.style_gcn

    def predict_params(self, clean_features: torch.Tensor, pixel_gt: torch.Tensor, model: "SAM2Base", img_batch: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
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

        clean_features_detached = clean_features.detach()

        # 1. Prepare inputs (extract original styles)
        pixel_gt_normalized, original_styles = self._prepare_style_adversary_inputs(img_batch, pixel_gt, model)

        # 2. Predict adversarial styles using neural network + GRL
        # Detach clean_features to prevent gradient fighting.
        # The backbone should not receive gradients from the GRL that attempt to maximize loss;
        # it should only update based on the final loss minimization on the adversarial example.
        adv_styles = self.style_net(clean_features_detached, original_styles, pixel_gt=pixel_gt_normalized)

        # 3. Optional GCN refinement for multi-object coordination
        if self.use_gcn and model.style_gcn is not None:
            # Compute style delta
            style_delta = adv_styles - original_styles

            # Build object graph (img_batch needed for future semantic features)
            # CRITICAL: Detach clean_features to prevent GRL gradients flowing to backbone
            with torch.no_grad():
                edge_index, edge_weight, _ = self._build_object_graph(
                    pixel_gt_normalized,
                    img_batch,
                    clean_features_detached,
                    model,
                )

            # Extract mask features if GCN uses visual features
            mask_features = None
            if model.style_gcn.feature_dim > 0 and clean_features is not None:
                # Explicitly detach to prevent gradient flow from GCN back to backbone
                with torch.no_grad():
                    mask_features = self._extract_mask_features(
                        clean_features_detached,
                        pixel_gt_normalized,
                    )
                mask_features = mask_features.detach()

            # Refine delta using GCN
            if edge_index is not None:
                refined_delta = model.style_gcn(
                    style_delta,
                    edge_index,
                    edge_weight,
                    mask_features=mask_features,
                )
                # Clip refined delta to epsilon ball
                refined_delta = torch.clamp(refined_delta, -self.epsilon, self.epsilon)

                # Apply GRL to the refined delta (since it was skipped in style_net)
                # This ensures gradients flow: Loss -> GRL(neg) -> GCN -> delta -> style_net
                # Both GCN and style_net update to maximize loss.
                grl = GRL()
                refined_delta = grl(refined_delta)

                adv_styles = original_styles + refined_delta

        return adv_styles

    def apply_transform(self, img_batch: torch.Tensor, params: torch.Tensor, model: "SAM2Base", pixel_gt: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        Apply style transformation to images.

        Args:
            img_batch: [B, 3, H, W] Input images (must preserve gradients!)
            params: [B, K, 6] Adversarial style parameters (already passed through GRL)
            model: SAM2Base model (required for accessing config)
            pixel_gt: [B, K, H, W] Ground truth masks (optional, for region-based style)

        Returns:
            styled_images: [B, 3, H, W] Styled images with gradients preserved

        Note:
            - params have already been processed by GRL in predict_params
            - img_batch MUST keep gradients for backbone training
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
        adv_styles = self.predict_params(clean_features=clean_features, pixel_gt=pixel_gt, model=model, img_batch=img_batch)

        # 2. Apply styles to images
        styled_images = self.apply_transform(img_batch=img_batch, params=adv_styles, pixel_gt=pixel_gt, model=model)

        # 3. Forward through backbone
        # Detach to avoid expensive checkpointing - attacker gradients flow through GRL
        backbone_out = model.forward_image(styled_images.detach(), use_checkpoint=False)
        styled_features = backbone_out["backbone_fpn"][-1]

        # 4. Extract high-res features
        styled_high_res = None
        if model.use_high_res_features_in_sam:
            styled_high_res = [
                backbone_out["backbone_fpn"][0],
                backbone_out["backbone_fpn"][1],
            ]

        # Explicitly delete backbone_out to free graph references
        del backbone_out

        # 6. Save styles for visualization
        enable_vis = getattr(model, "_enable_style_visualization", False)
        orig_styles_vis = None
        adv_styles_vis = None
        if enable_vis:
            _, original_styles = self._prepare_style_adversary_inputs(img_batch, pixel_gt, model)
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
            pixel_gt: [B, K, H, W] normalized GT masks (possibly merged if single-object)
            original_styles: [B, K, 6] original style statistics (detached)
        """
        # Ensure 4D: [B, K, H, W]
        if pixel_gt.ndim == 3:
            pixel_gt = pixel_gt.unsqueeze(1)

        # Enforce single-object mode if requested (merge all masks)
        if not self.use_multi_object and pixel_gt.shape[1] > 1:
            pixel_gt = pixel_gt.sum(dim=1, keepdim=True).clamp(0, 1)  # [B, 1, H, W]
        else:
            # In multi-object mode, drop background channel when background attacks are disabled.
            # Heuristic: the last channel is background if it covers the majority of pixels.
            if not self.enable_background and pixel_gt.shape[1] > 1:
                mask_area_ratio = pixel_gt.float().mean(dim=(2, 3))  # [B, K]
                bg_is_last = (mask_area_ratio[:, -1] > 0.5).all()
                if bg_is_last:
                    pixel_gt = pixel_gt[:, :-1]  # remove background channel

        B, K, H, W = pixel_gt.shape

        # Extract all objects' styles (vectorized)
        # CRITICAL: Detach to break gradient flow from deform_augmenter
        # Style PGD should only optimize style parameters, not deform offsets
        if model.style_adv_use_gt_region_style:
            # extract_gt_region_style now always returns [B, K, 6]
            original_styles = extract_gt_region_style(img_batch.detach(), pixel_gt)
        else:
            # Global style: extract_style_statistics now returns [B, 1, 6]
            global_style = extract_style_statistics(img_batch.detach())
            original_styles = global_style.expand(-1, K, -1)  # [B, 1, 6] -> [B, K, 6]

        return pixel_gt, original_styles

    def _apply_style_to_images(
        self,
        img_batch: torch.Tensor,
        style_stats: torch.Tensor | None,
        gt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply style statistics to images using Mask-Aware AdaIN.

        CRITICAL FIX (AUE 24/25): normalization is now tied to the specific object region.
        - Previous implementation used Global Normalization, which caused mismatch when applying Local Style.
        - New implementation computes Source Stats for the masked region, ensuring consistent Identity mapping.

        Args:
            img_batch: [B, 3, H, W] normalized images
            style_stats: [B, K, 6] or [B, 6] style statistics per object
            gt_mask: [B, K_mask, H, W] GT masks (optional)

        Returns:
            styled_images: [B, 3, H, W] styled images
        """
        # If no style stats provided, return original images
        if style_stats is None:
            return img_batch

        B, C, H, W = img_batch.shape

        # Handle backward compatibility for [B, 6] input
        if style_stats.ndim == 2:
            style_stats = style_stats.unsqueeze(1)  # [B, 1, 6]

        K = style_stats.shape[1]

        # 1. Mask Alignment & Source Statistics Extraction
        source_stats = None

        if gt_mask is not None:
            # Ensure mask acts as a float binary mask
            if gt_mask.shape[2:] != (H, W):
                gt_mask = F.interpolate(gt_mask.float(), size=(H, W), mode="nearest")
            gt_mask = (gt_mask > 0.5).float()

            # Align mask channels with style stats (handle drop/merge from predict pipeline)
            if gt_mask.shape[1] != K:
                if K == 1:
                    # Single-object style applied to Multi-object mask -> Merge all objects
                    gt_mask = gt_mask.max(dim=1, keepdim=True)[0]
                elif K == gt_mask.shape[1] - 1:
                    # Background drop detected (style has 1 less channel) -> Drop last channel
                    gt_mask = gt_mask[:, :-1]
                else:
                    # Fallback for unexpected mismatch: wrap or slice safely
                    logging.warning(f"Style/Mask channel mismatch: Style={K}, Mask={gt_mask.shape[1]}. Slicing mask.")
                    gt_mask = gt_mask[:, :K]

            # Compute Source Stats using the aligned mask
            # CRITICAL: Compute on img_batch (with gradient) to support full differentiability
            # extract_gt_region_style handles broadcasting and safe division
            # We use min_pixels=100 to MATCH the default used in 'predict_params' (_prepare_style_adversary_inputs)
            # This ensures that fallback-to-global decisions are identical between prediction and application,
            # preventing Global-vs-Local mismatch for small objects.
            source_stats = extract_gt_region_style(img_batch, gt_mask, min_pixels=100)  # [B, K, 6]

        else:
            # Global Application (no mask provided)
            source_stats = extract_style_statistics(img_batch)  # [B, 1, 6]
            if K > 1:
                source_stats = source_stats.expand(-1, K, -1)
            # Create dummy full-image masks for composition loop
            gt_mask = torch.ones(B, K, H, W, device=img_batch.device)

        # 2. Iterate and Apply Style per Object
        styled_regions = []
        masks_list = []

        for k in range(K):
            target_style = style_stats[:, k]  # [B, 6]
            source_style = source_stats[:, k]  # [B, 6]
            mask_k = gt_mask[:, k : k + 1]  # [B, 1, H, W]

            # Extract Means and Stds
            src_mean = source_style[:, :3].view(B, 3, 1, 1)
            src_std = source_style[:, 3:].view(B, 3, 1, 1)
            tgt_mean = target_style[:, :3].view(B, 3, 1, 1)
            tgt_std = target_style[:, 3:].view(B, 3, 1, 1)

            # Mask-Aware AdaIN: (x - mu_src) / sigma_src * sigma_tgt + mu_tgt
            # Using specific source stats ensures that if Tgt ~= Src, result ~= Input (Identity)
            # We add 1e-6 to std for numerical stability
            normalized = (img_batch - src_mean) / (src_std + 1e-6)
            object_styled = normalized * tgt_std + tgt_mean

            # Clamp output to safe image range
            object_styled = object_styled.clamp(min=-3.0, max=3.0)

            styled_regions.append(object_styled)
            masks_list.append(mask_k)

        # 3. Composition
        # Compose objects onto the original image
        # Using simple overwriting (Painter's Algorithm) based on the channel order
        styled_images = img_batch.clone()

        for k in range(K):
            m = masks_list[k]
            # Hard composition masked by region
            styled_images = m * styled_regions[k] + (1 - m) * styled_images

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
            masks_resized = F.interpolate(masks, size=(fH, fW), mode="bilinear", align_corners=False)
        else:
            masks_resized = masks

        # Binarize masks
        masks_binary = (masks_resized > 0.5).float()  # [B, K, fH, fW]

        # Compute masked average pooling for each mask
        mask_features = []
        for k in range(K):
            mask_k = masks_binary[:, k : k + 1, :, :]  # [B, 1, fH, fW]
            mask_area = mask_k.sum(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
            mask_area = mask_area + 1e-6

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
        if backbone_features is not None and backbone_features.requires_grad:
            backbone_features = backbone_features.detach()

        if pixel_gt is not None and logging.getLogger().isEnabledFor(logging.DEBUG):
            # DEBUG: Log detailed pixel_gt info (avoid any work unless debug is enabled)
            per_channel_sum = [pixel_gt[:, k].sum().item() for k in range(min(pixel_gt.shape[1], 5))]
            per_channel_nonzero = [(pixel_gt[:, k] > 0.5).sum().item() for k in range(min(pixel_gt.shape[1], 5))]
            logging.debug(
                f"DEBUG _build_style_graph: pixel_gt.shape={pixel_gt.shape}, "
                f"dtype={pixel_gt.dtype}, device={pixel_gt.device}, "
                f"min={pixel_gt.min():.3f}, max={pixel_gt.max():.3f}, "
                f"per_channel_sum[0:5]={per_channel_sum}, "
                f"per_channel_nonzero[0:5]={per_channel_nonzero}"
            )

            # Debug: Check input masks
            mask_areas = (pixel_gt > 0.5).float().sum(dim=(2, 3))  # [B, K]
            valid_masks = (mask_areas > 0).sum().item()
            logging.debug(
                f"GCN input: pixel_gt.shape={pixel_gt.shape}, valid_masks={valid_masks}/{pixel_gt.shape[0] * pixel_gt.shape[1]}, "
                f"edge_thresh={model.style_adv_gcn_edge_threshold}, dist_thresh={model.style_adv_gcn_distance_threshold}, "
                f"use_bg={model.adv_enable_background and model.style_adv_gcn_use_background_edges}"
            )

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
            use_background=(model.adv_enable_background and model.style_adv_gcn_use_background_edges),
            distance_threshold=model.style_adv_gcn_distance_threshold,
            use_boundary_distance=model.style_adv_gcn_use_boundary_distance,
            mask_features=mask_features_for_graph,  # Used to build semantic edges
            feature_sim_threshold=model.style_adv_gcn_feature_sim_threshold,
        )

        # Add self-loops to the graph
        num_nodes_total = pixel_gt.shape[0] * pixel_gt.shape[1]  # B * K
        edge_index, edge_weight = model.style_gcn._add_self_loops(edge_index, edge_weight, num_nodes_total)

        model._latest_gcn_stats = stats if stats else None
        if stats and stats.get("graphs", 0) == 0:
            # Check if pixel_gt has any content
            if pixel_gt is not None:
                valid_pixels = (pixel_gt > 0.5).sum().item()
                logging.warning(
                    f"GCN graph built but NO edges: graphs={stats['graphs']}, nodes_fg={stats['nodes_foreground']}, "
                    f"nodes_bg={stats['nodes_background']}, edges_iou={stats['edges_iou']}, edges_dist={stats['edges_distance']}, "
                    f"edges_bg={stats['edges_background']}, edges_semantic={stats.get('edges_semantic', 0)}, valid_pixels={valid_pixels}"
                )
            else:
                logging.warning("GCN graph built but NO edges (pixel_gt is None)")
        elif stats:
            # Show edge type breakdown for non-empty graphs
            logging.debug(
                f"GCN graph built: {stats['graphs']:.0f} graphs, {stats['edges_total']:.0f} edges "
                f"(IoU:{stats['edges_iou']:.0f}, Dist:{stats['edges_distance']:.0f}, Semantic:{stats.get('edges_semantic', 0):.0f}, BG:{stats['edges_background']:.0f}), "
                f"nodes: {stats['nodes_foreground']:.0f}fg+{stats['nodes_background']:.0f}bg, "
                f"avg_degree: {stats['avg_degree']:.2f}"
            )
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
        raise NotImplementedError("Feature-level style augmentation is not implemented yet. Use mode='image_level' for style augmentation.")

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
        freeze_encoder_components: Freeze all encoder components (mask_encoder, img_feat_proj, fuser)
        image_size: Target image resolution (default: 1024)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        epsilon: float = 0.15,
        use_soft_composite: bool = True,
        temperature: float = 1.0,
        use_multi_object: bool = False,
        use_gcn: bool = False,
        gcn_num_layers: int = 2,
        num_deform_groups: int = 4,
        init_from_memory_encoder: bool = True,
        freeze_encoder_components: bool = False,
        image_size: int = 1024,
        zero_mean_offsets: bool = False,
        local_offset_gain: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.use_soft_composite = use_soft_composite
        self.temperature = temperature
        self.use_multi_object = use_multi_object
        self.use_gcn = use_gcn
        self.init_from_memory_encoder = init_from_memory_encoder
        self.freeze_encoder_components = freeze_encoder_components

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
        self.fuser = Fuser(layer=CXBlock(dim=feature_dim, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True), num_layers=2)

        # 4. Deformation module (uses fused features)
        # Note: Produces both feature-level deformation and image-level offsets
        self.deform_module = FeatureBasedDeformModule(
            feature_dim=feature_dim,
            epsilon=epsilon,
            image_size=image_size,
            zero_mean_offsets=zero_mean_offsets,
            local_offset_gain=local_offset_gain,
            use_grl=not use_gcn,  # Disable internal GRL if GCN is used (to apply it after GCN)
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
        self.mask_encoder.load_state_dict(memory_encoder.mask_downsampler.state_dict())

        # Copy image feature projection weights
        self.img_feat_proj.load_state_dict(memory_encoder.pix_feat_proj.state_dict())

        # Copy fuser weights
        self.fuser.load_state_dict(memory_encoder.fuser.state_dict())

        logging.info("✓ Initialized deformation network from memory encoder weights")

        # Optionally freeze encoder components (mask_encoder, img_feat_proj, fuser)
        # to prevent GRL gradients from destabilizing pretrained representations
        if self.freeze_encoder_components:
            for param in self.mask_encoder.parameters():
                param.requires_grad = False
            for param in self.img_feat_proj.parameters():
                param.requires_grad = False
            for param in self.fuser.parameters():
                param.requires_grad = False
            logging.info("✓ Frozen all encoder components (mask_encoder, img_feat_proj, fuser)")

    def predict_params(self, clean_features: torch.Tensor, pixel_gt: torch.Tensor, model: nn.Module, img_batch: torch.Tensor | None = None, **kwargs) -> dict[str, torch.Tensor]:
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

        # CRITICAL: Binarize masks to prevent numerical instability from soft mask values
        pixel_gt_binary = (pixel_gt > 0.5).float()

        # Resize masks to feature resolution
        pixel_gt_resized = F.interpolate(pixel_gt_binary.flatten(0, 1).unsqueeze(1), size=(H_feat, W_feat), mode="nearest").view(B, K, H_feat, W_feat)

        # Identify valid objects
        mask_areas = pixel_gt_resized.sum(dim=(0, 2, 3))
        is_empty = mask_areas == 0

        # Background detection (only if background channel is enabled)
        area_ratios = mask_areas / (B * H_feat * W_feat)
        is_background = torch.zeros(K, dtype=torch.bool, device=device)
        if self.use_multi_object and getattr(model, "adv_enable_background", False):
            if K > 0 and area_ratios[-1] > 0.5:
                is_background[-1] = True

        valid_mask = ~(is_empty | is_background)
        valid_indices = torch.where(valid_mask)[0]

        # Initialize outputs
        feature_offsets_all = torch.zeros(B, K, 2, H_feat, W_feat, device=device)
        image_offsets_all = torch.zeros(B, K, 2, H_img, W_img, device=device)

        if len(valid_indices) == 0:
            return {"feature_offsets": feature_offsets_all, "image_offsets": image_offsets_all, "valid_mask": valid_mask}

        # Pre-compute image projection
        # Detach features to prevent gradient fighting:
        # The backbone should not receive gradients from the GRL (via offset_net)
        # that attempt to maximize loss. Backbone should only adapt to the *result*
        # of the deformation (task loss), not try to fool the offset predictor.
        clean_features_detached = clean_features.detach()

        img_proj = self.img_feat_proj(clean_features_detached)

        if img_proj.abs().max() > 1e3:
            import logging

            logging.warning(f"FeatureLevelDeformationImpl: img_proj has large values: {img_proj.abs().max().item():.4f}")

        # Process valid objects
        for k_idx in valid_indices.tolist():
            mask_k_original = pixel_gt_binary[:, k_idx : k_idx + 1]
            mask_k_resized = pixel_gt_resized[:, k_idx : k_idx + 1]

            # Encode mask
            mask_emb = self.mask_encoder(mask_k_original)

            # Fuse
            fused = self.fuser(img_proj + mask_emb)

            if fused.abs().max() > 1e3:
                import logging

                logging.warning(f"FeatureLevelDeformationImpl: fused features has large values: {fused.abs().max().item():.4f}")

            # Predict offsets
            feat_off, img_off = self.deform_module(fused, mask_k_resized)

            # Apply manual GRL if internal GRL was disabled (e.g. for GCN coordination)
            if self.use_gcn:
                grl = GRL()
                feat_off = grl(feat_off)
                img_off = grl(img_off)

            feature_offsets_all[:, k_idx] = feat_off
            image_offsets_all[:, k_idx] = img_off

        return {"feature_offsets": feature_offsets_all, "image_offsets": image_offsets_all, "valid_mask": valid_mask}

    def apply_transform(
        self, img_batch: torch.Tensor | None, clean_features: torch.Tensor, params: dict[str, torch.Tensor], pixel_gt: torch.Tensor | None = None, model: nn.Module | None = None, **kwargs
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
        norm_grid = torch.meshgrid(torch.linspace(-1, 1, H_feat, device=device), torch.linspace(-1, 1, W_feat, device=device), indexing="ij")
        norm_grid = torch.stack(norm_grid[::-1], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        deformed_list = []
        mask_list = []

        # If pixel_gt is provided, resize it for compositing
        pixel_gt_resized = None
        if pixel_gt is not None:
            # CRITICAL: Binarize masks before resize
            pixel_gt_binary = (pixel_gt > 0.5).float()
            pixel_gt_resized = F.interpolate(pixel_gt_binary.flatten(0, 1).unsqueeze(1), size=(H_feat, W_feat), mode="nearest").view(B, K, H_feat, W_feat)

        for k in range(K):
            # If object is not valid, use clean features
            if not valid_mask[k]:
                deformed_list.append(clean_features)
                if pixel_gt_resized is not None:
                    mask_list.append(pixel_gt_resized[:, k : k + 1])
                continue

            # Get offsets
            offset_k = feature_offsets[:, k]  # [B, 2, H, W]

            # Normalize offsets
            offset_norm = offset_k.permute(0, 2, 3, 1).clone()
            offset_norm[..., 0] = offset_norm[..., 0] / (W_feat / 2.0)
            offset_norm[..., 1] = offset_norm[..., 1] / (H_feat / 2.0)

            sampling_grid = norm_grid + offset_norm

            # Warp: CRITICAL - DO NOT detach clean_features here!
            # Offsets have already passed through GRL in predict_params,
            # so clean_features needs gradients for backbone training on adversarial features.
            deformed_k = F.grid_sample(clean_features, sampling_grid, mode="bilinear", padding_mode="border", align_corners=False)

            deformed_list.append(deformed_k)
            if pixel_gt_resized is not None:
                mask_list.append(pixel_gt_resized[:, k : k + 1])

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
                raise ValueError("Must provide either clean_features (Mode 1) or model (Mode 2)")

            logging.debug("Deformation Mode 2: Encoding image on-demand")
            # Remove no_grad to allow backbone updates during adversarial training
            backbone_out = model.forward_image(img_batch, use_checkpoint=True)
            clean_features = backbone_out["backbone_fpn"][-1]

            if model.use_high_res_features_in_sam and clean_high_res is None:
                clean_high_res = [backbone_out["backbone_fpn"][0], backbone_out["backbone_fpn"][1]]
            num_forwards = 1
        else:
            # Mode 1: Feature reuse (efficient)
            logging.debug("Deformation Mode 1: Reusing provided features")
            num_forwards = 0

        # 1. Predict
        params = self.predict_params(clean_features, pixel_gt, model)

        # 2. Apply
        deformed_features = self.apply_transform(img_batch, clean_features, params, pixel_gt, model)

        return AdversarialResult(
            features=deformed_features,
            high_res_features=clean_high_res,
            intermediate_images=img_batch,
            num_backbone_forwards=num_forwards,
            mode="feature_level",
            aug_type="deformation",
            deformation_offsets=params["image_offsets"],
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
        zero_mean_offsets: bool = False,
        local_offset_gain: float = 1.0,
        use_grl: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.image_size = image_size
        self.zero_mean_offsets = zero_mean_offsets
        self.local_offset_gain = local_offset_gain
        self.use_grl = use_grl

        # Offset predictor: fused_features -> dense offsets (2 channels)
        self.offset_net = nn.Sequential(
            nn.InstanceNorm2d(feature_dim),  # Normalize input (critical for stability)
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),  # 2 channels for (dx, dy)
        )

        # Initialize ALL layers with small weights for near-zero initial output
        # This prevents overly strong attacks at the start of training
        for m in self.offset_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        logging.info(f"FeatureBasedDeformModule: Initialized all conv layers with std=0.01 for near-zero initial output")

        # Gradient Reversal Layer - pure negation, no scaling
        # Note: Attack strength is controlled via LR and epsilon, not GRL alpha
        self.grl = GRL()

        # Track if we've logged initial output (for debugging)
        self._logged_initial_output = False

    def forward(self, fused_features: torch.Tensor, object_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        # Log initial output magnitude once (verify near-zero initialization)
        if not self._logged_initial_output:
            logging.info(
                f"DeformModule initial output: mean={raw_offsets.mean().item():.6f}, std={raw_offsets.std().item():.6f}, min={raw_offsets.min().item():.6f}, max={raw_offsets.max().item():.6f}"
            )
            self._logged_initial_output = True

        # 3. Apply GRL to the offsets (OUTPUT side)
        # This ensures OffsetNet receives inverted gradients (Maximize Loss)
        if self.use_grl:
            raw_offsets_adv = self.grl(raw_offsets)
        else:
            raw_offsets_adv = raw_offsets

        # === RELATIVE ENCODING (Jan 16 Evening) ===
        # Instead of tanh + clamp, use sigmoid for naturally bounded output.
        # Sigmoid outputs [0, 1], we shift to [-0.5, 0.5] for symmetric offsets.
        # This is smoother than tanh near the boundaries and avoids saturation.
        offset_ratio = torch.sigmoid(raw_offsets_adv) - 0.5  # [-0.5, 0.5]

        # Remove global shift while keeping local deformation energy
        if self.zero_mean_offsets:
            if object_mask is not None:
                # SAFETY: Ensure mask is binary to prevent numerical instability
                object_mask_binary = (object_mask > 0.5).float()
                mask_sum = object_mask_binary.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
                mean_offset = (offset_ratio * object_mask_binary).sum(dim=(2, 3), keepdim=True) / mask_sum
            else:
                mean_offset = offset_ratio.mean(dim=(2, 3), keepdim=True)
            offset_ratio = offset_ratio - mean_offset

        # After zero-mean, the range is still bounded (approx [-0.5, 0.5] in practice)
        # No need for local_offset_gain or additional clamp with this design

        # Use actual target resolution instead of a fixed scale factor
        if isinstance(self.image_size, (tuple, list)):
            target_h, target_w = self.image_size
        else:
            target_h = target_w = int(self.image_size)

        scale_y = target_h / float(fused_features.shape[2])
        scale_x = target_w / float(fused_features.shape[3])

        # 1. Compute Image-level Offsets (Target Resolution)
        # offset_ratio is in [-0.5, 0.5], multiply by 2*epsilon to get [-epsilon, +epsilon]
        # Use float32 here for stability under AMP.
        offset_ratio_f32 = offset_ratio.to(dtype=torch.float32)
        image_raw_offsets = F.interpolate(
            offset_ratio_f32,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        # Scale: offset_ratio * 2 * epsilon * scale_factor
        # This gives max pixel shift of ±epsilon * scale_factor
        scale_tensor = torch.tensor([scale_x, scale_y], device=fused_features.device, dtype=torch.float32).view(1, 2, 1, 1)
        image_offsets_f32 = image_raw_offsets * (2.0 * float(self.epsilon) * scale_tensor)

        image_offsets = image_offsets_f32.to(dtype=fused_features.dtype)

        # 2. Feature-level Offsets (Source Resolution)
        # Keep offsets in feature pixel units for direct warping on the feature map
        feature_offsets = (offset_ratio_f32 * 2.0 * float(self.epsilon)).to(dtype=fused_features.dtype)

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

    def forward(self, deformed_features_list: list[torch.Tensor], mask_list: list[torch.Tensor]) -> torch.Tensor:
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
        use_grl: Whether to apply Gradient Reversal Layer (default: True)
    """

    def __init__(self, feature_dim: int = 256, epsilon: float = 2.0, use_grl: bool = True, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.use_grl = use_grl

        # Shared MLP: [B, K, C] -> [B, K, 6]
        # LayerNorm normalizes across feature dimension (C), ensuring consistent
        # style residual magnitudes regardless of object size or feature statistics
        self.object_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )

        # Initialize ALL layers with small weights for near-zero initial output
        # This prevents overly strong attacks at the start of training
        # Previously only the last layer was initialized, causing strong initial attacks
        for m in self.object_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        logging.info(f"StyleAdversarialNetwork: Initialized all linear layers with std=0.01 for near-zero initial output")

        self.grl = GRL()

        # Track if we've logged initial output (for debugging)
        self._logged_initial_output = False

    def forward(
        self,
        features: torch.Tensor,
        original_styles: torch.Tensor,
        pixel_gt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, H, W = features.shape
        K_actual = original_styles.shape[1]

        if pixel_gt is not None:
            # CRITICAL: Binarize masks to prevent numerical instability from soft values
            pixel_gt_binary = (pixel_gt > 0.5).float()
            if pixel_gt_binary.shape[-2:] != (H, W):
                masks = F.interpolate(pixel_gt_binary, size=(H, W), mode="nearest")
            else:
                masks = pixel_gt_binary

            # Efficient Masked Pooling
            flat_features = features.flatten(2).transpose(1, 2)  # [B, N, C]
            flat_masks = masks.flatten(2)  # [B, K, N]

            # Use larger clamp value to prevent division instability
            mask_sums = flat_masks.sum(dim=2, keepdim=True).clamp(min=1.0)
            flat_masks_norm = flat_masks / mask_sums

            object_features = torch.bmm(flat_masks_norm, flat_features)  # [B, K, C]

        else:
            # Fallback: Global pooling
            global_feat = features.mean(dim=[2, 3])  # [B, C]
            object_features = global_feat.unsqueeze(1).expand(-1, K_actual, -1)

        style_residuals = self.object_mlp(object_features)

        # Log initial output magnitude once (verify near-zero initialization)
        if not self._logged_initial_output:
            logging.info(
                f"StyleNetwork initial output: mean={style_residuals.mean().item():.6f}, "
                f"std={style_residuals.std().item():.6f}, "
                f"min={style_residuals.min().item():.6f}, max={style_residuals.max().item():.6f}"
            )
            self._logged_initial_output = True

        # Apply GRL if enabled
        if self.use_grl:
            style_residuals_adv = self.grl(style_residuals)
        else:
            style_residuals_adv = style_residuals

        # Handle shape mismatch if pixel_gt K != original_styles K
        if style_residuals_adv.shape[1] != K_actual:
            if style_residuals_adv.shape[1] > K_actual:
                style_residuals_adv = style_residuals_adv[:, :K_actual, :]
            else:
                pad_k = K_actual - style_residuals_adv.shape[1]
                style_residuals_adv = F.pad(style_residuals_adv, (0, 0, 0, pad_k))

        # === RELATIVE ENCODING (Jan 16 Evening, Fixed Jan 18) ===
        # All transformations are now controlled by epsilon for proper regularization.
        # epsilon=0.0001 → minimal perturbation; epsilon=0.1 → stronger perturbation
        #
        # Split the 6-dim output into scale (first 3) and shift (last 3)
        raw_scale = style_residuals_adv[:, :, :3]  # For multiplicative factor
        raw_shift = style_residuals_adv[:, :, 3:]  # For additive factor

        # === FIX: Scale factor now controlled by epsilon ===
        # epsilon=0.0001 → scale_range=0.02 → [0.98, 1.02] (±2% brightness)
        # epsilon=0.01   → scale_range=0.20 → [0.80, 1.20] (±20% brightness)
        # epsilon=0.1    → scale_range=0.20 → [0.80, 1.20] (capped at ±20%)
        scale_range = min(0.2, self.epsilon * 200)
        scale_factor = 1.0 - scale_range + 2 * scale_range * torch.sigmoid(raw_scale)

        # Shift factor: tanh outputs [-1, 1], scale by epsilon for small shifts
        # Max shift is ±epsilon (e.g., ±0.5 for epsilon=0.5)
        shift_factor = self.epsilon * torch.tanh(raw_shift)  # [-epsilon, +epsilon]

        # Apply relative transformation to original styles
        original_means = original_styles[:, :, :3]  # [B, K, 3]
        original_stds = original_styles[:, :, 3:]  # [B, K, 3]

        # Means: scale + shift (both now bounded by epsilon)
        adv_means = original_means * scale_factor + shift_factor

        # === FIX: Std scale now controlled by epsilon ===
        # epsilon=0.0001 → std_range=0.05 → [0.95, 1.05] (±5% contrast)
        # epsilon=0.01   → std_range=0.50 → [0.50, 1.50] (±50% contrast)
        # epsilon=0.1    → std_range=0.50 → [0.50, 1.50] (capped at ±50%)
        std_range = min(0.5, self.epsilon * 500)
        std_scale = 1.0 - std_range + 2 * std_range * torch.sigmoid(raw_scale)
        adv_stds = original_stds * std_scale
        # Safety: ensure stds stay positive
        adv_stds = adv_stds.clamp(min=0.1)

        adv_styles_out = torch.cat([adv_means, adv_stds], dim=2)

        return adv_styles_out
