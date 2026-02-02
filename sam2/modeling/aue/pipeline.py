# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
AUE Adversarial Attack Pipeline.

Manages the execution of adversarial attacks (style, deformation) for AUE training.
Implements the cooperative attack strategy: parallel predict, sequential apply.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from sam2.modeling.sam2_base import SAM2Base


class AdversarialPipeline:
    """
    Manages adversarial attack execution for AUE training.

    Implements the cooperative attack strategy:
    1. Predict parameters for ALL attacks using CLEAN features (parallel)
    2. Apply transformations sequentially using predicted parameters

    This ensures stable gradient flow for Min-Max optimization.

    Args:
        model: Reference to SAM2Base model (for forward_image, attackers, etc.)
    """

    def __init__(self, model: "SAM2Base"):
        self._model = model

    def apply_attack_pipeline(
        self,
        img_batch: torch.Tensor,
        backbone_features: torch.Tensor,
        high_res_features: list[torch.Tensor],
        pixel_gt: torch.Tensor,
        enable_vis: bool = False,
        uq_sample_num: int = 8,
    ) -> dict[str, Any]:
        """
        Apply adversarial attack pipeline using cooperative mode.

        Strategy:
        1. Predict parameters for all active attackers using CLEAN features (parallel).
        2. Apply transformations sequentially using the predicted parameters.

        Args:
            img_batch: [B, 3, H, W] input images
            backbone_features: [B, C, H, W] backbone features (stride 16)
            high_res_features: List of high-res features [stride 4, stride 8]
            pixel_gt: [B, K, H, W] ground truth masks
            enable_vis: Whether to collect visualization data
            uq_sample_num: Number of samples for uncertainty estimation

        Returns:
            Dict containing:
                - calibration_loss_adversarial: Adversarial calibration loss
                - aug_metrics: Metrics for adversarial branch
                - adv_uncertainty: Adversarial uncertainty map
                - vis_refs: Visualization references (if enable_vis)
        """
        model = self._model
        # device and dtype used for zero tensor initialization if needed
        _ = img_batch.device  # Verify tensor is on correct device

        vis_refs = {}
        if enable_vis:
            vis_refs["img_batch"] = img_batch.detach().cpu()
            vis_refs["pixel_gt"] = pixel_gt.detach().cpu()

        # Initialize state for cooperative attack
        state = {
            "img": img_batch,
            "features": backbone_features,
            "high_res": high_res_features,
            "pixel_gt": pixel_gt,
        }

        # Apply cooperative attack (parallel predict, sequential apply)
        state = self._apply_cooperative_attack(state, enable_vis, vis_refs)

        # Record attack order for visualization
        if enable_vis:
            vis_refs["attack_order"] = list(model.adversarial_attack_order)

        # Compute adversarial calibration loss using augmented state
        calibration_loss_adversarial, aug_metrics, adv_uncertainty = self._compute_augmented_calibration_loss(
            state=state,
            uq_sample_num=uq_sample_num,
        )

        return {
            "calibration_loss_adversarial": calibration_loss_adversarial,
            "aug_metrics": aug_metrics,
            "adv_uncertainty": adv_uncertainty,
            "vis_refs": vis_refs,
        }

    def generate_adversarial_samples(
        self,
        img_batch: torch.Tensor,
        backbone_features: torch.Tensor,
        high_res_features: list[torch.Tensor],
        pixel_gt: torch.Tensor,
        single_obj_gt: torch.Tensor | None = None,
        enable_vis: bool = False,
    ) -> dict[str, Any]:
        """
        Generate adversarial samples without forward pass or loss computation.

        This is a lightweight method that only applies adversarial attacks and returns
        the transformed inputs. The calling code is responsible for:
        1. Running forward pass on adversarial samples
        2. Running iterative refinement (reusing existing code)
        3. Computing task loss and calibration loss

        Args:
            img_batch: [B, 3, H, W] input images
            backbone_features: [B, C, H, W] backbone features (used for attack prediction)
            high_res_features: List of high-res features [stride 4, stride 8]
            pixel_gt: [B, K, H, W] ground truth masks (all objects, for attack generation)
            single_obj_gt: [B, 1, H, W] single object GT (same as clean branch, for SAM task)
            enable_vis: Whether to collect visualization data

        Returns:
            Dict containing:
                - adv_img: Adversarially transformed images
                - adv_features: New backbone features from transformed images
                - adv_high_res: New high-res features from transformed images
                - adv_pixel_gt: Possibly warped GT masks (multi-object, for attack)
                - adv_single_obj_gt: Warped single object GT (for SAM task, matches clean branch)
                - vis_refs: Visualization references (if enable_vis)
        """
        model = self._model

        vis_refs = {}
        if enable_vis:
            vis_refs["img_batch"] = img_batch.detach().cpu()
            vis_refs["pixel_gt"] = pixel_gt.detach().cpu()

        # Initialize state for cooperative attack
        state = {
            "img": img_batch,
            "features": backbone_features,
            "high_res": high_res_features,
            "pixel_gt": pixel_gt,
            "single_obj_gt": single_obj_gt,  # Track single object GT separately
        }

        # Apply cooperative attack (parallel predict, sequential apply)
        state = self._apply_cooperative_attack(state, enable_vis, vis_refs)

        # Record attack order for visualization
        if enable_vis:
            vis_refs["attack_order"] = list(model.adversarial_attack_order)
            vis_refs["adv_img"] = state["img"].detach().cpu()

        return {
            "adv_img": state["img"],
            "adv_features": state["features"],
            "adv_high_res": state.get("high_res"),
            "adv_pixel_gt": state["pixel_gt"],
            "adv_single_obj_gt": state.get("single_obj_gt"),  # Warped single object GT
            "adv_images_for_attacker": state.get("adv_images_for_attacker"),  # For attacker loss
            "deform_offsets": state.get("deform_offsets"),  # For prompt coordinate transformation
            "vis_refs": vis_refs,
        }

    def _apply_cooperative_attack(
        self,
        state: dict,
        enable_vis: bool,
        vis_refs: dict,
    ) -> dict:
        """
        Apply cooperative adversarial attack.

        Strategy:
        1. Predict parameters for all active attackers using CLEAN features.
        2. Apply all image transformations sequentially (NO intermediate backbone forward).
        3. Do a SINGLE backbone forward at the end on the final transformed image.

        Optimization: Removed redundant backbone forwards after each attack.
        Previous: 3x forward (clean + style + deform)
        Current: 2x forward (clean + final transformed)
        """
        model = self._model
        aug_params = {}
        clean_features = state["features"].detach()
        pixel_gt = state["pixel_gt"]
        img_batch = state["img"]

        # === Phase 1: Predict (Parallel) ===
        # All predictions use CLEAN features (cooperative attack strategy)
        for aug_name in model.adversarial_attack_order:
            attacker = getattr(model, f"{aug_name}_attacker", None)
            if attacker is not None:
                params = attacker.predict_params(
                    clean_features=clean_features,
                    pixel_gt=pixel_gt,
                    model=model,
                    img_batch=img_batch,
                )
                aug_params[aug_name] = params

        # === Phase 2: Apply Image Transformations (NO backbone forward here) ===
        any_transform_applied = False
        for aug_name in model.adversarial_attack_order:
            if aug_name not in aug_params:
                continue

            attacker = getattr(model, f"{aug_name}_attacker")
            params = aug_params[aug_name]

            if attacker.mode == "image_level":
                # Image level attack (e.g. Style)
                # Save original styles BEFORE applying transform (for visualization)
                if enable_vis and attacker.aug_type == "style":
                    from sam2.modeling.style_utils import extract_gt_region_style

                    # extract_gt_region_style now always returns [B, K, 6]
                    orig_styles = extract_gt_region_style(state["img"], state["pixel_gt"])
                    vis_refs["original_styles"] = orig_styles.detach().cpu()

                styled_images = attacker.apply_transform(
                    img_batch=state["img"],
                    params=params,
                    pixel_gt=state["pixel_gt"],
                    model=model,
                )
                state["img"] = styled_images
                any_transform_applied = True

                # NOTE: Removed backbone forward here - will do once at the end

                if enable_vis:
                    vis_refs["styled_images"] = styled_images.detach().cpu()
                    if attacker.aug_type == "style":
                        vis_refs["adversarial_styles"] = params.detach().cpu()

            elif attacker.mode == "feature_level":
                # Feature level attack (e.g. Deform)
                offsets = params["image_offsets"] if isinstance(params, dict) else params

                warped_img, warped_gt, warped_single_gt = self._apply_deformation_to_images(
                    state["img"],
                    state["pixel_gt"],
                    offsets,
                    single_obj_gt=state.get("single_obj_gt"),
                    enable_vis=enable_vis,
                    vis_refs=vis_refs if enable_vis else {},
                )
                state["img"] = warped_img
                state["pixel_gt"] = warped_gt
                if warped_single_gt is not None:
                    state["single_obj_gt"] = warped_single_gt
                # Always save offsets for prompt coordinate transformation
                state["deform_offsets"] = offsets
                any_transform_applied = True

                # NOTE: Removed backbone forward here - will do once at the end

                if enable_vis:
                    vis_refs["deform_offsets"] = offsets.detach().cpu()
                    vis_refs.setdefault("warped_images", warped_img.detach().cpu())
                    vis_refs.setdefault("warped_pixel_gt", warped_gt.detach().cpu())

        # === Phase 3: Single Backbone Forward on Final Transformed Image ===
        # GRADIENT FLOW DESIGN:
        # - adv_images_for_attacker is saved BEFORE any processing
        # - Backbone forward WITHOUT detach to preserve gradient path for attacker
        # - Since backbone is frozen (freeze_image_encoder_epochs), no gradient conflict
        # - Gradient path: -attacker_loss → pred_masks → features → adv_img → params → Attacker
        if any_transform_applied:
            # Save adversarial images (with gradient to attacker params via style/deform transforms)
            state["adv_images_for_attacker"] = state["img"]
            # Forward through backbone (keep gradient for attacker, use checkpoint for memory)
            backbone_out = model.forward_image(state["img"], use_checkpoint=True)
            state["features"] = backbone_out["backbone_fpn"][-1]
            if model.use_high_res_features_in_sam:
                state["high_res"] = [backbone_out["backbone_fpn"][0], backbone_out["backbone_fpn"][1]]

        return state

    def _apply_deformation_to_images(
        self,
        img_batch: torch.Tensor,
        pixel_gt: torch.Tensor,
        deform_offsets: torch.Tensor | None,
        single_obj_gt: torch.Tensor | None = None,
        enable_vis: bool = False,
        vis_refs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply deformation offsets to images and GT masks (batched vectorized).

        Optimization: Uses batched grid_sample to warp all objects in 2 GPU calls
        instead of 2*K calls (K = number of valid objects).

        Also warps single_obj_gt if provided (for SAM task consistency with clean branch).
        """
        model = self._model
        if vis_refs is None:
            vis_refs = {}

        if deform_offsets is None:
            return img_batch, pixel_gt.clone(), single_obj_gt.clone() if single_obj_gt is not None else None

        B, K, _, H_img, W_img = deform_offsets.shape
        device = img_batch.device

        # Identify valid objects (non-empty, non-background)
        masks_float = (pixel_gt > 0.5).float()
        mask_areas = masks_float.sum(dim=(2, 3))
        is_empty = mask_areas.sum(dim=0) == 0
        mask_area_ratio = mask_areas / (H_img * W_img)
        is_bg_per_sample = mask_area_ratio > 0.5

        include_background = bool(getattr(model, "adv_enable_background", False))
        is_bg = torch.zeros(K, dtype=torch.bool, device=device)
        if K > 0:
            bg_candidate = is_bg_per_sample[:, -1].all()
            is_bg[-1] = include_background or bg_candidate
        valid_objects = ~(is_empty | is_bg)
        valid_indices = torch.where(valid_objects)[0]
        K_valid = len(valid_indices)

        if K_valid == 0:
            return img_batch, pixel_gt.clone(), single_obj_gt.clone() if single_obj_gt is not None else None

        # Remove valid objects from base image
        valid_masks = masks_float[:, valid_indices, :, :]  # [B, K_valid, H, W]
        valid_masks_union = valid_masks.sum(dim=1, keepdim=True).clamp(0, 1)
        base_img = img_batch * (1 - valid_masks_union)

        # ===== BATCHED WARPING OPTIMIZATION =====
        # Instead of K separate grid_sample calls, we do 2 batched calls:
        # 1. Warp images: [B*K_valid, 3, H, W]
        # 2. Warp masks: [B*K_valid, 1, H, W]

        # Gather offsets for valid objects: [B, K_valid, 2, H, W]
        valid_offsets = deform_offsets[:, valid_indices, :, :, :]

        # Reshape for batched processing
        # Images: expand [B, 3, H, W] -> [B, K_valid, 3, H, W] -> [B*K_valid, 3, H, W]
        img_expanded = img_batch.unsqueeze(1).expand(-1, K_valid, -1, -1, -1)
        img_flat = img_expanded.reshape(B * K_valid, 3, H_img, W_img)

        # Offsets: [B, K_valid, 2, H, W] -> [B*K_valid, 2, H, W]
        offsets_flat = valid_offsets.reshape(B * K_valid, 2, H_img, W_img)

        # Masks: [B, K_valid, H, W] -> [B*K_valid, 1, H, W]
        masks_flat = valid_masks.reshape(B * K_valid, 1, H_img, W_img)

        # Build sample grids once for all objects (batched)
        sample_grids = self._build_sample_grids_batched(offsets_flat, H_img, W_img)

        # Batched warp: 2 grid_sample calls instead of 2*K_valid
        warped_imgs_flat = F.grid_sample(img_flat, sample_grids, mode="bilinear", padding_mode="border", align_corners=True)  # [B*K_valid, 3, H, W]

        warped_masks_flat = F.grid_sample(masks_flat, sample_grids, mode="bilinear", padding_mode="border", align_corners=True)  # [B*K_valid, 1, H, W]

        # Reshape back: [B*K_valid, C, H, W] -> [B, K_valid, C, H, W]
        warped_imgs = warped_imgs_flat.reshape(B, K_valid, 3, H_img, W_img)
        warped_masks = warped_masks_flat.reshape(B, K_valid, 1, H_img, W_img)

        # Binarize masks after warping
        warped_masks_bin = (warped_masks > 0.5).float()  # [B, K_valid, 1, H, W]

        # Apply masks to get per-object warped images
        warped_objs = warped_imgs * warped_masks_bin  # [B, K_valid, 3, H, W]

        # ===== COMPOSITING (already vectorized) =====
        # mask_stack: [B, K_valid, 1, H, W]
        mask_stack = warped_masks_bin
        sum_mask = mask_stack.sum(dim=1).clamp(0, 1)  # [B, 1, H, W]
        background = base_img * (1 - sum_mask)

        overlap_count = mask_stack.sum(dim=1)  # [B, 1, H, W]
        foreground_sum = (warped_objs * mask_stack).sum(dim=1)  # [B, 3, H, W]
        foreground = foreground_sum / overlap_count.clamp(min=1.0)

        augmented_img = background + foreground

        # ===== UPDATE PIXEL_GT (vectorized scatter) =====
        warped_pixel_gt = pixel_gt.clone()
        # warped_masks_bin: [B, K_valid, 1, H, W] -> squeeze to [B, K_valid, H, W]
        warped_masks_squeezed = warped_masks_bin.squeeze(2)
        # Scatter warped masks back to their original positions
        for i, k_idx in enumerate(valid_indices):
            warped_pixel_gt[:, k_idx, :, :] = warped_masks_squeezed[:, i, :, :]

        if enable_vis:
            vis_refs["warped_images"] = augmented_img.detach().cpu()
            vis_refs["warped_pixel_gt"] = warped_pixel_gt.detach().cpu()

        # === WARP SINGLE OBJECT GT (for SAM task) ===
        # Use the combined offset field to warp the single object mask consistently
        warped_single_obj_gt = None
        if single_obj_gt is not None:
            # single_obj_gt: [B, 1, H, W]
            # Use a weighted average offset based on overlap with valid objects
            # For simplicity, use the offset from the object that has most overlap with single_obj_gt
            single_mask = (single_obj_gt > 0.5).float()  # [B, 1, H, W]

            # Find which valid object has most overlap with single_obj_gt per sample
            overlaps = (single_mask * valid_masks).sum(dim=(2, 3))  # [B, K_valid]
            best_obj_idx = overlaps.argmax(dim=1)  # [B]

            # Gather the corresponding offsets: [B, 2, H, W]
            batch_indices = torch.arange(B, device=device)
            selected_offsets = valid_offsets[batch_indices, best_obj_idx]  # [B, 2, H, W]

            # Build sample grids and warp single_obj_gt
            single_sample_grids = self._build_sample_grids_batched(selected_offsets, H_img, W_img)
            warped_single = F.grid_sample(
                single_obj_gt.float(),
                single_sample_grids,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            warped_single_obj_gt = (warped_single > 0.5).float()  # Binarize

        # Cleanup intermediate tensors
        del masks_float, img_expanded, img_flat, offsets_flat, masks_flat
        del sample_grids, warped_imgs_flat, warped_masks_flat
        del warped_imgs, warped_masks, warped_masks_bin, warped_objs

        return augmented_img, warped_pixel_gt, warped_single_obj_gt

    def _build_sample_grids_batched(
        self,
        offset_fields: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Build sample grids for batched grid_sample.

        Args:
            offset_fields: [N, 2, H, W] Offset fields (Δx, Δy) for N samples
            H, W: Spatial dimensions

        Returns:
            sample_grids: [N, H, W, 2] Sample grids for F.grid_sample
        """
        N = offset_fields.shape[0]
        device = offset_fields.device
        dtype = offset_fields.dtype

        # Create base grid once (shared across all samples)
        y_coords = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x_coords = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).expand(N, -1, -1, -1)  # [N, H, W, 2]

        # Normalize offsets to [-1, 1] range
        offset_normalized = offset_fields.clone()
        offset_normalized[:, 0] = offset_fields[:, 0] / (W / 2)  # x offset
        offset_normalized[:, 1] = offset_fields[:, 1] / (H / 2)  # y offset
        offset_normalized = offset_normalized.permute(0, 2, 3, 1)  # [N, H, W, 2]

        # Combine base grid with offsets
        sample_grids = base_grid + offset_normalized

        return sample_grids

    def _apply_offset_to_image(
        self,
        image: torch.Tensor,
        offset_field: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply spatial offset field to warp an image using grid_sample.

        Note: This method is kept for backward compatibility and single-object use.
        For multi-object batched warping, use _build_sample_grids_batched + F.grid_sample.

        Args:
            image: [B, C, H, W] Input image
            offset_field: [B, 2, H, W] Offset field (Δx, Δy)

        Returns:
            warped_image: [B, C, H, W] Warped image
        """
        B, C, H, W = image.shape
        sample_grid = self._build_sample_grids_batched(offset_field, H, W)

        # Warp image
        warped = F.grid_sample(
            image,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return warped

    def _compute_augmented_calibration_loss(
        self,
        state: dict,
        uq_sample_num: int,
    ) -> tuple[torch.Tensor, dict, torch.Tensor | None]:
        """
        Compute calibration loss for augmented features.

        Args:
            state: Current attack state with features, high_res, pixel_gt
            uq_sample_num: Number of samples for uncertainty

        Returns:
            calibration_loss, aug_metrics, adv_uncertainty
        """
        model = self._model
        device = state["features"].device
        dtype = state["features"].dtype

        # Get BNDL model
        pixel_bndl_model = getattr(model, "sam_mask_decoder", None)
        if pixel_bndl_model is not None:
            pixel_bndl_model = getattr(pixel_bndl_model, "pixel_bndl", None)

        if pixel_bndl_model is None:
            return torch.tensor(0.0, device=device, dtype=dtype), {}, None

        # Run mask decoder on augmented features to get aux_outputs
        high_res_features = state.get("high_res")
        backbone_features = state["features"]
        pixel_gt = state["pixel_gt"]

        # Prepare gt for prompts
        if pixel_gt.shape[1] > 1:
            pixel_gt_combined = pixel_gt.sum(dim=1, keepdim=True).clamp(0, 1)
        else:
            pixel_gt_combined = pixel_gt

        # Generate bbox prompts from GT (used for potential box-prompted inference)
        _ = model._generate_bbox_prompts_from_gt(pixel_gt_combined)  # Reserved for future use

        # Forward through SAM heads
        # Note: _forward_sam_heads expects backbone_features in [B, C, H, W] format
        # The method will internally handle the transformer input transformation
        _, _, _, _, _, _, _, aux_outputs = model._forward_sam_heads(
            backbone_features=backbone_features,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=high_res_features,
            multimask_output=False,
        )

        # Extract BNDL outputs
        from sam2.modeling.aue.loss import extract_bndl_outputs, prepare_gt_for_loss

        bndl_outputs = extract_bndl_outputs(
            aux_outputs=aux_outputs,
            pixel_bndl_model=pixel_bndl_model,
            compute_logits=True,
            compute_analytic_uncertainty=True,
        )

        # Prepare GT for loss
        H_feat, W_feat = bndl_outputs.pixel_logits.shape[1], bndl_outputs.pixel_logits.shape[2]
        pixel_gt_resized = prepare_gt_for_loss(pixel_gt, (H_feat, W_feat))

        # Compute calibration loss

        calibration_loss, aug_metrics, uncertainty = model._aue_module._loss_computer.compute_calibration_loss(
            bndl_outputs=bndl_outputs,
            pixel_gt=pixel_gt_resized,
            pixel_bndl_model=pixel_bndl_model,
            backbone_features=backbone_features,
            use_analytic_uncertainty=getattr(model, "aue_use_analytic_uncertainty", True),
            use_patches=getattr(model, "aue_use_patches", True),
            tag="adversarial",
        )

        return calibration_loss, aug_metrics, uncertainty


def cleanup_augmentation_results(results_to_cleanup: list) -> None:
    """
    Clean up augmentation results to free GPU memory.

    Args:
        results_to_cleanup: List of augmentation result objects with release_intermediate() method.
    """
    for result in reversed(results_to_cleanup):
        if hasattr(result, "release_intermediate"):
            result.release_intermediate()
