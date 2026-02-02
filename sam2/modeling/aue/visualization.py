# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
AUE Visualization Utilities.

Provides utilities for preparing visualization data during AUE training,
including adversarial images, style transformations, and deformation offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


@dataclass
class AUEVisualizationData:
    """Container for AUE visualization data."""

    original_images: torch.Tensor
    adversarial_images: torch.Tensor
    original_styles: torch.Tensor | None
    adversarial_styles: torch.Tensor | None
    num_objects: int
    object_masks: torch.Tensor | None
    object_bboxes: torch.Tensor | None
    object_area_ratios: torch.Tensor | None
    deform_offsets: torch.Tensor | None
    warped_images: torch.Tensor | None
    warped_object_masks: torch.Tensor | None = None
    attack_order: list[str] | None = None
    loss_object_indices: torch.Tensor | None = None


class AUEVisualizer:
    """
    Prepare visualization data for AUE training.

    Handles the collection and formatting of visualization payloads
    for tensorboard logging and debugging.
    """

    def prepare_visualization_data(
        self,
        img_batch: torch.Tensor,
        adv_images: torch.Tensor,
        pixel_gt: torch.Tensor,
        num_vis_samples: int = 4,
        original_styles: torch.Tensor | None = None,
        adv_styles: torch.Tensor | None = None,
        deform_offsets: torch.Tensor | None = None,
        warped_images: torch.Tensor | None = None,
        warped_masks: torch.Tensor | None = None,
        attack_order: list[str] | None = None,
        loss_object_indices: torch.Tensor | None = None,
    ) -> AUEVisualizationData:
        """
        Prepare visualization data for Style AUE multi-object training.

        This method computes all necessary visualization data from masks on-demand,
        avoiding extra memory overhead during training. Bounding boxes, area ratios,
        etc. are computed here for visualization purposes only.

        Args:
            img_batch: [B, 3, H, W] original images (CPU tensor)
            adv_images: [B, 3, H, W] adversarial images (CPU tensor)
            pixel_gt: [B, K, H, W] ground truth masks (CPU tensor)
            num_vis_samples: Number of samples to visualize
            original_styles: [B, K, C] original style features
            adv_styles: [B, K, C] adversarial style features
            deform_offsets: [B, K, 2, H, W] deformation offsets
            warped_images: [B, 3, H, W] warped images
            warped_masks: [B, K, H, W] warped masks
            attack_order: List of attack names in order
            loss_object_indices: Indices of objects used for loss

        Returns:
            AUEVisualizationData with all visualization payloads
        """
        B = min(num_vis_samples, img_batch.shape[0])
        K = pixel_gt.shape[1] if pixel_gt.ndim == 4 else 1

        # Ensure tensors are on CPU and sliced to num_vis_samples
        orig_imgs = img_batch[:B].cpu() if img_batch.is_cuda else img_batch[:B]
        adv_imgs = adv_images[:B].cpu() if adv_images.is_cuda else adv_images[:B]

        # Process masks
        if pixel_gt.ndim == 4:
            masks = pixel_gt[:B].cpu() if pixel_gt.is_cuda else pixel_gt[:B]
        else:
            masks = pixel_gt[:B].unsqueeze(1).cpu() if pixel_gt.is_cuda else pixel_gt[:B].unsqueeze(1)

        # Compute bounding boxes and area ratios from masks
        bboxes = self._compute_bboxes_from_masks(masks)
        area_ratios = self._compute_area_ratios(masks)

        # Process optional tensors
        orig_styles_vis = original_styles[:B].cpu() if original_styles is not None else None
        adv_styles_vis = adv_styles[:B].cpu() if adv_styles is not None else None
        deform_offsets_vis = deform_offsets[:B].cpu() if deform_offsets is not None else None
        warped_images_vis = warped_images[:B].cpu() if warped_images is not None else None
        warped_masks_vis = warped_masks[:B].cpu() if warped_masks is not None else None

        return AUEVisualizationData(
            original_images=orig_imgs,
            adversarial_images=adv_imgs,
            original_styles=orig_styles_vis,
            adversarial_styles=adv_styles_vis,
            num_objects=K,
            object_masks=masks,
            object_bboxes=bboxes,
            object_area_ratios=area_ratios,
            deform_offsets=deform_offsets_vis,
            warped_images=warped_images_vis,
            warped_object_masks=warped_masks_vis,
            attack_order=attack_order,
            loss_object_indices=loss_object_indices,
        )

    def _compute_bboxes_from_masks(
        self,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute bounding boxes from masks.

        Args:
            masks: [B, K, H, W] binary masks

        Returns:
            bboxes: [B, K, 4] bounding boxes in format [x1, y1, x2, y2]
        """
        B, K, H, W = masks.shape
        bboxes = torch.zeros(B, K, 4, dtype=torch.float32)

        for b in range(B):
            for k in range(K):
                mask = masks[b, k]
                nonzero = torch.nonzero(mask > 0.5, as_tuple=False)

                if nonzero.shape[0] > 0:
                    y_min = nonzero[:, 0].min().item()
                    y_max = nonzero[:, 0].max().item()
                    x_min = nonzero[:, 1].min().item()
                    x_max = nonzero[:, 1].max().item()
                    bboxes[b, k] = torch.tensor([x_min, y_min, x_max, y_max])
                else:
                    # Empty mask - use center box
                    bboxes[b, k] = torch.tensor([W // 2 - 10, H // 2 - 10, W // 2 + 10, H // 2 + 10])

        return bboxes

    def _compute_area_ratios(
        self,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute area ratios (mask area / image area) for each object.

        Args:
            masks: [B, K, H, W] binary masks

        Returns:
            area_ratios: [B, K] area ratios in [0, 1]
        """
        B, K, H, W = masks.shape
        total_pixels = H * W

        mask_areas = (masks > 0.5).float().sum(dim=(2, 3))
        area_ratios = mask_areas / total_pixels

        return area_ratios

    def select_adv_image_for_vis(self, vis_refs: dict) -> torch.Tensor:
        """
        Select the best adversarial image for visualization.

        Priority: styled_images > warped_images > img_batch

        Args:
            vis_refs: Dictionary containing visualization references

        Returns:
            Selected adversarial image tensor
        """
        if "styled_images" in vis_refs and vis_refs["styled_images"] is not None:
            return vis_refs["styled_images"]
        if "warped_images" in vis_refs and vis_refs["warped_images"] is not None:
            return vis_refs["warped_images"]
        return vis_refs["img_batch"]
