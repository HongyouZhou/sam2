# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
AUE Module - Main entry point for Adversarial Uncertainty Estimation.

Provides the AUEModule class that orchestrates AUE functionality
using the composition pattern for clean separation from SAM2Base.

NOTE: MMD loss computation has been moved to loss functions (loss_mmd.py, loss_aue.py)
following SAM's interface pattern. This module now only handles:
- Adversarial sample generation
- Visualization data preparation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sam2.modeling.aue.pipeline import AdversarialPipeline
from sam2.modeling.aue.visualization import AUEVisualizer

if TYPE_CHECKING:
    from sam2.modeling.sam2_base import SAM2Base


class AUEModule:
    """
    Main AUE module using composition pattern.

    This class orchestrates AUE functionality including:
    - Adversarial attack pipeline (Style/Deform)
    - Visualization data preparation

    NOTE: MMD loss computation is now handled by loss functions:
    - Clean MMD: loss_mmd.py (MMDCalibrationLoss)
    - Adversarial MMD: loss_aue.py (AUELoss)

    Usage:
        # In SAM2Base.__init__
        self._aue_module = AUEModule(self)

        # In training loop
        adv_samples = self._aue_module.generate_adversarial_samples(...)

    Args:
        model: Reference to SAM2Base model
    """

    def __init__(self, model: "SAM2Base"):
        self._model = model
        self._pipeline: AdversarialPipeline | None = None
        self._visualizer = AUEVisualizer()

    def initialize(self) -> None:
        """
        Initialize AUE components after model is fully constructed.
        """
        model = self._model

        if not getattr(model, "use_aue", False):
            return

        # Initialize adversarial pipeline (Style/Deform/PGD/RandomNoise attackers)
        if getattr(model, "use_style_adv", False) or getattr(model, "use_deform_adv", False) or getattr(model, "use_pgd_adv", False) or getattr(model, "use_patch_adv", False) or getattr(model, "use_random_noise_adv", False):
            self._pipeline = AdversarialPipeline(model)

    def generate_adversarial_samples(
        self,
        img_batch: torch.Tensor,
        backbone_features: torch.Tensor,
        high_res_features: list[torch.Tensor],
        pixel_gt: torch.Tensor,
        single_obj_gt: torch.Tensor | None = None,
        enable_vis: bool = False,
    ) -> dict:
        """
        Generate adversarial samples without forward pass or loss computation.

        This is a convenience method that delegates to the pipeline.

        Args:
            img_batch: [B, 3, H, W] input images
            backbone_features: [B, C, H, W] backbone features
            high_res_features: List of high-res features
            pixel_gt: [B, K, H, W] ground truth masks (all objects, for attack generation)
            single_obj_gt: [B, 1, H, W] single object GT (same as clean branch, for SAM task)
            enable_vis: Whether to collect visualization data

        Returns:
            Dict with adv_img, adv_features, adv_high_res, adv_pixel_gt, adv_single_obj_gt, vis_refs
        """
        if self._pipeline is None:
            return {
                "adv_img": img_batch,
                "adv_features": backbone_features,
                "adv_high_res": high_res_features,
                "adv_pixel_gt": pixel_gt,
                "adv_single_obj_gt": single_obj_gt,
                "vis_refs": {},
            }

        return self._pipeline.generate_adversarial_samples(
            img_batch=img_batch,
            backbone_features=backbone_features,
            high_res_features=high_res_features,
            pixel_gt=pixel_gt,
            single_obj_gt=single_obj_gt,
            enable_vis=enable_vis,
        )

    def _register_gradient_hooks(
        self,
        clean_loss: torch.Tensor,
        adv_loss: torch.Tensor,
    ) -> None:
        """Register gradient hooks for debugging."""

        def _make_grad_hook(name: str):
            def hook(grad):
                if grad is not None:
                    grad_norm = grad.norm().item()
                    grad_abs_max = grad.abs().max().item()
                    has_nan = torch.isnan(grad).any().item()
                    has_inf = torch.isinf(grad).any().item()
                    logging.debug(f"AUE Gradient [{name}]: norm={grad_norm:.6f}, max={grad_abs_max:.6f}, NaN={has_nan}, Inf={has_inf}")
                return grad

            return hook

        if isinstance(clean_loss, torch.Tensor) and clean_loss.requires_grad:
            clean_loss.register_hook(_make_grad_hook("clean_loss"))
        if isinstance(adv_loss, torch.Tensor) and adv_loss.requires_grad:
            adv_loss.register_hook(_make_grad_hook("adv_loss"))
