# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
MMD (Maximum Mean Discrepancy) Calibration Loss.

This loss aligns predicted uncertainty with true prediction error,
providing well-calibrated uncertainty estimates.

Design Pattern (SAM-style):
- Model outputs raw data (pixel_feat, pixel_logits, pixel_gt, etc.)
- Loss function computes MMD only when mmd_weight > 0
- No computation in model forward pass
"""

import torch
import torch.nn as nn

from training.trainer import CORE_LOSS_KEY


class MMDCalibrationLoss(nn.Module):
    """
    MMD Calibration Loss for uncertainty estimation on CLEAN samples only.

    SAM-style interface design:
    - Model outputs raw BNDL data to bndl_ns
    - This loss module computes MMD using those inputs
    - Creates its own DistributionMatcher from config (no model dependency)
    - Computation only happens when the loss is actually called

    Args:
        config: Distribution matching config (from scratch.mmd_config in YAML)
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = config or {}
        self._loss_computer = None
        self._distribution_matcher = None

        # Create distribution_matcher from config if provided
        if config:
            self._init_from_config(config)

    def _init_from_config(self, config: dict) -> None:
        """Initialize DistributionMatcher and AUELossComputer from config."""
        from sam2.modeling.distribution_matching import DistributionMatcher
        from sam2.modeling.aue.loss import AUELossComputer

        # Extract parameters from config
        method = config.get("method", "spatial_mmd")
        patch_size = config.get("patch_size", 16)
        kernel_bandwidth = config.get("kernel_bandwidth", 0.3)

        self._distribution_matcher = DistributionMatcher(
            method=method,
            patch_size=patch_size,
            kernel="rbf",
            bandwidth=kernel_bandwidth,
            # Use defaults for other parameters
            cka_use_linear_kernel=True,
            cka_use_minibatch=True,
            cka_minibatch_size=512,
            top_k_percent=0.25,
            max_samples=4096,
            diversity_weight=0.4,
            temperature=config.get("temperature", 0.5),
            diversity_method="channel_std",
            enable_monitoring=False,
            use_checkpoint=config.get("use_checkpoint", False),
        )

        self._loss_computer = AUELossComputer(
            distribution_matcher=self._distribution_matcher,
            config=config,
        )

    def forward(
        self,
        outs_batch: list[dict],
        targets_batch: torch.Tensor | None = None,
    ):
        """
        Compute MMD calibration loss on CLEAN samples only.

        Reads raw data from bndl_ns and computes MMD here (not in model).
        """
        device = targets_batch.device if targets_batch is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Early exit if distribution matcher not set
        if self._loss_computer is None:
            return {
                CORE_LOSS_KEY: torch.tensor(0.0, device=device, requires_grad=True),
                "mmd_scalar": torch.tensor(0.0, device=device),
            }

        mmd_losses = []
        all_metrics = {}

        for outs in outs_batch:
            # Get bndl_outputs list
            if "multistep_aux_outputs" in outs:
                aux_list = outs["multistep_aux_outputs"]
                bndl_outputs_list = [aux.get("bndl") if isinstance(aux, dict) else None for aux in aux_list]
            elif "multistep_bndl_outputs" in outs:
                bndl_outputs_list = outs["multistep_bndl_outputs"]
            else:
                continue

            # Use the last valid bndl_outputs (final refinement step)
            for bndl_ns in reversed(bndl_outputs_list):
                if bndl_ns is None:
                    continue

                # Extract raw data for MMD computation
                loss, metrics = self._compute_mmd_from_bndl_ns(bndl_ns)

                if loss is not None and loss.requires_grad:
                    mmd_losses.append(loss)
                    # Merge metrics
                    for k, v in metrics.items():
                        all_metrics[k] = v

                break  # Only use first valid bndl_outputs

        # Aggregate losses
        if mmd_losses:
            total_loss = torch.stack(mmd_losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        result = {
            CORE_LOSS_KEY: total_loss,
            "mmd_scalar": total_loss.detach(),
        }
        result.update(all_metrics)

        return result

    def _compute_mmd_from_bndl_ns(
        self,
        bndl_ns: dict,
    ) -> tuple[torch.Tensor | None, dict]:
        """
        Compute MMD loss from raw BNDL namespace data.

        This is where the actual MMD computation happens.
        """
        from sam2.modeling.aue.loss import prepare_gt_for_loss
        from sam2.modeling.bndl_utils import BNDLOutputs

        # Extract required data from bndl_ns
        pixel_feat = bndl_ns.get("pixel_feat_grad", bndl_ns.get("pixel_feat"))
        pixel_logits = bndl_ns.get(
            "pixel_logits",
            bndl_ns.get("masks_bndl_raw", bndl_ns.get("mean_pixel_logits")),
        )
        pixel_gt = bndl_ns.get("pixel_gt")
        external_w = bndl_ns.get("mask_tokens_out")

        # === EXPLICIT: Use ANALYTIC uncertainty for MMD calibration ===
        # Analytic uncertainty has gradients, enabling BNDL parameter updates
        # This aligns predicted uncertainty with true prediction error (calibration)
        pixel_uncertainty = bndl_ns.get("pixel_uncertainty_analytic")
        if pixel_uncertainty is None:
            # MMD calibration requires analytic uncertainty for gradient flow
            import logging

            logging.debug("[MMD Clean] pixel_uncertainty_analytic not available, skipping MMD loss")
            return None, {}

        # Validate required inputs
        if pixel_feat is None or pixel_logits is None:
            return None, {}

        if pixel_gt is None:
            return None, {}

        # Prepare GT for loss computation
        H_feat, W_feat = int(pixel_logits.shape[1]), int(pixel_logits.shape[2])

        # Handle pixel_gt shape - it might already be [B, H, W] or [B, K, H, W]
        if pixel_gt.ndim == 3:
            # Already [B, H, W], add channel dim for prepare_gt_for_loss
            pixel_gt_for_prep = pixel_gt.unsqueeze(1)
        else:
            pixel_gt_for_prep = pixel_gt

        pixel_gt_resized = prepare_gt_for_loss(pixel_gt_for_prep, (H_feat, W_feat))

        # Wrap inputs in BNDLOutputs
        bndl_outputs = BNDLOutputs(
            pixel_feat=pixel_feat,
            pixel_logits=pixel_logits,
            external_w=external_w,
            pixel_uncertainty=pixel_uncertainty,  # Pre-computed with gradients
        )

        # Compute calibration loss using pre-computed uncertainty
        loss, metrics, _ = self._loss_computer.compute_calibration_loss(
            bndl_outputs=bndl_outputs,
            pixel_gt=pixel_gt_resized,
            backbone_features=None,
            use_analytic_uncertainty=False,  # Use provided pixel_uncertainty
            use_patches=self.config.get("use_patches", True),
            tag="clean",
        )

        return loss, metrics
