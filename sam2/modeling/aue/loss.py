# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
AUE Loss Computation Utilities.

Provides loss computation functions for Adversarial Uncertainty Estimation (AUE).
Handles distribution matching between uncertainty and prediction error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from sam2.modeling.bndl_utils import BNDLOutputs
    from sam2.modeling.distribution_matching import DistributionMatcher


class AUELossComputer:
    """
    Compute uncertainty calibration losses for AUE training.

    This class handles the core loss computation logic, including:
    - Clean sample calibration loss (MMD between uncertainty and error)
    - Adversarial sample calibration loss
    - Prediction error computation
    - Confidence computation

    Args:
        distribution_matcher: DistributionMatcher instance for MMD/CKA computation
        config: AUE configuration dict
    """

    def __init__(
        self,
        distribution_matcher: "DistributionMatcher",
        config: dict,
    ):
        self.distribution_matcher = distribution_matcher
        self.config = config

    def compute_calibration_loss(
        self,
        bndl_outputs: "BNDLOutputs",
        pixel_gt: torch.Tensor | None,
        pixel_bndl_model=None,
        backbone_features: torch.Tensor | None = None,
        use_analytic_uncertainty: bool = True,
        use_patches: bool = True,
        tag: str = "unknown",
    ) -> tuple[torch.Tensor, dict, torch.Tensor | None]:
        """
        Compute uncertainty calibration loss via distribution matching.

        Theory: For zero-shot robustness, uncertainty distribution should match
        error distribution using Maximum Mean Discrepancy (MMD).

        Loss = MMD(P_U, P_Error) + 0.3 * MSE(U, Error)

        Key innovation: Uses analytic uncertainty (from Weibull parameters) to enable
        bidirectional optimization - both uncertainty and error are optimized to align.

        Args:
            bndl_outputs: BNDLOutputs containing pixel_feat, pixel_logits, external_w, pixel_uncertainty
            pixel_gt: [B, H, W] ground truth masks (already combined and resized)
            pixel_bndl_model: BNDL model (required for analytic uncertainty)
            backbone_features: [B, C, H, W] feature map for domain_aware_soft_mmd
            use_analytic_uncertainty: Whether to use analytic uncertainty (with gradients)
            use_patches: Whether to use patch-based distribution matching
            tag: Debug tag for logging

        Returns:
            calibration_loss: MMD-based distribution matching loss
            metrics: Dict of metrics for logging
            uncertainty: [B, H, W] uncertainty map
        """
        # Extract fields from bndl_outputs
        pixel_logits = bndl_outputs.pixel_logits
        pixel_feat = bndl_outputs.pixel_feat
        external_pre_out_w = bndl_outputs.external_w
        pixel_uncertainty = bndl_outputs.pixel_uncertainty

        device = pixel_feat.device
        dtype = pixel_feat.dtype

        if pixel_logits is None:
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True), {}, None

        # 1. Compute prediction error [B, H, W] in [0, 1]
        error = self.compute_prediction_error(
            pixel_logits=pixel_logits,
            pixel_gt=pixel_gt,
        )

        # 2. Get uncertainty [B, H, W] in [0, 1]
        if use_analytic_uncertainty and pixel_feat is not None and pixel_bndl_model is not None:
            # Analytic uncertainty (preserves gradients to BNDL)
            from sam2.modeling.bndl_utils import pixel_weibull_to_entropy_uncertainty

            uncertainty = pixel_weibull_to_entropy_uncertainty(
                pixel_bndl_model=pixel_bndl_model,
                pixel_feat=pixel_feat,
                external_pre_out_w=external_pre_out_w,
                per_channel=False,
            )
            uncertainty = uncertainty.clamp(0.0, 1.0)

        elif pixel_uncertainty is not None:
            # Sampling-based uncertainty (provided externally)
            uncertainty = pixel_uncertainty.clamp(0.0, 1.0)
        else:
            # Fallback: use 1 - confidence
            confidence = self.compute_confidence(pixel_logits, pixel_gt)
            uncertainty = (1.0 - confidence).clamp(0.0, 1.0)

        # 3. Distribution matching loss (primary: MMD/CKA/Gram)
        dist_loss, metrics = self.distribution_matcher.compute_loss(
            uncertainty=uncertainty,
            error=error,
            use_patches=use_patches,
            feature_map=backbone_features,
            tag=tag,
        )

        # 4. MSE loss (regularization)
        # CRITICAL: Detach error to enforce One-Way Alignment strategy
        # Only BNDL learns to predict uncertainty; backbone should not be affected
        mse_loss = F.mse_loss(uncertainty, error.detach(), reduction="mean")

        # 5. Combine losses
        total_loss = 1.0 * dist_loss + 0.3 * mse_loss

        metrics["mse_loss"] = mse_loss.item()

        return total_loss, metrics, uncertainty

    def compute_prediction_error(
        self,
        pixel_logits: torch.Tensor,
        pixel_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-pixel prediction error in [0, 1].

        Error = |sigmoid(logits) - GT|

        Args:
            pixel_logits: [B, H, W, K] or [B, H, W] logits
            pixel_gt: [B, H, W] or [B, 1, H, W] ground truth

        Returns:
            error: [B, H, W] in [0, 1]
        """
        # Extract logits value
        if pixel_logits.ndim == 4 and pixel_logits.shape[-1] >= 1:
            logits_val = pixel_logits.max(dim=-1).values
        elif pixel_logits.ndim == 3:
            logits_val = pixel_logits
        elif pixel_logits.ndim == 4 and pixel_logits.shape[1] == 1:
            logits_val = pixel_logits[:, 0]
        else:
            B, H, W = pixel_logits.shape[:3]
            logits_val = pixel_logits.view(B, H, W, -1).max(dim=-1).values

        # Extract GT mask
        H, W = logits_val.shape[1], logits_val.shape[2]
        B = logits_val.shape[0]
        gt_mask = self._extract_mask_from_gt(
            pixel_gt=pixel_gt,
            spatial_hw=(H, W),
            batch_size=B,
            device=logits_val.device,
        )

        # Compute prediction probability and error
        pred_prob = torch.sigmoid(logits_val)
        gt_float = gt_mask.float()
        error = torch.abs(pred_prob - gt_float)

        return error

    def compute_confidence(
        self,
        pixel_logits: torch.Tensor,
        pixel_gt: torch.Tensor,
        tau_conf: float = 2.0,
    ) -> torch.Tensor:
        """
        Compute GT-aligned per-pixel confidence in [0,1].

        Confidence reflects prediction correctness:
        - GT=1, logits=+5 → high confidence (correct foreground)
        - GT=0, logits=-5 → high confidence (correct background)

        Formula: c = sigmoid((logits * (2*gt - 1)) / tau_conf)
        """
        # Extract logits value
        if pixel_logits.ndim == 4 and pixel_logits.shape[-1] >= 1:
            logits_val = pixel_logits.max(dim=-1).values
        elif pixel_logits.ndim == 3:
            logits_val = pixel_logits
        elif pixel_logits.ndim == 4 and pixel_logits.shape[1] == 1:
            logits_val = pixel_logits[:, 0]
        else:
            B, H, W = pixel_logits.shape[0], pixel_logits.shape[1], pixel_logits.shape[2]
            logits_val = pixel_logits.view(B, H, W, -1).mean(dim=-1)

        H, W = logits_val.shape[1], logits_val.shape[2]
        B = logits_val.shape[0]
        gt_mask = self._extract_mask_from_gt(
            pixel_gt=pixel_gt,
            spatial_hw=(H, W),
            batch_size=B,
            device=logits_val.device,
        )

        # Convert GT to sign: [0,1] → [-1,+1]
        gt_sign = 2.0 * gt_mask.float() - 1.0
        aligned_logits = logits_val * gt_sign

        return torch.sigmoid(aligned_logits / float(tau_conf))

    def compute_confidence_from_logits(
        self,
        logits: torch.Tensor,
        tau_conf: float = 2.0,
    ) -> torch.Tensor:
        """Compute confidence from logits tensor [*, H, W, K] → [*, H, W]."""
        if logits.ndim < 3:
            raise ValueError("logits tensor rank too low")
        mag = logits.abs().max(dim=-1).values
        return torch.sigmoid(mag / float(tau_conf)).to(mag.dtype)

    def _extract_mask_from_gt(
        self,
        pixel_gt: torch.Tensor | None,
        spatial_hw: tuple[int, int],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Extract positive mask from GT tensor [B, H, W]."""
        H, W = spatial_hw

        if pixel_gt is None:
            return torch.ones((batch_size, H, W), device=device, dtype=torch.bool)

        gt = pixel_gt
        if gt.ndim == 4 and gt.shape[-1] > 1:
            pos = (gt > 0).any(dim=-1)
        elif gt.ndim == 4 and gt.shape[1] == 1 and gt.shape[2] == H and gt.shape[3] == W:
            pos = gt[:, 0] > 0
        elif gt.ndim == 3 and gt.shape[1] == H and gt.shape[2] == W:
            pos = gt > 0
        else:
            try:
                B = gt.shape[0]
                pos = (gt.view(B, H, W, -1) > 0).any(dim=-1)
            except Exception:
                pos = torch.ones((gt.shape[0], H, W), device=gt.device, dtype=torch.bool)

        return pos.to(torch.bool)


def extract_bndl_outputs(
    aux_outputs: dict,
    pixel_bndl_model,
    compute_logits: bool = True,
    use_sampling_uncertainty: bool = False,
    use_analytic_uncertainty: bool = False,
    uq_sample_num: int = 20,
) -> "BNDLOutputs":
    """
    Extract and process BNDL outputs from aux_outputs dict.

    Centralizes the pattern of extracting pixel_feat, external_w, logits, uncertainty.

    === UNCERTAINTY TYPES ===
    - use_sampling_uncertainty: Use pre-computed sampling uncertainty (no gradients)
      For: visualization, logging, attacker training
    - use_analytic_uncertainty: Use pre-computed analytic uncertainty (with gradients)
      For: MMD calibration, BNDL training

    NOTE: At most one of use_sampling_uncertainty/use_analytic_uncertainty should be True.
          If both are False, no uncertainty is extracted.

    Args:
        aux_outputs: Auxiliary outputs containing BNDL dict
        pixel_bndl_model: Pixel BNDL model (unused in new design, kept for compatibility)
        compute_logits: Whether to extract pixel logits
        use_sampling_uncertainty: Extract sampling-based uncertainty (no gradients)
        use_analytic_uncertainty: Extract analytic uncertainty (with gradients)
        uq_sample_num: Unused (kept for API compatibility)

    Returns:
        BNDLOutputs dataclass
    """
    from sam2.modeling.bndl_utils import BNDLOutputs

    bndl_dict = aux_outputs.get("bndl", {})
    if not bndl_dict:
        raise ValueError("aux_outputs must contain 'bndl' key")

    pixel_feat = bndl_dict.get("pixel_feat")
    # mask_tokens_out [B, K, 256] is the raw mask token embeddings
    external_w = bndl_dict.get("mask_tokens_out")

    if pixel_feat is None:
        raise ValueError("BNDL dict missing 'pixel_feat'")

    # Get pixel_logits directly from the bndl dict (already computed during forward)
    pixel_logits = None
    if compute_logits:
        # Prefer gradient-carrying logits if available
        pixel_logits = bndl_dict.get("pixel_logits", bndl_dict.get("masks_bndl_raw"))

    # === EXPLICIT uncertainty extraction - no fallbacks ===
    pixel_uncertainty = None
    if use_analytic_uncertainty:
        # Analytic uncertainty: has gradients, for MMD calibration
        pixel_uncertainty = bndl_dict.get("pixel_uncertainty_analytic")
        if pixel_uncertainty is None:
            import logging

            logging.debug("[extract_bndl_outputs] pixel_uncertainty_analytic not available")
    elif use_sampling_uncertainty:
        # Sampling uncertainty: no gradients, for visualization/logging/attacker
        pixel_uncertainty = bndl_dict.get("pixel_uncertainty_sampling")
        if pixel_uncertainty is None:
            # Legacy fallback to "pixel_uncertainty" (which should also be sampling)
            pixel_uncertainty = bndl_dict.get("pixel_uncertainty")
        if pixel_uncertainty is None:
            import logging

            logging.debug("[extract_bndl_outputs] pixel_uncertainty_sampling not available")

    return BNDLOutputs(
        pixel_feat=pixel_feat,
        external_w=external_w,
        pixel_logits=pixel_logits,
        pixel_uncertainty=pixel_uncertainty,
    )


def prepare_gt_for_loss(
    pixel_gt: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """
    Prepare GT masks: combine multi-object and resize to target resolution.

    Args:
        pixel_gt: [B, K, H, W] ground truth masks
        target_size: (H_feat, W_feat) target spatial size

    Returns:
        [B, H_feat, W_feat] combined and resized GT
    """
    if pixel_gt.shape[1] > 1:
        # Combine multiple objects into single mask
        pixel_gt_combined = pixel_gt.sum(dim=1, keepdim=True).clamp(0, 1)
    else:
        pixel_gt_combined = pixel_gt

    # Resize to feature map resolution
    pixel_gt_resized = F.interpolate(
        pixel_gt_combined.float(),
        size=target_size,
        mode="nearest",
    ).squeeze(1)

    return pixel_gt_resized
