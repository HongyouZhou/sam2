# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
AUE (Adversarial Uncertainty Estimation) Loss.

Computes losses on adversarial samples using the SAME loss functions as the
main branch (sam_loss, bndl_loss). These losses update:
- Model (Decoder): via normal gradient descent
- Attacker (Style/Deform): via GRL (gradient reversal)

MMD calibration is now computed here (not in model) following SAM's pattern.
"""

import logging

import torch
import torch.nn as nn

from training.trainer import CORE_LOSS_KEY


class AUELoss(nn.Module):
    """
    AUE Loss for adversarial training.

    Uses the same loss functions as the main branch but on adversarial samples.
    Gradients flow to Attacker via GRL in Style/Deform networks.

    SAM-style design:
    - Model outputs raw adversarial data to bndl_ns
    - This loss module computes all losses (task, BNDL, MMD)
    - Creates its own DistributionMatcher from config (no model dependency)
    """

    def __init__(
        self,
        task_weight: float = 1.0,
        bndl_weight: float = 0.0,
        mmd_weight: float = 0.0,
        attacker_uncertainty_weight: float = 0.0,  # Attacker maximizes raw uncertainty
        attacker_mmd_weight: float = 0.0,  # Attacker maximizes miscalibration (uncertainty ≠ error)
        attacker_task_weight: float = 0.0,  # NEW: Task loss only updates attacker (NO SAM update)
        mmd_config: dict | None = None,
        sam_loss_fn: nn.Module | None = None,
        bndl_loss_fn: nn.Module | None = None,
    ):
        """
        Args:
            task_weight: Weight for SAM task loss on adv samples (updates SAM + attacker via GRL)
            bndl_weight: Weight for BNDL KL loss on adv samples (updates BNDL + attacker via GRL)
            mmd_weight: Weight for MMD loss on adv samples (updates BNDL calibration + attacker via GRL)
            attacker_uncertainty_weight: Attacker maximizes uncertainty (NO BNDL update)
            attacker_mmd_weight: Attacker maximizes miscalibration (NO SAM/BNDL update)
            attacker_task_weight: Task loss only updates attacker (NO SAM update) - for complete isolation
            mmd_config: Distribution matching config for adversarial MMD
            sam_loss_fn: Reference to SAM loss (injected by CombinedLoss)
            bndl_loss_fn: Reference to BNDL loss (injected by CombinedLoss)
        """
        super().__init__()
        self.task_weight = task_weight
        self.bndl_weight = bndl_weight
        self.mmd_weight = mmd_weight
        self.attacker_uncertainty_weight = attacker_uncertainty_weight
        self.attacker_mmd_weight = attacker_mmd_weight
        self.attacker_task_weight = attacker_task_weight
        self.sam_loss_fn = sam_loss_fn
        self.bndl_loss_fn = bndl_loss_fn

        # Initialize MMD components from config
        self._loss_computer = None
        self._distribution_matcher = None

        if mmd_config and (mmd_weight > 0 or attacker_mmd_weight > 0):
            self._init_mmd_from_config(mmd_config)

    def _init_mmd_from_config(self, config: dict) -> None:
        """Initialize DistributionMatcher and AUELossComputer from config."""
        from sam2.modeling.distribution_matching import DistributionMatcher
        from sam2.modeling.aue.loss import AUELossComputer

        method = config.get("method", "spatial_mmd")
        patch_size = config.get("patch_size", 16)
        kernel_bandwidth = config.get("kernel_bandwidth", 0.3)

        self._distribution_matcher = DistributionMatcher(
            method=method,
            patch_size=patch_size,
            kernel="rbf",
            bandwidth=kernel_bandwidth,
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

    def forward(self, outs_batch: list[dict], targets_batch: torch.Tensor | None = None):
        """
        Compute AUE losses on adversarial samples.

        These losses update both Model and Attacker (via GRL).
        """
        device = targets_batch.device if targets_batch is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        task_losses = []
        bndl_losses = []
        mmd_losses = []
        attacker_uncertainty_losses = []
        attacker_mmd_losses = []  # NEW

        for outs in outs_batch:
            # Get bndl_outputs list
            if "multistep_aux_outputs" in outs:
                aux_list = outs["multistep_aux_outputs"]
                bndl_outputs_list = [aux.get("bndl") if isinstance(aux, dict) else None for aux in aux_list]
            elif "multistep_bndl_outputs" in outs:
                bndl_outputs_list = outs["multistep_bndl_outputs"]
            else:
                continue

            for b in reversed(bndl_outputs_list):
                if b is None:
                    continue

                # === Task loss (SAM loss on adversarial outputs) ===
                if "adv_outputs" in b and self.sam_loss_fn is not None and self.task_weight > 0:
                    adv_out = b["adv_outputs"]
                    result = self._compute_adv_task_loss(adv_out, device)
                    if result is not None:
                        task_loss, detailed_losses = result
                        if task_loss.requires_grad:
                            task_losses.append(task_loss)
                        # Collect detailed losses for logging (first sample only)
                        if not hasattr(self, "_aue_detailed_losses"):
                            self._aue_detailed_losses = detailed_losses

                # === BNDL loss (KL loss on adversarial outputs) ===
                if "adv_outputs" in b and self.bndl_loss_fn is not None and self.bndl_weight > 0:
                    adv_out = b["adv_outputs"]
                    adv_bndl_loss = self._compute_adv_bndl_loss(adv_out, device)
                    if adv_bndl_loss is not None and adv_bndl_loss.requires_grad:
                        bndl_losses.append(adv_bndl_loss)

                # === MMD loss (computed here, not in model) ===
                if "adv_outputs" in b and self._loss_computer is not None and self.mmd_weight > 0:
                    adv_out = b["adv_outputs"]
                    adv_mmd = self._compute_adv_mmd_loss(b, device)
                    if adv_mmd is not None and adv_mmd.requires_grad:
                        mmd_losses.append(adv_mmd)

                # === Attacker Uncertainty Loss (trains attacker to maximize uncertainty) ===
                if "adv_outputs" in b and self.attacker_uncertainty_weight > 0:
                    adv_out = b["adv_outputs"]
                    attacker_unc_loss = self._compute_attacker_uncertainty_loss(adv_out, device)
                    if attacker_unc_loss is not None and attacker_unc_loss.requires_grad:
                        attacker_uncertainty_losses.append(attacker_unc_loss)

                # === Attacker MMD Loss (trains attacker to maximize miscalibration, NO BNDL update) ===
                if "adv_outputs" in b and self.attacker_mmd_weight > 0:
                    adv_out = b["adv_outputs"]
                    attacker_mmd = self._compute_attacker_mmd_loss(b, device)
                    if attacker_mmd is not None and attacker_mmd.requires_grad:
                        attacker_mmd_losses.append(attacker_mmd)

                break  # Only use first valid bndl_outputs

        # Aggregate losses
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Track individual components for logging
        task_loss_val = torch.tensor(0.0, device=device)
        bndl_loss_val = torch.tensor(0.0, device=device)
        mmd_loss_val = torch.tensor(0.0, device=device)
        attacker_unc_loss_val = torch.tensor(0.0, device=device)

        if task_losses:
            task_loss_val = torch.stack(task_losses).mean()
            total_loss = total_loss + self.task_weight * task_loss_val

        if bndl_losses:
            bndl_loss_val = torch.stack(bndl_losses).mean()
            total_loss = total_loss + self.bndl_weight * bndl_loss_val

        if mmd_losses:
            mmd_loss_val = torch.stack(mmd_losses).mean()
            total_loss = total_loss + self.mmd_weight * mmd_loss_val

        if attacker_uncertainty_losses:
            attacker_unc_loss_val = torch.stack(attacker_uncertainty_losses).mean()
            total_loss = total_loss + self.attacker_uncertainty_weight * attacker_unc_loss_val

        # === SEPARATE OPTIMIZER DESIGN ===
        # attacker_mmd_loss is NOT added to total_loss (core_loss)
        # Instead, it's returned separately WITH gradients for the attacker optimizer
        # This ensures only attacker parameters (Style/Deform networks) are updated by this loss
        attacker_mmd_loss_val = None  # Will be tensor with gradients if available
        attacker_mmd_loss_scalar = torch.tensor(0.0, device=device)  # For logging only
        if attacker_mmd_losses:
            # Return the loss WITH gradients for separate optimizer
            attacker_mmd_loss_val = torch.stack(attacker_mmd_losses).mean()
            attacker_mmd_loss_scalar = attacker_mmd_loss_val.detach()
            # NOTE: We do NOT add to total_loss - separate optimizer handles this

        # === attacker_task_loss: Task loss only updates attacker (NO SAM update) ===
        # Same as attacker_mmd_loss - returned separately for attacker_optimizer
        attacker_task_loss_val = None
        attacker_task_loss_scalar = torch.tensor(0.0, device=device)
        if self.attacker_task_weight > 0 and task_losses:
            # Use the same task_loss but don't add to core_loss
            attacker_task_loss_val = torch.stack(task_losses).mean()
            attacker_task_loss_scalar = attacker_task_loss_val.detach()
            # NOTE: We do NOT add to total_loss - separate optimizer handles this

        # Build return dict
        result = {
            CORE_LOSS_KEY: total_loss,
            "aue_scalar": total_loss.detach(),
            "task_loss": task_loss_val.detach(),
            "bndl_loss": bndl_loss_val.detach(),
            "mmd_loss": mmd_loss_val.detach(),
            "attacker_unc_loss": attacker_unc_loss_val.detach(),
            "attacker_mmd_loss_scalar": attacker_mmd_loss_scalar,  # For logging
            "attacker_task_loss_scalar": attacker_task_loss_scalar,  # For logging
            # === Keys for separate optimizer ===
            # These tensors retain gradients and are used by attacker optimizer only
            "attacker_mmd_loss": attacker_mmd_loss_val,  # Tensor WITH gradients (or None)
            "attacker_task_loss": attacker_task_loss_val,  # Tensor WITH gradients (or None)
        }

        # Add detailed SAM losses if available
        if hasattr(self, "_aue_detailed_losses") and self._aue_detailed_losses:
            for key, val in self._aue_detailed_losses.items():
                if isinstance(val, torch.Tensor):
                    result[f"sam_{key}"] = val.detach()
                else:
                    result[f"sam_{key}"] = torch.tensor(val, device=device)
            # Clear for next iteration
            del self._aue_detailed_losses

        return result

    def _compute_adv_task_loss(self, adv_out: dict, device: torch.device) -> tuple[torch.Tensor, dict] | None:
        """Compute SAM task loss on adversarial outputs.

        Uses the SAME multi-step logic as clean branch for fair comparison:
        - Clean: sam_loss._forward() iterates over multistep_pred_multimasks_high_res
        - Adversarial: This method does the same

        Returns:
            (total_loss, detailed_losses) or None
        """
        gt = adv_out.get("gt")
        if gt is None:
            return None

        # Check for multi-step outputs (preferred, same as clean branch)
        multistep_masks = adv_out.get("multistep_pred_multimasks_high_res")
        multistep_ious = adv_out.get("multistep_pred_ious")
        multistep_obj_scores = adv_out.get("multistep_object_score_logits")

        if multistep_masks is not None and multistep_ious is not None and multistep_obj_scores is not None:
            # Use multi-step logic (same as clean branch)
            num_objects = max(float(gt.size(0)), 1.0)
            losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}

            target_masks = gt.float()  # [B, 1, H, W]
            if target_masks.dim() == 3:
                target_masks = target_masks.unsqueeze(1)

            for src_masks, ious, object_score_logits in zip(multistep_masks, multistep_ious, multistep_obj_scores):
                self.sam_loss_fn._update_losses(losses, src_masks, target_masks, ious, num_objects, object_score_logits)

            total_loss = self.sam_loss_fn.reduce_loss(losses)
            return total_loss, losses

        # Fallback: single-step (backward compatibility)
        pred_masks = adv_out.get("pred_masks")
        ious = adv_out.get("ious")
        object_score_logits = adv_out.get("object_score_logits")

        if pred_masks is None or ious is None:
            return None

        src_masks = pred_masks
        target_masks = gt.float()
        if target_masks.dim() == 3:
            target_masks = target_masks.unsqueeze(1)
        num_objects = max(float(src_masks.size(0)), 1.0)

        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        self.sam_loss_fn._update_losses(losses, src_masks, target_masks, ious, num_objects, object_score_logits)
        total_loss = self.sam_loss_fn.reduce_loss(losses)
        return total_loss, losses

    def _compute_adv_bndl_loss(self, adv_out: dict, device: torch.device) -> torch.Tensor | None:
        """Compute BNDL KL loss on adversarial outputs."""
        adv_aux = adv_out.get("aux_outputs")
        if adv_aux is None or "bndl" not in adv_aux:
            return None

        bndl_outputs = adv_aux["bndl"]
        if not isinstance(bndl_outputs, dict):
            return None

        return self.bndl_loss_fn._compute_kl_loss(bndl_outputs)

    def _compute_adv_mmd_loss(self, bndl_ns: dict, device: torch.device) -> torch.Tensor | None:
        """
        Compute adversarial MMD loss from raw data in bndl_ns.

        Uses the same logic as MMDCalibrationLoss but for adversarial samples.
        """
        from sam2.modeling.aue.loss import prepare_gt_for_loss
        from sam2.modeling.bndl_utils import BNDLOutputs

        # Get adversarial outputs
        adv_out = bndl_ns.get("adv_outputs")
        if adv_out is None:
            return None

        adv_aux = adv_out.get("aux_outputs")
        if adv_aux is None or "bndl" not in adv_aux:
            return None

        adv_bndl = adv_aux["bndl"]

        # Extract raw data
        pixel_feat = adv_bndl.get("pixel_feat_grad", adv_bndl.get("pixel_feat"))
        pixel_logits = adv_bndl.get("pixel_logits", adv_bndl.get("masks_bndl_raw"))
        pixel_gt = adv_out.get("gt")  # Adversarial GT (may be warped)
        external_w = adv_bndl.get("mask_tokens_out")

        # === EXPLICIT: Use ANALYTIC uncertainty for MMD calibration ===
        # Analytic uncertainty has gradients, enabling BNDL parameter updates
        pixel_uncertainty = adv_bndl.get("pixel_uncertainty_analytic")
        if pixel_uncertainty is None:
            # MMD calibration requires analytic uncertainty for gradient flow
            import logging

            logging.debug("[AUE MMD] pixel_uncertainty_analytic not available, skipping MMD loss")
            return None

        if pixel_feat is None or pixel_logits is None or pixel_gt is None:
            return None

        # Prepare GT
        H_feat, W_feat = int(pixel_logits.shape[1]), int(pixel_logits.shape[2])
        if pixel_gt.ndim == 3:
            pixel_gt_for_prep = pixel_gt.unsqueeze(1)
        else:
            pixel_gt_for_prep = pixel_gt
        pixel_gt_resized = prepare_gt_for_loss(pixel_gt_for_prep, (H_feat, W_feat))

        # Wrap inputs
        bndl_outputs = BNDLOutputs(
            pixel_feat=pixel_feat,
            pixel_logits=pixel_logits,
            external_w=external_w,
            pixel_uncertainty=pixel_uncertainty,  # ANALYTIC: has gradients
        )

        # Compute calibration loss using analytic uncertainty
        loss, metrics, _ = self._loss_computer.compute_calibration_loss(
            bndl_outputs=bndl_outputs,
            pixel_gt=pixel_gt_resized,
            backbone_features=None,
            use_analytic_uncertainty=False,  # Use provided pixel_uncertainty (already analytic)
            use_patches=True,
            tag="adversarial",
        )

        return loss

    def _compute_attacker_uncertainty_loss(self, adv_out: dict, device: torch.device) -> torch.Tensor | None:
        """
        Compute uncertainty-based loss for training the attacker.

        The attacker should learn to generate perturbations that MAXIMIZE model uncertainty.
        This loss returns the mean uncertainty, which will be added to total_loss.
        Since uncertainty flows through GRL in attacker networks, this effectively
        trains the attacker to maximize uncertainty.

        === DESIGN: Uses SAMPLING uncertainty (no gradients to BNDL) ===
        - Sampling uncertainty is detached (computed in torch.no_grad)
        - Gradients only flow to attacker networks (via GRL)
        - SAM's BNDL module is NOT updated by this loss
        - This ensures attacker training doesn't interfere with BNDL calibration

        Args:
            adv_out: Adversarial outputs containing BNDL predictions
            device: Computation device

        Returns:
            Mean uncertainty value (positive; attacker maximizes this via GRL)
        """
        adv_aux = adv_out.get("aux_outputs")
        if adv_aux is None or "bndl" not in adv_aux:
            return None

        bndl_outputs = adv_aux["bndl"]
        if not isinstance(bndl_outputs, dict):
            return None

        # === EXPLICIT: Use SAMPLING uncertainty for attacker training ===
        # Sampling uncertainty has NO gradients, so BNDL is not updated
        # Gradients only flow to attacker via GRL (gradient reversal layer)
        pixel_uncertainty = bndl_outputs.get("pixel_uncertainty_sampling")
        if pixel_uncertainty is None:
            # Legacy fallback to "pixel_uncertainty" (which should also be sampling)
            pixel_uncertainty = bndl_outputs.get("pixel_uncertainty")

        if pixel_uncertainty is None:
            # No uncertainty available - cannot compute attacker loss
            import logging

            logging.debug("[AUE Attacker] pixel_uncertainty_sampling not available, skipping attacker uncertainty loss")
            return None

        # Verify no gradients (sampling uncertainty should be detached)
        # This is a safety check - if uncertainty has gradients, it would incorrectly update BNDL
        if pixel_uncertainty.requires_grad:
            import logging

            logging.warning("[AUE Attacker] pixel_uncertainty has gradients! This may incorrectly update BNDL. Detaching to prevent unintended gradient flow to BNDL.")
            pixel_uncertainty = pixel_uncertainty.detach()

        # Return mean uncertainty (positive value)
        # Attacker maximizes this via GRL (gradients are reversed)
        return pixel_uncertainty.mean()

    def _compute_attacker_mmd_loss(self, bndl_ns: dict, device: torch.device) -> torch.Tensor | None:
        """
        Compute MMD-based miscalibration loss for training the attacker.

        === SEPARATE OPTIMIZER DESIGN ===
        This loss is returned separately (not added to core_loss) and used by a
        dedicated attacker optimizer. The attacker optimizer only updates
        Style/Deform network parameters, so even if gradients flow to SAM/BNDL,
        those parameters won't be updated.

        === GRADIENT FLOW ===
        miscalibration_loss → pixel_logits → mask_decoder → backbone → attacker

        The attacker optimizer's param_groups only include Style/Deform networks,
        so only those parameters receive updates. SAM/BNDL gradients are computed
        but not applied.

        Args:
            bndl_ns: BNDL namespace containing adversarial outputs
            device: Computation device

        Returns:
            MMD-like miscalibration loss value
        """
        from sam2.modeling.aue.loss import prepare_gt_for_loss

        # Get adversarial outputs
        adv_out = bndl_ns.get("adv_outputs")
        if adv_out is None:
            return None

        adv_aux = adv_out.get("aux_outputs")
        if adv_aux is None or "bndl" not in adv_aux:
            return None

        adv_bndl = adv_aux["bndl"]
        pixel_gt = adv_out.get("gt")  # Adversarial GT (may be warped)

        if pixel_gt is None:
            return None

        # === Get uncertainty (can use either sampling or analytic) ===
        # For attacker training, we use sampling uncertainty (no gradients to BNDL)
        # This ensures BNDL is not updated by this loss even without separate optimizer
        pixel_uncertainty = adv_bndl.get("pixel_uncertainty_sampling")
        if pixel_uncertainty is None:
            pixel_uncertainty = adv_bndl.get("pixel_uncertainty")
        if pixel_uncertainty is None:
            return None

        # Detach uncertainty to prevent BNDL updates (extra safety)
        if pixel_uncertainty.requires_grad:
            pixel_uncertainty = pixel_uncertainty.detach()

        # === Get pixel_logits (keep gradients for attacker) ===
        # With separate optimizer, we don't need to detach - only attacker params are updated
        pixel_logits = adv_bndl.get("pixel_logits", adv_bndl.get("masks_bndl_raw"))
        if pixel_logits is None:
            return None

        # Prepare GT
        H_feat, W_feat = int(pixel_logits.shape[1]), int(pixel_logits.shape[2])
        if pixel_gt.ndim == 3:
            pixel_gt_for_prep = pixel_gt.unsqueeze(1)
        else:
            pixel_gt_for_prep = pixel_gt
        pixel_gt_resized = prepare_gt_for_loss(pixel_gt_for_prep, (H_feat, W_feat))

        # === Compute prediction error with gradients ===
        # Gradients flow: error → sigmoid → logits → decoder → backbone → attacker
        if pixel_logits.ndim == 4 and pixel_logits.shape[-1] >= 1:
            logits_val = pixel_logits.max(dim=-1).values
        elif pixel_logits.ndim == 3:
            logits_val = pixel_logits
        else:
            logits_val = pixel_logits.view(pixel_logits.shape[0], H_feat, W_feat, -1).max(dim=-1).values

        pred_prob = torch.sigmoid(logits_val)
        error = torch.abs(pred_prob - pixel_gt_resized.float().detach())  # GT is detached

        # === Average uncertainty over masks to get [B, H, W] ===
        if pixel_uncertainty.ndim == 4:  # [B, H, W, K]
            uncertainty = pixel_uncertainty.mean(dim=-1)  # [B, H, W]
        else:
            uncertainty = pixel_uncertainty

        # Resize if needed
        if uncertainty.shape[1:] != error.shape[1:]:
            uncertainty = torch.nn.functional.interpolate(
                uncertainty.unsqueeze(1),
                size=error.shape[1:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        # === Compute miscalibration loss ===
        # MSE between uncertainty and error measures calibration quality
        # Attacker learns to maximize this (make predictions where uncertainty != error)
        miscalibration_loss = torch.nn.functional.mse_loss(
            uncertainty.clamp(0.0, 1.0).detach(),  # target (uncertainty) is detached
            error.clamp(0.0, 1.0),  # prediction (error) has gradients to attacker
            reduction="mean",
        )

        return miscalibration_loss
