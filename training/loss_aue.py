# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
AUE (Adversarial Uncertainty Estimation) Loss.

Computes losses on adversarial samples (task loss, BNDL KL, MMD/L_cal calibration).
All terms enter the single core_loss; backward through GRL in Style/Deform
networks flips the sign for attacker params, giving joint min-max in one pass.
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
        cal_weight: float = 0.0,  # λ from paper Eq. 5; coefficient on L_cal_sym in core_loss
        dual_sg_l2_weight: float = 1.0,  # Multiplier on L2 mirror term inside L_cal_sym (asymmetric Dual-SG)
        mmd_config: dict | None = None,
        sam_loss_fn: nn.Module | None = None,
        bndl_loss_fn: nn.Module | None = None,
    ):
        """
        Args:
            task_weight: Weight for SAM task loss on adv samples (updates SAM + attacker via GRL)
            bndl_weight: Weight for BNDL KL loss on adv samples (updates BNDL + attacker via GRL)
            mmd_weight: Weight for MMD calibration loss on adv samples (updates BNDL + attacker via GRL)
            cal_weight: Weight λ on L_cal_sym (paper Eq. 5); updates decoder/BNDL via L1/L2 paths, attacker via GRL
            dual_sg_l2_weight: Multiplier on L2 mirror term inside L_cal_sym
            mmd_config: Distribution matching config for adversarial MMD
            sam_loss_fn: Reference to SAM loss (injected by CombinedLoss)
            bndl_loss_fn: Reference to BNDL loss (injected by CombinedLoss)
        """
        super().__init__()
        self.task_weight = task_weight
        self.bndl_weight = bndl_weight
        self.mmd_weight = mmd_weight
        self.cal_weight = cal_weight
        self.dual_sg_l2_weight = dual_sg_l2_weight
        self.sam_loss_fn = sam_loss_fn
        self.bndl_loss_fn = bndl_loss_fn

        # Initialize MMD components from config
        self._loss_computer = None
        self._distribution_matcher = None

        if mmd_config and mmd_weight > 0:
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
        cal_losses = []

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

                # === Calibration Loss L_cal_sym (paper Eq. 5 + dual-SG L2 mirror) ===
                # Routed through GRL in Style/Deform networks: attacker maximizes L_cal,
                # decoder/BNDL minimize it (joint min-max via single backward).
                if "adv_outputs" in b and self.cal_weight > 0:
                    cal_loss = self._compute_calibration_loss(b, device)
                    if cal_loss is not None and cal_loss.requires_grad:
                        cal_losses.append(cal_loss)

                break  # Only use first valid bndl_outputs

        # Aggregate losses into a single core loss (paper Eq. 7 joint min-max via GRL).
        # All terms go through a single backward pass; GRL in Style/Deform networks
        # flips the sign for attacker params, so decoder/BNDL minimize and attackers
        # maximize the same scalar.
        total_loss = torch.tensor(0.0, device=device)

        task_loss_val = torch.tensor(0.0, device=device)
        bndl_loss_val = torch.tensor(0.0, device=device)
        mmd_loss_val = torch.tensor(0.0, device=device)
        cal_loss_val = torch.tensor(0.0, device=device)

        if task_losses:
            task_loss_val = torch.stack(task_losses).mean()
            total_loss = total_loss + self.task_weight * task_loss_val

        if bndl_losses:
            bndl_loss_val = torch.stack(bndl_losses).mean()
            total_loss = total_loss + self.bndl_weight * bndl_loss_val

        if mmd_losses:
            mmd_loss_val = torch.stack(mmd_losses).mean()
            total_loss = total_loss + self.mmd_weight * mmd_loss_val

        if cal_losses:
            cal_loss_val = torch.stack(cal_losses).mean()
            total_loss = total_loss + self.cal_weight * cal_loss_val

        result = {
            CORE_LOSS_KEY: total_loss,
            "aue_scalar": total_loss.detach(),
            "task_loss": task_loss_val.detach(),
            "bndl_loss": bndl_loss_val.detach(),
            "mmd_loss": mmd_loss_val.detach(),
            "cal_loss": cal_loss_val.detach(),
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

    def _compute_calibration_loss(self, bndl_ns: dict, device: torch.device) -> torch.Tensor | None:
        """
        Compute symmetric calibration loss L_cal_sym (Dual-SG extension of paper Eq. 5).

            L1 = e · exp(-sg[u]) + (1 - e) · exp(sg[u])         (paper Eq. 5, sampling u)
            L2 = sg[e] · exp(-u) + (1 - sg[e]) · exp(u)         (symmetric mirror, analytic u)
            L_cal_sym = (L1 + L2).mean()

        Motivation: original Eq. 5 has ∂L/∂e = exp(-u) - exp(u) ≈ 0 at u≈0, so the
        attacker (via GRL) has no gradient signal in the confident region and cannot
        drive the confident-wrong failure mode. L2 restores a ∂/∂u channel via the
        analytic uncertainty, symmetrically covering both UC and CW failure modes.

        === JOINT MIN-MAX (paper Eq. 7) ===
        Added to core_loss with weight ``cal_weight`` (= λ). Single backward pass:
        decoder/BNDL minimize L_cal_sym via the L1 (e) and L2 (u_analytic) gradient
        channels; Style/Deform attackers receive GRL-flipped gradients and so
        maximize the same scalar. No separate optimizer, no snapshot/restore.

        === GRADIENT FLOW ===
        L1 → e → sigmoid → pixel_logits → decoder → backbone → I^adv → GRL → attacker
        L2 → u_analytic → BNDL hyper_in → decoder → backbone → I^adv → GRL → attacker

        Args:
            bndl_ns: BNDL namespace containing adversarial outputs
            device: Computation device

        Returns:
            L_cal_sym scalar tensor (mean over pixels).
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

        # === Get uncertainty: sampling (no grad) for L1, analytic (with grad) for L2 ===
        # L1 (paper Eq.5) needs sg[u] => sampling is fine (already gradient-free).
        # L2 (mirror term) needs gradient on u to drive attacker via the u-channel
        # u_analytic -> BNDL hyper_in -> decoder -> backbone -> I^adv -> GRL -> attacker.
        # BNDL contamination from L2 is wiped by trainer.py snapshot/restore (4234079).
        pixel_u_sample = adv_bndl.get("pixel_uncertainty_sampling")
        if pixel_u_sample is None:
            pixel_u_sample = adv_bndl.get("pixel_uncertainty")
        if pixel_u_sample is None:
            return None
        if pixel_u_sample.requires_grad:
            pixel_u_sample = pixel_u_sample.detach()
        pixel_u_analy = adv_bndl.get("pixel_uncertainty_analytic")  # may be None in eval

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

        # === Average uncertainty over masks to get [B, H, W] (both sampling & analytic) ===
        def _reduce_and_resize(u, target_shape):
            if u is None:
                return None
            if u.ndim == 4:
                u = u.mean(dim=-1)
            if u.shape[1:] != target_shape:
                u = torch.nn.functional.interpolate(
                    u.unsqueeze(1), size=target_shape, mode="bilinear", align_corners=False
                ).squeeze(1)
            return u

        u_sample = _reduce_and_resize(pixel_u_sample, error.shape[1:])
        u_analy = _reduce_and_resize(pixel_u_analy, error.shape[1:])

        # === L1: paper Eq.5, sg[u] (sampling u, no grad) ===
        # L1 = e · exp(-sg[u]) + (1 - e) · exp(sg[u])
        u_sg = u_sample.clamp(0.0, 1.0).detach()
        e_clamp = error.clamp(0.0, 1.0)
        loss_e = e_clamp * torch.exp(-u_sg) + (1.0 - e_clamp) * torch.exp(u_sg)

        # === L2: symmetric mirror, sg[e] (analytic u, with grad) ===
        # L2 = sg[e] · exp(-u) + (1 - sg[e]) · exp(u)
        # Drives ∂/∂u channel: at confident-wrong (e=1, u≈0), ∂L2/∂u = -1, so attacker
        # (via GRL) is pushed to keep u low while raising e -> CW failure mode covered.
        if u_analy is not None:
            assert u_analy.requires_grad, "L2: u_analytic must carry gradient to drive attacker"
            assert u_analy.shape == u_sample.shape, (
                f"L2: u shape mismatch analytic={u_analy.shape} vs sampling={u_sample.shape}"
            )
            e_sg = e_clamp.detach()
            u_clamp = u_analy.clamp(0.0, 1.0)
            loss_u = e_sg * torch.exp(-u_clamp) + (1.0 - e_sg) * torch.exp(u_clamp)
            return (loss_e + self.dual_sg_l2_weight * loss_u).mean()

        # Fallback (eval-time or analytic missing): plain Eq.5 only
        return loss_e.mean()
