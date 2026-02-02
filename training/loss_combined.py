import torch
import torch.nn as nn

from training.trainer import CORE_LOSS_KEY


class CombinedSAMBNDLLoss(nn.Module):
    """
    Combined loss for SAM training with BNDL, MMD calibration, and AUE.

    Components:
    - sam_loss: Segmentation task loss (focal + dice + iou)
    - bndl_loss: BNDL KL divergence loss
    - mmd_loss: MMD calibration loss (NEW: separated from AUE)
    - aue_loss: Adversarial task/BNDL loss on adv samples (attacker via GRL)
    """

    def __init__(
        self,
        sam_loss,
        bndl_loss,
        ur_ern_loss=None,
        mmd_loss=None,  # NEW: Separated MMD calibration loss
        aue_loss=None,
        sam_weight=1.0,
        bndl_weight=1.0,
        ur_ern_weight=1.0,
        mmd_weight=1.0,  # NEW: Weight for MMD loss
        aue_weight=1.0,
        weight_schedule: list[dict] | None = None,
        prefix_keys: bool = True,
    ):
        super().__init__()
        self.sam_loss = sam_loss
        self.bndl_loss = bndl_loss
        self.ur_ern_loss = ur_ern_loss
        self.mmd_loss = mmd_loss
        self.aue_loss = aue_loss
        self.sam_weight = sam_weight
        self.bndl_weight = bndl_weight
        self.ur_ern_weight = ur_ern_weight
        self.mmd_weight = mmd_weight
        self.aue_weight = aue_weight
        self.prefix_keys = prefix_keys
        self._dbg_once = False
        self._initial_weights = {
            "sam_weight": sam_weight,
            "bndl_weight": bndl_weight,
            "ur_ern_weight": ur_ern_weight,
            "mmd_weight": mmd_weight,
            "aue_weight": aue_weight,
        }
        self._weight_schedule = self._prepare_weight_schedule(weight_schedule)
        if self._weight_schedule:
            # Ensure weights match the first stage for epoch 0
            self.apply_schedule(0)

        # Inject loss functions into AUELoss for reuse
        if self.aue_loss is not None:
            self.aue_loss.sam_loss_fn = sam_loss
            self.aue_loss.bndl_loss_fn = bndl_loss

    def _prepare_weight_schedule(self, schedule_cfg: list[dict] | None) -> list[dict]:
        if not schedule_cfg:
            return []
        prepared = []
        for stage in schedule_cfg:
            if not isinstance(stage, dict):
                raise TypeError(f"weight_schedule stage must be a dict, got {type(stage)}")
            stage_copy = dict(stage)
            until_epoch = stage_copy.pop("until_epoch", None)
            if until_epoch is not None:
                until_epoch = int(until_epoch)
            stage_copy["until_epoch"] = until_epoch
            prepared.append(stage_copy)
        prepared.sort(key=lambda s: float("inf") if s["until_epoch"] is None else s["until_epoch"])
        return prepared

    def _apply_stage_weights(self, stage: dict) -> None:
        self.sam_weight = stage.get("sam_weight", self._initial_weights["sam_weight"])
        self.bndl_weight = stage.get("bndl_weight", self._initial_weights["bndl_weight"])
        self.ur_ern_weight = stage.get("ur_ern_weight", self._initial_weights["ur_ern_weight"])
        self.mmd_weight = stage.get("mmd_weight", self._initial_weights["mmd_weight"])
        self.aue_weight = stage.get("aue_weight", self._initial_weights["aue_weight"])

    def apply_schedule(self, epoch: int) -> None:
        if not self._weight_schedule:
            return
        for stage in self._weight_schedule:
            limit = stage.get("until_epoch")
            if limit is None or epoch <= limit:
                self._apply_stage_weights(stage)
                return
        # If all stages had a finite limit smaller than epoch, fall back to the last stage
        self._apply_stage_weights(self._weight_schedule[-1])

    def forward(self, outs_batch: list[dict], targets_batch: torch.Tensor):
        # compute sam loss (always required)
        sam_losses = self.sam_loss(outs_batch, targets_batch)

        # compute optional losses
        # We compute them if the module exists, even if weight is 0 (for logging purposes)
        # But we only add to core_loss if weight > 0
        bndl_losses = None
        if self.bndl_loss is not None:
            bndl_losses = self.bndl_loss(outs_batch, targets_batch)

        ur_ern_losses = None
        if self.ur_ern_loss is not None:
            ur_ern_losses = self.ur_ern_loss(outs_batch, targets_batch)

        # NEW: MMD calibration loss (independent from AUE)
        # Only compute if module exists AND weight > 0 (skip unnecessary computation)
        mmd_losses = None
        if self.mmd_loss is not None and self.mmd_weight > 0:
            mmd_losses = self.mmd_loss(outs_batch, targets_batch)

        aue_losses = None
        # Only compute AUE loss when weight > 0 (forward pass is also skipped by model)
        if self.aue_loss is not None and self.aue_weight > 0:
            aue_losses = self.aue_loss(outs_batch, targets_batch)

            # === Gradient Flow Debug (one-time check) ===
            if not self._dbg_once and aue_losses is not None:
                aue_core = aue_losses.get(CORE_LOSS_KEY)
                if aue_core is not None:
                    has_grad = aue_core.requires_grad
                    grad_fn = aue_core.grad_fn
                    import logging

                    logging.info(f"[AUE Gradient Check] aue_core_loss.requires_grad={has_grad}, grad_fn={type(grad_fn).__name__ if grad_fn else None}")
                    self._dbg_once = True

        # merge loss
        combined_losses = {}

        # add sam loss
        for k, v in sam_losses.items():
            if k == CORE_LOSS_KEY:
                # Always add sam_core_loss for tracking
                combined_losses["sam_core_loss"] = v * self.sam_weight
            else:
                key_name = f"sam_{k}" if self.prefix_keys else k
                combined_losses[key_name] = v

        # add bndl loss
        if bndl_losses is not None:
            for k, v in bndl_losses.items():
                if k == CORE_LOSS_KEY:
                    combined_losses["bndl_core_loss"] = v * self.bndl_weight
                else:
                    key_name = f"bndl_{k}" if self.prefix_keys else k
                    combined_losses[key_name] = v
        else:
            # BNDL disabled: add zero placeholder if prefixing is on (for consistency)
            if self.prefix_keys:
                combined_losses["bndl_core_loss"] = torch.tensor(0.0, device=sam_losses[CORE_LOSS_KEY].device, requires_grad=False)

        # add ur_ern loss (optional)
        if ur_ern_losses is not None:
            for k, v in ur_ern_losses.items():
                if k == CORE_LOSS_KEY:
                    combined_losses["ur_ern_core_loss"] = v * self.ur_ern_weight
                else:
                    key_name = f"ur_ern_{k}" if self.prefix_keys else k
                    combined_losses[key_name] = v

        # NEW: add mmd loss (independent calibration loss)
        if mmd_losses is not None:
            for k, v in mmd_losses.items():
                if k == CORE_LOSS_KEY:
                    combined_losses["mmd_core_loss"] = v * self.mmd_weight
                else:
                    key_name = f"mmd_{k}" if self.prefix_keys else k
                    combined_losses[key_name] = v

        # add aue loss (adversarial task/BNDL loss, affects attacker via GRL)
        if aue_losses is not None:
            for k, v in aue_losses.items():
                if v is None:
                    continue  # Skip None values (e.g., attacker_mmd_loss when not computed)
                if k == CORE_LOSS_KEY:
                    combined_losses["aue_core_loss"] = v * self.aue_weight
                else:
                    key_name = f"aue_{k}" if self.prefix_keys else k
                    combined_losses[key_name] = v

        # compute total core loss
        # Start with sam component
        core = combined_losses["sam_core_loss"]

        # Add other components if their weights are > 0
        if "bndl_core_loss" in combined_losses and self.bndl_weight > 0:
            core = core + combined_losses["bndl_core_loss"]
        if "ur_ern_core_loss" in combined_losses and self.ur_ern_weight > 0:
            core = core + combined_losses["ur_ern_core_loss"]
        if "mmd_core_loss" in combined_losses and self.mmd_weight > 0:
            core = core + combined_losses["mmd_core_loss"]
        if "aue_core_loss" in combined_losses and self.aue_weight > 0:
            core = core + combined_losses["aue_core_loss"]

        combined_losses[CORE_LOSS_KEY] = core

        # === Attacker-only loss for separate optimizer ===
        # This loss is NOT added to core_loss. It's used by a separate attacker optimizer.
        # It allows training the attacker (Style/Deform networks) without affecting SAM/BNDL.
        attacker_only_losses = []
        if aue_losses is not None:
            # attacker_mmd_loss trains attacker to maximize miscalibration
            if "attacker_mmd_loss" in aue_losses and aue_losses["attacker_mmd_loss"] is not None:
                attacker_mmd = aue_losses["attacker_mmd_loss"]
                if torch.is_tensor(attacker_mmd) and attacker_mmd.requires_grad:
                    # Get weight from aue_loss module
                    attacker_mmd_weight = getattr(self.aue_loss, "attacker_mmd_weight", 1.0)
                    attacker_only_losses.append(attacker_mmd_weight * attacker_mmd)
                    combined_losses["_attacker_mmd_loss_for_optim"] = attacker_mmd.detach()

            # attacker_task_loss trains attacker with task loss (NO SAM update)
            if "attacker_task_loss" in aue_losses and aue_losses["attacker_task_loss"] is not None:
                attacker_task = aue_losses["attacker_task_loss"]
                if torch.is_tensor(attacker_task) and attacker_task.requires_grad:
                    attacker_task_weight = getattr(self.aue_loss, "attacker_task_weight", 1.0)
                    attacker_only_losses.append(attacker_task_weight * attacker_task)
                    combined_losses["_attacker_task_loss_for_optim"] = attacker_task.detach()

        if attacker_only_losses:
            combined_losses["_attacker_only_loss"] = sum(attacker_only_losses)
        else:
            # Zero placeholder if no attacker loss
            combined_losses["_attacker_only_loss"] = torch.tensor(0.0, device=core.device, requires_grad=False)

        # === Clean vs Adversarial comparison metrics ===
        # For monitoring adversarial training dynamics

        # Task loss comparison
        if "sam_core_loss" in combined_losses:
            combined_losses["clean_task_loss"] = combined_losses["sam_core_loss"].detach() / max(self.sam_weight, 1e-8)

        if aue_losses is not None and "task_loss" in aue_losses:
            combined_losses["adv_task_loss"] = aue_losses["task_loss"]
            # Ratio: adv/clean > 1 means attacker is effective
            if "clean_task_loss" in combined_losses and combined_losses["clean_task_loss"] > 1e-6:
                combined_losses["adv_clean_task_ratio"] = aue_losses["task_loss"] / combined_losses["clean_task_loss"]

        # BNDL loss comparison
        if bndl_losses is not None and CORE_LOSS_KEY in bndl_losses:
            combined_losses["clean_bndl_loss"] = bndl_losses[CORE_LOSS_KEY].detach()

        if aue_losses is not None and "bndl_loss" in aue_losses:
            combined_losses["adv_bndl_loss"] = aue_losses["bndl_loss"]
            # Ratio: adv/clean > 1 means attacker increases BNDL uncertainty
            if "clean_bndl_loss" in combined_losses and combined_losses["clean_bndl_loss"] > 1e-10:
                combined_losses["adv_clean_bndl_ratio"] = aue_losses["bndl_loss"] / combined_losses["clean_bndl_loss"]

        return combined_losses
