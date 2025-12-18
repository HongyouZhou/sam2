import logging
import torch
import torch.nn as nn

from training.trainer import CORE_LOSS_KEY


class CombinedSAMBNDLLoss(nn.Module):
    """
    combine sam and bndl loss
    """

    def __init__(
        self,
        sam_loss,
        bndl_loss,
        ur_ern_loss=None,
        aue_loss=None,
        sam_weight=1.0,
        bndl_weight=1.0,
        ur_ern_weight=1.0,
        aue_weight=1.0,
        weight_schedule: list[dict] | None = None,
        prefix_keys: bool = True,  # New: control whether to add prefix to keys
    ):
        super().__init__()
        self.sam_loss = sam_loss
        self.bndl_loss = bndl_loss
        self.ur_ern_loss = ur_ern_loss
        self.aue_loss = aue_loss
        self.sam_weight = sam_weight
        self.bndl_weight = bndl_weight
        self.ur_ern_weight = ur_ern_weight
        self.aue_weight = aue_weight
        self.prefix_keys = prefix_keys
        self._dbg_once = False
        self._initial_weights = {
            "sam_weight": sam_weight,
            "bndl_weight": bndl_weight,
            "ur_ern_weight": ur_ern_weight,
            "aue_weight": aue_weight,
        }
        self._weight_schedule = self._prepare_weight_schedule(weight_schedule)
        if self._weight_schedule:
            # Ensure weights match the first stage for epoch 0
            self.apply_schedule(0)

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
        
        aue_losses = None
        if self.aue_loss is not None:
            aue_losses = self.aue_loss(outs_batch, targets_batch)

        # merge loss
        combined_losses = {}

        # add sam loss
        for k, v in sam_losses.items():
            if k == CORE_LOSS_KEY:
                # Always add sam_core_loss for tracking
                combined_losses["sam_core_loss"] = v * self.sam_weight
                # If prefix_keys is False (e.g. legacy validation), we also keep the original "core_loss" key pointing to SAM loss
                # This will be overwritten later by total sum if we are strictly following combined logic,
                # but if weights are (1, 0, 0), it matches.
            else:
                key_name = f"sam_{k}" if self.prefix_keys else k
                combined_losses[key_name] = v

        # add bndl loss
        if bndl_losses is not None:
            for k, v in bndl_losses.items():
                if k == CORE_LOSS_KEY:
                    combined_losses["bndl_core_loss"] = v * self.bndl_weight
                else:
                    # BNDL keys usually don't conflict with SAM keys, but good to be safe
                    # If prefix_keys=True, use bndl_ prefix. Else keep original (e.g. kl_divergence)
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

        # add aue loss (optional)
        if aue_losses is not None:
            for k, v in aue_losses.items():
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
        if "aue_core_loss" in combined_losses and self.aue_weight > 0:
            core = core + combined_losses["aue_core_loss"]
            
        combined_losses[CORE_LOSS_KEY] = core

        return combined_losses
