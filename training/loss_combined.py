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
        
        # compute optional losses only if weight > 0 to avoid KeyError when components are disabled
        bndl_losses = None
        if self.bndl_weight > 0.0 and self.bndl_loss is not None:
            bndl_losses = self.bndl_loss(outs_batch, targets_batch)
        
        ur_ern_losses = None
        if self.ur_ern_weight > 0.0 and self.ur_ern_loss is not None:
            ur_ern_losses = self.ur_ern_loss(outs_batch, targets_batch)
        
        aue_losses = None
        if self.aue_weight > 0.0 and self.aue_loss is not None:
            aue_losses = self.aue_loss(outs_batch, targets_batch)

        # merge loss
        combined_losses = {}

        # add sam loss (add prefix to distinguish)
        for k, v in sam_losses.items():
            if k == CORE_LOSS_KEY:
                combined_losses["sam_core_loss"] = v * self.sam_weight
            else:
                combined_losses[f"sam_{k}"] = v

        # add bndl loss (add prefix to distinguish)
        if bndl_losses is not None:
            for k, v in bndl_losses.items():
                if k == CORE_LOSS_KEY:
                    combined_losses["bndl_core_loss"] = v * self.bndl_weight
                else:
                    combined_losses[f"bndl_{k}"] = v
        else:
            # BNDL disabled: add zero placeholder
            combined_losses["bndl_core_loss"] = torch.tensor(0.0, device=sam_losses[CORE_LOSS_KEY].device, requires_grad=False)

        # add ur_ern loss (optional)
        if ur_ern_losses is not None:
            for k, v in ur_ern_losses.items():
                if k == CORE_LOSS_KEY:
                    combined_losses["ur_ern_core_loss"] = v * self.ur_ern_weight
                else:
                    combined_losses[f"ur_ern_{k}"] = v

        # add aue loss (optional)
        if aue_losses is not None:
            for k, v in aue_losses.items():
                if k == CORE_LOSS_KEY:
                    combined_losses["aue_core_loss"] = v * self.aue_weight
                else:
                    combined_losses[f"aue_{k}"] = v

        # compute total core loss
        core = combined_losses["sam_core_loss"] + combined_losses["bndl_core_loss"]
        if "ur_ern_core_loss" in combined_losses:
            core = core + combined_losses["ur_ern_core_loss"]
        if "aue_core_loss" in combined_losses:
            core = core + combined_losses["aue_core_loss"]
        combined_losses[CORE_LOSS_KEY] = core

        return combined_losses
