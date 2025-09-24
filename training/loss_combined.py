import torch
import torch.nn as nn

from training.trainer import CORE_LOSS_KEY


class CombinedSAMBNDLLoss(nn.Module):
    """
    combine sam and bndl loss
    """

    def __init__(self, sam_loss, bndl_loss, adco_loss=None, proco_loss=None, moco_loss=None, sam_weight=1.0, bndl_weight=1.0, adco_weight=1.0, proco_weight=1.0, moco_weight=1.0):
        super().__init__()
        self.sam_loss = sam_loss
        self.bndl_loss = bndl_loss
        self.adco_loss = adco_loss
        self.proco_loss = proco_loss
        self.moco_loss = moco_loss
        self.sam_weight = sam_weight
        self.bndl_weight = bndl_weight
        self.adco_weight = adco_weight
        self.proco_weight = proco_weight
        self.moco_weight = moco_weight

    def forward(self, outs_batch: list[dict], targets_batch: torch.Tensor):
        # compute sam and bndl loss separately
        sam_losses = self.sam_loss(outs_batch, targets_batch)
        bndl_losses = self.bndl_loss(outs_batch, targets_batch)
        adco_losses = None
        proco_losses = None
        moco_losses = None
        if self.adco_loss is not None:
            adco_losses = self.adco_loss(outs_batch, targets_batch)
        if self.proco_loss is not None:
            proco_losses = self.proco_loss(outs_batch, targets_batch)
        if self.moco_loss is not None:
            moco_losses = self.moco_loss(outs_batch, targets_batch)

        # merge loss
        combined_losses = {}

        # add sam loss (add prefix to distinguish)
        for k, v in sam_losses.items():
            if k == CORE_LOSS_KEY:
                combined_losses["sam_core_loss"] = v * self.sam_weight
            else:
                combined_losses[f"sam_{k}"] = v

        # add bndl loss (add prefix to distinguish)
        for k, v in bndl_losses.items():
            if k == CORE_LOSS_KEY:
                combined_losses["bndl_core_loss"] = v * self.bndl_weight
            else:
                combined_losses[f"bndl_{k}"] = v

        # add adco loss (optional)
        if adco_losses is not None:
            for k, v in adco_losses.items():
                if k == CORE_LOSS_KEY:
                    combined_losses["adco_core_loss"] = v * self.adco_weight
                else:
                    combined_losses[f"adco_{k}"] = v

        # add proco loss (optional)
        if proco_losses is not None:
            for k, v in proco_losses.items():
                if k == CORE_LOSS_KEY:
                    combined_losses["proco_core_loss"] = v * self.proco_weight
                else:
                    combined_losses[f"proco_{k}"] = v

        # add moco loss (optional)
        if moco_losses is not None:
            for k, v in moco_losses.items():
                if k == CORE_LOSS_KEY:
                    combined_losses["moco_core_loss"] = v * self.moco_weight
                else:
                    combined_losses[f"moco_{k}"] = v

        # compute total core loss
        core = combined_losses["sam_core_loss"] + combined_losses["bndl_core_loss"]
        if "adco_core_loss" in combined_losses:
            core = core + combined_losses["adco_core_loss"]
        if "proco_core_loss" in combined_losses:
            core = core + combined_losses["proco_core_loss"]
        if "moco_core_loss" in combined_losses:
            core = core + combined_losses["moco_core_loss"]
        combined_losses[CORE_LOSS_KEY] = core

        return combined_losses
