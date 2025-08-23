import torch
import torch.nn as nn

from training.trainer import CORE_LOSS_KEY


class CombinedSAMBNDLLoss(nn.Module):
    """
    combine sam and bndl loss
    """

    def __init__(self, sam_loss, bndl_loss, sam_weight=1.0, bndl_weight=1.0):
        super().__init__()
        self.sam_loss = sam_loss
        self.bndl_loss = bndl_loss
        self.sam_weight = sam_weight
        self.bndl_weight = bndl_weight

    def forward(self, outs_batch: list[dict], targets_batch: torch.Tensor):
        # compute sam and bndl loss separately
        sam_losses = self.sam_loss(outs_batch, targets_batch)
        bndl_losses = self.bndl_loss(outs_batch, targets_batch)

        # merge loss
        combined_losses = {}

        # add sam loss (add prefix to distinguish)
        for k, v in sam_losses.items():
            if k == CORE_LOSS_KEY:
                combined_losses["core_loss"] = v * self.sam_weight
            else:
                combined_losses[f"{k}"] = v

        # add bndl loss (add prefix to distinguish)
        for k, v in bndl_losses.items():
            if k == CORE_LOSS_KEY:
                combined_losses["bndl_core_loss"] = v * self.bndl_weight
            else:
                combined_losses[f"bndl_{k}"] = v

        # compute total core loss
        combined_losses[CORE_LOSS_KEY] = combined_losses["core_loss"] + combined_losses["bndl_core_loss"]

        return combined_losses
