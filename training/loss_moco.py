import torch
import torch.nn as nn

from training.trainer import CORE_LOSS_KEY


class MoCoLoss(nn.Module):
    """
    Read MoCo auxiliary loss prepared by the model in outs_batch and expose
    it as a standard loss dict so it can be combined by a combined loss.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)

    def forward(self, outs_batch: list[dict], targets_batch: torch.Tensor | None = None):
        device = None
        total = 0.0
        count = 0
        for outs in outs_batch:
            bndl_list = outs.get("multistep_bndl_outputs")
            if not bndl_list:
                continue
            for b in reversed(bndl_list):
                if b is not None and ("moco_aux_loss" in b):
                    if device is None:
                        device = b["moco_aux_loss"].device
                    total = total + b["moco_aux_loss"]
                    count += 1
                    break
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        core = (total / count) if count > 0 else torch.tensor(0.0, device=device, requires_grad=True)
        return {CORE_LOSS_KEY: core, "moco_scalar": core.detach()}


