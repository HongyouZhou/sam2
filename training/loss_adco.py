# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch
import torch.nn as nn

from training.trainer import CORE_LOSS_KEY


class AdCoLoss(nn.Module):
    """
    Read AdCo auxiliary loss prepared by the model in outs_batch and expose
    it as a standard loss dict so it can be combined by a combined loss.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor | None = None):
        # Determine device consistently to avoid CPU/GPU add mismatch
        device = None
        if targets_batch is not None:
            device = targets_batch.device
        total = 0.0
        count = 0
        for outs in outs_batch:
            bndl_list = outs.get("multistep_bndl_outputs")
            if not bndl_list:
                continue
            for b in reversed(bndl_list):
                if b is not None and ("adco_aux_loss" in b):
                    if device is None:
                        device = b["adco_aux_loss"].device
                    total = total + b["adco_aux_loss"]  # no weighting here
                    count += 1
                    break
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        core = (total / count) if count > 0 else torch.tensor(0.0, device=device, requires_grad=True)
        return {CORE_LOSS_KEY: core, "adco_scalar": core.detach()}


