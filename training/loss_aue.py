# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch
import torch.nn as nn

from training.trainer import CORE_LOSS_KEY


class AUELoss(nn.Module):
    """
    Read AUE (Adversarial Uncertainty Estimation) auxiliary loss prepared by the model in outs_batch and expose
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
            # Support both new and old key names (like BNDLLoss does)
            if "multistep_aux_outputs" in outs:
                aux_list = outs["multistep_aux_outputs"]
                bndl_outputs_list = [aux.get("bndl") if isinstance(aux, dict) else None for aux in aux_list]
            elif "multistep_bndl_outputs" in outs:
                # Backward compatibility with old key name
                bndl_outputs_list = outs["multistep_bndl_outputs"]
            else:
                continue
            
            for b in reversed(bndl_outputs_list):
                if b is not None and ("aue_aux_loss" in b):
                    if device is None:
                        device = b["aue_aux_loss"].device
                    total = total + b["aue_aux_loss"]  # no weighting here
                    count += 1
                    break
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        core = (total / count) if count > 0 else torch.tensor(0.0, device=device, requires_grad=True)
        return {CORE_LOSS_KEY: core, "aue_scalar": core.detach()}


