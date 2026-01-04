# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# typing imports removed

import torch
import torch.nn as nn

from training.trainer import CORE_LOSS_KEY


class AUELoss(nn.Module):
    """
    Read AUE (Adversarial Uncertainty Estimation) auxiliary loss prepared by the model in outs_batch and expose
    it as a standard loss dict so it can be combined by a combined loss.
    """

    def __init__(self, clean_weight: float = 1.0, adv_weight: float = 1.0):
        super().__init__()
        self.clean_weight = clean_weight
        self.adv_weight = adv_weight

    def forward(self, outs_batch: list[dict], targets_batch: torch.Tensor | None = None):
        # Determine device consistently to avoid CPU/GPU add mismatch
        device = targets_batch.device if targets_batch is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        valid_losses = []
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
                if b is None:
                    continue

                # Option 1: Use separated losses (allows re-weighting here)
                if "aue_loss_clean" in b and "aue_loss_adv" in b:
                    clean_loss = b["aue_loss_clean"]
                    adv_loss = b["aue_loss_adv"]
                    
                    # Skip if no gradients (meaning no valid loss computed)
                    # Note: We check if EITHER has gradients, as one might be zero/detached
                    if not (clean_loss.requires_grad or adv_loss.requires_grad):
                        continue
                        
                    # Combine with configured weights
                    total_loss = self.clean_weight * clean_loss + self.adv_weight * adv_loss
                    valid_losses.append(total_loss)
                    break

                # Option 2: Fallback to pre-computed total loss
                elif "aue_aux_loss" in b:
                    aue_loss_value = b["aue_aux_loss"]
                    # Skip zero losses (from frames without pixel_gt)
                    # We want the first NON-ZERO loss with gradients
                    # Removed .item() == 0.0 check to avoid CPU synchronization
                    if not aue_loss_value.requires_grad:
                        continue  # Keep searching
                    
                    valid_losses.append(aue_loss_value)
                    break
        
        if len(valid_losses) > 0:
            core = torch.stack(valid_losses).mean()
        else:
            core = torch.tensor(0.0, device=device, requires_grad=True)
        return {CORE_LOSS_KEY: core, "aue_scalar": core.detach()}


