import torch
import torch.nn as nn

from training.trainer import CORE_LOSS_KEY


class CombinedSAMBNDLLoss(nn.Module):
    """
    Conflict-Aware Combined Loss with Staged Training
    
    理论改进:
    1. Staged training: 避免early stage的severe gradient conflicts
    2. Conflict-aware weighting: 动态调整weights based on loss magnitude
    3. 理论依据: Xue et al., "Conflict-Aware Adversarial Training" (2024)
    """

    def __init__(self, sam_loss, bndl_loss, ur_ern_loss=None, aue_loss=None, sam_weight=1.0, bndl_weight=1.0, ur_ern_weight=1.0, aue_weight=1.0, use_conflict_aware=True, use_staged_training=True):
        super().__init__()
        self.sam_loss = sam_loss
        self.bndl_loss = bndl_loss
        self.ur_ern_loss = ur_ern_loss
        self.aue_loss = aue_loss
        
        # Base weights (从config读取)
        self.sam_weight_base = sam_weight
        self.bndl_weight_base = bndl_weight
        self.ur_ern_weight_base = ur_ern_weight
        self.aue_weight_base = aue_weight
        
        # Conflict-aware settings
        self.use_conflict_aware = use_conflict_aware
        self.use_staged_training = use_staged_training
        
        # Training progress tracking
        self.current_epoch = 0
        self.max_epochs = 50  # Default, will be updated by trainer
        
        # EMA for loss magnitude (用于conflict-aware weighting)
        self.ema_momentum = 0.9
        self.ema_sam_loss = None
        self.ema_bndl_loss = None
        self.ema_aue_loss = None
    
    def set_training_progress(self, current_epoch: int, max_epochs: int):
        """Update training progress for staged training"""
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs

    def forward(self, outs_batch: list[dict], targets_batch: torch.Tensor):
        # Compute all individual losses
        sam_losses = self.sam_loss(outs_batch, targets_batch)
        
        # Compute optional losses
        bndl_losses = None
        if self.bndl_weight_base > 0.0 and self.bndl_loss is not None:
            bndl_losses = self.bndl_loss(outs_batch, targets_batch)
        
        ur_ern_losses = None
        if self.ur_ern_weight_base > 0.0 and self.ur_ern_loss is not None:
            ur_ern_losses = self.ur_ern_loss(outs_batch, targets_batch)
        
        aue_losses = None
        if self.aue_weight_base > 0.0 and self.aue_loss is not None:
            aue_losses = self.aue_loss(outs_batch, targets_batch)
        
        # Extract core loss values for conflict-aware weighting
        loss_sam = sam_losses[CORE_LOSS_KEY]
        loss_bndl = bndl_losses[CORE_LOSS_KEY] if bndl_losses is not None else torch.tensor(0.0, device=loss_sam.device)
        loss_aue = aue_losses[CORE_LOSS_KEY] if aue_losses is not None else torch.tensor(0.0, device=loss_sam.device)
        
        # ===== Staged Training + Conflict-Aware Weighting =====
        if self.use_staged_training and self.max_epochs > 0:
            stage = self.current_epoch / self.max_epochs
            
            # Stage 1: Warmup (0-20%): 只SAM + BNDL
            if stage < 0.2:
                alpha_sam = self.sam_weight_base
                alpha_bndl = self.bndl_weight_base
                alpha_aue = 0.0  # AUE暂时不加
            
            # Stage 2: Ramp-up (20%-50%): 逐渐引入AUE
            elif stage < 0.5:
                alpha_sam = self.sam_weight_base
                alpha_bndl = self.bndl_weight_base
                # Linear ramp-up
                ramp_progress = (stage - 0.2) / 0.3
                alpha_aue = self.aue_weight_base * ramp_progress
            
            # Stage 3: Full training (50%+): conflict-aware
            else:
                if self.use_conflict_aware and self.training:
                    # Update EMA of loss magnitudes
                    with torch.no_grad():
                        sam_mag = loss_sam.item()
                        bndl_mag = loss_bndl.item() if loss_bndl.item() > 0 else 1.0
                        aue_mag = loss_aue.item() if loss_aue.item() > 0 else 1.0
                        
                        # EMA update
                        if self.ema_sam_loss is None:
                            self.ema_sam_loss = sam_mag
                            self.ema_bndl_loss = bndl_mag
                            self.ema_aue_loss = aue_mag
                        else:
                            self.ema_sam_loss = self.ema_momentum * self.ema_sam_loss + (1 - self.ema_momentum) * sam_mag
                            self.ema_bndl_loss = self.ema_momentum * self.ema_bndl_loss + (1 - self.ema_momentum) * bndl_mag
                            self.ema_aue_loss = self.ema_momentum * self.ema_aue_loss + (1 - self.ema_momentum) * aue_mag
                    
                    # Conflict-aware dynamic weighting
                    # 基于loss magnitude的inverse scaling (平衡不同scale的losses)
                    alpha_sam = self.sam_weight_base
                    alpha_bndl = self.bndl_weight_base * (self.ema_sam_loss / (self.ema_bndl_loss + 1e-6))
                    alpha_aue = self.aue_weight_base * (self.ema_sam_loss / (self.ema_aue_loss + 1e-6))
                    
                    # Clamp to reasonable range (避免过度调整)
                    alpha_bndl = float(torch.clamp(
                        torch.tensor(alpha_bndl),
                        0.1 * self.bndl_weight_base,
                        2.0 * self.bndl_weight_base
                    ).item())
                    alpha_aue = float(torch.clamp(
                        torch.tensor(alpha_aue),
                        0.1 * self.aue_weight_base,
                        2.0 * self.aue_weight_base
                    ).item())
                else:
                    # No conflict-aware (e.g., eval mode)
                    alpha_sam = self.sam_weight_base
                    alpha_bndl = self.bndl_weight_base
                    alpha_aue = self.aue_weight_base
        else:
            # No staged training
            alpha_sam = self.sam_weight_base
            alpha_bndl = self.bndl_weight_base
            alpha_aue = self.aue_weight_base
        
        # ===== Merge Losses with Dynamic Weights =====
        combined_losses = {}
        
        # SAM loss
        for k, v in sam_losses.items():
            if k == CORE_LOSS_KEY:
                combined_losses["sam_core_loss"] = v * alpha_sam
            else:
                combined_losses[f"sam_{k}"] = v
        
        # BNDL loss
        if bndl_losses is not None:
            for k, v in bndl_losses.items():
                if k == CORE_LOSS_KEY:
                    combined_losses["bndl_core_loss"] = v * alpha_bndl
                else:
                    combined_losses[f"bndl_{k}"] = v
        else:
            combined_losses["bndl_core_loss"] = torch.tensor(0.0, device=loss_sam.device, requires_grad=False)
        
        # UR-ERN loss (optional)
        if ur_ern_losses is not None:
            for k, v in ur_ern_losses.items():
                if k == CORE_LOSS_KEY:
                    combined_losses["ur_ern_core_loss"] = v * self.ur_ern_weight_base
                else:
                    combined_losses[f"ur_ern_{k}"] = v
        
        # AUE loss (with staged + conflict-aware weights)
        if aue_losses is not None:
            for k, v in aue_losses.items():
                if k == CORE_LOSS_KEY:
                    combined_losses["aue_core_loss"] = v * alpha_aue
                else:
                    combined_losses[f"aue_{k}"] = v
        
        # Compute total core loss
        core = combined_losses["sam_core_loss"] + combined_losses["bndl_core_loss"]
        if "ur_ern_core_loss" in combined_losses:
            core = core + combined_losses["ur_ern_core_loss"]
        if "aue_core_loss" in combined_losses:
            core = core + combined_losses["aue_core_loss"]
        combined_losses[CORE_LOSS_KEY] = core
        
        # Log dynamic weights for monitoring
        combined_losses["loss_weight_sam"] = alpha_sam
        combined_losses["loss_weight_bndl"] = alpha_bndl
        combined_losses["loss_weight_aue"] = alpha_aue
        combined_losses["training_stage"] = self.current_epoch / max(self.max_epochs, 1)
        
        return combined_losses
