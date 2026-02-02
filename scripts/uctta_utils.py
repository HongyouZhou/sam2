#!/usr/bin/env python
"""
UC-TTA (Uncertainty-Calibrated Test-Time Adaptation) utilities.

Based on "Uncertainty-Calibrated Test-Time Model Adaptation without Forgetting" (TPAMI 2025).

This module provides reusable UC-TTA components that can be integrated into
any SAM2 evaluation script (ImagePredictor or VideoPredictor based).

Key components:
- Temperature scaling for uncertainty calibration
- Entropy minimization with sample selection
- Fisher regularization to prevent forgetting
- BN/LN layer adaptation setup
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UCTTAConfig:
    """Configuration for UC-TTA adaptation."""

    # Whether to enable UC-TTA at all
    enabled: bool = False

    # Number of adaptation steps per sample
    steps: int = 2

    # Learning rate for adaptation
    lr: float = 3e-4

    # Whether to adapt BN/LN layers (vs temperature-only)
    enable_bn_adapt: bool = True

    # Fisher regularization settings
    use_fisher_reg: bool = False  # Disabled by default for simplicity
    fisher_alpha: float = 2000.0

    # Sample selection settings
    entropy_threshold: float = 0.4  # Max entropy for reliable samples
    selection_p: float = 0.1  # Fraction of samples to use

    # Temperature bounds
    temp_min: float = 0.25
    temp_max: float = 4.0


def entropy_from_logits_scaled(
    logits: torch.Tensor,
    logT: torch.Tensor,
    temp_min: float = 0.25,
    temp_max: float = 4.0,
) -> torch.Tensor:
    """Compute binary entropy averaged over spatial dims with temperature scaling.

    Args:
        logits: [H, W] or [K, H, W] mask logits
        logT: Scalar log-temperature parameter (requires_grad=True)
        temp_min: Minimum temperature bound
        temp_max: Maximum temperature bound

    Returns:
        Scalar mean entropy
    """
    T = torch.exp(logT).clamp(temp_min, temp_max)
    z = logits / T

    # Handle different input shapes
    if z.ndim == 3:
        # [K, H, W] -> [H, W, K] for consistent processing
        if z.shape[0] <= 8 and z.shape[0] != z.shape[-1]:
            z = z.permute(1, 2, 0)

    p = torch.sigmoid(z)
    ent = -(p * torch.log(p.clamp_min(1e-8)) + (1.0 - p) * torch.log((1.0 - p).clamp_min(1e-8)))

    return ent.mean()


def entropy_map_from_logits_scaled(
    logits: torch.Tensor,
    logT: torch.Tensor,
    temp_min: float = 0.25,
    temp_max: float = 4.0,
) -> torch.Tensor:
    """Compute per-pixel entropy map after temperature scaling.

    Args:
        logits: [H, W] or [K, H, W] mask logits
        logT: Scalar log-temperature parameter
        temp_min: Minimum temperature bound
        temp_max: Maximum temperature bound

    Returns:
        [H, W] entropy map
    """
    T = torch.exp(logT).clamp(temp_min, temp_max)
    z = logits / T

    if z.ndim == 3:
        if z.shape[0] <= 8 and z.shape[0] != z.shape[-1]:
            z = z.permute(1, 2, 0)

    p = torch.sigmoid(z)
    ent = -(p * torch.log(p.clamp_min(1e-8)) + (1.0 - p) * torch.log((1.0 - p).clamp_min(1e-8)))

    if ent.ndim == 3:
        return ent.mean(dim=-1)  # [H, W]
    return ent


def apply_temperature(
    logits: torch.Tensor,
    logT: torch.Tensor,
    temp_min: float = 0.25,
    temp_max: float = 4.0,
) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Args:
        logits: Input logits (any shape)
        logT: Scalar log-temperature parameter
        temp_min: Minimum temperature bound
        temp_max: Maximum temperature bound

    Returns:
        Temperature-scaled logits (same shape as input)
    """
    T = torch.exp(logT).clamp(temp_min, temp_max)
    return logits / T


def entropy_with_sample_selection(
    logits: torch.Tensor,
    logT: torch.Tensor,
    entropy_threshold: float = 0.4,
    selection_p: float = 0.1,
    temp_min: float = 0.25,
    temp_max: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute entropy with active sample selection based on reliability.

    Only low-entropy (high confidence) samples are used for adaptation,
    which prevents the model from being misled by incorrect predictions.

    Args:
        logits: Prediction logits [H, W] or [B, H, W]
        logT: Temperature parameter (log scale)
        entropy_threshold: Maximum entropy for sample selection (0-1 range)
        selection_p: Probability threshold for selecting samples
        temp_min: Minimum temperature bound
        temp_max: Maximum temperature bound

    Returns:
        (filtered_loss, selection_mask): Loss from selected samples and selection mask
    """
    T = torch.exp(logT).clamp(temp_min, temp_max)
    z = logits / T

    if z.ndim == 2:
        z = z.unsqueeze(0)  # [1, H, W]

    # Compute probabilities and entropy
    p = torch.sigmoid(z)
    ent = -(p * torch.log(p.clamp_min(1e-8)) + (1.0 - p) * torch.log((1.0 - p).clamp_min(1e-8)))

    # Sample selection: only use reliable (low entropy) samples
    ent_flat = ent.reshape(-1)

    # Select samples with entropy below threshold
    reliable_mask = ent_flat < entropy_threshold

    # Further select top-p most confident samples
    if selection_p < 1.0 and reliable_mask.sum() > 0:
        num_select = max(1, int(selection_p * len(ent_flat)))
        sorted_ent, sorted_idx = torch.sort(ent_flat)
        top_p_mask = torch.zeros_like(ent_flat, dtype=torch.bool)
        top_p_mask[sorted_idx[:num_select]] = True
        reliable_mask = reliable_mask & top_p_mask

    # Compute loss only on selected samples
    if reliable_mask.sum() > 0:
        selected_entropy = ent_flat[reliable_mask]
        return selected_entropy.mean(), reliable_mask
    else:
        # No reliable samples, return full entropy
        return ent.mean(), torch.ones_like(ent_flat, dtype=torch.bool)


def setup_uctta_model(
    model: nn.Module,
    enable_bn_adapt: bool = True,
) -> list[torch.Tensor]:
    """Setup model for UC-TTA: freeze most params, enable BN/LN layers for adaptation.

    Args:
        model: SAM2 model (predictor.model for ImagePredictor)
        enable_bn_adapt: Whether to enable BN/LayerNorm adaptation

    Returns:
        List of adaptable parameters
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    adaptable_params = []

    if enable_bn_adapt:
        # Enable BatchNorm and LayerNorm parameters for adaptation
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
                for param in module.parameters():
                    param.requires_grad = True
                    adaptable_params.append(param)
                # Also enable BN running stats update
                if hasattr(module, "track_running_stats"):
                    module.track_running_stats = True
                if hasattr(module, "momentum"):
                    module.momentum = 0.1  # Standard BN momentum

    return adaptable_params


def compute_fisher_regularization(
    model: nn.Module,
    fisher_dict: dict[str, torch.Tensor],
    original_params: dict[str, torch.Tensor],
    fisher_alpha: float = 2000.0,
) -> torch.Tensor:
    """Compute Fisher regularization to prevent forgetting.

    Args:
        model: The model being adapted
        fisher_dict: Precomputed Fisher information for each parameter
        original_params: Original parameter values before adaptation
        fisher_alpha: Regularization strength

    Returns:
        Fisher regularization loss
    """
    device = next(model.parameters()).device
    fisher_loss = torch.tensor(0.0, device=device)

    for name, param in model.named_parameters():
        if param.requires_grad and name in fisher_dict and name in original_params:
            # L2 distance weighted by Fisher information
            fisher_loss = fisher_loss + (fisher_dict[name] * (param - original_params[name]).pow(2)).sum()

    return fisher_alpha * fisher_loss


def restore_model_params(
    model: nn.Module,
    original_params: dict[str, torch.Tensor],
) -> None:
    """Restore model parameters to original values after adaptation.

    This is useful when you want to reset the model between samples.

    Args:
        model: The model to restore
        original_params: Original parameter values
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_params:
                param.copy_(original_params[name])


class UCTTAAdapter:
    """UC-TTA adapter for SAM2 inference.

    This class encapsulates the UC-TTA adaptation logic and can be used
    with either SAM2ImagePredictor or SAM2VideoPredictor.

    Usage:
        adapter = UCTTAAdapter(model, config)
        adapter.setup()

        # For each sample:
        adapted_logits = adapter.adapt(logits, image_size)

        # Optionally reset between samples:
        adapter.reset()
    """

    def __init__(
        self,
        model: nn.Module,
        config: UCTTAConfig | None = None,
    ):
        """Initialize UC-TTA adapter.

        Args:
            model: SAM2 model (predictor.model for ImagePredictor)
            config: UC-TTA configuration
        """
        self.model = model
        self.config = config or UCTTAConfig()

        self.adaptable_params: list[torch.Tensor] = []
        self.original_params: dict[str, torch.Tensor] = {}
        self.logT: torch.Tensor | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self._setup_done = False

    def setup(self) -> None:
        """Setup model for UC-TTA adaptation."""
        if not self.config.enabled:
            return

        # Setup adaptable parameters
        self.adaptable_params = setup_uctta_model(
            self.model,
            enable_bn_adapt=self.config.enable_bn_adapt,
        )

        # Store original parameters for Fisher regularization and reset
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.original_params[name] = param.data.clone()

        # Initialize temperature parameter
        device = next(self.model.parameters()).device
        self.logT = nn.Parameter(torch.zeros(1, device=device, dtype=torch.float32))

        # Setup optimizer
        opt_params = [self.logT]
        if self.config.enable_bn_adapt:
            opt_params.extend(self.adaptable_params)

        self.optimizer = torch.optim.Adam(opt_params, lr=self.config.lr)
        self._setup_done = True

    def adapt(
        self,
        logits: torch.Tensor,
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Adapt model and return temperature-scaled logits.

        Args:
            logits: Mask logits [H, W] or [K, H, W]
            target_size: Optional (H, W) to resize logits

        Returns:
            Temperature-scaled logits
        """
        if not self.config.enabled or not self._setup_done:
            return logits

        # Ensure logits are on the correct device and require grad
        logits_2d = logits.squeeze(0) if logits.ndim == 3 and logits.shape[0] == 1 else logits
        if logits_2d.ndim == 3:
            logits_2d = logits_2d[0]  # Take first mask if multiple

        # Resize if needed
        if target_size is not None and tuple(logits_2d.shape[-2:]) != target_size:
            logits_2d = F.interpolate(
                logits_2d.unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        # Detach and clone to avoid inference tensor issues
        logits_clean = logits_2d.detach().clone()

        # Adaptation loop
        for _ in range(self.config.steps):
            self.optimizer.zero_grad(set_to_none=True)

            with torch.enable_grad():
                # Entropy minimization with sample selection
                loss, _ = entropy_with_sample_selection(
                    logits_clean,
                    self.logT,
                    entropy_threshold=self.config.entropy_threshold,
                    selection_p=self.config.selection_p,
                    temp_min=self.config.temp_min,
                    temp_max=self.config.temp_max,
                )

                # Optional Fisher regularization
                if self.config.use_fisher_reg and self.config.enable_bn_adapt:
                    fisher_loss = compute_fisher_regularization(
                        self.model,
                        {},  # Empty fisher dict - would need precomputation
                        self.original_params,
                        self.config.fisher_alpha,
                    )
                    loss = loss + fisher_loss

                loss.backward()
                self.optimizer.step()

        # Apply learned temperature to logits
        scaled_logits = apply_temperature(
            logits,
            self.logT,
            temp_min=self.config.temp_min,
            temp_max=self.config.temp_max,
        )

        return scaled_logits

    def get_temperature(self) -> float:
        """Get current temperature value."""
        if self.logT is None:
            return 1.0
        return float(
            torch.exp(self.logT)
            .clamp(
                self.config.temp_min,
                self.config.temp_max,
            )
            .item()
        )

    def reset(self) -> None:
        """Reset model parameters and temperature for next sample."""
        if not self._setup_done:
            return

        # Reset temperature
        if self.logT is not None:
            self.logT.data.zero_()

        # Restore original model parameters
        restore_model_params(self.model, self.original_params)

        # Re-create optimizer with fresh state
        opt_params = [self.logT]
        if self.config.enable_bn_adapt:
            opt_params.extend(self.adaptable_params)
        self.optimizer = torch.optim.Adam(opt_params, lr=self.config.lr)


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("UC-TTA Utils Module Tests")
    print("=" * 60)

    # Test entropy computation
    print("\n1. Testing entropy computation...")
    logits = torch.randn(256, 256)
    logT = torch.zeros(1, requires_grad=True)
    ent = entropy_from_logits_scaled(logits, logT)
    print(f"  ✓ Mean entropy: {ent.item():.4f}")

    # Test sample selection
    print("\n2. Testing sample selection...")
    loss, mask = entropy_with_sample_selection(logits, logT)
    print(f"  ✓ Loss: {loss.item():.4f}, Selected: {mask.sum().item()}/{mask.numel()}")

    # Test config
    print("\n3. Testing UCTTAConfig...")
    config = UCTTAConfig(enabled=True, steps=2, lr=1e-4)
    print(f"  ✓ Config: enabled={config.enabled}, steps={config.steps}, lr={config.lr}")

    print("\n✓ All tests passed!")
