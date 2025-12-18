# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
DoRA injection utilities.

Separated from standard LoRA to minimize coupling risk.
"""

import logging
from typing import Dict

import torch.nn as nn

from .dora_layer import LinearWithDoRA, Conv2dWithDoRA


def inject_dora(
    model: nn.Module,
    config: Dict,
    verbose: bool = True,
) -> nn.Module:
    """
    Inject DoRA layers into target Linear/Conv2d layers of the model.

    Args:
        model: target module (e.g., image_encoder.trunk)
        config: dict with keys rank, alpha, dropout, target_modules
        verbose: whether to log injection info
    """
    rank = config.get("rank", 8)
    alpha = config.get("alpha", 16.0)
    dropout = config.get("dropout", 0.0)
    target_modules = config.get("target_modules", [])

    if not target_modules:
        logging.warning("No target_modules specified for DoRA injection. Skipping.")
        return model

    injection_count = 0
    total_params_before = sum(p.numel() for p in model.parameters())

    def _inject_recursive(module: nn.Module, name: str):
        nonlocal injection_count
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            should_inject = any(target in full_name for target in target_modules)

            if should_inject and isinstance(child_module, nn.Linear):
                dora_layer = LinearWithDoRA(
                    original_linear=child_module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(module, child_name, dora_layer)
                injection_count += 1
                if verbose:
                    logging.info(f"  ✓ Injected DoRA (Linear): {full_name}")
            elif should_inject and isinstance(child_module, nn.Conv2d):
                if child_module.groups > 1:
                    if verbose:
                        logging.info(f"  x Skipping DoRA (Grouped Conv2d): {full_name}")
                    continue

                dora_layer = Conv2dWithDoRA(
                    original_conv=child_module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(module, child_name, dora_layer)
                injection_count += 1
                if verbose:
                    logging.info(f"  ✓ Injected DoRA (Conv2d): {full_name}")
            else:
                _inject_recursive(child_module, full_name)

    _inject_recursive(model, "")

    total_params_after = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        logging.info(f"\n{'='*60}")
        logging.info("DoRA Injection Summary:")
        logging.info(f"  Layers modified: {injection_count}")
        logging.info(f"  Total params: {total_params_before:,} → {total_params_after:,}")
        logging.info(
            f"  Trainable params: {trainable_params:,} "
            f"({100*trainable_params/total_params_after:.2f}%)"
        )
        logging.info(f"  Config: rank={rank}, alpha={alpha}")
        logging.info(f"{'='*60}\n")

    return model


def remove_dora(model: nn.Module) -> nn.Module:
    """Remove DoRA layers and restore original Linear/Conv2d layers."""

    def _remove_recursive(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, LinearWithDoRA):
                setattr(module, name, child.linear)
                logging.info(f"  ✓ Removed DoRA (Linear): {name}")
            elif isinstance(child, Conv2dWithDoRA):
                setattr(module, name, child.conv)
                logging.info(f"  ✓ Removed DoRA (Conv2d): {name}")
            else:
                _remove_recursive(child)

    _remove_recursive(model)
    return model


def get_dora_parameters(model: nn.Module):
    """Collect DoRA parameters (low-rank A/B and magnitude m)."""
    params = {}
    for name, param in model.named_parameters():
        if any(k in name for k in ["lora.lora_A", "lora.lora_B"]):
            params[name] = param
        if name.endswith(".m"):
            params[name] = param
    return params
