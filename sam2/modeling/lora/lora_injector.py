# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
LoRA injection utilities (standard LoRA only).

DoRA is handled separately in dora_injector to isolate behavior.
"""

import logging
import re
from typing import Dict, List, Optional

import torch.nn as nn

from .lora_layer import LinearWithLoRA, Conv2dWithLoRA


def inject_lora(
    model: nn.Module,
    config: Dict,
    verbose: bool = True,
) -> nn.Module:
    """
    Inject LoRA layers into target Linear/Conv2d layers of the model.
    
    Args:
        model: Model to inject LoRA into (typically image_encoder.trunk)
        config: Configuration dict with keys:
            - rank: LoRA rank
            - alpha: Scaling factor
            - dropout: Dropout rate
            - target_modules: List of module name patterns (e.g., ["attn.qkv", "mlp.layers.0"])
            - mode: "standard" (default) or "dora"
        verbose: Print injection information
    
    Returns:
        model: Modified model with LoRA injected
    """
    rank = config.get('rank', 8)
    alpha = config.get('alpha', 16.0)
    dropout = config.get('dropout', 0.0)
    target_modules = config.get('target_modules', [])

    if not target_modules:
        logging.warning("No target_modules specified for LoRA injection. Skipping.")
        return model
    
    injection_count = 0
    total_params_before = sum(p.numel() for p in model.parameters())
    
    # Recursively replace Linear/Conv2d layers
    def _inject_recursive(module: nn.Module, parent: nn.Module, name: str):
        nonlocal injection_count
        
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Check if this module matches any target pattern
            should_inject = any(
                target in full_name for target in target_modules
            )
            
            if should_inject:
                if isinstance(child_module, nn.Linear):
                    lora_layer = LinearWithLoRA(
                        original_linear=child_module,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    setattr(module, child_name, lora_layer)
                    injection_count += 1
                    if verbose:
                        logging.info(f"  ✓ Injected LoRA (Linear): {full_name}")

                elif isinstance(child_module, nn.Conv2d):
                    lora_layer = Conv2dWithLoRA(
                        original_conv=child_module,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    setattr(module, child_name, lora_layer)
                    injection_count += 1
                    if verbose:
                        logging.info(f"  ✓ Injected LoRA (Conv2d): {full_name}")
                
                else:
                    # Recurse into child modules even if matched (might be a container)
                    # But usually we target leaf layers. If it's a container and matched,
                    # we might want to inject into its children?
                    # Standard behavior: target_modules usually targets leaf layers.
                    # If a container is targeted, we recurse.
                    _inject_recursive(child_module, module, full_name)
            else:
                # Recurse into child modules
                _inject_recursive(child_module, module, full_name)
    
    _inject_recursive(model, None, "")
    
    total_params_after = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if verbose:
        logging.info(f"\n{'='*60}")
        logging.info(f"LoRA Injection Summary:")
        logging.info(f"  Layers modified: {injection_count}")
        logging.info(f"  Total params: {total_params_before:,} → {total_params_after:,}")
        logging.info(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params_after:.2f}%)")
        logging.info(f"  Config: rank={rank}, alpha={alpha}")
        logging.info(f"{'='*60}\n")
    
    return model


def remove_lora(model: nn.Module) -> nn.Module:
    """
    Remove LoRA layers and restore original Linear/Conv2d layers.
    
    Useful for debugging or exporting models without LoRA.
    
    Args:
        model: Model with LoRA injected
    
    Returns:
        model: Model with LoRA removed
    """
    def _remove_recursive(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, LinearWithLoRA):
                setattr(module, name, child.linear)
                logging.info(f"  ✓ Removed LoRA (Linear): {name}")
            elif isinstance(child, Conv2dWithLoRA):
                setattr(module, name, child.conv)
                logging.info(f"  ✓ Removed LoRA (Conv2d): {name}")
            else:
                _remove_recursive(child)
    
    _remove_recursive(model)
    return model


def get_lora_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    """
    Extract all LoRA parameters from the model.
    
    This is useful for:
    - Creating separate optimizer parameter groups
    - Saving only LoRA weights
    - Analyzing parameter statistics
    
    Args:
        model: Model with LoRA injected
    
    Returns:
        dict: Mapping of parameter names to parameters
    """
    lora_params = {}
    
    for name, param in model.named_parameters():
        # LoRA parameters
        if any(keyword in name for keyword in [
            'lora.lora_A',
            'lora.lora_B',
        ]):
            lora_params[name] = param

    return lora_params
