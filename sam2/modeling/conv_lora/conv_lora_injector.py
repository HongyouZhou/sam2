# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Conv-LoRA injection utilities.

Provides functions to inject/remove Conv-LoRA layers into existing models.
"""

import logging
import re
from typing import Dict, List, Optional

import torch.nn as nn

from .conv_lora_layer import LinearWithConvLoRA


def inject_conv_lora(
    model: nn.Module,
    config: Dict,
    verbose: bool = True,
) -> nn.Module:
    """
    Inject Conv-LoRA layers into target Linear layers of the model.
    
    Args:
        model: Model to inject Conv-LoRA into (typically image_encoder.trunk)
        config: Configuration dict with keys:
            - rank: LoRA rank
            - alpha: Scaling factor
            - dropout: Dropout rate
            - expert_scales: List of scales for MoE experts (e.g., [1.0, 0.5, 2.0])
            - target_modules: List of module name patterns (e.g., ["attn.qkv", "mlp.layers.0"])
        verbose: Print injection information
    
    Returns:
        model: Modified model with Conv-LoRA injected
    """
    rank = config.get('rank', 8)
    alpha = config.get('alpha', 16.0)
    dropout = config.get('dropout', 0.0)
    expert_scales = config.get('expert_scales', [1.0, 0.5, 2.0])
    top_k = config.get('top_k', 1)
    target_modules = config.get('target_modules', [])
    
    if not target_modules:
        logging.warning("No target_modules specified for Conv-LoRA injection. Skipping.")
        return model
    
    injection_count = 0
    total_params_before = sum(p.numel() for p in model.parameters())
    
    # Recursively replace Linear layers
    def _inject_recursive(module: nn.Module, parent: nn.Module, name: str):
        nonlocal injection_count
        
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Check if this module matches any target pattern
            should_inject = any(
                target in full_name for target in target_modules
            )
            
            if should_inject and isinstance(child_module, nn.Linear):
                # Replace with LinearWithConvLoRA
                conv_lora_layer = LinearWithConvLoRA(
                    original_linear=child_module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    expert_scales=expert_scales,
                    top_k=top_k,
                )
                
                setattr(module, child_name, conv_lora_layer)
                injection_count += 1
                
                if verbose:
                    logging.info(f"  ✓ Injected Conv-LoRA: {full_name}")
            else:
                # Recurse into child modules
                _inject_recursive(child_module, module, full_name)
    
    _inject_recursive(model, None, "")
    
    total_params_after = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if verbose:
        logging.info(f"\n{'='*60}")
        logging.info(f"Conv-LoRA Injection Summary:")
        logging.info(f"  Layers modified: {injection_count}")
        logging.info(f"  Total params: {total_params_before:,} → {total_params_after:,}")
        logging.info(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params_after:.2f}%)")
        logging.info(
            f"  Config: rank={rank}, alpha={alpha}, top_k={top_k}, "
            f"experts={len(expert_scales)} (scales={expert_scales})"
        )
        logging.info(f"{'='*60}\n")
    
    return model


def remove_conv_lora(model: nn.Module) -> nn.Module:
    """
    Remove Conv-LoRA layers and restore original Linear layers.
    
    Useful for debugging or exporting models without Conv-LoRA.
    
    Args:
        model: Model with Conv-LoRA injected
    
    Returns:
        model: Model with Conv-LoRA removed
    """
    def _remove_recursive(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, LinearWithConvLoRA):
                # Restore original Linear layer
                setattr(module, name, child.linear)
                logging.info(f"  ✓ Removed Conv-LoRA: {name}")
            else:
                _remove_recursive(child)
    
    _remove_recursive(model)
    return model


def get_conv_lora_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    """
    Extract all Conv-LoRA parameters from the model.
    
    This is useful for:
    - Creating separate optimizer parameter groups
    - Saving only Conv-LoRA weights
    - Analyzing parameter statistics
    
    Args:
        model: Model with Conv-LoRA injected
    
    Returns:
        dict: Mapping of parameter names to parameters
    """
    conv_lora_params = {}
    
    for name, param in model.named_parameters():
        # Check if parameter belongs to Conv-LoRA
        if any(keyword in name for keyword in [
            'conv_lora.lora_A',
            'conv_lora.lora_B',
            'conv_lora.gating_fc',
            'conv_lora.experts',
        ]):
            conv_lora_params[name] = param
    
    return conv_lora_params


def get_conv_lora_state_dict(model: nn.Module) -> Dict[str, any]:
    """
    Extract Conv-LoRA state dict for checkpointing.
    
    Args:
        model: Model with Conv-LoRA injected
    
    Returns:
        dict: State dict containing only Conv-LoRA parameters
    """
    conv_lora_params = get_conv_lora_parameters(model)
    return {name: param.data for name, param in conv_lora_params.items()}


def count_conv_lora_parameters(model: nn.Module) -> int:
    """
    Count total Conv-LoRA parameters.
    
    Args:
        model: Model with Conv-LoRA injected
    
    Returns:
        int: Total number of Conv-LoRA parameters
    """
    conv_lora_params = get_conv_lora_parameters(model)
    return sum(p.numel() for p in conv_lora_params.values())

