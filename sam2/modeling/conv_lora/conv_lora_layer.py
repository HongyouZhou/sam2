# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Conv-LoRA layer implementation.

Based on "Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model" (ICLR 2024)
https://openreview.net/forum?id=qpYoLWefgL
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ScaleExpert(nn.Module):
    """
    A single expert in the MoE Conv-LoRA architecture.
    
    It operates at a specific spatial scale:
    Input -> Interpolate(scale) -> Conv3x3 -> Interpolate(original) -> Output
    """
    
    def __init__(
        self,
        rank: int,
        scale: float,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.scale = scale
        
        # Depthwise convolution for parameter efficiency
        # Groups = rank ensures each channel is processed independently
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            rank, rank,
            kernel_size=kernel_size,
            padding=padding,
            groups=rank,
            bias=False
        )
        
        # Initialize with Kaiming Uniform
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, Rank, H, W]
            original_size: Tuple (H, W)
        """
        if self.scale == 1.0:
            return self.conv(x)
        
        H, W = original_size
        target_h = int(H * self.scale)
        target_w = int(W * self.scale)
        
        # 1. Downsample/Upsample to target scale
        x_scaled = F.interpolate(
            x, size=(target_h, target_w), mode='bilinear', align_corners=False
        )
        
        # 2. Apply convolution
        x_conv = self.conv(x_scaled)
        
        # 3. Restore original resolution
        x_out = F.interpolate(
            x_conv, size=(H, W), mode='bilinear', align_corners=False
        )
        
        return x_out


class ConvLoRALayer(nn.Module):
    """
    Conv-LoRA: Low-rank adaptation with convolutional inductive bias and Mixture-of-Experts.
    
    Architecture:
    x -> A (Linear) -> MoE(Multi-scale Convs) -> B (Linear) -> output
    
    The MoE block dynamically selects experts that operate at different spatial scales.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        expert_scales: List[float] = [1.0, 0.5, 2.0],
        top_k: int = 1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.expert_scales = expert_scales
        self.num_experts = len(expert_scales)
        self.top_k = max(1, min(top_k, self.num_experts))
        
        # 1. Down-projection (Linear A)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        
        # 2. Mixture-of-Experts (MoE)
        # Gating network: Computes weights for each expert based on global context
        self.gating_fc = nn.Linear(rank, self.num_experts)
        
        # Experts: Convolutional layers at different scales
        self.experts = nn.ModuleList([
            ScaleExpert(rank, scale) for scale in expert_scales
        ])
        
        # 3. Up-projection (Linear B)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialization
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        nn.init.zeros_(self.gating_fc.weight)
        nn.init.zeros_(self.gating_fc.bias)
        
        # Bias the gating towards the scale=1.0 expert (identity scale) for stability
        try:
            identity_idx = self.expert_scales.index(1.0)
            with torch.no_grad():
                self.gating_fc.bias[identity_idx] = 2.0 # Moderate bias towards identity
        except ValueError:
            pass # No identity scale found, stick to uniform
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, C] or [B, H, W, C]
            H, W: Spatial dimensions
        """
        # Handle input format
        is_4d = (x.ndim == 4)
        if is_4d:
            B, H_in, W_in, C = x.shape
            # Flatten for Linear layers: [B, H, W, C] -> [B, N, C]
            x = x.flatten(1, 2)
            N = H_in * W_in
            # Use dimensions from input if they match (sanity check) or rely on passed H, W
            # We rely on passed H, W for consistency with 3D case
        else:
            B, N, C = x.shape
            # x is [B, N, C]

        # 1. Linear Projection A
        x_low = self.lora_A(x) # [B, N, rank]
        
        # 2. MoE Processing
        # Reshape to image format for convolution: [B, N, rank] -> [B, rank, H, W]
        x_map = x_low.transpose(1, 2).reshape(B, self.rank, H, W)
        
        # Calculate gating weights
        # Global Average Pooling for context
        x_gap = x_map.mean(dim=(2, 3)) # [B, rank]
        gating_logits = self.gating_fc(x_gap) # [B, num_experts]
        # Top-k sparse gating (paper: top-k=1 by default)
        if self.top_k >= self.num_experts:
            gating_weights = F.softmax(gating_logits, dim=1)
        else:
            topk_vals, topk_idx = torch.topk(gating_logits, k=self.top_k, dim=1)
            masked_logits = torch.full_like(gating_logits, float('-inf'))
            masked_logits.scatter_(1, topk_idx, topk_vals)
            gating_weights = F.softmax(masked_logits, dim=1)
        
        # Apply experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            out = expert(x_map, (H, W)) # [B, rank, H, W]
            expert_outputs.append(out)
            
        # Weighted sum of experts
        # Stack: [B, num_experts, rank, H, W]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Expand weights: [B, num_experts, 1, 1, 1]
        weights = gating_weights.view(B, self.num_experts, 1, 1, 1)
        
        # Sum: [B, rank, H, W]
        x_moe = (expert_outputs * weights).sum(dim=1)
        
        # 3. Linear Projection B
        # [B, rank, H, W] -> [B, N, rank]
        x_moe_flat = x_moe.flatten(2).transpose(1, 2) 
        x_moe_flat = self.dropout(x_moe_flat)
        result = self.lora_B(x_moe_flat) # [B, N, out_features]
        
        # Restore original shape
        if is_4d:
            result = result.reshape(B, H, W, self.out_features)
            
        return result * self.scaling


class LinearWithConvLoRA(nn.Module):
    """
    Wrapper that combines a frozen Linear layer with trainable Conv-LoRA bypass.
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        expert_scales: List[float] = [1.0, 0.5, 2.0],
        top_k: int = 1,
    ):
        super().__init__()
        
        # Store original linear layer (frozen)
        self.linear = original_linear
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # Create Conv-LoRA bypass
        self.conv_lora = ConvLoRALayer(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            expert_scales=expert_scales,
            top_k=top_k,
        )
        
        # Store spatial dimensions (will be set during forward)
        self._spatial_shape = None
    
    def set_spatial_shape(self, H: int, W: int):
        """Set spatial dimensions for Conv-LoRA computation."""
        self._spatial_shape = (H, W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear output (frozen)
        out = self.linear(x)
        
        # Determine spatial shape
        if self._spatial_shape is not None:
            H, W = self._spatial_shape
        elif x.ndim == 4:
            # Assume (B, H, W, C) for Linear input
            H, W = x.shape[1], x.shape[2]
            self._spatial_shape = (H, W)
        elif x.ndim == 3:
            B, N, C = x.shape
            root = int(math.isqrt(N))
            if root * root == N:
                H = W = root
            else:
                # Find a factor pair (H, W) close to square to avoid hard failure on non-square inputs
                best_pair = None
                best_gap = None
                for h in range(1, root + 1):
                    if N % h == 0:
                        w = N // h
                        gap = abs(h - w)
                        if best_gap is None or gap < best_gap:
                            best_gap = gap
                            best_pair = (h, w)
                if best_pair is None:
                    raise ValueError(f"Cannot infer spatial dimensions from sequence length {N}.")
                H, W = best_pair
            self._spatial_shape = (H, W)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # Add Conv-LoRA bypass
        delta = self.conv_lora(x, H, W)
        
        return out + delta
    
    def __repr__(self):
        return (
            f"LinearWithConvLoRA(in_features={self.linear.in_features}, "
            f"out_features={self.linear.out_features}, "
            f"rank={self.conv_lora.rank}, "
            f"alpha={self.conv_lora.alpha}, "
            f"experts={self.conv_lora.num_experts}, "
            f"top_k={self.conv_lora.top_k})"
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """
        Handle backward compatibility for loading standard Linear weights.
        Remaps 'weight' and 'bias' to 'linear.weight' and 'linear.bias'.
        """
        for param_name in ["weight", "bias"]:
            old_key = prefix + param_name
            new_key = prefix + "linear." + param_name
            if old_key in state_dict:
                # Move the parameter to the new key so the child 'linear' module can find it
                state_dict[new_key] = state_dict.pop(old_key)
        
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

