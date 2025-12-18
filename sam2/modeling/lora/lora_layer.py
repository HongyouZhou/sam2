# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
LoRA layer implementation.

Based on "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
https://arxiv.org/abs/2106.09685
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class LoRALayer(nn.Module):
    """
    LoRA: Low-Rank Adaptation.
    
    Architecture:
    x -> A (Linear) -> B (Linear) -> output
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 1. Down-projection (Linear A)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        
        # 2. Up-projection (Linear B)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialization
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, C] or [B, H, W, C]
        """
        # 1. Dropout (Standard LoRA applies dropout to input)
        x = self.dropout(x)

        # 2. Linear Projection A
        x_low = self.lora_A(x) # [..., rank]
        
        # 3. Linear Projection B
        result = self.lora_B(x_low) # [..., out_features]
            
        return result * self.scaling


class LoRAConv2dLayer(nn.Module):
    """
    LoRA for Conv2d layers.
    
    Architecture:
    x -> A (Conv2d kxk) -> B (Conv2d 1x1) -> output
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 1. Down-projection (Conv2d A)
        # Preserves spatial kernel size, stride, padding
        self.lora_A = nn.Conv2d(
            in_channels, rank, 
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            bias=False
        )
        
        # 2. Up-projection (Conv2d B)
        # 1x1 convolution to restore channel dimension
        self.lora_B = nn.Conv2d(
            rank, out_channels, 
            kernel_size=1, stride=1, padding=0,
            bias=False
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialization
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Dropout
        x = self.dropout(x)
        
        # 2. Conv A
        x = self.lora_A(x)
        
        # 3. Conv B
        x = self.lora_B(x)
        
        return x * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Wrapper that combines a frozen Linear layer with trainable LoRA bypass.
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Store original linear layer (frozen)
        self.linear = original_linear
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # Create LoRA bypass
        self.lora = LoRALayer(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        self.merged = False

    def merge(self):
        """
        Merge LoRA weights into the original linear layer.
        Useful for inference acceleration.
        """
        if self.merged:
            return
        
        if self.lora.rank > 0:
            with torch.no_grad():
                # Compute delta weight: B @ A * scaling
                # lora_B.weight: [out, rank]
                # lora_A.weight: [rank, in]
                # Result: [out, in]
                delta_w = (self.lora.lora_B.weight @ self.lora.lora_A.weight) * self.lora.scaling
                self.linear.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """
        Unmerge LoRA weights from the original linear layer.
        Useful for resuming training or switching adapters.
        """
        if not self.merged:
            return
            
        if self.lora.rank > 0:
            with torch.no_grad():
                delta_w = (self.lora.lora_B.weight @ self.lora.lora_A.weight) * self.lora.scaling
                self.linear.weight.data -= delta_w
            self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.linear(x)

        # Original linear output (frozen)
        out = self.linear(x)
        
        # Add LoRA bypass
        delta = self.lora(x)
        
        return out + delta
    
    def __repr__(self):
        return (
            f"LinearWithLoRA(in_features={self.linear.in_features}, "
            f"out_features={self.linear.out_features}, "
            f"rank={self.lora.rank}, "
            f"alpha={self.lora.alpha})"
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


class Conv2dWithLoRA(nn.Module):
    """
    Wrapper that combines a frozen Conv2d layer with trainable LoRA bypass.
    """
    
    def __init__(
        self,
        original_conv: nn.Conv2d,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Store original conv layer (frozen)
        self.conv = original_conv
        for param in self.conv.parameters():
            param.requires_grad = False
        
        # Create LoRA bypass
        self.lora = LoRAConv2dLayer(
            in_channels=original_conv.in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size[0], # Assume square or symmetric handling
            stride=original_conv.stride[0],
            padding=original_conv.padding[0],
            dilation=original_conv.dilation[0],
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        self.merged = False

    def merge(self):
        if self.merged:
            return
        
        # Check for groups compatibility
        if self.conv.groups > 1:
            # If original conv has groups > 1, we cannot merge a dense LoRA update 
            # into the grouped weight matrix directly because the shapes don't match.
            # Original weight: [Out, In/Groups, K, K]
            # LoRA delta: [Out, In, K, K]
            # We would need to enforce sparsity in LoRA or expand original weight (which changes architecture).
            # For now, we disable merging for grouped convolutions.
            return

        if self.lora.rank > 0:
            with torch.no_grad():
                # Conv2d weight merging is more complex than Linear
                # W_orig: [Out, In, K, K]
                # A: [Rank, In, K, K]
                # B: [Out, Rank, 1, 1]
                # We need to convolve B with A to get [Out, In, K, K]
                
                # B * A = conv2d(A, B) ? No.
                # It's effectively a matrix multiplication if we view it as:
                # W_new = B @ A (if 1x1)
                # But A is KxK.
                # The operation is: Output = B(A(Input))
                # This is equivalent to a single convolution with weight W'
                # W' = B.weight * A.weight (convolution of kernels)
                # Since B is 1x1, it acts as a linear combination of A's kernels.
                
                # B: [O, R, 1, 1] -> view as [O, R]
                # A: [R, I, K, K]
                # Result: [O, I, K, K]
                
                weight_B = self.lora.lora_B.weight.squeeze(3).squeeze(2) # [O, R]
                weight_A = self.lora.lora_A.weight # [R, I, K, K]
                
                # Einstein summation: O,R * R,I,K,K -> O,I,K,K
                delta_w = torch.einsum('or,rikh->oikh', weight_B, weight_A) * self.lora.scaling
                
                self.conv.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        if not self.merged:
            return
        
        if self.conv.groups > 1:
            return
            
        if self.lora.rank > 0:
            with torch.no_grad():
                weight_B = self.lora.lora_B.weight.squeeze(3).squeeze(2)
                weight_A = self.lora.lora_A.weight
                delta_w = torch.einsum('or,rikh->oikh', weight_B, weight_A) * self.lora.scaling
                self.conv.weight.data -= delta_w
            self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.conv(x)

        return self.conv(x) + self.lora(x)
    
    def __repr__(self):
        return (
            f"Conv2dWithLoRA(in_channels={self.conv.in_channels}, "
            f"out_channels={self.conv.out_channels}, "
            f"kernel_size={self.conv.kernel_size}, "
            f"rank={self.lora.rank}, "
            f"alpha={self.lora.alpha})"
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """
        Handle backward compatibility for loading standard Conv2d weights.
        """
        for param_name in ["weight", "bias"]:
            old_key = prefix + param_name
            new_key = prefix + "conv." + param_name
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
        
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
