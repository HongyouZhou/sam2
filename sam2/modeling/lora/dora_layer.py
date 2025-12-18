# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
DoRA layer implementations.

Separated from standard LoRA to avoid mixing concerns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_layer import LoRALayer, LoRAConv2dLayer


class LinearWithDoRA(nn.Module):
    """
    DoRA: Weight-Decomposed Low-Rank Adaptation.

    W = m * (W0 + BA) / ||W0 + BA||
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.linear = original_linear
        for param in self.linear.parameters():
            param.requires_grad = False

        self.lora = LoRALayer(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        with torch.no_grad():
            weight_norm = self.linear.weight.norm(p=2, dim=1).detach()
        self.m = nn.Parameter(weight_norm)

        self.merged = False
        self.w0_cache = None

    def merge(self):
        if self.merged:
            return

        if self.lora.rank > 0:
            with torch.no_grad():
                delta_w = (self.lora.lora_B.weight @ self.lora.lora_A.weight) * self.lora.scaling
                V = self.linear.weight + delta_w
                V_norm = V.norm(p=2, dim=1, keepdim=True)
                V_normalized = V / (V_norm + 1e-6)
                W_final = self.m.view(-1, 1) * V_normalized
                self.w0_cache = self.linear.weight.detach().clone()
                self.linear.weight.data = W_final

            self.merged = True

    def unmerge(self):
        if not self.merged:
            return

        if self.w0_cache is not None:
            with torch.no_grad():
                self.linear.weight.data = self.w0_cache
                self.w0_cache = None
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.linear(x)

        # Note: This implementation applies the DoRA weight construction directly.
        # It effectively ignores the dropout in the LoRA layer because we don't
        # compute W0*x + BA*dropout(x), but rather (W0 + BA)*x.
        # This is consistent with DoRA as a weight reparameterization method.
        
        delta_w = (self.lora.lora_B.weight @ self.lora.lora_A.weight) * self.lora.scaling
        V = self.linear.weight + delta_w
        V_norm = V.norm(p=2, dim=1, keepdim=True)
        V_normalized = V / (V_norm + 1e-6)
        W_final = self.m.view(-1, 1) * V_normalized
        return F.linear(x, W_final, self.linear.bias)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for param_name in ["weight", "bias"]:
            old_key = prefix + param_name
            new_key = prefix + "linear." + param_name
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class Conv2dWithDoRA(nn.Module):
    """DoRA for Conv2d layers."""

    def __init__(
        self,
        original_conv: nn.Conv2d,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if original_conv.groups > 1:
            raise ValueError("DoRA does not support grouped convolutions (groups > 1).")

        self.conv = original_conv
        for param in self.conv.parameters():
            param.requires_grad = False

        self.lora = LoRAConv2dLayer(
            in_channels=original_conv.in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size[0],
            stride=original_conv.stride[0],
            padding=original_conv.padding[0],
            dilation=original_conv.dilation[0],
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        with torch.no_grad():
            weight_norm = self.conv.weight.norm(p=2, dim=(1, 2, 3)).detach()
        self.m = nn.Parameter(weight_norm)

        self.merged = False
        self.w0_cache = None

    def merge(self):
        if self.merged:
            return

        if self.conv.groups > 1:
            return

        if self.lora.rank > 0:
            with torch.no_grad():
                weight_B = self.lora.lora_B.weight.squeeze(3).squeeze(2)
                weight_A = self.lora.lora_A.weight
                delta_w = torch.einsum('or,rikh->oikh', weight_B, weight_A) * self.lora.scaling
                V = self.conv.weight + delta_w
                V_norm = V.norm(p=2, dim=(1, 2, 3), keepdim=True)
                V_normalized = V / (V_norm + 1e-6)
                W_final = self.m.view(-1, 1, 1, 1) * V_normalized
                self.w0_cache = self.conv.weight.detach().clone()
                self.conv.weight.data = W_final

            self.merged = True

    def unmerge(self):
        if not self.merged:
            return

        if self.w0_cache is not None:
            with torch.no_grad():
                self.conv.weight.data = self.w0_cache
                self.w0_cache = None
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.conv(x)

        weight_B = self.lora.lora_B.weight.squeeze(3).squeeze(2)
        weight_A = self.lora.lora_A.weight
        delta_w = torch.einsum('or,rikh->oikh', weight_B, weight_A) * self.lora.scaling
        V = self.conv.weight + delta_w
        V_norm = V.norm(p=2, dim=(1, 2, 3), keepdim=True)
        V_normalized = V / (V_norm + 1e-6)
        W_final = self.m.view(-1, 1, 1, 1) * V_normalized
        return F.conv2d(
            x, W_final, self.conv.bias,
            self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for param_name in ["weight", "bias"]:
            old_key = prefix + param_name
            new_key = prefix + "conv." + param_name
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
