"""
Non-adversarial style augmentation baselines for domain generalization.

MixStyle (Zhou et al., ICLR 2021): Mix instance-level feature statistics within a batch.
DSU (Li et al., ICLR 2022): Perturb channel-wise statistics with Gaussian noise.

Both are training-only augmentations applied to backbone FPN features [B, C, H, W].
"""

import random

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """MixStyle: mix instance-level feature statistics within a batch (Zhou et al., ICLR 2021)."""

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p:
            return x

        B, C, H, W = x.shape
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sigma = (var + self.eps).sqrt()
        x_normed = (x - mu) / sigma

        # Shuffle batch and mix statistics
        perm = torch.randperm(B, device=x.device)
        mu2, sigma2 = mu[perm], sigma[perm]

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu2
        sigma_mix = lam * sigma + (1 - lam) * sigma2

        return x_normed * sigma_mix + mu_mix


class DSU(nn.Module):
    """DSU: Distribution Shift-aware Uncertainty (Li et al., ICLR 2022)."""

    def __init__(self, p: float = 0.5, alpha: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p:
            return x

        B, C, H, W = x.shape
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sigma = (var + self.eps).sqrt()
        x_normed = (x - mu) / sigma

        # Batch-level variance of instance statistics
        mu_var = mu.var(dim=0, keepdim=True)
        sigma_var = sigma.var(dim=0, keepdim=True)

        # Perturb with Gaussian noise scaled by batch variance
        mu_noise = torch.randn_like(mu) * (mu_var + self.eps).sqrt() * self.alpha
        sigma_noise = torch.randn_like(sigma) * (sigma_var + self.eps).sqrt() * self.alpha

        return x_normed * (sigma + sigma_noise) + (mu + mu_noise)
