# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Distribution Matching Module for Uncertainty Calibration.

This module provides implementations of various distribution matching methods
for aligning uncertainty and error distributions in zero-shot scenarios:

- MMD (Maximum Mean Discrepancy): Kernel-based two-sample test
- CKA (Centered Kernel Alignment): Scale-invariant representation similarity
- Gram Matrix Matching: Direct covariance structure alignment

All methods support both global (pixel-level) and patch-based computation.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ============================================================================
# Auxiliary Functions (shared by multiple methods)
# ============================================================================

def extract_patches(
    tensor: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, int]:
    """
    Extract non-overlapping patches from 2D tensor.
    
    Args:
        tensor: [B, H, W] input tensor
        patch_size: size of square patches
    
    Returns:
        patches: [N, D] where N = B * num_patches, D = patch_size²
        n_total_patches: total number of patches extracted
    """
    B, H, W = tensor.shape
    device = tensor.device
    
    # Calculate number of patches
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size
    
    # Handle case where image size is not divisible by patch_size
    if n_patches_h == 0 or n_patches_w == 0:
        # Return flattened tensor as single "patch"
        return tensor.flatten().unsqueeze(-1), tensor.numel()
    
    # Crop to make divisible by patch_size
    H_crop = n_patches_h * patch_size
    W_crop = n_patches_w * patch_size
    tensor_crop = tensor[:, :H_crop, :W_crop]
    
    # Reshape to patches: [B, n_patches_h, patch_size, n_patches_w, patch_size]
    patches = tensor_crop.reshape(B, n_patches_h, patch_size, n_patches_w, patch_size)
    
    # Rearrange to [B, n_patches_h, n_patches_w, patch_size, patch_size]
    patches = patches.permute(0, 1, 3, 2, 4)
    
    # Flatten to [N, D] where N = total patches, D = patch_size²
    n_total_patches = B * n_patches_h * n_patches_w
    patches = patches.reshape(n_total_patches, patch_size * patch_size)
    
    return patches, n_total_patches


def rbf_kernel_matrix(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    """
    Compute RBF (Gaussian) kernel matrix K[i,j] = exp(-||x_i - x_j||² / (2σ²)).
    
    Memory-efficient implementation using mathematical expansion.
    
    Args:
        x1: [N, D] first set of samples
        x2: [M, D] second set of samples
        bandwidth: kernel bandwidth σ
    
    Returns:
        K: [N, M] kernel matrix
    """
    # Clamp inputs to reasonable range to prevent overflow
    x1 = torch.clamp(x1, min=-10.0, max=10.0)
    x2 = torch.clamp(x2, min=-10.0, max=10.0)
    
    # Compute squared norms
    x1_norm_sq = torch.sum(x1 ** 2, dim=1, keepdim=True)  # [N, 1]
    x2_norm_sq = torch.sum(x2 ** 2, dim=1, keepdim=True)  # [M, 1]
    
    # Compute inner product
    inner_prod = torch.mm(x1, x2.t())  # [N, M]
    
    # Compute squared distances: ||x-y||² = ||x||² + ||y||² - 2<x,y>
    dist_sq = x1_norm_sq + x2_norm_sq.t() - 2 * inner_prod  # [N, M]
    
    # Clamp to avoid negative values due to numerical errors
    dist_sq = torch.clamp(dist_sq, min=0.0)
    
    # RBF kernel with numerical stability
    sigma_sq = bandwidth ** 2 + 1e-8
    exponent = -dist_sq / (2 * sigma_sq)
    exponent = torch.clamp(exponent, min=-50.0, max=50.0)
    
    return torch.exp(exponent)


def center_kernel_matrix(K: torch.Tensor) -> torch.Tensor:
    """
    Center kernel matrix: K_centered = H @ K @ H where H = I - 1/N.
    
    Used in CKA with RBF kernels to ensure centering.
    
    Args:
        K: [N, N] kernel matrix
    
    Returns:
        K_centered: [N, N] centered kernel matrix
    """
    N = K.shape[0]
    device = K.device
    dtype = K.dtype
    
    # Create centering matrix H = I - 1/N * ones(N, N)
    H = torch.eye(N, device=device, dtype=dtype) - torch.ones(N, N, device=device, dtype=dtype) / N
    
    # Center: H @ K @ H
    K_centered = torch.mm(torch.mm(H, K), H)
    
    return K_centered


# ============================================================================
# MMD Computer (Maximum Mean Discrepancy)
# ============================================================================

class MMDComputer:
    """
    Maximum Mean Discrepancy computer.
    
    MMD is a kernel-based two-sample test that measures the distance between
    two probability distributions using kernel embeddings.
    
    References:
        - Gretton et al. (2012): "A Kernel Two-Sample Test"
        - Long et al. (2015): "Learning Transferable Features with DAN"
    """
    
    @staticmethod
    def compute_mmd(
        x: torch.Tensor,
        y: torch.Tensor,
        kernel: str = 'rbf',
        bandwidth: float = 0.1,
        batch_size: int = 256,
        n_batches: int = 10,
    ) -> torch.Tensor:
        """
        Compute MMD using mini-batch estimation.
        
        MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        
        Uses mini-batch sampling to avoid OOM while maintaining unbiased estimation.
        Memory: O(batch_size²) instead of O(N²)
        
        Args:
            x: [N, D] samples from distribution P (uncertainty)
            y: [M, D] samples from distribution Q (error)
            kernel: 'rbf' (Gaussian kernel, recommended)
            bandwidth: kernel bandwidth (σ in RBF kernel)
            batch_size: size of mini-batch for each iteration
            n_batches: number of mini-batches to average
        
        Returns:
            mmd: scalar MMD value (≥0)
        """
        N, M = x.shape[0], y.shape[0]
        
        # Safety check: need at least 2 samples
        if N < 2 or M < 2:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        # Adjust batch_size if dataset is smaller
        actual_batch_size = max(2, min(batch_size, N, M))
        
        # Mini-batch MMD estimation
        mmd_sum = 0.0
        
        for _ in range(n_batches):
            # Randomly sample mini-batches
            idx_x = torch.randperm(N, device=x.device)[:actual_batch_size]
            idx_y = torch.randperm(M, device=y.device)[:actual_batch_size]
            
            x_batch = x[idx_x]
            y_batch = y[idx_y]
            
            # Compute kernel matrices for this mini-batch
            k_xx = rbf_kernel_matrix(x_batch, x_batch, bandwidth)  # [B, B]
            k_yy = rbf_kernel_matrix(y_batch, y_batch, bandwidth)  # [B, B]
            k_xy = rbf_kernel_matrix(x_batch, y_batch, bandwidth)  # [B, B]
            
            # Unbiased MMD estimator: exclude diagonal terms
            b = k_xx.shape[0]
            
            # Sum all elements, subtract diagonal, normalize by b(b-1)
            k_xx_sum = k_xx.sum() - k_xx.diagonal().sum()
            k_yy_sum = k_yy.sum() - k_yy.diagonal().sum()
            k_xy_sum = k_xy.sum()
            
            # Unbiased estimator for this batch
            eps = 1e-8
            mmd_squared_batch = (
                k_xx_sum / (b * (b - 1) + eps) +
                k_yy_sum / (b * (b - 1) + eps) -
                2 * k_xy_sum / (b * b + eps)
            )
            
            # Accumulate
            mmd_sum += mmd_squared_batch
            
            # Free memory
            del k_xx, k_yy, k_xy
        
        # Average over batches
        mmd_squared_avg = mmd_sum / (n_batches + 1e-8)
        
        # Clamp before sqrt
        mmd_squared_avg = torch.clamp(mmd_squared_avg, min=0.0, max=10.0)
        
        # Take square root with epsilon for gradient stability at 0
        mmd = torch.sqrt(mmd_squared_avg + 1e-8)
        
        return mmd
    
    @staticmethod
    def compute_patch_based_mmd(
        uncertainty: torch.Tensor,
        error: torch.Tensor,
        patch_size: int = 16,
        kernel: str = 'rbf',
        bandwidth: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute patch-based MMD for local distribution matching.
        
        Instead of flattening all pixels, we:
        1. Extract patches from uncertainty and error maps
        2. Compute patch-level statistics (mean, std)
        3. Apply MMD on patch statistics for better local alignment
        
        This approach:
        - Preserves spatial locality
        - Reduces computational cost (fewer samples)
        - Better captures local uncertainty-error relationships
        
        Args:
            uncertainty: [B, H, W] uncertainty map
            error: [B, H, W] error map
            patch_size: size of square patches
            kernel: kernel type for MMD
            bandwidth: kernel bandwidth
        
        Returns:
            mmd_loss: scalar MMD value
        """
        # Extract patches
        u_patches, n_total_patches = extract_patches(uncertainty, patch_size)
        e_patches, _ = extract_patches(error, patch_size)
        
        # If extraction failed (too few patches), fallback to global MMD
        if n_total_patches <= 1:
            return MMDComputer.compute_mmd(
                uncertainty.flatten().unsqueeze(-1),
                error.flatten().unsqueeze(-1),
                kernel=kernel,
                bandwidth=bandwidth,
            )
        
        # Compute patch-level statistics as features
        # Using mean and std to capture distribution within each patch
        u_mean = u_patches.mean(dim=1, keepdim=True)  # [N, 1]
        u_std = u_patches.std(dim=1, keepdim=True)    # [N, 1]
        e_mean = e_patches.mean(dim=1, keepdim=True)  # [N, 1]
        e_std = e_patches.std(dim=1, keepdim=True)    # [N, 1]
        
        # Concatenate statistics as patch features
        uncertainty_features = torch.cat([u_mean, u_std], dim=1)  # [N, 2]
        error_features = torch.cat([e_mean, e_std], dim=1)        # [N, 2]
        
        # Compute MMD on patch features
        mmd_loss = MMDComputer.compute_mmd(
            uncertainty_features,
            error_features,
            kernel=kernel,
            bandwidth=bandwidth,
            batch_size=min(256, n_total_patches // 2),
            n_batches=10,
        )
        
        return mmd_loss


# ============================================================================
# Hard-Aware Pixel-based MMD Computer
# ============================================================================

class HardAwareMMD:
    """
    SOTA Implementation: Hard-Aware Pixel-based MMD for Uncertainty Calibration.
    
    Paper Contribution Points:
    1. Focuses on 'Epistemic Risk Areas' via Hard Patch Mining.
    2. Aligns full pixel-value histograms instead of compressed moments (mean/std).
    3. Maintains constant memory footprint via Stochastic Subsampling.
    """

    @staticmethod
    def compute_loss(
        uncertainty: torch.Tensor,
        error: torch.Tensor,
        patch_size: int = 16,
        top_k_percent: float = 0.25,  # 只关注前 25% 最难的区域
        max_samples: int = 4096,      # 显存优化：最大采样像素数 (4096^2 的矩阵约占显存 64MB)
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
    ) -> torch.Tensor:
        """
        Args:
            uncertainty: [B, H, W] Predicted uncertainty map (0~1)
            error: [B, H, W] Actual error map (0~1)
            patch_size: Size of local regions
            top_k_percent: Ratio of hard patches to mine
            max_samples: Max pixels to sample for MMD kernel computation
        """
        B, H, W = uncertainty.shape
        device = uncertainty.device
        
        # 1. Patch Extraction (提取 Patch)
        # [N_total, Patch_Area]
        u_patches, n_patches = HardAwareMMD._extract_patches(uncertainty, patch_size)
        e_patches, _ = HardAwareMMD._extract_patches(error, patch_size)
        
        # 2. Hard Patch Mining (困难挖掘)
        # 使用 detach 的 Error 计算难度，避免梯度流向索引选择机制
        patch_difficulty = e_patches.detach().mean(dim=1)  # [N_total]
        
        # 确定挖掘数量 k
        k = int(n_patches * top_k_percent)
        k = max(k, 2) # 至少选2个
        
        # 选出 Top-K 最难 Patch 的索引
        _, top_indices = torch.topk(patch_difficulty, k)
        
        # Gather data: 只取困难区域的数据
        u_hard = u_patches[top_indices] # [K, Patch_Area]
        e_hard = e_patches[top_indices] # [K, Patch_Area]
        
        # 3. Flatten to Pixel Space (展平到像素空间)
        # 我们将困难区域的所有像素视为一个分布集合
        u_samples = u_hard.flatten().unsqueeze(1) # [N_pixels, 1]
        e_samples = e_hard.flatten().unsqueeze(1) # [N_pixels, 1]
        
        # 4. Stochastic Subsampling (随机采样优化显存)
        # 如果像素点太多，随机采样一部分来计算 MMD
        num_pixels = u_samples.size(0)
        if num_pixels > max_samples:
            # 使用相同的随机排列，保持 U 和 E 在空间采样上的一致性 (虽然 MMD 是集合度量)
            perm = torch.randperm(num_pixels, device=device)[:max_samples]
            u_samples = u_samples[perm]
            e_samples = e_samples[perm]
            
        # 5. Compute Gaussian Kernel MMD (计算高斯核 MMD)
        # Optimization 3: Use gradient checkpointing to save memory during backward pass
        # This trades computation for memory by recomputing activations during backward
        loss = checkpoint(
            HardAwareMMD._gaussian_kernel_mmd,
            u_samples, 
            e_samples, 
            kernel_mul,
            kernel_num,
            use_reentrant=False
        )
        
        return loss

    @staticmethod
    def _extract_patches(tensor: torch.Tensor, patch_size: int):
        """Helper to extract non-overlapping patches efficiently."""
        B, H, W = tensor.shape
        
        # 简单的裁剪逻辑，确保能整除 (对应 1024x1024 输入，patch_size=16 无损)
        h_crop = (H // patch_size) * patch_size
        w_crop = (W // patch_size) * patch_size
        tensor = tensor[:, :h_crop, :w_crop]
        
        # [B, H//P, P, W//P, P] -> [B, H//P, W//P, P, P]
        patches = tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(B, -1, patch_size * patch_size)
        
        # Flatten batch dimension: [B*N_patches, Patch_Area]
        patches = patches.view(-1, patch_size * patch_size)
        return patches, patches.shape[0]

    @staticmethod
    def _gaussian_kernel_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        Computes MMD with Multi-Scale RBF Kernels.
        Memory-optimized implementation using chunking to avoid O(N^2) memory.
        
        Optimizations:
        1. Chunked computation to keep peak memory low (O(B^2) instead of O(N^2))
        2. Subsampling for bandwidth estimation
        3. Avoids large intermediate matrices
        """
        B_source = source.size(0)
        B_target = target.size(0)
        
        # Bandwidth estimation using subsampling to save memory
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            # Use at most 1024 samples for bandwidth estimation
            n_samples = B_source + B_target
            n_subset = min(n_samples, 1024)
            
            # Simple stratified sampling
            n_s = min(B_source, n_subset // 2)
            n_t = min(B_target, n_subset - n_s)
            
            # Use randperm for random subset
            idx_s = torch.randperm(B_source, device=source.device)[:n_s]
            idx_t = torch.randperm(B_target, device=target.device)[:n_t]
            
            s_sub = source[idx_s]
            t_sub = target[idx_t]
            total_sub = torch.cat([s_sub, t_sub], dim=0)
            
            # Compute approximate bandwidth
            # ||x - y||^2
            L2_dist_sub = (total_sub.unsqueeze(1) - total_sub.unsqueeze(0)).pow(2).sum(dim=-1)
            denominator = max(1.0, float(total_sub.shape[0]**2 - total_sub.shape[0]))
            bandwidth = torch.sum(L2_dist_sub.detach()) / (denominator + 1e-8)
        
        bandwidth = torch.clamp(bandwidth, min=1e-6)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

        # Chunked computation helper
        def compute_kernel_sum(X, Y, b_list, chunk_size=1024):
            total_sum = 0.0
            N = X.shape[0]
            M = Y.shape[0]
            
            # Loop over chunks
            for i in range(0, N, chunk_size):
                X_chunk = X[i:i+chunk_size]
                for j in range(0, M, chunk_size):
                    Y_chunk = Y[j:j+chunk_size]
                    
                    # Pairwise distance [Bi, Bj]
                    # (Bi, 1, D) - (1, Bj, D)
                    dist_chunk = (X_chunk.unsqueeze(1) - Y_chunk.unsqueeze(0)).pow(2).sum(dim=-1)
                    
                    # Sum multi-scale kernels
                    k_val = 0.0
                    for b_val in b_list:
                         k_val = k_val + torch.exp(-dist_chunk / (b_val + 1e-8))
                    
                    total_sum = total_sum + k_val.sum()
            return total_sum

        # Calculate sums (memory efficient)
        sum_XX = compute_kernel_sum(source, source, bandwidth_list)
        sum_YY = compute_kernel_sum(target, target, bandwidth_list)
        sum_XY = compute_kernel_sum(source, target, bandwidth_list)
        
        # MMD = E[K(X,X)] + E[K(Y,Y)] - 2E[K(X,Y)]
        if B_source == B_target:
             # Standard case: mean over total pairs
             loss = (sum_XX + sum_YY - 2 * sum_XY) / (B_source * B_source)
        else:
             # Generalized MMD for unequal batch sizes
             term1 = sum_XX / (B_source * B_source)
             term2 = sum_YY / (B_target * B_target)
             term3 = sum_XY / (B_source * B_target)
             loss = term1 + term2 - 2 * term3
             
        return torch.clamp(loss, min=0.0)


# ============================================================================
# Helper Functions for Domain-Aware Mining
# ============================================================================

def extract_patches_3d(
    tensor: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, int]:
    """
    Extract non-overlapping patches from 3D/4D feature tensor.
    
    Args:
        tensor: [B, C, H, W] input feature map
        patch_size: size of square patches
    
    Returns:
        patches: [N, C, Ph, Pw] where N = B * num_patches
        n_total_patches: total number of patches extracted
    """
    B, C, H, W = tensor.shape
    
    # Calculate number of patches
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size
    
    # Handle case where size is not divisible
    if n_patches_h == 0 or n_patches_w == 0:
        # Return as single patch
        return tensor.unsqueeze(0), 1
    
    # Crop to make divisible
    H_crop = n_patches_h * patch_size
    W_crop = n_patches_w * patch_size
    tensor_crop = tensor[:, :, :H_crop, :W_crop]
    
    # Extract patches: [B, C, H//P, P, W//P, P]
    patches = tensor_crop.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    
    # Rearrange to [B, H//P, W//P, C, P, P]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    
    # Flatten to [N, C, P, P]
    n_total_patches = B * n_patches_h * n_patches_w
    patches = patches.view(n_total_patches, C, patch_size, patch_size)
    
    return patches, n_total_patches


def compute_feature_diversity(
    feature_patches: torch.Tensor,
    method: str = "channel_std",
) -> torch.Tensor:
    """
    Compute feature diversity for each patch as a proxy for epistemic uncertainty.
    
    Theory: High feature diversity indicates encoder confusion, suggesting the
    input is near the boundary of the training distribution (OOD-like).
    
    Args:
        feature_patches: [N, C, Ph, Pw] feature patches
        method: "channel_std" (default) |  "spatial_var"
    
    Returns:
        diversity: [N] diversity score per patch
    
    References:
        - Hendrycks et al. (2019): "Using Self-Supervised Learning Can Improve
          Model Robustness and Uncertainty"
        - Lee et al. (2018): "A Simple Unified Framework for Detecting OOD"
    """
    if method == "channel_std":
        # Channel-wise standard deviation (encoder confusion across channels)
        channel_std = feature_patches.std(dim=1)  # [N, Ph, Pw]
        diversity = channel_std.mean(dim=(1, 2))  # [N]
    elif method == "spatial_var":
        # Spatial variance (patch uniformity)
        spatial_var = feature_patches.var(dim=(2, 3))  # [N, C]
        diversity = spatial_var.mean(dim=1)  # [N]
    else:
        raise ValueError(f"Unknown diversity_method: {method}")
    
    return diversity


# ============================================================================
# Domain-Aware Soft MMD (Theory-Driven Approach)
# ============================================================================

class DomainAwareSoftMMD:
    """
    Theoretically grounded hard mining combining:
    1. Domain-awareness (feature diversity for OOD detection)
    2. Soft weighting (information-theoretic optimal allocation)
    
    Key improvements over HardAwareMMD:
    - Value normalization: ensures error and diversity are on same scale
    - Correlation monitoring: validates complementary information
    - Explicit feature source: uses SAM2 encoder last layer
    - Soft weighting: MaxEnt principle (no arbitrary Top-K cutoff)
    
    Theoretical foundations:
    - OOD Detection: Lee et al. (2018), Hendrycks & Gimpel (2017)
    - Epistemic Uncertainty: Kendall & Gal (2017)
    - Maximum Entropy: Jaynes (1957), Hinton et al. (2015)
    """
    
    @staticmethod
    def compute_loss(
        uncertainty: torch.Tensor,
        error: torch.Tensor,
        feature_map: torch.Tensor,
        patch_size: int = 16,
        diversity_weight: float = 0.4,
        temperature: float = 0.1,
        max_samples: int = 4096,
        diversity_method: str = "channel_std",
        enable_monitoring: bool = False,
        tag: str = "unknow",
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute domain-aware soft MMD loss.
        
        Args:
            uncertainty: [B, H, W] predicted uncertainty
            error: [B, H, W] ground truth error
            feature_map: [B, C, H, W] from SAM2 Image Encoder's LAST layer
                        (vision_features[-1] from backbone_fpn, typically 256 channels)
                        Must be spatially aligned with uncertainty/error maps
            patch_size: Size of square patches (default: 16)
            diversity_weight: Weight for feature diversity component (0.0-1.0)
                             0.0 = pure error-based (aleatoric only)
                             0.4 = balanced (60% aleatoric, 40% epistemic)
                             1.0 = pure diversity-based (epistemic only)
            temperature: Softmax temperature for soft weighting
                        Lower T = more focused on hard patches
                        Higher T = more uniform distribution
            max_samples: Maximum number of pixels to sample (memory optimization)
            diversity_method: "channel_std" or "spatial_var"
            enable_monitoring: Enable correlation monitoring for validation
            tag: Debug string to identify caller (e.g. "clean" or "augmented")
        
        Returns:
            mmd_loss: Scalar MMD loss value
            metrics: Dict of metrics
        """
        import logging
        
        B, H, W = uncertainty.shape
        device = uncertainty.device
        
        # Optimization: Detach feature map to prevent graph retention
        # Feature map is used for weighting/sampling (non-differentiable), so gradients are not needed
        if feature_map is not None:
            feature_map = feature_map.detach()
        
        # Safeguard 1: Feature map is provided
        if feature_map is None:
            raise ValueError(
                f"[{tag}] domain_aware_soft_mmd requires feature_map. "
                "Ensure backbone features are passed from forward_image()."
            )
        
        # Safeguard 2: Spatial alignment
        if feature_map.shape[-2:] != (H, W):
            logging.warning(
                f"[{tag}] Feature map size {feature_map.shape[-2:]} != error size {(H, W)}. "
                f"Auto-interpolating feature map."
            )
            feature_map = F.interpolate(
                feature_map,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        
        # 1. Extract patches
        u_patches, n_patches = extract_patches(uncertainty, patch_size)  # [N, D]
        e_patches, _ = extract_patches(error, patch_size)  # [N, D]
        f_patches, _ = extract_patches_3d(feature_map, patch_size)  # [N, C, Ph, Pw]
        
        # 2. Compute domain-aware difficulty
        # (A) Aleatoric difficulty (error-based)
        error_difficulty = e_patches.mean(dim=1)  # [N]
        
        # (B) Epistemic difficulty (feature diversity)
        feature_diversity = compute_feature_diversity(f_patches, method=diversity_method)  # [N]
        
        
        # === CRITICAL: Normalize to [0, 1] for fair combination ===
        # Min-max normalization
        e_min, e_max = error_difficulty.min(), error_difficulty.max()
        f_min, f_max = feature_diversity.min(), feature_diversity.max()
        
        # Safeguard 4: Normalization validity
        if (e_max - e_min) < 1e-8:
            logging.warning("Error has near-zero range, using uniform weights")
            error_norm = torch.ones_like(error_difficulty) * 0.5
        else:
            error_norm = (error_difficulty - e_min) / (e_max - e_min + 1e-6)
        
        if (f_max - f_min) < 1e-8:
            logging.warning("Feature diversity has near-zero range, falling back to error-only")
            diversity_norm = torch.zeros_like(feature_diversity)
        else:
            diversity_norm = (feature_diversity - f_min) / (f_max - f_min + 1e-6)
        
        # === Optional: Monitor correlation for validation ===
        if enable_monitoring:
            # Pearson correlation
            if error_norm.numel() > 1 and diversity_norm.numel() > 1:
                try:
                    correlation = torch.corrcoef(
                        torch.stack([error_norm, diversity_norm])
                    )[0, 1].item()
                    
                    # Log for debugging
                    logging.info(
                        f"[DomainAware] Error-Diversity correlation: {correlation:.3f} "
                        f"(Error range: [{e_min:.3f}, {e_max:.3f}], "
                        f"Diversity range: [{f_min:.3f}, {f_max:.3f}])"
                    )
                    
                    # Expected: 0.2 < correlation < 0.6 (complementary info)
                    if correlation > 0.8:
                        logging.warning(
                            f"High correlation ({correlation:.3f}) detected! "
                            "Feature diversity may not provide additional information."
                        )
                    elif correlation < 0.1:
                        logging.warning(
                            f"Low correlation ({correlation:.3f}) detected! "
                            "Check if using correct feature layer (should be vision_features[-1])."
                        )
                except Exception as e:
                    logging.debug(f"Could not compute correlation: {e}")
        
        # (C) Combined difficulty (theory-driven, normalized)
        combined_difficulty = (
            (1 - diversity_weight) * error_norm +
            diversity_weight * diversity_norm
        )
        
        # Metrics for TensorBoard
        metrics = {}
        if error_norm.numel() > 1 and diversity_norm.numel() > 1:
            try:
                correlation = torch.corrcoef(
                    torch.stack([error_norm.detach(), diversity_norm.detach()])
                )[0, 1].item()
                metrics['error_diversity_corr'] = correlation
            except Exception:
                pass
        
        metrics['error_norm_mean'] = error_norm.mean().item()
        metrics['diversity_norm_mean'] = diversity_norm.mean().item()
        
        # 4. Soft weighting (MaxEnt principle) + Smart sampling
        # Boltzmann distribution on NORMALIZED difficulty
        weights = F.softmax(combined_difficulty / temperature, dim=0)  # [N]
        
        # Calculate optimal number of patches to sample
        # Goal: After flattening, total pixels should be <= max_samples
        # pixels_per_patch = patch_size^2
        # n_patches_to_sample = max_samples / pixels_per_patch
        pixels_per_patch = patch_size * patch_size
        n_patches_to_sample = max(1, max_samples // pixels_per_patch)  # At least 1 patch
        
        # Clamp to available patches
        n_patches_to_sample = min(n_patches_to_sample, u_patches.shape[0])
        
        # Single-stage weighted sampling
        if u_patches.shape[0] > n_patches_to_sample:
            # Sample according to difficulty weights
            sampled_indices = torch.multinomial(
                weights,
                num_samples=n_patches_to_sample,
                replacement=False
            )
            u_sampled = u_patches[sampled_indices]
            e_sampled = e_patches[sampled_indices]
        else:
            u_sampled = u_patches
            e_sampled = e_patches
        
        # 5. Compute MMD (using existing Gaussian kernel implementation)
        # After sampling, pixel count = n_patches_to_sample * patch_size^2 <= max_samples
        u_flat = u_sampled.flatten().unsqueeze(1)  # [M, 1]
        e_flat = e_sampled.flatten().unsqueeze(1)  # [M, 1]
        
        # Optimization 3: Use gradient checkpointing to save memory during backward pass
        mmd_loss = checkpoint(
            HardAwareMMD._gaussian_kernel_mmd,
            u_flat,
            e_flat,
            2.0,  # kernel_mul
            5,    # kernel_num
            use_reentrant=False
        )
        
        return mmd_loss, metrics


# ============================================================================
# CKA Computer (Centered Kernel Alignment)
# ============================================================================

class CKAComputer:
    """
    Centered Kernel Alignment computer.
    
    CKA is a scale-invariant similarity measure for representations, ideal for
    zero-shot scenarios where uncertainty and error scales may differ.
    
    Key properties:
    - Scale-invariant: CKA(αX, Y) = CKA(X, Y)
    - Translation-invariant: via centering
    - Captures correlation structure, not absolute values
    
    Reference:
        Kornblith et al. (2019): "Similarity of Neural Network Representations
        Revisited", ICML 2019
    """
    
    @staticmethod
    def compute_linear_cka_direct(
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Direct linear CKA computation (for moderate N).
        
        Linear CKA with centered features:
            CKA = <XX^T, YY^T>_F / (||XX^T||_F * ||YY^T||_F)
                = tr(XX^T @ YY^T) / sqrt(tr((XX^T)²) * tr((YY^T)²))
        
        Can simplify using: tr(XX^T @ YY^T) = ||X^T Y||²_F
        
        Args:
            X: [N, D] centered uncertainty features
            Y: [N, D] centered error features
        
        Returns:
            cka: scalar CKA similarity ∈ [0, 1]
        """
        # Use trace properties to avoid explicit Gram matrix computation
        # HSIC(X, Y) = tr(XX^T @ YY^T) = ||X^T @ Y||²_F
        XTY = torch.mm(X.T, Y)  # [D, D]
        hsic_xy = (XTY ** 2).sum()
        
        # HSIC(X, X) = tr((XX^T)²) = ||X^T X||²_F
        XTX = torch.mm(X.T, X)  # [D, D]
        hsic_xx = (XTX ** 2).sum()
        
        # HSIC(Y, Y)
        YTY = torch.mm(Y.T, Y)  # [D, D]
        hsic_yy = (YTY ** 2).sum()
        
        # CKA
        cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)
        
        # Clamp to [0, 1] for numerical stability
        cka = torch.clamp(cka, min=0.0, max=1.0)
        
        return cka
    
    @staticmethod
    def compute_linear_cka_minibatch(
        X: torch.Tensor,
        Y: torch.Tensor,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """
        Mini-batch linear CKA for large N.
        
        Computes CKA in chunks to avoid OOM on large patch sets.
        Uses the fact that HSIC can be computed incrementally.
        
        Args:
            X: [N, D] centered uncertainty features
            Y: [N, D] centered error features
            batch_size: batch size for mini-batch computation
        
        Returns:
            cka: scalar CKA similarity ∈ [0, 1]
        """
        N, D = X.shape
        
        # Accumulate cross-products incrementally
        XTY = torch.zeros(D, D, device=X.device, dtype=X.dtype)
        XTX = torch.zeros(D, D, device=X.device, dtype=X.dtype)
        YTY = torch.zeros(D, D, device=X.device, dtype=X.dtype)
        
        n_batches = (N + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, N)
            
            X_batch = X[start:end]  # [B, D]
            Y_batch = Y[start:end]  # [B, D]
            
            # Accumulate cross-products
            XTY += torch.mm(X_batch.T, Y_batch)
            XTX += torch.mm(X_batch.T, X_batch)
            YTY += torch.mm(Y_batch.T, Y_batch)
        
        # Compute HSIC from accumulated matrices
        hsic_xy = (XTY ** 2).sum()
        hsic_xx = (XTX ** 2).sum()
        hsic_yy = (YTY ** 2).sum()
        
        # CKA
        cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)
        cka = torch.clamp(cka, min=0.0, max=1.0)
        
        return cka
    
    @staticmethod
    def compute_rbf_cka(
        X: torch.Tensor,
        Y: torch.Tensor,
        bandwidth: float = None,
    ) -> torch.Tensor:
        """
        RBF kernel CKA (non-linear, more expensive).
        
        Use when you suspect non-linear relationships between uncertainty and error.
        Falls back to linear CKA if N is too large due to O(N²) memory.
        
        Args:
            X: [N, D] centered uncertainty features
            Y: [N, D] centered error features
            bandwidth: RBF kernel bandwidth (if None, use median heuristic)
        
        Returns:
            cka: scalar CKA similarity ∈ [0, 1]
        """
        N = X.shape[0]
        
        # Memory limit: only use RBF for moderate N
        if N > 2000:
            # Too expensive, fall back to linear
            return CKAComputer.compute_linear_cka_direct(X, Y)
        
        # Compute bandwidth using median heuristic if not provided
        if bandwidth is None:
            bandwidth = CKAComputer._compute_median_bandwidth(X)
        
        # Compute RBF kernel matrices
        K_X = rbf_kernel_matrix(X, X, bandwidth)  # [N, N]
        K_Y = rbf_kernel_matrix(Y, Y, bandwidth)  # [N, N]
        
        # Center kernel matrices
        K_X = center_kernel_matrix(K_X)
        K_Y = center_kernel_matrix(K_Y)
        
        # Compute HSIC
        hsic_xy = (K_X * K_Y).sum()
        hsic_xx = (K_X * K_X).sum()
        hsic_yy = (K_Y * K_Y).sum()
        
        # CKA
        cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)
        cka = torch.clamp(cka, min=0.0, max=1.0)
        
        return cka
    
    @staticmethod
    def _compute_median_bandwidth(X: torch.Tensor) -> float:
        """Compute median pairwise distance as bandwidth (median heuristic)."""
        N = X.shape[0]
        
        # Sample a subset to avoid O(N²) computation
        if N > 1000:
            idx = torch.randperm(N, device=X.device)[:1000]
            X_sample = X[idx]
        else:
            X_sample = X
        
        # Compute pairwise distances
        dist_sq = torch.cdist(X_sample, X_sample, p=2) ** 2
        median_dist_sq = dist_sq.flatten().median()
        
        # Bandwidth = median distance
        bandwidth = torch.sqrt(median_dist_sq + 1e-8).item()
        
        return max(bandwidth, 0.01)  # ensure non-zero
    
    @staticmethod
    def compute_patch_based_cka(
        uncertainty: torch.Tensor,
        error: torch.Tensor,
        patch_size: int = 16,
        use_linear_kernel: bool = True,
        use_minibatch: bool = True,
        minibatch_size: int = 512,
    ) -> torch.Tensor:
        """
        Compute patch-based CKA for local structure matching.
        
        CKA is scale-invariant and measures the similarity of representation
        structures, making it ideal for zero-shot robustness where the scale
        of uncertainty and error may differ between train and test domains.
        
        Args:
            uncertainty: [B, H, W] uncertainty map in [0, 1]
            error: [B, H, W] error map in [0, 1]
            patch_size: size of square patches
            use_linear_kernel: if True, use linear kernel (K=XX^T, faster);
                              if False, use RBF kernel (captures non-linearity)
            use_minibatch: if True, use mini-batch computation for memory efficiency
            minibatch_size: batch size for mini-batch computation
        
        Returns:
            cka_loss: 1 - CKA similarity, range [0, 2] (0 = perfect alignment)
        """
        # Extract patches
        u_patches, n_total_patches = extract_patches(uncertainty, patch_size)
        e_patches, _ = extract_patches(error, patch_size)
        
        # If extraction failed, fallback to pixel-level
        if n_total_patches <= 1:
            u_flat = uncertainty.flatten().unsqueeze(-1)
            e_flat = error.flatten().unsqueeze(-1)
            return CKAComputer._compute_cka_core(
                u_flat, e_flat, use_linear_kernel, use_minibatch, minibatch_size
            )
        
        # Compute CKA on patch features
        cka_loss = CKAComputer._compute_cka_core(
            u_patches,
            e_patches,
            use_linear_kernel,
            use_minibatch,
            minibatch_size,
        )
        
        return cka_loss
    
    @staticmethod
    def _compute_cka_core(
        X: torch.Tensor,
        Y: torch.Tensor,
        use_linear_kernel: bool = True,
        use_minibatch: bool = True,
        minibatch_size: int = 512,
    ) -> torch.Tensor:
        """
        Core CKA computation.
        
        Args:
            X: [N, D] uncertainty patch features
            Y: [N, D] error patch features
            use_linear_kernel: linear (True) or RBF (False) kernel
            use_minibatch: use mini-batch computation for large N
            minibatch_size: batch size for mini-batch
        
        Returns:
            cka_loss: 1 - CKA (range [0, 2], 0 is best)
        """
        N = X.shape[0]
        
        # Safety check
        if N < 2:
            return torch.tensor(0.0, device=X.device, dtype=X.dtype)
        
        # Step 1: Center features (critical for translation invariance)
        X_centered = X - X.mean(dim=0, keepdim=True)
        Y_centered = Y - Y.mean(dim=0, keepdim=True)
        
        # Step 2: Compute CKA based on kernel choice
        if use_linear_kernel:
            if use_minibatch and N > minibatch_size:
                cka_similarity = CKAComputer.compute_linear_cka_minibatch(
                    X_centered, Y_centered, minibatch_size
                )
            else:
                cka_similarity = CKAComputer.compute_linear_cka_direct(
                    X_centered, Y_centered
                )
        else:
            # RBF kernel CKA
            cka_similarity = CKAComputer.compute_rbf_cka(X_centered, Y_centered)
        
        # Step 3: Convert similarity [0, 1] to loss
        cka_loss = 1.0 - cka_similarity
        
        return cka_loss


# ============================================================================
# Gram Computer (Gram Matrix Matching)
# ============================================================================

class GramComputer:
    """
    Gram matrix matching computer.
    
    Directly matches the Gram matrices (correlation structure) of uncertainty
    and error. Simpler than CKA but lacks scale invariance.
    
    Gram(X) = X @ X.T captures the correlation structure between samples.
    Loss = ||Gram(U) - Gram(E)||²
    
    Reference:
        Gatys et al. (2016): "Image Style Transfer Using CNN" (Neural Style Transfer)
    """
    
    @staticmethod
    def compute_gram_loss(
        X: torch.Tensor,
        Y: torch.Tensor,
        center: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute Gram matrix matching loss.
        
        Loss = ||Gram(X) - Gram(Y)||²_F / N²
        
        Args:
            X: [N, D] uncertainty features
            Y: [N, D] error features
            center: if True, center features (recommended for translation invariance)
            normalize: if True, normalize by number of elements (for stability)
        
        Returns:
            loss: scalar Gram matching loss
        """
        raise NotImplementedError(
            "Gram matrix matching is not yet implemented. "
            "Use 'mmd' or 'cka' for now. "
            "To implement Gram matching, uncomment and complete the code below."
        )
        
        # N = X.shape[0]
        # 
        # # Optional: center features
        # if center:
        #     X = X - X.mean(dim=0, keepdim=True)
        #     Y = Y - Y.mean(dim=0, keepdim=True)
        # 
        # # Compute Gram matrices
        # gram_X = torch.mm(X, X.T)  # [N, N]
        # gram_Y = torch.mm(Y, Y.T)  # [N, N]
        # 
        # # Frobenius norm: ||A - B||²_F = sum((A - B)²)
        # gram_diff = gram_X - gram_Y
        # loss = (gram_diff ** 2).sum()
        # 
        # # Normalize by N² for scale invariance
        # if normalize:
        #     loss = loss / (N * N)
        # 
        # return loss
    
    @staticmethod
    def compute_patch_based_gram(
        uncertainty: torch.Tensor,
        error: torch.Tensor,
        patch_size: int = 16,
    ) -> torch.Tensor:
        """
        Compute patch-based Gram matrix matching.
        
        Args:
            uncertainty: [B, H, W] uncertainty map
            error: [B, H, W] error map
            patch_size: size of square patches
        
        Returns:
            gram_loss: scalar Gram matching loss
        """
        raise NotImplementedError(
            "Patch-based Gram matching is not yet implemented. "
            "Use 'mmd' or 'cka' for now."
        )


# ============================================================================
# Unified Distribution Matcher Interface
# ============================================================================

class DistributionMatcher:
    """
    Unified interface for distribution matching methods.
    
    Supports switching between MMD, CKA, Gram matrix matching, and Hard-Aware MMD
    via a simple configuration parameter.
    
    Usage:
        matcher = DistributionMatcher(method='cka', patch_size=16)
        loss = matcher.compute_loss(uncertainty, error, use_patches=True)
    """
    
    def __init__(
        self,
        method: str = 'mmd',
        patch_size: int = 16,
        kernel: str = 'rbf',
        bandwidth: float = 0.1,
        cka_use_linear_kernel: bool = True,
        cka_use_minibatch: bool = True,
        cka_minibatch_size: int = 512,
        # Hard-Aware MMD parameters
        top_k_percent: float = 0.25,
        max_samples: int = 4096,
        # Domain-Aware Soft MMD parameters (NEW)
        diversity_weight: float = 0.4,
        temperature: float = 0.1,
        diversity_method: str = "channel_std",
        enable_monitoring: bool = False,
        # Checkpointing for memory optimization
        use_checkpoint: bool = True,
    ):
        """
        Initialize distribution matcher.
        
        Args:
            method: 'mmd' | 'cka' | 'gram' | 'mmd_hard_aware' | 'domain_aware_soft_mmd'
            patch_size: size of square patches for patch-based methods
            kernel: kernel type for MMD ('rbf')
            bandwidth: kernel bandwidth for MMD
            cka_use_linear_kernel: use linear kernel for CKA (faster)
            cka_use_minibatch: use mini-batch for CKA (memory efficient)
            cka_minibatch_size: batch size for CKA mini-batch computation
            diversity_weight: weight for feature diversity (domain_aware_soft_mmd only)
            temperature: softmax temperature (domain_aware_soft_mmd only)
            diversity_method: 'channel_std' | 'spatial_var' (domain_aware_soft_mmd only)
            enable_monitoring: enable correlation monitoring (domain_aware_soft_mmd only)
            use_checkpoint: enable gradient checkpointing to save memory (default: True)
        """
        valid_methods = ['mmd', 'cka', 'gram', 'mmd_hard_aware', 'domain_aware_soft_mmd']
        assert method in valid_methods, f"Unknown method: {method}. Valid: {valid_methods}"
        
        self.method = method
        self.patch_size = patch_size
        self.kernel = kernel
        self.bandwidth = bandwidth
        
        # CKA-specific parameters
        self.cka_use_linear_kernel = cka_use_linear_kernel
        self.cka_use_minibatch = cka_use_minibatch
        self.cka_minibatch_size = cka_minibatch_size
        
        # Hard-Aware MMD parameters
        self.top_k_percent = top_k_percent
        self.max_samples = max_samples
        
        # Domain-Aware Soft MMD parameters (NEW)
        self.diversity_weight = diversity_weight
        self.temperature = temperature
        self.diversity_method = diversity_method
        self.enable_monitoring = enable_monitoring
        
        # Checkpointing to save memory during backward pass
        self.use_checkpoint = use_checkpoint
    
    def compute_loss(
        self,
        uncertainty: torch.Tensor,
        error: torch.Tensor,
        use_patches: bool = True,
        feature_map: torch.Tensor | None = None,  # NEW: Required for domain_aware_soft_mmd
        tag: str = "unknown",  # NEW: Debug tag
    ) -> torch.Tensor:
        """
        Compute distribution matching loss with optional gradient checkpointing.
        
        Gradient checkpointing trades computation for memory by recomputing activations
        during backward pass instead of storing them. This is crucial for AUE training
        where peak memory usage can cause OOM.
        
        Args:
            uncertainty: [B, H, W] uncertainty map in [0, 1]
            error: [B, H, W] error map in [0, 1]
            use_patches: if True, use patch-based computation;
                        if False, use global (pixel-level) computation
            feature_map: [B, C, H, W] feature map for domain_aware_soft_mmd
            tag: Debug string to identify caller
        
        Returns:
            loss: scalar distribution matching loss
            metrics: dict of metrics
        """
        # Optimization: We removed the outer checkpoint wrapper because:
        # 1. It causes issues with returning (Tensor, Dict) tuples (TypeError: iteration over a 0-d tensor)
        # 2. The heavy lifting (kernel matrix computation) is already checkpointed internally 
        #    within DomainAwareSoftMMD and HardAwareMMD.
        return self._compute_loss_impl(
            uncertainty, error, use_patches, feature_map, tag
        )
    
    def _compute_loss_impl(
        self,
        uncertainty: torch.Tensor,
        error: torch.Tensor,
        use_patches: bool,
        feature_map: torch.Tensor | None,
        tag: str,
    ) -> torch.Tensor:
        """Internal implementation without checkpointing."""
        if self.method == 'mmd':
            return self._compute_mmd_loss(uncertainty, error, use_patches)
        elif self.method == 'cka':
            return self._compute_cka_loss(uncertainty, error, use_patches)
        elif self.method == 'gram':
            return self._compute_gram_loss(uncertainty, error, use_patches)
        elif self.method == 'mmd_hard_aware':
            return self._compute_hard_aware_mmd_loss(uncertainty, error)
        elif self.method == 'domain_aware_soft_mmd':
            return self._compute_domain_aware_soft_mmd_loss(uncertainty, error, feature_map, tag=tag)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _compute_loss_with_checkpoint(
        self,
        uncertainty: torch.Tensor,
        error: torch.Tensor,
        use_patches: bool,
        feature_map: torch.Tensor | None,
        tag: str,
    ) -> torch.Tensor:
        """Compute loss with gradient checkpointing to save memory."""
        return checkpoint(
            self._compute_loss_impl,
            uncertainty,
            error,
            use_patches,
            feature_map,
            tag,
            use_reentrant=False
        )
    
    def _compute_mmd_loss(
        self,
        uncertainty: torch.Tensor,
        error: torch.Tensor,
        use_patches: bool,
    ) -> torch.Tensor:
        """Compute MMD loss (patch-based or global)."""
        if use_patches:
            return MMDComputer.compute_patch_based_mmd(
                uncertainty,
                error,
                patch_size=self.patch_size,
                kernel=self.kernel,
                bandwidth=self.bandwidth,
            ), {}
        else:
            # Global MMD: flatten to [N, 1]
            return MMDComputer.compute_mmd(
                uncertainty.flatten().unsqueeze(-1),
                error.flatten().unsqueeze(-1),
                kernel=self.kernel,
                bandwidth=self.bandwidth,
            ), {}
    
    def _compute_cka_loss(
        self,
        uncertainty: torch.Tensor,
        error: torch.Tensor,
        use_patches: bool,
    ) -> torch.Tensor:
        """Compute CKA loss (patch-based or global)."""
        if use_patches:
            return CKAComputer.compute_patch_based_cka(
                uncertainty,
                error,
                patch_size=self.patch_size,
                use_linear_kernel=self.cka_use_linear_kernel,
                use_minibatch=self.cka_use_minibatch,
                minibatch_size=self.cka_minibatch_size,
            ), {}
        else:
            # Global CKA
            return CKAComputer.compute_linear_cka_direct(
                uncertainty.flatten().unsqueeze(-1),
                error.flatten().unsqueeze(-1),
            ), {}

    def _compute_hard_aware_mmd_loss(
        self,
        uncertainty: torch.Tensor,
        error: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute Hard-Aware MMD loss."""
        # Hard-Aware MMD is inherently patch-based (mining hard patches)
        
        return HardAwareMMD.compute_loss(
            uncertainty,
            error,
            patch_size=self.patch_size,
            top_k_percent=self.top_k_percent,
            max_samples=self.max_samples,
        ), {}

    
    def _compute_gram_loss(
        self,
        uncertainty: torch.Tensor,
        error: torch.Tensor,
        use_patches: bool,
    ) -> torch.Tensor:
        """Compute Gram matrix matching loss (patch-based or global)."""
        if use_patches:
            return GramComputer.compute_patch_based_gram(
                uncertainty,
                error,
                patch_size=self.patch_size,
            ), {}
        else:
            # Global Gram: flatten to [N, 1]
            u_flat = uncertainty.flatten().unsqueeze(-1)
            e_flat = error.flatten().unsqueeze(-1)
            return GramComputer.compute_gram_loss(u_flat, e_flat), {}
    
    def _compute_domain_aware_soft_mmd_loss(
        self,
        uncertainty: torch.Tensor,
        error: torch.Tensor,
        feature_map: torch.Tensor | None,
        tag: str = "unknown",  # NEW: Debug tag
    ) -> torch.Tensor:
        """
        Compute Domain-Aware Soft MMD loss.
        
        This method requires feature_map from the image encoder to compute
        feature diversity as a proxy for epistemic uncertainty.
        """
        if feature_map is None:
            # Fallback logic is handled in compute_loss before calling this, 
            # or we can fallback here. But explicit fallback is better.
            # If we are here, it means we MUST have feature_map.
            # However, if it's None, we can try to fallback to hard-aware MMD?
            # No, let's fail loudly as requested.
            raise ValueError(
                f"[{tag}] domain_aware_soft_mmd requires feature_map parameter. "
                f"Please ensure feature maps are extracted from backbone and "
                f"passed to compute_loss(). Use vision_features[-1] from backbone_fpn."
            )
        
        return DomainAwareSoftMMD.compute_loss(
            uncertainty=uncertainty,
            error=error,
            feature_map=feature_map,
            patch_size=self.patch_size,
            diversity_weight=self.diversity_weight,
            temperature=self.temperature,
            max_samples=self.max_samples,
            diversity_method=self.diversity_method,
            enable_monitoring=self.enable_monitoring,
            tag=tag,  # Pass tag
        )
