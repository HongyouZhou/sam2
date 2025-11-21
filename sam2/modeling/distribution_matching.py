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
        
        # Take square root
        mmd = torch.sqrt(mmd_squared_avg)
        
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
    
    Supports switching between MMD, CKA, and Gram matrix matching
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
    ):
        """
        Initialize distribution matcher.
        
        Args:
            method: 'mmd' | 'cka' | 'gram'
            patch_size: size of square patches for patch-based methods
            kernel: kernel type for MMD ('rbf')
            bandwidth: kernel bandwidth for MMD
            cka_use_linear_kernel: use linear kernel for CKA (faster)
            cka_use_minibatch: use mini-batch for CKA (memory efficient)
            cka_minibatch_size: batch size for CKA mini-batch computation
        """
        assert method in ['mmd', 'cka', 'gram'], f"Unknown method: {method}"
        
        self.method = method
        self.patch_size = patch_size
        self.kernel = kernel
        self.bandwidth = bandwidth
        
        # CKA-specific parameters
        self.cka_use_linear_kernel = cka_use_linear_kernel
        self.cka_use_minibatch = cka_use_minibatch
        self.cka_minibatch_size = cka_minibatch_size
    
    def compute_loss(
        self,
        uncertainty: torch.Tensor,
        error: torch.Tensor,
        use_patches: bool = True,
    ) -> torch.Tensor:
        """
        Compute distribution matching loss.
        
        Args:
            uncertainty: [B, H, W] uncertainty map in [0, 1]
            error: [B, H, W] error map in [0, 1]
            use_patches: if True, use patch-based computation;
                        if False, use global (pixel-level) computation
        
        Returns:
            loss: scalar distribution matching loss
        """
        if self.method == 'mmd':
            return self._compute_mmd_loss(uncertainty, error, use_patches)
        elif self.method == 'cka':
            return self._compute_cka_loss(uncertainty, error, use_patches)
        elif self.method == 'gram':
            return self._compute_gram_loss(uncertainty, error, use_patches)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
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
            )
        else:
            # Global MMD: flatten to [N, 1]
            return MMDComputer.compute_mmd(
                uncertainty.flatten().unsqueeze(-1),
                error.flatten().unsqueeze(-1),
                kernel=self.kernel,
                bandwidth=self.bandwidth,
            )
    
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
            )
        else:
            # Global CKA: flatten to [N, 1]
            u_flat = uncertainty.flatten().unsqueeze(-1)
            e_flat = error.flatten().unsqueeze(-1)
            return CKAComputer._compute_cka_core(
                u_flat,
                e_flat,
                use_linear_kernel=self.cka_use_linear_kernel,
                use_minibatch=self.cka_use_minibatch,
                minibatch_size=self.cka_minibatch_size,
            )
    
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
            )
        else:
            # Global Gram: flatten to [N, 1]
            u_flat = uncertainty.flatten().unsqueeze(-1)
            e_flat = error.flatten().unsqueeze(-1)
            return GramComputer.compute_gram_loss(u_flat, e_flat)

