"""
BNDL utility functions for SAM2.

This module provides utility functions for working with BNDL (Bayesian Non-negative Decision Layer),
including analytic uncertainty computation from Weibull parameters.
"""

from dataclasses import dataclass

import torch

# Import precomputed constants from BNDL for consistency
from BNDL.BNDL_upload.ViT_Sparse.utils.bndl import LOG_2, PI_OVER_8


@dataclass
class BNDLOutputs:
    """Container for BNDL outputs with optional cached Weibull parameters.

    The cached parameters (wei_lambda, inv_k, etc.) can be populated from BNDL's
    forward pass to avoid redundant computation in analytic uncertainty estimation.
    """

    pixel_feat: torch.Tensor
    pixel_logits: torch.Tensor
    external_w: torch.Tensor | None
    pixel_uncertainty: torch.Tensor | None
    # Cached Weibull parameters to avoid redundant BNDL forward passes
    wei_lambda: torch.Tensor | None = None  # [B, H, W, C]
    inv_k: torch.Tensor | None = None  # [B, H, W, 1]
    wei_lambda_w: torch.Tensor | None = None  # [B, K, num_classes, C]
    inv_k_w: torch.Tensor | None = None  # [B, K, 1, C]
    pre_out: torch.Tensor | None = None  # [B, H, W, C] - reparameterized pixel features
    pre_out_w: torch.Tensor | None = None  # [B, K, num_classes, C] - reparameterized weights
    mean_logits: torch.Tensor | None = None  # [B, H, W, K] - same as pixel_logits for convenience


def pixel_weibull_to_entropy_uncertainty(
    pixel_bndl_model,
    pixel_feat: torch.Tensor,
    external_pre_out_w: torch.Tensor | None = None,
    per_channel: bool = False,
    cached_outputs: BNDLOutputs | None = None,
) -> torch.Tensor:
    """
    Compute Bernoulli entropy uncertainty analytically from Weibull parameters (with gradients).

    This is a differentiable approximation to the sampling-based uncertainty computation.
    The derivation uses the MacKay approximation for sigmoid expectation.

    Args:
        pixel_bndl_model: BNDL model (typically sam_mask_decoder.pixel_bndl)
        pixel_feat: [B, H, W, C] pixel features
        external_pre_out_w: [B, K, C'] mask_tokens_out (256-dim) - BNDL internally projects this via linear_add_w
        per_channel: If True, return per-channel entropy [B, H, W, K]
                     If False, return aggregated entropy [B, H, W]
        cached_outputs: Optional BNDLOutputs with cached Weibull parameters to avoid redundant forward pass

    Returns:
        uncertainty_map: [B, H, W, K] if per_channel=True, else [B, H, W]
                         Bernoulli entropy with gradients preserved

    References:
        - MacKay (1992): "The Evidence Framework Applied to Classification Networks"
        - Bishop (2006): "Pattern Recognition and Machine Learning", Section 4.5.2
    """
    B, H, W, C = pixel_feat.shape

    # Check if we can use cached parameters
    use_cache = (
        cached_outputs is not None
        and cached_outputs.wei_lambda is not None
        and cached_outputs.inv_k is not None
        and cached_outputs.wei_lambda_w is not None
        and cached_outputs.inv_k_w is not None
        and cached_outputs.pixel_logits is not None
    )

    if use_cache:
        # Use cached Weibull parameters (avoids redundant forward pass)
        wei_lambda = cached_outputs.wei_lambda
        inv_k = cached_outputs.inv_k
        wei_lambda_w = cached_outputs.wei_lambda_w
        inv_k_w = cached_outputs.inv_k_w
        out_sam = cached_outputs.pixel_logits
    else:
        # Fall back to computing via BNDL forward pass
        # Get factor parameters from model attributes or use defaults
        factor_z = getattr(pixel_bndl_model, "default_factor_z", 0.0)
        factor_w = getattr(pixel_bndl_model, "default_factor_w", 0.02)

        # external_pre_out_w (mask_tokens_out) is required for the current BNDL API
        if external_pre_out_w is None:
            raise ValueError("external_pre_out_w (mask_tokens_out) is required for pixel_weibull_to_entropy_uncertainty. Please pass the mask_tokens_out tensor from mask_decoder.")

        # Forward through BNDL to get Weibull parameters
        # Current BNDL API: forward(pixel_feat, mask_token, factor_z, factor_w, force_sample)
        # Returns: out_sam, z_out, wei_lambda, inv_k, wei_lambda_w, inv_k_w, pre_out, pre_out_w
        outputs = pixel_bndl_model(
            pixel_feat,
            external_pre_out_w,  # mask_tokens_out [B, K, 256]
            factor_z=factor_z,
            factor_w=factor_w,
            force_sample=False,  # Use deterministic mode (Weibull mean)
        )

        # Unpack the 6-tuple return value (019 API)
        out_sam, _z_out, wei_lambda, inv_k, wei_lambda_w, inv_k_w = outputs

    # out_sam is already the final logits [B, H, W, K]
    # We need to estimate uncertainty from the Weibull parameters

    # Step 2: Compute Weibull variance for pixel features
    # Weibull statistics:
    #   E[Z] = λ * Γ(1 + 1/κ)
    #   Var[Z] = λ² * [Γ(1 + 2/κ) - Γ²(1 + 1/κ)]
    # Since we have inv_k = 1/κ, we use it directly:

    # Compute lgamma values for variance
    lgamma_1 = torch.lgamma(1 + inv_k)  # Γ(1 + 1/κ)
    lgamma_2 = torch.lgamma(1 + 2 * inv_k)  # Γ(1 + 2/κ)

    # Weibull variance: λ² * [exp(lgamma_2) - exp(2*lgamma_1)]
    # = λ² * exp(lgamma_2) * [1 - exp(2*lgamma_1 - lgamma_2)]
    # Use log-space computation for numerical stability
    log_gamma_diff = lgamma_2 + torch.log1p(-torch.exp(torch.clamp(2 * lgamma_1 - lgamma_2, max=-1e-7)))
    var_z = wei_lambda**2 * torch.exp(log_gamma_diff)  # [B, H, W, C]

    # Step 3: Approximate variance of logits using Weibull weight variance
    # Since mask_tokens_out (256-dim) is projected by BNDL's linear_add_w internally,
    # we use the returned Weibull weight parameters (wei_lambda_w, inv_k_w) for variance
    #
    # wei_lambda_w: [B, K, num_classes, C] = [B, K, 2, 32]
    # inv_k_w: [B, K, 1, C] = [B, K, 1, 32]
    # var_z: [B, H, W, C] = [B, H, W, 32]

    mean_logits = out_sam  # [B, H, W, K]
    _ = mean_logits.shape[-1]  # K (number of masks) - not used but kept for reference

    # Compute variance of weight contributions from Weibull parameters
    # Var[W] = λ² * [Γ(1 + 2/κ) - Γ²(1 + 1/κ)]
    lgamma_1_w = torch.lgamma(1 + inv_k_w)
    lgamma_2_w = torch.lgamma(1 + 2 * inv_k_w)
    log_gamma_diff_w = lgamma_2_w + torch.log1p(-torch.exp(torch.clamp(2 * lgamma_1_w - lgamma_2_w, max=-1e-7)))
    var_w = wei_lambda_w**2 * torch.exp(log_gamma_diff_w)  # [B, K, num_classes, C]

    # Average over num_classes dimension: [B, K, C]
    var_w_avg = var_w.mean(dim=2)  # [B, K, 32]

    # Propagate variance: Var[logits] ≈ Σ_c (E[W_c]² * var_z_c + E[Z_c]² * var_w_c + var_z_c * var_w_c)
    # Simplified: use first-order approximation Var[logits] ≈ Σ W² * Var[Z] + Σ Z² * Var[W]
    # For stability, we use the pixel variance as the dominant term with weight variance as correction

    # var_logits = einsum("bhwc,bkc->bhwk", var_z, var_w_avg) - includes cross terms
    var_logits = torch.einsum("bhwc,bkc->bhwk", var_z, var_w_avg.clamp(min=1e-8))

    # Step 4: Compute E[sigmoid(logits)] via MacKay approximation
    # For X ~ N(μ, σ²): E[sigmoid(X)] ≈ sigmoid(κ * μ)
    # where κ = 1 / √(1 + π*σ²/8)
    denominator = 1.0 + PI_OVER_8 * var_logits
    kappa_sigmoid = 1.0 / torch.sqrt(denominator + 1e-8)

    sigmoid_input = kappa_sigmoid * mean_logits
    mean_probs = torch.sigmoid(sigmoid_input)  # [B, H, W, K]

    # Step 5: Compute Bernoulli entropy H(p) = -p*log(p) - (1-p)*log(1-p)
    # Use 1e-4 to avoid numerical issues in bfloat16 (consistent with entropy_uncertainty)
    mean_probs = torch.clamp(mean_probs, min=1e-4, max=1.0 - 1e-4)

    if hasattr(torch.special, "entr"):
        # PyTorch >= 1.9: use built-in entr function
        entropy_per_mask = torch.special.entr(mean_probs) + torch.special.entr(1.0 - mean_probs)
    else:
        # Fallback: manual implementation (use larger eps for bfloat16 safety)
        eps = 1e-8
        entropy_per_mask = -(mean_probs * torch.log(mean_probs + eps) + (1.0 - mean_probs) * torch.log(1.0 - mean_probs + eps))

    # Step 6: Normalize entropy to [0, 1] by dividing by max Bernoulli entropy log(2)
    entropy_per_mask = entropy_per_mask / LOG_2

    # Return per-channel or aggregated
    if per_channel:
        return entropy_per_mask  # [B, H, W, K], range [0, 1]
    else:
        # Average across masks
        return entropy_per_mask.mean(dim=-1)  # [B, H, W], range [0, 1]


def compute_analytic_sampling_correlation(
    pixel_bndl_model,
    pixel_feat: torch.Tensor,
    external_pre_out_w: torch.Tensor | None = None,
    sample_num: int = 100,
) -> dict:
    """
    Validate the analytic uncertainty against sampling-based uncertainty.

    This function computes both versions and returns correlation metrics
    to verify the accuracy of the analytic approximation.

    Args:
        pixel_bndl_model: BNDL model
        pixel_feat: [B, H, W, C] pixel features
        external_pre_out_w: [B, K, 256] mask_tokens_out embeddings
        sample_num: number of samples for ground truth (higher = more accurate)

    Returns:
        dict with keys:
            - 'correlation': Pearson correlation coefficient
            - 'mae': Mean absolute error
            - 'mse': Mean squared error
            - 'analytic': analytic uncertainty tensor
            - 'sampling': sampling uncertainty tensor
    """
    from BNDL.BNDL_upload.ViT_Sparse.utils.bndl import entropy_uncertainty, uncertainty_sample_parallel

    if external_pre_out_w is None:
        raise ValueError("external_pre_out_w (mask_tokens_out) is required")

    # Compute analytic version (with gradients)
    analytic = pixel_weibull_to_entropy_uncertainty(pixel_bndl_model, pixel_feat, external_pre_out_w=external_pre_out_w, per_channel=False)

    # Compute sampling version (ground truth, detached)
    with torch.no_grad():
        sampled_logits, _ = uncertainty_sample_parallel(
            pixel_bndl_model,
            pixel_feat,
            external_pre_out_w,
            sample_num=sample_num,
        )
        # entropy_uncertainty returns [B, H, W, K]
        sampling_entropy = entropy_uncertainty(sampled_logits)
        sampling = sampling_entropy.mean(dim=-1)  # [B, H, W]

    # Compute metrics
    analytic_flat = analytic.detach().flatten()
    sampling_flat = sampling.flatten()

    # Pearson correlation
    if analytic_flat.numel() > 1:
        correlation_matrix = torch.corrcoef(torch.stack([analytic_flat, sampling_flat]))
        correlation = correlation_matrix[0, 1].item()
    else:
        correlation = 1.0

    # MAE and MSE
    mae = (analytic_flat - sampling_flat).abs().mean().item()
    mse = (analytic_flat - sampling_flat).pow(2).mean().item()

    return {
        "correlation": correlation,
        "mae": mae,
        "mse": mse,
        "analytic": analytic,
        "sampling": sampling,
    }
