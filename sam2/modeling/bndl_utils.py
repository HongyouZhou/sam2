"""
BNDL utility functions for SAM2.

This module provides utility functions for working with BNDL (Bayesian Non-negative Decision Layer),
including analytic uncertainty computation from Weibull parameters.
"""

import torch
import torch.nn.functional as F


def pixel_weibull_to_entropy_uncertainty(
    pixel_bndl_model,
    pixel_feat: torch.Tensor,
    external_pre_out_w=None,
    per_channel: bool = False,
) -> torch.Tensor:
    """
    Compute Bernoulli entropy uncertainty analytically from Weibull parameters (with gradients).
    
    This is a differentiable approximation to the sampling-based uncertainty computation.
    The derivation follows the chain:
        Weibull(λ,κ) → E[Z], Var[Z] → E[logits], Var[logits] 
        → E[p] (via Probit approximation) → H(E[p]) (Bernoulli entropy)
    
    Key advantages:
    - Preserves gradients for BNDL parameters (λ, κ)
    - Single forward pass (no sampling loop)
    - Deterministic (no sampling noise)
    - Memory efficient (no large sampling buffers)
    
    Theoretical foundation:
    1. Weibull statistics (exact):
       E[Z] = λ * Γ(1 + 1/κ)
       Var[Z] = λ² * [Γ(1 + 2/κ) - Γ²(1 + 1/κ)]
    
    2. Linear propagation (exact):
       logits = W @ Z + b
       E[logits] = W @ E[Z] + b
       Var[logits_k] = Σ_i W_{ki}² * Var[Z_i]  (assuming independent Z_i)
    
    3. Sigmoid expectation (MacKay approximation):
       If X ~ N(μ, σ²), then E[sigmoid(X)] ≈ sigmoid(κ * μ)
       where κ = 1 / √(1 + π*σ²/8)
    
    4. Bernoulli entropy (exact):
       H(p) = -p*log(p) - (1-p)*log(1-p)
    
    Args:
        pixel_bndl_model: BNDL model (typically sam_mask_decoder.pixel_bndl)
        pixel_feat: [B, H, W, C] pixel features
        external_pre_out_w: external weights if using hyper_in
        per_channel: If True, return per-channel entropy [B, H, W, K]
                     If False, return aggregated entropy [B, H, W]
    
    Returns:
        uncertainty_map: [B, H, W, K] if per_channel=True, else [B, H, W]
                         Bernoulli entropy with gradients preserved
    
    References:
        - MacKay (1992): "The Evidence Framework Applied to Classification Networks"
        - Bishop (2006): "Pattern Recognition and Machine Learning", Section 4.5.2
    """
    B, H, W, C = pixel_feat.shape
    
    # Step 1: Forward through BNDL to get Weibull parameters
    # force_sample=False ensures deterministic mode (uses expectation)
    out, z_out, weibull_lambda, inv_k, pre_out_w, *_ = pixel_bndl_model(
        pixel_feat, force_sample=False, external_pre_out_w=external_pre_out_w
    )
    K = out.shape[-1]  # number of output classes/masks
    
    # Step 2: Compute Weibull statistics (E[Z] and Var[Z])
    eps_kappa = 1e-3
    
    # Recover kappa from inv_k
    # In BNDL: inv_k = 1 / (kappa + eps_kappa)
    # Therefore: kappa ≈ 1 / inv_k - eps_kappa
    kappa = 1.0 / (inv_k + 1e-8) - eps_kappa
    kappa = torch.clamp(kappa, min=0.5, max=8.0)  # Enforce KAPPA_MIN/MAX
    
    # Weibull expectation: E[Z] = λ * Γ(1 + 1/κ)
    lgamma_1 = torch.lgamma(1.0 + 1.0 / (kappa + eps_kappa))
    mean_z = weibull_lambda * torch.exp(lgamma_1)  # [B, H, W, C]
    
    # Weibull variance: Var[Z] = λ² * [Γ(1 + 2/κ) - Γ²(1 + 1/κ)]
    lgamma_2 = torch.lgamma(1.0 + 2.0 / (kappa + eps_kappa))
    var_z = (weibull_lambda ** 2) * (torch.exp(lgamma_2) - torch.exp(2 * lgamma_1))
    var_z = torch.clamp(var_z, min=0.0)  # Ensure non-negative
    
    # Step 3: Propagate to logits statistics
    # logits = linear_output(Z) = W @ Z + b
    # In eval mode, out = E[logits] = W @ E[Z] + b
    mean_logits = out  # [B, H, W, K]
    
    # Variance propagation: Var[logits_k] = Σ_i W_{ki}² * Var[Z_i]
    # (assumes independent Z_i, which is true by BNDL's design)
    if hasattr(pixel_bndl_model, 'linear_output'):
        output_weight = pixel_bndl_model.linear_output.weight  # [K, C]
        # Einsum: sum over C dimension with squared weights
        var_logits = torch.einsum('bhwc,kc->bhwk', var_z, output_weight ** 2)
    else:
        # Fallback: use mean variance across channels
        var_logits = var_z.mean(dim=-1, keepdim=True).expand(B, H, W, K)
    
    std_logits = torch.sqrt(var_logits + 1e-8)  # [B, H, W, K]
    
    # Step 4: Compute E[sigmoid(logits)] via MacKay approximation
    # For X ~ N(μ, σ²): E[sigmoid(X)] ≈ sigmoid(κ * μ)
    # where κ = 1 / √(1 + π*σ²/8)
    pi = 3.14159265359
    kappa_sigmoid = 1.0 / torch.sqrt(1.0 + pi * var_logits / 8.0)
    mean_probs = torch.sigmoid(kappa_sigmoid * mean_logits)  # [B, H, W, K]
    
    # Step 5: Compute Bernoulli entropy H(p) = -p*log(p) - (1-p)*log(1-p)
    # Use torch.special.entr for numerical stability (handles p=0 and p=1 gracefully)
    if hasattr(torch.special, 'entr'):
        # PyTorch >= 1.9: use built-in entr function
        # entr(p) = -p * log(p), with entr(0) = 0 by convention
        entropy_per_mask = torch.special.entr(mean_probs) + torch.special.entr(1.0 - mean_probs)
    else:
        # Fallback: manual implementation with epsilon for stability
        eps = 1e-10
        entropy_per_mask = -(
            mean_probs * torch.log(mean_probs + eps) + 
            (1.0 - mean_probs) * torch.log(1.0 - mean_probs + eps)
        )
    
    # Guard against NaN/Inf (should not happen with entr, but defensive)
    if not torch.isfinite(entropy_per_mask).all():
        raise RuntimeError("Analytic uncertainty computation produced NaN/Inf values")
    
    # Return per-channel or aggregated
    if per_channel:
        return entropy_per_mask  # [B, H, W, K]
    else:
        # Average across masks (backward compatible with sampling version)
        return entropy_per_mask.mean(dim=-1)  # [B, H, W]


def compute_analytic_sampling_correlation(
    pixel_bndl_model,
    pixel_feat: torch.Tensor,
    external_pre_out_w=None,
    sample_num: int = 100,
) -> dict:
    """
    Validate the analytic uncertainty against sampling-based uncertainty.
    
    This function computes both versions and returns correlation metrics
    to verify the accuracy of the analytic approximation.
    
    Args:
        pixel_bndl_model: BNDL model
        pixel_feat: [B, H, W, C] pixel features
        external_pre_out_w: external weights
        sample_num: number of samples for ground truth (higher = more accurate)
    
    Returns:
        dict with keys:
            - 'correlation': Pearson correlation coefficient
            - 'mae': Mean absolute error
            - 'mse': Mean squared error
            - 'analytic': analytic uncertainty tensor
            - 'sampling': sampling uncertainty tensor
    """
    # Import here to avoid circular dependency
    from BNDL.BNDL_upload.ViT_Sparse.utils.bndl import pixel_entropy_uncertainty
    
    # Compute analytic version (with gradients)
    analytic = pixel_weibull_to_entropy_uncertainty(
        pixel_bndl_model, pixel_feat,
        external_pre_out_w=external_pre_out_w,
        per_channel=False
    )
    
    # Compute sampling version (ground truth, detached)
    with torch.no_grad():
        sampling = pixel_entropy_uncertainty(
            pixel_bndl_model, pixel_feat,
            external_pre_out_w=external_pre_out_w,
            sample_num=sample_num,
            per_channel=False
        )
    
    # Compute metrics
    analytic_flat = analytic.detach().flatten()
    sampling_flat = sampling.flatten()
    
    # Pearson correlation
    correlation_matrix = torch.corrcoef(torch.stack([analytic_flat, sampling_flat]))
    correlation = correlation_matrix[0, 1].item()
    
    # MAE and MSE
    mae = (analytic_flat - sampling_flat).abs().mean().item()
    mse = (analytic_flat - sampling_flat).pow(2).mean().item()
    
    return {
        'correlation': correlation,
        'mae': mae,
        'mse': mse,
        'analytic': analytic,
        'sampling': sampling,
    }

