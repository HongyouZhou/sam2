"""
BNDL utility functions for SAM2.

This module provides utility functions for working with BNDL (Bayesian Non-negative Decision Layer),
including analytic uncertainty computation from Weibull parameters.
"""

from dataclasses import dataclass

import torch


@dataclass
class BNDLOutputs:
    pixel_feat: torch.Tensor
    pixel_logits: torch.Tensor
    external_w: torch.Tensor | None
    pixel_uncertainty: torch.Tensor | None


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
    # Use no_grad to prevent gradients from flowing back to BNDL parameters.
    
    # SAFETY: Clamp external weights to prevent gradient explosion
    if external_pre_out_w is not None:
        external_pre_out_w = torch.clamp(external_pre_out_w, min=-5.0, max=5.0)

    with torch.no_grad():
        out, z_out, weibull_lambda, inv_k, pre_out_w, *_ = pixel_bndl_model(pixel_feat, force_sample=False, external_pre_out_w=external_pre_out_w)
    K = out.shape[-1]  # number of output classes/masks

    # Early check: validate BNDL outputs
    if not torch.isfinite(out).all():
        raise RuntimeError("BNDL output contains NaN/Inf values")
    if not torch.isfinite(weibull_lambda).all():
        raise RuntimeError("BNDL weibull_lambda contains NaN/Inf values")
    if not torch.isfinite(inv_k).all():
        raise RuntimeError("BNDL inv_k contains NaN/Inf values")

    # Step 2: Compute Weibull statistics (E[Z] and Var[Z])
    # Revert to 013 version: no eps_kappa correction
    # In BNDL: inv_k = 1 / kappa (direct, no epsilon)
    # Therefore: kappa = 1 / inv_k
    # Clamp inv_k to prevent division by extremely small values
    inv_k_clamped = torch.clamp(inv_k, min=1e-6, max=1.0)
    kappa = 1.0 / inv_k_clamped
    kappa = torch.clamp(kappa, min=0.5, max=8.0)  # Enforce KAPPA_MIN/MAX

    # Early check: validate kappa
    if not torch.isfinite(kappa).all():
        raise RuntimeError("kappa computation produced NaN/Inf values")

    # Clamp kappa for lgamma stability (prevent overflow)
    kappa_safe = torch.clamp(kappa, min=0.5, max=8.0)
    kappa_reciprocal = 1.0 / kappa_safe
    kappa_reciprocal_2 = 2.0 / kappa_safe

    # Clamp reciprocal values to prevent lgamma overflow
    # lgamma(x) is stable for x < ~170, so we clamp to safe range
    kappa_reciprocal = torch.clamp(kappa_reciprocal, max=50.0)
    kappa_reciprocal_2 = torch.clamp(kappa_reciprocal_2, max=50.0)

    # Weibull expectation: E[Z] = λ * Γ(1 + 1/κ)
    # (computed but not directly used in variance calculation)
    lgamma_1_arg = 1.0 + kappa_reciprocal
    lgamma_1 = torch.lgamma(lgamma_1_arg)

    # Early check: validate lgamma_1
    if not torch.isfinite(lgamma_1).all():
        raise RuntimeError("lgamma(1 + 1/kappa) produced NaN/Inf values")

    # Weibull variance: Var[Z] = λ² * [Γ(1 + 2/κ) - Γ²(1 + 1/κ)]
    lgamma_2_arg = 1.0 + kappa_reciprocal_2
    lgamma_2 = torch.lgamma(lgamma_2_arg)

    # Early check: validate lgamma_2
    if not torch.isfinite(lgamma_2).all():
        raise RuntimeError("lgamma(1 + 2/kappa) produced NaN/Inf values")

    # Use log-space computation to avoid numerical instability
    # Var[Z] = λ² * [exp(lgamma_2) - exp(2*lgamma_1)]
    # Compute in log-space: log(Var[Z]) = 2*log(λ) + log(exp(lgamma_2) - exp(2*lgamma_1))
    # Use log-sum-exp trick for stability: log(exp(a) - exp(b)) = a + log(1 - exp(b-a)) when a > b
    lgamma_max = torch.maximum(lgamma_2, 2 * lgamma_1)
    lgamma_2_shifted = lgamma_2 - lgamma_max
    lgamma_1_shifted = 2 * lgamma_1 - lgamma_max

    # Compute log(exp(lgamma_2) - exp(2*lgamma_1)) = lgamma_max + log(exp(shifted_2) - exp(shifted_1))
    # Clamp the difference to prevent log(0) or log(negative)
    exp_diff = torch.clamp(
        torch.exp(lgamma_2_shifted) - torch.exp(lgamma_1_shifted),
        min=1e-12,  # Small positive value to prevent log(0)
    )
    log_gamma_diff = lgamma_max + torch.log(exp_diff)

    # Compute log(var_z) = 2*log(λ) + log_gamma_diff
    log_lambda = torch.log(torch.clamp(weibull_lambda, min=1e-8))
    log_var_z = 2 * log_lambda + log_gamma_diff

    # Convert back and clamp to reasonable range
    var_z = torch.exp(torch.clamp(log_var_z, max=15.0))  # exp(15) ≈ 3e6, more conservative
    var_z = torch.clamp(var_z, min=1e-10, max=1e6)  # Final safety clamp

    # Early check: validate var_z
    if not torch.isfinite(var_z).all():
        raise RuntimeError("var_z computation produced NaN/Inf values")

    # Step 3: Propagate to logits statistics
    # logits = linear_output(Z) = W @ Z + b
    # In eval mode, out = E[logits] = W @ E[Z] + b
    mean_logits = out  # [B, H, W, K]

    # Variance propagation: Var[logits_k] = Σ_i W_{ki}² * Var[Z_i]
    # (assumes independent Z_i, which is true by BNDL's design)
    if external_pre_out_w is not None:
        # Case with external weights (hyper_in)
        # external_pre_out_w: [B, K, C'] or [C', K]
        
        # Use _apply_external_weights from BNDL to handle shapes for variance
        # We need W^2 @ Var[Z]
        w_squared = external_pre_out_w ** 2
        var_logits = pixel_bndl_model._apply_external_weights(var_z, w_squared)
        
    elif hasattr(pixel_bndl_model, "linear"):
        # Standard BNDL with internal linear layer
        output_weight = pixel_bndl_model.linear.weight  # [K, C]
        # Einsum: sum over C dimension with squared weights
        var_logits = torch.einsum("bhwc,kc->bhwk", var_z, output_weight**2)
        
    elif hasattr(pixel_bndl_model, "linear_output"):
        output_weight = pixel_bndl_model.linear_output.weight  # [K, C]
        # Einsum: sum over C dimension with squared weights
        var_logits = torch.einsum("bhwc,kc->bhwk", var_z, output_weight**2)
    else:
        # Fallback: use mean variance across channels
        var_logits = var_z.mean(dim=-1, keepdim=True).expand(B, H, W, K)

    # Early check: validate var_logits
    if not torch.isfinite(var_logits).all():
        raise RuntimeError("var_logits computation produced NaN/Inf values")

    # Clamp var_logits to prevent overflow in sqrt and kappa_sigmoid
    var_logits = torch.clamp(var_logits, min=0.0, max=1e6)

    # Step 4: Compute E[sigmoid(logits)] via MacKay approximation
    # For X ~ N(μ, σ²): E[sigmoid(X)] ≈ sigmoid(κ * μ)
    # where κ = 1 / √(1 + π*σ²/8)
    pi = 3.14159265359
    # Clamp denominator to prevent division issues
    denominator = 1.0 + pi * var_logits / 8.0
    denominator = torch.clamp(denominator, min=1e-8)
    kappa_sigmoid = 1.0 / torch.sqrt(denominator)

    # Early check: validate kappa_sigmoid
    if not torch.isfinite(kappa_sigmoid).all():
        raise RuntimeError("kappa_sigmoid computation produced NaN/Inf values")

    # Clamp kappa_sigmoid * mean_logits to prevent sigmoid overflow
    sigmoid_input = kappa_sigmoid * torch.clamp(mean_logits, min=-50.0, max=50.0)
    mean_probs = torch.sigmoid(sigmoid_input)  # [B, H, W, K]

    # Early check: validate mean_probs
    if not torch.isfinite(mean_probs).all():
        raise RuntimeError("mean_probs computation produced NaN/Inf values")

    # Clamp mean_probs to valid probability range for numerical stability
    mean_probs = torch.clamp(mean_probs, min=1e-8, max=1.0 - 1e-8)

    # Step 5: Compute Bernoulli entropy H(p) = -p*log(p) - (1-p)*log(1-p)
    # Use torch.special.entr for numerical stability (handles p=0 and p=1 gracefully)
    if hasattr(torch.special, "entr"):
        # PyTorch >= 1.9: use built-in entr function
        # entr(p) = -p * log(p), with entr(0) = 0 by convention
        entropy_per_mask = torch.special.entr(mean_probs) + torch.special.entr(1.0 - mean_probs)
    else:
        # Fallback: manual implementation with epsilon for stability
        eps = 1e-10
        entropy_per_mask = -(mean_probs * torch.log(mean_probs + eps) + (1.0 - mean_probs) * torch.log(1.0 - mean_probs + eps))

    # Final clamp to ensure finite values
    entropy_per_mask = torch.clamp(entropy_per_mask, min=0.0, max=1.0)

    # Final check: guard against NaN/Inf
    if not torch.isfinite(entropy_per_mask).all():
        raise RuntimeError("Analytic uncertainty computation produced NaN/Inf values in final entropy")

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
    analytic = pixel_weibull_to_entropy_uncertainty(pixel_bndl_model, pixel_feat, external_pre_out_w=external_pre_out_w, per_channel=False)

    # Compute sampling version (ground truth, detached)
    with torch.no_grad():
        sampling = pixel_entropy_uncertainty(pixel_bndl_model, pixel_feat, external_pre_out_w=external_pre_out_w, sample_num=sample_num, per_channel=False)

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
        "correlation": correlation,
        "mae": mae,
        "mse": mse,
        "analytic": analytic,
        "sampling": sampling,
    }
