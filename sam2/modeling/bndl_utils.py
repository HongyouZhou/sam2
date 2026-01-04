"""
BNDL utility functions for SAM2.

This module provides utility functions for working with BNDL (Bayesian Non-negative Decision Layer),
including analytic uncertainty computation from Weibull parameters.
"""

from dataclasses import dataclass

import torch

# Import precomputed constants from BNDL for consistency
from BNDL.BNDL_upload.ViT_Sparse.utils.bndl import PI_OVER_8, LOG_2


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

    # Step 1: Forward through BNDL to get Weibull parameters.
    # force_sample=False ensures deterministic mode (uses expectation).
    # Returns: out, z_out, weibull_lambda, kappa, pre_out_w, wei_lambda_w, kappa_w, lgamma_cache
    # Note: BNDL now returns kappa directly (not inv_k) with scaled sigmoid constraint
    out, z_out, weibull_lambda, kappa_x, pre_out_w, wei_lambda_w, kappa_w, lgamma_cache = pixel_bndl_model(
        pixel_feat,
        force_sample=False,
        external_pre_out_w=external_pre_out_w,
    )
    K = out.shape[-1]  # number of output classes/masks

    # ============================================================
    # Step 2a: Compute Weibull statistics for PIXEL features (E[Z_x], Var[Z_x])
    # ============================================================
    # kappa_x is already in [KAPPA_MIN, KAPPA_MAX] range via scaled sigmoid, no clamp needed

    # Reuse lgamma values from BNDL forward pass to avoid recomputation
    lgamma_1_x = lgamma_cache["lgamma_1_k"]  # [B, H, W, 1]
    lgamma_2_x = lgamma_cache["lgamma_2_k"]  # [B, H, W, 1]

    # Weibull expectation: E[Z] = λ * Γ(1 + 1/κ)
    mean_z_x = weibull_lambda * torch.exp(lgamma_1_x)  # [B, H, W, C]

    # Weibull variance: Var[Z] = λ² * [Γ(1 + 2/κ) - Γ²(1 + 1/κ)]
    # Numerically stable: log(Γ(1+2/k) - Γ²(1+1/k))
    a_x = lgamma_2_x
    b_x = 2.0 * lgamma_1_x
    t_x = torch.clamp(b_x - a_x, max=-1e-7)
    log_gamma_diff_x = a_x + torch.log1p(-torch.exp(t_x))
    var_z_x = weibull_lambda**2 * torch.exp(log_gamma_diff_x)  # [B, H, W, C]

    # ============================================================
    # Step 2b: Compute Weibull statistics for HYPER_IN weights (E[Z_w], Var[Z_w])
    # ============================================================
    if wei_lambda_w is not None and kappa_w is not None:
        # kappa_w is already in [KAPPA_MIN, KAPPA_MAX] range via scaled sigmoid, no clamp needed

        # Reuse lgamma values from BNDL forward pass to avoid recomputation
        lgamma_1_w = lgamma_cache["lgamma_1_kw"]  # [B, K, C]
        lgamma_2_w = lgamma_cache["lgamma_2_kw"]  # [B, K, C]

        # Weibull expectation: E[W] = λ_w * Γ(1 + 1/κ_w)
        mean_z_w = wei_lambda_w * torch.exp(lgamma_1_w)  # [B, K, C]

        # Weibull variance: Var[W] = λ_w² * [Γ(1 + 2/κ_w) - Γ²(1 + 1/κ_w)]
        a_w = lgamma_2_w
        b_w = 2.0 * lgamma_1_w
        t_w = torch.clamp(b_w - a_w, max=-1e-7)
        log_gamma_diff_w = a_w + torch.log1p(-torch.exp(t_w))
        var_z_w = wei_lambda_w**2 * torch.exp(log_gamma_diff_w)  # [B, K, C]
    else:
        # No external weights, use deterministic weights
        mean_z_w = None
        var_z_w = None

    mean_logits = out  # [B, H, W, K]

    # ============================================================
    # Step 3: Variance propagation for product of two random variables
    # ============================================================
    # For independent X, W: Var[X·W] = E[X]²·Var[W] + E[W]²·Var[X] + Var[X]·Var[W]
    # The dot product: logits_k = Σ_c X_c · W_{k,c}
    # Var[logits_k] = Σ_c Var[X_c · W_{k,c}]
    #               = Σ_c [E[X_c]²·Var[W_{k,c}] + E[W_{k,c}]²·Var[X_c] + Var[X_c]·Var[W_{k,c}]]

    if pre_out_w is not None and var_z_w is not None:
        # Full variance propagation with both pixel and hyper_in variance
        # mean_z_x: [B, H, W, C], mean_z_w: [B, K, C]
        # var_z_x: [B, H, W, C], var_z_w: [B, K, C]

        # Var[X·W] = E[X]²·Var[W] + E[W]²·Var[X] + Var[X]·Var[W]
        # = E[X]²·Var[W] + Var[X]·(E[W]² + Var[W])
        # This combines term2 and term3 into a single matrix multiplication

        # Term 1: E[X]² · Var[W] → [B, H, W, C] @ [B, K, C] → [B, H, W, K]
        mean_z_x_sq = mean_z_x**2  # [B, H, W, C]
        term1 = pixel_bndl_model._apply_external_weights(mean_z_x_sq, var_z_w)  # [B, H, W, K]

        # Term 2+3 combined: Var[X] · (E[W]² + Var[W]) → reduces one matmul
        mean_z_w_sq_plus_var = mean_z_w**2 + var_z_w  # [B, K, C]
        term2_3 = pixel_bndl_model._apply_external_weights(var_z_x, mean_z_w_sq_plus_var)  # [B, H, W, K]

        # Total variance of the dot product (before logit scaling)
        var_logits = term1 + term2_3  # [B, H, W, K]

    elif pre_out_w is not None:
        # Only pixel variance (original implementation)
        w_squared = pre_out_w**2
        var_logits = pixel_bndl_model._apply_external_weights(var_z_x, w_squared)

    elif hasattr(pixel_bndl_model, "linear"):
        output_weight = pixel_bndl_model.linear.weight  # [K, C]
        var_logits = torch.einsum("bhwc,kc->bhwk", var_z_x, output_weight**2)

    elif hasattr(pixel_bndl_model, "linear_output"):
        output_weight = pixel_bndl_model.linear_output.weight  # [K, C]
        var_logits = torch.einsum("bhwc,kc->bhwk", var_z_x, output_weight**2)
    else:
        var_logits = var_z_x.mean(dim=-1, keepdim=True).expand(B, H, W, K)

    # Scale by logit_scale² (since logits = logit_scale * dot + bias)
    if hasattr(pixel_bndl_model, "logit_scale"):
        var_logits = var_logits * (pixel_bndl_model.logit_scale**2)

    # Step 4: Compute E[sigmoid(logits)] via MacKay approximation
    # For X ~ N(μ, σ²): E[sigmoid(X)] ≈ sigmoid(κ * μ)
    # where κ = 1 / √(1 + π*σ²/8)
    # Since var_logits >= 0, denominator = 1 + π*var/8 >= 1, no clamp needed
    denominator = 1.0 + PI_OVER_8 * var_logits
    kappa_sigmoid = 1.0 / torch.sqrt(denominator)

    sigmoid_input = kappa_sigmoid * mean_logits

    mean_probs = torch.sigmoid(sigmoid_input)  # [B, H, W, K]

    # Step 5: Compute Bernoulli entropy H(p) = -p*log(p) - (1-p)*log(1-p)
    # Use torch.special.entr for numerical stability (handles p=0 and p=1 gracefully)
    if hasattr(torch.special, "entr"):
        # PyTorch >= 1.9: use built-in entr function
        # entr(p) = -p * log(p), with entr(0) = 0 by convention
        # CRITICAL FIX: Clamp probabilities to avoid Infinite gradients at p=0 or p=1
        # d/dp(-p*ln(p)) = -ln(p) - 1, which diverges as p->0.
        mean_probs = torch.clamp(mean_probs, min=1e-6, max=1.0 - 1e-6)
        entropy_per_mask = torch.special.entr(mean_probs) + torch.special.entr(1.0 - mean_probs)
    else:
        # Fallback: manual implementation with epsilon for stability
        eps = 1e-10
        entropy_per_mask = -(mean_probs * torch.log(mean_probs + eps) + (1.0 - mean_probs) * torch.log(1.0 - mean_probs + eps))

    # Step 6: Normalize entropy to [0, 1] by dividing by max Bernoulli entropy log(2)
    # This makes uncertainty directly interpretable: 0 = certain, 1 = max uncertain
    # Also ensures consistent scale with other inputs (e.g., base_iou ∈ [0, 1])
    entropy_per_mask = entropy_per_mask / LOG_2

    # Return per-channel or aggregated
    if per_channel:
        return entropy_per_mask  # [B, H, W, K], range [0, 1]
    else:
        # Average across masks (backward compatible with sampling version)
        return entropy_per_mask.mean(dim=-1)  # [B, H, W], range [0, 1]


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
