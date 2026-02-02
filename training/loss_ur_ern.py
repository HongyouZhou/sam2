import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from training.trainer import CORE_LOSS_KEY


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x
    if x.ndim == 3:  # [B,H,W] -> [B,1,H,W]
        return x.unsqueeze(1)
    raise ValueError(f"Expected 3D/4D tensor, got shape {tuple(x.shape)}")


def _student_t_nll_paper(
    mu: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Student-t NLL loss per-pixel (UR-ERN paper Eq. 4).

    L^NLL = (1/2)log(π/v) - α·log(Ω) + (α + 1/2)·log((y-γ)²·v + Ω) + log(Γ(α)/Γ(α+1/2))

    where Ω = 2β(1+v).

    For numerical stability, we use:
    - log(Γ(α)/Γ(α+0.5)) ≈ 0.5·log(α) - 0.5·log(2π) (Stirling approx for large α)
    - For small α, use torch.lgamma directly
    """
    omega = 2.0 * beta * (1.0 + v)
    resid2 = (y - mu) ** 2

    # Term 1: (1/2) * log(π/v)
    term1 = 0.5 * torch.log(torch.tensor(math.pi, device=mu.device, dtype=mu.dtype) / (v + eps))

    # Term 2: -α * log(Ω)
    term2 = -alpha * torch.log(omega + eps)

    # Term 3: (α + 1/2) * log((y-γ)²·v + Ω)
    term3 = (alpha + 0.5) * torch.log(resid2 * v + omega + eps)

    # Term 4: log(Γ(α)) - log(Γ(α + 1/2))
    term4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

    nll = term1 + term2 + term3 + term4
    return nll


def _evidence_regularizer_paper(
    mu: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Original ERN evidence regularizer (Eq. 5): L^R = |y - γ| · (2v + α).

    Minimizes evidence on incorrect predictions.
    """
    resid = (y - mu).abs()
    lr = resid * (2.0 * v + alpha)
    return lr


def _ur_regularizer_paper(
    mu: torch.Tensor,
    alpha: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """UR-ERN uncertainty regularizer (Eq. 10): L^U = |y - γ|² / ln(α).

    Key innovation of UR-ERN paper: addresses zero gradient problem in High Uncertainty Area (HUA)
    where α → 1. The 1/ln(α) term provides non-zero gradients when α is close to 1.

    Note: α > 1 is guaranteed by the softplus + 1 activation in the head.
    """
    resid_sq = (y - mu) ** 2
    # ln(α) requires α > 1, which is ensured by head (softplus + 1 + eps)
    # Clamp to avoid log(1) = 0 division
    log_alpha = torch.log(torch.clamp(alpha, min=1.0 + eps))
    lu = resid_sq / (log_alpha + eps)
    return lu


class URERNLoss(nn.Module):
    """UR-ERN loss for SAM finetune (arXiv:2401.01484).

    Implements the full UR-ERN training objective (Eq. 11):
        L^UR-ERN = L^NLL + λ·L^R + λ₁·L^U

    where:
        - L^NLL: Student-t negative log-likelihood (Eq. 4)
        - L^R: Evidence regularizer |y-γ|·(2v+α) (Eq. 5)
        - L^U: Uncertainty regularizer |y-γ|²/ln(α) (Eq. 10) - core innovation

    This module expects aux_outputs['ur_ern'] = {nig_mu, nig_v, nig_alpha, nig_beta}.
    """

    def __init__(
        self,
        lambda_r: float = 0.01,  # λ for L^R (evidence regularizer)
        lambda_u: float = 0.001,  # λ₁ for L^U (uncertainty regularizer)
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_r = float(lambda_r)
        self.lambda_u = float(lambda_u)
        self.label_smoothing = float(label_smoothing)

    def forward(self, outs_batch: list[dict], targets_batch: torch.Tensor):
        total_nll = 0.0
        total_lr = 0.0  # L^R evidence regularizer
        total_lu = 0.0  # L^U uncertainty regularizer
        total_core = 0.0
        valid = 0

        for outs in outs_batch:
            # Read from unified multistep_aux_outputs
            if "multistep_aux_outputs" in outs:
                aux_list = outs["multistep_aux_outputs"]
            elif "aux_outputs" in outs:
                # Backward compatible with single-step output
                aux_list = [outs["aux_outputs"]]
            else:
                continue

            # Iterate all steps' aux_outputs, extract UR-ERN namespace
            for aux in aux_list:
                if not isinstance(aux, dict):
                    continue
                nig = aux.get("ur_ern", None)
                if not isinstance(nig, dict):
                    continue

                mu = _ensure_bchw(nig["nig_mu"])
                v = _ensure_bchw(nig["nig_v"])
                alpha = _ensure_bchw(nig["nig_alpha"])
                beta = _ensure_bchw(nig["nig_beta"])

                # Prepare targets to match [B,1,H,W]
                if targets_batch.ndim == 4 and targets_batch.shape[0] != mu.shape[0]:
                    # Assume [T,B,H,W] -> use first time index
                    y = targets_batch[0]
                elif targets_batch.ndim == 4 and targets_batch.shape[0] == mu.shape[0]:
                    y = targets_batch
                elif targets_batch.ndim == 3:
                    y = targets_batch
                else:
                    y = targets_batch.view(-1, *targets_batch.shape[-2:])
                y = _ensure_bchw(y.float())

                # Resize label to NIG spatial size if needed
                if y.shape[-2:] != mu.shape[-2:]:
                    y = F.interpolate(y, size=mu.shape[-2:], mode="bilinear", align_corners=False)

                if self.label_smoothing > 0.0:
                    y = y * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

                # Numerics: clamp alpha away from 1 (head already ensures α > 1 + eps)
                alpha = torch.clamp(alpha, min=1.0 + 1e-3)

                # Compute three loss terms (Eq. 11)
                nll_map = _student_t_nll_paper(mu, v, alpha, beta, y)
                lr_map = _evidence_regularizer_paper(mu, v, alpha, y)
                lu_map = _ur_regularizer_paper(mu, alpha, y)

                loss_nll = nll_map.mean()
                loss_lr = lr_map.mean()
                loss_lu = lu_map.mean()

                # L^UR-ERN = L^NLL + λ·L^R + λ₁·L^U (Eq. 11)
                loss_core = loss_nll + self.lambda_r * loss_lr + self.lambda_u * loss_lu

                total_nll += loss_nll
                total_lr += loss_lr
                total_lu += loss_lu
                total_core += loss_core
                valid += 1

        if valid == 0:
            device = targets_batch.device if isinstance(targets_batch, torch.Tensor) else torch.device("cpu")
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {CORE_LOSS_KEY: zero, "ur_ern_nll": zero, "ur_ern_lr": zero, "ur_ern_lu": zero}

        core = total_core / valid
        nll = total_nll / valid
        lr = total_lr / valid
        lu = total_lu / valid

        return {
            CORE_LOSS_KEY: core,
            "ur_ern_nll": nll,  # L^NLL
            "ur_ern_lr": lr,  # L^R (evidence regularizer)
            "ur_ern_lu": lu,  # L^U (uncertainty regularizer - UR-ERN innovation)
        }
