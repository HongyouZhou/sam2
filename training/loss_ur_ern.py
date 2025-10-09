import torch
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x
    if x.ndim == 3:  # [B,H,W] -> [B,1,H,W]
        return x.unsqueeze(1)
    raise ValueError(f"Expected 3D/4D tensor, got shape {tuple(x.shape)}")


def _student_t_nll(mu: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Student-t negative log-likelihood (stable surrogate) per-pixel.

    dof = 2*alpha; scale^2 = beta*(1+v)/(v*alpha)
    nll ≈ 0.5*(dof+1) * log1p( (y-mu)^2 / (dof*scale^2) )
    """
    dof = 2.0 * alpha
    scale2 = (beta * (1.0 + v)) / (v * alpha + eps)
    resid2 = (y - mu) ** 2
    denom = dof * scale2 + eps
    nll = 0.5 * (dof + 1.0) * torch.log1p(resid2 / denom)
    return nll


def _ur_regularizer(mu: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """UR-ERN style uncertainty regularizer: penalize overconfidence on large residuals.

    Use simple surrogate: (1 - exp(-|y-mu|)) * confidence, where confidence ~ 1/epistemic.
    epistemic = beta / (v * (alpha - 1)).
    """
    resid = (y - mu).abs()
    epistemic = beta / (v * (alpha - 1.0) + eps)
    confidence = torch.clamp(1.0 / (epistemic + eps), 0.0, 1e6)
    reg = (1.0 - torch.exp(-resid)) * confidence
    return reg


class URERNLoss(nn.Module):
    """UR-ERN loss for SAM: Student-t NLL + Uncertainty Regularizer.

    This module expects each element of outs_batch to be a dict produced by the SAM training loop,
    containing key 'aux_outputs' with namespace 'ur_ern':
        outs['aux_outputs']['ur_ern'] = { 'nig_mu','nig_v','nig_alpha','nig_beta' }  # [B,1,H,W]
    Targets are pixel masks [T, B, H, W] or [B, H, W] depending on dataset; we align to the NIG map size.
    """

    def __init__(self, lambda_ur: float = 1e-3, label_smoothing: float = 0.0):
        super().__init__()
        self.lambda_ur = float(lambda_ur)
        self.label_smoothing = float(label_smoothing)

    def forward(self, outs_batch: list[dict], targets_batch: torch.Tensor):
        total_nll = 0.0
        total_reg = 0.0
        total_core = 0.0
        valid = 0

        for outs in outs_batch:
            # 从统一的 multistep_aux_outputs 读取
            if "multistep_aux_outputs" in outs:
                aux_list = outs["multistep_aux_outputs"]
            elif "aux_outputs" in outs:
                # 向后兼容单步输出
                aux_list = [outs["aux_outputs"]]
            else:
                continue
            
            # 遍历所有步的 aux_outputs，提取 UR-ERN 命名空间
            for aux in aux_list:
                if not isinstance(aux, dict):
                    continue
                nig = aux.get("ur_ern", None)
                if not isinstance(nig, dict):
                    continue

                mu = _ensure_bchw(nig["nig_mu"])
                v = _ensure_bchw(nig["nig_v"])
                alpha = _ensure_bchw(nig["nig_alpha"])  # will clamp later
                beta = _ensure_bchw(nig["nig_beta"])    # >0 ensured by head

                # Prepare targets to match [B,1,H,W]
                # targets_batch typical shapes: [T,B,H,W] or [B,H,W]
                if targets_batch.ndim == 4 and targets_batch.shape[0] != mu.shape[0]:
                    # Assume [T,B,H,W] -> use first time index
                    y = targets_batch[0]
                elif targets_batch.ndim == 4 and targets_batch.shape[0] == mu.shape[0]:
                    y = targets_batch
                elif targets_batch.ndim == 3:
                    y = targets_batch
                else:
                    # Fallback: try last three dims as H,W or reduce
                    y = targets_batch.view(-1, *targets_batch.shape[-2:])
                y = _ensure_bchw(y.float())

                # Resize label to NIG spatial size if needed
                if y.shape[-2:] != mu.shape[-2:]:
                    y = F.interpolate(y, size=mu.shape[-2:], mode="bilinear", align_corners=False)

                if self.label_smoothing > 0.0:
                    y = y * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

                # Numerics: clamp alpha away from 1
                alpha = torch.clamp(alpha, min=1.0 + 1e-3)

                nll_map = _student_t_nll(mu, v, alpha, beta, y)
                reg_map = _ur_regularizer(mu, v, alpha, beta, y)

                loss_nll = nll_map.mean()
                loss_reg = reg_map.mean()
                loss_core = loss_nll + self.lambda_ur * loss_reg

                total_nll += loss_nll
                total_reg += loss_reg
                total_core += loss_core
                valid += 1

        if valid == 0:
            device = targets_batch.device if isinstance(targets_batch, torch.Tensor) else torch.device("cpu")
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {CORE_LOSS_KEY: zero, "ur_ern_nll": zero, "ur_ern_reg": zero}

        core = total_core / valid
        nll = total_nll / valid
        reg = total_reg / valid

        return {
            CORE_LOSS_KEY: core,
            "ur_ern_nll": nll,
            "ur_ern_reg": reg,
        }


