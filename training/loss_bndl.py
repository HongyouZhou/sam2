import torch
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY


def _connected_zero(*refs: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Return a scalar 0 that is connected to the autograd graph.

    Used for rare fallback paths (NaN/Inf) so backward() doesn't error,
    without creating many requires_grad=True leaf tensors.
    """
    valid_refs = [r for r in refs if isinstance(r, torch.Tensor)]
    if not valid_refs:
        if device is None:
            device = torch.device("cpu")
        return torch.zeros((), device=device, requires_grad=True)

    total = None
    for r in valid_refs:
        r_safe = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        s = r_safe.sum()
        total = s if total is None else (total + s)
    return total * 0.0


def _uncertainty_regularization(
    inv_k: torch.Tensor,
    pixel_logits: torch.Tensor,
    pixel_gt: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """UR-style uncertainty regularization for BNDL.

    Follows UR-ERN idea: ensure gradients exist in high uncertainty regions.

    Args:
        inv_k: [B, H, W, 1] - 1/kappa, larger = higher uncertainty
        pixel_logits: [B, H, W, K] - mask predictions (K=4 for multimask)
        pixel_gt: [B, 1, H_gt, W_gt] - ground truth mask in BCHW format
        eps: numerical stability

    Returns:
        Scalar loss encouraging learning from high-uncertainty mistakes
    """
    # Compute prediction error
    pred_prob = pixel_logits.sigmoid()  # [B, H, W, K]
    B, H, W, K = pred_prob.shape

    # Handle GT shape: convert from BCHW to BHW format first
    if pixel_gt.ndim == 4:
        # [B, C, H_gt, W_gt] -> [B, H_gt, W_gt]
        pixel_gt = pixel_gt.squeeze(1)  # Remove channel dim

    # Resize GT to match prediction resolution if needed
    if pixel_gt.shape[-2:] != (H, W):
        pixel_gt = F.interpolate(
            pixel_gt.unsqueeze(1).float(),  # [B, 1, H_gt, W_gt]
            size=(H, W),
            mode="nearest",
        ).squeeze(1)  # [B, H, W]

    # Expand GT to match K masks: [B, H, W] -> [B, H, W, K]
    pixel_gt = pixel_gt.unsqueeze(-1).expand(-1, -1, -1, K)  # [B, H, W, K]

    pred_error = (pred_prob - pixel_gt.float()).abs()  # [B, H, W, K]

    # inv_k is [B, H, W, 1], expand to match pred_error
    uncertainty = inv_k.expand_as(pred_error)  # 1/kappa = high uncertainty

    # UR loss: log(1 + uncertainty) * error
    # When uncertainty is high AND error is high, loss is large
    # This ensures gradients flow even in high-uncertainty regions
    ur_loss = torch.log1p(uncertainty + eps) * pred_error

    return ur_loss.mean()


class BNDLLoss(nn.Module):
    def __init__(
        self,
        kl_weight=1e-6,
        kl_weight_x: float | None = None,
        kl_weight_w: float | None = None,
        use_global_w_kl: bool = True,
        use_hyper_w_kl: bool = True,
        prior_gamma_shape: float = 1.0,
        prior_gamma_scale: float = 1.0,
        # UR-style regularization
        ur_weight: float = 0.0,  # 0 = disabled by default
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.kl_weight_x = kl_weight_x
        self.kl_weight_w = kl_weight_w
        self.use_global_w_kl = use_global_w_kl
        self.use_hyper_w_kl = use_hyper_w_kl
        self.prior_gamma_shape = prior_gamma_shape
        self.prior_gamma_scale = prior_gamma_scale
        self.ur_weight = ur_weight

    def forward(self, outs_batch: list[dict], targets_batch: torch.Tensor):
        """
        KL divergence loss for BNDL
        Args:
            outs_batch: sam ouputs
            targets_batch: targets, not used here
        """
        # Get device and initialize all accumulators as tensors to preserve gradients
        device = targets_batch.device if targets_batch is not None else torch.device("cpu")

        total_kl_loss = torch.zeros((), device=device)
        total_ur_loss = torch.zeros((), device=device)
        valid_samples = 0

        zero_ref: torch.Tensor | None = None

        for batch_idx, outs in enumerate(outs_batch):
            # 从统一的 aux_outputs 中提取 BNDL 命名空间
            if "multistep_aux_outputs" in outs:
                aux_list = outs["multistep_aux_outputs"]
                bndl_outputs_list = [aux.get("bndl") if isinstance(aux, dict) else None for aux in aux_list]
            elif "multistep_bndl_outputs" in outs:
                # 向后兼容旧键名
                bndl_outputs_list = outs["multistep_bndl_outputs"]
            else:
                continue

            # Initialize step accumulators as tensors
            step_kl_loss = torch.zeros((), device=device)
            step_ur_loss = torch.zeros((), device=device)
            valid_steps = 0

            for step_idx, bndl_outputs in enumerate(bndl_outputs_list):
                if bndl_outputs is not None:
                    if zero_ref is None and isinstance(bndl_outputs, dict):
                        if "wei_lambda_pos" in bndl_outputs:
                            zero_ref = bndl_outputs.get("wei_lambda_pos")
                        else:
                            zero_ref = bndl_outputs.get("wei_lambda")

                    # KL loss
                    kl_loss = self._compute_kl_loss(bndl_outputs)
                    step_kl_loss = step_kl_loss + kl_loss

                    # UR loss (if enabled and data available)
                    if self.ur_weight > 0:
                        inv_k = bndl_outputs.get("inv_k")
                        pixel_logits = bndl_outputs.get("pixel_logits")
                        pixel_gt = bndl_outputs.get("pixel_gt")

                        if inv_k is not None and pixel_logits is not None and pixel_gt is not None:
                            ur_loss = _uncertainty_regularization(inv_k, pixel_logits, pixel_gt)
                            step_ur_loss = step_ur_loss + ur_loss

                    valid_steps += 1

            if valid_steps > 0:
                avg_step_kl = step_kl_loss / valid_steps
                avg_step_ur = step_ur_loss / valid_steps

                total_kl_loss = total_kl_loss + avg_step_kl
                total_ur_loss = total_ur_loss + avg_step_ur

                valid_samples += 1

        if valid_samples > 0:
            kl_loss = total_kl_loss / valid_samples
            ur_loss = total_ur_loss / valid_samples
        else:
            kl_loss = _connected_zero(zero_ref, device=device)
            ur_loss = torch.zeros((), device=device)

        # Combine losses
        core_loss = kl_loss + self.ur_weight * ur_loss

        return {
            CORE_LOSS_KEY: core_loss,
            "kl_divergence": kl_loss,
            "ur_loss": ur_loss,
        }

    @staticmethod
    def KL_GamWei(Gam_shape, Gam_scale, Wei_shape_res, Wei_scale):
        def log_max(input, SMALL=1e-10):
            input_ = torch.maximum(input, input.new_tensor(SMALL))
            return torch.log(input_)

        # Simple parameter clamping
        Wei_shape_res = torch.clamp(Wei_shape_res, min=1e-10, max=1e3)
        Wei_scale = torch.clamp(Wei_scale, min=1e-10, max=1e6)

        eulergamma = torch.tensor(0.5772, dtype=torch.float32, requires_grad=False)

        part1 = Gam_shape * log_max(Wei_scale) - eulergamma.to(Wei_scale.device) * Gam_shape * Wei_shape_res + log_max(Wei_shape_res)
        part2 = -Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + Wei_shape_res))
        part3 = eulergamma.to(Wei_scale.device) + 1 + Gam_shape * log_max(Gam_scale) - torch.lgamma(Gam_shape)

        KL = part1 + part2 + part3

        # Simple NaN check - return 0 if computation failed
        if torch.any(torch.isnan(KL)) or torch.any(torch.isinf(KL)):
            return _connected_zero(Wei_shape_res, Wei_scale, device=KL.device)

        # Sum over the last dimension (feature dimension) and mean over batch/spatial
        if KL.dim() > 0:
            kl_mean = -torch.clamp(KL.sum(-1).mean(), min=-1000, max=1000)
        else:
            kl_mean = -torch.clamp(KL, min=-1000, max=1000)

        return kl_mean

    def _compute_kl_loss(self, bndl_outputs):
        # Use configurable prior parameters
        gamma_shape_val = self.prior_gamma_shape
        gamma_scale_val = self.prior_gamma_scale

        def kl_term(wei_lambda, inv_k):
            # Simple input check
            if torch.any(torch.isnan(wei_lambda)) or torch.any(torch.isnan(inv_k)):
                return _connected_zero(wei_lambda, inv_k, device=wei_lambda.device)

            wei_lambda = wei_lambda.float()
            inv_k = inv_k.float()

            gamma_shape = wei_lambda.new_tensor(gamma_shape_val, dtype=torch.float32)
            gamma_scale = wei_lambda.new_tensor(gamma_scale_val, dtype=torch.float32)

            kl_loss = BNDLLoss.KL_GamWei(gamma_shape, gamma_scale, inv_k, wei_lambda)

            # Simple NaN check
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                return _connected_zero(wei_lambda, inv_k, device=kl_loss.device)

            return kl_loss

        dev = None
        for v in bndl_outputs.values():
            if isinstance(v, torch.Tensor):
                dev = v.device
                break
        if dev is None:
            dev = torch.device("cpu")

        # 像素级KL散度 (Local sparsity)
        if "wei_lambda" in bndl_outputs and "inv_k" in bndl_outputs:
            KL_x = kl_term(bndl_outputs["wei_lambda"], bndl_outputs["inv_k"])
        elif "wei_lambda" in bndl_outputs and "kappa" in bndl_outputs:  # Legacy fallback
            KL_x = kl_term(bndl_outputs["wei_lambda"], 1.0 / bndl_outputs["kappa"])
        else:
            KL_x = torch.zeros((), device=dev)

        # Prompt-level KL散度 (Global sparsity via hyper_in)
        KL_w = torch.zeros((), device=dev)
        has_prompt_terms = False

        if self.use_global_w_kl and self.use_hyper_w_kl:
            if "wei_lambda_w" in bndl_outputs and "inv_k_w" in bndl_outputs:
                has_prompt_terms = True
                KL_w = kl_term(bndl_outputs["wei_lambda_w"], bndl_outputs["inv_k_w"])
            elif "wei_lambda_w" in bndl_outputs and "kappa_w" in bndl_outputs:  # Legacy fallback
                has_prompt_terms = True
                KL_w = kl_term(bndl_outputs["wei_lambda_w"], 1.0 / bndl_outputs["kappa_w"])

        # 像素KL保持主导地位，prompt KL作为辅助正则化
        pixel_kl_weight = self.kl_weight_x if self.kl_weight_x is not None else self.kl_weight
        if has_prompt_terms:
            prompt_kl_weight = self.kl_weight_w if self.kl_weight_w is not None else self.kl_weight
        else:
            prompt_kl_weight = 0.0

        total_loss = KL_x * pixel_kl_weight + KL_w * prompt_kl_weight

        return total_loss
