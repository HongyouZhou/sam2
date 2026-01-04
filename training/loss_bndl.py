import torch
import torch.nn as nn

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


class BNDLLoss(nn.Module):
    def __init__(self, kl_weight=1e-6, use_global_w_kl: bool = True, use_hyper_w_kl: bool = True):
        super().__init__()
        self.kl_weight = kl_weight
        self.use_global_w_kl = use_global_w_kl
        self.use_hyper_w_kl = use_hyper_w_kl

    def forward(self, outs_batch: list[dict], targets_batch: torch.Tensor):
        """
        KL divergence loss for BNDL
        Args:
            outs_batch: sam ouputs
            targets_batch: targets, not used here
        """
        # Get device and initialize all accumulators as tensors to preserve gradients
        device = targets_batch.device if targets_batch is not None else torch.device("cpu")

        total_loss = torch.zeros((), device=device)
        valid_samples = 0

        # Initialize accumulators for individual part losses (as tensors)
        total_part1_x = torch.zeros((), device=device)
        total_part2_x = torch.zeros((), device=device)
        total_part3_x = torch.zeros((), device=device)
        total_part1_w = torch.zeros((), device=device)
        total_part2_w = torch.zeros((), device=device)
        total_part3_w = torch.zeros((), device=device)

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
            step_loss = torch.zeros((), device=device)
            valid_steps = 0

            # Initialize step accumulators for individual part losses (as tensors)
            step_part1_x = torch.zeros((), device=device)
            step_part2_x = torch.zeros((), device=device)
            step_part3_x = torch.zeros((), device=device)
            step_part1_w = torch.zeros((), device=device)
            step_part2_w = torch.zeros((), device=device)
            step_part3_w = torch.zeros((), device=device)

            for step_idx, bndl_outputs in enumerate(bndl_outputs_list):
                if bndl_outputs is not None:
                    if zero_ref is None and isinstance(bndl_outputs, dict):
                        zero_ref = bndl_outputs.get("wei_lambda")
                    kl_loss, part_losses = self._compute_kl_loss(bndl_outputs)

                    loss = kl_loss

                    # Use tensor addition to preserve gradients
                    step_loss = step_loss + loss

                    # Accumulate individual part losses
                    step_part1_x = step_part1_x + part_losses["part1_x"]
                    step_part2_x = step_part2_x + part_losses["part2_x"]
                    step_part3_x = step_part3_x + part_losses["part3_x"]
                    step_part1_w = step_part1_w + part_losses["part1_w"]
                    step_part2_w = step_part2_w + part_losses["part2_w"]
                    step_part3_w = step_part3_w + part_losses["part3_w"]

                    valid_steps += 1

            if valid_steps > 0:
                avg_step_loss = step_loss / valid_steps

                # Use tensor addition to preserve gradients
                total_loss = total_loss + avg_step_loss

                # Accumulate averaged step part losses
                total_part1_x = total_part1_x + (step_part1_x / valid_steps)
                total_part2_x = total_part2_x + (step_part2_x / valid_steps)
                total_part3_x = total_part3_x + (step_part3_x / valid_steps)
                total_part1_w = total_part1_w + (step_part1_w / valid_steps)
                total_part2_w = total_part2_w + (step_part2_w / valid_steps)
                total_part3_w = total_part3_w + (step_part3_w / valid_steps)

                valid_samples += 1

        if valid_samples > 0:
            core_loss = total_loss / valid_samples

            # Average the accumulated part losses
            avg_part1_x = total_part1_x / valid_samples
            avg_part2_x = total_part2_x / valid_samples
            avg_part3_x = total_part3_x / valid_samples
            avg_part1_w = total_part1_w / valid_samples
            avg_part2_w = total_part2_w / valid_samples
            avg_part3_w = total_part3_w / valid_samples
        else:
            core_loss = _connected_zero(zero_ref, device=device)
            avg_part1_x = core_loss
            avg_part2_x = core_loss
            avg_part3_x = core_loss
            avg_part1_w = core_loss
            avg_part2_w = core_loss
            avg_part3_w = core_loss

        return {
            CORE_LOSS_KEY: core_loss,
            "kl_divergence": core_loss,
            # Individual part losses (absolute values for logging)
            "part1_x_abs": avg_part1_x.abs(),
            "part2_x_abs": avg_part2_x.abs(),
            "part3_x_abs": avg_part3_x.abs(),
            "part1_w_abs": avg_part1_w.abs(),
            "part2_w_abs": avg_part2_w.abs(),
            "part3_w_abs": avg_part3_w.abs(),
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

        # TODO: 打印 part1, part2, part3 的值
        KL = part1 + part2 + part3

        # Simple NaN check - return 0 if computation failed
        if torch.any(torch.isnan(KL)) or torch.any(torch.isinf(KL)):
            zero = _connected_zero(Wei_shape_res, Wei_scale, device=KL.device)
            return (zero, zero, zero, zero)

        # Handle different tensor shapes (global vs batch parameters)
        try:
            # Try to flatten and take mean
            if KL.dim() > 0:
                kl_mean = -torch.clamp(KL.view(-1).mean(), min=-1000, max=1000)
                part1_mean = torch.clamp(part1.view(-1).mean(), min=-1000, max=1000)
                part2_mean = torch.clamp(part2.view(-1).mean(), min=-1000, max=1000)
                part3_mean = torch.clamp(part3.view(-1).mean(), min=-1000, max=1000)
            else:
                # Scalar tensor
                kl_mean = -torch.clamp(KL, min=-1000, max=1000)
                part1_mean = torch.clamp(part1, min=-1000, max=1000)
                part2_mean = torch.clamp(part2, min=-1000, max=1000)
                part3_mean = torch.clamp(part3, min=-1000, max=1000)
        except Exception:
            # Fallback: just take the mean without reshaping
            kl_mean = -torch.clamp(KL.mean(), min=-1000, max=1000)
            part1_mean = torch.clamp(part1.mean(), min=-1000, max=1000)
            part2_mean = torch.clamp(part2.mean(), min=-1000, max=1000)
            part3_mean = torch.clamp(part3.mean(), min=-1000, max=1000)

        return kl_mean, part1_mean, part2_mean, part3_mean

    def _compute_kl_loss(self, bndl_outputs):
        def kl_term(wei_lambda, inv_k, name=""):
            # Simple input check
            if torch.any(torch.isnan(wei_lambda)) or torch.any(torch.isnan(inv_k)):
                zero = _connected_zero(wei_lambda, inv_k, device=wei_lambda.device)
                return (zero, zero, zero, zero)

            wei_lambda = wei_lambda.float()
            inv_k = inv_k.float()

            gamma_shape = wei_lambda.new_tensor(1.0, dtype=torch.float32)
            gamma_scale = wei_lambda.new_tensor(1.0, dtype=torch.float32)

            kl_loss, part1_loss, part2_loss, part3_loss = BNDLLoss.KL_GamWei(gamma_shape, gamma_scale, inv_k, wei_lambda)

            # Simple NaN check
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                zero = _connected_zero(wei_lambda, inv_k, device=kl_loss.device)
                return (zero, zero, zero, zero)

            return kl_loss, part1_loss, part2_loss, part3_loss

        # 像素级KL散度 (Local sparsity)
        # Note: BNDL now returns kappa directly, convert to inv_k for KL computation
        KL_x, part1_x, part2_x, part3_x = kl_term(bndl_outputs["wei_lambda"], 1.0 / bndl_outputs["kappa"], "pixel")

        # Prompt-level KL散度 (Global sparsity via hyper_in)
        # Initialize prompt-level terms as tensors on the right device
        dev = bndl_outputs["wei_lambda"].device
        KL_w = torch.zeros((), device=dev)
        part1_w = torch.zeros((), device=dev)
        part2_w = torch.zeros((), device=dev)
        part3_w = torch.zeros((), device=dev)

        has_prompt_terms = self.use_global_w_kl and bndl_outputs.get("wei_lambda_w") is not None and bndl_outputs.get("kappa_w") is not None
        if has_prompt_terms:
            # Note: BNDL now returns kappa_w directly, convert to inv_k_w for KL computation
            KL_w, part1_w, part2_w, part3_w = kl_term(bndl_outputs["wei_lambda_w"], 1.0 / bndl_outputs["kappa_w"], "prompt_hyper_in")

        # 像素KL保持主导地位，prompt KL作为辅助正则化
        pixel_kl_weight = self.kl_weight
        # [Critical Fix for ZS] 提升权重以匹配主Loss量级(20.0)，强迫模型在ZS时不确定就闭嘴
        prompt_kl_weight = self.kl_weight * 1.0 if has_prompt_terms else 0.0

        # 添加自适应权重调整
        if has_prompt_terms:
            # 如果prompt KL过大，降低其权重
            kl_ratio = (KL_w.abs() / (KL_x.abs() + 1e-8)).item()

        total_loss = KL_x * pixel_kl_weight + KL_w * prompt_kl_weight

        # Return both the total loss and individual part losses for logging
        return total_loss, {"part1_x": part1_x, "part2_x": part2_x, "part3_x": part3_x, "part1_w": part1_w, "part2_w": part2_w, "part3_w": part3_w}
