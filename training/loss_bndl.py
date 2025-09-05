import torch
import torch.nn as nn

from training.trainer import CORE_LOSS_KEY


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
        total_loss = 0.0
        valid_samples = 0
        
        # Initialize accumulators for individual part losses
        total_part1_x = 0.0
        total_part2_x = 0.0
        total_part3_x = 0.0
        total_part1_w = 0.0
        total_part2_w = 0.0
        total_part3_w = 0.0

        for outs in outs_batch:
            if "multistep_bndl_outputs" in outs:
                bndl_outputs_list = outs["multistep_bndl_outputs"]
                step_loss = 0.0
                valid_steps = 0
                
                # Initialize step accumulators for individual part losses
                step_part1_x = 0.0
                step_part2_x = 0.0
                step_part3_x = 0.0
                step_part1_w = 0.0
                step_part2_w = 0.0
                step_part3_w = 0.0
                
                for bndl_outputs in bndl_outputs_list:
                    if bndl_outputs is not None:
                        loss, part_losses = self._compute_kl_loss(bndl_outputs)
                        step_loss += loss
                        
                        # Accumulate individual part losses
                        step_part1_x += part_losses['part1_x']
                        step_part2_x += part_losses['part2_x']
                        step_part3_x += part_losses['part3_x']
                        step_part1_w += part_losses['part1_w']
                        step_part2_w += part_losses['part2_w']
                        step_part3_w += part_losses['part3_w']
                        
                        valid_steps += 1
                
                if valid_steps > 0:
                    total_loss += step_loss / valid_steps
                    
                    # Accumulate averaged step part losses
                    total_part1_x += step_part1_x / valid_steps
                    total_part2_x += step_part2_x / valid_steps
                    total_part3_x += step_part3_x / valid_steps
                    total_part1_w += step_part1_w / valid_steps
                    total_part2_w += step_part2_w / valid_steps
                    total_part3_w += step_part3_w / valid_steps
                    
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
            device = targets_batch.device if targets_batch is not None else torch.device("cpu")
            core_loss = torch.tensor(0.0, device=device, requires_grad=True)
            avg_part1_x = torch.tensor(0.0, device=device, requires_grad=True)
            avg_part2_x = torch.tensor(0.0, device=device, requires_grad=True)
            avg_part3_x = torch.tensor(0.0, device=device, requires_grad=True)
            avg_part1_w = torch.tensor(0.0, device=device, requires_grad=True)
            avg_part2_w = torch.tensor(0.0, device=device, requires_grad=True)
            avg_part3_w = torch.tensor(0.0, device=device, requires_grad=True)

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
            device = input.device
            input_ = torch.max(input, torch.tensor([SMALL]).to(device))
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
            return torch.tensor(0.0, device=KL.device, requires_grad=True), torch.tensor(0.0, device=KL.device, requires_grad=True), torch.tensor(0.0, device=KL.device, requires_grad=True), torch.tensor(0.0, device=KL.device, requires_grad=True)
        
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
                return torch.tensor(0.0, device=wei_lambda.device, requires_grad=True), torch.tensor(0.0, device=wei_lambda.device, requires_grad=True), torch.tensor(0.0, device=wei_lambda.device, requires_grad=True), torch.tensor(0.0, device=wei_lambda.device, requires_grad=True)
                
            wei_lambda = wei_lambda.float()
            inv_k = inv_k.float()
            
            gamma_shape = torch.tensor(1.0, dtype=torch.float32, device=wei_lambda.device)
            gamma_scale = torch.tensor(1.0, dtype=torch.float32, device=wei_lambda.device)
            
            kl_loss, part1_loss, part2_loss, part3_loss = BNDLLoss.KL_GamWei(gamma_shape, gamma_scale, inv_k, wei_lambda)
            
            # Simple NaN check
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                return torch.tensor(0.0, device=kl_loss.device, requires_grad=True), torch.tensor(0.0, device=kl_loss.device, requires_grad=True), torch.tensor(0.0, device=kl_loss.device, requires_grad=True), torch.tensor(0.0, device=kl_loss.device, requires_grad=True)
            
            return kl_loss, part1_loss, part2_loss, part3_loss

        # 像素级KL散度 (Local sparsity)
        KL_x, part1_x, part2_x, part3_x = kl_term(bndl_outputs["wei_lambda"], bndl_outputs["inv_k"], "pixel")

        # Prompt-level KL散度 (Global sparsity via hyper_in)
        KL_w, part1_w, part2_w, part3_w = 0.0, 0.0, 0.0, 0.0
        if (self.use_global_w_kl and 
            bndl_outputs.get("wei_lambda_w") is not None and 
            bndl_outputs.get("inv_k_w") is not None):
            KL_w, part1_w, part2_w, part3_w = kl_term(bndl_outputs["wei_lambda_w"], bndl_outputs["inv_k_w"], "prompt_hyper_in")

        # 像素KL保持主导地位，prompt KL作为辅助正则化
        pixel_kl_weight = self.kl_weight
        prompt_kl_weight = self.kl_weight * 0.05 if KL_w != 0 else 0  # 增加到0.05，但仍保持适中
        
        # 添加自适应权重调整
        if KL_w != 0:
            # 如果prompt KL过大，降低其权重
            kl_ratio = KL_w.abs() / (KL_x.abs() + 1e-8)
            if kl_ratio > 10.0:  # 如果prompt KL比pixel KL大10倍以上
                prompt_kl_weight = prompt_kl_weight * 0.1  # 进一步降低权重
        
        total_loss = KL_x * pixel_kl_weight + KL_w * prompt_kl_weight
        
        # Return both the total loss and individual part losses for logging
        return total_loss, {
            'part1_x': part1_x,
            'part2_x': part2_x, 
            'part3_x': part3_x,
            'part1_w': part1_w,
            'part2_w': part2_w,
            'part3_w': part3_w
        }



