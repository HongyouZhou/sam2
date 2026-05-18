# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from BNDL.BNDL_upload.ViT_Sparse.utils.bndl import BNDL, entropy_uncertainty, uncertainty_sample_parallel
from sam2.modeling.sam2_utils import MLP, LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        # BNDL related
        use_bndl_for_pixels: bool = False,
        bndl_factor_z: float = 0.0,
        bndl_factor_w: float = 0.0,
        use_multimask_token_for_obj_ptr: bool = False,
        # UR-ERN related (mutually exclusive with BNDL)
        use_ur_ern_for_pixels: bool = False,
        # BNDL eval mode control
        bndl_force_single_sample: bool = False,  # If True, use single sampling in eval mode (match training behavior)
        bndl_sample_num: int = 20,  # MC samples for sampling-based uncertainty (used by entropy_uncertainty path)
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.bndl_sample_num = bndl_sample_num

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1, stride=1)
            self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, stride=1)

        # BNDL related
        self.use_bndl_for_pixels = use_bndl_for_pixels
        self.bndl_factor_z = float(bndl_factor_z)
        self.bndl_factor_w = float(bndl_factor_w)
        if self.use_bndl_for_pixels:
            pixel_feat_dim = transformer_dim // 8  # C' after up-scaling (32)
            self.pixel_bndl = BNDL(
                pixel_feat_dim,  # num_features: 32 (for pixel_feat)
                2,  # num_classes: binary (fg/bg)
                mask_token_dim=transformer_dim,  # 256 (for mask_tokens_out)
            )
            self.bndl_force_single_sample = bndl_force_single_sample
        #########################################################

        # UR-ERN (NIG evidential regression) head - mutually exclusive with BNDL
        self.use_ur_ern_for_pixels = bool(use_ur_ern_for_pixels)
        if self.use_bndl_for_pixels and self.use_ur_ern_for_pixels:
            raise ValueError("use_bndl_for_pixels and use_ur_ern_for_pixels cannot be both True.")

        if self.use_ur_ern_for_pixels:
            # Use C' = transformer_dim // 8 features from output_upscaling; predict 4 channels: (mu, v, alpha, beta)
            pixel_feat_dim = transformer_dim // 8
            self.ur_ern_head = nn.Conv2d(pixel_feat_dim, 4, kernel_size=1, stride=1, bias=True)
            # Small epsilon to stabilize parameter transforms at inference/training time
            self.register_buffer("_ur_eps", torch.tensor(1e-6), persistent=False)

        self.output_hypernetworks_mlps = nn.ModuleList([MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for i in range(self.num_mask_tokens)])

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        masks, iou_pred, mask_tokens_out, object_score_logits, bndl_outputs = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits, bndl_outputs

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert image_pe.size(0) == 1, "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Calculate object_score_logits using the object score token (index 0 if present)
        if self.pred_obj_scores:
            assert s == 1
            obj_token = hs[:, 0, :]  # [B, 256]
            object_score_logits = self.pred_obj_score_head(obj_token)
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_token_out.new_ones(iou_token_out.shape[0], 1)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features

            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # Generate hyper network embeddings
        hyper_in_list: list[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        masks_sam = None
        if not self.use_bndl_for_pixels:
            b, c, h, w = upscaled_embedding.shape
            masks_sam = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
            masks = masks_sam

        aux_outputs = None
        if self.use_bndl_for_pixels:
            pixel_feat = upscaled_embedding.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
            # BNDL forward
            # force_sample=True in eval mode when bndl_force_single_sample is set
            # This ensures train-eval consistency (single sampling vs MC average)
            force_sample = (not self.training) and getattr(self, "bndl_force_single_sample", False)
            (
                masks_bndl_sam,  # [B, H, W, K]
                z_out,
                wei_lambda,
                inv_k,
                wei_lambda_w,
                inv_k_w,
            ) = self.pixel_bndl(
                pixel_feat,
                mask_tokens_out,  # [B, K, 256] - use full Transformer output
                factor_z=self.bndl_factor_z,
                factor_w=self.bndl_factor_w,
                force_sample=force_sample,
            )

            masks_bndl = masks_bndl_sam.permute(0, 3, 1, 2)  # [B, K, H, W]

            # === Uncertainty estimation ===
            # 1. Sampling-based uncertainty (no gradients, for visualization/logging)
            sampled_logits, mean_logits = uncertainty_sample_parallel(
                self.pixel_bndl,
                pixel_feat,
                mask_tokens_out,  # [B, K, 256] - use full Transformer output
                sample_num=getattr(self, "bndl_sample_num", 20),
                factor_z=self.bndl_factor_z,
                factor_w=self.bndl_factor_w,
            )
            # [B, H, W, K] - per-mask entropy (no gradients due to torch.no_grad in sampling)
            pixel_uncertainty_sampling = entropy_uncertainty(sampled_logits)

            # 2. Analytic uncertainty (with gradients, for training BNDL calibration)
            # Computed from Weibull parameters: H = -p*log(p) - (1-p)*log(1-p) with MacKay approx
            # Only compute during training to save memory
            pixel_uncertainty_analytic = None
            if self.training:
                from sam2.modeling.bndl_utils import pixel_weibull_to_entropy_uncertainty

                pixel_uncertainty_analytic = pixel_weibull_to_entropy_uncertainty(
                    pixel_bndl_model=self.pixel_bndl,
                    pixel_feat=pixel_feat,
                    external_pre_out_w=mask_tokens_out,
                    per_channel=True,  # [B, H, W, K] to match sampling shape
                )

            aux_outputs = {
                "bndl": {
                    "z_out": z_out,
                    "wei_lambda": wei_lambda,
                    "inv_k": inv_k,
                    "wei_lambda_w": wei_lambda_w,
                    "inv_k_w": inv_k_w,
                    "masks_bndl_raw": masks_bndl_sam.detach(),  # [B, H, W, K]
                    # Two types of uncertainty with clear semantics:
                    # - sampling: no gradients, use for visualization, logging, attacker loss
                    # - analytic: with gradients, use for BNDL calibration (MMD loss)
                    "pixel_uncertainty_sampling": pixel_uncertainty_sampling,  # [B, H, W, K], no grad
                    "pixel_uncertainty_analytic": pixel_uncertainty_analytic,  # [B, H, W, K], with grad (or None in eval)
                    # Legacy alias for backward compatibility (points to sampling)
                    "pixel_uncertainty": pixel_uncertainty_sampling,  # [B, H, W, K]
                    "pixel_logits": masks_bndl_sam if self.training else masks_bndl_sam.detach(),
                    "upscaled_shape": (b, c, h, w),
                    "mask_tokens_out": mask_tokens_out if self.training else mask_tokens_out.detach(),
                    "pixel_feat": pixel_feat.detach(),
                    "pixel_feat_grad": pixel_feat if self.training else None,
                    "masks_bndl": masks_bndl.detach(),
                    "masks_hyper": masks_sam.detach() if masks_sam is not None else None,
                }
            }

            masks = masks_bndl

        # UR-ERN evidential outputs (NIG) using upscaled feature map; independent of BNDL branch
        if self.use_ur_ern_for_pixels:
            # Input features to UR-ERN head must match transformer_dim//8 channels
            # If high-res path was used above, upscaled_embedding already fused; reuse it here.
            nig_params = self.ur_ern_head(upscaled_embedding)  # [B, 4, H, W]
            mu, v_raw, alpha_raw, beta_raw = torch.chunk(nig_params, 4, dim=1)
            # Constrain parameters to valid domains
            v = torch.nn.functional.softplus(v_raw) + self._ur_eps  # v > 0
            alpha = torch.nn.functional.softplus(alpha_raw) + 1.0 + self._ur_eps  # alpha > 1
            beta = torch.nn.functional.softplus(beta_raw) + self._ur_eps  # beta > 0

            # Student-t predictive variance decomposition
            # var = beta * (1 + v) / (v * (alpha - 1))
            # aleatoric = beta / (alpha - 1)
            # epistemic = beta / (v * (alpha - 1))
            denom = alpha - 1.0
            aleatoric = beta / denom
            epistemic = beta / (v * denom)
            var = (beta * (1.0 + v)) / (v * denom)

            # Pack into aux dict (fixed top-level key with method namespace)
            aux_outputs = {
                "ur_ern": {
                    "nig_mu": mu,
                    "nig_v": v,
                    "nig_alpha": alpha,
                    "nig_beta": beta,
                    "nig_var": var,
                    "nig_aleatoric": aleatoric,
                    "nig_epistemic": epistemic,
                }
            }

        iou_pred = self.iou_prediction_head(iou_token_out)

        if aux_outputs is None:
            aux_outputs = {}

        # Check: if use_bndl_for_pixels=True, aux_outputs must contain valid BNDL data
        if self.use_bndl_for_pixels:
            if "bndl" not in aux_outputs:
                raise RuntimeError(f"MaskDecoder.predict_masks: use_bndl_for_pixels=True but aux_outputs does not contain 'bndl'! aux_outputs keys: {list(aux_outputs.keys())}")
            bndl_data = aux_outputs["bndl"]
            if not isinstance(bndl_data, dict):
                raise RuntimeError(f"MaskDecoder.predict_masks: aux_outputs['bndl'] is not a dict! type: {type(bndl_data)}")
            required_keys = ["masks_bndl_raw", "pixel_uncertainty", "wei_lambda", "inv_k", "wei_lambda_w", "inv_k_w"]
            missing_keys = [k for k in required_keys if k not in bndl_data]
            if missing_keys:
                raise RuntimeError(f"MaskDecoder.predict_masks: aux_outputs['bndl'] missing required keys: {missing_keys}. Available keys: {list(bndl_data.keys())}")

        return masks, iou_pred, mask_tokens_out, object_score_logits, aux_outputs

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(multimask_iou_scores.size(0), device=all_iou_scores.device)
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
