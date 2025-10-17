# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import random

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames
from BNDL.BNDL_upload.ViT_Sparse.utils.bndl import (
    pixel_uncertain_sampling,
    pixel_entropy_uncertainty,
)

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class UAShiftGuidedPseudoPromptGenerator(torch.nn.Module):
    """
    UA-Shift-Guided Pseudo Prompt Generator
    
    受ICML'25 "Unlocking the Power of SAM 2 for Few-Shot Segmentation"启发
    但应用于adversarial training而非few-shot learning
    
    核心理论:
    - 在UA distribution shift最大的spatial regions生成prompts
    - 不是在uncertainty最大的regions (理论一致性)
    - Focus learning on critical shift bottlenecks
    """
    def __init__(self, feat_dim: int = 256, prompt_dim: int = 256):
        super().__init__()
        
        # Prompt aggregation MLP
        # Input: features from top UA-shift regions
        # Output: prompt embedding
        self.prompt_mlp = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, prompt_dim)
        )
    
    def forward(
        self,
        adv_feat: torch.Tensor,           # [M, H, W, C]
        ua_shift_map: torch.Tensor,       # [B, H, W]
        top_k: int = 10,
    ) -> torch.Tensor:
        """
        从UA shift map采样features生成pseudo prompts
        
        Args:
            adv_feat: adversarial features
            ua_shift_map: local UA distribution shift map (不是uncertainty map!)
            top_k: 采样top-k个shift最大的locations
        
        Returns:
            pseudo_prompts: [M, prompt_dim]
        """
        M, H, W, C = adv_feat.shape
        B = ua_shift_map.shape[0]
        
        pseudo_prompts = []
        
        for i in range(M):
            # 当前adversarial sample对应的shift map
            shift_map_i = ua_shift_map[i % B]  # [H, W]
            
            # Top-k UA shift locations
            # 关键理论: 不是top-k uncertainty，而是top-k distribution shift
            shift_flat = shift_map_i.flatten()
            
            # 确保有足够的locations
            k_actual = min(top_k, shift_flat.numel())
            if k_actual == 0:
                # Fallback: use mean features
                pseudo_prompts.append(adv_feat[i].mean(dim=[0, 1]))
                continue
            
            _, topk_indices = torch.topk(shift_flat, k_actual)
            
            # 这些locations的adversarial features
            adv_feat_flat = adv_feat[i].view(-1, C)  # [H*W, C]
            critical_features = adv_feat_flat[topk_indices]  # [k, C]
            
            # Aggregate通过MLP生成prompt
            # 类似Few-Shot论文的Pseudo Prompt Generator
            prompt = self.prompt_mlp(critical_features.mean(dim=0))  # [prompt_dim]
            
            pseudo_prompts.append(prompt)
        
        return torch.stack(pseudo_prompts, dim=0)  # [M, prompt_dim]


class SAM2Base(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,  # default 1 input frame + 6 previous frames
        image_size=512,
        backbone_stride=16,  # stride of the image backbone output
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed=False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder=16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs=True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs=False,
        # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
        # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        use_signed_tpos_enc_to_obj_ptrs=False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        # Soft no object, i.e. mix in no_obj_ptr softly,
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        # Whether to use BNDL for pixels
        use_bndl_for_pixels: bool = False,
        bndl_fuse_type: str = 'sum',
        bndl_replace_global_with_hyper: bool = False,
        bndl_hyper_in_sparse: bool = False,
        # Whether to use UR-ERN for pixels (mutually exclusive with BNDL)
        use_ur_ern_for_pixels: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
        # AUE options (optional auxiliary contrastive loss)
        use_aue: bool = False,
        aue_proj_dim: int = 256,
        aue_queue_size: int = 65536,
        aue_temperature: float = 0.2,
        aue_loss_weight: float = 0.1,
        # AUE adversarial image bank resolution (for memory safety)
        aue_adversarial_image_size: int = 128,
        # Whether AUE uses uncertainty for ROI weighting (can be disabled)
        aue_use_uncertainty: bool = True,
        # Uncertainty-aware controls
        aue_uncertainty_mask_threshold: float | None = None,
        # Diversity regularization for adversarial samples
        aue_diversity_loss_weight: float = 0.0,  # Weight for diversity regularization (0 = disabled)
        # MMD-based UA Distribution Invariance parameters
        ua_loss_alpha_corr: float = 1.0,      # Spearman correlation weight
        ua_loss_alpha_mmd: float = 0.5,       # MMD loss weight
        alpha_diversity: float = 0.01,        # Feature bank diversity weight
        ua_loss_use_spearman: bool = True,    # Use Spearman vs Pearson
        ua_loss_mmd_gamma: float = 1.0,       # RBF kernel bandwidth
    ):
        super().__init__()

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.use_bndl_for_pixels = use_bndl_for_pixels
        self.bndl_fuse_type = bndl_fuse_type
        self.bndl_replace_global_with_hyper = bndl_replace_global_with_hyper
        self.bndl_hyper_in_sparse = bndl_hyper_in_sparse
        self.use_ur_ern_for_pixels = use_ur_ern_for_pixels
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # AUE components
        self.use_aue = bool(use_aue)
        self.aue_proj_dim = int(aue_proj_dim)
        self.aue_queue_size = int(aue_queue_size)
        self.aue_temperature = float(aue_temperature)
        self.aue_loss_weight = float(aue_loss_weight)
        self.aue_adversarial_image_size = int(aue_adversarial_image_size)
        self.aue_use_uncertainty = bool(aue_use_uncertainty)
        self.aue_uncertainty_mask_threshold = aue_uncertainty_mask_threshold
        self.aue_diversity_loss_weight = float(aue_diversity_loss_weight)
        # MMD-based UA Distribution Invariance parameters
        self.ua_loss_alpha_corr = float(ua_loss_alpha_corr)
        self.ua_loss_alpha_mmd = float(ua_loss_alpha_mmd)
        self.alpha_diversity = float(alpha_diversity)
        self.ua_loss_use_spearman = bool(ua_loss_use_spearman)
        self.ua_loss_mmd_gamma = float(ua_loss_mmd_gamma)
        
        if self.use_aue:
            self._build_aue_components()

        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print(
                "Image encoder compilation is enabled. First forward pass will be slow."
            )
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    def _build_aue_components(self) -> None:
        """Build AUE projection head and feature-space adversarial bank."""
        # AUE operates on global vectors from pixel features (C' = hidden_dim // 8)
        aue_in_dim = max(8, self.hidden_dim // 8)
        self.aue_proj = torch.nn.Sequential(
            torch.nn.Linear(aue_in_dim, self.aue_proj_dim, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.aue_proj_dim, self.aue_proj_dim, bias=False),
        )
        
        # Feature-Space Adversarial Bank (替代Image Bank)
        # 理论优势:
        # 1. 短优化路径: Features → BNDL → UA → MMD (不经过encoder)
        # 2. 参数高效: K × C vs K × 3 × H × W (减少200倍)
        # 3. 理论直接: Feature perturbation直接影响UA distribution
        # 4. 训练稳定: 减少gradient path的non-linearity
        
        K = 16  # Number of perturbation modes
        C_feat = max(8, self.hidden_dim // 8)  # Feature dimension after upscaling
        
        # Learnable feature perturbations
        # Prior: Small Gaussian initialization (假设meaningful shifts是small的)
        self.feature_perturbations = torch.nn.Parameter(
            torch.randn(K, C_feat) * 0.01
        )
        
        # Gradient Reversal for adversarial learning
        # Bank objective: max_φ MMD(clean_UA, adv_UA)
        # 通过GRL实现: φ收到 -∇_φ MMD
        self.feature_perturbations.register_hook(
            lambda g: (-1.0 * g) if g is not None else g
        )
        
        # UA-Shift-Guided Pseudo Prompt Generator
        # 受ICML'25 Few-Shot论文启发
        # 在UA distribution shift最大的regions生成prompts
        self.ua_prompt_generator = UAShiftGuidedPseudoPromptGenerator(
            feat_dim=C_feat,
            prompt_dim=C_feat
        )

    @torch.no_grad()
    def _aue_roi_view(
        self,
        feat: torch.Tensor,                # [B, H, W, C]
        min_scale: float = 0.6,
        boundary_ignore: int = 2,
        uncert: torch.Tensor | None = None  # [B, H, W] or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked ROI pooling to form a global vector per sample without slicing.

        Returns (z, roi_weight): z is [B, C], roi_weight is [B] to reflect
        the effective proportion of confident pixels in the ROI when uncertainty is provided.
        """
        B, H, W, C = feat.shape
        device = feat.device

        tops, lefts, hs, ws = [], [], [], []
        for _ in range(B):
            ch = max(2, int(H * (min_scale + (1 - min_scale) * random.random())))
            cw = max(2, int(W * (min_scale + (1 - min_scale) * random.random())))
            t = 0 if ch == H else random.randint(0, H - ch)
            left_idx = 0 if cw == W else random.randint(0, W - cw)
            tops.append(t)
            lefts.append(left_idx)
            hs.append(ch)
            ws.append(cw)

        mask = torch.zeros(B, H, W, 1, device=device, dtype=feat.dtype)
        for i, (t, left, ch, cw) in enumerate(zip(tops, lefts, hs, ws, strict=True)):
            t0 = t + boundary_ignore
            l0 = left + boundary_ignore
            t1 = t + ch - boundary_ignore
            l1 = left + cw - boundary_ignore
            if t0 < t1 and l0 < l1:
                mask[i, t0:t1, l0:l1, 0] = 1.0
            else:
                mask[i, t:t + ch, left:left + cw, 0] = 1.0

        if uncert is not None:
            u = uncert.unsqueeze(-1).to(feat.dtype)  # [B, H, W, 1]
            # Base weight is (1 - uncertainty)
            w = torch.clamp(1.0 - u, 0.0, 1.0) * mask
            # Optional hard masking: keep only confident pixels (u <= t)
            if self.aue_uncertainty_mask_threshold is not None:
                thr = float(self.aue_uncertainty_mask_threshold)
                keep = (u <= thr).to(feat.dtype)
                w = w * keep
            num = (feat * w).sum(dim=(1, 2))          # [B, C]
            den = (w.sum(dim=(1, 2)) + 1e-6)          # [B, 1]
            z = num / den
            roi_weight = (w.sum(dim=(1, 2)).squeeze(-1) / (mask.sum(dim=(1, 2)).squeeze(-1) + 1e-6))
        else:
            num = (feat * mask).sum(dim=(1, 2))
            den = (mask.sum(dim=(1, 2)) + 1e-6)
            z = num / den
            roi_weight = torch.ones(B, device=device, dtype=feat.dtype)

        return z, roi_weight

    def compute_aue_loss(
        self,
        pixel_feat: torch.Tensor,                                # [B, H, W, C']
        pixel_uncertainty: torch.Tensor | None = None,
        pixel_gt: torch.Tensor | None = None,
        pixel_logits: torch.Tensor | None = None,
        adversarial_sample_M: int | None = 4,
        roi_z1: torch.Tensor | None = None,
        roi_z2: torch.Tensor | None = None,
        roi_w1: torch.Tensor | None = None,
        roi_w2: torch.Tensor | None = None,
        pixel_bndl_model=None,
        uq_sample_num: int = 20,
    ) -> torch.Tensor:
        """
        MMD-based UA Distribution Invariance Loss
        
        核心理论:
        1. 优化UA joint distribution的domain-invariance (不只是correlation)
        2. Feature-space adversarial bank (稳定+高效)
        3. UA-shift-guided pseudo prompts (理论一致)
        4. Min-Max game: Model minimizes MMD, Bank maximizes MMD (via GRL)
        
        Pipeline:
        1. Compute clean UA distribution
        2. Generate adversarial features from bank
        3. Compute initial adversarial UA
        4. Compute local UA shift map
        5. Generate prompts from top-shift regions
        6. Refine adversarial features with prompts
        7. Compute final MMD loss + Spearman correlation
        """
        assert pixel_feat is not None and pixel_feat.ndim == 4
        B, H, W, C = pixel_feat.shape
        device = pixel_feat.device
        dtype = pixel_feat.dtype
        
        # Default weights (可通过config override)
        alpha_corr = getattr(self, 'ua_loss_alpha_corr', 1.0)
        alpha_mmd = getattr(self, 'ua_loss_alpha_mmd', 0.5)
        alpha_diversity = getattr(self, 'alpha_diversity', 0.01)
        use_spearman = getattr(self, 'ua_loss_use_spearman', True)
        mmd_gamma = getattr(self, 'ua_loss_mmd_gamma', 1.0)
        
        # ===== Step 1: Clean UA Distribution =====
        if pixel_logits is None or pixel_uncertainty is None or pixel_gt is None:
            # Fallback: no loss if missing inputs
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        
        clean_error = torch.abs(torch.sigmoid(pixel_logits) - pixel_gt.float())
        
        # ===== Step 2: Correlation Loss (基础) =====
        if use_spearman:
            loss_corr = -self.spearman_correlation_differentiable(
                pixel_uncertainty.flatten(),
                clean_error.flatten(),
                temperature=0.1
            )
        else:
            loss_corr = -self.pearson_correlation(
                pixel_uncertainty.flatten(),
                clean_error.flatten()
            )
        
        # Regularization: 防止uncertainty collapse
        loss_reg = -torch.log(pixel_uncertainty.std() + 1e-6)
        loss_corr_total = loss_corr + 0.1 * loss_reg
        
        # ===== Step 3: Generate Adversarial Features (Feature Bank) =====
        if pixel_bndl_model is None or adversarial_sample_M is None or adversarial_sample_M <= 0:
            # No adversarial training
            return loss_corr_total
        
        adv_feat_init = self.sample_adversarial_features(
            pixel_feat,
            num_samples=adversarial_sample_M
        )  # [M, H, W, C]
        
        # ===== Step 4: Forward Initial Adversarial through BNDL =====
        M = adv_feat_init.shape[0]
        
        # Configure external_pre_out_w
        if pixel_bndl_model.enable_global_sparse:
            adv_external_w = None
        else:
            linear_weight = pixel_bndl_model.linear.weight
            adv_external_w = linear_weight.unsqueeze(0).expand(M, -1, -1).detach()
        
        adv_uq_pval_init, adv_logits_init = pixel_uncertain_sampling(
            pixel_bndl_model,
            adv_feat_init,
            external_pre_out_w=adv_external_w,
            sample_num=uq_sample_num
        )
        adv_uq_init = 1.0 - adv_uq_pval_init  # Convert p-value to uncertainty
        
        # Error proxy for adversarial (无GT)
        # 使用BNDL predictive variance
        adv_error_init = self._compute_bndl_predictive_variance(
            pixel_bndl_model,
            adv_feat_init,
            num_samples=20
        )
        
        # ===== Step 5: Compute Local UA Shift Map (理论关键!) =====
        ua_shift_map = self.compute_local_ua_shift_map(
            clean_uncertainty=pixel_uncertainty,
            adv_uncertainty=adv_uq_init,
            clean_error=clean_error,
            adv_error=adv_error_init,
            window_size=7
        )  # [B, H, W]
        
        # ===== Step 6: Generate Pseudo Prompts from UA Shift =====
        # 理论: 在UA shift最大的regions采样，不是uncertainty最大
        if hasattr(self, 'ua_prompt_generator'):
            pseudo_prompts = self.ua_prompt_generator(
                adv_feat_init,
                ua_shift_map,
                top_k=10
            )  # [M, C]
            
            # ===== Step 7: Prompt-modulated Adversarial Features =====
            # Prompt作为feature modulation
            prompt_modulation = pseudo_prompts.view(M, 1, 1, -1)
            adv_feat_final = adv_feat_init + 0.1 * prompt_modulation
        else:
            # No prompt generator (e.g., during testing or if disabled)
            adv_feat_final = adv_feat_init
        
        # ===== Step 8: Re-forward Final Adversarial Features =====
        adv_uq_pval_final, adv_logits_final = pixel_uncertain_sampling(
            pixel_bndl_model,
            adv_feat_final,
            external_pre_out_w=adv_external_w,
            sample_num=uq_sample_num
        )
        adv_uq_final = 1.0 - adv_uq_pval_final
        
        adv_error_final = self._compute_bndl_predictive_variance(
            pixel_bndl_model,
            adv_feat_final,
            num_samples=20
        )
        
        # ===== Step 9: Construct UA Joint Distributions =====
        clean_ua_joint = torch.stack([
            pixel_uncertainty.flatten(),
            clean_error.flatten()
        ], dim=1)  # [N_clean, 2]
        
        adv_ua_joint = torch.stack([
            adv_uq_final.flatten(),
            adv_error_final.flatten()
        ], dim=1)  # [N_adv, 2]
        
        # ===== Step 10: MMD Loss (核心) =====
        # Model: minimize MMD → 学习UA distribution invariance
        # Bank: maximize MMD (via GRL) → 找worst-case perturbations
        loss_mmd = self.compute_mmd(clean_ua_joint, adv_ua_joint, gamma=mmd_gamma)
        
        # ===== Step 11: Diversity Regularization =====
        loss_diversity = self._compute_feature_diversity_loss()
        
        # ===== Total Loss =====
        loss = (alpha_corr * loss_corr_total +      # 基础: UA correlation + regularization
                alpha_mmd * loss_mmd +              # 核心: distribution invariance
                alpha_diversity * loss_diversity)   # 正则: 防止mode collapse
        
        return loss

    @property
    def device(self):
        return next(self.parameters()).device

    # --------------------------- AUE (new) helpers ---------------------------
    def _compute_pos_ratios(
        self,
        pixel_logits: torch.Tensor | None,
        pixel_uncertainty: torch.Tensor | None,
        pixel_gt: torch.Tensor | None,
        spatial_hw: tuple[int, int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute global uncertainty-confidence ratio.
        
        With GT-aligned confidence, we no longer need separate TP/FP masks:
        - TP regions: confidence naturally high → ratio low
        - FP regions: confidence naturally low → ratio high
        
        Simply minimize global uncertainty/confidence ratio = maximize confidence globally.
        
        Returns:
            global_ratio: mean(uncertainty/confidence) over all pixels
        """
        if pixel_logits is None:
            return torch.tensor(0.0, device=device, dtype=dtype)
        
        # Compute GT-aligned confidence (automatically handles TP/FP/TN/FN)
        confidence = self._aue_compute_confidence(
            pixel_logits=pixel_logits,
            pixel_gt=pixel_gt,
        )
        
        uncertainty_p = pixel_uncertainty if pixel_uncertainty is not None else (1.0 - confidence)
        uncertainty_p = uncertainty_p.clamp(0.0, 1.0)
        
        # Global optimization: minimize uncertainty/confidence
        # - TP regions (high conf) → small ratio → small loss ✓
        # - FP regions (low conf) → large ratio → large loss (automatic penalty) ✓
        # - No need for separate masks!
        eps = 1e-6
        ratio = uncertainty_p / (confidence + eps)  # [B, H, W]
        return ratio.mean()  # Simple global mean
    
    def _extract_mask_from_gt(
        self,
        pixel_gt: torch.Tensor | None,
        spatial_hw: tuple[int, int],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Extract positive mask from GT tensor [B, H, W]."""
        H, W = spatial_hw
        
        if pixel_gt is None:
            return torch.ones((batch_size, H, W), device=device, dtype=torch.bool)
        
        gt = pixel_gt
        if gt.ndim == 4 and gt.shape[-1] > 1:
            pos = (gt > 0).any(dim=-1)
        elif gt.ndim == 4 and gt.shape[1] == 1 and gt.shape[2] == H and gt.shape[3] == W:
            pos = (gt[:, 0] > 0)
        elif gt.ndim == 3 and gt.shape[1] == H and gt.shape[2] == W:
            pos = (gt > 0)
        else:
            try:
                B = gt.shape[0]
                pos = (gt.view(B, H, W, -1) > 0).any(dim=-1)
            except Exception:
                pos = torch.ones((gt.shape[0], H, W), device=gt.device, dtype=torch.bool)
        
        return pos.to(torch.bool)

    def _aue_compute_confidence(
        self,
        pixel_logits: torch.Tensor,
        pixel_gt: torch.Tensor,
        tau_conf: float = 2.0,
    ) -> torch.Tensor:
        """Compute GT-aligned per-pixel confidence in [0,1], [B, H, W].

        Confidence reflects prediction correctness:
        - GT=1, logits=+5 → high confidence (correct foreground)
        - GT=0, logits=-5 → high confidence (correct background)
        - GT=1, logits=-5 → low confidence (wrong prediction)
        
        Formula: c = sigmoid((logits * (2*gt - 1)) / tau_conf)
        """
        # Extract logits value based on shape
        if pixel_logits.ndim == 4 and pixel_logits.shape[-1] >= 1:
            logits_val = pixel_logits.max(dim=-1).values  # [B, H, W, C] -> [B, H, W]
        elif pixel_logits.ndim == 3:
            logits_val = pixel_logits  # [B, H, W]
        elif pixel_logits.ndim == 4 and pixel_logits.shape[1] == 1:
            logits_val = pixel_logits[:, 0]  # [B, 1, H, W] -> [B, H, W]
        else:
            B, H, W = pixel_logits.shape[0], pixel_logits.shape[1], pixel_logits.shape[2]
            logits_val = pixel_logits.view(B, H, W, -1).mean(dim=-1).values
        
        # Extract GT mask [B, H, W] as boolean
        H, W = logits_val.shape[1], logits_val.shape[2]
        B = logits_val.shape[0]
        gt_mask = self._extract_mask_from_gt(
            pixel_gt=pixel_gt,
            spatial_hw=(H, W),
            batch_size=B,
            device=logits_val.device,
        )
        
        # Convert GT boolean mask to sign: [0,1] -> [-1,+1]
        # GT=1 (foreground) → +1, GT=0 (background) → -1
        gt_sign = 2.0 * gt_mask.float() - 1.0  # [B, H, W]
        
        # Align logits with GT direction
        aligned_logits = logits_val * gt_sign
        
        # Compute confidence from aligned logits
        return torch.sigmoid(aligned_logits / float(tau_conf))

    def sample_adversarial_features(
        self, 
        clean_feat: torch.Tensor,  # [B, H, W, C]
        num_samples: int = 4,
    ) -> torch.Tensor:
        """
        从Feature Bank采样adversarial features
        
        理论:
        - Bank参数化adversarial perturbation distribution
        - Random sampling探索不同的perturbation modes
        - 通过GRL，bank自动学习worst-case perturbations
        
        Args:
            clean_feat: clean样本的features
            num_samples: 采样数量
        
        Returns:
            adv_feat: [M, H, W, C] adversarial features
        """
        if not hasattr(self, 'feature_perturbations'):
            # Fallback: 如果没有feature bank，返回clean features
            return clean_feat[:num_samples]
        
        B, H, W, C = clean_feat.shape
        device = clean_feat.device
        K = self.feature_perturbations.shape[0]
        
        if num_samples <= 0:
            return clean_feat[:0]  # Empty tensor
        
        # Sample M perturbations from bank
        num_samples = min(num_samples, B)  # 不超过batch size
        indices = torch.randint(0, K, (num_samples,), device=device)
        deltas = self.feature_perturbations[indices]  # [M, C]
        
        # Broadcast to spatial dimensions
        deltas_spatial = deltas.view(num_samples, 1, 1, C)  # [M, 1, 1, C]
        
        # Apply to batch samples
        clean_sampled = clean_feat[:num_samples]  # [M, H, W, C]
        
        # Generate adversarial features
        adv_feat = clean_sampled + deltas_spatial
        
        return adv_feat
    
    def _aue_compute_conf_from_logits_tensor(self, logits: torch.Tensor, tau_conf: float = 2.0) -> torch.Tensor:
        """Compute confidence from logits tensor [*, H, W, K] -> [*, H, W] via sigmoid(max(|logit|)/tau)."""
        if logits.ndim < 3:
            raise ValueError("logits tensor rank too low")
        mag = logits.abs().max(dim=-1).values  # [..., H, W]
        return torch.sigmoid(mag / float(tau_conf)).to(mag.dtype)
    
    def _compute_feature_diversity_loss(self) -> torch.Tensor:
        """
        Diversity regularization for feature-space adversarial bank
        
        理论: 防止K个perturbations收敛到相同方向 (mode collapse)
        鼓励bank探索不同的perturbation directions
        
        Returns:
            Diversity loss (minimize to maximize diversity)
        """
        if not hasattr(self, 'feature_perturbations'):
            return torch.tensor(0.0, device=self.device)
        
        deltas = self.feature_perturbations  # [K, C]
        K = deltas.shape[0]
        
        if K < 2:
            return torch.tensor(0.0, device=deltas.device)
        
        # Normalize for cosine similarity
        deltas_norm = F.normalize(deltas, dim=1, p=2)
        
        # Pairwise similarity matrix
        sim_matrix = torch.mm(deltas_norm, deltas_norm.t())  # [K, K]
        
        # Mask diagonal
        mask = torch.eye(K, device=deltas.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, 0.0)
        
        # Average similarity (excluding diagonal)
        # Minimize this to maximize diversity
        num_pairs = K * (K - 1)
        avg_similarity = sim_matrix.sum() / num_pairs
        
        return avg_similarity
    
    # ==================== MMD and Correlation Methods ====================
    
    def compute_mmd(
        self, 
        X: torch.Tensor,  # [N_X, d]
        Y: torch.Tensor,  # [N_Y, d]
        gamma: float = 1.0
    ) -> torch.Tensor:
        """
        Maximum Mean Discrepancy with RBF kernel
        
        理论: MMD^2(P,Q) = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        当MMD=0时，P=Q (分布相同)
        
        Args:
            X: samples from distribution P
            Y: samples from distribution Q
            gamma: RBF kernel bandwidth
        
        Returns:
            mmd: scalar MMD distance
        """
        XX = self.rbf_kernel(X, X, gamma)
        YY = self.rbf_kernel(Y, Y, gamma)
        XY = self.rbf_kernel(X, Y, gamma)
        
        mmd_squared = XX.mean() + YY.mean() - 2 * XY.mean()
        
        # Ensure non-negative (numerical stability)
        return torch.sqrt(torch.clamp(mmd_squared, min=0.0) + 1e-6)
    
    def rbf_kernel(
        self, 
        X: torch.Tensor,  # [n, d]
        Y: torch.Tensor,  # [m, d]
        gamma: float
    ) -> torch.Tensor:
        """
        RBF (Gaussian) kernel: k(x,y) = exp(-gamma * ||x-y||^2)
        
        Args:
            X, Y: input tensors
            gamma: kernel bandwidth
        
        Returns:
            kernel_matrix: [n, m]
        """
        # Compute squared Euclidean distances efficiently
        XX = (X ** 2).sum(dim=1, keepdim=True)  # [n, 1]
        YY = (Y ** 2).sum(dim=1, keepdim=True).T  # [1, m]
        XY = X @ Y.T  # [n, m]
        
        sq_distances = XX + YY - 2 * XY  # [n, m]
        
        return torch.exp(-gamma * sq_distances)
    
    def pearson_correlation(
        self, 
        x: torch.Tensor,  # [n]
        y: torch.Tensor   # [n]
    ) -> torch.Tensor:
        """
        Pearson correlation coefficient
        
        PCC = Cov(X,Y) / (σ_X * σ_Y)
        
        Returns:
            correlation: scalar in [-1, 1]
        """
        mean_x = x.mean()
        mean_y = y.mean()
        
        xm = x - mean_x
        ym = y - mean_y
        
        r_num = (xm * ym).sum()
        r_den = torch.sqrt((xm ** 2).sum() * (ym ** 2).sum() + 1e-6)
        
        return r_num / r_den
    
    def spearman_correlation_differentiable(
        self,
        x: torch.Tensor,      # [n]
        y: torch.Tensor,      # [n]
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Differentiable Spearman correlation via soft-ranking
        
        理论:
        - Spearman比Pearson更鲁棒 (rank-based)
        - 只要求单调关系，不要求线性
        - 对outliers不敏感
        - 更适合domain shift场景
        
        Args:
            x, y: input tensors
            temperature: soft-rank temperature (smaller = closer to hard rank)
        
        Returns:
            spearman_rho: scalar correlation
        """
        rank_x = self.soft_rank(x, temperature)
        rank_y = self.soft_rank(y, temperature)
        
        return self.pearson_correlation(rank_x, rank_y)
    
    def soft_rank(
        self,
        x: torch.Tensor,      # [n]
        temperature: float
    ) -> torch.Tensor:
        """
        Differentiable soft ranking
        
        理论: rank(x_i) = Σ_j I(x_i > x_j)
        Soft version: rank(x_i) ≈ Σ_j sigmoid((x_i - x_j) / temp)
        
        Args:
            x: input tensor [n]
            temperature: controls softness
        
        Returns:
            soft_ranks: [n]
        """
        # Pairwise differences [n, n]
        diff = x.unsqueeze(0) - x.unsqueeze(1)  # x_i - x_j
        
        # Soft indicator via sigmoid
        soft_indicator = torch.sigmoid(diff / temperature)
        
        # Sum over j to get soft rank
        soft_ranks = soft_indicator.sum(dim=1)
        
        return soft_ranks
    
    def compute_local_ua_shift_map(
        self,
        clean_uncertainty: torch.Tensor,  # [B, H, W]
        adv_uncertainty: torch.Tensor,    # [M, H, W]
        clean_error: torch.Tensor,        # [B, H, W]
        adv_error: torch.Tensor,          # [M, H, W]
        window_size: int = 7,
    ) -> torch.Tensor:
        """
        计算spatial map of local UA distribution shift
        
        理论关键:
        - 不是看uncertainty的absolute value
        - 而是看UA joint distribution的local change
        - 这与MMD loss的目标完全对齐
        
        方法:
        - 对每个spatial location的local window
        - 计算clean vs adv的local UA distribution MMD
        - 返回spatial map of local MMD values
        
        Returns:
            shift_map: [B, H, W] 每个location的local UA shift强度
        """
        B, H, W = clean_uncertainty.shape
        M = adv_uncertainty.shape[0]
        device = clean_uncertainty.device
        
        shift_map = torch.zeros(B, H, W, device=device)
        pad = window_size // 2
        
        # 使用stride减少计算 (每隔stride计算一次)
        stride = max(1, window_size // 2)
        
        for b in range(B):
            for h in range(0, H, stride):
                for w in range(0, W, stride):
                    # Local window bounds
                    h_start = max(0, h - pad)
                    h_end = min(H, h + pad + 1)
                    w_start = max(0, w - pad)
                    w_end = min(W, w + pad + 1)
                    
                    # Clean local UA samples
                    clean_unc_local = clean_uncertainty[b, h_start:h_end, w_start:w_end].flatten()
                    clean_err_local = clean_error[b, h_start:h_end, w_start:w_end].flatten()
                    
                    if clean_unc_local.numel() == 0:
                        continue
                    
                    clean_ua_local = torch.stack([clean_unc_local, clean_err_local], dim=1)
                    
                    # Adversarial local UA samples (对应的adv sample)
                    adv_idx = b % M
                    adv_unc_local = adv_uncertainty[adv_idx, h_start:h_end, w_start:w_end].flatten()
                    adv_err_local = adv_error[adv_idx, h_start:h_end, w_start:w_end].flatten()
                    
                    if adv_unc_local.numel() == 0:
                        continue
                    
                    adv_ua_local = torch.stack([adv_unc_local, adv_err_local], dim=1)
                    
                    # Compute local MMD (小心: 小样本时MMD可能不稳定)
                    if clean_ua_local.shape[0] > 5 and adv_ua_local.shape[0] > 5:
                        local_mmd = self.compute_mmd(
                            clean_ua_local,
                            adv_ua_local,
                            gamma=1.0
                        )
                        
                        # 填充到shift map (填充整个window)
                        shift_map[b, h_start:h_end, w_start:w_end] = local_mmd.item()
        
        return shift_map
    
    def _compute_bndl_predictive_variance(
        self,
        bndl_model,
        feat: torch.Tensor,  # [M, H, W, C]
        num_samples: int = 20
    ) -> torch.Tensor:
        """
        通过BNDL多次sampling估计predictive variance
        
        理论: 作为adversarial samples的error proxy (无GT时使用)
        Var_p(w|D)[f_w(x)] ≈ E[(y - ŷ)²] (Bayesian expected squared error)
        
        Args:
            bndl_model: BNDL model
            feat: features [M, H, W, C]
            num_samples: number of BNDL samples
        
        Returns:
            variance: [M, H, W] predictive variance
        """
        predictions = []
        
        # Multiple forward passes through BNDL
        for _ in range(num_samples):
            _, logits = pixel_uncertain_sampling(
                bndl_model,
                feat,
                external_pre_out_w=None,
                sample_num=1
            )
            # Convert to probabilities
            predictions.append(torch.sigmoid(logits))
        
        # Stack and compute variance
        pred_stacked = torch.stack(predictions, dim=0)  # [num_samples, M, H, W, K]
        
        # Variance across samples, then average across output channels
        variance = pred_stacked.var(dim=0)  # [M, H, W, K]
        
        # Average over K (多个mask outputs)
        if variance.ndim == 4:
            variance = variance.mean(dim=-1)  # [M, H, W]
        
        return variance

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_bndl_for_pixels=self.use_bndl_for_pixels,
            bndl_fuse_type=self.bndl_fuse_type,
            bndl_replace_global_with_hyper=self.bndl_replace_global_with_hyper,
            bndl_hyper_in_sparse=self.bndl_hyper_in_sparse,
            use_ur_ern_for_pixels=self.use_ur_ern_for_pixels,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
        pixel_gt_for_aue: torch.Tensor | None = None,
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        mask_decoder_outputs = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        
        if len(mask_decoder_outputs) == 5:  # 包含辅助输出（BNDL/UR-ERN 等）
            (
                low_res_multimasks,
                ious,
                sam_output_tokens,
                object_score_logits,
                aux_outputs,
            ) = mask_decoder_outputs
        else:  # 标准输出
            (
                low_res_multimasks,
                ious,
                sam_output_tokens,
                object_score_logits,
            ) = mask_decoder_outputs

        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        if self.use_bndl_for_pixels:
            # Read and update BNDL namespace inside aux_outputs, then pass aux_outputs through unchanged
            aux_outputs = aux_outputs or {}
            bndl_outputs = aux_outputs.get("bndl", {})
            pixel_feat = bndl_outputs.get("pixel_feat_grad", bndl_outputs.get("pixel_feat", None))
            shared_z1 = None
            shared_z2 = None
            shared_w1 = None
            shared_w2 = None
            # Build one pair of ROI views once for AUE optimization; AUE will compute its own if None.
            if self.training and self.use_aue and (pixel_feat is not None):
                uncert = bndl_outputs.get("pixel_uncertainty", None) if self.aue_use_uncertainty else None
                shared_z1, shared_w1 = self._aue_roi_view(pixel_feat, min_scale=0.6, boundary_ignore=2, uncert=uncert)
                shared_z2, shared_w2 = self._aue_roi_view(pixel_feat, min_scale=0.6, boundary_ignore=2, uncert=uncert)
            # Optional per-frame uncertainty for AUE
            pixel_uncertainty = bndl_outputs.get("pixel_uncertainty", None) if self.aue_use_uncertainty else None
            if self.use_aue and self.training and (pixel_feat is not None):
                # Prefer gradient-carrying logits key when training
                pixel_logits_for_aue = bndl_outputs.get(
                    "pixel_logits",
                    bndl_outputs.get("masks_bndl_raw", bndl_outputs.get("mean_pixel_logits", None)),
                )
                aue_loss = self.compute_aue_loss(
                    pixel_feat=pixel_feat,
                    pixel_uncertainty=pixel_uncertainty,
                    pixel_gt=pixel_gt_for_aue,
                    pixel_logits=pixel_logits_for_aue,
                    roi_z1=shared_z1,
                    roi_z2=shared_z2,
                    roi_w1=shared_w1,
                    roi_w2=shared_w2,
                    pixel_bndl_model=self.sam_mask_decoder.pixel_bndl if getattr(self.sam_mask_decoder, "pixel_bndl", None) is not None else None,
                )
                bndl_outputs["aue_aux_loss"] = aue_loss
            # Write back updated BNDL namespace
            aux_outputs["bndl"] = bndl_outputs
            return (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
                aux_outputs,
            )
        else:
            return (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
                aux_outputs,
            )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _, *_ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs, strict=True)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list).to(
                            device=device, non_blocking=True
                        )
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
        pixel_gt_for_aue: torch.Tensor | None = None,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1], strict=True)
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
                pixel_gt_for_aue=pixel_gt_for_aue,
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
        pixel_gt_for_aue: torch.Tensor | None = None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
            pixel_gt_for_aue,
        )

        # If SAM head returned auxiliary outputs (BNDL/UR-ERN), sam_outputs will have 8 elements
        if isinstance(sam_outputs, tuple) and len(sam_outputs) == 8:
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
                aux_outputs,
            ) = sam_outputs
            # Optionally expose aux outputs for downstream consumers
            current_out["aux_outputs"] = aux_outputs
        else:
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks
