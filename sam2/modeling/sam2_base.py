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
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # DSU options
        use_dsu: bool = False,
        dsu_strength_mu: float = 0.1,
        dsu_strength_sigma: float = 0.1,
        dsu_prob: float = 1.0,
        dsu_apply_high_res: bool = False,
        dsu_stats_eps: float = 1e-6,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
        # AdCo options (optional auxiliary contrastive loss)
        use_adco: bool = False,
        adco_proj_dim: int = 256,
        adco_queue_size: int = 65536,
        adco_temperature: float = 0.2,
        adco_loss_weight: float = 0.1,
        # AdCo negative image bank resolution (for memory safety)
        adco_neg_image_size: int = 128,
        # Whether AdCo uses uncertainty for ROI weighting (can be disabled)
        adco_use_uncertainty: bool = True,
        # Uncertainty-aware controls
        adco_gate_by_uncertainty: bool = True,
        adco_gate_floor: float = 0.2,
        adco_tau_beta: float = 0.5,
        adco_uncertainty_mask_threshold: float | None = None,
        adco_scale_neg_by_uq: bool = True,
        adco_curriculum_warmup_steps: int = 0,
        # ProCo options (prototype contrastive, optional)
        use_proco: bool = False,
        proco_proj_dim: int = 256,
        proco_num_obj_prototypes: int = 128,
        proco_num_bg_prototypes: int = 32,
        proco_temperature: float = 0.1,
        proco_loss_weight: float = 0.1,
        # MoCo options (momentum contrast, optional)
        use_moco: bool = False,
        moco_proj_dim: int = 256,
        moco_queue_size: int = 65536,
        moco_momentum: float = 0.996,
        moco_temperature: float = 0.2,
        moco_loss_weight: float = 0.1,
        # ROI view sharing option
        share_roi_views: bool = True,
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

        # DSU settings
        self.use_dsu = use_dsu
        self.dsu_strength_mu = float(dsu_strength_mu)
        self.dsu_strength_sigma = float(dsu_strength_sigma)
        self.dsu_prob = float(dsu_prob)
        self.dsu_apply_high_res = bool(dsu_apply_high_res)
        self.dsu_stats_eps = float(dsu_stats_eps)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # AdCo components
        self.use_adco = bool(use_adco)
        self.adco_proj_dim = int(adco_proj_dim)
        self.adco_queue_size = int(adco_queue_size)
        self.adco_temperature = float(adco_temperature)
        self.adco_loss_weight = float(adco_loss_weight)
        self.adco_neg_image_size = int(adco_neg_image_size)
        ############################################################
        self.adco_use_uncertainty = bool(adco_use_uncertainty)
        self.adco_gate_by_uncertainty = bool(adco_gate_by_uncertainty)
        self.adco_gate_floor = float(adco_gate_floor)
        self.adco_tau_beta = float(adco_tau_beta)
        self.adco_uncertainty_mask_threshold = adco_uncertainty_mask_threshold
        self.adco_scale_neg_by_uq = bool(adco_scale_neg_by_uq)
        self.adco_curriculum_warmup_steps = int(adco_curriculum_warmup_steps)
        self._adco_step_count = 0
        self.adco_grl_scale = 1.0
        if self.use_adco:
            self._build_adco_components()

        # ProCo components
        self.use_proco = bool(use_proco)
        self.proco_proj_dim = int(proco_proj_dim)
        self.proco_num_obj_prototypes = int(proco_num_obj_prototypes)
        self.proco_num_bg_prototypes = int(proco_num_bg_prototypes)
        self.proco_temperature = float(proco_temperature)
        self.proco_loss_weight = float(proco_loss_weight)
        if self.use_proco:
            self._build_proco_components()

        # MoCo components
        self.use_moco = bool(use_moco)
        self.moco_proj_dim = int(moco_proj_dim)
        self.moco_queue_size = int(moco_queue_size)
        self.moco_momentum = float(moco_momentum)
        self.moco_temperature = float(moco_temperature)
        self.moco_loss_weight = float(moco_loss_weight)
        if self.use_moco:
            self._build_moco_components()

        # ROI sharing control
        self.share_roi_views = bool(share_roi_views)

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

    def _dsu_perturb(self, x: torch.Tensor) -> torch.Tensor:
        """
        DSU-style feature statistic perturbation on BCHW features.
        Active only during training and when `use_dsu=True`.
        """
        if not self.training or not self.use_dsu:
            return x
        if self.dsu_strength_mu <= 0.0 and self.dsu_strength_sigma <= 0.0:
            return x

        orig_dtype = x.dtype
        x_f32 = x.float()
        # Per-sample, per-channel stats
        mu = x_f32.mean(dim=(2, 3), keepdim=True)
        var = x_f32.var(dim=(2, 3), keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.dsu_stats_eps)

        x_hat = (x_f32 - mu) / (std + 1e-12)

        # Per-sample gating
        if self.dsu_prob >= 1.0:
            gate = 1.0
        elif self.dsu_prob <= 0.0:
            gate = 0.0
        else:
            gate = torch.bernoulli(
                torch.full((x.size(0), 1, 1, 1), float(self.dsu_prob), device=x.device, dtype=x_f32.dtype)
            )

        # Sample noise
        if isinstance(gate, torch.Tensor):
            eps_mu = torch.randn_like(mu) * self.dsu_strength_mu * gate
            eps_sigma = torch.randn_like(std) * self.dsu_strength_sigma * gate
        else:
            eps_mu = torch.randn_like(mu) * self.dsu_strength_mu * gate
            eps_sigma = torch.randn_like(std) * self.dsu_strength_sigma * gate

        mu_tilde = mu + eps_mu * std
        sigma_tilde = torch.clamp(std * (1.0 + eps_sigma), min=self.dsu_stats_eps)

        x_out = x_hat * sigma_tilde + mu_tilde
        return x_out.to(orig_dtype)

    def _build_adco_components(self) -> None:
        """Build AdCo projection head and adversarial negatives."""
        # AdCo operates on global vectors from pixel features (C' = hidden_dim // 8)
        adco_in_dim = max(8, self.hidden_dim // 8)
        self.adco_proj = torch.nn.Sequential(
            torch.nn.Linear(adco_in_dim, self.adco_proj_dim, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.adco_proj_dim, self.adco_proj_dim, bias=False),
        )
        # Scheme A: Learnable negative image bank for AdCo (sample → encode → BNDL)
        # Store at a reduced resolution for memory safety; upsample on use.
        # Shape: [K_eff, 3, Hneg, Wneg]
        Hneg = int(self.adco_neg_image_size)
        Wneg = int(self.adco_neg_image_size)
        # Budget ~128M elements to avoid OOM (float32 ~ 512MB). Clamp K accordingly.
        max_elems = 128 * 1024 * 1024
        per_item = 3 * Hneg * Wneg
        K_eff = max(64, min(int(self.adco_queue_size), int(max_elems // max(per_item, 1))))
        neg_images = torch.randn(K_eff, 3, Hneg, Wneg) * 0.01
        self.adco_negatives = torch.nn.Parameter(neg_images)
        # Gradient reversal on negatives to approximate adversarial ascent
        # Use fixed gradient reversal scale (standard GRL implementation)
        self.adco_negatives.register_hook(lambda g: (-1.0 * g) if g is not None else g)

    def _build_proco_components(self) -> None:
        """Build ProCo projection head and prototype banks (object/background)."""
        # Use the same pixel feature input dim heuristic as AdCo
        proco_in_dim = max(8, self.hidden_dim // 8)
        self.proco_proj = torch.nn.Sequential(
            torch.nn.Linear(proco_in_dim, self.proco_proj_dim, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.proco_proj_dim, self.proco_proj_dim, bias=False),
        )
        # Learnable prototypes (Proxy-NCA style)
        obj_proto = torch.randn(self.proco_num_obj_prototypes, self.proco_proj_dim) * 0.01
        bg_proto = torch.randn(self.proco_num_bg_prototypes, self.proco_proj_dim) * 0.01
        self.proco_obj_prototypes = torch.nn.Parameter(F.normalize(obj_proto, dim=1))
        self.proco_bg_prototypes = torch.nn.Parameter(F.normalize(bg_proto, dim=1))

    def _build_moco_components(self) -> None:
        """Build MoCo projection heads (query and key) and the queue."""
        in_dim = max(8, self.hidden_dim // 8)
        # Query encoder head
        self.moco_proj_q = torch.nn.Sequential(
            torch.nn.Linear(in_dim, self.moco_proj_dim, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.moco_proj_dim, self.moco_proj_dim, bias=False),
        )
        # Key encoder head (same arch), initialized as copy, not directly optimized
        self.moco_proj_k = torch.nn.Sequential(
            torch.nn.Linear(in_dim, self.moco_proj_dim, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.moco_proj_dim, self.moco_proj_dim, bias=False),
        )
        # Initialize key with query params
        for p_k, p_q in zip(self.moco_proj_k.parameters(), self.moco_proj_q.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False
        # Create the queue as a buffer (D x K)
        self.register_buffer("moco_queue", F.normalize(torch.randn(self.moco_proj_dim, self.moco_queue_size), dim=0))
        self.register_buffer("moco_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _moco_momentum_update_key_encoder(self) -> None:
        """Momentum update of the key encoder."""
        m = self.moco_momentum
        for p_k, p_q in zip(self.moco_proj_k.parameters(), self.moco_proj_q.parameters()):
            p_k.data = p_k.data * m + p_q.data * (1.0 - m)

    @torch.no_grad()
    def _moco_dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Enqueue keys and dequeue the oldest ones."""
        keys = F.normalize(keys, dim=1)  # [N, D]
        batch_size = keys.shape[0]
        K = self.moco_queue_size
        ptr = int(self.moco_queue_ptr.item())
        # Transpose to (D x N) to fit queue layout (D x K)
        keys_T = keys.t()  # [D, N]
        end = ptr + batch_size
        if end <= K:
            self.moco_queue[:, ptr:end] = keys_T
        else:
            first = K - ptr
            if first > 0:
                self.moco_queue[:, ptr:] = keys_T[:, :first]
            remain = batch_size - first
            self.moco_queue[:, :remain] = keys_T[:, first:]
        ptr = (ptr + batch_size) % K
        self.moco_queue_ptr[0] = ptr

    @torch.no_grad()
    def _gather_batchwise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Safe all_gather for variable batch sizes: pad to max B, gather, then unpad.

        Input:  tensor [B, D]
        Output: concatenated across ranks [sum_B, D]
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return tensor
        device = tensor.device
        B_local = torch.tensor([tensor.shape[0]], device=device, dtype=torch.long)
        sizes = [torch.zeros_like(B_local) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(sizes, B_local)
        sizes_int = [int(s.item()) for s in sizes]
        max_B = max(sizes_int) if len(sizes_int) > 0 else 0
        if max_B == 0:
            return tensor.new_zeros((0, tensor.shape[1]))
        D = tensor.shape[1]
        if tensor.shape[0] < max_B:
            pad = torch.zeros((max_B - tensor.shape[0], D), device=device, dtype=tensor.dtype)
            tensor_pad = torch.cat([tensor, pad], dim=0)
        else:
            tensor_pad = tensor
        gather_list = [torch.zeros((max_B, D), device=device, dtype=tensor.dtype) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gather_list, tensor_pad)
        # Unpad and concat
        parts = []
        for t, sz in zip(gather_list, sizes_int):
            if sz > 0:
                parts.append(t[:sz, :])
        if len(parts) == 0:
            return tensor.new_zeros((0, D))
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def _adco_random_crop(self, feat: torch.Tensor, min_scale: float = 0.6) -> torch.Tensor:
        """Random spatial crop on [B, H, W, C] features; fallback to center if tiny."""
        B, H, W, C = feat.shape
        if H < 4 or W < 4:
            return feat
        crop_h = max(2, int(H * (min_scale + (1 - min_scale) * random.random())))
        crop_w = max(2, int(W * (min_scale + (1 - min_scale) * random.random())))
        if crop_h >= H and crop_w >= W:
            return feat
        top = 0 if H == crop_h else random.randint(0, H - crop_h)
        left = 0 if W == crop_w else random.randint(0, W - crop_w)
        return feat[:, top:top + crop_h, left:left + crop_w, :]

    @torch.no_grad()
    def _adco_roi_view(
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
            if self.adco_uncertainty_mask_threshold is not None:
                thr = float(self.adco_uncertainty_mask_threshold)
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

    def compute_adco_loss(
        self,
        pixel_feat: torch.Tensor,                                # [B, H, W, C']
        pixel_uncertainty: torch.Tensor | None = None,
        pixel_gt: torch.Tensor | None = None,
        pixel_logits: torch.Tensor | None = None,
        neg_sample_M: int | None = 1024,
        roi_z1: torch.Tensor | None = None,
        roi_z2: torch.Tensor | None = None,
        roi_w1: torch.Tensor | None = None,
        roi_w2: torch.Tensor | None = None,
        pixel_bndl_model=None,
        external_pre_out_w=None,
        uq_sample_num: int = 50,
    ) -> torch.Tensor:
        
        # TODO
        # 计算logits与uncertainty的相关性
        
        # 1. 损失中要最大化相关性
        # 2. 采样neg样本
        # 3. 从neg样本中得到不确定性以及预测
        # 4. 最大化正/负样本的相关性 (分阶段 1. 正  2. 负 3. 正+负)
        
        # New AdCo implementation (independent of adco_bk):
        # - Use BNDL t-test pixel_uncertainty p-values directly
        # - Positive-only contribution now; negative path implemented but weight=0
        # - No gating/curriculum/temperature scheduling; no nested helper functions

        assert pixel_feat is not None and pixel_feat.ndim == 4
        B, H, W, _ = pixel_feat.shape
        device = pixel_feat.device
        dtype = pixel_feat.dtype

        # Ratio branch: if uncertainty missing, fallback to 1 - confidence from logits
        ratio_pos = torch.tensor(0.0, device=device, dtype=dtype)
        if pixel_logits is not None:
            # Build positive pixel mask
            # remove _adco_build_pos_neg_masks, simplfy
            pos_mask, _ = self._adco_build_pos_neg_masks(
                pixel_gt=pixel_gt,
                pixel_logits=pixel_logits,
                spatial_hw=(H, W),
                batch_size=B,
            )
            # Confidence from logits
            confidence = self._adco_compute_confidence(pixel_logits=pixel_logits)
            if confidence is not None:
                if pixel_uncertainty is None:
                    # Fallback uncertainty from logits: higher logit mag => lower uncertainty
                    uncertainty_p = (1.0 - confidence).clamp(0.0, 1.0)
                else:
                    uncertainty_p = torch.clamp(pixel_uncertainty, 0.0, 1.0)
                ratio_pos = self._adco_mean_ratio(uncertainty_p, confidence, pos_mask) # TODO 使用pos_mask的话，假阳性没有惩罚, 直接用gt

        # Neg off for now
        alpha_pos = 1.0
        alpha_neg = 1.0
        ratio_neg = torch.tensor(0.0, device=device, dtype=dtype)
        if (alpha_neg > 0.0) and (neg_sample_M is not None) and (neg_sample_M > 0) and (pixel_bndl_model is not None):
            neg_images = self._adco_sample_neg_images(neg_sample_M)  # [M, 3, Himg, Wimg]
            if neg_images is not None:
                if (neg_images.shape[-2] != self.image_size) or (neg_images.shape[-1] != self.image_size):
                    neg_images = F.interpolate(
                        neg_images,
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                enc_out = self.image_encoder(neg_images)
                feat = enc_out["backbone_fpn"][-1]
                neg_feat = feat.permute(0, 2, 3, 1).contiguous()
                neg_uq, neg_mean_logits = pixel_uncertain_sampling(
                    pixel_bndl_model,
                    neg_feat,
                    external_pre_out_w=external_pre_out_w, # TODO 是否是要生成对应负样本的prompt权重
                    sample_num=uq_sample_num,
                )
                neg_conf = self._adco_compute_conf_from_logits_tensor(neg_mean_logits)
                full_mask = torch.ones_like(neg_uq, dtype=torch.bool)
                ratio_neg = self._adco_mean_ratio(neg_uq, neg_conf, full_mask) # TODO 同上

        # Neg off for now
        alpha_pos = 1.0
        alpha_neg = 1.0
        # We want low uncertainty given high confidence => minimize the ratio
        loss = alpha_pos * ratio_pos + alpha_neg * ratio_neg

        return loss

    def compute_proco_loss(
        self,
        pixel_feat: torch.Tensor,                                # [B, H, W, C']
        use_background: bool = True,
        roi_z1: torch.Tensor | None = None,
        roi_z2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Prototype contrastive (Proxy-NCA/Proto-InfoNCE) with learnable prototypes.
        - Query q comes from ROI-averaged pixel feature (two views averaged for stability).
        - Positive is nearest object prototype; negatives are remaining object prototypes
          and optional background prototypes.
        """
        assert self.use_proco and pixel_feat is not None and pixel_feat.ndim == 4

        # Two ROI views for invariance, then average for a single query per sample
        if (roi_z1 is None) or (roi_z2 is None):
            with torch.no_grad():
                z1, _ = self._adco_roi_view(pixel_feat, min_scale=0.6, boundary_ignore=2, uncert=None)
                z2, _ = self._adco_roi_view(pixel_feat, min_scale=0.6, boundary_ignore=2, uncert=None)
                z = 0.5 * (z1 + z2)  # [B, C']
        else:
            z = 0.5 * (roi_z1 + roi_z2)

        # Projection + L2 normalize
        q = F.normalize(self.proco_proj(z), dim=1)  # [B, D]

        # Normalize prototype banks
        obj_bank = F.normalize(self.proco_obj_prototypes, dim=1)  # [Ko, D]
        if use_background and (self.proco_num_bg_prototypes > 0):
            bg_bank = F.normalize(self.proco_bg_prototypes, dim=1)  # [Kb, D]
        else:
            bg_bank = None

        # Similarity to object prototypes and choose nearest as positive
        sim_obj = q @ obj_bank.t()  # [B, Ko]
        pos_idx = torch.argmax(sim_obj, dim=1)  # [B]
        pos_sim = sim_obj.gather(dim=1, index=pos_idx.view(-1, 1))  # [B, 1]

        # Negatives: all other object prototypes + optional background prototypes
        if obj_bank.size(0) > 1:
            arange = torch.arange(obj_bank.size(0), device=q.device).view(1, -1)
            mask_other = (arange != pos_idx.view(-1, 1))  # [B, Ko]
            neg_obj = sim_obj[mask_other].view(q.size(0), -1)  # [B, Ko-1]
        else:
            neg_obj = torch.empty(q.size(0), 0, device=q.device, dtype=q.dtype)

        if bg_bank is not None:
            sim_bg = q @ bg_bank.t()  # [B, Kb]
            neg_all = torch.cat([neg_obj, sim_bg], dim=1) if neg_obj.numel() > 0 else sim_bg
        else:
            neg_all = neg_obj

        # Build logits and CE target
        tau = float(self.proco_temperature)
        if neg_all.numel() == 0:
            # Edge case: only one object prototype and no background prototypes
            logits = pos_sim / tau
        else:
            logits = torch.cat([pos_sim, neg_all], dim=1) / tau  # [B, 1+Kneg]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def compute_moco_loss(
        self,
        pixel_feat: torch.Tensor,                                # [B, H, W, C']
        roi_z1: torch.Tensor | None = None,
        roi_z2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        MoCo-style contrastive loss with momentum key encoder and queue negatives.
        """
        assert self.use_moco and pixel_feat is not None and pixel_feat.ndim == 4

        # Build or use shared two ROI views
        if (roi_z1 is None) or (roi_z2 is None):
            z1, _ = self._adco_roi_view(pixel_feat, min_scale=0.6, boundary_ignore=2, uncert=None)
            z2, _ = self._adco_roi_view(pixel_feat, min_scale=0.6, boundary_ignore=2, uncert=None)
        else:
            z1 = roi_z1
            z2 = roi_z2

        # Query projection
        q = F.normalize(self.moco_proj_q(z1), dim=1)  # [B, D]

        # Key projection with momentum update
        with torch.no_grad():
            self._moco_momentum_update_key_encoder()
            k = F.normalize(self.moco_proj_k(z2), dim=1)  # [B, D]

        # Positives: per-sample dot
        l_pos = (q * k).sum(dim=1, keepdim=True)  # [B, 1]
        # Negatives: queue
        l_neg = q @ self.moco_queue  # [B, K]

        logits = torch.cat([l_pos, l_neg], dim=1) / float(self.moco_temperature)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        # Update queue with keys from all GPUs if distributed
        with torch.no_grad():
            keys_to_enqueue = self._gather_batchwise(k)
            self._moco_dequeue_and_enqueue(keys_to_enqueue)

        return loss

    @property
    def device(self):
        return next(self.parameters()).device

    # --------------------------- AdCo (new) helpers ---------------------------
    def _adco_build_pos_neg_masks(
        self,
        pixel_gt: torch.Tensor | None,
        pixel_logits: torch.Tensor | None,
        spatial_hw: tuple[int, int],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build boolean masks for positive and negative pixels: [B, H, W].

        - If pixel_gt is provided, pos = any(gt>0) across channels.
        - Else if logits provided, pos = any(sigmoid(logits) >= 0.5) across channels.
        - Else fallback: all pixels as positive (negatives empty).
        """
        H, W = spatial_hw
        device = self.device
        if pixel_gt is not None:
            gt = pixel_gt
            if gt.ndim == 4 and gt.shape[-1] > 1:
                pos = (gt > 0).any(dim=-1)
            elif gt.ndim == 4 and gt.shape[1] == 1 and gt.shape[2] == H and gt.shape[3] == W:
                pos = (gt[:, 0] > 0)
            elif gt.ndim == 3 and gt.shape[1] == H and gt.shape[2] == W:
                pos = (gt > 0)
            else:
                # Attempt to reshape/interpret to [B, H, W]
                try:
                    B = gt.shape[0]
                    pos = (gt.view(B, H, W, -1) > 0).any(dim=-1)
                except Exception:
                    pos = torch.ones((gt.shape[0], H, W), device=gt.device, dtype=torch.bool)
        elif pixel_logits is not None:
            logits = pixel_logits
            if logits.ndim == 4 and logits.shape[-1] >= 1:
                p = torch.sigmoid(logits)
                pos = (p >= 0.5).any(dim=-1)
            elif logits.ndim == 3:
                pos = (torch.sigmoid(logits) >= 0.5)
            elif logits.ndim == 4 and logits.shape[1] == 1 and logits.shape[2] == H and logits.shape[3] == W:
                pos = (torch.sigmoid(logits[:, 0]) >= 0.5)
            else:
                try:
                    B = logits.shape[0]
                    p = torch.sigmoid(logits.view(B, H, W, -1))
                    pos = (p >= 0.5).any(dim=-1)
                except Exception:
                    pos = torch.ones((logits.shape[0], H, W), device=logits.device, dtype=torch.bool)
        else:
            # Fallback: consider all pixels as positive
            # (negative path exists but will not be used initially)
            pos = torch.ones((batch_size, H, W), device=device, dtype=torch.bool)

        neg = ~pos
        return pos.to(torch.bool), neg.to(torch.bool)

    def _adco_compute_confidence(
        self,
        pixel_logits: torch.Tensor | None,
        tau_conf: float = 2.0,
    ) -> torch.Tensor | None:
        """Compute per-pixel confidence in [0,1], [B, H, W] from logits only.

        c = sigmoid(max(|logits|) / tau_conf). Returns None if logits unavailable.
        # TODO 要在gt上取logits
        """
        if pixel_logits is None:
            return None
        logits = pixel_logits
        if logits.ndim == 4 and logits.shape[-1] >= 1:
            mag = torch.abs(logits)
            mag = mag.max(dim=-1).values  # [B, H, W]
        elif logits.ndim == 3:
            mag = torch.abs(logits)
        elif logits.ndim == 4 and logits.shape[1] == 1:
            mag = torch.abs(logits[:, 0])
        else:
            try:
                B2 = logits.shape[0]
                H2, W2 = logits.shape[1], logits.shape[2]
                mag = torch.abs(logits.view(B2, H2, W2, -1)).max(dim=-1).values
            except Exception:
                return None
        return torch.sigmoid(mag / float(tau_conf)).to(mag.dtype)

    def _adco_infonce_loss(
        self,
        pixel_feat: torch.Tensor,                                # [B, H, W, C']
        roi_z1: torch.Tensor | None,
        roi_z2: torch.Tensor | None,
        roi_w1: torch.Tensor | None,
        roi_w2: torch.Tensor | None,
        neg_sample_M: int | None,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE using two ROI queries and negatives from the image negative bank.

        - Queries/keys: two ROI pooled views from pixel features (self-supervised invariance), no GT/UQ.
        - Negatives: sample M images from `adco_negatives`, upsample to `image_size`, encode, global-pool, project.
        """
        assert pixel_feat is not None and pixel_feat.ndim == 4

        # 1) Build or reuse ROI views
        if (roi_z1 is None) or (roi_z2 is None):
            with torch.no_grad():
                z1, w1 = self._adco_roi_view(pixel_feat, min_scale=0.6, boundary_ignore=2, uncert=None)
                z2, w2 = self._adco_roi_view(pixel_feat, min_scale=0.6, boundary_ignore=2, uncert=None)
        else:
            z1, z2 = roi_z1, roi_z2
            # If external weights are provided, use them; else default to ones
            if (roi_w1 is None) or (roi_w2 is None):
                B = z1.shape[0]
                device = z1.device
                dtype = z1.dtype
                w1 = torch.ones(B, device=device, dtype=dtype)
                w2 = torch.ones(B, device=device, dtype=dtype)
            else:
                w1, w2 = roi_w1, roi_w2

        # 2) Project queries/keys
        q1 = F.normalize(self.adco_proj(z1), dim=1)
        q2 = F.normalize(self.adco_proj(z2), dim=1)
        k1 = F.normalize(self.adco_proj(z1).detach(), dim=1)
        k2 = F.normalize(self.adco_proj(z2).detach(), dim=1)

        # 3) Build negatives from image bank
        tau = float(self.adco_temperature)
        neg = None
        if (neg_sample_M is not None) and (neg_sample_M > 0) and (self.adco_negatives is not None):
            neg_images = self._adco_sample_neg_images(neg_sample_M)  # [M, 3, Hneg, Wneg]
            if neg_images is not None:
                if (neg_images.shape[-2] != self.image_size) or (neg_images.shape[-1] != self.image_size):
                    neg_images = F.interpolate(
                        neg_images,
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                with torch.no_grad():
                    enc_out = self.image_encoder(neg_images)
                    feat = enc_out["backbone_fpn"][-1]  # [M, C, Hf, Wf]
                # Global average pool spatial dims -> [M, C'] via simple mean over Hf,Wf then match adco_in dim
                m, c, hf, wf = feat.shape
                pooled = feat.mean(dim=(2, 3))  # [M, C]
                # Down-project to AdCo input dim if needed (use same heuristic as ROI views)
                # Our adco_proj expects in_dim = hidden_dim//8; ROI used z already in C' domain.
                # For negatives, approximate by a 1x linear into that domain via the first layer of adco_proj's in_features.
                # Simpler: map via the same adco_proj by faking a z-like feature using a linear adapter if sizes mismatch.
                # If pooled shape matches adco_proj[0].in_features we can feed directly; else use a 1x linear adapter.
                in_features = self.adco_proj[0].in_features
                if pooled.shape[1] != in_features:
                    adapter = torch.nn.Linear(pooled.shape[1], in_features, bias=False).to(pooled.device, pooled.dtype)
                    torch.nn.init.orthogonal_(adapter.weight)
                    pooled = adapter(pooled)
                neg = F.normalize(self.adco_proj(pooled), dim=1)  # [M, D]

        # 4) InfoNCE
        roi_w = (w1 + w2) * 0.5
        loss_1 = self._adco_infonce_once(q1, k2, neg, tau, weights=roi_w)
        loss_2 = self._adco_infonce_once(q2, k1, neg, tau, weights=roi_w)
        return 0.5 * loss_1 + 0.5 * loss_2

    def _adco_infonce_once(
        self,
        q: torch.Tensor,                # [B, D]
        k: torch.Tensor,                # [B, D]
        negatives: torch.Tensor | None, # [M, D] or None
        tau: float,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single-direction InfoNCE from q to k with optional negatives."""
        pos = (q * k).sum(dim=1, keepdim=True) / tau
        if negatives is None or negatives.numel() == 0:
            logits = pos
        else:
            logits = torch.cat([pos, (q @ negatives.t()) / tau], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        if weights is None:
            return F.cross_entropy(logits, labels)
        loss_per = F.cross_entropy(logits, labels, reduction="none")
        ww = weights / (weights.mean().detach() + 1e-6)
        return (loss_per * ww).mean()

    def _adco_mean_ratio(
        self,
        uncertainty_p: torch.Tensor,  # [B, H, W]
        confidence: torch.Tensor,     # [B, H, W]
        mask: torch.Tensor,           # [B, H, W] bool
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute masked mean of uncertainty/confidence ratio as scalar."""
        m = mask.to(uncertainty_p.dtype)
        denom = confidence * m + eps
        ratio = (uncertainty_p * m) / denom
        count = m.sum(dim=(1, 2)).clamp_min(1.0)
        per_sample = ratio.sum(dim=(1, 2)) / count
        return per_sample.mean()

    def _adco_sample_neg_images(self, M: int) -> torch.Tensor | None:
        """Sample M images from spatial negative bank as RGB tensors [M, 3, Himg, Wimg].

        Here we store adco_negatives as image-aligned RGB patches: [K, 3, Himg, Wimg].
        Returns None if the bank doesn't match this assumption.
        """
        neg = self.adco_negatives
        if neg is None or neg.ndim != 4 or neg.shape[1] != 3:
            return None
        K = neg.shape[0]
        device = neg.device
        if M <= 0:
            return None
        idx = torch.randint(0, K, (min(M, K),), device=device)
        return neg.index_select(0, idx)

    def _adco_compute_conf_from_logits_tensor(self, logits: torch.Tensor, tau_conf: float = 2.0) -> torch.Tensor:
        """Compute confidence from logits tensor [*, H, W, K] -> [*, H, W] via sigmoid(max(|logit|)/tau)."""
        if logits.ndim < 3:
            raise ValueError("logits tensor rank too low")
        mag = logits.abs().max(dim=-1).values  # [..., H, W]
        return torch.sigmoid(mag / float(tau_conf)).to(mag.dtype)

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
        pixel_gt_for_adco: torch.Tensor | None = None,
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
        
        if len(mask_decoder_outputs) == 5:  # 包含BNDL输出
            (
                low_res_multimasks,
                ious,
                sam_output_tokens,
                object_score_logits,
                bndl_outputs,
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
            pixel_feat = bndl_outputs.get("pixel_feat_grad", bndl_outputs.get("pixel_feat", None))
            shared_z1 = None
            shared_z2 = None
            shared_w1 = None
            shared_w2 = None
            # Build one pair of ROI views once (if enabled) and pass to all losses; methods handle None.
            if self.training and self.share_roi_views and (pixel_feat is not None):
                uncert = bndl_outputs.get("pixel_uncertainty", None) if self.adco_use_uncertainty else None
                shared_z1, shared_w1 = self._adco_roi_view(pixel_feat, min_scale=0.6, boundary_ignore=2, uncert=uncert)
                shared_z2, shared_w2 = self._adco_roi_view(pixel_feat, min_scale=0.6, boundary_ignore=2, uncert=uncert)
            # Optional per-frame uncertainty for AdCo
            pixel_uncertainty = bndl_outputs.get("pixel_uncertainty", None) if self.adco_use_uncertainty else None
            if self.use_adco and self.training and (pixel_feat is not None):
                # Prefer gradient-carrying logits key when training
                pixel_logits_for_adco = bndl_outputs.get(
                    "pixel_logits",
                    bndl_outputs.get("masks_bndl_raw", bndl_outputs.get("mean_pixel_logits", None)),
                )
                adco_loss = self.compute_adco_loss(
                    pixel_feat=pixel_feat,
                    pixel_uncertainty=pixel_uncertainty,
                    pixel_gt=pixel_gt_for_adco,
                    pixel_logits=pixel_logits_for_adco,
                    roi_z1=shared_z1,
                    roi_z2=shared_z2,
                    roi_w1=shared_w1,
                    roi_w2=shared_w2,
                    pixel_bndl_model=self.sam_mask_decoder.pixel_bndl if getattr(self.sam_mask_decoder, "pixel_bndl", None) is not None else None,
                    external_pre_out_w=bndl_outputs.get("wei_lambda_w", None),
                )
                bndl_outputs["adco_aux_loss"] = adco_loss
            if self.use_proco and self.training and (pixel_feat is not None):
                proco_loss = self.compute_proco_loss(
                    pixel_feat=pixel_feat,
                    use_background=True,
                    roi_z1=shared_z1,
                    roi_z2=shared_z2,
                )
                bndl_outputs["proco_aux_loss"] = proco_loss
            if self.use_moco and self.training and (pixel_feat is not None):
                moco_loss = self.compute_moco_loss(
                    pixel_feat=pixel_feat,
                    roi_z1=shared_z1,
                    roi_z2=shared_z2,
                )
                bndl_outputs["moco_aux_loss"] = moco_loss
            return (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
                bndl_outputs,
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
        pixel_gt_for_adco: torch.Tensor | None = None,
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
            # Apply DSU perturbation before SAM head
            if self.use_dsu:
                pix_feat = self._dsu_perturb(pix_feat)
                if self.dsu_apply_high_res and high_res_features is not None:
                    high_res_features = [self._dsu_perturb(f) for f in high_res_features]
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
                pixel_gt_for_adco=pixel_gt_for_adco,
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
        pixel_gt_for_adco: torch.Tensor | None = None,
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
            pixel_gt_for_adco,
        )

        if self.use_bndl_for_pixels:
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
                bndl_outputs,
            ) = sam_outputs
            # Optionally expose BNDL outputs for downstream consumers
            current_out["bndl_outputs"] = bndl_outputs
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
