# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from BNDL.BNDL_upload.ViT_Sparse.utils.bndl import (
    pixel_uncertain_sampling,
    pixel_entropy_uncertainty,
)
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

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
        # Whether to use UR-ERN for pixels (mutually exclusive with BNDL)
        use_ur_ern_for_pixels: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
        # AUE options (Adversarial Uncertainty Estimation)
        use_aue: bool = False,
        # Number of adversarial samples in the bank
        aue_num_adversarial_samples: int = 32,
        # Whether to initialize adversarial samples from dataset
        aue_init_from_dataset: bool = False,
        # Whether AUE uses uncertainty for ROI weighting (can be disabled)
        aue_use_uncertainty: bool = True,
        # Uncertainty-aware controls
        aue_uncertainty_mask_threshold: float | None = None,
        # Diversity regularization for adversarial samples
        aue_diversity_loss_weight: float = 0.0,  # Weight for diversity regularization (0 = disabled)
        # Constraint weight for adversarial samples (L1 distance to initial)
        aue_constraint_loss_weight: float = 0.0,  # Weight for constraint regularization (0 = disabled)
        # ROI view sharing option
        share_roi_views: bool = True,
        # Style Augmentation options (alternative to AUE, for domain generalization)
        use_style_aug: bool = False,
        style_aug_pgd_steps: int = 5,
        style_aug_pgd_step_size: float = 0.1,
        style_aug_pgd_epsilon: float = 2.0,
        style_aug_apply_at_layer: int = 0,  # Which backbone layer to apply AdaIN (0 = first FPN level)
        # Patch-based MMD configuration for AUE calibration loss
        aue_use_patch_mmd: bool = True,  # Enable patch-based MMD (better local distribution matching)
        aue_patch_size: int = 16,  # Patch size for patch-based MMD
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
        self.aue_num_adversarial_samples = int(aue_num_adversarial_samples)
        self.aue_init_from_dataset = bool(aue_init_from_dataset)
        self.aue_use_uncertainty = bool(aue_use_uncertainty)
        self.aue_uncertainty_mask_threshold = aue_uncertainty_mask_threshold
        self.aue_diversity_loss_weight = float(aue_diversity_loss_weight)
        self.aue_constraint_loss_weight = float(aue_constraint_loss_weight)
        # Patch-based MMD configuration
        self.aue_use_patch_mmd = aue_use_patch_mmd  # Enable patch-based MMD
        self.aue_patch_size = aue_patch_size  # Patch size for patch-based MMD
        if self.use_aue:
            self._build_aue_components()

        # Style Augmentation components (alternative to AUE)
        self.use_style_aug = bool(use_style_aug)
        self.style_aug_pgd_steps = int(style_aug_pgd_steps)
        self.style_aug_pgd_step_size = float(style_aug_pgd_step_size)
        self.style_aug_pgd_epsilon = float(style_aug_pgd_epsilon)
        self.style_aug_apply_at_layer = int(style_aug_apply_at_layer)
        if self.use_style_aug:
            self._build_style_aug_components()

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

    def _build_style_aug_components(self) -> None:
        """
        Build style augmentation components for adversarial training.
        
        Style-based adversarial training using:
        - AdaIN: Style transfer for domain augmentation
        - PGD: Iterative adversarial optimization in style space
        
        No pre-computed style bank needed - styles are extracted on-the-fly.
        """
        # AdaIN layer for style transfer
        from sam2.modeling.style_utils import AdaIN
        self.adain = AdaIN()
        
        logging.info(
            f"Style augmentation components built: "
            f"pgd_steps={self.style_aug_pgd_steps}, "
            f"epsilon={self.style_aug_pgd_epsilon}, "
            f"apply_at_layer={self.style_aug_apply_at_layer}"
        )
    
    def _build_aue_components(self) -> None:
        """
        Build components for Style-based AUE.
        
        Note: The old feature bank approach has been removed due to OOM issues.
        Style-based AUE does not require pre-allocated feature banks.
        """
        # Initialize SAM loss for PGD attacks (consistent with training config)
        from training.loss_fns import MultiStepMultiMasksAndIous
        
        # Use only segmentation-related losses for adversarial attacks
        # Focus on mask quality (focal + dice), not prediction heads (iou + class)
        weight_dict = {
            "loss_mask": 20.0,  # Higher weight for focal loss (segmentation quality)
            "loss_dice": 1.0,   # Dice loss (segmentation quality)
            "loss_iou": 0.0,    # Disable IoU prediction head loss
            "loss_class": 0.0   # Disable object existence prediction head loss
        }
        
        self.sam_loss_fn = MultiStepMultiMasksAndIous(
            weight_dict=weight_dict,
            focal_alpha=0.25,  # Default from loss_fns.py
            focal_gamma=2,     # Default from loss_fns.py
            supervise_all_iou=False,     # Disable IoU supervision for adversarial attacks
            iou_use_l1_loss=False,       # Not relevant when iou loss is disabled
            pred_obj_scores=False,       # Disable object score prediction for adversarial attacks
            focal_gamma_obj_score=0.0,   # Not relevant when pred_obj_scores=False
            focal_alpha_obj_score=-1.0,  # Not relevant when pred_obj_scores=False
        )
        
        mmd_type = "patch-based" if self.aue_use_patch_mmd else "global"
        logging.info(f"Style-based AUE enabled with SAM loss for PGD attacks (matching training config)")
        logging.info(f"AUE calibration loss: {mmd_type} MMD (patch_size={self.aue_patch_size if self.aue_use_patch_mmd else 'N/A'})")

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

    # Image-space constraint methods removed (not needed for feature-space AUE)

    def compute_aue_loss(
        self,
        pixel_feat: torch.Tensor,
        pixel_uncertainty: torch.Tensor | None = None,
        pixel_gt: torch.Tensor | None = None,
        pixel_logits: torch.Tensor | None = None,
        adversarial_sample_M: int | None = 1, 
        roi_z1: torch.Tensor | None = None,
        roi_z2: torch.Tensor | None = None,
        roi_w1: torch.Tensor | None = None,
        roi_w2: torch.Tensor | None = None,
        pixel_bndl_model=None,
        uq_sample_num: int = 8,
        # Style-based AUE parameters
        img_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, H, W, _ = pixel_feat.shape
        device = pixel_feat.device
        dtype = pixel_feat.dtype

        # Compute positive sample ratio (strict shape checks; logits must be provided)
        if pixel_logits is None:
            raise ValueError("AUE expects non-null pixel_logits of shape [B,H,W,K]")
        # Expect logits in channels-last format: [B, Hf, Wf, K]
        if not (pixel_logits.ndim == 4 and pixel_logits.shape[-1] >= 1):
            raise ValueError(f"AUE expects pixel_logits of shape [B,H,W,K], got {tuple(pixel_logits.shape)}")
        H_feat, W_feat = int(pixel_logits.shape[1]), int(pixel_logits.shape[2])

        # Optional GT: if provided, enforce [B,1,Hg,Wg] and resize; if None, ratio_pos will use default behavior
        pixel_gt_resized = None
        if pixel_gt is not None:
            if not (pixel_gt.ndim == 4 and pixel_gt.shape[1] == 1):
                raise ValueError(f"AUE expects pixel_gt of shape [B,1,H,W], got {tuple(pixel_gt.shape)}")
            pixel_gt_resized = F.interpolate(pixel_gt.float(), size=(H_feat, W_feat), mode='nearest').squeeze(1)

        calibration_loss_clean = self._compute_uncertainty_calibration_loss(
            pixel_logits=pixel_logits,
            pixel_uncertainty=pixel_uncertainty,
            pixel_gt=pixel_gt_resized,
            spatial_hw=(H_feat, W_feat),
            batch_size=B,
            device=device,
            dtype=dtype,
        )
 
        # Generate adversarial features using Style-based method
        if not (self.use_style_aug and img_batch is not None):
            # Style-based AUE requires img_batch
            logging.warning("Style-based AUE enabled but img_batch not provided, skipping AUE loss")
            return torch.tensor(0.0, device=device, dtype=dtype), {}
        
        adv_images, adv_gts, adv_prompts = self._generate_style_based_adversarial_images(
            img_batch, pixel_gt, adversarial_sample_M,
            pixel_bndl_model=pixel_bndl_model,
            uq_sample_num=uq_sample_num,
        )
        
        # Store images for visualization (detached, no gradients)
        vis_data = {}
        enable_vis = getattr(self, '_enable_style_visualization', False)
        if adv_images is not None and enable_vis:
            # Save first few samples for visualization
            num_vis_samples = min(4, img_batch.shape[0], adv_images.shape[0])
            vis_data['original_images'] = img_batch[:num_vis_samples].detach().cpu()
            vis_data['adversarial_images'] = adv_images[:num_vis_samples].detach().cpu()
            logging.info(f"Style AUE: Saved {num_vis_samples} samples for visualization, vis_data keys: {list(vis_data.keys())}")
        
        calibration_loss_adv = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
         
        if adv_images is not None:
            M = adv_images.shape[0]
            
            # Forward adversarial images through image encoder and get processed features
            adv_backbone_out = self.forward_image(adv_images)
            adv_backbone_feat = adv_backbone_out['backbone_fpn'][-1]  # [M, 256, 64, 64] - last FPN level
            
            # Extract high_res_features (already processed by forward_image)
            high_res_features = None
            if self.use_high_res_features_in_sam:
                # forward_image already processed these with conv_s0 and conv_s1
                high_res_features = [
                    adv_backbone_out['backbone_fpn'][0],  # [M, 32, 256, 256] - conv_s0 processed
                    adv_backbone_out['backbone_fpn'][1]   # [M, 64, 128, 128] - conv_s1 processed
                ]
            
            # Prompts are already in correct image_size space (e.g., 1024×1024), no scaling needed
            adv_prompts_scaled = adv_prompts
            
            adv_box_coords = torch.stack([adv_prompts_scaled[:, :2], adv_prompts_scaled[:, 2:]], dim=1)
            adv_point_inputs = {
                "point_coords": adv_box_coords,
                "point_labels": torch.tensor([[2, 3]], dtype=torch.int32, device=device).expand(M, 2),
            }
            
            # Forward through SAM heads; suppress nested AUE computation to avoid recursion
            prev_suppress = getattr(self, "_suppress_nested_aue", False)
            self._suppress_nested_aue = True
            try:
                *_, adv_aux_outputs = self._forward_sam_heads(
                    backbone_features=adv_backbone_feat,
                    point_inputs=adv_point_inputs,
                    high_res_features=high_res_features,
                    multimask_output=False,
                    pixel_gt_for_aue=None,
                )
            finally:
                self._suppress_nested_aue = prev_suppress
            
            # Extract BNDL tensors and prepare logits with gradients if available
            adv_bndl = adv_aux_outputs.get("bndl", {})
            adv_pixel_feat = adv_bndl.get("pixel_feat_grad", adv_bndl.get("pixel_feat"))
            adv_external_w = None
            if pixel_bndl_model is not None and not pixel_bndl_model.enable_global_sparse:
                adv_hyper_in = adv_bndl.get("hyper_in")
                adv_external_w = adv_hyper_in if adv_hyper_in is not None else pixel_bndl_model.linear.weight.unsqueeze(0).expand(M, -1, -1)

            adv_logits_grad = adv_bndl.get("pixel_logits", adv_bndl.get("masks_bndl_raw", None))
            if adv_logits_grad is None and (pixel_bndl_model is not None) and (adv_pixel_feat is not None):
                adv_logits_grad, *_ = pixel_bndl_model(
                    adv_pixel_feat, force_sample=False, external_pre_out_w=adv_external_w
                )

            # Compute uncertainty via sampling (no gradients)
            # Use entropy-based uncertainty for smoother gradients in adversarial training
            with torch.no_grad():
                adv_uq = pixel_entropy_uncertainty(
                    pixel_bndl_model, adv_pixel_feat, adv_external_w, uq_sample_num, per_channel=False
                )
                # Also get sampled logits for ratio computation
                _, adv_logits_sample = pixel_uncertain_sampling(
                    pixel_bndl_model, adv_pixel_feat, adv_external_w, uq_sample_num
                )
            
            # Resize GT to match logits (logits are at feature map resolution)
            adv_logits_for_ratio = adv_logits_grad if adv_logits_grad is not None else adv_logits_sample
            
            # Sanitize logits: replace NaN/Inf with safe values
            if not torch.isfinite(adv_logits_for_ratio).all():
                num_nan = (~torch.isfinite(adv_logits_for_ratio)).sum().item()
                logging.warning(f"[Style AUE] adv_logits contains {num_nan}/{adv_logits_for_ratio.numel()} NaN/Inf, sanitizing...")
                adv_logits_for_ratio = torch.nan_to_num(adv_logits_for_ratio, nan=0.0, posinf=20.0, neginf=-20.0)
                adv_logits_for_ratio = torch.clamp(adv_logits_for_ratio, min=-50.0, max=50.0)
            
            H_feat, W_feat = adv_logits_for_ratio.shape[1:3]
            adv_gts_resized = F.interpolate(adv_gts.unsqueeze(1).float(), size=(H_feat, W_feat), mode='nearest')
            
            # Compute calibration loss
            calibration_loss_adv = self._compute_uncertainty_calibration_loss(
                pixel_logits=adv_logits_for_ratio,
                pixel_uncertainty=adv_uq,
                pixel_gt=adv_gts_resized,
                spatial_hw=(H_feat, W_feat),
                batch_size=M,
                device=device,
                dtype=dtype,
            )
            
            # Clean up intermediate tensors to save memory
            del adv_bndl, adv_aux_outputs
            
        # Initialize loss dictionary for detailed logging
        loss_dict = {}
        
        # Main losses
        loss_dict['calibration_loss_clean'] = calibration_loss_clean
        loss_dict['calibration_loss_adv'] = calibration_loss_adv
        
        loss = calibration_loss_clean + calibration_loss_adv
        loss_dict['total_loss'] = loss
        
        # Add visualization data if available (check for keys, not truthiness)
        if 'original_images' in vis_data:
            loss_dict['style_aue_visualization'] = vis_data

        return loss, loss_dict

    def _generate_style_based_adversarial_images(
        self,
        img_batch: torch.Tensor,
        pixel_gt: torch.Tensor | None,
        sample_M: int,
        pixel_bndl_model=None,
        uq_sample_num: int = 8,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Generate adversarial images using PGD + AdaIN in style space.
        
        Args:
            img_batch: [B, 3, H, W] input images (normalized)
            pixel_gt: [B, 1, H, W] ground truth masks (optional)
            sample_M: number of adversarial samples to generate
            pixel_bndl_model: BNDL model for computing uncertainty
            uq_sample_num: number of samples for uncertainty estimation
        
        Returns:
            adv_images: [M, 3, H, W] adversarial images with style perturbation
            adv_gts: [M, H, W] ground truth masks
            adv_prompts: [M, 4] bounding box prompts
        """
        device = img_batch.device
        
        # 1. Extract original styles directly from current images (no bank needed!)
        from sam2.modeling.style_utils import extract_style_statistics
        original_styles = extract_style_statistics(img_batch)  # [B, 6]
        
        # 2. Run PGD to find adversarial styles
        adv_styles = self._pgd_find_adversarial_styles(
            img_batch, 
            original_styles,
            pixel_gt=pixel_gt,
            num_steps=self.style_aug_pgd_steps,
            step_size=self.style_aug_pgd_step_size,
            epsilon=self.style_aug_pgd_epsilon,
            pixel_bndl_model=pixel_bndl_model,
            uq_sample_num=uq_sample_num,
        )
        
        # 3. Apply adversarial styles to images using AdaIN
        # Note: forward_image_with_style applies AdaIN internally and returns features
        # We need to generate actual styled images, so we apply AdaIN to the input images
        adv_images = self._apply_style_to_images(img_batch, adv_styles)
        
        # 4. Generate prompts from ground truth (bounding boxes)
        if pixel_gt is not None:
            # Generate bounding box prompts from GT masks
            adv_prompts = self._generate_bbox_prompts_from_gt(pixel_gt)  # [B, 4]
            adv_gts = pixel_gt.squeeze(1)  # [B, H, W]
        else:
            # If no GT available, cannot generate prompts
            logging.warning("No GT available for style-based AUE, cannot generate prompts")
            return None, None, None
        
        # 5. Sample M instances if needed (for now, just use all B samples)
        # In the future, we could sample M < B for efficiency
        M = min(sample_M, img_batch.shape[0])
        if img_batch.shape[0] > M:
            # Randomly sample M instances
            indices = torch.randperm(img_batch.shape[0], device=device)[:M]
            adv_images = adv_images[indices]
            adv_gts = adv_gts[indices]
            adv_prompts = adv_prompts[indices]
        
        return adv_images, adv_gts, adv_prompts

    def _apply_style_to_images(
        self, 
        img_batch: torch.Tensor, 
        style_stats: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Apply style statistics to images using AdaIN in image space.
        
        Args:
            img_batch: [B, 3, H, W] normalized images
            style_stats: [B, 6] style statistics (means[:3], stds[3:]) or None
        
        Returns:
            styled_images: [B, 3, H, W] styled images (still normalized)
        """
        # If no style stats provided, return original images
        if style_stats is None:
            return img_batch
            
        B, C, H, W = img_batch.shape
        
        # Extract target means and stds from style_stats
        target_means = style_stats[:, :3].view(B, 3, 1, 1)  # [B, 3, 1, 1]
        target_stds = style_stats[:, 3:].view(B, 3, 1, 1)   # [B, 3, 1, 1]
        
        # Compute current image statistics
        current_means = img_batch.mean(dim=[2, 3], keepdim=True)  # [B, 3, 1, 1]
        current_stds = img_batch.std(dim=[2, 3], keepdim=True)   # [B, 3, 1, 1]
        
        # Apply AdaIN: normalize then denormalize with target stats
        # styled = target_std * (input - input_mean) / input_std + target_mean
        normalized = (img_batch - current_means) / (current_stds + 1e-8)
        styled_images = normalized * target_stds + target_means
        
        return styled_images

    def _pgd_find_adversarial_styles(
        self,
        img_batch: torch.Tensor,
        original_styles: torch.Tensor,
        pixel_gt: torch.Tensor | None = None,
        num_steps: int = 5,
        step_size: float = 0.1,
        epsilon: float = 2.0,
        pixel_bndl_model=None,
        uq_sample_num: int = 20,
    ) -> torch.Tensor:
        """
        Performs PGD to find adversarial style statistics that maximize the AUE calibration loss.
        Goal: Maximize uncertainty calibration loss (MMD + MSE between U and Error).
        
        Args:
            img_batch: [B, 3, H, W] original image batch
            original_styles: [B, 6] original style statistics (means[:3], stds[3:])
            pixel_gt: [B, H, W] ground truth masks
            num_steps: number of PGD iterations
            step_size: step size for each PGD iteration
            epsilon: L_inf perturbation budget
            pixel_bndl_model: BNDL model for computing uncertainty
            uq_sample_num: number of samples for uncertainty estimation
        
        Returns:
            adv_styles: [B, 6] adversarial style statistics
        """
        adv_styles = original_styles.clone().detach()
        
        for step in range(num_steps):
            adv_styles.requires_grad = True
            
            # 1. Apply style and forward through model
            styled_images = self._apply_style_to_images(img_batch, adv_styles)
            adv_backbone_out = self.forward_image(styled_images)
            adv_backbone_feat = adv_backbone_out['backbone_fpn'][-1]

            # 2. Extract high_res_features if needed
            high_res_features = None
            if self.use_high_res_features_in_sam:
                high_res_features = [
                    adv_backbone_out['backbone_fpn'][0],
                    adv_backbone_out['backbone_fpn'][1]
                ]
            
            # 3. Generate prompts from GT
            adv_prompts = self._generate_bbox_prompts_from_gt(pixel_gt)
            adv_box_coords = torch.stack([adv_prompts[:, :2], adv_prompts[:, 2:]], dim=1)
            adv_point_inputs = {
                "point_coords": adv_box_coords,
                "point_labels": torch.tensor([[2, 3]], dtype=torch.int32, device=img_batch.device).expand(img_batch.shape[0], 2),
            }
            
            # 4. Forward through SAM heads to get BNDL outputs
            prev_suppress = getattr(self, "_suppress_nested_aue", False)
            self._suppress_nested_aue = True
            try:
                *_, adv_aux_outputs = self._forward_sam_heads(
                    backbone_features=adv_backbone_feat,
                    point_inputs=adv_point_inputs,
                    high_res_features=high_res_features,
                    multimask_output=False,
                    pixel_gt_for_aue=None,
                )
            finally:
                self._suppress_nested_aue = prev_suppress
            
            # 5. Extract BNDL outputs
            adv_bndl = adv_aux_outputs.get("bndl", {})
            adv_pixel_feat = adv_bndl.get("pixel_feat_grad", adv_bndl.get("pixel_feat"))
            
            if adv_pixel_feat is None:
                logging.warning("PGD: Failed to extract pixel features, stopping early")
                break
             
            # 6. Get hyperparameters for BNDL
            adv_external_w = None
            if pixel_bndl_model is not None and not pixel_bndl_model.enable_global_sparse:
                adv_hyper_in = adv_bndl.get("hyper_in")
                adv_external_w = adv_hyper_in if adv_hyper_in is not None else pixel_bndl_model.linear.weight.unsqueeze(0).expand(adv_pixel_feat.shape[0], -1, -1)
            
            # 7. Compute logits with gradients
            adv_logits_grad = adv_bndl.get("pixel_logits", adv_bndl.get("masks_bndl_raw", None))
            if adv_logits_grad is None and pixel_bndl_model is not None:
                adv_logits_grad, *_ = pixel_bndl_model(
                    adv_pixel_feat, force_sample=False, external_pre_out_w=adv_external_w
                )
            
            if adv_logits_grad is None:
                logging.warning("PGD: Failed to extract logits, stopping early")
                break
            
            # 8. Compute uncertainty (no gradients to avoid nested grads)
            with torch.no_grad():
                adv_uq = pixel_entropy_uncertainty(
                    pixel_bndl_model, adv_pixel_feat, adv_external_w, uq_sample_num, per_channel=False
                )
            
            # Sanitize logits before computing calibration loss
            if not torch.isfinite(adv_logits_grad).all():
                num_nan = (~torch.isfinite(adv_logits_grad)).sum().item()
                logging.warning(f"[PGD Step {step}] adv_logits contains {num_nan}/{adv_logits_grad.numel()} NaN/Inf, sanitizing...")
                adv_logits_grad = torch.nan_to_num(adv_logits_grad, nan=0.0, posinf=20.0, neginf=-20.0)
                adv_logits_grad = torch.clamp(adv_logits_grad, min=-50.0, max=50.0)
            
            # 9. Compute calibration loss (maximize to find adversarial styles)
            H_feat, W_feat = adv_logits_grad.shape[1:3]
            adv_gts_resized = F.interpolate(pixel_gt.float(), size=(H_feat, W_feat), mode='nearest')
            
            calibration_loss_adv = self._compute_uncertainty_calibration_loss(
                pixel_logits=adv_logits_grad,
                pixel_uncertainty=adv_uq,
                pixel_gt=adv_gts_resized,
                spatial_hw=(H_feat, W_feat),
                batch_size=adv_logits_grad.shape[0],
                device=adv_logits_grad.device,
                dtype=adv_logits_grad.dtype,
            )
            
            # 10. Gradient ascent (maximize calibration loss to create hard samples)
            grad = torch.autograd.grad(calibration_loss_adv, adv_styles, create_graph=False)[0]
            
            with torch.no_grad():
                # Gradient ascent step
                adv_styles = adv_styles.detach() + step_size * grad.sign()
                
                # Project to epsilon ball
                delta = adv_styles - original_styles
                delta = torch.clamp(delta, -epsilon, epsilon)
                adv_styles = original_styles + delta
 
        return adv_styles.detach()

    def _generate_bbox_prompts_from_gt(self, gt_masks: torch.Tensor) -> torch.Tensor:
        """
        Generate bounding box prompts from ground truth masks.
        
        Args:
            gt_masks: [B, 1, H, W] ground truth masks
        
        Returns:
            bbox_prompts: [B, 4] bounding boxes in format [x1, y1, x2, y2]
        """
        B = gt_masks.shape[0]
        device = gt_masks.device
        H, W = gt_masks.shape[2], gt_masks.shape[3]
        
        bbox_prompts = []
        for i in range(B):
            mask = gt_masks[i, 0]  # [H, W]
            
            # Find non-zero pixels
            nonzero = torch.nonzero(mask > 0.5, as_tuple=False)  # [N, 2] (y, x)
            
            if nonzero.shape[0] > 0:
                # Compute bounding box
                y_min = nonzero[:, 0].min().item()
                y_max = nonzero[:, 0].max().item()
                x_min = nonzero[:, 1].min().item()
                x_max = nonzero[:, 1].max().item()
                
                # Convert to [x1, y1, x2, y2] format
                bbox = torch.tensor([x_min, y_min, x_max, y_max], device=device, dtype=torch.float32)
            else:
                # If mask is empty, use a default small box in the center
                bbox = torch.tensor([W // 2 - 10, H // 2 - 10, W // 2 + 10, H // 2 + 10], device=device, dtype=torch.float32)

            bbox_prompts.append(bbox)

        return torch.stack(bbox_prompts, dim=0)  # [B, 4]

    @property
    def device(self):
        return next(self.parameters()).device

    # --------------------------- AUE helpers ---------------------------
    def _compute_uncertainty_calibration_loss(
        self,
        pixel_logits: torch.Tensor | None,
        pixel_uncertainty: torch.Tensor | None,
        pixel_gt: torch.Tensor | None,
        spatial_hw: tuple[int, int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute uncertainty calibration loss via distribution matching.
        
        Theory: For zero-shot robustness, uncertainty distribution should match
        error distribution using Maximum Mean Discrepancy (MMD).
        
        Loss = MMD(P_U, P_Error) + 0.3 * MSE(U, Error)
        
        References:
        - Gretton et al. (2012): "A Kernel Two-Sample Test"
        - Long et al. (2015): "Learning Transferable Features with DAN"
        
        Returns:
            calibration_loss: MMD-based distribution matching loss
        """
        if pixel_logits is None:
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        
        # 1. Compute prediction error [B, H, W] in [0, 1]
        error = self._compute_prediction_error(
            pixel_logits=pixel_logits,
            pixel_gt=pixel_gt,
        )
        
        # Debug: Check error for NaN
        if not torch.isfinite(error).all():
            import logging
            num_nan = (~torch.isfinite(error)).sum().item()
            total = error.numel()
            logging.error(f"[AUE Debug] Error contains {num_nan}/{total} NaN/Inf values!")
            logging.error(f"[AUE Debug] Error stats: min={error[torch.isfinite(error)].min():.4f}, "
                         f"max={error[torch.isfinite(error)].max():.4f}, mean={error[torch.isfinite(error)].mean():.4f}")
            # Check logits
            if not torch.isfinite(pixel_logits).all():
                num_nan_logits = (~torch.isfinite(pixel_logits)).sum().item()
                logging.error(f"[AUE Debug] pixel_logits contains {num_nan_logits}/{pixel_logits.numel()} NaN/Inf!")
        
        # 2. Get uncertainty [B, H, W] in [0, 1]
        if pixel_uncertainty is not None:
            # Debug: Check uncertainty before clamping
            if not torch.isfinite(pixel_uncertainty).all():
                import logging
                num_nan = (~torch.isfinite(pixel_uncertainty)).sum().item()
                total = pixel_uncertainty.numel()
                logging.error(f"[AUE Debug] Uncertainty (before clamp) contains {num_nan}/{total} NaN/Inf values!")
                if torch.isfinite(pixel_uncertainty).any():
                    logging.error(f"[AUE Debug] Uncertainty stats: min={pixel_uncertainty[torch.isfinite(pixel_uncertainty)].min():.4f}, "
                                 f"max={pixel_uncertainty[torch.isfinite(pixel_uncertainty)].max():.4f}")
            
            uncertainty = pixel_uncertainty.clamp(0.0, 1.0)
            
            # Debug: Check uncertainty after clamping
            if not torch.isfinite(uncertainty).all():
                import logging
                num_nan = (~torch.isfinite(uncertainty)).sum().item()
                total = uncertainty.numel()
                logging.error(f"[AUE Debug] Uncertainty (after clamp) still contains {num_nan}/{total} NaN/Inf values!")
        else:
            # Fallback: use 1 - confidence
            confidence = self._aue_compute_confidence(pixel_logits, pixel_gt)
            
            # Debug: Check confidence
            if not torch.isfinite(confidence).all():
                import logging
                num_nan = (~torch.isfinite(confidence)).sum().item()
                logging.error(f"[AUE Debug] Confidence contains {num_nan}/{confidence.numel()} NaN/Inf values!")
            
            uncertainty = (1.0 - confidence).clamp(0.0, 1.0)
        
        # Input validation: check for NaN/Inf
        if not torch.isfinite(uncertainty).all() or not torch.isfinite(error).all():
            import logging
            logging.warning("[AUE] Uncertainty or Error contains NaN/Inf after processing, skipping calibration loss")
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        
        # 3. MMD loss (primary: distribution matching)
        # Choose between patch-based or global MMD
        if getattr(self, 'aue_use_patch_mmd', True):
            # Patch-based MMD for better local distribution matching
            mmd_loss = self._compute_patch_based_mmd(
                uncertainty.detach(),  # [B, H, W]
                error,                 # [B, H, W]
                patch_size=getattr(self, 'aue_patch_size', 16),
                kernel='rbf',
                bandwidth=0.1,
            )
        else:
            # Global MMD (original approach)
            mmd_loss = self._compute_mmd(
                uncertainty.detach().flatten().unsqueeze(-1),  # [N, 1]
                error.flatten().unsqueeze(-1),                 # [N, 1]
                kernel='rbf',
                bandwidth=0.1,
            )
        
        # 4. MSE loss (regularization: point-wise alignment)
        # Detach uncertainty to ensure gradients only flow through error
        mse_loss = F.mse_loss(uncertainty.detach(), error, reduction='mean')
        
        # 5. Combine losses
        total_loss = 1.0 * mmd_loss + 0.3 * mse_loss
        
        return total_loss
    
    def _compute_prediction_error(
        self,
        pixel_logits: torch.Tensor,
        pixel_gt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-pixel prediction error in [0, 1].
        
        Error = |sigmoid(logits) - GT|
        
        Args:
            pixel_logits: [B, H, W, K] or [B, H, W] logits
            pixel_gt: [B, H, W] or [B, 1, H, W] ground truth
        
        Returns:
            error: [B, H, W] in [0, 1], where 0=perfect, 1=wrong
        """
        # Sanitize input logits to prevent NaN propagation
        if not torch.isfinite(pixel_logits).all():
            num_nan = (~torch.isfinite(pixel_logits)).sum().item()
            logging.error(f"[Error Computation] Input logits contain {num_nan}/{pixel_logits.numel()} NaN/Inf values! Sanitizing...")
            pixel_logits = torch.nan_to_num(pixel_logits, nan=0.0, posinf=20.0, neginf=-20.0)
            pixel_logits = torch.clamp(pixel_logits, min=-50.0, max=50.0)
        
        # Extract logits value
        if pixel_logits.ndim == 4 and pixel_logits.shape[-1] >= 1:
            logits_val = pixel_logits.max(dim=-1).values  # [B, H, W]
        elif pixel_logits.ndim == 3:
            logits_val = pixel_logits
        elif pixel_logits.ndim == 4 and pixel_logits.shape[1] == 1:
            logits_val = pixel_logits[:, 0]
        else:
            B, H, W = pixel_logits.shape[:3]
            logits_val = pixel_logits.view(B, H, W, -1).max(dim=-1).values
        
        # Extract GT mask [B, H, W]
        H, W = logits_val.shape[1], logits_val.shape[2]
        B = logits_val.shape[0]
        gt_mask = self._extract_mask_from_gt(
            pixel_gt=pixel_gt,
            spatial_hw=(H, W),
            batch_size=B,
            device=logits_val.device,
        )
        
        # Compute prediction probability
        pred_prob = torch.sigmoid(logits_val)  # [B, H, W] in [0, 1]
        
        # Compute absolute error
        gt_float = gt_mask.float()  # [B, H, W] in {0, 1}
        error = torch.abs(pred_prob - gt_float)  # [B, H, W] in [0, 1]
        
        return error
    
    def _compute_patch_based_mmd(
        self,
        uncertainty: torch.Tensor,  # [B, H, W]
        error: torch.Tensor,        # [B, H, W]
        patch_size: int = 16,
        kernel: str = 'rbf',
        bandwidth: float = 0.1,
    ) -> torch.Tensor:
        """Compute patch-based MMD for local distribution matching.
        
        Instead of flattening all pixels, we:
        1. Extract patches from uncertainty and error maps
        2. Compute patch-level statistics (mean, std, etc.)
        3. Apply MMD on patch statistics for better local alignment
        
        This approach:
        - Preserves spatial locality
        - Reduces computational cost (fewer samples)
        - Better captures local uncertainty-error relationships
        
        Args:
            uncertainty: [B, H, W] uncertainty map
            error: [B, H, W] error map
            patch_size: size of square patches (default 16)
            kernel: kernel type for MMD
            bandwidth: kernel bandwidth
        
        Returns:
            mmd_loss: scalar MMD value
        """
        B, H, W = uncertainty.shape
        device = uncertainty.device
        dtype = uncertainty.dtype
        
        # Calculate number of patches
        n_patches_h = H // patch_size
        n_patches_w = W // patch_size
        
        # Handle case where image size is not divisible by patch_size
        if n_patches_h == 0 or n_patches_w == 0:
            # Fallback to global MMD if patches are too large
            return self._compute_mmd(
                uncertainty.flatten().unsqueeze(-1),
                error.flatten().unsqueeze(-1),
                kernel=kernel,
                bandwidth=bandwidth,
            )
        
        # Crop to make divisible by patch_size
        H_crop = n_patches_h * patch_size
        W_crop = n_patches_w * patch_size
        uncertainty_crop = uncertainty[:, :H_crop, :W_crop]
        error_crop = error[:, :H_crop, :W_crop]
        
        # Reshape to patches: [B, n_patches_h, patch_size, n_patches_w, patch_size]
        uncertainty_patches = uncertainty_crop.reshape(
            B, n_patches_h, patch_size, n_patches_w, patch_size
        )
        error_patches = error_crop.reshape(
            B, n_patches_h, patch_size, n_patches_w, patch_size
        )
        
        # Rearrange to [B, n_patches_h, n_patches_w, patch_size, patch_size]
        uncertainty_patches = uncertainty_patches.permute(0, 1, 3, 2, 4)
        error_patches = error_patches.permute(0, 1, 3, 2, 4)
        
        # Flatten to [B * n_patches_h * n_patches_w, patch_size * patch_size]
        n_total_patches = B * n_patches_h * n_patches_w
        uncertainty_patches = uncertainty_patches.reshape(n_total_patches, patch_size * patch_size)
        error_patches = error_patches.reshape(n_total_patches, patch_size * patch_size)
        
        # Compute patch-level statistics as features
        # Using mean and std to capture distribution within each patch
        u_mean = uncertainty_patches.mean(dim=1, keepdim=True)  # [N, 1]
        u_std = uncertainty_patches.std(dim=1, keepdim=True)    # [N, 1]
        e_mean = error_patches.mean(dim=1, keepdim=True)        # [N, 1]
        e_std = error_patches.std(dim=1, keepdim=True)          # [N, 1]
        
        # Concatenate statistics as patch features
        uncertainty_features = torch.cat([u_mean, u_std], dim=1)  # [N, 2]
        error_features = torch.cat([e_mean, e_std], dim=1)        # [N, 2]
        
        # Compute MMD on patch features
        mmd_loss = self._compute_mmd(
            uncertainty_features,
            error_features,
            kernel=kernel,
            bandwidth=bandwidth,
            batch_size=min(256, n_total_patches // 2),  # Adaptive batch size
            n_batches=10,
        )
        
        return mmd_loss
    
    def _compute_mmd(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        kernel: str = 'rbf',
        bandwidth: float = 0.1,
        batch_size: int = 256,  # Mini-batch size for memory efficiency
        n_batches: int = 10,    # Number of mini-batches to average
    ) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy using mini-batch estimation.
        
        MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        
        Uses mini-batch sampling to avoid OOM while maintaining unbiased estimation.
        Memory: O(batch_size²) instead of O(N²)
        
        Theory: MMD is a kernel-based distribution distance metric proven
        to be robust for domain adaptation tasks (Long et al. 2015, Gretton et al. 2012).
        
        Args:
            x: [N, D] samples from distribution P (uncertainty)
            y: [M, D] samples from distribution Q (error)
            kernel: 'rbf' (Gaussian kernel, recommended)
            bandwidth: kernel bandwidth (σ in RBF kernel)
            batch_size: size of mini-batch for each iteration (default 256)
            n_batches: number of mini-batches to average (default 10)
        
        Returns:
            mmd: scalar MMD value (≥0)
        
        References:
            - Gretton et al. (2012): "A Kernel Two-Sample Test"
            - Long et al. (2015): "Learning Transferable Features with DAN"
        """
        N, M = x.shape[0], y.shape[0]
        
        # Safety check: need at least 2 samples
        if N < 2 or M < 2:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        # Adjust batch_size if dataset is smaller (need at least 2 for unbiased estimator)
        actual_batch_size = max(2, min(batch_size, N, M))
        
        # RBF kernel using mathematical expansion (memory-efficient)
        # k(x, y) = exp(-||x-y||² / (2*σ²))
        def rbf_kernel_efficient(x1, x2, sigma):
            """
            Compute RBF kernel without using cdist (memory-efficient).
            
            Args:
                x1: [B, D]
                x2: [B, D]
                sigma: bandwidth
            
            Returns:
                kernel_matrix: [B, B]
            """
            # Clamp inputs to reasonable range to prevent overflow in squaring
            # (uncertainty and error should be in [0,1], but allow margin for safety)
            x1 = torch.clamp(x1, min=-10.0, max=10.0)
            x2 = torch.clamp(x2, min=-10.0, max=10.0)
            
            # Compute squared norms
            x1_norm_sq = torch.sum(x1 ** 2, dim=1, keepdim=True)  # [B, 1]
            x2_norm_sq = torch.sum(x2 ** 2, dim=1, keepdim=True)  # [B, 1]
            
            # Compute inner product
            inner_prod = torch.mm(x1, x2.t())  # [B, B]
            
            # Compute squared distances: ||x-y||² = ||x||² + ||y||² - 2<x,y>
            dist_sq = x1_norm_sq + x2_norm_sq.t() - 2 * inner_prod  # [B, B]
            
            # Clamp to avoid negative values due to numerical errors
            dist_sq = torch.clamp(dist_sq, min=0.0)
            
            # RBF kernel with numerical stability
            # Add epsilon to avoid division issues
            sigma_sq = sigma ** 2 + 1e-8
            
            # Clamp exponent to avoid overflow/underflow
            exponent = -dist_sq / (2 * sigma_sq)
            exponent = torch.clamp(exponent, min=-50.0, max=50.0)
            
            return torch.exp(exponent)
        
        # Mini-batch MMD estimation
        mmd_sum = 0.0
        
        for _ in range(n_batches):
            # Randomly sample mini-batches
            idx_x = torch.randperm(N, device=x.device)[:actual_batch_size]
            idx_y = torch.randperm(M, device=y.device)[:actual_batch_size]
            
            x_batch = x[idx_x]
            y_batch = y[idx_y]
            
            # Compute kernel matrices for this mini-batch
            k_xx = rbf_kernel_efficient(x_batch, x_batch, bandwidth)  # [B, B]
            k_yy = rbf_kernel_efficient(y_batch, y_batch, bandwidth)  # [B, B]
            k_xy = rbf_kernel_efficient(x_batch, y_batch, bandwidth)  # [B, B]
            
            # Unbiased MMD estimator: exclude diagonal terms
            b = k_xx.shape[0]
            
            # Safety check: should never happen due to actual_batch_size >= 2
            assert b >= 2, f"Batch size {b} < 2, this should not happen"
            
            # Sum all elements, subtract diagonal, normalize by b(b-1)
            k_xx_sum = k_xx.sum() - k_xx.diagonal().sum()
            k_yy_sum = k_yy.sum() - k_yy.diagonal().sum()
            k_xy_sum = k_xy.sum()
            
            # Unbiased estimator for this batch (add epsilon for numerical stability)
            eps = 1e-8
            mmd_squared_batch = k_xx_sum / (b * (b - 1) + eps) + k_yy_sum / (b * (b - 1) + eps) - 2 * k_xy_sum / (b * b + eps)
            
            # Accumulate (take sqrt later to reduce numerical issues)
            mmd_sum += mmd_squared_batch
            
            # Free memory
            del k_xx, k_yy, k_xy
        
        # Average over batches (add epsilon for numerical stability)
        mmd_squared_avg = mmd_sum / (n_batches + 1e-8)
        
        # Clamp before sqrt (ensure non-negative, allow reasonable upper bound)
        mmd_squared_avg = torch.clamp(mmd_squared_avg, min=0.0, max=10.0)
        
        # Take square root
        mmd = torch.sqrt(mmd_squared_avg)
        
        return mmd
    
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
            return torch.ones((batch_size, H, W), device=device, dtype=torch.bool, requires_grad=True)
        
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
            logits_val = pixel_logits.view(B, H, W, -1).mean(dim=-1)
        
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

    def _aue_compute_conf_from_logits_tensor(self, logits: torch.Tensor, tau_conf: float = 2.0) -> torch.Tensor:
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
        img_batch_for_style_aue: torch.Tensor | None = None,
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
            aux_outputs = {}  # 标准 SAM2 模式：空的辅助输出

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
            # Track selected index internally for downstream slicing (no external exposure)
            selected_mask_index = best_iou_inds.to(dtype=torch.long)
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks
            # Single-mask case: selected index is 0 for all samples
            selected_mask_index = torch.zeros(B, dtype=torch.long, device=device)

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
            # Also expose selected channel's hyper_in for single-channel UQ without revealing indices
            if 'hyper_in' in bndl_outputs and bndl_outputs['hyper_in'] is not None:
                hyper_in_full = bndl_outputs['hyper_in']  # [B, K, C'] (detached)
                if isinstance(hyper_in_full, torch.Tensor) and hyper_in_full.ndim == 3:
                    batch_inds = torch.arange(hyper_in_full.size(0), device=device)
                    hyper_in_selected = hyper_in_full[batch_inds, selected_mask_index]  # [B, C']
                    bndl_outputs['hyper_in_selected'] = hyper_in_selected.detach()
            shared_z1 = None
            shared_z2 = None
            shared_w1 = None
            shared_w2 = None
            # Build one pair of ROI views once (if enabled) and pass to all losses; methods handle None.
            if self.training and self.share_roi_views and (pixel_feat is not None):
                uncert = bndl_outputs.get("pixel_uncertainty", None) if self.aue_use_uncertainty else None
                shared_z1, shared_w1 = self._aue_roi_view(pixel_feat, min_scale=0.6, boundary_ignore=2, uncert=uncert)
                shared_z2, shared_w2 = self._aue_roi_view(pixel_feat, min_scale=0.6, boundary_ignore=2, uncert=uncert)
            # Optional per-frame uncertainty for AUE
            pixel_uncertainty = bndl_outputs.get("pixel_uncertainty", None) if self.aue_use_uncertainty else None
            
            # Adversarial training: unified AUE loss (supports both style-based and feature-based)
            if self.training and self.use_aue and (pixel_feat is not None) and not getattr(self, "_suppress_nested_aue", False):
                # Prefer gradient-carrying logits key when training
                pixel_logits_for_aue = bndl_outputs.get(
                    "pixel_logits",
                    bndl_outputs.get("masks_bndl_raw", bndl_outputs.get("mean_pixel_logits", None)),
                )
                
                # Unified AUE loss: automatically switches between style-based and feature-based
                aue_loss, aue_loss_dict = self.compute_aue_loss(
                    pixel_feat=pixel_feat,
                    pixel_uncertainty=pixel_uncertainty,
                    pixel_gt=pixel_gt_for_aue,
                    pixel_logits=pixel_logits_for_aue,
                    roi_z1=shared_z1,
                    roi_z2=shared_z2,
                    roi_w1=shared_w1,
                    roi_w2=shared_w2,
                    pixel_bndl_model=self.sam_mask_decoder.pixel_bndl if getattr(self.sam_mask_decoder, "pixel_bndl", None) is not None else None,
                    # Style-based AUE parameters (optional, used if use_style_aug=True)
                    img_batch=img_batch_for_style_aue,
                )
                bndl_outputs["aue_aux_loss"] = aue_loss
                bndl_outputs["aue_loss_dict"] = aue_loss_dict
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
            # Suppress nested AUE here to avoid noisy supervision in this "mask-as-output" path
            prev_suppress = getattr(self, "_suppress_nested_aue", False)
            self._suppress_nested_aue = True
            try:
                _, _, _, _, _, obj_ptr, _, *_ = self._forward_sam_heads(
                    backbone_features=backbone_features,
                    mask_inputs=self.mask_downsample(mask_inputs_float),
                    high_res_features=high_res_features,
                )
            finally:
                self._suppress_nested_aue = prev_suppress
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
        current_img_batch: torch.Tensor | None = None,
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
                img_batch_for_style_aue=current_img_batch,
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
