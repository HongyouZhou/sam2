# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


@dataclass
class AUEVisualizationData:
    """Container for AUE visualization data."""
    original_images: torch.Tensor  # [N, 3, H, W] CPU
    adversarial_images: torch.Tensor  # [N, 3, H, W] CPU
    original_styles: torch.Tensor | None  # [N, K, 6] CPU
    adversarial_styles: torch.Tensor | None  # [N, K, 6] CPU
    num_objects: int  # K
    all_object_masks: torch.Tensor | None  # [N, K, H, W] CPU
    all_bboxes: torch.Tensor | None  # [N, K, 4] CPU
    area_ratios: torch.Tensor | None  # [N, K] CPU
    epsilon_weights: torch.Tensor | None  # [N, K] CPU
    combined_mask_for_loss: torch.Tensor | None  # [N, 1, H, W] CPU
    # Deformation visualization data
    deform_offsets: torch.Tensor | None  # [N, K, 2, H, W] CPU (deformation offset fields)
    warped_images: torch.Tensor | None  # [N, 3, H, W] CPU (soft-composited warped images from offsets)
    # Style visualization data (already included above via original/adversarial images and styles)


@dataclass
class AUEAdversarialBatch:
    """Container for adversarial generation results."""
    adv_images: torch.Tensor | None  # [M, 3, H, W]
    adv_gts: torch.Tensor | None  # [M, H, W]
    adv_prompts: torch.Tensor | None  # [M, 4]
    visualization_data: AUEVisualizationData | None = None


@dataclass
class BNDLOutputs:
    """Container for BNDL model outputs."""
    pixel_feat: torch.Tensor  # [B, H, W, C]
    external_w: torch.Tensor | None  # [B, num_classes, C]
    pixel_logits: torch.Tensor | None  # [B, H, W, K]
    pixel_uncertainty: torch.Tensor | None = None  # [B, H, W]


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
        # Style Augmentation options (alternative to AUE, for domain generalization)
        use_style_aug: bool = False,
        style_aug_mode: str = "image_level",  # "image_level" or "feature_level"
        style_aug_pgd_steps: int = 5,
        style_aug_pgd_step_size: float = 0.1,
        style_aug_pgd_epsilon: float = 2.0,
        style_aug_use_gt_region_style: bool = False,  # Extract style only from GT region
        # Multi-object style attack control
        style_aug_use_multi_object: bool = False,  # true=attack with all objects, false=only current loss object
        style_aug_enable_background: bool = False,  # true=include background as K+1, false=objects only
        # Global-Local Mixed Style Augmentation
        style_aug_use_global_local_mix: bool = False,  # Enable global+local mixed style perturbation
        style_aug_global_epsilon: float = 1.5,  # Perturbation budget for global style
        style_aug_global_weight: float = 0.7,  # Weight of global style shift (0=local only, 1=global only)
        # Distribution matching configuration for AUE calibration loss
        aue_dist_matching_config: dict | None = None,  # Nested config dict
        # Analytic uncertainty computation (from Weibull parameters, with gradients)
        aue_use_analytic_uncertainty: bool = True,  # Use analytic uncertainty (enables bidirectional optimization)
        # GCN-based multi-object style refinement
        style_aug_use_gcn: bool = False,
        style_aug_gcn_hidden_dim: int = 32,  # Deprecated, kept for backward compatibility
        style_aug_gcn_num_layers: int = 2,
        style_aug_gcn_edge_threshold: float = 0.3,
        style_aug_gcn_use_semantic_edges: bool = True,
        style_aug_gcn_use_background_edges: bool = False,
        style_aug_gcn_distance_threshold: float | None = 0.35,
        style_aug_gcn_use_boundary_distance: bool = False,  # Use boundary distance instead of centroid distance
        style_aug_gcn_use_visual_features: bool = False,  # Enable visual features in GCN for semantic edges
        style_aug_gcn_feature_dim: int = 256,  # Feature dimension (matches backbone_fpn[-1])
        style_aug_gcn_feature_sim_threshold: float = 0.5,  # Cosine similarity threshold for semantic edges
        # Deformation Augmentation options (DG-Font style, feature-level)
        use_deform_aug: bool = False,
        deform_aug_epsilon: float = 30.0,  # Deformation strength in pixels (image space)
        deform_aug_pgd_steps: int = 3,  # PGD steps for deformation (placeholder, not used yet)
        deform_aug_use_soft_composite: bool = True,  # Use soft compositing for overlaps
        deform_aug_temperature: float = 1.0,  # Temperature for soft compositing
        deform_aug_use_gcn: bool = False,  # GCN coordination for multi-object deformations
        deform_aug_gcn_num_layers: int = 2,  # Number of GCN layers
        deform_aug_num_deform_groups: int = 4,  # Number of deformable convolution groups
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
        # Distribution matching configuration (nested config only)
        self.aue_dist_matching_config = self._parse_dist_matching_config(aue_dist_matching_config)
        
        # Extract commonly used values for convenience
        self.aue_calibration_method = self.aue_dist_matching_config['method']
        self.aue_use_patches = self.aue_dist_matching_config['use_patches']
        self.aue_patch_size = self.aue_dist_matching_config['patch_size']
        
        # Analytic uncertainty configuration
        self.aue_use_analytic_uncertainty = aue_use_analytic_uncertainty
        
        # Initialize distribution matcher for calibration loss
        if self.use_aue:
            from sam2.modeling.distribution_matching import DistributionMatcher
            
            cfg = self.aue_dist_matching_config
            
            self.distribution_matcher = DistributionMatcher(
                method=cfg['method'],
                patch_size=cfg['patch_size'],
                kernel=cfg.get('mmd', {}).get('kernel', 'rbf'),
                bandwidth=cfg.get('mmd', {}).get('bandwidth', 0.1),
                cka_use_linear_kernel=cfg.get('cka', {}).get('use_linear_kernel', True),
                cka_use_minibatch=cfg.get('cka', {}).get('use_minibatch', True),
                cka_minibatch_size=cfg.get('cka', {}).get('minibatch_size', 512),
            )
            self._build_aue_components()

        # Style Augmentation components (alternative to AUE)
        self.use_style_aug = bool(use_style_aug)
        self.style_aug_mode = str(style_aug_mode)
        self.style_aug_pgd_steps = int(style_aug_pgd_steps)
        self.style_aug_pgd_step_size = float(style_aug_pgd_step_size)
        self.style_aug_pgd_epsilon = float(style_aug_pgd_epsilon)
        self.style_aug_use_gt_region_style = bool(style_aug_use_gt_region_style)
        # Multi-object style attack control
        self.style_aug_use_multi_object = bool(style_aug_use_multi_object)
        self.style_aug_enable_background = bool(style_aug_enable_background)
        # Global-Local Mixed Style parameters
        self.style_aug_use_global_local_mix = bool(style_aug_use_global_local_mix)
        self.style_aug_global_epsilon = float(style_aug_global_epsilon)
        self.style_aug_global_weight = float(style_aug_global_weight)
        # GCN parameters
        self.style_aug_use_gcn = bool(style_aug_use_gcn)
        self.style_aug_gcn_hidden_dim = int(style_aug_gcn_hidden_dim)  # Deprecated
        self.style_aug_gcn_num_layers = int(style_aug_gcn_num_layers)
        self.style_aug_gcn_edge_threshold = float(style_aug_gcn_edge_threshold)
        self.style_aug_gcn_use_semantic_edges = bool(style_aug_gcn_use_semantic_edges)
        self.style_aug_gcn_use_background_edges = bool(style_aug_gcn_use_background_edges)
        self.style_aug_gcn_distance_threshold = (
            float(style_aug_gcn_distance_threshold)
            if style_aug_gcn_distance_threshold is not None
            else None
        )
        self.style_aug_gcn_use_boundary_distance = bool(style_aug_gcn_use_boundary_distance)
        self.style_aug_gcn_use_visual_features = bool(style_aug_gcn_use_visual_features)
        self.style_aug_gcn_feature_dim = int(style_aug_gcn_feature_dim)
        self.style_aug_gcn_feature_sim_threshold = float(style_aug_gcn_feature_sim_threshold)
        self._latest_gcn_stats: dict[str, float] | None = None
        if self.use_style_aug:
            self._build_style_aug_components()
        
        # Deformation Augmentation components (DG-Font style)
        self.use_deform_aug = bool(use_deform_aug)
        self.deform_aug_epsilon = float(deform_aug_epsilon)
        self.deform_aug_pgd_steps = int(deform_aug_pgd_steps)
        self.deform_aug_use_soft_composite = bool(deform_aug_use_soft_composite)
        self.deform_aug_temperature = float(deform_aug_temperature)
        self.deform_aug_use_gcn = bool(deform_aug_use_gcn)
        self.deform_aug_gcn_num_layers = int(deform_aug_gcn_num_layers)
        self.deform_aug_num_deform_groups = int(deform_aug_num_deform_groups)
        if self.use_deform_aug:
            self._build_deform_aug_components()

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

    def _parse_dist_matching_config(self, config: dict | None) -> dict:
        """
        Parse and validate distribution matching config.
        
        Expected config format:
        {
            'method': 'cka',  # 'mmd' | 'cka' | 'gram'
            'use_patches': True,
            'patch_size': 32,
            'mmd': {'kernel': 'rbf', 'bandwidth': 0.1, ...},
            'cka': {'use_linear_kernel': True, 'use_minibatch': True, ...},
            'gram': {'center': True, 'normalize': True, ...}
        }
        
        Args:
            config: Nested config dict (required)
        
        Returns:
            Parsed config dict with all defaults filled
        """
        if config is None:
            # Provide sensible defaults if no config specified
            config = {}
        
        # Make a copy to avoid mutating input
        config = config.copy()
        
        # Set top-level defaults
        config.setdefault('method', 'mmd')
        config.setdefault('use_patches', True)
        config.setdefault('patch_size', 16)
        
        # Set MMD defaults
        if 'mmd' not in config:
            config['mmd'] = {}
        config['mmd'].setdefault('kernel', 'rbf')
        config['mmd'].setdefault('bandwidth', 0.1)
        config['mmd'].setdefault('batch_size', 256)
        config['mmd'].setdefault('n_batches', 10)
        
        # Set CKA defaults
        if 'cka' not in config:
            config['cka'] = {}
        config['cka'].setdefault('use_linear_kernel', True)
        config['cka'].setdefault('use_minibatch', True)
        config['cka'].setdefault('minibatch_size', 512)
        
        # Set Gram defaults
        if 'gram' not in config:
            config['gram'] = {}
        config['gram'].setdefault('center', True)
        config['gram'].setdefault('normalize', True)
        
        # Validate method
        valid_methods = ['mmd', 'cka', 'gram']
        if config['method'] not in valid_methods:
            raise ValueError(
                f"Invalid distribution matching method: {config['method']}. "
                f"Must be one of {valid_methods}"
            )
        
        logging.info(
            f"Distribution matching config: "
            f"method={config['method']}, "
            f"use_patches={config['use_patches']}, "
            f"patch_size={config['patch_size']}"
        )
        
        return config
    
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
        
        # GCN module for multi-object style refinement
        if self.style_aug_use_gcn:
            if not self.style_aug_use_multi_object:
                raise ValueError("GCN requires style_aug_use_multi_object=True")
            if self.style_aug_use_global_local_mix:
                raise ValueError("GCN is incompatible with style_aug_use_global_local_mix")
            
            from sam2.modeling.style_gcn import AdversarialStyleGCN
            self.style_gcn = AdversarialStyleGCN(
                style_dim=6,
                feature_dim=self.style_aug_gcn_feature_dim if self.style_aug_gcn_use_visual_features else 0,
                num_layers=self.style_aug_gcn_num_layers,
            )
            logging.info(
                f"GCN module built: num_layers={self.style_aug_gcn_num_layers}, "
                f"use_visual_features={self.style_aug_gcn_use_visual_features}, "
                f"feature_dim={self.style_aug_gcn_feature_dim if self.style_aug_gcn_use_visual_features else 0}"
            )
        else:
            self.style_gcn = None
        
        # Build style augmenter (unified interface)
        from sam2.modeling.adversarial_augmentation import AdversarialAugmenter
        self.style_augmenter = AdversarialAugmenter(
            mode=self.style_aug_mode,
            aug_type="style",
            pgd_steps=self.style_aug_pgd_steps,
            epsilon=self.style_aug_pgd_epsilon,
            step_size=self.style_aug_pgd_step_size,
            use_multi_object=self.style_aug_use_multi_object,
            use_gcn=self.style_aug_use_gcn,
            use_gt_region_style=self.style_aug_use_gt_region_style,
            enable_background=self.style_aug_enable_background,
            use_global_local_mix=self.style_aug_use_global_local_mix,
            global_epsilon=self.style_aug_global_epsilon,
            global_weight=self.style_aug_global_weight,
        )
        
        logging.info(
            f"Style augmentation components built: "
            f"pgd_steps={self.style_aug_pgd_steps}, "
            f"epsilon={self.style_aug_pgd_epsilon}"
        )
    
    def _build_deform_aug_components(self) -> None:
        """
        Build deformation augmentation components for adversarial training.
        
        Deformation-based adversarial training using:
        - Deformable convolution: Spatial warping in feature space (DG-Font style)
        - Soft compositing: Handle multi-object overlaps gracefully
        - Optional GCN: Coordinate deformations across objects
        
        Key advantage: No extra backbone forward pass (uses clean features directly).
        """
        from sam2.modeling.adversarial_augmentation import AdversarialAugmenter
        
        # Build deformation augmenter (image-level for simplified GT alignment)
        self.deform_augmenter = AdversarialAugmenter(
            mode="image_level",  # Changed from feature_level to image_level
            aug_type="deformation",
            image_channels=3,  # RGB image
            epsilon=self.deform_aug_epsilon,
            use_soft_composite=self.deform_aug_use_soft_composite,
        )
        
        logging.info(
            f"Deformation augmentation components built (image-level): "
            f"epsilon={self.deform_aug_epsilon} pixels, "
            f"use_soft_composite={self.deform_aug_use_soft_composite}"
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
        
        dist_type = "patch-based" if self.aue_use_patches else "global"
        uncertainty_type = "analytic (Weibull-based, with gradients)" if self.aue_use_analytic_uncertainty else "sampling (entropy-based, detached)"
        logging.info("Style-based AUE enabled with SAM loss for PGD attacks (matching training config)")
        logging.info(f"AUE calibration loss: {self.aue_calibration_method.upper()} ({dist_type}, patch_size={self.aue_patch_size if self.aue_use_patches else 'N/A'})")
        logging.info(f"AUE uncertainty computation: {uncertainty_type}")

    # Image-space constraint methods removed (not needed for feature-space AUE)

    def _apply_offset_to_image(
        self,
        image: torch.Tensor,
        offset_field: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply spatial offset field to warp an image using grid_sample.
        
        Args:
            image: [B, 3, H, W] Input image
            offset_field: [B, 2, H, W] Offset field (Δx, Δy)
        
        Returns:
            warped_image: [B, 3, H, W] Warped image
        """
        B, _, H, W = image.shape
        device = image.device
        
        # Create identity grid (standard sampling positions)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Add offset (offset order is [Δx, Δy])
        offset_x = offset_field[:, 0, :, :]  # [B, H, W]
        offset_y = offset_field[:, 1, :, :]  # [B, H, W]
        
        # New sampling positions
        sampling_x = x_grid.unsqueeze(0) + offset_x  # [B, H, W]
        sampling_y = y_grid.unsqueeze(0) + offset_y  # [B, H, W]
        
        # Normalize to [-1, 1] (required by grid_sample)
        sampling_x_norm = 2.0 * sampling_x / (W - 1) - 1.0
        sampling_y_norm = 2.0 * sampling_y / (H - 1) - 1.0
        
        # Stack into grid format [B, H, W, 2] (last dimension is x, y)
        sampling_grid = torch.stack([sampling_x_norm, sampling_y_norm], dim=-1)
        
        # Apply warping using grid_sample
        warped_image = F.grid_sample(
            image,
            sampling_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped_image
    
    def _prepare_aue_visualization_data(
        self,
        img_batch: torch.Tensor,
        adv_images: torch.Tensor,
        pixel_gt: torch.Tensor | None,
        num_vis_samples: int,
        original_styles: torch.Tensor | None = None,
        adv_styles: torch.Tensor | None = None,
        deform_offsets: torch.Tensor | None = None,
        warped_images: torch.Tensor | None = None,
    ) -> AUEVisualizationData:
        """
        Prepare visualization data for Style AUE multi-object training.
        
        This method computes all necessary visualization data from masks on-demand,
        avoiding extra memory overhead during training. Bounding boxes, area ratios,
        and epsilon weights are computed here only when visualization is enabled.
        
        Args:
            img_batch: [B, 3, H, W] original images (normalized)
            adv_images: [M, 3, H, W] adversarial images (normalized)
            pixel_gt: [B, K, H, W] ground truth masks for K objects (optional)
            num_vis_samples: number of samples to visualize
        
        Returns:
            AUEVisualizationData containing all visualization tensors
        """
        from sam2.modeling.aue_utils import masks_to_boxes
        
        # Save images (move to CPU to free GPU memory)
        original_images_cpu = img_batch[:num_vis_samples].detach().cpu()
        adversarial_images_cpu = adv_images[:num_vis_samples].detach().cpu()
        
        # Process styles if provided (handle both GPU and CPU tensors)
        # Styles are [B, K, 6] where K is number of objects
        if original_styles is not None:
            if original_styles.is_cuda:
                original_styles_cpu = original_styles[:num_vis_samples].detach().cpu()
            else:
                original_styles_cpu = original_styles[:num_vis_samples].detach()
        else:
            original_styles_cpu = None
            
        if adv_styles is not None:
            if adv_styles.is_cuda:
                adv_styles_cpu = adv_styles[:num_vis_samples].detach().cpu()
            else:
                adv_styles_cpu = adv_styles[:num_vis_samples].detach()
        else:
            adv_styles_cpu = None
        
        # Add multi-object visualization data (masks and bboxes)
        if pixel_gt is not None and pixel_gt.ndim == 4:
            K = pixel_gt.shape[1]
            
            # Save multi-object masks for visualization
            all_object_masks = pixel_gt[:num_vis_samples].detach().cpu()  # [N, K, H, W]
            
            # Compute bounding boxes from masks (lightweight operation, <1ms per sample)
            all_bboxes_list = []
            all_area_ratios_list = []
            
            for i in range(num_vis_samples):
                sample_masks = pixel_gt[i]  # [K, H, W]
                
                # Compute bboxes
                sample_bboxes = masks_to_boxes(sample_masks)  # [K, 4]
                all_bboxes_list.append(sample_bboxes)
                
                # Compute area ratios for each object
                total_pixels = sample_masks.shape[1] * sample_masks.shape[2]
                object_areas = sample_masks.sum(dim=[1, 2])  # [K]
                area_ratio = object_areas.float() / total_pixels  # [K]
                all_area_ratios_list.append(area_ratio)
            
            all_bboxes = torch.stack(all_bboxes_list).detach().cpu()  # [N, K, 4]
            area_ratios = torch.stack(all_area_ratios_list).detach().cpu()  # [N, K]
            
            # Compute epsilon weights based on area ratios (larger objects get higher weight)
            all_epsilon_weights_list = []
            for area_ratio in all_area_ratios_list:
                max_area = area_ratio.max()
                if max_area > 0:
                    epsilon_weights = 0.5 + 0.5 * (area_ratio / max_area)
                else:
                    epsilon_weights = torch.ones_like(area_ratio)
                all_epsilon_weights_list.append(epsilon_weights)
            
            epsilon_weights = torch.stack(all_epsilon_weights_list).detach().cpu()  # [N, K]
            
            # Also save the combined mask used for loss
            if pixel_gt.shape[1] > 1:
                combined_mask = pixel_gt.sum(dim=1, keepdim=True).clamp(0, 1)  # [B, 1, H, W]
            else:
                combined_mask = pixel_gt
            combined_mask_for_loss = combined_mask[:num_vis_samples].detach().cpu()
        else:
            K = 1
            all_object_masks = None
            all_bboxes = None
            area_ratios = None
            epsilon_weights = None
            combined_mask_for_loss = None
         
        if deform_offsets is not None:
            if deform_offsets.is_cuda:
                deform_offsets_cpu = deform_offsets[:num_vis_samples].detach().cpu()
            else:
                deform_offsets_cpu = deform_offsets[:num_vis_samples].detach()
        else:
            deform_offsets_cpu = None
        
        # Process warped images if provided
        if warped_images is not None:
            if warped_images.is_cuda:
                warped_images_cpu = warped_images[:num_vis_samples].detach().cpu()
            else:
                warped_images_cpu = warped_images[:num_vis_samples].detach()
        else:
            warped_images_cpu = None
        
        return AUEVisualizationData(
            original_images=original_images_cpu,
            adversarial_images=adversarial_images_cpu,
            original_styles=original_styles_cpu,
            adversarial_styles=adv_styles_cpu,
            num_objects=K,
            all_object_masks=all_object_masks,
            all_bboxes=all_bboxes,
            area_ratios=area_ratios,
            epsilon_weights=epsilon_weights,
            combined_mask_for_loss=combined_mask_for_loss,
            deform_offsets=deform_offsets_cpu,
            warped_images=warped_images_cpu,
        )

    def _cleanup_augmentation_results(self, results_to_cleanup: list) -> None:
        """
        Clean up augmentation results to free GPU memory.
        
        Args:
            results_to_cleanup: List of augmentation result objects with release_intermediate() method.
                               Results are cleaned up in reverse order (LIFO).
        """
        for result in reversed(results_to_cleanup):
            result.release_intermediate()
    
    def _extract_bndl_outputs(
        self,
        aux_outputs: dict,
        pixel_bndl_model,
        compute_logits: bool = True,
        compute_uncertainty: bool = False,
        uq_sample_num: int = 20,
    ) -> BNDLOutputs | None:
        """
        Extract and process BNDL outputs from aux_outputs dict.
        
        Centralizes the pattern of extracting pixel_feat, external_w, logits, uncertainty.
        
        Args:
            aux_outputs: Auxiliary outputs containing BNDL dict
            pixel_bndl_model: BNDL model for computing logits/uncertainty
            compute_logits: Whether to compute logits if not present
            compute_uncertainty: Whether to compute sampling-based uncertainty
            uq_sample_num: Number of samples for uncertainty estimation
        
        Returns:
            BNDLOutputs containing pixel_feat, external_w, pixel_logits, pixel_uncertainty
            or None if extraction fails
        """
        bndl = aux_outputs.get("bndl", {})
        pixel_feat = bndl.get("pixel_feat_grad", bndl.get("pixel_feat"))
        
        if pixel_feat is None:
            return None
        
        # Extract external weights
        external_w = None
        if pixel_bndl_model is not None and not pixel_bndl_model.enable_global_sparse:
            hyper_in = bndl.get("hyper_in")
            if hyper_in is not None:
                external_w = hyper_in
            else:
                B = pixel_feat.shape[0]
                external_w = pixel_bndl_model.linear.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Compute logits if requested
        pixel_logits = None
        if compute_logits:
            pixel_logits = bndl.get("pixel_logits", bndl.get("masks_bndl_raw", None))
            if pixel_logits is None and pixel_bndl_model is not None:
                pixel_logits, *_ = pixel_bndl_model(
                    pixel_feat, force_sample=False, external_pre_out_w=external_w
                )
        
        # Compute uncertainty if requested
        pixel_uncertainty = None
        if compute_uncertainty:
            with torch.no_grad():
                from BNDL.BNDL_upload.ViT_Sparse.utils.bndl import pixel_entropy_uncertainty
                pixel_uncertainty = pixel_entropy_uncertainty(
                    pixel_bndl_model, pixel_feat, external_w, uq_sample_num, per_channel=False
                )
        
        return BNDLOutputs(
            pixel_feat=pixel_feat,
            external_w=external_w,
            pixel_logits=pixel_logits,
            pixel_uncertainty=pixel_uncertainty,
        )
    
    def _prepare_gt_for_loss(
        self,
        pixel_gt: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        Prepare GT masks: combine multi-object and resize to target resolution.
        
        Args:
            pixel_gt: [B, K, H, W] ground truth masks
            target_size: (H_feat, W_feat) target spatial size
        
        Returns:
            [B, H_feat, W_feat] combined and resized GT
        """
        # Combine K objects with sum(dim=1).clamp(0, 1) if K > 1
        if pixel_gt.shape[1] > 1:
            pixel_gt_combined = pixel_gt.sum(dim=1, keepdim=True).clamp(0, 1)
        else:
            pixel_gt_combined = pixel_gt
        
        # Resize to target_size using F.interpolate
        pixel_gt_resized = F.interpolate(
            pixel_gt_combined.float(), size=target_size, mode='nearest'
        ).squeeze(1)
        
        return pixel_gt_resized
    
    def _compute_augmented_calibration_loss(
        self,
        aux_outputs: dict,
        pixel_gt: torch.Tensor,
        pixel_bndl_model,
        uq_sample_num: int,
    ) -> torch.Tensor:
        """
        Compute calibration loss for augmented features (deformation or style).
        
        This is a helper method to avoid code duplication between style and deformation branches.
        
        Args:
            aux_outputs: Auxiliary outputs from _forward_sam_heads containing BNDL outputs
            pixel_gt: [B, K, H, W] Ground truth masks
            pixel_bndl_model: BNDL model for uncertainty estimation
            uq_sample_num: Number of samples for uncertainty estimation
        
        Returns:
            Calibration loss scalar
        """
        # 1. Extract BNDL outputs
        bndl_outputs = self._extract_bndl_outputs(
            aux_outputs, pixel_bndl_model,
            compute_logits=True, compute_uncertainty=True, uq_sample_num=uq_sample_num
        )
        if bndl_outputs is None:
            logging.warning("No pixel features in augmented outputs, returning zero loss")
            return torch.tensor(0.0, device=pixel_gt.device, dtype=torch.float32)
        
        # 2. Prepare GT
        if bndl_outputs.pixel_logits is not None:
            H, W = bndl_outputs.pixel_logits.shape[1:3]
        else:
            H, W = bndl_outputs.pixel_feat.shape[1:3]
        pixel_gt_prepared = self._prepare_gt_for_loss(pixel_gt, target_size=(H, W))
        
        # 3. Compute calibration loss
        return self._compute_uncertainty_calibration_loss(bndl_outputs, pixel_gt_prepared, pixel_bndl_model)
    
    def _apply_adversarial_augmentation_pipeline(
        self,
        img_batch: torch.Tensor,
        pixel_gt: torch.Tensor,
        backbone_features: torch.Tensor,
        high_res_features: list[torch.Tensor] | None,
        uq_sample_num: int,
        enable_vis: bool,
        vis_refs: dict,
    ) -> dict:
        """
        Apply adversarial augmentation pipeline: Deform → Style → Forward → Loss.
        
        Returns:
            dict with keys:
                - calibration_loss_adversarial: torch.Tensor
                - vis_refs: dict (optional, if visualization is enabled)
        """
        current_features = backbone_features
        current_high_res = high_res_features
        
        # Step 1: Generate deformation offsets
        deform_offsets = None
        deform_result = None
        if self.use_deform_aug and hasattr(self, 'deform_augmenter'):
            deform_result = self.deform_augmenter.apply(
                img_batch=img_batch,
                clean_features=current_features,
                clean_high_res=current_high_res,
                pixel_gt=pixel_gt,
                model=self,
            )
            deform_offsets = deform_result.deformation_offsets  # [B, K, 2, H, W]
            
            if enable_vis:
                vis_refs['deform_offsets'] = deform_offsets.detach().cpu()
        
        # Step 2: Apply deformation to images and GT masks
        augmented_img, warped_pixel_gt = self._apply_deformation_to_images(
            img_batch, pixel_gt, deform_offsets, enable_vis, vis_refs
        )
        
        # Update vis_refs with warped GT for correct visualization
        if enable_vis and deform_offsets is not None:
            vis_refs['pixel_gt'] = warped_pixel_gt.detach().cpu()
        
        # If deformation changed the images, update features to match deformed images
        if deform_result is not None and augmented_img is not img_batch:
            # Forward deformed images through backbone to get corresponding features
            backbone_out = self.forward_image(augmented_img, use_checkpoint=True)
            current_features = backbone_out['backbone_fpn'][-1]
            if self.use_high_res_features_in_sam:
                current_high_res = [
                    backbone_out['backbone_fpn'][0],
                    backbone_out['backbone_fpn'][1]
                ]
        
        # Step 3: Apply style augmentation (unified interface - like deform)
        style_result = None
        if self.use_style_aug and hasattr(self, 'style_augmenter'):
            augmented_img_detached = augmented_img.detach() if augmented_img is not img_batch else augmented_img
            style_result = self.style_augmenter.apply(
                img_batch=augmented_img_detached,
                clean_features=current_features,  # Now uses features corresponding to deformed images
                clean_high_res=current_high_res,
                pixel_gt=warped_pixel_gt,
                model=self,
            )
            
            # Update features and images from style result
            augmented_img = style_result.intermediate_images
            current_features = style_result.features
            if style_result.high_res_features is not None:
                current_high_res = style_result.high_res_features
            pixel_gt = warped_pixel_gt
            
            # Save visualization data
            if enable_vis:
                vis_refs['styled_images'] = augmented_img.detach().cpu()
                if style_result.original_styles is not None:
                    vis_refs['original_styles'] = style_result.original_styles
                if style_result.adversarial_styles is not None:
                    vis_refs['adversarial_styles'] = style_result.adversarial_styles
        
        # Step 4: Compute adversarial loss (no forward pass needed - style_result already has features)
        calibration_loss_adversarial = torch.tensor(0.0, device=img_batch.device, dtype=img_batch.dtype)
        if current_features is not backbone_features:
            prev_suppress = getattr(self, "_suppress_nested_aue", False)
            self._suppress_nested_aue = True
            try:
                *_, adv_aux_outputs = self._forward_sam_heads(
                    backbone_features=current_features,
                    high_res_features=current_high_res,
                    pixel_gt_for_aue=None,
                    multimask_output=False,
                )
            finally:
                self._suppress_nested_aue = prev_suppress
            
            pixel_bndl_model = None
            if hasattr(self, 'sam_mask_decoder') and hasattr(self.sam_mask_decoder, 'pixel_bndl'):
                pixel_bndl_model = self.sam_mask_decoder.pixel_bndl
            
            calibration_loss_adversarial = self._compute_augmented_calibration_loss(
                aux_outputs=adv_aux_outputs,
                pixel_gt=pixel_gt,
                pixel_bndl_model=pixel_bndl_model,
                uq_sample_num=uq_sample_num,
            )
            
            del adv_aux_outputs
        
        # Cleanup augmentation results
        if deform_result is not None:
            self._cleanup_augmentation_results([deform_result])
        if style_result is not None:
            self._cleanup_augmentation_results([style_result])
        
        return {
            'calibration_loss_adversarial': calibration_loss_adversarial,
            'vis_refs': vis_refs if enable_vis else {},
        }
    
    def _apply_deformation_to_images(
        self,
        img_batch: torch.Tensor,
        pixel_gt: torch.Tensor,
        deform_offsets: torch.Tensor | None,
        enable_vis: bool,
        vis_refs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply deformation offsets to images and GT masks (vectorized)."""
        if deform_offsets is None:
            return img_batch, pixel_gt.clone()
        
        B, K, _, H_img, W_img = deform_offsets.shape
        warped_pixel_gt = pixel_gt.clone()
        
        # Identify valid objects (non-empty, non-background)
        masks_float = pixel_gt.float()
        mask_areas = masks_float.sum(dim=(2, 3))
        is_empty = (mask_areas.sum(dim=0) == 0)
        mask_area_ratio = mask_areas / (H_img * W_img)
        is_bg_per_sample = (mask_area_ratio > 0.5)
        is_bg = torch.zeros(K, dtype=torch.bool, device=img_batch.device)
        is_bg[-1] = is_bg_per_sample[:, -1].all()
        valid_objects = ~(is_empty | is_bg)
        valid_indices = torch.where(valid_objects)[0]
        
        if len(valid_indices) == 0:
            return img_batch, warped_pixel_gt
        
        # Initialize: start with original image, then selectively replace deformed objects
        # Only remove objects that will be deformed (keep non-deformed objects in place)
        valid_masks = masks_float[:, valid_indices, :, :]  # [B, K_valid, H, W]
        valid_masks_union = valid_masks.sum(dim=1, keepdim=True).clamp(0, 1)  # [B, 1, H, W]
        
        # Start with image where valid objects are removed (keep other objects + background)
        augmented_img = img_batch * (1 - valid_masks_union)
        
        # Composite warped objects sequentially (forward order: later objects overwrite earlier ones)
        # This avoids storing all warped images, reducing memory from O(K) to O(1)
        for idx_pos, k_idx in enumerate(valid_indices):
            k_idx_scalar = k_idx.item()
            offset_k = deform_offsets[:, k_idx_scalar, :, :, :]  # [B, 2, H, W]
            mask_k = masks_float[:, k_idx_scalar, :, :].unsqueeze(1)  # [B, 1, H, W]
            
            # Warp image and mask
            warped_img_k = self._apply_offset_to_image(img_batch, offset_k)
            warped_mask_k = self._apply_offset_to_image(mask_k, offset_k)
            
            # Update GT
            warped_pixel_gt[:, k_idx_scalar, :, :] = (warped_mask_k.squeeze(1) > 0.5).float()
            
            # Composite: later objects overwrite earlier ones (simple overlay)
            augmented_img = augmented_img * (1 - warped_mask_k) + warped_img_k * warped_mask_k
            
            # Cleanup immediately (no storage needed)
            del offset_k, mask_k, warped_img_k, warped_mask_k
        
        # Save for visualization
        if enable_vis:
            vis_refs['warped_images'] = augmented_img.detach().cpu()
        
        # Cleanup
        del valid_masks, valid_masks_union
        
        return augmented_img, warped_pixel_gt
    
    def compute_aue_loss(
        self,
        pixel_feat: torch.Tensor,
        pixel_uncertainty: torch.Tensor | None = None,
        pixel_gt: torch.Tensor | None = None,
        pixel_logits: torch.Tensor | None = None,
        adversarial_sample_M: int | None = 1, 
        pixel_bndl_model=None,
        uq_sample_num: int = 8,
        # Style-based AUE parameters
        img_batch: torch.Tensor | None = None,
        backbone_features: torch.Tensor | None = None,  # [B, C, H, W] detached backbone features for GCN
        high_res_features: list[torch.Tensor] | None = None,  # High-res features for SAM decoder
        external_pre_out_w: torch.Tensor | None = None,  # [B, C', K] for hyper_in (analytic uncertainty)
    ) -> torch.Tensor:
        B, H, W, _ = pixel_feat.shape
        device = pixel_feat.device
        dtype = pixel_feat.dtype
        
        # Validate high_res_features when use_high_res_features_in_sam is True
        if self.use_high_res_features_in_sam and high_res_features is None:
            raise ValueError(
                "use_high_res_features_in_sam=True requires high_res_features to be provided, "
                "but got None. Please ensure high_res_features is passed to compute_aue_loss."
            )
        
        # Early exit if no GT is available (AUE requires GT for calibration)
        if pixel_gt is None:
            logging.debug("AUE: No pixel_gt provided, skipping AUE loss")
            return torch.tensor(0.0, device=device, dtype=dtype), {}

        # Compute positive sample ratio (strict shape checks; logits must be provided)
        if pixel_logits is None:
            raise ValueError("AUE expects non-null pixel_logits of shape [B,H,W,K]")
        # Expect logits in channels-last format: [B, Hf, Wf, K]
        if not (pixel_logits.ndim == 4 and pixel_logits.shape[-1] >= 1):
            raise ValueError(f"AUE expects pixel_logits of shape [B,H,W,K], got {tuple(pixel_logits.shape)}")
        H_feat, W_feat = int(pixel_logits.shape[1]), int(pixel_logits.shape[2])

        # Optional GT: support both single-object [B,1,H,W] and multi-object [B,K,H,W]
        pixel_gt_resized = None
        if pixel_gt is not None:
            if not (pixel_gt.ndim == 4 and pixel_gt.shape[1] >= 1):
                raise ValueError(f"AUE expects pixel_gt of shape [B,K,H,W], got {tuple(pixel_gt.shape)}")
            
            # For multi-object (K>1), combine all objects for calibration loss
            if pixel_gt.shape[1] > 1:
                # Combine multiple objects into single mask: [B, K, H, W] → [B, 1, H, W]
                pixel_gt_combined = pixel_gt.sum(dim=1, keepdim=True).clamp(0, 1)
            else:
                # Single object: use as is
                pixel_gt_combined = pixel_gt
            
            # Resize to feature map resolution
            pixel_gt_resized = F.interpolate(pixel_gt_combined.float(), size=(H_feat, W_feat), mode='nearest').squeeze(1)

        calibration_loss_clean = self._compute_uncertainty_calibration_loss(
            pixel_logits=pixel_logits,
            pixel_uncertainty=pixel_uncertainty,
            pixel_gt=pixel_gt_resized,
            spatial_hw=(H_feat, W_feat),
            batch_size=B,
            device=device,
            dtype=dtype,
            pixel_feat=pixel_feat,
            pixel_bndl_model=pixel_bndl_model,
            external_pre_out_w=external_pre_out_w,
        )
 
        # Adversarial augmentation branch (串行: 形变 → 风格)
        calibration_loss_adversarial = torch.tensor(0.0, device=device, dtype=dtype)
        vis_data = None  # Initialize outside try block for visualization
        
        if (self.use_deform_aug or self.use_style_aug) and backbone_features is not None:
            # ========================================================================
            # Adversarial Augmentation Pipeline (串行: 形变 → 风格)
            # ========================================================================
            vis_refs = {}  # Collect lightweight visualization data (CPU tensors)
            enable_vis = getattr(self, '_enable_style_visualization', False)
            
            if enable_vis:
                # Save original image and GT for visualization (move to CPU immediately)
                vis_refs['img_batch'] = img_batch.detach().cpu()
                vis_refs['pixel_gt'] = pixel_gt.detach().cpu()
            
            # Apply adversarial augmentations and compute loss
            augmentation_result = self._apply_adversarial_augmentation_pipeline(
                img_batch=img_batch,
                pixel_gt=pixel_gt,
                backbone_features=backbone_features,
                high_res_features=high_res_features,
                uq_sample_num=uq_sample_num,
                enable_vis=enable_vis,
                vis_refs=vis_refs,
            )
            
            calibration_loss_adversarial = augmentation_result['calibration_loss_adversarial']
            assert calibration_loss_adversarial is not None
            if 'vis_refs' in augmentation_result:
                vis_refs.update(augmentation_result['vis_refs'])
        
        # ========================================================================
        # Prepare visualization data (AFTER cleanup, using CPU tensors)
        # ========================================================================
        # All visualization data has been moved to CPU during computation,
        # so this can safely happen after GPU memory cleanup without conflicts.
        vis_data = None
        if vis_refs:
            with torch.no_grad():
                # Use styled images if available, otherwise use original images
                adv_images_for_vis = vis_refs.get('styled_images', vis_refs['img_batch'])
                
                vis_data = self._prepare_aue_visualization_data(
                    img_batch=vis_refs['img_batch'],  # Original image (before deformation)
                    adv_images=adv_images_for_vis,
                    pixel_gt=vis_refs['pixel_gt'],
                    num_vis_samples=min(4, vis_refs['img_batch'].shape[0]),
                    original_styles=vis_refs.get('original_styles'),
                    adv_styles=vis_refs.get('adversarial_styles'),
                    deform_offsets=vis_refs.get('deform_offsets'),
                    warped_images=vis_refs.get('warped_images'),
                )
        
        # ========================================================================
        # Assemble final loss dictionary
        # ========================================================================
        loss_dict = {}
        
        loss_dict['calibration_loss_clean'] = calibration_loss_clean
        loss_dict['calibration_loss_adversarial'] = calibration_loss_adversarial
        
        # Save visualization data if available (prepared after cleanup)
        # Note: This contains both style and deformation visualization data
        if vis_data is not None:
            loss_dict['aue_visualization'] = vis_data
        
        loss = calibration_loss_clean + calibration_loss_adversarial
        loss_dict['total_loss'] = loss

        return loss, loss_dict

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
        bndl_outputs: BNDLOutputs,
        pixel_gt: torch.Tensor | None,
        pixel_bndl_model=None,
    ) -> torch.Tensor:
        """Compute uncertainty calibration loss via distribution matching.
        
        Theory: For zero-shot robustness, uncertainty distribution should match
        error distribution using Maximum Mean Discrepancy (MMD).
        
        Loss = MMD(P_U, P_Error) + 0.3 * MSE(U, Error)
        
        Key innovation: Uses analytic uncertainty (from Weibull parameters) to enable
        bidirectional optimization - both uncertainty and error are optimized to align.
        
        Gradient flow:
        - Analytic mode: MMD(uncertainty, error) → gradients to BOTH BNDL and encoder
        - Sampling mode: MMD(uncertainty.detach(), error) → gradients only to encoder
        
        References:
        - Gretton et al. (2012): "A Kernel Two-Sample Test"
        - Long et al. (2015): "Learning Transferable Features with DAN"
        
        Args:
            bndl_outputs: BNDLOutputs containing pixel_feat, pixel_logits, external_w, pixel_uncertainty
            pixel_gt: [B, H, W] ground truth masks (already combined and resized)
            pixel_bndl_model: BNDL model (required for analytic uncertainty)
        
        Returns:
            calibration_loss: MMD-based distribution matching loss
        """
        # Extract fields from bndl_outputs
        pixel_logits = bndl_outputs.pixel_logits
        pixel_feat = bndl_outputs.pixel_feat
        external_pre_out_w = bndl_outputs.external_w
        pixel_uncertainty = bndl_outputs.pixel_uncertainty
        
        # Extract device and dtype from pixel_feat
        device = pixel_feat.device
        dtype = pixel_feat.dtype
        
        if pixel_logits is None:
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        
        # 1. Compute prediction error [B, H, W] in [0, 1]
        error = self._compute_prediction_error(
            pixel_logits=pixel_logits,
            pixel_gt=pixel_gt,
        )
        
        # 2. Get uncertainty [B, H, W] in [0, 1]
        # Choose between analytic (with gradients) or sampling (detached)
        use_analytic = getattr(self, 'aue_use_analytic_uncertainty', True)
        
        if use_analytic and pixel_feat is not None and pixel_bndl_model is not None:
            # ✅ Analytic uncertainty (preserves gradients to BNDL)
            from sam2.modeling.bndl_utils import pixel_weibull_to_entropy_uncertainty
            
            uncertainty = pixel_weibull_to_entropy_uncertainty(
                pixel_bndl_model=pixel_bndl_model,
                pixel_feat=pixel_feat,
                external_pre_out_w=external_pre_out_w,
                per_channel=False,
            )  # [B, H, W] with gradients!
            
            # Clamp to [0, 1] for numerical stability
            uncertainty = uncertainty.clamp(0.0, 1.0)
            
        elif pixel_uncertainty is not None:
            # Sampling-based uncertainty (provided externally, typically detached)
            uncertainty = pixel_uncertainty.clamp(0.0, 1.0)
        else:
            # Fallback: use 1 - confidence
            confidence = self._aue_compute_confidence(pixel_logits, pixel_gt)
            uncertainty = (1.0 - confidence).clamp(0.0, 1.0)
        
        # Guard against NaN/Inf
        if not torch.isfinite(uncertainty).all() or not torch.isfinite(error).all():
            raise RuntimeError("Uncertainty calibration: inputs contain NaN/Inf")
        
        # 3. Distribution matching loss (primary: MMD/CKA/Gram)
        # Gradients flow through error to encoder/decoder (and optionally through uncertainty to BNDL)
        dist_loss = self.distribution_matcher.compute_loss(
            uncertainty=uncertainty,
            error=error,
            use_patches=self.aue_use_patches,
        )
        
        # 4. MSE loss (regularization: point-wise alignment)
        mse_loss = F.mse_loss(uncertainty, error, reduction='mean')
        
        # 5. Combine losses
        total_loss = 1.0 * dist_loss + 0.3 * mse_loss
        
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
            # Optional per-frame uncertainty for AUE
            pixel_uncertainty = bndl_outputs.get("pixel_uncertainty", None) if self.aue_use_uncertainty else None
            
            # Adversarial training: unified AUE loss (supports both style-based and feature-based)
            if self.training and self.use_aue and (pixel_feat is not None) and not getattr(self, "_suppress_nested_aue", False):
                # Prefer gradient-carrying logits key when training
                pixel_logits_for_aue = bndl_outputs.get(
                    "pixel_logits",
                    bndl_outputs.get("masks_bndl_raw", bndl_outputs.get("mean_pixel_logits", None)),
                )
                
                # Extract external_pre_out_w (hyper_in) for analytic uncertainty computation
                external_w_for_aue = bndl_outputs.get('hyper_in', None)
                assert external_w_for_aue is not None
                
                # Unified AUE loss: automatically switches between style-based and feature-based
                aue_loss, aue_loss_dict = self.compute_aue_loss(
                    pixel_feat=pixel_feat,
                    pixel_uncertainty=pixel_uncertainty,
                    pixel_gt=pixel_gt_for_aue,
                    pixel_logits=pixel_logits_for_aue,
                    pixel_bndl_model=self.sam_mask_decoder.pixel_bndl if getattr(self.sam_mask_decoder, "pixel_bndl", None) is not None else None,
                    # Style-based AUE parameters (optional, used if use_style_aug=True)
                    img_batch=img_batch_for_style_aue,
                    backbone_features=backbone_features.detach() if backbone_features is not None else None,
                    high_res_features=high_res_features,  # Pass high_res_features for deform augmentation
                    external_pre_out_w=external_w_for_aue,
                )
                bndl_outputs["aue_aux_loss"] = aue_loss
                bndl_outputs["aue_loss_dict"] = aue_loss_dict
                
                # After AUE computation, check if GCN stats were generated and add them
                gcn_stats = getattr(self, "_latest_gcn_stats", None)
                if gcn_stats and isinstance(bndl_outputs, dict):
                    bndl_outputs["gcn_stats"] = gcn_stats
                    logging.debug(f"Added GCN stats to bndl_outputs: {gcn_stats}")
                    self._latest_gcn_stats = None
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

    def forward_image(self, img_batch: torch.Tensor, use_checkpoint: bool = False):
        """Get the image feature on the input batch."""
        if use_checkpoint and img_batch.requires_grad:
            backbone_out = self._forward_image_with_checkpoint(img_batch)
        else:
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

    def _forward_image_with_checkpoint(self, img_batch: torch.Tensor) -> dict:
        """Forward image encoder under gradient checkpointing."""
        def _encode(input_tensor: torch.Tensor):
            out = self.image_encoder(input_tensor)
            backbone = tuple(out["backbone_fpn"])
            pos = tuple(out["vision_pos_enc"])
            return (out["vision_features"], *backbone, *pos)

        outputs = checkpoint_fn(_encode, img_batch, use_reentrant=False)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        if len(outputs) < 3 or (len(outputs) - 1) % 2 != 0:
            raise RuntimeError(
                f"Unexpected number of tensors from image encoder checkpoint: {len(outputs)}"
            )

        num_levels = (len(outputs) - 1) // 2

        vision_features = outputs[0]
        idx = 1
        backbone_fpn = [outputs[idx + i] for i in range(num_levels)]
        idx += num_levels
        vision_pos_enc = [outputs[idx + i] for i in range(num_levels)]

        return {
            "vision_features": vision_features,
            "backbone_fpn": backbone_fpn,
            "vision_pos_enc": vision_pos_enc,
        }

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
