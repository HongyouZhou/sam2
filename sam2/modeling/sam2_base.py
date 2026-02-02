# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames
from sam2.modeling.aue.config import AUEConfig
from sam2.modeling.aue.module import AUEModule

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


@dataclass
class BNDLOutputs:
    """Container for BNDL model outputs."""

    pixel_feat: torch.Tensor  # [B, H, W, C]
    external_w: torch.Tensor | None  # [B, num_classes, C]
    pixel_logits: torch.Tensor | None  # [B, H, W, K]
    pixel_uncertainty: torch.Tensor | None = None  # [B, H, W]


import torch.nn as nn

# LoRA is now implemented via Hugging Face PEFT library for memory efficiency
# See _apply_lora_to_image_encoder() for the PEFT-based implementation


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
        # Whether to use BNDL for pixels (hyper_in only mode)
        use_bndl_for_pixels: bool = False,
        # Whether to use UR-ERN for pixels (mutually exclusive with BNDL)
        use_ur_ern_for_pixels: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
        # AUE options (Adversarial Uncertainty Estimation)
        use_aue: bool = False,
        # Alternating training: probability of running AUE branch per step (0~1)
        aue_probability: float = 0.5,
        # Number of adversarial samples in the bank
        aue_num_adversarial_samples: int = 32,
        # Whether to initialize adversarial samples from dataset
        aue_init_from_dataset: bool = False,
        # Diversity regularization for adversarial samples
        aue_diversity_loss_weight: float = 0.0,  # Weight for diversity regularization (0 = disabled)
        # Style Adversarial options (alternative to AUE, for domain generalization)
        use_style_adv: bool = False,
        style_adv_mode: str = "image_level",  # "image_level" or "feature_level"
        style_adv_epsilon: float = 2.0,  # Perturbation budget for style
        style_adv_use_gt_region_style: bool = False,  # Extract style only from GT region
        # Multi-object style attack control
        adv_use_multi_object: bool = False,  # true=attack with all objects, false=only current loss object
        adv_enable_background: bool = False,  # true=include background as K+1, false=objects only
        # Global-Local Mixed Style Adversarial
        style_adv_use_global_local_mix: bool = False,  # Enable global+local mixed style perturbation
        style_adv_global_epsilon: float = 1.5,  # Perturbation budget for global style
        style_adv_global_weight: float = 0.7,  # Weight of global style shift (0=local only, 1=global only)
        # Analytic uncertainty computation (from Weibull parameters, with gradients)
        aue_use_analytic_uncertainty: bool = True,  # Use analytic uncertainty (enables bidirectional optimization)
        # GCN-based multi-object style refinement
        style_adv_use_gcn: bool = False,
        style_adv_gcn_hidden_dim: int = 32,  # Deprecated, kept for backward compatibility
        style_adv_gcn_num_layers: int = 2,
        style_adv_gcn_edge_threshold: float = 0.3,
        style_adv_gcn_use_semantic_edges: bool = True,
        style_adv_gcn_use_background_edges: bool = False,
        style_adv_gcn_distance_threshold: float | None = 0.35,
        style_adv_gcn_use_boundary_distance: bool = False,  # Use boundary distance instead of centroid distance
        style_adv_gcn_use_visual_features: bool = False,  # Enable visual features in GCN for semantic edges
        style_adv_gcn_feature_dim: int = 256,  # Feature dimension (matches backbone_fpn[-1])
        style_adv_gcn_feature_sim_threshold: float = 0.5,  # Cosine similarity threshold for semantic edges
        # Deformation Adversarial options (DG-Font style, feature-level)
        use_deform_adv: bool = False,
        deform_adv_epsilon: float = 30.0,  # Deformation strength in pixels (image space)
        deform_adv_use_soft_composite: bool = True,  # Use soft compositing for overlaps
        deform_adv_temperature: float = 1.0,  # Temperature for soft compositing
        deform_adv_use_gcn: bool = False,  # GCN coordination for multi-object deformations
        deform_adv_gcn_num_layers: int = 2,  # Number of GCN layers
        deform_adv_num_deform_groups: int = 4,  # Number of deformable convolution groups
        # Feature-based deformation options
        deform_adv_init_from_memory_encoder: bool = True,  # Initialize from memory encoder
        deform_adv_freeze_encoder_components: bool = False,  # Freeze all encoder components (mask_encoder, img_feat_proj, fuser)
        deform_adv_zero_mean_offsets: bool = False,  # Remove global shift in flow fields
        deform_adv_local_offset_gain: float = 1.0,  # Boost local deformation after centering
        # Adversarial pipeline control
        adversarial_attack_order: list[str] | None = None,  # Order of attacks, e.g., ["deform", "style"]
        # Max number of objects (for AUE tensor allocation, matches dataset sampler)
        max_num_objects: int = 11,  # Default: 10 objects + 1 background
        # LoRA options (parameter-efficient fine-tuning)
        use_lora: bool = False,
        lora_mode: str = "standard",  # "standard", "dora", "convlora"
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_modules: list[str] | None = None,  # e.g., ["attn.qkv", "attn.proj", "mlp"]
        lora_expert_scales: list[float] | None = None,  # Conv-LoRA only
        lora_top_k: int = 1,  # Conv-LoRA only
        # Memory optimization: gradient checkpointing for AUE decoder
        use_aue_decoder_checkpoint: bool = True,  # Enable checkpoint for AUE branch decoder (reduces memory ~40%)
    ):
        super().__init__()
        self.use_aue_decoder_checkpoint = use_aue_decoder_checkpoint
        self.max_num_objects = max_num_objects

        # Helper function to instantiate modules from Hydra DictConfig
        def _maybe_instantiate(module, name: str):
            """Instantiate module from Hydra config if needed."""
            try:
                from omegaconf import DictConfig
                from hydra.utils import instantiate

                if isinstance(module, DictConfig):
                    logging.debug(f"Instantiating {name} from Hydra config...")
                    return instantiate(module)
            except ImportError:
                pass  # omegaconf not available
            return module

        # Part 1: the image backbone
        self.image_encoder = _maybe_instantiate(image_encoder, "image_encoder")

        # Apply LoRA if enabled (parameter-efficient fine-tuning)
        # Uses PEFT library for memory-efficient LoRA
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        if use_lora:
            self._apply_lora_to_image_encoder(lora_rank)
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
        self.memory_attention = _maybe_instantiate(memory_attention, "memory_attention")
        self.hidden_dim = self.image_encoder.neck.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = _maybe_instantiate(memory_encoder, "memory_encoder")
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(self.memory_encoder.out_proj, "weight"):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(torch.zeros(num_maskmem, 1, 1, self.mem_dim))
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

        # ========================================================================
        # AUE Configuration (Adversarial Uncertainty Estimation)
        # ========================================================================
        # Build AUEConfig from constructor kwargs (backward compatible with flat YAML)
        self.aue_config = AUEConfig.from_model_kwargs(
            use_aue=use_aue,
            aue_probability=aue_probability,
            aue_use_analytic_uncertainty=aue_use_analytic_uncertainty,
            adv_use_multi_object=adv_use_multi_object,
            adv_enable_background=adv_enable_background,
            aue_num_adversarial_samples=aue_num_adversarial_samples,
            aue_init_from_dataset=aue_init_from_dataset,
            aue_diversity_loss_weight=aue_diversity_loss_weight,
            max_num_objects=max_num_objects,
            # Style adversarial params
            use_style_adv=use_style_adv,
            style_adv_mode=style_adv_mode,
            style_adv_epsilon=style_adv_epsilon,
            style_adv_use_gt_region_style=style_adv_use_gt_region_style,
            style_adv_use_global_local_mix=style_adv_use_global_local_mix,
            style_adv_global_epsilon=style_adv_global_epsilon,
            style_adv_global_weight=style_adv_global_weight,
            style_adv_use_gcn=style_adv_use_gcn,
            style_adv_gcn_hidden_dim=style_adv_gcn_hidden_dim,
            style_adv_gcn_num_layers=style_adv_gcn_num_layers,
            style_adv_gcn_edge_threshold=style_adv_gcn_edge_threshold,
            style_adv_gcn_use_semantic_edges=style_adv_gcn_use_semantic_edges,
            style_adv_gcn_use_background_edges=style_adv_gcn_use_background_edges,
            style_adv_gcn_distance_threshold=style_adv_gcn_distance_threshold,
            style_adv_gcn_use_boundary_distance=style_adv_gcn_use_boundary_distance,
            style_adv_gcn_use_visual_features=style_adv_gcn_use_visual_features,
            style_adv_gcn_feature_dim=style_adv_gcn_feature_dim,
            style_adv_gcn_feature_sim_threshold=style_adv_gcn_feature_sim_threshold,
            # Deformation adversarial params
            use_deform_adv=use_deform_adv,
            deform_adv_epsilon=deform_adv_epsilon,
            deform_adv_use_soft_composite=deform_adv_use_soft_composite,
            deform_adv_temperature=deform_adv_temperature,
            deform_adv_use_gcn=deform_adv_use_gcn,
            deform_adv_gcn_num_layers=deform_adv_gcn_num_layers,
            deform_adv_num_deform_groups=deform_adv_num_deform_groups,
            deform_adv_init_from_memory_encoder=deform_adv_init_from_memory_encoder,
            deform_adv_freeze_encoder_components=deform_adv_freeze_encoder_components,
            deform_adv_zero_mean_offsets=deform_adv_zero_mean_offsets,
            deform_adv_local_offset_gain=deform_adv_local_offset_gain,
            # Attack ordering
            adversarial_attack_order=adversarial_attack_order,
        )

        # Apply AUE configuration to instance attributes (backward compatibility)
        self._apply_aue_config()

        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print("Image encoder compilation is enabled. First forward pass will be slow.")
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    def _apply_lora_to_image_encoder(self, lora_rank: int) -> None:
        """Apply LoRA to the Hiera image encoder using PEFT library.

        Uses Hugging Face PEFT for memory-efficient LoRA implementation.
        Following FS-SAM2's design: LoRA on image encoder with small rank.

        Args:
            lora_rank: Rank of the low-rank adaptation matrices
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("PEFT library is required for LoRA. Install with: pip install peft")

        # Configure LoRA for Hiera image encoder
        # Target 'qkv' and 'proj' layers in attention blocks
        peft_config = LoraConfig(
            inference_mode=False,
            r=lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["qkv", "proj"],  # Hiera attention layers
            bias="none",
        )

        # Apply PEFT to image encoder
        self.image_encoder = get_peft_model(self.image_encoder, peft_config)

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.image_encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.image_encoder.parameters())
        frozen_params = total_params - trainable_params

        logging.info(f"PEFT LoRA applied to image_encoder: rank={lora_rank}, alpha={self.lora_alpha}")
        logging.info(f"Image encoder: {trainable_params:,} trainable / {frozen_params:,} frozen / {total_params:,} total params")
        logging.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        """
        Override load_state_dict to handle PEFT LoRA parameters.

        When LoRA is enabled via PEFT, the model structure changes:
        - Vanilla: image_encoder.trunk.blocks.X.attn.qkv.weight
        - PEFT:    image_encoder.base_model.model.trunk.blocks.X.attn.qkv.base_layer.weight
                   + image_encoder.base_model.model.trunk.blocks.X.attn.qkv.lora_A.default.weight
                   + image_encoder.base_model.model.trunk.blocks.X.attn.qkv.lora_B.default.weight

        This method remaps vanilla checkpoint keys to the PEFT structure.
        """
        if self.use_lora:
            new_state_dict = {}
            remapped_count = 0

            for key, value in state_dict.items():
                # PEFT wraps the original module, so we need to remap keys
                # Vanilla: image_encoder.trunk.X -> PEFT: image_encoder.base_model.model.trunk.X.base_layer
                if key.startswith("image_encoder.") and not key.startswith("image_encoder.base_model."):
                    # Check if this is a LoRA target layer (qkv or proj)
                    if ".qkv." in key or ".proj." in key:
                        # Remap to PEFT's base_layer structure
                        # image_encoder.trunk.X.attn.qkv.weight -> image_encoder.base_model.model.trunk.X.attn.qkv.base_layer.weight
                        new_key = key.replace("image_encoder.", "image_encoder.base_model.model.")
                        # Insert .base_layer before .weight or .bias
                        if ".weight" in new_key:
                            new_key = new_key.replace(".weight", ".base_layer.weight")
                        elif ".bias" in new_key:
                            new_key = new_key.replace(".bias", ".base_layer.bias")
                        new_state_dict[new_key] = value
                        remapped_count += 1
                    else:
                        # Non-LoRA layers: just add base_model.model prefix
                        new_key = key.replace("image_encoder.", "image_encoder.base_model.model.")
                        new_state_dict[new_key] = value
                        remapped_count += 1
                else:
                    new_state_dict[key] = value

            if remapped_count > 0:
                logging.info(f"PEFT LoRA: Remapped {remapped_count} keys from vanilla checkpoint")

            state_dict = new_state_dict

        return super().load_state_dict(state_dict, strict=False)  # Use strict=False for PEFT compatibility

    def _apply_aue_config(self) -> None:
        """
        Apply AUEConfig to instance attributes for backward compatibility.

        This method maps the structured AUEConfig dataclass fields to the flat
        instance attributes that existing code expects. It also initializes
        the distribution matcher and builds AUE components.
        """
        cfg = self.aue_config

        # === Master flags ===
        self.use_aue = cfg.enabled
        self.aue_probability = cfg.probability  # Probability of running AUE per step (0~1)
        self.aue_use_analytic_uncertainty = cfg.use_analytic_uncertainty
        self.adv_use_multi_object = cfg.use_multi_object
        self.adv_enable_background = cfg.enable_background

        # === Legacy AUE options ===
        self.aue_num_adversarial_samples = cfg.num_adversarial_samples
        self.aue_init_from_dataset = cfg.init_from_dataset
        self.aue_diversity_loss_weight = cfg.diversity_loss_weight

        # === Style Adversarial ===
        self.use_style_adv = cfg.style.enabled
        self.style_adv_mode = cfg.style.mode
        self.style_adv_epsilon = cfg.style.epsilon
        self.style_adv_use_gt_region_style = cfg.style.use_gt_region_style
        self.style_adv_use_global_local_mix = cfg.style.use_global_local_mix
        self.style_adv_global_epsilon = cfg.style.global_epsilon
        self.style_adv_global_weight = cfg.style.global_weight
        # Style GCN
        self.style_adv_use_gcn = cfg.style.gcn.enabled
        self.style_adv_gcn_hidden_dim = cfg.style.gcn.hidden_dim
        self.style_adv_gcn_num_layers = cfg.style.gcn.num_layers
        self.style_adv_gcn_edge_threshold = cfg.style.gcn.edge_threshold
        self.style_adv_gcn_use_semantic_edges = cfg.style.gcn.use_semantic_edges
        self.style_adv_gcn_use_background_edges = cfg.style.gcn.use_background_edges
        self.style_adv_gcn_distance_threshold = cfg.style.gcn.distance_threshold
        self.style_adv_gcn_use_boundary_distance = cfg.style.gcn.use_boundary_distance
        self.style_adv_gcn_use_visual_features = cfg.style.gcn.use_visual_features
        self.style_adv_gcn_feature_dim = cfg.style.gcn.feature_dim
        self.style_adv_gcn_feature_sim_threshold = cfg.style.gcn.feature_sim_threshold
        self._latest_gcn_stats: dict[str, float] | None = None

        # === Deformation Adversarial ===
        self.use_deform_adv = cfg.deform.enabled
        self.deform_adv_epsilon = cfg.deform.epsilon
        self.deform_adv_use_soft_composite = cfg.deform.use_soft_composite
        self.deform_adv_temperature = cfg.deform.temperature
        self.deform_adv_use_gcn = cfg.deform.use_gcn
        self.deform_adv_gcn_num_layers = cfg.deform.gcn_num_layers
        self.deform_adv_num_deform_groups = cfg.deform.num_deform_groups
        self.deform_adv_init_from_memory_encoder = cfg.deform.init_from_memory_encoder
        self.deform_adv_freeze_encoder_components = cfg.deform.freeze_encoder_components
        self.deform_adv_zero_mean_offsets = cfg.deform.zero_mean_offsets
        self.deform_adv_local_offset_gain = cfg.deform.local_offset_gain

        # === Attack ordering ===
        self.adversarial_attack_order = cfg.attack_order

        # === Initialize components ===
        if self.use_aue:
            # Log AUE configuration
            uncertainty_type = "analytic (Weibull-based, with gradients)" if self.aue_use_analytic_uncertainty else "sampling (entropy-based, detached)"
            logging.info("Style-based AUE enabled (adversarial task loss computed inline in track_step)")
            logging.info(f"AUE uncertainty computation: {uncertainty_type}")
            logging.info(f"AUE alternating training probability: {self.aue_probability:.2f} (1.0 = every step, <1.0 = memory optimization)")
            logging.info("MMD calibration config: passed to loss modules via scratch.mmd_config")

            # Build adversarial attack components (only when AUE is enabled)
            if self.use_style_adv:
                self._build_style_adv_components()

            if self.use_deform_adv:
                self._build_deform_adv_components()

            # Initialize AUE module (composition pattern)
            self._aue_module = AUEModule(self)
            self._aue_module.initialize()
        else:
            self._aue_module = None

    def _build_style_adv_components(self) -> None:
        """
        Build style augmentation components for adversarial training.

        Style-based adversarial training using:
        - AdaIN: Style transfer for domain augmentation
        Style-based adversarial training using:
        - AdaIN: Style transfer for domain augmentation

        No pre-computed style bank needed - styles are extracted on-the-fly.
        """
        # AdaIN layer for style transfer
        from sam2.modeling.style_utils import AdaIN

        self.adain = AdaIN()

        # GCN module for multi-object style refinement
        if self.style_adv_use_gcn:
            if not self.adv_use_multi_object:
                raise ValueError("GCN requires adv_use_multi_object=True")
            if self.style_adv_use_global_local_mix:
                raise ValueError("GCN is incompatible with style_adv_use_global_local_mix")

            from sam2.modeling.style_gcn import AdversarialStyleGCN

            self.style_gcn = AdversarialStyleGCN(
                style_dim=6,
                feature_dim=self.style_adv_gcn_feature_dim if self.style_adv_gcn_use_visual_features else 0,
                num_layers=self.style_adv_gcn_num_layers,
            )
            logging.info(
                f"GCN module built: num_layers={self.style_adv_gcn_num_layers}, "
                f"use_visual_features={self.style_adv_gcn_use_visual_features}, "
                f"feature_dim={self.style_adv_gcn_feature_dim if self.style_adv_gcn_use_visual_features else 0}"
            )
        else:
            self.style_gcn = None

        # Build style augmenter (unified interface)
        from sam2.modeling.adversarial_augmentation import AdversarialAttacker

        self.style_attacker = AdversarialAttacker(
            mode=self.style_adv_mode,
            aug_type="style",
            epsilon=self.style_adv_epsilon,
            use_multi_object=self.adv_use_multi_object,
            use_gcn=self.style_adv_use_gcn,
            use_gt_region_style=self.style_adv_use_gt_region_style,
            enable_background=self.adv_enable_background,
            use_global_local_mix=self.style_adv_use_global_local_mix,
            global_epsilon=self.style_adv_global_epsilon,
            global_weight=self.style_adv_global_weight,
            num_objects=self.max_num_objects + (1 if self.adv_enable_background else 0),
        )

        logging.info(f"Style augmentation components built: mode={self.style_adv_mode}, epsilon={self.style_adv_epsilon}")

    def _build_deform_adv_components(self) -> None:
        """
        Build feature-level deformation augmentation components.

        Uses memory encoder components (MaskDownSampler, Fuser) for mask encoding.
        Produces dual outputs: feature-level deformation + image-level offsets.
        Optionally initializes weights from pretrained memory encoder.
        """
        from sam2.modeling.adversarial_augmentation import AdversarialAttacker

        # Build deformation augmenter (feature-level with image offset prediction)
        self.deform_attacker = AdversarialAttacker(
            mode="feature_level",
            aug_type="deformation",
            feature_dim=256,
            epsilon=self.deform_adv_epsilon,
            use_soft_composite=self.deform_adv_use_soft_composite,
            temperature=self.deform_adv_temperature,
            use_multi_object=self.adv_use_multi_object,
            use_gcn=self.deform_adv_use_gcn,
            gcn_num_layers=self.deform_adv_gcn_num_layers,
            num_deform_groups=self.deform_adv_num_deform_groups,
            init_from_memory_encoder=self.deform_adv_init_from_memory_encoder,
            freeze_encoder_components=self.deform_adv_freeze_encoder_components,
            image_size=self.image_size,  # Pass image_size for image-level offset prediction
            zero_mean_offsets=self.deform_adv_zero_mean_offsets,
            local_offset_gain=self.deform_adv_local_offset_gain,
        )

        # Load pretrained weights from memory encoder if enabled
        if self.deform_adv_init_from_memory_encoder:
            # Align deformation attacker with pretrained mask encoder weights
            self.deform_attacker.impl.load_memory_encoder_weights(self.memory_encoder)

        logging.info(
            f"Deformation augmentation built: "
            f"epsilon={self.deform_adv_epsilon}, "
            f"image_size={self.image_size}, "
            f"init_from_memory_encoder={self.deform_adv_init_from_memory_encoder}, "
            f"freeze_encoder_components={self.deform_adv_freeze_encoder_components}"
        )

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

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuningSee notebooks/video_predictor_example.ipynb for an inference example."
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
            use_ur_ern_for_pixels=self.use_ur_ern_for_pixels,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
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
        use_checkpoint=False,  # Enable gradient checkpointing for memory efficiency (used by AUE branch)
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

        # Apply gradient checkpointing to decoder if requested (for AUE memory optimization)
        if use_checkpoint and backbone_features.requires_grad:
            from torch.utils.checkpoint import checkpoint as checkpoint_fn

            def _decoder_forward(img_embed, img_pe, sparse_embed, dense_embed, multimask, high_res):
                return self.sam_mask_decoder(
                    image_embeddings=img_embed,
                    image_pe=img_pe,
                    sparse_prompt_embeddings=sparse_embed,
                    dense_prompt_embeddings=dense_embed,
                    multimask_output=multimask,
                    repeat_image=False,
                    high_res_features=high_res,
                )

            # Note: checkpoint requires tensors as inputs, so we pass high_res_features separately
            # and use use_reentrant=False for cleaner gradient handling
            mask_decoder_outputs = checkpoint_fn(
                _decoder_forward,
                backbone_features,
                self.sam_prompt_encoder.get_dense_pe(),
                sparse_embeddings,
                dense_embeddings,
                multimask_output,
                high_res_features,
                use_reentrant=False,
            )
        else:
            mask_decoder_outputs = self.sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=self.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=False,  # the image is already batched
                high_res_features=high_res_features,
            )

        # mask_decoder always returns 5 elements (masks, iou_pred, sam_tokens_out, object_score_logits, aux_outputs)
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
            aux_outputs,
        ) = mask_decoder_outputs

        # Check: if use_bndl_for_pixels=True, aux_outputs must contain valid BNDL data
        if self.use_bndl_for_pixels:
            if not isinstance(aux_outputs, dict):
                raise RuntimeError(f"SAM2Base._forward_sam_heads: use_bndl_for_pixels=True but aux_outputs is not a dict! type: {type(aux_outputs)}")
            if "bndl" not in aux_outputs:
                raise RuntimeError(f"SAM2Base._forward_sam_heads: use_bndl_for_pixels=True but aux_outputs does not contain 'bndl'! aux_outputs keys: {list(aux_outputs.keys())}")
            bndl_data = aux_outputs["bndl"]
            if not isinstance(bndl_data, dict):
                raise RuntimeError(f"SAM2Base._forward_sam_heads: aux_outputs['bndl'] is not a dict! type: {type(bndl_data)}")

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
            # Also expose selected channel's mask_tokens_out for single-channel UQ without revealing indices
            if "mask_tokens_out" in bndl_outputs and bndl_outputs["mask_tokens_out"] is not None:
                mask_tokens_out_full = bndl_outputs["mask_tokens_out"]  # [B, K, C'] (detached)
                if isinstance(mask_tokens_out_full, torch.Tensor) and mask_tokens_out_full.ndim == 3:
                    batch_inds = torch.arange(mask_tokens_out_full.size(0), device=device)
                    mask_tokens_out_selected = mask_tokens_out_full[batch_inds, selected_mask_index]  # [B, C']
                    bndl_outputs["mask_tokens_out_selected"] = mask_tokens_out_selected.detach()
            # Optional per-frame uncertainty for AUE (cached for later use in track_step level)
            pixel_uncertainty = bndl_outputs.get("pixel_uncertainty", None)

            # NOTE: AUE computation has been moved to track_step level (_compute_aue_with_refinement)
            # This enables:
            # 1. Adversarial branch to use refinement (iterative point sampling)
            # 2. Task loss computation on adversarial samples for OOD robustness
            # 3. Proper prompt generation for warped GT (deform attacks)
            #
            # The AUE-related data is still available in bndl_outputs for the track_step to use:
            # - pixel_feat, pixel_logits, mask_tokens_out, pixel_uncertainty

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
            obj_ptr = torch.zeros(mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device)
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

        aux_outputs = {}
        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            aux_outputs,
        )

    def forward_image(self, img_batch: torch.Tensor, use_checkpoint: bool = False):
        """Get the image feature on the input batch."""
        # Numerical stability hardening:
        # Run the image encoder (ViT + SDPA attention) in full FP32 to avoid AMP-related
        # backward NaNs that can appear when the backbone is unfrozen.
        from contextlib import nullcontext

        autocast_off = torch.cuda.amp.autocast(enabled=False) if getattr(img_batch, "is_cuda", False) else nullcontext()
        with autocast_off:
            if use_checkpoint and img_batch.requires_grad:
                backbone_out = self._forward_image_with_checkpoint(img_batch)
            else:
                backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        return backbone_out

    def _forward_image_with_checkpoint(self, img_batch: torch.Tensor) -> dict:
        """Forward image encoder under gradient checkpointing."""

        def _encode(input_tensor: torch.Tensor):
            from contextlib import nullcontext

            autocast_off = torch.cuda.amp.autocast(enabled=False) if getattr(input_tensor, "is_cuda", False) else nullcontext()
            with autocast_off:
                out = self.image_encoder(input_tensor)
            backbone = tuple(out["backbone_fpn"])
            pos = tuple(out["vision_pos_enc"])
            return (out["vision_features"], *backbone, *pos)

        outputs = checkpoint_fn(_encode, img_batch, use_reentrant=False)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        if len(outputs) < 3 or (len(outputs) - 1) % 2 != 0:
            raise RuntimeError(f"Unexpected number of tensors from image encoder checkpoint: {len(outputs)}")

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
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(frame_idx, cond_outputs, self.max_cond_frames_in_attn)
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
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {t: out for t, out in selected_cond_outputs.items() if (t >= frame_idx if track_in_reverse else t <= frame_idx)}
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        ((frame_idx - t) * tpos_sign_mul if self.use_signed_tpos_enc_to_obj_ptrs else abs(frame_idx - t)),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(t, unselected_cond_outputs.get(t, None))
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
                        obj_pos = torch.tensor(pos_list).to(device=device, non_blocking=True)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
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
            pred_masks_high_res = self._apply_non_overlapping_constraints(pred_masks_high_res)
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
            pix_feat,
            mask_for_mem,
            skip_mask_sigmoid=True,  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (1 - is_obj_appearing[..., None, None]) * self.no_obj_embed_spatial[..., None, None].expand(*maskmem_features.shape)

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
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [x.permute(1, 2, 0).view(x.size(1), x.size(2), *s) for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1], strict=True)]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            if self.use_bndl_for_pixels:
                raise RuntimeError(
                    f"SAM2Base._track_step (frame_idx={frame_idx}): "
                    f"use_bndl_for_pixels=True but use_mask_input_as_output_without_sam=True! "
                    f"This is incompatible - _use_mask_as_output does not compute BNDL outputs."
                )
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(pix_feat, high_res_features, mask_inputs)
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
        multimask_output = self.multimask_output_in_sam and (is_init_cond_frame or self.multimask_output_for_tracking) and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
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

    @contextmanager
    def _freeze_backbone_weights(self):
        """
        Context manager to temporarily freeze backbone weights.
        Allows gradients to flow THROUGH the backbone (to inputs) but not TO the backbone weights.
        """
        # Save original requires_grad states
        grads = {}
        for name, param in self.image_encoder.named_parameters():
            grads[name] = param.requires_grad
            param.requires_grad = False

        try:
            yield
        finally:
            # Restore original states
            for name, param in self.image_encoder.named_parameters():
                param.requires_grad = grads[name]
