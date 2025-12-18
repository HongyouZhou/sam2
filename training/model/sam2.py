# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import sample_box_points, get_next_point

from sam2.utils.misc import concat_points

from training.utils.data_utils import BatchedVideoDatapoint


class SAM2Train(SAM2Base):
    def __init__(
        self,
        image_encoder,
        memory_attention=None,
        memory_encoder=None,
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
        num_frames_to_correct_for_train=1,  # default: only iteratively sample on first frame
        num_frames_to_correct_for_eval=1,  # default: only iteratively sample on first frame
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
        # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
        # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
        # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
        # these are initial conditioning frames because as we track the video, more conditioning frames might be added
        # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
        num_init_cond_frames_for_train=1,  # default: only use the first frame as initial conditioning frame
        num_init_cond_frames_for_eval=1,  # default: only use the first frame as initial conditioning frame
        rand_init_cond_frames_for_train=True,  # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        rand_init_cond_frames_for_eval=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        # how many additional correction points to sample (on each frame selected to be corrected)
        # note that the first frame receives an initial input click (in addition to any correction clicks)
        num_correction_pt_per_frame=7,
        # method for point sampling during evaluation
        # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
        # default to "center" to be consistent with evaluation in the SAM paper
        pt_sampling_for_eval="center",
        # During training, we optionally allow sampling the correction points from GT regions
        # instead of the prediction error regions with a small probability. This might allow the
        # model to overfit less to the error regions in training datasets
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        forward_backbone_per_frame_for_eval=False,
        freeze_image_encoder=False,
        unfreeze_image_encoder_components=[],
        **kwargs,
    ):
        super().__init__(image_encoder, memory_attention, memory_encoder, **kwargs)
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval

        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)

        if freeze_image_encoder:
            # 1. Freeze everything by default
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            
            # 1.5. Auto-unfreeze LoRA parameters if use_lora=True (PEFT)
            if hasattr(self, 'use_lora') and self.use_lora:
                # LoRA parameters are those in LinearWithLoRA.lora or Conv2dWithLoRA.lora submodules
                lora_params_unfrozen = 0
                for name, module in self.image_encoder.named_modules():
                    # Check if this is a LoRA wrapper
                    if module.__class__.__name__ in ['LinearWithLoRA', 'Conv2dWithLoRA', 'LinearWithDoRA', 'Conv2dWithDoRA', 'LinearWithConvLoRA']:
                        # Unfreeze only the lora submodule, keep .linear/.conv frozen
                        if hasattr(module, 'lora'):
                            for p in module.lora.parameters():
                                p.requires_grad = True
                                lora_params_unfrozen += 1
                        # Unfreeze conv_lora submodule (for MoE Conv-LoRA)
                        if hasattr(module, 'conv_lora'):
                            for p in module.conv_lora.parameters():
                                p.requires_grad = True
                                lora_params_unfrozen += 1
                        # Unfreeze magnitude parameter for DoRA
                        if hasattr(module, 'm'):
                            module.m.requires_grad = True
                            lora_params_unfrozen += 1
                logging.info(f"Frozen image encoder, but unfrozen {lora_params_unfrozen} LoRA parameters for PEFT")
            
            # 2. Unfreeze specific components
            if "neck" in unfreeze_image_encoder_components:
                if hasattr(self.image_encoder, "neck"):
                    for p in self.image_encoder.neck.parameters():
                        p.requires_grad = True
                    logging.info("Unfrozen image encoder component: NECK")
                else:
                    logging.warning("Requested to unfreeze 'neck' but image_encoder has no 'neck' attribute.")

            if "trunk_last_stage" in unfreeze_image_encoder_components:
                if hasattr(self.image_encoder, "trunk") and hasattr(self.image_encoder.trunk, "blocks"):
                    # Hiera structure: blocks are in a ModuleList
                    # We need to find the blocks belonging to the last stage.
                    # Hiera stores stage_ends indices.
                    if hasattr(self.image_encoder.trunk, "stage_ends"):
                        last_stage_end = self.image_encoder.trunk.stage_ends[-1]
                        # The stage before the last one ends at stage_ends[-2]
                        # If there are 4 stages, stage_ends has 4 elements.
                        # Stage 3 (last) starts after stage_ends[2] + 1
                        if len(self.image_encoder.trunk.stage_ends) >= 2:
                            last_stage_start = self.image_encoder.trunk.stage_ends[-2] + 1
                        else:
                            last_stage_start = 0 # Fallback if only 1 stage
                        
                        # Unfreeze blocks from start to end
                        for i in range(last_stage_start, last_stage_end + 1):
                            for p in self.image_encoder.trunk.blocks[i].parameters():
                                p.requires_grad = True
                        logging.info(f"Unfrozen image encoder component: TRUNK_LAST_STAGE (blocks {last_stage_start}-{last_stage_end})")
                    else:
                         logging.warning("Requested to unfreeze 'trunk_last_stage' but trunk has no 'stage_ends'.")
                else:
                    logging.warning("Requested to unfreeze 'trunk_last_stage' but image_encoder structure is unknown.")

            # 3. Set modes
            # Set backbone to eval mode to disable DropPath and other training-only behaviors
            self.image_encoder.eval()
            
            # If we unfreeze something, we might want to keep those parts in train mode?
            # Usually for fine-tuning, we keep BN frozen (eval mode) but allow weights to update.
            # DropPath should probably be disabled for stability unless we have lots of data.
            # So keeping eval() mode is generally safer for partial fine-tuning.
            # However, if the user explicitly wants to train the neck, maybe they want BN updates?
            # For FPN neck, it usually has LayerNorm or GN, not BN. So eval/train mode matters less for stats,
            # but matters for Dropout/DropPath.
            # Let's keep it simple: Force eval() on everything to be safe (consistent with previous logic),
            # BUT if 'neck' is unfrozen, we might want to allow it to be in train mode if it has dropout?
            # The previous logic forced everything to eval. Let's stick to that for stability.
            # Wait, if we use `self.image_encoder.eval()`, it sets everything to eval.
            # If we want to train neck, we should probably set neck to train()?
            # Let's check Hiera FPN Neck. It has LayerNorm. No BN.
            # So eval() vs train() mainly affects Dropout (if any).
            # Let's stick to the safe bet: Global eval(), but unfreeze weights.
            
            # Prevent backbone from being set back to train mode by model.train()
            # We need to be careful: if we want to train neck, model.train() should ideally set neck to train.
            # But the monkey patch `self.image_encoder.train = lambda ...` prevents ANY part from going to train.
            
            # Revised Logic:
            # 1. Set global eval()
            self.image_encoder.eval()
            
            # 2. Define custom train function
            def custom_train(mode=True):
                # Always keep trunk in eval
                if hasattr(self.image_encoder, "trunk"):
                    self.image_encoder.trunk.eval()
                
                # If neck is unfrozen, allow it to follow mode
                if "neck" in unfreeze_image_encoder_components and hasattr(self.image_encoder, "neck"):
                    self.image_encoder.neck.train(mode)
                
                # If last stage is unfrozen, allow it to follow mode?
                # Hiera blocks have DropPath. Enabling DropPath (train mode) might be good for regularization
                # but risky for stability. Let's keep trunk fully eval for now to be safe.
                
                return self.image_encoder

            self.image_encoder.train = custom_train

        # If we are NOT freezing the encoder (joint training), we must ensure that
        # the base weights of LoRA layers (which are frozen by default upon injection)
        # are explicitly unfrozen.
        if not freeze_image_encoder and hasattr(self, 'use_lora') and self.use_lora:
            unfrozen_count = 0
            for name, module in self.image_encoder.named_modules():
                if module.__class__.__name__ in ['LinearWithLoRA', 'LinearWithDoRA', 'LinearWithConvLoRA']:
                     # Unfreeze the original linear layer
                     if hasattr(module, 'linear'):
                         for p in module.linear.parameters():
                             p.requires_grad = True
                             unfrozen_count += 1
                elif module.__class__.__name__ in ['Conv2dWithLoRA', 'Conv2dWithDoRA']:
                     # Unfreeze the original conv layer
                     if hasattr(module, 'conv'):
                         for p in module.conv.parameters():
                             p.requires_grad = True
                             unfrozen_count += 1
            if unfrozen_count > 0:
                logging.info(f"Joint Training: Unfrozen {unfrozen_count} base layers within LoRA modules.")

    def forward(self, input: BatchedVideoDatapoint):
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(input.flat_img_batch, use_checkpoint=True)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)

        return previous_stages_out

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image, use_checkpoint=True)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        """
        # Load the ground-truth masks on all frames (so that we can later
        # sample correction points from them)
        # gt_masks_per_frame = {
        #     stage_id: targets.segments.unsqueeze(1)  # [B, 1, H_im, W_im]
        #     for stage_id, targets in enumerate(input.find_targets)
        # }
        gt_masks_per_frame = {
            stage_id: masks.unsqueeze(1)  # [B, 1, H_im, W_im]
            for stage_id, masks in enumerate(input.masks)
        }
        # gt_masks_per_frame = input.masks.unsqueeze(2) # [T,B,1,H_im,W_im] keep everything in tensor form
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = input.num_frames
        backbone_out["num_frames"] = num_frames

        # Randomly decide whether to use point inputs or mask inputs
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        if num_frames == 1:
            # here we handle a special case for mixing video + SAM on image training,
            # where we force using point input for the SAM task on static images
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        assert num_init_cond_frames >= 1
        # (here `self.rng.random()` returns value in range 0.0 <= X < 1.0)
        use_pt_input = self.rng.random() < prob_to_use_pt_input
        if rand_init_cond_frames and num_init_cond_frames > 1:
            # randomly select 1 to `num_init_cond_frames` frames as initial conditioning frames
            num_init_cond_frames = self.rng.integers(
                1, num_init_cond_frames, endpoint=True
            )
        if (
            use_pt_input
            and rand_frames_to_correct
            and num_frames_to_correct > num_init_cond_frames
        ):
            # randomly select `num_init_cond_frames` to `num_frames_to_correct` frames to sample
            # correction clicks (only for the case of point input)
            num_frames_to_correct = self.rng.integers(
                num_init_cond_frames, num_frames_to_correct, endpoint=True
            )
        backbone_out["use_pt_input"] = use_pt_input

        # Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]  # starting frame
        else:
            # starting frame + randomly selected remaining frames (without replacement)
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        # Prepare mask or point inputs on initial conditioning frames
        backbone_out["mask_inputs_per_frame"] = {}  # {frame_idx: <input_masks>}
        backbone_out["point_inputs_per_frame"] = {}  # {frame_idx: <input_points>}
        
        # Construct multi-object masks grouped by (frame, video) for AUE
        # ============================================================
        # CONTROL: Toggle background mask support (from config)
        # ============================================================
        # Set style_aug_enable_background = True to include background as extra object
        # Set style_aug_enable_background = False for objects-only
        # 
        # When enabled:
        #   - K = max_num_objects + 1 (foreground objects + background)
        #   - Background = 1 - union(all objects)
        #   - Background participates in style attack
        #   - Visualization shows green dashed bbox for background
        # ============================================================
        ENABLE_BACKGROUND_MASK = self.style_adv_enable_background
        
        # Use max_num_objects from config (matches dataset sampler)
        # Add 1 slot for background if enabled
        max_num_objects = (self.max_num_objects + 1) if ENABLE_BACKGROUND_MASK else self.max_num_objects
        num_videos = input.num_videos
        H_mask, W_mask = input.masks.shape[-2:]
        
        # Initialize: [T, B_videos, K, H, W]
        # Force float32 for AUE masks (input.masks might be bool)
        mask_all_objs_for_aue = torch.zeros(
            num_frames, num_videos, max_num_objects, H_mask, W_mask,
            dtype=torch.float32, 
            device=input.masks.device
        )
        num_objs_per_video_frame = torch.zeros(
            num_frames, num_videos, 
            dtype=torch.int32, 
            device=input.masks.device
        )
        
        # Fill in masks for each (frame, video)
        for t in range(num_frames):
            frame_masks = input.masks[t]  # [O_t, H, W]
            frame_obj_to_img = input.obj_to_frame_idx[t]  # [O_t, 2]
            
            for video_idx in range(num_videos):
                obj_mask = (frame_obj_to_img[:, 1] == video_idx)
                obj_indices = obj_mask.nonzero(as_tuple=True)[0]
                K_actual = len(obj_indices)

                if ENABLE_BACKGROUND_MASK:
                    # Reserve final slot for background, fill foreground up to max_num_objects - 1
                    max_fg_slots = max_num_objects - 1
                    K_store = min(K_actual, max_fg_slots)

                    if K_store > 0:
                        video_frame_masks = frame_masks[obj_indices[:K_store]]
                        # Ensure float conversion when assigning bool masks to float tensor
                        mask_all_objs_for_aue[t, video_idx, :K_store] = video_frame_masks.float()
                        union_mask = video_frame_masks.float().sum(dim=0).clamp(0, 1)
                    else:
                        union_mask = torch.zeros(
                            H_mask, W_mask, dtype=mask_all_objs_for_aue.dtype, device=mask_all_objs_for_aue.device
                        )

                    background_mask = (1.0 - union_mask).clamp(0, 1)
                    background_slot = max_num_objects - 1
                    mask_all_objs_for_aue[t, video_idx, background_slot] = background_mask

                    # Total count = foreground (possibly zero) + background
                    num_objs_per_video_frame[t, video_idx] = K_store + 1
                else:
                    if K_actual == 0:
                        continue

                    video_frame_masks = frame_masks[obj_indices]
                    # No background: use all available slots for objects
                    K_store = min(K_actual, max_num_objects)
                    # Ensure float conversion when assigning bool masks to float tensor
                    mask_all_objs_for_aue[t, video_idx, :K_store] = video_frame_masks[:K_store].float()

                    # Total count = objects only
                    num_objs_per_video_frame[t, video_idx] = K_store
        
        backbone_out["mask_all_objs_for_aue"] = mask_all_objs_for_aue
        backbone_out["num_objs_per_video_frame"] = num_objs_per_video_frame
        
        for t in init_cond_frames:
            if not use_pt_input:
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
            else:
                # During training # P(box) = prob_to_use_pt_input * prob_to_use_box_input
                use_box_input = self.rng.random() < prob_to_use_box_input
                if use_box_input:
                    points, labels = sample_box_points(
                        gt_masks_per_frame[t],
                    )
                else:
                    # (here we only sample **one initial point** on initial conditioning frames from the
                    # ground-truth mask; we may sample more correction points on the fly)
                    points, labels = get_next_point(
                        gt_masks=gt_masks_per_frame[t],
                        pred_masks=None,
                        method=(
                            "uniform" if self.training else self.pt_sampling_for_eval
                        ),
                    )

                point_inputs = {"point_coords": points, "point_labels": labels}
                backbone_out["point_inputs_per_frame"][t] = point_inputs

        # Sample frames where we will add correction clicks on the fly
        # based on the error between prediction and ground-truth masks
        if not use_pt_input:
            # no correction points will be sampled when using mask inputs
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            assert num_frames_to_correct > num_init_cond_frames
            # initial cond frame + randomly selected remaining frames (without replacement)
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                init_cond_frames
                + self.rng.choice(
                    backbone_out["frames_not_in_init_cond"], extra_num, replace=False
                ).tolist()
            )
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt

        return backbone_out

    def forward_tracking(
        self, backbone_out, input: BatchedVideoDatapoint, return_dict=False
    ):
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)

        # Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        for stage_id in processing_order:
            # Save current frame's object mapping information (for track_step)
            self._current_obj_to_frame_idx = input.obj_to_frame_idx[stage_id]  # [O_t, 2]
            self._current_backbone_out = backbone_out  # Save backbone_out reference
            
            # Get the image features for the current frames
            # img_ids = input.find_inputs[stage_id].img_ids
            img_ids = input.flat_obj_to_img_idx[stage_id]
            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                (
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )

            # Get output masks based on this frame's prompts and previous memory
            # Extract current frame's image for style-based AUE
            if img_feats_already_computed:
                # If features already computed, we have all images in flat_img_batch
                current_img_batch = input.flat_img_batch[img_ids] if hasattr(input, 'flat_img_batch') else None
            else:
                # Images were computed on-the-fly in _prepare_backbone_features_per_frame
                current_img_batch = None  # Would need to store from earlier, skip for now
            
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
                current_img_batch=current_img_batch,
            )
            # Append the output, depending on whether it's a conditioning frame
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        if return_dict:
            return output_dict
        # turn `output_dict` into a list for loss function
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

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
        run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        frames_to_add_correction_pt=None,
        gt_masks=None,
        current_img_batch=None,  # Current frame images for style-based AUE
    ):
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
        
        # Construct multi-object masks: [O_t, K, H, W]
        # IMPORTANT: gt_masks should ONLY be used for prompt sampling (points/boxes)
        # AUE should use OTHER objects' masks, not the current object's GT
        # ============================================================
        # CONTROL: Toggle multi-object style attack (from config)
        # ============================================================
        # Set style_aug_use_multi_object = True to attack with all objects in the video
        # Set style_aug_use_multi_object = False to disable AUE (no pixel_gt_for_aue)
        # ============================================================
        USE_MULTI_OBJECT_AUE = self.style_adv_use_multi_object
        
        # DO NOT initialize pixel_gt_for_aue with gt_masks!
        # gt_masks is reserved for prompt sampling only
        pixel_gt_for_aue = None
        if self.use_aue and is_init_cond_frame:
            if USE_MULTI_OBJECT_AUE and hasattr(self, '_current_backbone_out'):
                backbone_out = self._current_backbone_out
                if "mask_all_objs_for_aue" in backbone_out:
                    mask_all_objs = backbone_out["mask_all_objs_for_aue"][frame_idx]  # [B_videos, K, H, W]
                    num_objs = backbone_out["num_objs_per_video_frame"][frame_idx]    # [B_videos]
                    obj_to_frame = self._current_obj_to_frame_idx  # [O_t, 2]
                    
                    # For each object, construct all object masks from its video
                    O_t = len(obj_to_frame)
                    K = mask_all_objs.shape[1]
                    H, W = mask_all_objs.shape[2:]
                    
                    multi_obj_masks = torch.zeros(
                        O_t, K, H, W,
                        dtype=mask_all_objs.dtype,
                        device=mask_all_objs.device
                    )
                    
                    for i in range(O_t):
                        video_idx = obj_to_frame[i, 1].item()
                        K_actual = num_objs[video_idx].item()
                        # Copy all non-empty masks from mask_all_objs (includes foreground + background if enabled)
                        # Background is at slot K-1 (e.g., index 10 when K=11), not necessarily in [:K_actual]
                        multi_obj_masks[i] = mask_all_objs[video_idx]
                    
                    # Use multi-object masks
                    pixel_gt_for_aue = multi_obj_masks
            elif not USE_MULTI_OBJECT_AUE and gt_masks is not None:
                # Single-object mode: use the current object's GT mask
                # gt_masks is [B, 1, H, W], which matches [B, K=1, H, W]
                pixel_gt_for_aue = gt_masks
                 
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
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
            pixel_gt_for_aue=pixel_gt_for_aue,
            current_img_batch=current_img_batch,
        )

        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            aux_outputs,
        ) = sam_outputs

        # 初始化多步辅助输出列表（统一为 aux_outputs）
        # Check: if use_bndl_for_pixels=True, aux_outputs must contain valid BNDL data
        if self.use_bndl_for_pixels:
            if aux_outputs is None:
                raise RuntimeError(
                    f"SAM2Train.track_step (frame_idx={frame_idx}): use_bndl_for_pixels=True but aux_outputs is None!"
                )
            if not isinstance(aux_outputs, dict):
                raise RuntimeError(
                    f"SAM2Train.track_step (frame_idx={frame_idx}): use_bndl_for_pixels=True but aux_outputs is not a dict! "
                    f"type: {type(aux_outputs)}"
                )
            if "bndl" not in aux_outputs:
                raise RuntimeError(
                    f"SAM2Train.track_step (frame_idx={frame_idx}): use_bndl_for_pixels=True but aux_outputs does not contain 'bndl'! "
                    f"aux_outputs keys: {list(aux_outputs.keys())}"
                )
            bndl_data = aux_outputs["bndl"]
            if not isinstance(bndl_data, dict):
                raise RuntimeError(
                    f"SAM2Train.track_step (frame_idx={frame_idx}): aux_outputs['bndl'] is not a dict! type: {type(bndl_data)}"
                )
        
        if aux_outputs is not None:
            # 将 GT 添加到 BNDL 命名空间（如果存在）
            bndl_ns = aux_outputs.get("bndl", None)
            if bndl_ns is not None and gt_masks is not None:
                bndl_ns["pixel_gt"] = gt_masks.detach().to(dtype=torch.float32)
            current_out["multistep_aux_outputs"] = [aux_outputs]
        else:
            current_out["multistep_aux_outputs"] = [None]

        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        # Optionally, sample correction points iteratively to correct the mask
        if frame_idx in frames_to_add_correction_pt:
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame,
                point_inputs,
                gt_masks,
                high_res_features,
                pix_feat,
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                object_score_logits,
                current_out,
                img_batch_for_aue=current_img_batch,
                pixel_gt_for_aue=pixel_gt_for_aue,
            )
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
                *_
            ) = final_sam_outputs

        # Use the final prediction (after all correction steps for output and eval)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

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

    def _iter_correct_pt_sampling(
        self,
        is_init_cond_frame,
        point_inputs,
        gt_masks,
        high_res_features,
        pix_feat_with_mem,
        low_res_multimasks,
        high_res_multimasks,
        ious,
        low_res_masks,
        high_res_masks,
        object_score_logits,
        current_out,
        img_batch_for_aue=None,
        pixel_gt_for_aue=None,
    ):

        assert gt_masks is not None
        all_pred_masks = [low_res_masks]
        all_pred_high_res_masks = [high_res_masks]
        all_pred_multimasks = [low_res_multimasks]
        all_pred_high_res_multimasks = [high_res_multimasks]
        all_pred_ious = [ious]
        all_point_inputs = [point_inputs]
        all_object_score_logits = [object_score_logits]
        # Collect aux outputs (BNDL/UR-ERN) if present
        all_aux_outputs = current_out.get("multistep_aux_outputs", [None])

        for _ in range(self.num_correction_pt_per_frame):
            # sample a new point from the error between prediction and ground-truth
            # (with a small probability, directly sample from GT masks instead of errors)
            if self.training and self.prob_to_sample_from_gt_for_train > 0:
                sample_from_gt = (
                    self.rng.random() < self.prob_to_sample_from_gt_for_train
                )
            else:
                sample_from_gt = False
            # if `pred_for_new_pt` is None, only GT masks will be used for point sampling
            pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
            new_points, new_labels = get_next_point(
                gt_masks=gt_masks,
                pred_masks=pred_for_new_pt,
                method="uniform" if self.training else self.pt_sampling_for_eval,
            )
            point_inputs = concat_points(point_inputs, new_points, new_labels)
            # Feed the mask logits of the previous SAM outputs in the next SAM decoder step.
            # For tracking, this means that when the user adds a correction click, we also feed
            # the tracking output mask logits along with the click as input to the SAM decoder.
            mask_inputs = low_res_masks
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            if self.use_act_ckpt_iterative_pt_sampling and not multimask_output:
                sam_outputs = torch.utils.checkpoint.checkpoint(
                    self._forward_sam_heads,
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    pixel_gt_for_aue=None,
                    img_batch_for_style_aue=None,
                    use_reentrant=False,
                )
            else:
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    pixel_gt_for_aue=None,
                    img_batch_for_style_aue=None,
                )

            # Unpack sam_outputs (always 8 elements now)
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                _,
                object_score_logits,
                aux_outputs,
            ) = sam_outputs
            
            # Check: if use_bndl_for_pixels=True, aux_outputs must contain valid BNDL data
            if self.use_bndl_for_pixels:
                if aux_outputs is None:
                    raise RuntimeError(
                        f"SAM2Train._iter_correct_pt_sampling (step {_}, frame_idx={current_out.get('frame_idx', 'unknown')}): "
                        f"use_bndl_for_pixels=True but aux_outputs is None!"
                    )
                if not isinstance(aux_outputs, dict):
                    raise RuntimeError(
                        f"SAM2Train._iter_correct_pt_sampling (step {_}, frame_idx={current_out.get('frame_idx', 'unknown')}): "
                        f"use_bndl_for_pixels=True but aux_outputs is not a dict! type: {type(aux_outputs)}"
                    )
                if "bndl" not in aux_outputs:
                    raise RuntimeError(
                        f"SAM2Train._iter_correct_pt_sampling (step {_}, frame_idx={current_out.get('frame_idx', 'unknown')}): "
                        f"use_bndl_for_pixels=True but aux_outputs does not contain 'bndl'! "
                        f"aux_outputs keys: {list(aux_outputs.keys())}"
                    )
                bndl_data = aux_outputs["bndl"]
                if not isinstance(bndl_data, dict):
                    raise RuntimeError(
                        f"SAM2Train._iter_correct_pt_sampling (step {_}, frame_idx={current_out.get('frame_idx', 'unknown')}): "
                        f"aux_outputs['bndl'] is not a dict! type: {type(bndl_data)}"
                    )
            
            all_aux_outputs.append(aux_outputs)
            all_pred_masks.append(low_res_masks)
            all_pred_high_res_masks.append(high_res_masks)
            all_pred_multimasks.append(low_res_multimasks)
            all_pred_high_res_multimasks.append(high_res_multimasks)
            all_pred_ious.append(ious)
            all_point_inputs.append(point_inputs)
            all_object_score_logits.append(object_score_logits)

        # Concatenate the masks along channel (to compute losses on all of them,
        # using `MultiStepIteractiveMasks`)
        current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
        current_out["multistep_pred_masks_high_res"] = torch.cat(
            all_pred_high_res_masks, dim=1
        )
        current_out["multistep_pred_multimasks"] = all_pred_multimasks
        current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
        current_out["multistep_pred_ious"] = all_pred_ious
        current_out["multistep_point_inputs"] = all_point_inputs
        current_out["multistep_object_score_logits"] = all_object_score_logits
        
        # 统一保存 aux_outputs（包含 BNDL/UR-ERN 等）
        current_out["multistep_aux_outputs"] = all_aux_outputs

        return point_inputs, sam_outputs
