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


def _log_branch_consistency(
    clean_gt: torch.Tensor,
    adv_gt: torch.Tensor,
    clean_features: torch.Tensor | None,
    adv_features: torch.Tensor | None,
    clean_prompts: dict | None,
    adv_prompts: dict | None,
    tag: str = "",
) -> None:
    """Log numerical consistency metrics between clean and adversarial branches.

    Call this periodically during training to verify that both branches
    are operating on equivalent data. Differences indicate potential bugs.
    """
    import logging

    logger = logging.getLogger()  # Use root logger to match setup_logging configuration
    if not logger.isEnabledFor(logging.DEBUG):
        return

    lines = [f"=== Branch Consistency Check {tag} ==="]

    # GT Mask comparison
    if clean_gt is not None and adv_gt is not None:
        clean_shape = tuple(clean_gt.shape)
        adv_shape = tuple(adv_gt.shape)
        shape_match = clean_shape == adv_shape

        clean_sum = clean_gt.float().sum().item()
        adv_sum = adv_gt.float().sum().item()
        sum_diff = abs(clean_sum - adv_sum)

        clean_range = (clean_gt.min().item(), clean_gt.max().item())
        adv_range = (adv_gt.min().item(), adv_gt.max().item())

        lines.append(f"GT Shape: clean={clean_shape}, adv={adv_shape}, match={shape_match}")
        lines.append(f"GT Sum: clean={clean_sum:.2f}, adv={adv_sum:.2f}, diff={sum_diff:.2f}")
        lines.append(f"GT Range: clean={clean_range}, adv={adv_range}")

        if shape_match:
            # Check pixel-wise difference
            diff = (clean_gt.float() - adv_gt.float()).abs()
            diff_mean = diff.mean().item()
            diff_max = diff.max().item()
            lines.append(f"GT Diff: mean={diff_mean:.6f}, max={diff_max:.6f}")

    # Feature comparison
    if clean_features is not None and adv_features is not None:
        clean_f_range = (clean_features.min().item(), clean_features.max().item())
        adv_f_range = (adv_features.min().item(), adv_features.max().item())
        clean_f_mean = clean_features.mean().item()
        adv_f_mean = adv_features.mean().item()
        clean_f_std = clean_features.std().item()
        adv_f_std = adv_features.std().item()

        lines.append(f"Features Range: clean={clean_f_range}, adv={adv_f_range}")
        lines.append(f"Features Stats: clean(mean={clean_f_mean:.4f}, std={clean_f_std:.4f}), adv(mean={adv_f_mean:.4f}, std={adv_f_std:.4f})")

    # Prompt comparison
    if clean_prompts is not None and adv_prompts is not None:
        clean_pts = clean_prompts.get("point_coords")
        adv_pts = adv_prompts.get("point_coords")
        if clean_pts is not None and adv_pts is not None:
            pts_shape_match = clean_pts.shape == adv_pts.shape
            lines.append(f"Prompts: clean_shape={tuple(clean_pts.shape)}, adv_shape={tuple(adv_pts.shape)}, match={pts_shape_match}")

    logger.debug("\n".join(lines))


def _warp_point_coords(
    point_coords: torch.Tensor,
    deform_offsets: torch.Tensor | None,
    image_size: tuple[int, int] = (1024, 1024),
) -> torch.Tensor:
    """Warp point coordinates using deformation offsets.

    When deformation attack is applied, the GT mask is warped. To maintain
    consistency with clean branch prompts, we need to warp the point coordinates
    by the same offsets.

    Args:
        point_coords: [B, N, 2] point coordinates in (x, y) format
        deform_offsets: [B, K, 2, H, W] or [B, 2, H, W] deformation offsets (None if no deformation)
        image_size: (H, W) image dimensions

    Returns:
        Warped point coordinates [B, N, 2]
    """
    if deform_offsets is None:
        return point_coords.clone()

    B, N, _ = point_coords.shape
    H_img, W_img = image_size

    # Handle both 4D [B, 2, H, W] and 5D [B, K, 2, H, W] tensors
    if deform_offsets.dim() == 5:
        # Average offsets across objects [B, K, 2, H, W] -> [B, 2, H, W]
        offsets_4d = deform_offsets.mean(dim=1)
    else:
        offsets_4d = deform_offsets

    _, _, H_off, W_off = offsets_4d.shape

    # point_coords are in pixel coordinates
    # We need to sample the offset at each point location

    # Normalize coordinates to [-1, 1] for grid_sample
    coords_normalized = point_coords.clone().float()
    coords_normalized[..., 0] = coords_normalized[..., 0] / W_img * 2 - 1  # x -> normalized
    coords_normalized[..., 1] = coords_normalized[..., 1] / H_img * 2 - 1  # y -> normalized

    # grid_sample expects [B, H_out, W_out, 2], we have [B, N, 2]
    # Reshape to [B, 1, N, 2] for sampling
    grid = coords_normalized.unsqueeze(1)  # [B, 1, N, 2]

    # Sample offsets at each point location
    # offsets_4d: [B, 2, H, W] -> need to interpolate
    sampled_offsets = torch.nn.functional.grid_sample(
        offsets_4d,  # [B, 2, H, W]
        grid,  # [B, 1, N, 2]
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )  # [B, 2, 1, N]

    # Reshape to [B, N, 2]
    sampled_offsets = sampled_offsets.squeeze(2).permute(0, 2, 1)  # [B, N, 2]

    # Apply offsets (offsets are in pixel space at offset resolution, scale to image resolution)
    scale_x = W_img / W_off
    scale_y = H_img / H_off

    # Use out-of-place operations to avoid gradient issues
    coords_x = point_coords[..., 0].float() + sampled_offsets[..., 0] * scale_x
    coords_y = point_coords[..., 1].float() + sampled_offsets[..., 1] * scale_y

    # Clamp to valid range (out-of-place)
    coords_x = coords_x.clamp(0, W_img - 1)
    coords_y = coords_y.clamp(0, H_img - 1)

    # Stack to form final coordinates [B, N, 2]
    warped_coords = torch.stack([coords_x, coords_y], dim=-1)

    return warped_coords


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
        freeze_image_encoder_epochs: int = 0,
        unfreeze_image_encoder_components=[],
        # Epsilon decay for adversarial training stability
        epsilon_decay_start_epoch: int = 0,  # Epoch to start decay (0 = disabled)
        epsilon_decay_end_epoch: int = 0,  # Epoch to reach min epsilon (0 = use max_epochs)
        epsilon_min_ratio: float = 0.3,  # Min epsilon as ratio of initial
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
            logging.info(f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}")
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

        self.freeze_image_encoder_epochs = int(freeze_image_encoder_epochs or 0)
        self._backbone_freeze_active = None

        # Epsilon decay initialization
        self.epsilon_decay_start_epoch = int(epsilon_decay_start_epoch or 0)
        self.epsilon_decay_end_epoch = int(epsilon_decay_end_epoch or 0)
        self.epsilon_min_ratio = float(epsilon_min_ratio if epsilon_min_ratio else 0.3)
        self._epsilon_decay_logged_epoch = -1
        # Initial epsilons captured from parent's AUE config (set after super().__init__)
        self._initial_style_epsilon = getattr(self, "style_adv_epsilon", 0.1)
        self._initial_deform_epsilon = getattr(self, "deform_adv_epsilon", 0.2)
        if self.freeze_image_encoder_epochs > 0 and getattr(self, "use_lora", False):
            logging.warning("freeze_image_encoder_epochs is set but use_lora=True; epoch-based freezing will be ignored to avoid freezing LoRA params.")
            self.freeze_image_encoder_epochs = 0

        if freeze_image_encoder:
            if self.freeze_image_encoder_epochs > 0:
                logging.warning("freeze_image_encoder_epochs is set but freeze_image_encoder=True; epoch-based freezing will be ignored.")
                self.freeze_image_encoder_epochs = 0
            if hasattr(self, "use_lora") and self.use_lora:
                # LoRA mode: PEFT/LoRA handles image_encoder freezing internally
                # (base weights frozen, only LoRA adapters trainable)
                # Other components (Mask Decoder, Prompt Encoder, BNDL, etc.) remain trainable
                logging.info("LoRA mode: image_encoder managed by PEFT (base frozen, LoRA trainable)")
                logging.info("Other components (Mask Decoder, Prompt Encoder, BNDL) remain trainable")

                # Count trainable params for logging
                lora_params = sum(p.numel() for n, p in self.named_parameters() if p.requires_grad and ("linear_a" in n or "linear_b" in n))
                other_trainable = sum(p.numel() for n, p in self.named_parameters() if p.requires_grad and not ("linear_a" in n or "linear_b" in n))
                logging.info(f"LoRA params: {lora_params:,} | Other trainable: {other_trainable:,}")

            else:
                # 2. Original Logic for standard Fine-Tuning (non-LoRA) or manual freezing
                # Freeze everything by default
                for p in self.image_encoder.parameters():
                    p.requires_grad = False

                # Unfreeze specific components
                if "neck" in unfreeze_image_encoder_components:
                    if hasattr(self.image_encoder, "neck"):
                        for p in self.image_encoder.neck.parameters():
                            p.requires_grad = True
                        logging.info("Unfrozen image encoder component: NECK")
                    else:
                        logging.warning("Requested to unfreeze 'neck' but image_encoder has no 'neck' attribute.")

                if "trunk_last_stage" in unfreeze_image_encoder_components:
                    if hasattr(self.image_encoder, "trunk") and hasattr(self.image_encoder.trunk, "blocks"):
                        if hasattr(self.image_encoder.trunk, "stage_ends"):
                            last_stage_end = self.image_encoder.trunk.stage_ends[-1]
                            if len(self.image_encoder.trunk.stage_ends) >= 2:
                                last_stage_start = self.image_encoder.trunk.stage_ends[-2] + 1
                            else:
                                last_stage_start = 0

                            for i in range(last_stage_start, last_stage_end + 1):
                                for p in self.image_encoder.trunk.blocks[i].parameters():
                                    p.requires_grad = True
                            logging.info(f"Unfrozen image encoder component: TRUNK_LAST_STAGE (blocks {last_stage_start}-{last_stage_end})")
                        else:
                            logging.warning("Requested to unfreeze 'trunk_last_stage' but trunk has no 'stage_ends'.")
                    else:
                        logging.warning("Requested to unfreeze 'trunk_last_stage' but image_encoder structure is unknown.")

                # Set global eval()
                self.image_encoder.eval()

                # Define custom train function
                def custom_train(mode=True):
                    if hasattr(self.image_encoder, "trunk"):
                        self.image_encoder.trunk.eval()
                    if "neck" in unfreeze_image_encoder_components and hasattr(self.image_encoder, "neck"):
                        self.image_encoder.neck.train(mode)
                    return self.image_encoder

                self.image_encoder.train = custom_train

    def apply_backbone_freeze(self, epoch: int) -> None:
        """Freeze image encoder for the first N epochs, then unfreeze."""
        if self.freeze_image_encoder_epochs <= 0:
            return

        should_freeze = epoch < self.freeze_image_encoder_epochs
        if self._backbone_freeze_active == should_freeze:
            return

        self._backbone_freeze_active = should_freeze
        for param in self.image_encoder.parameters():
            param.requires_grad = not should_freeze

        if should_freeze:
            self.image_encoder.eval()
            logging.info(f"Image encoder frozen for epoch {epoch} (unfreeze at epoch {self.freeze_image_encoder_epochs}).")
        else:
            self.image_encoder.train()
            logging.info(f"Image encoder unfrozen at epoch {epoch}.")

    def apply_epsilon_decay(self, epoch: int) -> None:
        """Decay adversarial epsilon based on epoch for training stability.

        Applies linear decay from initial epsilon to epsilon * min_ratio
        between start_epoch and end_epoch. Both style and deform attackers
        are decayed by the same factor to maintain relative strength.
        """
        if self.epsilon_decay_start_epoch <= 0:
            return  # Disabled

        # Calculate decay factor
        if epoch < self.epsilon_decay_start_epoch:
            decay_factor = 1.0  # Not started yet
        elif self.epsilon_decay_end_epoch <= self.epsilon_decay_start_epoch:
            decay_factor = self.epsilon_min_ratio  # Invalid range, use min
        elif epoch >= self.epsilon_decay_end_epoch:
            decay_factor = self.epsilon_min_ratio  # Reached minimum
        else:
            # Linear interpolation
            progress = (epoch - self.epsilon_decay_start_epoch) / (self.epsilon_decay_end_epoch - self.epsilon_decay_start_epoch)
            decay_factor = 1.0 - progress * (1.0 - self.epsilon_min_ratio)

        # Update Style Attacker epsilon
        if getattr(self, "use_style_adv", False) and hasattr(self, "style_attacker"):
            new_eps = self._initial_style_epsilon * decay_factor
            self.style_adv_epsilon = new_eps
            self.style_attacker.impl.epsilon = new_eps
            if hasattr(self.style_attacker.impl, "style_net"):
                self.style_attacker.impl.style_net.epsilon = new_eps

        # Update Deform Attacker epsilon
        if getattr(self, "use_deform_adv", False) and hasattr(self, "deform_attacker"):
            new_eps = self._initial_deform_epsilon * decay_factor
            self.deform_adv_epsilon = new_eps
            self.deform_attacker.impl.epsilon = new_eps
            if hasattr(self.deform_attacker.impl, "deform_module"):
                self.deform_attacker.impl.deform_module.epsilon = new_eps

        # Log only on epoch change
        if epoch != self._epsilon_decay_logged_epoch:
            self._epsilon_decay_logged_epoch = epoch
            style_eps = getattr(self, "style_adv_epsilon", 0)
            deform_eps = getattr(self, "deform_adv_epsilon", 0)
            logging.info(f"Epsilon decay: epoch={epoch}, factor={decay_factor:.3f}, style_eps={style_eps:.4f}, deform_eps={deform_eps:.4f}")

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
            num_init_cond_frames = self.rng.integers(1, num_init_cond_frames, endpoint=True)
        if use_pt_input and rand_frames_to_correct and num_frames_to_correct > num_init_cond_frames:
            # randomly select `num_init_cond_frames` to `num_frames_to_correct` frames to sample
            # correction clicks (only for the case of point input)
            num_frames_to_correct = self.rng.integers(num_init_cond_frames, num_frames_to_correct, endpoint=True)
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
        backbone_out["frames_not_in_init_cond"] = [t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames]
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
        ENABLE_BACKGROUND_MASK = self.adv_enable_background

        # Use max_num_objects from config (matches dataset sampler)
        # Add 1 slot for background if enabled
        max_num_objects = (self.max_num_objects + 1) if ENABLE_BACKGROUND_MASK else self.max_num_objects
        num_videos = input.num_videos
        H_mask, W_mask = input.masks.shape[-2:]

        # Initialize: [T, B_videos, K, H, W]
        # Force float32 for AUE masks (input.masks might be bool)
        mask_all_objs_for_aue = torch.zeros(num_frames, num_videos, max_num_objects, H_mask, W_mask, dtype=torch.float32, device=input.masks.device)
        num_objs_per_video_frame = torch.zeros(num_frames, num_videos, dtype=torch.int32, device=input.masks.device)

        # Fill in masks for each (frame, video)
        for t in range(num_frames):
            frame_masks = input.masks[t]  # [O_t, H, W]
            frame_obj_to_img = input.obj_to_frame_idx[t]  # [O_t, 2]

            for video_idx in range(num_videos):
                obj_mask = frame_obj_to_img[:, 1] == video_idx
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
                        union_mask = torch.zeros(H_mask, W_mask, dtype=mask_all_objs_for_aue.dtype, device=mask_all_objs_for_aue.device)

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
                        method=("uniform" if self.training else self.pt_sampling_for_eval),
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
            frames_to_add_correction_pt = init_cond_frames + self.rng.choice(backbone_out["frames_not_in_init_cond"], extra_num, replace=False).tolist()
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt

        return backbone_out

    def forward_tracking(self, backbone_out, input: BatchedVideoDatapoint, return_dict=False):
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
                ) = self._prepare_backbone_features_per_frame(input.flat_img_batch, img_ids)

            # Get output masks based on this frame's prompts and previous memory
            # Extract current frame's image for style-based AUE
            if img_feats_already_computed:
                # If features already computed, we have all images in flat_img_batch
                current_img_batch = input.flat_img_batch[img_ids] if hasattr(input, "flat_img_batch") else None
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
            add_output_as_cond_frame = stage_id in init_cond_frames or (self.add_all_frames_to_correct_as_cond and stage_id in frames_to_add_correction_pt)
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
        all_frame_outputs = [{k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs]

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
        USE_MULTI_OBJECT_AUE = self.adv_use_multi_object

        # DO NOT initialize pixel_gt_for_aue with gt_masks!
        # gt_masks is reserved for prompt sampling only
        pixel_gt_for_aue = None
        if self.use_aue:
            if USE_MULTI_OBJECT_AUE and is_init_cond_frame and hasattr(self, "_current_backbone_out"):
                backbone_out = self._current_backbone_out
                if "mask_all_objs_for_aue" in backbone_out:
                    mask_all_objs = backbone_out["mask_all_objs_for_aue"][frame_idx]  # [B_videos, K, H, W]
                    num_objs = backbone_out["num_objs_per_video_frame"][frame_idx]  # [B_videos]
                    obj_to_frame = self._current_obj_to_frame_idx  # [O_t, 2]

                    # For each object, construct all object masks from its video
                    O_t = len(obj_to_frame)
                    K = mask_all_objs.shape[1]
                    H, W = mask_all_objs.shape[2:]

                    # # DEBUG: Log mask construction
                    # print(f"DEBUG: AUE Track: Frame {frame_idx}, O_t={O_t}, K={K}")

                    multi_obj_masks = torch.zeros(O_t, K, H, W, dtype=mask_all_objs.dtype, device=mask_all_objs.device)

                    for i in range(O_t):
                        video_idx = obj_to_frame[i, 1].item()
                        K_actual = num_objs[video_idx].item()
                        # Copy all non-empty masks from mask_all_objs (includes foreground + background if enabled)
                        # Background is at slot K-1 (e.g., index 10 when K=11), not necessarily in [:K_actual]
                        multi_obj_masks[i] = mask_all_objs[video_idx]

                        # if i == 0 and logging.getLogger().isEnabledFor(logging.DEBUG):
                        #      logging.debug(f"AUE Track: Obj 0 (Video {video_idx}), K_actual={K_actual}, Mask sum={multi_obj_masks[i].sum().item()}")

                    # Use multi-object masks
                    pixel_gt_for_aue = multi_obj_masks
            elif not USE_MULTI_OBJECT_AUE and gt_masks is not None:
                # Single-object mode: always use current frame GT mask (all frames)
                pixel_gt_for_aue = gt_masks

        # Cache memory context for adversarial branch (used in AUE module)
        self._aue_memory_context = {
            "frame_idx": frame_idx,
            "is_init_cond_frame": is_init_cond_frame,
            "current_vision_feats": current_vision_feats,
            "current_vision_pos_embeds": current_vision_pos_embeds,
            "feat_sizes": feat_sizes,
            "output_dict": output_dict,
            "num_frames": num_frames,
            "track_in_reverse": track_in_reverse,
        }
        try:
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
            )
        finally:
            # Clear to avoid stale context on other code paths
            self._aue_memory_context = None

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
                raise RuntimeError(f"SAM2Train.track_step (frame_idx={frame_idx}): use_bndl_for_pixels=True but aux_outputs is None!")
            if not isinstance(aux_outputs, dict):
                raise RuntimeError(f"SAM2Train.track_step (frame_idx={frame_idx}): use_bndl_for_pixels=True but aux_outputs is not a dict! type: {type(aux_outputs)}")
            if "bndl" not in aux_outputs:
                raise RuntimeError(f"SAM2Train.track_step (frame_idx={frame_idx}): use_bndl_for_pixels=True but aux_outputs does not contain 'bndl'! aux_outputs keys: {list(aux_outputs.keys())}")
            bndl_data = aux_outputs["bndl"]
            if not isinstance(bndl_data, dict):
                raise RuntimeError(f"SAM2Train.track_step (frame_idx={frame_idx}): aux_outputs['bndl'] is not a dict! type: {type(bndl_data)}")

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
        # Store a DEEP COPY of initial point_inputs before iterative refinement modifies it
        # This is critical for AUE branch to get clean initial prompts
        if point_inputs is not None:
            initial_point_inputs_copy = {
                "point_coords": point_inputs["point_coords"].clone().detach(),
                "point_labels": point_inputs["point_labels"].clone().detach(),
            }
        else:
            initial_point_inputs_copy = None
        current_out["multistep_point_inputs"] = [initial_point_inputs_copy]
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
            )
            (_, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits, *_) = final_sam_outputs

        skip_aue = getattr(self, "_skip_aue_forward", False)

        if self.training and self.use_aue and is_init_cond_frame and pixel_gt_for_aue is not None and current_img_batch is not None and not skip_aue:
            # Get auxiliary outputs from the final refinement step
            final_aux_outputs = current_out.get("multistep_aux_outputs", [None])[-1]
            if final_aux_outputs is not None and "bndl" in final_aux_outputs:
                bndl_ns = final_aux_outputs["bndl"]
                pixel_feat = bndl_ns.get("pixel_feat_grad", bndl_ns.get("pixel_feat", None))
                pixel_logits_for_aue = bndl_ns.get(
                    "pixel_logits",
                    bndl_ns.get("masks_bndl_raw", bndl_ns.get("mean_pixel_logits", None)),
                )
                external_w_for_aue = bndl_ns.get("mask_tokens_out", None)
                pixel_uncertainty = bndl_ns.get("pixel_uncertainty", None)

                if pixel_feat is not None and pixel_logits_for_aue is not None and self._aue_module is not None:
                    # Initialize empty metrics (MMD computation moved to loss_mmd.py)
                    # Raw BNDL data is already in bndl_ns for loss function to use
                    aue_metrics = {}

                    # === Step 1: Generate adversarial samples ===
                    if self._aue_module._pipeline is not None:
                        enable_vis = getattr(self, "_enable_style_visualization", False)

                        adv_samples = self._aue_module.generate_adversarial_samples(
                            img_batch=current_img_batch,
                            backbone_features=pix_feat,
                            high_res_features=high_res_features,
                            pixel_gt=pixel_gt_for_aue,
                            single_obj_gt=gt_masks,  # Pass single object GT (same as clean branch)
                            enable_vis=enable_vis,
                        )

                        # === Step 2: Run forward + refinement on adversarial samples ===
                        # Reuse existing _track_step() and _iter_correct_pt_sampling() logic
                        adv_pixel_gt = adv_samples["adv_pixel_gt"]
                        adv_features = adv_samples["adv_features"]
                        adv_high_res = adv_samples["adv_high_res"]

                        # === CRITICAL: Use single object GT for SAM task (prompts + loss) ===
                        # adv_single_obj_gt is the warped version of gt_masks (if deformation was applied)
                        # This ensures adversarial branch learns the SAME task as clean branch
                        adv_single_obj_gt = adv_samples.get("adv_single_obj_gt")
                        if adv_single_obj_gt is not None:
                            adv_gt_for_task = adv_single_obj_gt.float()
                        elif gt_masks is not None:
                            # Fallback: use original gt_masks if no deformation
                            adv_gt_for_task = gt_masks.float()
                            if adv_gt_for_task.dim() == 3:
                                adv_gt_for_task = adv_gt_for_task.unsqueeze(1)
                        else:
                            # Legacy fallback: sum all objects (should not happen)
                            if adv_pixel_gt.shape[1] > 1:
                                adv_gt_for_task = adv_pixel_gt.sum(dim=1, keepdim=True).clamp(0, 1)
                            else:
                                adv_gt_for_task = adv_pixel_gt

                        # Add no_mem_embed to match clean branch behavior for init_cond_frames
                        # Clean branch: pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                        # See SAM2Base._prepare_memory_conditioned_features (L1470-1474)
                        if self.directly_add_no_mem_embed and hasattr(self, "no_mem_embed"):
                            # adv_features is [B, C, H, W], need to add no_mem_embed [1, 1, C]
                            # Reshape no_mem_embed for broadcasting
                            B, C, H, W = adv_features.shape
                            no_mem = self.no_mem_embed.view(1, C, 1, 1).expand(B, C, 1, 1)
                            adv_features = adv_features + no_mem

                        # Boolean GT for prompt sampling (only needed for iterative refinement fallback)
                        adv_gt_bool = adv_gt_for_task > 0.5

                        # === Use clean branch's INITIAL prompts with coordinate warping ===
                        # Get the initial prompts BEFORE iterative refinement (saved in multistep_point_inputs[0])
                        # Then warp coordinates and execute the same iterative refinement
                        deform_offsets = adv_samples.get("deform_offsets")  # [B, 2, H, W] or None

                        # Get initial prompts (before any correction points were added)
                        initial_point_inputs = current_out.get("multistep_point_inputs", [None])[0]

                        if initial_point_inputs is not None and initial_point_inputs.get("point_coords") is not None:
                            # Get image size from GT mask
                            H_img, W_img = adv_gt_for_task.shape[-2:]

                            # Clone initial coordinates to avoid inplace modification issues
                            # (clean branch's _iter_correct_pt_sampling modifies point_inputs inplace)
                            initial_coords = initial_point_inputs["point_coords"].clone().detach()
                            initial_labels = initial_point_inputs["point_labels"].clone().detach()

                            # Warp clean branch's initial coordinates using deformation offsets
                            warped_coords = _warp_point_coords(
                                initial_coords,
                                deform_offsets,
                                image_size=(H_img, W_img),
                            )
                            adv_point_inputs = {
                                "point_coords": warped_coords,
                                "point_labels": initial_labels,
                            }
                        else:
                            # Fallback: sample new prompts if clean branch didn't use points
                            from sam2.modeling.sam2_utils import get_next_point, sample_box_points

                            use_box = hasattr(self, "prob_to_use_box_input_for_train") and self.rng.random() < self.prob_to_use_box_input_for_train
                            if use_box:
                                adv_points, adv_labels = sample_box_points(adv_gt_bool)
                            else:
                                adv_points, adv_labels = get_next_point(
                                    gt_masks=adv_gt_bool,
                                    pred_masks=None,
                                    method="uniform",
                                )
                            adv_point_inputs = {"point_coords": adv_points, "point_labels": adv_labels}

                        # === Numerical Consistency Check (DEBUG level) ===
                        _log_branch_consistency(
                            clean_gt=gt_masks,
                            adv_gt=adv_gt_for_task,
                            clean_features=pix_feat,
                            adv_features=adv_features,
                            clean_prompts=initial_point_inputs,  # Compare with initial prompts
                            adv_prompts=adv_point_inputs,
                            tag=f"frame={frame_idx}",
                        )

                        # Forward through SAM heads with INITIAL prompts (same as clean branch step 1)
                        # Use gradient checkpointing for memory efficiency (AUE decoder recomputes during backward)
                        use_aue_decoder_checkpoint = getattr(self, "use_aue_decoder_checkpoint", True)
                        (
                            adv_low_res_multimasks,
                            adv_high_res_multimasks,
                            adv_ious,
                            adv_low_res_masks,
                            adv_high_res_masks,
                            _,
                            adv_object_score_logits,
                            adv_aux_outputs,
                        ) = self._forward_sam_heads(
                            backbone_features=adv_features,
                            point_inputs=adv_point_inputs,
                            mask_inputs=None,
                            high_res_features=adv_high_res,
                            multimask_output=self._use_multimask(is_init_cond_frame=True, point_inputs=adv_point_inputs),
                            use_checkpoint=use_aue_decoder_checkpoint,
                        )

                        # === Run iterative refinement (same as clean branch) ===
                        adv_multistep_multimasks = [adv_high_res_multimasks]
                        adv_multistep_ious = [adv_ious]
                        adv_multistep_object_score_logits = [adv_object_score_logits]

                        if frame_idx in frames_to_add_correction_pt and self.num_correction_pt_per_frame > 0:
                            # Prepare current_out for _iter_correct_pt_sampling
                            adv_current_out = {
                                "multistep_aux_outputs": [adv_aux_outputs],
                                "multistep_pred_multimasks_high_res": [adv_high_res_multimasks],
                                "multistep_pred_ious": [adv_ious],
                                "multistep_object_score_logits": [adv_object_score_logits],
                                "multistep_point_inputs": [adv_point_inputs],
                            }

                            # Reuse the same iterative refinement as clean branch
                            adv_point_inputs, final_adv_sam_outputs = self._iter_correct_pt_sampling(
                                is_init_cond_frame=True,
                                point_inputs=adv_point_inputs,
                                gt_masks=adv_gt_bool,
                                high_res_features=adv_high_res,
                                pix_feat_with_mem=adv_features,
                                low_res_multimasks=adv_low_res_multimasks,
                                high_res_multimasks=adv_high_res_multimasks,
                                ious=adv_ious,
                                low_res_masks=adv_low_res_masks,
                                high_res_masks=adv_high_res_masks,
                                object_score_logits=adv_object_score_logits,
                                current_out=adv_current_out,
                            )

                            # Extract multi-step outputs from the result
                            adv_multistep_multimasks = adv_current_out["multistep_pred_multimasks_high_res"]
                            adv_multistep_ious = adv_current_out["multistep_pred_ious"]
                            adv_multistep_object_score_logits = adv_current_out["multistep_object_score_logits"]
                            # Get the final aux_outputs from multistep
                            adv_aux_outputs = adv_current_out["multistep_aux_outputs"][-1]

                        # === Step 3: Store adversarial outputs for task/BNDL/MMD loss ===
                        # Store in SAME FORMAT as clean branch for consistent loss computation
                        bndl_ns["adv_outputs"] = {
                            # Multi-step outputs (for SAM loss via _forward)
                            "multistep_pred_multimasks_high_res": adv_multistep_multimasks,
                            "multistep_pred_ious": adv_multistep_ious,
                            "multistep_object_score_logits": adv_multistep_object_score_logits,
                            # Single-step outputs (for backward compatibility)
                            "pred_masks": adv_high_res_multimasks,  # [B, M, H, W] multimasks
                            "ious": adv_ious,  # [B, M]
                            "object_score_logits": adv_object_score_logits,  # [B, 1]
                            "gt": adv_gt_for_task,  # [B, 1, H, W] SINGLE object GT (same as clean branch)
                            "aux_outputs": adv_aux_outputs,  # For BNDL/MMD loss
                        }

                        # === Step 4: Store styled_images for attacker loss (SEPARATION DESIGN) ===
                        # This provides a gradient path from attacker loss back to Style/Deform networks
                        adv_images_for_attacker = adv_samples.get("adv_images_for_attacker")
                        if adv_images_for_attacker is not None:
                            bndl_ns["adv_images_for_attacker"] = adv_images_for_attacker
                            bndl_ns["attacker_pixel_gt"] = adv_pixel_gt  # For recomputing forward if needed

                        # === Prepare visualization data ===
                        vis_refs = adv_samples.get("vis_refs", {})
                        if enable_vis and vis_refs:
                            from sam2.modeling.aue.visualization import AUEVisualizer

                            visualizer = AUEVisualizer()
                            adv_img_for_vis = visualizer.select_adv_image_for_vis(vis_refs)
                            aue_metrics["aue_visualization"] = visualizer.prepare_visualization_data(
                                img_batch=vis_refs.get("img_batch", current_img_batch.detach().cpu()),
                                adv_images=adv_img_for_vis,
                                pixel_gt=vis_refs.get("pixel_gt", pixel_gt_for_aue.detach().cpu()),
                                original_styles=vis_refs.get("original_styles"),
                                adv_styles=vis_refs.get("adversarial_styles"),
                                deform_offsets=vis_refs.get("deform_offsets"),
                                warped_images=vis_refs.get("warped_images"),
                                warped_masks=vis_refs.get("warped_pixel_gt"),
                                attack_order=vis_refs.get("attack_order"),
                            )

                        bndl_ns["aue_metrics"] = aue_metrics
                    else:
                        # No adversarial pipeline available
                        bndl_ns["aue_loss_adv"] = torch.tensor(0.0, device=pixel_feat.device)

                    # Check for GCN stats
                    gcn_stats = getattr(self, "_latest_gcn_stats", None)
                    if gcn_stats and isinstance(bndl_ns, dict):
                        bndl_ns["gcn_stats"] = gcn_stats
                        self._latest_gcn_stats = None

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
                sample_from_gt = self.rng.random() < self.prob_to_sample_from_gt_for_train
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
                    use_reentrant=False,
                )
            else:
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
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
                    raise RuntimeError(f"SAM2Train._iter_correct_pt_sampling (step {_}, frame_idx={current_out.get('frame_idx', 'unknown')}): use_bndl_for_pixels=True but aux_outputs is None!")
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
                        f"SAM2Train._iter_correct_pt_sampling (step {_}, frame_idx={current_out.get('frame_idx', 'unknown')}): aux_outputs['bndl'] is not a dict! type: {type(bndl_data)}"
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
        current_out["multistep_pred_masks_high_res"] = torch.cat(all_pred_high_res_masks, dim=1)
        current_out["multistep_pred_multimasks"] = all_pred_multimasks
        current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
        current_out["multistep_pred_ious"] = all_pred_ious
        current_out["multistep_point_inputs"] = all_point_inputs
        current_out["multistep_object_score_logits"] = all_object_score_logits

        # 统一保存 aux_outputs（包含 BNDL/UR-ERN 等）
        current_out["multistep_aux_outputs"] = all_aux_outputs

        return point_inputs, sam_outputs
