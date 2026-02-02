# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import json
import logging
import math
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import cv2

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from training.optimizer import construct_optimizer

from training.utils.checkpoint_utils import (
    assert_skipped_parameters_are_frozen,
    exclude_params_matching_unix_pattern,
    load_state_dict_into_model,
    with_check_parameter_frozen,
)
from training.utils.data_utils import BatchedVideoDatapoint
from training.utils.distributed import all_reduce_max, barrier, get_rank

from training.utils.logger import Logger, setup_logging

# Import BNDL uncertainty and PAvPU functions
from BNDL.BNDL_upload.ViT_Sparse.utils.bndl import entropy_uncertainty, uncertainty_sample_parallel

# from training.utils.dataset_evaluator import DistributedDatasetEvaluator  # 暂时禁用

from training.utils.train_utils import (
    AverageMeter,
    collect_dict_keys,
    DurationMeter,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    human_readable_time,
    is_dist_avail_and_initialized,
    log_env_variables,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    set_seeds,
    setup_distributed_backend,
)


CORE_LOSS_KEY = "core_loss"


def unwrap_ddp_if_wrapped(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


@dataclass
class OptimAMPConf:
    enabled: bool = False
    amp_dtype: str = "float16"


@dataclass
class OptimConf:
    optimizer: torch.optim.Optimizer = None
    options: Optional[Dict[str, Any]] = None
    param_group_modifiers: Optional[List] = None
    amp: Optional[Dict[str, Any]] = None
    gradient_clip: Any = None
    gradient_logger: Any = None
    attacker_optim: Optional[Dict[str, Any]] = None  # Config for separate attacker optimizer

    def __post_init__(self):
        # amp
        if not isinstance(self.amp, OptimAMPConf):
            if self.amp is None:
                self.amp = {}
            assert isinstance(self.amp, Mapping)
            self.amp = OptimAMPConf(**self.amp)


@dataclass
class DistributedConf:
    backend: Optional[str] = None  # inferred from accelerator type
    comms_dtype: Optional[str] = None
    find_unused_parameters: bool = False
    timeout_mins: int = 30


@dataclass
class CudaConf:
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False
    # if not None, `matmul_allow_tf32` key will override `allow_tf32` for matmul
    matmul_allow_tf32: Optional[bool] = None
    # if not None, `cudnn_allow_tf32` key will override `allow_tf32` for cudnn
    cudnn_allow_tf32: Optional[bool] = None


@dataclass
class CheckpointConf:
    save_dir: str
    save_freq: int
    save_list: List[int] = field(default_factory=list)
    model_weight_initializer: Any = None
    save_best_meters: List[str] = None
    skip_saving_parameters: List[str] = field(default_factory=list)
    initialize_after_preemption: Optional[bool] = None
    # if not None, training will be resumed from this checkpoint
    resume_from: Optional[str] = None

    def infer_missing(self):
        if self.initialize_after_preemption is None:
            with_skip_saving = len(self.skip_saving_parameters) > 0
            self.initialize_after_preemption = with_skip_saving
        return self


@dataclass
class LoggingConf:
    log_dir: str
    log_freq: int  # In iterations
    tensorboard_writer: Any
    log_level_primary: str = "INFO"
    log_level_secondary: str = "ERROR"
    log_scalar_frequency: int = 100
    log_visual_frequency: int = 100
    scalar_keys_to_log: Optional[Dict[str, Any]] = None
    log_batch_stats: bool = False
    visualize_bndl: bool = False
    bndl_vis_sample_rate: float = 0.05  # Probability of saving BNDL visualization for each validation batch (0.05 = 5%)
    visualize_aue: bool = False  # Enable AUE visualization (style and/or deformation: original vs adversarial images)
    style_aue_visual_frequency: int = 5  # Log Style AUE visualization every N steps
    uncertainty_metric: set = field(default_factory=lambda: {"entropy"})  # Options: {"entropy"}, {"nll"}, {"sampling"}, {"entropy", "nll"}, {"entropy", "nll", "sampling"}, etc.
    visualize_pavpu_overlay: bool = True  # Enable PAvPU overlay visualization on original images
    enable_pavpu_eval: bool = True  # Enable PAvPU evaluator computation during validation
    uncertainty_sample_num: int = 50
    correlation_foreground_dilation: int = 0  # Foreground dilation radius (pixels), 0 means no dilation (only used when use_full_image=False)
    correlation_per_pixel: bool = True  # Use per-pixel statistics (vs per-image)
    correlation_use_full_image: bool = True  # Use full image statistics (True, default) or foreground only (False)


class Trainer:
    """
    Trainer supporting the DDP training strategies.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,  # the order of these args can change at any time, so they are keyword-only
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        accelerator: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        optim_overrides: Optional[List[Dict[str, Any]]] = None,
        meters: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        actual_max_epochs: Optional[int] = None,  # New: actual epochs to train (if different from max_epochs for lr scheduling)
        gradient_accumulation_steps: int = 1,  # Accumulate gradients over N steps before optimizer.step()
        gradient_conflict_monitor: Optional[Dict[str, Any]] = None,  # Gradient conflict monitoring config
    ):
        self._setup_env_variables(env_variables)
        self._setup_timers()

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._gradient_conflict_monitor_conf = gradient_conflict_monitor or {}

        self.data_conf = data
        self.model_conf = model
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint).infer_missing()
        self.max_epochs = max_epochs
        self.actual_max_epochs = actual_max_epochs if actual_max_epochs is not None else max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.optim_conf = OptimConf(**optim) if optim is not None else None
        self.meters_conf = meters
        self.loss_conf = loss
        distributed = DistributedConf(**distributed or {})
        cuda = CudaConf(**cuda or {})
        self.where = 0.0

        self._infer_distributed_backend_if_none(distributed, accelerator)

        self._setup_device(accelerator)

        self._setup_torch_dist_and_backend(cuda, distributed)

        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
        )

        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        log_env_variables()

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        self._setup_components()  # Except Optimizer everything is setup here.
        self._move_to_device()
        self._construct_optimizers()
        self._setup_dataloaders()

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")

        if self.checkpoint_conf.resume_from is not None:
            assert os.path.exists(self.checkpoint_conf.resume_from), f"The 'resume_from' checkpoint {self.checkpoint_conf.resume_from} does not exist!"
            dst = os.path.join(self.checkpoint_conf.save_dir, "checkpoint.pt")
            if self.distributed_rank == 0 and not os.path.exists(dst):
                # Copy the "resume_from" checkpoint to the checkpoint folder
                # if there is not a checkpoint to resume from already there
                makedir(self.checkpoint_conf.save_dir)
                g_pathmgr.copy(self.checkpoint_conf.resume_from, dst)
            barrier()

        # Initialize evaluators for PAvPU analysis during training and validation
        self.train_evaluator = None
        self.val_evaluator = None
        self._setup_evaluators()

        self.load_checkpoint()

        self._setup_ddp_distributed_training(distributed, accelerator)
        barrier()

    def _setup_timers(self):
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0
        self.est_epoch_time = dict.fromkeys([Phase.TRAIN, Phase.VAL], 0)

    def _get_meters(self, phase_filters=None):
        if self.meters is None:
            return {}
        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                if key_meters is None:
                    continue
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters

    def _infer_distributed_backend_if_none(self, distributed_conf, accelerator):
        if distributed_conf.backend is None:
            distributed_conf.backend = "nccl" if accelerator == "cuda" else "gloo"

    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.matmul_allow_tf32 if cuda_conf.matmul_allow_tf32 is not None else cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.cudnn_allow_tf32 if cuda_conf.cudnn_allow_tf32 is not None else cuda_conf.allow_tf32

        self.rank = setup_distributed_backend(distributed_conf.backend, distributed_conf.timeout_mins)

    def _setup_device(self, accelerator):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if accelerator == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")

    def _setup_ddp_distributed_training(self, distributed_conf, accelerator):
        assert isinstance(self.model, torch.nn.Module)

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if accelerator == "cuda" else [],
            find_unused_parameters=distributed_conf.find_unused_parameters,
        )
        if distributed_conf.comms_dtype is not None:  # noqa
            from torch.distributed.algorithms import ddp_comm_hooks

            amp_type = get_amp_type(distributed_conf.comms_dtype)
            if amp_type == torch.bfloat16:
                hook = ddp_comm_hooks.default_hooks.bf16_compress_hook
                logging.info("Enabling bfloat16 grad communication")
            else:
                hook = ddp_comm_hooks.default_hooks.fp16_compress_hook
                logging.info("Enabling fp16 grad communication")
            process_group = None
            self.model.register_comm_hook(process_group, hook)

    def _move_to_device(self):
        logging.info(f"Moving components to device {self.device} and local rank {self.local_rank}.")

        self.model.to(self.device)

        logging.info(f"Done moving components to device {self.device} and local rank {self.local_rank}.")

    def save_checkpoint(self, epoch, checkpoint_names=None):
        checkpoint_folder = self.checkpoint_conf.save_dir
        makedir(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (self.checkpoint_conf.save_freq > 0 and (int(epoch) % self.checkpoint_conf.save_freq == 0)) or int(epoch) in self.checkpoint_conf.save_list:
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_paths = []
        for ckpt_name in checkpoint_names:
            checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

        state_dict = unwrap_ddp_if_wrapped(self.model).state_dict()
        state_dict = exclude_params_matching_unix_pattern(patterns=self.checkpoint_conf.skip_saving_parameters, state_dict=state_dict)

        checkpoint = {
            "model": state_dict,
            "optimizer": self.optim.optimizer.state_dict(),
            "epoch": epoch,
            "loss": self.loss.state_dict(),
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "best_meter_values": self.best_meter_values,
        }
        if self.optim_conf.amp.enabled:
            checkpoint["scaler"] = self.scaler.state_dict()

        # DDP checkpoints are only saved on rank 0 (all workers are identical)
        if self.distributed_rank != 0:
            return

        for checkpoint_path in checkpoint_paths:
            self._save_checkpoint(checkpoint, checkpoint_path)

    def _save_checkpoint(self, checkpoint, checkpoint_path):
        """
        Save a checkpoint while guarding against the job being killed in the middle
        of checkpoint saving (which corrupts the checkpoint file and ruins the
        entire training since usually only the last checkpoint is kept per run).

        We first save the new checkpoint to a temp file (with a '.tmp' suffix), and
        and move it to overwrite the old checkpoint_path.
        """
        checkpoint_path_tmp = f"{checkpoint_path}.tmp"
        with g_pathmgr.open(checkpoint_path_tmp, "wb") as f:
            torch.save(checkpoint, f)
        # after torch.save is completed, replace the old checkpoint with the new one
        if g_pathmgr.exists(checkpoint_path):
            # remove the old checkpoint_path file first (otherwise g_pathmgr.mv fails)
            g_pathmgr.rm(checkpoint_path)
        success = g_pathmgr.mv(checkpoint_path_tmp, checkpoint_path)
        assert success

    def load_checkpoint(self):
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        if ckpt_path is None:
            self._init_model_state()
        else:
            if self.checkpoint_conf.initialize_after_preemption:
                self._call_model_initializer()
            self._load_resuming_checkpoint(ckpt_path)

    def _init_model_state(self):
        # Checking that parameters that won't be saved are indeed frozen
        # We do this check here before even saving the model to catch errors
        # are early as possible and not at the end of the first epoch
        assert_skipped_parameters_are_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
        )

        # Checking that parameters that won't be saved are initialized from
        # within the model definition, unless `initialize_after_preemption`
        # is explicitly set to `True`. If not, this is a bug, and after
        # preemption, the `skip_saving_parameters` will have random values
        allow_init_skip_parameters = self.checkpoint_conf.initialize_after_preemption
        with with_check_parameter_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
            disabled=allow_init_skip_parameters,
        ):
            self._call_model_initializer()

    def _call_model_initializer(self):
        model_weight_initializer = instantiate(self.checkpoint_conf.model_weight_initializer)
        if model_weight_initializer is not None:
            logging.info(f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}")
            self.model = model_weight_initializer(model=self.model)

    def _load_resuming_checkpoint(self, ckpt_path: str):
        logging.info(f"Resuming training from {ckpt_path}")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        load_state_dict_into_model(
            model=self.model,
            state_dict=checkpoint["model"],
            ignore_missing_keys=self.checkpoint_conf.skip_saving_parameters,
        )

        self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
        self.loss.load_state_dict(checkpoint["loss"], strict=True)
        self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"]
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed")

        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.best_meter_values = checkpoint.get("best_meter_values", {})

        if "train_dataset" in checkpoint and self.train_dataset is not None:
            self.train_dataset.load_checkpoint_state(checkpoint["train_dataset"])

    def is_intermediate_val_epoch(self, epoch):
        return epoch % self.val_epoch_freq == 0 and epoch < self.actual_max_epochs - 1

    def _step(
        self,
        batch: BatchedVideoDatapoint,
        model: nn.Module,
        phase: str,
    ):
        # Enable style visualization periodically BEFORE forward pass
        _model = unwrap_ddp_if_wrapped(self.model)
        style_aue_freq = 3  # Hardcoded: visualize every 5 steps

        # Track if we've already logged visualization for this step (to avoid duplicates with gradient accumulation)
        current_step = self.steps[phase]
        last_vis_step = getattr(self, "_last_vis_step", {}).get(phase, -1)
        already_logged_this_step = current_step == last_vis_step

        enable_vis = phase == "train" and current_step % style_aue_freq == 0 and not already_logged_this_step
        _model._enable_style_visualization = enable_vis

        # Mark this step as logged (will be used to prevent duplicate logging during gradient accumulation)
        if enable_vis:
            if not hasattr(self, "_last_vis_step"):
                self._last_vis_step = {}
            self._last_vis_step[phase] = current_step

        outputs = model(batch)
        targets = batch.masks
        batch_size = len(batch.img_batch)

        key = batch.dict_key  # key for dataset
        loss = self.loss[key](outputs, targets)
        loss_str = f"Losses/{phase}_{key}_loss"

        loss_log_str = os.path.join("Step_Losses", loss_str)

        # loss contains multiple sub-components we wish to log
        step_losses = {}

        # Initialize bndl_outputs to None to avoid NameError when use_bndl_for_pixels is False
        bndl_outputs = None

        # Log BNDL statistics from model outputs if available (only if BNDL is enabled)
        if getattr(_model, "use_bndl_for_pixels", False):
            bndl_outputs, step_index, frame_index = self._extract_bndl_outputs(outputs)
            if bndl_outputs is not None:
                # Calculate uncertainty and add to evaluator if in validation phase
                if phase == "val" and targets is not None:
                    # Extract the current frame targets
                    if frame_index is not None and targets.shape[0] > frame_index:
                        current_frame_targets = targets[frame_index]
                    else:
                        current_frame_targets = targets[0] if targets.shape[0] > 0 else targets

                    # Calculate uncertainty and add to evaluator
                    bndl_outputs = self._calculate_uncertainty_for_bndl(bndl_outputs, batch, current_frame_targets)
                    if self.val_evaluator is not None:
                        self._add_to_evaluator(bndl_outputs, current_frame_targets, self.val_evaluator)

                # Log BNDL statistics for both train and val
                self._log_bndl_statistics(bndl_outputs, self.steps[phase], phase)

                # Also check for aue_metrics in the FIRST aux_output (where it's set during _forward_sam_heads)
                # Since AUE is only computed on the first forward pass, we need to look there specifically
                self._log_aue_metrics_from_outputs(outputs, self.steps[phase], phase)

                # NOTE: AUE loss terms are now logged via loss_combined.py return dict
                # (Losses/train_all_aue_task_loss, etc.) - no manual extraction needed.

                if isinstance(bndl_outputs, dict):
                    pixel_nll = bndl_outputs.get("pixel_nll")
                    if isinstance(pixel_nll, torch.Tensor):
                        step_losses[f"BNDL/NLL/{phase}_{key}_pixel_nll"] = pixel_nll

        # Log Style AUE visualization if available and enabled (only on rank 0)
        if (
            phase == "train"
            and self.distributed_rank == 0
            and getattr(self.logging_conf, "visualize_aue", False)
            and hasattr(_model, "use_style_adv")
            and _model.use_style_adv
            and _model._enable_style_visualization
        ):
            self._log_style_aue_visualization(outputs, self.steps[phase])

        # Log Deformation AUE visualization if available and enabled (only on rank 0)
        if (
            phase == "train"
            and self.distributed_rank == 0
            and getattr(self.logging_conf, "visualize_aue", False)
            and hasattr(_model, "use_deform_adv")
            and _model.use_deform_adv
            and _model._enable_style_visualization
        ):
            self._log_deform_aue_visualization(outputs, self.steps[phase])

        # Log Clean vs Adversarial comparison (input, segmentation, uncertainty from both branches)
        # Replaces the old separate uncertainty visualization
        if phase == "train" and self.distributed_rank == 0 and getattr(self.logging_conf, "visualize_aue", False) and _model._enable_style_visualization and bndl_outputs is not None:
            self._log_clean_vs_adv_comparison(bndl_outputs, batch, outputs, self.steps[phase], frame_index=frame_index)

        if isinstance(loss, dict):
            if CORE_LOSS_KEY not in loss:
                raise KeyError(f"Missing {CORE_LOSS_KEY} in loss dict for phase={phase}, key={key}")

            # Separate normal losses and refinement metrics
            for k, v in loss.items():
                if k == CORE_LOSS_KEY:
                    continue

                # Preserve _attacker_only_loss for separate optimizer (keep tensor with gradients)
                # Also log with proper prefix for TensorBoard categorization
                if k == "_attacker_only_loss":
                    step_losses[k] = v  # Keep as-is for optimizer (tensor with gradients)
                    # Also add a prefixed version for TensorBoard logging
                    step_losses[f"Losses/{phase}_{key}{k}"] = v.detach() if torch.is_tensor(v) else v
                    continue

                ref_idx = k.find("_refinement_")
                if ref_idx != -1:
                    # Route refinement metrics to Refinement/ prefix
                    metric_name = k[ref_idx + 1 :]  # Remove leading underscore
                    step_losses[f"Refinement/{phase}_{key}_{metric_name}"] = v
                else:
                    # Normal losses go to Losses/ prefix
                    step_losses[f"Losses/{phase}_{key}_{k}"] = v

            # Compute gradient conflicts if monitor is enabled (training phase only)
            if phase == "train" and hasattr(self, "gradient_conflict_monitor") and self.gradient_conflict_monitor is not None:
                try:
                    # Log first invocation
                    if not hasattr(self, "_gc_logged_first_call"):
                        self._gc_logged_first_call = True
                        logging.info(f"GradConflict: First invocation at step {self.steps[phase]}, loss keys: {list(loss.keys())}")
                    # Get TensorBoard writer from Logger.tb_logger.writer
                    tb_writer = getattr(getattr(self.logger, "tb_logger", None), "writer", None)

                    self.gradient_conflict_monitor.compute_conflict_from_loss_dict(
                        loss_dict=loss,
                        writer=tb_writer,
                        step=self.steps[phase],
                    )
                except Exception as e:
                    # Log errors as warning to make them visible
                    if not hasattr(self, "_gc_error_logged"):
                        self._gc_error_logged = True
                        logging.warning(f"Gradient conflict monitor error: {e}", exc_info=True)

            loss = self._log_loss_detailed_and_return_core_loss(loss, loss_log_str, self.steps[phase], phase)

        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(
                loss_log_str,
                loss,
                self.steps[phase],
            )

        ret_tuple = {loss_str: loss}, batch_size, step_losses

        if phase in self.meters and key in self.meters[phase]:
            meters_dict = self.meters[phase][key]
            if meters_dict is not None:
                for _, meter in meters_dict.items():
                    meter.update(
                        find_stages=outputs,
                        find_metadatas=batch.metadata,
                    )

        return ret_tuple

    def run(self):
        assert self.mode in ["train", "train_only", "val"]
        if self.mode == "train":
            if self.epoch > 0:
                logging.info(f"Resuming training from epoch: {self.epoch}")
                # resuming from a checkpoint
                if self.is_intermediate_val_epoch(self.epoch - 1):
                    logging.info("Running previous val epoch")
                    self.epoch -= 1
                    self.run_val()
                    self.epoch += 1
            self.run_train()
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        elif self.mode == "train_only":
            self.run_train()

    def _setup_dataloaders(self):
        # Initialize if not already done by _setup_components for AUE
        if not hasattr(self, "train_dataset"):
            self.train_dataset = None
        if not hasattr(self, "val_dataset"):
            self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))

        if self.mode in ["train", "train_only"] and self.train_dataset is None:
            self.train_dataset = instantiate(self.data_conf.train)

    def run_train(self):
        while self.epoch < self.actual_max_epochs:
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            barrier()
            outs = self.train_epoch(dataloader)
            self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

            # log train to text file.
            if self.distributed_rank == 0:
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "train_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(outs) + "\n")

            # Save checkpoint before validating
            self.save_checkpoint(self.epoch + 1)

            del dataloader
            gc.collect()

            # Run val, not running on last epoch since will run after the
            # loop anyway
            if self.is_intermediate_val_epoch(self.epoch):
                self.run_val()

            if self.distributed_rank == 0:
                self.best_meter_values.update(self._get_trainer_state("train"))
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "best_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(self.best_meter_values) + "\n")

            self.epoch += 1

            # Step attacker LR scheduler (if exists)
            if hasattr(self, "attacker_scheduler") and self.attacker_scheduler is not None:
                self.attacker_scheduler.step()
        # epoch was incremented in the loop but the val step runs out of the loop
        self.epoch -= 1

    def run_val(self):
        if not self.val_dataset:
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        outs = self.val_epoch(dataloader, phase=Phase.VAL)
        del dataloader
        gc.collect()
        self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

        if self.distributed_rank == 0:
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "val_stats.json"),
                "a",
            ) as f:
                f.write(json.dumps(outs) + "\n")

            # WandB logging for sweep early stopping (Hyperband)
            # Enabled via WANDB_SWEEP_LOGGING=1 environment variable
            if os.environ.get("WANDB_SWEEP_LOGGING", "") == "1":
                self._log_to_wandb(outs, step=int(self.epoch))

    def _log_to_wandb(self, outs: dict, step: int):
        """Log metrics to wandb for sweep early stopping."""
        try:
            import wandb

            if wandb.run is None:
                return

            def _get_val_metric(metric_name: str):
                """Return the first matching val metric key if present."""
                candidates = [
                    f"Losses/val_val_{metric_name}",
                    f"Losses/val_{metric_name}",
                    f"Losses/val_all_{metric_name}",
                ]
                for k in candidates:
                    if k in outs:
                        return outs[k]
                return None

            # Extract key metrics for sweep comparison (robust to key naming)
            wandb_metrics = {
                "val/loss_dice": _get_val_metric("loss_dice"),
                "val/loss_mask": _get_val_metric("loss_mask"),
                "val/loss_iou": _get_val_metric("loss_iou"),
                "val/loss_class": _get_val_metric("loss_class"),
                "val/core_loss": _get_val_metric("core_loss"),
                "epoch": step,
            }
            # Filter None values
            wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}
            if wandb_metrics:
                wandb.log(wandb_metrics, step=step)
        except ImportError:
            pass
        except Exception as e:
            logging.warning(f"WandB logging failed: {e}")

    def val_epoch(self, val_loader, phase):
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = MemMeter("Mem (GB)", self.device, ":.2f")

        iters_per_epoch = len(val_loader)

        curr_phases = [phase]
        curr_models = [self.model]

        loss_names = []
        for p in curr_phases:
            for key in self.loss.keys():
                loss_names.append(f"Losses/{p}_{key}_loss")

        loss_mts = OrderedDict([(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names])
        extra_loss_mts = {}

        for model in curr_models:
            model.eval()
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_start"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_start()

        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, self.time_elapsed_meter, *loss_mts.values()],
            self._get_meters(curr_phases),
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        end = time.time()

        for data_iter, batch in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            batch = batch.to(self.device, non_blocking=True)

            # compute output
            with (
                torch.no_grad(),
                torch.cuda.amp.autocast(
                    enabled=(self.optim_conf.amp.enabled if self.optim_conf else False),
                    dtype=(get_amp_type(self.optim_conf.amp.amp_dtype) if self.optim_conf else None),
                ),
            ):
                for phase, model in zip(curr_phases, curr_models, strict=False):
                    loss_dict, batch_size, extra_losses = self._step(
                        batch,
                        model,
                        phase,
                    )

                    assert len(loss_dict) == 1
                    loss_key, loss = loss_dict.popitem()

                    loss_mts[loss_key].update(loss.item(), batch_size)

                    for k, v in extra_losses.items():
                        # Skip internal keys (start with _) - these are for optimizer use only
                        if k.startswith("_"):
                            continue
                        if k not in extra_loss_mts:
                            extra_loss_mts[k] = AverageMeter(k, self.device, ":.2e")
                        if torch.is_tensor(v):
                            update_val = v.item()
                        else:
                            update_val = v
                        extra_loss_mts[k].update(update_val, batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)

            if torch.cuda.is_available():
                mem.update(reset_peak_usage=True)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

            if data_iter % self.logging_conf.log_scalar_frequency == 0:
                # Log progress meters.
                for progress_meter in progress.meters:
                    self.logger.log(
                        os.path.join("Step_Stats", phase, progress_meter.name),
                        progress_meter.val,
                        self.steps[Phase.VAL],
                    )

            # Increment val step counter (no gradient accumulation in validation)
            self.steps[Phase.VAL] += 1

            # Randomly visualize BNDL parameters during validation - DISABLED
            # CRITICAL FIX: Run forward pass on ALL ranks to avoid SyncBatchNorm deadlock
            # Use deterministic sampling based on step count instead of random.random()
            # vis_interval = int(1.0 / max(self.logging_conf.bndl_vis_sample_rate, 0.001))
            # should_visualize = self.logging_conf.visualize_bndl and getattr(unwrap_ddp_if_wrapped(self.model), "use_bndl_for_pixels", False) and (data_iter % vis_interval == 0)
            should_visualize = False  # Disable visualization

            if should_visualize:
                # Extract BNDL outputs for visualization
                _model = unwrap_ddp_if_wrapped(self.model)
                if hasattr(batch, "img_batch"):
                    # Re-run forward pass to get outputs with BNDL data
                    # Must run on ALL ranks to keep SyncBatchNorm in sync
                    with torch.no_grad():
                        outputs_for_vis = _model(batch)

                    # Visualization only on rank 0
                    if self.distributed_rank == 0:
                        bndl_outputs, step_index, frame_index = self._extract_bndl_outputs(outputs_for_vis)
                        if bndl_outputs is not None:
                            # Calculate uncertainty for visualization
                            if frame_index is not None and batch.masks.shape[0] > frame_index:
                                current_frame_targets = batch.masks[frame_index]
                            else:
                                current_frame_targets = batch.masks[0] if batch.masks.shape[0] > 0 else batch.masks
                            bndl_outputs = self._calculate_uncertainty_for_bndl(bndl_outputs, batch, current_frame_targets)

                            vis_dir = os.path.join(self.logging_conf.log_dir, "bndl_visualizations", phase)
                            makedir(vis_dir)
                            # Create full visualization with uncertainty and PAvPU overlays
                            self._create_unified_visualization(bndl_outputs, batch, outputs_for_vis, vis_dir, data_iter, step_index, frame_index, layout_type="full")

            # Visualize AUE adversarial images periodically during validation
            # Note: AUE visualization is now handled by _log_style_aue_visualization
            # which extracts data from model outputs (loss_dict['aue_visualization'])

            if data_iter % 10 == 0:
                dist.barrier()

        # Use val_evaluator for validation epoch analysis
        if self.val_evaluator is not None:
            logging.info(f"Val evaluator: {len(self.val_evaluator)} images on rank {self.rank}")

            # Evaluate all metrics
            results = self.val_evaluator.evaluate()
            if results:
                logging.info(f"Evaluation completed: {len(results)} metrics")

                # Log to TensorBoard under "Uncertainty/" section
                for metric_name, metric_value in results.items():
                    if metric_name == "uncertainty_histogram":
                        # Log histogram for uncertainty distribution
                        self.logger.add_histogram("Uncertainty/distribution", metric_value, self.epoch)
                    elif isinstance(metric_value, (int, float)) and not metric_name.startswith("n_"):
                        self.logger.log(
                            f"Uncertainty/{metric_name}",
                            metric_value,
                            self.epoch,
                        )

            # Visualization and save
            self.val_evaluator.create_visualization(filename=f"epoch_{self.epoch}_unc_eval.png")
            self.val_evaluator.save_results(filename=f"epoch_{self.epoch}_unc_results.json")
            logging.info(f"Uncertainty evaluation completed for epoch {self.epoch}")

            # Reset for next epoch
            self.val_evaluator.reset()

        self.est_epoch_time[phase] = batch_time.avg * iters_per_epoch
        self._log_timers(phase)
        for model in curr_models:
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_end"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_end()

        out_dict = self._log_meters_and_save_best_ckpts(curr_phases)

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg

        for phase in curr_phases:
            out_dict.update(self._get_trainer_state(phase))
        self._reset_meters(curr_phases)
        logging.info(f"Meters: {out_dict}")
        return out_dict

    def _get_trainer_state(self, phase):
        return {
            "Trainer/where": self.where,
            "Trainer/epoch": self.epoch,
            f"Trainer/steps_{phase}": self.steps[phase],
        }

    def train_epoch(self, train_loader):
        # Init stat meters
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f")
        data_time_meter = AverageMeter("Data Time", self.device, ":.2f")
        mem_meter = MemMeter("Mem (GB)", self.device, ":.2f")
        data_times = []
        phase = Phase.TRAIN

        iters_per_epoch = len(train_loader)

        if "all" in self.loss:
            loss_all = self.loss["all"]
            if hasattr(loss_all, "apply_schedule"):
                loss_all.apply_schedule(self.epoch)

        loss_names = []
        for batch_key in self.loss.keys():
            loss_names.append(f"Losses/{phase}_{batch_key}_loss")

        loss_mts = OrderedDict([(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names])
        extra_loss_mts = {}

        progress = ProgressMeter(
            iters_per_epoch,
            [
                batch_time_meter,
                data_time_meter,
                mem_meter,
                self.time_elapsed_meter,
                *loss_mts.values(),
            ],
            self._get_meters([phase]),
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        # Model training loop
        self.model.train()
        model_unwrapped = unwrap_ddp_if_wrapped(self.model)
        if hasattr(model_unwrapped, "apply_backbone_freeze"):
            model_unwrapped.apply_backbone_freeze(self.epoch)
        if hasattr(model_unwrapped, "apply_epsilon_decay"):
            model_unwrapped.apply_epsilon_decay(self.epoch)

        if "all" in self.loss and hasattr(self.loss["all"], "aue_weight"):
            skip_aue = self.loss["all"].aue_weight == 0
            setattr(model_unwrapped, "_skip_aue_forward", skip_aue)
            if skip_aue and self.epoch == 0:
                logging.info(f"AUE forward pass skipped (aue_weight=0) - AUE networks will not train until weight > 0")

        end = time.time()

        for data_iter, batch in enumerate(train_loader):
            # measure data loading time
            data_time_meter.update(time.time() - end)
            data_times.append(data_time_meter.val)
            batch = batch.to(self.device, non_blocking=True)  # move tensors in a tensorclass

            # Gradient accumulation: zero gradients at start of accumulation window
            if data_iter % self.gradient_accumulation_steps == 0:
                self.optim.zero_grad(set_to_none=True)
                if self.attacker_optim is not None:
                    self.attacker_optim.zero_grad(set_to_none=True)

            try:
                is_accumulation_step = (data_iter + 1) % self.gradient_accumulation_steps != 0

                sync_context = self.model.no_sync() if is_accumulation_step else nullcontext()
                with sync_context:
                    self._run_step(batch, phase, loss_mts, extra_loss_mts)

                # Measure elapsed time and update progress (for all steps, including accumulation)
                batch_time_meter.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)
                mem_meter.update(reset_peak_usage=True)

                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)

                # Only do optimizer step at end of accumulation window
                if is_accumulation_step:
                    # Skip optimizer step, scheduler update, etc. - just accumulate gradients
                    continue

                # compute gradient and do optim step
                exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
                self.where = float(exact_epoch) / self.max_epochs
                assert self.where <= 1 + self.EPSILON
                if self.where < 1.0:
                    self.optim.step_schedulers(self.where, step=int(exact_epoch * iters_per_epoch))
                else:
                    logging.warning(f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1].")

                # Log schedulers
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for j, param_group in enumerate(self.optim.optimizer.param_groups):
                        for option in self.optim.schedulers[j]:
                            optim_prefix = "" + f"{j}_" if len(self.optim.optimizer.param_groups) > 1 else ""
                            self.logger.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                    # Log attacker optimizer LR (if exists)
                    if hasattr(self, "attacker_optim") and self.attacker_optim is not None:
                        attacker_lr = self.attacker_optim.param_groups[0]["lr"]
                        self.logger.log("Optim/attacker_lr", attacker_lr, self.steps[phase])

                # Clipping gradients and detecting diverging gradients
                # Unscale BOTH optimizers before clipping so gradient norms are correct
                if self.gradient_clipper is not None:
                    self.scaler.unscale_(self.optim.optimizer)
                    # Also unscale attacker optimizer if it exists and has gradients
                    if self.attacker_optim is not None and getattr(self, "_attacker_backward_done", False):
                        self.scaler.unscale_(self.attacker_optim)
                    self.gradient_clipper(model=self.model)

                # Debug: Check for NaN gradients after clipping
                has_nan_grad = False
                nan_param_names = []
                for name, param in self.model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        has_nan_grad = True
                        nan_count = torch.isnan(param.grad).sum().item()
                        inf_count = torch.isinf(param.grad).sum().item()
                        nan_param_names.append(f"{name}: nan={nan_count}, inf={inf_count}")
                        if len(nan_param_names) >= 10:  # Limit to first 10
                            break

                if has_nan_grad:
                    logging.error(f"[Trainer] NaN/Inf gradients detected after gradient clipping! First 10 problematic params: {nan_param_names}")
                    # Skip optimizer step to prevent corrupting weights
                    self.optim.zero_grad(set_to_none=True)
                    if self.attacker_optim is not None:
                        self.attacker_optim.zero_grad(set_to_none=True)
                    # NOTE: NOT calling scaler.update() here - let the error surface on next iteration
                    # This ensures we don't mask underlying NaN gradient issues
                    logging.warning("[Trainer] Skipping optimizer step due to NaN gradients")
                    continue  # Skip to next iteration

                if self.gradient_logger is not None:
                    self.gradient_logger(self.model, rank=self.distributed_rank, where=self.where)

                # Optimizer step: the scaler will make sure gradients are not
                # applied if the gradients are infinite
                self.scaler.step(self.optim.optimizer)

                # === Attacker Optimizer Step ===
                # If separate attacker optimizer is configured, step it too
                # Note: Already unscaled above (before gradient clipping)
                if self.attacker_optim is not None and getattr(self, "_attacker_backward_done", False):
                    self.scaler.step(self.attacker_optim)
                    self._attacker_backward_done = False  # Reset flag

                self.scaler.update()

                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    # Log progress meters.
                    for progress_meter in progress.meters:
                        self.logger.log(
                            os.path.join("Step_Stats", phase, progress_meter.name),
                            progress_meter.val,
                            self.steps[phase],
                        )

                # Increment step counter after optimizer step (not per forward pass)
                self.steps[phase] += 1

            # Catching NaN/Inf errors in the loss
            except FloatingPointError as e:
                raise e

        self.est_epoch_time[Phase.TRAIN] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(Phase.TRAIN)
        self._log_sync_data_times(Phase.TRAIN, data_times)

        out_dict = self._log_meters_and_save_best_ckpts([Phase.TRAIN])

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg
        out_dict.update(self._get_trainer_state(phase))
        logging.info(f"Losses and meters: {out_dict}")
        self._reset_meters([phase])
        return out_dict

    def _log_sync_data_times(self, phase, data_times):
        data_times = all_reduce_max(torch.tensor(data_times)).tolist()
        steps = range(self.steps[phase] - len(data_times), self.steps[phase])
        for step, data_time in zip(steps, data_times, strict=False):
            if step % self.logging_conf.log_scalar_frequency == 0:
                self.logger.log(
                    os.path.join("Step_Stats", phase, "Data Time Synced"),
                    data_time,
                    step,
                )

    def _run_step(
        self,
        batch: BatchedVideoDatapoint,
        phase: str,
        loss_mts: Dict[str, AverageMeter],
        extra_loss_mts: Dict[str, AverageMeter],
        raise_on_error: bool = True,
    ):
        """
        Run the forward / backward.

        Gradient accumulation is handled by the training loop based on
        gradient_accumulation_steps config.

        === Separate Attacker Optimizer ===
        If attacker_optim is configured, we run TWO backward passes:
        1. core_loss.backward(retain_graph=True) → main optimizer updates SAM+BNDL
        2. attacker_loss.backward() → attacker optimizer updates Style/Deform networks
        """
        with torch.cuda.amp.autocast(
            enabled=self.optim_conf.amp.enabled,
            dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
        ):
            loss_dict, batch_size, extra_losses = self._step(
                batch,
                self.model,
                phase,
            )

        assert len(loss_dict) == 1
        loss_key, loss = loss_dict.popitem()

        if not math.isfinite(loss.item()):
            error_msg = f"Loss is {loss.item()}, attempting to stop training"
            logging.error(error_msg)
            if raise_on_error:
                raise FloatingPointError(error_msg)
            else:
                return

        # Scale loss by gradient accumulation steps
        accum_steps = getattr(self, "gradient_accumulation_steps", 1)
        scaled_loss = loss / accum_steps

        # === Check for separate attacker backward ===
        # Get attacker_only_loss from extra_losses (put there by loss_combined.py)
        attacker_loss = None
        attacker_loss_key = "_attacker_only_loss"
        for k, v in extra_losses.items():
            if k == attacker_loss_key or attacker_loss_key in k:
                if torch.is_tensor(v) and v.requires_grad:
                    attacker_loss = v
                    break

        use_separate_attacker = self.attacker_optim is not None and attacker_loss is not None

        if use_separate_attacker:
            # Two backward passes: main + attacker
            # retain_graph=True for main backward because attacker backward needs the graph
            self.scaler.scale(scaled_loss).backward(retain_graph=True)

            # Scale attacker loss as well
            scaled_attacker_loss = attacker_loss / accum_steps
            self.scaler.scale(scaled_attacker_loss).backward()

            # Set flag so optimizer step knows backward was done
            self._attacker_backward_done = True
        else:
            # Single backward pass (normal case)
            self.scaler.scale(scaled_loss).backward()

        loss_mts[loss_key].update(loss.item(), batch_size)
        for extra_loss_key, extra_loss in extra_losses.items():
            # Skip internal keys (start with _)
            if extra_loss_key.startswith("_"):
                continue
            # Skip None values (e.g., optional AUE losses when not computed)
            if extra_loss is None:
                continue
            if extra_loss_key not in extra_loss_mts:
                extra_loss_mts[extra_loss_key] = AverageMeter(extra_loss_key, self.device, ":.2e")
            if torch.is_tensor(extra_loss):
                update_val = extra_loss.item()
            else:
                update_val = extra_loss
            extra_loss_mts[extra_loss_key].update(update_val, batch_size)

    def _log_meters_and_save_best_ckpts(self, phases: List[str]):
        logging.info("Synchronizing meters")
        out_dict = {}
        checkpoint_save_keys = []
        for key, meter in self._get_meters(phases).items():
            meter_output = meter.compute_synced()
            is_better_check = getattr(meter, "is_better", None)

            for meter_subkey, meter_value in meter_output.items():
                out_dict[os.path.join("Meters_train", key, meter_subkey)] = meter_value

                if is_better_check is None:
                    continue

                tracked_meter_key = os.path.join(key, meter_subkey)
                if tracked_meter_key not in self.best_meter_values or is_better_check(
                    meter_value,
                    self.best_meter_values[tracked_meter_key],
                ):
                    self.best_meter_values[tracked_meter_key] = meter_value

                    if self.checkpoint_conf.save_best_meters is not None and key in self.checkpoint_conf.save_best_meters:
                        checkpoint_save_keys.append(tracked_meter_key.replace("/", "_"))

        if len(checkpoint_save_keys) > 0:
            self.save_checkpoint(self.epoch + 1, checkpoint_save_keys)

        return out_dict

    def _log_timers(self, phase):
        time_remaining = 0
        epochs_remaining = self.actual_max_epochs - self.epoch - 1
        val_epochs_remaining = sum(n % self.val_epoch_freq == 0 for n in range(self.epoch, self.actual_max_epochs))

        # Adding the guaranteed val run at the end if val_epoch_freq doesn't coincide with
        # the end epoch.
        if (self.actual_max_epochs - 1) % self.val_epoch_freq != 0:
            val_epochs_remaining += 1

        # Remove the current val run from estimate
        if phase == Phase.VAL:
            val_epochs_remaining -= 1

        time_remaining += epochs_remaining * self.est_epoch_time[Phase.TRAIN] + val_epochs_remaining * self.est_epoch_time[Phase.VAL]

        self.logger.log(
            os.path.join("Step_Stats", phase, self.time_elapsed_meter.name),
            self.time_elapsed_meter.val,
            self.steps[phase],
        )

        logging.info(f"Estimated time remaining: {human_readable_time(time_remaining)}")

    def _reset_meters(self, phases: str) -> None:
        for meter in self._get_meters(phases).values():
            meter.reset()

    def _check_val_key_match(self, val_keys, phase):
        if val_keys is not None:
            # Check if there are any duplicates
            assert len(val_keys) == len(set(val_keys)), f"Duplicate keys in val datasets, keys: {val_keys}"

            # Check that the keys match the meter keys
            if self.meters_conf is not None and phase in self.meters_conf:
                assert set(val_keys) == set(self.meters_conf[phase].keys()), (
                    f"Keys in val datasets do not match the keys in meters."
                    f"\nMissing in meters: {set(val_keys) - set(self.meters_conf[phase].keys())}"
                    f"\nMissing in val datasets: {set(self.meters_conf[phase].keys()) - set(val_keys)}"
                )

            if self.loss_conf is not None:
                loss_keys = set(self.loss_conf.keys()) - set(["all"])
                assert all([k in loss_keys for k in val_keys]), (
                    f"Keys in val datasets do not match the keys in losses.\nMissing in losses: {set(val_keys) - loss_keys}\nMissing in val datasets: {loss_keys - set(val_keys)}"
                )

    def _setup_components(self):
        # Get the keys for all the val datasets, if any
        val_phase = Phase.VAL
        val_keys = None
        if self.data_conf.get(val_phase, None) is not None:
            val_keys = collect_dict_keys(self.data_conf[val_phase])
        # Additional checks on the sanity of the config for val datasets
        self._check_val_key_match(val_keys, phase=val_phase)

        logging.info("Setting up components: Model, loss, optim, meters etc.")
        self.epoch = 0
        self.steps = {Phase.TRAIN: 0, Phase.VAL: 0}

        self.logger = Logger(self.logging_conf)

        self.model = instantiate(self.model_conf, _convert_="all")

        # Style-based AUE: no initialization needed (styles extracted on-the-fly)

        print_model_summary(self.model)

        self.loss = None
        if self.loss_conf:
            self.loss = {
                key: el  # wrap_base_loss(el)
                for (key, el) in instantiate(self.loss_conf, _convert_="all").items()
            }
            self.loss = nn.ModuleDict(self.loss)

        self.meters = {}
        self.best_meter_values = {}
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")

        self.scaler = torch.amp.GradScaler(
            self.device,
            enabled=self.optim_conf.amp.enabled if self.optim_conf else False,
        )

        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip) if self.optim_conf else None
        self.gradient_logger = instantiate(self.optim_conf.gradient_logger) if self.optim_conf else None

        # Initialize gradient conflict monitor (for debugging AUE training)
        self._setup_gradient_conflict_monitor()

        logging.info("Finished setting up components: Model, loss, optim, meters etc.")

    def _setup_gradient_conflict_monitor(self):
        """Setup gradient conflict monitor for debugging AUE training.

        Configured via trainer.gradient_conflict_monitor in YAML:
            gradient_conflict_monitor:
              enabled: false
              sample_rate: 0.05
              target_params: ["bndl", "iou"]
        """
        self.gradient_conflict_monitor = None

        gc_config = getattr(self, "_gradient_conflict_monitor_conf", {}) or {}

        if not gc_config or not gc_config.get("enabled", False):
            logging.info("Gradient conflict monitor disabled (trainer.gradient_conflict_monitor.enabled=False)")
            return

        from training.utils.gradient_conflict import GradientConflictMonitor

        sample_rate = gc_config.get("sample_rate", 0.05)
        target_params = gc_config.get("target_params", ["bndl", "iou"])

        self.gradient_conflict_monitor = GradientConflictMonitor(
            model=self.model,
            target_params=target_params,
            sample_rate=sample_rate,
        )
        logging.info(f"Gradient conflict monitor enabled: sample_rate={sample_rate}, target_params={target_params}")

    def _setup_evaluators(self):
        """Setup evaluator for validation phase"""
        if not getattr(self.logging_conf, "enable_pavpu_eval", True):
            self.val_evaluator = None
            logging.info("Uncertainty evaluator disabled by config (logging.enable_pavpu_eval=False).")
            return

        # Get dilation from config (default: 15 pixels)
        foreground_dilation = getattr(self.logging_conf, "correlation_foreground_dilation", 15)

        # Create evaluator with simplified API
        from training.utils.dataset_evaluator import DistributedDatasetEvaluator

        self.val_evaluator = DistributedDatasetEvaluator(
            save_dir=os.path.join(self.logging_conf.log_dir, "uncertainty_eval"),
            distributed=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            foreground_dilation=foreground_dilation,
        )
        logging.info(f"Uncertainty evaluator initialized (dilation={foreground_dilation})")

    def _add_to_evaluator(self, bndl_outputs, targets, evaluator):
        """Add BNDL outputs to evaluator for PAvPU analysis"""
        try:
            uncertainty = bndl_outputs.get("pixel_uncertainty")
            pred_logits = bndl_outputs.get("mean_pixel_logits")

            # Normalize shapes for evaluator: uncertainty [B,H,W], logits [B,H,W,1]
            # Match training behavior: use channel average (consistent with loss computation and visualization)
            if isinstance(uncertainty, torch.Tensor) and uncertainty.ndim == 4:
                # [B,H,W,K] → prefer K=1, else use channel average (consistent with training)
                if uncertainty.shape[-1] == 1:
                    uncertainty = uncertainty.squeeze(-1)
                else:
                    uncertainty = uncertainty.mean(dim=-1)  # Cross-channel average, consistent with training
            if isinstance(pred_logits, torch.Tensor):
                if pred_logits.ndim == 3:
                    pred_logits = pred_logits.unsqueeze(-1)
                elif pred_logits.ndim == 4 and pred_logits.shape[-1] > 1:
                    pred_logits = pred_logits[..., 0:1]

            if uncertainty is not None and pred_logits is not None and targets is not None:
                evaluator.add_batch(uncertainty=uncertainty, pred_logits=pred_logits, gt_masks=targets)
        except Exception as e:
            logging.warning(f"Failed to add data to evaluator: {e}")

    def _construct_optimizers(self):
        # Enable anomaly detection to debug NaN gradients
        # torch.autograd.set_detect_anomaly(True)

        # === Exclude attacker params from main optimizer ===
        # Attacker networks are ALWAYS trained with a separate optimizer when they exist.
        # This ensures clean parameter isolation between SAM/BNDL and attacker networks.
        import fnmatch

        attacker_patterns = ["style_attacker.*", "deform_attacker.*", "style_gcn.*"]
        all_param_names = {name for name, _ in self.model.named_parameters()}
        excluded_params = set()
        for name in all_param_names:
            for pattern in attacker_patterns:
                if fnmatch.fnmatch(name, pattern):
                    excluded_params.add(name)
                    break

        if excluded_params:
            param_allowlist = all_param_names - excluded_params
            logging.info(f"[Main Optimizer] Excluding {len(excluded_params)} attacker params (handled by separate optimizer)")
        else:
            param_allowlist = None  # No attacker params to exclude

        self.optim = construct_optimizer(
            self.model,
            self.optim_conf.optimizer,
            self.optim_conf.options,
            self.optim_conf.param_group_modifiers,
            param_allowlist=param_allowlist,
            # Skip validation when using param_allowlist (attacker params handled separately)
            validate_param_groups=param_allowlist is None,
        )

        # === Separate Attacker Optimizer ===
        # Always create for attacker networks when they exist.
        self.attacker_optim = None
        self._construct_attacker_optimizer()

    def _construct_attacker_optimizer(self):
        """
        Construct separate optimizer for attacker networks (Style/Deform).

        This implements the "Separate Optimizer" pattern for isolated attacker training:
        - Main optimizer: updates SAM + BNDL (using core_loss)
        - Attacker optimizer: updates only Style/Deform networks (using _attacker_only_loss)

        Always created when attacker networks exist (AUE enabled).
        Uses CosineAnnealingLR scheduler matching the main optimizer's schedule.
        """
        import fnmatch

        model_unwrapped = unwrap_ddp_if_wrapped(self.model)

        # Collect attacker parameters
        attacker_patterns = ["style_attacker.*", "deform_attacker.*", "style_gcn.*"]
        attacker_params = []
        attacker_param_names = []

        for name, param in model_unwrapped.named_parameters():
            for pattern in attacker_patterns:
                if fnmatch.fnmatch(name, pattern):
                    if param.requires_grad:
                        attacker_params.append(param)
                        attacker_param_names.append(name)
                    break

        if not attacker_params:
            logging.info("[Attacker Optimizer] No attacker parameters found (AUE disabled)")
            return

        # Read config from optim_conf.attacker_optim (if available)
        attacker_optim_conf = getattr(self.optim_conf, "attacker_optim", None)
        if attacker_optim_conf is not None:
            lr_start = getattr(attacker_optim_conf, "lr_start", 1e-3)
            lr_end = getattr(attacker_optim_conf, "lr_end", 1e-4)
            weight_decay = getattr(attacker_optim_conf, "weight_decay", 0.0)
        else:
            # Fallback to legacy config or defaults
            lr_start = getattr(self.optim_conf, "attacker_lr", 1e-3)
            lr_end = lr_start / 10  # Default: 1/10 decay
            weight_decay = getattr(self.optim_conf, "attacker_weight_decay", 0.0)

        self.attacker_optim = torch.optim.AdamW(
            attacker_params,
            lr=lr_start,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Create CosineAnnealingLR scheduler
        # T_max = total training iterations (epochs * steps_per_epoch)
        # We'll step this scheduler alongside the main optimizer
        self.attacker_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.attacker_optim,
            T_max=self.max_epochs,  # Will be stepped per epoch
            eta_min=lr_end,
        )
        self._attacker_lr_start = lr_start
        self._attacker_lr_end = lr_end

        # Log component breakdown for ablation study clarity
        style_count = sum(1 for n in attacker_param_names if n.startswith("style_attacker."))
        deform_count = sum(1 for n in attacker_param_names if n.startswith("deform_attacker."))
        gcn_count = sum(1 for n in attacker_param_names if n.startswith("style_gcn."))
        components = []
        if style_count > 0:
            components.append(f"StyleAdv({style_count})")
        if deform_count > 0:
            components.append(f"DeformAdv({deform_count})")
        if gcn_count > 0:
            components.append(f"GCN({gcn_count})")

        logging.info(f"[Attacker Optimizer] Created with {len(attacker_params)} params [{', '.join(components)}], lr={lr_start}→{lr_end} (cosine), wd={weight_decay}")

    def _log_loss_detailed_and_return_core_loss(self, loss, loss_str, step, phase):
        core_loss = loss.pop(CORE_LOSS_KEY)
        if step % self.logging_conf.log_scalar_frequency == 0:
            for k in loss:
                # Log refinement stability metrics under dedicated namespace
                ref_idx = k.find("_refinement_")
                if ref_idx != -1:
                    metric_name = k[ref_idx + 1 :]  # Remove leading underscore
                    # Include phase in log path to separate train/val metrics
                    log_str = os.path.join("Refinement", f"{phase}_{metric_name}")
                else:
                    log_str = os.path.join(loss_str, k)
                self.logger.log(log_str, loss[k], step)
        return core_loss

    def _log_bndl_statistics(self, bndl_outputs, step, phase):
        """Log BNDL statistics including pixel-level uncertainty.

        Uses the global scalar logging frequency to avoid spamming tensorboard when
        running long validation epochs.
        """
        if bndl_outputs is None:
            return

        if step % self.logging_conf.log_scalar_frequency != 0:
            return

        bndl_prefix = "BNDL"
        stats_prefix = f"{bndl_prefix}/Stats"
        sign_prefix = f"{bndl_prefix}/Sign"
        gcn_prefix = f"{bndl_prefix}/GCN"

        # Pixel-level parameters (lambda and k)
        if bndl_outputs.get("wei_lambda_pos") is not None and bndl_outputs.get("kappa_pos") is not None:
            lambda_pos_mean = bndl_outputs["wei_lambda_pos"].mean().detach()
            k_pos_mean = bndl_outputs["kappa_pos"].mean().detach()
            self.logger.log(f"{stats_prefix}/{phase}_lambda_pixel_pos", lambda_pos_mean, step)
            self.logger.log(f"{stats_prefix}/{phase}_k_pixel_pos", k_pos_mean, step)

            lambda_sum = bndl_outputs["wei_lambda_pos"]
            k_sum = bndl_outputs["kappa_pos"]

            if bndl_outputs.get("wei_lambda_neg") is not None and bndl_outputs.get("kappa_neg") is not None:
                lambda_neg_mean = bndl_outputs["wei_lambda_neg"].mean().detach()
                k_neg_mean = bndl_outputs["kappa_neg"].mean().detach()
                self.logger.log(f"{stats_prefix}/{phase}_lambda_pixel_neg", lambda_neg_mean, step)
                self.logger.log(f"{stats_prefix}/{phase}_k_pixel_neg", k_neg_mean, step)
                lambda_sum = lambda_sum + bndl_outputs["wei_lambda_neg"]
                k_sum = 0.5 * (k_sum + bndl_outputs["kappa_neg"])

            self.logger.log(f"{stats_prefix}/{phase}_lambda_pixel", lambda_sum.mean().detach(), step)
            self.logger.log(f"{stats_prefix}/{phase}_k_pixel", k_sum.mean().detach(), step)

        elif bndl_outputs.get("wei_lambda") is not None and bndl_outputs.get("kappa") is not None:
            lambda_mean = bndl_outputs["wei_lambda"].mean().detach()
            k_mean = bndl_outputs["kappa"].mean().detach()
            self.logger.log(f"{stats_prefix}/{phase}_lambda_pixel", lambda_mean, step)
            self.logger.log(f"{stats_prefix}/{phase}_k_pixel", k_mean, step)

        # Log pixel uncertainty if available
        if bndl_outputs.get("pixel_uncertainty") is not None:
            uncertainty_mean = bndl_outputs["pixel_uncertainty"].mean().detach()
            self.logger.log(f"{stats_prefix}/{phase}_pixel_uncertainty", uncertainty_mean, step)

        # Global w statistics (mask_tokens_out)
        if bndl_outputs.get("wei_lambda_w_pos") is not None and bndl_outputs.get("kappa_w_pos") is not None:
            lambda_w_pos_mean = bndl_outputs["wei_lambda_w_pos"].mean().detach()
            k_w_pos_mean = bndl_outputs["kappa_w_pos"].mean().detach()
            self.logger.log(f"{stats_prefix}/{phase}_lambda_w_pos", lambda_w_pos_mean, step)
            self.logger.log(f"{stats_prefix}/{phase}_k_w_pos", k_w_pos_mean, step)

            lambda_w_sum = bndl_outputs["wei_lambda_w_pos"]
            k_w_sum = bndl_outputs["kappa_w_pos"]

            if bndl_outputs.get("wei_lambda_w_neg") is not None and bndl_outputs.get("kappa_w_neg") is not None:
                lambda_w_neg_mean = bndl_outputs["wei_lambda_w_neg"].mean().detach()
                k_w_neg_mean = bndl_outputs["kappa_w_neg"].mean().detach()
                self.logger.log(f"{stats_prefix}/{phase}_lambda_w_neg", lambda_w_neg_mean, step)
                self.logger.log(f"{stats_prefix}/{phase}_k_w_neg", k_w_neg_mean, step)
                lambda_w_sum = lambda_w_sum + bndl_outputs["wei_lambda_w_neg"]
                k_w_sum = 0.5 * (k_w_sum + bndl_outputs["kappa_w_neg"])

            self.logger.log(f"{stats_prefix}/{phase}_lambda_w", lambda_w_sum.mean().detach(), step)
            self.logger.log(f"{stats_prefix}/{phase}_k_w", k_w_sum.mean().detach(), step)

        elif bndl_outputs.get("wei_lambda_w") is not None and bndl_outputs.get("kappa_w") is not None:
            lambda_w_mean = bndl_outputs["wei_lambda_w"].mean().detach()
            k_w_mean = bndl_outputs["kappa_w"].mean().detach()
            self.logger.log(f"{stats_prefix}/{phase}_lambda_w", lambda_w_mean, step)
            self.logger.log(f"{stats_prefix}/{phase}_k_w", k_w_mean, step)

        # Sign statistics (for symmetric modeling)
        lgamma_cache = bndl_outputs.get("lgamma_cache") if isinstance(bndl_outputs, dict) else None
        if isinstance(lgamma_cache, dict):
            sign_x = lgamma_cache.get("sign_x")
            if sign_x is not None:
                sign_abs_mean_z = sign_x.detach().abs().mean()
                self.logger.log(f"{sign_prefix}/{phase}_sign_abs_mean_z", sign_abs_mean_z, step)
            sign_w = lgamma_cache.get("sign_w")
            if sign_w is not None:
                sign_abs_mean_w = sign_w.detach().abs().mean()
                self.logger.log(f"{sign_prefix}/{phase}_sign_abs_mean_w", sign_abs_mean_w, step)

        # Style GCN statistics (if available)
        gcn_stats = bndl_outputs.get("gcn_stats") if isinstance(bndl_outputs, dict) else None
        if gcn_stats:
            for key, value in gcn_stats.items():
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
                self.logger.log(f"{gcn_prefix}/{phase}_{key}", scalar, step)

    def _log_aue_metrics_from_outputs(self, outputs, step: int, phase: str):
        """Legacy method - AUE metrics are now logged via loss_combined.py.

        This method is kept as a no-op for backward compatibility.
        All AUE loss values are now returned by loss_combined.py and logged under Losses/ prefix.
        """
        pass

    def _log_style_aue_visualization(self, outputs, step):
        """Log Style AUE visualization: original vs adversarial images with style statistics"""
        # Extract visualization data from outputs
        vis_data = self._extract_vis_data_from_outputs(outputs)
        if vis_data is None:
            return

        # Denormalize images
        # If warped images are available (from deformation), use them as "before" for StyleAUE
        # This shows: deformed image -> styled image (the actual StyleAUE input/output)
        # Use the stored original and styled images directly (no fallback)
        original_denorm = self._denormalize_images(vis_data.original_images)
        adv_denorm = self._denormalize_images(vis_data.adversarial_images)

        # Use stored styles if available, otherwise extract from images
        # These are per-object styles from GT regions [N, K, 6]
        if vis_data.original_styles is not None and vis_data.adversarial_styles is not None:
            original_styles = vis_data.original_styles  # [N, K, 6] per-object styles
            adv_styles = vis_data.adversarial_styles  # [N, K, 6] per-object styles

            # For single-sample visualization, use global style (average across objects)
            original_styles_global = original_styles.mean(dim=1)  # [N, 6]
            adv_styles_global = adv_styles.mean(dim=1)  # [N, 6]

            # Log per-object style perturbations (this is the NEW meaningful metric)
            self._log_style_perturbations_per_object(original_styles, adv_styles, vis_data, step)

            # Use step % batch_size for diversity while ensuring consistency across visualizations
            batch_size = original_denorm.shape[0]
            sample_idx = step % batch_size

            # Log Style Delta Overlay and Object Graph Overlay (for paper figures)
            self._log_style_delta_overlay(original_denorm, original_styles, adv_styles, vis_data, sample_idx, step)
            self._log_object_graph_overlay(original_denorm, vis_data, sample_idx, step)

            # Log GCN comparison: generate no-GCN version for comparison
            self._log_gcn_comparison_visualization(original_denorm, original_styles, adv_styles, vis_data, sample_idx, step)
        else:
            from sam2.modeling.style_utils import extract_style_statistics

            # Fallback: extract global image style
            original_styles_global = extract_style_statistics(vis_data.original_images).mean(dim=1)
            adv_styles_global = extract_style_statistics(vis_data.adversarial_images).mean(dim=1)
            original_styles = None
            adv_styles = None

            # Use step % batch_size for diversity
            batch_size = original_denorm.shape[0]
            sample_idx = step % batch_size

        self._visualize_style_single_sample(
            original_denorm[sample_idx], adv_denorm[sample_idx], original_styles_global[sample_idx], adv_styles_global[sample_idx], vis_data, sample_idx=sample_idx, step=step
        )

        # Log global style perturbation statistics (for backward compatibility)
        self._log_style_perturbations(original_styles_global, adv_styles_global, step)

        # Log GCN parameters if available
        self._log_gcn_parameters(step)

    def _extract_vis_data_from_outputs(self, outputs):
        """Extract visualization data from model outputs."""
        import logging

        outputs_iter = [outputs] if isinstance(outputs, dict) else outputs

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"DeformAUE: _extract_vis_data_from_outputs - outputs type={type(outputs)}, is_dict={isinstance(outputs, dict)}")

        for outs in outputs_iter:
            if "multistep_aux_outputs" not in outs:
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug("DeformAUE: No 'multistep_aux_outputs' in outputs")
                continue

            aux_list = outs["multistep_aux_outputs"]
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"DeformAUE: Found multistep_aux_outputs with {len(aux_list)} entries")

            for aux in reversed(aux_list):
                if aux is None or not isinstance(aux, dict):
                    continue

                bndl_outputs = aux.get("bndl", None)
                if bndl_outputs is None:
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug("DeformAUE: No 'bndl' in aux_outputs")
                    continue

                # Use aue_metrics directly
                aue_metrics = bndl_outputs.get("aue_metrics")
                if aue_metrics is None:
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(f"DeformAUE: No 'aue_metrics' in bndl_outputs, keys={list(bndl_outputs.keys())}")
                    continue

                vis_data = aue_metrics.get("aue_visualization", None)

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"DeformAUE: aue_metrics keys={list(aue_metrics.keys())}, vis_data={'✓' if vis_data is not None else '✗'}")

                # Check if vis_data is AUEVisualizationData or dict
                if vis_data is not None:
                    # If it's a dataclass, check if it has data
                    if hasattr(vis_data, "original_images"):
                        # Debug: log what data is available
                        import logging

                        if logging.getLogger().isEnabledFor(logging.DEBUG):
                            has_deform_offsets = hasattr(vis_data, "deform_offsets") and vis_data.deform_offsets is not None
                            logging.debug(f"DeformAUE: vis_data found - deform_offsets={'✓' if has_deform_offsets else '✗'}")
                        return vis_data
                    # If it's a dict (legacy), check for keys
                    elif isinstance(vis_data, dict) and "original_images" in vis_data:
                        return vis_data

        return None

    def _denormalize_images(self, images):
        """Denormalize images from ImageNet normalization."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        denorm = images * std + mean
        return torch.clamp(denorm, 0, 1)

    def _save_image_to_disk(self, image, step, category, name):
        """Save an image directly to disk under the log directory.

        Images are saved to: {log_dir}/vis/{step}/{category}/{name}.png

        Args:
            image: Can be one of:
                - torch.Tensor [C, H, W] or [H, W, C] with values in [0, 1]
                - numpy.ndarray [H, W, C] with values in [0, 1] or [0, 255]
                - matplotlib figure (will be rendered)
            step: Training step number
            category: Category subdirectory (e.g., 'clean', 'adv', 'StyleAUE', 'DeformAUE')
            name: Image filename without extension
        """
        from PIL import Image as PILImage

        # Build the save path: {log_dir}/vis/{step}/{category}/{name}.png
        save_dir = os.path.join(self.logging_conf.log_dir, "vis", str(step), category)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{name}.png")

        try:
            # Convert to numpy array [H, W, C] in uint8
            if isinstance(image, torch.Tensor):
                img_np = image.detach().cpu()
                # Handle different tensor shapes
                if img_np.ndim == 3:
                    if img_np.shape[0] in [1, 3]:  # [C, H, W]
                        img_np = img_np.permute(1, 2, 0)  # -> [H, W, C]
                    # else assume [H, W, C]
                elif img_np.ndim == 2:  # [H, W] grayscale
                    img_np = img_np.unsqueeze(-1).expand(-1, -1, 3)  # -> [H, W, 3]
                img_np = img_np.numpy()
            elif isinstance(image, plt.Figure):
                # Render matplotlib figure to numpy
                image.canvas.draw()
                w, h = image.canvas.get_width_height()
                img_np = np.frombuffer(image.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
            else:
                img_np = np.asarray(image)

            # Ensure [H, W, C] shape
            if img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=-1)
            elif img_np.ndim == 3 and img_np.shape[0] in [1, 3] and img_np.shape[2] not in [1, 3]:
                # Likely [C, H, W], transpose
                img_np = np.transpose(img_np, (1, 2, 0))

            # Handle single-channel (grayscale) images
            if img_np.shape[-1] == 1:
                img_np = np.repeat(img_np, 3, axis=-1)

            # Convert to uint8 [0, 255]
            if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            elif img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)

            # Save using PIL
            pil_img = PILImage.fromarray(img_np)
            pil_img.save(save_path)

        except Exception as e:
            logging.debug(f"Failed to save image {save_path}: {e}")

    def _visualize_style_single_sample(self, original_img, adv_img, original_style, adv_style, vis_data, sample_idx, step, pred_mask=None):
        """Visualize a single sample for Style AUE with multi-object support."""
        # Extract multi-object data from vis_data (support new AUEVisualizationData dataclass)
        # New field names from refactored AUE visualization module
        if hasattr(vis_data, "object_bboxes"):
            # New AUEVisualizationData dataclass
            all_bboxes = vis_data.object_bboxes
            all_masks = vis_data.object_masks
            combined_mask = None  # Not stored in new dataclass (can be computed from object_masks if needed)
            area_ratios = vis_data.object_area_ratios
            epsilon_weights = None  # Not stored in new dataclass (no longer used for visualization)
        elif hasattr(vis_data, "all_bboxes"):
            # Legacy dataclass format (backward compatibility)
            all_bboxes = vis_data.all_bboxes
            all_masks = vis_data.all_object_masks
            combined_mask = getattr(vis_data, "combined_mask_for_loss", None)
            area_ratios = vis_data.area_ratios
            epsilon_weights = getattr(vis_data, "epsilon_weights", None)
        else:
            all_bboxes = None
            all_masks = None
            combined_mask = None
            area_ratios = None
            epsilon_weights = None

        # Extract data for this sample
        sample_bboxes = all_bboxes[sample_idx] if all_bboxes is not None else None
        sample_masks = all_masks[sample_idx] if all_masks is not None else None
        sample_combined_mask = combined_mask[sample_idx, 0] if combined_mask is not None and combined_mask.ndim >= 2 else None
        sample_area_ratios = area_ratios[sample_idx] if area_ratios is not None else None
        sample_epsilon_weights = epsilon_weights[sample_idx] if epsilon_weights is not None else None

        # Extract loss_object_idx for this sample (the object being trained)
        sample_loss_obj_idx = None
        if hasattr(vis_data, "loss_object_indices") and vis_data.loss_object_indices is not None:
            if sample_idx < len(vis_data.loss_object_indices):
                idx_val = vis_data.loss_object_indices[sample_idx]
                sample_loss_obj_idx = idx_val.item() if hasattr(idx_val, "item") else int(idx_val)

        # For single-object visualization, also pass bbox/mask so TensorBoard overlay can highlight the attack region
        single_obj_bbox = None
        single_obj_mask = None
        primary_idx = sample_loss_obj_idx if sample_loss_obj_idx is not None else 0
        if sample_masks is not None and sample_masks.shape[0] > primary_idx:
            single_obj_mask = sample_masks[primary_idx]
        elif sample_combined_mask is not None:
            single_obj_mask = sample_combined_mask
        if sample_bboxes is not None and sample_bboxes.shape[0] > primary_idx:
            single_obj_bbox = sample_bboxes[primary_idx]

        # Call unified visualization
        self._log_style_statistics_overlay(
            original_img,
            adv_img,
            original_style,
            adv_style,
            sample_idx,
            step,
            bbox=single_obj_bbox,
            gt_mask=single_obj_mask,
            all_bboxes=sample_bboxes,
            all_masks=sample_masks,
            combined_mask=sample_combined_mask,
            area_ratios=sample_area_ratios,
            epsilon_weights=sample_epsilon_weights,
            loss_object_idx=sample_loss_obj_idx,
            pred_mask=pred_mask,
        )

    def _log_style_perturbations(self, original_styles, adv_styles, step):
        """Log global style perturbation statistics (for backward compatibility)."""
        style_diff = (adv_styles - original_styles).abs().mean(dim=0)  # [6]
        channels = ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]
        for j, channel in enumerate(channels):
            self.logger.log(f"StyleAUE/global_perturbation_{channel}", style_diff[j].item(), step)

    def _log_style_perturbations_per_object(self, original_styles, adv_styles, vis_data, step):
        """
        Log per-object style perturbation statistics in attacked regions.

        This is the more meaningful metric: how much does the attack change the style
        in each GT region, weighted by object area.

        Args:
            original_styles: [N, K, 6] original style statistics per object
            adv_styles: [N, K, 6] adversarial style statistics per object
            vis_data: visualization data containing pixel_gt for area computation
            step: logging step
        """
        if original_styles is None or adv_styles is None:
            return

        N, K, _ = original_styles.shape

        # Per-object style difference: [N, K, 6]
        style_diff = (adv_styles - original_styles).abs()

        # Use pre-computed area ratios if available (more efficient)
        # AUEVisualizationData from visualization.py has: object_area_ratios [N, K]
        area_ratios = getattr(vis_data, "object_area_ratios", None)

        if area_ratios is not None and area_ratios.shape[1] == K:
            # Use pre-computed area ratios as weights
            # Normalize so they sum to 1 per sample
            area_weights = area_ratios / (area_ratios.sum(dim=1, keepdim=True) + 1e-8)
            # Weighted average style diff: [N, 6]
            weighted_diff = (style_diff * area_weights.unsqueeze(-1)).sum(dim=1)
        elif hasattr(vis_data, "object_masks") and vis_data.object_masks is not None:
            # Fallback: compute from masks
            pixel_gt = vis_data.object_masks  # [N, K, H, W]
            if pixel_gt.ndim == 3:
                pixel_gt = pixel_gt.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]

            # Compute area per object: [N, K]
            object_areas = (pixel_gt > 0.5).float().sum(dim=[2, 3])
            total_area = object_areas.sum(dim=1, keepdim=True) + 1e-8  # [N, 1]

            # Normalized weights: [N, K]
            area_weights = object_areas / total_area

            # Weighted average style diff: [N, 6]
            weighted_diff = (style_diff * area_weights.unsqueeze(-1)).sum(dim=1)
        else:
            # Fallback: simple average across objects
            weighted_diff = style_diff.mean(dim=1)  # [N, 6]

        # Average across batch: [6]
        mean_diff = weighted_diff.mean(dim=0)

        # Log per-channel metrics
        channels = ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]
        for j, channel in enumerate(channels):
            self.logger.log(f"StyleAUE/region_perturbation_{channel}", mean_diff[j].item(), step)

        # Log aggregate metrics
        mean_perturbation = mean_diff[:3].mean().item()  # Mean shift (channels 0-2)
        std_perturbation = mean_diff[3:].mean().item()  # Std shift (channels 3-5)
        total_perturbation = mean_diff.mean().item()  # Overall

        self.logger.log("StyleAUE/region_perturbation_mean_total", mean_perturbation, step)
        self.logger.log("StyleAUE/region_perturbation_std_total", std_perturbation, step)
        self.logger.log("StyleAUE/region_perturbation_total", total_perturbation, step)

        # Log per-object breakdown (sampled for efficiency)
        if K <= 5:  # Only log if reasonable number of objects
            for k in range(K):
                obj_diff = style_diff[:, k, :].mean(dim=0)  # [6] averaged over batch
                self.logger.log(f"StyleAUE/object_{k}_perturbation", obj_diff.mean().item(), step)

    def _log_style_delta_overlay(self, original_denorm, original_styles, adv_styles, vis_data, sample_idx, step):
        """Log Style Delta Overlay: per-object style perturbation intensity on original image.

        Creates a single image with color-coded overlay showing |Δμ| + |Δσ| intensity.
        NO TITLE - legend inside image for paper figures.
        """

        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import numpy as np

        if original_styles is None or adv_styles is None:
            return
        if not hasattr(vis_data, "object_masks") or vis_data.object_masks is None:
            return

        # Get data for this sample
        orig_img = original_denorm[sample_idx].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        orig_img = np.clip(orig_img, 0, 1)
        H, W = orig_img.shape[:2]

        sample_masks = vis_data.object_masks[sample_idx]  # [K, H, W]
        sample_orig_styles = original_styles[sample_idx]  # [K, 6]
        sample_adv_styles = adv_styles[sample_idx]  # [K, 6]

        K = sample_masks.shape[0]

        # Compute style delta for each object: |Δμ| + |Δσ|
        style_diff = (sample_adv_styles - sample_orig_styles).abs()  # [K, 6]
        delta_mu = style_diff[:, :3].mean(dim=1)  # [K] mean of RGB means
        delta_sigma = style_diff[:, 3:].mean(dim=1)  # [K] mean of RGB stds
        delta_total = (delta_mu + delta_sigma).cpu().numpy()  # [K]

        # Normalize to [0, 1] for colormap
        if delta_total.max() > delta_total.min():
            delta_norm = (delta_total - delta_total.min()) / (delta_total.max() - delta_total.min() + 1e-8)
        else:
            delta_norm = np.zeros_like(delta_total)

        # Create overlay - no margins
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(orig_img)
        ax.axis("off")

        # Create colormap (blue=low, yellow=mid, red=high)
        cmap = plt.cm.get_cmap("RdYlBu_r")  # Reversed: blue=low, red=high

        # Overlay each object mask with color based on delta intensity
        model = unwrap_ddp_if_wrapped(self.model)
        include_background = bool(getattr(model, "adv_enable_background", False))

        for k in range(K):
            # Skip background if it's the last channel and not visualizing background
            if not include_background and k == K - 1:
                continue

            mask_k = sample_masks[k].cpu().numpy()  # [H, W]
            if mask_k.shape != (H, W):
                # Resize mask if needed
                from skimage.transform import resize

                mask_k = resize(mask_k, (H, W), order=0, preserve_range=True)

            if mask_k.sum() < 10:  # Skip empty masks
                continue

            # Create colored overlay
            color = cmap(delta_norm[k])[:3]  # RGB
            overlay = np.zeros((H, W, 4))
            overlay[mask_k > 0.5, :3] = color
            overlay[mask_k > 0.5, 3] = 0.4  # Alpha

            ax.imshow(overlay)

        # Add colorbar INSIDE image (bottom-right corner)
        axins = inset_axes(ax, width="30%", height="3%", loc="lower right", borderpad=2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=axins, orientation="horizontal")
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["Low", "", "High"], fontsize=8)
        cbar.ax.tick_params(labelsize=8, colors="white")
        axins.set_title("|Δμ|+|Δσ|", fontsize=8, color="white", pad=2)

        # Remove all margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save to disk
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        self._save_image_to_disk(img_array, step, "StyleAUE", "style_delta_overlay")

        plt.close(fig)

    def _log_object_graph_overlay(self, original_denorm, vis_data, sample_idx, step):
        """Log Object Graph Overlay: nodes at centroids, edges showing object relationships.

        Creates a single image with graph overlay and mask contours.
        NO TITLE - legend inside image for paper figures.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        from skimage import measure

        if not hasattr(vis_data, "object_masks") or vis_data.object_masks is None:
            return

        # Get model for GCN config
        model = unwrap_ddp_if_wrapped(self.model)

        # Get data for this sample
        orig_img = original_denorm[sample_idx].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        orig_img = np.clip(orig_img, 0, 1)
        H, W = orig_img.shape[:2]

        sample_masks = vis_data.object_masks[sample_idx]  # [K, H, W]
        K = sample_masks.shape[0]

        include_background = bool(getattr(model, "adv_enable_background", False))

        # Create figure - no margins
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(orig_img)
        ax.axis("off")

        # Define colors for objects
        colors = plt.cm.tab10(np.linspace(0, 1, max(K, 10)))

        # Compute centroids and draw mask contours for each object
        centroids = []
        valid_objects = []
        for k in range(K):
            # Skip background if it's the last channel and not visualizing background
            if not include_background and k == K - 1:
                continue

            mask_k = sample_masks[k].cpu().numpy().astype(np.float32)  # [H, W]
            if mask_k.shape != (H, W):
                from skimage.transform import resize

                mask_k = resize(mask_k, (H, W), order=0, preserve_range=True)

            area = mask_k.sum()
            if area < 10:  # Skip empty masks
                continue

            # Draw mask contour
            node_color = colors[k % 10]
            contours = measure.find_contours(mask_k, 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color=node_color, linewidth=2, alpha=0.8)

            # Compute centroid
            y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            cy = (mask_k * y_coords).sum() / area
            cx = (mask_k * x_coords).sum() / area
            centroids.append((cx, cy, k))
            valid_objects.append(k)

        if len(centroids) < 1:
            plt.close(fig)
            return  # Need at least 1 object

        # Use ACTUAL build_object_graph function for accurate GCN edges
        from sam2.modeling.style_gcn import build_object_graph

        # Get GCN config from model
        edge_threshold = getattr(model, "style_adv_gcn_edge_threshold", 0.3)
        distance_threshold = getattr(model, "style_adv_gcn_distance_threshold", 0.2)
        use_semantic = getattr(model, "style_adv_gcn_use_semantic_edges", False)
        use_boundary_distance = getattr(model, "style_adv_gcn_use_boundary_distance", True)
        use_background_edges = include_background and getattr(model, "style_adv_gcn_use_background_edges", False)

        # Build graph for this sample (batch size = 1)
        sample_masks_4d = sample_masks.unsqueeze(0)  # [1, K, H, W]
        edge_index, edge_weight, gcn_stats = build_object_graph(
            sample_masks_4d,
            img_batch=None,
            edge_threshold=edge_threshold,
            use_semantic=use_semantic,
            use_background=use_background_edges,
            distance_threshold=distance_threshold,
            use_boundary_distance=use_boundary_distance,
        )

        # Parse edges from GCN's edge_index and edge_weight
        edges = []
        if edge_index.numel() > 0:
            edge_index_np = edge_index.cpu().numpy()  # [2, E]
            edge_weight_np = edge_weight.cpu().numpy()  # [E]

            # Create mapping from node index to centroid index
            node_to_centroid = {k: i for i, (_, _, k) in enumerate(centroids)}

            # Collect edges (skip self-loops and duplicates)
            drawn_edges = set()
            for e in range(edge_index_np.shape[1]):
                src, tgt = edge_index_np[0, e], edge_index_np[1, e]
                if src == tgt:
                    continue  # Skip self-loops

                # Normalize to single-sample indices (batch=1)
                src_k = src % K
                tgt_k = tgt % K

                # Skip if either node is not in our valid list
                if src_k not in node_to_centroid or tgt_k not in node_to_centroid:
                    continue

                # Skip duplicates (undirected)
                edge_key = tuple(sorted([src_k, tgt_k]))
                if edge_key in drawn_edges:
                    continue
                drawn_edges.add(edge_key)

                edges.append((node_to_centroid[src_k], node_to_centroid[tgt_k], edge_weight_np[e]))

        # Draw edges
        for i, j, weight in edges:
            cx_i, cy_i, _ = centroids[i]
            cx_j, cy_j, _ = centroids[j]

            # Edge color and width based on weight
            edge_color = plt.cm.Greens(0.3 + 0.7 * min(weight, 1.0))  # Green intensity
            line_width = 1 + 4 * min(weight, 1.0)  # 1-5 width

            ax.plot([cx_i, cx_j], [cy_i, cy_j], color=edge_color, linewidth=line_width, alpha=0.8, zorder=1)

            # Add weight label at midpoint
            mid_x, mid_y = (cx_i + cx_j) / 2, (cy_i + cy_j) / 2
            ax.text(mid_x, mid_y, f"{weight:.2f}", fontsize=8, color="white", ha="center", va="center", fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

        # Draw nodes
        for i, (cx, cy, k) in enumerate(centroids):
            node_color = colors[k % 10]
            circle = plt.Circle((cx, cy), radius=min(H, W) * 0.025, color=node_color, ec="white", linewidth=2, zorder=2)
            ax.add_patch(circle)
            ax.text(cx, cy, str(k), fontsize=9, color="white", ha="center", va="center", fontweight="bold", zorder=3)

        # Add legend INSIDE image (upper right corner)
        edge_info = []
        if gcn_stats:
            if gcn_stats.get("edges_iou", 0) > 0:
                edge_info.append("IoU")
            if gcn_stats.get("edges_distance", 0) > 0:
                edge_info.append("Dist")
            if gcn_stats.get("edges_semantic", 0) > 0:
                edge_info.append("Sem")
        edge_label = "Edge: " + "+".join(edge_info) if edge_info else "Edge"

        legend_elements = [
            mpatches.Patch(facecolor="green", alpha=0.6, label=edge_label),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="Node"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.7, facecolor="black", labelcolor="white")

        # Remove all margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save to disk
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        self._save_image_to_disk(img_array, step, "StyleAUE", "object_graph_overlay")

        plt.close(fig)

    def _log_gcn_comparison_visualization(self, original_denorm, original_styles, adv_styles, vis_data, sample_idx, step):
        """Generate no-GCN style perturbation visualization for comparison.

        Outputs separate pure images:
        - styled_image_with_gcn.png: styled image using GCN-coordinated perturbations
        - styled_image_no_gcn.png: styled image using uncoordinated perturbations
        - style_delta_overlay_no_gcn.png: style delta heatmap without GCN coordination
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import numpy as np

        model = unwrap_ddp_if_wrapped(self.model)

        # Check if GCN is enabled and we have the necessary components
        gcn_enabled = getattr(model, "style_adv_use_gcn", False)
        has_gcn = model.style_gcn is not None
        logging.info(f"[GCN Comparison] step={step}, gcn_enabled={gcn_enabled}, has_gcn={has_gcn}")

        if not gcn_enabled or not has_gcn:
            logging.info(f"[GCN Comparison] Skipping: GCN not enabled or not available")
            return  # GCN not enabled, no comparison needed

        has_masks = hasattr(vis_data, "object_masks") and vis_data.object_masks is not None
        has_images = hasattr(vis_data, "original_images") and vis_data.original_images is not None
        logging.info(f"[GCN Comparison] has_masks={has_masks}, has_images={has_images}")

        if not has_masks:
            logging.info(f"[GCN Comparison] Skipping: no object_masks in vis_data")
            return
        if not has_images:
            logging.info(f"[GCN Comparison] Skipping: no original_images in vis_data")
            return

        # Get the style attacker
        style_attacker = getattr(model, "style_attacker", None)
        has_impl = style_attacker is not None and hasattr(style_attacker, "impl")
        logging.info(f"[GCN Comparison] has_style_attacker={style_attacker is not None}, has_impl={has_impl}")

        if style_attacker is None or not has_impl:
            logging.info(f"[GCN Comparison] Skipping: no style_attacker or impl")
            return

        logging.info(f"[GCN Comparison] All checks passed, generating comparison images...")
        try:
            # === 1. Compute GLOBAL style perturbation (same delta for all objects) ===
            # This contrasts with GCN-coordinated per-object perturbations

            with torch.no_grad():
                # Ensure images are on the correct device (may be on CPU from vis_data)
                device = next(model.parameters()).device
                original_images = vis_data.original_images
                if original_images.device != device:
                    original_images = original_images.to(device)
                    logging.info(f"[GCN Comparison] Moved original_images to {device}")

                # Get backbone features for the original images
                backbone_out = model.forward_image(original_images, use_checkpoint=False)
                clean_features = backbone_out["backbone_fpn"][-1]  # [B, C, H, W]

                # Get pixel_gt from vis_data
                pixel_gt = vis_data.object_masks  # [B, K, H, W]
                if pixel_gt.device != device:
                    pixel_gt = pixel_gt.to(device)
                if pixel_gt.ndim == 3:
                    pixel_gt = pixel_gt.unsqueeze(1)

                # Use the style network directly
                style_impl = style_attacker.impl

                # === GLOBAL ATTACK: Merge all objects into single mask ===
                # This simulates global style attack (no per-object differentiation)
                B, K, H_mask, W_mask = pixel_gt.shape
                pixel_gt_global = pixel_gt.sum(dim=1, keepdim=True).clamp(0, 1)  # [B, 1, H, W]

                # Extract global style (average across entire image, not per-object)
                from sam2.modeling.style_utils import extract_style_statistics

                original_styles_global = extract_style_statistics(original_images)  # [B, 1, 6]

                # Get global style perturbation (single style for whole image)
                adv_styles_global = style_impl.style_net(clean_features.detach(), original_styles_global, pixel_gt=pixel_gt_global)  # [B, 1, 6]

                # Apply global style to entire image (no per-object regions)
                styled_img_global = style_impl._apply_style_to_images(original_images, adv_styles_global, gt_mask=pixel_gt_global)

                # Apply styles with GCN (already computed, stored in vis_data.adversarial_images)
                styled_img_with_gcn = vis_data.adversarial_images
                if styled_img_with_gcn.device != device:
                    styled_img_with_gcn = styled_img_with_gcn.to(device)

            # === 2. Save styled images as pure images ===
            # Denormalize images
            styled_global_denorm = self._denormalize_images(styled_img_global)
            styled_with_gcn_denorm = self._denormalize_images(styled_img_with_gcn)

            # Log style delta comparison (Global vs GCN-coordinated per-object)
            style_delta_global = (adv_styles_global - original_styles_global).abs()
            style_delta_with_gcn = (adv_styles - original_styles).abs()
            logging.info(f"[GCN Comparison] Style Delta (GLOBAL):    mean={style_delta_global.mean().item():.4f}, max={style_delta_global.max().item():.4f}")
            logging.info(f"[GCN Comparison] Style Delta (GCN multi): mean={style_delta_with_gcn.mean().item():.4f}, max={style_delta_with_gcn.max().item():.4f}")

            # Get sample
            sample_styled_global = styled_global_denorm[sample_idx].permute(1, 2, 0).cpu().numpy()
            sample_styled_with_gcn = styled_with_gcn_denorm[sample_idx].permute(1, 2, 0).cpu().numpy()
            sample_styled_global = np.clip(sample_styled_global, 0, 1)
            sample_styled_with_gcn = np.clip(sample_styled_with_gcn, 0, 1)

            # Save pure styled images (no extra layout)
            self._save_image_to_disk(sample_styled_with_gcn, step, "StyleAUE", "styled_image_gcn_multi")
            self._save_image_to_disk(sample_styled_global, step, "StyleAUE", "styled_image_global")

            # === 3. Generate style delta overlay for GLOBAL version ===
            # For global attack, all objects have the SAME delta (since there's only 1 style)
            orig_img = original_denorm[sample_idx].permute(1, 2, 0).cpu().numpy()
            orig_img = np.clip(orig_img, 0, 1)
            H, W = orig_img.shape[:2]

            sample_masks = vis_data.object_masks[sample_idx]  # [K, H, W]
            K = sample_masks.shape[0]

            # Global attack: same delta for all objects
            global_delta = (adv_styles_global[sample_idx, 0] - original_styles_global[sample_idx, 0]).abs().cpu()  # [6]
            delta_mu_global = global_delta[:3].mean()
            delta_sigma_global = global_delta[3:].mean()
            delta_total_global = (delta_mu_global + delta_sigma_global).item()

            # All objects get the same value (uniform color in heatmap)
            delta_total_per_obj = np.full(K, delta_total_global)

            # For global attack, normalization is trivial (all same value)
            # Use fixed color to indicate uniform perturbation
            delta_norm = np.ones(K) * 0.5  # Middle color for all objects (uniform)

            # Create overlay figure (same style as _log_style_delta_overlay)
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(orig_img)
            ax.axis("off")

            cmap = plt.cm.get_cmap("RdYlBu_r")
            include_background = bool(getattr(model, "adv_enable_background", False))

            for k in range(K):
                if not include_background and k == K - 1:
                    continue

                mask_k = sample_masks[k].cpu().numpy()
                if mask_k.shape != (H, W):
                    from skimage.transform import resize

                    mask_k = resize(mask_k, (H, W), order=0, preserve_range=True)

                if mask_k.sum() < 10:
                    continue

                color = cmap(delta_norm[k])[:3]
                overlay = np.zeros((H, W, 4))
                overlay[mask_k > 0.5, :3] = color
                overlay[mask_k > 0.5, 3] = 0.4

                ax.imshow(overlay)

            # Add colorbar INSIDE image
            axins = inset_axes(ax, width="30%", height="3%", loc="lower right", borderpad=2)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=axins, orientation="horizontal")
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels(["Low", "", "High"], fontsize=8)
            cbar.ax.tick_params(labelsize=8, colors="white")
            axins.set_title("|Δμ|+|Δσ| (Global)", fontsize=8, color="white", pad=2)

            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            self._save_image_to_disk(img_array, step, "StyleAUE", "style_delta_overlay_global")

            plt.close(fig)

        except Exception as e:
            logging.debug(f"GCN comparison visualization failed: {e}")
            import traceback

            logging.debug(traceback.format_exc())

    def _log_deform_aue_visualization(self, outputs, step):
        """Log Deformation AUE visualization: before/after predictions with offset fields"""
        import logging

        # Debug: Check all conditions
        _model = unwrap_ddp_if_wrapped(self.model)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(
                f"DeformAUE: Attempting to log at step {step} - "
                f"visualize_aue={getattr(self.logging_conf, 'visualize_aue', False)}, "
                f"use_deform_aug={hasattr(_model, 'use_deform_aug') and _model.use_deform_aug}, "
                f"_enable_style_visualization={getattr(_model, '_enable_style_visualization', False)}, "
                f"distributed_rank={self.distributed_rank}"
            )

        # Extract visualization data from outputs
        vis_data = self._extract_vis_data_from_outputs(outputs)
        if vis_data is None:
            logging.debug("DeformAUE: vis_data is None, skipping visualization")
            return

        # Check if deformation data is available (we need offsets for visualization)
        if vis_data.deform_offsets is None:
            logging.debug("DeformAUE: deform_offsets not available, skipping visualization")
            return

        # Denormalize images
        original_denorm = self._denormalize_images(vis_data.original_images)

        # Use step % batch_size for diversity while ensuring consistency across visualizations
        batch_size = original_denorm.shape[0]
        sample_idx = step % batch_size

        # Extract prediction for this sample
        pred_mask = self._extract_prediction(outputs, batch_idx=sample_idx)

        self._visualize_deform_single_sample(original_denorm[sample_idx], vis_data, sample_idx=sample_idx, step=step, pred_mask=pred_mask)

        # Log deformation statistics (offsets are guaranteed to exist at this point)
        self._log_deform_statistics(vis_data.deform_offsets, step)

    def _visualize_deform_single_sample(self, original_img, vis_data, sample_idx, step, pred_mask=None):
        """Visualize deformation with 2x2 layout (matching StyleAUE format)."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        model = unwrap_ddp_if_wrapped(self.model)
        include_background = bool(getattr(model, "adv_enable_background", False))

        # Extract multi-object data from vis_data (support new AUEVisualizationData dataclass)
        if hasattr(vis_data, "object_bboxes"):
            # New AUEVisualizationData dataclass
            all_bboxes = vis_data.object_bboxes
            orig_masks = vis_data.object_masks
            warped_masks_all = getattr(vis_data, "warped_object_masks", None)
            area_ratios = vis_data.object_area_ratios
            epsilon_weights = None  # Not stored in new dataclass
        elif hasattr(vis_data, "all_bboxes"):
            # Legacy dataclass format (backward compatibility)
            all_bboxes = vis_data.all_bboxes
            orig_masks = vis_data.all_object_masks
            warped_masks_all = getattr(vis_data, "warped_object_masks", None)
            area_ratios = getattr(vis_data, "area_ratios", None)
            epsilon_weights = getattr(vis_data, "epsilon_weights", None)
        else:
            all_bboxes = None
            orig_masks = None
            warped_masks_all = None
            area_ratios = None
            epsilon_weights = None

        # Extract data for this sample
        sample_bboxes = all_bboxes[sample_idx] if all_bboxes is not None else None
        sample_masks_orig = orig_masks[sample_idx] if orig_masks is not None else None
        sample_masks_warped = warped_masks_all[sample_idx] if warped_masks_all is not None else None
        sample_area_ratios = area_ratios[sample_idx] if area_ratios is not None else None
        sample_epsilon_weights = epsilon_weights[sample_idx] if epsilon_weights is not None else None

        # Get deformation data
        offsets = vis_data.deform_offsets[sample_idx] if vis_data.deform_offsets is not None else None  # [K, 2, H, W]
        warped_img = vis_data.warped_images[sample_idx] if hasattr(vis_data, "warped_images") and vis_data.warped_images is not None else None
        styled_img = vis_data.adversarial_images[sample_idx] if getattr(vis_data, "adversarial_images", None) is not None else None
        attack_order = getattr(vis_data, "attack_order", None)

        # Get GT masks for overlay
        # Use original masks for "before" overlay; warped masks for "after" if available
        gt_masks_orig = sample_masks_orig  # [K, H, W]
        gt_masks_warped = sample_masks_warped if sample_masks_warped is not None else sample_masks_orig
        K = gt_masks_orig.shape[0] if gt_masks_orig is not None else 0

        # Convert to numpy
        orig_np = original_img.permute(1, 2, 0).cpu().numpy()
        orig_np = np.clip(orig_np, 0, 1)
        H, W = orig_np.shape[0], orig_np.shape[1]

        # Convert styled/warped image to numpy if available (denormalize first!)
        if styled_img is not None:
            styled_img_denorm = self._denormalize_images(styled_img.unsqueeze(0))[0]
            styled_np = styled_img_denorm.permute(1, 2, 0).cpu().numpy()
            styled_np = np.clip(styled_np, 0, 1)
        else:
            styled_np = None

        if warped_img is not None:
            warped_img_denorm = self._denormalize_images(warped_img.unsqueeze(0))[0]
            warped_np = warped_img_denorm.permute(1, 2, 0).cpu().numpy()
            warped_np = np.clip(warped_np, 0, 1)
        else:
            warped_np = None

        # Convert GT masks to numpy if available (before deformation)
        if gt_masks_orig is not None:
            gt_masks_np = gt_masks_orig.cpu().numpy()  # [K, H, W]
        else:
            gt_masks_np = None

        # Determine primary object (the one being trained for loss computation)
        # Priority: 1) loss_object_indices from vis_data, 2) default to 0
        # (Matching StyleAUE behavior - sample_idx=0 corresponds to object 0 in the video)
        sample_loss_obj_idx = None
        if hasattr(vis_data, "loss_object_indices") and vis_data.loss_object_indices is not None:
            sample_loss_obj_idx = vis_data.loss_object_indices[sample_idx] if sample_idx < len(vis_data.loss_object_indices) else None

        if sample_loss_obj_idx is not None:
            primary_obj_idx = sample_loss_obj_idx.item() if hasattr(sample_loss_obj_idx, "item") else int(sample_loss_obj_idx)
        else:
            primary_obj_idx = 0

        # Create visualization: 2x2 layout (matching StyleAUE format)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Row 0, Col 0: Original Image with GT masks and bboxes (Before Deformation)
        # Row 0, Col 0: Original Image with GT masks and bboxes (Before Deformation)
        self._plot_image_with_bboxes(orig_np, sample_bboxes, primary_obj_idx, include_background=include_background, K=K, masks=gt_masks_orig, ax=axes[0, 0])
        axes[0, 0].set_title(f"Before Deformation (Sample {sample_idx})", fontsize=12, fontweight="bold")

        # Row 0, Col 1: Statistics (Top Right)
        axes[0, 1].axis("off")
        stats_text = f"Deformation Statistics:\n\n"
        stats_text += f"Objects: {K}\n"
        if sample_area_ratios is not None:
            stats_text += f"Primary Object: O{primary_obj_idx + 1}\n"
            stats_text += f"Area Ratios:\n"
            for k in range(min(K, 5)):  # Show first 5 objects
                ratio = sample_area_ratios[k].item()
                stats_text += f"  O{k + 1}: {ratio:.3f}\n"
        if offsets is not None:
            offsets_np = offsets.cpu().numpy()  # [K, 2, H, W]
            offset_magnitude = np.sqrt(offsets_np[:, 0] ** 2 + offsets_np[:, 1] ** 2)
            stats_text += f"\nOffset Statistics:\n"
            stats_text += f"  Max: {offset_magnitude.max():.4f}\n"
            stats_text += f"  Mean: {offset_magnitude.mean():.4f}\n"
            stats_text += f"  Std: {offset_magnitude.std():.4f}\n"
        axes[0, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment="center", family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # Row 1, Col 0: After attacks (respect configured order)
        def _pick_display_np(attack_order, warped_np, styled_np):
            mapping = {
                "style": styled_np,
                "deform": warped_np,
            }
            if attack_order is None:
                return warped_np or styled_np
            for name in reversed(attack_order):
                candidate = mapping.get(name)
                if candidate is not None:
                    return candidate
            return warped_np or styled_np

        display_np = _pick_display_np(attack_order, warped_np, styled_np)
        # Overlay warped masks if available (fallback to original masks)
        if gt_masks_warped is not None:
            gt_masks_warped_np = gt_masks_warped.cpu().numpy()
        else:
            gt_masks_warped_np = gt_masks_np
        if display_np is not None:
            self._plot_image_with_bboxes(
                display_np,
                None,
                primary_obj_idx,  # No bboxes on warped result?
                include_background=include_background,
                K=gt_masks_warped_np.shape[0] if gt_masks_warped_np is not None else K,
                masks=gt_masks_warped,
                ax=axes[1, 0],
            )
            axes[1, 0].set_title("After Attacks (Combined)", fontsize=12, fontweight="bold")
        else:
            axes[1, 0].imshow(orig_np)
            axes[1, 0].set_title("After Attacks (Not Available)", fontsize=12, fontweight="bold")
            axes[1, 0].axis("off")

        # Row 1, Col 1: Offset Magnitude Overlay
        if offsets is not None:
            offsets_np = offsets.cpu().numpy()  # [K, 2, H, W]
            offset_magnitude = np.sqrt(offsets_np[:, 0] ** 2 + offsets_np[:, 1] ** 2)  # [K, H, W]
            offset_mag_combined = offset_magnitude.sum(axis=0)  # [H, W]

            axes[1, 1].imshow(orig_np)
            im = axes[1, 1].imshow(offset_mag_combined, alpha=0.6, cmap="hot")
            axes[1, 1].set_title("Offset Magnitude (Overlay)", fontsize=12, fontweight="bold")
            axes[1, 1].axis("off")
            plt.colorbar(im, ax=axes[1, 1])
        else:
            axes[1, 1].axis("off")

        # Add overall title
        fig.suptitle(f"Deformation Augmentation ({K} objects)", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Convert figure to image and log to TensorBoard
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(height, width, 4)[:, :, :3]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0

        # Log to TensorBoard and save to disk
        writer = self.logger.tb_logger._writer if self.logger.tb_logger else None

        # Save combined figure to disk
        self._save_image_to_disk(fig, step, "DeformAUE", "combined_view")

        if writer:
            writer.add_image("DeformAUE/combined_view", img_tensor, step)

        # === Paper-quality individual images (no statistics overlay) ===
        # Log original image only
        orig_tensor = torch.from_numpy(orig_np).permute(2, 0, 1).float()
        orig_tensor = torch.clamp(orig_tensor, 0, 1)
        if writer:
            writer.add_image(f"DeformAUE/original", orig_tensor, step)
        # Save to disk
        self._save_image_to_disk(orig_tensor, step, "DeformAUE", "original")

        # Log original image with bbox overlays (for showing attack region)
        # Color scheme: Red = primary object (for loss), cyan = other objects
        if sample_bboxes is not None:
            orig_with_bbox = self._plot_image_with_bboxes(
                orig_np, sample_bboxes, primary_obj_idx, include_background=include_background, K=K, masks=gt_masks_orig, ax=None
            )  # ax=None -> returns uint8 image
            orig_bbox_tensor = torch.from_numpy(orig_with_bbox).permute(2, 0, 1).float() / 255.0
            if writer:
                writer.add_image(f"DeformAUE/original_with_bbox", orig_bbox_tensor, step)
            # Save to disk
            self._save_image_to_disk(orig_with_bbox, step, "DeformAUE", "original_with_bbox")

        # Log warped/styled image (the final result after all attacks)
        if display_np is not None:
            display_tensor = torch.from_numpy(display_np).permute(2, 0, 1).float()
            display_tensor = torch.clamp(display_tensor, 0, 1)
            if writer:
                writer.add_image(f"DeformAUE/deformed", display_tensor, step)
            # Save to disk
            self._save_image_to_disk(display_tensor, step, "DeformAUE", "deformed")

            # Log deformed image with bbox (same color scheme)
            if sample_bboxes is not None:
                display_with_bbox = self._plot_image_with_bboxes(display_np, sample_bboxes, primary_obj_idx, include_background=include_background, K=K, masks=gt_masks_warped, ax=None)
                display_bbox_tensor = torch.from_numpy(display_with_bbox).permute(2, 0, 1).float() / 255.0
                if writer:
                    writer.add_image(f"DeformAUE/deformed_with_bbox", display_bbox_tensor, step)
                # Save to disk
                self._save_image_to_disk(display_with_bbox, step, "DeformAUE", "deformed_with_bbox")

        # Log offset magnitude as heatmap
        if offsets is not None:
            offsets_np = offsets.cpu().numpy()  # [K, 2, H, W]
            offset_magnitude = np.sqrt(offsets_np[:, 0] ** 2 + offsets_np[:, 1] ** 2)  # [K, H, W]
            offset_mag_combined = offset_magnitude.sum(axis=0)  # [H, W]

            # Normalize and convert to colormap image
            if offset_mag_combined.max() > 0:
                offset_normalized = offset_mag_combined / offset_mag_combined.max()
            else:
                offset_normalized = offset_mag_combined

            # Apply 'hot' colormap
            import matplotlib.cm as cm_local

            cmap = cm_local.get_cmap("hot")
            offset_colored = cmap(offset_normalized)[:, :, :3]  # [H, W, 3]
            offset_tensor = torch.from_numpy(offset_colored).permute(2, 0, 1).float()
            if writer:
                writer.add_image(f"DeformAUE/offset_heatmap", offset_tensor, step)
            # Save to disk
            self._save_image_to_disk(offset_tensor, step, "DeformAUE", "offset_heatmap")

        # Log GT Mask (Standalone) - background stays black
        if gt_masks_orig is not None:
            gt_vis_np = np.zeros((*gt_masks_orig.shape[-2:], 3), dtype=np.uint8)
            gt_masks_np = gt_masks_orig.cpu().numpy()
            # Skip background object (last one when include_background is True)
            n_objs_to_draw = gt_masks_np.shape[0] - 1 if include_background else gt_masks_np.shape[0]
            colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]]
            for k in range(n_objs_to_draw):
                mask_k = gt_masks_np[k] > 0.5
                color = colors[k % len(colors)]
                for c in range(3):
                    gt_vis_np[..., c] = np.where(mask_k, color[c], gt_vis_np[..., c])

            gt_tensor = torch.from_numpy(gt_vis_np).permute(2, 0, 1).float() / 255.0
            if writer:
                writer.add_image(f"DeformAUE/gt_mask", gt_tensor, step)
            # Save to disk
            self._save_image_to_disk(gt_vis_np, step, "DeformAUE", "gt_mask")

        # Log Prediction Mask (Standalone)
        if pred_mask is not None:
            pred_np = pred_mask.detach().cpu().numpy()
            pred_bin = pred_np > 0.0
            pred_vis_np = np.zeros((*pred_bin.shape[-2:], 3), dtype=np.uint8)
            colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]]
            for k in range(pred_bin.shape[0]):
                mask_k = pred_bin[k]
                color = colors[k % len(colors)]
                for c in range(3):
                    pred_vis_np[..., c] = np.where(mask_k, color[c], pred_vis_np[..., c])

            pred_tensor = torch.from_numpy(pred_vis_np).permute(2, 0, 1).float() / 255.0
            if writer:
                writer.add_image(f"DeformAUE/pred_mask", pred_tensor, step)
            # Save to disk
            self._save_image_to_disk(pred_vis_np, step, "DeformAUE", "pred_mask")

        # Log Mask Difference Visualization (difference regions only, no contours)
        if gt_masks_orig is not None and gt_masks_warped is not None:
            # Create visualization on top of deformed image (or original if deformed not available)
            base_img = display_np if display_np is not None else orig_np
            H, W = base_img.shape[:2]

            # Create figure for difference visualization (no border/padding)
            fig_diff, ax_diff = plt.subplots(figsize=(8, 8))
            ax_diff.imshow(base_img)
            ax_diff.axis("off")

            gt_masks_orig_np = gt_masks_orig.cpu().numpy() if hasattr(gt_masks_orig, "cpu") else gt_masks_orig
            gt_masks_warped_np = gt_masks_warped.cpu().numpy() if hasattr(gt_masks_warped, "cpu") else gt_masks_warped

            # Accumulate difference regions across all objects
            added_combined = np.zeros((H, W), dtype=bool)
            removed_combined = np.zeros((H, W), dtype=bool)

            for k in range(gt_masks_orig_np.shape[0]):
                mask_orig = gt_masks_orig_np[k] > 0.5
                mask_warp = gt_masks_warped_np[k] > 0.5

                # Compute difference regions
                added_combined |= mask_warp & ~mask_orig  # Added region
                removed_combined |= mask_orig & ~mask_warp  # Removed region

            # Draw added regions (light green, semi-transparent)
            if added_combined.sum() > 0:
                added_overlay = np.zeros((H, W, 4), dtype=np.float32)
                added_overlay[added_combined] = [0.2, 0.8, 0.2, 0.5]  # Light green with alpha
                ax_diff.imshow(added_overlay)

            # Draw removed regions (light red, semi-transparent)
            if removed_combined.sum() > 0:
                removed_overlay = np.zeros((H, W, 4), dtype=np.float32)
                removed_overlay[removed_combined] = [0.9, 0.2, 0.2, 0.5]  # Light red with alpha
                ax_diff.imshow(removed_overlay)

            # Remove all padding/margins for clean output
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            ax_diff.xaxis.set_major_locator(plt.NullLocator())
            ax_diff.yaxis.set_major_locator(plt.NullLocator())

            fig_diff.canvas.draw()
            w_diff, h_diff = fig_diff.canvas.get_width_height()
            diff_array = np.frombuffer(fig_diff.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_diff, w_diff, 4)[:, :, :3]
            diff_tensor = torch.from_numpy(diff_array).permute(2, 0, 1).float() / 255.0
            if writer:
                writer.add_image(f"DeformAUE/mask_diff", diff_tensor, step)
            # Save to disk
            self._save_image_to_disk(diff_array, step, "DeformAUE", "mask_diff")
            plt.close(fig_diff)

        # IMPORTANT: Flush to disk immediately to prevent data loss from TensorBoard caching
        if writer:
            writer.flush()

        plt.close(fig)

    def _log_deform_statistics(self, deform_offsets, step):
        """Log deformation offset statistics."""
        # deform_offsets: [N, K, 2, H, W]
        offset_magnitude = torch.sqrt(deform_offsets[:, :, 0] ** 2 + deform_offsets[:, :, 1] ** 2)  # [N, K, H, W]
        max_offset = offset_magnitude.max().item()
        mean_offset = offset_magnitude.mean().item()
        std_offset = offset_magnitude.std().item()

        self.logger.log("DeformAUE/max_offset", max_offset, step)
        self.logger.log("DeformAUE/mean_offset", mean_offset, step)
        self.logger.log("DeformAUE/std_offset", std_offset, step)

    def _log_train_uncertainty_visualization(self, bndl_outputs, batch, targets, outputs, step, frame_index=0):
        """Log training uncertainty map visualization to TensorBoard under Uncertainty/ section.

        This provides visualization of pixel-level uncertainty from BNDL during training.
        Outputs a simple grayscale uncertainty map without borders.

        NOTE: pixel_uncertainty has shape [O_t, H, W, K] where O_t is the number of objects
        in the current frame (frame_index), NOT the number of videos. The batch dimension
        represents individual objects being processed by the mask decoder.

        Args:
            frame_index: The frame index within the video sequence. Used to correctly
                        map objects to images via flat_obj_to_img_idx[frame_index].
        """
        import cv2

        if bndl_outputs is None:
            return

        try:
            # Get or compute pixel uncertainty
            pixel_uncertainty = bndl_outputs.get("pixel_uncertainty")

            # If uncertainty not yet computed, compute it now
            if pixel_uncertainty is None:
                if targets is not None:
                    if targets.ndim >= 3 and targets.shape[0] > 0:
                        current_frame_targets = targets[0] if targets.shape[0] > 0 else targets
                    else:
                        current_frame_targets = targets
                    bndl_outputs = self._calculate_uncertainty_for_bndl(bndl_outputs, batch, current_frame_targets)
                    pixel_uncertainty = bndl_outputs.get("pixel_uncertainty")

            if pixel_uncertainty is None:
                return

            # Use pixel_uncertainty's batch size (object-level, O_t) as source of truth
            # This is the actual batch size processed by mask decoder
            unc_batch_size = pixel_uncertainty.shape[0]
            sample_idx = step % unc_batch_size

            # Get uncertainty for this sample
            uncertainty_sample = pixel_uncertainty[sample_idx]  # [H, W] or [H, W, K]
            if uncertainty_sample.ndim > 2:
                uncertainty_sample = uncertainty_sample.mean(dim=-1)  # Average across masks
            uncertainty_np = uncertainty_sample.detach().cpu().numpy()

            # Extract original image - prefer vis_data for consistency with StyleAUE/DeformAUE
            # vis_data.original_images has same batch dimension as pixel_uncertainty (object-level)
            original_img = None
            vis_data = self._extract_vis_data_from_outputs(outputs)
            if vis_data is not None and hasattr(vis_data, "original_images") and vis_data.original_images is not None:
                # Use denormalized original from vis_data (same source as StyleAUE/DeformAUE)
                original_denorm = self._denormalize_images(vis_data.original_images)
                if original_denorm is not None and len(original_denorm) > sample_idx:
                    orig_tensor = original_denorm[sample_idx]  # [C, H, W]
                    original_img = orig_tensor.cpu().permute(1, 2, 0).numpy()  # [H, W, C]
                    original_img = np.clip(original_img, 0, 1)

            # Fallback to batch.flat_img_batch using object-to-image mapping
            # This is needed when vis_data is not available (e.g., AUE not enabled)
            if original_img is None:
                # Use flat_img_batch [(B*T), C, H, W] and flat_obj_to_img_idx for correct mapping
                if hasattr(batch, "flat_img_batch") and hasattr(batch, "flat_obj_to_img_idx"):
                    flat_imgs = batch.flat_img_batch  # [(B*T), C, H, W]
                    # Get the current frame's object-to-image indices
                    flat_obj_idx = batch.flat_obj_to_img_idx[frame_index]  # [O_t] for frame frame_index
                    if sample_idx < len(flat_obj_idx):
                        img_idx = flat_obj_idx[sample_idx].item()
                        if img_idx < len(flat_imgs):
                            img_tensor = flat_imgs[img_idx]  # [C, H, W]
                            if img_tensor.is_cuda:
                                img_tensor = img_tensor.cpu()
                            img_np = img_tensor.numpy().transpose(1, 2, 0)  # [H, W, C]
                            # Denormalize ImageNet
                            mean = np.array([0.485, 0.456, 0.406])
                            std = np.array([0.229, 0.224, 0.225])
                            original_img = img_np * std + mean
                            original_img = np.clip(original_img, 0, 1)

                # Last resort: use simple extraction (for legacy compatibility)
                if original_img is None:
                    original_img = self._extract_original_image(batch, frame_idx=0, batch_idx=0)

            # Resize uncertainty to match original image size (smoothing out grid artifacts)
            if original_img is not None:
                H_img, W_img = original_img.shape[:2]
                H_unc, W_unc = uncertainty_np.shape
                if H_unc != H_img or W_unc != W_img:
                    import cv2

                    # Use INTER_LINEAR (Bilinear) to smooth out grid artifacts from components
                    uncertainty_np = cv2.resize(uncertainty_np, (W_img, H_img), interpolation=cv2.INTER_LINEAR)

            # Normalize uncertainty using robust percentiles to [0, 1]
            unc_p1 = np.percentile(uncertainty_np, 1)
            unc_p99 = np.percentile(uncertainty_np, 99)
            if unc_p99 - unc_p1 > 1e-6:
                uncertainty_norm = np.clip((uncertainty_np - unc_p1) / (unc_p99 - unc_p1), 0, 1)
            else:
                uncertainty_norm = np.clip(uncertainty_np, 0, 1)

            # Log to TensorBoard - only log if BOTH uncertainty and original image are available
            # This ensures consistent step numbers across all Uncertainty/* visualizations
            if self.logger.tb_logger and self.logger.tb_logger._writer and original_img is not None:
                # Log grayscale uncertainty map [1, H, W]
                unc_tensor = torch.from_numpy(uncertainty_norm.astype(np.float32)).unsqueeze(0)
                self.logger.tb_logger._writer.add_image(f"Uncertainty/train_uncertainty_map", unc_tensor, step)

                # Log original input image (always available if we reach here)
                # original_img is [H, W, 3] numpy array with values in [0, 1]
                orig_tensor = torch.from_numpy(original_img.astype(np.float32)).permute(2, 0, 1)  # [3, H, W]
                self.logger.tb_logger._writer.add_image(f"Uncertainty/train_original_input", orig_tensor, step)

                # Log uncertainty scalar statistics
                self.logger.log("Uncertainty/train_mean", float(uncertainty_np.mean()), step)
                self.logger.log("Uncertainty/train_max", float(uncertainty_np.max()), step)
                self.logger.log("Uncertainty/train_std", float(uncertainty_np.std()), step)

                # Flush to disk to prevent data loss from TensorBoard caching
                self.logger.tb_logger._writer.flush()

        except Exception as e:
            logging.debug(f"Uncertainty visualization failed at step {step}: {e}")

    def _log_clean_vs_adv_comparison(self, bndl_outputs, batch, outputs, step, frame_index=0):
        """Log raw images to Clean/ and Adv/ sections in TensorBoard.

        Logs individual images without any modification or overlay:
        - Clean/input, Clean/segmentation, Clean/uncertainty
        - Adv/input, Adv/segmentation, Adv/uncertainty

        Also saves images to disk at: {log_dir}/vis/{step}/clean/ and {log_dir}/vis/{step}/adv/
        """
        import cv2

        if bndl_outputs is None:
            return

        # Check if adversarial outputs are available
        adv_outputs = bndl_outputs.get("adv_outputs")
        if adv_outputs is None:
            return

        try:
            writer = self.logger.tb_logger._writer if self.logger.tb_logger else None

            # === Extract Clean Branch Data ===
            clean_uncertainty = bndl_outputs.get("pixel_uncertainty")
            clean_masks = bndl_outputs.get("masks_bndl")  # [B, K, H, W]

            # === Extract Adversarial Branch Data ===
            adv_aux = adv_outputs.get("aux_outputs", {})
            adv_bndl = adv_aux.get("bndl", {}) if isinstance(adv_aux, dict) else {}
            adv_uncertainty = adv_bndl.get("pixel_uncertainty")
            adv_pred_masks = adv_outputs.get("pred_masks")  # [B, M, H, W]

            if clean_uncertainty is None:
                return

            # === Get sample index ===
            unc_batch_size = clean_uncertainty.shape[0]
            sample_idx = step % unc_batch_size

            # === Extract images from vis_data ===
            vis_data = self._extract_vis_data_from_outputs(outputs)
            if vis_data is None:
                return

            original_denorm = self._denormalize_images(vis_data.original_images)
            adv_denorm = self._denormalize_images(vis_data.adversarial_images)

            if original_denorm is None or sample_idx >= len(original_denorm):
                return

            # === Log Clean Branch ===
            # Clean input image [3, H, W]
            clean_img_tensor = original_denorm[sample_idx].cpu().clamp(0, 1)
            if writer:
                writer.add_image("Clean/input", clean_img_tensor, step)
            # Save to disk
            self._save_image_to_disk(clean_img_tensor, step, "clean", "input")

            # Clean segmentation mask [1, H, W]
            if clean_masks is not None and sample_idx < clean_masks.shape[0]:
                clean_mask = clean_masks[sample_idx, 0:1].detach().cpu().float()  # [1, H, W]
                clean_mask = (clean_mask > 0).float()  # Binary mask
                if writer:
                    writer.add_image("Clean/segmentation", clean_mask, step)
                # Save to disk
                self._save_image_to_disk(clean_mask, step, "clean", "segmentation")

            # Clean uncertainty [1, H, W]
            clean_unc_sample = clean_uncertainty[sample_idx]
            if clean_unc_sample.ndim > 2:
                clean_unc_sample = clean_unc_sample.mean(dim=-1)  # [H, W]
            clean_unc_tensor = clean_unc_sample.detach().cpu().unsqueeze(0)  # [1, H, W]
            # Normalize to [0, 1]
            unc_min, unc_max = clean_unc_tensor.min(), clean_unc_tensor.max()
            if unc_max - unc_min > 1e-6:
                clean_unc_tensor = (clean_unc_tensor - unc_min) / (unc_max - unc_min)
            if writer:
                writer.add_image("Clean/uncertainty", clean_unc_tensor, step)
            # Save to disk
            self._save_image_to_disk(clean_unc_tensor, step, "clean", "uncertainty")

            # === Log Adversarial Branch ===
            if adv_denorm is not None and sample_idx < len(adv_denorm):
                # Adv input image [3, H, W]
                adv_img_tensor = adv_denorm[sample_idx].cpu().clamp(0, 1)
                if writer:
                    writer.add_image("Adv/input", adv_img_tensor, step)
                # Save to disk
                self._save_image_to_disk(adv_img_tensor, step, "adv", "input")

            # Adv segmentation mask [1, H, W]
            if adv_pred_masks is not None:
                adv_sample_idx = min(sample_idx, adv_pred_masks.shape[0] - 1)
                adv_mask = adv_pred_masks[adv_sample_idx, 0:1].detach().cpu().float()  # [1, H, W]
                adv_mask = (adv_mask > 0).float()  # Binary mask
                if writer:
                    writer.add_image("Adv/segmentation", adv_mask, step)
                # Save to disk
                self._save_image_to_disk(adv_mask, step, "adv", "segmentation")

            # Adv uncertainty [1, H, W]
            if adv_uncertainty is not None:
                adv_sample_idx = min(sample_idx, adv_uncertainty.shape[0] - 1)
                adv_unc_sample = adv_uncertainty[adv_sample_idx]
                if adv_unc_sample.ndim > 2:
                    adv_unc_sample = adv_unc_sample.mean(dim=-1)  # [H, W]
                adv_unc_tensor = adv_unc_sample.detach().cpu().unsqueeze(0)  # [1, H, W]
                # Normalize to [0, 1]
                unc_min, unc_max = adv_unc_tensor.min(), adv_unc_tensor.max()
                if unc_max - unc_min > 1e-6:
                    adv_unc_tensor = (adv_unc_tensor - unc_min) / (unc_max - unc_min)
                if writer:
                    writer.add_image("Adv/uncertainty", adv_unc_tensor, step)
                # Save to disk
                self._save_image_to_disk(adv_unc_tensor, step, "adv", "uncertainty")

            if writer:
                writer.flush()

        except Exception as e:
            logging.debug(f"Clean vs Adv logging failed at step {step}: {e}")

    def _plot_image_with_bboxes(self, img_np, bboxes=None, primary_idx=0, include_background=False, K=None, masks=None, epsilon_weights=None, ax=None):
        """Standardized plotting function for image with bboxes/masks using Matplotlib.

        This logic is shared between combined visualization and standalone paper images.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from skimage import measure

        # If no ax provided, create a standalone figure (for paper images)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            standalone = True
        else:
            fig = ax.figure
            standalone = False

        ax.imshow(img_np)
        ax.axis("off")

        # If no metadata, just return the image
        if bboxes is None and masks is None:
            if standalone:
                plt.tight_layout(pad=0)
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                img_out = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
                plt.close(fig)
                return img_out
            return

        # Draw Objects
        n_objs = 0
        if bboxes is not None:
            n_objs = bboxes.shape[0]
        elif masks is not None:
            n_objs = masks.shape[0]

        if K is None:
            K = n_objs

        # Handle tensor inputs
        if bboxes is not None and hasattr(bboxes, "cpu"):
            bboxes = bboxes.cpu().numpy()
        if masks is not None and hasattr(masks, "cpu"):
            masks = masks  # .cpu().numpy() # Keep tensor for sum() check sometimes, convert when needed

        for k in range(n_objs):
            # Skip empty masks if masks provided
            if masks is not None:
                mask_k = masks[k]
                if hasattr(mask_k, "cpu"):
                    mask_k = mask_k.cpu().numpy()
                if mask_k.sum() == 0:
                    continue

            # Determine logic
            is_valid_bbox = False
            if bboxes is not None:
                x1, y1, x2, y2 = bboxes[k]
                if x2 > x1 and y2 > y1:
                    is_valid_bbox = True

            if not is_valid_bbox and masks is None:
                continue

            # Style Configuration
            is_bg_obj = include_background and (k == K - 1)

            if is_bg_obj:
                color = "lime"
                linestyle = "--"
                linewidth = 1.0  # Thin contour for better visibility of small deformations
                label_suffix = " (BG)"
            elif k == primary_idx:
                color = "red"
                linestyle = "-"
                linewidth = 1.0  # Thin contour for better visibility of small deformations
                label_suffix = "*"
            else:
                color = "cyan"
                linestyle = "-"
                linewidth = 1.0  # Thin contour for better visibility of small deformations
                label_suffix = ""

            # Add epsilon info if available
            if epsilon_weights is not None:
                eps = epsilon_weights[k].item()
                label_suffix += f" ε{eps:.2f}"

            # BBox drawing removed for cleaner visualization
            # (bbox was too cluttered and didn't add useful info for deformation visualization)

            # Draw Contour (if masks provided)
            if masks is not None:
                mask_k = masks[k]
                if hasattr(mask_k, "cpu"):
                    mask_k = mask_k.cpu().numpy()
                contours = measure.find_contours(mask_k, 0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth, linestyle=linestyle, alpha=0.8)

        # Return logic
        if standalone:
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            img_out = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
            plt.close(fig)
            return img_out

    def _log_gcn_parameters(self, step):
        """Log GCN layer weight statistics if GCN is enabled."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model, "style_gcn") and model.style_gcn is not None:
            gcn = model.style_gcn
            # Log statistics for each GCN layer
            for layer_idx, layer in enumerate(gcn.layers):
                if hasattr(layer, "weight") and layer.weight is not None:
                    weight_mean = layer.weight.mean().item()
                    weight_std = layer.weight.std().item()
                    weight_norm = layer.weight.norm().item()
                    self.logger.log(f"GCN/layer_{layer_idx}_weight_mean", weight_mean, step)
                    self.logger.log(f"GCN/layer_{layer_idx}_weight_std", weight_std, step)
                    self.logger.log(f"GCN/layer_{layer_idx}_weight_norm", weight_norm, step)

    def _log_style_statistics_overlay(
        self,
        original_img,
        adv_img,
        original_style,
        adv_style,
        sample_id,
        step,
        bbox=None,
        gt_mask=None,
        all_bboxes=None,
        all_masks=None,
        combined_mask=None,
        area_ratios=None,
        epsilon_weights=None,
        length=None,
        loss_object_idx=None,
        pred_mask=None,
    ):
        """Create visualization with style statistics overlaid on images (supports both single and multi-object)

        Args:
            original_img: [3, H, W] original image tensor
            adv_img: [3, H, W] adversarial image tensor
            original_style: [6] original style statistics
            adv_style: [6] adversarial style statistics
            sample_id: sample index
            step: training step
            bbox: [4] single bbox (for single-object mode, optional)
            gt_mask: [1, H, W] or [H, W] single mask (for single-object mode, optional)
            all_bboxes: [K, 4] all bboxes (for multi-object mode, optional)
            all_masks: [K, H, W] all masks (for multi-object mode, optional)
            combined_mask: [H, W] combined mask for loss (for multi-object mode, optional)
            area_ratios: [K] area ratio for each object (for multi-object mode, optional)
            epsilon_weights: [K] epsilon weight for each object (for multi-object mode, optional)
            loss_object_idx: int, the object index being trained (for primary object highlighting)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.cm as cm
        import numpy as np

        model = unwrap_ddp_if_wrapped(self.model)
        include_background = bool(getattr(model, "adv_enable_background", False))

        # Detect if multi-object mode
        is_multi_object = all_masks is not None and all_masks.shape[0] > 1

        # Convert tensors to numpy
        orig_np = original_img.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        adv_np = adv_img.cpu().numpy().transpose(1, 2, 0)
        orig_style_np = original_style.cpu().numpy()  # [6]
        adv_style_np = adv_style.cpu().numpy()

        if is_multi_object:
            # Multi-object visualization with 2x3 layout
            K = all_masks.shape[0]
            H, W = all_masks.shape[1:]

            # Determine primary object (the one being trained)
            # Priority: 1) loss_object_idx parameter, 2) default to 0
            if loss_object_idx is not None:
                primary_obj_idx = loss_object_idx
            else:
                primary_obj_idx = 0

            # Define distinct colors for each object (non-primary: blue, primary for loss: red)
            colors = cm.get_cmap("tab10", K)
            object_colors = [colors(i)[:3] for i in range(K)]

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            self._plot_image_with_bboxes(orig_np, all_bboxes, primary_obj_idx, include_background=include_background, K=K, masks=all_masks, epsilon_weights=epsilon_weights, ax=axes[0, 0])
            axes[0, 0].set_title(f"Original Image (Sample {sample_id})", fontsize=12, fontweight="bold")

            # Top-middle: Original style stats
            axes[0, 1].bar(["R", "G", "B"], orig_style_np[:3], color=["red", "green", "blue"], alpha=0.7)
            axes[0, 1].set_title("Original Mean", fontsize=10)
            axes[0, 1].set_ylabel("Mean Value")
            axes[0, 1].set_ylim([orig_style_np[:3].min() - 0.1, orig_style_np[:3].max() + 0.1])
            axes[0, 1].grid(True, alpha=0.3)

            axes[0, 2].bar(["R", "G", "B"], orig_style_np[3:], color=["red", "green", "blue"], alpha=0.7)
            axes[0, 2].set_title("Original Std", fontsize=10)
            axes[0, 2].set_ylabel("Std Value")
            axes[0, 2].set_ylim([orig_style_np[3:].min() - 0.05, orig_style_np[3:].max() + 0.05])
            axes[0, 2].grid(True, alpha=0.3)

            # Bottom-left: Adversarial with all object bboxes (NO mask overlay)
            # Bottom-left: Adversarial with all object bboxes
            self._plot_image_with_bboxes(adv_np, all_bboxes, primary_obj_idx, include_background=include_background, K=K, masks=all_masks, epsilon_weights=epsilon_weights, ax=axes[1, 0])
            axes[1, 0].set_title("Adversarial (Multi-Object Style)", fontsize=12, fontweight="bold")

            # Bottom-middle & right: Adversarial style stats
            axes[1, 1].bar(["R", "G", "B"], adv_style_np[:3], color=["red", "green", "blue"], alpha=0.7)
            axes[1, 1].set_title("Adversarial Mean", fontsize=10)
            axes[1, 1].set_ylabel("Mean Value")
            axes[1, 1].set_ylim([adv_style_np[:3].min() - 0.1, adv_style_np[:3].max() + 0.1])
            axes[1, 1].grid(True, alpha=0.3)

            axes[1, 2].bar(["R", "G", "B"], adv_style_np[3:], color=["red", "green", "blue"], alpha=0.7)
            axes[1, 2].set_title("Adversarial Std", fontsize=10)
            axes[1, 2].set_ylabel("Std Value")
            axes[1, 2].set_ylim([adv_style_np[3:].min() - 0.05, adv_style_np[3:].max() + 0.05])
            axes[1, 2].grid(True, alpha=0.3)

            # Add overall title
            fig.suptitle(f"Multi-Object Style Attack ({K} objects)", fontsize=14, fontweight="bold")

        else:
            # Single-object visualization (original 2x3 layout)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            from skimage import measure  # Needed for contour drawing in single-object view

            # Row 1: Original image and its styles
            axes[0, 0].imshow(orig_np)
            axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
            axes[0, 0].axis("off")

            # Draw bbox and mask on original image if available
            if bbox is not None:
                bbox_np = bbox.cpu().numpy() if hasattr(bbox, "cpu") else bbox
                x1, y1, x2, y2 = bbox_np
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor="red", facecolor="none", linestyle="-")
                axes[0, 0].add_patch(rect)
                axes[0, 0].text(x1, y1 - 5, "Attack Region", color="red", fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            if gt_mask is not None:
                mask_np = gt_mask.cpu().numpy() if hasattr(gt_mask, "cpu") else gt_mask
                contours = measure.find_contours(mask_np, 0.5)
                for contour in contours:
                    axes[0, 0].plot(contour[:, 1], contour[:, 0], color="cyan", linewidth=2, linestyle="-", alpha=0.7)

            # Original style - means
            axes[0, 1].bar(["R", "G", "B"], orig_style_np[:3], color=["red", "green", "blue"], alpha=0.7)
            axes[0, 1].set_title("Original Mean (per channel)", fontsize=10)
            axes[0, 1].set_ylabel("Mean Value")
            axes[0, 1].set_ylim([orig_style_np[:3].min() - 0.1, orig_style_np[:3].max() + 0.1])
            axes[0, 1].grid(True, alpha=0.3)

            # Original style - stds
            axes[0, 2].bar(["R", "G", "B"], orig_style_np[3:], color=["red", "green", "blue"], alpha=0.7)
            axes[0, 2].set_title("Original Std (per channel)", fontsize=10)
            axes[0, 2].set_ylabel("Std Value")
            axes[0, 2].set_ylim([orig_style_np[3:].min() - 0.05, orig_style_np[3:].max() + 0.05])
            axes[0, 2].grid(True, alpha=0.3)

            # Row 2: Adversarial image and its styles
            axes[1, 0].imshow(adv_np)
            axes[1, 0].set_title("Adversarial Image (Style-Augmented)", fontsize=12, fontweight="bold")
            axes[1, 0].axis("off")

            # Draw bbox and mask on adversarial image if available
            if bbox is not None:
                bbox_np = bbox.cpu().numpy() if hasattr(bbox, "cpu") else bbox
                x1, y1, x2, y2 = bbox_np
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor="lime", facecolor="none", linestyle="-")
                axes[1, 0].add_patch(rect)
                axes[1, 0].text(x1, y1 - 5, "Attack Region (Styled)", color="lime", fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
            if gt_mask is not None:
                from skimage import measure

                mask_np = gt_mask.cpu().numpy() if hasattr(gt_mask, "cpu") else gt_mask
                contours = measure.find_contours(mask_np, 0.5)
                for contour in contours:
                    axes[1, 0].plot(contour[:, 1], contour[:, 0], color="cyan", linewidth=2, linestyle="-", alpha=0.7)

            # Adversarial style - means
            axes[1, 1].bar(["R", "G", "B"], adv_style_np[:3], color=["red", "green", "blue"], alpha=0.7)
            axes[1, 1].set_title("Adversarial Mean (per channel)", fontsize=10)
            axes[1, 1].set_ylabel("Mean Value")
            axes[1, 1].set_ylim([adv_style_np[:3].min() - 0.1, adv_style_np[:3].max() + 0.1])
            axes[1, 1].grid(True, alpha=0.3)

            # Adversarial style - stds
            axes[1, 2].bar(["R", "G", "B"], adv_style_np[3:], color=["red", "green", "blue"], alpha=0.7)
            axes[1, 2].set_title("Adversarial Std (per channel)", fontsize=10)
            axes[1, 2].set_ylabel("Std Value")
            axes[1, 2].set_ylim([adv_style_np[3:].min() - 0.05, adv_style_np[3:].max() + 0.05])
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Convert figure to image and log
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(height, width, 4)[:, :, :3]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0

        # Log to TensorBoard and save to disk
        writer = self.logger.tb_logger._writer if self.logger.tb_logger else None

        # Save combined figure to disk
        self._save_image_to_disk(fig, step, "StyleAUE", "combined_view")

        if writer:
            tag_suffix = "multi_object" if is_multi_object else "single_object"
            writer.add_image(f"StyleAUE/{tag_suffix}", img_tensor, step)

        # === Paper-quality individual images (no statistics overlay) ===
        # Log original image only
        orig_tensor = torch.from_numpy(orig_np).permute(2, 0, 1).float()
        orig_tensor = torch.clamp(orig_tensor, 0, 1)
        if writer:
            writer.add_image(f"StyleAUE/original", orig_tensor, step)
        # Save to disk
        self._save_image_to_disk(orig_tensor, step, "StyleAUE", "original")

        # Log original image with bbox overlays (for showing attack region)
        # Color scheme: Red = primary object (for loss), cyan = other objects
        # Use loss_object_idx if available (passed from training), otherwise default to 0
        primary_obj_idx = loss_object_idx if loss_object_idx is not None else 0

        if all_bboxes is not None:
            K_val = all_bboxes.shape[0]
            orig_with_bbox = self._plot_image_with_bboxes(orig_np, all_bboxes, primary_obj_idx, include_background=include_background, K=K_val, masks=all_masks, ax=None)
            orig_bbox_tensor = torch.from_numpy(orig_with_bbox).permute(2, 0, 1).float() / 255.0
            if writer:
                writer.add_image(f"StyleAUE/original_with_bbox", orig_bbox_tensor, step)
            # Save to disk
            self._save_image_to_disk(orig_with_bbox, step, "StyleAUE", "original_with_bbox")
        elif bbox is not None:
            # Single bbox - wrap in array for helper function
            single_bbox = bbox.unsqueeze(0) if hasattr(bbox, "unsqueeze") else np.expand_dims(bbox, 0)
            orig_with_bbox = self._plot_image_with_bboxes(
                orig_np,
                single_bbox,
                0,
                include_background=False,
                K=1,
                masks=gt_mask.unsqueeze(0) if hasattr(gt_mask, "unsqueeze") else np.expand_dims(gt_mask, 0) if gt_mask is not None else None,
                ax=None,
            )
            orig_bbox_tensor = torch.from_numpy(orig_with_bbox).permute(2, 0, 1).float() / 255.0
            if writer:
                writer.add_image(f"StyleAUE/original_with_bbox", orig_bbox_tensor, step)
            # Save to disk
            self._save_image_to_disk(orig_with_bbox, step, "StyleAUE", "original_with_bbox")

        # Log adversarial image only
        adv_tensor = torch.from_numpy(adv_np).permute(2, 0, 1).float()
        adv_tensor = torch.clamp(adv_tensor, 0, 1)
        if writer:
            writer.add_image(f"StyleAUE/adversarial", adv_tensor, step)
        # Save to disk
        self._save_image_to_disk(adv_tensor, step, "StyleAUE", "adversarial")

        # Log adversarial image with bbox overlays (same color scheme)
        if all_bboxes is not None:
            K_val = all_bboxes.shape[0]
            adv_with_bbox = self._plot_image_with_bboxes(adv_np, all_bboxes, primary_obj_idx, include_background=include_background, K=K_val, masks=all_masks, ax=None)
            adv_bbox_tensor = torch.from_numpy(adv_with_bbox).permute(2, 0, 1).float() / 255.0
            if writer:
                writer.add_image(f"StyleAUE/adversarial_with_bbox", adv_bbox_tensor, step)
            # Save to disk
            self._save_image_to_disk(adv_with_bbox, step, "StyleAUE", "adversarial_with_bbox")

        # Log difference image (amplified for visibility)
        diff = np.abs(adv_np - orig_np)
        diff_amplified = np.clip(diff * 5.0, 0, 1)  # Amplify by 5x
        diff_tensor = torch.from_numpy(diff_amplified).permute(2, 0, 1).float()
        if writer:
            writer.add_image(f"StyleAUE/difference", diff_tensor, step)
        # Save to disk
        self._save_image_to_disk(diff_tensor, step, "StyleAUE", "difference")

        # Log GT Mask (Standalone) - background stays black
        if all_masks is not None:
            gt_masks_np = all_masks.cpu().numpy()
            gt_vis_np = np.zeros((*gt_masks_np.shape[-2:], 3), dtype=np.uint8)
            # Skip background object (last one when include_background is True)
            n_objs_to_draw = gt_masks_np.shape[0] - 1 if include_background else gt_masks_np.shape[0]
            colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]]
            for k in range(n_objs_to_draw):
                mask_k = gt_masks_np[k] > 0.5
                color = colors[k % len(colors)]
                for c in range(3):
                    gt_vis_np[..., c] = np.where(mask_k, color[c], gt_vis_np[..., c])

            gt_tensor = torch.from_numpy(gt_vis_np).permute(2, 0, 1).float() / 255.0
            if writer:
                writer.add_image(f"StyleAUE/gt_mask", gt_tensor, step)
            # Save to disk
            self._save_image_to_disk(gt_vis_np, step, "StyleAUE", "gt_mask")
        elif gt_mask is not None:
            # Single mask
            gt_mask_np = gt_mask.cpu().numpy() if hasattr(gt_mask, "cpu") else gt_mask
            gt_vis_np = np.zeros((*gt_mask_np.shape[-2:], 3), dtype=np.uint8)
            mask_k = gt_mask_np > 0.5
            # Use cyan for single mask
            gt_vis_np[..., 1] = np.where(mask_k, 255, 0)
            gt_vis_np[..., 2] = np.where(mask_k, 255, 0)
            gt_tensor = torch.from_numpy(gt_vis_np).permute(2, 0, 1).float() / 255.0
            if writer:
                writer.add_image(f"StyleAUE/gt_mask", gt_tensor, step)
            # Save to disk
            self._save_image_to_disk(gt_vis_np, step, "StyleAUE", "gt_mask")

        # Log Prediction Mask (Standalone)
        if pred_mask is not None:
            pred_np = pred_mask.cpu().numpy()
            pred_bin = pred_np > 0.0
            pred_vis_np = np.zeros((*pred_bin.shape[-2:], 3), dtype=np.uint8)
            colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]]
            for k in range(pred_bin.shape[0]):
                mask_k = pred_bin[k]
                color = colors[k % len(colors)]
                for c in range(3):
                    pred_vis_np[..., c] = np.where(mask_k, color[c], pred_vis_np[..., c])

            pred_tensor = torch.from_numpy(pred_vis_np).permute(2, 0, 1).float() / 255.0
            if writer:
                writer.add_image(f"StyleAUE/pred_mask", pred_tensor, step)
            # Save to disk
            self._save_image_to_disk(pred_vis_np, step, "StyleAUE", "pred_mask")

        # IMPORTANT: Flush to disk immediately to prevent data loss from TensorBoard caching
        if writer:
            writer.flush()

        plt.close(fig)

    def _extract_pixel_bndl_model(self, model):
        """Extract the pixel_bndl model from the main SAM2 model"""
        try:
            # Handle DDP wrapper
            if hasattr(model, "module"):
                model = model.module

            # Try to find mask decoder with pixel_bndl
            mask_decoder = None

            # Check common paths
            if hasattr(model, "sam_mask_decoder"):
                mask_decoder = model.sam_mask_decoder
            elif hasattr(model, "mask_decoder"):
                mask_decoder = model.mask_decoder
            elif hasattr(model, "pixel_bndl"):
                mask_decoder = model

            # Return pixel_bndl if found
            if mask_decoder and hasattr(mask_decoder, "pixel_bndl"):
                return mask_decoder.pixel_bndl

            return None
        except Exception:
            return None

    def _extract_bndl_outputs(self, outputs):
        """提取BNDL输出（从统一的 aux_outputs）"""
        # Normalize outputs to an iterable of per-frame dicts
        if isinstance(outputs, dict):
            outputs_iter = [outputs]
        else:
            outputs_iter = outputs

        for frame_idx, outs in enumerate(outputs_iter):
            if "multistep_aux_outputs" in outs:
                aux_list = outs["multistep_aux_outputs"]
                # Use the last valid BNDL output (highest resolution)
                for i in reversed(range(len(aux_list))):
                    if aux_list[i] is not None and isinstance(aux_list[i], dict):
                        bndl_outputs = aux_list[i].get("bndl", None)
                        if bndl_outputs is not None:
                            return bndl_outputs, i, frame_idx
        return None, None, None

    def _calculate_uncertainty_for_bndl(self, bndl_outputs, batch, targets):
        """Calculate uncertainty for BNDL outputs during validation"""
        # Use no_grad to avoid gradient accumulation during uncertainty sampling
        with torch.no_grad():
            # Clear cache before uncertainty calculation to free up memory
            torch.cuda.empty_cache()
            # Extract pixel BNDL model for uncertainty sampling
            pixel_bndl_model = self._extract_pixel_bndl_model(self.model)
            if pixel_bndl_model is None:
                logging.warning("Could not extract pixel_bndl model for PAvPU calculation")
                return bndl_outputs

            factor_z = getattr(pixel_bndl_model, "default_factor_z", 0.0)
            factor_w = getattr(pixel_bndl_model, "default_factor_w", 0.0)

            # Extract pixel features from BNDL outputs
            pixel_feat = bndl_outputs["pixel_feat"]
            mask_tokens_out = bndl_outputs.get("mask_tokens_out", None)
            # If SAM has already selected the final channel, prefer single-channel weights
            mask_tokens_out_selected = bndl_outputs.get("mask_tokens_out_selected", None)
            single_channel_w = None
            if isinstance(mask_tokens_out_selected, torch.Tensor) and mask_tokens_out_selected.ndim == 2:
                # [B, C'] -> [B, 1, C'] for single-channel forward
                single_channel_w = mask_tokens_out_selected.unsqueeze(1)

            mask_tokens_to_use = single_channel_w if single_channel_w is not None else mask_tokens_out

            # Initialize all variables that may be used later
            uncertainty_data = {}
            sampled_logits = None
            mean_pixel_logits = None
            entropy_norm = None
            nll_norm = None
            nll_map = None
            pixel_uncertainty_p = None

            S = self.logging_conf.uncertainty_sample_num

            # Perform parallel sampling once if any metric needs it
            sampled_logits = None
            if any(metric in self.logging_conf.uncertainty_metric for metric in ["entropy", "sampling", "nll"]):
                sampled_logits, mean_logits = uncertainty_sample_parallel(
                    pixel_bndl_model,
                    pixel_feat,
                    mask_tokens_to_use,
                    sample_num=S,
                    factor_z=factor_z,
                    factor_w=factor_w,
                )

            # Compute requested uncertainty metrics
            if "entropy" in self.logging_conf.uncertainty_metric:
                entropy_map = entropy_uncertainty(sampled_logits)  # [B, H, W, K]

                # Reduce K=1 to [B,H,W]
                if entropy_map.ndim == 4 and entropy_map.shape[-1] == 1:
                    entropy_map = entropy_map.squeeze(-1)
                entropy_norm = torch.clamp(entropy_map / math.log(2.0), 0.0, 1.0)
                uncertainty_data["entropy"] = entropy_norm

            if "sampling" in self.logging_conf.uncertainty_metric:
                # Calculate p-value based uncertainty (simplified implementation directly here)
                # using simple variance/std across samples as a proxy if we don't want full t-test
                # or implementing simplified paired t-test logic

                probs = torch.sigmoid(sampled_logits)  # [B, H, W, K, S]
                if probs.shape[-2] == 1:  # Single mask
                    # Compare p vs 1-p
                    p = probs[..., 0, :]  # [B, H, W, S]
                    d = p - (1.0 - p)
                else:
                    # Compare top-2 masks (simplified: just take variance of max prob)
                    # Or stick to p-value between top-2
                    mean_p = probs.mean(-1)  # [B, H, W, K]
                    _, topk_idx = mean_p.topk(2, dim=-1)  # [B, H, W, 2]
                    # Gather top 2 prob samples
                    # This gets complicated to implement efficiently inline
                    # Fallback: simple std dev of max prob
                    max_prob_indices = mean_p.argmax(dim=-1, keepdim=True)  # [B, H, W, 1]
                    # [B, H, W, 1, S]
                    p_max = torch.gather(probs, -2, max_prob_indices.unsqueeze(-1).expand(-1, -1, -1, -1, S)).squeeze(-2)
                    d = p_max - (1.0 - p_max)  # Proxy: confidence in winning class

                # Compute t-statistic on d
                mean_d = d.mean(dim=-1)
                std_d = d.std(dim=-1, unbiased=True).clamp_min(1e-6)
                t_stat = mean_d / (std_d / (float(S) ** 0.5) + 1e-6)
                # Two-sided p-value
                z = t_stat.abs() / 1.4142135623730951
                pixel_uncertainty_p = 2.0 * (1.0 - (0.5 * (1.0 + torch.erf(z))))
                uncertainty_data["sampling"] = pixel_uncertainty_p

            if "nll" in self.logging_conf.uncertainty_metric:
                # Prepare targets: resize to match pixel_feat spatial dims
                B, H_feat, W_feat, C = pixel_feat.shape
                if len(targets.shape) == 4:
                    # [B, H, W, 1] -> [B, H, W]
                    targets_squeezed = targets.squeeze(-1)
                else:
                    targets_squeezed = targets

                if targets_squeezed.shape[-2:] != (H_feat, W_feat):
                    targets_resized = F.interpolate(
                        targets_squeezed.unsqueeze(1).float(),
                        size=(H_feat, W_feat),
                        mode="nearest",
                    ).squeeze(1)
                else:
                    targets_resized = targets_squeezed.float()

                # Expand targets to match K masks if needed
                K = sampled_logits.shape[3]
                targets_expanded = targets_resized.unsqueeze(-1).expand(-1, -1, -1, K)  # [B, H, W, K]

                probs = torch.sigmoid(sampled_logits)  # [B, H, W, K, S]
                mean_probs = probs.mean(dim=-1)  # [B, H, W, K]

                # NLL
                eps = 1e-6
                mean_probs = mean_probs.clamp(eps, 1.0 - eps)
                nll_map = -(targets_expanded * torch.log(mean_probs) + (1.0 - targets_expanded) * torch.log(1.0 - mean_probs))

                if nll_map.ndim == 4 and nll_map.shape[-1] == 1:
                    nll_map = nll_map.squeeze(-1)
                elif nll_map.ndim == 4:
                    # If multiple masks, take simple mean or min? Typically NLL is evaluated against the ground truth for the correct class.
                    # Here assuming targets are broadcasted, so maybe just mean.
                    nll_map = nll_map.mean(dim=-1)

                # Normalize NLL to [0, 1] range
                # Using 3*log(2) ≈ 2.08 as max: corresponds to ~6% probability (reasonable "very wrong" threshold)
                # This is theoretically justified unlike the previous magic number 10.0
                nll_norm = torch.clamp(nll_map / (3.0 * math.log(2)), 0.0, 1.0)
                uncertainty_data["nll"] = nll_norm

            # If no specific metrics requested, default to entropy
            if not uncertainty_data:
                if sampled_logits is None:
                    sampled_logits, _ = uncertainty_sample_parallel(
                        pixel_bndl_model,
                        pixel_feat,
                        mask_tokens_to_use,
                        sample_num=S,
                        factor_z=factor_z,
                        factor_w=factor_w,
                    )
                entropy_map = entropy_uncertainty(sampled_logits)
                if entropy_map.ndim == 4 and entropy_map.shape[-1] == 1:
                    entropy_map = entropy_map.squeeze(-1)
                entropy_norm = torch.clamp(entropy_map / math.log(2.0), 0.0, 1.0)
                uncertainty_data["entropy"] = entropy_norm

            # Get mean logits from sampling if not already computed
            if mean_pixel_logits is None:
                if sampled_logits is not None:
                    mean_pixel_logits = sampled_logits.mean(dim=-1)
                else:
                    # Deterministic forward
                    det_out_tuple = pixel_bndl_model(pixel_feat, mask_tokens_to_use, factor_z=factor_z, factor_w=factor_w)
                    mean_pixel_logits = det_out_tuple[0]
            elif "masks_bndl_raw" not in bndl_outputs:
                # Use sampling logits as fallback if masks_bndl_raw not available
                mean_pixel_logits = bndl_outputs.get("mean_pixel_logits", mean_pixel_logits)

            # Select primary uncertainty metric (prefer entropy)
            if entropy_norm is not None:
                chosen_uncertainty = entropy_norm
            elif nll_norm is not None:
                chosen_uncertainty = nll_norm
            else:
                chosen_uncertainty = pixel_uncertainty_p

            # Store uncertainty and logits for evaluator
            # Accuracy will be calculated by DistributedDatasetEvaluator for consistency
            # Ensure shapes: pixel_uncertainty [B,H,W]; logits [B,H,W,1]
            if chosen_uncertainty.ndim == 4 and chosen_uncertainty.shape[-1] == 1:
                chosen_uncertainty = chosen_uncertainty.squeeze(-1)
            if mean_pixel_logits.ndim == 3:
                mean_pixel_logits = mean_pixel_logits.unsqueeze(-1)
            bndl_outputs["pixel_uncertainty"] = chosen_uncertainty.detach()
            bndl_outputs["mean_pixel_logits"] = mean_pixel_logits.detach()

            # Add uncertainty-specific data
            if len(uncertainty_data) > 1:
                bndl_outputs["multi_uncertainty"] = {k: v.detach() for k, v in uncertainty_data.items()}
                bndl_outputs["uncertainty_type"] = "multi"
            elif "nll" in uncertainty_data:
                bndl_outputs["pixel_nll_raw"] = nll_map.detach()
                bndl_outputs["pixel_nll_normalized"] = nll_norm.detach()
                bndl_outputs["uncertainty_type"] = "nll"
            elif "entropy" in uncertainty_data:
                bndl_outputs["uncertainty_type"] = "entropy"
            else:
                bndl_outputs["uncertainty_type"] = "sampling"

            # Clear cache after uncertainty calculation to free up memory
            torch.cuda.empty_cache()

            return bndl_outputs

    def _has_global_params(self, bndl_outputs):
        """检查是否有全局权重参数"""
        has_new = "wei_lambda_w_pos" in bndl_outputs and "kappa_w_pos" in bndl_outputs and bndl_outputs["wei_lambda_w_pos"] is not None and bndl_outputs["kappa_w_pos"] is not None
        has_legacy = "wei_lambda_w" in bndl_outputs and "kappa_w" in bndl_outputs and bndl_outputs["wei_lambda_w"] is not None and bndl_outputs["kappa_w"] is not None
        return has_new or has_legacy

    def _extract_pixel_params(self, bndl_outputs, batch_idx=0):
        """提取并处理像素级参数"""
        b, c, h, w = bndl_outputs["upscaled_shape"]
        if bndl_outputs.get("wei_lambda_pos") is not None and bndl_outputs.get("kappa_pos") is not None:
            lambda_sum = bndl_outputs["wei_lambda_pos"]
            k_sum = bndl_outputs["kappa_pos"]
            if bndl_outputs.get("wei_lambda_neg") is not None and bndl_outputs.get("kappa_neg") is not None:
                lambda_sum = lambda_sum + bndl_outputs["wei_lambda_neg"]
                k_sum = 0.5 * (k_sum + bndl_outputs["kappa_neg"])
            lambda_vals = lambda_sum.detach().cpu().numpy()
            k_vals = k_sum.detach().cpu().numpy()
        else:
            lambda_vals = bndl_outputs["wei_lambda"].detach().cpu().numpy()  # [B, H, W, C]
            k_vals = bndl_outputs["kappa"].detach().cpu().numpy()

        # Extract specific batch - now working with [B, H, W, C] format
        lambda_batch = lambda_vals[batch_idx]  # [H, W, C]
        k_batch = k_vals[batch_idx]  # [H, W, C]

        # Handle channel dimension - average across channels if multiple channels
        if lambda_batch.shape[-1] > 1:
            lambda_img = lambda_batch.mean(axis=-1)  # [H, W]
            k_img = k_batch.mean(axis=-1)  # [H, W]
        else:
            lambda_img = lambda_batch.squeeze(-1)  # [H, W]
            k_img = k_batch.squeeze(-1)  # [H, W]

        return lambda_img, k_img

    def _extract_mask_prompt_info(self, outputs_for_vis, step_index=0, batch_idx=0):
        """从 mask_inputs 提取边界框或轮廓点用于可视化

        Args:
            outputs_for_vis: 模型输出列表
            step_index: 帧索引
            batch_idx: batch索引

        Returns:
            prompt_info 字典（包含 mask 的 bounding box）或 None
        """
        try:
            if outputs_for_vis is None or not isinstance(outputs_for_vis, list) or len(outputs_for_vis) == 0:
                return None

            if step_index >= len(outputs_for_vis):
                step_index = 0

            step_output = outputs_for_vis[step_index]
            if not isinstance(step_output, dict):
                return None

            # 提取 mask_inputs
            mask_inputs = step_output.get("mask_inputs", None)
            if mask_inputs is None:
                return None

            # mask_inputs 通常是 [B, 1, H, W] 格式
            if hasattr(mask_inputs, "shape") and len(mask_inputs.shape) >= 3:
                # Use batch_idx
                if len(mask_inputs.shape) == 4:
                    idx = min(batch_idx, mask_inputs.shape[0] - 1)
                    mask = mask_inputs[idx, 0]
                else:
                    mask = mask_inputs[0]  # Fallback if no batch dim

                # 转换为 numpy
                if hasattr(mask, "cpu"):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = mask

                # 计算 mask 的 bounding box
                fg_coords = np.argwhere(mask_np > 0.5)  # [[y, x], ...]
                if len(fg_coords) == 0:
                    return None

                # 获取边界框
                y_min, x_min = fg_coords.min(axis=0)
                y_max, x_max = fg_coords.max(axis=0)

                # 构造边界框的两个角点（SAM 使用 box corners）
                box_coords = torch.tensor(
                    [
                        [x_min, y_min],  # 左上角
                        [x_max, y_max],  # 右下角
                    ],
                    dtype=torch.float32,
                ).unsqueeze(0)  # [1, 2, 2]

                box_labels = torch.tensor([[2, 3]], dtype=torch.int32)  # SAM box 标签

                prompt_info = {
                    "point_coords": box_coords,
                    "point_labels": box_labels,
                    "is_box": True,  # 标记这是 box prompt
                }

                return prompt_info

            return None

        except Exception:
            return None

    def _extract_prompt_info(self, outputs_for_vis, step_index=0, batch_idx=0):
        """从模型输出中提取prompt信息（优先从第一帧）

        Args:
            outputs_for_vis: 模型输出列表，每个元素是一帧的输出字典
            step_index: 帧索引
            batch_idx: batch索引

        Returns:
            prompt_info字典，包含point_coords和point_labels，或None
        """
        try:
            if outputs_for_vis is None or not isinstance(outputs_for_vis, list) or len(outputs_for_vis) == 0:
                return None

            # 获取对应帧的输出
            if step_index >= len(outputs_for_vis):
                step_index = 0

            step_output = outputs_for_vis[step_index]
            if not isinstance(step_output, dict):
                return None

            # 优先尝试从 multistep_point_inputs 获取
            final_point_inputs = None
            if "multistep_point_inputs" in step_output:
                point_inputs_list = step_output["multistep_point_inputs"]
                if point_inputs_list and len(point_inputs_list) > 0:
                    # 取第一步的 point inputs（初始 prompts，不是 correction points）
                    if point_inputs_list[0] is not None and isinstance(point_inputs_list[0], dict):
                        final_point_inputs = point_inputs_list[0]

            # 如果 multistep_point_inputs 没有有效数据，尝试使用顶层的 point_inputs
            if final_point_inputs is None and "point_inputs" in step_output:
                top_level_pi = step_output["point_inputs"]
                if top_level_pi is not None and isinstance(top_level_pi, dict):
                    final_point_inputs = top_level_pi

            if final_point_inputs is None:
                return None

            # 验证包含必要的字段
            coords = final_point_inputs.get("point_coords", None)
            labels = final_point_inputs.get("point_labels", None)

            if coords is not None and labels is not None:
                # Slice specific batch index if available
                # coords: [B, N, 2], labels: [B, N]
                if hasattr(coords, "shape") and coords.ndim == 3 and coords.shape[0] > 1:
                    idx = min(batch_idx, coords.shape[0] - 1)
                    coords_slice = coords[idx : idx + 1]  # Keep batch dim [1, N, 2]
                    labels_slice = labels[idx : idx + 1]  # Keep batch dim [1, N]

                    return {"point_coords": coords_slice, "point_labels": labels_slice, "is_box": final_point_inputs.get("is_box", False)}

                return final_point_inputs

            return None

        except Exception:
            return None

    def _extract_original_image(self, batch, frame_idx=0, batch_idx=0):
        """提取并处理原始图像，对应指定的帧索引"""
        if not hasattr(batch, "img_batch"):
            return None

        try:
            img_batch = batch.img_batch
            if hasattr(img_batch, "cpu"):
                img_batch = img_batch.cpu().numpy()

            # Extract specific batch and frame
            if len(img_batch.shape) == 5:  # [T, B, C, H, W]
                T, B = img_batch.shape[0], img_batch.shape[1]
                safe_t = max(0, min(int(frame_idx), T - 1))
                safe_b = max(0, min(int(batch_idx), B - 1))
                orig_tensor = img_batch[safe_t, safe_b]  # [C, H, W]
            elif len(img_batch.shape) == 4:  # [B, C, H, W]
                B = img_batch.shape[0]
                safe_b = max(0, min(int(batch_idx), B - 1))
                orig_tensor = img_batch[safe_b]  # [C, H, W]
            else:
                return None

            # Convert [C, H, W] -> [H, W, C]
            if len(orig_tensor.shape) == 3 and orig_tensor.shape[0] in [1, 3]:
                original_img = orig_tensor.transpose(1, 2, 0)
            else:
                return None

            # Denormalize if needed (ImageNet normalization)
            if original_img.min() < -1 or original_img.max() > 2:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                if len(original_img.shape) == 3 and original_img.shape[-1] == 3:
                    original_img = original_img * std + mean

            # Clip to valid range
            original_img = np.clip(original_img, 0, 1)

            # Ensure 3 channels
            if len(original_img.shape) == 2:
                original_img = np.stack([original_img] * 3, axis=-1)
            elif len(original_img.shape) == 3 and original_img.shape[-1] == 1:
                original_img = np.repeat(original_img, 3, axis=-1)

            return original_img

        except Exception:
            return None

    def _extract_prediction(self, outputs, batch_idx, frame_idx=0):
        """Extract predicted mask for a specific batch index.

        NOTE: Uses high-resolution masks (pred_masks_high_res) for visualization,
        as low-resolution masks (pred_masks) are 256x256 while images are 1024x1024.
        Using low-res masks directly causes noise-like visualization artifacts.
        """
        try:
            # If outputs is a list of frames, get the specific frame
            if isinstance(outputs, list):
                if frame_idx < len(outputs):
                    outputs = outputs[frame_idx]
                else:
                    outputs = outputs[0]  # Fallback

            # Prefer high-resolution masks for visualization
            # pred_masks_high_res: [B, K, H_img, W_img] (1024x1024)
            # pred_masks: [B, K, H*4, W*4] (256x256) - too low for visualization
            if isinstance(outputs, dict):
                # Try high-res first (correct resolution for visualization)
                if "pred_masks_high_res" in outputs:
                    pred_masks = outputs["pred_masks_high_res"]
                    if pred_masks is not None and batch_idx < pred_masks.shape[0]:
                        return pred_masks[batch_idx]

                # Fallback to low-res (but note: will need upsampling for proper visualization)
                if "pred_masks" in outputs:
                    pred_masks = outputs["pred_masks"]
                    if pred_masks is not None and batch_idx < pred_masks.shape[0]:
                        return pred_masks[batch_idx]

            return None
        except Exception:
            return None

    def _extract_gt_mask(self, batch, batch_idx, frame_index=0):
        """Extract Ground Truth mask for validation visualization."""
        try:
            if not hasattr(batch, "masks"):
                return None

            masks = batch.masks  # [T, B, K, H, W] or [B, K, H, W] or [B, H, W]

            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()

            # Handle Temporal dimension
            if len(masks.shape) == 5:  # [T, B, K, H, W]
                # Default to frame_index, but check bounds
                t_idx = min(frame_index, masks.shape[0] - 1)
                masks = masks[t_idx]  # [B, K, H, W]

            # Handle Batch dimension
            if len(masks.shape) >= 2:  # At least [B, ...]
                if batch_idx < masks.shape[0]:
                    mask_b = masks[batch_idx]  # [K, H, W] or [H, W]
                    return mask_b

            return None
        except Exception:
            return None

    def _upsample_params_to_image_size(self, lambda_img, k_img, target_shape):
        """将参数图上采样到目标图像尺寸"""
        target_h, target_w = target_shape[:2]

        if lambda_img.shape != (target_h, target_w):
            lambda_img = cv2.resize(lambda_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            k_img = cv2.resize(k_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return lambda_img, k_img

    def _create_unified_visualization(self, bndl_outputs, batch, outputs_for_vis, vis_dir, data_iter, step_index, frame_index, layout_type="basic"):
        """统一的BNDL可视化方法，使用重构后的模块"""
        from .utils.bndl_visualizer import BNDLVisualizer

        # 初始化可视化器
        bndl_viz = BNDLVisualizer()

        # Determine batch size from batch.img_batch
        batch_size = 1
        if hasattr(batch, "img_batch"):
            img_batch = batch.img_batch
            if hasattr(img_batch, "shape"):
                if len(img_batch.shape) == 5:  # [T, B, C, H, W]
                    batch_size = img_batch.shape[1]
                elif len(img_batch.shape) == 4:  # [B, C, H, W]
                    batch_size = img_batch.shape[0]

        # Select random batch index
        batch_idx = random.randint(0, max(0, batch_size - 1))

        # 提取参数和图像
        lambda_img, k_img = self._extract_pixel_params(bndl_outputs, batch_idx=batch_idx)
        original_img = self._extract_original_image(batch, frame_idx=frame_index, batch_idx=batch_idx)
        gt_mask = self._extract_gt_mask(batch, batch_idx=batch_idx, frame_index=frame_index)

        # 提取prompt信息 - 总是从第一帧（frame 0）提取，因为第一帧是 init_cond_frame，有初始 prompts
        prompt_info = self._extract_prompt_info(outputs_for_vis, step_index=0, batch_idx=batch_idx)

        # 如果没有 point prompts，尝试从 mask_inputs 提取轮廓用于可视化
        if prompt_info is None:
            prompt_info = self._extract_mask_prompt_info(outputs_for_vis, step_index=0, batch_idx=batch_idx)

        if original_img is not None:
            lambda_img, k_img = self._upsample_params_to_image_size(lambda_img, k_img, original_img.shape)

        # 调用BNDLVisualizer的统一可视化方法
        bndl_viz.create_unified_visualization(
            vis_dir=vis_dir,
            data_iter=data_iter,
            step_index=step_index,
            original_img=original_img,
            lambda_img=lambda_img,
            k_img=k_img,
            bndl_outputs=bndl_outputs,
            prompt_info=prompt_info,
            layout_type=layout_type,
            save_individual=True,
            save_unified=True,
            visualize_pavpu_overlay=self.logging_conf.visualize_pavpu_overlay,
            uncertainty_metric=self.logging_conf.uncertainty_metric,
            epoch=self.epoch,
            gt_mask=gt_mask,
        )


def print_model_summary(model: torch.nn.Module, log_dir: str = ""):
    """
    Prints the model and the number of parameters in the model.
    # Multiple packages provide this info in a nice table format
    # However, they need us to provide an `input` (as they also write down the output sizes)
    # Our models are complex, and a single input is restrictive.
    # https://github.com/sksq96/pytorch-summary
    # https://github.com/nmhkahn/torchsummaryX
    """
    if get_rank() != 0:
        return
    param_kwargs = {}
    trainable_parameters = sum(p.numel() for p in model.parameters(**param_kwargs) if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters(**param_kwargs))
    non_trainable_parameters = total_parameters - trainable_parameters
    logging.info("==" * 10)
    logging.info(f"Summary for model {type(model)}")
    logging.info(f"Model is {model}")
    logging.info(f"\tTotal parameters {get_human_readable_count(total_parameters)}")
    logging.info(f"\tTrainable parameters {get_human_readable_count(trainable_parameters)}")
    logging.info(f"\tNon-Trainable parameters {get_human_readable_count(non_trainable_parameters)}")
    logging.info("==" * 10)

    if log_dir:
        output_fpath = os.path.join(log_dir, "model.txt")
        with g_pathmgr.open(output_fpath, "w") as f:
            print(model, file=f)


PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]


def get_human_readable_count(number: int) -> str:
    """
    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.
    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'
    Args:
        number: a positive integer number
    Return:
        A string formatted according to the pattern described above.
    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"
    else:
        return f"{number:,.1f} {labels[index]}"
