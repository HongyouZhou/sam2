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

matplotlib.use("Agg")  # 使用非交互式后端
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
from BNDL.BNDL_upload.ViT_Sparse.utils.bndl import pixel_uncertain_sampling, pixel_entropy_uncertainty, pixel_nll_uncertainty

from training.utils.dataset_evaluator import DistributedDatasetEvaluator

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
    uncertainty_metric: set = field(default_factory=lambda: {"entropy"})  # Options: {"entropy"}, {"nll"}, {"sampling"}, {"entropy", "nll"}, {"entropy", "nll", "sampling"}, etc.
    visualize_pavpu_overlay: bool = True  # Enable PAvPU overlay visualization on original images
    uncertainty_sample_num: int = 50
    correlation_foreground_dilation: int = 0  # Foreground dilation radius (pixels), 0 means no dilation (only used when use_full_image=False)
    correlation_per_pixel: bool = True  # Use per-pixel statistics (vs per-image)
    correlation_use_full_image: bool = True  # Use full image statistics (True, default) or foreground only (False)


@dataclass
class AlternatingBackwardConf:
    enabled: bool = False
    ddp_no_sync: bool = True
    symmetric_average: bool = True  # 0.5 per phase
    include_adv_seg: bool = False  # reserved; not used initially
    sample_adv_every: int = 1  # reserved; not used initially


@dataclass
class TrainStrategyConf:
    alternating_backward: AlternatingBackwardConf = field(default_factory=AlternatingBackwardConf)


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
        train_strategy: Optional[Dict[str, Any]] = None,
    ):
        self._setup_env_variables(env_variables)
        self._setup_timers()

        self.data_conf = data
        self.model_conf = model
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint).infer_missing()
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.optim_conf = OptimConf(**optim) if optim is not None else None
        self.meters_conf = meters
        self.loss_conf = loss
        distributed = DistributedConf(**distributed or {})
        cuda = CudaConf(**cuda or {})
        self.train_strategy = TrainStrategyConf(**(train_strategy or {}))
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
        return epoch % self.val_epoch_freq == 0 and epoch < self.max_epochs - 1

    def _step(
        self,
        batch: BatchedVideoDatapoint,
        model: nn.Module,
        phase: str,
    ):
        outputs = model(batch)
        targets = batch.masks
        batch_size = len(batch.img_batch)

        key = batch.dict_key  # key for dataset
        loss = self.loss[key](outputs, targets)
        loss_str = f"Losses/{phase}_{key}_loss"

        loss_log_str = os.path.join("Step_Losses", loss_str)

        # loss contains multiple sub-components we wish to log
        step_losses = {}

        # Log BNDL statistics from model outputs if available (only if BNDL is enabled)
        _model = unwrap_ddp_if_wrapped(self.model)
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

        if isinstance(loss, dict):
            step_losses.update({f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()})
            loss = self._log_loss_detailed_and_return_core_loss(loss, loss_log_str, self.steps[phase])

        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(
                loss_log_str,
                loss,
                self.steps[phase],
            )

        self.steps[phase] += 1

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
        if not hasattr(self, 'train_dataset'):
            self.train_dataset = None
        if not hasattr(self, 'val_dataset'):
            self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))

        if self.mode in ["train", "train_only"] and self.train_dataset is None:
            self.train_dataset = instantiate(self.data_conf.train)

    def _setup_dataloaders_early(self):
        """Setup train dataset early for AUE initialization (before moving model to GPU)"""
        if self.mode in ["train", "train_only"]:
            self.train_dataset = instantiate(self.data_conf.train)

    def run_train(self):
        while self.epoch < self.max_epochs:
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
                        if k not in extra_loss_mts:
                            extra_loss_mts[k] = AverageMeter(k, self.device, ":.2e")
                        extra_loss_mts[k].update(v.item(), batch_size)

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

            # Randomly visualize BNDL parameters during validation
            if (
                self.distributed_rank == 0
                and self.logging_conf.visualize_bndl
                and getattr(unwrap_ddp_if_wrapped(self.model), "use_bndl_for_pixels", False)
                and random.random() < self.logging_conf.bndl_vis_sample_rate
            ):
                # Extract BNDL outputs for visualization
                _model = unwrap_ddp_if_wrapped(self.model)
                if hasattr(batch, "img_batch"):
                    # Re-run forward pass to get outputs with BNDL data
                    with torch.no_grad():
                        outputs_for_vis = _model(batch)
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
            if (
                data_iter % self.logging_conf.log_visual_frequency == 0
                and self.distributed_rank == 0
                and hasattr(unwrap_ddp_if_wrapped(self.model), "use_aue")
                and unwrap_ddp_if_wrapped(self.model).use_aue
            ):
                self._visualize_aue_adversarials(phase, data_iter)

            if data_iter % 10 == 0:
                dist.barrier()

        # Use val_evaluator for validation epoch analysis
        if self.val_evaluator is not None:
            total_images = self.val_evaluator.get_total_images_across_all_processes()
            logging.info(f"Val evaluator status: {len(self.val_evaluator)} images on rank {self.rank}, {total_images} total across all processes")

            # 评估相关性
            correlation_results = self.val_evaluator.evaluate_dataset_correlation()
            logging.info(f"Correlation evaluation completed with {len(correlation_results)} metrics")

            # 生成可视化
            self.val_evaluator.create_dataset_correlation_visualization(title=f"Epoch {self.epoch} - Validation PAvPU Analysis", save_name=f"epoch_{self.epoch}_val_pavpu_analysis.png")
            # 保存结果
            self.val_evaluator.save_correlation_results(save_name=f"epoch_{self.epoch}_val_pavpu_results.json")
            logging.info(f"Validation PAvPU evaluation completed for epoch {self.epoch}")
            # 重置evaluator准备下一个epoch
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
        end = time.time()

        for data_iter, batch in enumerate(train_loader):
            # measure data loading time
            data_time_meter.update(time.time() - end)
            data_times.append(data_time_meter.val)
            batch = batch.to(self.device, non_blocking=True)  # move tensors in a tensorclass

            try:
                self._run_step(batch, phase, loss_mts, extra_loss_mts)

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

                # Clipping gradients and detecting diverging gradients
                if self.gradient_clipper is not None:
                    self.scaler.unscale_(self.optim.optimizer)
                    self.gradient_clipper(model=self.model)

                if self.gradient_logger is not None:
                    self.gradient_logger(self.model, rank=self.distributed_rank, where=self.where)

                # Optimizer step: the scaler will make sure gradients are not
                # applied if the gradients are infinite
                self.scaler.step(self.optim.optimizer)
                self.scaler.update()

                # measure elapsed time
                batch_time_meter.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)

                mem_meter.update(reset_peak_usage=True)
                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)

                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    # Log progress meters.
                    for progress_meter in progress.meters:
                        self.logger.log(
                            os.path.join("Step_Stats", phase, progress_meter.name),
                            progress_meter.val,
                            self.steps[phase],
                        )

                # (train) BNDL visualization disabled during training to avoid overhead

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
        Run the forward / backward
        """
        # it's important to set grads to None, especially with Adam since 0
        # grads will also update a model even if the step doesn't produce
        # gradients
        self.optim.zero_grad(set_to_none=True)

        ab_conf = getattr(self.train_strategy, "alternating_backward", AlternatingBackwardConf())

        def _get_component(loss_map: Dict[str, torch.Tensor], suffix: str) -> torch.Tensor:
            for k, v in loss_map.items():
                if k.endswith("_" + suffix):
                    return v
            return torch.tensor(0.0, device=self.device)

        if not ab_conf.enabled:
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

            self.scaler.scale(loss).backward()
            loss_mts[loss_key].update(loss.item(), batch_size)
            for extra_loss_key, extra_loss in extra_losses.items():
                if extra_loss_key not in extra_loss_mts:
                    extra_loss_mts[extra_loss_key] = AverageMeter(extra_loss_key, self.device, ":.2e")
                extra_loss_mts[extra_loss_key].update(extra_loss.item(), batch_size)
            return

        # Alternating two-phase backward within a single optimizer step
        ddp_no_sync = ab_conf.ddp_no_sync and isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
        ddp_ctx = self.model.no_sync() if ddp_no_sync else nullcontext()

        # Wrap both phases in the same no_sync context to ensure proper gradient accumulation
        with ddp_ctx:
            # Phase 1: Main
            unwrap_ddp_if_wrapped(self.model).__setattr__("_aue_phase", "main")
            with torch.cuda.amp.autocast(
                enabled=self.optim_conf.amp.enabled,
                dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
            ):
                loss_dict_main, batch_size_main, extra_losses_main = self._step(
                    batch,
                    self.model,
                    phase,
                )

            assert len(loss_dict_main) == 1
            loss_key, _ = loss_dict_main.popitem()

            sam_core = _get_component(extra_losses_main, "sam_core_loss")
            bndl_core = _get_component(extra_losses_main, "bndl_core_loss")
            ur_core = _get_component(extra_losses_main, "ur_ern_core_loss")
            aue_core_main = _get_component(extra_losses_main, "aue_core_loss")
            coef = 0.5 if ab_conf.symmetric_average else 1.0
            loss_main_adjusted = sam_core + bndl_core + ur_core + coef * aue_core_main

            if not math.isfinite(loss_main_adjusted.item()):
                error_msg = f"Adjusted main loss is {loss_main_adjusted.item()}, attempting to stop training"
                logging.error(error_msg)
                if raise_on_error:
                    raise FloatingPointError(error_msg)
                else:
                    return

            self.scaler.scale(loss_main_adjusted).backward()
            for extra_loss_key, extra_loss in extra_losses_main.items():
                if extra_loss_key not in extra_loss_mts:
                    extra_loss_mts[extra_loss_key] = AverageMeter(extra_loss_key, self.device, ":.2e")
                extra_loss_mts[extra_loss_key].update(extra_loss.item(), batch_size_main)

            # Phase 2: Adv
            unwrap_ddp_if_wrapped(self.model).__setattr__("_aue_phase", "adv")
            with torch.cuda.amp.autocast(
                enabled=self.optim_conf.amp.enabled,
                dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
            ):
                loss_dict_adv, batch_size_adv, extra_losses_adv = self._step(
                    batch,
                    self.model,
                    phase,
                )

            aue_core_adv = _get_component(extra_losses_adv, "aue_core_loss")
            coef = 0.5 if ab_conf.symmetric_average else 1.0
            loss_adv_adjusted = coef * aue_core_adv

            if not math.isfinite(loss_adv_adjusted.item()):
                error_msg = f"Adjusted adv loss is {loss_adv_adjusted.item()}, attempting to stop training"
                logging.error(error_msg)
                if raise_on_error:
                    raise FloatingPointError(error_msg)
                else:
                    return

            self.scaler.scale(loss_adv_adjusted).backward()
            for extra_loss_key, extra_loss in extra_losses_adv.items():
                if extra_loss_key not in extra_loss_mts:
                    extra_loss_mts[extra_loss_key] = AverageMeter(extra_loss_key, self.device, ":.2e")
                extra_loss_mts[extra_loss_key].update(extra_loss.item(), batch_size_adv)

        total_loss = (loss_main_adjusted + loss_adv_adjusted).detach()
        loss_mts[loss_key].update(total_loss.item(), batch_size_main)

        unwrap_ddp_if_wrapped(self.model).__setattr__("_aue_phase", None)

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
        epochs_remaining = self.max_epochs - self.epoch - 1
        val_epochs_remaining = sum(n % self.val_epoch_freq == 0 for n in range(self.epoch, self.max_epochs))

        # Adding the guaranteed val run at the end if val_epoch_freq doesn't coincide with
        # the end epoch.
        if (self.max_epochs - 1) % self.val_epoch_freq != 0:
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
        
        if hasattr(self.model, "use_aue") and self.model.use_aue:
            # 根据配置决定是否从数据集初始化对抗样本
            if self.model_conf.get("aue_init_from_dataset", True):
                if not hasattr(self, 'train_dataset'):
                    self.train_dataset = None
                    # Try to setup train dataset for AUE initialization (only in train modes)
                    if self.mode in ["train", "train_only"]:
                        self._setup_dataloaders_early()
                
                if self.train_dataset is not None:
                    logging.info("Initializing AUE adversarial samples from dataset...")
                    # Use multiprocessing with 'spawn' context (safe even when CUDA initialized)
                    self.model.init_aue_adversarials_from_dataset(
                        self.train_dataset, num_workers=4
                    )
                else:
                    logging.info("Skipping AUE adversarial samples initialization (train_dataset not available in val-only mode)")
            else:
                logging.info("AUE adversarial samples initialized with random values (aue_init_from_dataset=False)")
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

        self.gradient_clipper = (
            instantiate(self.optim_conf.gradient_clip) if self.optim_conf else None
        )
        self.gradient_logger = (
            instantiate(self.optim_conf.gradient_logger) if self.optim_conf else None
        )

        logging.info("Finished setting up components: Model, loss, optim, meters etc.")
        
    def _setup_evaluators(self):
        """Setup separate evaluators for training and validation phases"""
        # Get configuration
        foreground_dilation = getattr(self.logging_conf, "correlation_foreground_dilation", 0)
        per_pixel = getattr(self.logging_conf, "correlation_per_pixel", True)

        # Get use_full_image config (default: True for better calibration)
        use_full_image = getattr(self.logging_conf, "correlation_use_full_image", True)

        # Create evaluators for train and val phases
        self.val_evaluator = DistributedDatasetEvaluator(
            save_dir=os.path.join(self.logging_conf.log_dir, "val_pavpu_evaluation"),
            distributed=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            foreground_dilation=foreground_dilation,
            per_pixel_statistics=per_pixel,
            use_full_image=use_full_image,
        )

        # Train evaluator can be initialized similarly if needed
        # For now, we only use val_evaluator since PAvPU analysis is mainly done during validation
        logging.info("Initialized PAvPU evaluators for training")

    def _add_to_evaluator(self, bndl_outputs, targets, evaluator):
        """Add BNDL outputs to evaluator for PAvPU analysis"""
        try:
            uncertainty = bndl_outputs.get("pixel_uncertainty")
            pred_logits = bndl_outputs.get("mean_pixel_logits")

            # Normalize shapes for evaluator: uncertainty [B,H,W], logits [B,H,W,1]
            if isinstance(uncertainty, torch.Tensor) and uncertainty.ndim == 4:
                # [B,H,W,K] → prefer K=1, else fallback channel 0
                if uncertainty.shape[-1] == 1:
                    uncertainty = uncertainty.squeeze(-1)
                else:
                    uncertainty = uncertainty[..., 0]
            if isinstance(pred_logits, torch.Tensor):
                if pred_logits.ndim == 3:
                    pred_logits = pred_logits.unsqueeze(-1)
                elif pred_logits.ndim == 4 and pred_logits.shape[-1] > 1:
                    pred_logits = pred_logits[..., 0:1]

            if uncertainty is not None and pred_logits is not None and targets is not None:
                evaluator.add_batch_data(uncertainty=uncertainty, pred_logits=pred_logits, gt_masks=targets)
        except Exception as e:
            logging.warning(f"Failed to add data to evaluator: {e}")

    def _construct_optimizers(self):
        self.optim = construct_optimizer(
            self.model,
            self.optim_conf.optimizer,
            self.optim_conf.options,
            self.optim_conf.param_group_modifiers,
        )

    def _log_loss_detailed_and_return_core_loss(self, loss, loss_str, step):
        core_loss = loss.pop(CORE_LOSS_KEY)
        if step % self.logging_conf.log_scalar_frequency == 0:
            for k in loss:
                log_str = os.path.join(loss_str, k)
                self.logger.log(log_str, loss[k], step)
        return core_loss

    def _log_bndl_statistics(self, bndl_outputs, step, phase):
        """Log BNDL statistics including pixel-level uncertainty"""
        if bndl_outputs is None:
            return

        # Pixel-level parameters (lambda and k)
        if "wei_lambda" in bndl_outputs and "inv_k" in bndl_outputs and bndl_outputs["wei_lambda"] is not None and bndl_outputs["inv_k"] is not None:
            lambda_mean = bndl_outputs["wei_lambda"].mean().detach()
            k_mean = (1.0 / (bndl_outputs["inv_k"] + 1e-6)).mean().detach()
            self.logger.log(f"Stats/{phase}_lambda_pixel", lambda_mean, step)
            self.logger.log(f"Stats/{phase}_k_pixel", k_mean, step)

            # Log pixel uncertainty if available
            if "pixel_uncertainty" in bndl_outputs and bndl_outputs["pixel_uncertainty"] is not None:
                uncertainty_mean = bndl_outputs["pixel_uncertainty"].mean().detach()
                self.logger.log(f"Stats/{phase}_pixel_uncertainty", uncertainty_mean, step)

        # Global w statistics (original BNDL)
        if "wei_lambda_w" in bndl_outputs and "inv_k_w" in bndl_outputs and bndl_outputs["wei_lambda_w"] is not None and bndl_outputs["inv_k_w"] is not None:
            lambda_w_mean = bndl_outputs["wei_lambda_w"].mean().detach()
            k_w_mean = (1.0 / (bndl_outputs["inv_k_w"] + 1e-6)).mean().detach()
            self.logger.log(f"Stats/{phase}_lambda_w", lambda_w_mean, step)
            self.logger.log(f"Stats/{phase}_k_w", k_w_mean, step)

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
        except Exception as e:
            logging.warning(f"Failed to extract pixel_bndl model: {e}")
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

            # Extract pixel features from BNDL outputs
            pixel_feat = bndl_outputs["pixel_feat"]
            hyper_in = bndl_outputs.get("hyper_in", None)
            # If SAM has already selected the final channel, prefer single-channel weights
            hyper_in_selected = bndl_outputs.get("hyper_in_selected", None)
            single_channel_w = None
            if isinstance(hyper_in_selected, torch.Tensor) and hyper_in_selected.ndim == 2:
                # [B, C'] -> [B, 1, C'] for single-channel forward
                single_channel_w = hyper_in_selected.unsqueeze(1)

            # Decide which uncertainty to compute based on set configuration
            mean_pixel_logits = None
            pixel_uncertainty_p = None
            entropy_norm = None
            nll_map = None
            nll_norm = None
            uncertainty_data = {}

            # Compute requested uncertainty metrics
            if "entropy" in self.logging_conf.uncertainty_metric:
                entropy_map = pixel_entropy_uncertainty(
                    pixel_bndl_model,
                    pixel_feat,
                    external_pre_out_w=(single_channel_w if single_channel_w is not None else hyper_in),
                    sample_num=self.logging_conf.uncertainty_sample_num,
                    per_channel=True,  # [B, H, W, K] (K=1 if single channel)
                )
                # Reduce K=1 to [B,H,W]
                if entropy_map.ndim == 4 and entropy_map.shape[-1] == 1:
                    entropy_map = entropy_map.squeeze(-1)
                entropy_norm = torch.clamp(entropy_map / math.log(2.0), 0.0, 1.0)
                uncertainty_data["entropy"] = entropy_norm

            if "sampling" in self.logging_conf.uncertainty_metric:
                if single_channel_w is None:
                    # Multi-channel path: use top-2 paired test as before
                    pixel_uncertainty_pval, mean_pixel_logits = pixel_uncertain_sampling(
                        pixel_bndl_model,
                        pixel_feat,
                        external_pre_out_w=hyper_in,
                        sample_num=self.logging_conf.uncertainty_sample_num,
                    )
                    pixel_uncertainty_p = pixel_uncertainty_pval
                else:
                    # Single-channel paired test against background (p vs 1-p)
                    B, H, W, Cprime = pixel_feat.shape
                    S = self.logging_conf.uncertainty_sample_num
                    sampled_logits = torch.zeros(B, H, W, S, device=pixel_feat.device, dtype=pixel_feat.dtype)
                    for i in range(S):
                        out_i, *_ = pixel_bndl_model(
                            pixel_feat, force_sample=True, external_pre_out_w=single_channel_w
                        )  # [B,H,W,1]
                        sampled_logits[..., i] = out_i[..., 0]
                    # probs per sample
                    probs = torch.sigmoid(sampled_logits)  # [B,H,W,S]
                    # Paired t-test: A=probs, B=1-probs
                    d = probs - (1.0 - probs)  # [B,H,W,S]
                    mean_d = d.mean(dim=-1)
                    std_d = d.std(dim=-1, unbiased=True).clamp_min(1e-6)
                    t_stat = mean_d / (std_d / (float(S) ** 0.5) + 1e-6)
                    z = t_stat.abs() / 1.4142135623730951
                    phi = 0.5 * (1.0 + torch.erf(z))
                    pixel_uncertainty_p = (2.0 * (1.0 - phi)).to(mean_d.dtype)  # [B,H,W]
                uncertainty_data["sampling"] = pixel_uncertainty_p

            if "nll" in self.logging_conf.uncertainty_metric:
                # Prepare targets: resize to match pixel_feat spatial dims and add channel dim
                B, H_feat, W_feat, C = pixel_feat.shape

                # Ensure targets is 3D [B, H, W]
                if len(targets.shape) == 4:
                    targets = targets.squeeze(-1)  # [B, H, W, 1] -> [B, H, W]

                # Resize if needed
                if targets.shape[-2:] != (H_feat, W_feat):
                    targets_resized = F.interpolate(
                        targets.unsqueeze(1).float(),  # [B, H, W] -> [B, 1, H, W]
                        size=(H_feat, W_feat),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)  # [B, 1, H_feat, W_feat] -> [B, H_feat, W_feat]
                else:
                    targets_resized = targets

                # Add channel dimension: [B, H, W] -> [B, H, W, 1]
                targets_4d = targets_resized.unsqueeze(-1)

                nll_map, mean_pixel_logits = pixel_nll_uncertainty(
                    pixel_bndl_model,
                    pixel_feat,
                    gt_masks=targets_4d,
                    external_pre_out_w=(single_channel_w if single_channel_w is not None else hyper_in),
                    sample_num=self.logging_conf.uncertainty_sample_num,
                    per_channel=True,  # [B,H,W,K] (K=1 if single channel)
                )
                if nll_map.ndim == 4 and nll_map.shape[-1] == 1:
                    nll_map = nll_map.squeeze(-1)
                nll_norm = torch.clamp(nll_map / 10.0, 0.0, 1.0)
                uncertainty_data["nll"] = nll_norm

            # If no specific metrics requested, default to entropy
            if not uncertainty_data:
                entropy_map = pixel_entropy_uncertainty(
                    pixel_bndl_model,
                    pixel_feat,
                    external_pre_out_w=hyper_in,
                    sample_num=self.logging_conf.uncertainty_sample_num,
                    per_channel=True,  # Return [B, H, W, K] for per-channel statistics
                )
                entropy_norm = torch.clamp(entropy_map / math.log(2.0), 0.0, 1.0)
                uncertainty_data["entropy"] = entropy_norm

            # Get mean logits from sampling if not already computed
            if mean_pixel_logits is None:
                # Get one deterministic forward for logits on the chosen channel(s)
                if single_channel_w is not None:
                    det_out, *_ = pixel_bndl_model(
                        pixel_feat, force_sample=False, external_pre_out_w=single_channel_w
                    )  # [B,H,W,1]
                    mean_pixel_logits = det_out
                else:
                    _, mean_pixel_logits = pixel_uncertain_sampling(
                        pixel_bndl_model,
                        pixel_feat,
                        external_pre_out_w=hyper_in,
                        sample_num=1,
                    )
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

            # logging.info(f"PAvPU scores calculated: {pavpu_scores}")

            # Clear cache after uncertainty calculation to free up memory
            torch.cuda.empty_cache()

            return bndl_outputs

    def _has_global_params(self, bndl_outputs):
        """检查是否有全局权重参数"""
        return "wei_lambda_w" in bndl_outputs and "inv_k_w" in bndl_outputs and bndl_outputs["wei_lambda_w"] is not None and bndl_outputs["inv_k_w"] is not None

    def _extract_pixel_params(self, bndl_outputs, batch_idx=0):
        """提取并处理像素级参数"""
        b, c, h, w = bndl_outputs["upscaled_shape"]

        lambda_vals = bndl_outputs["wei_lambda"].detach().cpu().numpy()  # [B, H, W, C]
        inv_k_vals = bndl_outputs["inv_k"].detach().cpu().numpy()  # [B, H, W, C]
        k_vals = 1.0 / (inv_k_vals + 1e-6)

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

    def _extract_mask_prompt_info(self, outputs_for_vis, step_index=0):
        """从 mask_inputs 提取边界框或轮廓点用于可视化
        
        Args:
            outputs_for_vis: 模型输出列表
            step_index: 帧索引
            
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
            if hasattr(mask_inputs, 'shape') and len(mask_inputs.shape) >= 3:
                mask = mask_inputs[0, 0] if len(mask_inputs.shape) == 4 else mask_inputs[0]  # [H, W]
                
                # 转换为 numpy
                if hasattr(mask, 'cpu'):
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
                box_coords = torch.tensor([
                    [x_min, y_min],  # 左上角
                    [x_max, y_max]   # 右下角
                ], dtype=torch.float32).unsqueeze(0)  # [1, 2, 2]
                
                box_labels = torch.tensor([[2, 3]], dtype=torch.int32)  # SAM box 标签
                
                prompt_info = {
                    'point_coords': box_coords,
                    'point_labels': box_labels,
                    'is_box': True  # 标记这是 box prompt
                }
                
                logging.info(f"✓ Extracted bounding box from mask_inputs for visualization")
                return prompt_info
            
            return None
            
        except Exception as e:
            logging.warning(f"Failed to extract mask prompt info: {e}")
            return None
    
    def _extract_prompt_info(self, outputs_for_vis, step_index=0):
        """从模型输出中提取prompt信息（优先从第一帧）

        Args:
            outputs_for_vis: 模型输出列表，每个元素是一帧的输出字典
            step_index: 帧索引

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
                return final_point_inputs

            return None

        except Exception as e:
            logging.warning(f"Failed to extract prompt info: {e}")
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
                T = img_batch.shape[0]
                safe_t = max(0, min(int(frame_idx), T - 1))
                orig_tensor = img_batch[safe_t, batch_idx]  # [C, H, W]
            elif len(img_batch.shape) == 4:  # [B, C, H, W]
                orig_tensor = img_batch[batch_idx]  # [C, H, W]
            else:
                logging.warning(f"Unexpected img_batch shape: {img_batch.shape}")
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

        except Exception as e:
            logging.warning(f"Failed to process original image: {e}")
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
        from .utils.visualization_utils import VisualizationUtils
        from .utils.bndl_visualizer import BNDLVisualizer

        # 初始化可视化器
        viz_utils = VisualizationUtils()
        bndl_viz = BNDLVisualizer()

        # 提取参数和图像
        lambda_img, k_img = self._extract_pixel_params(bndl_outputs)
        original_img = self._extract_original_image(batch, frame_idx=frame_index)

        # 提取prompt信息 - 总是从第一帧（frame 0）提取，因为第一帧是 init_cond_frame，有初始 prompts
        prompt_info = self._extract_prompt_info(outputs_for_vis, step_index=0)
        
        # 如果没有 point prompts，尝试从 mask_inputs 提取轮廓用于可视化
        if prompt_info is None:
            prompt_info = self._extract_mask_prompt_info(outputs_for_vis, step_index=0)

        if original_img is not None:
            lambda_img, k_img = self._upsample_params_to_image_size(lambda_img, k_img, original_img.shape)

        has_uncertainty = "pixel_uncertainty" in bndl_outputs and bndl_outputs["pixel_uncertainty"] is not None
        has_pavpu = self.logging_conf.visualize_pavpu_overlay and "pixel_pavpu" in bndl_outputs and bndl_outputs["pixel_pavpu"] is not None

        # 检查是否有数据支持比值可视化
        has_ratio_data = (
            "pixel_uncertainty" in bndl_outputs and "mean_pixel_logits" in bndl_outputs and bndl_outputs["pixel_uncertainty"] is not None and bndl_outputs["mean_pixel_logits"] is not None
        )

        # 根据布局类型决定行数
        if layout_type == "full" and has_uncertainty:
            if has_pavpu and has_ratio_data:
                rows = 6  # 添加PAvPU overlay行和比值可视化行
            elif has_pavpu or has_ratio_data:
                rows = 5  # 添加其中一种可视化
            else:
                rows = 4
        else:
            rows = 3

        # 使用重构后的工具创建图表布局
        fig, axes = viz_utils.create_figure_layout(rows, 3, (18, 6 * rows))

        # 绘制通用元素（传递prompt信息）
        self._plot_common_elements_refactored(axes, original_img, lambda_img, k_img, step_index, bndl_outputs, has_uncertainty, batch, outputs_for_vis, bndl_viz, viz_utils, prompt_info)

        # 添加PAvPU overlay可视化
        current_row = 4
        if has_pavpu and rows >= 5:
            bndl_viz.plot_pavpu_overlay_visualization(axes[current_row, :], bndl_outputs, original_img, step_index)
            current_row += 1

        # 添加U/A比值可视化
        if has_ratio_data and rows >= current_row + 1:
            bndl_viz.plot_uncertainty_accuracy_ratio_visualization(axes[current_row, :], bndl_outputs, original_img, step_index, ratio_type="U/A")

        save_path = os.path.join(vis_dir, f"epoch_{self.epoch}_iter_{data_iter}_step_{step_index}_unified_{layout_type}.png")
        viz_utils.save_and_close_figure(fig, save_path, dpi=150)

        # logging.info(f"Unified BNDL visualization saved: {save_path}")

    def _plot_common_elements_refactored(
        self, axes, original_img, lambda_img, k_img, step_index, bndl_outputs, has_uncertainty=False, batch=None, outputs_for_vis=None, bndl_viz=None, viz_utils=None, prompt_info=None
    ):
        viz_utils.plot_original_image(axes[0, 0], original_img, prompt_info=prompt_info)
        viz_utils.plot_parameter_heatmap(axes[0, 1], lambda_img, f"Lambda (λ) Step {step_index}", "viridis")
        viz_utils.plot_parameter_heatmap(axes[0, 2], k_img, f"Shape (k) Step {step_index}", "plasma")

        if original_img is not None and original_img.shape[:2] == lambda_img.shape:
            if has_uncertainty:
                bndl_viz.plot_parameter_and_uncertainty_overlays(axes[1, :], original_img, lambda_img, k_img, bndl_outputs, step_index)
            else:
                viz_utils.plot_parameter_overlays(axes[1, :], original_img, lambda_img, k_img, step_index)
        else:
            viz_utils.plot_parameter_distributions(axes[1, :], lambda_img, k_img, step_index)

        bndl_viz.plot_global_parameters_in_layout(axes[2, :], bndl_outputs, step_index)
        if has_uncertainty:
            # Use multi-uncertainty visualization if multiple metrics are requested
            if len(self.logging_conf.uncertainty_metric) > 1:
                bndl_viz.plot_multi_uncertainty_visualization(axes[3, :], bndl_outputs, step_index)
            else:
                bndl_viz.plot_uncertainty_visualization(axes[3, :], bndl_outputs, step_index)

    def _visualize_aue_adversarials(self, phase: str, data_iter: int):
        """Visualization for AUE adversarial image bank with sample images and statistics.

        Saves plots under log_dir/bndl_visualizations/{phase}/aue_adversarials/:
        - epoch_{e}_iter_{i}_aue_images.png (actual adversarial sample images)
        - epoch_{e}_iter_{i}_aue_stats.png (image statistics and uncertainty)
        """
        # Locate underlying model (unwrap DDP)
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        # Build output directory
        vis_dir = os.path.join(self.logging_conf.log_dir, "bndl_visualizations", phase, "aue_adversarials")
        makedir(vis_dir)

        try:
            # Get adversarial image bank (expected shape: [K, 3, H, W])
            adv_images = getattr(model, "aue_adversarials", None)
            adv_uncertainty = getattr(model, "aue_adversarial_uncertainty", None)
            adv_prompts = getattr(model, "adv_prompts", None)  # [K, 4] bounding boxes
            adv_gt = getattr(model, "adv_gt", None)  # [K, H, W] ground truth masks

            if adv_images is None:
                return

            # Validate image format
            if adv_images.ndim != 4 or adv_images.shape[1] != 3:
                logging.warning(f"Unexpected aue_adversarials shape: {adv_images.shape}, expected [K, 3, H, W]")
                return

            K, C, H, W = adv_images.shape
            adv_imgs_np = adv_images.detach().cpu().numpy()

            # ==== Visualization 1: Display actual adversarial sample images ====
            fig1, axes1 = plt.subplots(3, 4, figsize=(16, 12))
            axes1 = axes1.flatten()

            num_display = min(12, K)
            for i in range(num_display):
                # 简单可视化：按初始化保持 [0,1]，否则直接裁剪
                img = adv_imgs_np[i].transpose(1, 2, 0)
                img = np.clip(img, 0.0, 1.0)

                axes1[i].imshow(img)

                # Add title with raw statistics (to detect changes even if visually similar)
                raw_img = adv_imgs_np[i]  # Original [3, H, W] values
                title = f"Adv #{i}\n"
                title += f"μ={raw_img.mean():.3f} σ={raw_img.std():.3f}\n"
                title += f"[{raw_img.min():.2f}, {raw_img.max():.2f}]"
                if adv_uncertainty is not None and i < len(adv_uncertainty):
                    unc_val = adv_uncertainty[i].item()
                    title += f"\nU={unc_val:.3f}"

                axes1[i].set_title(title, fontsize=9)

                # Draw bounding box if available
                if adv_prompts is not None and i < len(adv_prompts):
                    x1, y1, x2, y2 = adv_prompts[i].cpu().numpy()
                    # Convert normalized coords to pixel coords if needed
                    if x2 <= 1.0:  # Normalized coordinates
                        x1, x2 = x1 * W, x2 * W
                        y1, y2 = y1 * H, y2 * H
                    rect_w, rect_h = x2 - x1, y2 - y1
                    from matplotlib.patches import Rectangle

                    rect = Rectangle((x1, y1), rect_w, rect_h, linewidth=2, edgecolor="red", facecolor="none")
                    axes1[i].add_patch(rect)

                axes1[i].axis("off")

            # Hide unused subplots
            for i in range(num_display, 12):
                axes1[i].axis("off")

            plt.tight_layout()
            save_path1 = os.path.join(vis_dir, f"epoch_{self.epoch}_iter_{data_iter}_aue_images.png")
            plt.savefig(save_path1, dpi=150)
            plt.close(fig1)
            logging.info(f"AUE adversarial images saved: {save_path1}")

            # ==== Visualization 2: Statistics and uncertainty ====
            fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))

            # 2.1: Image pixel intensity distribution
            pixel_means = adv_imgs_np.reshape(K, -1).mean(axis=1)
            axes2[0, 0].hist(pixel_means, bins=50, alpha=0.7, color="blue", edgecolor="black")
            axes2[0, 0].set_title(f"Mean Pixel Intensity (K={K})")
            axes2[0, 0].set_xlabel("Mean Intensity")
            axes2[0, 0].set_ylabel("Count")
            axes2[0, 0].grid(True, alpha=0.3)

            # 2.2: Image pixel standard deviation
            pixel_stds = adv_imgs_np.reshape(K, -1).std(axis=1)
            axes2[0, 1].hist(pixel_stds, bins=50, alpha=0.7, color="green", edgecolor="black")
            axes2[0, 1].set_title(f"Pixel Std Dev (K={K})")
            axes2[0, 1].set_xlabel("Std Dev")
            axes2[0, 1].set_ylabel("Count")
            axes2[0, 1].grid(True, alpha=0.3)

            # 2.3: Per-channel mean across all images
            channel_means = adv_imgs_np.mean(axis=(0, 2, 3))  # [3]
            axes2[0, 2].bar(["R", "G", "B"], channel_means, color=["red", "green", "blue"], alpha=0.7)
            axes2[0, 2].set_title("Mean per Channel")
            axes2[0, 2].set_ylabel("Mean Intensity")
            axes2[0, 2].grid(True, alpha=0.3, axis="y")

            # Row 2: Uncertainty statistics (if available)
            if adv_uncertainty is not None and adv_uncertainty.numel() > 0:
                uq_np = adv_uncertainty.detach().cpu().numpy()

                # 2.4: Uncertainty distribution
                axes2[1, 0].hist(uq_np, bins=50, alpha=0.7, color="purple", edgecolor="black")
                axes2[1, 0].set_title(f"Adversarial Uncertainty Distribution (K={K})")
                axes2[1, 0].set_xlabel("Uncertainty")
                axes2[1, 0].set_ylabel("Count")
                axes2[1, 0].grid(True, alpha=0.3)

                # 2.5: Uncertainty vs pixel intensity
                axes2[1, 1].scatter(pixel_means, uq_np, alpha=0.5, s=20, c="purple")
                axes2[1, 1].set_title("Uncertainty vs Mean Pixel Intensity")
                axes2[1, 1].set_xlabel("Mean Pixel Intensity")
                axes2[1, 1].set_ylabel("Uncertainty")
                axes2[1, 1].grid(True, alpha=0.3)

                # 2.6: Top-k uncertain adversarials
                top_k = min(20, K)
                top_indices = np.argsort(uq_np)[-top_k:]
                axes2[1, 2].barh(range(top_k), uq_np[top_indices], alpha=0.7, color="orange")
                axes2[1, 2].set_title(f"Top-{top_k} Hardest Adversarials")
                axes2[1, 2].set_xlabel("Uncertainty")
                axes2[1, 2].set_ylabel("Sample Index (sorted)")
                axes2[1, 2].grid(True, alpha=0.3)
            else:
                # Display ground truth masks with original images if available
                if adv_gt is not None and adv_gt.numel() > 0:
                    gt_np = adv_gt.detach().cpu().numpy()
                    num_gt_display = min(3, len(gt_np))
                    for i in range(num_gt_display):
                        # 简单可视化背景图
                        img = adv_imgs_np[i].transpose(1, 2, 0)
                        img = np.clip(img, 0.0, 1.0)

                        # Display original image
                        axes2[1, i].imshow(img)

                        # Overlay mask with transparency
                        mask = gt_np[i]
                        # Create colored mask overlay (red for mask regions)
                        mask_overlay = np.zeros((*mask.shape, 4))
                        mask_overlay[mask > 0.5] = [1, 0, 0, 0.4]  # Red with 40% opacity
                        axes2[1, i].imshow(mask_overlay)

                        axes2[1, i].set_title(f"Image + GT Mask #{i}")
                        axes2[1, i].axis("off")
                else:
                    for ax in axes2[1, :]:
                        ax.text(0.5, 0.5, "Uncertainty not available", ha="center", va="center", transform=ax.transAxes)
                        ax.axis("off")

            plt.tight_layout()
            save_path2 = os.path.join(vis_dir, f"epoch_{self.epoch}_iter_{data_iter}_aue_stats.png")
            plt.savefig(save_path2, dpi=150)
            plt.close(fig2)
            logging.info(f"AUE adversarial statistics saved: {save_path2}")

        except Exception as e:
            logging.warning(f"AUE adversarial visualization failed: {e}")
            import traceback

            logging.warning(traceback.format_exc())


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
