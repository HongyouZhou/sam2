# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import cv2
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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
from BNDL.BNDL_upload.ViT_Sparse.utils.bndl import pixel_uncertain_sampling, pixel_pavpu_calculation, pixel_entropy_uncertainty, pixel_nll_uncertainty

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
    uncertainty_metric: set = field(default_factory=lambda: {"entropy"})  # Options: {"entropy"}, {"nll"}, {"sampling"}, {"entropy", "nll"}, {"entropy", "nll", "sampling"}, etc.
    visualize_pavpu_overlay: bool = True  # Enable PAvPU overlay visualization on original images
    uncertainty_sample_num: int = 50

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

        assert (
            is_dist_avail_and_initialized()
        ), "Torch distributed needs to be initialized before calling the trainer."

        self._setup_components()  # Except Optimizer everything is setup here.
        self._move_to_device()
        self._construct_optimizers()
        self._setup_dataloaders()

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")

        if self.checkpoint_conf.resume_from is not None:
            assert os.path.exists(
                self.checkpoint_conf.resume_from
            ), f"The 'resume_from' checkpoint {self.checkpoint_conf.resume_from} does not exist!"
            dst = os.path.join(self.checkpoint_conf.save_dir, "checkpoint.pt")
            if self.distributed_rank == 0 and not os.path.exists(dst):
                # Copy the "resume_from" checkpoint to the checkpoint folder
                # if there is not a checkpoint to resume from already there
                makedir(self.checkpoint_conf.save_dir)
                g_pathmgr.copy(self.checkpoint_conf.resume_from, dst)
            barrier()

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
            torch.backends.cuda.matmul.allow_tf32 = (
                cuda_conf.matmul_allow_tf32
                if cuda_conf.matmul_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )
            torch.backends.cudnn.allow_tf32 = (
                cuda_conf.cudnn_allow_tf32
                if cuda_conf.cudnn_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )

        self.rank = setup_distributed_backend(
            distributed_conf.backend, distributed_conf.timeout_mins
        )

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
        logging.info(
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )

        self.model.to(self.device)

        logging.info(
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )

    def save_checkpoint(self, epoch, checkpoint_names=None):
        checkpoint_folder = self.checkpoint_conf.save_dir
        makedir(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and (int(epoch) % self.checkpoint_conf.save_freq == 0)
            ) or int(epoch) in self.checkpoint_conf.save_list:
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_paths = []
        for ckpt_name in checkpoint_names:
            checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

        state_dict = unwrap_ddp_if_wrapped(self.model).state_dict()
        state_dict = exclude_params_matching_unix_pattern(
            patterns=self.checkpoint_conf.skip_saving_parameters, state_dict=state_dict
        )

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
        model_weight_initializer = instantiate(
            self.checkpoint_conf.model_weight_initializer
        )
        if model_weight_initializer is not None:
            logging.info(
                f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
            )
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
        if getattr(_model, 'use_bndl_for_pixels', False):
            bndl_outputs, step_index, frame_index = self._extract_bndl_outputs(outputs)
            if bndl_outputs is not None:
                # Calculate PAvPU if in validation phase and targets are available
                if phase == "val" and targets is not None:
                    # Use frame_index to get the corresponding mask for the current frame
                    # targets shape: [4, 2, 1024, 1024] -> [frames, batch_size, height, width]
                    # We need to extract the specific frame and transpose to [batch_size, height, width]
                    if frame_index is not None and targets.shape[0] > frame_index:
                        current_frame_targets = targets[frame_index]  # Shape: [2, 1024, 1024]
                    else:
                        current_frame_targets = targets[0] if targets.shape[0] > 0 else targets  # Fallback to first frame
                    bndl_outputs = self._calculate_pavpu_for_bndl(bndl_outputs, batch, current_frame_targets)
                self._log_bndl_statistics(bndl_outputs, self.steps[phase], phase)

        if isinstance(loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()}
            )
            loss = self._log_loss_detailed_and_return_core_loss(
                loss, loss_log_str, self.steps[phase]
            )

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
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))

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

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
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
            with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=(self.optim_conf.amp.enabled if self.optim_conf else False),
                dtype=(
                    get_amp_type(self.optim_conf.amp.amp_dtype)
                    if self.optim_conf
                    else None
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
                    
                    outputs = self.model(batch)

                    # Only process BNDL outputs if BNDL is enabled
                    _model = unwrap_ddp_if_wrapped(self.model)
                    if getattr(_model, 'use_bndl_for_pixels', False):
                        bndl_outputs, step_index, frame_index = self._extract_bndl_outputs(outputs)
                        if bndl_outputs is not None:
                            # Use frame_index to get the corresponding mask for the current frame
                            # batch.masks shape: [4, 2, 1024, 1024] -> [frames, batch_size, height, width]
                            # We need to extract the specific frame and transpose to [batch_size, height, width]
                            if frame_index is not None and batch.masks.shape[0] > frame_index:
                                current_frame_masks = batch.masks[frame_index]  # Shape: [2, 1024, 1024]
                            else:
                                current_frame_masks = batch.masks[0] if batch.masks.shape[0] > 0 else batch.masks  # Fallback to first frame
                            bndl_outputs = self._calculate_pavpu_for_bndl(bndl_outputs, batch, current_frame_masks)
                            # BNDL visualization and evaluation (moved inside the for loop)
                            if (random.random() < 0.15 and self.logging_conf.visualize_bndl) and get_rank() == 0:
                                logging.info(f"Starting BNDL visualization for iter {data_iter}")
                                # Use bndl_outputs already returned from _step instead of re-extracting
                                vis_dir = os.path.join(self.logging_conf.log_dir, "bndl_visualizations", phase)
                                makedir(vis_dir)
                                # Ensure PAvPU is calculated for visualization
                                self._create_unified_visualization(bndl_outputs, batch, outputs, vis_dir, data_iter, step_index, frame_index, 'full')
                                # After BNDL visualization, also visualize AdCo negatives (bank + sampled)
                                self._visualize_adco_negatives(phase=phase, data_iter=data_iter)
                            
                            # Dataset evaluation using BNDL outputs (use postprocessed masks)
                            pixel_predictions = bndl_outputs.get('masks_bndl_postprocessed', bndl_outputs.get('masks_bndl_raw'))
                            self.dataset_evaluator.add_batch_data(
                                uncertainty=bndl_outputs['pixel_uncertainty'],
                                pred_logits=pixel_predictions,
                                gt_masks=current_frame_masks
                            )
                            logging.info(f"Added batch {data_iter} to dataset evaluator (batch size: {pixel_predictions.shape[0]})")
                                                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

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

            if data_iter % 10 == 0:
                dist.barrier()

        total_images = self.dataset_evaluator.get_total_images_across_all_processes()
        logging.info(f"Dataset evaluator status: {len(self.dataset_evaluator)} images on rank {self.rank}, {total_images} total across all processes")
        
        # 评估相关性
        correlation_results = self.dataset_evaluator.evaluate_dataset_correlation()
        logging.info(f"Correlation evaluation completed with {len(correlation_results)} metrics")
        
        # 生成可视化
        self.dataset_evaluator.create_dataset_correlation_visualization(
            title=f"Epoch {self.epoch} - Dataset Analysis",
            save_name=f"epoch_{self.epoch}_dataset_analysis.png"
        )
        # 保存结果
        self.dataset_evaluator.save_correlation_results(
            save_name=f"epoch_{self.epoch}_results.json"
        )
        logging.info(f"Dataset evaluation completed for epoch {self.epoch}")
        # 重置evaluator
        self.dataset_evaluator.reset()

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

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
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
            batch = batch.to(
                self.device, non_blocking=True
            )  # move tensors in a tensorclass

            try:
                self._run_step(batch, phase, loss_mts, extra_loss_mts)

                # compute gradient and do optim step
                exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
                self.where = float(exact_epoch) / self.max_epochs
                assert self.where <= 1 + self.EPSILON
                if self.where < 1.0:
                    self.optim.step_schedulers(
                        self.where, step=int(exact_epoch * iters_per_epoch)
                    )
                else:
                    logging.warning(
                        f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                    )

                # Log schedulers
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for j, param_group in enumerate(self.optim.optimizer.param_groups):
                        for option in self.optim.schedulers[j]:
                            optim_prefix = (
                                "" + f"{j}_"
                                if len(self.optim.optimizer.param_groups) > 1
                                else ""
                            )
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
                    self.gradient_logger(
                        self.model, rank=self.distributed_rank, where=self.where
                    )

                # Optimizer step: the scaler will make sure gradients are not
                # applied if the gradients are infinite
                self.scaler.step(self.optim.optimizer)
                self.scaler.update()

                # measure elapsed time
                batch_time_meter.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )

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

                # (train) We keep AdCo negatives visualization off to avoid overhead

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
                extra_loss_mts[extra_loss_key] = AverageMeter(
                    extra_loss_key, self.device, ":.2e"
                )
            extra_loss_mts[extra_loss_key].update(extra_loss.item(), batch_size)

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

                    if (
                        self.checkpoint_conf.save_best_meters is not None
                        and key in self.checkpoint_conf.save_best_meters
                    ):
                        checkpoint_save_keys.append(tracked_meter_key.replace("/", "_"))

        if len(checkpoint_save_keys) > 0:
            self.save_checkpoint(self.epoch + 1, checkpoint_save_keys)

        return out_dict

    def _log_timers(self, phase):
        time_remaining = 0
        epochs_remaining = self.max_epochs - self.epoch - 1
        val_epochs_remaining = sum(
            n % self.val_epoch_freq == 0 for n in range(self.epoch, self.max_epochs)
        )

        # Adding the guaranteed val run at the end if val_epoch_freq doesn't coincide with
        # the end epoch.
        if (self.max_epochs - 1) % self.val_epoch_freq != 0:
            val_epochs_remaining += 1

        # Remove the current val run from estimate
        if phase == Phase.VAL:
            val_epochs_remaining -= 1

        time_remaining += (
            epochs_remaining * self.est_epoch_time[Phase.TRAIN]
            + val_epochs_remaining * self.est_epoch_time[Phase.VAL]
        )

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
            assert len(val_keys) == len(
                set(val_keys)
            ), f"Duplicate keys in val datasets, keys: {val_keys}"

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
                    f"Keys in val datasets do not match the keys in losses."
                    f"\nMissing in losses: {set(val_keys) - loss_keys}"
                    f"\nMissing in val datasets: {loss_keys - set(val_keys)}"
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

        # 获取像素级统计配置参数
        foreground_dilation = getattr(self.logging_conf, 'correlation_foreground_dilation', 10)
        per_pixel = getattr(self.logging_conf, 'correlation_per_pixel', True)

        self.dataset_evaluator = DistributedDatasetEvaluator(
            save_dir=os.path.join(self.logging_conf.log_dir, "dataset_evaluation"),
            distributed=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            foreground_dilation=foreground_dilation,
            per_pixel_statistics=per_pixel
        )

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
        """Log BNDL statistics including pixel-level uncertainty and PAvsPU"""
        if bndl_outputs is None:
            return
            
        # Pixel-level parameters (lambda and k)
        if ('wei_lambda' in bndl_outputs and 'inv_k' in bndl_outputs and 
            bndl_outputs['wei_lambda'] is not None and bndl_outputs['inv_k'] is not None):
            lambda_mean = bndl_outputs['wei_lambda'].mean().detach()
            k_mean = (1. / (bndl_outputs['inv_k'] + 1e-6)).mean().detach()
            self.logger.log(f"Stats/{phase}_lambda_pixel", lambda_mean, step)
            self.logger.log(f"Stats/{phase}_k_pixel", k_mean, step)
            
            # Log pixel uncertainty if available
            if 'pixel_uncertainty' in bndl_outputs and bndl_outputs['pixel_uncertainty'] is not None:
                uncertainty_mean = bndl_outputs['pixel_uncertainty'].mean().detach()
                self.logger.log(f"Stats/{phase}_pixel_uncertainty", uncertainty_mean, step)
            
            # Log PAvsPU scores if available
            if 'pixel_pavpu' in bndl_outputs and bndl_outputs['pixel_pavpu'] is not None:
                pavpu_scores = bndl_outputs['pixel_pavpu']
                thresholds = bndl_outputs.get('pavpu_thresholds', [0.01, 0.05, 0.1])
                for i, threshold in enumerate(thresholds):
                    if i < len(pavpu_scores):
                        score_val = pavpu_scores[i].item() if hasattr(pavpu_scores[i], 'item') else pavpu_scores[i]
                        self.logger.log(f"Stats/{phase}_pavpu_{threshold}", score_val, step)
        
        # Global w statistics (original BNDL)
        if ('wei_lambda_w' in bndl_outputs and 
            'inv_k_w' in bndl_outputs and
            bndl_outputs['wei_lambda_w'] is not None and 
            bndl_outputs['inv_k_w'] is not None):
            lambda_w_mean = bndl_outputs['wei_lambda_w'].mean().detach()
            k_w_mean = (1. / (bndl_outputs['inv_k_w'] + 1e-6)).mean().detach()
            self.logger.log(f"Stats/{phase}_lambda_w", lambda_w_mean, step)
            self.logger.log(f"Stats/{phase}_k_w", k_w_mean, step)

    def _extract_pixel_bndl_model(self, model):
        """Extract the pixel_bndl model from the main SAM2 model"""
        try:
            # Handle DDP wrapper
            if hasattr(model, 'module'):
                model = model.module
            
            # Try to find mask decoder with pixel_bndl
            mask_decoder = None
            
            # Check common paths
            if hasattr(model, 'sam_mask_decoder'):
                mask_decoder = model.sam_mask_decoder
            elif hasattr(model, 'mask_decoder'):
                mask_decoder = model.mask_decoder
            elif hasattr(model, 'pixel_bndl'):
                mask_decoder = model
            
            # Return pixel_bndl if found
            if mask_decoder and hasattr(mask_decoder, 'pixel_bndl'):
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

    def _calculate_pavpu_for_bndl(self, bndl_outputs, batch, targets):
        """Calculate PAvPU metric for BNDL outputs during validation"""
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
            pixel_feat = bndl_outputs['pixel_feat']
            hyper_in = bndl_outputs.get('hyper_in', None)
            
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
                    external_pre_out_w=hyper_in,
                    sample_num=self.logging_conf.uncertainty_sample_num,
                )
                entropy_norm = torch.clamp(entropy_map / math.log(2.0), 0.0, 1.0)
                uncertainty_data['entropy'] = entropy_norm
            
            if "sampling" in self.logging_conf.uncertainty_metric:
                pixel_uncertainty_p, mean_pixel_logits = pixel_uncertain_sampling(
                    pixel_bndl_model,
                    pixel_feat,
                    external_pre_out_w=hyper_in,
                    sample_num=self.logging_conf.uncertainty_sample_num,
                )
                uncertainty_data['sampling'] = pixel_uncertainty_p
            
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
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)  # [B, 1, H_feat, W_feat] -> [B, H_feat, W_feat]
                else:
                    targets_resized = targets
                
                # Add channel dimension: [B, H, W] -> [B, H, W, 1]
                targets_4d = targets_resized.unsqueeze(-1)
                
                nll_map, mean_pixel_logits = pixel_nll_uncertainty(
                    pixel_bndl_model,
                    pixel_feat,
                    gt_masks=targets_4d,
                    external_pre_out_w=hyper_in,
                    sample_num=self.logging_conf.uncertainty_sample_num,
                )
                nll_norm = torch.clamp(nll_map / 10.0, 0.0, 1.0)
                uncertainty_data['nll'] = nll_norm
            
            # If no specific metrics requested, default to entropy
            if not uncertainty_data:
                entropy_map = pixel_entropy_uncertainty(
                    pixel_bndl_model,
                    pixel_feat,
                    external_pre_out_w=hyper_in,
                    sample_num=self.logging_conf.uncertainty_sample_num,
                )
                entropy_norm = torch.clamp(entropy_map / math.log(2.0), 0.0, 1.0)
                uncertainty_data['entropy'] = entropy_norm
            # Also get mean logits for downstream use
            _, mean_pixel_logits = pixel_uncertain_sampling(
                pixel_bndl_model,
                pixel_feat,
                external_pre_out_w=hyper_in,
                sample_num=1,
            )
            
            # Prepare ground truth masks for PAvPU calculation
            pixel_targets = self._prepare_targets_for_pavpu(targets, bndl_outputs)
            
            # Calculate PAvPU scores
            pixel_predictions = bndl_outputs.get('masks_bndl_raw', mean_pixel_logits)

            if pixel_predictions.shape != pixel_targets.shape:
                if (pixel_predictions.shape[1:3] != pixel_targets.shape[1:3]):
                    pixel_targets = F.interpolate(
                        pixel_targets.permute(0, 3, 1, 2),
                        size=pixel_predictions.shape[1:3],
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                    logging.info("Aligned spatial dimensions for PAvPU calculation")
            
            # Select primary uncertainty metric for PAvPU calculation (prefer entropy)
            if entropy_norm is not None:
                chosen_uncertainty = entropy_norm
                thresholds = [0.10, 0.20, 0.30, 0.40, 0.60, 0.80]
            elif nll_norm is not None:
                chosen_uncertainty = nll_norm
                thresholds = [0.10, 0.20, 0.30, 0.40, 0.60, 0.80]
            else:
                chosen_uncertainty = pixel_uncertainty_p
                thresholds = [0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.300]

            pavpu_scores = pixel_pavpu_calculation(
                chosen_uncertainty,
                pixel_predictions,
                pixel_targets,
                thresholds=thresholds,
            )
            
            # Add PAvPU results to BNDL outputs
            bndl_outputs['pixel_uncertainty'] = chosen_uncertainty.detach()
            bndl_outputs['pixel_pavpu'] = pavpu_scores
            bndl_outputs['pavpu_thresholds'] = thresholds
            bndl_outputs['mean_pixel_logits'] = mean_pixel_logits.detach()
            
            # Add uncertainty-specific data
            if len(uncertainty_data) > 1:
                bndl_outputs['multi_uncertainty'] = {k: v.detach() for k, v in uncertainty_data.items()}
                bndl_outputs['uncertainty_type'] = 'multi'
            elif 'nll' in uncertainty_data:
                bndl_outputs['pixel_nll_raw'] = nll_map.detach()
                bndl_outputs['pixel_nll_normalized'] = nll_norm.detach()
                bndl_outputs['uncertainty_type'] = 'nll'
            elif 'entropy' in uncertainty_data:
                bndl_outputs['uncertainty_type'] = 'entropy'
            else:
                bndl_outputs['uncertainty_type'] = 'sampling'
            
            # logging.info(f"PAvPU scores calculated: {pavpu_scores}")
            
            # Clear cache after uncertainty calculation to free up memory
            torch.cuda.empty_cache()
            
            return bndl_outputs
 
    def _prepare_targets_for_pavpu(self, targets, bndl_outputs):
        """Prepare ground truth targets in the correct format for PAvPU calculation"""
        try:
            if targets is None:
                return None
            
            # Extract target tensor
            if isinstance(targets, torch.Tensor):
                target_tensor = targets
            elif isinstance(targets, list | tuple) and len(targets) > 0:
                target_tensor = targets[0]
            elif hasattr(targets, 'masks'):
                target_tensor = targets.masks
            else:
                logging.warning(f"Unknown target format: {type(targets)}")
                return None
            
            # Handle the new format: [B, H, W] -> [B, H, W, 1]
            # This is for single-class segmentation where targets are [batch_size, height, width]
            if len(target_tensor.shape) == 3:
                # Add channel dimension for single class: [B, H, W] -> [B, H, W, 1]
                target_tensor = target_tensor.unsqueeze(-1)
            elif len(target_tensor.shape) == 4:
                # Handle common format conversions
                if target_tensor.shape[0] < target_tensor.shape[1] and target_tensor.shape[2] == target_tensor.shape[3]:
                    target_tensor = target_tensor.permute(1, 2, 3, 0)  # [K, B, H, W] -> [B, H, W, K]
                elif target_tensor.shape[1] > target_tensor.shape[0] and target_tensor.shape[1] > target_tensor.shape[2]:
                    target_tensor = target_tensor.permute(0, 2, 3, 1)  # [B, K, H, W] -> [B, H, W, K]
            elif len(target_tensor.shape) == 5:
                target_tensor = target_tensor[:, 0, :, :, :].permute(0, 2, 3, 1)  # [B, T, K, H, W] -> [B, H, W, K]
            else:
                logging.warning(f"Unexpected target shape: {target_tensor.shape}")
                return None
            
            # Clean and validate tensor
            target_tensor = torch.nan_to_num(target_tensor, nan=0.0)
            target_tensor = torch.clamp(target_tensor, 0.0, 1.0)
            
            target_tensor = target_tensor.to(bndl_outputs['masks_bndl_raw'].device)
            
            return target_tensor.detach()
            
        except Exception as e:
            logging.warning(f"Failed to prepare targets for PAvPU: {e}")
            return None

    def _has_global_params(self, bndl_outputs):
        """检查是否有全局权重参数"""
        return ('wei_lambda_w' in bndl_outputs and 'inv_k_w' in bndl_outputs and
                bndl_outputs['wei_lambda_w'] is not None and 
                bndl_outputs['inv_k_w'] is not None)

    def _extract_pixel_params(self, bndl_outputs, batch_idx=0):
        """提取并处理像素级参数"""
        b, c, h, w = bndl_outputs['upscaled_shape']
        
        lambda_vals = bndl_outputs['wei_lambda'].detach().cpu().numpy()  # [B, H, W, C]
        inv_k_vals = bndl_outputs['inv_k'].detach().cpu().numpy()       # [B, H, W, C] 
        k_vals = 1.0 / (inv_k_vals + 1e-6)
        
        # Extract specific batch - now working with [B, H, W, C] format
        lambda_batch = lambda_vals[batch_idx]  # [H, W, C]
        k_batch = k_vals[batch_idx]           # [H, W, C]
        
        # Handle channel dimension - average across channels if multiple channels
        if lambda_batch.shape[-1] > 1:
            lambda_img = lambda_batch.mean(axis=-1)  # [H, W]
            k_img = k_batch.mean(axis=-1)           # [H, W]
        else:
            lambda_img = lambda_batch.squeeze(-1)   # [H, W]
            k_img = k_batch.squeeze(-1)            # [H, W]
        
        return lambda_img, k_img

    def _extract_original_image(self, batch, frame_idx=0, batch_idx=0):
        """提取并处理原始图像，对应指定的帧索引"""
        if not hasattr(batch, 'img_batch'):
            return None
            
        try:
            img_batch = batch.img_batch
            if hasattr(img_batch, 'cpu'):
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

    def _create_unified_visualization(self, bndl_outputs, batch, outputs_for_vis, vis_dir, data_iter, step_index, frame_index, layout_type='basic'):
        """统一的BNDL可视化方法，使用重构后的模块"""
        from .utils.visualization_utils import VisualizationUtils
        from .utils.bndl_visualizer import BNDLVisualizer
        
        # 初始化可视化器
        viz_utils = VisualizationUtils()
        bndl_viz = BNDLVisualizer()
        
        # 提取参数和图像
        lambda_img, k_img = self._extract_pixel_params(bndl_outputs)
        original_img = self._extract_original_image(batch, frame_idx=frame_index)
        
        if original_img is not None:
            lambda_img, k_img = self._upsample_params_to_image_size(lambda_img, k_img, original_img.shape)
        
        has_uncertainty = ('pixel_uncertainty' in bndl_outputs and 
                          bndl_outputs['pixel_uncertainty'] is not None)
        has_pavpu = (self.logging_conf.visualize_pavpu_overlay and 
                    'pixel_pavpu' in bndl_outputs and 
                    bndl_outputs['pixel_pavpu'] is not None)
        
        # 检查是否有数据支持比值可视化
        has_ratio_data = ('pixel_uncertainty' in bndl_outputs and 
                         'mean_pixel_logits' in bndl_outputs and
                         bndl_outputs['pixel_uncertainty'] is not None and
                         bndl_outputs['mean_pixel_logits'] is not None)
        
        # 根据布局类型决定行数
        if layout_type == 'full' and has_uncertainty:
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
        
        # 绘制通用元素
        self._plot_common_elements_refactored(axes, original_img, lambda_img, k_img, step_index, 
                                            bndl_outputs, has_uncertainty, batch, outputs_for_vis, bndl_viz, viz_utils)
        
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
            
    def _plot_common_elements_refactored(self, axes, original_img, lambda_img, k_img, step_index, 
                                        bndl_outputs, has_uncertainty=False, batch=None, outputs_for_vis=None, 
                                        bndl_viz=None, viz_utils=None):
        viz_utils.plot_original_image(axes[0, 0], original_img)
        viz_utils.plot_parameter_heatmap(axes[0, 1], lambda_img, f'Lambda (λ) Step {step_index}', 'viridis')
        viz_utils.plot_parameter_heatmap(axes[0, 2], k_img, f'Shape (k) Step {step_index}', 'plasma')
        
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

    def _visualize_adco_negatives(self, phase: str, data_iter: int):
        """Minimal visualization for AdCo negative bank and sampled negatives.

        Saves two optional grids under log_dir/bndl_visualizations/{phase}/adco_negatives/:
        - epoch_{e}_iter_{i}_adco_bank.png (learnable bank snapshot)
        - epoch_{e}_iter_{i}_adco_sampled.png (on-the-fly sampled negatives)
        """
        # Locate underlying model (unwrap DDP)
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        # Build output directory
        vis_dir = os.path.join(self.logging_conf.log_dir, "bndl_visualizations", phase, "adco_negatives")
        makedir(vis_dir)

        def _to_numpy_grid(img_tensor: torch.Tensor, max_items: int = 16) -> np.ndarray:
            """Convert a [N, 3, H, W] tensor to a grid numpy image [Hgrid, Wgrid, 3] in [0,1]."""
            n = min(int(max_items), int(img_tensor.shape[0]))
            sel = img_tensor[:n].detach().to("cpu")
            # Per-image min-max normalize to [0,1]
            imgs = []
            for t in sel:
                c, h, w = t.shape
                t_flat = t.view(c, -1)
                t_min = t_flat.min(dim=1, keepdim=True).values
                t_max = t_flat.max(dim=1, keepdim=True).values
                denom = torch.clamp(t_max - t_min, min=1e-6)
                t_norm = ((t - t_min.view(c, 1, 1)) / denom.view(c, 1, 1)).clamp(0.0, 1.0)
                imgs.append(t_norm.permute(1, 2, 0).numpy())  # H W C
            # Arrange into a square-ish grid
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))
            h, w, _ = imgs[0].shape
            grid = np.ones((rows * h, cols * w, 3), dtype=np.float32)
            idx = 0
            for r in range(rows):
                for cidx in range(cols):
                    if idx >= n:
                        break
                    grid[r * h:(r + 1) * h, cidx * w:(cidx + 1) * w, :] = imgs[idx]
                    idx += 1
            return grid

        # 1) Visualize the learnable negative bank if present
        try:
            neg_bank = getattr(model, "adco_negatives", None)
            if isinstance(neg_bank, torch.Tensor) and neg_bank.ndim == 4 and neg_bank.shape[1] == 3:
                bank_grid = _to_numpy_grid(neg_bank)
                plt.figure(figsize=(12, 12))
                plt.imshow(bank_grid)
                plt.axis("off")
                save_path = os.path.join(
                    vis_dir,
                    f"epoch_{self.epoch}_iter_{data_iter}_adco_bank.png",
                )
                plt.tight_layout()
                plt.savefig(save_path, dpi=150)
                plt.close()
        except Exception as e:
            logging.warning(f"AdCo bank visualization failed: {e}")

        # 2) Visualize a fresh sample via model._adco_sample_neg_images if available
        try:
            sample_fn = getattr(model, "_adco_sample_neg_images", None)
            if callable(sample_fn):
                sampled = sample_fn(16)
                if isinstance(sampled, torch.Tensor) and sampled.ndim == 4 and sampled.shape[1] == 3:
                    sampled_grid = _to_numpy_grid(sampled)
                    plt.figure(figsize=(12, 12))
                    plt.imshow(sampled_grid)
                    plt.axis("off")
                    save_path = os.path.join(
                        vis_dir,
                        f"epoch_{self.epoch}_iter_{data_iter}_adco_sampled.png",
                    )
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=150)
                    plt.close()
        except Exception as e:
            logging.warning(f"AdCo sampled visualization failed: {e}")


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
    trainable_parameters = sum(
        p.numel() for p in model.parameters(**param_kwargs) if p.requires_grad
    )
    total_parameters = sum(p.numel() for p in model.parameters(**param_kwargs))
    non_trainable_parameters = total_parameters - trainable_parameters
    logging.info("==" * 10)
    logging.info(f"Summary for model {type(model)}")
    logging.info(f"Model is {model}")
    logging.info(f"\tTotal parameters {get_human_readable_count(total_parameters)}")
    logging.info(
        f"\tTrainable parameters {get_human_readable_count(trainable_parameters)}"
    )
    logging.info(
        f"\tNon-Trainable parameters {get_human_readable_count(non_trainable_parameters)}"
    )
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
