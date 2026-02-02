# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Gradient Conflict Detection Utilities.

Provides tools to detect and monitor gradient conflicts in multi-task
and adversarial training settings. Useful for diagnosing issues where
different loss components push parameters in opposing directions.

Usage:
    from training.utils.gradient_conflict import GradientConflictMonitor

    monitor = GradientConflictMonitor(model, target_params=["bndl", "iou"])

    # In training loop:
    metrics = monitor.compute_conflict(
        losses={"sam": sam_loss, "bndl": bndl_loss, "aue": aue_loss},
        log_to_tensorboard=True,
        writer=tensorboard_writer,
        step=global_step,
    )
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientConflictMonitor:
    """Monitor gradient conflicts between different loss components."""

    def __init__(
        self,
        model: nn.Module,
        target_params: list[str] | None = None,
        sample_rate: float = 0.1,  # Only sample 10% of steps by default
    ):
        """
        Initialize gradient conflict monitor.

        Args:
            model: The model to monitor
            target_params: List of parameter name patterns to monitor (e.g., ["bndl", "iou"])
                          If None, monitors all parameters
            sample_rate: Fraction of steps to actually compute (expensive operation)
        """
        self.model = model
        self.target_params = target_params or []
        self.sample_rate = sample_rate
        self._step_count = 0
        self._logger = logging.getLogger(__name__)

    def _should_compute(self) -> bool:
        """Determine if we should compute this step (sampling for efficiency)."""
        # _step_count is incremented in compute_conflict_from_loss_dict
        return (self._step_count % max(1, int(1 / self.sample_rate))) == 0

    def _get_target_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Get parameters matching target patterns."""
        if not self.target_params:
            return list(self.model.named_parameters())

        params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(t in name.lower() for t in self.target_params):
                params.append((name, param))
        return params

    def compute_conflict(
        self,
        losses: dict[str, torch.Tensor],
        log_to_tensorboard: bool = False,
        writer: Any | None = None,
        step: int = 0,
    ) -> dict[str, float]:
        """
        Compute gradient conflicts between different loss components.

        Args:
            losses: Dict of loss name -> loss tensor (must have retain_graph=True capability)
            log_to_tensorboard: Whether to log to TensorBoard
            writer: TensorBoard SummaryWriter
            step: Current training step

        Returns:
            Dict of metrics:
            - GradConflict/{loss_a}_vs_{loss_b}/cosine_mean: Mean cosine similarity
            - GradConflict/{loss_a}_vs_{loss_b}/conflict_ratio: Fraction of params with cos < 0
            - GradNorm/{loss_name}/mean: Mean gradient norm for each loss
        """
        if not self._should_compute():
            return {}

        target_params = self._get_target_parameters()
        if not target_params:
            return {}

        metrics = {}
        gradients = {}

        # Collect gradients from each loss
        for loss_name, loss_val in losses.items():
            if loss_val is None or not loss_val.requires_grad:
                continue

            self.model.zero_grad()
            try:
                loss_val.backward(retain_graph=True)
            except RuntimeError:
                # Graph may have been freed
                self._logger.warning(f"Could not compute gradients for {loss_name}")
                continue

            grads = {}
            grad_norms = []
            for name, param in target_params:
                if param.grad is not None:
                    grads[name] = param.grad.clone().detach()
                    grad_norms.append(param.grad.norm().item())

            gradients[loss_name] = grads
            if grad_norms:
                metrics[f"GradNorm/{loss_name}/mean"] = sum(grad_norms) / len(grad_norms)
                metrics[f"GradNorm/{loss_name}/max"] = max(grad_norms)

        # Compute pairwise cosine similarities
        loss_names = list(gradients.keys())
        for i, name_a in enumerate(loss_names):
            for name_b in loss_names[i + 1 :]:
                grads_a = gradients[name_a]
                grads_b = gradients[name_b]

                cos_sims = []
                conflict_count = 0
                total_count = 0

                for param_name in grads_a:
                    if param_name not in grads_b:
                        continue

                    g_a = grads_a[param_name].flatten()
                    g_b = grads_b[param_name].flatten()

                    if g_a.numel() == 0 or g_b.numel() == 0:
                        continue

                    cos_sim = F.cosine_similarity(g_a.unsqueeze(0), g_b.unsqueeze(0)).item()
                    cos_sims.append(cos_sim)

                    total_count += 1
                    if cos_sim < 0:
                        conflict_count += 1

                if cos_sims:
                    key_prefix = f"GradConflict/{name_a}_vs_{name_b}"
                    metrics[f"{key_prefix}/cosine_mean"] = sum(cos_sims) / len(cos_sims)
                    metrics[f"{key_prefix}/cosine_min"] = min(cos_sims)
                    metrics[f"{key_prefix}/conflict_ratio"] = conflict_count / total_count if total_count > 0 else 0

        # Log to TensorBoard
        if log_to_tensorboard and writer is not None:
            for key, value in metrics.items():
                writer.add_scalar(key, value, step)

        return metrics

    def compute_conflict_from_loss_dict(
        self,
        loss_dict: dict[str, torch.Tensor],
        writer: Any | None = None,
        step: int = 0,
    ) -> dict[str, float]:
        """
        Compute gradient conflicts from CombinedSAMBNDLLoss output dict.

        This is the primary method to use in training. It extracts *_core_loss
        components and computes pairwise gradient conflicts.

        Args:
            loss_dict: Dict from CombinedSAMBNDLLoss.forward() containing
                      sam_core_loss, bndl_core_loss, aue_core_loss, etc.
            writer: TensorBoard SummaryWriter
            step: Current training step

        Returns:
            Dict of gradient conflict metrics
        """
        # Increment step count first
        self._step_count += 1

        # Log on first call to help debug
        if self._step_count == 1:
            all_core_losses = [k for k in loss_dict.keys() if "_core_loss" in k]
            self._logger.info(f"GradConflict first call: Found {len(all_core_losses)} core_loss keys: {all_core_losses}")

        if not self._should_compute():
            return {}

        # Extract core loss components
        loss_components = {}
        for key, value in loss_dict.items():
            if key.endswith("_core_loss") and isinstance(value, torch.Tensor) and value.requires_grad:
                # Extract the loss name: "sam_core_loss" -> "sam"
                loss_name = key.replace("_core_loss", "")
                loss_components[loss_name] = value

        if len(loss_components) < 2:
            # Log on first occurrence and every 100 steps
            if self._step_count <= 20 or self._step_count % 100 == 0:
                # Check requires_grad for each core_loss
                grad_status = {}
                for k, v in loss_dict.items():
                    if "_core_loss" in k and isinstance(v, torch.Tensor):
                        grad_status[k] = v.requires_grad
                self._logger.warning(
                    f"GradConflict: Only {len(loss_components)} loss component(s) with requires_grad=True (need >=2). Found: {list(loss_components.keys())}. requires_grad status: {grad_status}"
                )
            return {}  # Need at least 2 losses to compute conflict

        # Log that we're computing
        if self._step_count <= 20:
            self._logger.info(f"GradConflict computing: {list(loss_components.keys())}")

        return self.compute_conflict(
            losses=loss_components,
            log_to_tensorboard=writer is not None,
            writer=writer,
            step=step,
        )


def quick_gradient_check(
    model: nn.Module,
    loss_clean: torch.Tensor,
    loss_adv: torch.Tensor,
    param_pattern: str = "bndl",
) -> dict[str, float]:
    """
    Quick one-off gradient conflict check between clean and adversarial losses.

    Args:
        model: Model
        loss_clean: Clean branch loss
        loss_adv: Adversarial branch loss
        param_pattern: Pattern to match parameter names

    Returns:
        Dict with conflict metrics
    """
    results = {}

    # Get target params
    target_params = [(n, p) for n, p in model.named_parameters() if param_pattern in n.lower() and p.requires_grad]

    if not target_params:
        return {"warning": "no_matching_params"}

    # Get clean gradients
    model.zero_grad()
    try:
        loss_clean.backward(retain_graph=True)
    except RuntimeError:
        return {"error": "clean_backward_failed"}

    grads_clean = {n: p.grad.clone() for n, p in target_params if p.grad is not None}

    # Get adversarial gradients
    model.zero_grad()
    try:
        loss_adv.backward(retain_graph=True)
    except RuntimeError:
        return {"error": "adv_backward_failed"}

    grads_adv = {n: p.grad.clone() for n, p in target_params if p.grad is not None}

    # Compute cosine similarities
    cos_sims = []
    for name in grads_clean:
        if name not in grads_adv:
            continue
        g_c = grads_clean[name].flatten()
        g_a = grads_adv[name].flatten()
        cos_sim = F.cosine_similarity(g_c.unsqueeze(0), g_a.unsqueeze(0)).item()
        cos_sims.append(cos_sim)
        results[f"cos_sim/{name.split('.')[-1]}"] = cos_sim

    if cos_sims:
        results["cos_sim/mean"] = sum(cos_sims) / len(cos_sims)
        results["cos_sim/min"] = min(cos_sims)
        results["conflict_ratio"] = sum(1 for c in cos_sims if c < 0) / len(cos_sims)

    return results
