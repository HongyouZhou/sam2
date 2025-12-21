# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False, sample_weight=None):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        sample_weight: Optional float tensor of shape broadcastable to inputs (e.g. [N, 1, H, W])
                       Used to weight the loss per pixel.
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        
        if sample_weight is not None:
            sample_weight = sample_weight.flatten(2)
            numerator = 2 * (inputs * targets * sample_weight).sum(-1)
            denominator = (inputs * sample_weight).sum(-1) + (targets * sample_weight).sum(-1)
        else:
            numerator = 2 * (inputs * targets).sum(-1)
            denominator = inputs.sum(-1) + targets.sum(-1)
    else:
        inputs = inputs.flatten(1)
        
        if sample_weight is not None:
            sample_weight = sample_weight.flatten(1)
            numerator = 2 * (inputs * targets * sample_weight).sum(1)
            denominator = (inputs * sample_weight).sum(1) + (targets * sample_weight).sum(1)
        else:
            numerator = 2 * (inputs * targets).sum(1)
            denominator = inputs.sum(-1) + targets.sum(-1)
            
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
    sample_weight=None,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if sample_weight is not None:
        loss = loss * sample_weight

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects


def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
        uncertainty_weight=0.0,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
            uncertainty_weight: weight for uncertainty-based loss weighting (0.0 to 1.0)
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.uncertainty_weight = uncertainty_weight

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v

        # Normalize metrics by batch size
        batch_size = len(outs_batch)
        if batch_size > 0:
            if "sample_weight_mean" in losses:
                losses["sample_weight_mean"] /= batch_size
            if "uncertainty_mean" in losses:
                losses["uncertainty_mean"] /= batch_size
        
        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """

        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0, "sample_weight_mean": 0, "uncertainty_mean": 0}
        
        # Get auxiliary outputs (BNDL) if available
        # multistep_aux_outputs is a list of aux_outputs for each step
        aux_list = outputs.get("multistep_aux_outputs", [None] * len(src_masks_list))
        
        cnt_uncertainty = 0
        cnt_weight = 0
        
        for src_masks, ious, object_score_logits, aux_out in zip(
            src_masks_list, ious_list, object_score_logits_list, aux_list
        ):
            sample_weight = None
            
            # Check for BNDL uncertainty
            if aux_out is not None and "bndl" in aux_out and "pixel_uncertainty" in aux_out["bndl"]:
                pixel_uncertainty = aux_out["bndl"]["pixel_uncertainty"]    # [N, H, W]
                losses["uncertainty_mean"] += pixel_uncertainty.mean()
                cnt_uncertainty += 1
                
                # Apply weighting if enabled
                if self.uncertainty_weight > 0:
                    # Weight = 1.0 - alpha * U
                    # Detach U to prevent model from maximizing uncertainty to minimize loss
                    sample_weight = 1.0 - self.uncertainty_weight * pixel_uncertainty.detach()
                    sample_weight = torch.clamp(sample_weight, min=0.0, max=1.0)
                    
                    # Log mean weight
                    losses["sample_weight_mean"] += sample_weight.mean()
                    cnt_weight += 1
                    
                    # Expand for multimask: [N, H, W] -> [N, 1, H, W] for broadcasting to [N, M, H, W]
                    if sample_weight.dim() == 3:
                        sample_weight = sample_weight.unsqueeze(1)
            
            # If no weight computed (warmup or no uncertainty), effectively weight is 1.0
            if sample_weight is None:
                 # We don't add 1.0 to sample_weight_mean here to avoid shifting the average of actual weights
                 # Instead we handle the default case below
                 pass
            
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits, sample_weight
            )
            
        # Normalize metrics
        if cnt_uncertainty > 0:
            losses["uncertainty_mean"] /= cnt_uncertainty
            
        if cnt_weight > 0:
            losses["sample_weight_mean"] /= cnt_weight
        else:
            # If no weighting was applied (e.g. warmup), the effective weight is 1.0
            losses["sample_weight_mean"] = torch.tensor(1.0, device=targets.device)
            
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        losses["sam_core_loss"] = losses[CORE_LOSS_KEY]
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits, sample_weight=None
    ):
        # Resize sample_weight if needed (e.g. if uncertainty map is lower res than masks)
        if sample_weight is not None and sample_weight.shape[-2:] != src_masks.shape[-2:]:
            sample_weight = F.interpolate(
                sample_weight, 
                size=src_masks.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )

        target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
            sample_weight=sample_weight,
        )
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True, sample_weight=sample_weight
        )
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2
        if loss_multimask.size(1) > 1:
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                # ignore missing keys (e.g. sample_weight_mean)
                if loss_key in ["loss_mask", "loss_dice", "loss_iou", "loss_class"]:
                     raise ValueError(f"{type(self)} doesn't compute {loss_key}")
                continue
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss


 
