#!/usr/bin/env python
"""Unified click prompt generation using SAM2 sampling utilities."""

import numpy as np
import torch
import torch.nn.functional as F
from prompt_utils import sample_pos_neg, sample_error_click


def _euclidean_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points."""
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return float((dx * dx + dy * dy) ** 0.5)


def generate_click_prompts(
    predictor,
    state: dict,
    frame_idx: int,
    obj_id: int,
    gt_bool: np.ndarray,
    first_frame_mask_np: np.ndarray,
    score_thresh: float,
    click_protocol: str,
    min_click_dist: float,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Generate query points per click_protocol using SAM2 sampling utils and send to predictor.

    Args:
        predictor: SAM2 video predictor instance
        state: predictor inference state
        frame_idx: frame index (typically 0 for first frame)
        obj_id: object ID
        gt_bool: boolean mask of ground truth for this object [H, W]
        first_frame_mask_np: full first frame mask with all objects [H, W]
        score_thresh: threshold for mask logits
        click_protocol: one of "1click", "3click", "5click"
        min_click_dist: minimum distance between clicks (for 5click)

    Returns:
        (used_pts, used_labels): lists of (x, y) coordinates and labels (1=pos, 0=neg)
    """
    used_pts: list[tuple[int, int]] = []
    used_labels: list[int] = []
    total_clicks = 1 if click_protocol == "1click" else (5 if click_protocol == "5click" else 3)

    # First positive (and maybe negative) click using SAM2's sample_pos_neg
    try:
        pos_xy, neg_xy = sample_pos_neg(gt_bool, full_mask=first_frame_mask_np, current_obj_id=obj_id)
        cx, cy = int(pos_xy[0]), int(pos_xy[1])
    except Exception:
        ys, xs = np.where(gt_bool)
        cx = int(xs.mean()) if xs.size else 0
        cy = int(ys.mean()) if ys.size else 0
        neg_xy = None

    if click_protocol == "3click":
        # 3-click: positive, negative (if available), error-based
        if neg_xy is not None:
            pts = np.array([[cx, cy], [int(neg_xy[0]), int(neg_xy[1])]], dtype=np.float32)
            lbl = np.array([1, 0], dtype=np.int32)
        else:
            pts = np.array([[cx, cy]], dtype=np.float32)
            lbl = np.array([1], dtype=np.int32)
        _, obj_ids_after, masks = predictor.add_new_points_or_box(
            state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=pts,
            labels=lbl,
        )
        used_pts.extend([(cx, cy)])
        used_labels.extend([1])
        if neg_xy is not None:
            used_pts.append((int(neg_xy[0]), int(neg_xy[1])))
            used_labels.append(0)

        # Error-based third click
        cur_idx = obj_ids_after.index(obj_id)
        mask_logits_2d = masks[cur_idx, 0] if masks.dim() == 4 else (masks[0] if masks.dim() == 3 else masks.squeeze())
        if tuple(mask_logits_2d.shape[-2:]) != gt_bool.shape:
            h, w = gt_bool.shape
            mask_logits_2d = F.interpolate(
                mask_logits_2d.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False
            )[0, 0]
        pred_bool = (mask_logits_2d > score_thresh).detach().cpu().numpy().astype(bool)
        nx, ny_label = sample_error_click(gt_bool, pred_bool)
        nx_pt = (int(nx[0]), int(nx[1]))
        used_pts.append(nx_pt)
        used_labels.append(int(ny_label))
        predictor.add_new_points_or_box(
            state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=np.array([nx_pt], dtype=np.float32),
            labels=np.array([int(ny_label)], dtype=np.int32),
            clear_old_points=False,
        )
        return used_pts, used_labels

    # 1click / 5click: start with positive, then iterative error-based
    _, obj_ids_after, masks = predictor.add_new_points_or_box(
        state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=np.array([[cx, cy]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )
    used_pts.append((cx, cy))
    used_labels.append(1)

    while len(used_pts) < total_clicks:
        cur_idx = obj_ids_after.index(obj_id)
        mask_logits_2d = masks[cur_idx, 0] if masks.dim() == 4 else (masks[0] if masks.dim() == 3 else masks.squeeze())
        if tuple(mask_logits_2d.shape[-2:]) != gt_bool.shape:
            h, w = gt_bool.shape
            mask_logits_2d = F.interpolate(
                mask_logits_2d.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False
            )[0, 0]
        pred_bool = (mask_logits_2d > score_thresh).detach().cpu().numpy().astype(bool)
        nx, ny_label = sample_error_click(gt_bool, pred_bool)
        nx_pt = (int(nx[0]), int(nx[1]))
        
        # Enforce minimum distance constraint
        if min_click_dist > 0 and any(_euclidean_distance(nx_pt, p) < float(min_click_dist) for p in used_pts):
            ys, xs = np.where(gt_bool if ny_label == 1 else (~gt_bool & pred_bool))
            nx_pt = (int(xs.mean()) if xs.size else 0, int(ys.mean()) if ys.size else 0)
        
        used_pts.append(nx_pt)
        used_labels.append(int(ny_label))
        _, obj_ids_after, masks = predictor.add_new_points_or_box(
            state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=np.array([nx_pt], dtype=np.float32),
            labels=np.array([int(ny_label)], dtype=np.int32),
            clear_old_points=False,
        )

    return used_pts, used_labels

