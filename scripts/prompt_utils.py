#!/usr/bin/env python3
"""
Utilities for sampling prompts for SAM-2 zero-shot evaluation.

Currently supports:
- GT box from a boolean mask (tight bounding box)
- Helpers for simple visual metadata
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2
import torch
from sam2.modeling.sam2_utils import sample_one_point_from_error_center


def compute_tight_box_from_bool_mask(mask_bool: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute a tight bounding box for the foreground of a boolean mask.

    Returns (x0, y0, x1, y1) in integer pixel coordinates, inclusive bounds.
    Raises ValueError if the mask has no foreground pixels.
    """
    if mask_bool is None:
        raise ValueError("mask_bool is None")

    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        raise ValueError("Empty mask when computing tight box")

    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return x0, y0, x1, y1


def box_center_xy(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Return integer center (cx, cy) of (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    return int(cx), int(cy)


def sample_pos_neg(
    gt_mask: np.ndarray,
    dilate_iter: int = 5,
    full_mask: np.ndarray | None = None,
    current_obj_id: int | None = None,
) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """
    Sample a positive point inside GT and a negative point near boundary (or any background).
    Returns ((pos_x, pos_y), (neg_x, neg_y)|None)
    """
    ys, xs = np.where(gt_mask)
    if len(xs) == 0:
        raise ValueError("GT mask is empty.")
    idx = np.random.randint(len(xs))
    pos_xy = (int(xs[idx]), int(ys[idx]))

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(gt_mask.astype(np.uint8), kernel, iterations=dilate_iter) > 0
    ring = np.logical_and(dilated, ~gt_mask)
    ys_n, xs_n = np.where(ring)
    if len(xs_n) == 0:
        ys_n, xs_n = np.where(~gt_mask)

    if len(xs_n) == 0:
        if full_mask is not None and current_obj_id is not None:
            other_objects_mask = (full_mask > 0) & (full_mask != current_obj_id)
            if np.any(other_objects_mask):
                ys_other, xs_other = np.where(other_objects_mask)
                idx_other = np.random.randint(len(xs_other))
                neg_xy = (int(xs_other[idx_other]), int(ys_other[idx_other]))
                return pos_xy, neg_xy
        return pos_xy, None

    idx_n = np.random.randint(len(xs_n))
    neg_xy = (int(xs_n[idx_n]), int(ys_n[idx_n]))
    return pos_xy, neg_xy


def sample_error_click(
    gt_bool: np.ndarray,
    pred_bool: np.ndarray,
) -> tuple[tuple[int, int], int]:
    """
    Sample one error-based click from prediction vs GT difference.
    Returns ((x, y), label) where label is 1 for positive, 0 for negative.
    """
    if gt_bool.dtype != bool:
        gt_bool = gt_bool.astype(bool)
    if pred_bool.dtype != bool:
        pred_bool = pred_bool.astype(bool)

    gt_t = torch.from_numpy(gt_bool[None, None]).bool()
    pred_t = torch.from_numpy(pred_bool[None, None]).bool()
    pt3, lb3 = sample_one_point_from_error_center(gt_t, pred_t, padding=True)
    pt3_xy = tuple(map(int, pt3.squeeze().tolist()))
    lb3_i = int(lb3.squeeze().item())
    return pt3_xy, lb3_i


