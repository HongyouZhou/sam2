#!/usr/bin/env python
# Multi-dataset Zero-shot evaluation of SAM-2
# Supports TrashCan, GTEA, PIDRay, plittersdorf, Hypersim, DRAM, and CITYSCAPES datasets

import argparse
import shutil
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import random
import cv2
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Remove unused imports
import time

# ----------  SAM-2 -----------
from sam2.build_sam import build_sam2_video_predictor

# ----------  Tools -----------
from tools.vos_inference import (
    DAVIS_PALETTE,
    save_masks_to_dir,
)

# ----------  Click sampling ----------
from sam2.modeling.sam2_utils import (
    sample_one_point_from_error_center,
)

# ----------  Metric ----------
from sav_dataset.utils.sav_benchmark import benchmark


# ---------- Dataset Configurations ----------
from dataset_configs import DATASET_CONFIGS, DEFAULT_DATASETS
from prompt_utils import compute_tight_box_from_bool_mask, box_center_xy, sample_pos_neg, sample_error_click

# Distinct colors for different objects
OBJECT_COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
]

# Point colors for positive/negative/error clicks
POINT_COLORS = [
    (255, 0, 0),  # Red for positive
    (0, 0, 255),  # Blue for negative
    (255, 255, 0),  # Yellow for error-based
]


def draw_points_on_image(img: Image.Image, all_obj_pts: dict[int, list], point_colors=POINT_COLORS):
    """Draw query points from all objects on image with different colors per object"""
    print(f"Drawing points on image. Image size: {img.size}")
    print(f"Points data: {all_obj_pts}")

    img_with_points = img.copy()
    draw = ImageDraw.Draw(img_with_points)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    if not all_obj_pts:
        print("Warning: No query points found!")
        return img_with_points

    for obj_id, pts in all_obj_pts.items():
        print(f"Processing object {obj_id} with points: {pts}")
        obj_color = OBJECT_COLORS[obj_id % len(OBJECT_COLORS)]

        for i, pt in enumerate(pts):
            if pt is None or len(pt) != 2:
                print(f"Skipping invalid point: {pt}")
                continue

            x, y = int(pt[0]), int(pt[1])
            print(f"Drawing point {i} for obj {obj_id} at ({x}, {y})")
            r = 10  # Larger radius for visibility

            # Use point type color but with object-specific variations
            if i == 0:  # Positive point
                color = (255, 0, 0)  # Red
                label = f"O{obj_id}+"
            elif i == 1:  # Negative point
                color = (0, 0, 255)  # Blue
                label = f"O{obj_id}-"
            else:  # Error point
                color = (255, 255, 0)  # Yellow
                label = f"O{obj_id}E"

            # Draw larger point with thick outline for visibility
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=color,
                outline=(255, 255, 255),  # White outline for visibility
                width=4,
            )

            # Add a second outline with object color
            draw.ellipse(
                (x - r - 2, y - r - 2, x + r + 2, y + r + 2),
                fill=None,
                outline=obj_color,
                width=2,
            )

            # Add label with background
            text_x, text_y = x + r + 5, y - r
            # Draw text background
            text_bbox = draw.textbbox((text_x, text_y), label, font=font)
            draw.rectangle(text_bbox, fill=(0, 0, 0), outline=(255, 255, 255))
            draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

    print(f"Finished drawing {sum(len(pts) for pts in all_obj_pts.values())} points")
    return img_with_points


def create_multi_object_mask_visualization(all_masks: dict[int, np.ndarray], alpha: float = 0.6):
    """Create colored mask visualization with all objects using different colors"""
    if not all_masks:
        print("Warning: No masks provided for visualization")
        return None

    # Get image dimensions from any mask
    h, w = next(iter(all_masks.values())).shape
    print(f"Creating mask visualization for {len(all_masks)} objects, size: {h}x{w}")

    mask_vis = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA

    for obj_id, mask_bool in all_masks.items():
        if not np.any(mask_bool):
            print(f"Object {obj_id} has empty mask, skipping")
            continue

        color = OBJECT_COLORS[obj_id % len(OBJECT_COLORS)]
        print(f"Object {obj_id}: {np.sum(mask_bool)} pixels, color: {color}")
        mask_vis[mask_bool] = [*color, int(255 * alpha)]

    return Image.fromarray(mask_vis, mode="RGBA")


def create_error_analysis_multi_object(gt_masks: dict[int, np.ndarray], pred_masks: dict[int, np.ndarray]):
    """Create error analysis for multiple objects"""
    if not gt_masks and not pred_masks:
        print("Warning: No GT or prediction masks for error analysis")
        return None

    # Get all object IDs
    all_obj_ids = set(gt_masks.keys()) | set(pred_masks.keys())
    if not all_obj_ids:
        return None

    print(f"Creating error analysis for objects: {all_obj_ids}")

    # Get image dimensions
    h, w = next(iter((gt_masks or pred_masks).values())).shape
    error_vis = np.zeros((h, w, 3), dtype=np.uint8)

    for obj_id in all_obj_ids:
        gt_mask = gt_masks.get(obj_id, np.zeros((h, w), dtype=bool))
        pred_mask = pred_masks.get(obj_id, np.zeros((h, w), dtype=bool))

        obj_color = OBJECT_COLORS[obj_id % len(OBJECT_COLORS)]

        # True positives (both GT and pred) - use object color
        true_positive = gt_mask & pred_mask
        error_vis[true_positive] = obj_color

        # False positives (pred but not GT) - use darker version of object color
        false_positive = pred_mask & (~gt_mask)
        dark_color = [int(c * 0.5) for c in obj_color]  # Darker version
        error_vis[false_positive] = dark_color

        # False negatives (GT but not pred) - use lighter version of object color
        false_negative = gt_mask & (~pred_mask)
        light_color = [min(255, int(c * 1.5)) for c in obj_color]  # Lighter version
        error_vis[false_negative] = light_color

        print(f"Object {obj_id}: TP={np.sum(true_positive)}, FP={np.sum(false_positive)}, FN={np.sum(false_negative)}")

    return Image.fromarray(error_vis)


def create_comprehensive_multi_object_visualization(
    img_path: Path,
    pred_masks: dict[int, np.ndarray],
    gt_masks: dict[int, np.ndarray],
    all_obj_pts: dict[int, list],
    save_path: Path,
):
    """Create a comprehensive 2x3 grid visualization for all objects in one image"""

    print(f"\n=== Creating visualization for {img_path.name} ===")
    print(f"GT masks: {list(gt_masks.keys()) if gt_masks else 'None'}")
    print(f"Pred masks: {list(pred_masks.keys()) if pred_masks else 'None'}")
    print(f"Query points: {list(all_obj_pts.keys()) if all_obj_pts else 'None'}")

    # Load original image
    try:
        img = Image.open(img_path).convert("RGB")
        print(f"Loaded image: {img.size}")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return

    # Create individual visualizations
    img_with_points = draw_points_on_image(img, all_obj_pts)
    gt_mask_vis = create_multi_object_mask_visualization(gt_masks, alpha=0.8)
    pred_mask_vis = create_multi_object_mask_visualization(pred_masks, alpha=0.8)

    # Create overlay with all objects
    img_rgba = img.convert("RGBA")
    if pred_masks:
        pred_overlay = create_multi_object_mask_visualization(pred_masks, alpha=0.4)
        if pred_overlay:
            overlay_img = Image.alpha_composite(img_rgba, pred_overlay).convert("RGB")

            # Add all points to overlay
            draw = ImageDraw.Draw(overlay_img)
            for pts in all_obj_pts.values():
                for i, pt in enumerate(pts):
                    if pt is None or len(pt) != 2:
                        continue
                    x, y = int(pt[0]), int(pt[1])
                    r = 10

                    if i == 0:  # Positive
                        color = (255, 0, 0)
                    elif i == 1:  # Negative
                        color = (0, 0, 255)
                    else:  # Error
                        color = (255, 255, 0)

                    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(255, 255, 255), width=4)
        else:
            overlay_img = img
    else:
        overlay_img = img

    # Create error analysis
    error_img = create_error_analysis_multi_object(gt_masks, pred_masks)
    if error_img is None:
        error_img = Image.new("RGB", img.size, color=(128, 128, 128))

    # Handle None cases - create blank images with same size as original
    if gt_mask_vis is None:
        gt_mask_vis = Image.new("RGB", img.size, color=(64, 64, 64))
    else:
        gt_mask_vis = gt_mask_vis.convert("RGB")

    if pred_mask_vis is None:
        pred_mask_vis = Image.new("RGB", img.size, color=(64, 64, 64))
    else:
        pred_mask_vis = pred_mask_vis.convert("RGB")

    # Resize all images to same size
    target_size = (400, 300)
    images = [
        img.resize(target_size, Image.Resampling.LANCZOS),
        img_with_points.resize(target_size, Image.Resampling.LANCZOS),
        gt_mask_vis.resize(target_size, Image.Resampling.LANCZOS),
        pred_mask_vis.resize(target_size, Image.Resampling.LANCZOS),
        overlay_img.resize(target_size, Image.Resampling.LANCZOS),
        error_img.resize(target_size, Image.Resampling.LANCZOS),
    ]

    # Object count info
    num_gt_objs = len([oid for oid, mask in gt_masks.items() if np.any(mask)]) if gt_masks else 0
    num_pred_objs = len([oid for oid, mask in pred_masks.items() if np.any(mask)]) if pred_masks else 0
    num_query_objs = len(all_obj_pts) if all_obj_pts else 0

    titles = ["Original Image", f"Query Points ({num_query_objs} objs)", f"Ground Truth ({num_gt_objs} objects)", f"Prediction ({num_pred_objs} objects)", "Overlay Result", "Error Analysis"]

    # Create grid image
    grid_width = target_size[0] * 3
    grid_height = target_size[1] * 2 + 80  # Extra space for titles and legend

    grid_img = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        legend_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        title_font = ImageFont.load_default()
        legend_font = ImageFont.load_default()

    draw = ImageDraw.Draw(grid_img)

    # Add images and titles to grid
    for i, (img_item, title) in enumerate(zip(images, titles, strict=True)):
        row = i // 3
        col = i % 3

        x_offset = col * target_size[0]
        y_offset = row * (target_size[1] + 40) + 40

        # Paste image
        grid_img.paste(img_item, (x_offset, y_offset))

        # Add title
        text_width = draw.textlength(title, font=title_font)
        text_x = x_offset + (target_size[0] - text_width) // 2
        text_y = y_offset - 30
        draw.text((text_x, text_y), title, fill=(0, 0, 0), font=title_font)

    # Add legend at the bottom
    legend_y = grid_height - 35
    legend_text = "Legend: O{id}+ = Positive point, O{id}- = Negative point, O{id}E = Error-based point"
    draw.text((10, legend_y), legend_text, fill=(64, 64, 64), font=legend_font)

    # Create output directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the grid image
    try:
        grid_img.save(save_path, quality=95)
        print(f"Saved visualization to: {save_path}")
    except Exception as e:
        print(f"Error saving visualization to {save_path}: {e}")


def generate_enhanced_visualizations(
    jpeg_dir: Path,
    pred_dir: Path,
    ann_dir: Path,
    vis_root: Path,
    video_names: list[str] | None = None,
    max_frames_per_video: int = 5,
):
    """Generate enhanced visualizations with all objects in single images"""
    vis_root.mkdir(parents=True, exist_ok=True)

    if video_names is None:
        video_names = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    else:
        video_names = sorted(set(video_names))

    print(f"Generating enhanced multi-object visualizations for {len(video_names)} videos...")

    for vid in tqdm(video_names, desc="Processing videos"):
        jpg_dir = jpeg_dir / vid
        pred_dir_video = pred_dir / vid
        ann_dir_video = ann_dir / vid
        vis_dir_video = vis_root / vid
        vis_dir_video.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing video: {vid}")

        # Read stored points for all objects
        pts_json = pred_dir_video / "query_points.json"
        if pts_json.exists():
            with open(pts_json) as f:
                query_pts_dict = {int(k): v for k, v in json.load(f).items()}
            print(f"Loaded query points for objects: {list(query_pts_dict.keys())}")
        else:
            query_pts_dict = {}
            print("No query points file found!")

        # Get all image files and limit the number
        img_files = sorted([p for p in jpg_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]], key=lambda x: int(x.stem))

        print(f"Found {len(img_files)} image files")

        # Sample frames evenly if there are too many
        if len(img_files) > max_frames_per_video:
            step = len(img_files) // max_frames_per_video
            img_files = img_files[::step][:max_frames_per_video]
            print(f"Sampled {len(img_files)} frames")

        for img_path in img_files:
            print(f"Processing frame: {img_path.name}")

            # Check if corresponding masks exist
            mask_path = pred_dir_video / f"{img_path.stem}.png"
            gt_path = ann_dir_video / f"{img_path.stem}.png"

            if not mask_path.exists():
                print(f"  Prediction mask not found: {mask_path}")
                continue
            if not gt_path.exists():
                print(f"  GT mask not found: {gt_path}")
                continue

            try:
                # Load all object masks for this frame
                pred_mask_all = np.array(Image.open(mask_path))
                gt_mask_all = np.array(Image.open(gt_path))

                print(f"  Loaded masks. GT unique values: {np.unique(gt_mask_all)}, Pred unique values: {np.unique(pred_mask_all)}")

                # Extract individual object masks
                all_obj_ids = set(query_pts_dict.keys()) | set(np.unique(gt_mask_all)) | set(np.unique(pred_mask_all))
                all_obj_ids.discard(0)  # Remove background

                print(f"  All object IDs: {all_obj_ids}")

                pred_masks = {}
                gt_masks = {}

                for obj_id in all_obj_ids:
                    pred_masks[obj_id] = pred_mask_all == obj_id
                    gt_masks[obj_id] = gt_mask_all == obj_id
                    print(f"    Object {obj_id}: GT pixels={np.sum(gt_masks[obj_id])}, Pred pixels={np.sum(pred_masks[obj_id])}")

                # Create comprehensive visualization for all objects
                save_path = vis_dir_video / f"{img_path.stem}_all_objects.jpg"
                create_comprehensive_multi_object_visualization(img_path, pred_masks, gt_masks, query_pts_dict, save_path)

            except Exception as e:
                print(f"  Error processing frame {img_path.name}: {e}")
                continue


def overlay_mask_and_points(
    img_path: Path,
    mask_bool: np.ndarray,
    pts: list,
    save_path: Path,
    mask_color=(0, 255, 0, 128),
    point_colors=((255, 0, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255)),
):
    """Overlay mask and query points on image (legacy function for compatibility)"""
    img = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, mask_color)
    alpha = mask_bool.astype(np.uint8) * mask_color[3]
    overlay.putalpha(Image.fromarray(alpha, mode="L"))
    blended = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(blended)
    for pt, clr in zip(pts, point_colors, strict=False):
        if pt is None:
            continue
        x, y = pt
        r = 6
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=clr,
            outline=(255, 255, 255, 255),
            width=2,
        )

    blended.convert("RGB").save(save_path)


def generate_visualizations(
    jpeg_dir: Path,
    pred_dir: Path,
    ann_dir: Path,
    vis_root: Path,
    video_names: list[str] | None = None,
):
    """Generate standard visualizations (kept for compatibility, creates separate object images)"""
    vis_root.mkdir(parents=True, exist_ok=True)

    if video_names is None:
        video_names = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    else:
        video_names = sorted(set(video_names))

    for vid in video_names:
        jpg_dir = jpeg_dir / vid
        pred_dir_video = pred_dir / vid
        vis_dir_video = vis_root / vid
        vis_dir_video.mkdir(parents=True, exist_ok=True)

        # Read stored points if available
        pts_json = pred_dir_video / "query_points.json"
        if pts_json.exists():
            with open(pts_json) as f:
                query_pts_dict = {int(k): v for k, v in json.load(f).items()}
        else:
            query_pts_dict = {}

        for obj_id, pts in query_pts_dict.items():
            obj_vis_dir = vis_dir_video / f"obj_{obj_id}"
            obj_vis_dir.mkdir(parents=True, exist_ok=True)

            for img_path in sorted(jpg_dir.iterdir()):
                if img_path.suffix.lower() not in [".jpg", ".jpeg"]:
                    continue
                mask_path = pred_dir_video / f"{img_path.stem}.png"
                if not mask_path.exists():
                    continue
                mask_np = np.array(Image.open(mask_path)) == obj_id
                save_path = obj_vis_dir / f"{img_path.stem}.jpg"
                overlay_mask_and_points(img_path, mask_np, pts, save_path)


def _sample_pos_neg(gt_mask: np.ndarray, dilate_iter: int = 5, full_mask: np.ndarray | None = None, current_obj_id: int | None = None):
    return sample_pos_neg(gt_mask, dilate_iter=dilate_iter, full_mask=full_mask, current_obj_id=current_obj_id)


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def inference_3_clicks(
    predictor,
    jpeg_dir: Path,
    ann_dir: Path,
    out_dir: Path,
    score_thresh: float = 0.0,
    video_names: list[str] | None = None,
    max_objects: int | None = None,
    prompt_method: str = "gt_box",
):
    """
    3-click interactive inference:
    1) Random positive point inside GT
    2) Random negative point near GT boundary
    3) Error-based point from prediction vs GT difference
    """
    if video_names is None:
        video_names = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    else:
        video_names = sorted(set(video_names))

    print(f"3-click inference on {len(video_names)} videos")
    for v_idx, vid in enumerate(video_names, 1):
        print(f"[{v_idx:03}/{len(video_names)}] {vid}")
        video_dir = jpeg_dir / vid
        frame_names = sorted([p.stem for p in video_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]], key=lambda x: int(x))

        # Initialize predictor state
        state = predictor.init_state(str(video_dir))
        H, W = state["video_height"], state["video_width"]

        # Read first frame GT to determine object IDs
        first_mask_path = ann_dir / vid / f"{frame_names[0]}.png"
        if not first_mask_path.exists():
            print(f"Warning: First frame annotation not found: {first_mask_path}")
            continue

        first_mask = np.array(Image.open(first_mask_path))
        all_obj_ids = [oid for oid in np.unique(first_mask) if oid > 0]

        if len(all_obj_ids) == 0:
            print(f"Warning: No objects found in first frame of video {vid}")
            continue

        # Apply object limit if specified
        if max_objects and len(all_obj_ids) > max_objects:
            # Select objects with largest areas for more meaningful evaluation
            obj_areas = {}
            for oid in all_obj_ids:
                obj_areas[oid] = np.sum(first_mask == oid)

            # Sort by area and take top N objects
            sorted_objs = sorted(obj_areas.items(), key=lambda x: x[1], reverse=True)
            obj_ids = [oid for oid, _ in sorted_objs[:max_objects]]
            print(f"Limited to {max_objects} largest objects in video {vid} (from {len(all_obj_ids)} total)")
        else:
            obj_ids = all_obj_ids

        print(f"Processing {len(obj_ids)} objects in video {vid}: {obj_ids}")

        obj_points: dict[int, list] = {}

        for obj_id in obj_ids:
            gt_bool = first_mask == obj_id

            # Check if mask is empty
            if not np.any(gt_bool):
                print(f"Warning: Empty GT mask for object {obj_id} in video {vid}")
                continue

            # Clear GPU memory before processing each object to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if prompt_method == "gt_box":
                # Use tight GT bounding box as prompt
                try:
                    x0, y0, x1, y1 = compute_tight_box_from_bool_mask(gt_bool)
                except Exception as e:
                    print(f"Error computing box for object {obj_id} in video {vid}: {e}")
                    continue

                _, _, _ = predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=np.array([x0, y0, x1, y1], dtype=np.float32),
                )

                # For compatibility with existing visualization, store box center as a single point
                cx, cy = box_center_xy((x0, y0, x1, y1))
                obj_points[obj_id] = [
                    (int(cx), int(cy)),
                ]

            else:
                # three_clicks (legacy)
                try:
                    pos_xy, neg_xy = _sample_pos_neg(gt_bool, full_mask=first_mask, current_obj_id=obj_id)
                except Exception as e:
                    print(f"Error sampling points for object {obj_id} in video {vid}: {e}")
                    continue

                if neg_xy is None:
                    print(f"Warning: Object {obj_id} in video {vid} covers entire image and no other objects available for negative sampling, using only positive point")
                    pts = np.array([pos_xy], dtype=np.float32)
                    lbl = np.array([1], dtype=np.int32)
                else:
                    pts = np.array([[pos_xy, neg_xy]], dtype=np.float32)
                    lbl = np.array([[1, 0]], dtype=np.int32)
                    pts = pts.reshape(-1, 2)
                    lbl = lbl.reshape(-1)
                _, obj_ids_after, masks = predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=pts,
                    labels=lbl,
                )

                cur_idx = obj_ids_after.index(obj_id)
                pred_bool = (masks[cur_idx : cur_idx + 1] > score_thresh).cpu().numpy().astype(bool)[0, 0]
                pt3_xy, lb3_i = sample_error_click(gt_bool, pred_bool)

                predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=np.array(pt3_xy, dtype=np.float32),
                    labels=np.array([lb3_i], dtype=np.int32),
                    clear_old_points=False,
                )

                obj_points[obj_id] = [
                    (int(pos_xy[0]), int(pos_xy[1])),
                    (int(neg_xy[0]), int(neg_xy[1])) if neg_xy is not None else None,
                    (int(pt3_xy[0]), int(pt3_xy[1])),
                ]

        print(f"Generated query points for video {vid}: {obj_points}")

        # Propagate through entire video
        video_segments = {}
        for f_idx, out_obj_ids, out_logits in predictor.propagate_in_video(state):
            seg = {oid: (out_logits[i] > score_thresh).cpu().numpy() for i, oid in enumerate(out_obj_ids)}
            video_segments[f_idx] = seg

        # Save PNG masks
        for f_idx, seg in video_segments.items():
            save_masks_to_dir(
                output_mask_dir=str(out_dir),
                video_name=vid,
                frame_name=frame_names[f_idx],
                per_obj_output_mask=seg,
                height=H,
                width=W,
                per_obj_png_file=False,
                output_palette=DAVIS_PALETTE,
            )

        # Save query points
        (out_dir / vid).mkdir(parents=True, exist_ok=True)
        with open(out_dir / vid / "query_points.json", "w") as f:
            json.dump({int(k): v for k, v in obj_points.items()}, f, indent=2)

        torch.cuda.empty_cache()


def run_single_dataset(
    dataset_name: str,
    predictor,
    output_path: Path,
    split: str | None = None,
    score_thresh: float = 0.0,
    num_workers: int | None = None,
    video_subset: list[str] | None = None,
    save_vis: bool = True,
    enhanced_vis: bool = True,
    max_objects: int | None = None,
    prompt_method: str = "gt_box",
    first_frame_only: bool = False,
) -> tuple[float, float, float]:
    """Run evaluation on a single dataset and return metrics"""

    config = DATASET_CONFIGS[dataset_name]
    if split is None:
        split = config["default_split"]
    # For type-checkers: split is now a string
    assert isinstance(split, str)

    root = Path(config["root"])
    if config["has_split_subdir"]:
        jpeg_dir = root / split / "JPEGImages"
        ann_dir = root / split / "Annotations"
    else:
        jpeg_dir = root / "JPEGImages"
        ann_dir = root / "Annotations"

    if not jpeg_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"JPEGImages or Annotations not found for {dataset_name}: {jpeg_dir}, {ann_dir}")

    # Output directories
    out_dir = output_path / f"{dataset_name.lower()}_pred"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Running {dataset_name} dataset evaluation")
    print(f"{'=' * 60}")

    # Debug: Check data paths and availability
    print(f"JPEG directory: {jpeg_dir}")
    print(f"Annotation directory: {ann_dir}")
    print(f"JPEG dir exists: {jpeg_dir.exists()}")
    print(f"Ann dir exists: {ann_dir.exists()}")

    if jpeg_dir.exists():
        video_dirs = [d for d in jpeg_dir.iterdir() if d.is_dir()]
        print(f"Found {len(video_dirs)} video directories in JPEG dir")
        if video_dirs:
            print(f"First few videos: {[v.name for v in video_dirs[:3]]}")

    if ann_dir.exists():
        ann_dirs = [d for d in ann_dir.iterdir() if d.is_dir()]
        print(f"Found {len(ann_dirs)} annotation directories")
        if ann_dirs:
            print(f"First few annotation videos: {[v.name for v in ann_dirs[:3]]}")

    # Run inference
    start_time = time.time()
    try:
        inference_3_clicks(
            predictor,
            jpeg_dir,
            ann_dir,
            out_dir,
            score_thresh=score_thresh,
            video_names=video_subset,
            max_objects=max_objects,
            prompt_method=prompt_method,
        )
    except Exception as e:
        print(f"Error during inference for {dataset_name}: {e}")
        raise
    inference_time = time.time() - start_time

    # Run evaluation
    eval_start_time = time.time()

    # Prepare evaluation roots (support first-frame-only copy)
    if first_frame_only:
        base_videos = video_subset if video_subset is not None else [d.name for d in ann_dir.iterdir() if d.is_dir()]
        base_videos = sorted(base_videos)
        gt_tmp = output_path / f"{dataset_name.lower()}_tmp_gt_first"
        pred_tmp = output_path / f"{dataset_name.lower()}_tmp_pred_first"
        if gt_tmp.exists():
            shutil.rmtree(gt_tmp)
        if pred_tmp.exists():
            shutil.rmtree(pred_tmp)
        for v in base_videos:
            v_gt_dir = ann_dir / v
            v_pred_dir = out_dir / v
            if not v_gt_dir.exists() or not v_pred_dir.exists():
                continue
            # find first frame PNG in GT dir
            gt_pngs = sorted([p for p in v_gt_dir.iterdir() if p.suffix.lower() == ".png"])
            if not gt_pngs:
                continue
            first_png = gt_pngs[0].name
            # ensure corresponding pred exists
            if not (v_pred_dir / first_png).exists():
                continue
            (gt_tmp / v).mkdir(parents=True, exist_ok=True)
            (pred_tmp / v).mkdir(parents=True, exist_ok=True)
            shutil.copy2(v_gt_dir / first_png, gt_tmp / v / first_png)
            shutil.copy2(v_pred_dir / first_png, pred_tmp / v / first_png)
        gt_root_eval, pred_root_eval = gt_tmp, pred_tmp
    else:
        if video_subset is not None:
            gt_tmp = output_path / f"{dataset_name.lower()}_tmp_gt"
            pred_tmp = output_path / f"{dataset_name.lower()}_tmp_pred"
            if gt_tmp.exists():
                shutil.rmtree(gt_tmp)
            if pred_tmp.exists():
                shutil.rmtree(pred_tmp)
            gt_tmp.mkdir(parents=True, exist_ok=True)
            pred_tmp.mkdir(parents=True, exist_ok=True)
            for v in video_subset:
                if (ann_dir / v).exists() and (out_dir / v).exists():
                    shutil.copytree(ann_dir / v, gt_tmp / v, symlinks=True)
                    shutil.copytree(out_dir / v, pred_tmp / v, symlinks=True)
            gt_root_eval, pred_root_eval = gt_tmp, pred_tmp
        else:
            gt_root_eval, pred_root_eval = ann_dir, out_dir

    try:
        J_F, global_J, global_F, _ = benchmark(
            gt_roots=[str(gt_root_eval)],
            mask_roots=[str(pred_root_eval)],
            strict=False,
            num_processes=num_workers,
            skip_first_and_last=config["skip_first_and_last"],
            verbose=True,
        )

        # Check for NaN values and handle them
        if len(J_F) == 0 or len(global_J) == 0 or len(global_F) == 0:
            print(f"Warning: Empty evaluation results for {dataset_name}")
            return 0.0, 0.0, 0.0

        j_f_val = J_F[0] if not np.isnan(J_F[0]) else 0.0
        j_val = global_J[0] if not np.isnan(global_J[0]) else 0.0
        f_val = global_F[0] if not np.isnan(global_F[0]) else 0.0

        if np.isnan(j_f_val) or np.isnan(j_val) or np.isnan(f_val):
            print(f"Warning: NaN values detected in {dataset_name} evaluation results")
            print(f"  J&F: {J_F[0]}, J: {global_J[0]}, F: {global_F[0]}")
            # Return zeros instead of NaN
            j_f_val = 0.0 if np.isnan(j_f_val) else j_f_val
            j_val = 0.0 if np.isnan(j_val) else j_val
            f_val = 0.0 if np.isnan(f_val) else f_val

        # Removed PIDRay special-case macro averaging to keep a single-pass evaluation

    except Exception as e:
        print(f"Error during evaluation of {dataset_name}: {e}")
        return 0.0, 0.0, 0.0

    eval_time = time.time() - eval_start_time

    # Generate visualizations
    if save_vis:
        print("Saving visualizations...")
        if enhanced_vis:
            vis_dir = output_path / f"{dataset_name.lower()}_pred_vis_enhanced"
            generate_enhanced_visualizations(jpeg_dir, out_dir, ann_dir, vis_dir, video_subset)
            print(f"Enhanced multi-object visualizations saved to: {vis_dir}")
        else:
            vis_dir = output_path / f"{dataset_name.lower()}_pred_vis"
            generate_visualizations(jpeg_dir, out_dir, ann_dir, vis_dir, video_subset)
            print(f"Standard visualizations saved to: {vis_dir}")

    print(f"Inference time: {inference_time:.2f}s")
    print(f"Evaluation time: {eval_time:.2f}s")

    return j_f_val, j_val, f_val


def create_comparison_plots(results: dict[str, tuple[float, float, float]], output_path: Path):
    """Create comparison plots for all datasets"""

    # Prepare data for plotting
    datasets = list(results.keys())
    j_f_scores = [results[d][0] for d in datasets]
    j_scores = [results[d][1] for d in datasets]
    f_scores = [results[d][2] for d in datasets]

    # Create DataFrame for easier plotting
    df_data = []
    for dataset in datasets:
        j_f, j, f = results[dataset]
        df_data.extend(
            [
                {"Dataset": dataset, "Metric": "J&F", "Score": j_f},
                {"Dataset": dataset, "Metric": "J (IoU)", "Score": j},
                {"Dataset": dataset, "Metric": "F (Boundary)", "Score": f},
            ]
        )

    df = pd.DataFrame(df_data)

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("SAM-2 Zero-shot 3-Click Evaluation Results", fontsize=16, fontweight="bold")

    # 1. Bar plot comparing J&F scores
    ax1 = axes[0, 0]
    bars = ax1.bar(datasets, j_f_scores, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"])
    ax1.set_title("J&F Scores Comparison", fontweight="bold")
    ax1.set_ylabel("J&F Score")
    ax1.set_ylim(0, 100)

    # Add value labels on bars
    for bar, score in zip(bars, j_f_scores, strict=True):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{score:.1f}", ha="center", va="bottom", fontweight="bold")

    # 2. Grouped bar plot for J and F scores
    ax2 = axes[0, 1]
    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax2.bar(x - width / 2, j_scores, width, label="J (IoU)", color="#FF6B6B", alpha=0.8)
    bars2 = ax2.bar(x + width / 2, f_scores, width, label="F (Boundary)", color="#4ECDC4", alpha=0.8)

    ax2.set_title("J and F Scores Comparison", fontweight="bold")
    ax2.set_ylabel("Score")
    ax2.set_xlabel("Dataset")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    ax2.set_ylim(0, 100)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{height:.1f}", ha="center", va="bottom", fontsize=9)

    # 3. Heatmap of all metrics
    ax3 = axes[1, 0]
    heatmap_data = np.array([j_f_scores, j_scores, f_scores])
    im = ax3.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax3.set_xticks(range(len(datasets)))
    ax3.set_xticklabels(datasets)
    ax3.set_yticks(range(3))
    ax3.set_yticklabels(["J&F", "J (IoU)", "F (Boundary)"])
    ax3.set_title("Performance Heatmap", fontweight="bold")

    # Add text annotations
    for i in range(3):
        for j in range(len(datasets)):
            ax3.text(j, i, f"{heatmap_data[i, j]:.1f}", ha="center", va="center", color="black", fontweight="bold")

    plt.colorbar(im, ax=ax3, label="Score")

    # 4. Stacked bar chart showing metric breakdown
    ax4 = axes[1, 1]
    sns.barplot(data=df, x="Dataset", y="Score", hue="Metric", ax=ax4)
    ax4.set_title("Detailed Metrics Breakdown", fontweight="bold")
    ax4.set_ylabel("Score")
    ax4.set_ylim(0, 100)

    # Rotate x-axis labels if needed
    for ax in axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    plt.tight_layout()

    # Save plots
    plots_dir = output_path / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)

    plot_path = plots_dir / "dataset_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.savefig(plots_dir / "dataset_comparison.pdf", bbox_inches="tight")

    print(f"Comparison plots saved to: {plot_path}")

    # Save results table
    results_table = plots_dir / "results_table.csv"
    with open(results_table, "w") as file_handle:
        file_handle.write("Dataset,J&F,J (IoU),F (Boundary)\n")
        for dataset in datasets:
            j_f, j, f = results[dataset]
            file_handle.write(f"{dataset},{j_f:.2f},{j:.2f},{f:.2f}\n")

    print(f"Results table saved to: {results_table}")

    # plt.show()  # Commented out to avoid Qt issues in headless mode


def parse_args():
    p = argparse.ArgumentParser(description="Multi-dataset Zero-shot SAM-2 evaluation")

    # Dataset selection
    p.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=list(DATASET_CONFIGS.keys()),
        help="Datasets to evaluate (default: all)",
    )

    # SAM-2 configuration
    p.add_argument(
        "--sam2_cfg",
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM-2 config file",
    )
    p.add_argument(
        "--sam2_checkpoint",
        default="/home/hongyou/dev/ada_samp/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
        help="SAM-2 checkpoint path",
    )

    # Evaluation parameters
    p.add_argument("--device", default="cuda", help="Device to use")
    p.add_argument("--score_thresh", type=float, default=0.0, help="Mask logit threshold")
    p.add_argument(
        "--prompt_method",
        type=str,
        default="gt_box",
        choices=["gt_box", "three_clicks"],
        help="Prompting strategy: gt_box (default) or three_clicks",
    )
    p.add_argument(
        "--first_frame_only",
        action="store_true",
        help="Evaluate only the first frame per video by copying only the first PNG",
    )
    p.add_argument("--num_workers", type=int, default=None, help="Number of evaluation processes")
    p.add_argument("--output_path", default="./outputs/zs_04_09_sam", help="Root output directory")

    # Visualization and subset options
    p.add_argument("--save_vis", action="store_true", default=True, help="Save visualizations")
    p.add_argument("--enhanced_vis", action="store_true", default=True, help="Generate enhanced multi-object visualizations")
    p.add_argument("--video_limit", type=int, default=None, help="Limit number of videos per dataset (for quick testing)")
    p.add_argument("--max_objects", type=int, default=20, help="Maximum number of objects to process per video (default: 15)")

    return p.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load SAM-2 predictor
    print("Loading SAM-2 checkpoint...")
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=args.device,
    )
    print("SAM-2 loaded successfully!")

    # Run evaluation on each dataset
    results = {}
    total_start_time = time.time()

    for dataset_name in args.datasets:
        try:
            # Get video subset if limit is specified
            video_subset = None
            if args.video_limit is not None:
                config = DATASET_CONFIGS[dataset_name]
                root = Path(config["root"])
                split = config["default_split"]

                if config["has_split_subdir"]:
                    jpeg_dir = root / split / "JPEGImages"
                else:
                    jpeg_dir = root / "JPEGImages"

                if jpeg_dir.exists():
                    all_videos = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
                    video_subset = all_videos[: args.video_limit]
                    print(f"Limited to {len(video_subset)} videos for {dataset_name}")

            # Run evaluation
            j_f, j, f = run_single_dataset(
                dataset_name=dataset_name,
                predictor=predictor,
                output_path=output_path,
                score_thresh=args.score_thresh,
                num_workers=args.num_workers,
                video_subset=video_subset,
                save_vis=args.save_vis,
                enhanced_vis=args.enhanced_vis,
                max_objects=args.max_objects,
                prompt_method=args.prompt_method,
                first_frame_only=args.first_frame_only,
            )

            results[dataset_name] = (j_f, j, f)
            print(f"{dataset_name} Results - J&F: {j_f:.2f}, J: {j:.2f}, F: {f:.2f}")

        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            continue

    total_time = time.time() - total_start_time

    # Print summary
    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Dataset':<12} {'J&F':<8} {'J (IoU)':<8} {'F (Boundary)':<12}")
    print("-" * 80)

    for dataset_name, (j_f, j, f) in results.items():
        print(f"{dataset_name:<12} {j_f:<8.2f} {j:<8.2f} {f:<12.2f}")

    print(f"\nTotal evaluation time: {total_time:.2f}s")

    # Create comparison plots
    if len(results) > 1:
        print("\nGenerating comparison plots...")
        create_comparison_plots(results, output_path)

    print(f"\nAll outputs saved to: {output_path}")


if __name__ == "__main__":
    main()
