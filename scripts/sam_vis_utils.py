#!/usr/bin/env python
"""
SAM Baseline Visualization Utilities
Generate paper-quality visualizations for SAM baseline that match BNDL output format
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_object_colors(n_colors: int = 20):
    """Get a list of distinct colors for multi-object visualization.
    
    Uses colorcet's glasbey palette if available (up to 256 colors),
    otherwise falls back to matplotlib's tab20 (20 colors).
    
    Args:
        n_colors: Number of colors needed
        
    Returns:
        List of RGB tuples normalized to [0, 1]
    """
    try:
        import colorcet as cc
        # glasbey provides up to 256 perceptually distinct colors (as hex strings)
        hex_colors = cc.glasbey_dark[:min(n_colors, 256)]
        # Convert hex to RGB tuples normalized to [0, 1]
        colors = []
        for hex_color in hex_colors:
            hex_color = hex_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            colors.append((r, g, b))
        return colors
    except (ImportError, AttributeError):
        pass
    
    # Fallback to matplotlib's tab20 colormap (20 distinct colors)
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20')
    return [cmap(i / 20)[:3] for i in range(min(n_colors, 20))]


# Pre-generate 20 colors for common use
OBJECT_COLORS = get_object_colors(20)


def save_sam_individual_visualizations(
    original_img: np.ndarray,
    pred_mask: np.ndarray,
    point_coords: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
    vis_dir: str | Path,
    data_iter: int,
    step_index: int = 0,
    dpi: int = 150,
):
    """Save individual visualization images for SAM baseline
    
    Args:
        original_img: Original image as numpy array [H, W, 3] in range [0, 1] or [0, 255]
        pred_mask: Predicted binary mask [H, W] or combined mask from multiple objects
        point_coords: Click coordinates [N, 2] in (x, y) format
        point_labels: Click labels [N] where 1=positive, 0=negative
        vis_dir: Output directory for visualizations
        data_iter: Data iteration number (e.g., frame index)
        step_index: Step index within iteration
        dpi: DPI for saved images
    """
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    base_filename = f"iter_{data_iter}_step_{step_index}"
    
    # Normalize image to [0, 1] if needed
    if original_img.max() > 1.0:
        original_img = original_img.astype(np.float32) / 255.0
    
    # 1. Save original image
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(original_img)
    ax.set_title("Original Image")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(vis_dir / f"{base_filename}_original.png", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    # 2. Save original with predicted mask overlay
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(original_img)
    
    if pred_mask is not None:
        # Resize mask if needed
        if pred_mask.shape[:2] != original_img.shape[:2]:
            pred_mask = cv2.resize(
                pred_mask.astype(np.float32), 
                (original_img.shape[1], original_img.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        
        # Check if multi-object mask (values > 1 indicate different object IDs)
        unique_ids = np.unique(pred_mask)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background (0)
        
        if len(unique_ids) > 1:
            # Multi-object: use different colors for each object
            mask_overlay = np.zeros((*pred_mask.shape[:2], 4))
            for idx, obj_id in enumerate(unique_ids):
                color = OBJECT_COLORS[idx % len(OBJECT_COLORS)]
                obj_mask = pred_mask == obj_id
                mask_overlay[obj_mask, 0] = color[0]
                mask_overlay[obj_mask, 1] = color[1]
                mask_overlay[obj_mask, 2] = color[2]
                mask_overlay[obj_mask, 3] = 0.5  # Alpha
        else:
            # Single object or binary mask: use cyan
            mask_overlay = np.zeros((*pred_mask.shape[:2], 4))
            mask_bool = pred_mask > 0 if pred_mask.dtype != bool else pred_mask
            mask_overlay[..., 0] = 0.0    # R
            mask_overlay[..., 1] = 0.8    # G (cyan)
            mask_overlay[..., 2] = 0.8    # B (cyan)
            mask_overlay[..., 3] = mask_bool.astype(np.float32) * 0.5  # Alpha
        
        ax.imshow(mask_overlay)
    
    ax.set_title("Predicted Mask Overlay")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(vis_dir / f"{base_filename}_original_with_pred.png", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    # 3. Save standalone prediction mask (multi-object colored visualization)
    if pred_mask is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Resize mask if needed
        if pred_mask.shape[:2] != original_img.shape[:2]:
            pred_mask_resized = cv2.resize(
                pred_mask.astype(np.float32), 
                (original_img.shape[1], original_img.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        else:
            pred_mask_resized = pred_mask
        
        # Check if multi-object mask
        unique_ids = np.unique(pred_mask_resized)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background
        
        mask_vis = np.zeros((*pred_mask_resized.shape[:2], 3))
        
        if len(unique_ids) > 1:
            # Multi-object: use different colors for each object
            for idx, obj_id in enumerate(unique_ids):
                color = OBJECT_COLORS[idx % len(OBJECT_COLORS)]
                obj_mask = pred_mask_resized == obj_id
                mask_vis[obj_mask, 0] = color[0]
                mask_vis[obj_mask, 1] = color[1]
                mask_vis[obj_mask, 2] = color[2]
        else:
            # Single object: use cyan
            mask_bool = pred_mask_resized > 0
            mask_vis[mask_bool, 0] = 0.0   # R
            mask_vis[mask_bool, 1] = 0.9   # G (cyan)
            mask_vis[mask_bool, 2] = 0.9   # B (cyan)
        
        ax.imshow(mask_vis)
        ax.set_title(f"Predicted Mask ({len(unique_ids)} objects)" if len(unique_ids) > 1 else "Predicted Mask")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(vis_dir / f"{base_filename}_pred_mask.png", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    
    # 4. Save original with prompts
    if point_coords is not None and len(point_coords) > 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(original_img)
        
        # Draw prompt points with different colors based on position and label
        # Click order in 3click protocol:
        #   1st: positive (lime)
        #   2nd: negative (red)  
        #   3rd+: error-based (yellow for both FN/FP correction)
        for i, (x, y) in enumerate(point_coords):
            label = point_labels[i] if point_labels is not None and i < len(point_labels) else 1
            
            if i == 0:
                # First click: always positive (green)
                color = 'lime'
                marker = 'o'
            elif i == 1 and label == 0:
                # Second click: negative (red)
                color = 'red'
                marker = 'x'
            else:
                # 3rd+ clicks: error-based (yellow)
                # Also handles 5click protocol where 2nd+ might be error-based
                color = 'gold'  # Yellow for error-based
                marker = '*' if label == 1 else 'x'
            
            ax.scatter(x, y, c=color, s=200, marker=marker, edgecolors='white', linewidths=2)
        
        ax.set_title("Input Prompts")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(vis_dir / f"{base_filename}_original_with_prompts.png", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)


def generate_sam_paper_visualizations(
    jpeg_dir: Path,
    pred_dir: Path,
    ann_dir: Path,
    vis_root: Path,
    video_names: list[str] | None = None,
    max_frames_per_video: int = 2,
):
    """Generate paper-quality visualizations for SAM baseline
    
    Creates individual PNG files matching BNDL output format for direct comparison.
    
    Args:
        jpeg_dir: Directory containing source JPEG images
        pred_dir: Directory containing predicted masks
        ann_dir: Directory containing ground truth annotations
        vis_root: Output directory for visualizations
        video_names: Optional list of video names to process
        max_frames_per_video: Maximum frames to visualize per video
    """
    vis_root.mkdir(parents=True, exist_ok=True)
    
    if video_names is None:
        video_names = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    else:
        video_names = sorted(set(video_names))
    
    print(f"Generating SAM paper visualizations for {len(video_names)} videos...")
    
    for vid in video_names:
        jpg_dir = jpeg_dir / vid
        pred_dir_video = pred_dir / vid
        vis_dir_video = vis_root / vid
        vis_dir_video.mkdir(parents=True, exist_ok=True)
        
        # Read stored points for all objects
        import json
        pts_json = pred_dir_video / "query_points.json"
        prompts_json = pred_dir_video / "query_prompts.json"
        
        if prompts_json.exists():
            with open(prompts_json) as f:
                prompt_specs = {int(k): v for k, v in json.load(f).items()}
        elif pts_json.exists():
            with open(pts_json) as f:
                query_pts_dict = {int(k): v for k, v in json.load(f).items()}
            # Convert old format to new format
            prompt_specs = {}
            for obj_id, pts in query_pts_dict.items():
                if pts:
                    clicks = [{"xy": [p[0], p[1]], "label": p[2] if len(p) > 2 else 1} for p in pts]
                    prompt_specs[obj_id] = {"clicks": clicks}
        else:
            prompt_specs = {}
        
        # Get image files
        img_files = sorted(
            [p for p in jpg_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]], 
            key=lambda x: int(x.stem)
        )
        
        # Sample frames evenly
        if len(img_files) > max_frames_per_video:
            step = len(img_files) // max_frames_per_video
            img_files = img_files[::step][:max_frames_per_video]
        
        for img_path in img_files:
            mask_path = pred_dir_video / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue
            
            try:
                # Load image
                original_img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
                
                # Load predicted mask (combined for all objects)
                pred_mask_all = np.array(Image.open(mask_path))
                pred_mask = pred_mask_all > 0  # Binary mask
                
                # Combine prompts from all objects
                all_coords = []
                all_labels = []
                for obj_id, spec in prompt_specs.items():
                    if "clicks" in spec:
                        for click in spec["clicks"]:
                            all_coords.append(click["xy"])
                            all_labels.append(click.get("label", 1))
                
                point_coords = np.array(all_coords) if all_coords else None
                point_labels = np.array(all_labels) if all_labels else None
                
                # Generate visualizations
                save_sam_individual_visualizations(
                    original_img=original_img,
                    pred_mask=pred_mask,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    vis_dir=vis_dir_video,
                    data_iter=int(img_path.stem),
                    step_index=0,
                )
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    print(f"SAM paper visualizations saved to: {vis_root}")
