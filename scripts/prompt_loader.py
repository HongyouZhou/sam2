#!/usr/bin/env python
"""Unified prompt loading utilities for zero-shot evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path


def load_reused_prompts(
    reuse_prompts_root: Path | None,
    dataset_name: str,
    video_name: str,
) -> dict[int, dict] | None:
    """Load reused prompts from a previous run if available.
    
    Args:
        reuse_prompts_root: Root directory containing previous run outputs
        dataset_name: Name of the dataset (e.g., "GTEA")
        video_name: Name of the video
        
    Returns:
        Dictionary mapping object_id to prompt spec, or None if not found
        Format: {obj_id: {"type": "3click", "clicks": [{"xy": [x, y], "label": 1}, ...]}}
    """
    if reuse_prompts_root is None:
        return None
        
    prompt_file = (
        reuse_prompts_root 
        / f"{dataset_name.lower()}_pred" 
        / video_name 
        / "query_prompts.json"
    )
    
    if not prompt_file.exists():
        return None
    
    with open(prompt_file) as f:
        prompts_json = json.load(f)
    
    # Convert string keys to int
    return {int(k): v for k, v in prompts_json.items()}


def apply_reused_prompts(
    predictor,
    state,
    obj_id: int,
    prompt_spec: dict,
) -> bool:
    """Apply reused prompts to predictor state.
    
    Args:
        predictor: SAM2 predictor instance
        state: Predictor state
        obj_id: Object ID
        prompt_spec: Prompt specification from loaded JSON
        
    Returns:
        True if prompts were successfully applied, False otherwise
    """
    prompt_type = prompt_spec.get("type")
    clicks = prompt_spec.get("clicks")
    
    if prompt_type not in {"1click", "3click", "5click"} or not clicks:
        return False
    
    # Apply first click
    if len(clicks) >= 1:
        c0 = clicks[0]
        predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=obj_id,
            points=[[int(c0["xy"][0]), int(c0["xy"][1])]],
            labels=[int(c0["label"])],
        )
    
    # Apply subsequent clicks
    for c in clicks[1:]:
        predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=obj_id,
            points=[[int(c["xy"][0]), int(c["xy"][1])]],
            labels=[int(c["label"])],
            clear_old_points=False,
        )
    
    return True


def save_prompts_to_json(
    output_dir: Path,
    video_name: str,
    obj_prompts: dict[int, list[tuple[int, int, int]]],  # Now includes label
    click_protocol: str,
) -> None:
    """Save prompts to JSON file in standard format.
    
    Args:
        output_dir: Output directory root
        video_name: Video name
        obj_prompts: Dictionary mapping object_id to list of (x, y, label) tuples
        click_protocol: Protocol name (e.g., "3click")
    """
    video_dir = output_dir / video_name
    video_dir.mkdir(parents=True, exist_ok=True)
    
    prompts_to_save = {}
    for obj_id, points_list in obj_prompts.items():
        prompts_to_save[int(obj_id)] = {
            "type": click_protocol,
            "clicks": [
                {"xy": [int(x), int(y)], "label": int(label)} 
                for (x, y, label) in points_list
            ]
        }
    
    with open(video_dir / "query_prompts.json", "w") as f:
        json.dump(prompts_to_save, f, indent=2)

