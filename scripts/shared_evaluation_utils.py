#!/usr/bin/env python
"""
Minimal shared utilities for zero-shot evaluation scripts.
Only contains the essential duplicated functions.
"""

from __future__ import annotations

import pandas as pd
from sam2.build_sam import build_sam2_video_predictor


def build_predictor_with_overrides(
    cfg_file: str, 
    ckpt: str, 
    device: str = "cuda", 
    multimask: bool = True, 
    min_pts: int = 1, 
    max_pts: int = 2, 
    for_tracking: bool = False
):
    """Build SAM-2 predictor with consistent overrides."""
    hydra_overrides_extra = []
    if multimask:
        hydra_overrides_extra += [
            "++model.multimask_output_in_sam=true",
            f"++model.multimask_min_pt_num={min_pts}",
            f"++model.multimask_max_pt_num={max_pts}",
        ]
        if for_tracking:
            hydra_overrides_extra += ["++model.multimask_output_for_tracking=true"]
    
    return build_sam2_video_predictor(
        config_file=cfg_file,
        ckpt_path=ckpt,
        device=device,
        hydra_overrides_extra=hydra_overrides_extra,
    )


def create_append_shard_to_eval_callback(dataset_eval, downsample_max_samples: int = 100000):
    """Create callback for appending shard data to evaluator."""
    def _append_shard_to_eval(shard_data):
        if not shard_data:
            return
        
        # Collect all data
        data_dict = {}
        if shard_data.get('pixel_uncertainties'):
            data_dict['uncertainties'] = shard_data['pixel_uncertainties']
        if shard_data.get('pixel_accuracies'):
            data_dict['accuracies'] = shard_data['pixel_accuracies']
        if shard_data.get('pixel_ious'):
            data_dict['ious'] = shard_data['pixel_ious']
        if shard_data.get('pixel_dices'):
            data_dict['dices'] = shard_data['pixel_dices']
        if shard_data.get('pixel_nlls'):
            data_dict['nlls'] = shard_data['pixel_nlls']
        
        if data_dict:
            # Create new DataFrame and append
            new_data = pd.DataFrame(data_dict)
            dataset_eval.pixel_data = pd.concat([dataset_eval.pixel_data, new_data], ignore_index=True)
            
            # Downsample if needed
            if len(dataset_eval.pixel_data) > downsample_max_samples:
                dataset_eval.pixel_data = dataset_eval.pixel_data.sample(n=downsample_max_samples, random_state=42).reset_index(drop=True)
                print(f"  🔄 中间降采样: → {downsample_max_samples:,} 样本")
    
    return _append_shard_to_eval