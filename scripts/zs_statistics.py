#!/usr/bin/env python
"""
Unified statistics extraction for zero-shot evaluation.

Consolidates duplicated statistics extraction code from:
- zero_shot_multi_dataset_sam_bndl.py
- zero_shot_multi_dataset_uctta.py
- zero_shot_multi_dataset_ur_ern.py

Usage:
    from zs_statistics import (
        create_dataset_evaluator,
        extract_statistics_from_evaluator,
        build_statistics_dict,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Lazy imports to avoid loading heavy dependencies at module import
# These will be imported when actually used


def create_dataset_evaluator(
    output_dir: Path,
    dataset_name: str,
    method_name: str,
    per_pixel_statistics: bool = True,
) -> Any:
    """
    Factory for creating DistributedDatasetEvaluator with consistent settings.
    
    Args:
        output_dir: Root output directory
        dataset_name: Name of dataset (e.g., "GTEA")
        method_name: Name of method (e.g., "bndl_aue")
    
    Returns:
        DistributedDatasetEvaluator instance
    """
    from training.utils.dataset_evaluator import DistributedDatasetEvaluator
    
    # Use consistent path format: <output_root>/<dataset>_<method>_eval
    eval_dir = output_dir / f"{dataset_name.lower()}_{method_name.lower()}_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    return DistributedDatasetEvaluator(
        save_dir=str(eval_dir),
        distributed=False,  # Single process for zero-shot evaluation
        rank=0,
        world_size=1,
        foreground_dilation=4,  # Match training configuration
        use_full_image=False,
        per_pixel_statistics=per_pixel_statistics,
    )


def extract_statistics_from_evaluator(
    dataset_eval: Any,
    dataset_name: str,
    method_name: str,
    downsample_max_samples: int = 100000,
) -> dict[str, Any]:
    """
    Extract standardized statistics dict from evaluator.
    
    This consolidates the identical statistics extraction code from
    all three inference scripts (BNDL, UCTTA, UR-ERN).
    
    Args:
        dataset_eval: DistributedDatasetEvaluator instance
        dataset_name: Name of dataset
        method_name: Name of method (for logging)
        downsample_max_samples: Maximum samples for PAvPU scatter plots
    
    Returns:
        Statistics dictionary with standardized structure
    """
    from downsampling_utils import smart_downsample_samples
    
    # Get data source based on evaluator mode
    if dataset_eval.per_pixel_statistics:
        data_source = dataset_eval.pixel_data['uncertainties'].tolist()
        iou_source = dataset_eval.pixel_data['ious'].tolist()
        dice_source = dataset_eval.pixel_data['dices'].tolist()
        accuracy_source = dataset_eval.pixel_data['accuracies'].tolist()
    else:
        data_source = dataset_eval.image_uncertainties
        iou_source = dataset_eval.image_ious
        dice_source = dataset_eval.image_dices
        accuracy_source = dataset_eval.image_accuracies
    
    # Build base statistics
    statistics = build_statistics_dict(
        uncertainties=data_source,
        ious=iou_source,
        dices=dice_source,
        accuracies=accuracy_source,
    )
    
    # Add correlation results
    statistics['correlation_results'] = dataset_eval.correlation_results
    statistics['summary'] = dataset_eval.get_summary_statistics()
    statistics['num_samples'] = len(data_source)
    
    # Sample raw data for PAvPU scatter plot
    max_samples = min(10000, len(data_source))
    if data_source and accuracy_source:
        total_samples = min(len(data_source), len(accuracy_source))
        if total_samples > max_samples:
            indices = np.random.choice(total_samples, max_samples, replace=False)
            uncertainty_samples = [data_source[i] for i in indices]
            accuracy_samples = [accuracy_source[i] for i in indices]
        else:
            uncertainty_samples = list(data_source)
            accuracy_samples = list(accuracy_source)
    else:
        uncertainty_samples = []
        accuracy_samples = []
    
    # Downsample for storage
    unc_down, acc_down = smart_downsample_samples(
        uncertainty_samples, 
        accuracy_samples, 
        max_samples=downsample_max_samples
    )
    statistics['eval_pavpu_uncertainty_samples'] = unc_down
    statistics['eval_pavpu_accuracy_samples'] = acc_down
    
    # Log summary
    n_stored = len(unc_down)
    n_original = len(uncertainty_samples)
    if n_stored < n_original:
        print(f"{method_name} statistics: {statistics['num_samples']} samples, "
              f"mean uncertainty: {statistics['pixel_uncertainty_mean']:.4f}, "
              f"PAvPU: {n_original:,} → {n_stored:,} ({n_stored/n_original*100:.1f}%)")
    else:
        print(f"{method_name} statistics: {statistics['num_samples']} samples, "
              f"mean uncertainty: {statistics['pixel_uncertainty_mean']:.4f}")
    
    return statistics


def build_statistics_dict(
    uncertainties: list[float],
    ious: list[float] | None = None,
    dices: list[float] | None = None,
    accuracies: list[float] | None = None,
) -> dict[str, Any]:
    """
    Build standardized statistics dictionary from raw metric lists.
    
    Args:
        uncertainties: List of uncertainty values
        ious: Optional list of IoU values
        dices: Optional list of Dice values
        accuracies: Optional list of accuracy values
    
    Returns:
        Dictionary with mean, std, median, min, max for each metric
    """
    stats: dict[str, Any] = {}
    
    # Uncertainty statistics
    if uncertainties:
        stats['pixel_uncertainty_mean'] = float(np.mean(uncertainties))
        stats['pixel_uncertainty_std'] = float(np.std(uncertainties))
        stats['pixel_uncertainty_median'] = float(np.median(uncertainties))
        stats['pixel_uncertainty_min'] = float(np.min(uncertainties))
        stats['pixel_uncertainty_max'] = float(np.max(uncertainties))
    else:
        stats['pixel_uncertainty_mean'] = 0.0
        stats['pixel_uncertainty_std'] = 0.0
        stats['pixel_uncertainty_median'] = 0.0
        stats['pixel_uncertainty_min'] = 0.0
        stats['pixel_uncertainty_max'] = 0.0
    
    # Performance metrics
    stats['iou_mean'] = float(np.mean(ious)) if ious else 0.0
    stats['dice_mean'] = float(np.mean(dices)) if dices else 0.0
    stats['accuracy_mean'] = float(np.mean(accuracies)) if accuracies else 0.0
    
    return stats


def run_dataset_correlation_analysis(
    dataset_eval: Any,
    dataset_name: str,
    method_name: str,
) -> None:
    """
    Run correlation analysis and create visualizations.
    
    Args:
        dataset_eval: DistributedDatasetEvaluator instance
        dataset_name: Name of dataset
        method_name: Name of method
    """
    try:
        dataset_eval.evaluate_dataset_correlation()
        dataset_eval.create_dataset_correlation_visualization(
            title=f"{dataset_name} {method_name} - Dataset Correlation",
            save_name=f"{dataset_name.lower()}_{method_name.lower()}_dataset_analysis.png",
        )
        dataset_eval.save_correlation_results(
            save_name=f"{dataset_name.lower()}_{method_name.lower()}_results.json"
        )
    except Exception as e:
        print(f"Warning: {method_name} dataset evaluation failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("Statistics Module Tests")
    print("=" * 60)
    
    # Test build_statistics_dict
    test_unc = [0.1, 0.2, 0.3, 0.4, 0.5]
    test_iou = [0.8, 0.85, 0.9, 0.82, 0.88]
    
    stats = build_statistics_dict(test_unc, test_iou)
    print(f"\n1. build_statistics_dict:")
    print(f"  uncertainty mean: {stats['pixel_uncertainty_mean']:.3f}")
    print(f"  iou mean: {stats['iou_mean']:.3f}")
    
    print("\n✓ All tests passed!")
