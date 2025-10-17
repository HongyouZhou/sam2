#!/usr/bin/env python
"""
从缓存的results重新生成MMD plots
无需重新运行evaluation

用法:
    python scripts/plot_mmd_from_cache.py \
        --cached_json ./outputs/zs_001/comparison_plots/detailed_results.json \
        --output_dir ./outputs/replot_mmd \
        --methods BNDL BNDL_AUE UCTTA UR-ERN
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from zs import (
    compute_mmd_consistency_metrics,
    compute_correlation_consistency_metrics,
    create_mmd_comparison_across_methods,
)


def load_cached_results(json_path: Path) -> dict:
    """Load cached evaluation results"""
    if not json_path.exists():
        raise FileNotFoundError(f"Cache file not found: {json_path}")
    
    with open(json_path) as f:
        data = json.load(f)
    
    print(f"✓ Loaded cached results from: {json_path}")
    print(f"  - Methods: {[k for k in data.keys() if 'results' in k or 'statistics' in k]}")
    
    return data


def extract_statistics(data: dict, method_key: str) -> dict[str, Any] | None:
    """Extract statistics for a specific method"""
    stats = data.get(method_key, {})
    if stats:
        print(f"  ✓ {method_key}: {len(stats)} datasets")
        
        # 验证是否包含UA samples
        first_dataset = next(iter(stats.values())) if stats else {}
        if isinstance(first_dataset, dict):
            has_unc = 'eval_pavpu_uncertainty_samples' in first_dataset
            has_acc = 'eval_pavpu_accuracy_samples' in first_dataset
            if has_unc and has_acc:
                print(f"    ✓ Contains UA samples (uncertainty + accuracy)")
            else:
                print(f"    ⚠ Missing UA samples")
    else:
        print(f"  ✗ {method_key}: Not found")
    
    return stats if stats else None


def filter_methods(data: dict, selected_methods: list[str]) -> dict:
    """Filter to only selected methods"""
    method_map = {
        'SAM': 'sam2_statistics',
        'BNDL': 'bndl_statistics',
        'BNDL_AUE': 'bndl_aue_statistics',
        'UCTTA': 'uctta_statistics',
        'UR-ERN': 'ur_ern_statistics',
    }
    
    filtered = {}
    for method in selected_methods:
        stats_key = method_map.get(method)
        if stats_key:
            stats = data.get(stats_key, {})
            if stats:
                filtered[method] = stats
    
    return filtered


def plot_mmd_from_cache(
    cached_json: Path,
    output_dir: Path,
    selected_methods: list[str] | None = None,
    source_domain: str = "MOSE_train",
):
    """
    从缓存的results生成MMD plots
    
    Args:
        cached_json: Path to detailed_results.json
        output_dir: Output directory for new plots
        selected_methods: List of methods to include (None = all)
        source_domain: Source domain for MMD computation
    """
    # Load cached results
    data = load_cached_results(cached_json)
    
    # Extract statistics for all methods
    print("\n📊 Extracting statistics...")
    bndl_aue_stats = extract_statistics(data, 'bndl_aue_statistics')
    bndl_stats = extract_statistics(data, 'bndl_statistics')
    uctta_stats = extract_statistics(data, 'uctta_statistics')
    ur_ern_stats = extract_statistics(data, 'ur_ern_statistics')
    
    # Filter methods if specified
    if selected_methods:
        print(f"\n🔍 Filtering to selected methods: {selected_methods}")
        stats_map = {
            'BNDL': bndl_stats,
            'BNDL_AUE': bndl_aue_stats,
            'UCTTA': uctta_stats,
            'UR-ERN': ur_ern_stats,
        }
        
        bndl_stats = stats_map.get('BNDL') if 'BNDL' in selected_methods else None
        bndl_aue_stats = stats_map.get('BNDL_AUE') if 'BNDL_AUE' in selected_methods else None
        uctta_stats = stats_map.get('UCTTA') if 'UCTTA' in selected_methods else None
        ur_ern_stats = stats_map.get('UR-ERN') if 'UR-ERN' in selected_methods else None
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate MMD comparison plots
    print(f"\n📈 Generating MMD plots...")
    print(f"   Output: {output_dir}")
    
    if bndl_aue_stats or bndl_stats or uctta_stats or ur_ern_stats:
        create_mmd_comparison_across_methods(
            output_path=output_dir,
            bndl_aue_statistics=bndl_aue_stats,
            bndl_statistics=bndl_stats,
            uctta_statistics=uctta_stats,
            ur_ern_statistics=ur_ern_stats,
            source_domain=source_domain,
        )
        print(f"\n✓ Plots generated in: {output_dir / 'comparison_plots'}")
    else:
        print("\n⚠️  No valid statistics found for plotting")
    
    # Also compute and print MMD metrics
    print("\n📊 MMD Metrics Summary:")
    print("="*60)
    
    for method_name, stats in [
        ('BNDL', bndl_stats),
        ('BNDL_AUE', bndl_aue_stats),
        ('UCTTA', uctta_stats),
        ('UR-ERN', ur_ern_stats)
    ]:
        if stats:
            # Extract UA samples
            ua_samples = {}
            for dataset, dataset_stats in stats.items():
                if isinstance(dataset_stats, dict):
                    unc = dataset_stats.get('eval_pavpu_uncertainty_samples', [])
                    acc = dataset_stats.get('eval_pavpu_accuracy_samples', [])
                    if unc and acc and len(unc) == len(acc):
                        ua_samples[dataset] = {
                            'uncertainty': unc,
                            'error': [1.0 - a for a in acc]
                        }
            
            if ua_samples and source_domain in ua_samples:
                # Compute MMD metrics
                mmd_metrics = compute_mmd_consistency_metrics(
                    ua_samples,
                    source_domain=source_domain,
                    gamma=1.0
                )
                
                if mmd_metrics:
                    print(f"\n{method_name}:")
                    print(f"  MMD_mean: {mmd_metrics.get('mmd_mean', 0):.4f}")
                    print(f"  MMD_std: {mmd_metrics.get('mmd_std', 0):.4f}")
                    print(f"  MMD_min: {mmd_metrics.get('mmd_min', 0):.4f}")
                    print(f"  MMD_max: {mmd_metrics.get('mmd_max', 0):.4f}")
                    print(f"  UA_dist_consistency: {mmd_metrics.get('ua_dist_consistency', 0):.4f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="从缓存的results重新生成MMD plots"
    )
    
    parser.add_argument(
        "--cached_json",
        type=str,
        required=True,
        help="Path to detailed_results.json from previous run"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/replot_mmd",
        help="Output directory for regenerated plots"
    )
    
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=['SAM', 'BNDL', 'BNDL_AUE', 'UCTTA', 'UR-ERN'],
        help="Methods to include (default: all available)"
    )
    
    parser.add_argument(
        "--source_domain",
        type=str,
        default="MOSE_train",
        help="Source domain for MMD computation"
    )
    
    args = parser.parse_args()
    
    # Run plotting
    plot_mmd_from_cache(
        cached_json=Path(args.cached_json),
        output_dir=Path(args.output_dir),
        selected_methods=args.methods,
        source_domain=args.source_domain,
    )
    
    print("\n✅ Done! 无需重新运行evaluation")


if __name__ == "__main__":
    main()

