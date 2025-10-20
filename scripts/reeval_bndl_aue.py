#!/usr/bin/env python
"""Re-evaluate BNDL AUE results without re-running inference.

This script re-runs evaluation on existing prediction files after fixing
the temporary directory naming bug.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from evaluation_pipeline import run_benchmark_evaluation
from dataset_configs import DATASET_CONFIGS

def reeval_dataset(dataset_name, output_root, num_workers=8):
    """Re-evaluate a single dataset."""
    
    config = DATASET_CONFIGS[dataset_name]
    split = config["default_split"]
    if isinstance(split, list):
        split = split[0]
    
    root = Path(config["root"])
    if config["has_split_subdir"]:
        ann_dir = root / split / "Annotations"
    else:
        ann_dir = root / "Annotations"
    
    # Convert to absolute path to avoid path resolution issues
    output_path = Path(output_root).resolve()
    pred_dir = output_path / f"{dataset_name.lower()}_pred"
    
    if not pred_dir.exists():
        print(f"❌ Prediction directory not found: {pred_dir}")
        return None
    
    print(f"\n{'='*80}")
    print(f"Re-evaluating {dataset_name} with BNDL AUE")
    print(f"{'='*80}")
    print(f"Ground truth: {ann_dir}")
    print(f"Predictions: {pred_dir}")
    
    try:
        j_f_val, j_val, f_val = run_benchmark_evaluation(
            gt_dir=ann_dir,
            pred_dir=pred_dir,
            dataset_config=config,
            video_subset=None,
            first_frame_only=True,  # BNDL AUE uses first frame only
            num_workers=num_workers,
            output_path=output_path,
            use_symlinks=True,
            dataset_name=dataset_name,  # Fixed: pass dataset name explicitly
        )
        
        print(f"\n✅ {dataset_name} Results:")
        print(f"   J&F: {j_f_val:.2f}")
        print(f"   J:   {j_val:.2f}")
        print(f"   F:   {f_val:.2f}")
        
        return j_f_val, j_val, f_val
        
    except Exception as e:
        print(f"❌ Error evaluating {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Re-evaluate BNDL AUE results")
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Root directory containing prediction results (e.g., outputs/zs_parallel_10/BNDL_AUE_012_06/ADE20K/bndl_aue_results)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset name to re-evaluate"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers for evaluation (default: 8)"
    )
    
    args = parser.parse_args()
    
    result = reeval_dataset(
        dataset_name=args.dataset,
        output_root=args.output_root,
        num_workers=args.num_workers
    )
    
    if result is None:
        sys.exit(1)
    else:
        sys.exit(0)

