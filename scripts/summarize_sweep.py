#!/usr/bin/env python
"""
Summarize Sweep Results - Generate summary tables and CSVs from sweep output
Extracted from sweep_checkpoints.py

Usage:
    python scripts/summarize_sweep.py \
        --output_base outputs/sweep_results \
        --datasets GTEA MOSE_val ... \
        [--checkpoints_dir /path/to/checkpoints]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

# Add the script directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dataset_configs import DEFAULT_DATASETS
except ImportError:
    DEFAULT_DATASETS = [] # Fallback if not found

def collect_and_summarize_results(output_base: Path, checkpoint_names: list[str], datasets: list[str]):
    """收集所有 checkpoint 和数据集的结果"""
    results = defaultdict(dict)  # {checkpoint: {dataset: {jf, j, f}}}
    
    for ckpt_name in checkpoint_names:
        for dataset in datasets:
            # 构建结果文件路径
            # 优先尝试标准路径
            results_file = output_base / ckpt_name / dataset / "comparison_plots" / "detailed_results.json"
            
            if not results_file.exists():
                # 尝试搜索
                candidates = list((output_base / ckpt_name / dataset).glob("**/detailed_results.json"))
                if candidates:
                    results_file = candidates[0]
                else:
                    continue
            
            try:
                with open(results_file) as f:
                    data = json.load(f)
                
                # 提取 BNDL_AUE 结果
                if "bndl_aue_results" in data and dataset in data["bndl_aue_results"]:
                    scores = data["bndl_aue_results"][dataset]
                    results[ckpt_name][dataset] = {
                        "J&F": scores.get("jf", 0.0),
                        "J": scores.get("j", 0.0),
                        "F": scores.get("f", 0.0),
                    }
            except Exception as e:
                print(f"⚠️  Failed to load {results_file}: {e}")
    
    return results

def print_summary_table(results: dict):
    """打印汇总表格"""
    # 提取所有 checkpoint 和数据集
    def sort_key(x):
        parts = x.split('_')
        if parts[-1].isdigit():
            return int(parts[-1])
        return x

    checkpoints = sorted(results.keys(), key=sort_key)
    
    all_datasets = set()
    for ckpt_results in results.values():
        all_datasets.update(ckpt_results.keys())
    datasets = sorted(all_datasets)
    
    if not datasets:
        print("No datasets with results found!")
        return
    
    # 为每个指标创建表格
    for metric in ["J&F", "J", "F"]:
        print(f"\n{'=' * 80}")
        print(f"{metric} Metric")
        print(f"{'=' * 80}")
        
        # 创建表格数据
        table_data = []
        for dataset in datasets:
            row = {"Dataset": dataset}
            for ckpt in checkpoints:
                if dataset in results[ckpt]:
                    val = results[ckpt][dataset][metric]
                    row[ckpt] = f"{val:.2f}"
                else:
                    row[ckpt] = "-"
            table_data.append(row)
        
        # 添加平均行
        avg_row = {"Dataset": "AVERAGE"}
        for ckpt in checkpoints:
            scores = [results[ckpt][ds][metric] for ds in datasets if ds in results[ckpt]]
            if scores:
                avg = sum(scores) / len(scores)
                avg_row[ckpt] = f"{avg:.2f}"
            else:
                avg_row[ckpt] = "-"
        table_data.append(avg_row)
        
        df = pd.DataFrame(table_data)
        print(df.to_string(index=False))
    
    # 找到最佳 checkpoint
    print(f"\n{'=' * 80}")
    print("🏆 Best Checkpoints")
    print(f"{'=' * 80}")
    
    for metric in ["J&F", "J", "F"]:
        avg_scores = {}
        for ckpt in checkpoints:
            scores = [results[ckpt][ds][metric] for ds in datasets if ds in results[ckpt]]
            if scores:
                avg_scores[ckpt] = sum(scores) / len(scores)
        
        if avg_scores:
            best_ckpt = max(avg_scores, key=avg_scores.get)
            best_score = avg_scores[best_ckpt]
            print(f"  {metric:6s}: {best_ckpt:20s} ({best_score:.2f})")

def save_summary_csv(results: dict, output_base: Path):
    """保存汇总结果到 CSV"""
    def sort_key(x):
        parts = x.split('_')
        if parts[-1].isdigit():
            return int(parts[-1])
        return x

    checkpoints = sorted(results.keys(), key=sort_key)
    all_datasets = set()
    for ckpt_results in results.values():
        all_datasets.update(ckpt_results.keys())
    datasets = sorted(all_datasets)
    
    for metric in ["J&F", "J", "F"]:
        # 创建表格数据
        table_data = []
        for dataset in datasets:
            row = {"Dataset": dataset}
            for ckpt in checkpoints:
                if dataset in results[ckpt]:
                    row[ckpt] = results[ckpt][dataset][metric]
                else:
                    row[ckpt] = None
            table_data.append(row)
        
        # 添加平均行
        avg_row = {"Dataset": "AVERAGE"}
        for ckpt in checkpoints:
            scores = [results[ckpt][ds][metric] for ds in datasets if ds in results[ckpt]]
            avg_row[ckpt] = sum(scores) / len(scores) if scores else None
        table_data.append(avg_row)
        
        df = pd.DataFrame(table_data)
        csv_path = output_base / f"summary_{metric.replace('&', '_')}.csv"
        df.to_csv(csv_path, index=False, float_format="%.2f")
        print(f"✓ Saved {metric} summary to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Summarize SAM2 sweep results")
    parser.add_argument("--output_base", type=str, default="outputs/sweep_results_017_global", help="Base output directory")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Datasets to evaluate")
    parser.add_argument("--checkpoints_dir", type=str, help="Directory containing .pt checkpoints (optional, used to filter/order)")
    
    args = parser.parse_args()
    
    output_base = Path(args.output_base).resolve()
    if not output_base.exists():
        print(f"Error: Directory {output_base} does not exist.")
        
        # Try to help the user by listing available directories in the parent folder
        parent_dir = output_base.parent
        if parent_dir.exists():
            print(f"\nAvailable directories in {parent_dir}:")
            for p in sorted(parent_dir.iterdir()):
                if p.is_dir():
                    print(f"  - {p.name}")
        return

    checkpoint_names = []
    if args.checkpoints_dir:
        checkpoints_dir = Path(args.checkpoints_dir)
        if checkpoints_dir.exists():
            checkpoints = sorted(list(checkpoints_dir.glob("*.pt")), key=lambda p: p.stat().st_mtime)
            checkpoint_names = [ckpt.stem for ckpt in checkpoints]
            print(f"Found {len(checkpoint_names)} checkpoints in {checkpoints_dir}")
        else:
            print(f"Warning: Checkpoints directory {checkpoints_dir} not found. Scanning output_base instead.")
    
    if not checkpoint_names:
        # Scan output_base for directories
        print(f"Scanning {output_base} for results...")
        checkpoint_names = [d.name for d in output_base.iterdir() if d.is_dir()]
        # Sort them nicely
        def sort_key(x):
            parts = x.split('_')
            if parts[-1].isdigit():
                return int(parts[-1])
            return x
        checkpoint_names.sort(key=sort_key)
        print(f"Found {len(checkpoint_names)} potential checkpoint directories.")

    if not checkpoint_names:
        print("No checkpoints found to summarize.")
        return

    print(f"\n{'=' * 80}")
    print("📊 Summarizing results...")
    print(f"{'=' * 80}\n")
    
    try:
        summary_results = collect_and_summarize_results(output_base, checkpoint_names, args.datasets)
        if summary_results:
            print_summary_table(summary_results)
            save_summary_csv(summary_results, output_base)
        else:
            print("⚠️  No results found to summarize.")
    except Exception as e:
        print(f"⚠️  Failed to generate summary: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
