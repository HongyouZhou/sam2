#!/usr/bin/env python
"""
Parallel Comparison Runner
并行运行 compare_sam2_vs_bndl.py 的三个方法（SAM, UCTTA, BNDL）在不同GPU上

用法:
    python scripts/parallel_compare.py --datasets GTEA --gpu_ids 0 1 2
    python scripts/parallel_compare.py --datasets GTEA --gpu_ids 0 1 2 --first_frame_only
"""

import argparse
import subprocess
import multiprocessing
import time
import json
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, List, Any
import pandas as pd
import numpy as np

from dataset_configs import (
    DATASET_CONFIGS,
    DATASET_TO_TYPE,
    DATASET_TYPE_CATEGORIES,
    DEFAULT_DATASETS,
)

# 导入 compare_sam2_vs_bndl 中的可视化函数
sys.path.insert(0, str(Path(__file__).parent))
from compare_sam2_vs_bndl import create_comprehensive_comparison_plots, create_ua_shift_analysis_plots


# 方法配置
METHOD_CONFIGS = {
    "SAM": {
        "flags": ["--run_sam"],
        "color": "\033[94m",  # Blue
        "output_suffix": "sam",
    },
    "UCTTA": {
        "flags": ["--run_uctta"],
        "color": "\033[92m",  # Green
        "output_suffix": "uctta",
    },
    "BNDL": {
        "flags": ["--run_bndl"],
        "color": "\033[93m",  # Yellow
        "output_suffix": "bndl",
    },
}

RESET_COLOR = "\033[0m"


def build_command(method: str, gpu_id: int, output_dir: Path, args: argparse.Namespace) -> List[str]:
    """构建单个方法的命令行"""

    cmd = [
        sys.executable,
        "sam2/scripts/compare_sam2_vs_bndl.py",
        "--datasets",
        *args.datasets,
        "--output_path",
        str(output_dir),
        "--device",
        "cuda",
    ]

    # 添加方法特定的flags
    cmd.extend(METHOD_CONFIGS[method]["flags"])

    # SAM-2配置
    if args.sam2_cfg:
        cmd.extend(["--sam2_cfg", args.sam2_cfg])
    if args.sam2_checkpoint:
        cmd.extend(["--sam2_checkpoint", args.sam2_checkpoint])

    # BNDL配置 (只有BNDL方法需要)
    if method == "BNDL":
        if args.bndl_cfg:
            cmd.extend(["--bndl_cfg", args.bndl_cfg])
        if args.bndl_checkpoint:
            cmd.extend(["--bndl_checkpoint", args.bndl_checkpoint])

    # 评估参数
    cmd.extend([
        "--score_thresh",
        str(args.score_thresh),
        "--click_protocol",
        args.click_protocol,
        "--max_objects",
        str(args.max_objects),
        "--seed",
        str(args.seed),
    ])

    # 可选参数
    if args.first_frame_only:
        cmd.append("--first_frame_only")

    if args.video_limit:
        cmd.extend(["--video_limit", str(args.video_limit)])

    if args.num_workers:
        cmd.extend(["--num_workers", str(args.num_workers)])

    # UCTTA特定参数
    if method == "UCTTA":
        cmd.extend([
            "--uctta_steps",
            str(args.uctta_steps),
            "--uctta_lr",
            str(args.uctta_lr),
        ])
        if args.uctta_enable_bn:
            cmd.append("--uctta_enable_bn")
        if args.uctta_fisher_reg:
            cmd.append("--uctta_fisher_reg")

    # BNDL特定参数
    if method == "BNDL" and args.collect_bndl_stats:
        cmd.append("--collect_bndl_stats")

    return cmd


def run_method(method: str, gpu_id: int, output_base: Path, args: argparse.Namespace) -> Tuple[str, int, float, Path]:
    """在指定GPU上运行单个方法"""

    output_dir = output_base / f"output_{METHOD_CONFIGS[method]['output_suffix']}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建命令
    cmd = build_command(method, gpu_id, output_dir, args)

    # 设置环境变量指定GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    color = METHOD_CONFIGS[method]["color"]
    print(f"\n{color}{'=' * 80}{RESET_COLOR}")
    print(f"{color}[{method}] Starting on GPU {gpu_id}{RESET_COLOR}")
    print(f"{color}{'=' * 80}{RESET_COLOR}")
    print(f"{color}Command: {' '.join(cmd)}{RESET_COLOR}")
    print(f"{color}Output: {output_dir}{RESET_COLOR}\n")

    # 日志文件
    log_file = output_dir / f"{method.lower()}_run.log"

    # 运行
    start_time = time.time()

    try:
        with open(log_file, "w") as f:
            process = subprocess.run(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )

        elapsed = time.time() - start_time

        if process.returncode == 0:
            print(f"{color}[{method}] ✓ Completed successfully in {elapsed:.1f}s{RESET_COLOR}")
        else:
            print(f"{color}[{method}] ✗ Failed with code {process.returncode} after {elapsed:.1f}s{RESET_COLOR}")
            print(f"{color}[{method}] Check log: {log_file}{RESET_COLOR}")

        return method, process.returncode, elapsed, output_dir

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"{color}[{method}] ✗ Exception: {e}{RESET_COLOR}")
        return method, -1, elapsed, output_dir


def parse_results_from_output(output_dir: Path, method: str) -> Dict:
    """从输出目录解析结果

    Expected JSON structure from compare_sam2_vs_bndl.py:
    {
        "sam2_results": {"GTEA": {"jf": 91.81, "j": 89.65, "f": 93.96}},
        "uctta_results": {"GTEA": {"jf": 92.50, ...}},
        "bndl_results": {"GTEA": {"jf": 93.20, ...}},
        ...
    }
    """
    results = {}

    # 查找 detailed_results.json
    json_files = list(output_dir.glob("**/detailed_results.json"))

    if json_files:
        try:
            with open(json_files[0], "r") as f:
                data = json.load(f)

            # 映射方法名到 JSON 键
            method_key_map = {
                "SAM": "sam2_results",
                "UCTTA": "uctta_results",
                "BNDL": "bndl_results",
            }

            json_key = method_key_map.get(method)
            if json_key and json_key in data:
                method_data = data[json_key]

                # method_data 格式: {"GTEA": {"jf": 91.81, "j": 89.65, "f": 93.96}}
                for dataset_name, metrics in method_data.items():
                    if isinstance(metrics, dict):
                        results[dataset_name] = {
                            "J&F": metrics.get("jf", 0),
                            "J": metrics.get("j", 0),
                            "F": metrics.get("f", 0),
                        }

            print(f"[{method}] Parsed results for datasets: {list(results.keys())}")

        except Exception as e:
            print(f"[{method}] Warning: Could not parse JSON results: {e}")
            import traceback

            traceback.print_exc()

    # 如果JSON失败，尝试从CSV读取
    if not results:
        csv_files = list(output_dir.glob("**/comparison_results*.csv"))
        if csv_files:
            try:
                df = pd.read_csv(csv_files[0])
                # 处理CSV格式...
            except Exception as e:
                print(f"[{method}] Warning: Could not parse CSV results: {e}")

    return results


def parse_statistics_from_output(output_dir: Path, method: str) -> Dict[str, Any]:
    """从输出目录解析统计数据（用于UA分析）

    Expected JSON structure from compare_sam2_vs_bndl.py:
    {
        "bndl_statistics": {"GTEA": {...}, ...},
        "uctta_statistics": {"GTEA": {...}, ...},
        ...
    }
    """
    statistics = {}

    # 从 detailed_results.json 中提取统计信息
    json_files = list(output_dir.glob("**/detailed_results.json"))
    if json_files:
        try:
            with open(json_files[0], "r") as f:
                data = json.load(f)

            # 映射方法名到统计键
            stats_key_map = {
                "SAM": "sam2_statistics",  # 通常SAM没有统计数据
                "UCTTA": "uctta_statistics",
                "BNDL": "bndl_statistics",
            }

            stats_key = stats_key_map.get(method)
            if stats_key and stats_key in data:
                statistics = data[stats_key]
                if statistics:
                    print(f"[{method}] Parsed statistics for {len(statistics)} datasets")

        except Exception as e:
            print(f"[{method}] Warning: Could not parse statistics from detailed_results: {e}")

    return statistics


def create_parallel_comparison_wrapper(
    all_results: Dict[str, Dict],
    all_statistics: Dict[str, Dict],
    times: Dict[str, float],
    datasets: List[str],
    output_path: Path,
):
    """调用compare_sam2_vs_bndl中的可视化函数"""

    print("\n生成全面对比图表...")

    # 转换结果格式为 compare_sam2_vs_bndl 期望的格式
    # 格式: {dataset: (J&F, J, F)}
    sam_results = {}
    uctta_results = {}
    bndl_results = {}

    if "SAM" in all_results:
        for dataset, metrics in all_results["SAM"].items():
            sam_results[dataset] = (metrics["J&F"], metrics["J"], metrics["F"])

    if "UCTTA" in all_results:
        for dataset, metrics in all_results["UCTTA"].items():
            uctta_results[dataset] = (metrics["J&F"], metrics["J"], metrics["F"])

    if "BNDL" in all_results:
        for dataset, metrics in all_results["BNDL"].items():
            bndl_results[dataset] = (metrics["J&F"], metrics["J"], metrics["F"])

    # 获取统计数据
    uctta_statistics = all_statistics.get("UCTTA", {})
    bndl_statistics = all_statistics.get("BNDL", {})

    # 创建comprehensive对比图（复用原有函数）
    # 注意：原函数需要至少有SAM和BNDL的结果
    if sam_results and bndl_results:
        try:
            create_comprehensive_comparison_plots(
                sam2_results=sam_results,
                bndl_results=bndl_results,
                bndl_statistics=bndl_statistics,
                output_path=output_path,
                uctta_results=uctta_results if uctta_results else None,
                uctta_statistics=uctta_statistics if uctta_statistics else None,
            )
            print("✓ 综合对比图已生成 (包含SAM+BNDL+UCTTA)")
        except Exception as e:
            print(f"警告: 无法生成综合对比图: {e}")
            import traceback

            traceback.print_exc()
    elif sam_results or bndl_results or uctta_results:
        print(f"⚠ 跳过综合对比图: 需要至少有SAM和BNDL的结果")
        print(f"  当前有: SAM={'✓' if sam_results else '✗'}, UCTTA={'✓' if uctta_results else '✗'}, BNDL={'✓' if bndl_results else '✗'}")

    # 创建UA shift分析图（复用原有函数）
    if bndl_statistics and bndl_results and sam_results:
        try:
            create_ua_shift_analysis_plots(
                bndl_statistics=bndl_statistics,
                sam2_results=sam_results,
                bndl_results=bndl_results,
                output_path=output_path,
                uctta_statistics=uctta_statistics if uctta_statistics else None,
                uctta_results=uctta_results if uctta_results else None,
                source_domain="MOSE",  # 可以根据需要调整
            )
            print("✓ UA shift分析图已生成")
        except Exception as e:
            print(f"警告: 无法生成UA shift分析图: {e}")
            import traceback

            traceback.print_exc()
    elif bndl_statistics:
        print(f"⚠ 跳过UA shift分析: 需要SAM和BNDL的结果以及BNDL统计数据")


def create_merged_comparison(
    all_results: Dict[str, Dict],
    times: Dict[str, float],
    datasets: List[str],
    output_path: Path,
):
    """创建合并的对比表格"""

    print("\n" + "=" * 80)
    print("PARALLEL COMPARISON RESULTS")
    print("=" * 80 + "\n")

    # 创建DataFrame
    data = []

    for dataset in datasets:
        row = {"Dataset": dataset}

        for method in ["SAM", "UCTTA", "BNDL"]:
            if method in all_results and dataset in all_results[method]:
                metrics = all_results[method][dataset]
                row[f"{method}_J&F"] = f"{metrics['J&F']:.2f}"
                row[f"{method}_J"] = f"{metrics['J']:.2f}"
                row[f"{method}_F"] = f"{metrics['F']:.2f}"
            else:
                row[f"{method}_J&F"] = "N/A"
                row[f"{method}_J"] = "N/A"
                row[f"{method}_F"] = "N/A"

        data.append(row)

    # 添加运行时间
    time_row = {"Dataset": "Runtime (s)"}
    for method in ["SAM", "UCTTA", "BNDL"]:
        if method in times:
            time_row[f"{method}_J&F"] = f"{times[method]:.1f}"
            time_row[f"{method}_J"] = "-"
            time_row[f"{method}_F"] = "-"
        else:
            time_row[f"{method}_J&F"] = "N/A"
            time_row[f"{method}_J"] = "-"
            time_row[f"{method}_F"] = "-"
    data.append(time_row)

    df = pd.DataFrame(data)

    # 保存CSV
    csv_file = output_path / "parallel_comparison_merged.csv"
    df.to_csv(csv_file, index=False)

    # 打印表格
    print(df.to_string(index=False))
    print(f"\n✓ Results saved to: {csv_file}")

    # 创建详细报告
    create_detailed_report(all_results, times, datasets, output_path)

    return df


def create_detailed_report(
    all_results: Dict[str, Dict],
    times: Dict[str, float],
    datasets: List[str],
    output_path: Path,
):
    """创建详细的文本报告"""

    report_file = output_path / "parallel_comparison_summary.txt"

    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PARALLEL COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Datasets: {', '.join(datasets)}\n")
        f.write(f"Methods: {', '.join(all_results.keys())}\n\n")

        # 运行时间
        f.write("Runtime Analysis:\n")
        f.write("-" * 80 + "\n")
        for method, elapsed in sorted(times.items(), key=lambda x: x[1]):
            f.write(f"  {method:10s}: {elapsed:8.1f}s\n")

        total_seq = sum(times.values())
        total_par = max(times.values()) if times else 0
        speedup = total_seq / total_par if total_par > 0 else 0

        f.write(f"\n  Sequential (estimated): {total_seq:.1f}s\n")
        f.write(f"  Parallel (actual):      {total_par:.1f}s\n")
        f.write(f"  Speedup:                {speedup:.2f}x\n\n")

        # 每个数据集的详细结果
        f.write("=" * 80 + "\n")
        f.write("DETAILED RESULTS BY DATASET\n")
        f.write("=" * 80 + "\n\n")

        for dataset in datasets:
            f.write(f"{dataset}:\n")
            f.write("-" * 80 + "\n")

            best_jf = 0
            best_method = None

            for method in ["SAM", "UCTTA", "BNDL"]:
                if method in all_results and dataset in all_results[method]:
                    metrics = all_results[method][dataset]
                    jf = metrics["J&F"]
                    j = metrics["J"]
                    f_score = metrics["F"]

                    f.write(f"  {method:10s}: J&F={jf:6.2f}  J={j:6.2f}  F={f_score:6.2f}\n")

                    if jf > best_jf:
                        best_jf = jf
                        best_method = method

            if best_method:
                f.write(f"  → Best: {best_method} (J&F={best_jf:.2f})\n")
            f.write("\n")

        # 总体统计
        f.write("=" * 80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        for method in ["SAM", "UCTTA", "BNDL"]:
            if method in all_results and all_results[method]:
                jf_scores = [m["J&F"] for m in all_results[method].values()]

                if jf_scores:
                    f.write(f"{method}:\n")
                    f.write(f"  Average J&F: {sum(jf_scores) / len(jf_scores):.2f}\n")
                    f.write(f"  Min J&F:     {min(jf_scores):.2f}\n")
                    f.write(f"  Max J&F:     {max(jf_scores):.2f}\n\n")

    print(f"✓ Detailed report saved to: {report_file}\n")


def main():
    parser = argparse.ArgumentParser(description="并行运行compare_sam2_vs_bndl的三个方法在不同GPU上")

    # Dataset selection
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=list(DATASET_CONFIGS.keys()),
        help="Datasets to evaluate (default: all)",
    )

    # GPU配置
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0, 1, 2], help="分配给SAM, UCTTA, BNDL的GPU ID (默认: 0 1 2)")

    # SAM-2配置
    parser.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--sam2_checkpoint", default="/home/hongyou/dev/ada_samp/sam2/checkpoints/sam2.1_hiera_base_plus.pt")

    # BNDL配置
    parser.add_argument("--bndl_cfg", default="configs/sam2.1/sam2.1_hiera_b+_bndl.yaml")
    parser.add_argument("--bndl_checkpoint", default="/home/hongyou/dev/ada_samp/logs/sam2/sam2_bndl_011_02/checkpoints/checkpoint.pt")

    # 评估参数
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument("--click_protocol", default="3click", choices=["1click", "3click", "5click"])
    parser.add_argument("--max_objects", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--first_frame_only", action="store_true", help="只评估第一帧（快速模式）")
    parser.add_argument("--video_limit", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    # UCTTA参数
    parser.add_argument("--uctta_steps", type=int, default=2)
    parser.add_argument("--uctta_lr", type=float, default=3e-4)
    parser.add_argument("--uctta_enable_bn", action="store_true", default=True)
    parser.add_argument("--uctta_fisher_reg", action="store_true", default=True)

    # BNDL参数
    parser.add_argument("--collect_bndl_stats", action="store_true", default=True)

    # 输出
    parser.add_argument("--output_path", type=Path, default=Path("./outputs/parallel_compare"))

    args = parser.parse_args()

    # 验证GPU数量
    if len(args.gpu_ids) < 3:
        print("警告: GPU数量少于3个，将复用GPU")
        while len(args.gpu_ids) < 3:
            args.gpu_ids.append(args.gpu_ids[-1])

    # 打印配置
    print("=" * 80)
    print("并行对比评估")
    print("=" * 80)
    print(f"数据集: {', '.join(args.datasets)}")
    print(f"GPU分配: SAM→GPU{args.gpu_ids[0]}, UCTTA→GPU{args.gpu_ids[1]}, BNDL→GPU{args.gpu_ids[2]}")
    print(f"输出目录: {args.output_path}")
    print(f"模式: {'仅第一帧' if args.first_frame_only else '完整视频'}")
    print("=" * 80 + "\n")

    # 创建输出目录
    args.output_path.mkdir(parents=True, exist_ok=True)

    # 准备并行任务
    tasks = [
        ("SAM", args.gpu_ids[0], args.output_path, args),
        ("UCTTA", args.gpu_ids[1], args.output_path, args),
        ("BNDL", args.gpu_ids[2], args.output_path, args),
    ]

    # 并行运行
    print("启动并行执行...\n")
    start_time = time.time()

    with multiprocessing.Pool(processes=3) as pool:
        results = pool.starmap(run_method, tasks)

    total_time = time.time() - start_time

    # 处理结果
    print("\n" + "=" * 80)
    print("并行执行完成")
    print("=" * 80 + "\n")

    times = {}
    all_results = {}
    all_statistics = {}

    for method, return_code, elapsed, output_dir in results:
        times[method] = elapsed

        if return_code == 0:
            # 解析结果
            method_results = parse_results_from_output(output_dir, method)
            if method_results:
                all_results[method] = method_results

            # 解析统计数据（用于UA分析）
            method_statistics = parse_statistics_from_output(output_dir, method)
            if method_statistics:
                all_statistics[method] = method_statistics

            print(f"✓ {method:10s}: 成功 ({elapsed:.1f}s)")
        else:
            print(f"✗ {method:10s}: 失败 (代码 {return_code}, {elapsed:.1f}s)")

    print(f"\n总并行时间: {total_time:.1f}s")
    if times:
        est_seq = sum(times.values())
        print(f"预估顺序时间: {est_seq:.1f}s")
        print(f"加速比: {est_seq / total_time:.2f}x\n")

    # 创建合并的对比表格（只要有任何成功的结果）
    if all_results:
        # 显示哪些方法成功了
        successful_methods = list(all_results.keys())
        print(f"\n成功的方法: {', '.join(successful_methods)}")

        create_merged_comparison(all_results, times, args.datasets, args.output_path)

        # 调用复用的可视化函数（即使只有部分方法成功）
        create_parallel_comparison_wrapper(all_results, all_statistics, times, args.datasets, args.output_path)
    else:
        print("\n警告: 没有成功的结果可以对比")

    print(f"\n所有输出保存到: {args.output_path}")


if __name__ == "__main__":
    main()
