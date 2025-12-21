#!/usr/bin/env python
"""
Checkpoint Sweep Tool - Evaluate multiple checkpoints across datasets
基于 sam2/scripts/parallel_compare.py 修改，用于扫描指定目录下的所有 checkpiont 的 ZS 性能。

Usage:
    python scripts/sweep_checkpoints.py \
        --checkpoints_dir /path/to/checkpoints \
        --datasets GTEA MOSE_val TrashCan ... \
        --gpu_ids 0 1 2 3 4 5 6 7 \
        --video_limit 500
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any

import pandas as pd

# Rich imports for beautiful progress display
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Rich module not available. Install with: pip install rich")
    print("    Falling back to basic progress display.\n")

# Add the script directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))

from dataset_configs import DEFAULT_DATASETS

# Color codes
RESET_COLOR = "\033[0m"
COLOR_CYAN = "\033[96m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"

class ProgressMonitor:
    """Rich进度监控器 - 实时显示任务执行状态（支持子任务进度）"""
    
    def __init__(self, total_tasks: int, gpu_ids: list[int]):
        """初始化进度监控器
        
        Args:
            total_tasks: 总任务数
            gpu_ids: GPU ID列表
        """
        self.total_tasks = total_tasks
        self.gpu_ids = gpu_ids
        
        # 任务状态跟踪
        self.completed = 0
        self.running_tasks = {}  # {gpu_id: (dataset, checkpoint_name)}
        self.task_status = {}  # {(dataset, checkpoint_name): status}
        self.task_progress_bars = {}  # {task_id: progress_task_id} 子任务进度条
        self.lock = Lock()
        
        # Rich组件（Progress 和 Console 必须绑定）
        if RICH_AVAILABLE:
            self.console = Console()
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,  # 绑定 Console，确保输出同步
            )
            self.main_task = self.progress.add_task(
                "[cyan]Overall Progress", total=total_tasks
            )
        else:
            self.console = None
            self.progress = None
    
    def start_task(self, gpu_id: int, dataset: str, checkpoint_name: str, total_videos: int = 0):
        """标记任务开始并创建子进度条"""
        with self.lock:
            self.running_tasks[gpu_id] = (dataset, checkpoint_name)
            self.task_status[(dataset, checkpoint_name)] = "running"
            
            # 为此任务创建子进度条
            if RICH_AVAILABLE and total_videos > 0:
                task_id = f"{checkpoint_name}@{dataset}"
                sub_task = self.progress.add_task(
                    f"[green]GPU{gpu_id} {task_id}",
                    total=total_videos
                )
                self.task_progress_bars[task_id] = sub_task
    
    def update_task_progress(self, dataset: str, checkpoint_name: str, current: int, total: int):
        """更新任务的视频处理进度"""
        task_id = f"{checkpoint_name}@{dataset}"
        with self.lock:
            if RICH_AVAILABLE and task_id in self.task_progress_bars:
                sub_task = self.task_progress_bars[task_id]
                self.progress.update(sub_task, completed=current)
    
    def complete_task(self, gpu_id: int, dataset: str, checkpoint_name: str, success: bool):
        """标记任务完成并移除子进度条"""
        with self.lock:
            if gpu_id in self.running_tasks:
                del self.running_tasks[gpu_id]
            self.task_status[(dataset, checkpoint_name)] = "completed" if success else "failed"
            self.completed += 1
            
            # 移除子进度条
            task_id = f"{checkpoint_name}@{dataset}"
            if RICH_AVAILABLE and task_id in self.task_progress_bars:
                sub_task = self.task_progress_bars[task_id]
                self.progress.update(sub_task, visible=False)
                del self.task_progress_bars[task_id]
            
            if RICH_AVAILABLE:
                self.progress.update(self.main_task, advance=1)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if RICH_AVAILABLE and hasattr(self, 'progress'):
            self.progress.stop()
        return False


def build_task_command(
    dataset: str,
    checkpoint_path: Path,
    gpu_id: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """构建单个任务的命令"""
    cmd = [
        sys.executable,
        "sam2/scripts/zs.py",
        "--datasets", dataset,
        "--output_path", str(output_dir),
        "--device", "cuda",
        "--run_bndl_aue",
        "--bndl_aue_checkpoint", str(checkpoint_path),
        "--score_thresh", str(args.score_thresh),
        "--click_protocol", args.click_protocol,
        "--max_objects", str(args.max_objects),
        "--seed", str(args.seed),
    ]

    # Config: Use relative path (same as train_and_zs.sh approach)
    # Hydra will search in EXPERIMENT_CONFIG_DIR if it's set
    if args.config:
        config_path = args.config
        # Sanitize config path for Hydra: strip 'sam2/sam2/' prefix if present
        # Hydra expects path relative to search path (which is .../sam2/sam2)
        if config_path.startswith("sam2/sam2/"):
            config_path = config_path.replace("sam2/sam2/", "", 1)
        cmd.extend(["--bndl_aue_cfg", config_path])
    
    # Optional args
    if not args.first_frame_only:
        cmd.append("--process_full_video")
    
    if args.video_limit:
        cmd.extend(["--video_limit", str(args.video_limit)])
    
    if args.num_workers:
        cmd.extend(["--num_workers", str(args.num_workers)])
        
    if args.collect_bndl_stats:
        cmd.append("--collect_bndl_stats")

    if args.aue_version:
        cmd.extend(["--aue_version", args.aue_version])

    return cmd


def run_task(
    task: tuple[str, Path],
    gpu_id: int,
    output_base: Path,
    args: argparse.Namespace,
    progress_monitor: ProgressMonitor | None = None,
) -> tuple[str, str, int, float, Path]:
    """在指定GPU上运行单个(dataset, checkpoint)任务"""
    dataset_name, checkpoint_path = task
    checkpoint_name = checkpoint_path.stem  # e.g., "checkpoint_10"
    
    # Output structure: output_base / checkpoint_name / dataset
    # e.g. sweep_results/checkpoint_10/GTEA
    ckpt_output_dir = output_base / checkpoint_name
    output_dir = ckpt_output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = build_task_command(dataset_name, checkpoint_path, gpu_id, output_dir, args)
    
    # Environment setup
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONUNBUFFERED"] = "1"

    # Limit threads to prevent CPU oversubscription which causes extreme slowdowns
    # when running multiple PyTorch processes in parallel
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    
    task_id = f"{checkpoint_name}@{dataset_name}"
    
    # Infer experiment directories from checkpoint path
    # This ensures we use the exact code and config from training time
    # Checkpoint: .../sam2_bndl_aue_XXX/log/checkpoints/checkpoint.pt
    # Experiment Root: .../sam2_bndl_aue_XXX/
    # Config Base: .../sam2_bndl_aue_XXX/sam2/sam2
    # Source Code: .../sam2_bndl_aue_XXX/src
    if checkpoint_path.parent.name == "checkpoints" and checkpoint_path.parent.parent.name == "log":
        experiment_root = checkpoint_path.parent.parent.parent
        
        # Set EXPERIMENT_CONFIG_DIR for Hydra
        experiment_config_dir = experiment_root / "sam2" / "sam2"
        if experiment_config_dir.exists():
            env["EXPERIMENT_CONFIG_DIR"] = str(experiment_config_dir.resolve())
            print(f"[{task_id}] Set EXPERIMENT_CONFIG_DIR: {experiment_config_dir}")
        else:
            print(f"[{task_id}] ⚠️  Warning: Experiment config directory not found: {experiment_config_dir}")
        
        # Set PYTHONPATH to use experiment source code (same as train_and_zs.sh)
        # This ensures we use the exact code version from training time
        experiment_src = experiment_root / "src"
        if experiment_src.exists():
            # Prepend experiment src paths to PYTHONPATH (highest priority)
            experiment_sam2 = experiment_src / "sam2"
            experiment_bndl = experiment_src / "BNDL"
            
            pythonpath_parts = [str(experiment_src)]
            if experiment_sam2.exists():
                pythonpath_parts.append(str(experiment_sam2))
            if experiment_bndl.exists():
                pythonpath_parts.append(str(experiment_bndl))
            
            # Append original PYTHONPATH
            if "PYTHONPATH" in env:
                pythonpath_parts.append(env["PYTHONPATH"])
            
            env["PYTHONPATH"] = ":".join(pythonpath_parts)
            print(f"[{task_id}] Set PYTHONPATH to use experiment source code from: {experiment_src}")
        else:
            print(f"[{task_id}] ⚠️  Warning: Experiment src directory not found: {experiment_src}, using current code")
    else:
        print(f"[{task_id}] ⚠️  Warning: Could not infer experiment directory from checkpoint path")



    log_file = output_dir / f"{dataset_name.lower()}_{checkpoint_name}_run.log"
    
    if progress_monitor and RICH_AVAILABLE:
        progress_monitor.console.log(f"🚀 [{task_id}] Starting on GPU {gpu_id}")
    else:
        print(f"\n{COLOR_CYAN}{'=' * 80}{RESET_COLOR}")
        print(f"{COLOR_CYAN}[{task_id}] Starting on GPU {gpu_id}{RESET_COLOR}")
        print(f"{COLOR_CYAN}{'=' * 80}{RESET_COLOR}")

    progress_pattern = re.compile(r'Progress:\s*(\d+)/(\d+)\s*\([\d.]+%\)')
    total_videos_pattern = re.compile(r'inference on (\d+) videos')
    
    start_time = time.time()
    task_started = False
    process = None
    project_root = Path(__file__).parent.parent.parent
    
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                errors="replace",
                cwd=str(project_root),
            )
            
            try:
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                    line_stripped = line.strip()
                    
                    if not task_started and progress_monitor:
                        total_match = total_videos_pattern.search(line_stripped)
                        if total_match:
                            total_videos = int(total_match.group(1))
                            progress_monitor.start_task(gpu_id, dataset_name, checkpoint_name, total_videos)
                            task_started = True
                        else:
                            # Early progress catch
                            progress_match_early = progress_pattern.search(line_stripped)
                            if progress_match_early:
                                total_videos = int(progress_match_early.group(2))
                                if total_videos > 0:
                                    progress_monitor.start_task(gpu_id, dataset_name, checkpoint_name, total_videos)
                                    task_started = True
                                    progress_monitor.update_task_progress(dataset_name, checkpoint_name, int(progress_match_early.group(1)), total_videos)

                    if progress_monitor and task_started:
                        progress_match = progress_pattern.search(line_stripped)
                        if progress_match:
                            progress_monitor.update_task_progress(dataset_name, checkpoint_name, int(progress_match.group(1)), int(progress_match.group(2)))
                    
                    if not progress_monitor and line_stripped and any(k in line_stripped for k in ['Processing', 'Evaluating', 'video', 'Progress', 'Completed']):
                        print(f"[{task_id}] {line_stripped}")
                        
            except (BrokenPipeError, OSError):
                f.write("\nstdout closed unexpectedly\n")
            
            returncode = process.wait()
            f.write(f"\nProcess finished with return code {returncode}\n")
        
        elapsed = time.time() - start_time
        
        if progress_monitor:
            progress_monitor.complete_task(gpu_id, dataset_name, checkpoint_name, returncode == 0)
        
        if returncode == 0:
            if progress_monitor and RICH_AVAILABLE:
                progress_monitor.console.log(f"✓ [{task_id}] Completed in {elapsed:.1f}s")
            else:
                print(f"[{task_id}] ✓ Completed in {elapsed:.1f}s")
        else:
             if progress_monitor and RICH_AVAILABLE:
                progress_monitor.console.log(f"[red]✗ [{task_id}] Failed (code {returncode})[/red]")
             else:
                print(f"[{task_id}] ✗ Failed (code {returncode})")
        
        return dataset_name, checkpoint_name, returncode, elapsed, output_dir

    except Exception as e:
        if process and process.poll() is None:
            try:
                process.terminate()
            except:
                process.kill()
        
        elapsed = time.time() - start_time
        if progress_monitor:
            progress_monitor.complete_task(gpu_id, dataset_name, checkpoint_name, False)
        
        if progress_monitor and RICH_AVAILABLE:
            progress_monitor.console.log(f"[red]✗ [{task_id}] Exception: {e}[/red]")
        else:
            print(f"[{task_id}] ✗ Exception: {e}")
            
        return dataset_name, checkpoint_name, -1, elapsed, output_dir


def check_task_completed(output_dir: Path, dataset: str) -> bool:
    """Simple check if task is completed"""
    json_files = list(output_dir.glob("**/detailed_results.json"))
    if not json_files:
        return False
    try:
        with open(json_files[0]) as f:
            data = json.load(f)
        # BNDL_AUE results key
        if "bndl_aue_results" in data and dataset in data["bndl_aue_results"]:
             return True
    except:
        return False
    return False


def schedule_tasks_on_gpus(
    tasks: list[tuple[str, Path]],
    gpu_ids: list[int],
    output_base: Path,
    args: argparse.Namespace,
    reuse_cached: bool = False,
):
    """Schedule tasks"""
    print(f"\nScheduling {len(tasks)} tasks on {len(gpu_ids)} GPUs...")
    
    tasks_to_run = []
    skipped_tasks = []
    
    if reuse_cached:
        for task in tasks:
            dataset, ckpt_path = task
            ckpt_name = ckpt_path.stem
            output_dir = output_base / ckpt_name / dataset
            if check_task_completed(output_dir, dataset):
                skipped_tasks.append(task)
            else:
                tasks_to_run.append(task)
        print(f"Skipped {len(skipped_tasks)} cached tasks, {len(tasks_to_run)} to run.")
    else:
        tasks_to_run = tasks

    if not tasks_to_run:
        print("All tasks completed.")
        return

    # Create shared queue
    task_queue = Queue()
    for task in tasks_to_run:
        task_queue.put(task)
    
    with ProgressMonitor(len(tasks_to_run), gpu_ids) as progress:
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            if RICH_AVAILABLE:
                 progress.progress.start()
            
            futures = []
            for gpu_id in gpu_ids:
                futures.append(executor.submit(worker, gpu_id, task_queue, output_base, args, progress))
            
            for future in as_completed(futures):
                future.result()


def worker(gpu_id, queue, output_base, args, progress_monitor):
    """Worker thread"""
    while not queue.empty():
        try:
            task = queue.get_nowait()
        except:
            break
            
        run_task(task, gpu_id, output_base, args, progress_monitor)
        queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Sweep SAM2 checkpoints")
    parser.add_argument("--checkpoints_dir", type=str, required=True, help="Directory containing .pt checkpoints")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Datasets to evaluate")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0,1,2,3,4,5,6,7], help="GPUs to use")
    parser.add_argument("--output_base", type=str, default="outputs/sweep_results", help="Base output directory")
    parser.add_argument("--video_limit", type=int, default=1000, help="Max videos per dataset (default: 1000)")
    parser.add_argument("--config", type=str, default="configs/sam2.1/sam2.1_hiera_b+_bndl_aue.yaml", help="Path to config yaml")
    parser.add_argument("--reuse_cached", action="store_true", help="Skip completed tasks")
    
    # Eval args
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument("--click_protocol", type=str, default="3click", choices=["1click", "3click", "5click"])
    parser.add_argument("--max_objects", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--first_frame_only", action="store_true", default=True, help="Evaluate only the first frame (default: True, use --no-first_frame_only to process full video)")
    parser.add_argument("--no-first_frame_only", dest="first_frame_only", action="store_false", help="Process full video instead of just first frame")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for evaluation (default: 4)")
    parser.add_argument("--collect_bndl_stats", action="store_true")
    parser.add_argument("--aue_version", type=str, default=None)

    args = parser.parse_args()
    
    checkpoints_dir = Path(args.checkpoints_dir)
    if not checkpoints_dir.exists():
        print(f"Error: Directory {checkpoints_dir} does not exist.")
        return

    # Find checkpoints
    checkpoints = sorted(list(checkpoints_dir.glob("*.pt")), key=lambda p: p.stat().st_mtime)
    print(f"Found {len(checkpoints)} checkpoints in {checkpoints_dir}")
    
    # Create tasks
    tasks = []
    for ckpt in checkpoints:
        for dataset in args.datasets:
            tasks.append((dataset, ckpt))
    
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    schedule_tasks_on_gpus(tasks, args.gpu_ids, output_base, args, args.reuse_cached)
    
    # ===== 汇总所有 checkpoint 的结果 =====
    print(f"\n{'=' * 80}")
    print("📊 Summarizing results across all checkpoints...")
    print(f"{'=' * 80}\n")
    
    try:
        summary_results = collect_and_summarize_results(output_base, checkpoints, args.datasets)
        if summary_results:
            print_summary_table(summary_results)
            save_summary_csv(summary_results, output_base)
        else:
            print("⚠️  No results to summarize (tasks may still be running or failed)")
    except Exception as e:
        print(f"⚠️  Failed to generate summary: {e}")
        import traceback
        traceback.print_exc()


def collect_and_summarize_results(output_base: Path, checkpoints: list[Path], datasets: list[str]):
    """收集所有 checkpoint 和数据集的结果"""
    from collections import defaultdict
    import json
    
    results = defaultdict(dict)  # {checkpoint: {dataset: {jf, j, f}}}
    
    for ckpt in checkpoints:
        ckpt_name = ckpt.stem  # e.g., "checkpoint_10"
        
        for dataset in datasets:
            # 构建结果文件路径
            results_file = output_base / ckpt_name / dataset / "comparison_plots" / "detailed_results.json"
            
            if not results_file.exists():
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
    import pandas as pd
    
    # 提取所有 checkpoint 和数据集
    checkpoints = sorted(results.keys(), key=lambda x: int(x.split('_')[-1]) if '_' in x else 0)
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
                    row[ckpt] = f"{results[ckpt][dataset][metric]:.2f}"
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
    import pandas as pd
    
    checkpoints = sorted(results.keys(), key=lambda x: int(x.split('_')[-1]) if '_' in x else 0)
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


if __name__ == "__main__":
    main()
