#!/usr/bin/env python
"""
智能GPU任务调度器 - 动态并行评估
Dynamic GPU Task Scheduler for Zero-Shot Evaluation

核心特性:
1. 任务队列: 生成所有(数据集, 方法)组合
2. 🚀 动态调度: GPU完成任务后立即从共享队列取下一个任务
3. 🎯 智能负载均衡: 避免GPU空闲，最大化利用率（接近100%）
4. 📊 实时进度监控: 显示任务完成状态

总任务数 = len(datasets) × len(methods)
例如: 3个数据集 × 5个方法 = 15个任务，8个GPU并行执行

调度策略：
- 旧版（静态分配）：任务预先分配到各GPU队列，导致尾部GPU空闲
- 新版（动态队列）：共享任务队列，GPU完成后立即取新任务，GPU利用率接近100%

用法:
    # 使用8个GPU并行评估
    python scripts/parallel_compare.py \
        --datasets GTEA MOSE_val TrashCan \
        --methods SAM UCTTA BNDL_AUE BNDL UR-ERN \
        --gpu_ids 0 1 2 3 4 5 6 7
    
    # 只运行部分方法
    python scripts/parallel_compare.py \
        --datasets GTEA MOSE_val \
        --methods UCTTA BNDL_AUE \
        --gpu_ids 0 1
    
    # 🔥 智能续跑模式（中途中断后继续）
    python scripts/parallel_compare.py \
        --datasets GTEA MOSE_val TrashCan \
        --methods SAM UCTTA BNDL_AUE BNDL UR-ERN \
        --gpu_ids 0 1 2 3 4 5 6 7 \
        --reuse_cached
    # 自动检测已完成的任务，只运行未完成的部分
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

# 本地模块导入
sys.path.insert(0, str(Path(__file__).parent))

from dataset_configs import DATASET_CONFIGS, DEFAULT_DATASETS
from zs import create_comprehensive_comparison_plots, create_ua_shift_analysis_plots


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
    "BNDL_AUE": {
        "flags": ["--run_bndl_aue"],
        "color": "\033[93m",  # Yellow
        "output_suffix": "bndl_aue",
    },
    "BNDL": {
        "flags": ["--run_bndl"],
        "color": "\033[96m",  # Cyan
        "output_suffix": "bndl",
    },
    "UR-ERN": {
        "flags": ["--run_ur_ern"],
        "color": "\033[95m",  # Magenta
        "output_suffix": "ur_ern",
    },
}

ALL_METHODS = ["SAM", "UCTTA", "BNDL_AUE", "BNDL", "UR-ERN"]
RESET_COLOR = "\033[0m"

# 方法名到JSON键的映射（避免重复定义）
METHOD_RESULT_KEYS = {
    "SAM": "sam2_results",
    "UCTTA": "uctta_results",
    "BNDL_AUE": "bndl_aue_results",
    "BNDL": "bndl_results",
    "UR-ERN": "ur_ern_results",
}

METHOD_STATS_KEYS = {
    "SAM": "sam2_statistics",
    "UCTTA": "uctta_statistics",
    "BNDL_AUE": "bndl_aue_statistics",
    "BNDL": "bndl_statistics",
    "UR-ERN": "ur_ern_statistics",
}


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
        self.running_tasks = {}  # {gpu_id: (dataset, method)}
        self.task_status = {}  # {(dataset, method): status}
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
    
    def start_task(self, gpu_id: int, dataset: str, method: str, total_videos: int = 0):
        """标记任务开始并创建子进度条"""
        with self.lock:
            self.running_tasks[gpu_id] = (dataset, method)
            self.task_status[(dataset, method)] = "running"
            
            # 为此任务创建子进度条
            if RICH_AVAILABLE and total_videos > 0:
                task_id = f"{method}@{dataset}"
                color_map = {
                    "SAM": "blue",
                    "UCTTA": "green", 
                    "BNDL_AUE": "yellow",
                    "BNDL": "cyan",
                    "UR-ERN": "magenta",
                }
                color = color_map.get(method, "white")
                sub_task = self.progress.add_task(
                    f"[{color}]GPU{gpu_id} {task_id}",
                    total=total_videos
                )
                self.task_progress_bars[task_id] = sub_task
    
    def update_task_progress(self, dataset: str, method: str, current: int, total: int):
        """更新任务的视频处理进度"""
        task_id = f"{method}@{dataset}"
        with self.lock:
            if RICH_AVAILABLE and task_id in self.task_progress_bars:
                sub_task = self.task_progress_bars[task_id]
                self.progress.update(sub_task, completed=current)
    
    def complete_task(self, gpu_id: int, dataset: str, method: str, success: bool):
        """标记任务完成并移除子进度条"""
        with self.lock:
            if gpu_id in self.running_tasks:
                del self.running_tasks[gpu_id]
            self.task_status[(dataset, method)] = "completed" if success else "failed"
            self.completed += 1
            
            # 移除子进度条
            task_id = f"{method}@{dataset}"
            if RICH_AVAILABLE and task_id in self.task_progress_bars:
                sub_task = self.task_progress_bars[task_id]
                self.progress.update(sub_task, visible=False)
                del self.task_progress_bars[task_id]
            
            if RICH_AVAILABLE:
                self.progress.update(self.main_task, advance=1)
    
    def generate_status_table(self) -> Table:
        """生成GPU状态表格"""
        table = Table(title="GPU Task Status", show_header=True, header_style="bold magenta")
        table.add_column("GPU", style="cyan", width=6)
        table.add_column("Status", width=12)
        table.add_column("Current Task", width=40)
        
        for gpu_id in self.gpu_ids:
            if gpu_id in self.running_tasks:
                dataset, method = self.running_tasks[gpu_id]
                status = Text("🔄 Running", style="yellow")
                task_desc = f"{method} @ {dataset}"
            else:
                status = Text("💤 Idle", style="dim")
                task_desc = "-"
            
            table.add_row(f"GPU {gpu_id}", status, task_desc)
        
        return table
    
    def generate_summary_panel(self) -> Panel:
        """生成汇总面板"""
        completed_count = self.completed
        running_count = len(self.running_tasks)
        pending_count = self.total_tasks - completed_count - running_count
        
        summary_text = f"""
[bold cyan]Progress Summary[/bold cyan]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Completed: [green]{completed_count}[/green] / {self.total_tasks}
🔄 Running:   [yellow]{running_count}[/yellow]
⏳ Pending:   [dim]{pending_count}[/dim]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Progress: {completed_count / self.total_tasks * 100:.1f}%
        """
        
        return Panel(summary_text, title="📊 Task Overview", border_style="blue")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if RICH_AVAILABLE and hasattr(self, 'progress'):
            self.progress.stop()
        return False


def build_task_command(
    dataset: str,
    method: str,
    gpu_id: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """构建单个(数据集, 方法)任务的命令
    
    Args:
        dataset: 数据集名称
        method: 方法名称
        gpu_id: GPU ID
        output_dir: 输出目录
        args: 全局参数
    
    Returns:
        命令列表
    """
    cmd = [
        sys.executable,
        "sam2/scripts/zs.py",
        "--datasets",
        dataset,  # 单个数据集
        "--output_path",
        str(output_dir),
        "--device",
        "cuda",
    ]
    
    # 添加方法特定的flags
    cmd.extend(METHOD_CONFIGS[method]["flags"])
    
    # SAM-2配置 (所有方法都需要)
    cmd.extend(["--sam2_cfg", args.sam2_cfg])
    cmd.extend(["--sam2_checkpoint", args.sam2_checkpoint])
    
    # BNDL+AUE配置
    if method == "BNDL_AUE":
        cmd.extend(["--bndl_aue_cfg", args.bndl_aue_cfg])
        cmd.extend(["--bndl_aue_checkpoint", args.bndl_aue_checkpoint])
    
    # BNDL (pure)配置
    if method == "BNDL":
        cmd.extend(["--bndl_cfg", args.bndl_cfg])
        cmd.extend(["--bndl_checkpoint", args.bndl_checkpoint])
    
    # UR-ERN配置
    if method == "UR-ERN":
        cmd.extend(["--ur_ern_cfg", args.ur_ern_cfg])
        cmd.extend(["--ur_ern_checkpoint", args.ur_ern_checkpoint])
    
    # 评估参数
    cmd.extend([
        "--score_thresh", str(args.score_thresh),
        "--click_protocol", args.click_protocol,
        "--max_objects", str(args.max_objects),
        "--seed", str(args.seed),
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
            "--uctta_steps", str(args.uctta_steps),
            "--uctta_lr", str(args.uctta_lr),
        ])
        if args.uctta_enable_bn:
            cmd.append("--uctta_enable_bn")
        if args.uctta_fisher_reg:
            cmd.append("--uctta_fisher_reg")
    
    # BNDL/BNDL_AUE/UR-ERN特定参数
    if method in ["BNDL", "BNDL_AUE", "UR-ERN"] and args.collect_bndl_stats:
        cmd.append("--collect_bndl_stats")
    
    return cmd


def run_task(
    task: tuple[str, str],
    gpu_id: int,
    output_base: Path,
    args: argparse.Namespace,
    method_versions: dict[str, str],
    progress_monitor: ProgressMonitor | None = None,
) -> tuple[str, str, int, float, Path]:
    """在指定GPU上运行单个(数据集, 方法)任务（支持Rich子进度条）
    
    Args:
        task: (dataset_name, method_name) 元组
        gpu_id: 分配的GPU ID
        output_base: 输出基础路径
        args: 命令行参数
        method_versions: 方法版本映射
        progress_monitor: 进度监控器（可选）
    
    Returns:
        (dataset, method, return_code, elapsed_time, output_dir)
    """
    dataset_name, method = task
    version = method_versions[method]
    
    # 按方法组织目录: METHOD_VERSION/DATASET/
    method_dir = output_base / f"{method}_{version}"
    output_dir = method_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建命令
    cmd = build_task_command(dataset_name, method, gpu_id, output_dir, args)
    
    # 设置环境变量指定GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # 强制子进程的 Python 使用无缓冲输出，确保进度信息实时传递
    env["PYTHONUNBUFFERED"] = "1"
    
    color = METHOD_CONFIGS[method]["color"]
    task_id = f"{method}@{dataset_name}"
    
    # 日志文件
    log_file = output_dir / f"{dataset_name.lower()}_{method.lower()}_run.log"
    
    # 使用 Rich Console 避免破坏进度条显示
    # Rich Console.log() 会在进度条上方正确输出，不会错位
    if progress_monitor and RICH_AVAILABLE:
        progress_monitor.console.log(f"🚀 [{task_id}] Starting on GPU {gpu_id}")
    else:
        # 无进度条时使用传统输出
        print(f"\n{color}{'=' * 80}{RESET_COLOR}")
        print(f"{color}[{task_id}] Starting on GPU {gpu_id}{RESET_COLOR}")
        print(f"{color}{'=' * 80}{RESET_COLOR}")
    
    # 正则表达式匹配进度信息: "Progress: 5/28 (17.9%)"
    progress_pattern = re.compile(r'Progress:\s*(\d+)/(\d+)\s*\([\d.]+%\)')
    total_videos_pattern = re.compile(r'inference on (\d+) videos')
    
    # 运行
    start_time = time.time()
    total_videos = 0
    task_started = False
    
    try:
        # 使用Popen实时捕获输出，同时写入日志文件
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )
            
            # 实时读取并显示输出
            for line in process.stdout:
                # 写入日志文件
                f.write(line)
                f.flush()
                
                line_stripped = line.strip()
                
                # 尝试提取总视频数（用于初始化进度条）
                if not task_started and progress_monitor:
                    total_match = total_videos_pattern.search(line_stripped)
                    if total_match:
                        total_videos = int(total_match.group(1))
                        progress_monitor.start_task(gpu_id, dataset_name, method, total_videos)
                        task_started = True
                    else:
                        # 兼容：如果没有打印 "inference on X videos"，
                        # 则从第一条 Progress: a/b 中提取总数并初始化
                        progress_match_early = progress_pattern.search(line_stripped)
                        if progress_match_early:
                            current_early = int(progress_match_early.group(1))
                            total_videos = int(progress_match_early.group(2))
                            if total_videos > 0:
                                progress_monitor.start_task(gpu_id, dataset_name, method, total_videos)
                                task_started = True
                                # 立即更新一次当前进度
                                progress_monitor.update_task_progress(dataset_name, method, current_early, total_videos)
                
                # 尝试解析进度更新
                if progress_monitor and task_started:
                    progress_match = progress_pattern.search(line_stripped)
                    if progress_match:
                        current = int(progress_match.group(1))
                        total = int(progress_match.group(2))
                        progress_monitor.update_task_progress(dataset_name, method, current, total)
                
                # 实时显示关键进度（只在无progress_monitor时，否则进度条已显示）
                if not progress_monitor and line_stripped and any(keyword in line_stripped for keyword in [
                    'Processing', 'Evaluating', 'video', 'Progress', 
                    'Completed', 'Dataset', 'Inference', '✓', '✗', '⚠️'
                ]):
                    print(f"{color}[{task_id}] {line_stripped}{RESET_COLOR}")
            
            # 等待进程完成
            process.wait()
            returncode = process.returncode
        
        elapsed = time.time() - start_time
        
        # 完成任务
        if progress_monitor:
            progress_monitor.complete_task(gpu_id, dataset_name, method, returncode == 0)
        
        # 使用 Rich Console 或传统输出
        if returncode == 0:
            if progress_monitor and RICH_AVAILABLE:
                progress_monitor.console.log(f"✓ [{task_id}] Completed in {elapsed:.1f}s on GPU {gpu_id}")
            else:
                print(f"{color}[{task_id}] ✓ Completed in {elapsed:.1f}s on GPU {gpu_id}{RESET_COLOR}")
        else:
            # 错误信息始终显示（重要）
            if progress_monitor and RICH_AVAILABLE:
                progress_monitor.console.log(f"[red]✗ [{task_id}] Failed (code {returncode}) on GPU {gpu_id}[/red]")
            else:
                print(f"{color}[{task_id}] ✗ Failed (code {returncode}) on GPU {gpu_id}{RESET_COLOR}")
            
            # 检查OOM错误
            with open(log_file) as f:
                log_content = f.read()
                if 'CUDA out of memory' in log_content or 'OutOfMemoryError' in log_content:
                    if progress_monitor and RICH_AVAILABLE:
                        progress_monitor.console.log(f"[yellow]⚠️  [{task_id}] CUDA OOM detected[/yellow]")
                    else:
                        print(f"{color}[{task_id}] ⚠️  CUDA OOM detected{RESET_COLOR}")
        
        return dataset_name, method, returncode, elapsed, output_dir
        
    except Exception as e:
        elapsed = time.time() - start_time
        if progress_monitor and task_started:
            progress_monitor.complete_task(gpu_id, dataset_name, method, False)
        
        # 异常信息使用 Rich Console 或传统输出
        if progress_monitor and RICH_AVAILABLE:
            progress_monitor.console.log(f"[red]✗ [{task_id}] Exception: {e}[/red]")
        else:
            print(f"{color}[{task_id}] ✗ Exception: {e}{RESET_COLOR}")
        return dataset_name, method, -1, elapsed, output_dir


def check_task_completed(output_dir: Path, method: str, dataset: str) -> tuple[bool, str]:
    """智能检测任务是否真正完成
    
    检查标准：
    1. detailed_results.json 文件存在
    2. JSON 中包含对应方法的结果键
    3. JSON 中包含对应数据集的数据
    4. 数据有效（J&F > 0）
    
    Args:
        output_dir: 任务输出目录
        method: 方法名称
        dataset: 数据集名称
    
    Returns:
        (is_completed, reason): 是否完成和原因说明
    """
    # 查找 detailed_results.json
    json_files = list(output_dir.glob("**/detailed_results.json"))
    if not json_files:
        return False, "结果文件不存在"
    
    try:
        with open(json_files[0]) as f:
            data = json.load(f)
        
        # 检查方法对应的结果键是否存在
        result_key = METHOD_RESULT_KEYS.get(method)
        if not result_key:
            return False, f"未知方法: {method}"
        
        if result_key not in data:
            return False, f"结果中缺少 {result_key}"
        
        method_results = data[result_key]
        
        # 检查数据集是否存在且有效
        if dataset not in method_results:
            return False, f"结果中缺少数据集 {dataset}"
        
        dataset_metrics = method_results[dataset]
        if not isinstance(dataset_metrics, dict):
            return False, "数据格式错误"
        
        # 检查 J&F 值是否有效（> 0）
        jf_value = dataset_metrics.get("jf", 0)
        if not isinstance(jf_value, (int, float)):
            return False, "J&F 值类型错误"
        
        if jf_value <= 0:
            return False, f"J&F={jf_value:.2f} 无效"
        
        # 所有检查通过 - 显示结果
        j_value = dataset_metrics.get("j", 0)
        f_value = dataset_metrics.get("f", 0)
        return True, f"J&F={jf_value:.2f}, J={j_value:.2f}, F={f_value:.2f}"
        
    except Exception as e:
        # 任何解析错误都视为未完成
        return False, f"解析错误: {str(e)[:30]}"


def generate_all_tasks(
    datasets: list[str],
    methods: list[str],
) -> list[tuple[str, str]]:
    """生成所有(数据集, 方法)任务组合
    
    Args:
        datasets: 数据集列表
        methods: 方法列表
    
    Returns:
        任务列表 [(dataset, method), ...]
    """
    tasks = []
    for dataset in datasets:
        for method in methods:
            tasks.append((dataset, method))
    return tasks


def schedule_tasks_on_gpus(
    tasks: list[tuple[str, str]],
    gpu_ids: list[int],
    output_base: Path,
    args: argparse.Namespace,
    method_versions: dict[str, str],
    reuse_cached: bool = False,
) -> tuple[dict, float]:
    """智能调度任务到GPU池（动态任务队列策略）
    
    调度策略：
    - 创建共享任务队列，包含所有待执行任务
    - 每个GPU一个专用Worker线程
    - Worker从队列中动态获取任务，完成后立即取下一个
    - 优势：避免静态分配导致的GPU空闲，最大化利用率
    
    Args:
        tasks: 所有任务列表
        gpu_ids: 可用GPU ID列表
        output_base: 输出基础路径
        args: 命令行参数
        method_versions: 方法版本映射
        reuse_cached: 是否复用缓存结果
    
    Returns:
        (结果字典, 实际执行时间)
        - 结果字典: {(dataset, method): (returncode, elapsed, output_dir)}
        - 实际执行时间: 墙上时间（秒）
    """
    print("\n" + "=" * 80)
    print("智能GPU任务调度")
    print("=" * 80)
    print(f"总任务数: {len(tasks)}")
    print(f"可用GPU: {len(gpu_ids)} ({gpu_ids})")
    print(f"并发度: {min(len(tasks), len(gpu_ids))}")
    print("=" * 80 + "\n")
    
    # 智能检查并跳过已完成的任务
    tasks_to_run = []
    skipped_tasks = []
    
    if reuse_cached:
        print("\n" + "=" * 80)
        print("🔍 智能任务续跑检测（自动跳过已完成任务）")
        print("=" * 80)
        
        for task in tasks:
            dataset, method = task
            version = method_versions[method]
            method_dir = output_base / f"{method}_{version}"
            output_dir = method_dir / dataset
            
            # 智能检查任务是否真正完成（版本已包含在路径中）
            is_completed, reason = check_task_completed(output_dir, method, dataset)
            
            # 显示方法版本信息
            method_with_ver = f"{method}_{version}"
            
            if is_completed:
                print(f"   ✓ 跳过: {method_with_ver:18s} @ {dataset:15s} - {reason}")
                skipped_tasks.append(task)
            else:
                if reason:
                    print(f"   ⚙️  待运行: {method_with_ver:18s} @ {dataset:15s} - {reason}")
                tasks_to_run.append(task)
        
        print("=" * 80)
        print(f"📊 检测结果: 已完成 {len(skipped_tasks)}/{len(tasks)}, 待运行 {len(tasks_to_run)}/{len(tasks)}")
        if len(tasks_to_run) == 0:
            print("🎉 所有任务已完成！无需重新运行。")
        print("=" * 80 + "\n")
    else:
        tasks_to_run = tasks
    
    # 如果所有任务都已完成，直接返回
    if len(tasks_to_run) == 0:
        print("所有任务都已完成，直接使用缓存结果。\n")
        results = {}
        for task in skipped_tasks:
            dataset, method = task
            method_dir = output_base / f"{method}_{method_versions[method]}"
            output_dir = method_dir / dataset
            results[task] = {
                "returncode": 0,
                "elapsed": 0.0,
                "output_dir": output_dir,
                "cached": True,
            }
        return results, 0.0
    
    # 显示初始状态
    if RICH_AVAILABLE:
        console = Console()
        console.print("\n[bold cyan]" + "=" * 40 + "[/bold cyan]")
        console.print("[bold cyan]智能GPU任务调度启动[/bold cyan]")
        console.print("[bold cyan]" + "=" * 40 + "[/bold cyan]")
        console.print(f"[green]待运行:[/green] {len(tasks_to_run)}")
        console.print(f"[yellow]已缓存:[/yellow] {len(skipped_tasks)}")
        console.print(f"[cyan]可用GPU:[/cyan] {len(gpu_ids)} ({gpu_ids})")
        console.print(f"[magenta]并发度:[/magenta] {min(len(tasks_to_run), len(gpu_ids))}")
        console.print("[bold cyan]" + "=" * 40 + "[/bold cyan]\n")
    else:
        print(f"\n待运行: {len(tasks_to_run)} 任务")
        print(f"已缓存: {len(skipped_tasks)} 任务\n")
    
    # ✅ 动态任务队列：GPU完成一个任务后立即取下一个，最大化利用率
    # 创建共享任务队列
    task_queue = Queue()
    for task in tasks_to_run:
        task_queue.put(task)
    
    print("动态任务调度模式:")
    print(f"  共享任务队列: {len(tasks_to_run)} 个任务")
    print(f"  可用GPU: {len(gpu_ids)} 个")
    print("  调度策略: 动态分配（GPU空闲时自动取下一个任务）")
    print()
    
    # 使用每GPU一个专用Worker（线程），动态从共享队列获取任务
    results = {}
    results_lock = Lock()
    start_time = time.time()
    max_workers = len(gpu_ids)  # 固定为GPU数量

    # 定义GPU专用worker：从共享队列动态获取任务
    def gpu_worker(gpu_id: int, task_queue: Queue, monitor):
        """GPU worker: 持续从队列中取任务执行，直到队列为空"""
        from queue import Empty
        local_results = []
        while True:
            try:
                task = task_queue.get_nowait()
            except Empty:
                break
            
            try:
                # 统一使用 run_task，传递 monitor 参数
                dataset, method, returncode, elapsed, output_dir = run_task(
                    task, gpu_id, output_base, args, method_versions, monitor
                )
                local_results.append((task, returncode, elapsed, output_dir))
            finally:
                task_queue.task_done()
        
        return local_results

    if tasks_to_run:
        # 初始化进度监控
        progress_monitor = ProgressMonitor(len(tasks_to_run), gpu_ids) if RICH_AVAILABLE else None

        # 执行任务（统一逻辑）
        if progress_monitor:
            context_manager = progress_monitor.progress
        else:
            from contextlib import nullcontext
            context_manager = nullcontext()
        
        with context_manager, ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有GPU workers
            future_to_gpu = {
                executor.submit(gpu_worker, gpu_id, task_queue, progress_monitor): gpu_id
                for gpu_id in gpu_ids
            }

            # 收集结果
            completed = 0
            total_tasks = len(tasks_to_run)
            for future in as_completed(future_to_gpu):
                for task, returncode, elapsed, output_dir in future.result():
                    with results_lock:
                        results[task] = {
                            "returncode": returncode,
                            "elapsed": elapsed,
                            "output_dir": output_dir,
                        }
                        completed += 1
                        # 无progress时显示简单进度
                        if not progress_monitor:
                            progress_pct = completed / total_tasks * 100
                            print(f"进度: {completed}/{total_tasks} ({progress_pct:.1f}%)")
    
    total_time = time.time() - start_time
    
    # 显示完成信息
    if RICH_AVAILABLE:
        console = Console()
        console.print("\n[bold green]✓ 所有任务完成！[/bold green]")
        console.print(f"[cyan]总时间:[/cyan] {total_time:.1f}s ({total_time / 60:.1f}min)")
    else:
        print(f"\n{'=' * 80}")
        print(f"✓ 所有任务完成！总时间: {total_time:.1f}s")
        print(f"{'=' * 80}\n")
    
    # 添加跳过的任务到结果
    for task in skipped_tasks:
        dataset, method = task
        method_dir = output_base / f"{method}_{method_versions[method]}"
        output_dir = method_dir / dataset
        results[task] = {
            "returncode": 0,  # 假定缓存有效
            "elapsed": 0.0,
            "output_dir": output_dir,
            "cached": True,
        }
    
    return results, total_time


def parse_results_from_output(output_dir: Path, method: str, dataset: str) -> dict:
    """从输出目录解析结果
    
    Args:
        output_dir: 输出目录
        method: 方法名称
        dataset: 数据集名称
    
    Returns:
        结果字典 {dataset: {J&F, J, F}}
    """
    json_files = list(output_dir.glob("**/detailed_results.json"))
    if not json_files:
        print(f"⚠️  [{method}@{dataset}] No detailed_results.json found")
        return {}
    
    with open(json_files[0]) as f:
        data = json.load(f)
    
    json_key = METHOD_RESULT_KEYS.get(method)
    if not json_key or json_key not in data:
        return {}
    
    method_data = data[json_key]
    
    # 提取当前数据集的结果
    if dataset in method_data and isinstance(method_data[dataset], dict):
        metrics = method_data[dataset]
        return {
            dataset: {
                "J&F": metrics.get("jf", 0),
                "J": metrics.get("j", 0),
                "F": metrics.get("f", 0),
            }
        }
    
    return {}


def parse_statistics_from_output(output_dir: Path, method: str) -> dict[str, Any]:
    """从输出目录解析统计数据"""
    json_files = list(output_dir.glob("**/detailed_results.json"))
    if not json_files:
        return {}
    
    with open(json_files[0]) as f:
        data = json.load(f)
    
    stats_key = METHOD_STATS_KEYS.get(method)
    return data.get(stats_key, {}) if stats_key else {}


def merge_results_by_method(
    task_results: dict,
    output_base: Path,
    method_versions: dict[str, str],
) -> tuple[dict[str, dict], dict[str, dict], dict[str, float]]:
    """合并所有任务结果，按方法组织
    
    Args:
        task_results: 任务结果字典
        output_base: 输出基础路径
        method_versions: 方法版本映射
    
    Returns:
        (all_results, all_statistics, times)
        - all_results: {method: {dataset: {J&F, J, F}}}
        - all_statistics: {method: {dataset: stats}}
        - times: {(dataset, method): elapsed_time}
    """
    all_results = {method: {} for method in ALL_METHODS}
    all_statistics = {method: {} for method in ALL_METHODS}
    times = {}
    
    for (dataset, method), result in task_results.items():
        output_dir = result["output_dir"]
        returncode = result["returncode"]
        elapsed = result["elapsed"]
        
        times[(dataset, method)] = elapsed
        
        if returncode == 0:
            # 解析结果
            method_results = parse_results_from_output(output_dir, method, dataset)
            if method_results:
                all_results[method].update(method_results)
            
            # 解析统计数据
            method_statistics = parse_statistics_from_output(output_dir, method)
            if method_statistics and dataset in method_statistics:
                all_statistics[method][dataset] = method_statistics[dataset]
    
    return all_results, all_statistics, times


def create_comprehensive_summary(
    all_results: dict[str, dict],
    times: dict[tuple[str, str], float],
    datasets: list[str],
    methods: list[str],
    output_path: Path,
    gpu_ids: list[int],
    actual_parallel_time: float = None,
    method_versions: dict[str, str] = None,
):
    """创建综合汇总报告
    
    Args:
        all_results: 所有方法的结果
        times: 每个任务的运行时间
        datasets: 数据集列表
        methods: 方法列表
        output_path: 输出路径
        gpu_ids: GPU ID列表（用于计算利用率）
        actual_parallel_time: 实际并行执行时间（墙上时间，秒）
        method_versions: 方法版本映射（用于显示版本信息）
    """
    print("\n" + "=" * 80)
    print("智能并行评估结果汇总")
    if method_versions:
        print("方法版本: " + ", ".join([f"{m}_{method_versions[m]}" for m in methods if m in method_versions]))
    print("=" * 80 + "\n")
    
    # 创建DataFrame（列名包含版本信息）
    data = []
    for dataset in datasets:
        row = {"Dataset": dataset}
        for method in methods:
            # 构建包含版本的列名
            if method_versions and method in method_versions:
                col_prefix = f"{method}_{method_versions[method]}"
            else:
                col_prefix = method
            
            if method in all_results and dataset in all_results[method]:
                metrics = all_results[method][dataset]
                row[f"{col_prefix}_J&F"] = f"{metrics['J&F']:.2f}"
                row[f"{col_prefix}_J"] = f"{metrics['J']:.2f}"
                row[f"{col_prefix}_F"] = f"{metrics['F']:.2f}"
            else:
                row[f"{col_prefix}_J&F"] = "N/A"
                row[f"{col_prefix}_J"] = "N/A"
                row[f"{col_prefix}_F"] = "N/A"
        data.append(row)
    
    # 添加时间统计行
    time_row = {"Dataset": "Avg Time (s)"}
    for method in methods:
        # 构建包含版本的列名
        if method_versions and method in method_versions:
            col_prefix = f"{method}_{method_versions[method]}"
        else:
            col_prefix = method
        
        method_times = [times.get((d, method), 0) for d in datasets]
        valid_times = [t for t in method_times if t > 0]
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            time_row[f"{col_prefix}_J&F"] = f"{avg_time:.1f}"
            time_row[f"{col_prefix}_J"] = "-"
            time_row[f"{col_prefix}_F"] = "-"
        else:
            time_row[f"{col_prefix}_J&F"] = "N/A"
            time_row[f"{col_prefix}_J"] = "-"
            time_row[f"{col_prefix}_F"] = "-"
    data.append(time_row)
    
    df = pd.DataFrame(data)
    
    # 保存CSV（包含版本信息）
    csv_file = output_path / "parallel_results_summary.csv"
    
    # 添加版本信息作为注释
    with open(csv_file, 'w') as f:
        if method_versions:
            f.write("# Method Versions:\n")
            for method in methods:
                if method in method_versions:
                    f.write(f"# {method}: {method_versions[method]}\n")
            f.write("#\n")
        # 写入DataFrame
        df.to_csv(f, index=False)
    
    # 打印表格
    print(df.to_string(index=False))
    print(f"\n✓ Results saved to: {csv_file}")
    
    # 打印时间统计
    print("\n" + "=" * 80)
    print("任务执行时间统计")
    print("=" * 80)
    
    # 按方法统计
    print("\n按方法统计:")
    for method in methods:
        method_times = [times.get((d, method), 0) for d in datasets if times.get((d, method), 0) > 0]
        if method_times:
            print(f"  {method:12s}: 平均 {sum(method_times) / len(method_times):.1f}s, "
                  f"最小 {min(method_times):.1f}s, 最大 {max(method_times):.1f}s")
    
    # 按数据集统计
    print("\n按数据集统计:")
    for dataset in datasets:
        dataset_times = [times.get((dataset, m), 0) for m in methods if times.get((dataset, m), 0) > 0]
        if dataset_times:
            print(f"  {dataset:12s}: 平均 {sum(dataset_times) / len(dataset_times):.1f}s, "
                  f"总计 {sum(dataset_times):.1f}s")
    
    # 总体统计
    all_times = [t for t in times.values() if t > 0]
    if all_times:
        total_sequential = sum(all_times)
        
        # 使用实际的墙上时间，如果未提供则使用理论最大值（每个GPU队列的最大时间）
        if actual_parallel_time is not None and actual_parallel_time > 0:
            total_parallel = actual_parallel_time
        else:
            # 回退方案：计算每个GPU队列的总时间，取最大值
            gpu_times = {}
            for (dataset, method), elapsed in times.items():
                # 简单估算：假设任务按顺序分配到GPU
                task_idx = list(times.keys()).index((dataset, method))
                gpu_id = task_idx % len(gpu_ids)
                if gpu_id not in gpu_times:
                    gpu_times[gpu_id] = 0
                gpu_times[gpu_id] += elapsed
            total_parallel = max(gpu_times.values()) if gpu_times else max(all_times)
        
        speedup = total_sequential / total_parallel if total_parallel > 0 else 1
        
        print("\n总体统计:")
        print(f"  顺序执行（预估）: {total_sequential:.1f}s ({total_sequential / 60:.1f}min)")
        print(f"  并行执行（实际）: {total_parallel:.1f}s ({total_parallel / 60:.1f}min)")
        print(f"  加速比: {speedup:.2f}x")
        gpu_utilization = total_sequential / (total_parallel * len(gpu_ids)) * 100
        print(f"  GPU利用率: {gpu_utilization:.1f}%")


def merge_detailed_results(
    all_results: dict[str, dict],
    all_statistics: dict[str, dict],
    output_path: Path,
    method_to_output: dict[tuple[str, str], Path],
):
    """合并所有详细结果到单个JSON文件
    
    Args:
        all_results: 所有方法的结果
        all_statistics: 所有统计数据
        output_path: 输出路径
        method_to_output: (dataset, method) -> output_dir映射
    """
    # 转换结果格式为zs.py期望的格式
    def convert_results(method: str) -> dict:
        if method not in all_results:
            return {}
        return {
            dataset: (metrics["J&F"], metrics["J"], metrics["F"]) 
            for dataset, metrics in all_results[method].items()
        }
    
    sam_results = convert_results("SAM")
    uctta_results = convert_results("UCTTA")
    bndl_aue_results = convert_results("BNDL_AUE")
    bndl_results = convert_results("BNDL")
    ur_ern_results = convert_results("UR-ERN")
    
    # 获取统计数据
    uctta_statistics = all_statistics.get("UCTTA", {})
    bndl_aue_statistics = all_statistics.get("BNDL_AUE", {})
    bndl_statistics = all_statistics.get("BNDL", {})
    
    # 生成可视化
    if sam_results and bndl_aue_results:
        try:
            print("\n生成综合对比图...")
            create_comprehensive_comparison_plots(
                sam2_results=sam_results,
                bndl_results=bndl_aue_results,
                bndl_statistics=bndl_aue_statistics,
                output_path=output_path,
                uctta_results=uctta_results or None,
                uctta_statistics=uctta_statistics or None,
            )
            print("✓ 综合对比图已生成")
        except Exception as e:
            print(f"警告: 无法生成综合对比图: {e}")
    
    # 生成UA shift分析
    if bndl_aue_statistics and bndl_aue_results and sam_results:
        try:
            print("\n生成UA shift分析...")
            # 智能选择源域
            available_datasets = list(bndl_aue_results.keys())
            source_domain = "MOSE_train" if "MOSE_train" in available_datasets else (
                available_datasets[0] if available_datasets else "MOSE_train"
            )
            
            # 构建root路径映射
            sam2_root = None
            bndl_aue_root = None
            uctta_root = None
            ur_ern_root = None
            bndl_root = None
            
            for key, output_dir in method_to_output.items():
                # 提取方法名（兼容两种键类型）
                method = key[1] if isinstance(key, tuple) else key if isinstance(key, str) else None
                if not method:
                    continue

                if method == "SAM":
                    sam2_root = output_dir / "sam2_results"
                elif method == "BNDL_AUE":
                    bndl_aue_root = output_dir / "bndl_aue_results"
                elif method == "UCTTA":
                    uctta_root = output_dir / "sam2_uctta_results"
                elif method == "UR-ERN":
                    ur_ern_root = output_dir / "sam2_ur_ern_results"
                elif method == "BNDL":
                    bndl_root = output_dir / "bndl_results"
            
            create_ua_shift_analysis_plots(
                bndl_statistics=bndl_aue_statistics,
                sam2_results=sam_results,
                bndl_results=bndl_aue_results,
                output_path=output_path,
                uctta_statistics=uctta_statistics or None,
                uctta_results=uctta_results or None,
                ur_ern_results=ur_ern_results or None,
                bndl_pure_statistics=bndl_statistics or None,
                bndl_pure_results=bndl_results or None,
                source_domain=source_domain,
                sam2_root_override=sam2_root,
                bndl_root_override=bndl_aue_root,
                uctta_root_override=uctta_root,
                ur_ern_root_override=ur_ern_root,
                bndl_pure_root_override=bndl_root,
            )
            print("✓ UA shift分析图已生成")
        except Exception as e:
            print(f"警告: 无法生成UA shift分析: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="智能GPU任务调度器 - 动态并行评估(数据集, 方法)组合"
    )
    
    # Dataset selection
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=list(DATASET_CONFIGS.keys()),
        help="要评估的数据集列表",
    )
    
    # Method selection
    parser.add_argument(
        "--methods",
        nargs="+",
        default=ALL_METHODS,
        choices=ALL_METHODS,
        help="要运行的方法列表（默认：所有方法）",
    )
    
    # GPU配置
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7],
        help="可用GPU ID列表（默认：0-7）",
    )
    
    # SAM-2配置
    parser.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--sam2_checkpoint", default="/home/hongyou/dev/ada_samp/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    
    # BNDL+AUE配置
    parser.add_argument("--bndl_aue_cfg", default="configs/sam2.1/sam2.1_hiera_b+_bndl_aue.yaml")
    parser.add_argument("--bndl_aue_checkpoint", default="/home/hongyou/dev/ada_samp/logs/sam2/sam2_bndl_aue_012_09/checkpoints/checkpoint.pt")
    
    # BNDL (pure)配置
    parser.add_argument("--bndl_cfg", default="configs/sam2.1/sam2.1_hiera_b+_bndl.yaml")
    parser.add_argument("--bndl_checkpoint", default="/home/hongyou/dev/ada_samp/logs/sam2/sam2_bndl_013_01/checkpoints/checkpoint.pt")
    
    # UR-ERN配置
    parser.add_argument("--ur_ern_cfg", default="configs/sam2.1/sam2.1_hiera_b+_ur_ern.yaml")
    parser.add_argument("--ur_ern_checkpoint", default="/home/hongyou/dev/ada_samp/logs/sam2/sam2_ur_ern_001_01/checkpoints/checkpoint.pt")
    
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
    parser.add_argument("--output_path", type=Path, default=Path("./outputs/parallel_smart"))
    
    # 版本号配置
    parser.add_argument("--sam_version", type=str, default="001_01")
    parser.add_argument("--uctta_version", type=str, default="001_01")
    parser.add_argument("--bndl_aue_version", type=str, default="012_09")
    parser.add_argument("--bndl_version", type=str, default="013_01")
    parser.add_argument("--ur_ern_version", type=str, default="001_01")
    
    # 智能续跑功能
    parser.add_argument(
        "--reuse_cached", 
        action="store_true", 
        default=False,
        help="智能续跑模式：自动检测并跳过已完成的(方法,数据集)组合，中途中断后可续跑"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    args.output_path.mkdir(parents=True, exist_ok=True)
    
    # 打印配置
    print("=" * 80)
    print("智能GPU任务调度器")
    print("=" * 80)
    print(f"数据集: {args.datasets} ({len(args.datasets)}个)")
    print(f"方法: {args.methods} ({len(args.methods)}个)")
    
    # 方法版本映射
    method_versions = {
        "SAM": args.sam_version,
        "UCTTA": args.uctta_version,
        "BNDL_AUE": args.bndl_aue_version,
        "BNDL": args.bndl_version,
        "UR-ERN": args.ur_ern_version,
    }
    
    # 显示当前运行使用的方法版本
    print("\n方法版本配置（当前运行）:")
    for method in args.methods:
        if method in method_versions:
            version = method_versions[method]
            print(f"  • {method:12s} → 版本 {version}")
    
    print(f"\n总任务数: {len(args.datasets)} × {len(args.methods)} = {len(args.datasets) * len(args.methods)}")
    print(f"可用GPU: {len(args.gpu_ids)} ({args.gpu_ids})")
    print(f"输出目录: {args.output_path}")
    print(f"模式: {'仅第一帧' if args.first_frame_only else '完整视频'}")
    print(f"智能续跑: {'启用 - 将跳过已完成任务' if args.reuse_cached else '禁用 - 将运行所有任务'}")
    print("=" * 80 + "\n")
    
    # 生成所有任务
    all_tasks = generate_all_tasks(args.datasets, args.methods)
    
    print(f"生成的任务列表（共{len(all_tasks)}个）:")
    for i, (dataset, method) in enumerate(all_tasks, 1):
        version = method_versions[method]
        method_with_ver = f"{method}_{version}"
        print(f"  {i:2d}. {method_with_ver:18s} @ {dataset}")
    print()
    
    # 智能调度任务到GPU
    task_results, actual_parallel_time = schedule_tasks_on_gpus(
        tasks=all_tasks,
        gpu_ids=args.gpu_ids,
        output_base=args.output_path,
        args=args,
        method_versions=method_versions,
        reuse_cached=args.reuse_cached,
    )
    
    # 合并结果
    # 注意：all_results 只包含当前运行指定版本的结果
    # 版本号已嵌入在 output_dir 路径中（METHOD_VERSION/DATASET/）
    all_results, all_statistics, times = merge_results_by_method(
        task_results,
        args.output_path,
        method_versions,
    )
    
    # 创建method_to_output映射（用于可视化函数）
    # 注意：这里的 output_dir 包含版本号，例如：
    #   outputs/parallel_smart/BNDL_AUE_012_06/GTEA/
    method_to_output = {}
    for (dataset, method), result in task_results.items():
        if method not in method_to_output:
            # 使用第一个数据集的输出目录作为该方法的根目录
            method_to_output[method] = result["output_dir"]
        # 也保存完整的(dataset, method)映射
        method_to_output[(dataset, method)] = result["output_dir"]
    
    # 创建综合汇总
    create_comprehensive_summary(
        all_results,
        times,
        args.datasets,
        args.methods,
        args.output_path,
        args.gpu_ids,
        actual_parallel_time,
        method_versions,  # 传递版本信息
    )
    
    # 生成可视化
    if all_results:
        merge_detailed_results(
            all_results,
            all_statistics,
            args.output_path,
            method_to_output,
        )
    
    print(f"\n所有输出保存到: {args.output_path}")
    print("✓ 智能并行评估完成！")


if __name__ == "__main__":
    main()
