#!/usr/bin/env python
"""
Script to analyze Soft Accuracy and Confidence histograms for SAM2 across 23 datasets.
Focuses on decision-critical regions (Ground Truth Neighborhood).

Reference: parallel_compare.py, zs.py
"""

import argparse
import pickle
import sys
import os
import shutil
import warnings
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
import scipy.ndimage
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# Adjust path to find sam2 modules
# Assuming script is in sam2/scripts/, we need to add the root (../../)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also add the scripts directory itself to path to import sibling modules easily
if str(current_file.parent) not in sys.path:
    sys.path.insert(0, str(current_file.parent))

from sam2.build_sam import build_sam2_video_predictor
from dataset_configs import DATASET_CONFIGS, DEFAULT_DATASETS
from prompt_generation import generate_click_prompts


def get_roi_mask(gt_mask: np.ndarray, dilate_iter: int = 10) -> np.ndarray:
    """
    Create a Region of Interest (ROI) mask by dilating the GT mask.
    The ROI focuses on the object and its immediate neighborhood.
    """
    struct = scipy.ndimage.generate_binary_structure(2, 2)
    dilated = scipy.ndimage.binary_dilation(gt_mask, structure=struct, iterations=dilate_iter)
    return dilated


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def analyze_dataset(
    dataset_name: str,
    predictor,
    output_pkl: Path,
    num_bins: int = 50,
    dilate_iter: int = 4,
    max_videos: int | None = None,
    seed: int = 0,
    click_protocol: str = "3click",
    min_click_dist: float = 12.0,
) -> Dict[str, Any]:
    
    if dataset_name not in DATASET_CONFIGS:
        print(f"Dataset {dataset_name} not found in configs.")
        return {}

    config = DATASET_CONFIGS[dataset_name]
    split = config["default_split"]
    if isinstance(split, list):
        split = split[0]
        
    root = Path(config["root"])
    if config["has_split_subdir"]:
        jpeg_dir = root / split / "JPEGImages"
        ann_dir = root / split / "Annotations"
    else:
        jpeg_dir = root / "JPEGImages"
        ann_dir = root / "Annotations"

    if not jpeg_dir.is_dir() or not ann_dir.is_dir():
        print(f"Skipping {dataset_name}: Data not found at {jpeg_dir}")
        return {}

    video_names = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    if max_videos and len(video_names) > max_videos:
        video_names = video_names[:max_videos]

    print(f"Analyzing {dataset_name}: {len(video_names)} videos (First Frame Only)")
    
    # Histograms
    # Bins for raw probability [0, 1]
    bins = np.linspace(0, 1, num_bins + 1)
    
    # Accumulators (counts per bin)
    hist_roi_all = np.zeros(num_bins, dtype=np.int64)
    hist_roi_correct = np.zeros(num_bins, dtype=np.int64)
    hist_roi_incorrect = np.zeros(num_bins, dtype=np.int64)
    
    total_pixels_roi = 0
    pixels_in_ambiguous_range = 0 # range [0.4, 0.6]

    for vid in tqdm(video_names, desc=f"{dataset_name}", leave=False):
        video_dir = jpeg_dir / vid
        # Only process first frame for Zero-shot analysis (standard primitive SAM behavior)
        frame_names = sorted([p.stem for p in video_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]], key=lambda x: int(x))
        
        if not frame_names:
            continue
            
        first_frame_name = frame_names[0]
        
        # init state with max_frames=1 to strictly load only the first frame
        state = predictor.init_state(str(video_dir), max_frames=1)
        H, W = state["video_height"], state["video_width"]
        
        # Load GT
        gt_path = ann_dir / vid / f"{first_frame_name}.png"
        if not gt_path.exists():
            continue
            
        gt_mask_all = np.array(Image.open(gt_path))
        obj_ids = [oid for oid in np.unique(gt_mask_all) if oid > 0]
        
        for obj_id in obj_ids:
            gt_bool = gt_mask_all == obj_id
            if not np.any(gt_bool):
                continue
                
            # Generate prompts
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                _, _, _ = predictor.reset_state(state) # clear previous prompts if any, but keep image features
                
                # Mock prompt generation or use helper
                # Using generate_click_prompts from scripts
                used_pts, used_labels = generate_click_prompts(
                    predictor,
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    gt_bool=gt_bool,
                    first_frame_mask_np=gt_mask_all,
                    click_protocol=click_protocol,
                    min_click_dist=min_click_dist
                )
                
                # Propagate (Run Model)
                # First frame only -> run propagate_in_video for 1 step or use run_step if accessible?
                # propagate_in_video works well
                for _, out_obj_ids, out_logits in predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=0):
                    # Find our object channel
                    if obj_id in out_obj_ids:
                        idx = out_obj_ids.index(obj_id)
                        logits = out_logits[idx] 
                        if logits.ndim == 3:
                            logits = logits.squeeze(0)
                        
                        # Interpolate if needed
                        if tuple(logits.shape[-2:]) != (H, W):
                             logits = torch.nn.functional.interpolate(
                                 logits.unsqueeze(0).unsqueeze(0), 
                                 size=(H, W), 
                                 mode="bilinear", 
                                 align_corners=False
                             )[0, 0]
                        
                        # Sigmoid for probability
                        probs = torch.sigmoid(logits).float().cpu().numpy()
                        
                        # ROI Mask
                        roi_mask = get_roi_mask(gt_bool, dilate_iter=dilate_iter)
                        
                        # Extract ROI pixels
                        probs_roi = probs[roi_mask]
                        gt_roi = gt_bool[roi_mask].astype(int) # 0 or 1
                        
                        if len(probs_roi) == 0:
                            continue
                            
                        # Prediction (0.5 threshold)
                        preds_roi = (probs_roi > 0.5).astype(int)
                        
                        # Correct/Incorrect
                        correct_mask = (preds_roi == gt_roi)
                        incorrect_mask = ~correct_mask
                        
                        # Update Histograms (Raw Probabilities)
                        counts_all, _ = np.histogram(probs_roi, bins=bins)
                        hist_roi_all += counts_all
                        
                        if np.any(correct_mask):
                            counts_correct, _ = np.histogram(probs_roi[correct_mask], bins=bins)
                            hist_roi_correct += counts_correct
                            
                        if np.any(incorrect_mask):
                            counts_incorrect, _ = np.histogram(probs_roi[incorrect_mask], bins=bins)
                            hist_roi_incorrect += counts_incorrect
                        
                        # Stats
                        total_pixels_roi += len(probs_roi)
                        pixels_in_ambiguous_range += np.sum((probs_roi >= 0.4) & (probs_roi <= 0.6))
                        
                    break # Only first frame loop
            except Exception as e:
                print(f"Error processing {vid} obj {obj_id}: {e}")
                continue

        predictor.reset_state(state)
        
    # Save results
    stats = {
        "dataset": dataset_name,
        "hist_bins": bins,
        "hist_all": hist_roi_all,
        "hist_correct": hist_roi_correct,
        "hist_incorrect": hist_roi_incorrect,
        "total_pixels_roi": total_pixels_roi,
        "pixels_ambiguous": pixels_in_ambiguous_range,
        "percent_ambiguous": (pixels_in_ambiguous_range / total_pixels_roi * 100) if total_pixels_roi > 0 else 0
    }
    
    with open(output_pkl, "wb") as f:
        pickle.dump(stats, f)
        
    print(f"Saved stats for {dataset_name} to {output_pkl}")
    print(f"  Ambiguous [0.4, 0.6] %: {stats['percent_ambiguous']:.4f}%")
    return stats


def plot_results(stats_files: List[Path], output_dir: Path):
    """
    Generate the 3 visualizations requested.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_stats = []
    for p in stats_files:
        if p.exists():
            with open(p, "rb") as f:
                all_stats.append(pickle.load(f))
                
    if not all_stats:
        print("No stats loaded to plot.")
        return

    # 1. Ambiguous Region Table/Chart
    print("\n=== Ambiguous Region Statistics ([0.4, 0.6]) ===")
    ambig_data = []
    for s in all_stats:
        ambig_data.append({
            "Dataset": s["dataset"],
            "Ambiguous %": s["percent_ambiguous"]
        })
        print(f"{s['dataset']}: {s['percent_ambiguous']:.4f}%")
    
    df_ambig = pd.DataFrame(ambig_data) # Removed set_index to keep Dataset as column
    
    # 2. Correct vs Incorrect Soft Acc Distribution (The Killer)
    # We aggregate ALL datasets or plot per dataset?
    # User said "Correct vs Incorrect prediction... on same graph".
    # User said "Metric: % of pixels... across 23 datasets".
    # Ideally, we show one aggregated plot or multiple.
    # Let's do an AGGREGATED plot (pooling all pixels from all datasets) to be very strong.
    # But weighted by pixel count? Yes, simply summing histograms works.
    
    agg_hist_correct = np.zeros_like(all_stats[0]["hist_correct"])
    agg_hist_incorrect = np.zeros_like(all_stats[0]["hist_incorrect"])
    bins = all_stats[0]["hist_bins"]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    for s in all_stats:
        agg_hist_correct += s["hist_correct"]
        agg_hist_incorrect += s["hist_incorrect"]
        
    # Convert to probability density (normalized)
    # Note: Correct counts >> Incorrect counts usually.
    # User wants to show SHAPE/Distribution.
    # So normalize each separately.
    
    def normalize(h):
        s = h.sum()
        return (h / s) if s > 0 else h

    dens_correct = normalize(agg_hist_correct)
    dens_incorrect = normalize(agg_hist_incorrect)
    
    # Process "Soft Acc" / Confidence
    # Raw probs P are in [0, 1].
    # For Correct pixels:
    #   If P ~ 1 (FG correctly predicted as FG), Conf ~ 1.
    #   If P ~ 0 (BG correctly predicted as BG), Conf ~ 1 (since 1-P ~ 1).
    # so we should Fold the raw probability histogram around 0.5 to get Confidence?
    # User: "x axis: soft acc ... σ(|logit_fg - logit_bg|)".
    # This implies Confidence (0.5 to 1.0) or Raw Logit.
    # If the user says "Incorrect prediction... soft acc still concentrated in high region",
    # and "soft acc" for incorrect usually means P(predicted).
    
    # Let's TRANSFORM the raw histograms to Confidence histograms [0.5, 1.0]
    # Raw bins: [0.0, 0.02, ... 1.0]
    # We want bins [0.5, 1.0]
    # Conf = max(p, 1-p)
    # P in [0, 0.5] -> Conf in [0.5, 1.0] (maps 0->1, 0.5->0.5)
    # P in [0.5, 1] -> Conf in [0.5, 1.0]
    
    mid_idx = len(bin_centers) // 2
    # Ensure even bins for symmetry
    
    # Simply iterate raw bins and map to conf bins
    # But simpler: we have Raw Probs histograms for Correct and Incorrect.
    # Let's plot Raw Probs first to check. If "Incorrect" is bimodal at 0 and 1, then Conf is at 1.
    
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, dens_correct, label="Correct Predictions", color="green", linewidth=2)
    plt.plot(bin_centers, dens_incorrect, label="Incorrect Predictions", color="red", linewidth=2, linestyle="--")
    plt.title("Distribution of Raw Probabilities (Aggregated)", fontsize=14)
    plt.xlabel("Raw Sigmoid Probability", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "raw_prob_dist_correct_vs_incorrect.png")
    plt.close()
    
    # Now Confidence Plot (The Killer)
    # Fold the densities
    num_half = len(bin_centers) // 2
    # split at 0.5
    # For equal bin width
    
    # Construct High Confidence histogram manually from bin counts
    # It is hard to fold perfectly if bins are not aligned, but with 50 bins [0,1], 0.5 is at index 25.
    
    half_bins = bins[mid_idx:] # [0.5 ... 1.0]
    half_centers = 0.5 * (half_bins[:-1] + half_bins[1:])
    
    # Map [0, 0.5] to [1.0, 0.5] and add to [0.5, 1.0]
    # counts[0] (p=0..0.02) -> corresponds to conf 1.0..0.98. Add to last bin.
    # index i < mid_idx: dist from 0.5 is (0.5 - p). Conf = 0.5 + dist = 1 - p.
    # index j >= mid_idx: dist from 0.5 is (p - 0.5). Conf = p.
    
    def to_confidence_hist(counts):
        # counts has N bins. N should be even.
        N = len(counts)
        mid = N // 2
        conf_counts = np.zeros(mid, dtype=np.float64)
        
        # Upper half (0.5 to 1.0)
        conf_counts += counts[mid:]
        
        # Lower half (0 to 0.5) -> reversed maps to (0.5 to 1.0)
        # e.g. bins 0..24. bin 24 is 0.48-0.5. Mapped to 0.5-0.52 (idx 0 of upper).
        conf_counts += counts[:mid][::-1] 
        
        return conf_counts

    conf_correct_counts = to_confidence_hist(agg_hist_correct)
    conf_incorrect_counts = to_confidence_hist(agg_hist_incorrect)
    
    conf_dens_correct = normalize(conf_correct_counts)
    conf_dens_incorrect = normalize(conf_incorrect_counts)
    
    plt.figure(figsize=(8, 6))
    plt.plot(half_centers, conf_dens_correct, label="Correct Predictions", color="green", linewidth=2.5)
    plt.plot(half_centers, conf_dens_incorrect, label="Incorrect Predictions", color="red", linewidth=2.5, linestyle="--")
    plt.fill_between(half_centers, conf_dens_incorrect, alpha=0.1, color="red")
    
    plt.title("Soft Accuracy (Confidence) Distribution\nCorrect vs Incorrect Predictions", fontsize=14, fontweight='bold')
    plt.xlabel("Confidence (max(p, 1-p))", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(0.5, 1.0)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Annotate "Overconfidence"
    plt.text(0.8, 0.5 * max(conf_dens_incorrect), "Incorrect predictions\nstill high confidence!", 
             color="red", fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    
    plt.savefig(output_dir / "soft_acc_distribution_killer.png", dpi=300)
    print(f"Saved killer plot to {output_dir / 'soft_acc_distribution_killer.png'}")
    plt.close()

    # 3. Soft Acc Histogram per Dataset (Grouped or individual?)
    # User: "One plot per dataset (or grouped). x axis: soft acc...".
    # We can make a grid of 23 small plots.
    
    rows = 4
    cols = 6
    fig, axes = plt.subplots(rows, cols, figsize=(24, 16))
    axes = axes.flatten()
    
    for i, s in enumerate(all_stats):
        if i >= len(axes): break
        ax = axes[i]
        
        # Calculate conf hist for this dataset
        c_conf = to_confidence_hist(s["hist_all"])
        d_conf = normalize(c_conf)
        
        ax.bar(half_centers, d_conf, width=1.0/len(half_centers)/2, color='royalblue', alpha=0.8)
        ax.set_title(s["dataset"], fontsize=10)
        ax.set_xlim(0.5, 1.0)
        ax.set_yticks([])
        
        # Shade the empty middle region if visible
        # Middle region in raw probs was 0.3-0.7.
        # Here in confidence, that corresponds to 0.5-0.7 (since 0.3->0.7, 0.7->0.7).
        # User said "Middle range (0.3-0.7) almost empty".
        # If plotted as Confidence, 0.5-0.7 should be empty.
        
    plt.tight_layout()
    plt.savefig(output_dir / "per_dataset_soft_acc_histograms.png")
    plt.close()


def run_worker(args):
    """Worker process: analyzes a single dataset on a specific GPU."""
    import os
    
    # Note: CUDA_VISIBLE_DEVICES is now set by the manager via env vars
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    print(f"[Worker] GPU {args.gpu_id} processing {args.datasets[0]}...")
    
    # Build predictor (only one needed per worker)
    try:
        # device="cuda" will map to the visible device (always cuda:0 inside this isolated process)
        predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_checkpoint, device="cuda")
    except Exception as e:
        print(f"[Worker Error] Failed to build predictor: {e}")
        sys.exit(1)
        
    dataset = args.datasets[0] # Worker only gets one dataset
    stats_dir = args.output_dir / "stats"
    pkl_path = stats_dir / f"{dataset}_stats.pkl"
    
    try:
        analyze_dataset(
            dataset,
            predictor,
            output_pkl=pkl_path,
            max_videos=args.max_videos,
            # Pass other args if needed
        )
    except Exception as e:
        print(f"[Worker Error] Failed analyzing {dataset}: {e}")
        sys.exit(1)
        
    print(f"[Worker] GPU {args.gpu_id} finished {dataset}.")


def run_manager(args):
    """Manager process: schedules tasks on available GPUs."""
    import subprocess
    import time
    
    datasets = args.datasets
    gpu_ids = args.gpu_ids
    
    print(f"\n[Manager] Starting parallel analysis on {len(datasets)} datasets using GPUs: {gpu_ids}")
    
    # Queue of tasks
    task_queue = list(datasets)
    
    # Track running processes: {gpu_id: subprocess.Popen}
    running_procs = {} 
    
    # Stats plotting directory
    stats_dir = args.output_dir / "stats"
    
    with tqdm(total=len(datasets), desc="Overall Progress") as pbar:
        while task_queue or running_procs:
            # 1. Check for finished processes
            finished_gpus = []
            for gpu, proc in running_procs.items():
                if proc.poll() is not None: # Process finished
                    finished_gpus.append(gpu)
                    pbar.update(1)
                    if proc.returncode != 0:
                        print(f"\n[Manager] Warning: Task on GPU {gpu} failed with code {proc.returncode}")
            
            # Clean up finished
            for gpu in finished_gpus:
                del running_procs[gpu]
                
            # 2. Assign new tasks to free GPUs
            # Find free GPUs
            free_gpus = [g for g in gpu_ids if g not in running_procs]
            
            while free_gpus and task_queue:
                gpu = free_gpus.pop(0)
                dataset = task_queue.pop(0)
                
                # Check if result already exists
                pkl_path = stats_dir / f"{dataset}_stats.pkl"
                if pkl_path.exists():
                    pbar.write(f"Skipping {dataset} (already exists)")
                    pbar.update(1)
                    # Don't use the GPU, loop again to use it for next task or release it
                    free_gpus.insert(0, gpu) 
                    continue
                
                # Construct command
                # Must replicate current args but switch to worker mode
                cmd = [sys.executable, str(Path(__file__).resolve())]
                cmd.extend(["--worker_mode"])
                cmd.extend(["--gpu_id", str(gpu)])
                cmd.extend(["--datasets", dataset])
                cmd.extend(["--output_dir", str(args.output_dir)])
                cmd.extend(["--max_videos", str(args.max_videos)])
                cmd.extend(["--sam2_cfg", args.sam2_cfg])
                cmd.extend(["--sam2_checkpoint", args.sam2_checkpoint])
                
                # Set environment variable for this specific subprocess
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                
                # Limit threads to prevent CPU oversubscription
                env["OMP_NUM_THREADS"] = "1"
                env["MKL_NUM_THREADS"] = "1"
                env["OPENBLAS_NUM_THREADS"] = "1"
                env["VECLIB_MAXIMUM_THREADS"] = "1"
                env["NUMEXPR_NUM_THREADS"] = "1"
                
                # Launch
                # pbar.write(f"Launching {dataset} on GPU {gpu}")
                # Use DEVNULL for stderr to prevent deadlock from buffer filling (tqdm output)
                proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 
                
                running_procs[gpu] = proc
                
            time.sleep(0.5)

    print("[Manager] All tasks completed.")


def main():
    parser = argparse.ArgumentParser(description="Analyze Soft Acc / Confidence for SAM2")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Datasets to analyze")
    parser.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml", help="SAM2 config")
    parser.add_argument("--sam2_checkpoint", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="SAM2 checkpoint")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/soft_acc_analysis"), help="Output directory")
    parser.add_argument("--max_videos", type=int, default=50, help="Max videos per dataset (for speed)")
    parser.add_argument("--device", default="cuda", help="Device (ignored in parallel mode, uses --gpu_ids)")
    parser.add_argument("--plot_only", action="store_true", help="Only plot from existing pkls")
    
    # Parallel execution args
    parser.add_argument("--parallel", action="store_true", help="Run in parallel manager mode")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7], help="GPUs to use in parallel mode")
    
    # Internal args for worker
    parser.add_argument("--worker_mode", action="store_true", help="Internal flag: act as worker")
    parser.add_argument("--gpu_id", type=int, default=0, help="Internal flag: gpu id for worker")
    
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = args.output_dir / "stats"
    stats_dir.mkdir(exist_ok=True)

    if args.worker_mode:
        run_worker(args)
        return

    if not args.plot_only:
        if args.parallel:
            run_manager(args)
        else:
            # Serial execution
            print(f"Loading SAM2 model: {args.sam2_cfg}")
            try:
                predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_checkpoint, device=args.device)
            except Exception as e:
                print(f"Error loading model: {e}")
                sys.exit(1)
                
            for dataset in args.datasets:
                pkl_path = stats_dir / f"{dataset}_stats.pkl"
                if pkl_path.exists():
                    print(f"Stats for {dataset} already exist. Skipping.")
                    continue
                    
                analyze_dataset(
                    dataset, 
                    predictor, 
                    output_pkl=pkl_path,
                    max_videos=args.max_videos
                )
            
    # Generate Plots (Manager or Serial mode runs this at end)
    pkl_files = list(stats_dir.glob("*_stats.pkl"))
    if pkl_files:
        print("Generating plots...")
        plot_results(pkl_files, args.output_dir)
    else:
        print("No stats files found to plot.")

if __name__ == "__main__":
    main()
