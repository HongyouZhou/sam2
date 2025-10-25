#!/usr/bin/env python
"""Checkpoint manager for incremental saving during long-running evaluations.

This module provides efficient checkpoint management to prevent OOM errors
during evaluation of large datasets by periodically saving intermediate results.
"""

from __future__ import annotations

import gc
import json
import pickle
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch


class CheckpointManager:
    """Manages incremental checkpoint saving and merging for evaluation data.
    
    This class helps prevent OOM errors during long evaluations by:
    1. Periodically saving accumulated data to disk
    2. Clearing memory after saves
    3. Merging all checkpoints at the end
    
    Performance optimizations:
    - Reduced checkpoint frequency (default: 50 videos, further increased from 10)
    - Faster serialization using numpy.savez_compressed
    - Automatic cleanup of checkpoint files
    - Progress tracking during merge to monitor for hangs
    - Memory monitoring to prevent OOM errors
    """
    
    def __init__(
        self,
        output_dir: Path,
        dataset_name: str,
        checkpoint_type: str = "eval",
        interval: int = 50,
        use_numpy: bool = True,
    ):
        """Initialize checkpoint manager.
        
        Args:
            output_dir: Directory to save checkpoints
            dataset_name: Name of dataset (for checkpoint naming)
            checkpoint_type: Type of checkpoint ('eval' or 'stats')
            interval: Number of videos between checkpoints (default: 50, increased from 10 to reduce memory pressure)
            use_numpy: Use numpy.savez instead of pickle (faster, default: True)
        """
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.checkpoint_type = checkpoint_type
        self.interval = interval
        self.use_numpy = use_numpy
        self.checkpoint_files: list[Path] = []
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _checkpoint_glob_patterns(self) -> list[str]:
        """Return filename glob patterns for this manager's checkpoints."""
        # Support both numpy and pickle for eval, and json for stats
        base = f".{self.checkpoint_type}_checkpoint_{self.dataset_name}_*."
        if self.checkpoint_type == "stats":
            return [base + "json"]
        return [base + "npz", base + "pkl"]

    def discover_existing_checkpoints(self) -> None:
        """Discover leftover checkpoint files on disk and register them.

        Useful for resumable runs or when this object is created fresh and
        needs to merge previously saved shards.
        """
        discovered: list[Path] = []
        for pattern in self._checkpoint_glob_patterns():
            discovered.extend(sorted(self.output_dir.glob(pattern)))

        # Merge with any already-tracked files, de-duplicate
        existing = set(p.resolve() for p in self.checkpoint_files)
        all_files = existing.union(p.resolve() for p in discovered)

        # Sort by trailing numeric index if present
        def _index_key(p: Path) -> int:
            # Examples: .eval_checkpoint_MOSE_train_0010.npz
            m = re.search(r"_(\d{3,6})\.[^.]+$", p.name)
            try:
                return int(m.group(1)) if m else 0
            except Exception:
                return 0

        self.checkpoint_files = sorted([Path(x) for x in all_files], key=_index_key)
    
    def should_checkpoint(self, video_idx: int) -> bool:
        """Check if we should save a checkpoint at this video index.
        
        Args:
            video_idx: Current video index (1-based)
        
        Returns:
            True if checkpoint should be saved
        """
        return video_idx % self.interval == 0
    
    def save_checkpoint(
        self, 
        data: dict[str, Any], 
        video_idx: int,
        force_pickle: bool = False,
    ) -> Path:
        """Save checkpoint to disk.
        
        Args:
            data: Data dictionary to save
            video_idx: Current video index (for naming)
            force_pickle: Force use of pickle even if use_numpy=True
        
        Returns:
            Path to saved checkpoint file
        """
        # Generate checkpoint filename
        if self.use_numpy and not force_pickle:
            suffix = ".npz"
            checkpoint_file = (
                self.output_dir 
                / f".{self.checkpoint_type}_checkpoint_{self.dataset_name}_{video_idx:04d}{suffix}"
            )
            
            # Convert data to numpy-compatible format
            numpy_data = self._convert_to_numpy(data)
            
            # Save using numpy (faster compression)
            np.savez_compressed(checkpoint_file, **numpy_data)
        else:
            suffix = ".pkl"
            checkpoint_file = (
                self.output_dir 
                / f".{self.checkpoint_type}_checkpoint_{self.dataset_name}_{video_idx:04d}{suffix}"
            )
            
            # Save using pickle
            with open(checkpoint_file, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Track checkpoint file
        self.checkpoint_files.append(checkpoint_file)
        
        return checkpoint_file
    
    def load_checkpoint(self, checkpoint_file: Path) -> dict[str, Any]:
        """Load checkpoint from disk.
        
        Args:
            checkpoint_file: Path to checkpoint file
        
        Returns:
            Data dictionary
        """
        if checkpoint_file.suffix == ".npz":
            # Load numpy checkpoint
            loaded = np.load(checkpoint_file, allow_pickle=True)
            
            # Convert numpy arrays back to lists
            data = {}
            for key in loaded.files:
                value = loaded[key]
                # Handle numpy object arrays (contain lists)
                if value.dtype == object:
                    data[key] = value.item() if value.shape == () else value.tolist()
                else:
                    data[key] = value.tolist()
            
            return data
        else:
            # Load pickle checkpoint
            with open(checkpoint_file, "rb") as f:
                return pickle.load(f)
    
    def merge_checkpoints(self) -> dict[str, Any]:
        """Merge all saved checkpoints into a single dictionary.
        
        This method loads all checkpoint files, merges their data,
        and cleans up the checkpoint files.
        
        Returns:
            Merged data dictionary
        """
        if not self.checkpoint_files:
            # Try to discover leftover shards on disk
            self.discover_existing_checkpoints()
        if not self.checkpoint_files:
            return {}
        
        total_files = len(self.checkpoint_files)
        print(f"🔄 Merging {total_files} checkpoint files...")
        
        merged_data = {}
        
        for idx, checkpoint_file in enumerate(self.checkpoint_files, 1):
            try:
                checkpoint_data = self.load_checkpoint(checkpoint_file)
                
                # Merge data
                for key, value in checkpoint_data.items():
                    if key not in merged_data:
                        merged_data[key] = []
                    
                    # Handle different data types
                    if isinstance(value, list):
                        merged_data[key].extend(value)
                    elif isinstance(value, dict):
                        if not isinstance(merged_data[key], dict):
                            merged_data[key] = {}
                        merged_data[key].update(value)
                    else:
                        merged_data[key].append(value)
                
                # Explicitly delete checkpoint data
                del checkpoint_data
                
                # Clean up checkpoint file
                checkpoint_file.unlink()
                
                # Progress update and memory monitoring every 10 files or at the end
                if idx % 10 == 0 or idx == total_files:
                    print(f"  ├─ Processed {idx}/{total_files} checkpoint files")
                    # Show memory usage every 10 files
                    self.force_memory_cleanup(verbose=True)
                else:
                    # Silent cleanup for other files
                    self.force_memory_cleanup(verbose=False)
                
            except Exception as e:
                print(f"⚠️  Warning: Failed to merge checkpoint {checkpoint_file}: {e}")
                continue
        
        # Clear checkpoint list
        self.checkpoint_files.clear()
        
        print("✓ Merged all checkpoints")
        return merged_data

    def merge_checkpoints_streaming(self, on_data: Callable[[dict[str, Any]], None]) -> None:
        """Stream-merge checkpoints by processing each shard via a callback.

        This avoids materializing the fully merged dictionary in memory.

        Args:
            on_data: Function called with the loaded dict for each shard.
        """
        if not self.checkpoint_files:
            # Try to discover leftover shards on disk
            self.discover_existing_checkpoints()
        if not self.checkpoint_files:
            return

        total_files = len(self.checkpoint_files)
        print(f"🔄 Merging {total_files} checkpoint files...")
        
        # Process checkpoints with progress tracking
        for idx, checkpoint_file in enumerate(self.checkpoint_files, 1):
            try:
                # Load checkpoint data
                checkpoint_data = self.load_checkpoint(checkpoint_file)
                
                # Process this shard
                on_data(checkpoint_data)
                
                # Explicitly delete the data to free memory immediately
                del checkpoint_data
                
                # Remove shard after successful processing
                checkpoint_file.unlink()
                
                # Progress update and memory monitoring every 10 files or at the end
                if idx % 10 == 0 or idx == total_files:
                    print(f"  ├─ Processed {idx}/{total_files} checkpoint files")
                    # Show memory usage every 10 files
                    self.force_memory_cleanup(verbose=True)
                else:
                    # Silent cleanup for other files
                    self.force_memory_cleanup(verbose=False)
                    
            except Exception as e:
                print(f"⚠️  Warning: Failed to process checkpoint {checkpoint_file}: {e}")
                continue

        self.checkpoint_files.clear()
        print("✓ Merged all checkpoints")
    
    def cleanup(self) -> None:
        """Clean up any remaining checkpoint files."""
        for checkpoint_file in self.checkpoint_files:
            try:
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete checkpoint {checkpoint_file}: {e}")
        
        self.checkpoint_files.clear()
    
    @staticmethod
    def _convert_to_numpy(data: dict[str, Any]) -> dict[str, np.ndarray]:
        """Convert data dictionary to numpy-compatible format.
        
        Args:
            data: Data dictionary with mixed types
        
        Returns:
            Dictionary with numpy arrays
        """
        numpy_data = {}
        
        for key, value in data.items():
            if isinstance(value, list):
                # Try to convert list to numpy array
                try:
                    numpy_data[key] = np.array(value)
                except (ValueError, TypeError):
                    # Can't convert to regular array, use object array
                    numpy_data[key] = np.array(value, dtype=object)
            elif isinstance(value, np.ndarray):
                numpy_data[key] = value
            elif isinstance(value, dict):
                # Store dict as object array
                numpy_data[key] = np.array(value, dtype=object)
            else:
                # Store other types as object array
                numpy_data[key] = np.array(value, dtype=object)
        
        return numpy_data
    
    @staticmethod
    def get_memory_info() -> dict[str, float]:
        """Get current memory usage information.
        
        Returns:
            Dictionary with memory usage in GB (system RAM and GPU memory)
        """
        import psutil
        
        mem_info = {
            'system_used_gb': psutil.virtual_memory().used / (1024**3),
            'system_available_gb': psutil.virtual_memory().available / (1024**3),
            'system_percent': psutil.virtual_memory().percent,
        }
        
        if torch.cuda.is_available():
            mem_info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            mem_info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        
        return mem_info
    
    @staticmethod
    def force_memory_cleanup(verbose: bool = False) -> None:
        """Force aggressive memory cleanup (garbage collection + CUDA cache clear).
        
        Args:
            verbose: If True, print memory usage before and after cleanup
        """
        if verbose:
            mem_before = CheckpointManager.get_memory_info()
            print(f"  ├─ Memory before cleanup: System {mem_before['system_used_gb']:.1f}GB used ({mem_before['system_percent']:.1f}%)", end='')
            if 'gpu_allocated_gb' in mem_before:
                print(f", GPU {mem_before['gpu_allocated_gb']:.1f}GB allocated", end='')
            print()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if verbose:
            mem_after = CheckpointManager.get_memory_info()
            print(f"  ├─ Memory after cleanup: System {mem_after['system_used_gb']:.1f}GB used ({mem_after['system_percent']:.1f}%)", end='')
            if 'gpu_allocated_gb' in mem_after:
                print(f", GPU {mem_after['gpu_allocated_gb']:.1f}GB allocated", end='')
            print()


class StatisticsCheckpointManager(CheckpointManager):
    """Specialized checkpoint manager for JSON-serializable statistics."""
    
    def __init__(self, output_dir: Path, dataset_name: str, interval: int = 50):
        """Initialize statistics checkpoint manager.
        
        Args:
            output_dir: Directory to save checkpoints
            dataset_name: Name of dataset
            interval: Checkpoint interval (default: 50 videos, increased from 10)
        """
        super().__init__(
            output_dir=output_dir,
            dataset_name=dataset_name,
            checkpoint_type="stats",
            interval=interval,
            use_numpy=False,  # Use JSON for statistics
        )
    
    def save_checkpoint(self, data: dict[str, Any], video_idx: int) -> Path:
        """Save statistics checkpoint as JSON.
        
        Args:
            data: Statistics dictionary
            video_idx: Current video index
        
        Returns:
            Path to saved checkpoint file
        """
        checkpoint_file = (
            self.output_dir
            / f".stats_checkpoint_{self.dataset_name}_{video_idx:04d}.json"
        )
        
        with open(checkpoint_file, "w") as f:
            json.dump(data, f)
        
        self.checkpoint_files.append(checkpoint_file)
        return checkpoint_file
    
    def load_checkpoint(self, checkpoint_file: Path) -> dict[str, Any]:
        """Load statistics checkpoint from JSON.
        
        Args:
            checkpoint_file: Path to checkpoint file
        
        Returns:
            Statistics dictionary
        """
        with open(checkpoint_file) as f:
            return json.load(f)
    
    def merge_checkpoints(self) -> dict[str, Any]:
        """Merge all statistics checkpoints.
        
        Returns:
            Merged statistics dictionary
        """
        if not self.checkpoint_files:
            return {}
        
        print(f"🔄 Merging {len(self.checkpoint_files)} statistics checkpoint files...")
        
        merged_stats = {}
        
        for checkpoint_file in self.checkpoint_files:
            try:
                checkpoint_data = self.load_checkpoint(checkpoint_file)
                merged_stats.update(checkpoint_data)
                checkpoint_file.unlink()
            except Exception as e:
                print(f"Warning: Failed to merge statistics checkpoint {checkpoint_file}: {e}")
        
        self.checkpoint_files.clear()
        print("✓ Merged all statistics checkpoints")
        
        # 🎯 关键修复: 合并后立即降采样，防止内存爆炸
        from downsampling_utils import downsample_statistics_pavpu
        if merged_stats:
            _, n_orig, n_down = downsample_statistics_pavpu(merged_stats, max_samples=100000)
            if n_down < n_orig:
                print(f"  💾 合并后降采样: {n_orig:,} → {n_down:,} PAvPU样本 ({n_down/n_orig*100:.1f}%)")
        
        return merged_stats

