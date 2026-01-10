"""
Simplified Uncertainty Evaluation for BNDL.

Evaluates uncertainty quality using:
- Image-level Pearson/Spearman correlations (statistically valid)
- Pixel-level AUROC (error detection capability)
- ECE (calibration quality)
- PAvPU (uncertainty-accuracy alignment)
"""

import json
import logging
import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")


def _dilate_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Dilate a binary mask using max pooling."""
    if kernel_size <= 1:
        return mask
    mask_4d = mask.unsqueeze(0).unsqueeze(0).float()
    padding = (kernel_size - 1) // 2
    dilated = F.max_pool2d(mask_4d, kernel_size=kernel_size, stride=1, padding=padding)
    return dilated.squeeze() > 0.5


class DistributedDatasetEvaluator:
    """Simplified evaluator for uncertainty quality metrics."""

    def __init__(
        self,
        save_dir: str,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        foreground_dilation: int = 15,  # Pixels to dilate GT region for boundary focus
    ):
        self.save_dir = save_dir
        self.distributed = distributed and dist.is_initialized()
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        self.foreground_dilation = foreground_dilation

        # Simple image-level storage (sufficient for correlations)
        self.data: dict[str, list[float]] = {
            "uncertainty": [],
            "accuracy": [],
            "correctness": [],  # Binary: pred == gt
            "nll": [],
        }

        # Pixel-level samples for AUROC (subsampled to limit memory)
        self.pixel_samples: dict[str, list[float]] = {
            "uncertainty": [],
            "is_correct": [],  # Binary: 0 or 1
        }
        self.max_pixel_samples = 100000  # Limit memory usage

        # Results cache
        self.results: dict[str, Any] = {}

        if self.is_main:
            os.makedirs(save_dir, exist_ok=True)
            logging.info(f"Evaluator initialized: distributed={distributed}, dilation={foreground_dilation}")

    def add_batch(
        self,
        uncertainty: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> None:
        """Add a batch of data. Computes image-level metrics from each sample."""
        try:
            # Ensure tensors are on same device and aligned
            B = pred_logits.shape[0]

            for i in range(B):
                unc = uncertainty[i] if uncertainty.ndim > 2 else uncertainty
                pred = pred_logits[i]
                gt = gt_masks[i]

                # Align spatial dims if needed
                if pred.shape[:2] != gt.shape[:2]:
                    gt = F.interpolate(
                        gt.unsqueeze(0).unsqueeze(0).float(),
                        size=pred.shape[:2],
                        mode="nearest",
                    ).squeeze()

                # Compute image-level metrics
                pred_binary = pred > 0
                gt_binary = gt > 0.5

                # Accuracy (soft): P(correct class)
                pred_prob = torch.sigmoid(pred)
                accuracy = torch.where(gt_binary, pred_prob, 1 - pred_prob).mean()

                # Correctness (binary): fraction of correct pixels
                correctness = (pred_binary == gt_binary).float().mean()

                # NLL
                pred_prob_safe = pred_prob.clamp(1e-7, 1 - 1e-7)
                nll = -(gt_binary.float() * torch.log(pred_prob_safe) + (1 - gt_binary.float()) * torch.log(1 - pred_prob_safe)).mean()

                # Uncertainty (mean over spatial dims)
                if unc.ndim >= 2:
                    unc_scalar = unc.float().mean()
                else:
                    unc_scalar = unc.float()

                # Store image-level
                self.data["uncertainty"].append(float(unc_scalar.cpu()))
                self.data["accuracy"].append(float(accuracy.cpu()))
                self.data["correctness"].append(float(correctness.cpu()))
                self.data["nll"].append(float(nll.cpu()))

                # Sample pixels for AUROC (from dilated GT region only)
                if len(self.pixel_samples["uncertainty"]) < self.max_pixel_samples:
                    # Create dilated GT mask (focus on boundary region)
                    gt_mask = gt_binary.any(dim=-1) if gt_binary.ndim == 3 else gt_binary
                    if self.foreground_dilation > 0:
                        region_mask = _dilate_mask(gt_mask, self.foreground_dilation)
                    else:
                        region_mask = gt_mask

                    # Get indices of pixels in the region
                    region_indices = torch.where(region_mask.flatten())[0].cpu().numpy()

                    if len(region_indices) > 0:
                        # Get uncertainty and correctness for region pixels
                        unc_flat = unc.float().flatten().cpu().numpy()
                        correct_flat = (pred_binary == gt_binary).float()
                        if correct_flat.ndim == 3:
                            correct_flat = correct_flat.any(dim=-1)  # Any channel correct
                        correct_flat = correct_flat.flatten().cpu().numpy()

                        # Sample from region only
                        n_sample = min(1000, len(region_indices), self.max_pixel_samples - len(self.pixel_samples["uncertainty"]))
                        if n_sample > 0:
                            sample_idx = np.random.choice(len(region_indices), n_sample, replace=False)
                            pixel_idx = region_indices[sample_idx]
                            self.pixel_samples["uncertainty"].extend(unc_flat[pixel_idx].tolist())
                            self.pixel_samples["is_correct"].extend(correct_flat[pixel_idx].tolist())

        except Exception as e:
            logging.warning(f"add_batch failed: {e}")

    def _gather_all(self) -> dict[str, np.ndarray]:
        """Gather data from all processes."""
        if not self.distributed:
            return {k: np.array(v, dtype=np.float32) for k, v in self.data.items()}

        gathered = {}
        for key, values in self.data.items():
            # Use all_gather_object for simplicity
            all_values = [None] * self.world_size
            dist.all_gather_object(all_values, values)
            merged = []
            for v in all_values:
                if v:
                    merged.extend(v)
            gathered[key] = np.array(merged, dtype=np.float32)

        return gathered

    def _gather_pixel_samples(self) -> dict[str, np.ndarray]:
        """Gather pixel samples from all processes for AUROC."""
        if not self.distributed:
            return {k: np.array(v, dtype=np.float32) for k, v in self.pixel_samples.items()}

        gathered = {}
        for key, values in self.pixel_samples.items():
            all_values = [None] * self.world_size
            dist.all_gather_object(all_values, values)
            merged = []
            for v in all_values:
                if v:
                    merged.extend(v)
            gathered[key] = np.array(merged, dtype=np.float32)

        return gathered

    def evaluate(self) -> dict[str, Any]:
        """Compute all uncertainty quality metrics."""
        if len(self.data["uncertainty"]) == 0:
            logging.warning("No data for evaluation")
            return {}

        # Gather from all processes
        data = self._gather_all()
        pixel_data = self._gather_pixel_samples()

        if not self.is_main:
            return {}

        n_samples = len(data["uncertainty"])
        logging.info(f"Evaluating {n_samples} images, {len(pixel_data['uncertainty'])} pixel samples")

        results: dict[str, Any] = {"n_images": n_samples}

        # Filter valid values
        valid = np.isfinite(data["uncertainty"]) & np.isfinite(data["accuracy"])
        unc = data["uncertainty"][valid]
        acc = data["accuracy"][valid]
        corr = data["correctness"][valid]

        if len(unc) < 2:
            logging.warning("Not enough valid samples")
            return results

        # === Correlations (image-level) ===
        # Pearson
        results["pearson_acc"] = float(np.corrcoef(unc, acc)[0, 1])
        results["pearson_corr"] = float(np.corrcoef(unc, corr)[0, 1])

        # Spearman
        sp_acc, _ = spearmanr(unc, acc)
        sp_corr, _ = spearmanr(unc, corr)
        results["spearman_acc"] = float(sp_acc)
        results["spearman_corr"] = float(sp_corr)

        # === AUROC (pixel-level): Can uncertainty detect errors? ===
        try:
            pix_unc = pixel_data["uncertainty"]
            pix_correct = pixel_data["is_correct"]
            valid_pix = np.isfinite(pix_unc) & np.isfinite(pix_correct)
            pix_unc = pix_unc[valid_pix]
            pix_correct = pix_correct[valid_pix]

            if len(pix_unc) > 100:
                is_error = 1.0 - pix_correct  # 1 = error, 0 = correct
                results["auroc"] = float(roc_auc_score(is_error, pix_unc))
                results["aupr"] = float(average_precision_score(is_error, pix_unc))
                results["n_pixels"] = len(pix_unc)
                results["error_rate"] = float(is_error.mean())
                logging.info(f"Pixel-level AUROC: {results['auroc']:.4f} ({len(pix_unc)} pixels, {results['error_rate'] * 100:.1f}% errors)")
        except Exception as e:
            logging.warning(f"AUROC failed: {e}")

        # === ECE: Calibration (image-level) ===
        results["ece"] = self._compute_ece(acc, unc)

        # === PAvPU (image-level) ===
        results.update(self._compute_pavpu(unc, corr))

        # Log summary
        logging.info(f"Results: Pearson(acc)={results.get('pearson_acc', 0):.3f}, AUROC={results.get('auroc', 0):.3f}, ECE={results.get('ece', 0):.3f}")

        # Add histogram data for TensorBoard (subsample to limit size)
        if len(pixel_data["uncertainty"]) > 0:
            unc_samples = pixel_data["uncertainty"]
            if len(unc_samples) > 10000:
                unc_samples = np.random.choice(unc_samples, 10000, replace=False)
            results["uncertainty_histogram"] = unc_samples

        self.results = results
        return results

    def _compute_ece(self, accuracy: np.ndarray, uncertainty: np.ndarray, n_bins: int = 15) -> float:
        """Expected Calibration Error."""
        # Convert uncertainty to confidence (invert)
        unc_range = uncertainty.max() - uncertainty.min()
        if unc_range > 1e-8:
            confidence = 1.0 - (uncertainty - uncertainty.min()) / unc_range
        else:
            return 0.0

        ece = 0.0
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            mask = (confidence >= lo) & (confidence < hi)
            if mask.sum() > 0:
                avg_conf = confidence[mask].mean()
                avg_acc = accuracy[mask].mean()
                ece += np.abs(avg_conf - avg_acc) * mask.sum()

        return float(ece / len(accuracy))

    def _compute_pavpu(self, uncertainty: np.ndarray, correctness: np.ndarray) -> dict[str, float]:
        """PAvPU at multiple thresholds."""
        results = {}

        # Normalize uncertainty
        unc_range = uncertainty.max() - uncertainty.min()
        if unc_range > 1e-8:
            unc_norm = (uncertainty - uncertainty.min()) / unc_range
        else:
            return {"pavpu_0.05": 50.0}

        for thresh in [0.01, 0.05, 0.10]:
            # Top thresh% uncertainty = "uncertain"
            thresh_val = np.percentile(unc_norm, (1 - thresh) * 100)
            uncertain = unc_norm >= thresh_val
            certain = ~uncertain
            accurate = correctness > 0.5

            ac = (accurate & certain).sum()  # Accurate & Certain
            iu = (~accurate & uncertain).sum()  # Inaccurate & Uncertain
            pavpu = (ac + iu) / len(correctness) * 100

            results[f"pavpu_{thresh:.2f}"] = float(pavpu)

        return results

    def save_results(self, filename: str = "evaluation_results.json") -> str:
        """Save results to JSON."""
        if not self.is_main or not self.results:
            return ""

        path = os.path.join(self.save_dir, filename)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)

        logging.info(f"Results saved: {path}")
        return path

    def create_visualization(self, filename: str = "correlation_plot.png") -> str:
        """Create simple scatter plot of uncertainty vs accuracy."""
        if not self.is_main or len(self.data["uncertainty"]) == 0:
            return ""

        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            unc = np.array(self.data["uncertainty"])
            acc = np.array(self.data["accuracy"])
            corr = np.array(self.data["correctness"])

            # Uncertainty vs Accuracy
            axes[0].scatter(unc, acc, alpha=0.3, s=10)
            axes[0].set_xlabel("Uncertainty")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_title(f"Unc vs Acc (r={self.results.get('pearson_acc', 0):.3f})")

            # Uncertainty vs Correctness
            axes[1].scatter(unc, corr, alpha=0.3, s=10)
            axes[1].set_xlabel("Uncertainty")
            axes[1].set_ylabel("Correctness")
            axes[1].set_title(f"Unc vs Corr (r={self.results.get('pearson_corr', 0):.3f})")

            plt.tight_layout()
            path = os.path.join(self.save_dir, filename)
            plt.savefig(path, dpi=100)
            plt.close(fig)

            logging.info(f"Visualization saved: {path}")
            return path

        except Exception as e:
            logging.warning(f"Visualization failed: {e}")
            return ""

    def reset(self) -> None:
        """Clear all data."""
        for key in self.data:
            self.data[key].clear()
        for key in self.pixel_samples:
            self.pixel_samples[key].clear()
        self.results.clear()

    def __len__(self) -> int:
        return len(self.data["uncertainty"])


# Backward compatibility alias
DatasetEvaluator = DistributedDatasetEvaluator
