"""
BNDL可视化器模块
专门处理BNDL模型的可视化逻辑
"""

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from metric_calculator import MetricCalculator
from visualization_utils import VisualizationUtils


class BNDLVisualizer:
    """BNDL可视化器类"""

    def __init__(self):
        self.viz_utils = VisualizationUtils()
        self.metric_calc = MetricCalculator()

    def plot_parameter_and_uncertainty_overlays(self, axes, original_img: np.ndarray, lambda_img: np.ndarray, k_img: np.ndarray, bndl_outputs: Dict[str, Any], step_index: int) -> None:
        """参数和不确定性叠加图，包含PAvPU可视化"""
        try:
            lambda_norm, k_norm = self.viz_utils.normalize_parameters_robust(lambda_img, k_img)

            # 提取不确定性用于叠加
            uncertainty = None
            if "pixel_uncertainty" in bndl_outputs and bndl_outputs["pixel_uncertainty"] is not None:
                uncertainty_tensor = bndl_outputs["pixel_uncertainty"].detach().cpu().numpy()

                if len(uncertainty_tensor.shape) == 4:  # [B, H, W, C]
                    uncertainty = uncertainty_tensor[0].mean(axis=-1)  # 跨通道平均
                elif len(uncertainty_tensor.shape) == 3:  # [B, H, W]
                    uncertainty = uncertainty_tensor[0]
                else:
                    uncertainty = uncertainty_tensor

                # 如果需要，将不确定性调整为与原始图像匹配
                if uncertainty.shape != lambda_img.shape:
                    uncertainty = cv2.resize(uncertainty, (lambda_img.shape[1], lambda_img.shape[0]), interpolation=cv2.INTER_LINEAR)

                # 归一化不确定性
                uncertainty_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)

            # Lambda叠加
            axes[0].imshow(original_img)
            axes[0].imshow(lambda_norm, cmap="viridis", alpha=0.6, interpolation="nearest")
            axes[0].set_title(f"Lambda Overlay (Step {step_index})")
            axes[0].axis("off")

            # 不确定性叠加
            if uncertainty is not None:
                axes[1].imshow(original_img)
                axes[1].imshow(uncertainty_norm, cmap="hot", alpha=0.7, interpolation="nearest")
                axes[1].set_title(f"Uncertainty Overlay (Step {step_index})\nMean: {uncertainty.mean():.4f}")
                axes[1].axis("off")
            else:
                # 如果没有不确定性，回退到K叠加
                axes[1].imshow(original_img)
                axes[1].imshow(k_norm, cmap="plasma", alpha=0.6, interpolation="nearest")
                axes[1].set_title(f"K Overlay (Step {step_index})")
                axes[1].axis("off")

            # 包含PAvPU信息的组合叠加
            axes[2].imshow(original_img)

            if uncertainty is not None:
                # 创建RGB叠加: Red=uncertainty, Green=lambda, Blue=k
                combined = np.zeros((*lambda_img.shape, 3))
                combined[:, :, 0] = uncertainty_norm  # Red for uncertainty
                combined[:, :, 1] = lambda_norm  # Green for lambda
                combined[:, :, 2] = k_norm  # Blue for k
                axes[2].imshow(combined, alpha=0.6, interpolation="nearest")

                # 如果可用，添加PAvPU文本
                pavpu_text = ""
                if "pixel_pavpu" in bndl_outputs and bndl_outputs["pixel_pavpu"] is not None:
                    pavpu_scores = bndl_outputs["pixel_pavpu"]
                    thresholds = [0.01, 0.05, 0.1]
                    pavpu_text = "\nPAvPU: "
                    for thresh, score in zip(thresholds, pavpu_scores, strict=False):
                        pavpu_text += f"p={thresh:.2f}:{score:.1f}% "

                axes[2].set_title(f"Multi-layer Overlay (Step {step_index}){pavpu_text}")
            else:
                # 后备组合叠加
                combined = np.zeros((*lambda_img.shape, 3))
                combined[:, :, 1] = lambda_norm  # Green for lambda
                combined[:, :, 0] = k_norm  # Red for k
                axes[2].imshow(combined, alpha=0.6, interpolation="nearest")
                axes[2].set_title(f"Combined Overlay (Step {step_index})")

            axes[2].axis("off")

        except Exception as e:
            logging.warning(f"Failed to plot parameter and uncertainty overlays: {e}")
            # 回退到常规参数叠加
            self.viz_utils.plot_parameter_overlays(axes, original_img, lambda_img, k_img, step_index)

    def plot_global_parameters_in_layout(self, axes, bndl_outputs: Dict[str, Any], step_index: int) -> None:
        """在统一布局中绘制全局权重参数"""
        try:
            lambda_w = bndl_outputs["wei_lambda_w"].detach().cpu().numpy()
            k_w = (1.0 / (bndl_outputs["inv_k_w"] + 1e-6)).detach().cpu().numpy()

            if len(lambda_w.shape) == 3:  # [B, K, C']
                lambda_w_vis = lambda_w[0]  # 使用第一个批次
                k_w_vis = k_w[0] if len(k_w.shape) == 3 else k_w[0:1]

                # Lambda_w热图
                im1 = axes[0].imshow(lambda_w_vis, cmap="viridis", interpolation="nearest", aspect="auto")
                axes[0].set_title(f"Global Lambda_w (Step {step_index})\nMean: {lambda_w_vis.mean():.4f}")
                axes[0].set_xlabel("Feature Dimension")
                axes[0].set_ylabel("Mask Token")
                plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

                # K_w热图
                im2 = axes[1].imshow(k_w_vis, cmap="plasma", interpolation="nearest", aspect="auto")
                axes[1].set_title(f"Global K_w (Step {step_index})\nMean: {k_w_vis.mean():.4f}")
                axes[1].set_xlabel("Feature Dimension" if len(k_w_vis.shape) == 2 and k_w_vis.shape[1] > 1 else "Single Value")
                axes[1].set_ylabel("Mask Token")
                plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

                # 全局参数统计
                axes[2].text(
                    0.5,
                    0.5,
                    f"Global Parameters Summary:\n\nLambda_w:\nMean: {lambda_w_vis.mean():.4f}\nStd: {lambda_w_vis.std():.4f}\n\nK_w:\nMean: {k_w_vis.mean():.4f}\nStd: {k_w_vis.std():.4f}",
                    ha="center",
                    va="center",
                    transform=axes[2].transAxes,
                    fontsize=10,
                )
                axes[2].set_title(f"Global Parameters Stats (Step {step_index})")
                axes[2].axis("off")

            else:
                # 处理其他形状
                for i in range(3):
                    axes[i].text(0.5, 0.5, f"Global Parameters\nShape: {lambda_w.shape}\nNot visualized", ha="center", va="center", transform=axes[i].transAxes, fontsize=10)
                    axes[i].set_title(f"Global Params {i + 1} (Step {step_index})")
                    axes[i].axis("off")

        except Exception as e:
            logging.warning(f"Failed to plot global parameters in layout: {e}")
            for i in range(3):
                axes[i].text(0.5, 0.5, "Global Parameters\nVisualization\nFailed", ha="center", va="center", transform=axes[i].transAxes)
                axes[i].set_title("Error")
                axes[i].axis("off")

    def plot_uncertainty_visualization(self, axes, bndl_outputs: Dict[str, Any], step_index: int) -> None:
        """绘制不确定性和PAvPU可视化"""
        try:
            if "pixel_uncertainty" in bndl_outputs and bndl_outputs["pixel_uncertainty"] is not None:
                uncertainty = bndl_outputs["pixel_uncertainty"].detach().cpu().numpy()

                if len(uncertainty.shape) == 4:  # [B, H, W, C]
                    uncertainty_vis = uncertainty[0].mean(axis=-1)  # 跨通道平均
                elif len(uncertainty.shape) == 3:  # [B, H, W]
                    uncertainty_vis = uncertainty[0]
                else:
                    uncertainty_vis = uncertainty

                # 不确定性热图
                im1 = axes[0].imshow(uncertainty_vis, cmap="hot", interpolation="nearest")
                axes[0].set_title(f"Pixel Uncertainty (Step {step_index})\nMean: {uncertainty_vis.mean():.4f}")
                axes[0].axis("off")
                plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

                # PAvPU可视化
                if "pixel_pavpu" in bndl_outputs and bndl_outputs["pixel_pavpu"] is not None:
                    pavpu_scores = bndl_outputs["pixel_pavpu"]
                    thresholds = [0.01, 0.05, 0.1]

                    # PAvPU柱状图
                    bars = axes[1].bar(range(len(thresholds)), pavpu_scores, color=["lightblue", "skyblue", "deepskyblue"], alpha=0.8)
                    axes[1].set_xlabel("Uncertainty Threshold")
                    axes[1].set_ylabel("PAvPU Score (%)")
                    axes[1].set_title(f"PAvPU Scores (Step {step_index})")
                    axes[1].set_xticks(range(len(thresholds)))
                    axes[1].set_xticklabels([f"{t:.2f}" for t in thresholds])

                    # 在柱子上添加值标签
                    for bar, score in zip(bars, pavpu_scores, strict=False):
                        height = bar.get_height()
                        axes[1].text(bar.get_x() + bar.get_width() / 2.0, height + 0.5, f"{score:.1f}%", ha="center", va="bottom", fontsize=9)
                else:
                    # 如果没有PAvPU，显示不确定性直方图
                    axes[1].hist(uncertainty_vis.flatten(), bins=50, alpha=0.7, color="orange")
                    axes[1].set_title(f"Uncertainty Distribution (Step {step_index})")
                    axes[1].set_xlabel("Uncertainty Value")
                    axes[1].set_ylabel("Frequency")

                # 包含PAvPU的组合统计
                stats_text = f"Uncertainty Summary:\nMean: {uncertainty_vis.mean():.4f}\nStd: {uncertainty_vis.std():.4f}\nMin: {uncertainty_vis.min():.4f}\nMax: {uncertainty_vis.max():.4f}"

                if "pixel_pavpu" in bndl_outputs and bndl_outputs["pixel_pavpu"] is not None:
                    pavpu_scores = bndl_outputs["pixel_pavpu"]
                    stats_text += "\n\nPAvPU Scores:\n"
                    for thresh, score in zip([0.01, 0.05, 0.1], pavpu_scores, strict=False):
                        stats_text += f"p={thresh:.2f}: {score:.1f}%\n"

                axes[2].text(0.5, 0.5, stats_text, ha="center", va="center", transform=axes[2].transAxes, fontsize=9)
                axes[2].set_title(f"Statistics (Step {step_index})")
                axes[2].axis("off")

            else:
                # 没有不确定性数据可用
                for i in range(3):
                    axes[i].text(0.5, 0.5, "No Uncertainty\nData Available", ha="center", va="center", transform=axes[i].transAxes, fontsize=10)
                    axes[i].set_title(f"Uncertainty {i + 1} (Step {step_index})")
                    axes[i].axis("off")

        except Exception as e:
            logging.warning(f"Failed to plot uncertainty visualization: {e}")
            for i in range(3):
                axes[i].text(0.5, 0.5, "Uncertainty\nVisualization\nFailed", ha="center", va="center", transform=axes[i].transAxes)
                axes[i].set_title("Error")
                axes[i].axis("off")

    def plot_correlation_analysis(self, axes, bndl_outputs: Dict[str, Any], step_index: int, batch: Any, outputs_for_vis: Optional[Dict[str, Any]] = None) -> None:
        """计算IoU、DICE和掩码准确率指标，并绘制它们与不确定性值的相关性"""
        try:
            # 提取预测和目标
            pred_masks = None
            gt_masks = None
            uncertainty = None

            # 从bndl_outputs或outputs_for_vis获取预测
            if "masks_bndl_raw" in bndl_outputs and bndl_outputs["masks_bndl_raw"] is not None:
                pred_logits = bndl_outputs["masks_bndl_raw"]
            elif "mean_pixel_logits" in bndl_outputs and bndl_outputs["mean_pixel_logits"] is not None:
                pred_logits = bndl_outputs["mean_pixel_logits"]
            elif outputs_for_vis is not None and "masks" in outputs_for_vis:
                pred_logits = outputs_for_vis["masks"]
            else:
                # 没有可用的预测
                for i in range(3):
                    axes[i].text(0.5, 0.5, "No Predictions\nAvailable", ha="center", va="center", transform=axes[i].transAxes)
                    axes[i].set_title(f"Correlation Analysis {i + 1}")
                    axes[i].axis("off")
                return

            # 从batch获取真实掩码
            if hasattr(batch, "masks") and batch.masks is not None:
                gt_masks = batch.masks
            else:
                # 没有可用的真实标签
                for i in range(3):
                    axes[i].text(0.5, 0.5, "No Ground Truth\nAvailable", ha="center", va="center", transform=axes[i].transAxes)
                    axes[i].set_title(f"Correlation Analysis {i + 1}")
                    axes[i].axis("off")
                return

            # 获取不确定性值
            if "pixel_uncertainty" in bndl_outputs and bndl_outputs["pixel_uncertainty"] is not None:
                uncertainty = bndl_outputs["pixel_uncertainty"].detach().cpu()
            else:
                # 没有可用的不确定性
                for i in range(3):
                    axes[i].text(0.5, 0.5, "No Uncertainty\nData Available", ha="center", va="center", transform=axes[i].transAxes)
                    axes[i].set_title(f"Correlation Analysis {i + 1}")
                    axes[i].axis("off")
                return

            # 转换为张量并确保正确的格式
            if hasattr(pred_logits, "detach"):
                pred_logits = pred_logits.detach().cpu()
            if hasattr(gt_masks, "detach"):
                gt_masks = gt_masks.detach().cpu()

            # 处理不同的张量形状并转换为 [B, H, W, K] 格式
            pred_logits = self.metric_calc.normalize_tensor_format(pred_logits, "predictions")
            gt_masks = self.metric_calc.normalize_tensor_format(gt_masks, "targets")

            # 修复：对于相关性分析，uncertainty应该保持多通道结构
            # 不要跨通道平均，这样每个像素位置都有不同的uncertainty值
            if len(uncertainty.shape) == 4:  # [B, H, W, C]
                # 保持多通道结构，不进行平均
                pass
            elif len(uncertainty.shape) == 3:  # [B, H, W]
                # 保持3D结构，不添加通道维度
                pass
            else:
                # 其他情况，尝试标准化
                uncertainty = self.metric_calc.normalize_tensor_format(uncertainty, "uncertainty")

            if pred_logits is None or gt_masks is None or uncertainty is None:
                for i in range(3):
                    axes[i].text(0.5, 0.5, "Format Error\nCheck Logs", ha="center", va="center", transform=axes[i].transAxes)
                    axes[i].set_title(f"Correlation Analysis {i + 1}")
                    axes[i].axis("off")
                return

            # 确保批次维度首先匹配
            min_batch = min(pred_logits.shape[0], gt_masks.shape[0], uncertainty.shape[0])
            pred_logits = pred_logits[:min_batch]
            gt_masks = gt_masks[:min_batch]
            uncertainty = uncertainty[:min_batch]

            # 确保空间维度匹配
            pred_logits, gt_masks, uncertainty = self.metric_calc.align_spatial_dimensions(pred_logits, gt_masks, uncertainty)

            if pred_logits is None or gt_masks is None or uncertainty is None:
                for i in range(3):
                    axes[i].text(0.5, 0.5, "Alignment\nFailed", ha="center", va="center", transform=axes[i].transAxes)
                    axes[i].set_title(f"Correlation Analysis {i + 1}")
                    axes[i].axis("off")
                return

            # 计算类似于loss_fns.py的指标
            iou_scores = self.metric_calc.calculate_iou_metric(pred_logits, gt_masks)
            dice_scores = self.metric_calc.calculate_dice_metric(pred_logits, gt_masks)
            mask_acc = self.metric_calc.calculate_mask_accuracy(pred_logits, gt_masks)

            if iou_scores is None or dice_scores is None or mask_acc is None:
                for i in range(3):
                    axes[i].text(0.5, 0.5, "Metric Calculation\nFailed", ha="center", va="center", transform=axes[i].transAxes)
                    axes[i].set_title(f"Correlation Analysis {i + 1}")
                    axes[i].axis("off")
                return

            # 展平用于相关性分析
            uncertainty_flat = uncertainty.flatten().numpy()
            iou_flat = iou_scores.flatten().numpy()
            dice_flat = dice_scores.flatten().numpy()
            acc_flat = mask_acc.flatten().numpy()

            # 确保所有数组具有相同的大小
            min_size = min(len(uncertainty_flat), len(iou_flat), len(dice_flat), len(acc_flat))
            uncertainty_flat = uncertainty_flat[:min_size]
            iou_flat = iou_flat[:min_size]
            dice_flat = dice_flat[:min_size]
            acc_flat = acc_flat[:min_size]

            # 移除任何无效值
            valid_mask = ~(
                np.isnan(uncertainty_flat)
                | np.isnan(iou_flat)
                | np.isnan(dice_flat)
                | np.isnan(acc_flat)
                | np.isinf(uncertainty_flat)
                | np.isinf(iou_flat)
                | np.isinf(dice_flat)
                | np.isinf(acc_flat)
            )

            if np.sum(valid_mask) < 10:  # 至少需要10个有效点
                for i in range(3):
                    axes[i].text(0.5, 0.5, "Insufficient\nValid Data", ha="center", va="center", transform=axes[i].transAxes)
                    axes[i].set_title(f"Correlation Analysis {i + 1}")
                    axes[i].axis("off")
                return

            uncertainty_valid = uncertainty_flat[valid_mask]
            iou_valid = iou_flat[valid_mask]
            dice_valid = dice_flat[valid_mask]
            acc_valid = acc_flat[valid_mask]

            # 图1: IoU vs Uncertainty
            self.viz_utils.plot_metric_uncertainty_correlation(axes[0], uncertainty_valid, iou_valid, "IoU vs Uncertainty", "Uncertainty", "IoU Score", step_index)

            # 图2: DICE vs Uncertainty
            self.viz_utils.plot_metric_uncertainty_correlation(axes[1], uncertainty_valid, dice_valid, "DICE vs Uncertainty", "Uncertainty", "DICE Score", step_index)

            # 图3: Mask Accuracy vs Uncertainty
            self.viz_utils.plot_metric_uncertainty_correlation(axes[2], uncertainty_valid, acc_valid, "Mask Accuracy vs Uncertainty", "Uncertainty", "Mask Accuracy", step_index)

            logging.info(f"Correlation analysis completed for step {step_index}")

        except Exception as e:
            logging.warning(f"Failed to plot correlation analysis: {e}")
            import traceback

            logging.warning(f"Traceback: {traceback.format_exc()}")
            for i in range(3):
                axes[i].text(0.5, 0.5, "Correlation\nAnalysis\nFailed", ha="center", va="center", transform=axes[i].transAxes)
                axes[i].set_title("Error")
                axes[i].axis("off")
