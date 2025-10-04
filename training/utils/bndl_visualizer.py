"""
BNDL可视化器模块
专门处理BNDL模型的可视化逻辑
"""

import logging
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .metric_calculator import MetricCalculator
    from .visualization_utils import VisualizationUtils
except ImportError:
    # Fallback for when imported as standalone module
    from metric_calculator import MetricCalculator
    from visualization_utils import VisualizationUtils


class BNDLVisualizer:
    """BNDL可视化器类"""

    def __init__(self):
        self.viz_utils = VisualizationUtils()
        self.metric_calc = MetricCalculator()

    def plot_parameter_and_uncertainty_overlays(self, axes, original_img: np.ndarray, lambda_img: np.ndarray, k_img: np.ndarray, bndl_outputs: dict[str, Any], step_index: int) -> None:
        """参数和不确定性叠加图，包含PAvPU可视化"""
        try:
            # 优先尝试使用 hyper_in/out_w 与预测加权后的有效参数图
            lambda_eff_np = None
            k_eff_np = None

            try:
                wei_lambda = bndl_outputs.get("wei_lambda")
                inv_k = bndl_outputs.get("inv_k")
                out_w = bndl_outputs.get("out_w")
                logits = bndl_outputs.get("masks_bndl_raw") if bndl_outputs.get("masks_bndl_raw") is not None else bndl_outputs.get("mean_pixel_logits")

                if wei_lambda is not None and inv_k is not None:
                    # 转为torch并在CPU上计算
                    wl = wei_lambda.detach().float().cpu()  # [B,H,W,C']
                    invk = inv_k.detach().float().cpu()
                    k_val = 1.0 / (invk + 1e-6)            # [B,H,W,C']

                    # 处理权重矩阵：优先使用 out_w；若无则尝试 hyper_in
                    w = None
                    if out_w is not None:
                        if hasattr(out_w, "detach"):
                            w = out_w.detach().float().cpu()
                        else:
                            # numpy -> torch
                            w = torch.from_numpy(out_w).float().cpu()
                    if w is None and "hyper_in" in bndl_outputs and bndl_outputs["hyper_in"] is not None:
                        w = bndl_outputs["hyper_in"].detach().float().cpu()  # [B,K,C']

                    lambda_eff = None
                    k_eff = None

                    if w is not None:
                        # 归一化并广播到像素
                        if w.ndim == 3:  # [B,K,C']
                            B, H, W, C = wl.shape
                            Bb, K, Cp = w.shape
                            if Bb == B and Cp == C:
                                w_sum = w.sum(dim=2, keepdim=True) + 1e-8
                                w_norm = w / w_sum  # [B,K,C']

                                # 展平像素做批矩阵乘：([B*H*W,C'] @ [B,C',K])
                                wl_flat = wl.view(B, H * W, C)
                                k_flat = k_val.view(B, H * W, C)
                                w_bt = w_norm.transpose(1, 2)  # [B,C',K]
                                lambda_w_flat = torch.bmm(wl_flat, w_bt)  # [B,HW,K]
                                k_w_flat = torch.bmm(k_flat, w_bt)        # [B,HW,K]
                                lambda_w = lambda_w_flat.view(B, H, W, K)
                                k_w = k_w_flat.view(B, H, W, K)
                            else:
                                lambda_w = None
                                k_w = None
                        elif w.ndim == 2:  # [C',K] 或 [K,C']
                            Ck0, Ck1 = w.shape
                            # 统一为 [C',K]
                            if Ck0 >= Ck1:
                                w_ck = w  # 可能已是 [C',K]
                            else:
                                w_ck = w.transpose(0, 1)  # [C',K]

                            # 安全检查wl的维度
                            if wl.ndim != 4:
                                logging.debug(f"wl has unexpected shape {wl.shape}, expected 4D [B,H,W,C], skipping weighted overlay")
                                lambda_w = None
                                k_w = None
                            else:
                                B, H, W, C = wl.shape
                                if w_ck.shape[0] == C:
                                    w_sum = w_ck.sum(dim=0, keepdim=True) + 1e-8  # [1,K]
                                    w_norm = w_ck / w_sum                           # [C',K]
                                    wl_flat = wl.view(B * H * W, C)
                                    k_flat = k_val.view(B * H * W, C)
                                    lambda_w_flat = torch.matmul(wl_flat, w_norm)  # [BHW,K]
                                    k_w_flat = torch.matmul(k_flat, w_norm)
                                    lambda_w = lambda_w_flat.view(B, H, W, -1)
                                    k_w = k_w_flat.view(B, H, W, -1)
                                else:
                                    lambda_w = None
                                    k_w = None
                        else:
                            lambda_w = None
                            k_w = None

                        # 沿K聚合：优先使用概率加权，其次赢家法，否则均匀平均
                        if lambda_w is not None and k_w is not None:
                            if logits is not None:
                                lg = logits.detach().float().cpu()  # [B,H,W,K]
                                p = torch.sigmoid(lg)
                                psum = p.sum(dim=-1, keepdim=True) + 1e-8
                                p_norm = p / psum
                                lambda_eff = (lambda_w * p_norm).sum(dim=-1)  # [B,H,W]
                                k_eff = (k_w * p_norm).sum(dim=-1)
                            else:
                                # 无logits，使用均匀
                                lambda_eff = lambda_w.mean(dim=-1)
                                k_eff = k_w.mean(dim=-1)

                    # 若无法使用权重，回退到传入的未加权图
                    if lambda_eff is not None and k_eff is not None:
                        lambda_eff_np = lambda_eff[0].numpy() if lambda_eff.ndim == 3 else lambda_eff.numpy()
                        k_eff_np = k_eff[0].numpy() if k_eff.ndim == 3 else k_eff.numpy()
            except Exception as e:
                logging.warning(f"Weighted overlay computation failed, fallback to raw params: {e}")

            # 选择用于显示的参数图
            if lambda_eff_np is not None and k_eff_np is not None:
                lambda_vis = lambda_eff_np
                k_vis = k_eff_np
            else:
                lambda_vis = lambda_img
                k_vis = k_img

            # 尺寸对齐
            if original_img is not None and (lambda_vis.shape != original_img.shape[:2]):
                lambda_vis = cv2.resize(lambda_vis, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                k_vis = cv2.resize(k_vis, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)

            lambda_norm, k_norm = self.viz_utils.normalize_parameters_robust(lambda_vis, k_vis)

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
                    thresholds = bndl_outputs.get("pavpu_thresholds", [0.01, 0.05, 0.1])
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

    def plot_global_parameters_in_layout(self, axes, bndl_outputs: dict[str, Any], step_index: int) -> None:
        """在统一布局中绘制全局权重参数"""
        try:
            lambda_w = bndl_outputs["wei_lambda_w"].detach().cpu().numpy()
            k_w = (1.0 / (bndl_outputs["inv_k_w"] + 1e-6)).detach().cpu().numpy()
            out_w = bndl_outputs.get("out_w")
            if out_w is not None and hasattr(out_w, "detach"):
                out_w = out_w.detach().cpu().numpy()

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

                # 第三列：可视化权重矩阵（如果提供 out_w），否则显示统计
                if out_w is not None:
                    # 统一为 [C', K] 以便显示：若为 [B, K, C'] 取 batch 0 并转置到 [C', K]
                    if len(out_w.shape) == 3:
                        ow = out_w[0]  # [K, C']
                        if ow.shape[0] != 1 and ow.shape[1] != 1:
                            ow = ow.transpose(1, 0)  # -> [C', K]
                    elif len(out_w.shape) == 2:
                        # 可能是 [C', K] 或 [K, C']，尽量转为 [C', K]
                        ow = out_w
                        if ow.shape[0] < ow.shape[1]:
                            # 假设 [K, C']，转置
                            ow = ow.transpose(1, 0)
                    else:
                        ow = None

                    if ow is not None:
                        im3 = axes[2].imshow(ow, cmap="magma", interpolation="nearest", aspect="auto")
                        axes[2].set_title(f"Used Weight Matrix (Step {step_index})\nShape: {ow.shape[0]}×{ow.shape[1]}")
                        axes[2].set_xlabel("Mask Token (K)")
                        axes[2].set_ylabel("Feature Dim (C')")
                        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
                    else:
                        axes[2].text(
                            0.5,
                            0.5,
                            "Weight matrix shape\nnot supported",
                            ha="center",
                            va="center",
                            transform=axes[2].transAxes,
                            fontsize=10,
                        )
                        axes[2].set_title(f"Weight Matrix (Step {step_index})")
                        axes[2].axis("off")
                else:
                    # 回退到全局参数统计
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

    def plot_uncertainty_visualization(self, axes, bndl_outputs: dict[str, Any], step_index: int) -> None:
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
                pavpu_scores = bndl_outputs.get("pixel_pavpu")
                thresholds = bndl_outputs.get("pavpu_thresholds", [0.01, 0.05, 0.1]) if pavpu_scores is not None else None

                if pavpu_scores is not None and thresholds is not None:
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

                if pavpu_scores is not None and thresholds is not None:
                    stats_text += "\n\nPAvPU Scores:\n"
                    for thresh, score in zip(thresholds, pavpu_scores, strict=False):
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
            # logging.warning(f"Failed to plot uncertainty visualization: {e}")
            for i in range(3):
                axes[i].text(0.5, 0.5, "Uncertainty\nVisualization\nFailed", ha="center", va="center", transform=axes[i].transAxes)
                axes[i].set_title("Error")
                axes[i].axis("off")

    def plot_correlation_analysis(self, axes, bndl_outputs: dict[str, Any], step_index: int, batch: Any, outputs_for_vis: dict[str, Any] | None = None) -> None:
        """计算IoU、DICE和掩码准确率指标，并绘制它们与不确定性值的相关性"""
        try:
            # 提取预测和目标
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

            # logging.info(f"Correlation analysis completed for step {step_index}")

        except Exception as e:
            logging.warning(f"Failed to plot correlation analysis: {e}")
            import traceback

            logging.warning(f"Traceback: {traceback.format_exc()}")
            for i in range(3):
                axes[i].text(0.5, 0.5, "Correlation\nAnalysis\nFailed", ha="center", va="center", transform=axes[i].transAxes)
                axes[i].set_title("Error")
                axes[i].axis("off")
