import os
import logging
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .metric_calculator import MetricCalculator
from .visualization_utils import VisualizationUtils
from BNDL.BNDL_upload.ViT_Sparse.utils.bndl import calculate_nll_from_logits


def _dilate_mask(mask: torch.Tensor, kernel_size: int = 10) -> torch.Tensor:
    """
    对二值mask进行形态学膨胀操作，用于扩展前景区域

    Args:
        mask: [H, W] 二值mask张量
        kernel_size: 膨胀核大小（像素），推荐5-15

    Returns:
        dilated_mask: [H, W] 膨胀后的二值mask
    """
    if kernel_size <= 0:
        return mask

    # 使用max_pool2d实现膨胀：将mask reshape为[1, 1, H, W]
    mask_4d = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]

    # max_pool2d with padding实现膨胀效果
    # padding = (kernel_size - 1) // 2 保证输出尺寸不变
    padding = (kernel_size - 1) // 2
    dilated = F.max_pool2d(mask_4d, kernel_size=kernel_size, stride=1, padding=padding)

    # 转回[H, W]并转为bool
    dilated_mask = dilated.squeeze(0).squeeze(0) > 0.5

    return dilated_mask


class DistributedDatasetEvaluator:
    
    def __init__(self, save_dir: str, distributed: bool = False, rank: int = 0, world_size: int = 1,
                 foreground_dilation: int = 10, per_pixel_statistics: bool = True):
        """
        初始化分布式数据集评估器

        Args:
            save_dir: 结果保存目录
            distributed: 是否启用分布式训练
            rank: 当前进程的rank
            world_size: 总进程数
            foreground_dilation: 前景区域膨胀半径（像素），0表示不膨胀
            per_pixel_statistics: 是否使用像素级统计（vs 图片级）
        """
        self.save_dir = save_dir
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0
        self.foreground_dilation = foreground_dilation
        self.per_pixel_statistics = per_pixel_statistics

        self.metric_calculator = MetricCalculator()
        self.viz_utils = VisualizationUtils()

        if per_pixel_statistics:
            # 存储每个像素的指标值（每个数据点代表一个像素）
            self.pixel_uncertainties = []  # 存储前景区域每个像素的uncertainty值
            self.pixel_ious = []          # 存储前景区域每个像素的准确性值
            self.pixel_dices = []         # 存储前景区域每个像素的DICE相关值
            self.pixel_accuracies = []    # 存储前景区域每个像素的accuracy值
            self.pixel_nlls = []          # 存储前景区域每个像素的NLL值
        else:
            # 向后兼容：存储每张图片的标量指标值
            self.image_uncertainties = []  # 存储每张图片的平均uncertainty值
            self.image_ious = []          # 存储每张图片的平均IoU值
            self.image_dices = []         # 存储每张图片的平均DICE值
            self.image_accuracies = []    # 存储每张图片的平均accuracy值
            self.image_nlls = []          # 存储每张图片的平均NLL值（有监督不确定性）

        # 存储最终结果
        self.correlation_results = {}

        # 分布式训练相关
        if self.distributed:
            self._setup_distributed()

        # 只在主进程创建保存目录
        if self.is_main_process:
            os.makedirs(save_dir, exist_ok=True)
            logging.info(f"Dataset evaluator initialized: per_pixel={per_pixel_statistics}, dilation={foreground_dilation}")
    
    def _setup_distributed(self):
        """设置分布式训练相关配置"""
        if not dist.is_initialized():
            logging.warning("Distributed training not initialized, falling back to single GPU mode")
            self.distributed = False
            return
        
        logging.info(f"Initializing distributed evaluator: rank {self.rank}/{self.world_size}")
    
    def add_batch_data(self, 
                       uncertainty: torch.Tensor,
                       pred_logits: torch.Tensor,
                       gt_masks: torch.Tensor) -> None:
        """添加一个批次的数据用于后续评估，为每张图片计算标量指标"""
        try:
            # 数据验证
            if uncertainty is None or pred_logits is None or gt_masks is None:
                logging.warning("One or more inputs are None, skipping batch")
                return
            
            # 检查张量形状
            if uncertainty.numel() == 0 or pred_logits.numel() == 0 or gt_masks.numel() == 0:
                logging.warning("One or more inputs have zero elements, skipping batch")
                return
            
            # 确保张量在正确的设备上
            device = uncertainty.device
            
            # 标准化格式
            uncertainty_norm = self.metric_calculator.normalize_tensor_format(uncertainty, "uncertainty")
            pred_norm = self.metric_calculator.normalize_tensor_format(pred_logits, "pred_logits")
            gt_norm = self.metric_calculator.normalize_tensor_format(gt_masks, "gt_masks")
            
            if uncertainty_norm is None or pred_norm is None or gt_norm is None:
                logging.warning("Failed to normalize batch data, skipping")
                return
            
            # 对齐空间维度
            pred_aligned, gt_aligned, uncertainty_aligned = self.metric_calculator.align_spatial_dimensions(
                pred_norm, gt_norm, uncertainty_norm
            )
            
            if pred_aligned is None or gt_aligned is None or uncertainty_aligned is None:
                logging.warning("Failed to align spatial dimensions, skipping")
                return
            
            # 获取batch size
            B = pred_aligned.shape[0]

            if self.per_pixel_statistics:
                # 像素级统计：收集前景扩展区域内的所有像素
                total_pixels = 0
                for i in range(B):
                    # 提取单张图片的数据
                    single_uncertainty = uncertainty_aligned[i]  # [H, W, K]
                    single_pred = pred_aligned[i]                # [H, W, K]
                    single_gt = gt_aligned[i]                    # [H, W, K]

                    # 计算像素级指标
                    pixel_unc, pixel_acc, pixel_iou, pixel_dice, pixel_nll = \
                        self._calculate_pixel_wise_metrics(single_uncertainty, single_pred, single_gt)

                    # 存储所有前景区域像素（转换为CPU并转为list以节省内存）
                    if pixel_unc.numel() > 0:
                        self.pixel_uncertainties.extend(pixel_unc.detach().cpu().tolist())
                        self.pixel_accuracies.extend(pixel_acc.detach().cpu().tolist())
                        self.pixel_ious.extend(pixel_iou.detach().cpu().tolist())
                        self.pixel_dices.extend(pixel_dice.detach().cpu().tolist())
                        self.pixel_nlls.extend(pixel_nll.detach().cpu().tolist())
                        total_pixels += pixel_unc.numel()

                if self.rank == 0:  # 只在主进程记录日志
                    logging.info(f"Added {B} images ({total_pixels} foreground pixels) to evaluation data")
            else:
                # 图片级统计（向后兼容）：对每张图片计算标量指标
                for i in range(B):
                    # 提取单张图片的数据
                    single_uncertainty = uncertainty_aligned[i]  # [H, W, K]
                    single_pred = pred_aligned[i]                # [H, W, K]
                    single_gt = gt_aligned[i]                    # [H, W, K]

                    # 计算单张图片的标量指标
                    iou_scalar = self._calculate_single_image_iou_scalar(single_pred, single_gt)
                    dice_scalar = self._calculate_single_image_dice_scalar(single_pred, single_gt)
                    accuracy_scalar = self._calculate_single_image_accuracy_scalar(single_pred, single_gt)
                    uncertainty_scalar = self._calculate_single_image_uncertainty_scalar(single_uncertainty, single_pred, single_gt)

                    # 计算NLL（复用BNDL模块的方法）
                    nll_scalar = calculate_nll_from_logits(single_pred, single_gt, foreground_only=True)

                    # 存储单张图片的标量指标（转换为CPU以节省GPU内存）
                    self.image_uncertainties.append(uncertainty_scalar.detach().cpu())
                    self.image_ious.append(iou_scalar.detach().cpu())
                    self.image_dices.append(dice_scalar.detach().cpu())
                    self.image_accuracies.append(accuracy_scalar.detach().cpu())
                    self.image_nlls.append(nll_scalar.detach().cpu())

                if self.rank == 0:  # 只在主进程记录日志
                    logging.info(f"Added {B} images to evaluation data")
            
        except Exception as e:
            logging.warning(f"Failed to add batch data: {e}")
            import traceback
            logging.warning(f"Traceback: {traceback.format_exc()}")
    
    def _calculate_single_image_iou_scalar(self, pred_logits: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """计算单张图片的平均IoU值（标量）"""
        try:
            # 确保输入形状匹配
            if pred_logits.shape != gt_masks.shape:
                logging.warning(f"Shape mismatch in single image IoU: pred {pred_logits.shape} vs gt {gt_masks.shape}")
                return torch.tensor(0.0)
            
            # 对logits应用 > 0 阈值
            pred_binary = pred_logits > 0
            gt_binary = gt_masks > 0
            
            # 计算IoU：|A ∩ B| / |A ∪ B|
            intersection = (pred_binary & gt_binary).float()
            union = (pred_binary | gt_binary).float()
            
            # 对空间维度和通道维度求和，得到整张图片的IoU
            intersection_sum = intersection.sum()  # 标量
            union_sum = union.sum()  # 标量
            
            # 避免除零，计算整张图片的IoU
            iou = intersection_sum / (union_sum + 1e-8)  # 标量
            
            # 确保值在[0, 1]范围内
            iou = torch.clamp(iou, 0.0, 1.0)
            
            return iou
            
        except Exception as e:
            logging.warning(f"Failed to calculate single image IoU scalar: {e}")
            return torch.tensor(0.0)
    
    def _calculate_single_image_dice_scalar(self, pred_logits: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """计算单张图片的平均DICE值（标量）"""
        try:
            if pred_logits.shape != gt_masks.shape:
                logging.warning(f"Shape mismatch in single image DICE: pred {pred_logits.shape} vs gt {gt_masks.shape}")
                return torch.tensor(0.0)
            
            # 先应用sigmoid
            pred_probs = torch.sigmoid(pred_logits)
            
            # 计算DICE：2 * |A ∩ B| / (|A| + |B|)
            numerator = 2 * (pred_probs * gt_masks.float()).sum()  # 标量
            denominator = pred_probs.sum() + gt_masks.float().sum()  # 标量
            
            # 避免除零，计算整张图片的DICE
            dice = numerator / (denominator + 1e-8)  # 标量
            
            # 处理边界情况
            gt_empty = (gt_masks.float().sum() < 1e-8)
            pred_empty = (pred_probs.sum() < 1e-8)
            
            # 当两者都为空时，DICE = 1；当只有预测为空时，DICE = 0
            if gt_empty and pred_empty:
                dice = torch.tensor(1.0)
            elif gt_empty and not pred_empty:
                dice = torch.tensor(0.0)
            
            # 确保值在[0, 1]范围内
            dice = torch.clamp(dice, 0.0, 1.0)
            
            return dice
            
        except Exception as e:
            logging.warning(f"Failed to calculate single image DICE scalar: {e}")
            return torch.tensor(0.0)
    
    def _calculate_single_image_accuracy_scalar(self, pred_logits: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """计算单张图片的平均准确率值（标量）；优先在前景区域上统计，避免背景主导"""
        try:
            if pred_logits.shape != gt_masks.shape:
                logging.warning(f"Shape mismatch in single image accuracy: pred {pred_logits.shape} vs gt {gt_masks.shape}")
                return torch.tensor(0.0)
            
            # 对logits应用 > 0 阈值，与 IoU 一致
            pred_binary = pred_logits > 0
            gt_binary = gt_masks > 0
            
            # 前景掩膜（任一通道为真）
            fg_mask = gt_binary.any(dim=-1)  # [H, W]

            # 计算准确率：(TP + TN) / (TP + TN + FP + FN)
            correct_predictions = (pred_binary == gt_binary).float()  # [H, W, K]

            if fg_mask.any():
                # 仅在前景区域统计，避免背景像素占比过大
                correct_sum = correct_predictions[fg_mask].sum()
                total_pixels = correct_predictions[fg_mask].numel()
            else:
                # 无前景时退化为全图
                correct_sum = correct_predictions.sum()
                total_pixels = correct_predictions.numel()

            # 计算准确率
            pixel_acc = correct_sum / (total_pixels + 1e-8)  # 标量
            
            # 确保值在[0, 1]范围内
            pixel_acc = torch.clamp(pixel_acc, 0.0, 1.0)
            
            return pixel_acc
            
        except Exception as e:
            logging.warning(f"Failed to calculate single image accuracy scalar: {e}")
            return torch.tensor(0.0)
    
    def _calculate_single_image_uncertainty_scalar(self, uncertainty: torch.Tensor, pred_logits: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        """计算单张图片的不确定性标量，直接使用传入的不确定性并在前景区域取中位数。"""
        # 统一到 [H, W]
        if uncertainty.ndim == 3:
            pixel_uncertainty = uncertainty.mean(dim=-1)
        else:
            pixel_uncertainty = uncertainty
        pixel_uncertainty = pixel_uncertainty.float()

        # 前景掩膜 [H, W]
        fg_mask = (gt_masks > 0)
        if fg_mask.ndim == 3:
            fg_mask = fg_mask.any(dim=-1)

        values = pixel_uncertainty[fg_mask] if fg_mask.any() else pixel_uncertainty.reshape(-1)
        return values.median()

    def _calculate_pixel_wise_metrics(self, uncertainty: torch.Tensor, pred_logits: torch.Tensor,
                                      gt_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算像素级指标，返回前景扩展区域内每个像素的值

        Args:
            uncertainty: [H, W, K] or [H, W] 不确定性
            pred_logits: [H, W, K] 预测logits
            gt_masks: [H, W, K] GT masks

        Returns:
            pixel_uncertainties: [N] 前景区域像素的uncertainty
            pixel_accuracies: [N] 前景区域像素的accuracy（二值正确性）
            pixel_ious: [N] 前景区域像素的IoU贡献
            pixel_dices: [N] 前景区域像素的DICE贡献
            pixel_nlls: [N] 前景区域像素的NLL值
        """
        try:
            # 1. 构建前景mask并膨胀
            fg_mask = (gt_masks > 0)
            if fg_mask.ndim == 3:
                fg_mask = fg_mask.any(dim=-1)  # [H, W]

            # 膨胀前景区域
            if self.foreground_dilation > 0:
                fg_mask_expanded = _dilate_mask(fg_mask, kernel_size=self.foreground_dilation)
            else:
                fg_mask_expanded = fg_mask

            # 2. 处理uncertainty到[H, W]
            if uncertainty.ndim == 3:
                pixel_uncertainty = uncertainty.mean(dim=-1)  # [H, W]
            else:
                pixel_uncertainty = uncertainty
            pixel_uncertainty = pixel_uncertainty.float()

            # 3. 计算像素级accuracy（二值正确性）
            pred_binary_bool = (pred_logits > 0)  # [H, W, K] - keep as bool
            gt_binary_bool = (gt_masks > 0)       # [H, W, K] - keep as bool
            pred_binary = pred_binary_bool.float()  # [H, W, K] - convert to float for comparison
            gt_binary = gt_binary_bool.float()       # [H, W, K]
            pixel_correct = (pred_binary == gt_binary).all(dim=-1).float()  # [H, W] 所有通道都正确

            # 4. 计算像素级IoU贡献（intersection / union per pixel across channels）
            intersection = (pred_binary_bool & gt_binary_bool).float().sum(dim=-1)  # [H, W]
            union = (pred_binary_bool | gt_binary_bool).float().sum(dim=-1)  # [H, W]
            pixel_iou = intersection / (union + 1e-8)  # [H, W]
            pixel_iou = torch.clamp(pixel_iou, 0.0, 1.0)

            # 5. 计算像素级DICE贡献
            pred_probs = torch.sigmoid(pred_logits)  # [H, W, K]
            numerator = 2 * (pred_probs * gt_binary).sum(dim=-1)  # [H, W]
            denominator = pred_probs.sum(dim=-1) + gt_binary.sum(dim=-1)  # [H, W]
            pixel_dice = numerator / (denominator + 1e-8)
            pixel_dice = torch.clamp(pixel_dice, 0.0, 1.0)

            # 6. 计算像素级NLL（使用logits和gt的交叉熵）
            # 对每个像素的K个通道计算平均NLL
            pred_probs_safe = torch.clamp(pred_probs, 1e-7, 1 - 1e-7)
            pixel_nll_per_channel = - (gt_binary * torch.log(pred_probs_safe) +
                                       (1 - gt_binary) * torch.log(1 - pred_probs_safe))  # [H, W, K]
            pixel_nll = pixel_nll_per_channel.mean(dim=-1)  # [H, W]

            # 7. 提取前景扩展区域的像素
            if fg_mask_expanded.any():
                extracted_uncertainty = pixel_uncertainty[fg_mask_expanded]
                extracted_accuracy = pixel_correct[fg_mask_expanded]
                extracted_iou = pixel_iou[fg_mask_expanded]
                extracted_dice = pixel_dice[fg_mask_expanded]
                extracted_nll = pixel_nll[fg_mask_expanded]
            else:
                # 如果没有前景，返回空张量
                extracted_uncertainty = torch.tensor([], dtype=torch.float32)
                extracted_accuracy = torch.tensor([], dtype=torch.float32)
                extracted_iou = torch.tensor([], dtype=torch.float32)
                extracted_dice = torch.tensor([], dtype=torch.float32)
                extracted_nll = torch.tensor([], dtype=torch.float32)

            return extracted_uncertainty, extracted_accuracy, extracted_iou, extracted_dice, extracted_nll

        except Exception as e:
            logging.warning(f"Failed to calculate pixel-wise metrics: {e}")
            import traceback
            logging.warning(f"Traceback: {traceback.format_exc()}")
            # 返回空张量
            empty = torch.tensor([], dtype=torch.float32)
            return empty, empty, empty, empty, empty
    
    def _gather_distributed_data(self) -> tuple[list[float], list[float],
                                               list[float], list[float], list[float]]:
        """收集所有GPU进程的数据（使用NCCL-safe all_gather + padding）"""
        if not self.distributed or not dist.is_initialized():
            if self.per_pixel_statistics:
                # 像素级统计：直接返回列表
                return (self.pixel_uncertainties, self.pixel_ious,
                        self.pixel_dices, self.pixel_accuracies, self.pixel_nlls)
            else:
                # 图片级统计：将张量转为标量列表
                return (
                    [t.item() if hasattr(t, 'item') else t for t in self.image_uncertainties],
                    [t.item() if hasattr(t, 'item') else t for t in self.image_ious],
                    [t.item() if hasattr(t, 'item') else t for t in self.image_dices],
                    [t.item() if hasattr(t, 'item') else t for t in self.image_accuracies],
                    [t.item() if hasattr(t, 'item') else t for t in self.image_nlls]
                )

        def gather_float_list(values: list[float]) -> list[float]:
            """聚合float列表（支持像素级和图片级）"""
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            local_vals = torch.tensor(values, device=device, dtype=torch.float32)
            local_len = torch.tensor([local_vals.numel()], device=device, dtype=torch.int64)

            # 收集各rank长度
            len_list = [torch.zeros_like(local_len) for _ in range(self.world_size)]
            dist.all_gather(len_list, local_len)
            max_len = int(torch.stack(len_list).max().item())

            # padding到相同长度
            if local_vals.numel() < max_len:
                pad = torch.full((max_len - local_vals.numel(),), float("nan"), device=device, dtype=torch.float32)
                local_pad = torch.cat([local_vals, pad], dim=0)
            else:
                local_pad = local_vals

            # all_gather实际数据
            gathered = [torch.empty_like(local_pad) for _ in range(self.world_size)]
            dist.all_gather(gathered, local_pad)

            # 根据各rank真实长度去掉padding
            out: list[float] = []
            for r, tensor_r in enumerate(gathered):
                n = int(len_list[r].item())
                if n > 0:
                    out.extend(tensor_r[:n].detach().cpu().tolist())
            return out

        if self.per_pixel_statistics:
            # 像素级统计：直接聚合float列表
            all_uncertainties = gather_float_list(self.pixel_uncertainties)
            all_ious = gather_float_list(self.pixel_ious)
            all_dices = gather_float_list(self.pixel_dices)
            all_accuracies = gather_float_list(self.pixel_accuracies)
            all_nlls = gather_float_list(self.pixel_nlls)
        else:
            # 图片级统计：先转换再聚合
            all_uncertainties = gather_float_list([t.item() if hasattr(t, 'item') else float(t) for t in self.image_uncertainties])
            all_ious = gather_float_list([t.item() if hasattr(t, 'item') else float(t) for t in self.image_ious])
            all_dices = gather_float_list([t.item() if hasattr(t, 'item') else float(t) for t in self.image_dices])
            all_accuracies = gather_float_list([t.item() if hasattr(t, 'item') else float(t) for t in self.image_accuracies])
            all_nlls = gather_float_list([t.item() if hasattr(t, 'item') else float(t) for t in self.image_nlls])

        return all_uncertainties, all_ious, all_dices, all_accuracies, all_nlls

    def _calculate_correlation_from_lists(self, uncertainties: list[float],
                                          metrics: list[float],
                                          metric_name: str) -> Dict[str, float]:
        """
        从float列表计算相关性（支持像素级和图片级统计）

        Args:
            uncertainties: uncertainty值列表
            metrics: metric值列表
            metric_name: 指标名称

        Returns:
            相关性结果字典
        """
        if not uncertainties or not metrics:
            logging.warning(f"Empty data for {metric_name} correlation")
            return {}

        if len(uncertainties) != len(metrics):
            logging.warning(f"Length mismatch for {metric_name}: {len(uncertainties)} vs {len(metrics)}")
            return {}

        # 转换为numpy数组
        uncertainty_np = np.array(uncertainties, dtype=np.float32)
        metric_np = np.array(metrics, dtype=np.float32)

        # 移除无效值（nan/inf）
        valid_mask = np.isfinite(uncertainty_np) & np.isfinite(metric_np)
        uncertainty_valid = uncertainty_np[valid_mask]
        metric_valid = metric_np[valid_mask]

        if len(uncertainty_valid) < 2:
            logging.warning(f"Not enough valid data for {metric_name} correlation: {len(uncertainty_valid)}")
            return {}

        # 计算相关性
        correlation = np.corrcoef(uncertainty_valid, metric_valid)[0, 1]

        # 计算线性回归
        slope, intercept = np.polyfit(uncertainty_valid, metric_valid, 1)

        # 计算统计信息
        results = {
            'correlation': float(correlation),
            'slope': float(slope),
            'intercept': float(intercept),
            'uncertainty_mean': float(np.mean(uncertainty_valid)),
            'uncertainty_std': float(np.std(uncertainty_valid)),
            'metric_mean': float(np.mean(metric_valid)),
            'metric_std': float(np.std(metric_valid)),
            'num_valid_points': int(np.sum(valid_mask)),
            'total_points': len(uncertainty_np),
            # 保存原始数据用于绘图（采样以节省内存，如果数据量很大）
            'uncertainty_data': self._sample_data_for_plotting(uncertainty_valid),
            'metric_data': self._sample_data_for_plotting(metric_valid)
        }

        return results

    def _sample_data_for_plotting(self, data: np.ndarray, max_points: int = 10000) -> list[float]:
        """对数据进行采样以用于绘图，避免存储过多数据点"""
        if len(data) <= max_points:
            return data.tolist()
        else:
            # 随机采样
            indices = np.random.choice(len(data), max_points, replace=False)
            return data[indices].tolist()

    def evaluate_dataset_correlation(self) -> Dict[str, Dict[str, float]]:
        """评估整个数据集的指标与不确定性相关性"""
        try:
            # 检查是否有数据
            data_source = self.pixel_uncertainties if self.per_pixel_statistics else self.image_uncertainties
            if not data_source:
                logging.warning("No data available for evaluation")
                return {}

            data_unit = "pixels" if self.per_pixel_statistics else "images"
            if self.is_main_process:
                logging.info(f"Evaluating correlation for {len(data_source)} {data_unit}")

            # 收集分布式数据（返回float列表）
            all_uncertainties, all_ious, all_dices, all_accuracies, all_nlls = self._gather_distributed_data()

            # 只在主进程上进行评估
            if not self.is_main_process:
                return {}

            # 记录各指标列表长度用于调试
            logging.info(f"Gathered data lengths - Uncertainties: {len(all_uncertainties)}, "
                        f"IoU: {len(all_ious)}, DICE: {len(all_dices)}, "
                        f"Accuracy: {len(all_accuracies)}, NLL: {len(all_nlls)}")

            # 分别计算每个指标的相关性（使用numpy数组进行计算）
            correlation_results = {}

            # IoU相关性
            if all_ious:
                logging.info(f"Calculating IoU correlation with {len(all_ious)} {data_unit}")
                iou_corr = self._calculate_correlation_from_lists(
                    all_uncertainties, all_ious, 'IoU'
                )
                if iou_corr:
                    correlation_results['IoU'] = iou_corr
                    logging.info(f"IoU correlation calculated: {iou_corr.get('correlation', 'N/A'):.4f}")

            # DICE相关性
            if all_dices:
                logging.info(f"Calculating DICE correlation with {len(all_dices)} {data_unit}")
                dice_corr = self._calculate_correlation_from_lists(
                    all_uncertainties, all_dices, 'DICE'
                )
                if dice_corr:
                    correlation_results['DICE'] = dice_corr
                    logging.info(f"DICE correlation calculated: {dice_corr.get('correlation', 'N/A'):.4f}")

            # Accuracy相关性
            if all_accuracies:
                logging.info(f"Calculating Accuracy correlation with {len(all_accuracies)} {data_unit}")
                acc_corr = self._calculate_correlation_from_lists(
                    all_uncertainties, all_accuracies, 'Accuracy'
                )
                if acc_corr:
                    correlation_results['Accuracy'] = acc_corr
                    logging.info(f"Accuracy correlation calculated: {acc_corr.get('correlation', 'N/A'):.4f}")

            # NLL相关性（NLL本身就是不确定性度量，与uncertainty的相关性）
            if all_nlls:
                logging.info(f"Calculating NLL correlation with {len(all_nlls)} {data_unit}")
                nll_corr = self._calculate_correlation_from_lists(
                    all_uncertainties, all_nlls, 'NLL'
                )
                if nll_corr:
                    correlation_results['NLL'] = nll_corr
                    logging.info(f"NLL correlation calculated: {nll_corr.get('correlation', 'N/A'):.4f}")
            
            # 保存结果
            self.correlation_results = correlation_results
            
            # 记录汇总信息
            if correlation_results:
                logging.info("Dataset correlation evaluation completed: " + 
                           str(list(correlation_results.keys())))
                for metric_name, results in correlation_results.items():
                    correlation = results.get('correlation', 'N/A')
                    valid_points = results.get('num_valid_points', 'N/A')
                    logging.info(f"{metric_name}: correlation={correlation}, valid_points={valid_points}")
            
            return correlation_results
            
        except Exception as e:
            logging.warning(f"Failed to evaluate dataset correlation: {e}")
            import traceback
            logging.warning(f"Correlation evaluation traceback: {traceback.format_exc()}")
            return {}
    
    def create_dataset_correlation_visualization(self, 
                                               title: str = "Dataset Metric-Uncertainty Correlation Analysis",
                                               save_name: str = "dataset_correlation_analysis.png") -> str:
        """
        创建数据集相关性分析的可视化图表
        
        Args:
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        try:
            # 只在主进程上创建可视化
            if not self.is_main_process:
                return ""
            
            if not self.correlation_results:
                logging.warning("No correlation results available for visualization")
                return ""
            
            # 创建图形
            fig = plt.figure(figsize=(18, 12))
            
            # 使用可视化工具创建数据集相关性图
            self.viz_utils.plot_dataset_metric_uncertainty_correlation(
                fig, self.correlation_results, title
            )
            
            # 保存图表
            save_path = os.path.join(self.save_dir, save_name)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logging.info(f"Dataset correlation visualization saved: {save_path}")
            return save_path
            
        except Exception as e:
            logging.warning(f"Failed to create dataset correlation visualization: {e}")
            import traceback
            logging.warning(f"Traceback: {traceback.format_exc()}")
            return ""
    
    def save_correlation_results(self, save_name: str = "correlation_results.json") -> str:
        """
        保存相关性分析结果到JSON文件
        
        Args:
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        try:
            # 只在主进程上保存结果
            if not self.is_main_process:
                return ""
            
            if not self.correlation_results:
                logging.warning("No correlation results available for saving")
                return ""
            
            import json
            
            # 处理numpy数据类型，确保可以序列化
            serializable_results = {}
            for metric_name, results in self.correlation_results.items():
                serializable_results[metric_name] = {}
                for key, value in results.items():
                    if isinstance(value, (np.integer, np.floating)):
                        serializable_results[metric_name][key] = float(value)
                    elif isinstance(value, np.ndarray):
                        # 将numpy数组转换为Python列表
                        serializable_results[metric_name][key] = value.tolist()
                    elif isinstance(value, list):
                        serializable_results[metric_name][key] = [
                            float(x) if isinstance(x, (np.integer, np.floating)) else x 
                            for x in value
                        ]
                    else:
                        serializable_results[metric_name][key] = value
            
            save_path = os.path.join(self.save_dir, save_name)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Correlation results saved: {save_path}")
            logging.info(f"Saved metrics: {list(serializable_results.keys())}")
            return save_path
            
        except Exception as e:
            logging.warning(f"Failed to save correlation results: {e}")
            import traceback
            logging.warning(f"Traceback: {traceback.format_exc()}")
            return ""
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        获取汇总统计信息
        
        Returns:
            汇总统计字典
        """
        try:
            if not self.correlation_results:
                return {}
            
            summary = {
                'total_batches': len(self.image_uncertainties),
                'metrics_evaluated': list(self.correlation_results.keys()),
                'correlation_summary': {},
                'overall_statistics': {},
                'distributed_info': {
                    'distributed': self.distributed,
                    'rank': self.rank,
                    'world_size': self.world_size
                }
            }
            
            # 为每个指标计算汇总统计
            for metric_name, results in self.correlation_results.items():
                summary['correlation_summary'][metric_name] = {
                    'correlation': results.get('correlation', np.nan),
                    'slope': results.get('slope', np.nan),
                    'valid_points': results.get('num_valid_points', 0),
                    'total_points': results.get('total_points', 0)
                }
            
            # 计算整体统计
            correlations = [results.get('correlation', np.nan) for results in self.correlation_results.values()]
            valid_correlations = [c for c in correlations if not np.isnan(c)]
            
            if valid_correlations:
                summary['overall_statistics'] = {
                    'mean_correlation': np.mean(valid_correlations),
                    'std_correlation': np.std(valid_correlations),
                    'min_correlation': np.min(valid_correlations),
                    'max_correlation': np.max(valid_correlations)
                }
            
            return summary
            
        except Exception as e:
            logging.warning(f"Failed to get summary statistics: {e}")
            return {}
    
    def reset(self) -> None:
        """重置评估器状态，清除所有数据"""
        if self.per_pixel_statistics:
            self.pixel_uncertainties.clear()
            self.pixel_ious.clear()
            self.pixel_dices.clear()
            self.pixel_accuracies.clear()
            self.pixel_nlls.clear()
        else:
            self.image_uncertainties.clear()
            self.image_ious.clear()
            self.image_dices.clear()
            self.image_accuracies.clear()
            self.image_nlls.clear()

        self.correlation_results.clear()

        if self.is_main_process:
            data_unit = "pixel" if self.per_pixel_statistics else "image"
            logging.info(f"Distributed dataset evaluator reset ({data_unit}-level statistics)")

    def __len__(self) -> int:
        """返回当前进程已添加的数据点数（图片数或像素数）"""
        if self.per_pixel_statistics:
            return len(self.pixel_uncertainties)
        else:
            return len(self.image_uncertainties)

    def get_total_images_across_all_processes(self) -> int:
        """获取所有进程的总数据点数（all_reduce 求和）"""
        if not self.distributed or not dist.is_initialized():
            return len(self)

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        local_count = torch.tensor([len(self)], device=device, dtype=torch.int64)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        return int(local_count.item())


# 保持向后兼容性
class DatasetEvaluator(DistributedDatasetEvaluator):
    """向后兼容的DatasetEvaluator类"""
    
    def __init__(self, save_dir: str):
        super().__init__(save_dir, distributed=False, rank=0, world_size=1)
