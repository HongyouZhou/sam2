"""
指标计算模块
提供IoU、DICE、准确率等评估指标的计算方法
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict


class MetricCalculator:
    """指标计算器类"""
    
    @staticmethod
    def normalize_tensor_format(tensor, name: str) -> Optional[torch.Tensor]:
        """将张量标准化为 [B, H, W, K] 格式"""
        try:
            if tensor is None:
                return None
            
            if not hasattr(tensor, 'shape'):
                tensor = torch.tensor(tensor)
            
            shape = tensor.shape
            logging.info(f"{name} original shape: {shape}")
            
            if len(shape) == 5:  # [B, T, K, H, W] or [T, B, K, H, W] or [K, B, T, H, W]
                # 更明确的维度判断
                if shape[0] <= 4 and shape[1] <= 4:  # [K, B, T, H, W] 或 [T, B, K, H, W]
                    if shape[0] <= shape[1]:  # [K, B, T, H, W]
                        tensor = tensor[:, :, 0]  # 取第一帧 [K, B, H, W]
                        tensor = tensor.permute(1, 0, 2, 3)  # [B, K, H, W]
                    else:  # [T, B, K, H, W]
                        tensor = tensor[0]  # 取第一帧 [B, K, H, W]
                else:  # [B, T, K, H, W]
                    tensor = tensor[:, 0]  # 取第一帧 [B, K, H, W]
                shape = tensor.shape
            
            if len(shape) == 4:
                B, dim1, dim2, dim3 = shape
                # 更好的启发式方法来确定格式
                if dim1 <= 4 and dim2 > 16 and dim3 > 16:  # [B, K, H, W]
                    tensor = tensor.permute(0, 2, 3, 1)  # [B, H, W, K]
                elif dim2 == dim3 and dim1 > 16:  # [B, H, W, K] (已经正确)
                    pass
                elif dim1 > 16 and dim2 > 16 and dim3 <= 4:  # [B, H, W, K] (已经正确)
                    pass
                else:
                    # 后备方案：如果不确定，假设最大的维度是空间维度
                    spatial_dims = sorted([(i, d) for i, d in enumerate(shape[1:], 1)], key=lambda x: x[1], reverse=True)
                    if spatial_dims[0][1] == spatial_dims[1][1]:  # 两个最大维度相等（可能是H, W）
                        h_idx, w_idx = spatial_dims[0][0], spatial_dims[1][0]
                        remaining_idx = [i for i in [1, 2, 3] if i not in [h_idx, w_idx]][0]
                        # 重新排序为 [B, H, W, K]
                        perm = [0, h_idx, w_idx, remaining_idx]
                        tensor = tensor.permute(*perm)
            elif len(shape) == 3:  # [B, H, W]
                tensor = tensor.unsqueeze(-1)  # [B, H, W, 1]
            elif len(shape) == 2:  # [H, W]
                tensor = tensor.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
            else:
                logging.warning(f"Unsupported {name} shape: {shape}")
                return None
                
            logging.info(f"{name} normalized shape: {tensor.shape}")
            return tensor
            
        except Exception as e:
            logging.warning(f"Failed to normalize {name} tensor: {e}")
            return None
    
    @staticmethod
    def align_spatial_dimensions(pred_logits: torch.Tensor, gt_masks: torch.Tensor, 
                                uncertainty: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """对齐所有张量的空间维度"""
        try:
            # 获取目标空间维度（使用预测作为参考）
            B_pred, target_h, target_w, K_pred = pred_logits.shape
            
            # 处理gt_masks
            B_gt, H_gt, W_gt, K_gt = gt_masks.shape
            if (H_gt, W_gt) != (target_h, target_w):
                # 转换为float进行插值，然后转换回原始dtype
                gt_dtype = gt_masks.dtype
                gt_masks_float = gt_masks.float()
                
                # 重塑为 [B, K, H, W] 进行插值
                gt_masks_reshaped = gt_masks_float.permute(0, 3, 1, 2)
                gt_masks_resized = F.interpolate(
                    gt_masks_reshaped, size=(target_h, target_w), 
                    mode='bilinear', align_corners=False
                )
                gt_masks = gt_masks_resized.permute(0, 2, 3, 1).to(gt_dtype)
                logging.info(f"Resized gt_masks from {H_gt}x{W_gt} to {target_h}x{target_w}")
            
            # 处理uncertainty
            if len(uncertainty.shape) == 4:  # [B, H, W, C]
                B_unc, H_unc, W_unc, C_unc = uncertainty.shape
                if (H_unc, W_unc) != (target_h, target_w):
                    uncertainty_reshaped = uncertainty.permute(0, 3, 1, 2)
                    uncertainty_resized = F.interpolate(
                        uncertainty_reshaped, size=(target_h, target_w),
                        mode='bilinear', align_corners=False
                    )
                    uncertainty = uncertainty_resized.permute(0, 2, 3, 1)
                    logging.info(f"Resized uncertainty from {H_unc}x{W_unc} to {target_h}x{target_w}")
            elif len(uncertainty.shape) == 3:  # [B, H, W]
                B_unc, H_unc, W_unc = uncertainty.shape
                if (H_unc, W_unc) != (target_h, target_w):
                    uncertainty_resized = F.interpolate(
                        uncertainty.unsqueeze(1), size=(target_h, target_w),
                        mode='bilinear', align_corners=False
                    )
                    uncertainty = uncertainty_resized.squeeze(1)
                    logging.info(f"Resized uncertainty from {H_unc}x{W_unc} to {target_h}x{target_w}")
            
            # 处理通道维度不匹配
            if gt_masks.shape[-1] != pred_logits.shape[-1]:
                K_min = min(gt_masks.shape[-1], pred_logits.shape[-1])
                gt_masks = gt_masks[..., :K_min]
                pred_logits = pred_logits[..., :K_min]
                logging.info(f"Truncated to {K_min} channels for consistency")
            
            return pred_logits, gt_masks, uncertainty
            
        except Exception as e:
            logging.warning(f"Failed to align spatial dimensions: {e}")
            import traceback
            logging.warning(f"Alignment traceback: {traceback.format_exc()}")
            return None, None, None
    
    @staticmethod
    def calculate_iou_metric(pred_logits: torch.Tensor, gt_masks: torch.Tensor) -> Optional[torch.Tensor]:
        """计算标准IoU指标，与训练时保持一致"""
        try:
            # 确保输入形状匹配
            if pred_logits.shape != gt_masks.shape:
                logging.warning(f"Shape mismatch in IoU calculation: pred {pred_logits.shape} vs gt {gt_masks.shape}")
                return None
            
            # 与训练时保持一致：直接对logits应用 > 0 阈值（与iou_loss一致）
            pred_binary = pred_logits > 0
            gt_binary = gt_masks > 0
            
            # 标准IoU计算：|A ∩ B| / |A ∪ B|
            intersection = (pred_binary & gt_binary).float()
            union = (pred_binary | gt_binary).float()
            
            # 对空间维度求和，得到每个batch和通道的IoU
            intersection_sum = intersection.sum(dim=(1, 2))  # [B, K]
            union_sum = union.sum(dim=(1, 2))  # [B, K]
            
            # 避免除零，计算IoU
            iou = intersection_sum / (union_sum + 1e-8)  # [B, K]
            
            # 确保值在[0, 1]范围内
            iou = torch.clamp(iou, 0.0, 1.0)
            
            # 将IoU值扩展到原始空间维度，用于后续处理
            B, H, W, K = pred_logits.shape
            iou_expanded = iou.unsqueeze(1).unsqueeze(2).repeat(1, H, W, 1)  # [B, H, W, K]
            
            return iou_expanded
            
        except Exception as e:
            logging.warning(f"Failed to calculate IoU metric: {e}")
            import traceback
            logging.warning(f"IoU calculation traceback: {traceback.format_exc()}")
            return None
    
    @staticmethod
    def calculate_dice_metric(pred_logits: torch.Tensor, gt_masks: torch.Tensor) -> Optional[torch.Tensor]:
        """计算标准DICE指标，与训练时保持一致"""
        try:
            if pred_logits.shape != gt_masks.shape:
                logging.warning(f"Shape mismatch in DICE calculation: pred {pred_logits.shape} vs gt {gt_masks.shape}")
                return None
            
            # 与训练时保持一致：先应用sigmoid
            pred_probs = torch.sigmoid(pred_logits)
            
            # 标准DICE计算：2 * |A ∩ B| / (|A| + |B|)
            numerator = 2 * (pred_probs * gt_masks.float()).sum(dim=(1, 2), keepdim=True)  # [B, 1, 1, K]
            denominator = pred_probs.sum(dim=(1, 2), keepdim=True) + gt_masks.float().sum(dim=(1, 2), keepdim=True)  # [B, 1, 1, K]
            
            # 避免除零，计算DICE
            dice = numerator / (denominator + 1e-8)  # [B, 1, 1, K]
            
            # 处理边界情况：当预测和真实都为空时，DICE应该为1
            # 当只有预测为空时，DICE应该为0
            gt_empty = (gt_masks.float().sum(dim=(1, 2), keepdim=True) < 1e-8)
            pred_empty = (pred_probs.sum(dim=(1, 2), keepdim=True) < 1e-8)
            
            # 当两者都为空时，DICE = 1；当只有预测为空时，DICE = 0
            dice = torch.where(gt_empty & pred_empty, torch.ones_like(dice), dice)
            dice = torch.where(gt_empty & ~pred_empty, torch.zeros_like(dice), dice)
            
            # 确保值在[0, 1]范围内
            dice = torch.clamp(dice, 0.0, 1.0)
            
            # 扩展到与输入相同的空间维度
            B, H, W, K = pred_logits.shape
            dice_expanded = dice.expand(B, H, W, K).contiguous()  # [B, H, W, K]
            
            return dice_expanded
            
        except Exception as e:
            logging.warning(f"Failed to calculate DICE metric: {e}")
            import traceback
            logging.warning(f"DICE calculation traceback: {traceback.format_exc()}")
            return None
    
    @staticmethod
    def calculate_mask_accuracy(pred_logits: torch.Tensor, gt_masks: torch.Tensor) -> Optional[torch.Tensor]:
        """计算标准掩码准确率，与训练时保持一致"""
        try:
            if pred_logits.shape != gt_masks.shape:
                logging.warning(f"Shape mismatch in accuracy calculation: pred {pred_logits.shape} vs gt {gt_masks.shape}")
                return None
            
            # 与训练时保持一致：先应用sigmoid，再应用 > 0 阈值
            pred_probs = torch.sigmoid(pred_logits)
            pred_binary = pred_probs > 0
            gt_binary = gt_masks > 0
            
            # 标准准确率计算：(TP + TN) / (TP + TN + FP + FN)
            correct_predictions = (pred_binary == gt_binary).float()  # [B, H, W, K]
            
            # 对空间维度求和，得到每个batch和通道的准确率
            correct_sum = correct_predictions.sum(dim=(1, 2))  # [B, K]
            total_pixels = correct_predictions.shape[1] * correct_predictions.shape[2]  # H * W
            
            # 计算准确率
            pixel_acc = correct_sum / (total_pixels + 1e-8)  # [B, K]
            
            # 确保值在[0, 1]范围内
            pixel_acc = torch.clamp(pixel_acc, 0.0, 1.0)
            
            # 扩展到与输入相同的空间维度
            B, H, W, K = pred_logits.shape
            acc_expanded = pixel_acc.unsqueeze(1).unsqueeze(2).repeat(1, H, W, 1)  # [B, H, W, K]
            
            return acc_expanded
            
        except Exception as e:
            logging.warning(f"Failed to calculate mask accuracy: {e}")
            import traceback
            logging.warning(f"Accuracy calculation traceback: {traceback.format_exc()}")
            return None

    @staticmethod
    def calculate_image_metric_uncertainty_correlation(
        uncertainties: List[torch.Tensor], 
        metrics: List[torch.Tensor],
        metric_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        计算单张图片指标与不确定性的相关性（每个数据点代表一张图片）
        
        Args:
            uncertainties: 每张图片的uncertainty标量张量列表
            metrics: 每张图片的指标标量张量列表
            metric_names: 指标名称列表
            
        Returns:
            相关性结果字典
        """
        try:
            if len(uncertainties) != len(metrics) or len(uncertainties) != len(metric_names):
                logging.warning("Input lists have different lengths")
                return {}
            
            if not uncertainties:
                logging.warning("No data provided for correlation calculation")
                return {}
            
            # 将所有图片的标量数据合并
            all_uncertainties = []
            all_metrics = []
            
            for uncertainty, metric in zip(uncertainties, metrics):
                if uncertainty.numel() > 0 and metric.numel() > 0:
                    # 将标量张量转换为numpy标量
                    uncertainty_scalar = uncertainty.detach().cpu().item()
                    metric_scalar = metric.detach().cpu().item()
                    
                    all_uncertainties.append(uncertainty_scalar)
                    all_metrics.append(metric_scalar)
            
            if not all_uncertainties:
                logging.warning("No valid data after processing")
                return {}
            
            # 转换为numpy数组
            uncertainty_np = np.array(all_uncertainties)
            metric_np = np.array(all_metrics)
            
            # 移除无效值
            valid_mask = np.isfinite(uncertainty_np) & np.isfinite(metric_np)
            uncertainty_valid = uncertainty_np[valid_mask]
            metric_valid = metric_np[valid_mask]
            
            if len(uncertainty_valid) < 2:
                logging.warning("Not enough valid data points for correlation")
                return {}
            
            # 计算相关性
            correlation = np.corrcoef(uncertainty_valid, metric_valid)[0, 1]
            
            # 计算线性回归
            if len(uncertainty_valid) > 1:
                slope, intercept = np.polyfit(uncertainty_valid, metric_valid, 1)
            else:
                slope, intercept = 0.0, 0.0
            
            # 计算统计信息
            uncertainty_mean = np.mean(uncertainty_valid)
            uncertainty_std = np.std(uncertainty_valid)
            metric_mean = np.mean(metric_valid)
            metric_std = np.std(metric_valid)
            
            # 保存结果
            results = {}
            for metric_name in set(metric_names):
                results[metric_name] = {
                    'correlation': float(correlation),
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'uncertainty_mean': float(uncertainty_mean),
                    'uncertainty_std': float(uncertainty_std),
                    'metric_mean': float(metric_mean),
                    'metric_std': float(metric_std),
                    'num_valid_points': int(np.sum(valid_mask)),
                    'total_points': len(uncertainty_np),
                    # 保存原始数据用于绘图
                    'uncertainty_data': uncertainty_valid.tolist(),
                    'metric_data': metric_valid.tolist()
                }
                
                logging.info(f"{metric_name} correlation: {correlation:.4f}, "
                           f"slope: {slope:.4f}, valid points: {np.sum(valid_mask)}/{len(uncertainty_np)}")
            
            return results
            
        except Exception as e:
            logging.warning(f"Failed to calculate image metric uncertainty correlation: {e}")
            import traceback
            logging.warning(f"Correlation calculation traceback: {traceback.format_exc()}")
            return {}
    
    @staticmethod
    def aggregate_metrics_across_dataset(
        metric_batches: List[torch.Tensor],
        uncertainty_batches: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        聚合整个数据集的指标和不确定性数据
        
        Args:
            metric_batches: 批次指标张量列表
            uncertainty_batches: 批次不确定性张量列表
            
        Returns:
            聚合后的指标和不确定性张量列表
        """
        try:
            aggregated_metrics = []
            aggregated_uncertainties = []
            
            for metric_batch, uncertainty_batch in zip(metric_batches, uncertainty_batches):
                if metric_batch is None or uncertainty_batch is None:
                    continue
                
                # 标准化格式
                metric_normalized = MetricCalculator.normalize_tensor_format(metric_batch, "metric_batch")
                uncertainty_normalized = MetricCalculator.normalize_tensor_format(uncertainty_batch, "uncertainty_batch")
                
                if metric_normalized is not None and uncertainty_normalized is not None:
                    aggregated_metrics.append(metric_normalized)
                    aggregated_uncertainties.append(uncertainty_normalized)
            
            return aggregated_metrics, aggregated_uncertainties
            
        except Exception as e:
            logging.warning(f"Failed to aggregate metrics across dataset: {e}")
            return [], []
