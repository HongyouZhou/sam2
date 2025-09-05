import os
import logging
import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .metric_calculator import MetricCalculator
from .visualization_utils import VisualizationUtils


class DistributedDatasetEvaluator:
    
    def __init__(self, save_dir: str, distributed: bool = False, rank: int = 0, world_size: int = 1):
        """
        初始化分布式数据集评估器
        
        Args:
            save_dir: 结果保存目录
            distributed: 是否启用分布式训练
            rank: 当前进程的rank
            world_size: 总进程数
        """
        self.save_dir = save_dir
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0
        
        self.metric_calculator = MetricCalculator()
        self.viz_utils = VisualizationUtils()
        
        # 存储每张图片的标量指标值（每个数据点代表一张图片）
        self.image_uncertainties = []  # 存储每张图片的平均uncertainty值
        self.image_ious = []          # 存储每张图片的平均IoU值
        self.image_dices = []         # 存储每张图片的平均DICE值
        self.image_accuracies = []    # 存储每张图片的平均accuracy值
        
        # 存储最终结果
        self.correlation_results = {}
        
        # 分布式训练相关
        if self.distributed:
            self._setup_distributed()
        
        # 只在主进程创建保存目录
        if self.is_main_process:
            os.makedirs(save_dir, exist_ok=True)
    
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
            
            # 对每张图片分别计算标量指标
            for i in range(B):
                # 提取单张图片的数据
                single_uncertainty = uncertainty_aligned[i]  # [H, W, K]
                single_pred = pred_aligned[i]                # [H, W, K]
                single_gt = gt_aligned[i]                    # [H, W, K]
                
                # 计算单张图片的标量指标
                iou_scalar = self._calculate_single_image_iou_scalar(single_pred, single_gt)
                dice_scalar = self._calculate_single_image_dice_scalar(single_pred, single_gt)
                accuracy_scalar = self._calculate_single_image_accuracy_scalar(single_pred, single_gt)
                uncertainty_scalar = self._calculate_single_image_uncertainty_scalar(single_uncertainty)
                
                # 存储单张图片的标量指标（转换为CPU以节省GPU内存）
                self.image_uncertainties.append(uncertainty_scalar.detach().cpu())
                self.image_ious.append(iou_scalar.detach().cpu())
                self.image_dices.append(dice_scalar.detach().cpu())
                self.image_accuracies.append(accuracy_scalar.detach().cpu())
            
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
        """计算单张图片的平均准确率值（标量）"""
        try:
            if pred_logits.shape != gt_masks.shape:
                logging.warning(f"Shape mismatch in single image accuracy: pred {pred_logits.shape} vs gt {gt_masks.shape}")
                return torch.tensor(0.0)
            
            # 先应用sigmoid，再应用 > 0 阈值
            pred_probs = torch.sigmoid(pred_logits)
            pred_binary = pred_probs > 0
            gt_binary = gt_masks > 0
            
            # 计算准确率：(TP + TN) / (TP + TN + FP + FN)
            correct_predictions = (pred_binary == gt_binary).float()  # [H, W, K]
            
            # 对空间维度和通道维度求和，得到整张图片的准确率
            correct_sum = correct_predictions.sum()  # 标量
            total_pixels = correct_predictions.numel()  # 总像素数
            
            # 计算整张图片的准确率
            pixel_acc = correct_sum / (total_pixels + 1e-8)  # 标量
            
            # 确保值在[0, 1]范围内
            pixel_acc = torch.clamp(pixel_acc, 0.0, 1.0)
            
            return pixel_acc
            
        except Exception as e:
            logging.warning(f"Failed to calculate single image accuracy scalar: {e}")
            return torch.tensor(0.0)
    
    def _calculate_single_image_uncertainty_scalar(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """计算单张图片的平均uncertainty值（标量）"""
        try:
            # 对空间维度和通道维度求平均，得到整张图片的平均uncertainty
            uncertainty_mean = uncertainty.mean()  # 标量
            return uncertainty_mean
            
        except Exception as e:
            logging.warning(f"Failed to calculate single image uncertainty scalar: {e}")
            return torch.tensor(0.0)
    
    def _gather_distributed_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], 
                                               List[torch.Tensor], List[torch.Tensor]]:
        """收集所有GPU进程的数据（使用NCCL-safe all_gather + padding）"""
        if not self.distributed or not dist.is_initialized():
            return (self.image_uncertainties, self.image_ious, 
                    self.image_dices, self.image_accuracies)

        def gather_scalar_list(values: List[torch.Tensor]) -> List[torch.Tensor]:
            # 将标量列表打包为CUDA张量，使用padding后all_gather
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            local_vals = torch.tensor([v.item() for v in values], device=device, dtype=torch.float32)
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
            out: List[torch.Tensor] = []
            for r, tensor_r in enumerate(gathered):
                n = int(len_list[r].item())
                if n > 0:
                    vals = tensor_r[:n].detach().cpu().tolist()
                    out.extend([torch.tensor(v) for v in vals if np.isfinite(v)])
            return out

        try:
            all_uncertainties = gather_scalar_list(self.image_uncertainties)
            all_ious = gather_scalar_list(self.image_ious)
            all_dices = gather_scalar_list(self.image_dices)
            all_accuracies = gather_scalar_list(self.image_accuracies)
            return all_uncertainties, all_ious, all_dices, all_accuracies
        except Exception as e:
            logging.warning(f"Failed to gather distributed data: {e}")
            return (self.image_uncertainties, self.image_ious, 
                    self.image_dices, self.image_accuracies)
    
    def evaluate_dataset_correlation(self) -> Dict[str, Dict[str, float]]:
        """评估整个数据集的指标与不确定性相关性"""
        try:
            if not self.image_uncertainties:
                logging.warning("No image data available for evaluation")
                return {}
            
            if self.is_main_process:
                logging.info(f"Evaluating correlation for {len(self.image_uncertainties)} images")
            
            # 收集分布式数据
            all_uncertainties, all_ious, all_dices, all_accuracies = self._gather_distributed_data()
            
            # 只在主进程上进行评估
            if not self.is_main_process:
                return {}
            
            # 分别计算每个指标的相关性
            correlation_results = {}
            
            # IoU相关性
            if all_ious:
                logging.info(f"Calculating IoU correlation with {len(all_ious)} images")
                iou_corr = self.metric_calculator.calculate_image_metric_uncertainty_correlation(
                    uncertainties=all_uncertainties,
                    metrics=all_ious,
                    metric_names=['IoU'] * len(all_ious)
                )
                if 'IoU' in iou_corr:
                    correlation_results['IoU'] = iou_corr['IoU']
                    logging.info(f"IoU correlation calculated: {iou_corr['IoU'].get('correlation', 'N/A')}")
                else:
                    logging.warning("IoU correlation calculation failed")
            
            # DICE相关性
            if all_dices:
                logging.info(f"Calculating DICE correlation with {len(all_dices)} images")
                dice_corr = self.metric_calculator.calculate_image_metric_uncertainty_correlation(
                    uncertainties=all_uncertainties,
                    metrics=all_dices,
                    metric_names=['DICE'] * len(all_dices)
                )
                if 'DICE' in dice_corr:
                    correlation_results['DICE'] = dice_corr['DICE']
                    logging.info(f"DICE correlation calculated: {dice_corr['DICE'].get('correlation', 'N/A')}")
                else:
                    logging.warning("DICE correlation calculation failed")
            
            # Accuracy相关性
            if all_accuracies:
                logging.info(f"Calculating Accuracy correlation with {len(all_accuracies)} images")
                acc_corr = self.metric_calculator.calculate_image_metric_uncertainty_correlation(
                    uncertainties=all_uncertainties,
                    metrics=all_accuracies,
                    metric_names=['Accuracy'] * len(all_accuracies)
                )
                if 'Accuracy' in acc_corr:
                    correlation_results['Accuracy'] = acc_corr['Accuracy']
                    logging.info(f"Accuracy correlation calculated: {acc_corr['Accuracy'].get('correlation', 'N/A')}")
                else:
                    logging.warning("Accuracy correlation calculation failed")
            
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
        """重置评估器状态，清除所有图片数据"""
        self.image_uncertainties.clear()
        self.image_ious.clear()
        self.image_dices.clear()
        self.image_accuracies.clear()
        self.correlation_results.clear()
        
        if self.is_main_process:
            logging.info("Distributed dataset evaluator reset")
    
    def __len__(self) -> int:
        """返回当前进程已添加的图片数"""
        return len(self.image_uncertainties)
    
    def get_total_images_across_all_processes(self) -> int:
        """获取所有进程的总图片数（all_reduce 求和）"""
        if not self.distributed or not dist.is_initialized():
            return len(self.image_uncertainties)
        try:
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            local_count = torch.tensor([len(self.image_uncertainties)], device=device, dtype=torch.int64)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            return int(local_count.item())
        except Exception as e:
            logging.warning(f"Failed to get total images across processes: {e}")
            return len(self.image_uncertainties)


# 保持向后兼容性
class DatasetEvaluator(DistributedDatasetEvaluator):
    """向后兼容的DatasetEvaluator类"""
    
    def __init__(self, save_dir: str):
        super().__init__(save_dir, distributed=False, rank=0, world_size=1)
