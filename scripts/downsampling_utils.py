#!/usr/bin/env python
"""
降采样工具函数 - 用于减少checkpoint和可视化数据量

共享给所有评估脚本使用，避免代码重复
"""

import numpy as np


def smart_downsample_samples(uncertainty, accuracy, max_samples=10000):
    """
    智能降采样：保持数据分布特性的同时减少点数
    
    支持多种输入格式：numpy数组、tensor、Python列表
    
    策略：
    - 小数据集 (<5k): 不降采样
    - 中等数据集 (5k-50k): 均匀随机采样到max_samples
    - 大数据集 (>50k): 分层采样（保持边界和异常值）
    
    Args:
        uncertainty: 不确定性值（numpy数组/tensor/list）
        accuracy: 准确度值（numpy数组/tensor/list）
        max_samples: 最大保留样本数
    
    Returns:
        (降采样后的uncertainty, 降采样后的accuracy)
        返回类型与输入类型一致（numpy→numpy, list→list）
    """
    # 记录输入类型
    input_is_list = isinstance(uncertainty, list)
    
    # 转换为numpy数组
    if hasattr(uncertainty, 'detach'):
        uncertainty = uncertainty.detach().cpu().numpy()
    if hasattr(accuracy, 'detach'):
        accuracy = accuracy.detach().cpu().numpy()
    
    uncertainty = np.asarray(uncertainty).flatten()
    accuracy = np.asarray(accuracy).flatten()
    
    n = len(uncertainty)
    
    # 策略1: 小数据集不降采样
    if n <= 5000:
        return (uncertainty.tolist(), accuracy.tolist()) if input_is_list else (uncertainty, accuracy)
    
    # 策略2: 已在目标范围内
    if n <= max_samples:
        return (uncertainty.tolist(), accuracy.tolist()) if input_is_list else (uncertainty, accuracy)
    
    # 策略3: 中等数据集 - 简单随机采样
    if n <= 50000:
        indices = np.random.choice(n, max_samples, replace=False)
        unc_down = uncertainty[indices]
        acc_down = accuracy[indices]
        return (unc_down.tolist(), acc_down.tolist()) if input_is_list else (unc_down, acc_down)
    
    # 策略4: 大数据集 - 分层采样（保持极值）
    n_boundary = min(1000, max_samples // 10)
    
    # 保留边界值（极端不确定性）
    unc_sorted_idx = np.argsort(uncertainty)
    boundary_idx = np.concatenate([
        unc_sorted_idx[:n_boundary // 2],  # 最低不确定性
        unc_sorted_idx[-n_boundary // 2:],  # 最高不确定性
    ])
    boundary_idx = np.unique(boundary_idx)
    
    # 从剩余数据中随机采样
    remaining_mask = np.ones(n, dtype=bool)
    remaining_mask[boundary_idx] = False
    remaining_idx = np.where(remaining_mask)[0]
    
    n_random = max_samples - len(boundary_idx)
    if n_random > 0 and len(remaining_idx) > 0:
        random_idx = np.random.choice(remaining_idx, min(n_random, len(remaining_idx)), replace=False)
        selected_idx = np.concatenate([boundary_idx, random_idx])
    else:
        selected_idx = boundary_idx
    
    unc_down = uncertainty[selected_idx]
    acc_down = accuracy[selected_idx]
    
    return (unc_down.tolist(), acc_down.tolist()) if input_is_list else (unc_down, acc_down)


def downsample_checkpoint_data(checkpoint_data, max_samples=10000):
    """
    对checkpoint数据进行降采样（用于保存前优化）
    
    Args:
        checkpoint_data: 包含pixel_uncertainties和pixel_accuracies的字典
        max_samples: 每个checkpoint最多保留的样本数
    
    Returns:
        降采样后的checkpoint_data（就地修改）
    """
    unc = checkpoint_data.get('pixel_uncertainties', [])
    acc = checkpoint_data.get('pixel_accuracies', [])
    
    if unc and acc and len(unc) > max_samples:
        unc_down, acc_down = smart_downsample_samples(unc, acc, max_samples=max_samples)
        checkpoint_data['pixel_uncertainties'] = unc_down
        checkpoint_data['pixel_accuracies'] = acc_down
        return True, len(unc), len(unc_down)
    
    return False, len(unc) if unc else 0, len(unc) if unc else 0


def downsample_statistics_pavpu(statistics_dict, max_samples=10000):
    """
    对statistics字典中的所有PAvPU样本进行降采样（BNDL专用）
    
    Args:
        statistics_dict: 统计数据字典
        max_samples: 最多保留的样本数
    
    Returns:
        降采样后的statistics_dict（就地修改）
    """
    if not statistics_dict:
        return statistics_dict
    
    # 收集所有PAvPU样本
    all_uncertainty = []
    all_accuracy = []
    pavpu_keys = []
    
    for key in list(statistics_dict.keys()):
        if key.endswith("_pavpu_uncertainty_samples"):
            uncertainty = statistics_dict[key]
            accuracy_key = key.replace("_uncertainty_", "_accuracy_")
            if accuracy_key in statistics_dict:
                accuracy = statistics_dict[accuracy_key]
                all_uncertainty.extend(uncertainty)
                all_accuracy.extend(accuracy)
                pavpu_keys.append((key, accuracy_key))
    
    # 如果有PAvPU样本，进行降采样
    if all_uncertainty and all_accuracy:
        n_original = len(all_uncertainty)
        uncertainty_down, accuracy_down = smart_downsample_samples(
            all_uncertainty, all_accuracy, max_samples=max_samples
        )
        
        # 清除旧的分散样本
        for unc_key, acc_key in pavpu_keys:
            del statistics_dict[unc_key]
            del statistics_dict[acc_key]
        
        # 存储降采样后的合并样本
        statistics_dict["checkpoint_pavpu_uncertainty_samples"] = uncertainty_down if isinstance(uncertainty_down, list) else uncertainty_down.tolist()
        statistics_dict["checkpoint_pavpu_accuracy_samples"] = accuracy_down if isinstance(accuracy_down, list) else accuracy_down.tolist()
        
        return statistics_dict, n_original, len(uncertainty_down) if isinstance(uncertainty_down, list) else len(uncertainty_down)
    
    return statistics_dict, 0, 0

