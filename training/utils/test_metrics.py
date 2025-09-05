#!/usr/bin/env python3
"""
测试修复后的指标计算函数
"""

import torch
import numpy as np
from metric_calculator import MetricCalculator

def test_metric_calculations():
    """测试IoU、DICE和accuracy指标计算"""
    
    # 创建测试数据
    batch_size = 2
    height = 64
    width = 64
    channels = 3
    
    # 创建预测logits (随机值)
    pred_logits = torch.randn(batch_size, height, width, channels)
    
    # 创建ground truth masks (二值)
    gt_masks = torch.randint(0, 2, (batch_size, height, width, channels)).float()
    
    # 创建一些重叠的预测
    pred_probs = torch.sigmoid(pred_logits)
    pred_binary = (pred_probs > 0.5).float()
    
    # 手动计算期望的IoU
    intersection = (pred_binary & gt_masks).float()
    union = (pred_binary | gt_masks).float()
    
    intersection_sum = intersection.sum(dim=(1, 2))  # [B, K]
    union_sum = union.sum(dim=(1, 2))  # [B, K]
    expected_iou = intersection_sum / (union_sum + 1e-8)
    
    print(f"Test data shapes:")
    print(f"  pred_logits: {pred_logits.shape}")
    print(f"  gt_masks: {gt_masks.shape}")
    print(f"  pred_binary: {pred_binary.shape}")
    print(f"  Expected IoU shape: {expected_iou.shape}")
    print(f"  Expected IoU values:\n{expected_iou}")
    
    # 测试修复后的指标计算
    calculator = MetricCalculator()
    
    print("\n" + "="*50)
    print("Testing IoU calculation:")
    print("="*50)
    
    iou_result = calculator.calculate_iou_metric(pred_logits, gt_masks)
    if iou_result is not None:
        print(f"IoU result shape: {iou_result.shape}")
        # 提取每个batch和通道的IoU值
        iou_values = iou_result[:, 0, 0, :]  # [B, K]
        print(f"Calculated IoU values:\n{iou_values}")
        
        # 验证结果
        iou_diff = torch.abs(iou_values - expected_iou)
        print(f"IoU differences:\n{iou_diff}")
        print(f"Max IoU difference: {iou_diff.max().item():.6f}")
        
        if iou_diff.max() < 1e-6:
            print("✅ IoU calculation is correct!")
        else:
            print("❌ IoU calculation has errors!")
    else:
        print("❌ IoU calculation failed!")
    
    print("\n" + "="*50)
    print("Testing DICE calculation:")
    print("="*50)
    
    dice_result = calculator.calculate_dice_metric(pred_logits, gt_masks)
    if dice_result is not None:
        print(f"DICE result shape: {dice_result.shape}")
        # 提取每个batch和通道的DICE值
        dice_values = dice_result[:, 0, 0, :]  # [B, K]
        print(f"Calculated DICE values:\n{dice_values}")
        
        # 验证DICE值在合理范围内
        if torch.all((dice_values >= 0) & (dice_values <= 1)):
            print("✅ DICE values are in valid range [0, 1]!")
        else:
            print("❌ DICE values are out of range!")
    else:
        print("❌ DICE calculation failed!")
    
    print("\n" + "="*50)
    print("Testing Accuracy calculation:")
    print("="*50)
    
    acc_result = calculator.calculate_mask_accuracy(pred_logits, gt_masks)
    if acc_result is not None:
        print(f"Accuracy result shape: {acc_result.shape}")
        # 提取每个batch和通道的accuracy值
        acc_values = acc_result[:, 0, 0, :]  # [B, K]
        print(f"Calculated Accuracy values:\n{acc_values}")
        
        # 验证accuracy值在合理范围内
        if torch.all((acc_values >= 0) & (acc_values <= 1)):
            print("✅ Accuracy values are in valid range [0, 1]!")
        else:
            print("❌ Accuracy values are out of range!")
    else:
        print("❌ Accuracy calculation failed!")
    
    print("\n" + "="*50)
    print("Test completed!")
    print("="*50)

def test_edge_cases():
    """测试边界情况"""
    print("\n" + "="*50)
    print("Testing edge cases:")
    print("="*50)
    
    calculator = MetricCalculator()
    
    # 测试全零的情况
    print("\n1. Testing all zeros:")
    pred_zeros = torch.zeros(1, 32, 32, 2)
    gt_zeros = torch.zeros(1, 32, 32, 2)
    
    iou_zero = calculator.calculate_iou_metric(pred_zeros, gt_zeros)
    print(f"  IoU for all zeros: {iou_zero[0, 0, 0, :] if iou_zero is not None else 'None'}")
    
    # 测试全一的情况
    print("\n2. Testing all ones:")
    pred_ones = torch.ones(1, 32, 32, 2)
    gt_ones = torch.ones(1, 32, 32, 2)
    
    iou_ones = calculator.calculate_iou_metric(pred_ones, gt_ones)
    print(f"  IoU for all ones: {iou_ones[0, 0, 0, :] if iou_ones is not None else 'None'}")
    
    # 测试完全相反的情况
    print("\n3. Testing opposite masks:")
    pred_opp = torch.zeros(1, 32, 32, 2)
    gt_opp = torch.ones(1, 32, 32, 2)
    
    iou_opp = calculator.calculate_iou_metric(pred_opp, gt_opp)
    print(f"  IoU for opposite masks: {iou_opp[0, 0, 0, :] if iou_opp is not None else 'None'}")

if __name__ == "__main__":
    print("Testing metric calculations...")
    test_metric_calculations()
    test_edge_cases()
