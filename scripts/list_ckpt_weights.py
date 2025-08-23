#!/usr/bin/env python3
"""
检查预训练权重中的参数
"""

import torch
import fnmatch
import argparse
from pathlib import Path


def check_checkpoint_params(checkpoint_path, target_patterns=None):
    """
    检查预训练权重中的参数
    
    Args:
        checkpoint_path: 预训练权重文件路径
        target_patterns: 要检查的模式列表
    """
    print(f"正在检查预训练权重: {checkpoint_path}")
    print("=" * 60)
    
    # 加载权重文件
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ 成功加载权重文件")
    except Exception as e:
        print(f"❌ 加载权重文件失败: {e}")
        return
    
    # 检查权重文件的结构
    print(f"\n📁 权重文件结构:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # 获取模型权重
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print(f"\n📊 模型权重包含 {len(state_dict)} 个参数")
    else:
        print(f"\n❌ 权重文件中没有找到 'model' 键")
        print(f"可用的键: {list(checkpoint.keys())}")
        return
    
    # 默认检查的模式
    if target_patterns is None:
        target_patterns = [
            "sam_mask_decoder.output_hypernetworks_mlps.*",
            "sam_mask_decoder.pixel_bndl.*",
            "sam_mask_decoder.*",
            "image_encoder.*",
            "sam_prompt_encoder.*"
        ]
    
    # 检查每个模式
    print(f"\n🔍 检查模式匹配:")
    for pattern in target_patterns:
        matching_keys = fnmatch.filter(state_dict.keys(), pattern)
        print(f"\n模式: {pattern}")
        if matching_keys:
            print(f"  ✅ 找到 {len(matching_keys)} 个匹配的参数:")
            for key in sorted(matching_keys):
                param_shape = state_dict[key].shape
                print(f"    - {key}: {param_shape}")
        else:
            print(f"  ❌ 没有找到匹配的参数")
    
    # 显示所有参数的前缀统计
    print(f"\n📈 参数前缀统计:")
    prefix_count = {}
    for key in state_dict.keys():
        parts = key.split('.')
        if len(parts) >= 2:
            prefix = f"{parts[0]}.{parts[1]}"
            prefix_count[prefix] = prefix_count.get(prefix, 0) + 1
    
    for prefix, count in sorted(prefix_count.items()):
        print(f"  - {prefix}.*: {count} 个参数")
    
    # 检查特定的 output_hypernetworks_mlps 参数
    print(f"\n🎯 详细检查 output_hypernetworks_mlps:")
    mlp_keys = [k for k in state_dict.keys() if 'output_hypernetworks_mlps' in k]
    if mlp_keys:
        print(f"  ✅ 找到 {len(mlp_keys)} 个 output_hypernetworks_mlps 参数:")
        for key in sorted(mlp_keys):
            param_shape = state_dict[key].shape
            print(f"    - {key}: {param_shape}")
    else:
        print(f"  ❌ 没有找到 output_hypernetworks_mlps 参数")
    
    # 检查 pixel_bndl 参数
    print(f"\n🎯 详细检查 pixel_bndl:")
    bndl_keys = [k for k in state_dict.keys() if 'pixel_bndl' in k]
    if bndl_keys:
        print(f"  ✅ 找到 {len(bndl_keys)} 个 pixel_bndl 参数:")
        for key in sorted(bndl_keys):
            param_shape = state_dict[key].shape
            print(f"    - {key}: {param_shape}")
    else:
        print(f"  ❌ 没有找到 pixel_bndl 参数")


def main():
    parser = argparse.ArgumentParser(description="检查预训练权重中的参数")
    parser.add_argument("checkpoint_path", help="预训练权重文件路径")
    parser.add_argument("--patterns", nargs="+", help="要检查的模式列表")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ 文件不存在: {checkpoint_path}")
        return
    
    check_checkpoint_params(checkpoint_path, args.patterns)


if __name__ == "__main__":
    main()
