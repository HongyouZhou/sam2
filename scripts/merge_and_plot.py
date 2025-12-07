#!/usr/bin/env python
"""
合并 parallel_compare.py 生成的多个子目录结果，并使用 zs.py 的可视化函数生成图表

用法:
    python scripts/merge_and_plot.py --output_path ../outputs/zs_parallel_9
"""

import argparse
import json
from pathlib import Path
import sys

# 导入 zs.py 的可视化函数
sys.path.insert(0, str(Path(__file__).parent))
from zs import (
    create_comprehensive_comparison_plots,
    create_ua_shift_analysis_plots,
)


def load_json_from_subdir(subdir: Path) -> dict:
    """从子目录加载 detailed_results.json"""
    json_path = subdir / "comparison_plots" / "detailed_results.json"
    if not json_path.exists():
        return {}
    
    with open(json_path) as f:
        return json.load(f)


def convert_results(data: dict, key: str) -> dict:
    """将 JSON 格式的结果转换为 (J&F, J, F) 元组格式"""
    if key not in data:
        return {}
    
    results = {}
    for dataset, metrics in data[key].items():
        if isinstance(metrics, dict):
            results[dataset] = (
                float(metrics.get("jf", 0)),
                float(metrics.get("j", 0)),
                float(metrics.get("f", 0))
            )
    return results


def merge_results(output_path: Path) -> tuple:
    """合并所有方法的结果
    
    Returns:
        (sam2_results, bndl_aue_results, bndl_aue_statistics, bndl_results, 
         bndl_statistics, uctta_results, ur_ern_results, uctta_statistics, ur_ern_statistics)
    """
    print("\n" + "=" * 80)
    print("🔄 合并所有方法的评估结果...")
    print("=" * 80)
    
    # 方法名到子目录的映射
    method_dirs = {
        "SAM": "output_sam_001_01",
        "UCTTA": "output_uctta_001_01",
        "BNDL_AUE": "output_bndl_aue_012_02",
        "BNDL": "output_bndl_012_02",
        "UR-ERN": "output_ur_ern_001_01",
    }
    
    # JSON 键名映射
    key_mapping = {
        "SAM": "sam2_results",
        "UCTTA": "uctta_results",
        "BNDL_AUE": "bndl_aue_results",
        "BNDL": "bndl_results",
        "UR-ERN": "ur_ern_results",
    }
    
    stats_mapping = {
        "UCTTA": "uctta_statistics",
        "BNDL_AUE": "bndl_aue_statistics",
        "BNDL": "bndl_statistics",
        "UR-ERN": "ur_ern_statistics",
    }
    
    # 初始化结果容器
    sam2_results = {}
    bndl_aue_results = {}
    bndl_aue_statistics = {}
    bndl_results = {}
    bndl_statistics = {}
    uctta_results = {}
    ur_ern_results = {}
    uctta_statistics = {}
    ur_ern_statistics = {}
    
    # 加载每个方法的结果
    for method, dirname in method_dirs.items():
        subdir = output_path / dirname
        if not subdir.exists():
            print(f"⚠️  跳过 {method}: 目录不存在 ({dirname})")
            continue
        
        print(f"📂 加载 {method} 的结果...")
        data = load_json_from_subdir(subdir)
        
        if not data:
            print(f"   ⚠️  未找到 detailed_results.json")
            continue
        
        # 提取结果
        json_key = key_mapping[method]
        results = convert_results(data, json_key)
        
        if results:
            print(f"   ✓ 找到 {len(results)} 个数据集的结果")
            
            # 根据方法名分配到相应的变量
            if method == "SAM":
                sam2_results.update(results)
            elif method == "UCTTA":
                uctta_results.update(results)
            elif method == "BNDL_AUE":
                bndl_aue_results.update(results)
            elif method == "BNDL":
                bndl_results.update(results)
            elif method == "UR-ERN":
                ur_ern_results.update(results)
        
        # 提取统计数据
        if method in stats_mapping:
            stats_key = stats_mapping[method]
            if stats_key in data:
                stats = data[stats_key]
                if stats:
                    print(f"   ✓ 找到统计数据")
                    if method == "UCTTA":
                        uctta_statistics.update(stats)
                    elif method == "BNDL_AUE":
                        bndl_aue_statistics.update(stats)
                    elif method == "BNDL":
                        bndl_statistics.update(stats)
                    elif method == "UR-ERN":
                        ur_ern_statistics.update(stats)
    
    print("\n" + "=" * 80)
    print("✓ 结果合并完成！")
    print("=" * 80)
    print(f"\n统计:")
    print(f"  SAM-2:     {len(sam2_results)} 个数据集")
    print(f"  UCTTA:     {len(uctta_results)} 个数据集")
    print(f"  BNDL+AUE:  {len(bndl_aue_results)} 个数据集")
    print(f"  BNDL:      {len(bndl_results)} 个数据集")
    print(f"  UR-ERN:    {len(ur_ern_results)} 个数据集")
    print()
    
    return (
        sam2_results,
        bndl_aue_results,
        bndl_aue_statistics,
        bndl_results,
        bndl_statistics,
        uctta_results if uctta_results else None,
        ur_ern_results if ur_ern_results else None,
        uctta_statistics,
        ur_ern_statistics,
    )


def main():
    parser = argparse.ArgumentParser(
        description="合并 parallel_compare.py 的结果并生成可视化图表"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="parallel_compare.py 的输出目录 (包含 output_sam_*, output_bndl_* 等子目录)",
    )
    
    args = parser.parse_args()
    
    if not args.output_path.exists():
        print(f"❌ 错误: 输出目录不存在: {args.output_path}")
        return 1
    
    # 合并所有结果
    (
        sam2_results,
        bndl_aue_results,
        bndl_aue_statistics,
        bndl_results,
        bndl_statistics,
        uctta_results,
        ur_ern_results,
        uctta_statistics,
        ur_ern_statistics,
    ) = merge_results(args.output_path)
    
    # 生成可视化图表
    print("\n" + "=" * 80)
    print("🎨 生成可视化图表...")
    print("=" * 80)
    
    # 1. 综合对比图
    if sam2_results and bndl_aue_results:
        print("\n📊 生成综合对比图...")
        try:
            create_comprehensive_comparison_plots(
                sam2_results=sam2_results,
                bndl_results=bndl_aue_results,
                bndl_statistics=bndl_aue_statistics,
                output_path=args.output_path,
                uctta_results=uctta_results,
                uctta_statistics=uctta_statistics,
                bndl_baseline_results=bndl_results if bndl_results else None,
            )
            print("✓ 综合对比图已生成")
        except Exception as e:
            print(f"⚠️  警告: 生成综合对比图失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  跳过综合对比图: 缺少必要的结果数据")
    
    # 2. UA shift 分析图
    if bndl_aue_statistics and sam2_results and bndl_aue_results:
        print("\n📊 生成 UA shift 分析图...")
        try:
            # 构建各方法的结果目录路径
            sam2_root = args.output_path / "output_sam_001_01" / "sam2_results"
            bndl_aue_root = args.output_path / "output_bndl_aue_012_02" / "bndl_aue_results"
            bndl_root = args.output_path / "output_bndl_012_02" / "bndl_results"
            uctta_root = args.output_path / "output_uctta_001_01" / "sam2_uctta_results"
            ur_ern_root = args.output_path / "output_ur_ern_001_01" / "sam2_ur_ern_results"
            
            create_ua_shift_analysis_plots(
                bndl_statistics=bndl_aue_statistics,
                sam2_results=sam2_results,
                bndl_results=bndl_aue_results,
                output_path=args.output_path,
                uctta_statistics=uctta_statistics if uctta_statistics else None,
                uctta_results=uctta_results,
                ur_ern_results=ur_ern_results,
                bndl_pure_statistics=bndl_statistics if bndl_statistics else None,
                bndl_pure_results=bndl_results if bndl_results else None,
                sam2_root_override=sam2_root if sam2_root.exists() else None,
                bndl_root_override=bndl_aue_root if bndl_aue_root.exists() else None,
                uctta_root_override=uctta_root if uctta_root.exists() else None,
                ur_ern_root_override=ur_ern_root if ur_ern_root.exists() else None,
                bndl_pure_root_override=bndl_root if bndl_root.exists() else None,
            )
            print("✓ UA shift 分析图已生成")
        except Exception as e:
            print(f"⚠️  警告: 生成 UA shift 分析图失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  跳过 UA shift 分析图: 缺少必要的统计数据")
    
    print("\n" + "=" * 80)
    print("✅ 所有图表生成完成！")
    print("=" * 80)
    print(f"\n图表保存位置: {args.output_path / 'comparison_plots'}")
    print("\n生成的文件:")
    print("  • sam2_vs_bndl_comprehensive_comparison.png")
    print("  • sam2_vs_bndl_comprehensive_comparison.pdf")
    print("  • ua_shift_analysis.png  (修复后无重叠)")
    print("  • ua_shift_analysis.pdf")
    print("  • ua_pcc_summary.csv")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

