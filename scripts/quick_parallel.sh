#!/bin/bash
# 快速并行测试脚本
# 使用compare_sam2_vs_bndl统一入口，在3个GPU上并行运行SAM、UCTTA、BNDL

cd /home/hongyou/dev/ada_samp

echo "======================================================================"
echo "并行对比评估 - 使用 compare_sam2_vs_bndl 统一入口"
echo "======================================================================"
echo "GPU分配: SAM→GPU0, UCTTA→GPU1, BNDL→GPU2"
echo "数据集: GTEA"
echo "模式: 仅第一帧（快速测试）"
echo "======================================================================"
echo ""

python sam2/scripts/parallel_compare.py \
    --datasets GTEA \
    --gpu_ids 0 1 2 \
    --first_frame_only \
    --video_limit 5 \
    --output_path ./outputs/parallel_quick_test \
    --collect_bndl_stats

echo ""
echo "======================================================================"
echo "测试完成！"
echo "结果保存在: ./outputs/parallel_quick_test/"
echo "查看对比表格: ./outputs/parallel_quick_test/parallel_comparison_merged.csv"
echo "======================================================================"

