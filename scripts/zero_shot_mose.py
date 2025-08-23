#!/usr/bin/env python
# Zero-shot evaluation of SAM-2 on the MOSE dataset
# Author: <your name>

import argparse
import os
import shutil
import sys
from pathlib import Path
import tempfile
import torch
import numpy as np
from PIL import Image, ImageDraw
import json  # 新增

# ----------  SAM-2 -----------
from sam2.build_sam import build_sam2_video_predictor  # 构建视频版 predictor

# ↓ 仅保留工具函数，去掉原来直接用 mask prompt 的 vos_inference
from tools.vos_inference import vos_inference
from tools.vos_inference import (
    DAVIS_PALETTE,
    save_masks_to_dir,
)

# 新增：点击采样工具
from sam2.modeling.sam2_utils import (
    sample_one_point_from_error_center,
    sample_random_points_from_errors,
)

# ----------  Metric ----------
from sav_dataset.utils.sav_benchmark import benchmark  # 评测 J & F（IOU + Boundary F）

# 新增：通用依赖
import random
import cv2
from tqdm import tqdm


def overlay_mask_and_point(
    img_path: Path,
    mask_path: Path,
    point_xy,
    save_path: Path,
    mask_color=(0, 255, 0, 128),  # 半透明绿色
    point_color=(255, 0, 0, 255),  # 红点
):
    """把 mask 和 query point 叠加到原图并保存"""
    img = Image.open(img_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")  # 单通道
    mask_np = np.array(mask) > 0  # bool 掩码

    # 生成彩色半透明掩码图层
    overlay = Image.new("RGBA", img.size, mask_color)
    alpha = (mask_np * mask_color[3]).astype(np.uint8)  # 0/128 α 通道
    overlay.putalpha(Image.fromarray(alpha, mode="L"))

    blended = Image.alpha_composite(img, overlay)

    # 画 query point
    if point_xy is not None:
        draw = ImageDraw.Draw(blended)
        r = 6
        x, y = point_xy
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=point_color,
            outline=(255, 255, 255, 255),
            width=2,
        )

    blended.convert("RGB").save(save_path)


def overlay_mask_and_points(
    img_path: Path,
    mask_bool: np.ndarray,
    pts: list,
    save_path: Path,
    mask_color=(0, 255, 0, 128),
    point_colors=((255, 0, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255)),
):
    """
    把 bool 掩码与若干查询点一起画到原图
    pts           : [(x, y), ...]   查询点
    point_colors  : 与 pts 对应的 RGBA 颜色
    """
    img = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, mask_color)
    alpha = mask_bool.astype(np.uint8) * mask_color[3]
    overlay.putalpha(Image.fromarray(alpha, mode="L"))
    blended = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(blended)
    for pt, clr in zip(pts, point_colors):
        if pt is None:
            continue
        x, y = pt
        r = 6
        draw.ellipse((x - r, y - r, x + r, y + r), fill=clr, outline=(255, 255, 255, 255), width=2)

    blended.convert("RGB").save(save_path)


def generate_visualizations(jpeg_dir: Path, pred_dir: Path, ann_dir: Path, vis_root: Path, video_names: list[str] | None = None):
    vis_root.mkdir(parents=True, exist_ok=True)

    if video_names is None:
        video_names = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    else:
        video_names = sorted(set(video_names))

    for vid in video_names:
        jpg_dir = jpeg_dir / vid
        pred_dir_video = pred_dir / vid
        vis_dir_video = vis_root / vid
        vis_dir_video.mkdir(parents=True, exist_ok=True)

        # ---- 读取查询点 ----
        pts_json = pred_dir_video / "query_points.json"
        if pts_json.exists():
            with open(pts_json) as f:
                query_pts_dict = {int(k): v for k, v in json.load(f).items()}
        else:  # 兼容旧结果：若无文件则跳过
            query_pts_dict = {}

        # 遍历 obj
        for obj_id, pts in query_pts_dict.items():
            obj_vis_dir = vis_dir_video / f"obj_{obj_id}"
            obj_vis_dir.mkdir(parents=True, exist_ok=True)

            pt_colors = [(255, 0, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255)]

            for img_path in sorted(jpg_dir.iterdir()):
                if img_path.suffix.lower() not in [".jpg", ".jpeg"]:
                    continue
                mask_path = pred_dir_video / f"{img_path.stem}.png"
                if not mask_path.exists():
                    continue
                mask_np = np.array(Image.open(mask_path)) == obj_id
                save_path = obj_vis_dir / f"{img_path.stem}.jpg"
                overlay_mask_and_points(img_path, mask_np, pts, save_path, point_colors=pt_colors)


# ---------- 3-click 评估核心 ---------- #
def _sample_pos_neg(gt_mask: np.ndarray, dilate_iter: int = 5):
    """
    在单个 obj 的 GT mask 上采样 1 个正点 + 1 个负点
    返回: (pos_xy, neg_xy)，均为 (x, y) 像素坐标
    """
    # 正点：GT 内随机像素
    ys, xs = np.where(gt_mask)
    assert len(xs) > 0, "GT mask 为空！"
    idx = random.randrange(len(xs))
    pos_xy = (int(xs[idx]), int(ys[idx]))

    # 负点：目标边界外 (但靠近边界) 的背景像素
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(gt_mask.astype(np.uint8), kernel, iterations=dilate_iter) > 0
    ring = np.logical_and(dilated, ~gt_mask)  # dilate 后减去自身得到“边界邻域”
    ys_n, xs_n = np.where(ring)
    if len(xs_n) == 0:  # 极端情况兜底：任意背景
        ys_n, xs_n = np.where(~gt_mask)
    idx_n = random.randrange(len(xs_n))
    neg_xy = (int(xs_n[idx_n]), int(ys_n[idx_n]))
    return pos_xy, neg_xy


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def inference_3_clicks(
    predictor,
    jpeg_dir: Path,
    ann_dir: Path,
    out_dir: Path,
    score_thresh: float = 0.0,
    video_names: list[str] | None = None,
    third_click_method: str = "center",
):
    """
    对每个 obj 执行 3-click 交互：
      1) GT 内随机正点
      2) GT 边界邻域随机负点
      3) 基于当前预测与 GT 的 error 区域，再采 1 点（center / uniform）
    推理完成后将 PNG 结果与 `vos_inference` 相同格式保存到 out_dir
    """
    if video_names is None:
        video_names = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    else:
        video_names = sorted(set(video_names))

    print(f"3-click 推理，共 {len(video_names)} 个 video")
    for v_idx, vid in enumerate(video_names, 1):
        print(f"[{v_idx:03}/{len(video_names)}] {vid}")
        video_dir = jpeg_dir / vid
        frame_names = sorted([p.stem for p in video_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]], key=lambda x: int(x))
        # 初始化 predictor state
        state = predictor.init_state(str(video_dir))
        H, W = state["video_height"], state["video_width"]

        # 读取首帧 GT，确定 object ids
        first_mask = np.array(Image.open(ann_dir / vid / f"{frame_names[0]}.png"))
        obj_ids = [oid for oid in np.unique(first_mask) if oid > 0]

        obj_points: dict[int, list[tuple[int, int]]] = {}  # ← 记录 {obj_id: [(x,y)*3]}

        for obj_id in obj_ids:
            gt_bool = first_mask == obj_id
            # Step-1/2：正 + 负
            pos_xy, neg_xy = _sample_pos_neg(gt_bool)
            pts = np.array([[pos_xy, neg_xy]], dtype=np.float32)
            lbl = np.array([[1, 0]], dtype=np.int32)
            _, obj_ids_after, masks = predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=obj_id,
                points=pts.reshape(-1, 2),
                labels=lbl.reshape(-1),
            )

            # Step-3：基于 error 采点
            cur_idx = obj_ids_after.index(obj_id)
            pred_bool = (masks[cur_idx : cur_idx + 1] > score_thresh).cpu().numpy().astype(bool)[0, 0]
            gt_t = torch.from_numpy(gt_bool[None, None]).bool()
            pred_t = torch.from_numpy(pred_bool[None, None]).bool()
            pt3, lb3 = sample_one_point_from_error_center(gt_t, pred_t, padding=True)

            pt3_xy = list(map(int, pt3.squeeze().tolist()))  # ← 转成纯 Python list[int]

            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=obj_id,
                points=pt3.squeeze(0).cpu().numpy(),
                labels=lb3.squeeze(0).cpu().numpy(),
                clear_old_points=False,
            )

            # 全部转换为 list[int, int]，确保可 JSON 序列化
            obj_points[obj_id] = [
                list(map(int, pos_xy)),
                list(map(int, neg_xy)),
                pt3_xy,
            ]

        # 完整 propagate
        video_segments = {}
        for f_idx, out_obj_ids, out_logits in predictor.propagate_in_video(state):
            seg = {oid: (out_logits[i] > score_thresh).cpu().numpy() for i, oid in enumerate(out_obj_ids)}
            video_segments[f_idx] = seg

        # 保存 PNG
        out_video_dir = out_dir / vid
        for f_idx, seg in video_segments.items():
            save_masks_to_dir(
                output_mask_dir=str(out_dir),
                video_name=vid,
                frame_name=frame_names[f_idx],
                per_obj_output_mask=seg,
                height=H,
                width=W,
                per_obj_png_file=False,
                output_palette=DAVIS_PALETTE,
            )
        # ------ 保存查询点到 JSON ------
        (out_dir / vid).mkdir(parents=True, exist_ok=True)  # 若未创建
        with open(out_dir / vid / "query_points.json", "w") as f:
            json.dump({int(k): v for k, v in obj_points.items()}, f, indent=2)

        torch.cuda.empty_cache()


# -------- run_inference 别名，主函数里直接调用 --------
def run_inference(
    predictor,
    jpeg_dir: Path,
    ann_dir: Path,
    out_dir: Path,
    score_thresh: float = 0.0,
    video_names: list[str] | None = None,
):
    """
    对接 main()，内部调用 3-click 推理
    """
    inference_3_clicks(
        predictor,
        jpeg_dir,
        ann_dir,
        out_dir,
        score_thresh=score_thresh,
        video_names=video_names,
    )


# -------------  评测 ------------------
def evaluate(
    gt_root: Path,
    pred_root: Path,
    video_subset: list[str] | None = None,
    workers: int | None = None,
    output_path: Path = Path("./outputs"),  # 新增参数，默认与主程序一致
):
    # use mose_tmp to avoid the problem of the same name of the video
    if video_subset is not None:
        gt_tmp = output_path / "mose_tmp_gt"
        pred_tmp = output_path / "mose_tmp_pred"
        # 先清空旧的临时目录（如果存在）
        if gt_tmp.exists():
            shutil.rmtree(gt_tmp)
        if pred_tmp.exists():
            shutil.rmtree(pred_tmp)
        gt_tmp.mkdir(parents=True, exist_ok=True)
        pred_tmp.mkdir(parents=True, exist_ok=True)
        for v in video_subset:
            shutil.copytree(gt_root / v, gt_tmp / v, symlinks=True)
            shutil.copytree(pred_root / v, pred_tmp / v, symlinks=True)
        gt_root, pred_root = gt_tmp, pred_tmp

    J_F, global_J, global_F, _ = benchmark(
        gt_roots=[str(gt_root)],
        mask_roots=[str(pred_root)],
        strict=False,  # allow missing videos
        num_processes=workers,
        skip_first_and_last=True,  # or True if 00000.png is absent
        verbose=True,
    )


# -------------  主函数 -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot SAM-2 evaluation on MOSE")
    p.add_argument(
        "--mose_root",
        required=False,
        default="/ssdArray/hongyou/dev/data/MOSE_release",
        help="MOSE_release 目录（包含 train / val / test 子目录）",
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="使用哪个子集做推理与评测",
    )
    p.add_argument(
        "--sam2_cfg",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    p.add_argument(
        "--sam2_checkpoint",
        required=False,
        default="/home/hongyou/dev/sam2/checkpoints/sam2.1_hiera_large.pt",
        help="SAM-2 预训练权重 .pt",
    )
    p.add_argument("--output_dir", default="./mose_pred", help="保存预测 PNG 的根目录")
    p.add_argument("--device", default="cuda")
    p.add_argument("--score_thresh", type=float, default=0.0, help="mask logit 阈值")
    p.add_argument("--num_workers", type=int, default=None, help="评测时并⾏进程数")
    p.add_argument(
        "--save_vis",
        default=True,
        action="store_true",
        help="保存可视化图 (原图 + mask + query point)",
    )
    p.add_argument(
        "--file_list",
        default="training/assets/MOSE_sample_val_list.txt",
        help="TXT file containing video names (one per line) to run evaluation on",
    )
    p.add_argument("--output_path", default="./outputs", help="所有输出的根目录")
    return p.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    mose_split_dir = Path(args.mose_root) / args.split
    jpeg_dir = mose_split_dir / "JPEGImages"
    ann_dir = mose_split_dir / "Annotations"
    out_dir = output_path / "mose_pred"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------- 构建 SAM-2 predictor ------------
    print("Loading SAM-2 checkpoint…")
    predictor = build_sam2_video_predictor(config_file=args.sam2_cfg, ckpt_path=args.sam2_checkpoint, device=args.device)

    # ------------------ 读取 video 列表 ----------------------
    if args.file_list and Path(args.file_list).is_file():
        with open(args.file_list, "r") as f:
            subset_videos = [ln.strip() for ln in f if ln.strip()]
        print(f"Using {len(subset_videos)} videos from {args.file_list}")
    else:
        subset_videos = None

    # ------------------ 推理 ----------------------
    run_inference(
        predictor,
        jpeg_dir,
        ann_dir,
        out_dir,
        score_thresh=args.score_thresh,
        video_names=subset_videos,
    )

    # ------------------ 评测 ----------------------
    evaluate(ann_dir, out_dir, video_subset=subset_videos, workers=args.num_workers, output_path=output_path)

    if args.save_vis:
        print("Saving visualizations...")
        vis_dir = output_path / "mose_pred_vis"
        generate_visualizations(jpeg_dir, out_dir, ann_dir, vis_dir, subset_videos)
        print("Visualizations saved to:", vis_dir)


if __name__ == "__main__":
    main()
