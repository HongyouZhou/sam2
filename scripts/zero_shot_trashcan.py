#!/usr/bin/env python
# Zero-shot evaluation of SAM-2 on the TranshCan/GTEA-style dataset
# Author: <your name>

import argparse
from pathlib import Path
import shutil
import json
import random

import torch
import numpy as np
from PIL import Image

# ----------  SAM-2 -----------
from sam2.build_sam import build_sam2_video_predictor

# ----------  Tools -----------
from tools.vos_inference import (
    DAVIS_PALETTE,
    save_masks_to_dir,
)
from tools.vos_inference import vos_inference  # kept for parity, not used in 3-click

# ----------  Click sampling ----------
from sam2.modeling.sam2_utils import (
    sample_one_point_from_error_center,
)

# ----------  Metric ----------
from sav_dataset.utils.sav_benchmark import benchmark

# ----------  Optional viz ----------
from PIL import ImageDraw
import cv2
from tqdm import tqdm


def overlay_mask_and_points(img_path: Path, mask_bool: np.ndarray, pts: list, save_path: Path,
                            mask_color=(0, 255, 0, 128),
                            point_colors=((255, 0, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255))):
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


def generate_visualizations(jpeg_dir: Path, pred_dir: Path, ann_dir: Path, vis_root: Path,
                            video_names: list[str] | None = None):
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

        # read stored points if available
        pts_json = pred_dir_video / "query_points.json"
        if pts_json.exists():
            with open(pts_json) as f:
                query_pts_dict = {int(k): v for k, v in json.load(f).items()}
        else:
            query_pts_dict = {}

        for obj_id, pts in query_pts_dict.items():
            obj_vis_dir = vis_dir_video / f"obj_{obj_id}"
            obj_vis_dir.mkdir(parents=True, exist_ok=True)
            for img_path in sorted(jpg_dir.iterdir()):
                if img_path.suffix.lower() not in [".jpg", ".jpeg"]:
                    continue
                mask_path = pred_dir_video / f"{img_path.stem}.png"
                if not mask_path.exists():
                    continue
                mask_np = np.array(Image.open(mask_path)) == obj_id
                save_path = obj_vis_dir / f"{img_path.stem}.jpg"
                overlay_mask_and_points(img_path, mask_np, pts, save_path)


def _sample_pos_neg(gt_mask: np.ndarray, dilate_iter: int = 5):
    ys, xs = np.where(gt_mask)
    assert len(xs) > 0, "GT mask is empty."
    idx = random.randrange(len(xs))
    pos_xy = (int(xs[idx]), int(ys[idx]))

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(gt_mask.astype(np.uint8), kernel, iterations=dilate_iter) > 0
    ring = np.logical_and(dilated, ~gt_mask)
    ys_n, xs_n = np.where(ring)
    if len(xs_n) == 0:
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
):
    if video_names is None:
        video_names = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
    else:
        video_names = sorted(set(video_names))

    print(f"3-click inference on {len(video_names)} videos")
    for v_idx, vid in enumerate(video_names, 1):
        print(f"[{v_idx:03}/{len(video_names)}] {vid}")
        video_dir = jpeg_dir / vid
        frame_names = sorted([p.stem for p in video_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg"]],
                             key=lambda x: int(x))

        state = predictor.init_state(str(video_dir))
        H, W = state["video_height"], state["video_width"]

        first_mask = np.array(Image.open(ann_dir / vid / f"{frame_names[0]}.png"))
        obj_ids = [oid for oid in np.unique(first_mask) if oid > 0]

        obj_points: dict[int, list[tuple[int, int]]] = {}
        for obj_id in obj_ids:
            gt_bool = first_mask == obj_id
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

            cur_idx = obj_ids_after.index(obj_id)
            pred_bool = (masks[cur_idx : cur_idx + 1] > score_thresh).cpu().numpy().astype(bool)[0, 0]
            gt_t = torch.from_numpy(gt_bool[None, None]).bool()
            pred_t = torch.from_numpy(pred_bool[None, None]).bool()
            pt3, lb3 = sample_one_point_from_error_center(gt_t, pred_t, padding=True)
            pt3_xy = list(map(int, pt3.squeeze().tolist()))

            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=obj_id,
                points=pt3.squeeze(0).cpu().numpy(),
                labels=lb3.squeeze(0).cpu().numpy(),
                clear_old_points=False,
            )

            obj_points[obj_id] = [
                list(map(int, pos_xy)),
                list(map(int, neg_xy)),
                pt3_xy,
            ]

        video_segments = {}
        for f_idx, out_obj_ids, out_logits in predictor.propagate_in_video(state):
            seg = {oid: (out_logits[i] > score_thresh).cpu().numpy() for i, oid in enumerate(out_obj_ids)}
            video_segments[f_idx] = seg

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
        (out_dir / vid).mkdir(parents=True, exist_ok=True)
        with open(out_dir / vid / "query_points.json", "w") as f:
            json.dump({int(k): v for k, v in obj_points.items()}, f, indent=2)

        torch.cuda.empty_cache()


def run_inference(
    predictor,
    jpeg_dir: Path,
    ann_dir: Path,
    out_dir: Path,
    score_thresh: float = 0.0,
    video_names: list[str] | None = None,
):
    inference_3_clicks(
        predictor,
        jpeg_dir,
        ann_dir,
        out_dir,
        score_thresh=score_thresh,
        video_names=video_names,
    )


def evaluate(
    gt_root: Path,
    pred_root: Path,
    video_subset: list[str] | None = None,
    workers: int | None = None,
    output_path: Path = Path("./outputs"),
    skip_first_and_last: bool = False,
):
    if video_subset is not None:
        gt_tmp = output_path / "transhcan_tmp_gt"
        pred_tmp = output_path / "transhcan_tmp_pred"
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

    benchmark(
        gt_roots=[str(gt_root)],
        mask_roots=[str(pred_root)],
        strict=False,
        num_processes=workers,
        skip_first_and_last=skip_first_and_last,
        verbose=True,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot SAM-2 evaluation on TrashCan dataset")
    p.add_argument(
        "--dataset_root",
        default="/ssdArray/hongyou/dev/data/sam2_data/TrashCan_SAM2",
        help="Dataset root. Either contains JPEGImages/Annotations directly or under a split subfolder.",
    )
    p.add_argument(
        "--split",
        default="val",
        help="Optional split subfolder (e.g., train/val/test). If empty, the script looks for JPEGImages/Annotations directly under dataset_root.",
    )
    p.add_argument(
        "--sam2_cfg",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    p.add_argument(
        "--sam2_checkpoint",
        default="/home/hongyou/dev/ada_samp/sam2/checkpoints/sam2.1_hiera_large.pt",
        help="SAM-2 .pt checkpoint",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--score_thresh", type=float, default=0.0, help="mask logit threshold")
    p.add_argument("--num_workers", type=int, default=None, help="evaluation processes")
    p.add_argument("--output_path", default="./outputs", help="root for all outputs")
    p.add_argument("--output_dirname", default="trashcan_pred", help="subdir under output_path for masks")
    p.add_argument("--save_vis", default=True, action="store_true", help="save simple visualizations")
    p.add_argument("--vis_dirname", default="trashcan_pred_vis", help="subdir under output_path for visualizations")
    p.add_argument("--file_list", default="", help="optional TXT file: one video name per line to run on")
    p.add_argument("--skip_first_and_last", action="store_true", default=False, help="skip first/last frame in eval")
    return p.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    root = Path(args.dataset_root)
    if args.split:
        jpeg_dir = root / args.split / "JPEGImages"
        ann_dir = root / args.split / "Annotations"
    else:
        jpeg_dir = root / "JPEGImages"
        ann_dir = root / "Annotations"

    if not jpeg_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"JPEGImages or Annotations not found under: {jpeg_dir} and {ann_dir}")

    out_dir = output_path / args.output_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SAM-2 checkpoint…")
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=args.device,
    )

    if args.file_list and Path(args.file_list).is_file():
        with open(args.file_list, "r") as f:
            subset_videos = [ln.strip() for ln in f if ln.strip()]
        print(f"Using {len(subset_videos)} videos from {args.file_list}")
    else:
        subset_videos = None

    run_inference(
        predictor,
        jpeg_dir,
        ann_dir,
        out_dir,
        score_thresh=args.score_thresh,
        video_names=subset_videos,
    )

    evaluate(
        ann_dir,
        out_dir,
        video_subset=subset_videos,
        workers=args.num_workers,
        output_path=output_path,
        skip_first_and_last=args.skip_first_and_last,
    )

    if args.save_vis:
        print("Saving visualizations...")
        vis_dir = output_path / args.vis_dirname
        generate_visualizations(jpeg_dir, out_dir, ann_dir, vis_dir, subset_videos)
        print("Visualizations saved to:", vis_dir)


if __name__ == "__main__":
    main()
