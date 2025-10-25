# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for AUE (Adversarial Uncertainty Estimation) initialization."""

import logging

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    从 binary masks 计算紧凑的 bounding boxes。
    
    Args:
        masks: [N, H, W] - binary masks (bool or float)
    
    Returns:
        boxes: [N, 4] - boxes in (x1, y1, x2, y2) format
    """
    N, H, W = masks.shape
    boxes = torch.zeros(N, 4, dtype=torch.float32, device=masks.device)
    
    for i in range(N):
        mask = masks[i] > 0.5 if masks[i].dtype == torch.float32 else masks[i]
        if mask.sum() == 0:
            # 空 mask，返回中心的小框
            boxes[i] = torch.tensor([W // 2 - 1, H // 2 - 1, W // 2 + 1, H // 2 + 1], dtype=torch.float32)
            continue
        
        # 找到非零位置
        ys, xs = torch.where(mask)
        x1, x2 = xs.min().item(), xs.max().item()
        y1, y2 = ys.min().item(), ys.max().item()
        boxes[i] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
    
    return boxes


def _collate_adversarial_samples(batch, H_adv, W_adv):
    """自定义 collate function 用于批量处理对抗样本。"""
    valid_samples = []
    
    for idx, sample in enumerate(batch):
        try:
            # 提取第一帧第一个对象
            if not hasattr(sample, 'frames') or len(sample.frames) == 0:
                continue
            
            frame = sample.frames[0]
            if len(frame.objects) == 0:
                continue
                
            img = frame.data  # PIL Image 或 Tensor
            obj = frame.objects[0]
            mask = obj.segment  # [H, W] uint8 tensor
            
            if mask is None:
                continue
            
            # 转换图像为 tensor
            if isinstance(img, torch.Tensor):
                img_t = img.float() / 255.0 if img.dtype == torch.uint8 else img
                if img_t.dim() == 2:  # 灰度图
                    img_t = img_t.unsqueeze(0).repeat(3, 1, 1)
            else:  # PIL Image
                from PIL import Image  # noqa: F401
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            
            valid_samples.append((img_t, mask.float()))
            
        except Exception as e:
            # More detailed error logging
            import traceback
            logging.warning(f"Failed to process sample {idx} in batch: {e}\n{traceback.format_exc()}")
            continue
    
    if not valid_samples:
        return None, None
    
    # 分离图像和掩码
    imgs, masks = zip(*valid_samples, strict=False)
    
    # 检查是否所有图像尺寸相同
    shapes = [img.shape for img in imgs]
    all_same_shape = all(s == shapes[0] for s in shapes)
    
    if all_same_shape and len(imgs) > 1:
        # 批量 resize
        stacked_imgs = torch.stack(list(imgs))
        imgs_resized = F.interpolate(stacked_imgs, size=(H_adv, W_adv), mode='bilinear', align_corners=False)
        
        stacked_masks = torch.stack(list(masks)).unsqueeze(1)
        masks_resized = F.interpolate(stacked_masks, size=(H_adv, W_adv), mode='nearest').squeeze(1)
    else:
        # 逐个 resize
        imgs_resized = torch.stack([
            F.interpolate(img.unsqueeze(0), size=(H_adv, W_adv), mode='bilinear', align_corners=False)[0]
            for img in imgs
        ])
        masks_resized = torch.stack([
            F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(H_adv, W_adv), mode='nearest')[0, 0]
            for mask in masks
        ])
    
    return imgs_resized, masks_resized


@torch.no_grad()
def sample_adversarials_from_dataset(
    dataset,
    K_eff: int,
    device: torch.device,
    num_workers: int = 4,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    从训练数据集随机采样初始化对抗样本。
    
    直接使用 dataset 配置中的 transforms（RandomResize(1024), ToTensor, Normalize），
    采用与训练循环相同的数据访问方式，无需手动重复实现预处理逻辑。
    
    Args:
        dataset: 训练数据集 (TorchTrainMixedDataset 或 VOSDataset)
                 数据已经过 transforms: RandomResize + ToTensor + ImageNet Normalize
        K_eff: 要采样的样本数量
        device: 目标设备（保留参数，向后兼容）
        num_workers: 保留参数，向后兼容（直接访问 dataset 不需要 workers）
    
    Returns:
        adv_images: List of [3, H, W] tensors - 已归一化的图像（来自 dataset）
        adv_boxes: List of [4] tensors - bounding box prompts（从 GT mask 计算）
        adv_masks: List of [H, W] tensors - ground truth masks（保留用于将来扩展）
    """
    # 获取实际的 dataset（处理 TorchTrainMixedDataset 包装）
    actual_dataset = dataset.datasets[0] if hasattr(dataset, 'datasets') else dataset
    dataset_len = len(actual_dataset)
    
    # 随机采样索引（1.5 倍备用，以防部分样本无效）
    oversample_ratio = 1.5
    num_to_sample = min(int(K_eff * oversample_ratio), dataset_len)
    indices = torch.randperm(dataset_len)[:num_to_sample].tolist()
    
    logging.info(f"Sampling {K_eff} adversarial samples from dataset (trying {num_to_sample} samples)...")
    
    images_list = []
    boxes_list = []
    masks_list = []
    
    # 直接访问 dataset，与训练时相同的方式
    for idx in indices:
        if len(images_list) >= K_eff:
            break
            
        try:
            # 获取样本 (已经过 dataset transforms)
            # 这会调用 VOSDataset._get_datapoint -> construct -> transforms
            sample = actual_dataset[idx]
            
            # 提取第一帧第一个对象
            frame = sample.frames[0]
            if len(frame.objects) == 0:
                continue
                
            # frame.data 已经是 normalized tensor [3, H, W]（来自 dataset transforms）
            img_t = frame.data
            obj = frame.objects[0]
            mask_t = obj.segment.float()  # [H, W]
            
            # 从 GT mask 生成 bbox
            bbox = masks_to_boxes(mask_t.unsqueeze(0))[0]  # [4]
            
            images_list.append(img_t)
            masks_list.append(mask_t)
            boxes_list.append(bbox)
            
        except Exception as e:
            logging.warning(f"Failed to load sample {idx}: {e}")
            continue
    
    # 如果采样不足，复制填充到 K_eff
    num_sampled = len(images_list)
    if num_sampled < K_eff and num_sampled > 0:
        logging.warning(f"Only sampled {num_sampled}/{K_eff}, replicating to fill")
        repeat_times = (K_eff + num_sampled - 1) // num_sampled
        images_list = (images_list * repeat_times)[:K_eff]
        boxes_list = (boxes_list * repeat_times)[:K_eff]
        masks_list = (masks_list * repeat_times)[:K_eff]
    elif num_sampled == 0:
        raise RuntimeError("No adversarial samples could be loaded from dataset!")
    
    logging.info(f"AUE adversarials sampled: {len(images_list)}/{K_eff} from dataset")
    
    return images_list, boxes_list, masks_list

