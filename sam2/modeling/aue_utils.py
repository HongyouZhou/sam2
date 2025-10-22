# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for AUE (Adversarial Uncertainty Estimation) initialization."""

import logging
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


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
    H_adv: int,
    W_adv: int,
    device: torch.device,
    use_dataloader: bool = True,
    num_workers: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从训练数据集随机采样初始化对抗样本。
    
    Args:
        dataset: 训练数据集 (TorchTrainMixedDataset 或 VOSDataset)
        K_eff: 要采样的样本数量
        H_adv: 对抗图像的目标高度
        W_adv: 对抗图像的目标宽度
        device: 目标设备
        use_dataloader: 是否使用 DataLoader 进行并行加载（更快）
        num_workers: DataLoader 的 worker 数量
    
    Returns:
        adv_images: [K_eff, 3, H_adv, W_adv] - 对抗图像
        adv_boxes: [K_eff, 4] - bounding box prompts
        adv_masks: [K_eff, H_adv, W_adv] - ground truth masks
    """
    # 获取实际的 dataset（处理 TorchTrainMixedDataset 包装）
    actual_dataset = dataset.datasets[0] if hasattr(dataset, 'datasets') else dataset
    dataset_len = len(actual_dataset)
    
    # 随机采样索引（1.5 倍备用，减少过度采样）
    oversample_ratio = min(1.5, 1.0 + 100 / max(K_eff, 1))  # 动态调整
    num_to_sample = min(int(K_eff * oversample_ratio), dataset_len)
    indices = torch.randperm(dataset_len)[:num_to_sample].tolist()
    
    logging.info(f"Sampling {K_eff} adversarial samples from dataset (trying {num_to_sample} samples)...")
    
    images_list = []
    boxes_list = []
    masks_list = []
    
    # 快速路径：使用 DataLoader 并行加载
    if use_dataloader and num_workers > 0:
        try:
            subset = Subset(actual_dataset, indices)
            
            collate_fn = partial(_collate_adversarial_samples, H_adv=H_adv, W_adv=W_adv)
            
            batch_size = min(16, max(1, K_eff // 4))
            
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                timeout=30,
            )
            
            # 批量加载
            for batch_imgs, batch_masks in loader:
                if batch_imgs is None:
                    continue
                
                # 生成 bboxes
                batch_boxes = masks_to_boxes(batch_masks)
                
                images_list.extend(list(batch_imgs))
                masks_list.extend(list(batch_masks))
                boxes_list.extend(list(batch_boxes))
                
                if len(images_list) >= K_eff:
                    break
            
            logging.info(f"Fast path: loaded {len(images_list)} samples using DataLoader")
            
        except Exception as e:
            logging.warning(f"DataLoader failed ({e}), falling back to sequential loading")
            images_list = []
            boxes_list = []
            masks_list = []
    
    # 回退路径或补充采样：串行加载
    if len(images_list) < K_eff:
        batch_imgs = []
        batch_masks = []
        
        start_idx = len(images_list)
        for idx in indices[start_idx:]:
            if len(batch_imgs) >= K_eff - len(images_list):
                break
                
            try:
                # 获取样本
                sample = actual_dataset[idx]
                
                # 提取第一帧第一个对象
                frame = sample.frames[0]
                if len(frame.objects) == 0:
                    continue
                    
                img = frame.data  # PIL Image 或 Tensor
                obj = frame.objects[0]
                mask = obj.segment  # [H, W] uint8 tensor
                
                # 快速转换图像为 tensor（避免多次类型检查）
                if isinstance(img, torch.Tensor):
                    img_t = img.float() / 255.0 if img.dtype == torch.uint8 else img
                    if img_t.dim() == 2:  # 灰度图
                        img_t = img_t.unsqueeze(0).repeat(3, 1, 1)
                else:  # PIL Image
                    from PIL import Image  # noqa: F401
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                
                # 收集到批次中（延迟 resize）
                batch_imgs.append(img_t)
                batch_masks.append(mask.float())
                
            except Exception as e:
                logging.warning(f"Failed to load sample {idx}: {e}")
                continue
        
        # 批量 resize（关键优化：真正的批量处理）
        if batch_imgs:
            # 找到所有图像的尺寸，看是否可以直接堆叠
            shapes = [img.shape for img in batch_imgs]
            all_same_shape = all(s == shapes[0] for s in shapes)
            
            if all_same_shape and len(batch_imgs) > 1:
                # 所有图像尺寸相同，可以真正批量处理
                stacked_imgs = torch.stack(batch_imgs)  # [N, 3, H, W]
                batch_imgs_tensor = F.interpolate(
                    stacked_imgs, 
                    size=(H_adv, W_adv), 
                    mode='bilinear', 
                    align_corners=False
                )  # [N, 3, H_adv, W_adv]
                
                stacked_masks = torch.stack(batch_masks).unsqueeze(1)  # [N, 1, H, W]
                batch_masks_tensor = F.interpolate(
                    stacked_masks,
                    size=(H_adv, W_adv),
                    mode='nearest'
                ).squeeze(1)  # [N, H_adv, W_adv]
            else:
                # 尺寸不同，需要逐个处理（但仍比原来的方式快）
                batch_imgs_tensor = torch.stack([
                    F.interpolate(img.unsqueeze(0), size=(H_adv, W_adv), mode='bilinear', align_corners=False)[0]
                    for img in batch_imgs
                ])
                batch_masks_tensor = torch.stack([
                    F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(H_adv, W_adv), mode='nearest')[0, 0]
                    for mask in batch_masks
                ])
            
            # 批量生成 bboxes（向量化操作）
            batch_boxes = masks_to_boxes(batch_masks_tensor)  # [N, 4]
            
            images_list.extend(list(batch_imgs_tensor))
            masks_list.extend(list(batch_masks_tensor))
            boxes_list.extend(list(batch_boxes))
    
    # 如果采样不足，使用已有样本复制填充到 K_eff（避免随机噪声）
    num_sampled = len(images_list)
    if num_sampled < K_eff and num_sampled > 0:
        repeat_times = (K_eff + num_sampled - 1) // num_sampled
        images_list = (images_list * repeat_times)[:K_eff]
        boxes_list = (boxes_list * repeat_times)[:K_eff]
        masks_list = (masks_list * repeat_times)[:K_eff]
    elif num_sampled == 0:
        logging.warning("No adversarial samples could be loaded from dataset; falling back to zeros.")
        images_list = [torch.zeros(3, H_adv, W_adv)] * K_eff
        boxes_list = [torch.tensor([W_adv // 4, H_adv // 4, W_adv * 3 // 4, H_adv * 3 // 4], dtype=torch.float32)] * K_eff
        masks_list = [torch.zeros(H_adv, W_adv)] * K_eff
    
    # 一次性堆叠并移到设备（减少内存拷贝）
    adv_images = torch.stack(images_list[:K_eff]).to(device)
    adv_boxes = torch.stack(boxes_list[:K_eff]).to(device)
    adv_masks = torch.stack(masks_list[:K_eff]).to(device)
    
    logging.info(f"AUE adversarials sampled: {min(num_sampled, K_eff)}/{K_eff} from dataset, filled by replication if needed")
    
    return adv_images, adv_boxes, adv_masks

