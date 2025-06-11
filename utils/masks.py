# utils/masks.py
"""Mask 生成与应用工具

该模块收集了 Phase-2 图像修复(inpainting)所需的常用 mask 生成函数，
并提供 create_mask 与 apply_mask 的统一入口，避免在不同脚本中重复实现。
所有实现均不依赖第三方库，仅使用 PyTorch 原生 API，
以满足"作业必须自行实现"的要求。
"""

from __future__ import annotations

import torch
from typing import Tuple

__all__ = [
    "center_mask",
    "random_mask",
    "left_half_mask",
    "top_half_mask",
    "stripes_mask",
    "create_mask",
    "apply_mask",
]

def _prepare_mask_tensor(image_shape: Tuple[int, int, int, int], device: torch.device | None) -> torch.Tensor:
    """创建全 1 的初始 mask 张量。"""
    batch_size, channels, height, width = image_shape
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.ones(image_shape, device=device)

def center_mask(image_shape: Tuple[int, int, int, int], mask_size: float = 0.5, device: torch.device | None = None) -> torch.Tensor:
    """在图像中心生成方形 mask。

    参数
    ----
    image_shape: 输入张量形状 (B, C, H, W)
    mask_size: 方形大小相对于长宽的比例 (0,1]
    device: 目标设备
    返回
    ----
    mask: 1 表示已知区域，0 表示需要修复的区域
    """
    mask = _prepare_mask_tensor(image_shape, device)
    _, _, height, width = image_shape

    mask_h = int(height * mask_size)
    mask_w = int(width * mask_size)
    start_h = (height - mask_h) // 2
    start_w = (width - mask_w) // 2
    mask[:, :, start_h : start_h + mask_h, start_w : start_w + mask_w] = 0
    return mask

def random_mask(image_shape: Tuple[int, int, int, int], mask_ratio: float = 0.5, device: torch.device | None = None) -> torch.Tensor:
    """随机散布遮挡像素的 mask。"""
    mask = _prepare_mask_tensor(image_shape, device)
    batch_size, _, height, width = image_shape
    num_pixels_to_mask = int(height * width * mask_ratio)
    for b in range(batch_size):
        # 在单张图片上随机选取像素位置
        indices = torch.randperm(height * width, device=mask.device)[:num_pixels_to_mask]
        flat = mask[b, 0].view(-1)
        flat[indices] = 0
        # 将同一通道的 mask 复制到其它通道
        mask[b] = flat.view(height, width).unsqueeze(0).repeat(mask.shape[1], 1, 1)
    return mask

def left_half_mask(image_shape: Tuple[int, int, int, int], device: torch.device | None = None) -> torch.Tensor:
    """遮挡左半幅图像。"""
    mask = _prepare_mask_tensor(image_shape, device)
    _, _, _, width = image_shape
    mask[:, :, :, : width // 2] = 0
    return mask

def top_half_mask(image_shape: Tuple[int, int, int, int], device: torch.device | None = None) -> torch.Tensor:
    """遮挡上半幅图像。"""
    mask = _prepare_mask_tensor(image_shape, device)
    _, _, height, _ = image_shape
    mask[:, :, : height // 2, :] = 0
    return mask

def stripes_mask(image_shape: Tuple[int, int, int, int], stripe_width: int = 2, gap: int = 2, device: torch.device | None = None) -> torch.Tensor:
    """生成水平条纹 mask (遮挡某些行)。

    stripe_width: 被遮挡的条纹宽度
    gap: 两条遮挡之间的间隙宽度
    """
    mask = _prepare_mask_tensor(image_shape, device)
    _, _, height, _ = image_shape
    # 每隔 (stripe_width + gap) 行遮挡 stripe_width 行
    step = stripe_width + gap
    for start in range(0, height, step):
        mask[:, :, start : start + stripe_width, :] = 0
    return mask

def create_mask(
    image_shape: Tuple[int, int, int, int],
    mask_type: str = "center",
    mask_size: float = 0.5,
    device: torch.device | None = None,
) -> torch.Tensor:
    """统一入口函数，根据 mask_type 调度对应 mask 生成器。"""
    mask_type = mask_type.lower()
    if mask_type == "center":
        return center_mask(image_shape, mask_size, device)
    if mask_type == "random":
        return random_mask(image_shape, mask_size, device)
    if mask_type == "left_half":
        return left_half_mask(image_shape, device)
    if mask_type == "top_half":
        return top_half_mask(image_shape, device)
    if mask_type == "stripes":
        return stripes_mask(image_shape, device=device)

    raise ValueError(f"Unsupported mask_type: {mask_type}")


def apply_mask(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """按元素相乘应用 mask。"""
    if image.shape != mask.shape:
        raise ValueError("image 和 mask 的形状必须一致")
    return image * mask 