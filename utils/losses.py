"""基础图像质量评估指标实现（PSNR / SSIM）。

完全基于 PyTorch，避免使用第三方库，以符合课程要求。
当前实现面向通用张量格式 (B, C, H, W)，支持 CPU / GPU 运算。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple

__all__ = ["psnr", "ssim"]

_EPS = 1e-8  # 避免除零


def _to_float32(t: torch.Tensor) -> torch.Tensor:
    """将输入转换到 float32，保持梯度。"""
    if t.dtype != torch.float32:
        t = t.float()
    return t


def _denormalize(img: torch.Tensor) -> torch.Tensor:
    """将 [-1,1] 或其它范围的张量压缩至 [0,1]。

    该函数假设输入大部分数据位于 [-1,1] 或 [0,1]，
    若超出该区间，则会被 clamp。"""
    return ((img + 1) / 2).clamp(0, 1) if img.min() < 0 else img.clamp(0, 1)


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_value: float | None = 1.0) -> torch.Tensor:
    """计算 Peak Signal-to-Noise Ratio (PSNR)。

    参数
    ----
    img1, img2: 形状相同的张量 (B, C, H, W) 或 (C, H, W)。
    max_value: 数据的最大可能值；若为 None，则自动根据数据范围确定。

    返回
    ----
    psnr 值 (逐张计算后求平均)
    """
    img1 = _to_float32(_denormalize(img1))
    img2 = _to_float32(_denormalize(img2))
    mse = F.mse_loss(img1, img2, reduction="none")
    mse = mse.view(mse.size(0), -1).mean(dim=1)  # 每张图像的 mse
    if max_value is None:
        max_value = 1.0  # 归一化后
    psnr_val = 20 * torch.log10(max_value / torch.sqrt(mse + _EPS))
    return psnr_val.mean()


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    kernel_size: int = 7,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """计算结构相似性 (SSIM)。

    使用简单的均值高斯核近似，避免外部依赖；
    与 skimage 的实现在数值上略有差异，但趋势一致。
    """
    img1 = _to_float32(_denormalize(img1))
    img2 = _to_float32(_denormalize(img2))

    # 构造均值卷积核 (box filter)
    channel = img1.size(1)
    kernel = torch.ones((channel, 1, kernel_size, kernel_size), device=img1.device) / (kernel_size ** 2)

    mu1 = F.conv2d(img1, kernel, padding=kernel_size // 2, groups=channel)
    mu2 = F.conv2d(img2, kernel, padding=kernel_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=kernel_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=kernel_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel_size // 2, groups=channel) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() 