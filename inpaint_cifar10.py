#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CIFAR-10 批量 Inpainting 测试脚本

在最少改动的基础上，使用现有 `InpaintDDPM` 模型对 CIFAR-10 *测试集* 进行图像修复，
并统计整体 PSNR / SSIM。

运行方式 (示例)：
    python inpaint_cifar10.py --model_path diffusion_model.pth \
                              --mask_type center \
                              --mask_size 0.5 \
                              --limit 1000

该脚本不会进行任何训练，仅加载预训练权重，如未提供则采用随机初始化（效果可能较差）。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from inpaint_ddpm import InpaintDDPM

# ----------------------------- CLI ----------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 批量 Inpainting 测试")
    parser.add_argument("--model_path", type=str, default="diffusion_model.pth", help="预训练模型路径")
    parser.add_argument("--batch_size", type=int, default=64, help="推断批大小")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto", help="运行设备")
    parser.add_argument("--mask_type", type=str, default="center", help="遮罩类型，详见 README")
    parser.add_argument("--mask_size", type=float, default=0.5, help="遮罩大小比例 (0,1]")
    parser.add_argument("--use_improved", action="store_true", default=False, help="使用改进算法")
    parser.add_argument("--output", type=str, default="cifar_inpaint_results", help="输出目录，用于保存示例图像")
    parser.add_argument("--limit", type=int, default=500, help="处理前 N 张图片 (<=0 表示全部)")
    return parser.parse_args()

# --------------------------- Helpers --------------------------- #

def to_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    return batch.to(device, non_blocking=True)

# ----------------------------- Main ---------------------------- #

def main() -> None:
    args = parse_args()

    # 设备选择
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] 使用设备: {device}")

    # CIFAR-10 测试集
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 转换至 [-1,1]
    ])
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    if args.limit > 0:
        test_set = torch.utils.data.Subset(test_set, range(args.limit))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型（DataLoader 仅用于兼容，实际不训练）
    dummy_loader = DataLoader(test_set, batch_size=args.batch_size)
    model = InpaintDDPM(dummy_loader, T=200, device=device, output_dir=args.output)
    model.load_pretrained_model(args.model_path)

    os.makedirs(args.output, exist_ok=True)

    # 统计指标
    total_psnr, total_ssim, count = 0.0, 0.0, 0

    print("[INFO] 开始批量 Inpainting...")
    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(tqdm(test_loader)):
            imgs = to_device(imgs, device)
            # 创建并应用 mask
            mask = model.create_mask(imgs.shape, args.mask_type, args.mask_size)
            imgs_masked = model.apply_mask(imgs, mask)

            # 推断
            if args.use_improved:
                recon = model.inpaint_improved(imgs, mask, num_samples=1)
            else:
                recon = model.inpaint(imgs, mask, num_samples=1)

            # 评估
            metrics = model.evaluate_quality(imgs, recon)
            batch_size = imgs.size(0)
            total_psnr += metrics["PSNR"] * batch_size
            total_ssim += metrics["SSIM"] * batch_size
            count += batch_size

            # 保存前 8 张示例（仅第一次迭代）
            if batch_idx == 0:
                for i in range(min(8, batch_size)):
                    save_path = os.path.join(args.output, f"sample_{i}.png")
                    model.save_inpaint_results(
                        imgs[i:i+1].cpu(),
                        imgs_masked[i:i+1].cpu(),
                        recon[i:i+1].cpu(),
                        mask[i:i+1].cpu(),
                        save_path,
                    )

    # 输出平均指标
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"[RESULT] 平均 PSNR: {avg_psnr:.2f} dB, 平均 SSIM: {avg_ssim:.4f}")
    print(f"示例结果已保存至: {args.output}/sample_*.png")


if __name__ == "__main__":
    main() 