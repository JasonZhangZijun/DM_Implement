#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CelebA-HQ 批量 Inpainting 测试脚本

用法示例：
    python inpaint_celeba.py --model_path diffusion_model_celeba.pth \
                             --mask_type center --mask_size 0.4 \
                             --use_improved --limit 1000

默认假设图片存放于  ./data/celeba/img  并已统一调整为对齐人脸 (178×178) 或更大尺寸。
如果你采用 Kaggle / 官方目录，可通过 --data_root 指定路径。
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from inpaint_ddpm import InpaintDDPM

# ---------------- CLI ---------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CelebA 批量 Inpainting")
    parser.add_argument("--data_root", type=str, default="./data/celeba", help="包含 img 子目录的根路径")
    parser.add_argument("--model_path", type=str, default="diffusion_model_celeba.pth", help="预训练模型路径")
    parser.add_argument("--batch_size", type=int, default=32, help="推断批大小")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto", help="运行设备")
    parser.add_argument("--mask_type", type=str, default="center", help="遮罩类型")
    parser.add_argument("--mask_size", type=float, default=0.5, help="遮罩大小比例 (0,1]")
    parser.add_argument("--use_improved", action="store_true", default=False, help="使用改进算法")
    parser.add_argument("--output", type=str, default="celeba_inpaint_results", help="输出目录")
    parser.add_argument("--limit", type=int, default=500, help="处理前 N 张图片 (<=0 表示全部)")
    return parser.parse_args()

# -------------- Helpers -------------- #

def build_dataloader(root: str, batch_size: int, limit: int) -> DataLoader:
    transform = T.Compose([
        T.CenterCrop(178),
        T.Resize((64, 64)),   # 与训练分辨率匹配
        T.ToTensor(),
        T.Normalize((0.5,)*3, (0.5,)*3),
    ])
    ds = ImageFolder(root=root, transform=transform)
    if limit > 0:
        indices = list(range(min(limit, len(ds))))
        ds = torch.utils.data.Subset(ds, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def to_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    return batch.to(device, non_blocking=True)

# -------------- Main ----------------- #

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # 加载数据
    data_root = os.path.join(args.data_root, "img") if os.path.isdir(os.path.join(args.data_root, "img")) else args.data_root
    dataloader = build_dataloader(data_root, args.batch_size, args.limit)

    # Dummy loader 仅为兼容 InpaintDDPM
    dummy_loader = DataLoader(dataloader.dataset, batch_size=args.batch_size)
    model = InpaintDDPM(dummy_loader, T=1000, device=device, output_dir=args.output)
    model.load_pretrained_model(args.model_path)
    os.makedirs(args.output, exist_ok=True)

    total_psnr = total_ssim = count = 0.0
    print("[INFO] 开始 CelebA 批量 Inpainting…")

    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(tqdm(dataloader)):
            imgs = to_device(imgs, device)
            mask = model.create_mask(imgs.shape, args.mask_type, args.mask_size)
            imgs_masked = model.apply_mask(imgs, mask)

            recon = model.inpaint_improved(imgs, mask) if args.use_improved else model.inpaint(imgs, mask)

            metrics = model.evaluate_quality(imgs, recon)
            bs = imgs.size(0)
            total_psnr += metrics["PSNR"] * bs
            total_ssim += metrics["SSIM"] * bs
            count += bs

            if batch_idx == 0:
                for i in range(min(8, bs)):
                    save_path = os.path.join(args.output, f"sample_{i}.png")
                    model.save_inpaint_results(imgs[i:i+1].cpu(), imgs_masked[i:i+1].cpu(), recon[i:i+1].cpu(), mask[i:i+1].cpu(), save_path)

    print(f"[RESULT] 平均 PSNR: {total_psnr / count:.2f} dB, 平均 SSIM: {total_ssim / count:.4f}")
    print(f"示例结果保存在: {args.output}/sample_*.png")


if __name__ == "__main__":
    main() 