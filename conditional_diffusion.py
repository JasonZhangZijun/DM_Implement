#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conditional Diffusion on CIFAR-10

示例用法：
    python conditional_diffusion.py --train --epochs 20
    python conditional_diffusion.py --sample --num_samples 10 --class_id 3
"""
from __future__ import annotations

import argparse
import os

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from models.conditional_ddpm import ConditionalDDPM_Model


def get_dataloader(batch_size: int = 128):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Conditional DDPM on CIFAR-10")
    parser.add_argument("--train", action="store_true", help="是否执行训练")
    parser.add_argument("--epochs", type=int, default=20, help="训练 epoch 数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--model_path", type=str, default="conditional_diffusion.pth", help="模型保存/加载路径")

    parser.add_argument("--sample", action="store_true", help="是否执行采样")
    parser.add_argument("--num_samples", type=int, default=10, help="采样张数")
    parser.add_argument("--class_id", type=int, default=0, help="要生成的 CIFAR10 类别 (0-9)")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"], help="设备")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"使用设备: {device}")

    # DataLoader
    dataloader = get_dataloader() if args.train else None

    # 初始化模型
    model = ConditionalDDPM_Model(dataloader if dataloader is not None else [], T=1000, device=device)

    # 如果存在模型文件就加载
    if os.path.exists(args.model_path):
        print(f"加载已有模型: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    # 训练
    if args.train:
        model.train_model(num_epochs=args.epochs, lr=args.lr, save_path=args.model_path)

    # 采样
    if args.sample:
        class_id = torch.full((args.num_samples,), args.class_id, device=device, dtype=torch.long)
        samples = model.sample(class_id).cpu()
        os.makedirs("cond_samples", exist_ok=True)
        save_path = f"cond_samples/class_{args.class_id}.png"
        vutils.save_image(samples, save_path, nrow=min(10, args.num_samples), normalize=True, value_range=(-1, 1))
        print(f"采样结果已保存到 {save_path}")


if __name__ == "__main__":
    main() 