#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conditional Diffusion on CIFAR-10

示例用法：
    python conditional_diffusion.py --train --epochs 20 --dataset "cifar"
    python conditional_diffusion.py --sample --num_samples 10 --class_id 3 --dataset "cifar"
"""
from __future__ import annotations

import argparse
import os

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
from models.conditional_ddpm import ConditionalDDPM_Model_CIFAR, ConditionalDDPM_Model_Celeba


def get_dataloader_cifar(batch_size: int = 128):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def get_dataloader_celeba(batch_size: int = 128):
    transform = T.Compose([
        T.CenterCrop(178),        # CelebA 的人脸裁剪
        T.Resize((64, 64)),       # or 32×32 视网络而定
        T.ToTensor(),
        T.Normalize((0.5,)*3, (0.5,)*3),
    ])
    dataset = ImageFolder(root="./data/celeba", transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Conditional DDPM on CIFAR-10")
    parser.add_argument("--train", action="store_true", help="是否执行训练")
    parser.add_argument("--epochs", type=int, default=50, help="训练 epoch 数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--model_path", type=str, default="", help="模型保存/加载路径（为空时自动根据数据集选择）")

    parser.add_argument("--sample", action="store_true", help="是否执行采样")
    parser.add_argument("--num_samples", type=int, default=10, help="采样张数")
    parser.add_argument("--class_id", type=int, default=0, help="要生成的 CIFAR10 类别 (0-9)")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"], help="设备")
    parser.add_argument("--dataset", type=str, default="cifar", choices=["cifar", "celeba"], help="数据集")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"使用设备: {device}")

    # 根据数据集自动选择模型文件名
    if not args.model_path:
        if args.dataset == "cifar":
            args.model_path = "conditional_diffusion_cifar.pth"
        elif args.dataset == "celeba":
            args.model_path = "conditional_diffusion_celeba.pth"

    # DataLoader
    if args.dataset == "cifar":
        dataloader = get_dataloader_cifar() if args.train else None
        model = ConditionalDDPM_Model_CIFAR(dataloader if dataloader is not None else [], T=1000, device=device)
    elif args.dataset == "celeba":
        dataloader = get_dataloader_celeba() if args.train else None
        model = ConditionalDDPM_Model_Celeba(dataloader if dataloader is not None else [], T=1000, device=device)
    
    # 如果存在模型文件就加载
    if os.path.exists(args.model_path):
        print(f"加载已有模型: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"模型文件不存在，将从头开始训练: {args.model_path}")

    # 训练
    if args.train:
        model.train_model(num_epochs=args.epochs, lr=args.lr, save_path=args.model_path)

    # 采样
    if args.sample:
        class_id = torch.full((args.num_samples,), args.class_id, device=device, dtype=torch.long)
        samples = model.sample(class_id).cpu()
        os.makedirs("cond_samples", exist_ok=True)
        save_path = f"cond_samples/class_{args.class_id}_{args.dataset}.png"
        vutils.save_image(samples, save_path, nrow=min(10, args.num_samples), normalize=True, value_range=(-1, 1))
        print(f"采样结果已保存到 {save_path}")


if __name__ == "__main__":
    main() 