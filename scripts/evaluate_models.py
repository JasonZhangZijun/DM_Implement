#!/usr/bin/env python3
"""
模型评估脚本
评估预训练模型的生成质量
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import argparse
from pathlib import Path
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import DataLoader
from models.ddpm import DDPM_Model
from models.ddim import DDIM_Model
from models.conditional_ddpm import ConditionalDDPM_Model_CIFAR, ConditionalDDPM_Model_Celeba
from utils.fid import calculate_fid_from_folders

def setup_cifar10_dataloader(batch_size=100):
    """设置 CIFAR-10 数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def setup_celeba_dataloader(batch_size=100):
    """设置 CelebA 数据加载器"""
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(root='./data/celeba', transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def generate_samples(model, num_samples, output_dir, dataset_name, is_conditional=False):
    """生成样本并保存"""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        if is_conditional:
            # 为每个类别生成相同数量的样本
            samples_per_class = num_samples // 10
            all_samples = []
            for label in range(10):
                labels = torch.full((samples_per_class,), label, device=model.device)
                samples = model.sample(labels)
                all_samples.append(samples)
            samples = torch.cat(all_samples, dim=0)
        else:
            if isinstance(model, DDIM_Model):
                samples = model.sample(num_samples, ddim_steps=100, eta=0.0)
            else:
                samples = model.sample(num_samples, dataset_name)
        
        # 保存生成的图像
        for i in range(num_samples):
            img = samples[i]
            img = (img + 1) / 2  # 从 [-1, 1] 转换到 [0, 1]
            torchvision.utils.save_image(img, f"{output_dir}/sample_{i:04d}.png")

def evaluate_model(model_name, model, dataloader, dataset_name, num_samples=1000):
    """评估单个模型"""
    print(f"\n评估 {model_name}...")
    
    # 生成样本
    output_dir = f"./evaluation/generated_{model_name.lower().replace(' ', '_')}"
    generate_samples(model, num_samples, output_dir, dataset_name, 
                    is_conditional='conditional' in model_name.lower())
    
    # 准备真实图像
    real_dir = "./evaluation/real_images"
    os.makedirs(real_dir, exist_ok=True)
    
    # 保存一些真实图像用于计算 FID
    if not os.path.exists(f"{real_dir}/image_0000.png"):
        print("保存真实图像样本...")
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i * dataloader.batch_size >= num_samples:
                    break
                for j, img in enumerate(images):
                    idx = i * dataloader.batch_size + j
                    if idx >= num_samples:
                        break
                    img = (img + 1) / 2  # 从 [-1, 1] 转换到 [0, 1]
                    torchvision.utils.save_image(img, f"{real_dir}/image_{idx:04d}.png")
    
    # 计算 FID 分数
    fid_score = calculate_fid_from_folders(real_dir, output_dir)
    return fid_score

def main():
    """主评测函数"""
    print("开始评测扩散模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 准备数据加载器
    cifar_loader = setup_cifar10_dataloader()
    celeba_loader = setup_celeba_dataloader()
    
    # 创建评测结果目录
    os.makedirs("./evaluation", exist_ok=True)
    
    # 模型配置
    models = {
        "DDPM CIFAR-10": {
            "model": DDPM_Model(cifar_loader, device=device),
            "path": "../../diffusion_model_cifar.pth",
            "loader": cifar_loader,
            "dataset": "cifar"
        },
        "DDIM CIFAR-10": {
            "model": DDIM_Model(cifar_loader, device=device),
            "path": "../../ddim_model_cifar.pth",
            "loader": cifar_loader,
            "dataset": "cifar"
        },
        "Conditional DDPM CIFAR-10": {
            "model": ConditionalDDPM_Model_CIFAR(cifar_loader, device=device),
            "path": "../../conditional_diffusion_cifar.pth",
            "loader": cifar_loader,
            "dataset": "cifar"
        },
        "DDPM CelebA": {
            "model": DDPM_Model(celeba_loader, device=device),
            "path": "../../diffusion_model_celeba.pth",
            "loader": celeba_loader,
            "dataset": "celeba"
        },
        "DDIM CelebA": {
            "model": DDIM_Model(celeba_loader, device=device),
            "path": "../../ddim_model_celeba.pth",
            "loader": celeba_loader,
            "dataset": "celeba"
        },
    }
    
    # 评测结果
    results = {}
    
    # 评估每个模型
    for name, config in models.items():
        if os.path.exists(config["path"]):
            print(f"\n加载模型 {name}...")
            config["model"].load_state_dict(torch.load(config["path"], map_location=device))
            fid = evaluate_model(name, config["model"], config["loader"], config["dataset"])
            results[name] = fid
            print(f"{name} FID 分数: {fid:.2f}")
        else:
            print(f"\n⚠️ 找不到模型文件: {config['path']}")
            print(f"跳过 {name} 的评测")
    
    # 绘制结果对比图
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    fid_scores = list(results.values())
    
    # CIFAR-10 和 CelebA 分开显示
    cifar_models = [m for m in models if "CIFAR-10" in m]
    celeba_models = [m for m in models if "CelebA" in m]
    cifar_scores = [results[m] for m in cifar_models]
    celeba_scores = [results[m] for m in celeba_models]
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(cifar_scores)), cifar_scores)
    plt.xticks(range(len(cifar_scores)), [m.replace("CIFAR-10", "").strip() for m in cifar_models], rotation=45)
    plt.title("CIFAR-10 Models FID Scores")
    plt.ylabel("FID Score")
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(celeba_scores)), celeba_scores)
    plt.xticks(range(len(celeba_scores)), [m.replace("CelebA", "").strip() for m in celeba_models], rotation=45)
    plt.title("CelebA Models FID Scores")
    plt.ylabel("FID Score")
    
    plt.tight_layout()
    plt.savefig("./evaluation/fid_comparison.png")
    print("\n已保存 FID 分数对比图到 ./evaluation/fid_comparison.png")
    
    # 保存数值结果
    with open("./evaluation/fid_scores.txt", "w") as f:
        f.write("FID Scores:\n")
        f.write("=" * 50 + "\n")
        for name, score in results.items():
            f.write(f"{name}: {score:.2f}\n")
    print("已保存详细评测结果到 ./evaluation/fid_scores.txt")

if __name__ == "__main__":
    main() 