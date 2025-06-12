#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDIM vs DDPM 对比测试脚本
比较两种模型在 CIFAR-10 和 CelebA 数据集上的生成效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

from models.ddpm import DDPM_Model
from models.ddim import DDIM_Model

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class ModelComparison:
    def __init__(self):
        self.device = device
        self.results = {}
        
    def load_cifar10_models(self):
        """加载 CIFAR-10 模型"""
        print("正在加载 CIFAR-10 模型...")
        
        # 准备 CIFAR-10 数据加载器
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = CIFAR10(root='./data', train=True, download=False, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        
        # 加载 DDPM 模型
        try:
            self.ddpm_cifar = DDPM_Model(dataloader)
            if os.path.exists("../../diffusion_model_cifar.pth"):
                self.ddpm_cifar.load_state_dict(torch.load("../../diffusion_model_cifar.pth", map_location=device))
                print("✅ DDPM CIFAR-10 模型加载成功")
            else:
                print("⚠️ 未找到 DDPM CIFAR-10 模型权重文件")
                self.ddpm_cifar = None
        except Exception as e:
            print(f"❌ DDPM CIFAR-10 模型加载失败: {e}")
            self.ddpm_cifar = None
            
        # 加载 DDIM 模型
        try:
            self.ddim_cifar = DDIM_Model(dataloader, device=device)
            if os.path.exists("../../ddim_model_cifar.pth"):
                self.ddim_cifar.load_state_dict(torch.load("../../ddim_model_cifar.pth", map_location=device))
                print("✅ DDIM CIFAR-10 模型加载成功")
            else:
                print("⚠️ 未找到 DDIM CIFAR-10 模型权重文件")
                self.ddim_cifar = None
        except Exception as e:
            print(f"❌ DDIM CIFAR-10 模型加载失败: {e}")
            self.ddim_cifar = None
    
    def load_celeba_models(self):
        """加载 CelebA 模型"""
        print("正在加载 CelebA 模型...")
        
        # 准备 CelebA 数据加载器
        transform = T.Compose([
            T.CenterCrop(178),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        if os.path.exists('./data/celeba'):
            dataset = ImageFolder(root='./data/celeba', transform=transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        else:
            print("⚠️ 未找到 CelebA 数据集，使用虚拟数据加载器")
            dataloader = None
        
        # 加载 DDPM CelebA 模型
        try:
            if dataloader:
                self.ddpm_celeba = DDPM_Model(dataloader)
            else:
                self.ddpm_celeba = None
            if os.path.exists("../../diffusion_model_celeba.pth"):
                if self.ddpm_celeba:
                    self.ddpm_celeba.load_state_dict(torch.load("../../diffusion_model_celeba.pth", map_location=device))
                print("✅ DDPM CelebA 模型权重文件找到")
            else:
                print("⚠️ 未找到 DDPM CelebA 模型权重文件")
                self.ddpm_celeba = None
        except Exception as e:
            print(f"❌ DDPM CelebA 模型加载失败: {e}")
            self.ddpm_celeba = None
            
        # 加载 DDIM CelebA 模型
        try:
            if dataloader:
                self.ddim_celeba = DDIM_Model(dataloader, device=device)
            else:
                self.ddim_celeba = None
            if os.path.exists("../../ddim_model_celeba.pth"):
                if self.ddim_celeba:
                    self.ddim_celeba.load_state_dict(torch.load("../../ddim_model_celeba.pth", map_location=device))
                print("✅ DDIM CelebA 模型权重文件找到")
            else:
                print("⚠️ 未找到 DDIM CelebA 模型权重文件")
                self.ddim_celeba = None
        except Exception as e:
            print(f"❌ DDIM CelebA 模型加载失败: {e}")
            self.ddim_celeba = None

    def generate_samples(self, model, model_type, dataset_name, num_samples=8, ddim_steps=50):
        """生成样本并记录时间"""
        if model is None:
            print(f"⚠️ {model_type} {dataset_name} 模型不可用，生成随机样本")
            if dataset_name == "CIFAR-10":
                return torch.randn(num_samples, 3, 32, 32), 0.0
            else:  # CelebA
                return torch.randn(num_samples, 3, 64, 64), 0.0
        
        print(f"正在生成 {model_type} {dataset_name} 样本...")
        start_time = time.time()
        
        try:
            with torch.no_grad():
                if model_type == "DDPM":
                    if dataset_name == "CIFAR-10":
                        samples = model.sample(num_samples, "cifar")
                    else:  # CelebA
                        samples = model.sample(num_samples, "celeba")
                else:  # DDIM
                    if dataset_name == "CIFAR-10":
                        samples = model.sample(num_samples, ddim_steps=ddim_steps)
                    else:  # CelebA
                        # 为 CelebA 重写采样方法
                        shape = (num_samples, 3, 64, 64)
                        samples = model.p_sample_loop(shape, ddim_steps=ddim_steps)
        except Exception as e:
            print(f"❌ 生成样本时出错: {e}")
            if dataset_name == "CIFAR-10":
                return torch.randn(num_samples, 3, 32, 32), 0.0
            else:
                return torch.randn(num_samples, 3, 64, 64), 0.0
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        return samples.cpu(), generation_time

    def create_comparison_grid(self):
        """创建对比网格图像"""
        print("正在创建对比网格...")
        
        # 生成所有样本
        samples_data = {}
        
        # CIFAR-10 样本
        if self.ddpm_cifar or self.ddim_cifar:
            ddpm_cifar_samples, ddpm_cifar_time = self.generate_samples(
                self.ddpm_cifar, "DDPM", "CIFAR-10", num_samples=8)
            ddim_cifar_samples, ddim_cifar_time = self.generate_samples(
                self.ddim_cifar, "DDIM", "CIFAR-10", num_samples=8, ddim_steps=50)
            
            samples_data["CIFAR-10"] = {
                "DDPM": (ddpm_cifar_samples, ddpm_cifar_time),
                "DDIM": (ddim_cifar_samples, ddim_cifar_time)
            }
        
        # CelebA 样本
        if self.ddpm_celeba or self.ddim_celeba:
            ddpm_celeba_samples, ddpm_celeba_time = self.generate_samples(
                self.ddpm_celeba, "DDPM", "CelebA", num_samples=8)
            ddim_celeba_samples, ddim_celeba_time = self.generate_samples(
                self.ddim_celeba, "DDIM", "CelebA", num_samples=8, ddim_steps=50)
            
            samples_data["CelebA"] = {
                "DDPM": (ddpm_celeba_samples, ddpm_celeba_time),
                "DDIM": (ddim_celeba_samples, ddim_celeba_time)
            }
        
        # 创建图像网格
        self.create_visual_comparison(samples_data)
    
    def create_visual_comparison(self, samples_data):
        """创建可视化对比图"""
        # 设置图像大小和布局
        fig_width = 16
        fig_height = 12
        fig, axes = plt.subplots(4, 8, figsize=(fig_width, fig_height))
        fig.suptitle('DDPM vs DDIM 生成效果对比', fontsize=20, fontweight='bold')
        
        row_labels = []
        datasets = list(samples_data.keys())
        
        current_row = 0
        
        for dataset in datasets:
            for model_type in ["DDPM", "DDIM"]:
                if model_type in samples_data[dataset]:
                    samples, gen_time = samples_data[dataset][model_type]
                    
                    # 添加行标签
                    row_labels.append(f"{dataset}\n{model_type}\n({gen_time:.1f}s)")
                    
                    # 显示8个样本
                    for col in range(8):
                        ax = axes[current_row, col]
                        
                        if col < samples.shape[0]:
                            # 反归一化图像
                            img = samples[col]
                            img = (img + 1) / 2  # 从 [-1,1] 到 [0,1]
                            img = torch.clamp(img, 0, 1)
                            
                            # 转换为 numpy 并调整维度
                            img_np = img.permute(1, 2, 0).numpy()
                            
                            ax.imshow(img_np)
                        else:
                            ax.axis('off')
                        
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                        # 添加列标题（仅第一行）
                        if current_row == 0:
                            ax.set_title(f'样本 {col+1}', fontsize=10)
                    
                    current_row += 1
        
        # 添加行标签
        for i, label in enumerate(row_labels):
            axes[i, 0].set_ylabel(label, rotation=0, labelpad=80, 
                                  fontsize=12, ha='right', va='center')
        
        # 隐藏未使用的子图
        for i in range(current_row, 4):
            for j in range(8):
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, left=0.15)
        
        # 保存图像
        output_path = "../../ddim_vs_ddpm_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 对比图像已保存为: {output_path}")
        
        # 显示图像
        plt.show()
        
        return output_path
    
    def run_comparison(self):
        """运行完整的对比测试"""
        print("🚀 开始 DDIM vs DDPM 对比测试")
        print("=" * 50)
        
        # 加载模型
        self.load_cifar10_models()
        self.load_celeba_models()
        
        print("\n" + "=" * 50)
        print("📊 生成对比样本...")
        
        # 创建对比网格
        self.create_comparison_grid()
        
        print("\n" + "=" * 50)
        print("🎉 对比测试完成！")

def main():
    """主函数"""
    print("DDIM vs DDPM 效果对比测试")
    print("本脚本将对比两种扩散模型在不同数据集上的生成效果")
    print("请确保已有训练好的模型权重文件")
    print("-" * 60)
    
    # 检查模型文件
    model_files = [
        "../../diffusion_model_cifar.pth",
        "../../ddim_model_cifar.pth", 
        "../../diffusion_model_celeba.pth",
        "../../ddim_model_celeba.pth"
    ]
    
    print("📁 检查模型文件...")
    for file in model_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"⚠️ {file} (不存在)")
    
    print("\n" + "-" * 60)
    
    # 运行对比
    comparison = ModelComparison()
    comparison.run_comparison()

if __name__ == "__main__":
    main() 