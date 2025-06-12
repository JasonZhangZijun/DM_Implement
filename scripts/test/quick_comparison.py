#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDIM vs DDPM Quick Comparison - 简化版本
对比 DDIM 和 DDPM 的采样速度和质量
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, ImageFolder
import matplotlib.pyplot as plt
import time
import numpy as np

from models.ddpm import DDPM_Model
from models.ddim import DDIM_Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(name):
    """加载数据集，返回 DataLoader 或 None"""
    try:
        if name == 'cifar':
            transform = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
            dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
            return torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        elif name == 'celeba' and os.path.exists('./data/celeba'):
            transform = T.Compose([
                T.CenterCrop(178), T.Resize((64, 64)), T.ToTensor(),
                T.Normalize((0.5,)*3, (0.5,)*3)
            ])
            dataset = ImageFolder(root='./data/celeba', transform=transform)
            return torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    except Exception as e:
        print(f"❌ 无法加载 {name} 数据集: {e}")
    return None

def load_model(model_type, dataset_name, model_path, dataloader):
    """加载单个模型"""
    if not dataloader or not os.path.exists(model_path):
        return None
    
    try:
        model_class = DDPM_Model if model_type == "DDPM" else DDIM_Model
        model = model_class(dataloader, device=DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✅ 已加载 {model_type} 模型 ({dataset_name})")
        return model
    except Exception as e:
        print(f"⚠️ 加载 {model_path} 失败: {e}")
        return None

def generate_samples(model, model_type, dataset_name, num_samples=8):
    """为单个模型生成样本"""
    print(f"生成 {model_type} 样本 ({dataset_name})...")
    
    size = (32, 32) if 'cifar' in dataset_name else (64, 64)
    shape = (num_samples, 3, *size)
    
    # DDPM: 只有一种配置 (1000 steps)
    if model_type == "DDPM":
        start_time = time.time()
        with torch.no_grad():
            samples = model.sample(num_samples, "cifar" if "cifar" in dataset_name else "celeba")
        gen_time = time.time() - start_time
        return [{"samples": samples.cpu(), "time": gen_time, "steps": 1000, "name": "Standard"}]
    
    # DDIM: 多种配置
    else:
        configs = [("Fast", 100), ("Standard", 500), ("High Quality", 1000)]
        results = []
        for name, steps in configs:
            start_time = time.time()
            with torch.no_grad():
                samples = model.p_sample_loop(shape, ddim_steps=steps, eta=0.0)
            gen_time = time.time() - start_time
            results.append({
                "samples": samples.cpu(), "time": gen_time, 
                "steps": steps, "name": name
            })
            print(f"  - {name} ({steps} 步): {gen_time:.2f}秒")
        return results

def create_comparison_plot(all_results):
    """创建对比图"""
    print("🎨 生成对比图...")
    
    # 计算布局
    total_rows = sum(len(results) for results in all_results.values())
    if total_rows == 0:
        print("❌ 没有结果可显示")
        return
    
    fig, axes = plt.subplots(total_rows, 8, figsize=(16, 2.5 * total_rows + 1))
    if total_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('DDIM vs DDPM Sampling Comparison', fontsize=16, fontweight='bold')
    
    row_idx = 0
    for (model_type, dataset), results in all_results.items():
        for result in results:
            # 显示样本
            for col_idx in range(8):
                ax = axes[row_idx, col_idx]
                if col_idx < len(result['samples']):
                    img = (result['samples'][col_idx].permute(1, 2, 0) + 1) / 2
                    ax.imshow(torch.clamp(img, 0, 1))
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(f'Sample {col_idx+1}', fontsize=10)
            
            # 行标签
            label = f"{model_type} @ {dataset.upper()}\n{result['name']} ({result['steps']} steps)\n{result['time']:.1f}s"
            axes[row_idx, 0].set_ylabel(label, rotation=0, labelpad=120, fontsize=11, ha='right', va='center')
            row_idx += 1
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, left=0.22)
    
    output_path = "ddim_ddpm_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存: {output_path}")

def print_performance_report(all_results):
    """打印性能报告"""
    print("\n📈 性能报告:")
    print("=" * 50)
    
    for (model_type, dataset), results in all_results.items():
        print(f"\n--- {dataset.upper()} Dataset ---")
        if model_type == "DDPM":
            result = results[0]
            print(f"🔵 DDPM: {result['time']:.2f}s ({result['steps']} steps)")
        else:
            print("🟢 DDIM:")
            for result in results:
                print(f"  - {result['name']}: {result['time']:.2f}s ({result['steps']} steps)")

def main():
    """主函数"""
    print("🚀 DDIM vs DDPM 快速对比")
    print(f"使用设备: {DEVICE}")
    
    # 加载数据集和模型
    datasets = {'cifar': load_dataset('cifar'), 'celeba': load_dataset('celeba')}
    model_configs = [
        ("DDPM", "cifar", "../../diffusion_model_cifar.pth"),
        ("DDIM", "cifar", "../../ddim_model_cifar.pth"),
        ("DDPM", "celeba", "../../diffusion_model_celeba.pth"),
        ("DDIM", "celeba", "../../ddim_model_celeba.pth"),
    ]
    
    all_results = {}
    
    for model_type, dataset_name, model_path in model_configs:
        dataloader = datasets.get(dataset_name)
        model = load_model(model_type, dataset_name, model_path, dataloader)
        
        if model:
            try:
                results = generate_samples(model, model_type, dataset_name)
                all_results[(model_type, dataset_name)] = results
            except Exception as e:
                print(f"❌ 生成样本失败 ({model_type}, {dataset_name}): {e}")
    
    if not all_results:
        print("❌ 没有可用的模型和数据集")
        return
    
    # 生成结果
    create_comparison_plot(all_results)
    print_performance_report(all_results)
    print("\n🎉 对比完成!")

if __name__ == "__main__":
    main() 