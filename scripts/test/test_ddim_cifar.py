#!/usr/bin/env python3
"""
DDIM CIFAR-10 模型测试脚本
测试不同采样方法的效果和时间
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torchvision
import torchvision.transforms as T
import os
import matplotlib.pyplot as plt
import numpy as np
from models.ddim import DDIM_Model

def create_comparison_grid(model_path="../../ddim_model_cifar.pth", num_samples=4, output_path="../../ddim_step_comparison.png"):
    """
    创建展示不同步数采样质量变化的对比图
    包含初始噪声和四个不同步数的采样结果
    
    Args:
        model_path: 预训练模型路径
        num_samples: 每种配置的样本数量
        output_path: 输出对比图路径
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行 train_ddim_cifar.py 训练模型")
        return
    
    class FakeDataset(torch.utils.data.Dataset):
        def __getitem__(self, index):
            return torch.randn(3, 32, 32), 0
        def __len__(self):
            return num_samples
            
    dataloader = torch.utils.data.DataLoader(FakeDataset(), batch_size=num_samples)

    # 加载 DDIM 模型
    print(f"加载模型: {model_path}")
    model = DDIM_Model(dataloader, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 生成初始噪声图像
    torch.manual_seed(42)  # 固定随机种子确保可复现性
    initial_noise = torch.randn(num_samples, 3, 32, 32).to(device)
    
    # 采样配置
    sampling_configs = [
        {"name": "20 step", "steps": 20, "eta": 0.0},
        {"name": "50 step", "steps": 50, "eta": 0.0},
        {"name": "100 step", "steps": 100, "eta": 0.0},
        {"name": "50 step(random)", "steps": 50, "eta": 0.5}
    ]
    
    # 存储所有生成的图像
    all_images = []
    all_titles = []
    
    # 添加初始噪声
    all_images.append(initial_noise.cpu())
    all_titles.append("Initial Noise")
    
    print("开始生成不同步数的采样对比...")
    
    with torch.no_grad():
        for config in sampling_configs:
            print(f"生成 {config['name']} 采样...")
            
            # 使用相同的初始噪声
            torch.manual_seed(42)
            
            # 生成样本
            if hasattr(model, 'p_sample_loop'):
                shape = (num_samples, 3, 32, 32)
                samples = model.p_sample_loop(shape, ddim_steps=config["steps"], eta=config["eta"])
            else:
                samples = model.sample(num_samples, ddim_steps=config["steps"], eta=config["eta"])
            
            all_images.append(samples.cpu())
            all_titles.append(f"{config['name']}")
    
    # 创建对比图
    fig, axes = plt.subplots(len(all_images), num_samples, figsize=(num_samples * 3, len(all_images) * 3))
    fig.suptitle('DDIM Sampling Quality Evolution', fontsize=16, fontweight='bold')
    
    for row_idx, (images, title) in enumerate(zip(all_images, all_titles)):
        for col_idx in range(num_samples):
            ax = axes[row_idx, col_idx] if num_samples > 1 else axes[row_idx]
            
            # 转换图像格式用于显示
            img = images[col_idx]
            img = torch.clamp((img + 1) / 2, 0, 1)  # 从 [-1,1] 转换到 [0,1]
            img = img.permute(1, 2, 0).numpy()
            
            ax.imshow(img)
            ax.axis('off')
            
            # 只在第一列添加行标题
            if col_idx == 0:
                ax.text(-0.1, 0.5, title, transform=ax.transAxes, 
                       rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
            
            # 只在第一行添加列标题
            if row_idx == 0:
                ax.set_title(f'Sample {col_idx + 1}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 对比图已保存: {output_path}")
    print("对比图展示了从初始噪声到不同步数采样的质量变化过程")

def test_cifar_ddim(model_path="../../ddim_model_cifar.pth", num_samples=16, output_prefix="ddim_cifar_test"):
    """
    测试 DDIM CIFAR-10 模型，生成多种采样配置的样本
    
    Args:
        model_path: 预训练模型路径
        num_samples: 生成样本数量
        output_prefix: 输出文件前缀
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行 train_ddim_cifar.py 训练模型")
        return
    
    class FakeDataset(torch.utils.data.Dataset):
        def __getitem__(self, index):
            # 返回与模型输入匹配的随机张量
            return torch.randn(3, 32, 32), 0
        def __len__(self):
            return num_samples
            
    dataloader = torch.utils.data.DataLoader(FakeDataset(), batch_size=num_samples)

    # 加载 DDIM 模型
    print(f"加载模型: {model_path}")
    model = DDIM_Model(dataloader, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"开始生成 {num_samples} 个样本...")

    # 测试不同的 DDIM 采样配置
    sampling_configs = [
        {"name": "fast", "steps": 200, "eta": 0.0, "desc": "fast (200 step)"},
        {"name": "standard", "steps": 500, "eta": 0.0, "desc": "standard (500 step)"},
        {"name": "high_quality", "steps": 1000, "eta": 0.0, "desc": "high quality (100 step)"},
        {"name": "stochastic", "steps": 50, "eta": 0.5, "desc": "stochastic (500 step, eta=0.5)"}
    ]
    
    with torch.no_grad():
        for config in sampling_configs:
            print(f"生成 {config['desc']}...")
            
            # 生成样本
            if hasattr(model, 'p_sample_loop'):
                # 使用 p_sample_loop 方法
                shape = (num_samples, 3, 32, 32)
                samples = model.p_sample_loop(shape, ddim_steps=config["steps"], eta=config["eta"])
            else:
                # 备用方法：使用 sample 方法
                samples = model.sample(num_samples, ddim_steps=config["steps"], eta=config["eta"])
            
            # 保存图像
            output_path = f"{output_prefix}_{config['name']}.png"
            torchvision.utils.save_image(samples, output_path, nrow=4, normalize=True, value_range=(-1, 1))
            print(f"✅ 已保存: {output_path}")
    
    print("\n🎉 DDIM CIFAR-10 测试完成!")
    print("生成的图像文件:")
    for config in sampling_configs:
        print(f"  - {output_prefix}_{config['name']}.png ({config['desc']})")

def compare_ddim_speeds(model_path="../../ddim_model_cifar.pth", num_samples=8):
    """
    对比不同 DDIM 步数的生成速度
    """
    import time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    class FakeDataset(torch.utils.data.Dataset):
        def __getitem__(self, index):
            return torch.randn(3, 32, 32), 0
        def __len__(self):
            return num_samples
            
    dataloader = torch.utils.data.DataLoader(FakeDataset(), batch_size=num_samples)
    
    model = DDIM_Model(dataloader, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("\n⏱️  DDIM 速度对比测试:")
    print("=" * 40)
    
    step_configs = [10, 20, 50, 100, 200, 500]
    shape = (num_samples, 3, 32, 32)
    
    with torch.no_grad():
        for steps in step_configs:
            start_time = time.time()
            
            if hasattr(model, 'p_sample_loop'):
                samples = model.p_sample_loop(shape, ddim_steps=steps, eta=0.0)
            else:
                samples = model.sample(num_samples, ddim_steps=steps, eta=0.0)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            print(f"{steps:3d} 步: {elapsed:.2f} 秒 ({elapsed/num_samples:.3f} 秒/图)")

if __name__ == "__main__":
    # 创建步数对比图
    print("创建 DDIM 步数质量对比图...")
    create_comparison_grid()
    
    print("\n" + "="*50)
    
    # 基本测试
    test_cifar_ddim()
    
    # 速度对比测试
    print("\n" + "="*50)
    compare_ddim_speeds() 