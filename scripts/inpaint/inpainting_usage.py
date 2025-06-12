#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的图像修复工具
支持多种掩码类型和数据集

作者: 助手
日期: 2024
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import torch
from inpaint_ddpm import InpaintDDPM

# 创建虚拟数据加载器
class DummyDataLoader:
    def __init__(self):
        self.batch_size = 4
    def __iter__(self):
        for i in range(2):
            fake_data = torch.randn(self.batch_size, 3, 32, 32)
            fake_labels = torch.randint(0, 10, (self.batch_size,))
            yield fake_data, fake_labels

def setup_model():
    """初始化inpainting模型"""
    print("🚀 正在初始化DDPM inpainting模型...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = DummyDataLoader()
    
    # 创建模型（使用较小的T进行快速测试）
    model = InpaintDDPM(dataloader, T=50, device=device, output_dir="inpainting_outputs")
    
    # 尝试加载预训练模型
    model_path = 'ddpm_model.pth'
    if os.path.exists(model_path):
        model.load_pretrained_model(model_path)
        print(f"✅ 加载预训练模型: {model_path}")
    else:
        print("⚠️ 未找到预训练模型，使用随机初始化（效果可能不佳）")
    
    return model

def demo_basic_inpainting():
    """基本inpainting演示"""
    print("\n" + "="*50)
    print("📸 基本Inpainting演示")
    print("="*50)
    
    model = setup_model()
    
    # 运行基本演示
    model.demo_inpainting(use_improved=True)
    print("✅ 基本演示完成！查看生成的图像文件。")

def demo_custom_mask():
    """自定义mask演示"""
    print("\n" + "="*50)
    print("🎭 自定义Mask演示")
    print("="*50)
    
    model = setup_model()
    
    # 创建一个随机图像
    device = model.device
    x_0 = torch.randn(1, 3, 32, 32).to(device)
    
    # 创建不同类型的mask
    mask_configs = [
        ("center", 0.3, "小的中心方形mask"),
        ("center", 0.7, "大的中心方形mask"),
        ("random", 0.4, "随机40%遮罩"),
        ("left_half", 0.5, "左半边遮罩"),
        ("stripes", 0.5, "条纹遮罩")
    ]
    
    for mask_type, mask_size, description in mask_configs:
        print(f"\n正在处理: {description}")
        
        # 创建mask
        mask = model.create_mask(x_0.shape, mask_type, mask_size)
        
        # 应用mask
        x_masked = model.apply_mask(x_0, mask)
        
        # 执行inpainting
        result = model.inpaint_improved(x_0, mask, num_samples=1)
        
        # 保存结果
        save_path = f"custom_mask_{mask_type}_{mask_size}.png"
        model.save_inpaint_results(x_0, x_masked, result, mask, save_path, category="custom_mask")
        print(f"  💾 结果保存到: {save_path}")

def demo_multiple_samples():
    """多样本生成演示"""
    print("\n" + "="*50)
    print("🎲 多样本生成演示")
    print("="*50)
    
    model = setup_model()
    
    # 创建一个图像和mask
    device = model.device
    x_0 = torch.randn(1, 3, 32, 32).to(device)
    mask = model.create_mask(x_0.shape, "center", 0.5)
    x_masked = model.apply_mask(x_0, mask)
    
    # 生成多个不同的修复结果
    num_samples = 3
    print(f"为同一张图像生成 {num_samples} 个不同的修复结果...")
    
    results = model.inpaint_improved(x_0, mask, num_samples=num_samples)
    
    # 保存每个结果
    for i in range(num_samples):
        save_path = f"multiple_samples_result_{i+1}.png"
        model.save_inpaint_results(x_0, x_masked, results[i:i+1], mask, save_path, category="multiple_samples")
        print(f"  💾 样本 {i+1} 保存到: {save_path}")

def demo_algorithm_comparison():
    """算法对比演示"""
    print("\n" + "="*50)
    print("⚖️ 算法对比演示")
    print("="*50)
    
    model = setup_model()
    
    # 创建测试图像
    device = model.device
    x_0 = torch.randn(1, 3, 32, 32).to(device)
    mask = model.create_mask(x_0.shape, "center", 0.4)
    x_masked = model.apply_mask(x_0, mask)
    
    print("比较标准算法和改进算法的效果...")
    
    # 标准算法
    print("  🔄 运行标准算法...")
    result_standard = model.inpaint(x_0, mask, num_samples=1)
    model.save_inpaint_results(x_0, x_masked, result_standard, mask, 
                              "algorithm_comparison_standard.png", category="algorithm_comparison")
    
    # 改进算法
    print("  🔄 运行改进算法...")
    result_improved = model.inpaint_improved(x_0, mask, num_samples=1)
    model.save_inpaint_results(x_0, x_masked, result_improved, mask, 
                              "algorithm_comparison_improved.png", category="algorithm_comparison")
    
    print("  💾 对比结果已保存")
    print("  📊 你可以比较两个算法的效果差异")

def demo_with_real_image():
    """真实图像演示（如果有的话）"""
    print("\n" + "="*50)
    print("🖼️ 真实图像Inpainting演示")
    print("="*50)
    
    model = setup_model()
    
    # 检查是否有现有的图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            # 排除我们生成的演示图像，以及输出目录中的图像
            if (not file.startswith('inpaint_demo_') and 
                not file.startswith('custom_mask_') and
                not file.startswith('multiple_samples_') and
                not file.startswith('algorithm_comparison_') and
                not file.startswith('real_image_inpaint_') and
                not file.startswith('interactive_demo_')):
                image_files.append(file)
    
    if image_files:
        print(f"找到 {len(image_files)} 个图像文件，将进行inpainting演示:")
        for img_file in image_files[:3]:  # 最多处理3个文件
            print(f"  📷 处理图像: {img_file}")
            try:
                # 使用中心mask进行inpainting
                result_path = f"real_image_inpaint_{img_file}"
                model.inpaint_from_image(
                    img_file, 
                    mask_type="center", 
                    mask_size=0.4,
                    save_path=result_path,
                    use_improved=True,
                    category="real_images"
                )
                print(f"    ✅ 完成，结果保存到: real_images/{result_path}")
            except Exception as e:
                print(f"    ❌ 处理失败: {e}")
    else:
        print("未找到合适的图像文件进行演示")
        print("你可以将图像文件放在当前目录下，然后重新运行此演示")

def interactive_demo():
    """交互式演示"""
    print("\n" + "="*50)
    print("🎮 交互式Inpainting演示")
    print("="*50)
    
    model = setup_model()
    
    print("可用的mask类型:")
    print("1. center - 中心方形")
    print("2. left_half - 左半边") 
    print("3. top_half - 上半边")
    print("4. random - 随机遮罩")
    print("5. stripes - 条纹遮罩")
    
    # 这里可以添加用户输入，但为了自动演示，我们使用预设值
    mask_type = "center"
    mask_size = 0.5
    use_improved = True
    
    print(f"\n使用设置: mask_type={mask_type}, mask_size={mask_size}, improved={use_improved}")
    
    # 创建并处理图像
    device = model.device
    x_0 = torch.randn(1, 3, 32, 32).to(device)
    mask = model.create_mask(x_0.shape, mask_type, mask_size)
    x_masked = model.apply_mask(x_0, mask)
    
    if use_improved:
        result = model.inpaint_improved(x_0, mask, num_samples=1)
    else:
        result = model.inpaint(x_0, mask, num_samples=1)
    
    save_path = "interactive_demo_result.png"
    model.save_inpaint_results(x_0, x_masked, result, mask, save_path, category="interactive")
    print(f"✅ 交互式演示完成！结果保存到: interactive/{save_path}")

def main():
    """主演示函数"""
    print("🎨 DDPM图像修复(Inpainting)完整演示")
    print("=" * 60)
    
    demos = [
        ("基本演示", demo_basic_inpainting),
        ("自定义Mask", demo_custom_mask),
        ("多样本生成", demo_multiple_samples),
        ("算法对比", demo_algorithm_comparison),
        ("真实图像", demo_with_real_image),
        ("交互式演示", interactive_demo)
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"❌ {name}演示失败: {e}")
            continue
    
    print("\n" + "="*60)
    print("🎉 所有演示完成！")
    print("📁 检查 inpainting_outputs/ 目录下的分类文件夹来查看结果:")
    print("   ├── demo/ - 基本演示结果")
    print("   ├── custom_masks/ - 自定义遮罩结果")  
    print("   ├── multiple_samples/ - 多样本生成")
    print("   ├── algorithm_comparison/ - 算法对比")
    print("   ├── real_images/ - 真实图像修复")
    print("   └── interactive/ - 交互式演示")
    print("📚 查看代码了解如何自定义使用inpainting功能")

if __name__ == "__main__":
    main() 