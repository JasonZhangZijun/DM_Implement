#!/usr/bin/env python3
"""
简单的图像修复演示
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import argparse
from inpaint_ddpm import InpaintDDPM

# 创建一个简单的虚拟数据加载器
class DummyDataLoader:
    def __init__(self):
        self.batch_size = 4
    def __iter__(self):
        for i in range(2):
            fake_data = torch.randn(self.batch_size, 3, 32, 32)
            fake_labels = torch.randint(0, 10, (self.batch_size,))
            yield fake_data, fake_labels

def main():
    parser = argparse.ArgumentParser(description='DDPM Inpainting Demo')
    parser.add_argument('--model_path', type=str, default='ddpm_model.pth', 
                        help='预训练模型路径')
    parser.add_argument('--image_path', type=str, default=None,
                        help='输入图像路径（可选）')
    parser.add_argument('--use_improved', action='store_true', default=True,
                        help='使用改进的inpainting算法')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备选择 (cpu/cuda/auto)')
    parser.add_argument('--output_dir', type=str, default='main_outputs',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    print(f"输出目录: {args.output_dir}")
    
    # 创建虚拟数据加载器
    dataloader = DummyDataLoader()
    
    # 初始化模型
    print("正在初始化DDPM inpainting模型...")
    ddpm = InpaintDDPM(dataloader, T=50, device=device, output_dir=args.output_dir)  # 使用较小的T进行快速测试
    
    # 尝试加载预训练模型
    if ddpm.load_pretrained_model(args.model_path):
        print(f"✅ 成功加载预训练模型: {args.model_path}")
    else:
        print("⚠️ 未找到预训练模型，将使用随机初始化的权重")
    
    # 运行演示
    print("\n开始运行inpainting演示...")
    ddpm.demo_inpainting(image_path=args.image_path, use_improved=args.use_improved)
    
    print(f"\n🎉 演示完成！检查 {args.output_dir}/demo/ 目录下的生成图像文件。")

if __name__ == "__main__":
    main() 