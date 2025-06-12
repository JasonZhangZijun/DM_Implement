#!/usr/bin/env python3
"""图像修复DDPM模型实现"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ddpm import DDPM_Model
from PIL import Image
import numpy as np
from utils.masks import create_mask as _create_mask_util, apply_mask as _apply_mask_util
from utils.losses import psnr as _psnr_metric, ssim as _ssim_metric

class InpaintDDPM(DDPM_Model):
    """
    基于DDPM的图像修复模型
    继承DDPM_Model，添加inpainting特有的功能
    """
    def __init__(self, dataloader, T=1000, beta_start=0.0001, beta_end=0.02, device=None, output_dir="inpainting_outputs"):
        # 继承父类的初始化
        super(InpaintDDPM, self).__init__(dataloader, T, beta_start, beta_end, device)
        
        # 设置输出目录
        self.output_dir = output_dir
        self.setup_output_dirs()
    
    def setup_output_dirs(self):
        """
        设置输出目录结构
        """
        # 主输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 子目录结构
        subdirs = [
            "demo",
            "custom_masks", 
            "multiple_samples",
            "algorithm_comparison",
            "improved_method",
            "random_images"
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        
        print(f"输出目录已设置：{self.output_dir}")
    
    def create_mask(self, image_shape, mask_type="center", mask_size=0.5):
        """
        创建不同类型的mask（委托给 utils.masks.create_mask）
        """
        return _create_mask_util(image_shape, mask_type, mask_size, self.device)
    
    def apply_mask(self, image, mask):
        """
        应用mask到图像（委托给 utils.masks.apply_mask）
        """
        return _apply_mask_util(image, mask)
    
    @torch.no_grad()
    def inpaint_sample_step(self, x_t, t, x_0_known, mask):
        """
        Inpainting的单步采样
        在每一步中将已知区域约束到原始图像
        """
        # 标准DDPM采样步骤
        t_tensor = torch.full((x_t.size(0),), t, dtype=torch.long, device=self.device)
        predicted_noise = self.predict_noise(x_t, t_tensor)
        
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        
        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_bar_t)
        
        # 预测的前一步
        x_t_prev = coef1 * (x_t - coef2 * predicted_noise)
        
        if t > 0:
            noise = torch.randn_like(x_t)
            x_t_prev += torch.sqrt(beta_t) * noise
            
            # 关键修复：将已知区域约束到原始图像的正确噪声版本
            # 计算t-1时间步的参数
            t_prev = t - 1
            t_prev_tensor = torch.full((x_t.size(0),), t_prev, dtype=torch.long, device=self.device)
            
            # 对原始图像添加t-1时间步的噪声
            noise_known = torch.randn_like(x_0_known)
            x_0_known_noisy = self.q_sample(x_0_known, t_prev_tensor, noise_known)
            
            # 在已知区域使用原始图像的噪声版本，在未知区域使用预测结果
            x_t_prev = mask * x_0_known_noisy + (1 - mask) * x_t_prev
        else:
            # t=0时直接使用原始图像的已知部分
            x_t_prev = mask * x_0_known + (1 - mask) * x_t_prev
        
        return x_t_prev
    
    @torch.no_grad()
    def inpaint(self, x_0_known, mask, num_samples=1):
        """
        执行图像修复
        Args:
            x_0_known: 已知部分的原始图像 (batch_size, 3, 32, 32)
            mask: mask张量 (1=已知, 0=未知)
            num_samples: 生成样本数量
        Returns:
            修复后的图像
        """
        self.unet.eval()
        
        batch_size = x_0_known.size(0)
        if num_samples > 1:
            # 如果要生成多个样本，复制输入
            x_0_known = x_0_known.repeat(num_samples, 1, 1, 1)
            mask = mask.repeat(num_samples, 1, 1, 1)
        
        # 从随机噪声开始
        x = torch.randn_like(x_0_known).to(self.device)
        
        # 反向采样过程
        for t in reversed(range(self.T)):
            x = self.inpaint_sample_step(x, t, x_0_known, mask)
        
        return x
    
    @torch.no_grad()
    def inpaint_improved(self, x_0_known, mask, num_samples=1):
        """
        改进的图像修复方法
        使用更稳定的约束策略
        """
        self.unet.eval()
        
        batch_size = x_0_known.size(0)
        if num_samples > 1:
            # 如果要生成多个样本，复制输入
            x_0_known = x_0_known.repeat(num_samples, 1, 1, 1)
            mask = mask.repeat(num_samples, 1, 1, 1)
        
        # 从随机噪声开始
        x = torch.randn_like(x_0_known).to(self.device)
        
        # 反向采样过程
        for t in reversed(range(self.T)):
            t_tensor = torch.full((x.size(0),), t, dtype=torch.long, device=self.device)
            
            # 标准DDPM采样步骤
            predicted_noise = self.predict_noise(x, t_tensor)
            
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1 - alpha_bar_t)
            
            # 预测的前一步
            x_prev = coef1 * (x - coef2 * predicted_noise)
            
            if t > 0:
                noise = torch.randn_like(x)
                x_prev += torch.sqrt(beta_t) * noise
            
            # 改进的约束策略：在每一步直接约束到带噪声的原始图像
            if t > 0:
                # 将原始图像添加当前时间步的噪声
                noise_orig = torch.randn_like(x_0_known)
                t_prev_tensor = torch.full((x.size(0),), max(0, t-1), dtype=torch.long, device=self.device)
                x_0_noisy = self.q_sample(x_0_known, t_prev_tensor, noise_orig)
                
                # 约束已知区域
                x = mask * x_0_noisy + (1 - mask) * x_prev
            else:
                # 最后一步使用原始图像
                x = mask * x_0_known + (1 - mask) * x_prev
        
        return x
    
    def load_and_preprocess_image(self, image_path, target_size=(32, 32)):
        """
        加载并预处理图像
        """
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # 调整大小
        image = image.resize(target_size)
        
        # 转换为张量并归一化到[-1, 1]
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # (3, H, W)
        image_tensor = (image_tensor - 0.5) * 2  # 归一化到[-1, 1]
        
        return image_tensor.unsqueeze(0)  # 添加batch维度
    
    def save_inpaint_results(self, original, masked, result, mask, save_path, category=None):
        """
        保存inpainting结果的对比图
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # 转换到[0, 1]范围
        def denormalize(tensor):
            return torch.clamp((tensor + 1) / 2, 0, 1)
        
        original = denormalize(original[0])
        masked = denormalize(masked[0])
        result = denormalize(result[0])
        
        # 创建对比图：原图 | 遮罩图 | 修复图 | mask可视化
        mask_vis = mask[0, 0].unsqueeze(0).repeat(3, 1, 1)
        
        # 水平拼接
        comparison = torch.cat([original, masked, result, mask_vis], dim=2)
        
        # 转换为PIL图像并保存
        comparison_np = (comparison.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        comparison_img = Image.fromarray(comparison_np)
        comparison_img.save(save_path)
        
        print(f"Inpainting结果已保存到: {save_path}")
    
    def inpaint_from_image(self, image_path, mask_type="center", mask_size=0.5, 
                          num_samples=1, save_path="inpaint_result.png", use_improved=True):
        """
        从图像文件执行inpainting的完整流程
        """
        print(f"开始对图像 {image_path} 执行inpainting...")
        
        # 加载图像
        x_0 = self.load_and_preprocess_image(image_path).to(self.device)
        
        # 创建mask
        mask = self.create_mask(x_0.shape, mask_type, mask_size)
        
        # 应用mask
        x_masked = self.apply_mask(x_0, mask)
        
        print(f"使用mask类型: {mask_type}, mask大小: {mask_size}")
        print(f"使用{'改进' if use_improved else '标准'}算法...")
        print("开始inpainting过程...")
        
        # 执行inpainting
        if use_improved:
            result = self.inpaint_improved(x_0, mask, num_samples)
        else:
            result = self.inpaint(x_0, mask, num_samples)
        
        # 保存结果
        if num_samples == 1:
            self.save_inpaint_results(x_0, x_masked, result, mask, save_path)
        else:
            # 保存多个结果
            for i in range(num_samples):
                save_path_i = save_path.replace('.png', f'_sample_{i}.png')
                self.save_inpaint_results(x_0, x_masked, result[i:i+1], mask, save_path_i)
        
        return result
    
    def load_pretrained_model(self, model_path):
        """
        加载预训练的DDPM模型
        """
        try:
            print(f"加载预训练模型: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.load_state_dict(state_dict)
            print("✅ 预训练模型加载成功！")
            return True
        except Exception as e:
            print(f"❌ 加载预训练模型失败: {e}")
            print("将使用随机初始化的模型（效果可能不佳）")
            return False
    
    def demo_inpainting(self, image_path=None, use_improved=True):
        """
        演示不同类型的inpainting效果
        """
        if image_path is None:
            print("没有提供图像路径，生成随机图像进行演示...")
            # 生成一个随机图像用于演示
            x_0 = torch.randn(1, 3, 32, 32).to(self.device)
        else:
            x_0 = self.load_and_preprocess_image(image_path).to(self.device)
        
        mask_types = ["center", "left_half", "top_half", "random", "stripes"]
        
        print(f"演示不同类型的inpainting效果（使用{'改进' if use_improved else '标准'}算法）...")
        
        for mask_type in mask_types:
            print(f"\n正在处理 {mask_type} mask...")
            mask = self.create_mask(x_0.shape, mask_type, 0.5)
            x_masked = self.apply_mask(x_0, mask)
            
            if use_improved:
                result = self.inpaint_improved(x_0, mask, num_samples=1)
            else:
                result = self.inpaint(x_0, mask, num_samples=1)
            
            save_path = os.path.join(self.output_dir, "demo", f"inpaint_demo_{mask_type}_{'improved' if use_improved else 'standard'}.png")
            self.save_inpaint_results(x_0, x_masked, result, mask, save_path)
        
        print("\n✅ Inpainting演示完成！")

    def evaluate_quality(self, original, reconstructed):
        """计算单组图像的 PSNR 与 SSIM 指标（原图与修复图应已对齐且 shape 相同）。"""
        if original.dim() == 3:
            original = original.unsqueeze(0)
        if reconstructed.dim() == 3:
            reconstructed = reconstructed.unsqueeze(0)
        with torch.no_grad():
            psnr_val = _psnr_metric(original, reconstructed).item()
            ssim_val = _ssim_metric(original, reconstructed).item()
        return {
            "PSNR": psnr_val,
            "SSIM": ssim_val,
        }


# 测试和使用示例
if __name__ == "__main__":
    # 创建简单的数据加载器用于测试
    class DummyDataLoader:
        def __init__(self):
            self.batch_size = 4
        def __iter__(self):
            for i in range(2):
                fake_data = torch.randn(self.batch_size, 3, 32, 32)
                fake_labels = torch.randint(0, 10, (self.batch_size,))
                yield fake_data, fake_labels
    
    print("初始化Inpainting DDPM模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DummyDataLoader()
    
    # 创建inpainting模型
    inpaint_model = InpaintDDPM(dataloader, T=50, device=device)  # 使用较小的T进行测试
    
    # 演示inpainting功能
    print("\n开始inpainting演示...")
    inpaint_model.demo_inpainting()
    
    print("\n现在你可以使用以下方法进行inpainting:")
    print("1. inpaint_model.inpaint_from_image('path/to/image.jpg')")
    print("2. inpaint_model.demo_inpainting('path/to/image.jpg')")
    print("3. 手动创建mask: mask = inpaint_model.create_mask(shape, 'center')") 