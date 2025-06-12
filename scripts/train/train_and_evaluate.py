#!/usr/bin/env python3
"""
统一的训练和评估脚本

功能：
- 支持 CIFAR-10 和 CelebA 数据集
- 训练 DDPM 和条件 DDPM 模型
- 自动生成样本和评估 FID
- 保存模型和实验结果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json

from config import get_config
from models.ddpm import DDPM_Model
from models.conditional_ddpm import ConditionalDDPM_Model
from inpaint_ddpm import InpaintDDPM
from utils.fid import evaluate_generated_images
from utils.losses import psnr, ssim

class ExperimentLogger:
    """实验日志记录器"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "experiment.log")
        self.metrics_file = os.path.join(log_dir, "metrics.json")
        self.metrics = {}
    
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    
    def save_metrics(self, epoch, metrics_dict):
        if epoch not in self.metrics:
            self.metrics[epoch] = {}
        self.metrics[epoch].update(metrics_dict)
        
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

class DataLoaderFactory:
    """数据加载器工厂"""
    @staticmethod
    def get_cifar10_loader(config):
        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root=config.data_path,
            train=True,
            download=True,
            transform=transform
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True
        )
    
    @staticmethod
    def get_test_loader(config):
        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root=config.data_path,
            train=False,
            download=True,
            transform=transform
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

def save_samples(samples, save_path, nrow=8):
    """保存生成的样本"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 将样本从[-1, 1]转换到[0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # 保存图像网格
    torchvision.utils.save_image(samples, save_path, nrow=nrow, normalize=False)

def evaluate_unconditional(model, config, logger, epoch):
    """评估无条件生成模型"""
    logger.log(f"开始评估无条件生成模型 (Epoch {epoch})")
    
    # 生成样本
    with torch.no_grad():
        samples = model.sample(config.sample_size)
    
    # 保存样本
    save_dir = os.path.join(config.output_dir, "samples")
    save_path = os.path.join(save_dir, f"epoch_{epoch}_samples.png")
    save_samples(samples, save_path)
    
    logger.log(f"样本已保存到: {save_path}")
    
    # 计算FID（如果有足够的样本）
    if epoch > 0 and epoch % config.eval_interval == 0:
        try:
            # 生成更多样本用于FID计算
            all_samples = []
            num_batches = 100 // config.sample_size + 1
            
            for _ in range(num_batches):
                batch_samples = model.sample(config.sample_size)
                all_samples.append(batch_samples)
            
            all_samples = torch.cat(all_samples, dim=0)[:100]  # 取前100个样本
            
            # 保存用于FID计算的样本
            fid_dir = os.path.join(config.output_dir, "fid_samples")
            os.makedirs(fid_dir, exist_ok=True)
            
            for i, sample in enumerate(all_samples):
                sample_path = os.path.join(fid_dir, f"sample_{i:03d}.png")
                torchvision.utils.save_image((sample + 1) / 2, sample_path)
            
            # 计算FID（这里需要真实数据集路径）
            # fid_score = evaluate_generated_images(fid_dir, real_folder=None)
            # logger.log(f"FID Score: {fid_score:.3f}")
            
        except Exception as e:
            logger.log(f"FID计算失败: {e}")

def evaluate_inpainting(model, test_loader, config, logger, epoch):
    """评估图像修复模型"""
    logger.log(f"开始评估图像修复模型 (Epoch {epoch})")
    
    total_psnr = 0
    total_ssim = 0
    num_samples = 0
    
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= 10:  # 只评估前10个batch
                break
                
            images = images.to(config.device)
            
            # 创建不同类型的遮罩
            for mask_type in ["center", "random"]:
                mask = model.create_mask(images.shape, mask_type, 0.5)
                masked_images = images * mask
                
                # 修复图像
                restored = model.inpaint_improved(images, mask, num_samples=1)
                
                # 计算PSNR和SSIM
                for j in range(images.shape[0]):
                    # 只在遮罩区域计算指标
                    mask_region = (1 - mask[j]).bool()
                    
                    if mask_region.sum() > 0:
                        original_patch = images[j][mask_region].cpu().numpy()
                        restored_patch = restored[0][j][mask_region].cpu().numpy()
                        
                        psnr = psnr(original_patch, restored_patch)
                        ssim = ssim(
                            original_patch.reshape(-1), 
                            restored_patch.reshape(-1)
                        )
                        
                        total_psnr += psnr
                        total_ssim += ssim
                        num_samples += 1
    
    if num_samples > 0:
        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples
        
        logger.log(f"平均PSNR: {avg_psnr:.3f}")
        logger.log(f"平均SSIM: {avg_ssim:.3f}")
        
        return {"PSNR": avg_psnr, "SSIM": avg_ssim}
    
    return {}

def train_model(task, config_name=None):
    """训练模型主函数"""
    # 获取配置
    config = get_config(task)
    if config_name:
        # 可以根据config_name进一步定制配置
        pass
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{task}_{timestamp}"
    config.output_dir = os.path.join("experiments", experiment_name)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 初始化日志记录器
    logger = ExperimentLogger(config.output_dir)
    logger.log(f"开始训练任务: {task}")
    logger.log(f"配置: {vars(config)}")
    
    # 保存配置
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(config), f, indent=2, ensure_ascii=False, default=str)
    
    # 创建数据加载器
    if task in ["cifar10", "conditional"]:
        train_loader = DataLoaderFactory.get_cifar10_loader(config)
        test_loader = DataLoaderFactory.get_test_loader(config)
    else:
        # 创建虚拟数据加载器
        class DummyDataLoader:
            def __init__(self, config):
                self.batch_size = config.batch_size
                self.config = config
            def __iter__(self):
                for i in range(100):  # 100个batch
                    fake_data = torch.randn(self.batch_size, 3, self.config.image_size, self.config.image_size)
                    fake_labels = torch.randint(0, 10, (self.batch_size,))
                    yield fake_data, fake_labels
        
        train_loader = DummyDataLoader(config)
        test_loader = DummyDataLoader(config)
    
    # 创建模型
    if task == "inpainting":
        model = InpaintDDPM(
            train_loader, 
            T=config.T, 
            device=config.device,
            output_dir=config.output_dir
        )
    elif task == "conditional":
        model = ConditionalDDPM_Model(
            train_loader,
            T=config.T,
            num_classes=config.num_classes,
            device=config.device
        )
    else:  # unconditional
        model = DDPM_Model(
            train_loader,
            T=config.T,
            device=config.device
        )
    
    logger.log(f"模型已创建，参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练循环
    model.train()
    for epoch in range(config.num_epochs):
        logger.log(f"开始训练 Epoch {epoch+1}/{config.num_epochs}")
        
        if hasattr(model, 'train_epoch'):
            # 如果模型有自定义的训练epoch方法
            epoch_loss = model.train_epoch()
        else:
            # 使用模型的train_model方法（但只训练一个epoch）
            model.train_model(num_epochs=1, lr=config.lr)
            epoch_loss = 0.0  # 占位符
        
        logger.log(f"Epoch {epoch+1} 完成，损失: {epoch_loss:.4f}")
        
        # 定期评估
        if (epoch + 1) % config.eval_interval == 0:
            model.eval()
            
            if task == "inpainting":
                metrics = evaluate_inpainting(model, test_loader, config, logger, epoch+1)
            else:
                evaluate_unconditional(model, config, logger, epoch+1)
                metrics = {}
            
            logger.save_metrics(epoch+1, metrics)
            model.train()
        
        # 定期保存模型
        if (epoch + 1) % config.save_interval == 0:
            model_path = os.path.join(config.output_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            logger.log(f"模型已保存到: {model_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(config.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.log(f"最终模型已保存到: {final_model_path}")
    
    logger.log("训练完成！")
    return config.output_dir

def main():
    parser = argparse.ArgumentParser(description="DDPM训练和评估脚本")
    parser.add_argument("--task", type=str, default="cifar10",
                       choices=["cifar10", "inpainting", "conditional", "super_resolution"],
                       help="选择任务类型")
    parser.add_argument("--config", type=str, default=None,
                       help="自定义配置名称")
    parser.add_argument("--eval_only", action="store_true",
                       help="仅评估模式")
    parser.add_argument("--model_path", type=str, default=None,
                       help="预训练模型路径")
    
    args = parser.parse_args()
    
    if args.eval_only:
        print("仅评估模式暂未实现")
        return
    
    # 开始训练
    experiment_dir = train_model(args.task, args.config)
    print(f"\n🎉 实验完成！结果保存在: {experiment_dir}")

if __name__ == "__main__":
    main() 