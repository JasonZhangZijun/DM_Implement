import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.improved_unet import ImprovedUNet
from models.ddpm import DDPM_Model

class SuperResolutionUNet(ImprovedUNet):
    """专门用于超分辨率的UNet"""
    def __init__(self, low_res_channels=3, high_res_channels=3, scale_factor=4):
        # 输入通道是低分辨率图像+噪声图像
        super().__init__(
            in_channels=low_res_channels + high_res_channels,  # 6通道输入
            out_channels=high_res_channels,  # 3通道输出
            time_embed_dim=256,
            channels=[64, 128, 256, 512],
            attention_resolutions=[16, 8],
            num_res_blocks=2,
            dropout=0.1
        )
        self.scale_factor = scale_factor
        
        # 低分辨率图像的上采样层
        self.lowres_upsampler = nn.Sequential(
            nn.ConvTranspose2d(low_res_channels, 64, 4, stride=scale_factor, padding=scale_factor//2),
            nn.ReLU(),
            nn.Conv2d(64, low_res_channels, 3, padding=1)
        )
    
    def forward(self, x_noisy, x_lowres, timesteps):
        """
        Args:
            x_noisy: [B, C, H, W] 噪声图像（高分辨率）
            x_lowres: [B, C, H//scale, W//scale] 低分辨率输入图像
            timesteps: [B] 时间步
        """
        # 将低分辨率图像上采样到高分辨率
        x_lowres_up = self.lowres_upsampler(x_lowres)
        
        # 连接噪声图像和上采样的低分辨率图像
        x_combined = torch.cat([x_noisy, x_lowres_up], dim=1)
        
        # 通过UNet预测噪声
        return super().forward(x_combined, timesteps)

class SuperResolutionDDPM(DDPM_Model):
    """超分辨率DDPM模型"""
    def __init__(self, dataloader, scale_factor=4, T=1000, beta_start=0.0001, beta_end=0.02, device=None):
        # 初始化父类但不使用父类的UNet
        super().__init__(dataloader, T, beta_start, beta_end, device)
        
        # 替换为超分辨率UNet
        self.unet = SuperResolutionUNet(scale_factor=scale_factor).to(self.device)
        self.scale_factor = scale_factor
        
    def create_lowres_data(self, high_res_images):
        """创建低分辨率训练数据"""
        # 使用双线性插值下采样
        low_res = F.interpolate(
            high_res_images, 
            scale_factor=1/self.scale_factor, 
            mode='bilinear', 
            align_corners=False
        )
        return low_res
    
    def calculate_loss(self, x_0, t):
        """计算超分辨率损失"""
        batch_size = x_0.shape[0]
        
        # 创建对应的低分辨率图像
        x_lowres = self.create_lowres_data(x_0)
        
        # 添加噪声到高分辨率图像
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        # 预测噪声
        predicted_noise = self.unet(x_t, x_lowres, t)
        
        # 计算MSE损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def super_resolve(self, low_res_images, num_inference_steps=None):
        """
        超分辨率推理
        Args:
            low_res_images: [B, C, H, W] 低分辨率输入图像
            num_inference_steps: 推理步数，None表示使用完整的T步
        Returns:
            [B, C, H*scale, W*scale] 超分辨率图像
        """
        self.unet.eval()
        
        if num_inference_steps is None:
            num_inference_steps = self.T
            
        batch_size = low_res_images.shape[0]
        high_res_shape = (
            batch_size, 
            3, 
            low_res_images.shape[2] * self.scale_factor,
            low_res_images.shape[3] * self.scale_factor
        )
        
        # 从随机噪声开始
        x = torch.randn(high_res_shape, device=self.device)
        
        # 计算时间步
        timesteps = torch.linspace(self.T-1, 0, num_inference_steps, dtype=torch.long, device=self.device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            
            # 预测噪声
            with torch.no_grad():
                predicted_noise = self.unet(x, low_res_images, t_batch)
            
            # DDPM采样步骤
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1 - alpha_bar_t)
            
            x = coef1 * (x - coef2 * predicted_noise)
            
            # 添加噪声（除了最后一步）
            if i < len(timesteps) - 1:
                noise = torch.randn_like(x)
                x += torch.sqrt(beta_t) * noise
        
        return x
    
    @torch.no_grad()
    def ddim_super_resolve(self, low_res_images, num_inference_steps=50, eta=0.0):
        """
        使用DDIM进行快速超分辨率推理
        """
        self.unet.eval()
        
        batch_size = low_res_images.shape[0]
        high_res_shape = (
            batch_size, 
            3, 
            low_res_images.shape[2] * self.scale_factor,
            low_res_images.shape[3] * self.scale_factor
        )
        
        # 从随机噪声开始
        x = torch.randn(high_res_shape, device=self.device)
        
        # 计算时间步
        timesteps = torch.linspace(self.T-1, 0, num_inference_steps, dtype=torch.long, device=self.device)
        
        for i in range(len(timesteps)):
            t = timesteps[i]
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            
            # 预测噪声
            predicted_noise = self.unet(x, low_res_images, t_batch)
            
            # DDIM更新
            alpha_bar_t = self.alpha_bars[t]
            
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_bar_t_next = self.alpha_bars[t_next]
            else:
                alpha_bar_t_next = torch.tensor(1.0, device=self.device)
            
            # 预测x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            
            # DDIM更新规则
            sigma_t = eta * torch.sqrt((1 - alpha_bar_t_next) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_next)
            
            direction_pointing_to_xt = torch.sqrt(1 - alpha_bar_t_next - sigma_t**2) * predicted_noise
            
            noise = torch.randn_like(x) if i < len(timesteps) - 1 else torch.zeros_like(x)
            
            x = torch.sqrt(alpha_bar_t_next) * pred_x0 + direction_pointing_to_xt + sigma_t * noise
        
        return x
    
    def evaluate_super_resolution(self, test_loader, num_samples=10):
        """评估超分辨率质量"""
        from utils.losses import calculate_psnr, calculate_ssim
        
        total_psnr = 0
        total_ssim = 0
        count = 0
        
        self.unet.eval()
        with torch.no_grad():
            for i, (high_res_batch, _) in enumerate(test_loader):
                if i >= num_samples:
                    break
                    
                high_res_batch = high_res_batch.to(self.device)
                
                # 创建低分辨率版本
                low_res_batch = self.create_lowres_data(high_res_batch)
                
                # 超分辨率重建
                sr_batch = self.ddim_super_resolve(low_res_batch, num_inference_steps=50)
                
                # 计算指标
                for j in range(high_res_batch.shape[0]):
                    hr_img = high_res_batch[j].cpu().numpy()
                    sr_img = sr_batch[j].cpu().numpy()
                    
                    psnr = calculate_psnr(hr_img, sr_img)
                    ssim = calculate_ssim(hr_img.flatten(), sr_img.flatten())
                    
                    total_psnr += psnr
                    total_ssim += ssim
                    count += 1
        
        avg_psnr = total_psnr / count if count > 0 else 0
        avg_ssim = total_ssim / count if count > 0 else 0
        
        return {"PSNR": avg_psnr, "SSIM": avg_ssim}
    
    def save_sr_comparison(self, low_res_images, save_path="sr_comparison.png"):
        """保存超分辨率对比图"""
        import torchvision.utils as vutils
        import os
        
        with torch.no_grad():
            # 超分辨率重建
            sr_images = self.ddim_super_resolve(low_res_images, num_inference_steps=50)
            
            # 双线性上采样作为对比
            bicubic_images = F.interpolate(
                low_res_images, 
                scale_factor=self.scale_factor, 
                mode='bicubic', 
                align_corners=False
            )
            
            # 拼接图像：低分辨率 | 双线性 | 超分辨率
            low_res_repeated = F.interpolate(low_res_images, scale_factor=self.scale_factor, mode='nearest')
            
            comparison = torch.cat([low_res_repeated, bicubic_images, sr_images], dim=0)
            
            # 归一化到[0,1]
            comparison = (comparison + 1) / 2
            comparison = torch.clamp(comparison, 0, 1)
            
            # 保存
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            vutils.save_image(comparison, save_path, nrow=low_res_images.shape[0], normalize=False)
            
            print(f"超分辨率对比图已保存到: {save_path}")

# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    class DummyDataLoader:
        def __init__(self):
            self.batch_size = 4
        def __iter__(self):
            for _ in range(2):
                # 生成32x32的高分辨率图像
                data = torch.randn(self.batch_size, 3, 32, 32)
                labels = torch.randint(0, 10, (self.batch_size,))
                yield data, labels
    
    # 初始化模型
    dataloader = DummyDataLoader()
    sr_model = SuperResolutionDDPM(dataloader, scale_factor=2, T=100)
    
    print(f"模型参数数量: {sum(p.numel() for p in sr_model.parameters()):,}")
    
    # 测试超分辨率
    low_res_test = torch.randn(2, 3, 16, 16).to(sr_model.device)
    print(f"输入低分辨率图像形状: {low_res_test.shape}")
    
    sr_result = sr_model.ddim_super_resolve(low_res_test, num_inference_steps=10)
    print(f"输出超分辨率图像形状: {sr_result.shape}")
    
    print("超分辨率模型测试成功！") 