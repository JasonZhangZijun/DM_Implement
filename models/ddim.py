import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.unet import UNet

class DDIM_Model(nn.Module):
    """
    DDIM (Denoising Diffusion Implicit Models) 类
    forward: q(x_t|x_0) - 与DDPM相同的前向过程
    backward: 确定性采样过程，可以使用更少的步数
    """
    def __init__(self, dataloader, T=1000, beta_start=0.0001, beta_end=0.02, device=None):
        super(DDIM_Model, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.T = T
        self.dataloader = dataloader
        
        self.unet = UNet().to(self.device)
        
        # 噪声调度参数
        self.betas = torch.linspace(beta_start, beta_end, T).to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def q_sample(self, x_0, t, noise):
        """前向过程：给干净图像添加噪声"""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    def calculate_loss(self, x_0, t):
        """计算训练损失"""
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.unet(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    def train_model(self, num_epochs=10, lr=1e-4, save_path="ddim_model.pth"):
        """训练模型"""
        self.unet.train()
        optimizer = optim.Adam(self.unet.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            for batch_idx, (x_0, _) in enumerate(self.dataloader):
                x_0 = x_0.to(self.device)
                t = torch.randint(0, self.T, (x_0.size(0),), device=self.device)
                loss = self.calculate_loss(x_0, t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}")

        torch.save(self.state_dict(), save_path)
    
    def ddim_step(self, x_t, t, t_prev, eta=0.0):
        """DDIM采样步骤"""
        # 预测噪声
        predicted_noise = self.unet(x_t, t)
        
        # 获取alpha值，确保维度正确
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        alpha_bar_t_prev = self.alpha_bars[t_prev].view(-1, 1, 1, 1) if t_prev >= 0 else torch.tensor(1.0).to(self.device).view(1, 1, 1, 1)
        
        # 预测x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        
        # 计算方向向量
        direction = torch.sqrt(1 - alpha_bar_t_prev - eta**2 * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)) * predicted_noise
        
        # 随机性项（eta=0时为确定性采样）
        noise = torch.randn_like(x_t) if eta > 0 else 0
        random_noise = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev) * noise
        
        # DDIM更新公式
        x_t_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + direction + random_noise
        
        return x_t_prev
    
    @torch.no_grad()
    def p_sample_loop(self, shape, ddim_steps=50, eta=0.0):
        """DDIM采样循环，可以使用更少的步数"""
        self.unet.eval()
        x = torch.randn(shape).to(self.device)
        
        # 创建采样时间步序列
        step_size = self.T // ddim_steps
        timesteps = list(range(0, self.T, step_size))[:ddim_steps]
        timesteps = timesteps[::-1]  # 反向
        
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=self.device)
            x = self.ddim_step(x, t_tensor, t_prev, eta)
        
        return x
    
    @torch.no_grad()
    def sample(self, num_samples, ddim_steps=50, eta=0.0):
        """生成样本的便捷方法"""
        shape = (num_samples, 3, 32, 32)  # CIFAR-10图像尺寸
        return self.p_sample_loop(shape, ddim_steps, eta)
    
    @torch.no_grad()
    def fast_sample(self, num_samples, ddim_steps=10, eta=0.0):
        """快速采样，使用更少的步数"""
        return self.sample(num_samples, ddim_steps, eta)
    
    def evaluate(self, num_samples=1000):
        """评估模型（简单实现，返回模拟的FID分数）"""
        # 这里应该实现真正的FID计算，现在返回一个模拟值
        import random
        return random.uniform(10.0, 50.0) 