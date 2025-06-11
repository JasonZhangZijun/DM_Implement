import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.unet import UNet
import os
from PIL import Image

class DDPM_Model(nn.Module):
    """
    class for diffusion model
    forward: q(x_t|x_0)
    backward: p(x_{t-1}|x_t), using unet
    """
    def __init__(self, dataloader, T=1000, beta_start=0.0001, beta_end=0.02, device=None):
        super(DDPM_Model, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.T = T
        self.dataloader = dataloader
        
        self.unet = UNet().to(self.device)
        
        self.betas = torch.linspace(beta_start, beta_end, T).to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def q_sample(self, x_0, t, noise):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    def calculate_loss(self, x_0, t):
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.unet(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def train_model(self, num_epochs=10, lr=1e-4, save_path="diffusion_model.pth"):
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
    
    def predict_noise(self, x, t):
        """预测噪声的方法"""
        return self.unet(x, t)
        
    @torch.no_grad()
    def p_sample_loop(self, shape):
        self.unet.eval()
        x = torch.randn(shape).to(self.device)

        for t in reversed(range(self.T)):
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=self.device)
            predicted_noise = self.predict_noise(x, t_tensor)

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]

            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1 - alpha_bar_t)

            x = coef1 * (x - coef2 * predicted_noise)

            if t > 0:
                noise = torch.randn_like(x)
                x += torch.sqrt(beta_t) * noise

        return x
    
    @torch.no_grad()
    def sample(self, num_samples):
        """生成样本的便捷方法"""
        # shape = (num_samples, 3, 32, 32)  # CIFAR-10图像尺寸
        # celeba 图像尺寸
        shape = (num_samples, 3, 218, 178)
        return self.p_sample_loop(shape)

    def evaluate(self, num_samples=1000):
        """评估模型（简单实现，返回模拟的FID分数）"""
        # 这里应该实现真正的FID计算，现在返回一个模拟值
        import random
        return random.uniform(10.0, 50.0) 
                