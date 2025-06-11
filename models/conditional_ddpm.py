import torch
import torch.nn.functional as F
from models.ddpm import DDPM_Model
from models.conditional_unet import ConditionalUNet

class ConditionalDDPM_Model(DDPM_Model):
    """类别条件 DDPM，实现与 DDPM_Model 基本一致，但在 UNet 前向中加入标签。"""

    def __init__(self, dataloader, num_classes: int = 10, **kwargs):
        super().__init__(dataloader, **kwargs)
        # 用条件 UNet 替换
        self.unet = ConditionalUNet(num_classes=num_classes).to(self.device)
        self.num_classes = num_classes

    # ----------- 损失计算 ----------- #
    def calculate_loss(self, x_0: torch.Tensor, t: torch.Tensor, labels: torch.Tensor):
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.unet(x_t, t, labels)
        return F.mse_loss(predicted_noise, noise)

    # ----------- 训练 ----------- #
    def train_model(self, num_epochs=10, lr=1e-4, save_path="conditional_diffusion.pth"):
        self.unet.train()
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=lr)
        for epoch in range(num_epochs):
            for batch_idx, (x_0, y) in enumerate(self.dataloader):
                x_0 = x_0.to(self.device)
                y = y.to(self.device)
                t = torch.randint(0, self.T, (x_0.size(0),), device=self.device)
                loss = self.calculate_loss(x_0, t, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}")
        torch.save(self.state_dict(), save_path)

    # ----------- 预测噪声 ----------- #
    def predict_noise(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor):
        return self.unet(x, t, labels)

    # ----------- 采样 ----------- #
    @torch.no_grad()
    def p_sample_loop(self, shape, labels: torch.Tensor):
        """labels: (B,) 与 shape[0] 对齐"""
        self.unet.eval()
        x = torch.randn(shape).to(self.device)

        for t in reversed(range(self.T)):
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=self.device)
            predicted_noise = self.predict_noise(x, t_tensor, labels)

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
    def sample(self, labels: torch.Tensor):
        """labels: (B,) 指定类别"""
        shape = (labels.size(0), 3, 32, 32)
        return self.p_sample_loop(shape, labels) 