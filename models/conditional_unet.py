import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalUNet(nn.Module):
    """带类别条件的简易 UNet（继承自原架构手动复制，保持兼容）。"""

    def __init__(self, num_classes: int = 10, label_embed_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim

        # ----------- 编码器 ----------- #
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )

        # ----------- 时间 & 标签嵌入 ----------- #
        # 时间嵌入保持与原 UNet 一致 (1 -> 256)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        # 类别嵌入
        self.label_emb = nn.Embedding(num_classes, label_embed_dim)
        self.label_mlp = nn.Sequential(
            nn.Linear(label_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # ----------- 解码器 ----------- #
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """x: (B,3,H,W)  t: (B,)  y: (B,)"""
        # 时间和类别嵌入
        t_embed = self.time_mlp(t.float().view(-1, 1))  # (B,256)
        y_embed = self.label_mlp(self.label_emb(y))      # (B,256)
        cond = (t_embed + y_embed).view(-1, 256, 1, 1)   # (B,256,1,1)

        # ------- 编码 ------- #
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # 在最深层注入条件
        e3 = e3 * cond

        # ------- 解码 ------- #
        d3 = self.dec3(e3)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return d1

class ConditionalUNet_Celeba(nn.Module):
    """带类别条件的 UNet，专门用于 CelebA (3, 64, 64) 图像。"""

    def __init__(self, num_classes: int = 10, label_embed_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim

        # ----------- 编码器 ----------- #
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )  # -> 64x64
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )  # -> 32x32
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )  # -> 16x16
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
        )  # -> 8x8

        # ----------- 时间 & 标签嵌入 ----------- #
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.label_emb = nn.Embedding(num_classes, label_embed_dim)
        self.label_mlp = nn.Sequential(
            nn.Linear(label_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        # ----------- 解码器 ----------- #
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),  # 8x8 -> 16x16
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 2, stride=2),  # 16x16 -> 32x32
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, stride=2),  # 32x32 -> 64x64
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """x: (B,3,64,64)  t: (B,)  y: (B,)"""
        # 时间和类别嵌入
        t_embed = self.time_mlp(t.float().view(-1, 1))  # (B,512)
        y_embed = self.label_mlp(self.label_emb(y))      # (B,512)
        cond = (t_embed + y_embed).view(-1, 512, 1, 1)   # (B,512,1,1)

        # ------- 编码 ------- #
        e1 = self.enc1(x)  # (B,64,64,64)
        e2 = self.enc2(e1)  # (B,128,32,32)
        e3 = self.enc3(e2)  # (B,256,16,16)
        e4 = self.enc4(e3)  # (B,512,8,8)

        # 在最深层注入条件
        e4 = e4 * cond

        # ------- 解码 ------- #
        d4 = self.dec4(e4)  # (B,256,16,16)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))  # (B,128,32,32)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # (B,64,64,64)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # (B,3,64,64)
        return d1

if __name__ == "__main__":
    # 测试原始 ConditionalUNet
    net = ConditionalUNet()
    x = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 100, (4,))
    y = torch.randint(0, 10, (4,))
    print("Original UNet output shape:", net(x, t, y).shape)

    # 测试 CelebA UNet
    net_celeba = ConditionalUNet_Celeba()
    x_celeba = torch.randn(4, 3, 64, 64)
    t_celeba = torch.randint(0, 100, (4,))
    y_celeba = torch.randint(0, 10, (4,))
    print("CelebA UNet output shape:", net_celeba(x_celeba, t_celeba, y_celeba).shape) 