import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # 编码器部分
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # 解码器部分 - 修改为使用插值上采样
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )
        
    def forward(self, x, t):
        # 时间嵌入
        t = t.float().view(-1, 1)
        t = self.time_mlp(t)
        t = t.view(-1, 256, 1, 1)
        
        # 编码器前向传播
        e1 = self.enc1(x)  # (B, 64, 32, 32)
        e2 = self.enc2(e1)  # (B, 128, 16, 16)
        e3 = self.enc3(e2)  # (B, 256, 8, 8)
        
        # 添加时间信息
        e3 = e3 * t
        
        # 解码器前向传播 - 使用插值上采样确保尺寸匹配
        # 上采样到与 e2 相同的尺寸
        d3_up = F.interpolate(e3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3_conv(d3_up)
        
        # 跳跃连接并上采样到与 e1 相同的尺寸
        d2_cat = torch.cat([d3, e2], dim=1)
        d2_up = F.interpolate(d2_cat, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2_conv(d2_up)
        
        # 最终跳跃连接和输出
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        return d1
    
if __name__ == "__main__":
    unet = UNet()
    x = torch.randn(1, 3, 32, 32)
    t = torch.randint(0, 100, (1,))
    print(unet(x, t).shape)