import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbedding(nn.Module):
    """改进的时间嵌入层"""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
    def forward(self, timesteps):
        """
        使用正弦位置编码生成时间嵌入
        Args:
            timesteps: [batch_size] 时间步
        Returns:
            [batch_size, embed_dim] 时间嵌入
        """
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if self.embed_dim % 2 == 1:  # 如果embed_dim是奇数
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        
        return emb

class AttentionBlock(nn.Module):
    """自注意力块"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        # 计算注意力
        b, c, height, width = q.shape
        q = q.reshape(b, c, height * width).permute(0, 2, 1)  # [b, hw, c]
        k = k.reshape(b, c, height * width)  # [b, c, hw]
        v = v.reshape(b, c, height * width).permute(0, 2, 1)  # [b, hw, c]
        
        attn = torch.bmm(q, k) * (int(c) ** (-0.5))  # [b, hw, hw]
        attn = F.softmax(attn, dim=2)
        
        out = torch.bmm(attn, v)  # [b, hw, c]
        out = out.permute(0, 2, 1).reshape(b, c, height, width)
        
        return x + self.proj_out(out)

class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, time_embed_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 添加时间嵌入
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip_connection(x)

class ImprovedUNet(nn.Module):
    """改进的UNet架构"""
    def __init__(
        self, 
        in_channels=3, 
        out_channels=3,
        time_embed_dim=256,
        channels=[64, 128, 256, 512],
        attention_resolutions=[16, 8],
        num_res_blocks=2,
        dropout=0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embed_dim = time_embed_dim
        self.channels = channels
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )
        
        # 初始卷积
        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        current_resolution = 32  # 假设输入是32x32
        for i, ch in enumerate(channels):
            in_ch = channels[i-1] if i > 0 else channels[0]
            
            # 残差块
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, ch, time_embed_dim, dropout))
                in_ch = ch
            self.down_blocks.append(blocks)
            
            # 注意力块
            if current_resolution in attention_resolutions:
                self.down_attentions.append(AttentionBlock(ch))
            else:
                self.down_attentions.append(nn.Identity())
            
            # 下采样
            if i < len(channels) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                current_resolution //= 2
            else:
                self.down_samples.append(nn.Identity())
        
        # 中间块
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_embed_dim, dropout)
        self.mid_attention = AttentionBlock(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_embed_dim, dropout)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        reversed_channels = list(reversed(channels))
        for i, ch in enumerate(reversed_channels):
            out_ch = reversed_channels[i+1] if i < len(reversed_channels)-1 else channels[0]
            
            # 上采样
            if i > 0:
                self.up_samples.append(nn.ConvTranspose2d(ch, ch, 2, stride=2))
                current_resolution *= 2
            else:
                self.up_samples.append(nn.Identity())
            
            # 残差块（输入通道是ch + ch，因为有skip connection）
            blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):  # +1因为第一个块需要处理skip connection
                in_ch = ch + ch if j == 0 else out_ch
                blocks.append(ResBlock(in_ch, out_ch, time_embed_dim, dropout))
            self.up_blocks.append(blocks)
            
            # 注意力块
            if current_resolution in attention_resolutions:
                self.up_attentions.append(AttentionBlock(out_ch))
            else:
                self.up_attentions.append(nn.Identity())
        
        # 输出层
        self.output_norm = nn.GroupNorm(8, channels[0])
        self.output_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)
    
    def forward(self, x, timesteps):
        """
        Args:
            x: [batch_size, in_channels, H, W]
            timesteps: [batch_size] 时间步
        Returns:
            [batch_size, out_channels, H, W]
        """
        # 时间嵌入
        time_emb = self.time_embedding(timesteps)
        time_emb = self.time_mlp(time_emb)
        
        # 初始卷积
        h = self.input_conv(x)
        
        # 保存skip connections
        skip_connections = [h]
        
        # 下采样路径
        for blocks, attention, downsample in zip(self.down_blocks, self.down_attentions, self.down_samples):
            for block in blocks:
                h = block(h, time_emb)
            h = attention(h)
            skip_connections.append(h)
            h = downsample(h)
        
        # 中间处理
        h = self.mid_block1(h, time_emb)
        h = self.mid_attention(h)
        h = self.mid_block2(h, time_emb)
        
        # 上采样路径
        for i, (blocks, attention, upsample) in enumerate(zip(self.up_blocks, self.up_attentions, self.up_samples)):
            h = upsample(h)
            
            # 添加skip connection
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            
            for block in blocks:
                h = block(h, time_emb)
            h = attention(h)
        
        # 输出
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        
        return h

# 兼容性包装器
class UNet(ImprovedUNet):
    """保持向后兼容的UNet类"""
    def __init__(self):
        super().__init__(
            in_channels=3,
            out_channels=3,
            time_embed_dim=256,
            channels=[64, 128, 256, 512],
            attention_resolutions=[16, 8],
            num_res_blocks=2,
            dropout=0.1
        )

if __name__ == "__main__":
    # 测试代码
    unet = ImprovedUNet()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    
    print(f"输入形状: {x.shape}")
    print(f"时间步形状: {t.shape}")
    
    output = unet(x, t)
    print(f"输出形状: {output.shape}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"模型参数数量: {total_params:,}") 